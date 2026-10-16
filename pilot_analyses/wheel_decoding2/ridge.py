"""Batched GPU ridge decoding of wheel speed with options, to test the
dorsal_maps wheel decoder (gpu_decode.decode_wheel).

decode(F, sets, y, inner="loo"|"trial", lags=(0,), target="speed"|"residual",
       shuffle=False)
- outer 5-fold CV over trials (same folds as gpu_decode / sklearn KFold);
- features standardised on the outer training rows;
- ridge penalty chosen over ALPHAS by exact leave-one-row-out error
  (inner="loo", as RidgeCV and the dorsal_maps decoder), by exact
  leave-one-trial-out error (inner="loto"; rows of one trial are never split,
  same cost as "loo"), or by 4-fold CV over the training trials (inner="trial");
- lags: neural bins b+L (L in lags, clipped at the window edges) are
  concatenated as features for wheel bin b;
- target "residual": wheel speed minus its mean per bin over the outer
  training trials (trial-specific speed; the stereotyped time course is
  removed), R^2 against that residual;
- shuffle: trials of y are permuted (bins kept), a null for trial-specific
  information.
Returns R^2 of the pooled outer-test predictions per set.
"""
from functools import lru_cache

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHAS = np.logspace(-1, 5, 7)


@lru_cache(maxsize=32)
def _folds(nt, seed=0, n_folds=5):
    from sklearn.model_selection import KFold
    return [(tr, te) for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(nt))]


def lagged(F, lags):
    """F (nt, nb, n) -> (nt, nb, n * len(lags)); lag L uses bin b+L, clipped."""
    nt, nb, n = F.shape
    out = [F[:, np.clip(np.arange(nb) + L, 0, nb - 1)] for L in lags]
    return torch.cat(out, 2) if torch.is_tensor(F) else np.concatenate(out, 2)


def _fit_predict(A, b, Q, alphas):
    """Ridge with centring. A (G, m, p), b (m,), Q (G, q, p) -> predictions
    (G, n_alpha, q) for every alpha."""
    Am, bm = A.mean(1, keepdim=True), b.mean()
    Ac = A - Am
    s, V = torch.linalg.eigh(Ac.transpose(1, 2) @ Ac)
    s = s.clamp(min=0)
    zy = torch.einsum("gmp,gpk,m->gk", Ac, V, b - bm)
    inv = 1 / (s[:, None, :] + alphas[None, :, None])           # (G, a, p)
    W = torch.einsum("gpk,gak->gap", V, inv * zy[:, None])       # (G, a, p)
    return torch.einsum("gqp,gap->gaq", Q - Am, W) + bm


def _loo_best(A, b, alphas):
    Am, bm = A.mean(1, keepdim=True), b.mean()
    Ac, bc = A - Am, b - bm
    s, V = torch.linalg.eigh(Ac.transpose(1, 2) @ Ac)
    s = s.clamp(min=0)
    Z = Ac @ V
    zy = torch.einsum("gnp,n->gp", Z, bc)
    inv = 1 / (s[:, None, :] + alphas[None, :, None])
    yhat = bm + torch.einsum("gnp,gap->gan", Z, inv * zy[:, None])
    h = torch.einsum("gnp,gap->gan", Z ** 2, inv) + 1 / A.shape[1]
    return (((b - yhat) / (1 - h)) ** 2).mean(2).argmin(1)


def _loto_best(A, b, alphas, nb):
    """Alpha with the smallest exact leave-one-trial-out error. Rows of A are
    trials x nb bins (trial-major). For ridge with intercept the held-out
    residuals of trial g are (I - H_gg)^-1 e_g, with H the hat matrix of the
    centred fit plus 1/m (intercept); one eigendecomposition serves all
    alphas and trials."""
    G, m, p = A.shape
    Am, bm = A.mean(1, keepdim=True), b.mean()
    Ac, bc = A - Am, b - bm
    s, V = torch.linalg.eigh(Ac.transpose(1, 2) @ Ac)
    s = s.clamp(min=0)
    Z = Ac @ V                                                    # (G, m, p)
    zy = torch.einsum("gmp,m->gp", Z, bc)
    inv = 1 / (s[:, None, :] + alphas[None, :, None])             # (G, a, p)
    e = b - (bm + torch.einsum("gmp,gap->gam", Z, inv * zy[:, None]))   # (G, a, m)
    nt = m // nb
    Zt = Z.reshape(G, nt, nb, p)
    H = torch.einsum("gtbp,gap,gtcp->gatbc", Zt, inv, Zt) + 1 / m        # (G, a, nt, nb, nb)
    eye = torch.eye(nb, dtype=A.dtype, device=A.device)
    el = torch.linalg.solve(eye - H, e.reshape(G, len(alphas), nt, nb, 1))
    return (el ** 2).mean((2, 3, 4)).argmin(1)


def _spectral(A, b, Q, alphas, nb, inner):
    """Ridge with intercept for all alphas from one eigendecomposition,
    primal (p x p) when p <= m, dual (m x m Gram matrix) otherwise.
    A (G, m, p) training rows (trial-major, nb bins per trial), b (m,),
    Q (G, q, p) rows to predict. Returns predictions (G, n_alpha, q) and,
    for inner "loo"/"loto", the index of the best alpha per set."""
    G, m, p = A.shape
    Am, bm = A.mean(1, keepdim=True), b.mean()
    Ac, bc, Qc = A - Am, b - bm, Q - Am
    if p <= m:
        s, V = torch.linalg.eigh(Ac.transpose(1, 2) @ Ac)
        s = s.clamp(min=0)
        L = Ac @ V                                               # H = L diag(1/(s+a)) L^T
        Ly = torch.einsum("gmr,m->gr", L, bc)
        inv = 1 / (s[:, None, :] + alphas[None, :, None])        # (G, a, r)
        w = inv
        pred = bm + torch.einsum("gqr,gar->gaq", Qc @ V, inv * Ly[:, None])
    else:
        s, U = torch.linalg.eigh(Ac @ Ac.transpose(1, 2))
        s = s.clamp(min=0)
        L = U                                                    # H = U diag(s/(s+a)) U^T
        Ly = torch.einsum("gmr,m->gr", U, bc)
        inv = 1 / (s[:, None, :] + alphas[None, :, None])
        w = s[:, None, :] * inv
        pred = bm + torch.einsum("gqr,gar->gaq", Qc @ Ac.transpose(1, 2) @ U, inv * Ly[:, None])
    if inner not in ("loo", "loto"):
        return pred, None
    e = b - (bm + torch.einsum("gmr,gar->gam", L, w * Ly[:, None]))   # training residuals (G, a, m)
    if inner == "loo":
        h = torch.einsum("gmr,gar->gam", L ** 2, w) + 1 / m
        return pred, (((e / (1 - h)) ** 2).mean(2)).argmin(1)
    nt = m // nb
    Lt = L.reshape(G, nt, nb, -1)
    H = torch.einsum("gtbr,gar,gtcr->gatbc", Lt, w, Lt) + 1 / m
    eye = torch.eye(nb, dtype=A.dtype, device=A.device)
    el = torch.linalg.solve(eye - H, e.reshape(G, len(alphas), nt, nb, 1))
    return pred, ((el ** 2).mean((2, 3, 4))).argmin(1)


def decode(F, sets, y, inner="loo", lags=(0,), target="speed", shuffle=False, seed=0, alphas=ALPHAS,
           return_alpha=False):
    F = np.asarray(F, np.float64)
    y = np.asarray(y, np.float64)
    nt, nb, n = F.shape
    if shuffle:
        y = y[np.random.default_rng(seed + 1).permutation(nt)]
    L = len(lags)
    if lags != (0,):
        F = lagged(F, lags)
        sets = [np.concatenate([np.asarray(s) + k * n for k in range(L)]) for s in sets]
    Ft = torch.as_tensor(F, device=DEVICE)
    yt = torch.as_tensor(y, device=DEVICE)
    al = torch.as_tensor(alphas, dtype=torch.float64, device=DEVICE)
    out = np.full(len(sets), np.nan)
    best_alpha = np.full((len(sets), 5), np.nan)
    sizes = np.array([len(s) for s in sets])
    for p in np.unique(sizes):                                   # equal-size sets in one batch
        gi = np.flatnonzero(sizes == p)
        for c0 in range(0, len(gi), max(1, int(2e8 // (nt * nb * p * 8)))):
            g = gi[c0:c0 + max(1, int(2e8 // (nt * nb * p * 8)))]
            idx = torch.as_tensor(np.stack([sets[i] for i in g]), device=DEVICE)
            X = Ft[:, :, idx].permute(2, 0, 1, 3)                 # (G, nt, nb, p)
            pred = torch.zeros(len(g), nt, nb, dtype=torch.float64, device=DEVICE)
            tgt = torch.zeros(nt, nb, dtype=torch.float64, device=DEVICE)
            for k, (tr, te) in enumerate(_folds(nt)):
                tr_t, te_t = torch.as_tensor(tr, device=DEVICE), torch.as_tensor(te, device=DEVICE)
                Xtr = X[:, tr_t].reshape(len(g), -1, p)
                mu, sd = Xtr.mean(1, keepdim=True), Xtr.std(1, unbiased=False, keepdim=True)
                sd = torch.where(sd == 0, torch.ones_like(sd), sd)
                Xs = (X - mu[:, None]) / sd[:, None]
                ytr = yt[tr_t]
                if target == "residual":
                    bmean = ytr.mean(0)
                    ytr = ytr - bmean
                    tgt[te_t] = yt[te_t] - bmean
                else:
                    tgt[te_t] = yt[te_t]
                A, b = Xs[:, tr_t].reshape(len(g), -1, p), ytr.reshape(-1)
                Qt = Xs[:, te_t].reshape(len(g), -1, p)
                if inner in ("loo", "loto"):
                    P, kb = _spectral(A, b, Qt, al, nb, inner)
                else:                                            # 4-fold CV over training trials
                    err = torch.zeros(len(g), len(alphas), dtype=torch.float64, device=DEVICE)
                    for j, (itr, ite) in enumerate(_folds(len(tr), seed=1, n_folds=4)):
                        a_tr, a_te = Xs[:, tr_t[itr]].reshape(len(g), -1, p), Xs[:, tr_t[ite]].reshape(len(g), -1, p)
                        b_tr, b_te = ytr[itr].reshape(-1), ytr[ite].reshape(-1)
                        err += ((_spectral(a_tr, b_tr, a_te, al, nb, None)[0] - b_te) ** 2).mean(2)
                    kb = err.argmin(1)
                    P = _spectral(A, b, Qt, al, nb, None)[0]
                best_alpha[g, k] = alphas[kb.cpu().numpy()]
                pred[:, te_t] = P[torch.arange(len(g)), kb].reshape(len(g), len(te), nb)
            r2 = 1 - ((tgt - pred) ** 2).sum((1, 2)) / ((tgt - tgt.mean()) ** 2).sum()
            out[g] = r2.cpu().numpy()
    return (out, best_alpha) if return_alpha else out


def bin_baseline(y):
    """R^2 of predicting each bin's wheel speed by its mean over the outer
    training trials (the stereotyped time course alone)."""
    y = np.asarray(y, np.float64)
    pred = np.zeros_like(y)
    for tr, te in _folds(len(y)):
        pred[te] = y[tr].mean(0)
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
