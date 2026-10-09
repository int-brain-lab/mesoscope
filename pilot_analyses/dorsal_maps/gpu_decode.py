"""GPU (torch) versions of common.decode_binary / common.decode_wheel,
batched over many neuron sets of one session.

Same procedure, folds and hyper-parameter grids as the scikit-learn
versions, so scores agree up to solver tolerance:

decode_binary: outer StratifiedKFold(5, shuffle, seed); features
standardised on the outer training set; inside it,
LogisticRegressionCV-style selection of C over Cs by 3-fold (unshuffled
StratifiedKFold) balanced accuracy; L2 logistic regression with
class_weight='balanced' (objective C * sum(sw * logloss) + ||w||^2 / 2,
intercept unpenalised) refit on the outer training set with the best C;
balanced accuracy of the pooled outer-test predictions.

decode_wheel: outer KFold(5, shuffle, seed) over trials; features
standardised on the training rows; ridge penalty chosen by exact
leave-one-out error (as RidgeCV with cv=None) over alphas; R^2 of the
pooled outer-test predictions.

Batching: a session's features F (n_trials, n_neurons) stay on the GPU;
each neuron set (one hex bin, or a random draw from it) is a row of an
index matrix padded with a zero column, so sets of different size are
solved together (a zero column standardises to zero and gets zero weight,
which leaves the fit unchanged - the same as the reference dropping
constant columns). For one set all 5 folds x (3 inner + 1 outer fit) x
len(Cs) logistic fits are solved at once by batched Newton steps; sets are
grouped into chunks bounded by GPU memory. Folds depend only on the labels
and are computed once per label vector.
"""
from functools import lru_cache

import numpy as np
import torch

CS = np.logspace(-4, 1, 6)
ALPHAS = np.logspace(-1, 5, 7)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_ELEMS = 1.5e8  # per chunk: bound on the largest intermediate (floats)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _gather(F, sets):
    """F (N, n) tensor, sets: list of index arrays -> (G, N, pmax) with zero
    padding, plus the number of non-constant columns per set."""
    pmax = max(len(s) for s in sets)
    pad = F.shape[1]
    idx = np.full((len(sets), pmax), pad)
    for g, s in enumerate(sets):
        idx[g, :len(s)] = s
    Fz = torch.cat([F, torch.zeros(F.shape[0], 1, dtype=F.dtype, device=F.device)], 1)
    X = Fz[:, torch.as_tensor(idx, device=F.device)].permute(1, 0, 2)  # (G, N, pmax)
    n_var = (X.std(1) > 0).sum(1)
    return X, n_var


def _chunks(sets, cost):
    """Greedy chunks of set indices (sorted by size) with sum of cost(pmax) per set bounded."""
    order = np.argsort([len(s) for s in sets], kind="stable")
    chunks, cur = [], []
    for g in order:
        p = len(sets[g])
        if cur and (len(cur) + 1) * cost(p) > MAX_ELEMS:
            chunks.append(cur)
            cur = []
        cur.append(g)
    if cur:
        chunks.append(cur)
    return chunks


def _standardise(X, train):
    """X (G, N, p) float64, train (k, N) bool -> (G, k, N, p): standardised on
    each fold's training rows (population sd, sd 0 -> 1, as StandardScaler)."""
    w = train.to(X.dtype)[None, :, :, None]                    # (1, k, N, 1)
    n = w.sum(2, keepdim=True)
    mu = (X[:, None] * w).sum(2, keepdim=True) / n             # (G, k, 1, p)
    sd = (((X[:, None] - mu) ** 2) * w).sum(2, keepdim=True).div(n).sqrt()
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    return (X[:, None] - mu) / sd


# ---------------------------------------------------------------------------
# logistic regression
# ---------------------------------------------------------------------------
def _newton_logreg(Xa, y, sw, lam, pen, n_iter=50, tol=1e-6):
    """Batched L2 logistic regression, K fits per design matrix.

    Xa: (B, N, P) features incl. intercept column; y: (N,); sw: (B, K, N)
    sample weights (0 = not in this fit's training set); lam: (K,) = 1 / C;
    pen: (P,) 1 for penalised coefficients, 0 for the intercept.
    Minimises sum(sw * logloss) + lam/2 ||w||^2. Returns theta (B, K, P).
    """
    B, N, P = Xa.shape
    K = sw.shape[1]
    theta = torch.zeros(B, K, P, device=Xa.device, dtype=Xa.dtype)
    reg = lam[:, None] * pen[None, :]                           # (K, P)
    ridge = torch.diag_embed(reg + 1e-8)                        # (K, P, P)

    def loss(th):
        z = torch.einsum("bnp,bkp->bkn", Xa, th)
        ll = torch.nn.functional.softplus(z) - y * z
        return (sw * ll).sum(2) + 0.5 * (reg * th ** 2).sum(2)

    f = loss(theta)
    for _ in range(n_iter):
        z = torch.einsum("bnp,bkp->bkn", Xa, theta)
        s = torch.sigmoid(z)
        g = torch.einsum("bnp,bkn->bkp", Xa, sw * (s - y)) + reg * theta
        Wt = sw * s * (1 - s)
        H = torch.einsum("bnp,bkn,bnq->bkpq", Xa, Wt, Xa) + ridge
        L, info = torch.linalg.cholesky_ex(H)
        if bool(info.any()):
            step = torch.linalg.solve(H, g[..., None])[..., 0]
        else:
            step = torch.cholesky_solve(g[..., None], L)[..., 0]
        t = torch.ones(B, K, 1, device=Xa.device, dtype=Xa.dtype)
        for _ in range(20):  # backtracking: halve the step where the loss went up
            f_new = loss(theta - t * step)
            bad = f_new > f + 1e-7 * f.abs()
            if not bad.any():
                break
            t = torch.where(bad[..., None], t / 2, t)
        theta = theta - t * step
        f = loss(theta)
        if (t * step).abs().max() < tol:
            break
    return theta


@lru_cache(maxsize=64)
def _binary_folds(y_bytes, n, seed, n_folds):
    """Masks for the outer folds and the 3 inner + 1 refit per outer fold:
    train (k, N); weights (k, 4, N) (class_weight='balanced' from the outer
    training set, 0 outside the fit); test (k, 4, N)."""
    from sklearn.model_selection import StratifiedKFold
    y = np.frombuffer(y_bytes, dtype=np.int64)
    Z = np.zeros((n, 1))
    train = np.zeros((n_folds, n), bool)
    W = np.zeros((n_folds, 4, n))
    test = np.zeros((n_folds, 4, n), bool)
    for k, (tr, te) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(Z, y)):
        train[k, tr] = True
        inner = list(StratifiedKFold(3).split(Z[tr], y[tr]))
        cnt = np.bincount(y[tr], minlength=2)
        for j, (sub, tst) in enumerate([(tr[a], tr[b]) for a, b in inner] + [(tr, te)]):
            W[k, j, sub] = len(tr) / (2 * cnt[y[sub]])
            test[k, j, tst] = True
    return train, W, test


def _bal_acc(y, pred, mask):
    """Balanced accuracy over the last axis restricted to mask; y (N,)."""
    acc = 0
    for c in (0, 1):
        m = mask & (y == c)
        acc = acc + ((pred == c) & m).sum(-1) / m.sum(-1).clamp(min=1)
    return acc / 2


def decode_binary(F, sets, y, seed=0, n_folds=5, Cs=CS, dtype=torch.float32):
    """Cross-validated balanced accuracy for each neuron set.

    F: (n_trials, n_neurons) array or tensor; sets: list of index arrays
    into the neuron axis; y: (n_trials,) binary labels. Returns (len(sets),).
    """
    y = np.asarray(y).astype(np.int64)
    out = np.full(len(sets), np.nan)
    if np.bincount(y, minlength=2).min() < n_folds or not sets:
        return out
    N, nC = len(y), len(Cs)
    train, W, test = (torch.as_tensor(a, device=DEVICE) for a in _binary_folds(y.tobytes(), N, seed, n_folds))
    sw = W.to(dtype).repeat_interleave(nC, 1)                   # (k, 4 nC, N)
    test_k = test.repeat_interleave(nC, 1)
    lam = torch.as_tensor(np.tile(1 / np.asarray(Cs), 4), dtype=dtype, device=DEVICE)
    yt = torch.as_tensor(y, device=DEVICE)
    yf = yt.to(dtype)
    F = torch.as_tensor(F, dtype=torch.float64, device=DEVICE)
    K = 4 * nC
    cost = lambda p: n_folds * K * max(N * (p + 1), (p + 1) ** 2)  # noqa: E731
    for chunk in _chunks(sets, cost):
        X, n_var = _gather(F, [sets[g] for g in chunk])
        G, _, p = X.shape
        Xo = _standardise(X, train).to(dtype)                    # (G, k, N, p)
        Xa = torch.cat([Xo, torch.ones(G, n_folds, N, 1, dtype=dtype, device=DEVICE)], 3)
        Xa = Xa.reshape(G * n_folds, N, p + 1)
        pen = torch.ones(p + 1, dtype=dtype, device=DEVICE)
        pen[-1] = 0
        swb = sw.repeat(G, 1, 1)                                 # (G k, K, N)
        theta = _newton_logreg(Xa, yf, swb, lam, pen)
        pred = (torch.einsum("bnp,bkp->bkn", Xa, theta) > 0).long().view(G, n_folds, 4, nC, N)
        score = _bal_acc(yt, pred, test_k.view(1, n_folds, 4, nC, N))      # (G, k, 4, nC)
        best = score[:, :, :3].mean(2).argmax(2)  # first maximum = smallest C on ties, as sklearn
        final = torch.gather(pred[:, :, 3], 2, best[:, :, None, None].expand(G, n_folds, 1, N))[:, :, 0]
        pooled = (final * test[:, 3].long()).sum(1)              # (G, N): each trial tested once
        acc = _bal_acc(yt, pooled, torch.ones_like(pooled, dtype=torch.bool))
        acc = torch.where(n_var > 0, acc, torch.full_like(acc, float("nan")))
        out[chunk] = acc.cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# ridge regression of wheel speed
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _trial_folds(nt, nb, seed, n_folds):
    from sklearn.model_selection import KFold
    trial = np.repeat(np.arange(nt), nb)
    return np.stack([np.isin(trial, tr) for tr, _ in
                     KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(nt))])


def decode_wheel(F, sets, y, seed=0, n_folds=5, alphas=ALPHAS):
    """Cross-validated R^2 for each neuron set.

    F: (n_trials, n_bins, n_neurons); y: (n_trials, n_bins). Returns (len(sets),).
    """
    out = np.full(len(sets), np.nan)
    if not sets:
        return out
    nt, nb, nn = F.shape
    n = nt * nb
    F = torch.as_tensor(np.asarray(F).reshape(n, nn) if not torch.is_tensor(F) else F.reshape(n, nn),
                        dtype=torch.float64, device=DEVICE)
    yf = torch.as_tensor(np.asarray(y, np.float64).reshape(-1), device=DEVICE)
    train = torch.as_tensor(_trial_folds(nt, nb, seed, n_folds), device=DEVICE)
    al = torch.as_tensor(alphas, dtype=torch.float64, device=DEVICE)
    cost = lambda p: n * max(p + 1, 2 * len(alphas)) + 2 * (p + 1) ** 2  # noqa: E731
    for chunk in _chunks(sets, cost):
        X, n_var = _gather(F, [sets[g] for g in chunk])          # (G, n, p)
        G = X.shape[0]
        pred = torch.empty(G, n, dtype=torch.float64, device=DEVICE)
        for k in range(n_folds):
            itr = train[k]
            Xs = _standardise(X, itr[None])[:, 0]                 # (G, n, p)
            A, b = Xs[:, itr], yf[itr]
            Am, bm = A.mean(1, keepdim=True), b.mean()
            Ac, bc = A - Am, b - bm
            s, V = torch.linalg.eigh(Ac.transpose(1, 2) @ Ac)
            s = s.clamp(min=0)
            Z = Ac @ V                                           # (G, ntr, p)
            zy = torch.einsum("gnp,n->gp", Z, bc)
            inv = 1 / (s[:, None, :] + al[None, :, None])        # (G, n_alpha, p)
            yhat = bm + torch.einsum("gnp,gap->gan", Z, inv * zy[:, None])
            h = torch.einsum("gnp,gap->gan", Z ** 2, inv) + 1 / A.shape[1]
            loo = (((b - yhat) / (1 - h)) ** 2).mean(2)          # (G, n_alpha)
            kbest = loo.argmin(1)
            w = torch.einsum("gpq,gq->gp", V, inv[torch.arange(G), kbest] * zy)
            pred[:, ~itr] = torch.einsum("gnp,gp->gn", Xs[:, ~itr] - Am, w) + bm
        r2 = 1 - ((yf - pred) ** 2).sum(1) / ((yf - yf.mean()) ** 2).sum()
        r2 = torch.where(n_var > 0, r2, torch.full_like(r2, float("nan")))
        out[chunk] = r2.cpu().numpy()
    return out
