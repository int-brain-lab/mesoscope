"""REML variance components for crossed random intercepts, in torch (GPU when available).

Model: y = X b + sum_k Z_k u_k + e, u_k ~ N(0, s_k^2 I), e ~ N(0, s_e^2 I), so
V = sum_k s_k^2 Z_k Z_k' + s_e^2 I. The restricted log likelihood
-0.5 [log|V| + log|X'V^-1 X| + y'P y] is maximised over log variances with L-BFGS,
with dense n x n matrices (fine for n up to a few thousand). Same model as
statsmodels MixedLM with one variance component per factor (region_variance.mixed),
about 100x faster, for bootstraps; validated against statsmodels in region_variance.
"""
import numpy as np
import torch
from scipy import stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _neg_reml(phi, y, X, Ks):
    th = torch.exp(phi)
    V = th[-1] * torch.eye(len(y), dtype=y.dtype, device=y.device)
    for t, K in zip(th[:-1], Ks):
        V = V + t * K
    L = torch.linalg.cholesky(V)
    Vi_X = torch.cholesky_solve(X, L)
    Vi_y = torch.cholesky_solve(y[:, None], L)[:, 0]
    XtViX = X.T @ Vi_X
    L2 = torch.linalg.cholesky(XtViX)
    b = torch.cholesky_solve((X.T @ Vi_y)[:, None], L2)[:, 0]
    r = y - X @ b
    quad = r @ torch.cholesky_solve(r[:, None], L)[:, 0]
    return 0.5 * (2 * torch.log(torch.diagonal(L)).sum() + 2 * torch.log(torch.diagonal(L2)).sum() + quad), b


def fit(y, X, groups, n_iter=200):
    """groups: list of integer label arrays (one per random factor).
    Returns (variances [s_1..s_K, s_e], fixed effects b, restricted log likelihood)."""
    y = torch.as_tensor(y, dtype=torch.float64, device=DEVICE)
    X = torch.as_tensor(X, dtype=torch.float64, device=DEVICE)
    Ks = []
    for g in groups:
        g = torch.as_tensor(np.unique(g, return_inverse=True)[1], device=DEVICE)
        Ks.append((g[:, None] == g[None, :]).to(torch.float64))
    v0 = float(y.var())
    phi = torch.full((len(groups) + 1,), np.log(v0 / (len(groups) + 1)), dtype=torch.float64, device=DEVICE,
                     requires_grad=True)
    opt = torch.optim.LBFGS([phi], max_iter=n_iter, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = _neg_reml(phi.clamp(np.log(v0) - 30, np.log(v0) + 5), y, X, Ks)[0]
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        p = phi.clamp(np.log(v0) - 30, np.log(v0) + 5)
        loss, b = _neg_reml(p, y, X, Ks)
    return torch.exp(p).cpu().numpy(), b.cpu().numpy(), -float(loss)


def components(df, comps, covariate="logN", value="all", lrt=()):
    """Variance shares of comps and residual; likelihood-ratio p-values (chi-square 0/1 mixture)
    for the components in lrt."""
    comps = [c for c in comps if df[c].nunique() > 1]
    X = np.c_[np.ones(len(df)), df[covariate].values]
    y = df[value].values
    var, b, ll = fit(y, X, [df[c].values for c in comps])
    out = {f"frac_{c}": v / var.sum() for c, v in zip(comps, var[:-1])}
    out["frac_residual"] = var[-1] / var.sum()
    out["slope"] = b[1]
    for c in lrt:
        if c in comps:
            rest = [k for k in comps if k != c]
            ll0 = fit(y, X, [df[k].values for k in rest])[2]
            out[f"p_lrt_{c}"] = 0.5 * stats.chi2.sf(max(2 * (ll - ll0), 0), 1)
    return out
