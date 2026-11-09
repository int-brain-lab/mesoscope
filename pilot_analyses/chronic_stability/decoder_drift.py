"""How does the population code for the stimulus change across days?

Per session pair (a, b) of a mouse, on the neurons tracked (QC-passing) in
both, ridge decoders (same features, labels and penalty as stability.py) are
fitted once per session half and then used three ways:

1. Transfer matrix: decoder trained on even (odd) trials of day i, tested on
   odd (even) trials of day j, kept per direction (i -> j and j -> i).
2. Recalibrated vs fixed: the test data are z-scored either with their own
   statistics (as in stability.py: a session-wide change of gain or offset
   is undone) or with the statistics of the training half (the decoder is
   applied exactly as trained).
3. Weight similarity: cosine between decoder weight vectors (on z-scored
   activity) of day a and day b, against the even-vs-odd cosine of the same
   day (baseline).

Within-day baseline for 1 and 2: train on one half of a session, test on the
other half of the same session, same neurons. Fraction of baseline kept:
(acc_x - 0.5) / (acc_w - 0.5), with acc_w the mean over the two sessions;
for weights cos_x / cos_w.

Output: output/decoder_drift.parquet (one row per ordered pair and variable).
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stability import MIN_SHARED, OUT, RIDGE, _date, session_summary  # noqa: E402


def _fit(X, y, dev):
    """Dual-form ridge (as stability.ridge_acc); returns z-scoring stats,
    primal weights on z-scored activity and intercept."""
    import torch
    X = torch.as_tensor(X, dtype=torch.float32, device=dev)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    A = (X - mu) / sd
    yy = torch.as_tensor(np.where(y, 1.0, -1.0), dtype=torch.float32, device=dev)
    K = A @ A.T
    ym = yy.mean()
    alpha = torch.linalg.solve(K + RIDGE * torch.diagonal(K).mean() * torch.eye(len(yy), device=dev), yy - ym)
    return mu, sd, A.T @ alpha, ym


def _acc(model, X, y, own, dev):
    import torch
    mu, sd, w, ym = model
    X = torch.as_tensor(X, dtype=torch.float32, device=dev)
    B = (X - X.mean(0)) / (X.std(0) + 1e-6) if own else (X - mu) / sd
    pred = ((B @ w + ym) > 0).cpu().numpy()
    return 0.5 * (pred[y].mean() + (~pred[~y]).mean())


def _cos(m1, m2):
    w1, w2 = m1[2], m2[2]
    return float((w1 @ w2) / (w1.norm() * w2.norm()))


def pair_rows(a, b, var, dev):
    ua, ia, ib = np.intersect1d(a["uid"], b["uid"], return_indices=True)
    if ua.size < MIN_SHARED:
        return []
    X = {"a": [x[:, ia] for x in a["X"][var]], "b": [x[:, ib] for x in b["X"][var]]}
    y = {"a": a["y"][var], "b": b["y"][var]}
    M = {(s, h): _fit(X[s][h], y[s][h], dev) for s in "ab" for h in (0, 1)}

    def acc(tr, te, own):   # train halves of tr, test the other half of te
        return np.mean([_acc(M[(tr, h)], X[te][1 - h], y[te][1 - h], own, dev) for h in (0, 1)])

    w_own = 0.5 * (acc("a", "a", True) + acc("b", "b", True))
    w_fix = 0.5 * (acc("a", "a", False) + acc("b", "b", False))
    cos_w = 0.5 * (_cos(M[("a", 0)], M[("a", 1)]) + _cos(M[("b", 0)], M[("b", 1)]))
    cos_x = 0.5 * (_cos(M[("a", 0)], M[("b", 1)]) + _cos(M[("a", 1)], M[("b", 0)]))
    out = []
    for tr, te in (("a", "b"), ("b", "a")):
        out.append(dict(var=var, train=a["name"] if tr == "a" else b["name"],
                        test=b["name"] if te == "b" else a["name"], n_shared=ua.size,
                        acc_w_own=w_own, acc_x_own=acc(tr, te, True),
                        acc_w_fixed=w_fix, acc_x_fixed=acc(tr, te, False),
                        cos_w=cos_w, cos_x=cos_x))
    return out


def run():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    qc = pd.read_parquet(OUT / "qc_positions.parquet")
    rows = []
    for sdir in sorted((OUT / "sessions").iterdir()):
        S = {}
        for f in sorted(sdir.glob("*.npz")):
            q = qc[(qc.subject == sdir.name) & (qc.session == f.stem)].sort_values("row")
            s = session_summary(f, qc_ok=q.qc_pass.values)
            if s is not None:
                S[f.stem] = dict(s, name=f.stem)
        for ka, kb in combinations(sorted(S, key=_date), 2):
            for var in ("stim", "choice"):
                for r in pair_rows(S[ka], S[kb], var, dev):
                    r["subject"] = sdir.name
                    r["dt"] = (_date(r["test"]) - _date(r["train"])).days   # > 0: tested later
                    rows.append(r)
        print(sdir.name, len(S), "sessions", flush=True)
    d = pd.DataFrame(rows)
    for k in ("own", "fixed"):
        d[f"kept_{k}"] = (d[f"acc_x_{k}"] - 0.5) / (d[f"acc_w_{k}"] - 0.5)
    d["kept_cos"] = d.cos_x / d.cos_w
    d.to_parquet(OUT / "decoder_drift.parquet")
    return d


if __name__ == "__main__":
    run()
