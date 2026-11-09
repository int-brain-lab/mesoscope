"""Does a neuron's response profile stay the same across days?

Input: OUT/sessions/<subject>/*.npz from data.py (per-trial window means of
the deconvolved trace, 9 windows: stim 0-0.3 s, pre-move -0.3-0 s, feedback
0-0.3 s, six 0.2 s bins -0.2..1.0 s around first movement; ROICaT UIDs) and
OUT/qc_positions.parquet from qc.py (only (UID, session) entries passing the
spatial tracking QC count as tracked).

Baseline for every metric: even vs odd trials of the same session. Every
across-day comparison uses the same amount of data: even trials of day a
against odd trials of day b (and odd a against even b), averaged.

Metrics per session pair (a, b) and tracked neuron:
  profile: trial-mean activity in the 9 windows for rewarded-left,
    rewarded-right and unrewarded trials (27 values; errors pooled over sides
    because trained mice make too few errors on one side).
      r_w = corr(even_a, odd_a)              within-day (baseline)
      r_x = corr(even_a, odd_b)              across days
      r_0 = as r_x, partner a random other neuron of the same FOV
      r_nn = as r_x, partner the spatially nearest other tracked neuron
    stability = (med r_x - med r_0) / (med r_w - med r_0)
  activity level: correlation across neurons of log mean activity.
  selectivity maps: correlation across neurons of d' for stimulus side,
    choice and reward (stimulus, pre-move and feedback windows).
  sensitivity persistence: neurons significantly stimulus- (contra vs ipsi)
    or choice-selective on even trials of day a (Mann-Whitney p < ALPHA);
    fraction also significant with the same sign on odd trials of day b,
    against odd trials of day a (baseline) and against the fraction
    significant on odd trials overall (chance).
  decoder: ridge classifier (dual form, torch/GPU, each half z-scored with its
    own statistics) for stimulus side and choice from the tracked neurons;
    train even a -> test odd b, against train even a -> test odd a.
    Balanced accuracy. Chance: the across-day decoders retrained on N_SHUF
    permutations of the training labels (mean and 95th percentile).
Per-neuron, per-session values (full session) go to neuron_sessions.parquet.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
MIN_PER_HALF = 5      # trials per trial type and half
MIN_SHARED = 50       # tracked neurons per session pair
ALPHA = 0.01          # significance for sensitivity persistence
N_SHUF = 50          # label shuffles per decoder for chance level
RIDGE = 0.01          # ridge penalty relative to mean(diag(X X^T)); best of 0.01-10 on one session
SEL_WIN = {"stim": 0, "choice": 1, "reward": 2}


def _date(name):
    return pd.Timestamp(name.split("_")[0])


def _rowcorr(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1) + 1e-12)


def _corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def _dprime(a, b):
    return (a.mean(1) - b.mean(1)) / (np.sqrt(0.5 * (a.var(1) + b.var(1))) + 1e-6)


def _selectivity(Xw, lab):
    """d' and Mann-Whitney p per neuron; lab is (trials,) or (n, trials) bool."""
    if lab.ndim == 1:
        a, b = Xw[:, lab], Xw[:, ~lab]
        return _dprime(a, b), mannwhitneyu(a, b, axis=1).pvalue
    d, p = np.empty(len(Xw)), np.empty(len(Xw))
    keys, inv = np.unique(lab, axis=0, return_inverse=True)   # neurons sharing a labelling (hemisphere)
    for i, key in enumerate(keys):
        r = np.flatnonzero(inv.ravel() == i)
        d[r], p[r] = _selectivity(Xw[r], key)
    return d, p


def session_summary(f, qc_ok=None):
    """Per-neuron even/odd quantities of one session (None if too few trials)."""
    z = np.load(f, allow_pickle=True)
    X, ch, rw = z["X"], z["choice"], z["reward"]
    odd = np.arange(len(ch)) % 2 == 1
    types = [(ch == -1) & (rw == 1), (ch == 1) & (rw == 1), rw == 0]
    if any(min((t & odd).sum(), (t & ~odd).sum()) < MIN_PER_HALF for t in types):
        return None
    halves = [~odd, odd]                                               # even, odd
    contra = (z["stim_left"][None, :] == 1) == (z["xyz"][:, :1] > 0)  # (n, trials)
    labs = {"stim": contra, "choice": ch == 1, "reward": rw == 1}

    prof = np.stack([np.concatenate([X[:, t & h].mean(1) for t in types], 1) for h in halves])
    act = np.stack([np.log(X[:, h].mean((1, 2)) + 1e-3) for h in halves])
    d, p = {}, {}
    for v, w in SEL_WIN.items():
        res = [_selectivity(X[:, h, w], labs[v][..., h]) for h in halves]
        d[v] = np.stack([r[0] for r in res])                           # (2, n)
        p[v] = np.nan_to_num(np.stack([r[1] for r in res]), nan=1.0)
    full = {v: _selectivity(X[:, :, SEL_WIN[v]], labs[v]) for v in ("stim", "choice")}

    uid = z["uid"]
    ok = uid != ""
    u, cnt = np.unique(uid[ok], return_counts=True)
    ok &= ~np.isin(uid, u[cnt > 1])                  # UIDs assigned to two ROIs
    if qc_ok is not None:
        ok &= qc_ok
    return dict(uid=uid[ok], row=np.flatnonzero(ok), fov=z["fov"][ok], xyz=z["xyz"][ok],
                prof=prof[:, ok], act=act[:, ok], d={k: v[:, ok] for k, v in d.items()},
                p={k: v[:, ok] for k, v in p.items()},
                full={k: (v[0][ok], v[1][ok]) for k, v in full.items()},
                X={"stim": [np.ascontiguousarray(X[ok][:, h, 0].T) for h in halves],
                   "choice": [np.ascontiguousarray(X[ok][:, h, 1].T) for h in halves]},
                y={"stim": [z["stim_left"][h] == 1 for h in halves],
                   "choice": [ch[h] == 1 for h in halves]},
                n_all=len(X), n_trials=len(ch))


def _partner(fov, rng):
    """Random different neuron of the same FOV (or -1)."""
    out = np.full(fov.size, -1)
    for f in np.unique(fov):
        i = np.flatnonzero(fov == f)
        if i.size > 1:
            out[i] = i[(np.arange(i.size) + rng.integers(1, i.size)) % i.size]
    return out


def _nearest(fov, xyz):
    """Spatially nearest other neuron of the same FOV (or -1)."""
    out = np.full(fov.size, -1)
    for f in np.unique(fov):
        i = np.flatnonzero(fov == f)
        if i.size > 1:
            out[i] = i[cKDTree(xyz[i]).query(xyz[i], k=2)[1][:, 1]]
    return out


def ridge_acc(Xtr, ytr, Xte, yte, n_shuf=0, rng=None):
    """Balanced accuracy of a dual-form ridge classifier on the GPU (matches
    sklearn RidgeClassifier with the same z-scoring and alpha).
    n_shuf > 0: instead, the accuracies (n_shuf,) of the same decoder trained on
    randomly permuted training labels (all permutations solved in one call)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def _z(M):
        M = torch.as_tensor(M, dtype=torch.float32, device=dev)
        return (M - M.mean(0)) / (M.std(0) + 1e-6)
    A, B = _z(Xtr), _z(Xte)
    y = np.where(ytr, 1.0, -1.0)
    Y = (np.stack([rng.permutation(y) for _ in range(n_shuf)], 1) if n_shuf else y[:, None])
    Y = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    K = A @ A.T
    ym = Y.mean(0)
    alpha = torch.linalg.solve(K + RIDGE * torch.diagonal(K).mean() * torch.eye(len(y), device=dev),
                               Y - ym)
    pred = ((B @ (A.T @ alpha) + ym) > 0).cpu().numpy()          # (n_test, n_labelsets)
    acc = 0.5 * (pred[yte].mean(0) + (~pred[~yte]).mean(0))
    return acc if n_shuf else float(acc[0])


def _kept(dsrc, psrc, ddst, pdst):
    """Of the neurons significant in src, fraction significant with the same sign in dst."""
    s = psrc < ALPHA
    return ((pdst < ALPHA) & (np.sign(ddst) == np.sign(dsrc)))[s].mean() if s.any() else np.nan


def pair_stats(a, b, rng, rng_shuf=None):
    rng_shuf = rng_shuf if rng_shuf is not None else np.random.default_rng(1)
    ua, ia, ib = np.intersect1d(a["uid"], b["uid"], return_indices=True)
    if ua.size < MIN_SHARED:
        return None, None
    Pa, Pb = a["prof"][:, ia], b["prof"][:, ib]
    rw_a, rw_b = _rowcorr(Pa[0], Pa[1]), _rowcorr(Pb[0], Pb[1])
    rx = 0.5 * (_rowcorr(Pa[0], Pb[1]) + _rowcorr(Pa[1], Pb[0]))
    j = _partner(a["fov"][ia], rng)
    v = j >= 0
    r0 = 0.5 * (_rowcorr(Pa[0][v], Pb[1][j[v]]) + _rowcorr(Pa[1][v], Pb[0][j[v]]))
    k = _nearest(b["fov"][ib], b["xyz"][ib])
    u = k >= 0
    rnn = 0.5 * (_rowcorr(Pa[0][u], Pb[1][k[u]]) + _rowcorr(Pa[1][u], Pb[0][k[u]]))
    row = dict(n_shared=ua.size, rw=0.5 * (np.median(rw_a) + np.median(rw_b)),
               rx=np.median(rx), r0=np.median(r0), rnn=np.median(rnn),
               d_same=1e3 * np.median(np.linalg.norm(a["xyz"][ia] - b["xyz"][ib], axis=1)),
               d_other=1e3 * np.median(np.linalg.norm(a["xyz"][ia][v] - b["xyz"][ib][j[v]], axis=1)))
    row["stability"] = (row["rx"] - row["r0"]) / (row["rw"] - row["r0"])

    # activity level and selectivity maps: across-neuron correlations
    Aa, Ab = a["act"][:, ia], b["act"][:, ib]
    row["act_w"] = 0.5 * (_corr(Aa[0], Aa[1]) + _corr(Ab[0], Ab[1]))
    row["act_x"] = 0.5 * (_corr(Aa[0], Ab[1]) + _corr(Aa[1], Ab[0]))
    for var in SEL_WIN:
        Da, Db = a["d"][var][:, ia], b["d"][var][:, ib]
        row[f"sel_{var}_w"] = 0.5 * (_corr(Da[0], Da[1]) + _corr(Db[0], Db[1]))
        row[f"sel_{var}_x"] = 0.5 * (_corr(Da[0], Db[1]) + _corr(Da[1], Db[0]))

    # sensitivity persistence: significant on one half -> other half, same day vs other day
    for var in ("stim", "choice"):
        da, pa = a["d"][var][:, ia], a["p"][var][:, ia]
        db, pb = b["d"][var][:, ib], b["p"][var][:, ib]
        row[f"{var}_sig"] = 0.5 * ((pa < ALPHA).mean() + (pb < ALPHA).mean())
        row[f"{var}_kept_w"] = np.nanmean([_kept(da[0], pa[0], da[1], pa[1]), _kept(da[1], pa[1], da[0], pa[0]),
                                           _kept(db[0], pb[0], db[1], pb[1]), _kept(db[1], pb[1], db[0], pb[0])])
        row[f"{var}_kept_x"] = np.nanmean([_kept(da[0], pa[0], db[1], pb[1]), _kept(da[1], pa[1], db[0], pb[0]),
                                           _kept(db[0], pb[0], da[1], pa[1]), _kept(db[1], pb[1], da[0], pa[0])])

    # decoders: train one half -> test the other half, same day vs other day
    for var in ("stim", "choice"):
        Xa = [x[:, ia] for x in a["X"][var]]
        Xb = [x[:, ib] for x in b["X"][var]]
        ya, yb = a["y"][var], b["y"][var]
        row[f"dec_{var}_w"] = np.mean([ridge_acc(Xa[h], ya[h], Xa[1 - h], ya[1 - h]) for h in (0, 1)]
                                      + [ridge_acc(Xb[h], yb[h], Xb[1 - h], yb[1 - h]) for h in (0, 1)])
        row[f"dec_{var}_x"] = np.mean([ridge_acc(Xa[h], ya[h], Xb[1 - h], yb[1 - h]) for h in (0, 1)]
                                      + [ridge_acc(Xb[h], yb[h], Xa[1 - h], ya[1 - h]) for h in (0, 1)])
        # chance: same across-day decoders trained on shuffled labels
        sh = np.concatenate([ridge_acc(Xa[h], ya[h], Xb[1 - h], yb[1 - h], N_SHUF, rng_shuf) for h in (0, 1)]
                            + [ridge_acc(Xb[h], yb[h], Xa[1 - h], ya[1 - h], N_SHUF, rng_shuf) for h in (0, 1)])
        row[f"dec_{var}_shuf"] = sh.mean()
        row[f"dec_{var}_shuf95"] = np.quantile(sh, 0.95)

    neur = pd.DataFrame(dict(rx=rx.astype(np.float32), rw=np.sqrt(
        np.clip(rw_a, 0, None) * np.clip(rw_b, 0, None)).astype(np.float32)))
    return row, neur


def run(seed=0):
    rng = np.random.default_rng(seed)
    rng_shuf = np.random.default_rng(seed + 1)   # separate stream: r0 partners unchanged
    qc = pd.read_parquet(OUT / "qc_positions.parquet")
    rows, neurons, sess_rows, per_neuron = [], [], [], []
    for sdir in sorted((OUT / "sessions").iterdir()):
        S = {}
        for f in sorted(sdir.glob("*.npz")):
            q = qc[(qc.subject == sdir.name) & (qc.session == f.stem)].sort_values("row")
            s = session_summary(f, qc_ok=q.qc_pass.values)
            if s is None:
                print(f"{sdir.name} {f.stem}: too few trials per type and half, skipped")
                continue
            S[f.stem] = s
            sess_rows.append(dict(subject=sdir.name, session=f.stem, n_neurons=s["n_all"],
                                  n_tracked=s["uid"].size, n_trials=s["n_trials"],
                                  rw=np.median(_rowcorr(s["prof"][0], s["prof"][1]))))
            per_neuron.append(pd.DataFrame(dict(
                subject=sdir.name, session=f.stem, uid=s["uid"], row=s["row"], fov=s["fov"],
                rel=_rowcorr(s["prof"][0], s["prof"][1]), act=s["act"].mean(0),
                d_stim=s["full"]["stim"][0], p_stim=np.nan_to_num(s["full"]["stim"][1], nan=1.0),
                d_choice=s["full"]["choice"][0],
                p_choice=np.nan_to_num(s["full"]["choice"][1], nan=1.0))))
        for ka, kb in combinations(sorted(S, key=_date), 2):
            row, neur = pair_stats(S[ka], S[kb], rng, rng_shuf)
            if row is None:
                continue
            dt = (_date(kb) - _date(ka)).days
            rows.append(dict(subject=sdir.name, a=ka, b=kb, dt=dt, **row))
            neurons.append(neur.assign(subject=sdir.name, dt=dt))
        print(sdir.name, len(S), "sessions", sum(r["subject"] == sdir.name for r in rows), "pairs",
              flush=True)
    pairs, sess = pd.DataFrame(rows), pd.DataFrame(sess_rows)
    pairs.to_csv(OUT / "pairs.csv", index=False)
    sess.to_csv(OUT / "sessions.csv", index=False)
    pd.concat(neurons, ignore_index=True).to_parquet(OUT / "neurons.parquet")
    pd.concat(per_neuron, ignore_index=True).to_parquet(OUT / "neuron_sessions.parquet")
    return pairs, sess


if __name__ == "__main__":
    run()
