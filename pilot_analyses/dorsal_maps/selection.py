#!/usr/bin/env python3
"""Decoding from selected neurons and from high-contrast trials.

Per (session, hex bin) with >= MIN_NEURONS neurons, as in decode.py:
- `top`: the MIN_NEURONS neurons of the bin most selective for the target,
  chosen separately in every outer cross-validation fold from that fold's
  training trials only (so the held-out trials never influence the choice).
  Selectivity: |AUC - 0.5| of the two classes (Mann-Whitney U / (n1 n0),
  ties counted half; not stratified) for the binary targets, |Pearson r|
  with wheel speed over all wheel bins for wheel speed. A cheap stand-in
  for a sparse (L1) decoder.
- `active`: the MIN_NEURONS neurons of the bin with the highest fraction of
  non-zero window values (all task windows and wheel bins, labels not
  used); a stand-in for a quality selection of mesoscope ROIs.
- `hc_all` / `hc_n10` / `hc_top`: stimulus side decoded from trials with
  contrast >= HC only (all neurons / random draws / top neurons).
Also `n10` (random draws, as in decode.py) for reference.
With --pool N (pool-size control) each bin is first reduced to N randomly
drawn neurons and only `top` and `active` are computed (output
selection_<modality>_pool<N>.pqt): mesoscope bins hold many more neurons
than BWM bins, which gives the selection more to choose from.

Usage: python selection.py meso bwm [--pool 24]
Writes output/selection_<modality>.pqt (session, hex, n_neurons, target,
version, score).
"""
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from common import HEX_SIZE, MIN_NEURONS, OUT, bin_neurons
from decode import MIN_TRIALS, N_DRAWS
from gpu_decode import DEVICE, decode_binary, decode_wheel, outer_train, outer_train_wheel
from sessions import load_session, session_files

HC = 0.25  # high-contrast trials: the two highest contrasts (0.25, 1)


def avg_ranks(X):
    """Average ranks (1-based, ties averaged) along dim 0 of X (N, n)."""
    S = X.T.contiguous().sort(1).values                     # (n, N)
    lo = torch.searchsorted(S, X.T.contiguous(), right=False)
    hi = torch.searchsorted(S, X.T.contiguous(), right=True)
    return ((lo + hi + 1) / 2).T.to(torch.float64)          # (N, n)


def auc_selectivity(F, y, train):
    """|AUC - 0.5| per fold and neuron on the training trials.
    F (N, n) tensor, y (N,) 0/1, train (k, N) bool -> (k, n)."""
    out = []
    for tr in train:
        Xt, yt = F[tr], torch.as_tensor(y[tr.cpu().numpy()], device=F.device)
        R = avg_ranks(Xt)
        n1, n0 = int(yt.sum()), int((1 - yt).sum())
        U = R[yt == 1].sum(0) - n1 * (n1 + 1) / 2
        out.append((U / (n1 * n0) - 0.5).abs())
    return torch.stack(out)


def corr_selectivity(F, y, train):
    """|Pearson r| with y per fold and neuron. F (M, n), y (M,), train (k, M)."""
    out = []
    for tr in train:
        X, t = F[tr], y[tr]
        X = X - X.mean(0)
        t = t - t.mean()
        r = (X * t[:, None]).sum(0) / (X.norm(dim=0) * t.norm()).clamp(min=1e-12)
        out.append(r.abs())
    return torch.stack(out)


def top_sets(sel, bins):
    """Fold-specific index arrays (k, MIN_NEURONS): per bin the neurons with
    the largest selectivity in each fold. sel (k, n) numpy."""
    return [np.stack([idx[np.argsort(-s[idx], kind="stable")[:MIN_NEURONS]] for s in sel])
            for idx in bins]


def run_session(sid, s, pool=None):
    N = len(s["choice"])
    if N < MIN_TRIALS:
        return pd.DataFrame()
    keys = bin_neurons(s["xyz"], size=HEX_SIZE)
    bins, info = [], []
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        if idx.size >= MIN_NEURONS:
            bins.append(idx)
            info.append((int(k), idx.size))
    if not bins:
        return pd.DataFrame()
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    if pool:
        bins = [np.sort(rng.choice(idx, min(pool, idx.size), replace=False)) for idx in bins]
    draws = [[rng.choice(idx, MIN_NEURONS, replace=False) for _ in range(N_DRAWS)] for idx in bins]
    flat_draws = [d for b in draws for d in b]

    act = np.concatenate([(s["X_task"] > 0).reshape(len(s["xyz"]), -1),
                          (s["X_wheel"] > 0).reshape(len(s["xyz"]), -1)], 1).mean(1)
    active = [idx[np.argsort(-act[idx], kind="stable")[:MIN_NEURONS]] for idx in bins]

    Xt = torch.as_tensor(s["X_task"], device=DEVICE, dtype=torch.float64)  # (n, T, 3)
    labels = {"stim": s["stim_left"], "choice": (s["choice"] == 1).astype(int),
              "feedback": s["reward"]}
    res = {}
    for j, t in enumerate(("stim", "choice", "feedback")):
        F, y = Xt[:, :, j].T, labels[t]
        tr = torch.as_tensor(outer_train(y), device=DEVICE)
        top = top_sets(auc_selectivity(F, y, tr).cpu().numpy(), bins)
        res[t] = {"top": decode_binary(F, top, y), "active": decode_binary(F, active, y)}
        if not pool:
            res[t]["n10"] = decode_binary(F, flat_draws, y)
    Xw = torch.as_tensor(s["X_wheel"], device=DEVICE).permute(1, 2, 0)   # (T, nb, n)
    nt, nb, n = Xw.shape
    yw = torch.as_tensor(np.asarray(s["wheel_y"], np.float64).reshape(-1), device=DEVICE)
    tr = torch.as_tensor(outer_train_wheel(nt, nb), device=DEVICE)
    top = top_sets(corr_selectivity(Xw.reshape(nt * nb, n).to(torch.float64), yw, tr).cpu().numpy(),
                   bins)
    res["wheel"] = {"top": decode_wheel(Xw, top, s["wheel_y"]),
                    "active": decode_wheel(Xw, active, s["wheel_y"])}
    if not pool:
        res["wheel"]["n10"] = decode_wheel(Xw, flat_draws, s["wheel_y"])

    hc = np.maximum(s["c_L"], s["c_R"]) >= HC
    if hc.sum() >= MIN_TRIALS and not pool:
        F, y = Xt[:, hc, 0].T, s["stim_left"][hc]
        tr = torch.as_tensor(outer_train(y), device=DEVICE)
        top = top_sets(auc_selectivity(F, y, tr).cpu().numpy(), bins)
        res["stim"].update(hc_all=decode_binary(F, bins, y), hc_top=decode_binary(F, top, y),
                           hc_n10=decode_binary(F, flat_draws, y))
    rows = []
    for t, vers in res.items():
        for ver, sc in vers.items():
            if ver.endswith("n10"):
                sc = np.nanmean(np.reshape(sc, (len(bins), N_DRAWS)), 1)
            for (k, nn), v in zip(info, sc):
                rows.append(dict(session=sid, hex=k, n_neurons=nn, n_trials=N,
                                 n_hc=int(hc.sum()), target=t, version=ver, score=v))
    return pd.DataFrame(rows)


def run(modality, pool=None):
    warnings.filterwarnings("ignore")
    groups = list(session_files(modality).items())
    res = []
    with ThreadPoolExecutor(2) as ex:
        loaded = ex.map(lambda g: load_session(g[1]), groups)
        for i, ((sid, _), s) in enumerate(zip(groups, loaded)):
            df = run_session(sid, s, pool)
            res.append(df)
            print(modality, i + 1, len(groups), sid, len(df), flush=True)
    out = pd.concat(res, ignore_index=True)
    out.to_parquet(OUT / f"selection_{modality}{f'_pool{pool}' if pool else ''}.pqt")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("modality", nargs="*", default=["meso", "bwm"])
    ap.add_argument("--pool", type=int, default=None, help="reduce each bin to this many random neurons")
    a = ap.parse_args()
    for m in a.modality:
        run(m, a.pool)
