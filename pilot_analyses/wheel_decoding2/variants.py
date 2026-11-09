#!/usr/bin/env python3
"""Wheel-decoder variants on every (session, 0.4 mm bin) of both modalities.

Sets per bin: N_DRAWS random draws of 10 neurons (as dorsal_maps "n10") and
all neurons of the bin. Variants (ridge.decode):
  current      leave-one-row-out alpha, same-bin features, wheel speed (= dorsal_maps)
  loto         alpha by exact leave-one-trial-out error instead
  lag          + neural bins b-1, b, b+1 as features (loto)
  resid        trial-specific speed (bin means of the training trials removed), loto
  resid_lag    + lags
  resid_null   resid with trials of y shuffled (should be <= 0)
Per session also the bin-only baseline R^2 (bin means of the training trials).

Usage: python variants.py  -> output/variants.pqt, output/variants_sessions.pqt
"""
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import ridge
from wd_common import MODS, OUT, load_session, session_files
from common import MIN_NEURONS, bin_neurons
from decode import MIN_TRIALS

N_DRAWS = 3
VARIANTS = {
    "current": dict(inner="loo"),
    "loto": dict(inner="loto"),
    "lag": dict(inner="loto", lags=(-1, 0, 1)),
    "resid": dict(inner="loto", target="residual"),
    "resid_lag": dict(inner="loto", target="residual", lags=(-1, 0, 1)),
    "resid_null": dict(inner="loto", target="residual", shuffle=True),
}


def run_session(sid, s, m):
    if len(s["choice"]) < MIN_TRIALS:
        return pd.DataFrame(), None
    F = s["X_wheel"].astype(np.float64).transpose(1, 2, 0)
    y = s["wheel_y"]
    keys = bin_neurons(s["xyz"])
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    bins = [(int(k), np.flatnonzero(keys == k)) for k in np.unique(keys) if (keys == k).sum() >= MIN_NEURONS]
    if not bins:
        return pd.DataFrame(), None
    draws = [rng.choice(idx, MIN_NEURONS, replace=False) for _, idx in bins for _ in range(N_DRAWS)]
    rows = []
    for v, kw in VARIANTS.items():
        sc = ridge.decode(F, draws, y, **kw).reshape(len(bins), N_DRAWS).mean(1)
        sa = ridge.decode(F, [idx for _, idx in bins], y, **kw)
        rows += [dict(modality=m, session=sid, hex=k, n_neurons=len(idx), variant=v, n10=a, all=b)
                 for (k, idx), a, b in zip(bins, sc, sa)]
    yr = y - y.mean(0)
    sess = dict(modality=m, session=sid, n_trials=len(y), bin_baseline=ridge.bin_baseline(y),
                frac_trial_specific=float((yr ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    return pd.DataFrame(rows), sess


def main():
    warnings.filterwarnings("ignore")
    res, ses = [], []
    for m in MODS:
        groups = list(session_files(m).items())
        with ThreadPoolExecutor(2) as ex:
            for i, ((sid, _), s) in enumerate(zip(groups, ex.map(lambda g: load_session(g[1]), groups))):
                d, info = run_session(sid, s, m)
                res.append(d)
                if info:
                    ses.append(info)
                print(m, i + 1, len(groups), sid, len(d), flush=True)
    pd.concat(res, ignore_index=True).to_parquet(OUT / "variants.pqt")
    pd.DataFrame(ses).to_parquet(OUT / "variants_sessions.pqt")


if __name__ == "__main__":
    main()
