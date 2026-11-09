#!/usr/bin/env python3
"""Decode stim side / choice / feedback / wheel speed from the neurons of
each (session, hex bin) with >= MIN_NEURONS neurons, for both modalities.

Two versions per (session, bin):
- `all`: every neuron in the bin (what was asked for),
- `n10`: mean over N_DRAWS random draws of MIN_NEURONS neurons, so meso
  bins (often hundreds of neurons) and BWM bins (tens) are compared at
  the same population size.
Binary variables: balanced accuracy (chance 0.5) of L2 logistic regression;
wheel: R^2 of ridge regression on wheel speed in 0.2 s bins.

Runs on the GPU (gpu_decode; identical to the scikit-learn reference
decoders in common.py run at tight solver tolerance): all bins and random
draws of a session are decoded together in batched GPU calls, one session
at a time, with the next sessions' feature files loaded in background
threads. Sessions with fewer than MIN_TRIALS valid trials are skipped.

Writes output/decoding_<modality>.pqt.
"""
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from common import HEX_SIZE, MIN_NEURONS, OUT, bin_neurons
from gpu_decode import DEVICE, decode_binary, decode_wheel
from sessions import load_session, session_files

N_DRAWS = 3
MIN_TRIALS = 50
TARGETS = ("stim", "choice", "feedback", "wheel")


def decode_sets(s, sets):
    """Scores of every neuron set (index arrays into the session's neurons)
    for each target, decoded together on the GPU -> {target: (len(sets),)}."""
    Xt = torch.as_tensor(s["X_task"], device=DEVICE)  # (n, T, 3)
    return {"stim": decode_binary(Xt[:, :, 0].T, sets, s["stim_left"]),
            "choice": decode_binary(Xt[:, :, 1].T, sets, (s["choice"] == 1).astype(int)),
            "feedback": decode_binary(Xt[:, :, 2].T, sets, s["reward"]),
            "wheel": decode_wheel(torch.as_tensor(s["X_wheel"], device=DEVICE).permute(1, 2, 0),
                                  sets, s["wheel_y"])}


def run_session(sid, s, hex_size=HEX_SIZE, n10_only=False):
    if len(s["choice"]) < MIN_TRIALS:
        return pd.DataFrame()
    keys = bin_neurons(s["xyz"], size=hex_size)
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    bins, sets = [], []  # per bin: the full set (unless n10_only), then N_DRAWS random subsets
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        if idx.size < MIN_NEURONS:
            continue
        bins.append((int(k), idx.size))
        sets += ([] if n10_only else [idx]) + [rng.choice(idx, MIN_NEURONS, replace=False)
                                               for _ in range(N_DRAWS)]
    if not bins:
        return pd.DataFrame()
    sc = decode_sets(s, sets)
    per = N_DRAWS + (0 if n10_only else 1)
    rows = []
    for b, (k, n) in enumerate(bins):
        for t in TARGETS:
            v = sc[t][b * per:(b + 1) * per]
            rows.append(dict(session=sid, hex=k, n_neurons=n, n_trials=len(s["choice"]),
                             target=t, all=np.nan if n10_only else v[0],
                             n10=np.nanmean(v[-N_DRAWS:])))
    return pd.DataFrame(rows)


def out_name(modality, features="", hex_size=HEX_SIZE):
    tag = modality + features + ("" if hex_size == HEX_SIZE else f"_hex{hex_size:g}")
    return OUT / f"decoding_{tag}.pqt"


def run(modality, features="", hex_size=HEX_SIZE, n10_only=False):
    """features: suffix of the feature arrays to use (e.g. "_fr" for the
    frame-sampled BWM control)."""
    warnings.filterwarnings("ignore")
    groups = list(session_files(modality).items())
    res = []
    with ThreadPoolExecutor(2) as pool:  # load ahead while the GPU decodes
        loaded = pool.map(lambda g: load_session(g[1], features), groups)
        for i, ((sid, _), s) in enumerate(zip(groups, loaded)):
            df = run_session(sid, s, hex_size, n10_only)
            res.append(df)
            print(modality, features, i + 1, len(groups), sid, len(df), flush=True)
    out = pd.concat(res, ignore_index=True)
    out.to_parquet(out_name(modality, features, hex_size))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("modality", nargs="*", default=["meso", "bwm"],
                    help="meso, bwm, or bwm_w03 (BWM with the mesoscope windows)")
    ap.add_argument("--features", default="", help='feature suffix, e.g. "_fr"')
    ap.add_argument("--hex", type=float, default=HEX_SIZE, help="hexagon centre-to-vertex size, mm")
    ap.add_argument("--n10-only", action="store_true", help="skip the all-neuron decoders")
    a = ap.parse_args()
    for m in a.modality:
        run(m, a.features, a.hex, a.n10_only)
