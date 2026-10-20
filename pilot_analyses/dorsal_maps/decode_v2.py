#!/usr/bin/env python3
"""Revised decoding (v2) for the dorsal maps.

Changes from decode.py:
- 1 mm hexagons (centre-to-vertex 0.5 mm) are the default spatial bins;
- BWM is also decoded from units within 640 um of the pial surface
  (modality bwm_d640, make_bwm_depth.py), the depth range the mesoscope could
  plausibly see; bwm (all depths) is kept as a reference;
- wheel speed (wheel_decoding2): wheel targets with encoder glitches fixed
  (fix_wheel_glitches.py); ridge penalty chosen by exact leave-one-trial-out
  error instead of leave-one-row-out (rows of a trial are correlated); neural
  activity of the previous, same and next wheel bin as features (mesoscope
  activity follows wheel speed by 0.15-0.25 s, BWM spikes lead it by 0.1 s);
  two targets: "wheel" (wheel speed, as before) and "wheel_trial" (wheel speed
  minus its mean per bin over the training trials, i.e. trial-to-trial speed
  differences without the stereotyped time course after movement onset,
  which alone gives R^2 ~0.4-0.6 with no neural data).
Binary targets (stimulus side, choice, feedback) are decoded as before
(gpu_decode.decode_binary). Per (session, bin): all neurons ("all") and the
mean over N_DRAWS (10) draws of N_PICK (20) neurons ("n20"); bins need >= N_PICK
neurons of the session. Mesoscope trials outside the imaging (imaging_coverage.py)
are dropped, and sessions with fewer than half of their trials covered
(38438b7d: imaging 840-1840 s with a 333 s gap, task 192-1924 s; its covered
trials decode at chance) and sessions with task-imaging clock offsets
(CLOCK_OFFSET) are left out (maps_v2 and wheel_decoding2 filter them).

Usage: python decode_v2.py [meso bwm_d640 bwm] [--hex 0.5] [--fill-binary-all]
Writes output/v2/decoding_<modality>[_hex<size>].pqt and
output/v2/wheel_baseline_<modality>.pqt (per-session bin-only R^2).
"""
import sys
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import FEATURE_ROOT, OUT, bin_neurons
from gpu_decode import DEVICE, decode_binary
from sessions import load_session, session_files

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wheel_decoding2"))
import ridge  # noqa: E402

HEX_V2 = 0.5
N_PICK = 20       # neurons per random draw (and minimum neurons per session and bin)
N_DRAWS = 10
MIN_TRIALS = 50
WHEEL = dict(inner="loto", lags=(-1, 0, 1))
BINARY = ("stim", "choice", "feedback")
TARGETS = BINARY + ("wheel", "wheel_trial")
OUT_V2 = OUT / "v2"


def decode_sets(s, sets, binary=True):
    out = {}
    if binary:
        labels = {"stim": s["stim_left"], "choice": (s["choice"] == 1).astype(int), "feedback": s["reward"]}
        Xt = torch.as_tensor(s["X_task"], device=DEVICE, dtype=torch.float64)
        out = {t: decode_binary(Xt[:, :, j].T, sets, labels[t]) for j, t in enumerate(BINARY)}
    F = s["X_wheel"].astype(np.float64).transpose(1, 2, 0)
    out["wheel"] = ridge.decode(F, sets, s["wheel_y"], **WHEEL)
    out["wheel_trial"] = ridge.decode(F, sets, s["wheel_y"], target="residual", **WHEEL)
    return out


TRIAL_KEYS = ("X_task", "X_wheel")
MIN_COVERED = 0.5


def apply_coverage(sid, s, modality):
    """Drop mesoscope trials whose windows are not covered by imaging."""
    f = FEATURE_ROOT / "meso_trial_coverage" / f"{sid}.npy"
    if modality != "meso" or not f.exists():
        return s
    ok = np.load(f)
    if ok.all() or len(ok) != len(s["choice"]):
        return s
    if ok.mean() < MIN_COVERED:      # mostly outside the imaging: clock alignment suspect, drop the session
        ok[:] = False
    s = dict(s)
    for k, v in s.items():
        if k in TRIAL_KEYS:
            s[k] = v[:, ok]
        elif k not in ("xyz", "atlas_id") and np.ndim(v) and len(v) == len(ok):
            s[k] = v[ok]
    return s


# Mesoscope sessions whose task and imaging clocks disagree: the cross-correlation of
# population activity (FOV_00) with wheel speed peaks far from lag 0 (good sessions peak
# within +-0.4 s), and all-neuron decoding of every variable is at chance.
CLOCK_OFFSET = {"bd92d5ef-0f42-4643-a7d2-110a8843e557": -146.0,   # s, r 0.11 there vs -0.01 at 0
                "38438b7d-dc9d-4540-9b85-60571629b886": -337.6}   # 333 s gap in mpci.times


def excluded_sessions():
    """Mesoscope sessions left out: clock offsets (CLOCK_OFFSET) and sessions with fewer
    than MIN_COVERED of their trials inside the imaging."""
    d = FEATURE_ROOT / "meso_trial_coverage"
    low = {f.stem for f in d.glob("*.npy") if 0 < np.load(f).size and np.load(f).mean() < MIN_COVERED} \
        if d.exists() else set()
    return low | set(CLOCK_OFFSET)


def run_session(sid, s, hex_size, binary=True):
    if len(s["choice"]) < MIN_TRIALS:
        return pd.DataFrame()
    keys = bin_neurons(s["xyz"], size=hex_size)
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    bins, sets = [], []
    for k in np.unique(keys):
        idx = np.flatnonzero(keys == k)
        if idx.size < N_PICK:
            continue
        bins.append((int(k), idx.size))
        sets += [idx] + [rng.choice(idx, N_PICK, replace=False) for _ in range(N_DRAWS)]
    if not bins:
        return pd.DataFrame()
    sc = decode_sets(s, sets, binary)
    per = N_DRAWS + 1
    rows = []
    for b, (k, n) in enumerate(bins):
        for t in sc:
            v = sc[t][b * per:(b + 1) * per]
            rows.append(dict(session=sid, hex=k, n_neurons=n, n_trials=len(s["choice"]), target=t,
                             all=v[0], n20=np.nanmean(v[1:])))
    return pd.DataFrame(rows)


def out_name(modality, hex_size=HEX_V2):
    return OUT_V2 / f"decoding_{modality}{'' if hex_size == HEX_V2 else f'_hex{hex_size:g}'}.pqt"


def run(modality, hex_size=HEX_V2):
    warnings.filterwarnings("ignore")
    OUT_V2.mkdir(exist_ok=True)
    groups = list(session_files(modality).items())
    res, base = [], []
    with ThreadPoolExecutor(2) as pool:
        loaded = pool.map(lambda g: load_session(g[1]), groups)
        for i, ((sid, _), s) in enumerate(zip(groups, loaded)):
            s = apply_coverage(sid, s, modality)
            df = run_session(sid, s, hex_size)
            res.append(df)
            if len(s["choice"]) >= MIN_TRIALS:
                base.append(dict(session=sid, bin_baseline=ridge.bin_baseline(s["wheel_y"])))
            print(modality, i + 1, len(groups), sid, len(df), flush=True)
    out = pd.concat(res, ignore_index=True)
    out.to_parquet(out_name(modality, hex_size))
    pd.DataFrame(base).to_parquet(OUT_V2 / f"wheel_baseline_{modality}.pqt")
    return out


def fill_binary_all(modality, hex_size=HEX_V2):
    """Fill the all-neuron scores of the binary targets where they are missing
    (decode.py results reused from an n10-only run)."""
    warnings.filterwarnings("ignore")
    path = out_name(modality, hex_size)
    d = pd.read_parquet(path)
    miss = d.target.isin(BINARY) & d["all"].isna()
    todo = d[miss].session.unique()
    files = session_files(modality)
    for i, sid in enumerate(todo):
        s = load_session(files[sid])
        keys = bin_neurons(s["xyz"], size=hex_size)
        rows = d[miss & (d.session == sid)].drop_duplicates("hex")
        sets = [np.flatnonzero(keys == k) for k in rows.hex]
        labels = {"stim": s["stim_left"], "choice": (s["choice"] == 1).astype(int), "feedback": s["reward"]}
        Xt = torch.as_tensor(s["X_task"], device=DEVICE, dtype=torch.float64)
        for j, t in enumerate(BINARY):
            sc = decode_binary(Xt[:, :, j].T, sets, labels[t])
            for k, v in zip(rows.hex, sc):
                d.loc[(d.session == sid) & (d.hex == k) & (d.target == t), "all"] = v
        print(modality, "all-neuron binary", i + 1, len(todo), sid, flush=True)
        if (i + 1) % 10 == 0:
            d.to_parquet(path)                                   # checkpoint; reruns skip filled rows
    d.to_parquet(path)
    return d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("modality", nargs="*", default=["meso", "bwm_d640", "bwm"])
    ap.add_argument("--hex", type=float, default=HEX_V2, help="hexagon centre-to-vertex size, mm")
    ap.add_argument("--fill-binary-all", action="store_true",
                    help="only fill missing all-neuron binary scores of an existing output")
    a = ap.parse_args()
    for m in a.modality:
        if not a.fill_binary_all:
            run(m, a.hex)
        fill_binary_all(m, a.hex)
