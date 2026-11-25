"""Refit fixed-size decoders for Figure 5.

Run python decode_bwm_depth.py to cache one draw-averaged row per eligible bin
and BWM session. Add --meso-reference to fit the depth-pooled mesoscope scores.
"""
import argparse
import zlib

import numpy as np
import pandas as pd

from decode_v2 import MIN_TRIALS, OUT_V2, TARGETS, apply_coverage, decode_sets, excluded_sessions
from make_bwm_depth import streamline_depth_percent
from sessions import load_session, session_files


DEPTH_EDGES = np.linspace(0, 100, 6)
N_UNITS = 10
N_DRAWS = 5
SCORES = OUT_V2 / "report_v7_bwm_depth_scores.pqt"
MESO_SCORES = OUT_V2 / "report_v7_meso_depth_reference.pqt"


def depth_bins(depth_percent):
    """Use left-closed bins with 100% in the last bin and -1 for invalid depths."""
    index = np.searchsorted(DEPTH_EDGES[1:-1], depth_percent, side="right")
    return np.where((depth_percent >= 0) & (depth_percent <= 100), index, -1)


def eligible_bins(groups):
    """Read positions and trial counts before loading the larger feature arrays."""
    positions = {}
    for sid, paths in groups.items():
        xyz, n_trials = [], set()
        for path in paths:
            with np.load(path) as z:
                xyz.append(z["xyz"])
                n_trials.add(len(z["choice"]))
        if len(n_trials) != 1:
            raise ValueError(f"Trial counts differ across probes in {sid}.")
        if n_trials.pop() >= MIN_TRIALS:
            positions[sid] = np.concatenate(xyz)
    if not positions:
        raise ValueError("No eligible BWM session features.")
    # The atlas loads its lookup table per call, so map all BWM units together.
    depths = streamline_depth_percent(np.concatenate(list(positions.values())))
    assignments = np.split(depth_bins(depths), np.cumsum([len(xyz) for xyz in positions.values()])[:-1])
    for sid, bins in zip(positions, assignments):
        members = [np.flatnonzero(bins == i) for i in range(len(DEPTH_EDGES) - 1)]
        eligible = {i: units for i, units in enumerate(members) if len(units) >= N_UNITS}
        if eligible:
            yield sid, eligible


def fit_session(sid, session, bins):
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    sets = [rng.choice(units, N_UNITS, replace=False)
            for units in bins.values() for _ in range(N_DRAWS)]
    scores = decode_sets(session, sets)
    averages = {target: np.asarray(scores[target]).reshape(len(bins), N_DRAWS).mean(axis=1)
                for target in TARGETS}
    return pd.DataFrame({
        "session": sid,
        "bin": list(bins),
        "n_available": [len(units) for units in bins.values()],
        "n_units": N_UNITS,
        "n_draws": N_DRAWS,
        **averages,
    })


def bwm_scores():
    groups = session_files("bwm")
    eligible = list(eligible_bins(groups))
    if not eligible:
        raise ValueError(f"No BWM depth bins have {N_UNITS} units.")
    scores = []
    for i, (sid, bins) in enumerate(eligible, start=1):
        scores.append(fit_session(sid, load_session(groups[sid]), bins))
        if i % 10 == 0 or i == len(eligible):
            print(f"{i}/{len(eligible)} BWM sessions", flush=True)
    return pd.concat(scores, ignore_index=True)


def meso_scores():
    excluded = excluded_sessions()
    groups = {sid: files for sid, files in session_files("meso").items() if sid not in excluded}
    scores = []
    for i, (sid, files) in enumerate(groups.items(), start=1):
        session = apply_coverage(sid, load_session(files), "meso")
        # The feature cache already contains only isocortex ROIs.
        units = np.arange(len(session["xyz"]))
        if len(session["choice"]) >= MIN_TRIALS and len(units) >= N_UNITS:
            scores.append(fit_session(sid, session, {0: units}).drop(columns="bin"))
        if i % 10 == 0 or i == len(groups):
            print(f"{i}/{len(groups)} mesoscope sessions", flush=True)
    if not scores:
        raise ValueError(f"No eligible mesoscope sessions with {N_UNITS} units.")
    return pd.concat(scores, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meso-reference", action="store_true",
                        help="Fit the mesoscope reference using whole-session ROI pools.")
    args = parser.parse_args()
    scores, path = (meso_scores(), MESO_SCORES) if args.meso_reference else (bwm_scores(), SCORES)
    OUT_V2.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pqt")
    scores.to_parquet(tmp, index=False)
    tmp.replace(path)
