"""Refit BWM decoders for fixed-size populations in five relative depth bins.

Run python decode_bwm_depth.py to cache one draw-averaged row per eligible bin
and session. depth_profile.py uses those scores to plot Figure 5.
"""
import zlib

import numpy as np
import pandas as pd

from decode_v2 import MIN_TRIALS, OUT_V2, TARGETS, decode_sets
from make_bwm_depth import streamline_depth_percent
from sessions import load_session, session_files


DEPTH_EDGES = np.linspace(0, 100, 6)
N_UNITS = 10
N_DRAWS = 5
SCORES = OUT_V2 / "report_v7_bwm_depth_scores.pqt"


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


def main():
    groups = session_files("bwm")
    eligible = list(eligible_bins(groups))
    if not eligible:
        raise ValueError(f"No BWM depth bins have {N_UNITS} units.")
    scores = []
    for i, (sid, bins) in enumerate(eligible, start=1):
        scores.append(fit_session(sid, load_session(groups[sid]), bins))
        if i % 10 == 0 or i == len(eligible):
            print(f"{i}/{len(eligible)} BWM sessions", flush=True)
    OUT_V2.mkdir(parents=True, exist_ok=True)
    tmp = SCORES.with_suffix(".tmp.pqt")
    pd.concat(scores, ignore_index=True).to_parquet(tmp, index=False)
    tmp.replace(SCORES)


if __name__ == "__main__":
    main()
