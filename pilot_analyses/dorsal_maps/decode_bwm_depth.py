"""Refit fixed-size VISa and VISam decoders by cortical layer for Figure 5.

Run python decode_bwm_depth.py to cache one draw-averaged row per eligible BWM
session and layer. Add --meso-reference to fit the pooled mesoscope scores.
"""
import argparse
import zlib

import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions

from decode_v2 import MIN_TRIALS, OUT_V2, TARGETS, apply_coverage, decode_sets, excluded_sessions
from sessions import load_session, session_files


N_UNITS = 8
N_DRAWS = 5
REGIONS = ("VISa", "VISam")
REGION_LABEL = " and ".join(REGIONS)
REGION_STEM = "_".join(REGIONS)
LAYERS = ("L2/3", "L4", "L5", "L6")
SCORES = OUT_V2 / f"report_v7_bwm_{REGION_STEM}_layer_n{N_UNITS}_scores.pqt"
MESO_SCORES = OUT_V2 / f"report_v7_meso_{REGION_STEM}_layer_n{N_UNITS}_reference.pqt"
ALLEN_LAYERS = {
    f"{region}{suffix}": layer
    for region in REGIONS
    for suffix, layer in zip(("2/3", "4", "5", "6a", "6b"), (*LAYERS, "L6"))
}


def unit_layers(atlas_ids, regions, modality):
    """Return measured BWM layers and the assumed mesoscope layer in the selected regions."""
    in_region = np.isin(regions.id2acronym(atlas_ids, mapping="Beryl"), REGIONS)
    if modality == "meso":
        return pd.Series(np.where(in_region, LAYERS[0], None))
    allen = pd.Series(regions.id2acronym(atlas_ids, mapping="Allen"))
    return allen.where(in_region).map(ALLEN_LAYERS)


def layer_units(modality):
    """Find units and their layer labels in sessions with enough usable trials."""
    groups = session_files(modality)
    excluded = excluded_sessions() if modality == "meso" else set()
    regions = BrainRegions()
    rows = []
    for sid, paths in groups.items():
        if sid in excluded:
            continue
        session_units, n_trials, offset = [], set(), 0
        for path in paths:
            with np.load(path) as z:
                choice = apply_coverage(sid, {"choice": z["choice"]}, modality)["choice"]
                n_trials.add(len(choice))
                layers = unit_layers(z["atlas_id"], regions, modality)
                keep = np.flatnonzero(layers.notna())
                if keep.size:
                    session_units.append(pd.DataFrame({
                        "session": sid, "unit": offset + keep, "layer": layers.iloc[keep].to_numpy()
                    }))
                offset += len(layers)
        if len(n_trials) != 1:
            raise ValueError(f"Trial counts differ across probes in {sid}.")
        if n_trials.pop() >= MIN_TRIALS:
            rows.extend(session_units)
    if not rows:
        raise ValueError(f"No eligible {modality} features in {REGION_LABEL}.")
    return groups, pd.concat(rows, ignore_index=True)


def fit_session(sid, session, pools):
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    sets = [rng.choice(units, N_UNITS, replace=False)
            for units in pools.values() for _ in range(N_DRAWS)]
    scores = decode_sets(session, sets)
    averages = {target: np.asarray(scores[target]).reshape(len(pools), N_DRAWS).mean(axis=1)
                for target in TARGETS}
    return pd.DataFrame({
        "session": sid,
        "layer": list(pools),
        "n_available": [len(units) for units in pools.values()],
        "n_units": N_UNITS,
        "n_draws": N_DRAWS,
        **averages,
    })


def regional_scores(modality):
    groups, units = layer_units(modality)
    pools = units.groupby(["session", "layer"], sort=False).unit.agg(list)
    eligible = pools[pools.str.len().ge(N_UNITS)].groupby(level="session")
    scores = []
    for i, (sid, layers) in enumerate(eligible, start=1):
        session = apply_coverage(sid, load_session(groups[sid]), modality)
        scores.append(fit_session(sid, session, layers.droplevel("session").to_dict()))
        if i % 10 == 0 or i == len(eligible):
            print(f"{i}/{len(eligible)} {modality} sessions", flush=True)
    if not scores:
        raise ValueError(f"No eligible {modality} pools have {N_UNITS} units.")
    return pd.concat(scores, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meso-reference", action="store_true",
                        help="Fit the mesoscope reference using whole-session ROI pools.")
    args = parser.parse_args()
    modality, path = ("meso", MESO_SCORES) if args.meso_reference else ("bwm", SCORES)
    scores = regional_scores(modality)
    OUT_V2.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pqt")
    scores.to_parquet(tmp, index=False)
    tmp.replace(path)
