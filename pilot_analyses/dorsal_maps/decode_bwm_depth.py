"""Refit fixed-size VISa and VISam decoders by cortical layer for Figure 5.

Run python decode_bwm_depth.py to cache one draw-averaged row per eligible BWM
session and layer. Add --meso-reference to fit the pooled mesoscope scores.
"""
import argparse
import zlib
from functools import lru_cache

import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions

from common import bin_neurons, hex_center, hex_unkey
from decode_v2 import MIN_TRIALS, OUT_V2, REPORT_HEX_SIZE, TARGETS, apply_coverage, decode_sets, excluded_sessions
from sessions import load_session, session_files


N_UNITS = 10
N_DRAWS = 5
REGIONS = ("VISa", "VISam")
REGION_LABEL = " and ".join(REGIONS)
REGION_STEM = "_".join(REGIONS)
LAYERS = ("L2/3", "L4", "L5", "L6")
SCORES = OUT_V2 / f"report_v7_bwm_{REGION_STEM}_layer_scores.pqt"
MESO_SCORES = OUT_V2 / f"report_v7_meso_{REGION_STEM}_layer_reference.pqt"
VISP_LABEL = "VISp top 3"
VISP_SCORES = OUT_V2 / "report_v7_bwm_VISp_top3_layer_scores.pqt"
MESO_VISP_SCORES = OUT_V2 / "report_v7_meso_VISp_top3_layer_scores.pqt"
ALLEN_LAYERS = {
    f"{region}{suffix}": layer
    for region in (*REGIONS, "VISp")
    for suffix, layer in zip(("2/3", "4", "5", "6a", "6b"), (*LAYERS, "L6"))
}


@lru_cache(maxsize=1)
def visp_bins():
    """Rank majority-VISp hexagons using the matched Figure 2 stimulus scores."""
    from maps_v2 import load_matched_decoding

    scores = load_matched_decoding()[0]["meso"].query("target == 'stim'")
    regions = BrainRegions()
    files = session_files("meso")
    counts = []
    for sid, rows in scores.groupby("session"):
        with np.load(files[sid][0]) as z:
            keys = bin_neurons(z["xyz"], size=REPORT_HEX_SIZE)
            keep = np.isin(keys, rows.hex)
            is_visp = regions.id2acronym(z["atlas_id"][keep], mapping="Beryl") == "VISp"
        counts.append(pd.DataFrame({"hex": keys[keep], "visp": is_visp}))
    ranked = scores.groupby("hex")["all"].mean().rename("mean_auc").to_frame()
    ranked["visp_fraction"] = pd.concat(counts).groupby("hex").visp.mean()
    selected = ranked.query("visp_fraction > 0.5").sort_values("mean_auc", ascending=False,
                                                               kind="stable").head(3).copy()
    if len(selected) != 3:
        raise ValueError("Figure 2 does not contain three majority-VISp bins.")
    selected["ml_mm"], selected["ap_mm"] = hex_center(*hex_unkey(selected.index), size=REPORT_HEX_SIZE)
    return selected


def unit_layers(atlas_ids, regions, modality, *, visp=False):
    """Return measured BWM layers and the assumed mesoscope layer in the selected regions."""
    region_names = ("VISp",) if visp else REGIONS
    in_region = np.isin(regions.id2acronym(atlas_ids, mapping="Beryl"), region_names)
    if modality == "meso":
        return pd.Series(np.where(in_region, LAYERS[0], None))
    allen = pd.Series(regions.id2acronym(atlas_ids, mapping="Allen"))
    return allen.where(in_region).map(ALLEN_LAYERS)


def layer_units(modality, *, visp=False):
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
                layers = unit_layers(z["atlas_id"], regions, modality, visp=visp)
                if visp:
                    keys = bin_neurons(z["xyz"], size=REPORT_HEX_SIZE)
                    layers = layers.where(np.isin(keys, visp_bins().index))
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
        raise ValueError(f"No eligible {modality} features in {VISP_LABEL if visp else REGION_LABEL}.")
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


def regional_scores(modality, *, visp=False):
    groups, units = layer_units(modality, visp=visp)
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
    parser.add_argument("--visp", action="store_true", help="Use the three Figure 2 VISp hexagons.")
    args = parser.parse_args()
    modality, path = ("meso", MESO_SCORES) if args.meso_reference else ("bwm", SCORES)
    if args.visp:
        path = MESO_VISP_SCORES if args.meso_reference else VISP_SCORES
    scores = regional_scores(modality, visp=args.visp)
    OUT_V2.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pqt")
    scores.to_parquet(tmp, index=False)
    tmp.replace(path)
