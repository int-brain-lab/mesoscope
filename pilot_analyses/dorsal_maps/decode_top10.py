"""Decode the ten most selective units per mesoscope bin or BWM insertion.

Usage: python decode_top10.py [meso bwm_pial640 bwm]
Uses the same candidate pools, trial QC and outer folds as the v7 decoding maps.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from common import FEATURE_ROOT, MIN_NEURONS, bin_neurons
from decode_probe import probe_output
from decode_v2 import (BINARY, MIN_TRIALS, N_PICK, OUT_V2, REPORT_HEX_SIZE,
                       WHEEL, apply_coverage, excluded_sessions, ridge)
from gpu_decode import DEVICE, decode_binary, outer_train, outer_train_wheel
from make_bwm_depth import STREAMLINE_MODALITY
from selection import auc_selectivity, corr_selectivity, top_sets
from sessions import load_session


MODALITIES = ("meso", STREAMLINE_MODALITY, "bwm")


def top10_output(modality, *, legacy=False):
    suffix = "" if legacy else "_auc"
    return OUT_V2 / f"decoding_{modality}_top10{suffix}.pqt"


def selected_scores(s, pools, *, wheel=True):
    """Rank units inside each outer training fold, then score its held-out trials."""
    y_binary = (s["stim_left"], (s["choice"] == 1).astype(int), s["reward"])
    scores = {}
    for i, (name, y) in enumerate(zip(BINARY, y_binary)):
        if np.bincount(y, minlength=2).min() < 5:
            scores[name] = np.full(len(pools), np.nan)
            continue
        x = torch.as_tensor(s["X_task"][:, :, i].T, device=DEVICE, dtype=torch.float64)
        train = torch.as_tensor(outer_train(y), device=DEVICE)
        rank = auc_selectivity(x, y, train).cpu().numpy()
        scores[name] = decode_binary(x, top_sets(rank, pools), y, roc_auc=True)

    if not wheel:
        return scores
    f = s["X_wheel"].transpose(1, 2, 0)
    y = s["wheel_y"]
    nt, nb, n = f.shape
    x = torch.as_tensor(f.reshape(nt * nb, n), device=DEVICE, dtype=torch.float64)
    yw = torch.as_tensor(y.reshape(-1), device=DEVICE, dtype=torch.float64)
    train = torch.as_tensor(outer_train_wheel(nt, nb), device=DEVICE)
    rank = corr_selectivity(x, yw, train, absolute=False).cpu().numpy()
    valid = [j for j, pool in enumerate(pools)
             if (np.isfinite(rank[:, pool]).sum(axis=1) >= MIN_NEURONS).all()]
    selected = top_sets(np.nan_to_num(rank, nan=-np.inf), [pools[j] for j in valid])
    for target, mode in (("wheel", "speed"), ("wheel_trial", "residual")):
        scores[target] = np.full(len(pools), np.nan)
        if selected:
            scores[target][valid] = ridge.decode(f, selected, y, target=mode, **WHEEL)
    return scores


def run(modality, *, wheel=True):
    exclude = excluded_sessions() if modality == "meso" else set()
    probes = None if modality == "meso" else pd.read_parquet(probe_output(modality, legacy=not wheel))
    if probes is not None:
        probes = probes.drop_duplicates("pid").set_index("pid")
    rows = []
    for path in sorted((FEATURE_ROOT / modality).glob("*.npz")):
        if path.stem in exclude or (probes is not None and path.stem not in probes.index):
            continue
        s = apply_coverage(path.stem, load_session([path]), modality)
        if len(s["choice"]) < MIN_TRIALS:
            continue
        if probes is None:
            keys = bin_neurons(s["xyz"], size=REPORT_HEX_SIZE)
            bins, counts = np.unique(keys, return_counts=True)
            bins = bins[counts >= N_PICK]
            pools = [np.flatnonzero(keys == k) for k in bins]
            loc = [dict(session=path.stem, hex=k) for k in bins]
        else:
            p = probes.loc[path.stem]
            pools = [np.arange(len(s["xyz"]))]
            loc = [dict(pid=path.stem, session=p.session, entry_x=p.entry_x, entry_y=p.entry_y)]
        if not pools:
            continue
        scores = selected_scores(s, pools, wheel=wheel)
        rows.extend(dict(**place, n_neurons=len(pool), n_trials=len(s["choice"]), target=t, top10=v)
                    for t, vals in scores.items() for place, pool, v in zip(loc, pools, vals))
        print(modality, path.stem, len(pools), flush=True)
    if not rows:
        raise ValueError(f"No eligible cached features for {modality}")
    result = pd.DataFrame(rows)
    if wheel:
        result.to_parquet(top10_output(modality), index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modality", nargs="*", default=MODALITIES)
    for modality in parser.parse_args().modality:
        if modality not in MODALITIES:
            parser.error(f"unsupported modality: {modality}")
        run(modality)
