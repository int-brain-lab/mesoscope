#!/usr/bin/env python3
"""Per-insertion features for BWM probes that record isocortex.

For every BWM insertion (2023_12 release, `bwm_query`) with at least one
good isocortex unit in the clusters aggregate table: load good units
(`brainwidemap.load_good_units`), keep isocortex units, and save per-trial
firing rates in the stim / choice / feedback windows and wheel bins (see
common.WINDOWS / WHEEL_EDGES) plus trial labels and wheel speed to
FEATURE_ROOT/bwm/<pid>.npz. Spike files this run downloaded (spikes.*,
passingSpikes.table.pqt) are deleted
afterwards (disk space).
"""
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (FEATURE_ROOT, isocortex_mask, spike_window_rates, task_intervals,  # noqa: E402
                    trial_labels, trial_mask, wheel_speed)

OUTDIR = FEATURE_ROOT / "bwm"
CLUSTERS_PQT = FEATURE_ROOT.parent / "bwm_tables" / "clusters.pqt"


def cortical_insertions(one):
    """bwm_query rows whose pid has >= 1 good isocortex unit."""
    from brainwidemap import bwm_query
    from iblatlas.regions import BrainRegions
    q = bwm_query(one)
    df = pd.read_parquet(CLUSTERS_PQT, columns=["pid", "label", "atlas_id"])
    df = df[(df.label == 1) & df.pid.isin(q.pid)]
    df = df[isocortex_mask(df.atlas_id.values, BrainRegions())]
    return q[q.pid.isin(df.pid.unique())].reset_index(drop=True)


def extract(row):
    pid, eid = row["pid"], row["eid"]
    out = OUTDIR / f"{pid}.npz"
    if out.exists():
        return pid, "exists"
    from one.api import ONE
    from iblatlas.regions import BrainRegions
    from brainwidemap import load_good_units
    one = ONE()
    try:
        sess_path = one.eid2path(eid)
        before = set(sess_path.rglob("*pikes*")) if sess_path and sess_path.exists() else set()
        spikes, clusters = load_good_units(one, pid)
        keep = isocortex_mask(clusters["atlas_id"].values, BrainRegions())
        cids = clusters.index.values[keep]
        xyz = clusters[["x", "y", "z"]].values[keep] * 1e3  # m -> mm

        trials = one.load_object(eid, "trials", collection="alf")
        mask = trial_mask(trials)
        lab = trial_labels(trials, mask)
        iv = task_intervals(trials, mask, "bwm")
        wheel = one.load_object(eid, "wheel", collection="alf")
        wy = wheel_speed(wheel["timestamps"], wheel["position"], iv[:, 3:])

        X = spike_window_rates(spikes["times"], spikes["clusters"], cids, iv)  # (n, T, 9)
        np.savez_compressed(
            out, pid=pid, eid=eid, modality="bwm", xyz=xyz,
            atlas_id=clusters["atlas_id"].values[keep],
            X_task=X[:, :, :3].astype(np.float32), X_wheel=X[:, :, 3:].astype(np.float16),
            wheel_y=wy.astype(np.float32), **lab)
        del spikes
        for f in set(one.eid2path(eid).rglob("*pikes*")) - before:
            f.unlink()
        return pid, f"ok n={keep.sum()} trials={mask.sum()}"
    except Exception:
        return pid, "FAILED\n" + traceback.format_exc()


if __name__ == "__main__":
    from one.api import ONE
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = cortical_insertions(ONE()).to_dict("records")
    if len(sys.argv) > 1:
        rows = rows[: int(sys.argv[1])]
    print(len(rows), "insertions", flush=True)
    with ProcessPoolExecutor(3) as ex:
        for pid, status in ex.map(extract, rows):
            print(pid, status, flush=True)
