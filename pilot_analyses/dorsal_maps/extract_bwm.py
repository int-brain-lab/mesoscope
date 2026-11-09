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

`python extract_bwm.py w03` writes the control variant to
FEATURE_ROOT/bwm_w03/<pid>.npz: X_task / X_wheel are firing rates in the
mesoscope windows (0.3 s for stim / choice / feedback), and X_task_fr /
X_wheel_fr are "frame-sampled" features: spikes are binned into
FRAME_PERIOD frames, giving a rate trace at frame resolution, which is then
read out like a mesoscope trace (linear interpolation between frame
centres, mean over the window; common.interp_window_means).
"""
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (FEATURE_ROOT, interp_window_means, isocortex_mask,  # noqa: E402
                    spike_window_rates, task_intervals, trial_labels, trial_mask, wheel_speed)

OUTDIR = FEATURE_ROOT / "bwm"
VARIANT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else None
if VARIANT == "w03":
    OUTDIR = FEATURE_ROOT / "bwm_w03"
FRAME_PERIOD = 0.205  # s, the mesoscope frame period
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


def frame_sampled(spike_times, spike_clusters, cluster_ids, intervals):
    """Window means of a frame-resolution rate trace (see module docstring)."""
    if spike_times.size == 0 or len(cluster_ids) == 0:
        return np.zeros((len(cluster_ids), *intervals.shape[:-1]), np.float32)
    t0 = spike_times.min()
    n_frames = int(np.ceil((spike_times.max() - t0) / FRAME_PERIOD)) + 1
    lut = np.full(max(spike_clusters.max(), cluster_ids.max()) + 1, -1)
    lut[cluster_ids] = np.arange(len(cluster_ids))
    unit = lut[spike_clusters]
    sel = unit >= 0
    unit = unit[sel]
    frame = ((spike_times[sel] - t0) // FRAME_PERIOD).astype(np.int64)
    trace = np.bincount(unit * n_frames + frame, minlength=len(cluster_ids) * n_frames)
    trace = trace.reshape(len(cluster_ids), n_frames).astype(np.float32) / FRAME_PERIOD
    centres = t0 + (np.arange(n_frames) + 0.5) * FRAME_PERIOD
    return interp_window_means(trace, centres, np.zeros(len(cluster_ids)), intervals)


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
        # only this probe's folder: probes of one session may run in parallel workers
        probe_path = one.eid2path(eid) / "alf" / row["probe_name"]
        before = set(probe_path.rglob("*pikes*")) if probe_path.exists() else set()
        spikes, clusters = load_good_units(one, pid)
        keep = isocortex_mask(clusters["atlas_id"].values, BrainRegions())
        cids = clusters.index.values[keep]
        xyz = clusters[["x", "y", "z"]].values[keep] * 1e3  # m -> mm

        trials = one.load_object(eid, "trials", collection="alf")
        mask = trial_mask(trials)
        lab = trial_labels(trials, mask)
        iv = task_intervals(trials, mask, "meso" if VARIANT == "w03" else "bwm")
        wheel = one.load_object(eid, "wheel", collection="alf")
        wy = wheel_speed(wheel["timestamps"], wheel["position"], iv[:, 3:])

        X = spike_window_rates(spikes["times"], spikes["clusters"], cids, iv)  # (n, T, 9)
        extra = {}
        if VARIANT == "w03":
            Xf = frame_sampled(spikes["times"], spikes["clusters"], cids, iv)
            extra = dict(X_task_fr=Xf[:, :, :3].astype(np.float32),
                         X_wheel_fr=Xf[:, :, 3:].astype(np.float16))
        np.savez_compressed(
            out, pid=pid, eid=eid, modality="bwm", xyz=xyz,
            atlas_id=clusters["atlas_id"].values[keep],
            X_task=X[:, :, :3].astype(np.float32), X_wheel=X[:, :, 3:].astype(np.float16),
            wheel_y=wy.astype(np.float32), **lab, **extra)
        del spikes
        for f in set(probe_path.rglob("*pikes*")) - before:
            f.unlink()
        return pid, f"ok n={keep.sum()} trials={mask.sum()}"
    except Exception:
        return pid, "FAILED\n" + traceback.format_exc()


if __name__ == "__main__":
    from one.api import ONE
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = cortical_insertions(ONE()).to_dict("records")
    if sys.argv[-1].isdigit():
        rows = rows[: int(sys.argv[-1])]
    print(len(rows), "insertions", flush=True)
    with ProcessPoolExecutor(3) as ex:
        for pid, status in ex.map(extract, rows):
            print(pid, status, flush=True)
