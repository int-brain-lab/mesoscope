#!/usr/bin/env python3
"""Per-session features for the canonical mesoscope sessions.

For every canonical session: keep isocortex neurons and save per-trial
window means of the deconvolved trace (stim / choice / feedback windows and
the wheel bins, see common.WINDOWS / WHEEL_EDGES) plus trial labels and
wheel speed to FEATURE_ROOT/meso/<eid>.npz.

Memory: sessions are processed one FOV at a time, the signal file is
memory-mapped and read in chunks of ROI_CHUNK ROIs, so a worker never holds
a whole session (the stacked `meso_loader` arrays are ~1 GB per large
session). Dataset choice, ROI timing and neuron/region definitions follow
`meso_loader._load_fov`: deconvolved signal (else fluorescence), per-ROI
times = frame times + mpciStack.timeshift[stackPos], mpciROITypes neurons,
histology region ids (else the estimate).

Positions: histology-registered `mpciROIs.mlapdv` when every FOV has it,
else `mlapdv_estimate`.

Signal files this run downloaded are deleted after the session is done
(disk space); files already in the ONE cache are left alone.
"""
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from common import (FEATURE_ROOT, interp_window_means, isocortex_mask,  # noqa: E402
                    task_intervals, trial_labels, trial_mask, wheel_speed)
from meso_loader import _REGION_KEYS, _SIGNAL_DATASETS  # noqa: E402

OUTDIR = FEATURE_ROOT / "meso"
ROI_CHUNK = 256
N_WORKERS = 2


class SkipFOV(Exception):
    """An FOV whose datasets are missing or inconsistent; the session is
    kept without it."""


def task_collection(one, eid):
    """The alf/task_XX collection holding the choice-world trials (usually
    task_00; a few sessions have the passive protocol first)."""
    cols = sorted({d.rsplit("/", 1)[0] for d in one.list_datasets(eid) if "_ibl_trials." in d})
    if not cols:
        raise FileNotFoundError("no trials object")
    return cols[0]


def canonical_eids(one):
    paths = [s.strip() for s in (REPO / "canonical_sessions.txt").read_text().split(",") if s.strip()]
    return [str(one.path2eid(p)) for p in paths]


def frame_times_for(one, eid, fov, n_timepoints):
    """mpci.times, subsampled when it is an exact multiple of the signal
    length (multi-plane sessions; same rule as meso_loader._load_fov)."""
    t = one.load_dataset(eid, "mpci.times.npy", collection=fov)
    if len(t) != n_timepoints:
        ratio = len(t) / n_timepoints
        if not (ratio.is_integer() and ratio > 1):
            raise ValueError(f"{fov}: mpci.times has {len(t)} entries, signal {n_timepoints} "
                             f"(ratio {ratio:.3f})")
        t = t[:: int(ratio)]
    return t.astype(np.float64)


def fov_features(one, eid, fov, names, regions, iv, xyz_key, transform=None):
    """Features for the isocortex neurons of one FOV, in ROI chunks.
    transform: optional function applied to the whole (T, n_rois) signal of the
    FOV before feature extraction (e.g. a compression round trip).
    Returns (X (n, T, 9) float32, xyz (n, 3) um, region ids (n,), signal name)."""
    load = lambda d: one.load_dataset(eid, d, collection=fov)  # noqa: E731
    region_key = next((f"mpciROIs.{k}.npy" for k in _REGION_KEYS if f"mpciROIs.{k}.npy" in names), None)
    if region_key is None or xyz_key not in names:
        raise SkipFOV("no region / position dataset")
    sig_name = next((s for s in _SIGNAL_DATASETS if s in names), None)
    if sig_name is None:
        raise SkipFOV("no signal dataset")
    path = one.load_dataset(eid, sig_name, collection=fov, download_only=True)
    sig = np.load(path, mmap_mode="r")  # (T, n_rois)
    if transform is not None:
        sig = transform(np.asarray(sig, dtype=np.float32))

    region_ids = load(region_key)
    is_neuron = load("mpciROIs.mpciROITypes.npy").astype(bool)
    xyz = load(xyz_key)
    stack_pos = load("mpciROIs.stackPos.npy")
    timeshift = load("mpciStack.timeshift.npy")
    offsets = timeshift[stack_pos[:, len(timeshift.shape)]].astype(np.float64)
    frame_times = frame_times_for(one, eid, fov, sig.shape[0])
    if not len(region_ids) == len(is_neuron) == len(xyz) == sig.shape[1]:
        raise SkipFOV(f"ROI count mismatch (signal {sig.shape[1]}, regions {len(region_ids)}, "
                      f"types {len(is_neuron)}, xyz {len(xyz)})")

    keep = np.flatnonzero(is_neuron & isocortex_mask(region_ids, regions)
                          & np.all(np.isfinite(xyz), 1))
    X = np.empty((keep.size, iv.shape[0], iv.shape[1]), np.float32)
    for i0 in range(0, keep.size, ROI_CHUNK):
        idx = keep[i0:i0 + ROI_CHUNK]
        chunk = np.ascontiguousarray(sig[:, idx].T, dtype=np.float32)
        X[i0:i0 + ROI_CHUNK] = interp_window_means(chunk, frame_times, offsets[idx], iv)
    del sig
    return X, xyz[keep], region_ids[keep], sig_name


def extract(eid):
    out = OUTDIR / f"{eid}.npz"
    if out.exists():
        return eid, "exists"
    from one.api import ONE
    from iblatlas.regions import BrainRegions
    one = ONE()
    regions = BrainRegions()
    sess_path = Path(one.eid2path(eid))
    before = set(sess_path.rglob("mpci.ROIActivity*.npy"))
    try:
        task = task_collection(one, eid)
        trials = one.load_object(eid, "trials", collection=task)
        mask = trial_mask(trials)
        lab = trial_labels(trials, mask)
        iv = task_intervals(trials, mask, "meso")
        wheel = one.load_object(eid, "wheel", collection=task)
        wy = wheel_speed(wheel["timestamps"], wheel["position"], iv[:, 3:])

        fovs = sorted(one.list_collections(eid, collection="alf/FOV_*"), key=lambda c: int(c[-2:]))
        names = {f: {d.split("/")[-1] for d in one.list_datasets(eid, collection=f)} for f in fovs}
        xyz_key = ("mpciROIs.mlapdv.npy" if all("mpciROIs.mlapdv.npy" in n for n in names.values())
                   else "mpciROIs.mlapdv_estimate.npy")
        parts, skipped = [], []
        for f in fovs:
            try:
                parts.append(fov_features(one, eid, f, names[f], regions, iv, xyz_key))
            except SkipFOV as e:
                skipped.append(f"{f.split('/')[-1]}: {e}")
        if not parts:
            raise ValueError(f"no usable FOV ({skipped})")
        sig_types = {p[3] for p in parts}
        if len(sig_types) > 1:
            raise ValueError(f"mixed signal types across FOVs: {sig_types}")

        X = np.concatenate([p[0] for p in parts])
        np.savez_compressed(
            out, eid=eid, modality="meso", xyz=np.concatenate([p[1] for p in parts]) / 1e3,
            atlas_id=np.concatenate([p[2] for p in parts]),
            X_task=X[:, :, :3], X_wheel=X[:, :, 3:].astype(np.float16),
            wheel_y=wy.astype(np.float32), xyz_source=xyz_key.split(".")[1],
            signal=sig_types.pop(), task_collection=task, skipped_fovs=np.array(skipped, dtype=str),
            **lab)
        return eid, (f"ok n={len(X)} trials={mask.sum()} xyz={xyz_key.split('.')[1]}"
                     + (f" skipped {skipped}" if skipped else ""))
    except Exception:
        return eid, "FAILED\n" + traceback.format_exc()
    finally:
        for f in set(sess_path.rglob("mpci.ROIActivity*.npy")) - before:
            f.unlink(missing_ok=True)  # free disk: drop the signal files this run downloaded


if __name__ == "__main__":
    from one.api import ONE
    OUTDIR.mkdir(parents=True, exist_ok=True)
    eids = canonical_eids(ONE())
    if len(sys.argv) > 1:
        eids = eids[: int(sys.argv[1])]
    with ProcessPoolExecutor(N_WORKERS) as ex:
        for eid, status in ex.map(extract, eids):
            print(eid, status, flush=True)
