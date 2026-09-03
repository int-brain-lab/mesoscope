"""Clean, minimal-dependency loader for basic IBL mesoscope session data.

This module has one job: given an experiment ID (eid), return the ROI
signal, per-ROI corrected timestamps, brain-region assignment, and 3D
location for every field of view (FOV) in that session, stacked into a
single array per quantity.

It intentionally does *not* pull in matplotlib or rastermap — for
Rastermap-sorted rasters and plotting utilities see `meso.py`; for
cross-day tracked-neuron alignment see `meso_chronic.py`.

Default settings (see README for the reasoning):
- neuron-only ROIs (`filter_neurons=True`)
- deconvolved spike-rate signal, falling back to raw fluorescence
- histology-registered brain regions, falling back to the pipeline estimate
- on-disk caching under `<ONE.cache_dir>/meso/basic_data/`
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from one.api import ONE
from iblatlas.atlas import AllenAtlas

# ALF attribute names for brain-region assignment, in preference order:
# final histology-registered CCF ids first, falling back to the
# registration-pipeline estimate when histology isn't available yet.
_REGION_KEYS = ("brainLocationIds_ccf_2017", "brainLocationIds_ccf_2017_estimate")

# Fixed, small set of datasets this loader actually needs, per FOV. Deliberately
# NOT using one.load_object(eid, 'mpci'/'mpciROIs', ...): those namespaces also
# contain mpci.ROIActivityF.npy, mpci.ROINeuropilActivityF.npy,
# mpciROIs.masks.sparse_npz, mpciROIs.neuropilMasks.sparse_npz etc. -- each
# several hundred MB per FOV -- which would multiply download size and time for
# no benefit when we only need one signal array.
_REQUIRED_DATASETS = (
    "mpci.times.npy",
    "mpciStack.timeshift.npy",
    "mpciROIs.stackPos.npy",
    "mpciROIs.mpciROITypes.npy",
    "mpciROIs.mlapdv_estimate.npy",
)
_SIGNAL_DATASETS = ("mpci.ROIActivityDeconvolved.npy", "mpci.ROIActivityF.npy")


@dataclass
class MesoscopeSession:
    """Stacked, ready-to-use data for one mesoscope session (all FOVs)."""

    eid: str
    roi_signal: np.ndarray       # (n_rois, n_timepoints) float32
    roi_times: np.ndarray        # (n_rois, n_timepoints) float32, per-ROI corrected
    region_ids: np.ndarray       # (n_rois,) Allen CCF structure ids
    region_labels: np.ndarray    # (n_rois,) Allen acronyms
    xyz: np.ndarray              # (n_rois, 3) mm, MLAPDV estimate
    fov: np.ndarray              # (n_rois,) originating FOV, e.g. 'FOV_00'
    signal_type: str             # 'deconvolved' or 'fluorescence'
    filtered_to_neurons: bool


def _cache_path(one: ONE, eid: str, filter_neurons: bool) -> Path:
    d = Path(one.cache_dir, "meso", "basic_data")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{eid}_filter_{filter_neurons}.npz"


def _load_fov(eid: str, fov: str, one: ONE, atlas: AllenAtlas) -> dict:
    """Load and lightly process the ROI data for a single FOV.

    Fetches only the specific datasets this loader needs (see
    `_REQUIRED_DATASETS` / `_SIGNAL_DATASETS` / `_REGION_KEYS`), not whole
    ALF objects, to keep downloads small and predictable.
    """
    available = {d.split("/")[-1] for d in one.list_datasets(eid, collection=fov)}

    missing = [d for d in _REQUIRED_DATASETS if d not in available]
    if missing:
        raise FileNotFoundError(f"{fov} for {eid} is missing required dataset(s): {missing}")

    def _load(dataset: str):
        return one.load_dataset(eid, dataset, collection=fov)

    signal_type = next((s for s in _SIGNAL_DATASETS if s in available), None)
    if signal_type is None:
        raise FileNotFoundError(f"{fov} for {eid} has neither of {_SIGNAL_DATASETS}")
    roi_signal = _load(signal_type).T
    signal_type = "deconvolved" if "Deconvolved" in signal_type else "fluorescence"

    region_dataset = next((f"mpciROIs.{k}.npy" for k in _REGION_KEYS if f"mpciROIs.{k}.npy" in available), None)
    if region_dataset is None:
        raise FileNotFoundError(f"{fov} for {eid} has no brain-region dataset (tried {_REGION_KEYS})")
    region_ids = _load(region_dataset)
    region_labels = atlas.regions.id2acronym(region_ids)

    # Per-ROI time correction: mesoscope FOVs are scanned with a small
    # sub-frame offset per ROI (line/plane-dependent). This matches the
    # timeshift-indexing convention already used in meso.py.
    frame_times = _load("mpci.times.npy")
    roi_stack_pos = _load("mpciROIs.stackPos.npy")
    timeshift = _load("mpciStack.timeshift.npy")
    roi_offsets = timeshift[roi_stack_pos[:, len(timeshift.shape)]]
    roi_times = np.tile(frame_times, (roi_offsets.size, 1)) + roi_offsets[:, None]

    is_neuron = _load("mpciROIs.mpciROITypes.npy").astype(bool)
    xyz = _load("mpciROIs.mlapdv_estimate.npy")

    n = roi_signal.shape[0]
    assert region_ids.shape[0] == n and xyz.shape[0] == n and is_neuron.shape[0] == n, (
        f"ROI count mismatch in {fov} for {eid}: signal={n}, "
        f"regions={region_ids.shape[0]}, xyz={xyz.shape[0]}, mask={is_neuron.shape[0]}"
    )

    return dict(
        roi_signal=roi_signal.astype(np.float32, copy=False),
        roi_times=roi_times.astype(np.float32, copy=False),
        region_ids=region_ids,
        region_labels=region_labels,
        xyz=xyz,
        is_neuron=is_neuron,
        signal_type=signal_type,
    )


def load_mesoscope_session(
    eid: str,
    one: Optional[ONE] = None,
    filter_neurons: bool = True,
    use_cache: bool = True,
    rerun: bool = False,
) -> MesoscopeSession:
    """Load and stack all FOVs of a mesoscope session.

    Parameters
    ----------
    eid : str
        Experiment/session id.
    one : ONE, optional
        An authenticated ONE instance. A default one is created if omitted.
    filter_neurons : bool, default True
        Keep only ROIs classified as neurons (`mpciROITypes`), dropping
        non-neuronal / artifact ROIs.
    use_cache : bool, default True
        Read/write the on-disk cache at `<ONE.cache_dir>/meso/basic_data/`.
    rerun : bool, default False
        Force a fresh load even if a cache file exists.

    Returns
    -------
    MesoscopeSession
    """
    one = one if one is not None else ONE()
    cache_file = _cache_path(one, eid, filter_neurons)

    if use_cache and cache_file.exists() and not rerun:
        with np.load(cache_file, allow_pickle=True) as z:
            return MesoscopeSession(
                eid=eid,
                roi_signal=z["roi_signal"],
                roi_times=z["roi_times"],
                region_ids=z["region_ids"],
                region_labels=z["region_labels"],
                xyz=z["xyz"],
                fov=z["fov"],
                signal_type=str(z["signal_type"]),
                filtered_to_neurons=bool(z["filtered_to_neurons"]),
            )

    atlas = AllenAtlas()
    fov_collections = sorted(
        one.list_collections(eid, collection="alf/FOV_*"),
        key=lambda s: int(s[-2:]),
    )
    if not fov_collections:
        raise ValueError(f"No FOV collections found for session {eid}")

    signal_chunks, times_chunks = [], []
    region_id_chunks, region_label_chunks, xyz_chunks, fov_chunks = [], [], [], []
    signal_types = set()

    for fov in fov_collections:
        fov_name = fov.split("/")[-1]
        d = _load_fov(eid, fov, one, atlas)
        signal_types.add(d["signal_type"])

        mask = d["is_neuron"] if filter_neurons else np.ones_like(d["is_neuron"], dtype=bool)

        signal_chunks.append(d["roi_signal"][mask])
        times_chunks.append(d["roi_times"][mask])
        region_id_chunks.append(d["region_ids"][mask])
        region_label_chunks.append(d["region_labels"][mask])
        xyz_chunks.append(d["xyz"][mask])
        fov_chunks.append(np.full(int(mask.sum()), fov_name))

    if len(signal_types) > 1:
        raise ValueError(
            f"Mixed signal types across FOVs for {eid}: {signal_types}. "
            "Some FOVs have deconvolved traces and others don't; re-extract "
            "consistently before using this loader."
        )

    session = MesoscopeSession(
        eid=eid,
        roi_signal=np.vstack(signal_chunks),
        roi_times=np.vstack(times_chunks),
        region_ids=np.hstack(region_id_chunks),
        region_labels=np.hstack(region_label_chunks),
        xyz=np.vstack(xyz_chunks),
        fov=np.hstack(fov_chunks),
        signal_type=signal_types.pop(),
        filtered_to_neurons=filter_neurons,
    )

    if use_cache:
        np.savez(
            cache_file,
            roi_signal=session.roi_signal,
            roi_times=session.roi_times,
            region_ids=session.region_ids,
            region_labels=session.region_labels,
            xyz=session.xyz,
            fov=session.fov,
            signal_type=session.signal_type,
            filtered_to_neurons=session.filtered_to_neurons,
        )

    return session
