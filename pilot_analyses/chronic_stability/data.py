"""Data for the cross-day stability analysis: ROICaT neuron IDs attached to
the per-trial window features that `dorsal_maps/extract_meso.py` already
saved for every canonical session.

Feature rows. `extract_meso.py` keeps, per FOV (in FOV order), the ROIs with
  mpciROITypes == neuron  &  Cosmos(region) == Isocortex  &  finite mlapdv
and saves their per-trial means in 9 windows (stim 0-0.3 s, pre-move
-0.3-0 s, feedback 0-0.3 s, six 0.2 s bins from -0.2 to 1.0 s around first
movement). It does not save ROI indices, so `row_uids` rebuilds the same mask
from the small per-FOV datasets and checks that the row count matches.

ROICaT IDs. One `mpciROIs.clusterUIDs.csv` per session and FOV on the
FlatIron ROICaT mirror, one line per ROI (all ROIs, not only neurons; an
empty line = ROI not assigned to a tracked cluster). Only subjects SP054,
SP058, SP061, SP066, SP067 and SP072 have them.
"""
import re
import sys
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pilot_analyses" / "dorsal_maps"))  # common, extract_meso
from common import FEATURE_ROOT, isocortex_mask  # noqa: E402
from meso_loader import _REGION_KEYS  # noqa: E402

ROICAT_URL = "https://ibl.flatironinstitute.org/resources/mesoscope/ROICaT/"
UID_ROOT = Path.home() / "chronic_csv" / "ROICaT"   # local mirror of the csv files
OUT = HERE / "output"
_HREF = re.compile(r'href="([^"/?][^"]*)/"')


def _auth(one):
    p = one.alyx._par
    return p.HTTP_DATA_SERVER_LOGIN, p.HTTP_DATA_SERVER_PWD


def _subdirs(url, auth):
    r = requests.get(url, auth=auth, timeout=30)
    r.raise_for_status()
    return sorted(set(_HREF.findall(r.text)))


def roicat_sessions(one):
    """{subject: ['YYYY-MM-DD/NNN', ...]} available on the ROICaT mirror."""
    auth, out = _auth(one), {}
    for subj in _subdirs(ROICAT_URL, auth):
        out[subj] = [f"{d}/{n}" for d in _subdirs(f"{ROICAT_URL}{subj}/", auth)
                     for n in _subdirs(f"{ROICAT_URL}{subj}/{d}/", auth)]
    return out


def canonical_paths():
    txt = (REPO / "canonical_sessions.txt").read_text()
    return [s.strip() for s in txt.split(",") if s.strip()]


def fetch_uids(one, session_path, fov):
    """ROICaT cluster UIDs (one per ROI, '' = untracked) for one FOV; cached
    under UID_ROOT. Returns None when the mirror has no file for the FOV."""
    local = UID_ROOT / session_path / "alf" / fov / "mpciROIs.clusterUIDs.csv"
    if not local.exists():
        r = requests.get(f"{ROICAT_URL}{session_path}/alf/{fov}/mpciROIs.clusterUIDs.csv",
                         auth=_auth(one), timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(r.text)
    lines = local.read_text().splitlines()
    return np.array([s.strip() for s in lines], dtype=object)


def _keep_mask(one, eid, col, names, xyz_key, regions):
    """The ROI selection of dorsal_maps/extract_meso.fov_features, and the region ids."""
    load = lambda d: one.load_dataset(eid, d, collection=col)  # noqa: E731
    rkey = next(f"mpciROIs.{k}.npy" for k in _REGION_KEYS if f"mpciROIs.{k}.npy" in names)
    rid = load(rkey)
    keep = (load("mpciROIs.mpciROITypes.npy").astype(bool) & isocortex_mask(rid, regions)
            & np.all(np.isfinite(load(xyz_key)), 1))
    return keep, rid


def extract_fresh(one, eid, session_path, regions):
    """Re-extract the dorsal_maps features from the deconvolved traces with the
    current histology (for sessions whose saved rows can no longer be mapped
    to ROIs because the histology files changed after the extraction).
    Signal files downloaded here are deleted afterwards."""
    from common import task_intervals, trial_labels, trial_mask
    from extract_meso import SkipFOV, fov_features, task_collection
    sess = Path(one.eid2path(eid))
    before = set(sess.rglob("mpci.ROIActivity*.npy"))
    try:
        task = task_collection(one, eid)
        trials = one.load_object(eid, "trials", collection=task)
        mask = trial_mask(trials)
        lab = trial_labels(trials, mask)
        iv = task_intervals(trials, mask, "meso")
        fovs = sorted(one.list_collections(eid, collection="alf/FOV_*"), key=lambda c: int(c[-2:]))
        names = {f: {d.split("/")[-1] for d in one.list_datasets(eid, collection=f)} for f in fovs}
        xyz_key = ("mpciROIs.mlapdv.npy" if all("mpciROIs.mlapdv.npy" in n for n in names.values())
                   else "mpciROIs.mlapdv_estimate.npy")
        X, xyz, rid, uids, fov_of = [], [], [], [], []
        for col in fovs:
            try:
                x, p, r, _ = fov_features(one, eid, col, names[col], regions, iv, xyz_key)
            except SkipFOV:
                continue
            keep, _ = _keep_mask(one, eid, col, names[col], xyz_key, regions)
            fov = col.split("/")[-1]
            u = fetch_uids(one, session_path, fov)
            u = np.full(keep.size, "", dtype=object) if u is None else u
            if u.size != keep.size or keep.sum() != len(x):
                raise ValueError(f"{session_path} {fov}: {u.size} UIDs, {keep.size} ROIs, "
                                 f"{keep.sum()} kept, {len(x)} feature rows")
            X.append(x), xyz.append(p / 1e3), rid.append(r), uids.append(u[keep])
            fov_of.append(np.full(len(x), int(fov[-2:])))
        return (np.concatenate(X), np.concatenate(xyz), np.concatenate(rid),
                np.concatenate(uids), np.concatenate(fov_of), lab)
    finally:
        for f in set(sess.rglob("mpci.ROIActivity*.npy")) - before:
            f.unlink(missing_ok=True)


def row_uids(one, eid, session_path, z, regions):
    """UID and FOV of every feature row of a dorsal_maps session file `z`.
    Raises if the rebuilt neuron mask does not reproduce the saved rows, or if
    a UID file length differs from the FOV's ROI count."""
    fovs = sorted(one.list_collections(eid, collection="alf/FOV_*"), key=lambda c: int(c[-2:]))
    skipped = {s.split(":")[0] for s in z["skipped_fovs"]} if "skipped_fovs" in z.files else set()
    xyz_key = f"mpciROIs.{z['xyz_source']}.npy"
    uids, fov_of, atlas = [], [], []
    for col in fovs:
        fov = col.split("/")[-1]
        if fov in skipped:
            continue
        names = {d.split("/")[-1] for d in one.list_datasets(eid, collection=col)}
        load = lambda d: one.load_dataset(eid, d, collection=col)  # noqa: E731
        keep, rid = _keep_mask(one, eid, col, names, xyz_key, regions)
        atlas.append(rid[keep])
        u = fetch_uids(one, session_path, fov)
        if u is None:
            u = np.full(keep.size, "", dtype=object)
        if u.size != keep.size:
            raise ValueError(f"{session_path} {fov}: {u.size} UIDs for {keep.size} ROIs")
        uids.append(u[keep])
        fov_of.append(np.full(keep.sum(), int(fov[-2:])))
    uids, fov_of = np.concatenate(uids), np.concatenate(fov_of)
    if uids.size != z["X_task"].shape[0]:
        raise ValueError(f"{session_path}: rebuilt mask has {uids.size} rows, features have "
                         f"{z['X_task'].shape[0]}")
    if not np.array_equal(np.concatenate(atlas), z["atlas_id"]):
        raise ValueError(f"{session_path}: region ids of the rebuilt rows differ from the saved ones")
    return uids, fov_of


def build(one=None):
    """Per tracked session: features + UIDs, saved to OUT/sessions/<subject>/<date>_<num>.npz."""
    from iblatlas.regions import BrainRegions
    from one.api import ONE
    one = one or ONE()
    regions = BrainRegions()
    roicat = roicat_sessions(one)
    for path in canonical_paths():
        subj, rest = path.split("/", 1)
        if rest not in roicat.get(subj, []):
            continue
        out = OUT / "sessions" / subj / f"{rest.replace('/', '_')}.npz"
        if out.exists():
            continue
        eid = str(one.path2eid(path))
        f = FEATURE_ROOT / "meso" / f"{eid}.npz"
        if not f.exists():
            print(path, "no dorsal_maps features")
            continue
        z = np.load(f, allow_pickle=True)
        try:
            uids, fov = row_uids(one, eid, path, z, regions)
            X = np.concatenate([z["X_task"], z["X_wheel"].astype(np.float32)], axis=2)
            xyz, rid, lab, src = z["xyz"], z["atlas_id"], z, "cached"
        except ValueError as e:
            print(path, "cached rows unusable:", e, "-> re-extracting", flush=True)
            try:
                X, xyz, rid, uids, fov, lab = extract_fresh(one, eid, path, regions)
                src = "fresh"
            except Exception as e2:  # noqa: BLE001
                print(path, "SKIP", type(e2).__name__, e2)
                continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, eid=eid, path=path, uid=uids.astype(str), fov=fov, xyz=xyz,
                            atlas_id=rid, X=X.astype(np.float32), source=src,
                            **{k: lab[k] for k in ("choice", "reward", "stim_left", "c_L", "c_R",
                                                   "block")})
        print(path, src, f"n={uids.size} tracked={np.mean(uids != ''):.2f} trials={X.shape[1]}", flush=True)


if __name__ == "__main__":
    build()
