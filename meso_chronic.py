from __future__ import annotations
from pathlib import Path
import numpy as np
from one.api import ONE

# ---------------------------
# Helpers to read/write UIDs
# ---------------------------

def _read_uid_csv(p: Path) -> np.ndarray:
    """
    Robust one-column CSV reader that preserves empty lines as ''.
    Returns dtype=object 1D array of strings.
    """
    if not p.exists():
        raise FileNotFoundError(p)
    # Avoid np.loadtxt() since it drops empty fields.
    with p.open('r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n\r') for ln in f]
    # Normalize None to '' for safety
    return np.array([ln if ln is not None else '' for ln in lines], dtype=object)

def _session_components_from_eid(one: ONE, eid: str):
    """
    Extract (subject, date_str 'YYYY-MM-DD', number 'NNN') from Alyx for path building.
    """
    meta = one.alyx.rest('sessions', 'read', id=eid)
    subject = meta['subject']
    date_str = str(meta['start_time'])[:10]
    number = int(meta.get('number', 1))
    return subject, date_str, f'{number:03d}'

def _candidate_fov_dir_paths(
    one: ONE,
    eid: str,
    fov_name: str,
    roicat_root: Path | None,
    server_root: Path | None
) -> list[Path]:
    """
    Return candidate directories that may contain alf/FOV_XX files.

    Priority order:
      1) roicat_root/ROICaT/<subject>/<date>/<number>/alf/<FOV_XX>   (your local mirror)
      2) server_root/Subjects/<subject>/<date>/<number>/alf/<FOV_XX> (e.g., network share)
      3) one.cache_dir/Subjects/<subject>/<date>/<number>/alf/<FOV_XX> (ONE cache)
    """
    subject, date_str, number = _session_components_from_eid(one, eid)
    cands = []
    if roicat_root is not None:
        cands.append(Path(roicat_root) / 'ROICaT' / subject / date_str / number / 'alf' / fov_name)
    if server_root is not None:
        cands.append(Path(server_root) / 'Subjects' / subject / date_str / number / 'alf' / fov_name)
    cands.append(Path(one.cache_dir) / 'Subjects' / subject / date_str / number / 'alf' / fov_name)
    return cands

# ----------------------------------------------
# Load per-session clusterUIDs aligned to neurons
# ----------------------------------------------

def get_cluster_uids_neuronal(
    one: ONE,
    eid: str,
    roicat_root: str | Path | None = None,
    server_root: str | Path | None = None
) -> np.ndarray:
    roicat_root = Path(roicat_root) if roicat_root is not None else None
    server_root = Path(server_root) if server_root is not None else None

    # 1) Try listing FOV collections via ONE
    try:
        fov_cols = one.list_collections(eid, collection='alf/FOV_*')       
        # ensure deterministic order like the local glob path
        fov_cols = sorted(fov_cols, key=lambda s: Path(s).name)
    except Exception:
        fov_cols = []

    # 2) Fallback: enumerate FOV_* dirs locally if ONE didn’t yield anything
    if (not fov_cols) and (roicat_root is not None):
        subject, date_str, number = _session_components_from_eid(one, eid)
        local_alf = roicat_root / 'ROICaT' / subject / date_str / number / 'alf'
        if local_alf.is_dir():
            fov_cols = [f'alf/{p.name}' for p in sorted(local_alf.glob('FOV_*')) if p.is_dir()]

    out = []
    for fov_col in fov_cols:
        fov_name = Path(fov_col).name  # e.g., 'FOV_03'

        # Load ROITypes (from ONE cache), we need this to align and build neuron mask
        dd = one.load_collection(eid, fov_col, object=['mpciROIs', 'mpciROITypes'])
        roitypes = dd['mpciROIs']['mpciROITypes'] if 'mpciROIs' in dd else dd['mpciROITypes']
        mask_neuron = roitypes.astype(bool)
        n_rois = int(roitypes.shape[0])

        uid_vec = None
        for cand_dir in _candidate_fov_dir_paths(one, eid, fov_name, roicat_root, server_root):
            csv_path = cand_dir / 'mpciROIs.clusterUIDs.csv'
            if csv_path.exists():
                uid_vec = _read_uid_csv(csv_path)
                break

        if uid_vec is None:
            uid_vec = np.full(n_rois, '', dtype=object)
        else:
            # Defensive alignment
            if uid_vec.shape[0] < n_rois:
                pad = np.full(n_rois - uid_vec.shape[0], '', dtype=object)
                uid_vec = np.concatenate([uid_vec, pad], axis=0)
            elif uid_vec.shape[0] > n_rois:
                uid_vec = uid_vec[:n_rois]

        # normalize UIDs to stable strings (guards against stray spaces/None)
        uid_vec = np.array(
            [("" if (x is None) else str(x).strip()) for x in uid_vec],
            dtype=object
        )

        out.append(uid_vec[mask_neuron])

    if not out:
        return np.array([], dtype=object)

    return np.concatenate(out, axis=0).astype(object)


# --------------------------------------------------------
# Match tracked neurons between two or many sessions (eids)
# --------------------------------------------------------

def match_tracked_indices_across_sessions(
    one: ONE,
    anchor_eid: str,
    other_eids: list[str],
    roicat_root: str | Path | None = None,
    server_root: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Return a mapping {eid: indices} such that indexing rr['roi_signal'][indices, :]
    selects the same tracked neurons (same UIDs) in the same order across all sessions.

    The order is lexicographic by UID. If no common UIDs exist, each array is length 0.
    """

    def _uid_first_index_map(u: np.ndarray) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
        """Build UID -> absolute first index map for non-empty UIDs."""
        nz = (u != '')
        if not np.any(nz):
            return {}, np.empty(0, dtype=object), np.empty(0, dtype=int)
        u_nz = u[nz]
        # first occurrence among duplicates
        u_unique, idx_first = np.unique(u_nz, return_index=True)
        idx_abs = np.flatnonzero(nz)[idx_first]
        uid2abs = {uid: int(ix) for uid, ix in zip(u_unique, idx_abs)}
        return uid2abs, u_unique, idx_abs

    # 1) Collect UIDs
    u_anchor = get_cluster_uids_neuronal(one, anchor_eid, roicat_root=roicat_root, server_root=server_root)
    anchor_uid2abs, anchor_unique, _ = _uid_first_index_map(u_anchor)

    # Start with all non-empty anchor UIDs
    shared = anchor_unique.copy()

    # Intersect with each other session's non-empty UIDs
    per_session_uidmaps: dict[str, dict[str, int]] = {anchor_eid: anchor_uid2abs}
    for e in other_eids:
        u_e = get_cluster_uids_neuronal(one, e, roicat_root=roicat_root, server_root=server_root)
        uid2abs_e, u_e_unique, _ = _uid_first_index_map(u_e)
        shared = np.intersect1d(shared, u_e_unique)  # lexicographic, unique
        per_session_uidmaps[e] = uid2abs_e

    # Handle empty intersection early
    if shared.size == 0:
        out = {eid: np.empty(0, dtype=int) for eid in [anchor_eid, *other_eids]}
        return out

    # 2) Shared UIDs are already sorted lexicographically by np.intersect1d
    shared_uids = shared

    # 3) Build aligned indices per session (first occurrence per UID)
    out: dict[str, np.ndarray] = {}
    for eid, uid2abs in per_session_uidmaps.items():
        try:
            idx = np.array([uid2abs[uid] for uid in shared_uids], dtype=int)
        except KeyError as e:
            missing = str(e).strip("'")
            raise ValueError(f"Internal mismatch: shared UID not found in session {eid}: {missing}")
        out[eid] = idx

    return out


# ------------------------------
# Example usage (many days; min 2 eids)
# ------------------------------

# from meso import load_or_embed  # your existing function

# one = ONE()
# roicat_root = '/home/mic/chronic_csv'

# anchor = '0c60b2f3-455f-41d9-ac91-ebb51ec51de5'
# others = ['4778be48-3f5e-4802-9062-0046aced36df']

# idx_map = match_tracked_indices_across_sessions(
#     one, anchor, others, roicat_root=roicat_root, server_root=None
# )

# # Direct restriction:
# rr0 = load_or_embed(anchor, restrict=idx_map[anchor])
# rr1 = load_or_embed(others[0], restrict=idx_map[others[0]])


## get all eids for a subject:
# subject = 'SP058'
# eids, details = one.search(subject=subject, details=True)
## get all eids for which you have a chronic csv file locally
# 1) Find session dirs that contain at least one FOV_* with mpciROIs.clusterUIDs.csv
# def find_session_dirs_with_uids(roicat_root: str | Path) -> list[Path]:
#     root = Path(roicat_root) / 'ROICaT'
#     found = set()
#     # Matches: ROICaT/<subject>/<YYYY-MM-DD>/<NNN>/alf/FOV_XX/mpciROIs.clusterUIDs.csv
#     for csv in root.glob('*/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/*/alf/FOV_*/mpciROIs.clusterUIDs.csv'):
#         # session dir is .../<subject>/<date>/<number>
#         sess_dir = csv.parents[2]
#         found.add(sess_dir)
#     return sorted(found)

# # 2) Convert a session dir to the relative Alyx path expected by ONE.path2eid
# def session_dir_to_relpath(sess_dir: Path) -> str:
#     subject, date, number = sess_dir.parts[-3:]
#     return f'/{subject}/{date}/{number}/'

# # 3) Map session dirs -> EIDs (skip any that cannot be resolved)
# def eids_from_session_dirs(paths: Iterable[Path], one: ONE) -> dict[Path, str]:
#     mapping = {}
#     for p in paths:
#         rel = session_dir_to_relpath(Path(p))
#         try:
#             eid = str(one.path2eid(rel))
#             mapping[Path(p)] = eid
#         except Exception:
#             # not resolvable in Alyx/ONE; skip
#             continue
#     return mapping

# 
