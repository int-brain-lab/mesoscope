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

        out.append(uid_vec[mask_neuron])

    if not out:
        return np.array([], dtype=object)

    return np.concatenate(out, axis=0).astype(object)
# --------------------------------------------------------
# Match tracked neurons between two or many sessions (eids)
# --------------------------------------------------------

def match_tracked_between_two(
    one: ONE,
    eid0: str,
    eid1: str,
    roicat_root: str | Path | None = None,
    server_root: str | Path | None = None
):
    """
    Strict 1-1 mapping between sessions:
      - deduplicate per-session UID->index maps (first occurrence kept),
      - intersect UIDs,
      - return aligned indices sorted by UID.
    """
    u0 = get_cluster_uids_neuronal(one, eid0, roicat_root=roicat_root, server_root=server_root)
    u1 = get_cluster_uids_neuronal(one, eid1, roicat_root=roicat_root, server_root=server_root)

    # Keep only non-empty UIDs
    nz0 = (u0 != '')
    nz1 = (u1 != '')

    # Build UID->first_index maps (deduplicate inside each session)
    # Using return_index ensures first occurrence is kept with stable order
    u0_unique, idx0_first = np.unique(u0[nz0], return_index=True)
    u1_unique, idx1_first = np.unique(u1[nz1], return_index=True)

    # Intersect and sort lexicographically (np.intersect1d returns sorted)
    shared = np.intersect1d(u0_unique, u1_unique)

    # Map back to absolute row indices in the full u0/u1 arrays
    # (idx*_first are relative to the compressed nz* arrays)
    # Compute positions of shared in the unique arrays:
    pos0 = np.searchsorted(u0_unique, shared)
    pos1 = np.searchsorted(u1_unique, shared)

    # Convert to indices into nz* arrays, then to absolute indices
    idx0_nz = idx0_first[pos0]
    idx1_nz = idx1_first[pos1]
    idx0 = np.flatnonzero(nz0)[idx0_nz]
    idx1 = np.flatnonzero(nz1)[idx1_nz]

    # Boolean masks over all neurons (True if tracked)
    mask0 = np.zeros(u0.size, dtype=bool); mask0[idx0] = True
    mask1 = np.zeros(u1.size, dtype=bool); mask1[idx1] = True

    return dict(
        shared_uids=shared,
        idx0=idx0, idx1=idx1,
        mask0=mask0, mask1=mask1,
        uids0=u0, uids1=u1
    )


def match_tracked_across_many(
    one: ONE,
    anchor_eid: str,
    other_eids: list[str],
    roicat_root: str | Path | None = None,
    server_root: str | Path | None = None
):
    u_anchor = get_cluster_uids_neuronal(one, anchor_eid, roicat_root=roicat_root, server_root=server_root)
    shared = u_anchor[u_anchor != ''].copy()

    for e in other_eids:
        u_e = get_cluster_uids_neuronal(one, e, roicat_root=roicat_root, server_root=server_root)
        shared = np.intersect1d(shared, u_e[u_e != ''])

    is_tracked = np.isin(u_anchor, shared)
    order = np.argsort(shared.astype(str))
    iROIs = np.array([np.flatnonzero(u_anchor == uid)[0] for uid in shared[order]], dtype=int)

    return dict(shared_uids=shared[order], is_tracked=is_tracked, iROIs=iROIs)

# ------------------------
# Example usage (two days)
# ------------------------

# from one.api import ONE
# from meso import load_or_embed  # your existing function  :contentReference[oaicite:1]{index=1}
# from meso_chronic import match_tracked_between_two  # with the patch above  :contentReference[oaicite:2]{index=2}
#  roicat_root='/home/mic/chronic_csv'
# one = ONE()
# eid0 = '0c60b2f3-455f-41d9-ac91-ebb51ec51de5'   # earlier session of SP058
# eid1 = '4778be48-3f5e-4802-9062-0046aced36df'   # subsequent session of SP058

# m = match_tracked_between_two(one, eid0, eid1, roicat_root='/home/mic/chronic_csv')  # '.' contains ROICaT/

# rr0 = load_or_embed(eid0)   # do NOT sort rows before indexing
# rr1 = load_or_embed(eid1)

# sig0_tracked = rr0['roi_signal'][m['idx0'], :]
# sig1_tracked = rr1['roi_signal'][m['idx1'], :]
