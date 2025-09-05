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

def _candidate_fov_dir_paths(one: ONE, eid: str, fov_name: str, server_root: Path | None) -> list[Path]:
    """
    Return candidate directories that may contain the alf/FOV_XX files.
    Priority 1: server_root/Subjects/<sub>/<date>/<num>/alf/<FOV_XX>
    Fallback : one.cache_dir/Subjects/<sub>/<date>/<num>/alf/<FOV_XX>
    """
    subject, date_str, number = _session_components_from_eid(one, eid)
    candidates = []
    if server_root is not None:
        candidates.append(server_root / 'Subjects' / subject / date_str / number / 'alf' / fov_name)
    # fallback to the local ONE cache layout (many IBL caches mirror 'Subjects/...'):
    candidates.append(Path(one.cache_dir) / 'Subjects' / subject / date_str / number / 'alf' / fov_name)
    return candidates

# ----------------------------------------------
# Load per-session clusterUIDs aligned to neurons
# ----------------------------------------------

def get_cluster_uids_neuronal(one: ONE, eid: str, server_root: str | Path | None = None) -> np.ndarray:
    """
    Concatenate clusterUIDs across all FOVs for a given eid and
    filter to neuronal ROIs using mpciROITypes (bool mask), so that the
    resulting array aligns with rows in rr['roi_signal'] built by embed_meso().

    Returns
    -------
    uids_neuronal : (N_neurons,) array of dtype=object ('' if missing)
    """
    server_root = Path(server_root) if server_root is not None else None

    # Discover per-FOV collections exactly as in your embed_meso
    fov_cols = ONE().list_collections(eid, collection='alf/FOV_*')  # keep same order as your code

    out = []
    for fov_col in fov_cols:
        fov_name = Path(fov_col).name  # e.g., 'FOV_00'

        # Load mpciROITypes to get the neuron mask (same mask you use in embed_meso)
        dd = ONE().load_collection(eid, fov_col, object=['mpciROIs', 'mpciROITypes'])
        roitypes = dd['mpciROIs']['mpciROITypes'] if 'mpciROIs' in dd else dd['mpciROITypes']
        mask_neuron = roitypes.astype(bool)

        # Find the clusterUIDs.csv file across candidates
        uid_vec = None
        for cand_dir in _candidate_fov_dir_paths(one, eid, fov_name, server_root):
            csv_path = cand_dir / 'mpciROIs.clusterUIDs.csv'
            if csv_path.exists():
                try:
                    uid_vec = _read_uid_csv(csv_path)
                    break
                except Exception:
                    pass

        if uid_vec is None:
            # No CSV found for this FOV: fill with empty strings, length = total ROIs in this FOV
            uid_vec = np.full(roitypes.shape[0], '', dtype=object)

        # Filter to neuronal rows so indexing matches roi_signal stacking
        out.append(uid_vec[mask_neuron])

    if not out:
        return np.array([], dtype=object)

    return np.concatenate(out, axis=0).astype(object)

# --------------------------------------------------------
# Match tracked neurons between two or many sessions (eids)
# --------------------------------------------------------

def match_tracked_between_two(one: ONE, eid0: str, eid1: str, server_root: str | Path | None = None):
    """
    Identify same neurons between two sessions via clusterUIDs intersection.

    Returns a dict with:
        shared_uids : (K,) array of UIDs present in both
        idx0       : (K,) array of row indices in eid0's rr['roi_signal']
        idx1       : (K,) array of row indices in eid1's rr['roi_signal']
        mask0      : (N0,) boolean mask True for tracked in eid0
        mask1      : (N1,) boolean mask True for tracked in eid1
    """
    u0 = get_cluster_uids_neuronal(one, eid0, server_root=server_root)
    u1 = get_cluster_uids_neuronal(one, eid1, server_root=server_root)

    # Keep only non-empty UIDs
    v0 = (u0 != '')
    v1 = (u1 != '')
    shared = np.intersect1d(u0[v0], u1[v1])

    # Build index lists; handle rare duplicates defensively (map all occurrences)
    idx0_list, idx1_list = [], []
    for uid in shared:
        i0s = np.flatnonzero(u0 == uid)
        i1s = np.flatnonzero(u1 == uid)
        # In the usual case both are length-1; if not, create all pairs
        for i0 in i0s:
            for i1 in i1s:
                idx0_list.append(i0)
                idx1_list.append(i1)

    idx0 = np.array(idx0_list, dtype=int)
    idx1 = np.array(idx1_list, dtype=int)

    mask0 = np.isin(np.arange(u0.size), idx0)
    mask1 = np.isin(np.arange(u1.size), idx1)

    return dict(
        shared_uids=shared,
        idx0=idx0, idx1=idx1,
        mask0=mask0, mask1=mask1,
        uids0=u0, uids1=u1
    )

def match_tracked_across_many(one: ONE, anchor_eid: str, other_eids: list[str], server_root: str | Path | None = None):
    """
    Return UIDs and indices for ROIs in anchor_eid that are present in *all* other_eids.
    Mirrors MATLAB get_trackedROIs() semantics.

    Returns dict with:
        shared_uids  : UIDs present in anchor and all others
        is_tracked   : boolean mask over anchor ROIs
        iROIs        : indices (sorted by UID) into anchor ROIs
    """
    u_anchor = get_cluster_uids_neuronal(one, anchor_eid, server_root=server_root)
    shared = u_anchor[u_anchor != ''].copy()

    for e in other_eids:
        u_e = get_cluster_uids_neuronal(one, e, server_root=server_root)
        shared = np.intersect1d(shared, u_e[u_e != ''])

    is_tracked = np.isin(u_anchor, shared)
    # Sort indices by alphabetical order of UID (as in MATLAB)
    order = np.argsort(shared.astype(str))
    iROIs = np.array([np.flatnonzero(u_anchor == uid)[0] for uid in shared[order]], dtype=int)

    return dict(shared_uids=shared[order], is_tracked=is_tracked, iROIs=iROIs)

# ------------------------
# Example usage (two days)
# ------------------------

if __name__ == "__main__":
    one = ONE()
    # Example animal/session EIDs
    eid0 = '71e53fd1-38f2-49bb-93a1-3c826fbe7c13'   # day 0
    eid1 = 'PUT-EID-OF-LATER-SESSION-HERE'         # day 1

    # Point this to your server if the CSVs live only on "whiterussian"
    # On Windows: r"Y:\"
    # On Linux/macOS (mounted): "/mnt/whiterussian"
    server_root = r"Y:\\"  # or Path("/mnt/whiterussian")

    m = match_tracked_between_two(one, eid0, eid1, server_root=server_root)

    # Now align your rr dicts to tracked pairs
    rr0 = load_or_embed(eid0)  # your function
    rr1 = load_or_embed(eid1)

    # IMPORTANT: do NOT apply rsort before indexing; we index raw rows.
    sig0_tracked = rr0['roi_signal'][m['idx0'], :]
    sig1_tracked = rr1['roi_signal'][m['idx1'], :]
    # You can then compare longitudinal trajectories of the same neurons.