from __future__ import annotations
from pathlib import Path
import numpy as np
from one.api import ONE
from itertools import combinations
from collections import defaultdict
import concurrent.futures as cf
from typing import Iterable, List, Tuple, Optional, Dict
from datetime import date, timedelta
import re, os, sys, threading, requests
from urllib.parse import urljoin
import math
import matplotlib.pyplot as plt
from uuid import UUID

one = ONE()

MESO_DIR = Path.home() / "Dropbox/scripts/IBL"
if str(MESO_DIR) not in sys.path:
    sys.path.insert(0, str(MESO_DIR))
from meso import plot_raster, plot_sparseness_res, load_or_embed

BASE_ROOT = "https://ibl.flatironinstitute.org/resources/mesoscope/ROICaT/"
FNAME = "mpciROIs.clusterUIDs.csv"
_HREF_RE = re.compile(r'href="([^"]+)"')

def list_dirs(url: str, auth: tuple[str,str]) -> List[str]:
    r = requests.get(url, auth=auth, timeout=30, allow_redirects=True)
    if r.status_code != 200: return []
    hrefs = _HREF_RE.findall(r.text)
    return sorted({h for h in hrefs if h.endswith('/') and h not in ('../','./')})

class Progress:
    def __init__(self, total:int):
        self.total=total; self.done=0; self.ok=0; self.miss=0
        self.lock=threading.Lock()
    def update(self, ok: bool):
        with self.lock:
            self.done+=1; self.ok+=int(ok); self.miss+=int(not ok)
            bar_len=30; filled=int(bar_len*self.done/max(1,self.total))
            bar="#"*filled + "-"*(bar_len-filled)
            pct=100*self.done/max(1,self.total)
            print(f"\r[{bar}] {self.done}/{self.total} ({pct:5.1f}%)  OK:{self.ok}  MISS:{self.miss}",
                  end="", file=sys.stderr, flush=True)
    def close(self): print("", file=sys.stderr)

def fetch_file(url: str, auth: tuple[str,str]) -> tuple[bytes|None,int]:
    try:
        r = requests.get(url, auth=auth, timeout=60, allow_redirects=True)
        if r.status_code != 200 or not r.content:
            return None, r.status_code
        return r.content, r.status_code  # accept any non-empty 200
    except requests.RequestException:
        return None, -1

def mirror_subject(subject: str,
                   out_root: str = "~/chronic_csv/ROICaT",
                   username: str = "iblmember",
                   password: str = "GrayMatter19",
                   overwrite: bool = False,
                   fov_regex: str = r"^FOV_\d{2}/$",
                   debug_show_misses: int = 20) -> None:
    auth = (username, password)
    out_base = Path(os.path.expanduser(out_root)) / subject
    subject_url = urljoin(BASE_ROOT, f"{subject}/")

    # 1) dates
    date_dirs = [d for d in list_dirs(subject_url, auth) if re.match(r"^\d{4}-\d{2}-\d{2}/$", d)]
    if not date_dirs:
        print("[DIAG] No dates found at subject index.", file=sys.stderr); return

    # 2) triples (date, number, fov)
    triples: List[Tuple[str,int,str]] = []
    for d in date_dirs:
        date_str = d.rstrip("/")
        num_url = urljoin(subject_url, d)
        nums = [n for n in list_dirs(num_url, auth) if re.match(r"^\d{3}/$", n)]
        for n in nums:
            num_int = int(n.rstrip("/"))
            alf_url = urljoin(num_url, f"{n}alf/")
            fov_dirs = [f for f in list_dirs(alf_url, auth) if re.match(fov_regex, f)]
            for fov in fov_dirs:
                triples.append((date_str, num_int, fov.rstrip("/")))
    if not triples:
        print("[DIAG] No (date, number, FOV) combinations found.", file=sys.stderr); return

    # 3) download
    prog = Progress(len(triples))
    misses: List[str] = []
    for date_str, num_int, fov in triples:
        dest = out_base / date_str / f"{num_int:03d}" / "alf" / fov / FNAME
        if dest.exists() and not overwrite:
            prog.update(True); continue
        src_url = urljoin(subject_url, f"{date_str}/{num_int:03d}/alf/{fov}/{FNAME}")
        blob, status = fetch_file(src_url, auth)
        if blob is None:
            if len(misses) < debug_show_misses:
                misses.append(f"[{status}] {src_url}")
            prog.update(False); continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        prog.update(True)
    prog.close()

    if misses:
        print("[DIAG] First misses:", file=sys.stderr)
        for m in misses: print(m, file=sys.stderr)
    print("Done.")



#####################################
######################################



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
    Return (subject, date_str 'YYYY-MM-DD', number 'NNN', lab_name).
    Tries Alyx; if missing/unpublished, falls back to parsing one.eid2path(eid).
    """
    # A) Alyx (preferred when available)
    try:
        meta = one.alyx.rest('sessions', 'read', id=eid)
        subject = meta['subject']
        date_str = str(meta['start_time'])[:10]
        number = int(meta.get('number', 1))
        lab = meta.get('lab')
        if isinstance(lab, dict):
            lab = lab.get('name')
        return subject, date_str, f'{number:03d}', (lab or '')
    except Exception:
        pass

    # B) Local path parse: .../FlatIron/<lab>/Subjects/<subject>/<YYYY-MM-DD>/<NNN>
    spath = one.eid2path(eid)
    if spath is None:
        raise RuntimeError(f"Cannot resolve session components for EID {eid}: Alyx lookup failed and eid2path is None.")
    spath = Path(spath)
    number = spath.name
    date_str = spath.parent.name
    subject = spath.parent.parent.name
    lab = ''
    try:
        lab = spath.parents[3].name  # .../<lab>/Subjects/<subject>/<date>/<number>
    except Exception:
        lab = ''
    # normalize number to 3 digits if numeric
    try:
        number = f"{int(number):03d}"
    except Exception:
        pass
    return subject, date_str, number, lab


def _candidate_fov_dir_paths(
    one: ONE,
    eid: str,
    fov_name: str,
    roicat_root: Path | None,
    server_root: Path | None
) -> List[Path]:
    """
    Priority order:
      1) roicat_root/ROICaT/<subject>/<date>/<number>/alf/<FOV_XX>
      2) server_root/Subjects/<subject>/<date>/<number>/alf/<FOV_XX>
      3) <one.cache_dir>/FlatIron/<lab>/Subjects/<subject>/<date>/<number>/alf/<FOV_XX>
    """
    subject, date_str, number, lab = _session_components_from_eid(one, eid)
    cands: List[Path] = []
    if roicat_root is not None:
        cands.append(Path(roicat_root) / 'ROICaT' / subject / date_str / number / 'alf' / fov_name)
    if server_root is not None:
        cands.append(Path(server_root) / 'Subjects' / subject / date_str / number / 'alf' / fov_name)
    if lab:
        cands.append(Path(one.cache_dir) / 'FlatIron' / lab / 'Subjects' / subject / date_str / number / 'alf' / fov_name)
    return cands


def _load_roitypes_bool(one: ONE, eid: str, fov_col: str) -> Optional[np.ndarray]:
    """Return boolean mask of neuronal ROIs for a FOV collection or None."""
    for ds in ('mpciROIs.mpciROITypes', 'mpciROITypes'):
        try:
            arr = one.load_dataset(eid, ds, collection=fov_col)
            if arr is not None:
                return np.asarray(arr).astype(bool)
        except Exception:
            pass
    try:
        obj = one.load_object(eid, 'mpciROIs', collection=fov_col)
        for key in ('mpciROITypes', 'ROITypes', 'roi_types', 'roiType'):
            if key in obj:
                return np.asarray(obj[key]).astype(bool)
    except Exception:
        pass
    return None

# ----------------------------------------------
# Load per-session clusterUIDs aligned to neurons
# ----------------------------------------------

def get_cluster_uids_neuronal(
    one: ONE,
    eid: str,
    roicat_root: str | Path | None = None,
    server_root: str | Path | None = None,
    filter_neurons: bool = True) -> np.ndarray:


    roicat_root = Path(roicat_root) if roicat_root is not None else None
    server_root = Path(server_root) if server_root is not None else None

    # FOV discovery via ONE; fallback to local mirror
    try:
        cols = one.list_collections(eid)
        fov_cols = [c for c in cols if '/FOV_' in c]
        fov_cols = sorted(fov_cols, key=lambda s: Path(s).name)
    except Exception:
        fov_cols = []

    if (not fov_cols) and (roicat_root is not None):
        subject, date_str, number, _ = _session_components_from_eid(one, eid)
        local_alf = Path(roicat_root) / 'ROICaT' / subject / date_str / number / 'alf'
        if local_alf.is_dir():
            fov_cols = [f'alf/{p.name}' for p in sorted(local_alf.glob('FOV_*')) if p.is_dir()]

    out = []
    for fov_col in fov_cols:
        fov_name = Path(fov_col).name

        mask_neuron = _load_roitypes_bool(one, eid, fov_col)  # may be None

        # read UID vector from any candidate location
        uid_vec = None
        for cand_dir in _candidate_fov_dir_paths(one, eid, fov_name, roicat_root, server_root):
            csv_path = cand_dir / 'mpciROIs.clusterUIDs.csv'
            if csv_path.exists():
                uid_vec = _read_uid_csv(csv_path)
                break

        # skip if nothing available for this FOV
        if uid_vec is None and mask_neuron is None:
            continue

        # if only mask is known, create empty-string UIDs of same length
        if uid_vec is None and mask_neuron is not None:
            uid_vec = np.full(mask_neuron.shape[0], '', dtype=object)

        # align lengths
        if mask_neuron is not None:
            n = int(mask_neuron.shape[0])
            if uid_vec.shape[0] < n:
                uid_vec = np.concatenate([uid_vec, np.full(n - uid_vec.shape[0], '', dtype=object)])
            elif uid_vec.shape[0] > n:
                uid_vec = uid_vec[:n]

        uid_vec = np.array(['' if (x is None) else str(x).strip() for x in uid_vec], dtype=object)
        if filter_neurons and (mask_neuron is not None):
            out.append(uid_vec[mask_neuron])
        else:
            out.append(uid_vec)

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
    roicat_root: Path = Path.home() / "chronic_csv",
    server_root: str | Path | None = None,
    *,
    filter_neurons: bool = True,
    sanity_check: bool = False,
) -> dict[str, np.ndarray]:
    """
    Map each session to row indices in rr['roi_signal'] that select the SAME tracked neurons,
    aligned by ROICaT cluster UIDs.

    - Shared UIDs are the lexicographic intersection across all sessions.
    - If filter_neurons=True, only ROIs flagged neuronal by mpciROITypes are considered.
    - If no shared UIDs exist, returns zero-length int arrays for ALL sessions.
    - If sanity_check=True, verifies indices are within [0, N-1] using load_or_embed(eid).

    Returns
    -------
    dict {eid: np.ndarray[int]} with identical lengths and order across sessions.
    """

    def _uid_first_index_map(u: np.ndarray) -> tuple[dict[str, int], np.ndarray]:
        """Map UID -> first absolute row index (only for non-empty UIDs)."""
        if u.size == 0:
            return {}, np.empty(0, dtype=object)
        nz = (u != '')
        if not np.any(nz):
            return {}, np.empty(0, dtype=object)
        u_nz = u[nz]
        u_unique, idx_first = np.unique(u_nz, return_index=True)
        idx_abs = np.flatnonzero(nz)[idx_first]
        return {uid: int(ix) for uid, ix in zip(u_unique, idx_abs)}, u_unique

    # ---------- collect UIDs ----------
    u_anchor = get_cluster_uids_neuronal(
        one, anchor_eid, roicat_root=roicat_root, server_root=server_root, filter_neurons=filter_neurons
    )
    anchor_uid2abs, anchor_unique = _uid_first_index_map(u_anchor)

    per_session_uidmaps: dict[str, dict[str, int]] = {anchor_eid: anchor_uid2abs}
    shared = anchor_unique.copy()  # running intersection (lexicographic)

    for e in other_eids:
        u_e = get_cluster_uids_neuronal(
            one, e, roicat_root=roicat_root, server_root=server_root, filter_neurons=filter_neurons
        )
        uid2abs_e, u_e_unique = _uid_first_index_map(u_e)
        per_session_uidmaps[e] = uid2abs_e
        # intersection across sessions; ensures uniqueness & lexicographic order
        shared = np.intersect1d(shared, u_e_unique, assume_unique=False)

        # early exit if empty
        if shared.size == 0:
            return {eid: np.empty(0, dtype=int) for eid in [anchor_eid, *other_eids]}

    # ---------- build aligned indices ----------
    out: dict[str, np.ndarray] = {}
    for eid, uid2abs in per_session_uidmaps.items():
        try:
            idx = np.fromiter((uid2abs[uid] for uid in shared), dtype=int, count=shared.size)
        except KeyError as ke:
            # This should not happen after intersect1d; signal internal inconsistency clearly.
            missing = str(ke).strip("'")
            raise ValueError(f"Shared UID not found in {eid}: {missing}") from None
        out[eid] = idx

    # ---------- optional sanity check against rr['roi_signal'].shape[0] ----------
    if sanity_check:
        from meso import load_or_embed  # lazy import to avoid heavy deps unless requested
        for eid, idx in out.items():
            if idx.size == 0:
                continue
            rr = load_or_embed(eid, rerun=False)
            N = int(rr["roi_signal"].shape[0])
            mx = int(idx.max()) if idx.size else -1
            mn = int(idx.min()) if idx.size else 0
            if mn < 0 or mx >= N:
                raise IndexError(
                    f"{eid}: tracked indices out of bounds (min={mn}, max={mx}, N={N}). "
                    "Ensure mappings are per-session row indices aligned to roi_signal."
                )

    return out



def _build_presence_matrix(
    one: ONE,
    eids: List[str],
    roicat_root: str | Path | None = None,
    server_root: str | Path | None = None,
    filter_neurons: bool = True
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """
    Build a boolean matrix M [n_sessions x n_uids], where M[i, j] = True
    iff UID j appears in session i. Returns (M, uid_list, eid_index).
    """
    # Gather unique UIDs per session
    sess_uids: List[np.ndarray] = []
    for eid in eids:
        u = get_cluster_uids_neuronal(one, eid, roicat_root=roicat_root, server_root=server_root, filter_neurons=filter_neurons)
        sess_uids.append(np.unique(u[u != '']).astype(str))

    # Global UID vocabulary
    uid_list = np.unique(np.concatenate([x for x in sess_uids if x.size], axis=0)).tolist()
    uid_index = {uid: j for j, uid in enumerate(uid_list)}

    # Fill matrix
    M = np.zeros((len(eids), len(uid_list)), dtype=bool)
    for i, arr in enumerate(sess_uids):
        if arr.size == 0:
            continue
        idx = np.fromiter((uid_index[uid] for uid in arr), dtype=int, count=arr.size)
        M[i, idx] = True

    eid_index = {eid: i for i, eid in enumerate(eids)}
    return M, uid_list, eid_index


def find_best_subsets_by_greedy_intersection(
    one: ONE,
    eids: List[str],
    roicat_root: str | Path | None = None,
    server_root: str | Path | None = None,
    k_min: int = 10,
    k_max: Optional[int] = None,
    n_starts: int = 10,
    random_starts: int = 0,
    rng: Optional[np.random.Generator] = None,
    filter_neurons: bool = True,
) -> Dict[int, Dict[str, object]]:
    """
    For each k in [k_min, k_max], greedily select k sessions that maximize
    |intersection of tracked UIDs|. Multi-start to reduce local optima.
    Returns: { k: {'eids': [..], 'n_shared': int} } with strictly nonincreasing n_shared as k grows.
    """
    assert k_min >= 1, "k_min must be >= 1"
    if k_max is None:
        k_max = len(eids)
    assert k_max <= len(eids) and k_min <= k_max

    M, uid_list, eid_index = _build_presence_matrix(one, eids, roicat_root, server_root , filter_neurons=filter_neurons)
    n_sess, n_uid = M.shape
    if n_uid == 0:
        return {k: {'eids': [], 'n_shared': 0} for k in range(k_min, k_max + 1)}

    # Seed selection: top by individual coverage + optional random seeds
    cover = M.sum(axis=1)  # UIDs per session
    order_top = np.argsort(-cover)[:min(n_starts, n_sess)]
    seeds = list(order_top)
    if random_starts > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        pool = np.setdiff1d(np.arange(n_sess), order_top, assume_unique=True)
        if pool.size > 0:
            seeds += rng.choice(pool, size=min(random_starts, pool.size), replace=False).tolist()

    best_for_k: Dict[int, Tuple[int, List[int]]] = {}  # k -> (n_shared, sel_idx_list)

    for seed in seeds:
        selected = [seed]
        inter_mask = M[seed].copy()  # current intersection mask over UIDs
        # Greedy forward selection
        while len(selected) < k_max:
            cand_idx = np.setdiff1d(np.arange(n_sess), np.array(selected), assume_unique=False)
            # Intersection size if we add each candidate: AND with current mask
            # Vectorized: for all candidates compute (M[cand] & inter_mask).sum(axis=1)
            inter_counts = (M[cand_idx] & inter_mask).sum(axis=1)
            j_best = cand_idx[np.argmax(inter_counts)]
            selected.append(int(j_best))
            inter_mask &= M[j_best]

            k = len(selected)
            if k >= k_min:
                n_shared = int(inter_mask.sum())
                prev = best_for_k.get(k, (-1, []))
                # Keep best n_shared; tie-breaker: lexicographic on eid strings
                if (n_shared > prev[0]) or (
                    n_shared == prev[0] and
                    [eids[i] for i in selected] < [eids[i] for i in prev[1]]
                ):
                    best_for_k[k] = (n_shared, selected.copy())

    # Format output
    out: Dict[int, Dict[str, object]] = {}
    for k in range(k_min, k_max + 1):
        if k not in best_for_k:
            out[k] = {'eids': [], 'n_shared': 0}
            continue
        n_shared, sel_idx = best_for_k[k]
        out[k] = {'eids': [eids[i] for i in sel_idx], 'n_shared': n_shared}
    return out

def find_session_dirs_with_uids_for_subject(roicat_root: str | Path, subject: str) -> List[Path]:
    root = Path(roicat_root) / 'ROICaT' / subject
    found = set()
    # ROICaT/<subject>/<YYYY-MM-DD>/<NNN>/alf/FOV_XX/mpciROIs.clusterUIDs.csv
    for csv in root.glob('[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/*/alf/FOV_*/mpciROIs.clusterUIDs.csv'):
        found.add(csv.parents[2])  # .../<subject>/<date>/<number>
    return sorted(found)

def session_dir_to_relpath(sess_dir: Path) -> str:
    subject, date, number = sess_dir.parts[-3:]
    return f'/{subject}/{date}/{number}/'

def eids_from_session_dirs(paths: Iterable[Path], one: ONE) -> Dict[Path, str]:
    mapping: Dict[Path, str] = {}
    seen: set[str] = set()
    for p in paths:
        rel = session_dir_to_relpath(Path(p))
        eid_obj = None
        try:
            eid_obj = one.path2eid(rel)   # may return None without raising
        except Exception:
            continue
        if not eid_obj:
            continue
        eid = str(eid_obj)
        if not eid or eid.lower() == 'none':
            continue
        if eid in seen:
            continue
        seen.add(eid)
        mapping[Path(p)] = eid
    return mapping


def aligned_indices_for_subset(
    one: ONE,
    eids_subset: List[str],
    roicat_root: str | Path,

    server_root: str | Path | None = None
) -> Dict[str, np.ndarray]:
    anchor = eids_subset[0]
    others = eids_subset[1:]
    return match_tracked_indices_across_sessions(
        one, anchor, others, roicat_root=roicat_root, server_root=server_root,filter_neurons=True
    )


def safe_plot_subset(eids_subset, idx_map):
    
    for eid in eids_subset:
        idx = idx_map.get(eid)
        n = int(idx.size) if idx is not None else 0
        if n == 0:
            print(f"[skip] {eid}: no shared neurons")
            continue
        try:
            plot_raster(eid, restrict=idx); print(f"raster done for {eid}"); plt.close()
        except Exception as e:
            print(f"raster error for {eid}\n{type(e).__name__}: {e}")
        try:
            plot_sparseness_res(eid, restrict=idx); print(f"sparseness done for {eid}"); plt.close()
        except Exception as e:
            print(f"sparseness error for {eid}\n{type(e).__name__}: {e}")


def best_eid_subsets_for_animal(
    subject: str,
    *,
    one: Optional[ONE] = None,
    roicat_root: Path = Path.home() / "chronic_csv",
    server_root: Optional[Path] = None,
    k_min: int = 5,
    k_max: Optional[int] = None,
    n_starts: int = 10,
    random_starts: int = 5,
    enforce_monotone: bool = True,
    min_trials: int = 400,
    trial_key: str = "stimOn_times",
    filter_neurons=True,
) -> Dict[int, Dict[str, object]]:
    """
    Return dict: {k: {'eids': [...], 'n_shared': int}} for a subject, using
    the greedy shared-neuron subset finder, after filtering sessions by trial count.

    Sessions are kept only if trials[trial_key].size >= min_trials (default 400).
    Falls back to skipping sessions where 'trials' are missing.

    Parameters
    ----------
    subject : str
        Animal/subject name (e.g., 'SP072').
    one : ONE, optional
        Preconfigured ONE instance. If None, a default ONE() is created.
    roicat_root : Path
        Local root containing 'ROICaT' exports.
    server_root : Path or None
        Optional server path to ROICaT.
    k_min, k_max : int
        Min/max subset sizes to evaluate. If k_max is None, uses min(10, n_sessions_filtered).
    n_starts : int
        Number of deterministic starts for the greedy procedure.
    random_starts : int
        Number of random starts (adds robustness).
    enforce_monotone : bool
        If True, post-process to make n_shared non-increasing with k.
    min_trials : int
        Minimum number of trials required to include a session (default 400).
    trial_key : str
        Trials field to count (default 'stimOn_times').

    Returns
    -------
    Dict[int, Dict[str, object]]
        Mapping k -> {'eids': [EID...], 'n_shared': int}.
    """
    if one is None:
        one = ONE()

    # 1) discover sessions for this subject that have ROICaT UID CSVs
    sess_dirs = find_session_dirs_with_uids_for_subject(roicat_root, subject=subject)
    path2eid = eids_from_session_dirs(sess_dirs, one)
    eids_all = sorted(set([e for e in path2eid.values() if e and str(e).lower() != "none"]))
    if not eids_all:
        raise RuntimeError(f"No EIDs with UID CSVs found for subject '{subject}' under {roicat_root}.")

    # 2) filter by trial count
    eids = []
    for eid in eids_all:
        try:
            trials = one.load_object(eid, "trials")
        except Exception:
            # missing trials object; skip
            continue
        arr = trials.get(trial_key, None)
        if arr is None:
            # optional fallback to another key (comment in/out as needed)
            # arr = trials.get("goCue_times", None)
            # if arr is None:
            continue
        try:
            n = int(arr.size)
        except Exception:
            # if the loaded field isn't a numpy array; try to coerce
            try:
                n = len(arr)
            except Exception:
                n = 0
        if n >= min_trials:
            eids.append(eid)

    if not eids:
        raise RuntimeError(
            f"No sessions for subject '{subject}' passed the trial filter: "
            f"{trial_key}.size >= {min_trials}."
        )

    # 3) set k bounds after filtering
    if k_max is None:
        k_max = min(10, len(eids))
    else:
        k_max = min(k_max, len(eids))
    if k_min < 1 or k_min > k_max:
        raise ValueError(f"Invalid k_min/k_max: {k_min}/{k_max} for n_sessions={len(eids)} after filtering.")

    # 4) run greedy finder on the filtered EIDs
    res = find_best_subsets_by_greedy_intersection(
        one, eids,
        roicat_root=roicat_root,
        server_root=server_root,
        k_min=k_min,
        k_max=k_max,
        n_starts=n_starts,
        random_starts=random_starts,filter_neurons=filter_neurons,
    )

    # 5) enforce monotone non-increasing n_shared across k (optional)
    if enforce_monotone and res:
        last = math.inf
        for k in sorted(res):
            n = int(res[k].get("n_shared", 0))
            n = min(n, last)
            res[k]["n_shared"] = int(n)
            last = n

    return res


def _eid_date(one: ONE, eid: str) -> str:
    try:
        meta = one.alyx.rest("sessions", "read", id=eid)
        return str(meta["start_time"])[:10]
    except Exception:
        return "9999-99-99"

def pairwise_shared_indices_for_animal(
    subject: str,
    *,
    one: Optional[ONE] = None,
    roicat_root: Path = Path.home() / "chronic_csv",
    server_root: Optional[Path] = None,
    min_trials: int = 400,
    trial_key: str = "stimOn_times",
    require_nonzero: bool = True, 
    filter_neurons=True,
) -> Dict[str, Dict[str, np.ndarray]]:
    if one is None:
        one = ONE()
    sess_dirs = find_session_dirs_with_uids_for_subject(roicat_root, subject=subject)
    path2eid = eids_from_session_dirs(sess_dirs, one)
    eids_all = sorted({e for e in path2eid.values() if e and str(e).lower() != "none"})
    if not eids_all:
        return {}

    # Filter by trials
    eids_keep = []
    for eid in eids_all:
        try:
            trials = one.load_object(eid, "trials")
            arr = trials.get(trial_key, None)
            n = int(arr.size) if arr is not None else 0
            if (n >= min_trials) and (not all(trials['probabilityLeft'] == 0.5)):
                eids_keep.append(eid)
        except Exception:
            continue
    if len(eids_keep) < 2:
        return {}

    # Chronological order
    eids_sorted = sorted(eids_keep, key=lambda e: (_eid_date(one, e), e))

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for i in range(len(eids_sorted) - 1):
        e0, e1 = eids_sorted[i], eids_sorted[i+1]
        try:
            idx_map = match_tracked_indices_across_sessions(
                one, e0, [e1], roicat_root=roicat_root, server_root=server_root,
                filter_neurons=filter_neurons,
            )
        except Exception:
            continue

        a = np.asarray(idx_map.get(e0, np.array([], dtype=int)), dtype=int)
        b = np.asarray(idx_map.get(e1, np.array([], dtype=int)), dtype=int)
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]

        if require_nonzero and n == 0:
            continue

        key = f"{e0[:3]}_{e1[:3]}"
        out[key] = {e0: a, e1: b}

    return out


