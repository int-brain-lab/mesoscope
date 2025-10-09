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

MESO_DIR = Path.home() / "Dropbox/scripts/IBL"
if str(MESO_DIR) not in sys.path:
    sys.path.insert(0, str(MESO_DIR))
from meso import plot_raster, plot_sparseness_res

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
    server_root: str | Path | None = None
) -> np.ndarray:
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
        out.append(uid_vec[mask_neuron] if mask_neuron is not None else uid_vec)

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


def _build_presence_matrix(
    one: ONE,
    eids: List[str],
    roicat_root: str | Path | None = None,
    server_root: str | Path | None = None,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """
    Build a boolean matrix M [n_sessions x n_uids], where M[i, j] = True
    iff UID j appears in session i. Returns (M, uid_list, eid_index).
    """
    # Gather unique UIDs per session
    sess_uids: List[np.ndarray] = []
    for eid in eids:
        u = get_cluster_uids_neuronal(one, eid, roicat_root=roicat_root, server_root=server_root)
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

    M, uid_list, eid_index = _build_presence_matrix(one, eids, roicat_root, server_root)
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
        one, anchor, others, roicat_root=roicat_root, server_root=server_root
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


# if __name__ == "__main__":
#     one = ONE(base_url='https://alyx.internationalbrainlab.org')
#     roicat_root = Path.home() / 'chronic_csv'   # parent of 'ROICaT'

#     # discover EIDs for SP072 from your local mirror
#     sess_dirs = find_session_dirs_with_uids_for_subject(roicat_root, subject='SP072')
#     path2eid = eids_from_session_dirs(sess_dirs, one)
#     eids = sorted(set([e for e in path2eid.values() if e and e.lower() != 'none']))
#     print(f"Found {len(eids)} EIDs with UID CSVs for SP072.")

#     # compute best-k subsets
#     res = find_best_subsets_by_greedy_intersection(
#         one, eids,
#         roicat_root=roicat_root,
#         server_root=None,
#         k_min=5,
#         k_max=min(10, len(eids)),
#         n_starts=10, random_starts=5
#     )
#     for k in sorted(res):
#         print(f"{k}: {{'eids': {res[k]['eids']}, 'n_shared': {res[k]['n_shared']}}}")

#     last = math.inf
#     for k in sorted(res):
#         n = int(res[k].get('n_shared', 0))
#         n = min(n, last)          # if last is inf, this returns n (safe)
#         res[k]['n_shared'] = int(n)
#         last = n

#     # Choose the largest k with nonzero intersection; bail cleanly otherwise
#     valid_ks = [k for k in sorted(res) if int(res[k]['n_shared']) > 0]
#     if not valid_ks:
#         raise SystemExit("No subset with non-zero shared neurons was found.")
#     k = max(valid_ks)

#     idx_map = match_tracked_indices_across_sessions(
#         one, eids_k[0], eids_k[1:], roicat_root=roicat_root, server_root=None
#     )

#     # your plotting loop
    
#     for eid in idx_map:
#         safe_plot_subset(eids_k, idx_map)

