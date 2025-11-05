import gc
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Optional, List, Literal, Union
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import pickle
import json
from datetime import datetime
from one.api import ONE
import re

# --- progress support (no hard dependency) ---
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # fallback: identity iterator
        return x

# ------------------------
# Local IBL helper imports
# ------------------------
MESO_DIR = Path.home() / "Dropbox/scripts/IBL"
if str(MESO_DIR) not in sys.path:
    sys.path.insert(0, str(MESO_DIR))
from meso import get_win_times, load_or_embed
from meso_chronic import (pairwise_shared_indices_for_animal, match_tracked_indices_across_sessions, best_eid_subsets_for_animal)

one = ONE()
pth_meso = Path(one.cache_dir) / "meso" / "decoding"
pth_meso.mkdir(parents=True, exist_ok=True)
print(f"[meso] Using global cache directory: {pth_meso}")

Target = Literal["choice", "feedback", "stimulus", "block"]
ShuffleMode = Literal["permute", "circular", "blockwise"]


# =============
# Helper utils
# =============

def _as_stats_tuple(v) -> Tuple[float, float, int]:
    """Return (mean, sd, n) whether v is an array or a compact dict."""
    if isinstance(v, dict):
        return float(v.get("mean", np.nan)), float(v.get("sd", 0.0)), int(v.get("n", 0))
    a = np.asarray(v, float)
    if a.size == 0:
        return (np.nan, 0.0, 0)
    m = float(a.mean())
    s = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return (m, s, int(a.size))

@dataclass
class NDCResult:
    ks: np.ndarray
    scores: Dict[int, np.ndarray]
    metric: str
    ceiling: float
    shuffled: Optional[Dict[int, np.ndarray]] = None

class OnlineStats:
    __slots__ = ("n","mean","M2")
    def __init__(self): self.n=0; self.mean=0.0; self.M2=0.0
    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
    def result(self):
        sd = (self.M2 / (self.n - 1))**0.5 if self.n > 1 else 0.0
        return dict(mean=float(self.mean), sd=float(sd), n=int(self.n))

def _nan_to_num(a):
    return np.where(np.isfinite(a), a, 0.0)

def _event_window_indices(times: np.ndarray, ev: np.ndarray, win: Tuple[float,float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return i0,i1 indices (inclusive-exclusive) and a mask for trials with at least 1 sample."""
    starts = np.clip(ev + win[0], times[0], times[-1])
    ends   = np.clip(ev + win[1], times[0], times[-1])
    i0 = np.searchsorted(times, starts, side="left")
    i1 = np.searchsorted(times, ends,   side="right")
    good = i1 > i0
    return i0, i1, good

def _windowed_mean(rr, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    sig = rr["roi_signal"]  # (N,T)
    N = sig.shape[0]
    Tn = len(i0)
    X = np.empty((Tn, N), dtype=np.float32)
    for t in range(Tn):
        X[t] = sig[:, i0[t]:i1[t]].mean(axis=1)
    return X

# =====================
# Label constructors
# =====================

def _permute_labels(rng, y):
    return rng.permutation(y)

def _circular_shift_labels(rng, y):
    if y.size <= 1: return y.copy()
    s = rng.integers(1, y.size)  # 1..n-1
    return np.roll(y, s)

def _blockwise_permute_labels(rng, y, block_ids):
    # Permute y within each block id (e.g., session block or contrast block)
    if block_ids is None:
        return _permute_labels(rng, y)
    y_perm = y.copy()
    for b in np.unique(block_ids):
        idx = np.flatnonzero(block_ids == b)
        if idx.size > 1:
            y_perm[idx] = rng.permutation(y[idx])
    return y_perm



def _labels_choice(trials):
    choice = np.asarray(trials["choice"], float)  # -1,0,1
    valid = np.isfinite(choice) & (choice != 0)
    y = (choice[valid] == 1).astype(int)
    return y, valid

def _labels_feedback(trials):
    fb = np.asarray(trials["feedbackType"], float)  # 1,-1,0
    valid = np.isfinite(fb) & (fb != 0)
    y = (fb[valid] == 1).astype(int)
    return y, valid

def _labels_stimulus_side(trials):
    # Prefer signed contrast if present; else derive from CL/CR.
    if "signed_contrast" in trials:
        sc = np.asarray(trials["signed_contrast"], float)
        valid = np.isfinite(sc) & (sc != 0)
        y = (sc[valid] > 0).astype(int)  # right if positive
        return y, valid

    cl = np.asarray(trials.get("contrastLeft"), float)
    cr = np.asarray(trials.get("contrastRight"), float)

    # Treat NaNs as 0, then keep trials where exactly one side > 0
    cl = _nan_to_num(cl)
    cr = _nan_to_num(cr)
    valid = np.isfinite(cl) & np.isfinite(cr)
    one_side = ((cl > 0) ^ (cr > 0))
    valid &= one_side
    y = (cr[valid] > 0).astype(int)  # right=1, left=0
    return y, valid

def _labels_block(trials):
    p = np.asarray(trials["probabilityLeft"], float)
    valid = np.isfinite(p) & ((p == 0.2) | (p == 0.8))  # drop 0.5 and NaNs
    y = (p[valid] == 0.8).astype(int)  # 0.8 -> 1, 0.2 -> 0
    return y, valid

# ===========================
# Feature builder (robust)
# ===========================

def build_trial_features(
    eid: str,
    target: Target = "choice",
    rerun = False,
    restrict: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, Tuple[float, float]]]:

    rr = load_or_embed(eid, restrict=restrict, rerun=rerun)
    times = rr["roi_times"][0]
    trials, _tts = get_win_times(eid)

    if target == "choice":
        event = "firstMovement_times"; win = (-0.1, 0.0)
        y, valid_y = _labels_choice(trials)
        ev = np.asarray(trials[event], float)[valid_y]
    elif target == "feedback":
        event = "feedback_times"; win = (0.0, 0.20)
        y, valid_y = _labels_feedback(trials)
        ev = np.asarray(trials[event], float)[valid_y]
    elif target == "stimulus":
        event = "stimOn_times"; win = (0.0, 0.10)
        y, valid_y = _labels_stimulus_side(trials)
        ev = np.asarray(trials[event], float)[valid_y]
    elif target == "block":
        event = "stimOn_times"; win = (-0.40, -0.10)  # pre-stim window
        y, valid_y = _labels_block(trials)
        ev = np.asarray(trials[event], float)[valid_y]
    else:
        raise ValueError(f"Unknown target: {target}")

    # Drop trials with invalid/NaN event time
    good_ev = np.isfinite(ev)
    if not np.any(good_ev):
        raise RuntimeError(f"{eid} [{target}]: no finite event times")
    y = y[good_ev]
    ev = ev[good_ev]

    # Compute window indices and drop empty windows
    i0, i1, good_win = _event_window_indices(times, ev, win)
    if not np.any(good_win):
        raise RuntimeError(f"{eid} [{target}]: all windows empty after alignment")
    y = y[good_win]; i0 = i0[good_win]; i1 = i1[good_win]

    # Build features
    X = _windowed_mean(rr, i0, i1)

    # Remove rows with any non-finite values
    row_ok = np.all(np.isfinite(X), axis=1)
    X, y = X[row_ok], y[row_ok]
    if len(y) < 4:
        raise RuntimeError(f"{eid} [{target}]: too few valid trials after filtering (n={len(y)})")

    # Per-neuron z-score across trials (robust)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    X = (X - mu) / sd

    # Class balance (only if both classes exist)
    n0 = int(np.sum(y == 0)); n1 = int(np.sum(y == 1))
    if n0 == 0 or n1 == 0:
        raise RuntimeError(f"{eid} [{target}]: not enough balanced trials (class counts 0={n0},1={n1})")

    n = min(n0, n1)
    rng = np.random.default_rng(0)
    keep0 = rng.choice(np.flatnonzero(y == 0), n, replace=False)
    keep1 = rng.choice(np.flatnonzero(y == 1), n, replace=False)
    keep = np.sort(np.concatenate([keep0, keep1]))
    X, y = X[keep], y[keep]

    return X.astype(np.float32), y.astype(int), (event, win)



# ===============================
# 1) Decoding + NDC routines
# ===============================

def _stats_from_vals(vals: np.ndarray):
    # Handle empty safely (shouldn't happen, but defensive)
    if vals is None or len(vals) == 0:
        return dict(mean=np.nan, sd=np.nan, n=0)
    return dict(mean=float(np.mean(vals)),
                sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                n=int(len(vals)))

def compact_ndc_result(res: NDCResult) -> NDCResult:
    """
    Replace res.scores[k] and res.shuffled[k] arrays by dict(mean, sd, n)
    to drastically reduce memory footprint on disk and in RAM.
    """
    res.scores = {int(k): _stats_from_vals(v) for k, v in res.scores.items()}
    if res.shuffled is not None:
        res.shuffled = {int(k): _stats_from_vals(v) for k, v in res.shuffled.items()}
    return res


def _cv_score(clf, X, y, metric: str, n_splits: int, seed: int) -> float:
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values after preprocessing")
    n_splits = max(2, min(n_splits, int(np.min(np.bincount(y)))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    outs = []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        if metric == "auc":
            if hasattr(clf, "decision_function"):
                s = clf.decision_function(X[te])
            else:
                s = clf.predict_proba(X[te])[:, 1]
            outs.append(roc_auc_score(y[te], s))
        else:
            outs.append(accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(outs))

def neuron_dropping_curve(
    X: np.ndarray,
    y: np.ndarray,
    ks: Sequence[int] = (8,16,32,64,128,256,512,1024,2048),
    R: int = 50,
    metric: str = "auc",
    cv_splits: int = 5,
    seed: int = 0,
    ceiling_mode: str = "maxk",
    n_label_shuffles: int = 0,
    shuffle_seed: Optional[int] = None,
    shuffle_mode: ShuffleMode = "circular",
    block_ids: Optional[np.ndarray] = None,
    scores_dtype: np.dtype = np.float16,
    stats_only: bool = False,
    progress: bool = True,
) -> NDCResult:
    # --- MISSING LINES (restore) ---
    N = X.shape[1]
    ks = np.array(sorted([k for k in ks if 1 <= k <= N]), dtype=int)
    if ks.size == 0:
        raise ValueError("No valid k in ks <= number of neurons")

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def make_pipe():
        # scaler refit inside each CV split (no leakage)
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000))
        ])

    rng = np.random.default_rng(seed)

    scores: Dict[int, np.ndarray] = {}
    for k in tqdm(ks, desc=f"NDC[{metric}] ks", disable=not progress):
        if stats_only:
            acc = OnlineStats()
            for r in tqdm(range(R), desc=f"k={k} reps", leave=False, disable=not progress):
                cols = rng.choice(N, size=k, replace=False)
                val = _cv_score(make_pipe(), X[:, cols], y, metric=metric, n_splits=cv_splits, seed=seed + r)
                acc.update(float(val))
            scores[k] = acc.result()
        else:
            sc = np.empty(R, dtype=scores_dtype)
            for r in tqdm(range(R), desc=f"k={k} reps", leave=False, disable=not progress):
                cols = rng.choice(N, size=k, replace=False)
                sc[r] = _cv_score(make_pipe(), X[:, cols], y, metric=metric, n_splits=cv_splits, seed=seed + r)
            scores[k] = sc
            print(f"k={k:4d} | {metric} mean={sc.mean():.3f} ± {sc.std(ddof=1):.3f}")

    if n_label_shuffles and n_label_shuffles > 0:
        shuffled = {}
        rng_shuf = np.random.default_rng(seed if shuffle_seed is None else shuffle_seed)
        for k in tqdm(ks, desc="Shuffle ks", disable=not progress):
            if stats_only:
                acc = OnlineStats()
                for s in tqdm(range(n_label_shuffles), desc=f"k={k} shuf", leave=False, disable=not progress):
                    cols = rng_shuf.choice(N, size=k, replace=False)
                    if shuffle_mode == "permute":
                        y_perm = _permute_labels(rng_shuf, y)
                    elif shuffle_mode == "circular":
                        y_perm = _circular_shift_labels(rng_shuf, y)
                    else:
                        y_perm = _blockwise_permute_labels(rng_shuf, y, block_ids)
                    val = _cv_score(make_pipe(), X[:, cols], y_perm, metric=metric, n_splits=cv_splits, seed=seed + 10_000 + s)
                    acc.update(float(val))
                shuffled[k] = acc.result()
            else:
                scs = np.empty(n_label_shuffles, dtype=scores_dtype)
                for s in tqdm(range(n_label_shuffles), desc=f"k={k} shuf", leave=False, disable=not progress):
                    cols = rng_shuf.choice(N, size=k, replace=False)
                    if shuffle_mode == "permute":
                        y_perm = _permute_labels(rng_shuf, y)
                    elif shuffle_mode == "circular":
                        y_perm = _circular_shift_labels(rng_shuf, y)
                    else:
                        y_perm = _blockwise_permute_labels(rng_shuf, y, block_ids)
                    scs[s] = _cv_score(make_pipe(), X[:, cols], y_perm, metric=metric, n_splits=cv_splits, seed=seed + 10_000 + s)
                shuffled[k] = scs

    return NDCResult(ks=ks, scores=scores, metric=metric, ceiling=float(ceiling), shuffled=shuffled)

# ============================
# 2) Save/Load (fast plotting)
# ============================


def _tmp_session_name(prefix: str, target: str, eid: str) -> str:
    return f"{prefix}__{target}__{eid[:8]}.tmp.pkl"

def save_tmp_session(cache_dir: Path, prefix: str, target: str, eid: str, res: NDCResult, meta: dict, compact: bool = True):
    if compact:
        res = compact_ndc_result(res)
    payload = dict(prefix=prefix, target=target, eid=eid, result=res, meta=meta)
    with open(Path(cache_dir) / _tmp_session_name(prefix, target, eid), "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_and_merge_tmp_sessions(cache_dir: Path, prefix: str, target: str) -> Dict[str, dict]:
    sess = {}
    pat = f"{prefix}__{target}__*.tmp.pkl"
    for p in Path(cache_dir).glob(pat):
        with open(p, "rb") as f:
            pay = pickle.load(f)
        eid = pay["eid"]
        sess[eid] = {"result": pay["result"], "meta": pay["meta"]}
    return sess

def list_cached_groups(cache_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Scan cache_dir for group files saved by save_group_ndc_result and return
    a mapping: prefix -> {target -> path} for any targets that exist.
    """
    cache_dir = Path(cache_dir)
    out: Dict[str, Dict[str, Path]] = {}
    pat = re.compile(r"^(?P<prefix>.+)_(?P<target>choice|feedback|stimulus|block)\.pkl$")
    for p in cache_dir.glob("*.pkl"):
        m = pat.match(p.name)
        if not m:
            continue
        g = out.setdefault(m["prefix"], {})
        g[m["target"]] = p
    return out


def plot_cached_ndcs(
    cache_dir: Path,
    prefix: Optional[str] = None,
    eids: Optional[Sequence[str]] = None,
    targets: Sequence[Target] = ("choice", "feedback", "stimulus", "block"),
    delta: float = 0.02,
    show_shuffle: bool = True,
):
    """
    Plot previously cached NDCs without recomputing anything.

    Resolution order for the group identifier:
      1) If `prefix` is given, use it directly.
      2) Else if `eids` is given, derive the prefix as in compute_and_cache_ndcs.
      3) Else pick the most complete/most recent prefix from index.json or from files.
    """
    cache_dir = Path(cache_dir)

    # --- (1) choose/derive prefix ---
    chosen_prefix: Optional[str] = None
    if prefix:
        chosen_prefix = prefix
    elif eids:
        # same rule as compute_and_cache_ndcs
        eids_sorted = sorted(list(eids), key=_eid_date)
        chosen_prefix = "_".join([e[:3] for e in eids_sorted])
    else:
        # try index.json first for a deterministic choice
        idx_path = cache_dir / "index.json"
        candidates = {}
        if idx_path.exists():
            try:
                import json
                idx = json.loads(idx_path.read_text())
                # Prefer groups with most targets present, then latest in file order
                file_map = list_cached_groups(cache_dir)
                for row in idx:
                    pref = row.get("prefix")
                    if not pref:
                        continue
                    present = len(file_map.get(pref, {}))
                    candidates[pref] = present
                if candidates:
                    chosen_prefix = max(candidates.items(), key=lambda kv: kv[1])[0]
            except Exception:
                pass
        # fallback: scan files
        if not chosen_prefix:
            file_map = list_cached_groups(cache_dir)
            if not file_map:
                raise FileNotFoundError(f"No cached group files found in {cache_dir}.")
            # pick the prefix with the largest number of targets present
            chosen_prefix = max(file_map.items(), key=lambda kv: len(kv[1]))[0]

    # --- (2) verify that at least one requested target file exists ---
    missing = []
    have_any = False
    for t in targets:
        p = cache_dir / _group_cache_name(chosen_prefix, t)
        if p.exists():
            have_any = True
        else:
            missing.append(t)
    if not have_any:
        raise FileNotFoundError(
            f"No cached files for prefix '{chosen_prefix}' in {cache_dir}. "
            f"Expected at least one of: {[f'{chosen_prefix}_{t}.pkl' for t in targets]}"
        )
    if missing and len(missing) == len(targets):
        # defensive; should be caught by have_any above
        raise FileNotFoundError(f"All requested targets missing for prefix '{chosen_prefix}': {missing}")

    # --- (3) plot from cache only ---
    return plot_ndc_grid_from_cache(
        cache_dir=cache_dir,
        prefix_or_metas=chosen_prefix,
        targets=targets,
        delta=delta,
        show_shuffle=show_shuffle,
    )


def _infer_prefix_from_metas(metas: Dict[tuple, dict]) -> str:
    """Extract the common prefix from any meta dict in the metas mapping."""
    for meta in metas.values():
        if isinstance(meta, dict) and "prefix" in meta:
            return meta["prefix"]
    raise ValueError("No prefix found in metas dict.")

def _eid_date(eid: str) -> str:
    try:
        meta = one.alyx.rest("sessions", "read", id=eid)
        return str(meta["start_time"])[:10]
    except Exception:
        try:
            p = one.eid2path(eid)
            return p.parent.name if p else "9999-99-99"
        except Exception:
            return "9999-99-99"

def session_performance(eid: str) -> float:
    trials, _ = get_win_times(eid)
    fb = np.asarray(trials["feedbackType"])
    return float(np.sum(fb == 1) / len(fb)) if len(fb) else np.nan


def _subject_of_eid(eid: str) -> str:
    try:
        meta = one.alyx.rest("sessions", "read", id=eid)
        return str(meta.get("subject", ""))
    except Exception:
        return ""

def save_pairwise_ndc_result(
    cache_dir: Path,
    subject: str,
    prefix: str,
    target: Target,
    sessions: Dict[str, dict],   # eid -> {"result": NDCResult, "meta": dict}
    group_meta: dict
):
    """
    Save pairwise payload to:
        <cache_dir>/pair_summaries/<subject>/res/<prefix>_<target>.pkl
    """
    res_dir = Path(cache_dir) / "pair_summaries" / subject / "res"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = res_dir / f"{prefix}_{target}.pkl"

    payload = dict(
        prefix=prefix,
        target=target,
        group_meta=group_meta,
        sessions=sessions,
        saved_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def _group_cache_name(prefix: str, target: str) -> str:
    return f"{prefix}_{target}.pkl"

def save_group_ndc_result(
    cache_dir: Path,
    prefix: str,
    target: Target,
    sessions: Dict[str, dict],   # eid -> {"result": NDCResult, "meta": dict}
    group_meta: dict             # {"all_eids": [...], "n_shared": int, "prefix": str}
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / _group_cache_name(prefix, target)
    payload = dict(
        prefix=prefix,
        target=target,
        group_meta=group_meta,     # includes all_eids, n_shared, prefix
        sessions=sessions,         # per-session results+metadata
        saved_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Update index (one row per (prefix,target))
    index_path = cache_dir / "index.json"
    idx = []
    if index_path.exists():
        try:
            idx = json.loads(index_path.read_text())
        except Exception:
            idx = []

    # Compute a brief summary for the index
    # Use the first available session’s result to record metric/ceiling
    any_eid = next(iter(sessions.keys()))
    any_res: NDCResult = sessions[any_eid]["result"]
    brief = dict(
        prefix=prefix,
        target=target,
        all_eids=group_meta.get("all_eids"),
        n_shared=int(group_meta.get("n_shared")),
        ks_max=int(max(any_res.ks)),
        metric=any_res.metric,
        # store per-session perf in a small dict
        perf_by_eid={e: float(sessions[e]["meta"].get("performance", float("nan"))) for e in sessions.keys()},
    )
    idx = [b for b in idx if not (b.get("prefix")==prefix and b.get("target")==target)] + [brief]
    index_path.write_text(json.dumps(idx, indent=2))

def load_group_ndc_result(cache_dir: Path, prefix: str, target: Target):
    p = cache_dir / _group_cache_name(prefix, target)
    with open(p, "rb") as f:
        return pickle.load(f)

# ===================================
# 3) High-level pipeline (compute & save)
# ===================================

def compute_and_cache_ndcs(
    eids: Sequence[str],
    targets: Sequence[Target] = ("choice", "feedback", "stimulus", "block"),
    roicat_root: Path = Path.home() / "chronic_csv",
    cache_dir: Path = pth_meso,
    ks: tuple = (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    R: int = 50,
    metric: str = "auc",
    cv_splits: int = 5,
    equalize_trials: bool = True,
    seed: int = 0,
    rerun = False,
    ceiling_mode: str = "maxk",
    n_label_shuffles: int = 100,
    shuffle_seed: Optional[int] = None,
    idx_map_override: Optional[Dict[str, np.ndarray]] = None,   # NEW
) -> Dict[str, dict]:


    eids = sorted(list(eids), key=_eid_date)
    if len(eids) < 2:
        raise ValueError("Provide at least two EIDs.")
    subject = _subject_of_eid(eids[0])

    # Shared neurons across sessions
    if idx_map_override is not None:
        # use provided indices
        idx_map = {e: np.asarray(idx_map_override[e], dtype=int) for e in eids if e in idx_map_override}
        if len(idx_map) != len(eids):
            missing = [e for e in eids if e not in idx_map]
            raise ValueError(f"idx_map_override missing entries for: {missing}")
    else:
        idx_map = match_tracked_indices_across_sessions(one, eids[0], eids[1:], roicat_root=roicat_root)

    lens = {e: int(idx_map[e].size) for e in eids}
    if len(set(lens.values())) != 1:
        n_shared = min(lens.values())
        for e in eids:
            if lens[e] != n_shared:
                idx_map[e] = idx_map[e][:n_shared]
    n_shared = int(idx_map[eids[0]].size)
    if n_shared == 0:
        raise RuntimeError("No shared neurons across the provided sessions.")

    # Prefix like 3c7_4a4_bc3_...
    prefix = "_".join([e[:3] for e in eids])

    # ks up to n_shared (ensure max point present)
    ks_grid = tuple(sorted({k for k in ks if 1 <= k <= n_shared}))
    if not ks_grid or ks_grid[-1] != n_shared:
        base = [k for k in ks_grid] if ks_grid else []
        if n_shared not in base:
            base.append(n_shared)
        ks_grid = tuple(sorted(set(base)))

    # Build features per (eid,target)
    Xy_map: Dict[tuple, Tuple[np.ndarray, np.ndarray, Tuple[str, Tuple[float,float]]]] = {}
    for e in tqdm(eids, desc="Build features: sessions"):
        for t in tqdm(targets, desc="Targets", leave=False):
            try:
                X, y, (event, win) = build_trial_features(e, target=t, restrict=idx_map[e], rerun=rerun)
                Xy_map[(e,t)] = (X, y, (event, win))
            except Exception as ex:
                print(f"[skip] {e} [{t}]: {type(ex).__name__}: {ex}")

    if not Xy_map:
        raise RuntimeError("No sessions/targets yielded valid trial features.")

    # Equalize trials across sessions (per target)
    if equalize_trials:
        for t in targets:
            sizes = [len(Xy_map[(e,t)][1]) for e in eids if (e,t) in Xy_map]
            if not sizes:
                continue
            n_common = min(sizes)
            rng = np.random.default_rng(seed)
            for e in eids:
                if (e,t) not in Xy_map: continue
                X, y, meta_ev = Xy_map[(e,t)]
                if len(y) > n_common:
                    # stratify: keep class proportions ~equal
                    idx0 = np.flatnonzero(y == 0)
                    idx1 = np.flatnonzero(y == 1)
                    n_each = n_common // 2
                    take0 = rng.choice(idx0, size=min(n_each, len(idx0)), replace=False)
                    take1 = rng.choice(idx1, size=min(n_each, len(idx1)), replace=False)
                    take  = np.sort(np.concatenate([take0, take1]))
                    # if odd remainder, fill from the larger class
                    if take.size < n_common:
                        rest = np.setdiff1d(np.arange(len(y)), take, assume_unique=False)
                        extra = rng.choice(rest, size=(n_common - take.size), replace=False)
                        take = np.sort(np.concatenate([take, extra]))
                    Xy_map[(e,t)] = (X[take], y[take], meta_ev)

    metas = {}
    group_meta = dict(all_eids=list(eids), n_shared=n_shared, prefix=prefix)

    for t in tqdm(targets, desc="Compute NDC by target"):
        sessions: Dict[str, dict] = {}
        for e in tqdm(eids, desc=f"{t}: sessions", leave=False):
            if (e,t) not in Xy_map:
                continue
            X, y, (event, win) = Xy_map[(e,t)]
            res = neuron_dropping_curve(
                X, y, ks=ks_grid, R=R, metric=metric, cv_splits=cv_splits, seed=seed,
                ceiling_mode=ceiling_mode, n_label_shuffles=n_label_shuffles, shuffle_seed=shuffle_seed,
                scores_dtype=np.float16, stats_only=False, progress=True   # <- show bars
            )
            perf = session_performance(e)
            meta = dict(
                eid=e, target=t, date=_eid_date(e), n_shared=n_shared,
                ks_max=int(max(res.ks)), trials=len(y), performance=float(perf),
                event=event, win=tuple(win), equalize_trials=bool(equalize_trials),
                prefix=prefix, all_eids=list(eids), subject=subject,
            )
            sessions[e] = {"result": compact_ndc_result(res), "meta": meta}  # compact to save RAM
            metas[(e,t)] = meta
            del X, y, res
            gc.collect()

        if sessions:
            # If this is a pair, also save to the new per-subject pairwise layout
            if len(eids) == 2 and subject:
                save_pairwise_ndc_result(
                    cache_dir=cache_dir,
                    subject=subject,
                    prefix=prefix,
                    target=t,
                    sessions=sessions,
                    group_meta=group_meta,
                )
            # Keep legacy group file for backward compatibility (grid plotters, etc.)
            save_group_ndc_result(cache_dir, prefix=prefix, target=t, sessions=sessions, group_meta=group_meta)

    return metas

# =====================
# 4) Plotting from cache
# =====================


def summarize_k_star(res: NDCResult, delta: float = 0.02) -> Optional[int]:
    target = float(res.ceiling) - float(delta)
    ks_iter = res.ks if isinstance(res.ks, np.ndarray) else np.asarray(res.ks, dtype=int)
    means_by_k = {}
    for k in ks_iter:
        m, s, n = _as_stats_tuple(res.scores[int(k)])
        means_by_k[int(k)] = m
    ks_ok = sorted([k for k, m in means_by_k.items() if np.isfinite(m) and m >= target])
    return ks_ok[0] if ks_ok else None

def plot_ndc_grid_from_cache(
    cache_dir: Path,
    prefix_or_metas: Union[str, Dict[tuple, dict]],
    targets: Sequence[Target] = ("choice", "feedback", "stimulus", "block"),
    delta: float = 0.02,
    show_shuffle: bool = True,
    save_path: Optional[Path] = None,   # keep save option
    show: bool = True,                  # optionally suppress display
):
    """
    Load cached NDC results for a given prefix and render a 2x2 grid plot.
    Works with both array-based caches and compact dict caches {"mean","sd","n"}.
    """
    # Resolve prefix
    if isinstance(prefix_or_metas, dict):
        prefix = _infer_prefix_from_metas(prefix_or_metas)
    else:
        prefix = prefix_or_metas

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.ravel()

    # Stable panel order
    target_order = ["choice", "feedback", "stimulus", "block"]
    target_order = [t for t in target_order if t in targets] + [t for t in targets if t not in target_order]

    last_res = None  # for axis label fallback

    for ax, t in zip(axes, target_order):
        try:
            payload = load_group_ndc_result(cache_dir, prefix=prefix, target=t)
        except FileNotFoundError:
            ax.set_visible(False)
            continue

        sessions = payload.get("sessions", {})
        if not sessions:
            ax.set_visible(False)
            continue

        # Sort sessions chronologically (if dates present)
        def _date_of(eid):
            return sessions[eid]["meta"].get("date", "")
        eids_sorted = sorted(sessions.keys(), key=_date_of)

        first_shuffle_drawn = False

        for eid in eids_sorted:
            res: NDCResult = sessions[eid]["result"]
            meta = sessions[eid]["meta"]
            last_res = res

            ks_arr = np.asarray(res.ks, dtype=int)

            # Means / CIs using unified accessor
            means, sds, ns = [], [], []
            for k in ks_arr:
                m, s, n = _as_stats_tuple(res.scores[int(k)])
                means.append(m); sds.append(s); ns.append(n)
            means = np.asarray(means)
            sds   = np.asarray(sds)
            ns    = np.asarray(ns, dtype=int)
            cis   = 1.96 * sds / np.sqrt(np.maximum(1, ns))

            # Label with k* summary
            k_star = summarize_k_star(res, delta=delta)
            lbl = f"{meta.get('date','NA')} (k*={k_star})" if k_star is not None else f"{meta.get('date','NA')} (k*=NA)"
            ax.errorbar(ks_arr, means, yerr=cis, fmt="-o", capsize=3, label=lbl, zorder=3)

            # Shuffle band once per panel
            if show_shuffle and (res.shuffled is not None) and (not first_shuffle_drawn):
                sh_means, sh_sds, sh_ns = [], [], []
                for k in ks_arr:
                    m, s, n = _as_stats_tuple(res.shuffled[int(k)])
                    sh_means.append(m); sh_sds.append(s); sh_ns.append(n)
                sh_means = np.asarray(sh_means)
                sh_sds   = np.asarray(sh_sds)
                sh_ns    = np.asarray(sh_ns, dtype=int)
                sh_cis   = 1.96 * sh_sds / np.sqrt(np.maximum(1, sh_ns))

                ax.fill_between(ks_arr, sh_means - sh_cis, sh_means + sh_cis, alpha=0.10, color="gray",
                                label="shuffle ±95% CI", zorder=1)
                ax.plot(ks_arr, sh_means, "--", lw=1.0, color="gray", zorder=2)
                first_shuffle_drawn = True

            # Annotate behavioral performance at k_max
            perf = meta.get("performance", np.nan)
            if np.isfinite(perf):
                kmax = int(np.max(ks_arr))
                y_at_kmax = _as_stats_tuple(res.scores[kmax])[0]
                ax.annotate(f"{perf:.2f}", xy=(kmax, y_at_kmax), xytext=(0, 6),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8)

        # Ax cosmetics
        ax.set_xscale("log", base=2)
        ax.set_xlabel("neurons (k)")
        ylabel = (last_res.metric.upper() if last_res else "SCORE")
        ax.set_ylabel(ylabel)
        ax.set_title(t.capitalize())
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    # Hide any unused panels
    for j in range(len(target_order), 4):
        axes[j].set_visible(False)

    fig.suptitle(prefix, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

# =========================
# Convenience entrypoint
# =========================

def neuron_dropping_curves_cached(
    eids: Sequence[str],
    targets: Sequence[Target] = ("choice", "feedback", "stimulus", "block"),
    roicat_root: Path = Path.home() / "chronic_csv",
    cache_dir: Path = pth_meso,
    ks: tuple = (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    R: int = 50,
    metric: str = "auc",
    cv_splits: int = 5,
    delta: float = 0.02,
    equalize_trials: bool = True,
    seed: int = 0,
    ceiling_mode: str = "maxk",
    n_label_shuffles: int = 100,
    shuffle_seed: Optional[int] = None,
    plot: bool = True,
    rerun: bool = False,
):
    """
    If rerun=False (default):
        - derive the order-deterministic prefix from EIDs (same rule as compute),
        - if any cache files '<prefix>_<target>.pkl' exist, just plot from cache and return a minimal metas dict,
        - otherwise compute, cache, and (optionally) plot.

    If rerun=True:
        - force recomputation, overwrite caches, and (optionally) plot.
    """
    # Use same deterministic ordering as compute_and_cache_ndcs (by session date),
    # so the derived prefix matches the files on disk.
    eids_sorted = sorted(list(eids), key=_eid_date)
    prefix = "_".join([e[:3] for e in eids_sorted])

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing cache files for any of the requested targets
    existing = {t: (cache_dir / f"{prefix}_{t}.pkl").exists() for t in targets}
    have_any = any(existing.values())

    if not rerun and have_any:
        # Fast path: plot from cache only
        if plot:
            plot_ndc_grid_from_cache(
                cache_dir=cache_dir,
                prefix_or_metas=prefix,   # string -> loader reads '<prefix>_<target>.pkl'
                targets=targets,
                delta=delta,
                show_shuffle=True,
            )
        # Return a minimal metas-like dict (prefix + which targets were found),
        # so callers can inspect what was plotted without recomputation.
        return {
            "prefix": prefix,
            "found_targets": [t for t, ok in existing.items() if ok],
            "missing_targets": [t for t, ok in existing.items() if not ok],
        }

    # Otherwise, compute and cache (force if rerun=True or nothing is cached)
    metas = compute_and_cache_ndcs(
        eids=eids_sorted,
        targets=targets,
        roicat_root=roicat_root,
        cache_dir=cache_dir,
        ks=ks,
        R=R,
        metric=metric,
        cv_splits=cv_splits,
        equalize_trials=equalize_trials,
        seed=seed,
        ceiling_mode=ceiling_mode,
        n_label_shuffles=n_label_shuffles,
        shuffle_seed=shuffle_seed,
    )

    if plot:
        plot_ndc_grid_from_cache(
            cache_dir=cache_dir,
            prefix_or_metas=metas,   # metas contains the correct prefix
            targets=targets,
            delta=delta,
            show_shuffle=True,
        )

    return metas


def neuron_dropping_curves_for_session_pairs(
    subject: str,
    *,
    targets: Sequence[Target] = ("choice", "feedback", "stimulus", "block"),
    roicat_root: Path = Path.home() / "chronic_csv",
    cache_dir: Path = pth_meso,
    ks: tuple = (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    R: int = 50,
    metric: str = "auc",
    cv_splits: int = 5,
    delta: float = 0.02,
    equalize_trials: bool = True,
    seed: int = 0,
    ceiling_mode: str = "maxk",
    n_label_shuffles: int = 100,
    shuffle_seed: Optional[int] = None,
    min_trials: int = 400,
    trial_key: str = "stimOn_times",
    rerun: bool = False,
    show: bool = False,   # default off when batch-saving images
):
    """
    For a subject, compute NDCs for every consecutive session pair (after trial filtering),
    using shared-neuron indices per pair.

    Saves:
      - new payloads:  <cache_dir>/pair_summaries/<subject>/res/<prefix>_<target>.pkl
      - per-pair grid: <cache_dir>/pair_summaries/<subject>/imgs/<prefix>.png
    """
    one = ONE()
    pairs = pairwise_shared_indices_for_animal(
        subject, one=one, roicat_root=roicat_root, min_trials=min_trials, trial_key=trial_key
    )
    if not pairs:
        print(f"[info] No valid consecutive pairs found for subject {subject}.")
        return {}

    saved = {}

    # new layout roots
    pair_root = Path(cache_dir) / "pair_summaries" / subject
    img_root = pair_root / "imgs"
    res_root = pair_root / "res"
    img_root.mkdir(parents=True, exist_ok=True)
    res_root.mkdir(parents=True, exist_ok=True)

    for pair_key, idx_map in pairs.items():
        eids = list(idx_map.keys())
        eids_sorted = sorted(eids, key=_eid_date)
        prefix = "_".join([e[:3] for e in eids_sorted])

        # Already computed? Prefer new res/ structure; fall back to legacy.
        existing_new = {t: (res_root / f"{prefix}_{t}.pkl").exists() for t in targets}
        existing_legacy = {t: (Path(cache_dir) / f"{prefix}_{t}.pkl").exists() for t in targets}
        have_any = any(existing_new.values()) or any(existing_legacy.values())

        if not rerun and have_any:
            save_path = img_root / f"{prefix}.png"
            plot_ndc_grid_from_cache(
                cache_dir=cache_dir,
                prefix_or_metas=prefix,
                targets=targets,
                delta=delta,
                show_shuffle=True,
                save_path=save_path,
                show=show,
            )
            saved[pair_key] = {"prefix": prefix, "from_cache": True, "save_path": str(save_path)}
            continue

        # Compute & cache using the provided shared indices for this pair
        metas = compute_and_cache_ndcs(
            eids=eids_sorted,
            targets=targets,
            roicat_root=roicat_root,
            cache_dir=cache_dir,
            ks=ks,
            R=R,
            metric=metric,
            cv_splits=cv_splits,
            equalize_trials=equalize_trials,
            seed=seed,
            ceiling_mode=ceiling_mode,
            n_label_shuffles=n_label_shuffles,
            shuffle_seed=shuffle_seed,
            idx_map_override=idx_map,   # use pair-specific shared indices
        )

        # Save (or overwrite) the grid figure (legacy loader still works)
        save_path = img_root / f"{prefix}.png"
        plot_ndc_grid_from_cache(
            cache_dir=cache_dir,
            prefix_or_metas=metas,
            targets=targets,
            delta=delta,
            show_shuffle=True,
            save_path=save_path,
            show=show,
        )
        saved[pair_key] = {"prefix": prefix, "from_cache": False, "save_path": str(save_path)}

    return saved

def plot_session_pair_kpanels_for_subject(
    subject: str,
    *,
    targets: Sequence[Target] = ("choice", "feedback", "stimulus", "block"),
    cache_dir: Path = pth_meso,
    show: bool = True,
    save: bool = True,
) -> Dict[str, Optional[Path]]:
    """
    Load pairwise decoding payloads from:
        cache_dir / 'pair_summaries' / subject / 'res' / '*.pkl'
    (fallback to legacy cache_dir/*_<target>.pkl if none found),
    then plot 3 fixed ks panels (k = 64, 256, 1024) for each target.

    Style:
      - data markers black; shuffle markers grey (x, dashed)
      - legend only in top panel
      - shared x and y axes across rows
      - no grid; remove top/right spines
    """
    cache_dir = Path(cache_dir)
    out_paths: Dict[str, Optional[Path]] = {}

    # I/O roots in the new structure
    res_dir = cache_dir / "pair_summaries" / subject / "res"
    img_dir = cache_dir / "pair_summaries" / subject / "imgs"
    if save:
        img_dir.mkdir(parents=True, exist_ok=True)

    ks_fixed = [64, 256, 1024]

    def _mean_ci(stats_like):
        m, sd, n = _as_stats_tuple(stats_like)
        if n <= 1 or not np.isfinite(sd):
            return m, 0.0, n
        return m, 1.96 * sd / np.sqrt(max(1, n)), n

    def _sessions_belong_to_subject_by_meta(sessions_dict: dict) -> bool:
        subs = []
        for eid, sd in sessions_dict.items():
            meta = sd.get("meta", {})
            sub = str(meta.get("subject", ""))  # may or may not be present
            if sub:
                subs.append(sub)
        return (len(set(subs)) == 1 and subs[0] == subject) if subs else True

    def _load_pair_payloads_for_target(target: str) -> Dict[str, dict]:
        """
        Returns {prefix: payload} for this target.
        Prefers new res_dir; falls back to legacy cache_dir files.
        """
        out = {}

        # --- new layout: res_dir contains per-pair payloads like 'f7c_002_stimulus.pkl'
        if res_dir.exists():
            for p in res_dir.glob(f"*_{target}.pkl"):
                try:
                    with open(p, "rb") as f:
                        pay = pickle.load(f)
                except Exception:
                    continue
                sessions = pay.get("sessions", {})
                if len(sessions) != 2:
                    continue
                if not _sessions_belong_to_subject_by_meta(sessions):
                    continue
                # prefix from filename (stem without _target)
                stem = p.stem
                prefix = stem[: -len(f"_{target}")]
                out[prefix] = pay

        # --- fallback: legacy layout in cache_dir
        if not out:
            for p in cache_dir.glob(f"*_{target}.pkl"):
                try:
                    pay = load_group_ndc_result(cache_dir, prefix=p.stem[: -len(f"_{target}")], target=target)
                except Exception:
                    continue
                sessions = pay.get("sessions", {})
                if len(sessions) != 2:
                    continue
                if not _sessions_belong_to_subject_by_meta(sessions):
                    continue
                prefix = p.stem[: -len(f"_{target}")]
                out[prefix] = pay

        return out

    for target in targets:
        payloads = _load_pair_payloads_for_target(target)
        if not payloads:
            print(f"[info] No cached pair payloads found for {subject} [{target}].")
            out_paths[target] = None
            continue

        # Build global session timeline (dates) from payload metas
        eid_dates: Dict[str, str] = {}
        for pay in payloads.values():
            for eid, sd in pay.get("sessions", {}).items():
                meta = sd.get("meta", {})
                eid_dates[eid] = meta.get("date") or _eid_date(eid)

        unique_eids = sorted(eid_dates.keys(), key=lambda e: eid_dates[e])
        x_pos = {eid: i for i, eid in enumerate(unique_eids)}
        x_ticks = [eid_dates[e] for e in unique_eids]

        # Figure with shared axes
        fig, axes = plt.subplots(
            len(ks_fixed), 1, figsize=(6.26, 4.55), sharex=True, sharey=True,
            constrained_layout=True
        )
        metric_label = None

        def _pair_date(prefix):
            try:
                sess = payloads[prefix]["sessions"]
                eids_here = sorted(sess.keys(), key=lambda e: sess[e]["meta"].get("date", ""))
                return sess[eids_here[0]]["meta"].get("date", "")
            except Exception:
                return ""

        for i, (ax, k) in enumerate(zip(axes, ks_fixed)):
            drawn_any = False
            for prefix in sorted(payloads.keys(), key=_pair_date):
                sessions = payloads[prefix]["sessions"]
                # chronological within pair
                eids_here = sorted(sessions.keys(), key=lambda e: sessions[e]["meta"].get("date", ""))
                if len(eids_here) != 2:
                    continue
                eid_a, eid_b = eids_here
                res_a = sessions[eid_a]["result"]
                res_b = sessions[eid_b]["result"]
                if metric_label is None:
                    metric_label = (res_a.metric.upper() if hasattr(res_a, "metric") else "SCORE")

                ks_a = set(map(int, (res_a.ks if isinstance(res_a.ks, np.ndarray) else list(res_a.ks))))
                ks_b = set(map(int, (res_b.ks if isinstance(res_b.ks, np.ndarray) else list(res_b.ks))))
                if k not in ks_a or k not in ks_b:
                    continue

                # data
                m_a, ci_a, _ = _mean_ci(res_a.scores[int(k)])
                m_b, ci_b, _ = _mean_ci(res_b.scores[int(k)])
                # shuffle
                sh_a = getattr(res_a, "shuffled", None)
                sh_b = getattr(res_b, "shuffled", None)
                if sh_a and k in sh_a:
                    sm_a, sci_a, _ = _mean_ci(sh_a[int(k)])
                else:
                    sm_a, sci_a = np.nan, 0.0
                if sh_b and k in sh_b:
                    sm_b, sci_b, _ = _mean_ci(sh_b[int(k)])
                else:
                    sm_b, sci_b = np.nan, 0.0

                xa = x_pos.get(eid_a); xb = x_pos.get(eid_b)
                if xa is None or xb is None:
                    continue

                # main (black)
                ax.errorbar(xa + 0.25, m_a, yerr=ci_a, fmt="o", capsize=2.5, color="black", zorder=3)
                ax.errorbar(xb - 0.25, m_b, yerr=ci_b, fmt="o", capsize=2.5, color="black", zorder=3)
                # shuffle (grey)
                if np.isfinite(sm_a):
                    ax.errorbar(xa + 0.25, sm_a, yerr=sci_a, fmt="x", capsize=2.5,
                                linestyle="--", color="grey", zorder=2)
                if np.isfinite(sm_b):
                    ax.errorbar(xb - 0.25, sm_b, yerr=sci_b, fmt="x", capsize=2.5,
                                linestyle="--", color="grey", zorder=2)

                drawn_any = True

            # per-axis cosmetics
            ax.set_title(f"{target.capitalize()} — k={k}", fontsize=10)
            ax.set_ylabel(metric_label if metric_label else "Score", fontsize=9)
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if drawn_any and i == 0:
                data_line = ax.errorbar([], [], fmt="o", color="black", label="data")[0]
                shuf_line = ax.errorbar([], [], fmt="x", linestyle="--", color="grey", label="shuffle")[0]
                ax.legend(handles=[data_line, shuf_line], frameon=False, fontsize=9, loc="best")

        # shared x-axis
        axes[-1].set_xticks(list(range(len(unique_eids))))
        axes[-1].set_xticklabels(x_ticks, rotation=45, ha="right", fontsize=8)
        axes[-1].set_xlabel("Session date", fontsize=9)
        axes[-1].set_xlim(-0.5, len(unique_eids) - 0.5)
        for ax in axes:
            ax.tick_params(labelsize=8)

        
        fig.suptitle(f"{subject} — {target}", fontsize=12, y=0.995)
        fig.tight_layout()
        save_path = None
        if save:
            save_path = img_dir / f"{subject}_{target}_kpanels_k64_256_1024_black.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

        out_paths[target] = save_path

    return out_paths


def run_cross_session_train_first_test_rest(
    subject: str,
    *,
    one: Optional[ONE] = None,
    roicat_root: Path = Path.home() / "chronic_csv",
    min_sessions: int = 10,
    min_k_shared: int = 1349,
    targets: Sequence[Target] = ("choice", "feedback", "stimulus", "block"),
    n_shuffle: int = 200,              # shuffle control on the train/test accuracies
    # --- regularization / model selection controls ---
    inner_cv_splits: int = 5,          # CV inside FIRST session for train accuracy & tuning
    Cs: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0),
    use_elasticnet: bool = False,      # if True → elastic-net (solver='saga')
    l1_ratio_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    dimreduce: Literal["none", "pca", "selectk"] = "pca",
    dim_k: int = 512,                  # #PCs or #features for SelectKBest
    max_iter: int = 2000,
    seed: int = 0,
    rerun = False
) -> dict:
    """
    Select a subset of sessions for `subject` via best_eid_subsets_for_animal,
    requiring at least `min_sessions` sessions and global shared neurons >= `min_k_shared`.
    Train a single decoder on the FIRST session (all trials; k = n_shared), then
    test on each remaining session (same neuron indices). Compute accuracy and a
    shuffle-control accuracy (permute labels on the test set).
    
    Returns a dict with:
      'subject', 'eids' (ordered), 'dates', 'n_shared', 'results' (per target, per session).
    """
    if one is None:
        one = ONE()
    rng = np.random.default_rng(seed)

    # --- pick sessions via best_eid_subsets_for_animal (first outcome) ---
    subset_map = best_eid_subsets_for_animal(
        subject,
        one=one,
        roicat_root=roicat_root,
        k_min=min_sessions,
        k_max=min_sessions,   # exactly min_sessions (e.g., 10)
        n_starts=10,
        random_starts=5,
        enforce_monotone=True,
        min_trials=400,
        trial_key="stimOn_times",
    )
    if not subset_map:
        raise RuntimeError(f"{subject}: best_eid_subsets_for_animal returned no subsets.")

    # "Pick first outcome": take the first (lowest k) entry
    k = sorted(subset_map.keys())[0]
    eids = list(subset_map[k]["eids"])
    n_shared = int(subset_map[k]["n_shared"])

    if len(eids) < min_sessions:
        raise RuntimeError(f"{subject}: only {len(eids)} sessions returned; need ≥ {min_sessions}.")
    if n_shared < min_k_shared:
        raise RuntimeError(f"{subject}: n_shared={n_shared} < required {min_k_shared}.")

    # order by date
    def _date_of(eid: str) -> str:
        try:
            m = one.alyx.rest("sessions", "read", id=eid)
            return str(m.get("start_time", ""))[:10]
        except Exception:
            return ""
    eids = sorted(eids, key=_date_of)
    dates = [_date_of(e) for e in eids]

    # print the chosen eids
    print(f"[subject] {subject}")
    print(f"[sessions] {len(eids)} sessions; global shared neurons k = {n_shared}")
    for i, (e, d) in enumerate(zip(eids, dates)):
        print(f"  {i:02d}  {d}  {e}")

    # --- get the actual shared-neuron indices aligned to n_shared for the chosen set ---
    idx_map = match_tracked_indices_across_sessions(one, eids[0], eids[1:], roicat_root=roicat_root)
    if eids[0] not in idx_map:
        # identity mapping for train eid
        idx_map[eids[0]] = np.arange(n_shared, dtype=int)
    # truncate defensively to n_shared
    for e in eids:
        arr = np.asarray(idx_map[e], dtype=int)
        if arr.size < n_shared:
            raise RuntimeError(f"{subject}: mapping produced {arr.size} < n_shared({n_shared}) for {e}.")
        idx_map[e] = arr[:n_shared]

    # --- build features for all sessions/targets with the global shared neurons ---
    Xy = {}
    for e in tqdm(eids, desc="Build features: sessions"):
        restrict = np.asarray(idx_map[e], dtype=int)
        for t in tqdm(targets, desc="Targets", leave=False):
            try:
                X, y, _ = build_trial_features(e, target=t, restrict=restrict, rerun=rerun)
                Xy[(e, t)] = (X, y)
            except Exception as ex:
                print(f"[skip] build features {e} [{t}]: {type(ex).__name__}: {ex}")

    # --- train on first, test on rest (accuracy + shuffle control) ---
    results = {t: [] for t in targets}
    train_eid = eids[0]

    for t in targets:
        if (train_eid, t) not in Xy:
            print(f"[warn] no training features for {train_eid} [{t}] → skipping target.")
            continue

        Xtr, ytr = Xy[(train_eid, t)]

        from sklearn.model_selection import GridSearchCV

        # ---- determine effective CV splits and smallest training-fold size
        min_class = int(np.min(np.bincount(ytr)))
        cv_splits_eff = max(2, min(inner_cv_splits, min_class))
        # floor((cv_splits_eff - 1) / cv_splits_eff * n_samples)
        n_train_min = (cv_splits_eff - 1) * len(ytr) // cv_splits_eff
        if n_train_min <= 2:
            raise ValueError(f"Too few trials for CV with {cv_splits_eff} folds (n_train_min={n_train_min}).")

        # ---- build preprocessing steps
        steps = [("scaler", StandardScaler(with_mean=True, with_std=True))]
        if dimreduce == "pca":
            # cap PCs to <= min(n_features, min training samples - 1)
            n_comp_cap = max(2, min(dim_k, Xtr.shape[1], n_train_min - 1))
            steps.append(("pca", PCA(n_components=n_comp_cap,
                                     svd_solver="randomized", random_state=seed)))
        elif dimreduce == "selectk":
            k_sel = max(2, min(dim_k, Xtr.shape[1], n_train_min - 1))
            steps.append(("skb", SelectKBest(score_func=f_classif, k=k_sel)))

        base = Pipeline(steps + [("clf", LogisticRegression(
            penalty=("elasticnet" if use_elasticnet else "l2"),
            solver=("saga" if use_elasticnet else "liblinear"),
            max_iter=max_iter,
            random_state=seed
        ))])

        param_grid = {"clf__C": list(Cs)}
        if use_elasticnet:
            param_grid["clf__l1_ratio"] = list(l1_ratio_grid)

        skf = StratifiedKFold(
            n_splits=cv_splits_eff, shuffle=True, random_state=seed
        )
        gs = GridSearchCV(base, param_grid=param_grid, verbose=1,
                          scoring="roc_auc", cv=skf, refit=True)
        gs.fit(Xtr, ytr)
        final_pipe = gs.best_estimator_
        # ---- training-session accuracy: cross-validated (prevents inflated ~1.0)
        acc_tr = _cv_score(final_pipe, Xtr, ytr, metric="accuracy",
                           n_splits=cv_splits_eff, seed=seed)

        # Shuffle-control on training (same CV protocol)
        sh_tr = []
        rng_tr = np.random.default_rng(seed)
        for _ in range(n_shuffle):
            y_perm = _permute_labels(rng_tr, ytr)
            sh_tr.append(_cv_score(final_pipe, Xtr, y_perm, metric="accuracy",
                                   n_splits=cv_splits_eff, seed=seed))

        acc_tr_shuf = float(np.mean(sh_tr)) if sh_tr else np.nan

        # Save training record
        results[t].append(dict(eid=train_eid, date=_date_of(train_eid), which="train",
                            n_trials=int(len(ytr)), acc=float(acc_tr),
                            acc_shuffle=acc_tr_shuf, k=n_shared))

        # ---- fit final pipeline on FULL training session once for testing on other days
        final_pipe.fit(Xtr, ytr)

        # test sessions
        for e in eids[1:]:
            if (e, t) not in Xy:
                print(f"[skip] no test features for {e} [{t}]")
                continue
            Xte, yte = Xy[(e, t)]
            yhat = final_pipe.predict(Xte)
            acc  = float(accuracy_score(yte, yhat))
            sh   = [float(accuracy_score(_permute_labels(rng, yte), yhat)) for _ in range(n_shuffle)]
            results[t].append(dict(eid=e, date=_date_of(e), which="test",
                                    n_trials=int(len(yte)), acc=acc, acc_shuffle=float(np.mean(sh)), k=n_shared))

    return dict(subject=subject, eids=eids, dates=dates, n_shared=n_shared, targets=list(targets), results=results)  


def plot_cross_session_train_first_test_rest(payload: dict, *, show: bool = True, save_path: Optional[Path] = None):
    """
    Plot 4 panels (choice/feedback/stimulus/block) for k = n_shared only.
    Each panel shows per-session accuracy and shuffle-control accuracy (markers).
    """
    subject = payload["subject"]
    eids = payload["eids"]
    dates = payload["dates"]
    n_shared = payload["n_shared"]
    results = payload["results"]

    # x-axis order matches eids
    xs = np.arange(len(eids))

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)
    axes = axes.ravel()
    target_order = ["choice", "feedback", "stimulus", "block"]

    for ax, t in zip(axes, target_order):
        recs = results.get(t, [])
        if not recs:
            ax.set_visible(False)
            continue

        # gather in eids order
        accs, shfs, ntr = [], [], []
        for e in eids:
            r = next((row for row in recs if row["eid"] == e), None)
            if r is None:
                accs.append(np.nan); shfs.append(np.nan); ntr.append(0)
            else:
                accs.append(float(r["acc"])); shfs.append(float(r["acc_shuffle"])); ntr.append(int(r["n_trials"]))

        accs = np.asarray(accs, float)
        shfs = np.asarray(shfs, float)

        ax.plot(xs, accs, "o-", label="accuracy (k = n_shared)")
        ax.plot(xs, shfs, "x--", label="shuffle (permute labels)")

        ax.set_title(t.capitalize())
        ax.set_ylabel("Accuracy")
        ax.set_xticks(xs)
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # annotate trial counts near points
        for xi, (a, n) in enumerate(zip(accs, ntr)):
            if np.isfinite(a):
                ax.annotate(f"n={n}", xy=(xi, a), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=7)

        # legend only on first visible panel
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(frameon=False, fontsize=8, loc="lower right")

    # hide unused panels (if any)
    for j in range(len(target_order), 4):
        axes[j].set_visible(False)

    fig.suptitle(f"{subject} — k (global shared) = {n_shared} — {len(eids)} sessions", y=0.995, fontsize=11)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)



# payload = run_cross_session_train_first_test_rest(
#     subject="SP072",
#     min_sessions=10,
#     min_k_shared=3000,
#     targets=("choice","feedback","stimulus","block"),
#     n_shuffle=200,
#     seed=0,
# )

# plot_cross_session_train_first_test_rest(payload, show=True, save_path=None)       
