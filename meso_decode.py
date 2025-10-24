import gc
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Optional, List, Literal, Union
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
import pickle
import json
from datetime import datetime
from one.api import ONE

# ------------------------
# Local IBL helper imports
# ------------------------
MESO_DIR = Path.home() / "Dropbox/scripts/IBL"
if str(MESO_DIR) not in sys.path:
    sys.path.insert(0, str(MESO_DIR))
from meso import get_win_times, load_or_embed
from meso_chronic import match_tracked_indices_across_sessions

one = ONE()
pth_meso = Path(one.cache_dir) / "meso" / "decoding"
pth_meso.mkdir(parents=True, exist_ok=True)
print(f"[meso] Using global cache directory: {pth_meso}")

Target = Literal["choice", "feedback", "stimulus", "block"]

# =============
# Helper utils
# =============

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
    restrict: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, Tuple[float, float]]]:
    rr = load_or_embed(eid, restrict=restrict)
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


def _extract_stats(res: NDCResult, ks_arr: np.ndarray):
    means, sds, ns = [], [], []
    for k in ks_arr:
        v = res.scores[k]
        if isinstance(v, dict):
            means.append(v["mean"]); sds.append(v["sd"]); ns.append(v["n"])
        else:
            means.append(float(np.mean(v)))
            sds.append(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            ns.append(int(len(v)))
    return np.array(means), np.array(sds), np.array(ns)

def _extract_shuffle_stats(res: NDCResult, ks_arr: np.ndarray):
    sh_means, sh_sds, sh_ns = [], [], []
    for k in ks_arr:
        v = res.shuffled[k]
        if isinstance(v, dict):
            sh_means.append(v["mean"]); sh_sds.append(v["sd"]); sh_ns.append(v["n"])
        else:
            sh_means.append(float(np.mean(v)))
            sh_sds.append(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            sh_ns.append(int(len(v)))
    return np.array(sh_means), np.array(sh_sds), np.array(sh_ns)

@dataclass
class NDCResult:
    ks: np.ndarray
    scores: Dict[int, np.ndarray]
    metric: str
    ceiling: float
    shuffled: Optional[Dict[int, np.ndarray]] = None

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
    scores_dtype: np.dtype = np.float16,   # NEW: smaller arrays
    stats_only: bool = False,              # NEW: don’t keep per-replicate arrays
) -> NDCResult:
    ...
    scores: Dict[int, np.ndarray] = {}
    # if stats_only: keep OnlineStats in place of arrays, convert at end
    stats_scores: Dict[int, OnlineStats] = {} if stats_only else None

    for k in ks:
        if stats_only:
            acc = OnlineStats()
            for r in range(R):
                cols = rng.choice(N, size=k, replace=False)
                val = _cv_score(make_clf(), Xs[:, cols], y, metric=metric, n_splits=cv_splits, seed=seed + r)
                acc.update(float(val))
            scores[k] = acc.result()  # directly store dict(mean,sd,n)
        else:
            sc = np.empty(R, dtype=scores_dtype)  # np.float16 saves 2–4× RAM vs float64/32
            for r in range(R):
                cols = rng.choice(N, size=k, replace=False)
                sc[r] = _cv_score(make_clf(), Xs[:, cols], y, metric=metric, n_splits=cv_splits, seed=seed + r)
            scores[k] = sc
            print(f"k={k:4d} | {metric} mean={sc.mean():.3f} ± {sc.std(ddof=1):.3f}")

    if ceiling_mode == "all" and N > ks.max():
        ceiling = _cv_score(make_clf(), Xs, y, metric=metric, n_splits=cv_splits, seed=seed + 12345)
    else:
        if stats_only:
            ceiling = scores[int(ks.max())]["mean"]
        else:
            ceiling = float(np.mean(scores[ks.max()]))

    shuffled: Optional[Dict[int, np.ndarray]] = None
    if n_label_shuffles and n_label_shuffles > 0:
        shuffled = {}
        rng_shuf = np.random.default_rng(seed if shuffle_seed is None else shuffle_seed)
        for k in ks:
            if stats_only:
                acc = OnlineStats()
                for s in range(n_label_shuffles):
                    cols = rng_shuf.choice(N, size=k, replace=False)
                    y_perm = rng_shuf.permutation(y)
                    val = _cv_score(make_clf(), Xs[:, cols], y_perm, metric=metric, n_splits=cv_splits, seed=seed + 10_000 + s)
                    acc.update(float(val))
                shuffled[k] = acc.result()
            else:
                scs = np.empty(n_label_shuffles, dtype=scores_dtype)
                for s in range(n_label_shuffles):
                    cols = rng_shuf.choice(N, size=k, replace=False)
                    y_perm = rng_shuf.permutation(y)
                    scs[s] = _cv_score(make_clf(), Xs[:, cols], y_perm, metric=metric, n_splits=cv_splits, seed=seed + 10_000 + s)
                shuffled[k] = scs
                print(f"[shuffle] k={k:4d} | {metric} mean={scs.mean():.3f} ± {scs.std(ddof=1):.3f}")

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
    ceiling_mode: str = "maxk",
    n_label_shuffles: int = 100,
    shuffle_seed: Optional[str] = None,
) -> Dict[str, dict]:
    eids = sorted(list(eids), key=_eid_date)
    if len(eids) < 2:
        raise ValueError("Provide at least two EIDs.")

    # Shared neurons across sessions
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

    # Prefix like 3c7_4a4_bc3_f89_23c
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
    for e in eids:
        for t in targets:
            try:
                X, y, (event, win) = build_trial_features(e, target=t, restrict=idx_map[e])
                Xy_map[(e,t)] = (X, y, (event, win))
            except Exception as ex:
                print(f"[skip] {e} [{t}]: {type(ex).__name__}: {ex}")

    if not Xy_map:
        raise RuntimeError("No sessions/targets yielded valid trial features.")

    # Equalize trial counts per target across sessions
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
                    keep = rng.choice(len(y), size=n_common, replace=False)
                    Xy_map[(e,t)] = (X[keep], y[keep], meta_ev)

    metas = {}
    group_meta = dict(all_eids=list(eids), n_shared=n_shared, prefix=prefix)

    for t in targets:
        # build & equalize as you already do (Xy_map ready)
        for e in eids:
            if (e, t) not in Xy_map:
                continue
            X, y, (event, win) = Xy_map[(e, t)]

            # --- compute with small dtypes and/or stats-only ---
            res = neuron_dropping_curve(
                X, y,
                ks=ks_grid, R=R, metric=metric, cv_splits=cv_splits, seed=seed,
                ceiling_mode=ceiling_mode, n_label_shuffles=n_label_shuffles, shuffle_seed=shuffle_seed,
                scores_dtype=np.float16,          # saves RAM
                stats_only=False                  # set True if you only need mean/SD/N
            )

            perf = session_performance(e)
            meta = dict(
                eid=e, target=t, date=_eid_date(e), n_shared=n_shared,
                ks_max=int(max(res.ks)), trials=len(y), performance=float(perf),
                event=event, win=tuple(win), equalize_trials=bool(equalize_trials),
                prefix=prefix, all_eids=list(eids),
            )
            metas[(e, t)] = meta

            # --- write immediately and drop from RAM ---
            save_tmp_session(cache_dir, prefix, t, e, res, meta, compact=True)
            del X, y, res
            gc.collect()

        # after all eids for this target, assemble small sessions dict and save once
        sessions = load_and_merge_tmp_sessions(cache_dir, prefix, t)
        if sessions:
            save_group_ndc_result(cache_dir, prefix=prefix, target=t, sessions=sessions, group_meta=group_meta)

        # optional: clean tmp files now to reclaim disk
        for p in Path(cache_dir).glob(f"{prefix}__{t}__*.tmp.pkl"):
            try: p.unlink()
            except Exception: pass

    return metas

# =====================
# 4) Plotting from cache
# =====================


def summarize_k_star(res: NDCResult, delta: float = 0.02) -> Optional[int]:
    target = res.ceiling - float(delta)
    means = {k: float(v.mean()) for k, v in res.scores.items()}
    ks_ok = sorted([k for k, m in means.items() if m >= target])
    return ks_ok[0] if ks_ok else None

def plot_ndc_grid_from_cache(
    cache_dir: Path,
    prefix_or_metas: Union[str, Dict[tuple, dict]],
    targets: Sequence[Target] = ("choice", "feedback", "stimulus", "block"),
    delta: float = 0.02,
    show_shuffle: bool = True,
):
    import math
    # --- automatically infer prefix if a metas dict was passed ---
    if isinstance(prefix_or_metas, dict):
        prefix = _infer_prefix_from_metas(prefix_or_metas)
    else:
        prefix = prefix_or_metas

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.ravel()

    target_order = ["choice", "feedback", "stimulus", "block"]
    target_order = [t for t in target_order if t in targets] + [t for t in targets if t not in target_order]

    for ax, t in zip(axes, target_order):
        try:
            payload = load_group_ndc_result(cache_dir, prefix=prefix, target=t)
        except FileNotFoundError:
            ax.set_visible(False)
            continue

        sessions = payload["sessions"]
        if not sessions:
            ax.set_visible(False)
            continue

        def _date_of(eid):
            return sessions[eid]["meta"].get("date","")
        eids_sorted = sorted(sessions.keys(), key=_date_of)

        first_shuffle_drawn = False
        last_res = None
        for i, e in enumerate(eids_sorted):
            res: NDCResult = sessions[e]["result"]
            meta = sessions[e]["meta"]
            last_res = res

            ks_arr = res.ks
            means = np.array([res.scores[k].mean() for k in ks_arr])
            sds   = np.array([res.scores[k].std(ddof=1) for k in ks_arr])
            ns    = np.array([len(res.scores[k]) for k in ks_arr], dtype=int)
            cis   = 1.96 * sds / np.sqrt(np.maximum(1, ns))

            k_star = summarize_k_star(res, delta=delta)
            lbl = f"{meta.get('date','NA')} (k*={k_star})" if k_star is not None else f"{meta.get('date','NA')} (k*=NA)"
            ax.errorbar(ks_arr, means, yerr=cis, fmt="-o", capsize=3, label=lbl, zorder=3)

            # Shuffle band once per panel
            if show_shuffle and (res.shuffled is not None) and (not first_shuffle_drawn):
                sh_means = np.array([res.shuffled[k].mean() for k in ks_arr])
                sh_sds   = np.array([res.shuffled[k].std(ddof=1) for k in ks_arr])
                sh_ns    = np.array([len(res.shuffled[k]) for k in ks_arr], dtype=int)
                sh_cis   = 1.96 * sh_sds / np.sqrt(np.maximum(1, sh_ns))
                ax.fill_between(ks_arr, sh_means - sh_cis, sh_means + sh_cis, alpha=0.10, color="gray", label="shuffle ±95% CI", zorder=1)
                ax.plot(ks_arr, sh_means, "--", lw=1.0, color="gray", zorder=2)
                first_shuffle_drawn = True

            # Annotate behavioral performance at k_max
            perf = meta.get("performance", np.nan)
            if np.isfinite(perf):
                kmax = int(np.max(ks_arr))
                y_at_kmax = float(res.scores[kmax].mean())
                ax.annotate(f"{perf:.2f}", xy=(kmax, y_at_kmax), xytext=(0, 6),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("neurons (k)")
        ylabel = (last_res.metric.upper() if last_res else "SCORE")
        ax.set_ylabel(ylabel)
        ax.set_title(t.capitalize())
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    for j in range(len(target_order), 4):
        axes[j].set_visible(False)

    fig.suptitle(prefix, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

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
