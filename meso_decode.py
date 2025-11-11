import gc
import json
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from one.api import ONE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------
# Local IBL helper imports
# ------------------------
MESO_DIR = Path.home() / "Dropbox/scripts/IBL"
if str(MESO_DIR) not in sys.path:
    sys.path.insert(0, str(MESO_DIR))
from meso import get_win_times, load_or_embed
from meso_chronic import (
    pairwise_shared_indices_for_animal,
    match_tracked_indices_across_sessions,
    best_eid_subsets_for_animal,
)

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# ------------------------
# Globals & types
# ------------------------
one = ONE()
pth_meso = Path(one.cache_dir) / "meso" / "decoding"
pth_meso.mkdir(parents=True, exist_ok=True)
print(f"[meso] Using global cache directory: {pth_meso}")

Target = Literal["choice", "feedback", "stimulus", "block"]
ShuffleMode = Literal["permute", "circular", "blockwise"]

# =====================
# Small helpers
# =====================

def _nan_to_num(a):
    return np.where(np.isfinite(a), a, 0.0)


def _event_window_indices(times: np.ndarray, ev: np.ndarray, win: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return i0,i1 indices (inclusive-exclusive) and a mask for trials with ≥1 sample."""
    starts = np.clip(ev + win[0], times[0], times[-1])
    ends = np.clip(ev + win[1], times[0], times[-1])
    i0 = np.searchsorted(times, starts, side="left")
    i1 = np.searchsorted(times, ends, side="right")
    good = i1 > i0
    return i0, i1, good


def _windowed_mean(rr, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    sig = rr["roi_signal"]  # (N,T)
    N = sig.shape[0]
    Tn = len(i0)
    X = np.empty((Tn, N), dtype=np.float32)
    for t in range(Tn):
        X[t] = sig[:, i0[t]: i1[t]].mean(axis=1)
    return X


# =====================
# Label constructors
# =====================

def _permute_labels(rng, y):
    return rng.permutation(y)


def _circular_shift_labels(rng, y):
    if y.size <= 1:
        return y.copy()
    s = rng.integers(1, y.size)
    return np.roll(y, s)


def _blockwise_permute_labels(rng, y, block_ids):
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
    if "signed_contrast" in trials:
        sc = np.asarray(trials["signed_contrast"], float)
        valid = np.isfinite(sc) & (sc != 0)
        y = (sc[valid] > 0).astype(int)  # right if positive
        return y, valid
    cl = np.asarray(trials.get("contrastLeft"), float)
    cr = np.asarray(trials.get("contrastRight"), float)
    cl = _nan_to_num(cl)
    cr = _nan_to_num(cr)
    valid = np.isfinite(cl) & np.isfinite(cr)
    one_side = (cl > 0) ^ (cr > 0)
    valid &= one_side
    y = (cr[valid] > 0).astype(int)  # right=1, left=0
    return y, valid


def _labels_block(trials):
    p = np.asarray(trials["probabilityLeft"], float)
    valid = np.isfinite(p) & ((p == 0.2) | (p == 0.8))
    y = (p[valid] == 0.8).astype(int)  # 0.8 -> 1, 0.2 -> 0
    return y, valid


# ===========================
# Feature builder (robust, unified)
# ===========================

def build_trial_features(
    eid: str,
    target: Target = "choice",
    rerun: bool = False,
    restrict: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, Tuple[float, float]]]:
    """Return (X,y,(event,win)) of per-trial mean activity for a target.

    IMPORTANT: We *load* the full cached embedding first and then apply
    `restrict` to the neuron axis (rows). This mirrors meso_decode2.py and
    avoids inadvertent reshuffling inside load_or_embed.
    """
    rr = load_or_embed(eid, rerun=rerun)  # do not pass restrict to loader

    # Validate and apply restrict (row selection)
    if restrict is not None:
        r = np.asarray(restrict, dtype=int)
        if r.ndim != 1 or r.size == 0:
            raise ValueError(f"{eid}: restrict must be a 1D non-empty integer array.")
        N = int(rr["roi_signal"].shape[0])
        if r.min() < 0 or r.max() >= N:
            raise IndexError(f"{eid}: restrict indices out of bounds (min={r.min()}, max={r.max()}, N={N}).")
        rr["roi_signal"] = rr["roi_signal"][r, :]
        for key in ("uuids", "acs"):
            if key in rr and len(rr[key]) == N:
                rr[key] = rr[key][r]

    times = rr["roi_times"][0]
    trials, _ = get_win_times(eid)

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
        event = "stimOn_times"; win = (-0.40, -0.10)
        y, valid_y = _labels_block(trials)
        ev = np.asarray(trials[event], float)[valid_y]
    else:
        raise ValueError(f"Unknown target: {target}")

    good_ev = np.isfinite(ev)
    if not np.any(good_ev):
        raise RuntimeError(f"{eid} [{target}]: no finite event times")
    y = y[good_ev]
    ev = ev[good_ev]

    i0, i1, good_win = _event_window_indices(times, ev, win)
    if not np.any(good_win):
        raise RuntimeError(f"{eid} [{target}]: all windows empty after alignment")
    y = y[good_win]; i0 = i0[good_win]; i1 = i1[good_win]

    X = _windowed_mean(rr, i0, i1)

    row_ok = np.all(np.isfinite(X), axis=1)
    X, y = X[row_ok], y[row_ok]
    if len(y) < 4:
        raise RuntimeError(f"{eid} [{target}]: too few valid trials after filtering (n={len(y)})")

    # z-score per neuron across trials
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    X = (X - mu) / sd

    # balance classes
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


# =====================================
# NDC computation 
# =====================================

@dataclass
class NDCResult:
    ks: np.ndarray
    scores: Dict[int, Union[np.ndarray, Dict[str, float]]]
    metric: str
    ceiling: float
    shuffled: Optional[Dict[int, Union[np.ndarray, Dict[str, float]]]] = None


class OnlineStats:
    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)

    def result(self):
        sd = (self.M2 / (self.n - 1)) ** 0.5 if self.n > 1 else 0.0
        return dict(mean=float(self.mean), sd=float(sd), n=int(self.n))


def _as_stats_tuple(v) -> Tuple[float, float, int]:
    if isinstance(v, dict):
        return float(v.get("mean", np.nan)), float(v.get("sd", 0.0)), int(v.get("n", 0))
    a = np.asarray(v, float)
    if a.size == 0:
        return (np.nan, 0.0, 0)
    m = float(a.mean())
    s = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return (m, s, int(a.size))


def _cv_score(clf, X, y, metric: str, n_splits: int, seed: int) -> float:
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


def compact_ndc_result(res: NDCResult) -> NDCResult:
    def _stats(vals):
        if vals is None or len(vals) == 0:
            return dict(mean=np.nan, sd=np.nan, n=0)
        return dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, n=int(len(vals)))
    res.scores = {int(k): _stats(v) if not isinstance(v, dict) else v for k, v in res.scores.items()}
    if res.shuffled is not None:
        res.shuffled = {int(k): _stats(v) if not isinstance(v, dict) else v for k, v in res.shuffled.items()}
    return res


def neuron_dropping_curve(
    X: np.ndarray,
    y: np.ndarray,
    ks: Sequence[int] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
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
    N = X.shape[1]
    ks = np.array(sorted([k for k in ks if 1 <= k <= N]), dtype=int)
    if ks.size == 0:
        raise ValueError("No valid k in ks <= number of neurons")

    def make_pipe():
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000)),
        ])

    rng = np.random.default_rng(seed)

    scores: Dict[int, Union[np.ndarray, Dict[str, float]]] = {}
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

    shuffled = None
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

    kmax = int(ks.max())
    if stats_only:
        m, _, _ = _as_stats_tuple(scores[kmax])
        ceiling = float(m)
    else:
        sc = np.asarray(scores[kmax], dtype=float)
        ceiling = float(np.mean(sc)) if sc.size else np.nan

    return NDCResult(ks=ks, scores=scores, metric=metric, ceiling=float(ceiling), shuffled=shuffled)


# ===================================
# Save/Load (group cache for fast plotting)
# ===================================

def _group_cache_name(prefix: str, target: str) -> str:
    return f"{prefix}_{target}.pkl"


def save_group_ndc_result(
    cache_dir: Path,
    prefix: str,
    target: Target,
    sessions: Dict[str, dict],  # eid -> {"result": NDCResult, "meta": dict}
    group_meta: dict,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / _group_cache_name(prefix, target)
    payload = dict(
        prefix=prefix,
        target=target,
        group_meta=group_meta,
        sessions=sessions,
        saved_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    index_path = cache_dir / "index.json"
    idx = []
    if index_path.exists():
        try:
            idx = json.loads(index_path.read_text())
        except Exception:
            idx = []

    any_eid = next(iter(sessions.keys()))
    any_res: NDCResult = sessions[any_eid]["result"]
    brief = dict(
        prefix=prefix,
        target=target,
        all_eids=group_meta.get("all_eids"),
        n_shared=int(group_meta.get("n_shared")),
        ks_max=int(max(any_res.ks)),
        metric=any_res.metric,
        perf_by_eid={e: float(sessions[e]["meta"].get("performance", float("nan"))) for e in sessions.keys()},
    )
    idx = [b for b in idx if not (b.get("prefix") == prefix and b.get("target") == target)] + [brief]
    index_path.write_text(json.dumps(idx, indent=2))


def load_group_ndc_result(cache_dir: Path, prefix: str, target: Target):
    p = cache_dir / _group_cache_name(prefix, target)
    with open(p, "rb") as f:
        return pickle.load(f)


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


# ===================================
# High-level: compute and cache NDCs for a set of sessions
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
    rerun: bool = False,
    ceiling_mode: str = "maxk",
    n_label_shuffles: int = 100,
    shuffle_seed: Optional[int] = None,
    idx_map_override: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, dict]:

    eids = sorted(list(eids), key=_eid_date)
    if len(eids) < 2:
        raise ValueError("Provide at least two EIDs.")
    subject = _subject_of_eid(eids[0])

    if idx_map_override is not None:
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

    prefix = "_".join([e[:3] for e in eids])

    ks_grid = tuple(sorted({k for k in ks if 1 <= k <= n_shared}))
    if not ks_grid or ks_grid[-1] != n_shared:
        base = [k for k in ks_grid] if ks_grid else []
        if n_shared not in base:
            base.append(n_shared)
        ks_grid = tuple(sorted(set(base)))

    Xy_map: Dict[tuple, Tuple[np.ndarray, np.ndarray, Tuple[str, Tuple[float, float]]]] = {}
    for e in tqdm(eids, desc="Build features: sessions"):
        for t in tqdm(targets, desc="Targets", leave=False):
            try:
                X, y, (event, win) = build_trial_features(e, target=t, restrict=idx_map[e], rerun=rerun)
                Xy_map[(e, t)] = (X, y, (event, win))
            except Exception as ex:
                print(f"[skip] {e} [{t}]: {type(ex).__name__}: {ex}")

    if not Xy_map:
        raise RuntimeError("No sessions/targets yielded valid trial features.")

    if equalize_trials:
        for t in targets:
            sizes = [len(Xy_map[(e, t)][1]) for e in eids if (e, t) in Xy_map]
            if not sizes:
                continue
            n_common = min(sizes)
            rng = np.random.default_rng(seed)
            for e in eids:
                if (e, t) not in Xy_map:
                    continue
                X, y, meta_ev = Xy_map[(e, t)]
                if len(y) > n_common:
                    idx0 = np.flatnonzero(y == 0)
                    idx1 = np.flatnonzero(y == 1)
                    n_each = n_common // 2
                    take0 = rng.choice(idx0, size=min(n_each, len(idx0)), replace=False)
                    take1 = rng.choice(idx1, size=min(n_each, len(idx1)), replace=False)
                    take = np.sort(np.concatenate([take0, take1]))
                    if take.size < n_common:
                        rest = np.setdiff1d(np.arange(len(y)), take, assume_unique=False)
                        extra = rng.choice(rest, size=(n_common - take.size), replace=False)
                        take = np.sort(np.concatenate([take, extra]))
                    Xy_map[(e, t)] = (X[take], y[take], meta_ev)

    metas = {}
    group_meta = dict(all_eids=list(eids), n_shared=n_shared, prefix=prefix)

    for t in tqdm(targets, desc="Compute NDC by target"):
        sessions: Dict[str, dict] = {}
        for e in tqdm(eids, desc=f"{t}: sessions", leave=False):
            if (e, t) not in Xy_map:
                continue
            X, y, (event, win) = Xy_map[(e, t)]
            res = neuron_dropping_curve(
                X, y, ks=ks_grid, R=50, metric=metric, cv_splits=cv_splits, seed=seed,
                ceiling_mode=ceiling_mode, n_label_shuffles=n_label_shuffles, shuffle_seed=shuffle_seed,
                scores_dtype=np.float16, stats_only=False, progress=True,
            )
            perf = session_performance(e)
            meta = dict(
                eid=e, target=t, date=_eid_date(e), n_shared=n_shared,
                ks_max=int(max(res.ks)), trials=len(y), performance=float(perf),
                event=event, win=tuple(win), equalize_trials=bool(equalize_trials),
                prefix=prefix, all_eids=list(eids), subject=subject,
            )
            sessions[e] = {"result": compact_ndc_result(res), "meta": meta}
            metas[(e, t)] = meta
            del X, y, res
            gc.collect()
        if sessions:
            save_group_ndc_result(cache_dir, prefix=prefix, target=t, sessions=sessions, group_meta=group_meta)

    return metas


# ===================================
# Plotting cached NDCs (2x2 grid)
# ===================================

def _infer_prefix_from_metas(metas: Dict[tuple, dict]) -> str:
    for meta in metas.values():
        if isinstance(meta, dict) and "prefix" in meta:
            return meta["prefix"]
    raise ValueError("No prefix found in metas dict.")


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
    save_path: Optional[Path] = None,
    show: bool = True,
):
    if isinstance(prefix_or_metas, dict):
        prefix = _infer_prefix_from_metas(prefix_or_metas)
    else:
        prefix = prefix_or_metas

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.ravel()
    target_order = ["choice", "feedback", "stimulus", "block"]
    target_order = [t for t in target_order if t in targets] + [t for t in targets if t not in target_order]

    last_res = None
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

        def _date_of(eid):
            return sessions[eid]["meta"].get("date", "")
        eids_sorted = sorted(sessions.keys(), key=_date_of)

        first_shuffle_drawn = False
        for eid in eids_sorted:
            res: NDCResult = sessions[eid]["result"]
            meta = sessions[eid]["meta"]
            last_res = res

            ks_arr = np.asarray(res.ks, dtype=int)
            means, sds, ns = [], [], []
            for k in ks_arr:
                m, s, n = _as_stats_tuple(res.scores[int(k)])
                means.append(m); sds.append(s); ns.append(n)
            means = np.asarray(means)
            sds = np.asarray(sds)
            ns = np.asarray(ns, dtype=int)
            cis = 1.96 * sds / np.sqrt(np.maximum(1, ns))

            k_star = summarize_k_star(res, delta=delta)
            lbl = f"{meta.get('date','NA')} (k*={k_star})" if k_star is not None else f"{meta.get('date','NA')} (k*=NA)"
            ax.errorbar(ks_arr, means, yerr=cis, fmt="-o", capsize=3, label=lbl, zorder=3)

            if show_shuffle and (res.shuffled is not None) and (not first_shuffle_drawn):
                sh_means, sh_sds, sh_ns = [], [], []
                for k in ks_arr:
                    m, s, n = _as_stats_tuple(res.shuffled[int(k)])
                    sh_means.append(m); sh_sds.append(s); sh_ns.append(n)
                sh_means = np.asarray(sh_means)
                sh_sds = np.asarray(sh_sds)
                sh_ns = np.asarray(sh_ns, dtype=int)
                sh_cis = 1.96 * sh_sds / np.sqrt(np.maximum(1, sh_ns))
                ax.fill_between(ks_arr, sh_means - sh_cis, sh_means + sh_cis, alpha=0.10, color="gray", label="shuffle ±95% CI", zorder=1)
                ax.plot(ks_arr, sh_means, "--", lw=1.0, color="gray", zorder=2)
                first_shuffle_drawn = True

            perf = meta.get("performance", np.nan)
            if np.isfinite(perf):
                kmax = int(np.max(ks_arr))
                y_at_kmax = _as_stats_tuple(res.scores[kmax])[0]
                ax.annotate(f"{perf:.2f}", xy=(kmax, y_at_kmax), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=8)

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

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ===================================
# Convenience wrapper: compute if needed, else just plot
# ===================================

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
    eids_sorted = sorted(list(eids), key=_eid_date)
    prefix = "_".join([e[:3] for e in eids_sorted])

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = {t: (cache_dir / f"{prefix}_{t}.pkl").exists() for t in targets}
    have_any = any(existing.values())

    if not rerun and have_any:
        if plot:
            plot_ndc_grid_from_cache(
                cache_dir=cache_dir,
                prefix_or_metas=prefix,
                targets=targets,
                delta=delta,
                show_shuffle=True,
            )
        return {
            "prefix": prefix,
            "found_targets": [t for t, ok in existing.items() if ok],
            "missing_targets": [t for t, ok in existing.items() if not ok],
        }

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
            prefix_or_metas=metas,
            targets=targets,
            delta=delta,
            show_shuffle=True,
        )
    return metas


# ===================================
# Cross-session: train on first, test the rest (from meso_decode2.py)
# ===================================

class PCACap(BaseEstimator, TransformerMixin):
    """Fit-time capping of PCA components to avoid CV failures."""
    def __init__(self, dim_k=128, random_state=None, svd_solver="randomized"):
        self.dim_k = int(dim_k)
        self.random_state = random_state
        self.svd_solver = svd_solver
        self._pca = None

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        n_comp = max(2, min(self.dim_k, n_features, n_samples - 1))
        self._pca = PCA(n_components=n_comp, svd_solver=self.svd_solver, random_state=self.random_state)
        self._pca.fit(X, y)
        return self

    def transform(self, X):
        return self._pca.transform(X)


def run_cross_session_train_first_test_rest(
    subject: str,
    *,
    one: Optional[ONE] = None,
    roicat_root: Path = Path.home() / "chronic_csv",
    min_sessions: int = 10,
    min_k_shared: int = 1349,
    targets: Sequence[Target] = ("choice", "feedback", "stimulus", "block"),
    n_shuffle: int = 50,
    inner_cv_splits: int = 3,
    Cs: Sequence[float] = (0.03, 0.3, 1.0),
    use_elasticnet: bool = False,
    l1_ratio_grid: Sequence[float] = (0.0, 0.5),
    dimreduce: Literal["none", "pca", "selectk"] = "pca",
    dim_k: int = 128,
    max_iter: int = 2000,
    seed: int = 0,
    rerun: bool = False,
    eid_train: Optional[str] = None,
    eid_test: Optional[Sequence[str]] = None,
    filter_neurons: bool = True,
) -> dict:

    if one is None:
        one = ONE()
    rng = np.random.default_rng(seed)

    out_dir = pth_meso / "cross_session_trainFirst"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _date_of(eid: str) -> str:
        try:
            m = one.alyx.rest("sessions", "read", id=eid)
            return str(m.get("start_time", ""))[:10]
        except Exception:
            return ""

    # Session selection
    if (eid_train is not None) or (eid_test is not None):
        if (eid_train is None) or (eid_test is None):
            raise ValueError("Provide both eid_train and eid_test or neither.")
        if not isinstance(eid_test, (list, tuple)):
            raise TypeError("eid_test must be a list/tuple of EIDs.")
        eids = [eid_train] + list(eid_test)
        dates = [_date_of(e) for e in eids]
        print(f"[subject] {subject} (explicit EIDs)")
        print(f"[sessions] {len(eids)} sessions (train on first; test on the rest)")
        for i, (e, d) in enumerate(zip(eids, dates)):
            print(f"  {i:02d}  {d}  {e}")
        idx_map = match_tracked_indices_across_sessions(one, eids[0], eids[1:], roicat_root=roicat_root, filter_neurons=filter_neurons)
        if eids[0] not in idx_map:
            raise RuntimeError("Tracking returned no indices for anchor train EID.")
        lengths = [int(np.asarray(idx_map.get(e, np.empty(0, int)), int).size) for e in eids]
        if not lengths or min(lengths) == 0:
            raise RuntimeError("Shared-neuron intersection is empty across selected sessions.")
        n_shared = int(min(lengths))
        for e in eids:
            arr = np.asarray(idx_map[e], dtype=int)
            if arr.size < n_shared:
                raise RuntimeError(f"mapping produced {arr.size} < n_shared({n_shared}) for {e}.")
            idx_map[e] = arr[:n_shared]
        train_eid = eid_train
        test_eids = list(eid_test)
    else:
        subset_map = best_eid_subsets_for_animal(
            subject,
            one=one,
            roicat_root=roicat_root,
            k_min=min_sessions,
            k_max=min_sessions,
            n_starts=10,
            random_starts=5,
            enforce_monotone=True,
            min_trials=400,
            trial_key="stimOn_times",
        )
        if not subset_map:
            raise RuntimeError(f"{subject}: best_eid_subsets_for_animal returned no subsets.")
        k = sorted(subset_map.keys())[0]
        eids = list(subset_map[k]["eids"])
        n_shared = int(subset_map[k]["n_shared"])
        if len(eids) < min_sessions:
            raise RuntimeError(f"{subject}: only {len(eids)} sessions returned; need ≥ {min_sessions}.")
        if n_shared < min_k_shared:
            raise RuntimeError(f"{subject}: n_shared={n_shared} < required {min_k_shared}.")
        eids = sorted(eids, key=_date_of)
        dates = [_date_of(e) for e in eids]
        print(f"[subject] {subject}")
        print(f"[sessions] {len(eids)} sessions; global shared neurons k = {n_shared}")
        for i, (e, d) in enumerate(zip(eids, dates)):
            print(f"  {i:02d}  {d}  {e}")
        idx_map = match_tracked_indices_across_sessions(one, eids[0], eids[1:], roicat_root=roicat_root, filter_neurons=filter_neurons)
        if eids[0] not in idx_map:
            idx_map[eids[0]] = np.arange(n_shared, dtype=int)
        for e in eids:
            arr = np.asarray(idx_map[e], dtype=int)
            if arr.size < n_shared:
                raise RuntimeError(f"mapping produced {arr.size} < n_shared({n_shared}) for {e}.")
            idx_map[e] = arr[:n_shared]
        train_eid = eids[0]
        test_eids = eids[1:]

    def make_pipeline(n_features: int, n_train_samples: int):
        steps = [("scaler", StandardScaler(with_mean=True, with_std=True))]
        if dimreduce == "pca":
            steps.append(("pca", PCACap(dim_k=dim_k, random_state=seed, svd_solver="randomized")))
        elif dimreduce == "selectk":
            k_sel = max(2, min(dim_k, n_features))
            steps.append(("skb", SelectKBest(score_func=f_classif, k=k_sel)))
        clf = LogisticRegression(
            penalty=("elasticnet" if use_elasticnet else "l2"),
            solver=("saga" if use_elasticnet else "liblinear"),
            max_iter=max_iter,
            random_state=seed,
        )
        steps.append(("clf", clf))
        return Pipeline(steps)

    results = {t: [] for t in targets}
    first_session_cv = {t: dict(mean_acc=np.nan, fold_accs=[], picked_fold=-1) for t in targets}

    print(f"training on {train_eid}")
    print(f"testing on {test_eids} sessions")
    restrict_train = np.asarray(idx_map[train_eid], dtype=int)
    for t in targets:
        try:
            X, y, _ = build_trial_features(train_eid, target=t, restrict=restrict_train, rerun=rerun)
        except Exception as ex:
            print(f"[skip] train features {train_eid} [{t}]: {type(ex).__name__}: {ex}")
            continue

        min_class = int(np.min(np.bincount(y)))
        outer_splits = max(2, min(3, min_class))
        outer_skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

        fold_accs, fold_models = [], []
        for fidx, (tr_idx, va_idx) in enumerate(outer_skf.split(X, y)):
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xva, yva = X[va_idx], y[va_idx]

            base = make_pipeline(n_features=Xtr.shape[1], n_train_samples=Xtr.shape[0])
            param_grid = {"clf__C": list(Cs)}
            if use_elasticnet:
                param_grid["clf__l1_ratio"] = list(l1_ratio_grid)

            min_class_tr = int(np.min(np.bincount(ytr)))
            inner_splits_eff = max(2, min(inner_cv_splits, min_class_tr))
            inner_skf = StratifiedKFold(n_splits=inner_splits_eff, shuffle=True, random_state=seed)
            gs = GridSearchCV(base, param_grid=param_grid, scoring="accuracy", cv=inner_skf, refit=True, n_jobs=1, verbose=0)
            gs.fit(Xtr, ytr)
            model = gs.best_estimator_
            yhat = model.predict(Xva)
            acc = float(accuracy_score(yva, yhat))
            fold_accs.append(acc)
            fold_models.append(model)

            del Xtr, ytr, Xva, yva, gs, model, yhat
            gc.collect()

        if fold_accs:
            order = np.argsort(fold_accs)
            picked_fold = int(order[len(order) // 2])
            transfer_model = fold_models[picked_fold]
            mean_cv_acc = float(np.mean(fold_accs))
            first_session_cv[t]["mean_acc"] = mean_cv_acc
            first_session_cv[t]["fold_accs"] = [float(a) for a in fold_accs]
            first_session_cv[t]["picked_fold"] = picked_fold

            results[t].append(dict(
                eid=train_eid, date=_eid_date(train_eid), which="train_session_cv",
                n_trials=int(len(y)), acc=mean_cv_acc, acc_shuffle=np.nan, k=int(len(restrict_train)), picked_fold=picked_fold,
            ))
        else:
            print(f"[warn] no CV folds available for {train_eid} [{t}]")
            transfer_model = None

        del X, y
        gc.collect()

        if transfer_model is None:
            continue
        for e in test_eids:
            restrict = np.asarray(idx_map[e], dtype=int)
            try:
                Xte, yte, _ = build_trial_features(e, target=t, restrict=restrict, rerun=rerun)
            except Exception as ex:
                print(f"[skip] test features {e} [{t}]: {type(ex).__name__}: {ex}")
                continue
            try:
                yhat = transfer_model.predict(Xte)
                acc = float(accuracy_score(yte, yhat))
                sh = [float(accuracy_score(_permute_labels(rng, yte), yhat)) for _ in range(n_shuffle)]
                acc_shuf = float(np.mean(sh))
            except Exception as ex:
                print(f"[warn] inference failed on {e} [{t}]: {type(ex).__name__}: {ex}")
                acc, acc_shuf = np.nan, np.nan

            results[t].append(dict(
                eid=e, date=_eid_date(e), which="test",
                n_trials=int(len(yte)), acc=acc, acc_shuffle=acc_shuf, k=int(len(restrict)), from_fold=first_session_cv[t]["picked_fold"],
            ))
            del Xte, yte
            gc.collect()
        del transfer_model
        gc.collect()

    payload = dict(
        subject=subject,
        eids=[train_eid] + list(test_eids),
        train_eid=train_eid,
        dates=[_eid_date(e) for e in [train_eid] + list(test_eids)],
        n_shared=int(len(np.asarray(idx_map[train_eid], int))),
        targets=list(targets),
        results=results,
        first_session_cv=first_session_cv,
        protocol="trainFirst_streamed_outer3fold_keepMedianFold",
        dimreduce=dimreduce,
        dim_k=dim_k,
        use_elasticnet=use_elasticnet,
        Cs=list(Cs),
        l1_ratio_grid=list(l1_ratio_grid),
        inner_cv_splits=inner_cv_splits,
        seed=seed,
    )

    prefix = "trainFirst_stream_" + "_".join([e[:3] for e in [train_eid] + list(test_eids)])
    pkl_path = out_dir / f"{prefix}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    json_path = out_dir / f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {pkl_path}")
    print(f"[saved] {json_path}")
    return payload


# ===================================
# Plotting for the train-first payload
# ===================================

def load_trainFirst_payload(eid_train: str, eid_test: Sequence[str], one: Optional[ONE] = None, prefer: Literal["json", "pkl"] = "json") -> dict:
    if one is None:
        one = ONE()
    eids = [eid_train] + list(eid_test)
    prefix = "trainFirst_stream_" + "_".join([e[:3] for e in eids])
    out_dir = Path(one.cache_dir) / "meso" / "decoding" / "cross_session_trainFirst"
    json_path = out_dir / f"{prefix}.json"
    pkl_path = out_dir / f"{prefix}.pkl"
    if prefer == "json" and json_path.exists():
        return json.loads(json_path.read_text())
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    if json_path.exists():
        return json.loads(json_path.read_text())
    raise FileNotFoundError(f"No payload found for {prefix}")


def plot_trainFirst(payload: dict, targets: Sequence[Target] = ("choice", "feedback", "stimulus", "block")):
    eids = payload["eids"]
    dates = payload["dates"]
    results = payload["results"]
    first_cv = payload["first_session_cv"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    order = ["choice", "feedback", "stimulus", "block"]
    order = [t for t in order if t in targets] + [t for t in targets if t not in order]

    for ax, t in zip(axes, order):
        rows = results.get(t, [])
        if not rows:
            ax.set_visible(False)
            continue
        xs, ys, ysh = [], [], []
        for r in rows:
            xs.append(r.get("date", ""))
            ys.append(r.get("acc", np.nan))
            ysh.append(r.get("acc_shuffle", np.nan))
        xs_tick = np.arange(len(xs))
        ax.plot(xs_tick, ys, marker="o", label="acc")
        if np.isfinite(np.nanmean(ysh)):
            ax.plot(xs_tick, ysh, linestyle="--", label="shuffle")
        ax.set_xticks(xs_tick)
        ax.set_xticklabels(xs, rotation=45, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(t)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        cv = first_cv.get(t, {})
        if cv:
            ax.annotate(f"train-CV={cv.get('mean_acc', np.nan):.2f}", xy=(0, ys[0]), xytext=(5, 8), textcoords="offset points", fontsize=8)

    fig.suptitle("Train-first transfer accuracy across sessions", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


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


# In [1]: run Dropbox/scripts/IBL/meso_chronic.py

#    ...:     "SP072/2025-08-22/001",
#    ...:     "SP072/2025-08-26/001",
#    ...:     "SP072/2025-08-27/002",
#    ...:     "SP072/2025-08-28/001",
#    ...:     "SP072/2025-09-02/002",
#    ...:     "SP072/2025-09-03/001",
#    ...:     "SP072/2025-09-04/001",
#    ...:     "SP072/2025-09-05/001",
#    ...: ]
#    ...: 
#    ...: eids = [one.path2eid(p) for p in paths]

# In [3]: res = match_tracked_indices_across_sessions(one,eids[0], eids[1:])
