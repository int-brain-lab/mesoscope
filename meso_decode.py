
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Optional, List, Literal
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

Target = Literal["choice", "feedback", "stimulus"]

# =============
# Helper utils
# =============

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
        event = "firstMovement_times"; win = (0.0, 0.15)
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
    ks: Sequence[int] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    R: int = 50,
    metric: str = "auc",
    cv_splits: int = 5,
    seed: int = 0,
    ceiling_mode: str = "maxk",
    n_label_shuffles: int = 0,
    shuffle_seed: Optional[int] = None,
) -> NDCResult:
    rng = np.random.default_rng(seed)
    N = X.shape[1]
    ks = np.array(sorted([k for k in ks if 1 <= k <= N]), dtype=int)
    if ks.size == 0:
        raise ValueError("No valid k in ks <= number of neurons")

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    def make_clf():
        return LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000)

    scores: Dict[int, np.ndarray] = {}
    for k in ks:
        sc = np.empty(R, dtype=float)
        for r in range(R):
            cols = rng.choice(N, size=k, replace=False)
            sc[r] = _cv_score(make_clf(), Xs[:, cols], y, metric=metric, n_splits=cv_splits, seed=seed + r)
        scores[k] = sc
        print(f"k={k:4d} | {metric} mean={sc.mean():.3f} ± {sc.std(ddof=1):.3f}")

    if ceiling_mode == "all" and N > ks.max():
        ceiling = _cv_score(make_clf(), Xs, y, metric=metric, n_splits=cv_splits, seed=seed + 12345)
    else:
        ceiling = float(np.mean(scores[ks.max()]))

    shuffled: Optional[Dict[int, np.ndarray]] = None
    if n_label_shuffles and n_label_shuffles > 0:
        shuffled = {}
        rng_shuf = np.random.default_rng(seed if shuffle_seed is None else shuffle_seed)
        for k in ks:
            scs = np.empty(n_label_shuffles, dtype=float)
            for s in range(n_label_shuffles):
                cols = rng_shuf.choice(N, size=k, replace=False)
                y_perm = rng_shuf.permutation(y)
                scs[s] = _cv_score(make_clf(), Xs[:, cols], y_perm, metric=metric, n_splits=cv_splits, seed=seed + 10_000 + s)
            shuffled[k] = scs
            print(f"[shuffle] k={k:4d} | {metric} mean={scs.mean():.3f} ± {scs.std(ddof=1):.3f}")

    return NDCResult(ks=ks, scores=scores, metric=metric, ceiling=ceiling, shuffled=shuffled)

# ============================
# 2) Save/Load (fast plotting)
# ============================

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

def _cache_name(eid: str, target: Target) -> str:
    return f"{eid}__{target}.pkl"

def save_ndc_result(cache_dir: Path, eid: str, target: Target, res: NDCResult, meta: dict):
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / _cache_name(eid, target)
    payload = dict(eid=eid, target=target, result=res, meta=meta,
                   saved_at=datetime.utcnow().isoformat(timespec="seconds") + "Z")
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    index_path = cache_dir / "index.json"
    idx = []
    if index_path.exists():
        try:
            idx = json.loads(index_path.read_text())
        except Exception:
            idx = []
    brief = dict(
        eid=eid, target=target, date=meta.get("date"), n_shared=meta.get("n_shared"),
        ks_max=int(meta.get("ks_max")) if meta.get("ks_max") is not None else None,
        metric=res.metric, ceiling=float(res.ceiling),
        perf=float(meta.get("performance", np.nan)), trials=int(meta.get("trials", 0)),
    )
    idx = [b for b in idx if not (b.get("eid")==eid and b.get("target")==target)] + [brief]
    index_path.write_text(json.dumps(idx, indent=2))

def load_ndc_result(cache_dir: Path, eid: str, target: Target):
    p = cache_dir / _cache_name(eid, target)
    with open(p, "rb") as f:
        return pickle.load(f)

def list_cached(cache_dir: Path, target: Optional[Target]=None) -> List[dict]:
    p = cache_dir / "index.json"
    if not p.exists():
        return []
    rows = json.loads(p.read_text())
    if target is None:
        return rows
    return [r for r in rows if r.get("target")==target]

# ===================================
# 3) High-level pipeline (compute & save)
# ===================================

def compute_and_cache_ndcs(
    eids: Sequence[str],
    targets: Sequence[Target] = ("choice", "feedback", "stimulus"),
    roicat_root: Path = Path.home() / "chronic_csv",
    cache_dir: Path = Path.home() / "ndc_cache",
    ks: tuple = (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    R: int = 50,
    metric: str = "auc",
    cv_splits: int = 5,
    equalize_trials: bool = True,
    seed: int = 0,
    ceiling_mode: str = "maxk",
    n_label_shuffles: int = 100,
    shuffle_seed: Optional[int] = None,
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

    ok = [(e,t) for (e,t) in Xy_map.keys()]
    if len(ok) == 0:
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

    # Compute + save
    metas = {}
    for e in eids:
        for t in targets:
            if (e,t) not in Xy_map:
                continue
            X, y, (event, win) = Xy_map[(e,t)]
            res = neuron_dropping_curve(
                X, y, ks=ks_grid, R=R, metric=metric, cv_splits=cv_splits, seed=seed,
                ceiling_mode=ceiling_mode, n_label_shuffles=n_label_shuffles, shuffle_seed=shuffle_seed
            )
            perf = session_performance(e)
            meta = dict(
                eid=e, target=t, date=_eid_date(e), n_shared=n_shared,
                ks_max=int(max(res.ks)), trials=len(y), performance=float(perf),
                event=event, win=tuple(win), equalize_trials=bool(equalize_trials),
            )
            save_ndc_result(cache_dir, e, t, res, meta)
            metas[(e,t)] = meta
            print(f"[saved] {e} [{t}]  date={meta['date']}  perf={perf:.3f}  ceiling={res.ceiling:.3f}")
    return metas

# =====================
# 4) Plotting from cache
# =====================

@dataclass
class _PlotCfg:
    title: str = ""
    delta: float = 0.02
    show_shuffle: bool = True

def summarize_k_star(res: NDCResult, delta: float = 0.02) -> Optional[int]:
    target = res.ceiling - float(delta)
    means = {k: float(v.mean()) for k, v in res.scores.items()}
    ks_ok = sorted([k for k, m in means.items() if m >= target])
    return ks_ok[0] if ks_ok else None

def plot_ndc_overlay_from_cache(cache_dir: Path, target: Target = "choice",
                                eids: Optional[Sequence[str]] = None,
                                title: Optional[str] = None,
                                delta: float = 0.02,
                                show_shuffle: bool = True):
    idx = list_cached(cache_dir, target=target)
    if eids is None:
        eids = [d["eid"] for d in sorted(idx, key=lambda d: d.get("date",""))]
    else:
        eids = list(eids)

    if title is None:
        title = f"{target.capitalize()} decoding — neuron-dropping curves"

    plt.figure(figsize=(7.6, 4.8))
    last_payload = None
    for i, e in enumerate(eids):
        payload = load_ndc_result(cache_dir, e, target)
        last_payload = payload
        res: NDCResult = payload["result"]
        meta = payload["meta"]

        ks_arr = res.ks
        means = np.array([res.scores[k].mean() for k in ks_arr])
        sds   = np.array([res.scores[k].std(ddof=1) for k in ks_arr])
        ns    = np.array([len(res.scores[k]) for k in ks_arr], dtype=int)
        cis   = 1.96 * sds / np.sqrt(np.maximum(1, ns))

        k_star = summarize_k_star(res, delta=delta)
        label = f"{meta.get('date','NA')} (k*={k_star})" if k_star is not None else f"{meta.get('date','NA')} (k*=NA)"
        plt.errorbar(ks_arr, means, yerr=cis, fmt="-o", capsize=3, label=label, zorder=3)

        if show_shuffle and res.shuffled is not None:
            sh_means = np.array([res.shuffled[k].mean() for k in ks_arr])
            sh_sds   = np.array([res.shuffled[k].std(ddof=1) for k in ks_arr])
            sh_ns    = np.array([len(res.shuffled[k]) for k in ks_arr], dtype=int)
            sh_cis   = 1.96 * sh_sds / np.sqrt(np.maximum(1, sh_ns))
            if i == 0:
                plt.fill_between(ks_arr, sh_means - sh_cis, sh_means + sh_cis, alpha=0.08, color="gray", label="shuffle mean ± 95% CI", zorder=1)
                plt.plot(ks_arr, sh_means, "--", lw=1.0, color="gray", zorder=2)
            else:
                plt.fill_between(ks_arr, sh_means - sh_cis, sh_means + sh_cis, alpha=0.06, color="gray", zorder=1)
                plt.plot(ks_arr, sh_means, "--", lw=1.0, color="gray", zorder=2)

    plt.xscale("log", base=2)
    plt.xlabel("number of neurons (k)")
    ylabel = (last_payload["result"].metric.upper() if last_payload else "SCORE")
    plt.ylabel(ylabel)
    plt.title(title + " (shared cells, equalized trials)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def plot_score_vs_performance_from_cache(cache_dir: Path, target: Target = "choice",
                                         eids: Optional[Sequence[str]] = None,
                                         use_metric: Optional[str] = None,
                                         title: Optional[str] = None):
    if title is None:
        title = f"{target.capitalize()} decoding (kmax) vs behavioral performance"

    idx = list_cached(cache_dir, target=target)
    if eids is None:
        eids = [d["eid"] for d in sorted(idx, key=lambda d: d.get("date",""))]
    else:
        eids = list(eids)

    xs, ys, labels = [], [], []
    metric = None
    for e in eids:
        payload = load_ndc_result(cache_dir, e, target)
        res: NDCResult = payload["result"]
        meta = payload["meta"]
        km = int(max(res.ks))
        score_mean = float(res.scores[km].mean())
        perf = float(meta.get("performance", np.nan))
        xs.append(perf); ys.append(score_mean); labels.append(meta.get("date","NA"))
        metric = res.metric if use_metric is None else use_metric

    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    plt.figure(figsize=(5.8, 4.8))
    plt.scatter(xs, ys)
    msk = np.isfinite(xs) & np.isfinite(ys)
    if np.sum(msk) >= 2:
        Xd = np.column_stack([np.ones(np.sum(msk)), xs[msk]])
        beta, *_ = np.linalg.lstsq(Xd, ys[msk], rcond=None)
        xr = np.linspace(np.nanmin(xs), np.nanmax(xs), 100)
        yr = beta[0] + beta[1] * xr
        plt.plot(xr, yr, linestyle="--")
        r = np.corrcoef(xs[msk], ys[msk])[0,1]
        plt.annotate(f"r = {r:.2f}", xy=(0.05, 0.95), xycoords="axes fraction", va="top")

    for (x, y, lab) in zip(xs, ys, labels):
        if np.isfinite(x) and np.isfinite(y):
            plt.annotate(lab, (x, y), textcoords="offset points", xytext=(4,4), fontsize=8)

    plt.xlabel("Session performance  (mean(feedbackType == 1))")
    plt.ylabel(f"{(metric or 'SCORE').upper()} at k_max")
    plt.title(title + "  (largest neuron count only)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =========================
# Convenience entrypoint
# =========================

def neuron_dropping_curves_cached(
    eids: Sequence[str],
    targets: Sequence[Target] = ("choice", "feedback", "stimulus"),
    roicat_root: Path = Path.home() / "chronic_csv",
    cache_dir: Path = Path.home() / "ndc_cache",
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
):
    metas = compute_and_cache_ndcs(
        eids=eids, targets=targets, roicat_root=roicat_root, cache_dir=cache_dir,
        ks=ks, R=R, metric=metric, cv_splits=cv_splits, equalize_trials=equalize_trials,
        seed=seed, ceiling_mode=ceiling_mode, n_label_shuffles=n_label_shuffles, shuffle_seed=shuffle_seed,
    )

    for t in targets:
        eids_avail = [e for e in eids if (e,t) in metas]
        if not eids_avail:
            continue
        plot_ndc_overlay_from_cache(cache_dir, target=t, eids=eids_avail, delta=delta, show_shuffle=True)
        plot_score_vs_performance_from_cache(cache_dir, target=t, eids=eids_avail)

