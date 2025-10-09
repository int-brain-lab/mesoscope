import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
from one.api import ONE


MESO_DIR = Path.home() / "Dropbox/scripts/IBL"
if str(MESO_DIR) not in sys.path:
    sys.path.insert(0, str(MESO_DIR))
from meso import get_win_times, load_or_embed
from meso_chronic import match_tracked_indices_across_sessions

one = ONE()


# ---------------------------
# 1) Trial-feature extraction
# ---------------------------

def build_trial_features_choice(
    eid: str,
    restrict: Optional[np.ndarray] = None,
    event: str = "firstMovement_times",
    win: Tuple[float, float] = (0.0, 0.15),
    class_balance: str = "downsample",  # 'downsample' | 'none'
    scaling: bool = True,               # per-neuron z-score across trials
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns X (n_trials x n_neurons), y in {0,1} for choice decoding.
    Window is [event + win[0], event + win[1]] in seconds; trials with choice==0 are dropped.
    """
    rr = load_or_embed(eid, restrict=restrict)
    times = rr['roi_times'][0]
    T = times.size

    trials, _tts = get_win_times(eid)
    # valid trials: event is finite, choice is ±1
    event_times = np.asarray(trials[event], float)
    choice = np.asarray(trials['choice'], float)   # -1, 0, 1
    valid = np.isfinite(event_times) & np.isfinite(choice) & (choice != 0)
    if not np.any(valid):
        raise RuntimeError(f"{eid}: no valid trials for event '{event}' and choice decoding")

    event_times = event_times[valid]
    y = (choice[valid] == 1).astype(int)  # 1 = right, 0 = left (adjust if you prefer)

    # per-trial window bounds
    starts = np.clip(event_times + win[0], times[0], times[-1])
    ends   = np.clip(event_times + win[1], times[0], times[-1])
    i0 = np.searchsorted(times, starts, side='left')
    i1 = np.searchsorted(times, ends,   side='right')  # exclusive
    # ensure at least one sample
    i1 = np.maximum(i1, i0 + 1)

    # build features: mean over window for each neuron
    sig = rr['roi_signal']  # (N, T)
    N = sig.shape[0]
    X = np.empty((len(event_times), N), dtype=np.float32)
    for t in range(len(event_times)):
        X[t] = sig[:, i0[t]:i1[t]].mean(axis=1)

    # optional per-neuron z-score across trials
    if scaling:
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)

    # class balance
    if class_balance == "downsample":
        idx0 = np.flatnonzero(y == 0)
        idx1 = np.flatnonzero(y == 1)
        n = min(len(idx0), len(idx1))
        if n < 2:
            raise RuntimeError(f"{eid}: not enough trials after balancing (min class count = {n})")
        rng = np.random.default_rng(0)
        keep = np.concatenate([rng.choice(idx0, n, replace=False),
                               rng.choice(idx1, n, replace=False)])
        keep.sort()
        X, y = X[keep], y[keep]

    return X.astype(np.float32), y.astype(int)


# ------------------------------------
# 2) Neuron-dropping curves (many k's)
# ------------------------------------

@dataclass
class NDCResult:
    ks: np.ndarray                 # array of k values
    scores: Dict[int, np.ndarray]  # k -> array of scores shape (R,)
    metric: str                    # 'auc' or 'acc'
    ceiling: float                 # accuracy using max k (or all), CV estimate
    shuffled: Optional[Dict[int, np.ndarray]] = None  # k -> array (n_label_shuffles,)

def _cv_score(clf, X, y, metric: str, n_splits: int, seed: int) -> float:
    n_splits = max(2, min(n_splits, np.min(np.bincount(y))))  # guard
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    outs = []
    for tr, te in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        if metric == "auc":
            # use decision_function if available else proba
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
    ks: Sequence[int] = (8, 16, 32, 64, 128, 256, 512),
    R: int = 50,
    metric: str = "auc",
    cv_splits: int = 5,
    seed: int = 0,
    ceiling_mode: str = "maxk",
    n_label_shuffles: int = 0,
    shuffle_seed: Optional[int] = None,
) -> NDCResult:
    """
    For each k, sample R random neuron sets, fit decoder with CV (true labels).
    If n_label_shuffles>0, also compute a shuffled-label baseline per k
    by permuting y (independently per shuffle) and re-running CV once per shuffle.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[1]
    ks = np.array(sorted([k for k in ks if 1 <= k <= N]), dtype=int)
    if ks.size == 0:
        raise ValueError("No valid k in ks <= number of neurons")

    # Standardize once
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    def make_clf():
        return LogisticRegression(
            penalty="l2", solver="liblinear", max_iter=2000, class_weight=None
        )

    # --- true-label scores (R per k) ---
    scores: Dict[int, np.ndarray] = {}
    for k in ks:
        sc = np.empty(R, dtype=float)
        for r in range(R):
            cols = rng.choice(N, size=k, replace=False)
            sc[r] = _cv_score(make_clf(), Xs[:, cols], y,
                              metric=metric, n_splits=cv_splits, seed=seed + r)
        scores[k] = sc
        print(f"k={k:4d} | {metric} mean={sc.mean():.3f} ± {sc.std(ddof=1):.3f}")

    # --- ceiling ---
    if ceiling_mode == "all" and N > ks.max():
        ceiling = _cv_score(make_clf(), Xs, y, metric=metric,
                            n_splits=cv_splits, seed=seed + 12345)
    else:
        ceiling = float(np.mean(scores[ks.max()]))

    # --- shuffled-label baseline (S per k) ---
    shuffled: Optional[Dict[int, np.ndarray]] = None
    if n_label_shuffles and n_label_shuffles > 0:
        shuffled = {}
        rng_shuf = np.random.default_rng(seed if shuffle_seed is None else shuffle_seed)
        for k in ks:
            scs = np.empty(n_label_shuffles, dtype=float)
            for s in range(n_label_shuffles):
                cols = rng_shuf.choice(N, size=k, replace=False)
                y_perm = rng_shuf.permutation(y)
                scs[s] = _cv_score(make_clf(), Xs[:, cols], y_perm,
                                   metric=metric, n_splits=cv_splits, seed=seed + 10_000 + s)
            shuffled[k] = scs
            print(f"[shuffle] k={k:4d} | {metric} mean={scs.mean():.3f} ± {scs.std(ddof=1):.3f}")

    return NDCResult(ks=ks, scores=scores, metric=metric, ceiling=ceiling, shuffled=shuffled)


# -------------------------
# 3) Summaries and plotting
# -------------------------

def summarize_k_star(res: NDCResult, delta: float = 0.02) -> Optional[int]:
    """
    k* = smallest k whose mean score >= ceiling - delta.
    Returns None if no k reaches the target.
    """
    target = res.ceiling - float(delta)
    means = {k: float(v.mean()) for k, v in res.scores.items()}
    ks_ok = sorted([k for k, m in means.items() if m >= target])
    return ks_ok[0] if ks_ok else None

def plot_ndc(res: NDCResult, title: str = "", show_shuffle: bool = True):
    ks = res.ks
    means = np.array([res.scores[k].mean() for k in ks])
    sds   = np.array([res.scores[k].std(ddof=1) for k in ks])
    ns    = np.array([len(res.scores[k]) for k in ks], dtype=int)
    cis   = 1.96 * sds / np.sqrt(np.maximum(1, ns))

    plt.figure(figsize=(6.4, 4.2))
    # true labels
    plt.errorbar(ks, means, yerr=cis, fmt="-o", capsize=3, label="true", zorder=3)
    plt.axhline(res.ceiling, ls="--", lw=1, color="gray",
                label=f"ceiling ({res.metric}={res.ceiling:.3f})", zorder=2)

    # shuffled baseline
    if show_shuffle and res.shuffled is not None:
        sh_means = np.array([res.shuffled[k].mean() for k in ks])
        sh_sds   = np.array([res.shuffled[k].std(ddof=1) for k in ks])
        sh_ns    = np.array([len(res.shuffled[k]) for k in ks], dtype=int)
        sh_cis   = 1.96 * sh_sds / np.sqrt(np.maximum(1, sh_ns))
        # band
        plt.fill_between(ks, sh_means - sh_cis, sh_means + sh_cis,
                         alpha=0.2, color="tab:gray", label="shuffle 95% CI", zorder=1)
        plt.plot(ks, sh_means, "--", color="tab:gray", lw=1.2, label="shuffle mean", zorder=2)

    plt.xscale("log", base=2)
    plt.xticks(ks, ks)
    plt.xlabel("number of neurons (k)")
    plt.ylabel(res.metric.upper())
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

########################################################

    # eids = ['38bc2e7c-d160-4887-86f6-4368bfd58c5f',
    # '74ffa405-3e23-47d9-972b-11bea1c3c2f6',
    # '1322edbf-5c42-4b9a-aecd-7ddaf4f44387',
    # '20ebc2b9-5b4c-42cd-8e4b-65ddb427b7ff']


def neuron_dropping_curves(
    eids,
    roicat_root: Path = Path.home() / "chronic_csv",
    event: str = "firstMovement_times",
    win: tuple = (0.0, 0.15),
    ks: tuple = (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    R: int = 50,
    metric: str = "auc",
    cv_splits: int = 5,
    delta: float = 0.02,
    equalize_trials: bool = True,
    seed: int = 0,
    ceiling_mode: str = "maxk",
):
    """
    For the given EIDs, align the SAME neurons across sessions and compute
    neuron-dropping curves (NDCs) for choice decoding. Plots an overlay and
    prints a per-session efficiency summary (k*).

    Returns a dict with results and metadata.
    """
    # ---- 0) chronological order helpers ----
    def _eid_date(e):
        try:
            meta = one.alyx.rest('sessions', 'read', id=e)
            return str(meta['start_time'])[:10]
        except Exception:
            p = one.eid2path(e)
            return p.parent.name if p else "9999-99-99"

    eids = sorted(list(eids), key=_eid_date)
    if len(eids) < 2:
        raise ValueError("Provide at least two EIDs.")

    # ---- 1) align shared neurons across all sessions ----
    idx_map = match_tracked_indices_across_sessions(
        one, eids[0], eids[1:], roicat_root=roicat_root
    )
    # Sanity + shared count
    lens = {e: int(idx_map[e].size) for e in eids}
    if len(set(lens.values())) != 1:
        # Shouldn't happen with the helper, but guard anyway
        n_shared = min(lens.values())
        for e in eids:
            if lens[e] != n_shared:
                idx_map[e] = idx_map[e][:n_shared]
    n_shared = int(idx_map[eids[0]].size)
    if n_shared == 0:
        raise RuntimeError("No shared neurons across the provided sessions.")

    # ---- 2) choose k grid up to n_shared (ensure it’s non-empty and ends at n_shared) ----
    ks_grid = tuple(sorted({k for k in ks if 1 <= k <= n_shared}))
    if not ks_grid:
        # fallback: powers of two up to n_shared, ensure final point is n_shared
        k = 2
        ks_grid = []
        while k < n_shared:
            ks_grid.append(k)
            k *= 2
        ks_grid.append(n_shared)
        ks_grid = tuple(ks_grid)
    else:
        # ensure the largest point is n_shared (better “ceiling” comparability)
        if ks_grid[-1] != n_shared:
            ks_grid = tuple(list(ks_grid) + [n_shared])

    # ---- 3) build trial features (same shared neurons) ----
    def make_Xy(eid):
        X, y = build_trial_features_choice(
            eid,
            restrict=idx_map[eid],
            event=event,
            win=win,
            class_balance="downsample",
            scaling=True,
        )
        return X, y

    Xy = {}
    failed = []
    for e in eids:
        try:
            Xy[e] = make_Xy(e)
        except Exception as ex:
            print(f"[skip] {e}: {type(ex).__name__}: {ex}")
            failed.append(e)
    if len(Xy) < 2:
        raise RuntimeError("Fewer than two sessions yielded valid trial features.")

    eids = [e for e in eids if e in Xy]  # drop failed ones while preserving order

    # ---- 4) (optional) equalize trial counts across sessions ----
    if equalize_trials:
        n_common = min(len(y) for (_, y) in Xy.values())
        rng = np.random.default_rng(seed)
        for e in eids:
            X, y = Xy[e]
            if len(y) > n_common:
                keep = rng.choice(len(y), size=n_common, replace=False)
                Xy[e] = (X[keep], y[keep])

    # ---- 5) compute neuron-dropping curves per session ----
    ndc = {}
    for e in eids:
        X, y = Xy[e]
        ndc[e] = neuron_dropping_curve(
            X, y,
            ks=ks_grid,
            R=R,
            metric=metric,
            cv_splits=cv_splits,
            seed=seed,
            ceiling_mode=ceiling_mode,
            n_label_shuffles=100,          # << add: 100 shuffled-label baselines per k
            shuffle_seed=seed + 999,       # << add: independent RNG for permutations
        )

    # ---- 6) summarize efficiency (k*) and plot overlay ----
    kstars = []
    dates = [_eid_date(e) for e in eids]

    plt.figure(figsize=(7.6, 4.8))
    for e, d in zip(eids, dates):
        res = ndc[e]
        k_star = summarize_k_star(res, delta=delta)
        kstars.append(k_star if k_star is not None else np.nan)

        ks_arr = res.ks
        means = np.array([res.scores[k].mean() for k in ks_arr])
        sds   = np.array([res.scores[k].std(ddof=1) for k in ks_arr])
        ns    = np.array([len(res.scores[k]) for k in ks_arr], dtype=int)
        cis   = 1.96 * sds / np.sqrt(np.maximum(1, ns))

        label = f"{d} (k*={k_star})" if k_star is not None else f"{d} (k*=NA)"
        plt.errorbar(ks_arr, means, yerr=cis, fmt='-o', capsize=3, label=label)
        if res.shuffled is not None:
            sh_means = np.array([res.shuffled[k].mean() for k in ks_arr])
            sh_sds   = np.array([res.shuffled[k].std(ddof=1) for k in ks_arr])
            sh_ns    = np.array([len(res.shuffled[k]) for k in ks_arr], dtype=int)
            sh_cis   = 1.96 * sh_sds / np.sqrt(np.maximum(1, sh_ns))
            lbl = "shuffle mean ± 95% CI" if e == eids[0] else None
            plt.fill_between(ks_arr, sh_means - sh_cis, sh_means + sh_cis,
                            alpha=0.08, color="gray", label=lbl)
            plt.plot(ks_arr, sh_means, "--", lw=1.0, color="gray")

            plt.xscale("log", base=2)
            plt.xlabel("number of neurons (k)")
            plt.ylabel(metric.upper())
            plt.title("Choice decoding — neuron-dropping curves (shared cells, equalized trials)")
            plt.grid(True, alpha=0.3)
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.show()

    # ---- 7) table summary ----
    print("\nSession efficiency summary (smaller k* = more efficient):")
    for e, d, kstar in zip(eids, dates, kstars):
        ceil = ndc[e].ceiling
        print(f"{d}   {e}   k*={kstar}   ceiling={ceil:.3f}")

    return {
        "eids_ordered": eids,
        "dates": dates,
        "n_shared": n_shared,
        "ks_grid": ks_grid,
        "kstars": kstars,
        "ndc": ndc,                         # dict: eid -> NDCResult
        "trials_per_session": {e: len(Xy[e][1]) for e in eids},
        "failed": failed,
        "params": dict(
            event=event, win=win, R=R, metric=metric, cv_splits=cv_splits,
            delta=delta, equalize_trials=equalize_trials, seed=seed,
            ceiling_mode=ceiling_mode
        ),
    }