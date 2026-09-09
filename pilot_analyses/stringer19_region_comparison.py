"""Does SVCA structure vary by brain region? Two comparisons, both matched
for neuron count and recording duration across sessions/regions so
differences reflect the region, not sample size:

1. **Dimensionality / power-law decay** ("Tier 1", cheap -- no video):
   for several sessions per region, restrict SVCA to that region's neurons
   only and compute the reliable-variance spectrum (via
   `stringer19_svca_prediction.compute_reliable_variance`). Fit a power-law
   exponent to each spectrum (adapted from MouseLand/critical_init's
   `fit_powerlaw_exp` -- a weighted log-log linear fit, same idea as this
   repo's own SVCA-efficiency work) and compute how many top dimensions
   are needed to reach half of the captured reliable variance ("effective
   dimensionality"). Compare exponents and effective-dimensionality across
   regions.

2. **Behavioral/task contribution by region** ("Tier 2", needs video):
   for one session per region, restrict `compute_svca_prediction` to that
   region's neurons and compare the video-PC / wheel+whisker / block /
   choice variance-explained curves across regions. Reuses this repo's
   video-PC cache -- if a session has enough neurons in two target
   regions, the same cached video answers both, with zero extra decoding.

Matching: a fixed neuron count (`N_TOTAL`, subsampled per region/session
with a fixed seed) and a fixed time window (`WINDOW`) are used everywhere,
found in `stringer19_region_scan.py`'s reconnaissance to be safely below
every candidate session's actual region-neuron-count and duration.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

from one.api import ONE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meso_loader import load_mesoscope_session, MesoscopeSession  # noqa: E402
from stringer19_svca_prediction import compute_reliable_variance, compute_svca_prediction  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "stringer19" / "region_comparison"

# Matched across every session/region analyzed below (see module docstring).
N_TOTAL = 1000          # neurons per region-session, before the checkerboard split (500/side)
WINDOW = (100.0, 3100.0)  # session-relative seconds; 3000s, safely under every candidate's duration
BIN_SECONDS = 1.25

REGION_COLORS = {"VISp": "tab:blue", "SSp-bfd": "tab:orange", "MOp": "tab:green", "MOs": "tab:red"}

# Top-5-by-neuron-count sessions per region among canonical sessions with
# >=1000 post-filter neurons in that region (from stringer19_region_scan.py).
TIER1_SESSIONS = {
    "VISp": [
        "29063845-ba7a-4dc1-a4ab-61d285365fa9", "76f74094-99bb-467f-a89a-f20e15656048",
        "97334cc3-059f-4fb9-8413-3fe0288a7144", "be5909b6-0df7-415c-bd54-1e84eaadf591",
        "460a4c0f-f3f3-49a4-b0da-ddcb44322cbe",
    ],
    "SSp-bfd": [
        "0377646d-a970-44f8-806a-712c4214e0ce", "bd8565ba-35fe-4c85-93b6-54be4f2b4ccc",
        "98e9c22f-1006-4bb4-b573-6e4676c5f61d", "dd45de54-6b3c-443a-949d-e74954fc33b2",
        "aac478d1-7dc3-4acc-b965-c350c73e3610",
    ],
    "MOp": [
        "c14efa12-5931-490b-b445-3552044a9c54", "52ad9bfb-583a-496c-8d2a-5b8dd20c6af2",
        "d094fdec-fc93-48d4-892c-4b0dd17c553d", "16d4e507-d20f-4808-9584-fab050643077",
        "fa10ff03-be1f-4b67-9e1e-87bf6979c967",
    ],
    "MOs": [
        "a4afea1e-ad72-433d-9498-95ddc54252fe", "367aff30-254d-4ea9-ac09-b25feace139a",
        "29220169-a779-4519-b131-de403fe87507", "77141718-8120-4931-bec2-f59ca47f7603",
        "66a5f0a1-828d-422e-9181-9432c3f094b2",
    ],
}

# One session per (region pair) for Tier 2 -- each has >=1000 neurons in *both*
# listed regions, so one video decode/cache answers both.
TIER2_SESSIONS = {
    "16d4e507-d20f-4808-9584-fab050643077": ["MOp", "MOs"],
    "9e67c687-856d-4b94-938c-83c5775a1fff": ["VISp", "SSp-bfd"],
}


def select_region_neurons(
    session: MesoscopeSession,
    region_prefix: str,
    n_total: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Indices of neurons whose region acronym starts with `region_prefix`
    (so e.g. "VISp" matches "VISp1", "VISp2/3", "VISp5", ...), optionally
    randomly subsampled down to `n_total` (reproducibly, fixed `seed`) so
    different sessions/regions can be matched to the same neuron count.
    """
    labels = np.asarray(session.region_labels)
    idx = np.where(np.char.startswith(labels.astype(str), region_prefix))[0]
    if n_total is not None:
        if len(idx) < n_total:
            raise ValueError(f"only {len(idx)} '{region_prefix}' neurons available, need {n_total}")
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=n_total, replace=False)
    return idx


def fit_powerlaw_exp(ss: np.ndarray, fit_range: np.ndarray) -> Tuple[float, np.ndarray, float]:
    """Weighted log-log fit of a power-law decay ss[k] ~ (k+1)^-alpha,
    adapted from MouseLand/critical_init/powerlaw.py's `fit_powerlaw_exp`
    (same math: weighted least squares in log-log space, weights 1/rank
    so the fit isn't dominated by the noisiest high-rank tail).

    Parameters
    ----------
    ss : array
        The spectrum (e.g. `reliable_frac`), rank 0 = SVC 1.
    fit_range : array of int
        Which rank *indices* (0-based) to fit to -- typically excludes the
        very top (steepest, most idiosyncratic) and bottom (noisiest) ranks.

    Returns
    -------
    (alpha, ypred, log_log_corr): fitted exponent, the power-law curve
    evaluated at every rank in `ss`, and a log-log correlation over
    `fit_range` as a rough goodness-of-fit check.
    """
    logss = np.log(np.abs(ss))
    ranks = fit_range + 1  # 1-indexed rank, as in the original
    y = logss[fit_range][:, np.newaxis]
    nt = ranks.size
    x = np.concatenate((-np.log(ranks)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    w = 1.0 / ranks.astype(np.float64)[:, np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    all_ranks = np.arange(1, ss.size + 1)
    x_all = np.concatenate((-np.log(all_ranks)[:, np.newaxis], np.ones((ss.size, 1))), axis=1)
    ypred = np.exp((x_all * b).sum(axis=1))
    alpha = b[0]

    corr = float((zscore(np.log(ranks)) * zscore(logss[fit_range])).mean())
    return alpha, ypred, corr


def effective_dimensionality(ss: np.ndarray, frac: float = 0.5) -> int:
    """How many top ranks are needed for their cumulative sum to reach
    `frac` of the *captured* total (sum over all ranks in `ss`) -- a
    decay-rate/"effective dimensionality" measure, decoupled from SVC 1's
    absolute magnitude (which varies with e.g. binning, unrelated to shape).
    """
    ss = np.clip(ss, 0, None)
    cum = np.cumsum(ss) / ss.sum()
    return int(np.argmax(cum >= frac)) + 1


def run_tier1_dimensionality(
    sessions_by_region: Dict[str, Sequence[str]] = TIER1_SESSIONS,
    n_total: int = N_TOTAL,
    window: Tuple[float, float] = WINDOW,
    bin_seconds: float = BIN_SECONDS,
    fit_range: Optional[np.ndarray] = None,
    one: Optional[ONE] = None,
    save: bool = True,
) -> dict:
    one = one if one is not None else ONE()
    fit_range = fit_range if fit_range is not None else np.arange(9, 100)  # ranks 10-100

    records = []
    for region, eids in sessions_by_region.items():
        for eid in eids:
            print(f"[{region}] {eid}: loading...")
            session = load_mesoscope_session(eid, one=one)
            idx = select_region_neurons(session, region, n_total=n_total)
            res = compute_reliable_variance(
                session, window=window, bin_seconds=bin_seconds, neuron_subset=idx
            )
            r = res["reliable_frac"]
            fr = fit_range[fit_range < len(r)]
            alpha, ypred, corr = fit_powerlaw_exp(r, fr)
            dim50 = effective_dimensionality(r, 0.5)
            records.append(dict(region=region, eid=eid, reliable_frac=r, alpha=alpha, dim50=dim50,
                                 loglog_corr=corr, n_train=res["n_train"], n_test=res["n_test"]))
            print(f"  SVC1={100*r[0]:.1f}%  alpha={alpha:.3f}  dim50={dim50}  n_train/test={res['n_train']}/{res['n_test']}")

    _plot_tier1(records, save=save)
    return dict(records=records)


def _plot_tier1(records: List[dict], save: bool = True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    for region, color in REGION_COLORS.items():
        rs = [r["reliable_frac"] for r in records if r["region"] == region]
        if not rs:
            continue
        rs = np.stack(rs)
        rank = np.arange(1, rs.shape[1] + 1)
        mean_r = rs.mean(axis=0)
        sem_r = rs.std(axis=0) / np.sqrt(rs.shape[0])
        ax.plot(rank, 100 * mean_r, color=color, lw=1.5, label=f"{region} (n={rs.shape[0]})")
        ax.fill_between(rank, 100 * (mean_r - sem_r), 100 * (mean_r + sem_r), color=color, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("SVC dimension")
    ax.set_ylabel("% reliable variance")
    ax.set_title("reliable-variance spectrum\n(mean +/- SEM across sessions)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    regions = list(REGION_COLORS.keys())
    for i, region in enumerate(regions):
        alphas = [r["alpha"] for r in records if r["region"] == region]
        ax.scatter([i] * len(alphas), alphas, color=REGION_COLORS[region], s=40, zorder=2)
        if alphas:
            ax.scatter([i], [np.mean(alphas)], color="black", marker="_", s=400, zorder=3)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=30)
    ax.set_ylabel("power-law exponent (alpha)")
    ax.set_title("SVCA spectrum decay rate")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    for i, region in enumerate(regions):
        dims = [r["dim50"] for r in records if r["region"] == region]
        ax.scatter([i] * len(dims), dims, color=REGION_COLORS[region], s=40, zorder=2)
        if dims:
            ax.scatter([i], [np.mean(dims)], color="black", marker="_", s=400, zorder=3)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=30)
    ax.set_ylabel("dimensions for 50% of reliable variance")
    ax.set_title("effective dimensionality")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DIR / "tier1_dimensionality_by_region.png", dpi=200, bbox_inches="tight")
        print("Saved:", OUT_DIR / "tier1_dimensionality_by_region.png")
    return fig


def run_tier2_behavior_by_region(
    session_region_map: Dict[str, Sequence[str]] = TIER2_SESSIONS,
    n_total: int = N_TOTAL,
    window: Tuple[float, float] = WINDOW,
    bin_seconds: float = BIN_SECONDS,
    one: Optional[ONE] = None,
    save: bool = True,
) -> dict:
    one = one if one is not None else ONE()
    records = []
    for eid, regions in session_region_map.items():
        session = load_mesoscope_session(eid, one=one)
        for region in regions:
            print(f"[{region}] {eid}: computing SVCA + behavior/video (video PCs reused if cached)...")
            idx = select_region_neurons(session, region, n_total=n_total)
            res = compute_svca_prediction(
                eid, one=one, session=session, window=window, bin_seconds=bin_seconds, neuron_subset=idx,
            )
            records.append(dict(region=region, eid=eid, **res))
            print(f"  SVC1: reliable={100*res['reliable_frac'][0]:.1f}% video={100*res['video_var_explained'][0]:.1f}% "
                  f"behav={100*res['behav_var_explained'][0]:.1f}% block={100*res['block_var_explained'][0]:.1f}% "
                  f"choice={100*res['choice_var_explained'][0]:.1f}%")

    _plot_tier2(records, save=save)
    return dict(records=records)


def _plot_tier2(records: List[dict], save: bool = True):
    fig, axes = plt.subplots(1, len(records), figsize=(4.5 * len(records), 4.5), sharey=True)
    if len(records) == 1:
        axes = [axes]

    for ax, rec in zip(axes, records):
        rank = np.arange(1, len(rec["reliable_frac"]) + 1)
        ax.plot(rank, 100 * np.clip(rec["reliable_frac"], 0, None), color="gray", lw=1.5, label="max explainable")
        ax.plot(rank, 100 * np.clip(rec["video_var_explained"], 0, None), color="tab:blue", lw=1.5, label="video PCs")
        behav_label = "wheel+whisker" if rec.get("behav_predictors_used") == "wheel+whisker" else "wheel only"
        ax.plot(rank, 100 * np.clip(rec["behav_var_explained"], 0, None), color="tab:green", lw=1.5, label=behav_label)
        ax.plot(rank, 100 * np.clip(rec["block_var_explained"], 0, None), color="tab:purple", lw=1.2, label="block")
        ax.plot(rank, 100 * np.clip(rec["choice_var_explained"], 0, None), color="tab:red", lw=1.2, label="choice")
        ax.set_xscale("log")
        ax.set_xlabel("SVC dimension")
        ax.set_title(rec["region"], fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("% variance explained")
    axes[0].set_ylim(0, 100)
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Behavioral/task contribution to shared variance, by region", y=1.02)
    fig.tight_layout()

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DIR / "tier2_behavior_by_region.png", dpi=200, bbox_inches="tight")
        print("Saved:", OUT_DIR / "tier2_behavior_by_region.png")
    return fig


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("tier1", "both"):
        run_tier1_dimensionality()
    if which in ("tier2", "both"):
        run_tier2_behavior_by_region()
