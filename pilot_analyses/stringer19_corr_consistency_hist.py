"""Reproduce Stringer et al. 2019 (Science) supplementary Fig. S2 D for IBL
mesoscope data: a histogram, across recordings, of the Pearson r between a
session's first-half and second-half pairwise neural correlations (see
`stringer19_corr_consistency.py` for panels A-C, one session at a time).

Sessions are drawn at random from `canonical_sessions.txt`. Not every
canonical session has fully processed mesoscope data yet, so each candidate
is first cheaply checked (dataset listing only, no download) before the
(expensive) full load is attempted; sessions that fail either check are
skipped and logged, and sampling continues until `n_sessions` succeed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from one.api import ONE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meso_loader import load_mesoscope_session  # noqa: E402
from stringer19_corr_consistency import compute_correlation_consistency  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "stringer19"
CANONICAL_SESSIONS_PATH = Path(__file__).resolve().parent.parent / "canonical_sessions.txt"

_REQUIRED_DATASETS = (
    "mpci.times.npy",
    "mpciStack.timeshift.npy",
    "mpciROIs.stackPos.npy",
    "mpciROIs.mpciROITypes.npy",
    "mpciROIs.mlapdv_estimate.npy",
)
_SIGNAL_DATASETS = ("mpci.ROIActivityDeconvolved.npy", "mpci.ROIActivityF.npy")


def _load_canonical_paths(path: Path = CANONICAL_SESSIONS_PATH) -> List[str]:
    return [p.strip() for p in path.read_text().split(",") if p.strip()]


def _is_session_processed(eid: str, one: ONE) -> bool:
    """Cheap check (dataset listing, no downloads): does this session have
    at least one FOV with everything `meso_loader.load_mesoscope_session`
    needs?
    """
    cols = one.list_collections(eid, collection="alf/FOV_*")
    if not cols:
        return False
    for fov in cols:
        names = {d.split("/")[-1] for d in one.list_datasets(eid, collection=fov)}
        if set(_REQUIRED_DATASETS).issubset(names) and any(s in names for s in _SIGNAL_DATASETS):
            return True
    return False


def collect_r_across_sessions(
    n_sessions: int = 10,
    one: Optional[ONE] = None,
    seed: int = 0,
    n_scatter_neurons: int = 500,
    corr_seed: int = 0,
    bin_seconds: Optional[float] = None,
    canonical_sessions_path: Path = CANONICAL_SESSIONS_PATH,
    verbose: bool = True,
) -> dict:
    """Randomly sample sessions from the canonical list and compute the
    first-half-vs-second-half pairwise-correlation r for each.

    Parameters
    ----------
    n_sessions : int, default 10
        Number of sessions to successfully process.
    one : ONE, optional
    seed : int, default 0
        RNG seed for shuffling the canonical session list (reproducible by
        default -- pass a different value for a different random draw).
    n_scatter_neurons, corr_seed, bin_seconds
        Passed to `compute_correlation_consistency` for each session.
    canonical_sessions_path : Path
        Defaults to `canonical_sessions.txt` at the repo root.
    verbose : bool, default True
        Print progress/skip reasons as sessions are processed.

    Returns
    -------
    dict with `r_values`, `durations` (session length in seconds), and
    `n_neurons` (each an array, len `n_sessions`, aligned with `used`), plus
    `used` (list of (path, eid) actually used) and `skipped` (list of
    (path, eid_or_None, reason)).
    """
    one = one if one is not None else ONE()
    paths = _load_canonical_paths(canonical_sessions_path)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))

    r_values, durations, n_neurons, used, skipped = [], [], [], [], []

    for idx in order:
        if len(used) >= n_sessions:
            break
        path = paths[idx]
        try:
            eid = str(one.path2eid(path))
            if eid is None or eid == "None":
                skipped.append((path, None, "no eid"))
                if verbose:
                    print(f"[skip] {path}: no eid")
                continue
            if not _is_session_processed(eid, one):
                skipped.append((path, eid, "no processed FOV data"))
                if verbose:
                    print(f"[skip] {path} ({eid}): no processed FOV data")
                continue

            session = load_mesoscope_session(eid, one=one)
            info = compute_correlation_consistency(
                session, n_scatter_neurons=n_scatter_neurons, seed=corr_seed, bin_seconds=bin_seconds
            )
            r_values.append(info["r"])
            durations.append(float(session.roi_times[0][-1] - session.roi_times[0][0]))
            n_neurons.append(session.roi_signal.shape[0])
            used.append((path, eid))
            if verbose:
                print(f"[ok]   {path} ({eid}): r={info['r']:.3f}  ({len(used)}/{n_sessions})")
        except Exception as e:
            skipped.append((path, None, f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"[skip] {path}: {type(e).__name__}: {e}")

    if len(used) < n_sessions:
        print(
            f"Warning: only found {len(used)}/{n_sessions} usable sessions "
            f"out of {len(paths)} canonical sessions."
        )

    return dict(
        r_values=np.array(r_values),
        durations=np.array(durations),
        n_neurons=np.array(n_neurons),
        used=used,
        skipped=skipped,
    )


def plot_r_histogram(
    n_sessions: int = 10,
    one: Optional[ONE] = None,
    seed: int = 0,
    n_scatter_neurons: int = 500,
    corr_seed: int = 0,
    bin_seconds: Optional[float] = None,
    save: bool = True,
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, dict]:
    """Reproduce Stringer et al. 2019 Fig. S2 D: histogram of the
    half1-vs-half2 pairwise-correlation r across `n_sessions` randomly
    chosen canonical mesoscope sessions.

    See `collect_r_across_sessions` for the sampling parameters. Returns
    `(fig, result)` where `result` is `collect_r_across_sessions`'s return
    value.
    """
    one = one if one is not None else ONE()
    result = collect_r_across_sessions(
        n_sessions=n_sessions, one=one, seed=seed,
        n_scatter_neurons=n_scatter_neurons, corr_seed=corr_seed, bin_seconds=bin_seconds,
    )
    r_values = result["r_values"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(r_values, bins=bins, color="gray", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("r of correlations")
    ax.set_ylabel("# of recordings")
    ax.set_xlim(0, 1)
    ax.set_title(f"n={len(r_values)} sessions, mean r={np.mean(r_values):.2f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save:
        out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / f"canonical_n{len(r_values)}_stringer19_figS2d.png"
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        print("Saved:", fpath)

    return fig, result


def plot_r_vs_duration(
    n_sessions: int = 10,
    one: Optional[ONE] = None,
    seed: int = 0,
    n_scatter_neurons: int = 500,
    corr_seed: int = 0,
    bin_seconds: Optional[float] = None,
    result: Optional[dict] = None,
    save: bool = True,
    out_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, dict]:
    """Diagnostic scatter (not from the paper): r vs. session duration.

    Not every canonical session gives an equally consistent correlation
    structure across its two halves -- some `r` values in the Fig. S2 D
    histogram are much lower than others. The reason turns out to be
    mundane: `r` is the Pearson correlation between two *estimates* of a
    pairwise-correlation matrix, each computed from only half the
    recording. Shorter recordings give each half fewer timepoints, so each
    per-pair correlation is a noisier estimate purely from sampling error
    (estimation noise in a Pearson r shrinks only as ~1/sqrt(T)) -- not
    because the underlying neural correlation structure is actually less
    real or less stable. This plot checks that: across 10 random canonical
    sessions, r vs. duration had corr=0.82 (n_neurons vs. r was much weaker,
    -0.34, and largely a side effect of longer/shorter sessions also having
    different neuron counts).

    Parameters
    ----------
    result : dict, optional
        Reuse an existing `collect_r_across_sessions` result instead of
        computing a new one (must include `durations`).
    Other parameters are passed to `collect_r_across_sessions` when
    `result` is not given.
    """
    one = one if one is not None else ONE()
    result = result if result is not None else collect_r_across_sessions(
        n_sessions=n_sessions, one=one, seed=seed,
        n_scatter_neurons=n_scatter_neurons, corr_seed=corr_seed, bin_seconds=bin_seconds,
    )
    r_values, durations = result["r_values"], result["durations"]
    corr = float(np.corrcoef(r_values, durations)[0, 1])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(durations / 60, r_values, color="steelblue", edgecolor="black", linewidth=0.5, s=40)
    ax.set_xlabel("session duration (min)")
    ax.set_ylabel("r of correlations")
    ax.set_ylim(0, 1)
    ax.set_title(f"n={len(r_values)} sessions, corr(r, duration)={corr:.2f}", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save:
        out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / f"canonical_n{len(r_values)}_stringer19_r_vs_duration.png"
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        print("Saved:", fpath)

    return fig, result


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    plot_r_histogram(n_sessions=n)
