"""Reproduce Stringer et al. 2019 (Science) supplementary Fig. S2 (A-C) for an
IBL mesoscope session: "Correlation matrices are structured and consistent
across time."

Fig. S2 shows, for one example recording: (A,B) a pseudocolor correlation
matrix between two disjoint sets of 10 neurons each, computed independently
from the first and second half of the recording; (C) a scatter of every
cell pair's correlation in the first half against the second half, with
their Pearson r annotated; (D) a histogram of that r value across many
recordings.

`plot_correlation_consistency` adapts this directly to IBL mesoscope data
using the deconvolved ROI signal from `meso_loader.load_mesoscope_session`.
Panel D (`include_histogram=True`, the default) draws on a random sample of
other canonical sessions via `stringer19_corr_consistency_hist.py`, with
this session's own r marked -- see that module's docstring for why some
sessions land far lower than others (short recordings give a noisier r
purely from having fewer timepoints per half, not "worse" data).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from one.api import ONE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meso_loader import load_mesoscope_session, MesoscopeSession  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "stringer19"


def _half_split(signal: np.ndarray):
    mid = signal.shape[1] // 2
    return signal[:, :mid], signal[:, mid:]


def _upper_triangle(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def _cross_block(sig: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Correlation of each `rows` neuron against each `cols` neuron."""
    sub = sig[np.concatenate([rows, cols])]
    c = np.corrcoef(sub)
    n = len(rows)
    return c[:n, n:]


def compute_correlation_consistency(
    session: MesoscopeSession,
    n_display: int = 10,
    n_scatter_neurons: int = 500,
    seed: int = 0,
) -> dict:
    """Data half of Fig. S2 A-C: no plotting, so this is cheap to call in bulk
    (e.g. once per session when building a histogram of `r` across sessions).

    Parameters
    ----------
    session : MesoscopeSession
    n_display : int, default 10
        Neurons per side of the panel A/B cross-correlation block (10x10,
        matching the paper's "neurons 1-10" x "neurons 11-20").
    n_scatter_neurons : int, default 500
        Size of the random neuron pool used for the pairwise-correlation
        comparison. The paper uses "all cell pairs"; with ~7,000 neurons
        that's ~24M pairs, impractical to compute and plot, so this instead
        uses all pairs (n_scatter_neurons choose 2) within a random subset --
        500 neurons gives ~124,750 pairs, a representative sample at a
        reasonable compute cost. The panel A/B neurons are the first
        `2 * n_display` of this same pool.
    seed : int, default 0
        RNG seed for the neuron subsample (reproducible by default).

    Returns
    -------
    dict with `r` (Pearson correlation between the two halves' pairwise
    correlations), `block1`/`block2` (the 10x10 panel A/B matrices),
    `pair_corr_half1`/`pair_corr_half2` (pairwise-correlation vectors), and
    the neuron indices used (`row_idx`, `col_idx`, `scatter_idx`).
    """
    n_neurons = session.roi_signal.shape[0]
    n_scatter_neurons = min(n_scatter_neurons, n_neurons)
    if n_scatter_neurons < 2 * n_display:
        raise ValueError(f"n_scatter_neurons ({n_scatter_neurons}) must be >= 2 * n_display ({2 * n_display})")

    rng = np.random.default_rng(seed)
    pool = rng.choice(n_neurons, size=n_scatter_neurons, replace=False)
    row_idx, col_idx = pool[:n_display], pool[n_display : 2 * n_display]

    sig_half1, sig_half2 = _half_split(session.roi_signal)

    block1 = _cross_block(sig_half1, row_idx, col_idx)
    block2 = _cross_block(sig_half2, row_idx, col_idx)

    c1_full = np.corrcoef(sig_half1[pool])
    c2_full = np.corrcoef(sig_half2[pool])
    pair_c1 = _upper_triangle(c1_full)
    pair_c2 = _upper_triangle(c2_full)
    r = float(np.corrcoef(pair_c1, pair_c2)[0, 1])

    return dict(
        r=r,
        block1=block1,
        block2=block2,
        pair_corr_half1=pair_c1,
        pair_corr_half2=pair_c2,
        row_idx=row_idx,
        col_idx=col_idx,
        scatter_idx=pool,
    )


def plot_correlation_consistency(
    eid: str,
    one: Optional[ONE] = None,
    session: Optional[MesoscopeSession] = None,
    n_display: int = 10,
    n_scatter_neurons: int = 500,
    seed: int = 0,
    include_histogram: bool = True,
    n_hist_sessions: int = 10,
    hist_seed: int = 0,
    canonical_sessions_path: Optional[Path] = None,
    save: bool = True,
    out_dir: Optional[Path] = None,
):
    """Reproduce Stringer et al. 2019 Fig. S2 A-D for one mesoscope session.

    Parameters
    ----------
    eid : str
    one : ONE, optional
    session : MesoscopeSession, optional
        Pass an already-loaded session to avoid reloading.
    n_display, n_scatter_neurons, seed
        See `compute_correlation_consistency`.
    include_histogram : bool, default True
        Add panel D: a histogram of `r` across `n_hist_sessions` other
        random canonical sessions (via `stringer19_corr_consistency_hist
        .collect_r_across_sessions`), with this session's own r marked.
        Set False to skip it and only reproduce panels A-C (much faster
        the first time, since it avoids loading ~10 other sessions).
    n_hist_sessions, hist_seed, canonical_sessions_path
        Passed to `collect_r_across_sessions` when `include_histogram`.
    save : bool, default True
        Save the figure as a PNG under `out_dir`.
    out_dir : Path, optional
        Defaults to `<repo_root>/stringer19/`.

    Returns
    -------
    (fig, info) -- see `compute_correlation_consistency` for `info`'s keys;
    when `include_histogram`, `info` also has `hist_r_values`, `hist_used`,
    and `hist_skipped` (see `collect_r_across_sessions`).
    """
    one = one if one is not None else ONE()
    session = session if session is not None else load_mesoscope_session(eid, one=one)

    info = compute_correlation_consistency(
        session, n_display=n_display, n_scatter_neurons=n_scatter_neurons, seed=seed
    )
    block1, block2 = info["block1"], info["block2"]
    pair_c1, pair_c2, r = info["pair_corr_half1"], info["pair_corr_half2"], info["r"]
    n_scatter_neurons = len(info["scatter_idx"])

    if include_histogram:
        from stringer19_corr_consistency_hist import collect_r_across_sessions  # noqa: E402 (avoids circular import)

        hist_kwargs = dict(n_sessions=n_hist_sessions, one=one, seed=hist_seed, n_scatter_neurons=n_scatter_neurons, corr_seed=seed)
        if canonical_sessions_path is not None:
            hist_kwargs["canonical_sessions_path"] = canonical_sessions_path
        hist_result = collect_r_across_sessions(**hist_kwargs)
        info["hist_r_values"] = hist_result["r_values"]
        info["hist_durations"] = hist_result["durations"]
        info["hist_n_neurons"] = hist_result["n_neurons"]
        info["hist_used"] = hist_result["used"]
        info["hist_skipped"] = hist_result["skipped"]

    vmax = max(float(np.nanmax(np.abs(np.concatenate([block1.ravel(), block2.ravel()])))), 1e-6)

    ncols = 5 if include_histogram else 4
    width_ratios = [1, 1, 1.3, 0.05, 1.3] if include_histogram else [1, 1, 1.3, 0.05]
    fig = plt.figure(figsize=(18 if include_histogram else 15, 4.2))
    gs = gridspec.GridSpec(1, ncols, width_ratios=width_ratios, wspace=0.55)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])

    im = ax_a.imshow(block1, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax_a.set_title("correlation half 1")
    ax_a.set_xlabel("neurons 11-20")
    ax_a.set_ylabel("neurons 1-10")
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    ax_b.imshow(block2, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax_b.set_title("correlation half 2")
    ax_b.set_xlabel("neurons 11-20")
    ax_b.set_xticks([])
    ax_b.set_yticks([])

    fig.colorbar(im, cax=cax)

    ax_c.scatter(pair_c1, pair_c2, s=4, color="gold", edgecolor="none", alpha=0.4, rasterized=True)
    lim = max(np.abs(pair_c1).max(), np.abs(pair_c2).max()) * 1.05
    ax_c.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8, ls="--", zorder=0)
    ax_c.set_xlim(-lim, lim)
    ax_c.set_ylim(-lim, lim)
    ax_c.set_xlabel("correlation half 1")
    ax_c.set_ylabel("correlation half 2")
    ax_c.set_title("pairwise correlations")
    ax_c.text(0.05, 0.9, f"r={r:.2f}", transform=ax_c.transAxes, fontsize=12)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    if include_histogram:
        hist_r = info["hist_r_values"]
        ax_d = fig.add_subplot(gs[4])
        ax_d.hist(hist_r, bins=np.linspace(0, 1, 21), color="gray", edgecolor="black", linewidth=0.5)
        ax_d.axvline(r, color="crimson", lw=1.5, ls="--")
        ax_d.text(r, ax_d.get_ylim()[1], " this session", color="crimson", fontsize=8, va="top", ha="left" if r < 0.85 else "right")
        ax_d.set_xlabel("r of correlations")
        ax_d.set_ylabel("# of recordings")
        ax_d.set_xlim(0, 1)
        ax_d.set_title(f"n={len(hist_r)} sessions\nmean r={np.mean(hist_r):.2f}", fontsize=10)
        ax_d.spines["top"].set_visible(False)
        ax_d.spines["right"].set_visible(False)

    fig.suptitle(
        f"{eid}  —  correlation consistency across time "
        f"({n_scatter_neurons} neurons, {len(pair_c1)} pairs)"
    )

    if save:
        out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "figS2abcd" if include_histogram else "figS2abc"
        fpath = out_dir / f"{eid}_stringer19_{suffix}.png"
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        print("Saved:", fpath)

    return fig, info


if __name__ == "__main__":
    eid_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if eid_arg is None:
        raise SystemExit("usage: python stringer19_corr_consistency.py <eid>")
    plot_correlation_consistency(eid_arg)
