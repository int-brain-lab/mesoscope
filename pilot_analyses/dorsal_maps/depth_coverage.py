"""Plot depth and insertion coverage for the v7 decoding maps.

Usage: python depth_coverage.py
Reads positions and trial counts from the cached meso and bwm feature files.
"""
import matplotlib.pyplot as plt
import numpy as np

from common import FEATURE_ROOT
from decode_v2 import MIN_TRIALS, N_PICK, OUT_V2, excluded_sessions
from make_bwm_depth import MAX_DEPTH_UM, streamline_depth_um
from plotstyle import NATURE_DOUBLE_COL_IN, panel_label, save_with_caption, style_axes


MESO_COLOR = "tab:green"
BWM_COLOR = "tab:purple"
ALL_COLOR = "0.3"
DEPTH_BIN_UM = 80
COUNT_BIN = 5
COUNT_OVERFLOW = 100


def load_registered_meso_xyz():
    """Return the per-session ROI positions with measured histological depths."""
    meso_xyz = []
    excluded = excluded_sessions()
    for path in sorted((FEATURE_ROOT / "meso").glob("*.npz")):
        if path.stem in excluded:
            continue
        with np.load(path) as z:
            # Projected ROI positions carry no measured depth.
            if str(z["xyz_source"]) == "mlapdv":
                meso_xyz.append(z["xyz"])
    return meso_xyz


def load_depths():
    meso_xyz = load_registered_meso_xyz()
    bwm_xyz, trials = [], []
    for path in sorted((FEATURE_ROOT / "bwm").glob("*.npz")):
        with np.load(path) as z:
            bwm_xyz.append(z["xyz"])
            trials.append(len(z["choice"]))
    if not meso_xyz or not bwm_xyz or not any(n >= MIN_TRIALS for n in trials):
        raise ValueError("Depth coverage needs cached mesoscope and BWM features.")
    meso = streamline_depth_um(np.concatenate(meso_xyz))
    bwm = streamline_depth_um(np.concatenate(bwm_xyz))
    by_probe = np.split(bwm, np.cumsum([len(x) for x in bwm_xyz])[:-1])
    counts = [(d.size, np.count_nonzero(d <= MAX_DEPTH_UM))
              for n, d in zip(trials, by_probe) if n >= MIN_TRIALS]
    return meso, bwm, len(meso_xyz), np.asarray(counts)

def depth_hist(ax, depths, edges, color, title):
    freq, _ = np.histogram(depths, bins=edges)
    ax.axvspan(edges[0], MAX_DEPTH_UM, color=BWM_COLOR, alpha=0.08, lw=0)
    ax.stairs(100 * freq / freq.sum(), edges, color=color, fill=True, alpha=0.8, lw=0)
    ax.axvline(MAX_DEPTH_UM, color="0.25", ls="--", lw=0.8)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("Units per bin (%)")
    style_axes(ax)


def main():
    meso, bwm, n_meso, counts = load_depths()
    meso, bwm = meso[np.isfinite(meso)], bwm[np.isfinite(bwm)]
    lo = DEPTH_BIN_UM * np.floor(min(meso.min(), bwm.min()) / DEPTH_BIN_UM)
    hi = DEPTH_BIN_UM * np.ceil(max(meso.max(), bwm.max()) / DEPTH_BIN_UM)
    depth_edges = np.arange(lo, hi + DEPTH_BIN_UM, DEPTH_BIN_UM)
    n_shallow = np.count_nonzero(bwm <= MAX_DEPTH_UM)
    n_all_probes, n_shallow_probes = (counts >= N_PICK).sum(axis=0)

    fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.05), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.13, 1))
    a = fig.add_subplot(grid[0, 0])
    b = fig.add_subplot(grid[1, 0], sharex=a)
    c = fig.add_subplot(grid[:, 1])

    depth_hist(a, meso, depth_edges, MESO_COLOR, "Mesoscope ROIs")
    depth_hist(b, bwm, depth_edges, BWM_COLOR, "BWM isocortex units")
    a.set_ylabel("ROIs per bin (%)")
    a.tick_params(labelbottom=False)
    a.text(MAX_DEPTH_UM + DEPTH_BIN_UM, 0.94, f"{MAX_DEPTH_UM} µm cutoff",
           transform=a.get_xaxis_transform(), va="top")
    b.text(0.98, 0.94, f"{n_shallow / bwm.size:.1%} within cutoff",
           transform=b.transAxes, ha="right", va="top")
    b.set_xlabel("Depth below pial surface (µm)")
    b.set_xticks([0, MAX_DEPTH_UM, 1000, 1500])

    count_edges = np.arange(0, COUNT_OVERFLOW + 2 * COUNT_BIN, COUNT_BIN)
    for values, color, label in (
        (counts[:, 0], ALL_COLOR, f"All depths ({n_all_probes} eligible)"),
        (counts[:, 1], BWM_COLOR, f"≤{MAX_DEPTH_UM} µm ({n_shallow_probes} eligible)"),
    ):
        freq, _ = np.histogram(np.minimum(values, COUNT_OVERFLOW), bins=count_edges)
        c.stairs(freq, count_edges, color=color, lw=1.2, label=label)
    c.axvline(N_PICK, color="0.25", ls="--", lw=0.8)
    c.text(N_PICK + 2, 0.6, f"{N_PICK}-unit minimum", transform=c.get_xaxis_transform())
    c.set_xlim(0, count_edges[-1])
    c.set_ylim(bottom=0)
    c.set_title("Units per BWM insertion", loc="left", fontweight="bold")
    c.set_xlabel("Isocortex units")
    c.set_ylabel("Insertions per bin")
    c.legend(frameon=False, loc="upper right")
    style_axes(c)
    c.set_xticks([0, N_PICK, 50, 75, COUNT_OVERFLOW + COUNT_BIN / 2],
                 ["0", str(N_PICK), "50", "75", f"{COUNT_OVERFLOW}+"])
    for ax, letter in zip((a, b, c), "abc"):
        panel_label(ax, letter, x=-0.08, y=1.08)

    pct = f"{100 * n_shallow / bwm.size:.1f}"
    caps = [
        f"Depths of {meso.size:,} isocortex ROIs pooled over {n_meso} histology-registered "
        "mesoscope sessions. Sessions with projected positions are omitted.",
        f"Depths of {bwm.size:,} good BWM isocortex units. The shaded range ends at the "
        f"dashed {MAX_DEPTH_UM}\\,$\\mu$m cutoff and contains {n_shallow:,} units ({pct}\\,\\%). "
        f"Both depth histograms use {DEPTH_BIN_UM}\\,$\\mu$m bins and are normalized separately.",
        f"Unit counts per insertion for {len(counts)} BWM insertions in sessions with at least "
        f"{MIN_TRIALS} eligible trials. The dashed line marks the {N_PICK}-unit requirement "
        f"for the decoding maps. {n_all_probes} insertions meet it at all depths and "
        f"{n_shallow_probes} meet it after the depth cutoff. Counts of at least "
        f"{COUNT_OVERFLOW} units share the final bin.",
    ]
    save_with_caption(fig, OUT_V2 / "si_depth_coverage", caps, "depth_coverage",
                      intro="Recording depth and probe eligibility. The IBL 25-micron "
                            "Allen CCF streamline lookup estimates local pial depth.")
    plt.close(fig)


if __name__ == "__main__":
    main()
