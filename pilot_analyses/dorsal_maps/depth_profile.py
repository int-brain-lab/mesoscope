#!/usr/bin/env python3
"""Depth sensitivity of the BWM comparison: how do task tuning and wheel-speed
correlation of BWM isocortex units change with depth below the pial surface,
relative to the mesoscope?

Per BWM unit: depth (Allen CCF cortical streamlines), tuning p-values
(output/tuning_bwm.pqt) and the correlation with wheel speed over all wheel
bins on even and odd trials (feature files). Fractions per 160 um depth band,
with 95 % binomial (Wilson) intervals; mesoscope fractions over all neurons
as reference lines; shading: the 5--95% depth range of histology-registered mesoscope ROIs.
The 640 um cutoff uses local pial depth.

Usage: python depth_profile.py -> output/v2/si_depth_profile.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ALPHA, OUT, VARIABLES
from depth_coverage import load_depths
from decode_v2 import OUT_V2
from make_bwm_depth import MAX_DEPTH_UM, streamline_depth_um
from plotstyle import NATURE_DOUBLE_COL_IN, panel_label, save_with_caption, style_axes
from sessions import session_files

EDGES = np.arange(0, 2081, 160)
TITLE = {"stim": "Stimulus side", "choice": "Choice", "feedback": "Feedback", "wheel": "Wheel speed"}


def wheel_frac(X, y):
    """Per neuron: r > 0.1 with wheel speed on both even and odd trials."""
    def corr(Xs, ys):
        A = Xs.reshape(len(Xs), -1)
        A = A - A.mean(1, keepdims=True)
        b = ys.reshape(-1) - ys.mean()
        den = np.linalg.norm(A, axis=1) * np.linalg.norm(b)
        return np.where(den > 0, A @ b / np.where(den > 0, den, 1), np.nan)
    ev, od = slice(0, None, 2), slice(1, None, 2)
    return (corr(X[:, ev], y[ev]) > 0.1) & (corr(X[:, od], y[od]) > 0.1)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return c - h, c + h


def collect_wheel():
    rows = []
    xyz = []
    for m in ("bwm", "meso"):
        for sid, files in session_files(m).items():
            traces = []
            positions = []
            for i, path in enumerate(files):
                with np.load(path) as z:
                    traces.append(z["X_wheel"])
                    positions.append(z["xyz"])
                    if i == 0:
                        y = np.asarray(z["wheel_y"], dtype=np.float64)
            X = np.concatenate(traces).astype(np.float64)
            if len(X) == 0 or len(y) < 50:
                continue
            ok = X.reshape(len(X), -1).std(1) > 0
            if not ok.any():
                continue
            wheel = wheel_frac(X[ok], y)
            if m == "bwm":
                xyz.append(np.concatenate(positions)[ok])
            rows.append(pd.DataFrame(dict(modality=m, session=sid, wheel=wheel, depth=np.nan)))
    wh = pd.concat(rows, ignore_index=True)
    wh.loc[wh.modality == "bwm", "depth"] = streamline_depth_um(np.concatenate(xyz))
    return wh


def main(wh=None):
    if wh is None:
        wh = collect_wheel()
    tun = pd.read_parquet(OUT / "tuning_bwm.pqt")
    tun["depth"] = streamline_depth_um(tun[["x", "y", "z"]].values)
    meso_depth, _, _, _ = load_depths()
    meso_lo, meso_hi = np.nanpercentile(meso_depth, [5, 95])
    meso_t = pd.read_parquet(OUT / "tuning_meso.pqt")
    ref = {v: np.mean(meso_t[f"p_{v}"].dropna() < ALPHA) for v in VARIABLES}
    ref["wheel"] = wh[wh.modality == "meso"].wheel.mean()
    fig, axs = plt.subplots(1, 4, figsize=(NATURE_DOUBLE_COL_IN, 1.8), gridspec_kw=dict(wspace=0.45))
    centres = (EDGES[:-1] + EDGES[1:]) / 2
    stats = {}
    for c, v in enumerate(VARIABLES + ("wheel",)):
        ax = axs[c]
        if v == "wheel":
            b = wh[wh.modality == "bwm"]
            sig, dep = b.wheel.values, b.depth.values
        else:
            ok = tun[f"p_{v}"].notna()
            sig, dep = (tun[f"p_{v}"][ok] < ALPHA).values, tun.depth[ok].values
        valid = np.isfinite(dep)
        sig, dep = sig[valid], dep[valid]
        band = np.digitize(dep, EDGES) - 1
        k = np.array([sig[band == i].sum() for i in range(len(centres))])
        n = np.array([(band == i).sum() for i in range(len(centres))])
        good = n >= 30
        lo, hi = wilson(k[good], n[good])
        ax.axvspan(meso_lo, meso_hi, color="tab:green", alpha=0.15, lw=0)
        ax.axvline(MAX_DEPTH_UM, color="0.4", lw=0.6, ls="--")
        ax.fill_between(centres[good], lo, hi, color="tab:purple", alpha=0.25, lw=0)
        ax.plot(centres[good], k[good] / n[good], "o-", color="tab:purple", ms=2, lw=0.8)
        ax.axhline(ref[v], color="tab:green", lw=1.0)
        ax.set_ylim(0, 1.2 * max(hi.max(), ref[v]))
        ax.set_title(TITLE[v], fontsize=6)
        ax.set_xlabel("depth below surface (µm)")
        if c == 0:
            ax.set_ylabel("fraction of units")
        style_axes(ax)
        panel_label(ax, "abcd"[c])
        shallow, deep = dep <= MAX_DEPTH_UM, dep > MAX_DEPTH_UM
        stats[v] = (sig[shallow].mean(), sig[deep].mean(), ref[v], shallow.sum(), deep.sum())
    pd.DataFrame(stats, index=["bwm_le640", "bwm_gt640", "meso", "n_le640", "n_gt640"]).T.to_csv(
        OUT_V2 / "si_depth_profile.csv")
    f = lambda v: (f"{stats[v][0]:.1%} of BWM units within {MAX_DEPTH_UM}\\,$\\mu$m, {stats[v][1]:.1%} deeper, "  # noqa
                   f"mesoscope {stats[v][2]:.1%}").replace("%", "\\%")
    caps = [f"Fraction of BWM isocortex units tuned to stimulus side ($p<{ALPHA}$, same test as the maps) per "
            f"160\\,$\\mu$m depth band (purple; band: 95\\,\\% Wilson interval; bands with $\\geq$30 units): {f('stim')}.",
            f"Choice: {f('choice')}.", f"Feedback: {f('feedback')}.",
            "Fraction of units correlated with wheel speed ($r>0.1$ over the wheel bins of both the even and "
            f"the odd trials): {f('wheel')}."]
    intro = ("Depth sensitivity of the BWM comparison. Green line: mesoscope fraction over all neurons; green "
             f"shading: 5th--95th percentiles of histology-registered mesoscope ROI depths "
             f"({meso_lo:.0f}--{meso_hi:.0f}\\,$\\mu$m). The dashed line marks the "
             f"{MAX_DEPTH_UM}\\,$\\mu$m cutoff. Depth uses the 25\\,$\\mu$m Allen CCF streamline lookup.")
    save_with_caption(fig, OUT_V2 / "si_depth_profile", caps, "v2_depth_profile", intro=intro)


if __name__ == "__main__":
    main()
