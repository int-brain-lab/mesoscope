#!/usr/bin/env python3
"""Plot single-unit tuning and fixed-size BWM decoding by relative pial depth.

First run decode_bwm_depth.py, then run python depth_profile.py to regenerate
output/v2/si_depth_profile.pdf from the refitted decoder scores and tuning cache.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from common import ALPHA, OUT, VARIABLES
from decode_bwm_depth import DEPTH_EDGES, N_DRAWS, N_UNITS, SCORES, depth_bins
from depth_coverage import load_registered_meso_xyz
from decode_v2 import MIN_TRIALS, OUT_V2, TARGETS
from make_bwm_depth import streamline_depth_percent
from plotstyle import NATURE_DOUBLE_COL_IN, panel_label, save_with_caption, style_axes
from sessions import session_files

CENTRES = (DEPTH_EDGES[:-1] + DEPTH_EDGES[1:]) / 2
TITLE = {"stim": "Stimulus side", "choice": "Choice", "feedback": "Feedback",
         "wheel": "Wheel speed", "wheel_trial": "Trial-specific speed"}
WHEEL_CACHE = OUT_V2 / "report_v7_relative_depth_wheel.pqt"


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
            if len(X) == 0 or len(y) < MIN_TRIALS:
                continue
            ok = X.reshape(len(X), -1).std(1) > 0
            if not ok.any():
                continue
            wheel = wheel_frac(X[ok], y)
            if m == "bwm":
                xyz.append(np.concatenate(positions)[ok])
            rows.append(pd.DataFrame(dict(modality=m, session=sid, wheel=wheel, depth=np.nan)))
    wh = pd.concat(rows, ignore_index=True)
    wh.loc[wh.modality == "bwm", "depth"] = streamline_depth_percent(np.concatenate(xyz))
    return wh


def plot_tuning(axes, wh):
    tun = pd.read_parquet(OUT / "tuning_bwm.pqt")
    tun["bin"] = depth_bins(streamline_depth_percent(tun[["x", "y", "z"]].values))
    meso_depth = streamline_depth_percent(np.concatenate(load_registered_meso_xyz()))
    meso_lo, meso_hi = np.nanpercentile(meso_depth, [5, 95])
    meso_t = pd.read_parquet(OUT / "tuning_meso.pqt")
    ref = {v: np.mean(meso_t[f"p_{v}"].dropna() < ALPHA) for v in VARIABLES}
    ref["wheel"] = wh[wh.modality == "meso"].wheel.mean()
    stats = []
    for ax, v in zip(axes, VARIABLES + ("wheel",)):
        if v == "wheel":
            b = wh[wh.modality == "bwm"]
            sig, bins = b.wheel.values, depth_bins(b.depth.values)
        else:
            ok = tun[f"p_{v}"].notna()
            sig, bins = (tun.loc[ok, f"p_{v}"] < ALPHA).values, tun.loc[ok, "bin"].values
        k = np.array([sig[bins == i].sum() for i in range(len(CENTRES))])
        n = np.array([(bins == i).sum() for i in range(len(CENTRES))])
        good = n >= 30
        lo, hi = np.full(len(n), np.nan), np.full(len(n), np.nan)
        lo[good], hi[good] = wilson(k[good], n[good])
        fraction = np.full(len(n), np.nan)
        np.divide(k, n, out=fraction, where=n > 0)
        ax.axvspan(meso_lo, meso_hi, color="tab:green", alpha=0.15, lw=0)
        ax.fill_between(CENTRES, lo, hi, color="tab:purple", alpha=0.25, lw=0)
        ax.plot(CENTRES, np.where(good, fraction, np.nan), "o-", color="tab:purple", ms=3, lw=0.8)
        ax.axhline(ref[v], color="tab:green", lw=1.0)
        ax.set_ylim(0, 1.15 * max(hi[good].max(initial=0), ref[v]))
        stats.append(pd.DataFrame({
            "target": v, "depth_lo_pct": DEPTH_EDGES[:-1], "depth_hi_pct": DEPTH_EDGES[1:],
            "bwm_n": n, "bwm_fraction": fraction, "wilson_lo": lo, "wilson_hi": hi,
            "meso_fraction": ref[v],
        }))
    pd.concat(stats, ignore_index=True).to_csv(OUT_V2 / "si_depth_profile.csv", index=False,
                                                float_format="%.6f")
    return meso_lo, meso_hi


def plot_decoding(axes, scores):
    grouped = scores.groupby("bin")[list(TARGETS)]
    bins = np.arange(len(CENTRES))
    means = grouped.mean().reindex(bins)
    sem = grouped.sem().reindex(bins)
    counts = grouped.count().reindex(bins, fill_value=0)
    summary = pd.concat([
        pd.DataFrame({
            "target": v, "depth_lo_pct": DEPTH_EDGES[:-1], "depth_hi_pct": DEPTH_EDGES[1:],
            "n_sessions": counts[v], "n_units": N_UNITS, "n_draws": N_DRAWS,
            "mean_score": means[v], "sem": sem[v],
        })
        for v in TARGETS
    ], ignore_index=True)
    summary.to_csv(OUT_V2 / "si_depth_decoding.csv", index=False, float_format="%.6f")
    for ax, v in zip(axes, TARGETS):
        chance = 0 if v.startswith("wheel") else 0.5
        ax.axhline(chance, color="0.5", lw=0.6, ls="--")
        ax.fill_between(CENTRES, means[v] - sem[v], means[v] + sem[v],
                        color="tab:purple", alpha=0.25, lw=0)
        ax.plot(CENTRES, means[v], "o-", color="tab:purple", ms=3, lw=0.8)
    for ax_group, variables, chance in ((axes[:3], TARGETS[:3], 0.5),
                                       (axes[3:4], ("wheel",), 0),
                                       (axes[4:], ("wheel_trial",), 0)):
        values = means[list(variables)]
        error = sem[list(variables)].fillna(0)
        low = min(chance, (values - error).min().min())
        high = (values + error).max().max()
        for ax in ax_group:
            ax.set_ylim(low, high + 0.1 * (high - low))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
            if low == chance:
                ax.spines["bottom"].set_visible(False)
    return counts


def format_row(axes, variables, letters):
    for ax, v, letter in zip(axes, variables, letters):
        ax.set_title(TITLE[v], fontsize=7)
        ax.set_xlim(0, 100)
        style_axes(ax)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        panel_label(ax, letter, x=-0.10, y=1.08)


def main():
    scores = pd.read_parquet(SCORES)
    if set(scores.n_units) != {N_UNITS} or set(scores.n_draws) != {N_DRAWS}:
        raise ValueError("Decoder settings changed. Run python decode_bwm_depth.py.")
    if WHEEL_CACHE.exists():
        wh = pd.read_parquet(WHEEL_CACHE)
    else:
        wh = collect_wheel()
        wh.to_parquet(WHEEL_CACHE, index=False)

    fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.9), layout="constrained")
    top, bottom = fig.subfigures(2, 1, hspace=0.10)
    a = top.subplots(1, 4)
    b = bottom.subplots(1, 5)
    format_row(a, VARIABLES + ("wheel",), "abcd")
    format_row(b, TARGETS, "efghi")
    meso_lo, meso_hi = plot_tuning(a, wh)
    counts = plot_decoding(b, scores)
    top.suptitle("Single-unit tuning", fontsize=8)
    bottom.suptitle(f"BWM decoding with {N_UNITS} units", fontsize=8)
    a[0].set_ylabel("Fraction of units")
    b[0].set_ylabel("ROC-AUC")
    b[3].set_ylabel("$R^2$")
    bottom.supxlabel("Depth below pia (% of local cortical thickness)", fontsize=7)

    if not counts.eq(counts.iloc[:, 0], axis=0).all().all():
        support = " ".join(f"{TITLE[v]} included " + ", ".join(map(str, counts[v]))
                           + " sessions from shallowest to deepest." for v in TARGETS)
    else:
        support = "Session counts from shallowest to deepest were " + ", ".join(map(str, counts.iloc[:, 0])) + "."
    caps = [
        f"Fraction of BWM isocortex units tuned to stimulus side ($p<{ALPHA}$). "
        "Purple bands show 95\\,\\% Wilson intervals for bins with at least 30 units. "
        "Green lines show the fraction across all mesoscope neurons, and green shading shows "
        f"the 5th--95th percentiles of histology-registered mesoscope ROI depth ({meso_lo:.0f}--{meso_hi:.0f}\\,\\%).",
        "Choice.", "Feedback.",
        "Replicable wheel-speed correlation ($r>0.1$ in both even and odd trials).",
        f"Mean held-out stimulus-side ROC-AUC. For each session and bin with at least {N_UNITS} units, "
        f"we refit the decoders on {N_DRAWS} random draws of {N_UNITS} units without replacement within "
        "each draw, using the same cross-validation folds across depths within a session for each "
        "target. We averaged draws within session, then averaged sessions with equal weight. "
        f"Shading shows the standard error across sessions. {support} The session cohort varies by bin.",
        "Choice ROC-AUC.", "Feedback ROC-AUC.",
        "Wheel-speed $R^2$.", "Trial-specific wheel-speed $R^2$. Dashed lines show chance.",
    ]
    save_with_caption(fig, OUT_V2 / "si_depth_profile", caps, "v2_depth_profile",
                      intro="BWM tuning and decoding across five equal bins of relative pial depth. "
                            "The 25\\,$\\mu$m Allen CCF streamlines define 0\\,\\% at the pia "
                            "and 100\\,\\% at the local base of isocortex.")
    plt.close(fig)


if __name__ == "__main__":
    main()
