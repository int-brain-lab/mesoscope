#!/usr/bin/env python3
"""Figure 2: can the wheel-speed decoding be trusted, and what does it measure?

Inputs: output/checks_*.csv (checks.py), output/timing.pqt + timing_summary.csv
(timing.py), output/variants.pqt + variants_sessions.pqt (variants.py).

Usage: python fig_checks.py -> output/fig2_checks.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wd_common import COL, MODS, NAME, NATURE_DOUBLE_COL_IN, OUT, panel_label, save_with_caption, style_axes

def sci(x):
    e = int(np.floor(np.log10(x)))
    return f"${x / 10 ** e:.0f}\\times10^{{{e}}}$"


VAR_ORDER = ["current", "loto", "lag", "resid", "resid_lag", "resid_null"]
VAR_LABEL = {"current": "current", "loto": "LOTO $\\alpha$", "lag": "+ lags", "resid": "trial-\nspecific",
             "resid_lag": "trial-spec.\n+ lags", "resid_null": "trial-spec.\nshuffled"}


def main():
    syn = pd.read_csv(OUT / "checks_synthetic.csv")
    dec = pd.read_csv(OUT / "checks_decoder.csv")
    tgt = pd.read_csv(OUT / "checks_target.csv")
    tim = pd.read_parquet(OUT / "timing.pqt")
    tsum = pd.read_csv(OUT / "timing_summary.csv")
    var = pd.read_parquet(OUT / "variants.pqt")
    ses = pd.read_parquet(OUT / "variants_sessions.pqt")

    fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 4.6))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.45)

    # a: synthetic recovery
    ax = fig.add_subplot(gs[0, 0])
    for m in MODS:
        g = syn[syn.modality == m].groupby("snr")
        ax.errorbar(g.expected.mean(), g.r2.mean(), yerr=g.r2.std(), fmt="o", ms=3, color=COL[m],
                    lw=0.7, capsize=1.5, label=NAME[m])
        ax.plot(g.expected.mean(), g.r2_without.mean(), "x", ms=3, color=COL[m], alpha=0.6)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.5, ls="--")
    ax.set_xlabel("expected $R^2$ = SNR/(1+SNR)")
    ax.set_ylabel("decoded $R^2$")
    ax.legend(frameon=False, fontsize=5)
    style_axes(ax)
    panel_label(ax, "a")

    # b: timing
    ax = fig.add_subplot(gs[0, 1])
    for (m, sid), g in tim.groupby(["modality", "session"]):
        ax.plot(g.lag, g.top, color=COL[m], lw=0.8, alpha=0.9)
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("lag of neural activity (s)")
    ax.set_ylabel("$r$ with wheel speed")
    style_axes(ax)
    panel_label(ax, "b")

    # c: bin-only baseline vs neural R^2 per session
    ax = fig.add_subplot(gs[0, 2])
    cur = var[var.variant == "current"].groupby(["modality", "session"])["all"].mean().rename("neural")
    d = ses.set_index(["modality", "session"]).join(cur)
    for m in MODS:
        x = d.loc[m]
        ax.scatter(x.bin_baseline, x.neural, s=4, color=COL[m], alpha=0.6, lw=0)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.5, ls="--")
    ax.set_xlabel("$R^2$ of bin means only")
    ax.set_ylabel("$R^2$ from neurons (current)")
    style_axes(ax)
    panel_label(ax, "c")

    # d, e: variants, 10 neurons and all neurons
    stats = {}
    for j, ver in enumerate(("n10", "all")):
        ax = fig.add_subplot(gs[1, j])
        for k, m in enumerate(MODS):
            v = var[var.modality == m]
            means = [v[v.variant == x][ver].mean() for x in VAR_ORDER]
            sem = [v[v.variant == x].groupby("session")[ver].mean().sem() for x in VAR_ORDER]
            stats[m, ver] = dict(zip(VAR_ORDER, means))
            ax.bar(np.arange(len(VAR_ORDER)) + (k - 0.5) * 0.38, means, 0.36, yerr=sem, color=COL[m],
                   alpha=0.75, error_kw=dict(lw=0.5, capsize=1), label=NAME[m])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(VAR_ORDER)), [VAR_LABEL[x] for x in VAR_ORDER], fontsize=5, rotation=45, ha="right")
        ax.set_ylabel(f"$R^2$ ({'10 random neurons' if ver == 'n10' else 'all neurons'})")
        style_axes(ax, categorical_x=True)
        panel_label(ax, "de"[j])
        if j == 0:
            ax.legend(frameon=False, fontsize=5)

    # f: per-bin gain from lags, meso vs BWM (all neurons)
    ax = fig.add_subplot(gs[1, 2])
    for m in MODS:
        v = var[var.modality == m].pivot_table(index=["session", "hex"], columns="variant", values="all")
        ax.hist(v["lag"] - v["loto"], bins=np.linspace(-0.1, 0.3, 41), color=COL[m], alpha=0.5, density=True,
                label=NAME[m])
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("$\\Delta R^2$ from lags (all neurons, per bin)")
    ax.set_ylabel("density")
    style_axes(ax)
    panel_label(ax, "f")

    tm = tsum.groupby("modality").peak_lag_pop.agg(["min", "max"])
    bb = ses.groupby("modality").bin_baseline.median()
    fs = ses.groupby("modality").frac_trial_specific.median()
    s = lambda m, ver, x: f"{stats[m, ver][x]:.3f}"  # noqa: E731
    caps = [
        "Decoder check: a synthetic neuron (wheel speed plus Gaussian noise at signal-to-noise ratios 0.1--4) "
        "replacing one of 10 random neurons of a session is recovered at the expected $R^2$ (dots: mean $\\pm$ "
        "s.d. over 5 draws and 3 sessions per modality; crosses: the same 10 neurons without it). The decoder "
        f"reproduces the dorsal-maps GPU decoder (max difference {sci(dec.max_diff_gpu.max())}) and the "
        f"scikit-learn reference ({sci(dec.max_diff_sklearn.max())}). The wheel-speed target agrees with "
        f"brainbox's filtered speed ($r\\geq${np.floor(tgt.r.min() * 1e4) / 1e4:.4f}) once encoder glitches are removed.",
        "Timing at full resolution: correlation of neural activity with wheel speed at lags $-1$ to 1\\,s "
        "(positive: neural activity follows the wheel), mean over the 5\\,\\% most wheel-correlated neurons "
        "selected on the first half of each session and measured on the second half; 2 sessions per modality. "
        f"Mesoscope deconvolved activity peaks {tm.loc['meso', 'min']:.2f}--{tm.loc['meso', 'max']:.2f}\\,s "
        f"after wheel speed, BWM spikes {abs(tm.loc['bwm', 'max']):.2f}\\,s before it.",
        "What the $R^2$ measures: predicting each bin from its mean over the training trials (no neural data) "
        f"gives $R^2$ {bb['meso']:.2f} (mesoscope sessions, median) and {bb['bwm']:.2f} (BWM), far above "
        "decoding from all neurons of a bin (current decoder; one dot per session). Most of the variance is the "
        f"stereotyped time course after movement onset; trial-to-trial differences are "
        f"{fs['meso']:.0%} (mesoscope) and {fs['bwm']:.0%} (BWM) of the variance.".replace("%", "\\%"),
        "Decoder variants, 10 random neurons per 0.4\\,mm bin (mean $\\pm$ s.e.m. over sessions): current "
        "(leave-one-row-out choice of the ridge penalty), leave-one-trial-out (LOTO) choice, neural activity "
        "of the previous, same and next bin as features (lags), trial-specific speed (bin means of the "
        "training trials removed) without and with lags, and the trial-specific target with trials "
        f"shuffled (null). Mesoscope {s('meso', 'n10', 'current')} $\\to$ {s('meso', 'n10', 'lag')} with lags, "
        f"BWM {s('bwm', 'n10', 'current')} $\\to$ {s('bwm', 'n10', 'lag')}.",
        f"Same from all neurons of a bin: mesoscope {s('meso', 'all', 'current')} $\\to$ "
        f"{s('meso', 'all', 'lag')}, BWM {s('bwm', 'all', 'current')} $\\to$ {s('bwm', 'all', 'lag')}; "
        f"trial-specific speed {s('meso', 'all', 'resid_lag')} vs {s('bwm', 'all', 'resid_lag')}.",
        "Gain from the lagged features per (session, bin), all neurons.",
    ]
    intro = ("Checks of the wheel-speed decoding. The decoder and target are correct (after removing encoder "
             "glitches in 16 mesoscope sessions), mesoscope activity lags the wheel by about one 0.2\\,s bin "
             "more than spikes, and the $R^2$ is dominated by the stereotyped speed profile after movement "
             "onset rather than by trial-to-trial differences in speed.")
    save_with_caption(fig, OUT / "fig2_checks", caps, "wd2_checks", intro=intro)


if __name__ == "__main__":
    main()
