#!/usr/bin/env python3
"""Figure 3: wheel-speed correlates are rare among mesoscope neurons, but the
best of a whole session track wheel speed as well as the best BWM units.

Input: output/population.pqt (population.py).
Usage: python fig_population.py -> output/fig3_population.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from wd_common import COL, NAME, NATURE_DOUBLE_COL_IN, OUT, panel_label, save_with_caption, style_axes

MODS = ("meso", "bwm", "bwm_d640")
NAME = dict(NAME, bwm_d640="BWM, \u2264640 \u00b5m")
COL = dict(COL, bwm_d640="tab:pink")


def main():
    d = pd.read_parquet(OUT / "population.pqt")
    med = d.groupby("modality").median(numeric_only=True)
    cnt = d.groupby("modality").size()
    fig, axs = plt.subplots(1, 4, figsize=(NATURE_DOUBLE_COL_IN, 1.9), gridspec_kw=dict(wspace=0.55))

    ax = axs[0]
    bins = np.linspace(0, 0.6, 31)
    for m in MODS:
        ax.hist(d[d.modality == m].frac_pos, bins=bins, color=COL[m], alpha=0.5, density=True, label=NAME[m])
    ax.set_xlabel("fraction of neurons\nwith $r>0.1$ in both halves")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=5)
    style_axes(ax)
    panel_label(ax, "a")

    ax = axs[1]
    for m in MODS:
        x = d[d.modality == m]
        ax.scatter(x.n_neurons, x.r_top10, s=4, color=COL[m], alpha=0.6, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("neurons in session")
    ax.set_ylabel("held-out $r$, top-10 mean")
    style_axes(ax, max_xticks=None)
    ax.set_xticks([10, 100, 1000, 10000], ["10", "100", "1000", "10000"])
    ax.minorticks_off()
    panel_label(ax, "b")

    for j, cols in enumerate((["r_top10", "r_top10_match"], ["r_top10_trial", "r_top10_trial_match"])):
        ax = axs[2 + j]
        for k, m in enumerate(MODS):
            x = d[d.modality == m]
            for i, c in enumerate(cols):
                v = x[c].dropna()
                pos = i + (k - 1) * 0.27
                ax.boxplot(v, positions=[pos], widths=0.22, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=COL[m], alpha=0.6, lw=0.5), medianprops=dict(color="k", lw=0.8),
                           whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
        ax.set_xticks([0, 1], ["all\nneurons", "30\nneurons"])
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_ylabel("held-out $r$, top-10 mean\n" + ("(wheel speed)" if j == 0 else "(trial-specific speed)"))
        style_axes(ax, categorical_x=True)
        panel_label(ax, "cd"[j])

    p = lambda c: mannwhitneyu(d[d.modality == "meso"][c].dropna(), d[d.modality == "bwm"][c].dropna()).pvalue  # noqa
    f = lambda m, c: f"{med.loc[m, c]:.2f}"  # noqa: E731
    caps = [
        "Fraction of neurons per session whose correlation with wheel speed (over all wheel bins) exceeds 0.1 "
        "in both the even and the odd trials: median "
        f"{med.loc['meso', 'frac_pos']:.1%} (mesoscope, {cnt['meso']} sessions), "
        f"{med.loc['bwm', 'frac_pos']:.1%} (BWM isocortex, {cnt['bwm']}), "
        f"{med.loc['bwm_d640', 'frac_pos']:.1%} (BWM units within 640\\,$\\mu$m of the surface, "
        f"{cnt['bwm_d640']}).".replace("%", "\\%"),
        "Correlation with wheel speed in the odd trials of the mean z-scored activity of the 10 neurons that "
        "correlate best on the even trials, against the number of neurons in the session (one dot per session).",
        f"The same with all neurons of a session (median mesoscope {f('meso', 'r_top10')}, BWM "
        f"{f('bwm', 'r_top10')}, $p$ = {p('r_top10'):.2f}) and with sessions subsampled to 30 random neurons "
        f"(mean over 20 draws; {f('meso', 'r_top10_match')} vs {f('bwm', 'r_top10_match')}, "
        f"$p<10^{{{int(np.ceil(np.log10(p('r_top10_match'))))}}}$). Boxes: quartiles; whiskers: 1.5 inter-quartile ranges.",
        "Same against trial-specific speed (wheel speed minus its per-bin mean over the even trials): "
        f"{f('meso', 'r_top10_trial')} vs {f('bwm', 'r_top10_trial')} (all neurons), "
        f"{f('meso', 'r_top10_trial_match')} vs {f('bwm', 'r_top10_trial_match')} (30 neurons).",
    ]
    intro = ("Wheel-speed correlates are rare in the mesoscope but present: among thousands of neurons per "
             "session the best 10 track wheel speed in held-out trials as well as the best BWM units, but a "
             "random set of 30 neurons, as in a small spatial bin, contains far fewer of them than a BWM set "
             "of the same size. Sessions with $\\geq$30 neurons and $\\geq$50 trials.")
    save_with_caption(fig, OUT / "fig3_population", caps, "wd2_population", intro=intro)


if __name__ == "__main__":
    main()
