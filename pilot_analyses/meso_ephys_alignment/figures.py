#!/usr/bin/env python3
"""Figures for the report (report/figures/*.pdf)."""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import OUT, HERE, K, N_NEURONS

FIG = HERE / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COL = {"sensory": "#2a78d6", "motor": "#eb6834"}
MARK = {"sensory": "o", "motor": "^"}
VAR_LABEL = {"stim": "Stimulus side", "choice": "Choice", "feedback": "Feedback", "prior": "Block prior"}
MM = 1 / 25.4

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": .5, "xtick.major.width": .5, "ytick.major.width": .5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def letter(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", fontsize=8)


def log_r_axis(ax, lo=.3, hi=4):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_yticks([.3, 1, 3])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.axhline(1, color="0.5", lw=.5, ls="--", zorder=0)


def strip(ax, d, positions):
    """Region dots per (modality, class); filled = significant coding; black bar = median."""
    rng = np.random.default_rng(0)
    for (modality, cls), x in positions.items():
        s = d[(d.modality == modality) & (d.cls == cls)]
        if s.empty:
            continue
        jit = rng.uniform(-.15, .15, len(s))
        for (_, r), j in zip(s.iterrows(), jit):
            ax.scatter(x + j, r.R, s=12, marker=MARK[cls], lw=.6, zorder=3,
                       facecolor=COL[cls] if r.coding else "white", edgecolor=COL[cls])
        ax.plot([x - .25, x + .25], [s.R.median()] * 2, color="black", lw=1.2, zorder=4)


def fig_main():
    df = pd.read_parquet(OUT / "alignment.pqt")
    d = df[df.cls != "other"]
    pos = {("bwm", "sensory"): 0, ("bwm", "motor"): 1, ("meso", "sensory"): 2.5, ("meso", "motor"): 3.5}
    fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 50 * MM), constrained_layout=True, sharey=True)
    for ax, v, l in zip(axes, ("choice", "stim", "feedback", "prior"), "abcd"):
        strip(ax, d[d.variable == v], pos)
        log_r_axis(ax)
        ax.set_xticks([.5, 3])
        ax.set_xticklabels(["BWM\n(spikes)", "Meso\n(calcium)"])
        ax.set_xlim(-.6, 4.1)
        ax.set_title(VAR_LABEL[v], fontsize=7)
        letter(ax, l)
        if v == "prior":
            ax.text(3, .5, "no biased\nblocks", ha="center", fontsize=6, color="0.4")
    axes[0].set_ylabel(f"Alignment vs noise, R (k = {K}, N = {N_NEURONS})")
    for cls in COL:
        axes[0].scatter([], [], marker=MARK[cls], color=COL[cls], s=12, label=cls)
    axes[0].scatter([], [], marker="o", facecolor="white", edgecolor="0.3", s=12, label="no sig. coding")
    axes[0].legend(frameon=False, loc="upper left", handletextpad=.2, borderaxespad=0)
    fig.savefig(FIG / "fig_main.pdf")


def fig_k():
    z = np.load(OUT / "kcurves.npz", allow_pickle=True)
    keys = pd.DataFrame(list(z["keys"]), columns=["modality", "eid", "region", "variable"])
    df = pd.read_parquet(OUT / "alignment.pqt")
    cls = dict(zip(zip(df.modality, df.eid, df.region), df.cls))
    keys["cls"] = [cls[(a, b, c)] for a, b, c in zip(keys.modality, keys.eid, keys.region)]
    R = z["real"] / z["null"]
    k = np.arange(1, R.shape[1] + 1)
    fig, axes = plt.subplots(1, 3, figsize=(136 * MM, 45 * MM), constrained_layout=True)
    for ax, modality, l in zip(axes[:2], ("bwm", "meso"), "ab"):
        for c in COL:
            m = ((keys.modality == modality) & (keys.cls == c) & (keys.variable == "choice")).values
            med = np.median(R[m], 0)
            q1, q3 = np.percentile(R[m], [25, 75], axis=0)
            ax.fill_between(k, q1, q3, color=COL[c], alpha=.15, lw=0)
            ax.plot(k, med, color=COL[c], lw=1.2, marker=MARK[c], ms=3, label=f"{c} (n={m.sum()})")
        ax.axhline(1, color="0.5", lw=.5, ls="--")
        ax.set_xticks([1, 5, 10])
        ax.set_xlabel("Spontaneous PCs, k")
        ax.set_ylim(0, 2.5)
        ax.set_yticks([0, 1, 2])
        ax.set_title(("BWM" if modality == "bwm" else "Meso") + ", choice", fontsize=7)
        ax.legend(frameon=False, loc="upper right")
        letter(ax, l)
    axes[0].set_ylabel("Alignment vs noise, R")
    ax = axes[2]
    for modality, ls in (("bwm", "-"), ("meso", "--")):
        m = (keys.modality == modality).values
        ax.plot(k, np.median(z["null"][m], 0), color="black", ls=ls, lw=1, label=f"noise null, {modality.upper() if modality == 'bwm' else 'meso'}")
    ax.plot(k, k / N_NEURONS, color="0.6", lw=1, ls=":", label="isotropic, k/N")
    ax.set_xticks([1, 5, 10])
    ax.set_xlabel("Spontaneous PCs, k")
    ax.set_ylabel("Null alignment, A")
    ax.set_ylim(0, .4)
    ax.set_yticks([0, .2, .4])
    ax.legend(frameon=False, loc="upper left")
    letter(ax, "c")
    fig.savefig(FIG / "fig_k.pdf")


def fig_nsweep():
    d = pd.read_parquet(OUT / "n_sweep.pqt")
    d = d[d.cls != "other"]
    fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 48 * MM), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    for ax, v, l in zip(axes[:3], ("choice", "stim", "feedback"), "abc"):
        for c in COL:
            s = d[(d.variable == v) & (d.cls == c)]
            g = s.groupby("N").R
            ax.plot(g.median().index, g.median().values, color=COL[c], marker=MARK[c], ms=3, lw=1.2,
                    label=f"{c} (n={s.region.size // s.N.nunique()})")
        ax.axhline(1, color="0.5", lw=.5, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([30, 100, 200])
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlabel("Neurons per draw, N")
        ax.set_ylim(0, 4)
        ax.set_yticks([0, 2, 4])
        ax.set_title(VAR_LABEL[v], fontsize=7)
        letter(ax, l)
    axes[0].set_ylabel("Median R (k = 3), meso")
    axes[0].legend(frameon=False, loc="upper left")
    ax = axes[3]
    s = d[(d.subject == "SP075") & (d.N == 200)]
    regions = ["MOp", "MOs", "SSp-bfd", "SSp-ll", "SSp-n", "SSp-ul", "SSp-un"]
    w = .26
    for j, (v, hatch) in enumerate((("choice", None), ("stim", "////"), ("feedback", "...."))):
        vals = [s[(s.region == r) & (s.variable == v)].R.iloc[0] for r in regions]
        cols = [COL["motor"] if r.startswith("MO") else COL["sensory"] for r in regions]
        ax.bar(np.arange(len(regions)) + (j - 1) * w, vals, w * .9, color=cols if v == "choice" else "white",
               edgecolor=cols, hatch=hatch, lw=.6, label=VAR_LABEL[v])
    ax.axhline(1, color="0.5", lw=.5, ls="--")
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha="right")
    ax.set_ylim(0, 6)
    ax.set_yticks([0, 3, 6])
    ax.set_ylabel("R (k = 3, N = 200)")
    ax.set_title("SP075, both classes", fontsize=7)
    ax.legend(frameon=False, loc="upper left", ncol=3, handlelength=1.2, columnspacing=.8)
    letter(ax, "d")
    fig.savefig(FIG / "fig_nsweep.pdf")


if __name__ == "__main__":
    fig_main()
    fig_k()
    fig_nsweep()
