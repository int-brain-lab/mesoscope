#!/usr/bin/env python3
"""Side-by-side dorsal cortex maps, meso (top row) vs BWM (bottom row).

Inputs: output/tuning_<modality>.pqt (one row per neuron, also used for the
neuron/session counts) and output/decoding_<modality>.pqt.

Figures (each as <name>_panels.pdf, <name>.tex, <name>.pdf with caption):
- dorsal_counts: neurons per bin; sessions with >= MIN_NEURONS neurons per bin
- dorsal_decoding_all / dorsal_decoding_n10: mean decoding score across
  sessions per bin, all neurons / MIN_NEURONS random neurons
- dorsal_tuning: fraction of neurons with p < ALPHA per bin
Also writes output/dorsal_maps_summary.csv with the headline numbers.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, Normalize

from common import (ALPHA, HEX_SIZE, MIN_NEURONS, N_SHUF, OUT, VARIABLES, WHEEL_EDGES, WINDOWS,
                    bin_neurons, hex_center, hex_unkey, top_boundaries)
from decode import MIN_TRIALS, N_DRAWS
from plotstyle import NATURE_DOUBLE_COL_IN, NATURE_ONEHALF_COL_IN, save_with_caption, style_axes

MODS = ("meso", "bwm")
ROW_NAME = {"meso": "Mesoscope", "bwm": "BWM (Neuropixels)"}
XLIM, YLIM = (-0.4, 5.6), (-5.8, 3.9)
CMAP = "magma"
_ANG = np.deg2rad(30 + 60 * np.arange(6))
HEX_VERTS = HEX_SIZE * np.c_[np.cos(_ANG), np.sin(_ANG)]
TARGET_LABEL = {"stim": "stimulus side", "choice": "choice", "feedback": "feedback",
                "wheel": "wheel speed"}

_BND = None


def outline(ax):
    global _BND
    if _BND is None:
        _BND = top_boundaries()
    bnd, xs, ys = _BND
    ax.imshow(np.ma.masked_where(bnd == 0, bnd), extent=[xs[0], xs[-1], ys[-1], ys[0]],
              origin="upper", cmap="Greys", vmin=0, vmax=1.6, interpolation="antialiased",
              zorder=3, alpha=0.6)
    ax.axvline(0, color="0.55", lw=0.25)


def hexmap(ax, keys, values, norm, cmap=CMAP, size=HEX_SIZE, alpha=None):
    """alpha: optional per-bin opacity (array)."""
    q, r = hex_unkey(np.asarray(keys))
    cx, cy = hex_center(q, r, size)
    hv = HEX_VERTS * size / HEX_SIZE
    verts = np.stack([np.c_[x + hv[:, 0], y + hv[:, 1]] for x, y in zip(cx, cy)]) \
        if len(cx) else np.zeros((0, 6, 2))
    pc = PolyCollection(verts, array=np.asarray(values, float), cmap=cmap, norm=norm,
                        edgecolors="none", zorder=2)
    if alpha is not None:
        pc.set_alpha(np.asarray(alpha, float))
    ax.add_collection(pc)
    return pc


def style_map(ax, row, col, nrows, xlim=XLIM):
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([0, 2.5, 5] if xlim[0] > -1 else [-5, 0, 5])
    ax.set_yticks([-5, 0, 2.5])
    if row == nrows - 1:
        ax.set_xlabel("ML (mm)")
    else:
        ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("AP (mm)")
    else:
        ax.set_yticklabels([])


def map_grid(per_mod, titles, cbar_labels, norms, out_stem, captions, label, intro, xlim=XLIM,
             rows=MODS, row_names=ROW_NAME, size=HEX_SIZE, out_dir=None, cmap=CMAP):
    """per_mod[row][c] = (keys, values) or (keys, values, alpha) for row
    (modality or session) and column c."""
    from matplotlib.cm import ScalarMappable
    ncol, nrow = len(titles), len(rows)
    w = NATURE_DOUBLE_COL_IN if ncol > 2 else NATURE_DOUBLE_COL_IN * 0.6
    aspect = (YLIM[1] - YLIM[0]) / (xlim[1] - xlim[0])
    fig, axs = plt.subplots(nrow, ncol, figsize=(w, w / ncol * nrow * aspect + 0.6), squeeze=False)
    for c in range(ncol):
        for r, m in enumerate(rows):
            ax = axs[r, c]
            outline(ax)
            keys, vals, *alpha = per_mod[m][c]
            hexmap(ax, keys, vals, norms[c], cmap=cmap, size=size, alpha=alpha[0] if alpha else None)
            pc = ScalarMappable(norm=norms[c], cmap=cmap)
            style_map(ax, r, c, nrow, xlim)
            if r == 0:
                ax.set_title(titles[c])
                ax.text(-0.12 if c else -0.3, 1.1, chr(ord("a") + c), transform=ax.transAxes,
                        fontsize=8, fontweight="bold", va="bottom")
            if c == 0:
                ax.text(-0.42, 0.5, row_names[m], transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=7)
        cb = fig.colorbar(pc, ax=axs[:, c], orientation="horizontal", fraction=0.04, pad=0.09,
                          aspect=15)
        cb.set_label(cbar_labels[c], fontsize=6)
        cb.ax.tick_params(labelsize=5)
        if not isinstance(norms[c], LogNorm):
            cb.locator = plt.MaxNLocator(3)
            cb.update_ticks()
    save_with_caption(fig, (out_dir or OUT) / out_stem, captions, label=label, intro=intro)
    plt.close(fig)


def joint_norm(arrays, log=False, lo=None, pct=(1, 99)):
    v = np.concatenate([np.asarray(a, float) for a in arrays])
    v = v[np.isfinite(v)]
    if log:
        return LogNorm(vmin=max(1, v.min()), vmax=v.max())
    a = np.percentile(v, pct[0]) if lo is None else lo
    return Normalize(vmin=a, vmax=np.percentile(v, pct[1]))


def main():
    tun = {m: pd.read_parquet(OUT / f"tuning_{m}.pqt") for m in MODS}
    dec = {m: pd.read_parquet(OUT / f"decoding_{m}.pqt") for m in MODS}
    for m in MODS:
        tun[m]["hex"] = bin_neurons(tun[m][["x", "y", "z"]].values)
    summary = []
    windows_figure()

    # ---- 1. counts ------------------------------------------------------
    per = {}
    for m in MODS:
        n = tun[m].groupby("hex").size()
        ns = tun[m].groupby(["hex", "session"]).size()
        ses = (ns >= MIN_NEURONS).groupby("hex").sum()
        ses = ses[ses > 0]
        per[m] = [(n.index, n.values), (ses.index, ses.values)]
        summary.append(dict(modality=m, neurons=len(tun[m]), sessions=tun[m].session.nunique(),
                            bins_with_neurons=len(n), bins_with_session=len(ses)))
    norms = [joint_norm([per[m][0][1] for m in MODS], log=True),
             joint_norm([per[m][1][1] for m in MODS], log=True)]
    sm = {d["modality"]: d for d in summary}
    map_grid(per, ["Neurons", f"Sessions ($\\geq${MIN_NEURONS} neurons)"],
             ["neurons per bin", "sessions per bin"], norms, "dorsal_counts",
             [f"Total number of isocortex neurons recorded in each spatial bin; mesoscope "
              f"(top; {sm['meso']['neurons']:,} neurons, {sm['meso']['sessions']} sessions) and BWM "
              f"(bottom; {sm['bwm']['neurons']:,} good units, {sm['bwm']['sessions']} sessions).",
              f"Number of sessions contributing at least {MIN_NEURONS} neurons to each bin."],
             "dorsal_counts", intro=_intro())

    # ---- 2. decoding ----------------------------------------------------
    targets = ["stim", "choice", "feedback", "wheel"]
    for ver in ("all", "n10"):
        per = {}
        for m in MODS:
            g = dec[m].groupby(["target", "hex"])[ver].mean()
            per[m] = [(g[t].dropna().index, g[t].dropna().values) for t in targets]
        # one scale for the three binary targets (0.5 = chance), so colour differences
        # between variables are not exaggerated by per-panel scales
        binary = joint_norm([per[m][i][1] for m in MODS for i, t in enumerate(targets) if t != "wheel"],
                            lo=0.5)
        norms = [binary if t != "wheel" else joint_norm([per[m][i][1] for m in MODS], lo=0)
                 for i, t in enumerate(targets)]
        who = ("all neurons of the session in the bin" if ver == "all" else
               f"{MIN_NEURONS} randomly drawn neurons of the session in the bin (mean of {N_DRAWS} draws)")
        map_grid(per, [TARGET_LABEL[t].capitalize() for t in targets],
                 ["balanced accuracy"] * 3 + ["$R^2$"], norms, f"dorsal_decoding_{ver}",
                 [f"Stimulus side (left vs right, including 0\\% contrast) decoded from {who}: "
                  f"cross-validated balanced accuracy (chance 0.5), averaged across sessions with "
                  f"$\\geq${MIN_NEURONS} neurons in the bin; activity in the {_win('stim')}.",
                  f"Same for choice (left vs right wheel turn), {_win('choice')}.",
                  f"Same for feedback (reward vs error), {_win('feedback')}.",
                  f"Wheel speed (mean absolute wheel velocity) in 0.2\\,s bins from "
                  f"${WHEEL_EDGES[0]:g}$ to ${WHEEL_EDGES[-1]:g}$\\,s around movement onset, decoded "
                  f"from the activity in the same bins: cross-validated $R^2$ of ridge regression "
                  f"(chance 0)."],
                 f"dorsal_decoding_{ver}", intro=_intro() + " " + _decoding_methods(ver))
        for m in MODS:
            for t in targets:
                d = dec[m][dec[m].target == t]
                summary.append(dict(modality=m, measure=f"decoding_{ver}_{t}",
                                    session_bins=len(d), mean=d[ver].mean()))
        summary += decoding_scatter(dec, ver, targets)

    # ---- 3. tuning --------------------------------------------------------
    for m in MODS:
        for v in VARIABLES:
            summary.append(dict(modality=m, measure=f"frac_sig_{v}",
                                mean=float(np.mean(tun[m][f"p_{v}"].dropna() < ALPHA))))
    tuning_maps(tun, "dorsal_tuning", fold=True)
    tuning_maps(tun, "dorsal_tuning_unfolded", fold=False)
    pd.DataFrame(summary).to_csv(OUT / "dorsal_maps_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string())


def windows_figure():
    """Analysis windows per variable and modality, relative to the aligning event."""
    from matplotlib.patches import Rectangle
    ev = {"stimOn_times": "stimulus onset", "firstMovement_times": "movement onset",
          "feedback_times": "feedback"}
    col = {"bwm": "0.25", "meso": plt.get_cmap(CMAP)(0.6)}
    rows = [(v, {m: [WINDOWS[m][v][1:]] for m in MODS}, ev[WINDOWS["bwm"][v][0]]) for v in VARIABLES]
    wb = list(zip(WHEEL_EDGES[:-1], WHEEL_EDGES[1:]))
    rows.append(("wheel", {m: wb for m in MODS}, "movement onset"))
    fig, axs = plt.subplots(len(rows), 1, figsize=(NATURE_ONEHALF_COL_IN, 2.6), sharex=True)
    for i, (ax, (v, wins, event)) in enumerate(zip(axs, rows)):
        ax.text(-0.2, 1.02, chr(ord("a") + i), transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom")
        for yi, m in enumerate(("meso", "bwm")):
            for a, b in wins[m]:
                ax.add_patch(Rectangle((a, yi + 0.15), b - a, 0.7, facecolor=col[m],
                                       edgecolor="white", lw=0.6))
        ax.axvline(0, color="k", lw=0.6)
        ax.set_ylim(0, 2)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Mesoscope", "BWM"], fontsize=6)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"{TARGET_LABEL[v].capitalize()}, aligned to {event}", fontsize=7, loc="left",
                     pad=2)
    axs[-1].set_xlim(-0.45, 1.05)
    axs[-1].set_xticks([-0.2, 0, 0.5, 1])
    axs[-1].set_xlabel("time from event (s)")
    fig.tight_layout(h_pad=0.4)
    save_with_caption(
        fig, OUT / "dorsal_windows",
        [f"Stimulus side: activity in {_win('stim')}.",
         f"Choice: {_win('choice')}.",
         f"Feedback: {_win('feedback')}.",
         f"Wheel speed: six 0.2\\,s bins from ${WHEEL_EDGES[0]:g}$ to ${WHEEL_EDGES[-1]:g}$\\,s "
         f"around movement onset, identical for both modalities; each bin's activity is paired "
         f"with the mean absolute wheel velocity in the same bin."],
        label="dorsal_windows",
        intro=("Analysis windows used for decoding and for the single-neuron tuning test. BWM "
               "windows follow the Brain Wide Map single-cell analyses. Mesoscope frames are about "
               "0.2\\,s apart, so the mesoscope stimulus, choice and feedback windows are widened to "
               "0.3\\,s to always contain at least one frame; the mesoscope value is the mean of the "
               "deconvolved trace, linearly interpolated at each neuron's own sampling times, over "
               "the window. BWM values are firing rates (spike count / window length)."))
    plt.close(fig)


def decoding_scatter(dec, ver, targets):
    """Per hex bin: session-averaged decoding score from BWM (x) vs mesoscope
    (y), for bins that have both. Returns summary rows (Spearman r per target)."""
    from scipy.stats import spearmanr
    g = {m: dec[m].groupby(["target", "hex"])[ver].agg(["mean", "size"]) for m in MODS}
    w = NATURE_DOUBLE_COL_IN
    fig, axs = plt.subplots(1, len(targets), figsize=(w, w / len(targets) + 0.55))
    rows, stats = [], []
    for c, (ax, t) in enumerate(zip(axs, targets)):
        both = g["bwm"].loc[t].join(g["meso"].loc[t], lsuffix="_bwm", rsuffix="_meso", how="inner").dropna()
        x, y = both["mean_bwm"].values, both["mean_meso"].values
        _, ap = hex_center(*hex_unkey(both.index.values))
        sc = ax.scatter(x, y, c=ap, cmap="viridis", vmin=-5, vmax=3, s=7, lw=0, alpha=0.85)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        pad = 0.05 * (hi - lo)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color="0.6", lw=0.5, ls="--", zorder=0)
        chance = 0.5 if t != "wheel" else 0.0
        ax.axhline(chance, color="0.85", lw=0.5, zorder=0)
        ax.axvline(chance, color="0.85", lw=0.5, zorder=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        style_axes(ax)
        r, p = spearmanr(x, y)
        ax.text(0.04, 0.97, f"$r_s$ = {r:.2f}\nn = {len(x)}", transform=ax.transAxes, va="top",
                fontsize=6)
        ax.set_title(TARGET_LABEL[t].capitalize())
        ax.set_xlabel("BWM " + ("$R^2$" if t == "wheel" else "bal. acc."))
        if c == 0:
            ax.set_ylabel("Mesoscope score")
        ax.text(-0.12 if c else -0.3, 1.1, chr(ord("a") + c), transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="bottom")
        rows.append(dict(modality="both", measure=f"scatter_{ver}_{t}", session_bins=len(x),
                         mean=r, p=p))
        stats.append((r, p, len(x)))
    fig.tight_layout()
    cb = fig.colorbar(sc, ax=axs, orientation="vertical", fraction=0.015, pad=0.02, aspect=20)
    cb.set_label("AP (mm)", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    cb.locator = plt.MaxNLocator(3)
    cb.update_ticks()
    who = ("all neurons of the session in the bin" if ver == "all" else
           f"{MIN_NEURONS} randomly drawn neurons of the session in the bin (mean of {N_DRAWS} draws)")
    def fmt(i):
        r, p, n = stats[i]
        m, e = f"{p:.1e}".split("e")
        ptex = f"{p:.2g}" if p >= 1e-3 else f"{m}\\times 10^{{{int(e)}}}"
        return f"Spearman $r_s = {r:.2f}$, $p = {ptex}$, {n} bins"
    save_with_caption(
        fig, OUT / f"dorsal_decoding_{ver}_scatter",
        [f"Stimulus side ({fmt(0)}).", f"Choice ({fmt(1)}).", f"Feedback ({fmt(2)}).",
         f"Wheel speed, cross-validated $R^2$ ({fmt(3)})."],
        label=f"dorsal_decoding_{ver}_scatter",
        intro=(f"Decoding per spatial bin, BWM Neuropixels (x) vs mesoscope (y), decoded from {who}. "
               f"Each point is one hexagonal bin (0.4\\,mm, hemispheres folded) with at least one "
               f"session of each modality contributing $\\geq${MIN_NEURONS} neurons; its value is the "
               f"score averaged across those sessions (same values as the decoding maps). Colour: "
               f"AP position of the bin. Dashed line: identity; grey lines: chance (0.5 balanced "
               f"accuracy, 0 for $R^2$). The $p$-values ignore spatial autocorrelation between "
               f"neighbouring bins and are far too small; tested against surrogate maps that keep the "
               f"spatial autocorrelation, none of these correlations is significant at $p<0.05$ except "
               f"wheel speed with 10 neurons ($p=0.044$; Supplementary Information, spatial null)."
               + (" With all neurons, mesoscope bins hold far more neurons than BWM bins, which "
                  "inflates the mesoscope scores." if ver == "all" else "")))
    plt.close(fig)
    return rows


def tuning_maps(tun, out_stem, fold):
    """Fraction of significantly tuned neurons per bin (the original
    meso-vs-BWM figure); fold=False keeps both hemispheres as recorded."""
    per = {}
    for m in MODS:
        keys = bin_neurons(tun[m][["x", "y", "z"]].values, fold=fold)
        cols = []
        for v in VARIABLES:
            ok = tun[m][f"p_{v}"].notna().values  # neurons the test could be run on
            g = (tun[m][f"p_{v}"][ok] < ALPHA).groupby(keys[ok])
            frac = g.mean()[g.count() >= MIN_NEURONS]
            cols.append((frac.index, frac.values))
        per[m] = cols
    norms = [joint_norm([per[m][i][1] for m in MODS], lo=0) for i in range(3)]
    map_grid(per, ["Stimulus side", "Choice", "Feedback"],
             ["fraction significant"] * 3, norms, out_stem,
             [f"Fraction of neurons in each bin significantly tuned to stimulus side "
              f"($p<{ALPHA}$; bins with $\\geq${MIN_NEURONS} neurons pooled across sessions); "
              f"left vs right stimulus, compared within block $\\times$ choice; {_win('stim')}.",
              f"Same for choice (left vs right turn, within block $\\times$ stimulus side); "
              f"{_win('choice')}.",
              f"Same for feedback (reward vs error, within block $\\times$ stimulus side); "
              f"{_win('feedback')}."],
             out_stem,
             intro=_intro(fold) + " " + _tuning_methods(),
             xlim=XLIM if fold else (-5.8, 5.8))


def _intro(fold=True):
    return (f"Dorsal cortex maps, mesoscope two-photon imaging (top row; deconvolved activity "
            f"of isocortex neurons, canonical sessions) vs Brain Wide Map Neuropixels recordings "
            f"(bottom row; good isocortex units, 2023 release). Each neuron is assigned to a "
            f"hexagonal bin ({HEX_SIZE*2:.1f}\\,mm vertex to vertex) by its medio-lateral (ML) and "
            f"antero-posterior (AP) Allen CCF position relative to bregma; "
            + ("both hemispheres are folded onto the right one (ML $\\to$ $|$ML$|$). " if fold
               else "both hemispheres shown as recorded. ")
            + "Grey lines: Allen top-view region boundaries. Trials: choice made and reaction time "
              "0.08--2\\,s (BWM criteria). Colour scales are shared between the two rows.")


def _win(v):
    """Caption text for the analysis windows of variable v, both modalities."""
    ev = {"stimOn_times": "stimulus onset", "firstMovement_times": "movement onset",
          "feedback_times": "feedback"}
    rng = lambda w: f"$[{w[1]:g}, {w[2]:g}]$\\,s"  # noqa: E731
    b, m = WINDOWS["bwm"][v], WINDOWS["meso"][v]
    return f"window {rng(b)} (BWM) or {rng(m)} (mesoscope) relative to {ev[b[0]]}"


def _tuning_methods():
    return (f"Single-neuron tuning: the BWM single-cell test (brainwidemap single\\_cell\\_stats, "
            f"re-implemented on the GPU and checked against the original), applied identically to "
            f"both modalities to the mean activity in each window. Mann--Whitney $U$ statistics are "
            f"computed within condition strata, summed and normalised; the null distribution comes "
            f"from {N_SHUF} permutations of the condition labels within consecutive blocks, and $p$ "
            f"is the rank of the observed statistic. Mesoscope windows are 0.3\\,s long because "
            f"the frame period is about 0.2\\,s.")


def _decoding_methods(ver):
    return (f"Decoding per session and bin, for every session with $\\geq${MIN_TRIALS} trials "
            f"and $\\geq${MIN_NEURONS} neurons in the bin: 5-fold stratified cross-validation; "
            f"binary variables with L2-regularised logistic regression (balanced class weights, "
            f"regularisation chosen by inner 3-fold cross-validation), wheel speed with ridge "
            f"regression (penalty chosen by leave-one-out error, folds split by trial). The three "
            f"binary variables share one colour scale starting at chance (0.5)."
            + (" With all neurons, mesoscope bins hold far more neurons than BWM bins, so scores "
               "are not comparable across rows; see the 10-neuron version." if ver == "all" else
               " A fixed population size makes the two rows comparable."))


if __name__ == "__main__":
    main()
