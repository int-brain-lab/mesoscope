#!/usr/bin/env python3
"""Revised dorsal maps (v2): 1 mm hexagons, depth-limited BWM, revised wheel decoding.

Rows: mesoscope; BWM units within 640 um of the pial surface (bwm_d640, the
main comparison); BWM units at all depths (reference).
Inputs: output/tuning_meso.pqt, output/tuning_bwm.pqt (one row per neuron;
the bwm_d640 rows are the BWM neurons within 640 um, same test), and
output/v2/decoding_<modality>.pqt (decode_v2.py).

Figures in output/v2/ (each _panels.pdf, .tex, .pdf with caption):
- v2_counts: neurons and sessions (>= N_PICK neurons) per bin
- v2_decoding_n20 / v2_decoding_all: stimulus side, choice, feedback, wheel speed,
  trial-specific wheel speed
- v2_tuning: fraction of tuned neurons
- v2_summary: mean score per target and row, and per-bin mesoscope vs BWM (<=640 um)
Also output/v2/summary_v2.csv.

Usage: python maps_v2.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from common import ALPHA, MIN_NEURONS, OUT, VARIABLES, WHEEL_EDGES, bin_neurons
from decode_v2 import HEX_V2, N_DRAWS, N_PICK, OUT_V2
from make_bwm_depth import MAX_DEPTH_UM, depth_below_surface
from maps import _tuning_methods, _win, joint_norm, map_grid
from plotstyle import NATURE_DOUBLE_COL_IN, save_with_caption, style_axes

ROWS = ("meso", "bwm_d640", "bwm")
ROW_NAME = {"meso": "Mesoscope", "bwm_d640": f"BWM $\\leq${MAX_DEPTH_UM} µm",
            "bwm": "BWM all depths", "bwm_probe": "BWM per probe"}
ROW_TEXT = {"meso": "mesoscope", "bwm_d640": f"BWM $\\leq${MAX_DEPTH_UM}\\,$\\mu$m", "bwm": "BWM all depths",
            "bwm_probe": "BWM per probe"}
COL = {"meso": "tab:green", "bwm_d640": "tab:purple", "bwm": "0.6"}
TARGETS = ("stim", "choice", "feedback", "wheel", "wheel_trial")
TITLE = {"stim": "Stimulus side", "choice": "Choice", "feedback": "Feedback", "wheel": "Wheel speed",
         "wheel_trial": "Wheel speed,\ntrial-specific"}
CHANCE = {"stim": 0.5, "choice": 0.5, "feedback": 0.5, "wheel": 0.0, "wheel_trial": 0.0}


def load():
    from iblatlas.atlas import AllenAtlas
    from decode_v2 import excluded_sessions
    tun = {m: pd.read_parquet(OUT / f"tuning_{m}.pqt") for m in ("meso", "bwm")}
    tun["meso"] = tun["meso"][~tun["meso"].session.isin(excluded_sessions())]
    ba = AllenAtlas()
    ba.compute_surface()
    d = depth_below_surface(tun["bwm"][["x", "y", "z"]].values, ba)
    tun["bwm_d640"] = tun["bwm"][d <= MAX_DEPTH_UM].copy()
    for m in ROWS:
        tun[m]["hex"] = bin_neurons(tun[m][["x", "y", "z"]].values, size=HEX_V2)
    dec = {m: pd.read_parquet(OUT_V2 / f"decoding_{m}.pqt") for m in ROWS}
    dec["meso"] = dec["meso"][~dec["meso"].session.isin(excluded_sessions())]
    dec["bwm_probe"] = pd.read_parquet(OUT_V2 / "decoding_bwm_probe.pqt")   # decode_probe.py
    return tun, dec


def intro(title="Dorsal cortex maps"):
    return (f"{title}: mesoscope two-photon imaging (top row; deconvolved activity of "
            "isocortex neurons) vs Brain Wide Map Neuropixels units within "
            f"{MAX_DEPTH_UM}\\,$\\mu$m of the pial surface (middle row; twice the depth range the "
            "mesoscope images, 174--318\\,$\\mu$m) and at all depths (bottom row; reference). Each neuron is "
            f"assigned to a hexagonal bin ({HEX_V2 * 2:.0f}\\,mm vertex to vertex) by its ML and AP Allen "
            "CCF position; hemispheres folded onto the right. Grey lines: Allen top-view region "
            "boundaries. Colour scales are shared across rows.")


def decoding_methods(ver):
    return ("Decoding per session and bin (sessions with $\\geq$50 trials and $\\geq$"
            f"{N_PICK} neurons in the bin), 5-fold cross-validation over trials. Binary variables: "
            "L2 logistic regression, balanced accuracy (chance 0.5). Wheel speed in six 0.2\\,s bins from "
            f"${WHEEL_EDGES[0]:g}$ to ${WHEEL_EDGES[-1]:g}$\\,s around movement onset: ridge regression on "
            "the activity of the previous, same and next bin (penalty by exact leave-one-trial-out error), "
            "cross-validated $R^2$; trial-specific: the same with the mean speed profile of the training "
            "trials subtracted, so only trial-to-trial differences in speed count."
            + (f" {N_PICK} randomly drawn neurons per session and bin (mean of {N_DRAWS} draws), so "
               "rows are compared at the same population size." if ver == "n20" else
               " All neurons of the session in the bin: mesoscope bins hold far more neurons than BWM bins, "
               "so rows are not comparable in absolute terms."))


def counts(tun, summary):
    per = {}
    for m in ROWS:
        n = tun[m].groupby("hex").size()
        ns = tun[m].groupby(["hex", "session"]).size()
        ses = (ns >= N_PICK).groupby("hex").sum()
        ses = ses[ses > 0]
        per[m] = [(n.index, n.values), (ses.index, ses.values)]
        summary.append(dict(row=m, measure="neurons", value=len(tun[m])))
        summary.append(dict(row=m, measure="sessions", value=tun[m].session.nunique()))
        summary.append(dict(row=m, measure="bins_with_session", value=len(ses)))
    norms = [joint_norm([per[m][0][1] for m in ROWS], log=True), joint_norm([per[m][1][1] for m in ROWS], log=True)]
    nn = {m: (len(tun[m]), tun[m].session.nunique()) for m in ROWS}
    map_grid(per, ["Neurons", f"Sessions ($\\geq${N_PICK} neurons)"], ["neurons per bin", "sessions per bin"],
             norms, "v2_counts",
             ["Number of isocortex neurons per bin: " + "; ".join(
                 f"{ROW_TEXT[m]} {nn[m][0]:,} neurons, {nn[m][1]} sessions" for m in ROWS) + ".",
              f"Number of sessions contributing at least {N_PICK} neurons to each bin."],
             "v2_counts", intro=intro("Coverage"), rows=ROWS, row_names=ROW_NAME, size=HEX_V2, out_dir=OUT_V2)


def decoding(dec, ver, summary):
    from controls2 import SUBJ, opacity
    per = {}
    for m in ROWS:
        d = dec[m].dropna(subset=[ver]).assign(subject=lambda x: x.session.map(SUBJ))
        g = d.groupby(["target", "hex"]).agg(v=(ver, "mean"), na=("subject", "nunique"))
        per[m] = [(g.loc[t].index, g.loc[t].v.values, opacity(g.loc[t].na.values)) if t in g.index.levels[0]
                  else ([], [], []) for t in TARGETS]
    rows = ROWS
    if ver == "all":     # extra row: BWM decoded per probe, a circle at the probe's entry point
        rows = ROWS + ("bwm_probe",)
        p = dec["bwm_probe"].sort_values("all")          # strongest drawn on top where probes overlap
        per["bwm_probe"] = [dict(x=np.abs(p.entry_x[p.target == t].values), y=p.entry_y[p.target == t].values,
                                 v=p["all"][p.target == t].values) for t in TARGETS]
    # colour scales from the hexagon rows only (probe circles above the top saturate)
    binary = joint_norm([per[m][i][1] for m in ROWS for i, t in enumerate(TARGETS[:3]) if len(per[m][i][1])], lo=0.5)
    norms = [binary] * 3 + [joint_norm([per[m][3][1] for m in ROWS], lo=0),
                            joint_norm([per[m][4][1] for m in ROWS], lo=0)]
    for m in ROWS:
        for t in TARGETS:
            d = dec[m][dec[m].target == t][ver]
            summary.append(dict(row=m, measure=f"decoding_{ver}_{t}", value=d.mean(), session_bins=d.notna().sum()))
    if ver == "n20":
        title = f"Decoding from {N_PICK} neurons"
        who = (f"{N_PICK} random neurons of one session in that bin (mean of {N_DRAWS} draws; "
               f"sessions with $\\geq${N_PICK} neurons there)")
    else:
        title = "Decoding from all neurons"
        who = (f"all neurons of one session in that bin (sessions with $\\geq${N_PICK} neurons there); "
               "mesoscope bins hold far more neurons than BWM bins")
        p = dec["bwm_probe"].drop_duplicates("pid")
        probe = (f" Bottom row: BWM decoded per probe from all its isocortex units at all depths, whichever bins "
                 f"they fall in ({len(p)} probes with $\\geq${N_PICK} units, median {p.n_neurons.median():.0f}); "
                 "each circle sits at the probe's brain entry point (histology-aligned Alyx trajectory, "
                 "hemispheres folded), higher scores drawn on top where circles overlap; colour scales as in the rows "
                 "above, so circles beyond the top of a scale saturate.")
    head = (f"\\textbf{{{title}}}. Rows: mesoscope; BWM units within {MAX_DEPTH_UM}\\,$\\mu$m of the pial "
            "surface; BWM at all depths. Each bin (1\\,mm hexagon, hemispheres folded) shows the mean over "
            f"sessions of decoders trained on {who}. Opacity: number of animals (full at $\\geq$4)."
            + (probe if ver == "all" else ""))
    map_grid(per, [TITLE[t] for t in TARGETS], ["balanced accuracy"] * 3 + ["$R^2$"] * 2, norms,
             f"v2_decoding_{ver}",
             ["Stimulus side, balanced accuracy (chance 0.5; colour scale shared by \\textbf{a}--\\textbf{c}).",
              "Choice.", "Feedback.", "Wheel speed, cross-validated $R^2$.", "Trial-specific wheel speed, $R^2$."],
             f"v2_decoding_{ver}", intro=head, rows=rows, row_names=ROW_NAME, size=HEX_V2, out_dir=OUT_V2)


def tuning(tun, summary):
    per = {}
    for m in ROWS:
        cols = []
        for v in VARIABLES:
            ok = tun[m][f"p_{v}"].notna().values
            g = (tun[m][f"p_{v}"][ok] < ALPHA).groupby(tun[m]["hex"].values[ok])
            frac = g.mean()[g.count() >= MIN_NEURONS]
            cols.append((frac.index, frac.values))
            summary.append(dict(row=m, measure=f"frac_tuned_{v}", value=float(np.mean(tun[m][f"p_{v}"].dropna() < ALPHA))))
        per[m] = cols
    norms = [joint_norm([per[m][i][1] for m in ROWS], lo=0) for i in range(3)]
    map_grid(per, ["Stimulus side", "Choice", "Feedback"], ["fraction significant"] * 3, norms, "v2_tuning",
             [f"Fraction of neurons per bin tuned to stimulus side ($p<{ALPHA}$; bins with $\\geq${MIN_NEURONS} "
              f"neurons pooled across sessions); {_win('stim')}.", f"Choice; {_win('choice')}.",
              f"Feedback; {_win('feedback')}."],
             "v2_tuning", intro=intro("Tuning results") + " " + _tuning_methods(), rows=ROWS, row_names=ROW_NAME, size=HEX_V2,
             out_dir=OUT_V2)


def summary_figure(dec, summary):
    from controls2 import spatial_test
    fig, axs = plt.subplots(2, 5, figsize=(NATURE_DOUBLE_COL_IN, 3.6), gridspec_kw=dict(hspace=0.75, wspace=0.55))
    stats = {}
    for c, t in enumerate(TARGETS):
        ax = axs[0, c]
        for k, m in enumerate(ROWS):
            for j, ver in enumerate(("n20", "all")):
                d = dec[m][dec[m].target == t]
                ses = d.groupby("session")[ver].mean()
                ax.bar(j + (k - 1) * 0.27, ses.mean(), 0.25, yerr=ses.sem(), color=COL[m], alpha=0.8 if j == 0 else 0.5,
                       error_kw=dict(lw=0.5, capsize=1), label=ROW_TEXT[m] if (c == 0 and j == 0) else None)
                stats[m, t, ver] = ses.mean()
        ax.axhline(CHANCE[t], color="k", lw=0.5)
        ax.set_xticks([0, 1], [f"{N_PICK} neurons", "all"], fontsize=5)
        if CHANCE[t]:
            ax.set_ylim(CHANCE[t] - 0.02, None)
        ax.set_title(TITLE[t], fontsize=6)
        style_axes(ax, categorical_x=True)
        if c == 0:
            ax.set_ylabel("score (mean over sessions)")
        ax = axs[1, c]
        g = {m: dec[m][dec[m].target == t].groupby("hex")["n20"].mean() for m in ("meso", "bwm_d640")}
        both = pd.concat(g, axis=1, join="inner").dropna()
        ax.scatter(both["bwm_d640"], both["meso"], s=6, color="k", lw=0)
        lo, hi = np.nanmin(both.values), np.nanmax(both.values)
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls="--")
        st = spatial_test(f"v2_{t}", both.index.values, both["bwm_d640"].values, both["meso"].values, HEX_V2)
        r, p = st["r"], st["p_spatial"]
        stats[t, "r"] = (r, p, len(both), st["p_naive"])
        ax.text(0.04, 0.97, f"$r_s$={r:.2f}\n$p_{{sp}}$={p:.2f}", transform=ax.transAxes, va="top", fontsize=5)
        ax.set_xlabel("BWM $\\leq$640 µm", fontsize=6)
        if c == 0:
            ax.set_ylabel("Mesoscope", fontsize=6)
        style_axes(ax)
        summary.append(dict(row="meso_vs_bwm_d640", measure=f"spearman_n20_{t}", value=r, p=p, session_bins=len(both),
                            p_naive=st["p_naive"]))
    axs[0, 0].legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(0, -0.35), ncol=3)
    for c in range(5):
        axs[0, c].text(-0.3, 1.15, "abcde"[c], transform=axs[0, c].transAxes, fontsize=8, fontweight="bold")
        axs[1, c].text(-0.3, 1.15, "fghij"[c], transform=axs[1, c].transAxes, fontsize=8, fontweight="bold")
    f = lambda t, ver: ", ".join(f"{ROW_TEXT[m]} {stats[m, t, ver]:.3f}" for m in ROWS)  # noqa: E731
    caps = [f"{TITLE[t].replace(chr(10), ' ')}, {N_PICK} neurons: {f(t, 'n20')}; all neurons: {f(t, 'all')}."
            for t in TARGETS]
    caps += [f"{TITLE[t].replace(chr(10), ' ')}: $r_s$ = {stats[t, 'r'][0]:.2f}, naive $p$ = {stats[t, 'r'][3]:.2f}, "
             f"$p_{{sp}}$ = {stats[t, 'r'][1]:.2f}." for t in TARGETS]
    head = ("Summary of the revised decoding. \\textbf{a--e} Mean over sessions ($\\pm$ s.e.m.) of each "
            "session's mean score over its 1\\,mm bins, from " + str(N_PICK) + " random neurons (left, solid) and all neurons "
            "(right, light). \\textbf{f--j} Per 1\\,mm bin, " + str(N_PICK) + " neurons: BWM $\\leq$640\\,$\\mu$m (x) vs "
            f"mesoscope (y), the {stats[TARGETS[0], 'r'][2]} bins covered by both; Spearman $r_s$, naive $p$, and "
            "$p_{sp}$ against 1000 variogram-matched surrogates of either map (larger $p$ reported). ")
    save_with_caption(fig, OUT_V2 / "v2_summary", caps, "v2_summary",
                      intro=head + decoding_methods("n20").split(f" {N_PICK} randomly")[0])
    plt.close(fig)


def main():
    tun, dec = load()
    summary = []
    counts(tun, summary)
    for ver in ("n20", "all"):
        decoding(dec, ver, summary)
    tuning(tun, summary)
    summary_figure(dec, summary)
    s = pd.DataFrame(summary)
    s.to_csv(OUT_V2 / "summary_v2.csv", index=False)
    print(s.to_string())


if __name__ == "__main__":
    main()
