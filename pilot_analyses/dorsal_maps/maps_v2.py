#!/usr/bin/env python3
"""Revised dorsal maps (v2): 1 mm hexagons, depth-limited BWM, revised wheel decoding.

Rows: mesoscope; BWM units within 640 um of the pial surface (bwm_pial640, the
main comparison); BWM units at all depths (reference).
Inputs: output/tuning_meso.pqt, output/tuning_bwm.pqt (one row per neuron;
the bwm_pial640 rows are the BWM neurons within 640 um, same test), and
output/v2/decoding_<modality>_auc.pqt (decode_v2.py). The report decoding
figures also use the 0.5 mm mesoscope bins and the BWM per-probe tables.

Figures in output/v2/ (each _panels.pdf, .tex, .pdf with caption):
- v2_counts: neurons and sessions (>= N_PICK neurons) per bin
- v2_decoding_n20 / v2_decoding_all: stimulus side, choice, feedback, wheel speed,
  trial-specific wheel speed
- v2_tuning: fraction of tuned neurons
- v2_summary: mean score per target and row, and per-bin mesoscope vs BWM (<=640 um)
Also output/v2/summary_v2.csv.

Usage: python maps_v2.py
"""
import colorcet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from common import ALPHA, MIN_NEURONS, OUT, VARIABLES, WHEEL_EDGES, bin_neurons
from decode_probe import probe_output
from decode_top10 import top10_output
from decode_v2 import HEX_V2, MIN_TRIALS, N_DRAWS, N_PICK, OUT_V2, REPORT_HEX_SIZE, out_name
from make_bwm_depth import MAX_DEPTH_UM, STREAMLINE_MODALITY, streamline_depth_um
from maps import _tuning_methods, _win, joint_norm, map_grid
from plotstyle import NATURE_DOUBLE_COL_IN, save_with_caption, style_axes
from sessions import session_files

# Keep score hue separate from the lightness change caused by support opacity.
DORSAL_CMAP = colorcet.cm["CET_I3"]
REPORT_CMAP = "cool"
BWM_MARKER_ALPHA = 0.8
ROWS = ("meso", STREAMLINE_MODALITY, "bwm")
ROW_NAME = {"meso": "Mesoscope", STREAMLINE_MODALITY: f"BWM $\\leq${MAX_DEPTH_UM} µm",
            "bwm": "BWM all depths"}
ROW_TEXT = {"meso": "mesoscope", STREAMLINE_MODALITY: f"BWM $\\leq${MAX_DEPTH_UM}\\,$\\mu$m", "bwm": "BWM all depths"}
COL = {"meso": "tab:green", STREAMLINE_MODALITY: "tab:purple", "bwm": "0.6"}
TARGETS = ("stim", "choice", "feedback", "wheel", "wheel_trial")
TITLE = {"stim": "Stimulus side", "choice": "Choice", "feedback": "Feedback", "wheel": "Wheel speed",
         "wheel_trial": "Wheel speed,\ntrial-specific"}
CHANCE = {"stim": 0.5, "choice": 0.5, "feedback": 0.5, "wheel": 0.0, "wheel_trial": 0.0}


def load_tuning():
    from decode_v2 import excluded_sessions

    tun = {m: pd.read_parquet(OUT / f"tuning_{m}.pqt") for m in ("meso", "bwm")}
    tun["meso"] = tun["meso"][~tun["meso"].session.isin(excluded_sessions())]
    d = streamline_depth_um(tun["bwm"][["x", "y", "z"]].values)
    tun[STREAMLINE_MODALITY] = tun["bwm"][d <= MAX_DEPTH_UM].copy()
    for m in ROWS:
        tun[m]["hex"] = bin_neurons(tun[m][["x", "y", "z"]].values, size=HEX_V2)
    return tun


def load_binned_scores():
    from decode_v2 import excluded_sessions

    dec = {m: pd.read_parquet(out_name(m)) for m in ROWS}
    dec["meso"] = dec["meso"][~dec["meso"].session.isin(excluded_sessions())]
    return dec


def coverage_data():
    """One row per unit from cached coordinates, independent of tuning scores."""
    from decode_v2 import excluded_sessions

    out = {}
    excluded = excluded_sessions()
    for m in ROWS:
        rows = []
        for sid, files in session_files(m).items():
            if m == "meso" and sid in excluded:
                continue
            for path in files:
                with np.load(path) as z:
                    keys = bin_neurons(z["xyz"], size=HEX_V2)
                rows.append(pd.DataFrame(dict(hex=keys, session=sid)))
        out[m] = pd.concat(rows, ignore_index=True)
    return out


def intro(title="Dorsal cortex maps"):
    return (f"{title}: mesoscope two-photon imaging (top row; deconvolved activity of "
            "isocortex neurons) vs Brain Wide Map Neuropixels units within "
            f"{MAX_DEPTH_UM}\\,$\\mu$m of the pia by Allen CCF streamline depth (middle row) "
            "and at all depths (bottom row). Each neuron is "
            f"assigned to a hexagonal bin ({HEX_V2 * 2:.0f}\\,mm vertex to vertex) by its ML and AP Allen "
            "CCF position; hemispheres folded onto the right. Grey lines: Allen top-view region "
            "boundaries. Colour scales are shared across rows.")


def decoding_methods(ver):
    return ("Decoding per session and bin (sessions with $\\geq$50 trials and $\\geq$"
            f"{N_PICK} neurons in the bin), 5-fold cross-validation over trials. Binary variables: "
            "L2 logistic regression, ROC-AUC (chance 0.5). Wheel speed in six 0.2\\,s bins from "
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
             "v2_counts", intro=intro("Coverage"), rows=ROWS, row_names=ROW_NAME, size=HEX_V2,
             out_dir=OUT_V2, cmap=DORSAL_CMAP)


def score_norm(arrays, chance):
    """Share a chance-anchored colour scale across modalities for one target."""
    values = np.concatenate(arrays)
    finite = values[np.isfinite(values)]
    hi = float(np.percentile(finite, 99)) if finite.size else chance
    return Normalize(vmin=chance, vmax=hi if hi > chance else chance + 0.01, clip=True)


def report_decoding(cmap=REPORT_CMAP, *, top10=False):
    """Render the requested v7 maps from fine-bin and per-probe scores."""
    from decode_v2 import excluded_sessions

    cmap = plt.get_cmap(cmap)
    meso_path = top10_output("meso") if top10 else out_name("meso", REPORT_HEX_SIZE)
    probe_path = top10_output if top10 else probe_output
    meso = pd.read_parquet(meso_path)
    meso = meso[~meso.session.isin(excluded_sessions())]
    probes = {STREAMLINE_MODALITY: pd.read_parquet(probe_path(STREAMLINE_MODALITY)),
              "bwm": pd.read_parquet(probe_path("bwm"))}
    for ver in (("top10",) if top10 else ("n20", "all")):
        decoding(meso, probes, ver, cmap=cmap)


def decoding(meso, probes, ver, cmap=REPORT_CMAP):
    from controls2 import SUBJ, opacity

    d = meso.dropna(subset=[ver]).assign(subject=lambda x: x.session.map(SUBJ))
    n_sessions = d.session.nunique()
    g = d.groupby(["target", "hex"]).agg(v=(ver, "mean"), na=("subject", "nunique"))
    per = {"meso": [(a.index, a.v.values, opacity(a.na.values))
                    for a in (g.xs(t) for t in TARGETS)]}
    n_probes = {}
    for m in ROWS[1:]:
        d = probes[m].dropna(subset=[ver, "entry_x", "entry_y"]).sort_values(ver)
        n_probes[m] = d.pid.nunique()
        per[m] = [dict(x=np.abs(p.entry_x.values), y=p.entry_y.values, v=p[ver].values,
                       alpha=BWM_MARKER_ALPHA)
                  for p in (d[d.target == t] for t in TARGETS)]

    values = [[per["meso"][i][1]] + [per[m][i]["v"] for m in ROWS[1:]]
              for i in range(len(TARGETS))]
    norms = [score_norm(v, CHANCE[t]) for t, v in zip(TARGETS, values)]
    ticks = [np.linspace(norm.vmin, norm.vmax, 3) for norm in norms]
    if ver == "n20":
        title = f"Decoding from {N_PICK} neurons"
        population = f"{N_PICK} randomly drawn neurons (mean of {N_DRAWS} draws)"
    elif ver == "top10":
        title = f"Decoding from the best {MIN_NEURONS} neurons"
        population = f"the {MIN_NEURONS} highest-ranked neurons"
    else:
        title = "Decoding from all neurons"
        population = "all eligible neurons"

    head = (f"\\textbf{{{title}}}. Top row: mesoscope decoding from {population} in each "
            f"session and {2 * REPORT_HEX_SIZE:g}\\,mm hexagon (vertex to vertex), averaged over "
            "sessions. Opacity represents the number of animals (full at $\\geq$4). Middle and "
            f"bottom rows: BWM decoders fit to {population} on each probe, using units within "
            f"{MAX_DEPTH_UM}\\,$\\mu$m of the pia by Allen CCF streamline depth and at all depths, "
            "respectively. Each "
            "semi-transparent dot is at the probe's Alyx trajectory entry point. Hemispheres are folded onto the "
            f"right. Sessions require at least {MIN_TRIALS} trials and bins or probes need at least {N_PICK} "
            f"neurons. Decoding scores are available for {n_sessions} mesoscope sessions, "
            f"{n_probes[STREAMLINE_MODALITY]} superficial BWM insertions and {n_probes['bwm']} "
            "all-depth BWM insertions. Each variable has a colour scale shared across rows, "
            "from chance to the 99th percentile of plotted scores. Below-chance scores "
            "use the lowest colour.")
    if ver == "top10":
        head += (r" Ranking was repeated on the training trials of each outer fold. "
                 r"Binary variables use $|\mathrm{AUC}-0.5|$. Both wheel targets use "
                 r"the highest signed Pearson $r$ with raw wheel speed across all wheel bins. "
                 r"Wheel scores require ten defined correlations per training fold.")
    map_grid(per, [TITLE[t] for t in TARGETS], ["ROC-AUC"] * 3 + ["$R^2$"] * 2, norms,
             f"v2_decoding_{ver}",
             ["Stimulus side, ROC-AUC (chance 0.5).",
              "Choice.", "Feedback.", "Wheel speed, cross-validated $R^2$.",
              "Trial-specific wheel speed, $R^2$."],
             f"v2_decoding_{ver}", intro=head, rows=ROWS, row_names=ROW_NAME, size=REPORT_HEX_SIZE,
             out_dir=OUT_V2, cmap=cmap, cbar_ticks=ticks)


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
             out_dir=OUT_V2, cmap=DORSAL_CMAP)


def summary_figure(dec, summary, meso_all_estimates=None):
    from controls2 import spatial_test

    fig, axs = plt.subplots(2, 5, figsize=(NATURE_DOUBLE_COL_IN, 3.6), gridspec_kw=dict(hspace=0.75, wspace=0.55))
    stats = {}
    for c, t in enumerate(TARGETS):
        ax = axs[0, c]
        for k, m in enumerate(ROWS):
            for j, ver in enumerate(("n20", "all")):
                d = dec[m][dec[m].target == t]
                ses = d.groupby("session")[ver].mean()
                mean, sem = ses.mean(), ses.sem()
                if meso_all_estimates is not None and m == "meso" and ver == "all" and t in meso_all_estimates:
                    mean, sem = meso_all_estimates[t]
                ax.bar(j + (k - 1) * 0.27, mean, 0.25, yerr=sem, color=COL[m], alpha=0.8 if j == 0 else 0.5,
                       error_kw=dict(lw=0.5, capsize=1), label=ROW_TEXT[m] if (c == 0 and j == 0) else None)
                stats[m, t, ver] = mean
        ax.axhline(CHANCE[t], color="k", lw=0.5)
        ax.set_xticks([0, 1], [f"{N_PICK} neurons", "all"], fontsize=5)
        if CHANCE[t]:
            ax.set_ylim(CHANCE[t] - 0.02, None)
        ax.set_title(TITLE[t], fontsize=6)
        style_axes(ax, categorical_x=True)
        if c == 0:
            ax.set_ylabel("score (mean over sessions)")
        ax = axs[1, c]
        g = {m: dec[m][dec[m].target == t].groupby("hex")["n20"].mean() for m in ("meso", STREAMLINE_MODALITY)}
        both = pd.concat(g, axis=1, join="inner").dropna()
        ax.scatter(both[STREAMLINE_MODALITY], both["meso"], s=6, color="k", lw=0)
        lo, hi = np.nanmin(both.values), np.nanmax(both.values)
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls="--")
        st = spatial_test(f"v2_{t}", both.index.values, both[STREAMLINE_MODALITY].values, both["meso"].values, HEX_V2)
        r, p = st["r"], st["p_spatial"]
        stats[t, "r"] = (r, p, len(both), st["p_naive"])
        ax.text(0.04, 0.97, f"$r_s$={r:.2f}\n$p_{{sp}}$={p:.2f}", transform=ax.transAxes, va="top", fontsize=5)
        ax.set_xlabel("BWM $\\leq$640 µm", fontsize=6)
        if c == 0:
            ax.set_ylabel("Mesoscope", fontsize=6)
        style_axes(ax)
        summary.append(dict(row=f"meso_vs_{STREAMLINE_MODALITY}", measure=f"spearman_n20_{t}", value=r, p=p, session_bins=len(both),
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
            "$p_{sp}$ against 1000 variogram-matched surrogates of either map (larger $p$ reported). "
            "BWM depth uses the Allen CCF streamline lookup. ")
    if meso_all_estimates is not None:
        head += "All-neuron mesoscope bars for targets with archived estimates reuse the original v7 session summaries. "
    save_with_caption(fig, OUT_V2 / "v2_summary", caps, "v2_summary",
                      intro=head + decoding_methods("n20").split(f" {N_PICK} randomly")[0])
    plt.close(fig)


def main(cmap=REPORT_CMAP):
    tun, dec = load_tuning(), load_binned_scores()
    summary = []
    counts(coverage_data(), summary)
    for ver in ("n20", "all"):
        for m in ROWS:
            for t in TARGETS:
                d = dec[m][dec[m].target == t][ver]
                summary.append(dict(row=m, measure=f"decoding_{ver}_{t}", value=d.mean(),
                                    session_bins=d.notna().sum()))
    report_decoding(cmap=cmap)
    tuning(tun, summary)
    summary_figure(dec, summary)
    s = pd.DataFrame(summary)
    s.to_csv(OUT_V2 / "summary_v2.csv", index=False)
    print(s.to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--report-decoding", action="store_true",
                           help="Render the random-20 and all-neuron v7 maps.")
    selection.add_argument("--top10", action="store_true",
                           help="Render the best-10 v7 map.")
    selection.add_argument("--supplement", choices=("counts", "tuning", "summary"),
                           help="Render just the selected supplement figure.")
    parser.add_argument("--cmap", default=REPORT_CMAP,
                        help="Matplotlib colormap for the v7 decoding maps (default: %(default)s).")
    args = parser.parse_args()
    if args.report_decoding or args.top10:
        report_decoding(cmap=args.cmap, top10=args.top10)
    elif args.supplement == "counts":
        counts(coverage_data(), [])
    elif args.supplement == "tuning":
        tuning(load_tuning(), [])
    elif args.supplement == "summary":
        summary_figure(load_binned_scores(), [])
    else:
        main(cmap=args.cmap)
