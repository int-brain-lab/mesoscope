#!/usr/bin/env python3
"""Supplementary controls for the dorsal maps (questions raised on the first
version of the report).

1. Are the mesoscope 'fraction tuned' maps shaped by session-to-session
   differences rather than by regions?
   - si_tuning_sessions: fraction tuned per bin in single sessions
   - si_tuning_variance: variance of the per-(session, bin) fraction
     explained by session vs bin; similarity between the variable maps
     before and after removing session effects
   - si_tuning_corrected: session-corrected fraction maps (bin effects of an
     additive session + bin model)
   - si_tuning_strength: median and 90th percentile of the tuning strength
     (shuffle-corrected U statistic) per bin
2. Larger spatial bins (1 mm across): si_hex1mm_counts / _tuning / _decoding.
3. Window length and sampling (needs extract_bwm.py w03 and the
   corresponding tuning / decode / layers runs): si_windows.
4. Recording depth: si_layers.

Writes figures to output/ (via maps.map_grid / plotstyle.save_with_caption)
and output/controls_summary.csv.
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

from common import ALPHA, MIN_NEURONS, OUT, VARIABLES, bin_neurons
from maps import (MODS, ROW_NAME, TARGET_LABEL, XLIM, joint_norm, map_grid)
from plotstyle import NATURE_DOUBLE_COL_IN, save_with_caption, style_axes

MIN_CELL = 20      # neurons per (session, bin) for a per-session fraction
HEX_LARGE = 0.5    # centre-to-vertex, mm (1 mm across)
VAR_TITLE = {"stim": "Stimulus side", "choice": "Choice", "feedback": "Feedback"}
COL = {"meso": plt.get_cmap("magma")(0.6), "bwm": "0.25", "bwm_w03": "0.5",
       "bwm_w03_fr": "0.72", "superficial": plt.get_cmap("viridis")(0.75),
       "L4": plt.get_cmap("viridis")(0.45), "deep": plt.get_cmap("viridis")(0.15)}
SUMMARY = []


def load_tuning(m, features=""):
    t = pd.read_parquet(OUT / f"tuning_{m}{features}.pqt")
    t["hex"] = bin_neurons(t[["x", "y", "z"]].values)
    return t


def cell_fractions(t, v, min_cell=MIN_CELL):
    """Fraction tuned per (session, bin) with >= min_cell testable neurons."""
    ok = t[f"p_{v}"].notna()
    g = (t.loc[ok, f"p_{v}"] < ALPHA).groupby([t.loc[ok, "session"], t.loc[ok, "hex"]])
    f = pd.DataFrame({"f": g.mean(), "n": g.size()})
    return f[f.n >= min_cell].reset_index()


def additive_fit(f, n_iter=200):
    """f = mu + a[session] + b[bin], least squares by backfitting.
    Returns (a, b, mu, fitted)."""
    mu = f.f.mean()
    a = pd.Series(0.0, index=f.session.unique())
    b = pd.Series(0.0, index=f.hex.unique())
    for _ in range(n_iter):
        a = (f.f - mu - b.loc[f.hex].values).groupby(f.session.values).mean()
        b = (f.f - mu - a.loc[f.session].values).groupby(f.hex.values).mean()
        b -= b.mean()
    a = (f.f - mu - b.loc[f.hex].values).groupby(f.session.values).mean()
    return a, b, mu, mu + a.loc[f.session].values + b.loc[f.hex].values


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


# ---------------------------------------------------------------------------
# 1. tuning: sessions vs bins
# ---------------------------------------------------------------------------
def tuning_sessions(t, n_sessions=4):
    """Single-session fraction maps for the sessions covering the most bins."""
    fs = {v: cell_fractions(t, v) for v in VARIABLES}
    cover = fs["stim"].groupby("session").size().sort_values(ascending=False)
    pick = list(cover.index[:n_sessions])
    try:
        from one.api import ONE
        one = ONE()
        names = {s: "{subject} {date}".format(**one.eid2ref(s)) for s in pick}
    except Exception:
        names = {s: s[:8] for s in pick}
    per = {s: [(fs[v][fs[v].session == s].hex.values, fs[v][fs[v].session == s].f.values)
               for v in VARIABLES] for s in pick}
    norms = [joint_norm([per[s][i][1] for s in pick], lo=0) for i in range(3)]
    map_grid(per, [VAR_TITLE[v] for v in VARIABLES], ["fraction significant"] * 3, norms,
             "si_tuning_sessions",
             ["Fraction of neurons significantly tuned to stimulus side in each bin, one session per "
              "row.", "Same for choice.", "Same for feedback."],
             "si_tuning_sessions",
             intro=(f"Single-session tuning maps, mesoscope. Rows: the {n_sessions} sessions covering "
                    f"the most bins (each bin with $\\geq${MIN_CELL} testable neurons in that "
                    f"session); hemispheres folded, 0.4\\,mm bins. Colour scales are shared across "
                    f"sessions within a column."),
             rows=pick, row_names=names)
    return pick


def co_tuning(tables):
    """Per-neuron overlap between variables: P(tuned to choice | tuned to stim)
    relative to P(tuned to choice), etc., and the Spearman correlation of the
    tuning strengths across neurons."""
    rows = []
    for name, t in tables.items():
        t = t.dropna(subset=["p_stim", "p_choice", "p_feedback"])
        for a, b in (("stim", "choice"), ("stim", "feedback"), ("choice", "feedback")):
            sa, sb = t[f"p_{a}"] < ALPHA, t[f"p_{b}"] < ALPHA
            rows.append(dict(control="co_tuning", version=name, target=f"{a}-{b}",
                             mean=spearmanr(t[f"u_{a}"], t[f"u_{b}"])[0],
                             enrichment=sb[sa].mean() / sb.mean()))
    SUMMARY.extend(rows)
    return pd.DataFrame(rows)


def tuning_variance(t_meso, t_bwm, t_w03=None):
    """Variance decomposition and cross-variable map similarity."""
    rows = []
    corr = {}
    bin_eff = {}
    for v in VARIABLES:
        f = cell_fractions(t_meso, v)
        a, b, mu, fit = additive_fit(f)
        y = f.f.values
        sess_mean = f.groupby("session").f.transform("mean").values
        bin_mean = f.groupby("hex").f.transform("mean").values
        noise = np.mean(y * (1 - y) / f.n.values)  # binomial sampling variance
        # spread across bins within a session vs across sessions within a bin
        within_sess = f.groupby("session").f.std().mean()
        shared = f.groupby("hex").filter(lambda d: len(d) >= 3)
        within_bin = shared.groupby("hex").f.std().mean()
        rows.append(dict(variable=v, cells=len(f), sessions=f.session.nunique(),
                         bins=f.hex.nunique(), r2_session=r2(y, sess_mean),
                         r2_bin=r2(y, bin_mean), r2_additive=r2(y, fit),
                         noise_share=noise / y.var(), sd_bins_within_session=within_sess,
                         sd_sessions_within_bin=within_bin))
        bin_eff[v] = b + mu
    vt = pd.DataFrame(rows)
    # similarity of the variable maps across bins
    pairs = [("stim", "choice"), ("stim", "feedback"), ("choice", "feedback")]

    def pooled(t, v):
        ok = t[f"p_{v}"].notna()
        g = (t.loc[ok, f"p_{v}"] < ALPHA).groupby(t.loc[ok, "hex"])
        return g.mean()[g.size() >= MIN_NEURONS]

    def strength(t, v):
        g = t.groupby("hex")[f"u_{v}"]
        return g.median()[g.size() >= MIN_NEURONS]

    versions = [("meso pooled", {v: pooled(t_meso, v) for v in VARIABLES}),
                ("meso session-corrected", bin_eff),
                ("meso strength (median)", {v: strength(t_meso, v) for v in VARIABLES}),
                ("BWM pooled", {v: pooled(t_bwm, v) for v in VARIABLES})]
    if t_w03 is not None:
        versions.append(("BWM 0.3 s pooled", {v: pooled(t_w03, v) for v in VARIABLES}))
    for name, maps_ in versions:
        corr[name] = []
        for x, y in pairs:
            j = maps_[x].to_frame("x").join(maps_[y].to_frame("y"), how="inner").dropna()
            corr[name].append(spearmanr(j.x, j.y)[0])
    corr = pd.DataFrame(corr, index=[f"{x}--{y}" for x, y in pairs])

    w = NATURE_DOUBLE_COL_IN
    fig, axs = plt.subplots(1, 3, figsize=(w, 2.1))
    xs = np.arange(len(VARIABLES))
    for k, (col, lab) in enumerate((("r2_session", "session"), ("r2_bin", "bin"),
                                    ("r2_additive", "session + bin"))):
        axs[0].bar(xs + (k - 1) * 0.27, vt[col], 0.27, label=lab,
                   color=plt.get_cmap("magma")(0.2 + 0.3 * k))
    axs[0].plot(xs, vt.noise_share, "k_", ms=10, mew=1, label="binomial noise")
    axs[0].set_xticks(xs, [VAR_TITLE[v] for v in VARIABLES], fontsize=6)
    axs[0].set_ylabel("variance explained")
    axs[0].legend(fontsize=5, frameon=False, loc="upper left")
    axs[0].set_ylim(0, 1)
    k = 0
    for col, lab in (("sd_bins_within_session", "across bins,\nsame session"),
                     ("sd_sessions_within_bin", "across sessions,\nsame bin")):
        axs[1].bar(xs + (k - 0.5) * 0.36, vt[col], 0.36, label=lab,
                   color=plt.get_cmap("magma")(0.3 + 0.4 * k))
        k += 1
    axs[1].set_xticks(xs, [VAR_TITLE[v] for v in VARIABLES], fontsize=6)
    axs[1].set_ylabel("SD of fraction tuned")
    axs[1].legend(fontsize=5, frameon=False, loc="upper left")
    im = axs[2].imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axs[2].set_xticks(range(corr.shape[1]), ["meso", "meso\ncorr.", "meso\nstrength",
                                             "BWM", "BWM\n0.3 s"][:corr.shape[1]],
                      fontsize=5)
    axs[2].set_yticks(range(3), [p.replace("--", " vs ") for p in corr.index], fontsize=6)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            axs[2].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="w" if abs(corr.values[i, j]) > 0.6 else "k")
    fig.colorbar(im, ax=axs[2], fraction=0.05, pad=0.03).ax.tick_params(labelsize=5)
    for ax in axs[:2]:
        style_axes(ax, categorical_x=True)
    for c, ax in enumerate(axs):
        ax.text(-0.25, 1.05, "abc"[c], transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.tight_layout()
    save_with_caption(
        fig, OUT / "si_tuning_variance",
        [f"Variance of the fraction tuned per (session, bin) explained by session means, by bin "
         f"means and by an additive session + bin model; black ticks: share expected from binomial "
         f"sampling noise alone ({MIN_CELL} or more neurons per cell).",
         "Standard deviation of the fraction tuned across bins of the same session (averaged over "
         "sessions) and across sessions recording the same bin (bins with $\\geq$3 sessions).",
         "Spearman correlation across bins between the maps of two variables: pooled fraction "
         "tuned (as in the main figure), the session-corrected bin effects of the additive model, "
         "the median tuning strength, and the BWM pooled fractions with the BWM windows and with "
         "the 0.3\\,s mesoscope windows."],
        label="si_tuning_variance",
        intro=(f"Mesoscope tuning: session vs region effects. Each cell is one session $\\times$ one "
               f"bin (0.4\\,mm, hemispheres folded) with $\\geq${MIN_CELL} testable neurons."))
    plt.close(fig)
    vt.to_csv(OUT / "si_tuning_variance.csv", index=False)
    corr.to_csv(OUT / "si_tuning_map_correlations.csv")
    for _, r in vt.iterrows():
        SUMMARY.append(dict(control="tuning_variance", **r.to_dict()))
    return vt, corr, bin_eff


def tuning_corrected(bin_eff):
    per = {"meso": [(b.index.values, b.values) for b in (bin_eff[v] for v in VARIABLES)]}
    norms = [joint_norm([per["meso"][i][1]], lo=None) for i in range(3)]
    map_grid(per, [VAR_TITLE[v] for v in VARIABLES], ["fraction significant (corrected)"] * 3, norms,
             "si_tuning_corrected",
             ["Session-corrected fraction tuned to stimulus side.", "Same for choice.",
              "Same for feedback."],
             "si_tuning_corrected",
             intro=(f"Mesoscope tuning maps with session effects removed: bin effects of the "
                    f"additive model fraction = mean + session + bin, fitted to all (session, bin) "
                    f"cells with $\\geq${MIN_CELL} testable neurons, plus the grand mean. "
                    f"Hemispheres folded, 0.4\\,mm bins."),
             rows=["meso"], row_names={"meso": "Mesoscope"})


def tuning_strength(tun):
    per = {}
    for stat, q in (("median", 0.5), ("90th percentile", 0.9)):
        for m in MODS:
            cols = []
            for v in VARIABLES:
                g = tun[m].groupby("hex")[f"u_{v}"]
                val = g.quantile(q)[g.count() >= MIN_NEURONS]
                cols.append((val.index.values, val.values))
            per[f"{m} {stat}"] = cols
    rows = [f"{m} {s}" for s in ("median", "90th percentile") for m in MODS]
    norms = [joint_norm([per[r][i][1] for r in rows], lo=0) for i in range(3)]
    map_grid(per, [VAR_TITLE[v] for v in VARIABLES], ["tuning strength"] * 3, norms,
             "si_tuning_strength",
             ["Tuning strength for stimulus side.", "Same for choice.", "Same for feedback."],
             "si_tuning_strength",
             intro=("Tuning strength instead of the fraction of significant neurons. For each "
                    "neuron, the strength is the condition-combined separation of the two "
                    "conditions (|AUC $-$ 0.5|, pooled over strata, from the same Mann--Whitney "
                    "statistic as the test) minus its mean over the block-wise shuffles, so 0 is "
                    "chance and the value does not grow with fewer trials. Rows: median and 90th "
                    "percentile per bin, mesoscope and BWM; bins with $\\geq$10 neurons pooled "
                    "across sessions; hemispheres folded, 0.4\\,mm bins. Colour scales shared "
                    "within a column."),
             rows=rows, row_names={r: r.replace("meso", "Meso").replace("bwm", "BWM")
                                   .replace(" 90th percentile", "\n90th pct") for r in rows})


# ---------------------------------------------------------------------------
# 2. larger bins
# ---------------------------------------------------------------------------
def large_bins(tun):
    size = HEX_LARGE
    across = 2 * size
    keys = {m: bin_neurons(tun[m][["x", "y", "z"]].values, size=size) for m in MODS}
    per = {}
    for m in MODS:
        n = tun[m].groupby(keys[m]).size()
        ns = tun[m].groupby([keys[m], tun[m].session.values]).size()
        ses = (ns >= MIN_NEURONS).groupby(level=0).sum()
        ses = ses[ses > 0]
        per[m] = [(n.index, n.values), (ses.index, ses.values)]
    norms = [joint_norm([per[m][0][1] for m in MODS], log=True),
             joint_norm([per[m][1][1] for m in MODS], log=True)]
    intro = (f"Same as the main figure with larger bins ({across:g}\\,mm vertex to vertex), "
             f"hemispheres folded; mesoscope top row, BWM bottom row.")
    map_grid(per, ["Neurons", f"Sessions ($\\geq${MIN_NEURONS} neurons)"],
             ["neurons per bin", "sessions per bin"], norms, "si_hex1mm_counts",
             ["Neurons per bin.", f"Sessions contributing $\\geq${MIN_NEURONS} neurons."],
             "si_hex1mm_counts", intro=intro, size=size)
    per = {}
    for m in MODS:
        cols = []
        for v in VARIABLES:
            ok = tun[m][f"p_{v}"].notna().values
            g = (tun[m][f"p_{v}"][ok] < ALPHA).groupby(keys[m][ok])
            frac = g.mean()[g.count() >= MIN_NEURONS]
            cols.append((frac.index, frac.values))
        per[m] = cols
    norms = [joint_norm([per[m][i][1] for m in MODS], lo=0) for i in range(3)]
    map_grid(per, [VAR_TITLE[v] for v in VARIABLES], ["fraction significant"] * 3, norms,
             "si_hex1mm_tuning",
             ["Fraction of neurons tuned to stimulus side.", "Same for choice.", "Same for feedback."],
             "si_hex1mm_tuning", intro=intro + f" Bins with $\\geq${MIN_NEURONS} testable neurons.",
             size=size)
    dec = {m: pd.read_parquet(OUT / f"decoding_{m}_hex{size:g}.pqt") for m in MODS}
    targets = ["stim", "choice", "feedback", "wheel"]
    per = {}
    for m in MODS:
        g = dec[m].groupby(["target", "hex"]).n10.mean()
        per[m] = [(g[t].dropna().index, g[t].dropna().values) for t in targets]
    binary = joint_norm([per[m][i][1] for m in MODS for i in range(3)], lo=0.5)
    norms = [binary] * 3 + [joint_norm([per[m][3][1] for m in MODS], lo=0)]
    map_grid(per, [TARGET_LABEL[t].capitalize() for t in targets],
             ["balanced accuracy"] * 3 + ["$R^2$"], norms, "si_hex1mm_decoding",
             ["Stimulus side.", "Choice.", "Feedback.", "Wheel speed ($R^2$)."],
             "si_hex1mm_decoding",
             intro=intro + f" Decoding from {MIN_NEURONS} randomly drawn neurons of a session in "
                           f"the bin (mean of 3 draws), averaged across sessions; otherwise as in "
                           f"the main decoding figure. The binary variables share one colour scale.",
             size=size)
    for m in MODS:
        for t in targets:
            d = dec[m][dec[m].target == t]
            SUMMARY.append(dict(control="hex1mm_decoding_n10", modality=m, target=t,
                                session_bins=len(d), mean=d.n10.mean()))


# ---------------------------------------------------------------------------
# 3. windows and frame sampling
# ---------------------------------------------------------------------------
VERSIONS = [("meso", "", "Mesoscope"), ("bwm", "", "BWM, BWM windows"),
            ("bwm_w03", "", "BWM, 0.3 s windows"), ("bwm_w03", "_fr", "BWM, 0.3 s, frame-sampled")]


def windows_control():
    """Decoding (10 neurons, 0.4 mm bins) and fraction tuned for the BWM
    window / sampling variants next to the mesoscope."""
    from scipy.stats import sem
    targets = ["stim", "choice", "feedback", "wheel"]
    dec, tun = {}, {}
    for m, f, _ in VERSIONS:
        dec[m + f] = pd.read_parquet(OUT / f"decoding_{m}{f}.pqt")
        tun[m + f] = pd.read_parquet(OUT / f"tuning_{m}{f}.pqt")
    w = NATURE_DOUBLE_COL_IN
    fig, axs = plt.subplots(1, 3, figsize=(w, 2.2), gridspec_kw=dict(width_ratios=[3, 1.2, 3]))
    nv = len(VERSIONS)
    width = 0.8 / nv
    for k, (m, f, lab) in enumerate(VERSIONS):
        key = m + f
        d = dec[key]
        mean = [d[d.target == t].n10.mean() for t in targets[:3]]
        err = [sem(d[d.target == t].n10, nan_policy="omit") for t in targets[:3]]
        axs[0].bar(np.arange(3) + (k - (nv - 1) / 2) * width, mean, width, yerr=err,
                   color=COL[key], label=lab, error_kw=dict(lw=0.5))
        wv = d[d.target == "wheel"].n10
        axs[1].bar(k, wv.mean(), 0.7, yerr=sem(wv, nan_policy="omit"), color=COL[key],
                   error_kw=dict(lw=0.5))
        fr = [np.mean(tun[key][f"p_{v}"].dropna() < ALPHA) for v in VARIABLES]
        axs[2].bar(np.arange(3) + (k - (nv - 1) / 2) * width, fr, width, color=COL[key])
        for t, mm in zip(targets, mean + [wv.mean()]):
            SUMMARY.append(dict(control="windows_decoding_n10", version=lab, target=t, mean=mm))
        for v, x in zip(VARIABLES, fr):
            SUMMARY.append(dict(control="windows_frac_tuned", version=lab, target=v, mean=x))
    axs[0].set_ylim(0.5, None)
    axs[0].set_xticks(range(3), [VAR_TITLE[v] for v in VARIABLES], fontsize=6)
    axs[0].set_ylabel("balanced accuracy")
    axs[0].legend(fontsize=5, frameon=False, loc="upper left")
    axs[1].set_xticks(range(nv), ["Meso", "BWM", "0.3 s", "frames"], fontsize=5, rotation=30)
    axs[1].set_ylabel("wheel speed $R^2$")
    axs[2].set_xticks(range(3), [VAR_TITLE[v] for v in VARIABLES], fontsize=6)
    axs[2].set_ylabel("fraction tuned")
    for c, ax in enumerate(axs):
        style_axes(ax, categorical_x=True)
        ax.text(-0.22 if c != 1 else -0.5, 1.04, "abc"[c], transform=ax.transAxes, fontsize=8,
                fontweight="bold")
    fig.tight_layout()
    save_with_caption(
        fig, OUT / "si_windows",
        [f"Decoding from {MIN_NEURONS} neurons per session and 0.4\\,mm bin (mean $\\pm$ s.e.m. over "
         f"session $\\times$ bin pairs), binary variables.",
         "Same for wheel speed ($R^2$). The wheel bins are 0.2\\,s in all versions; the frame-"
         "sampled version reads them from the frame-resolution trace.",
         "Fraction of neurons significantly tuned ($p<0.05$), same test as in the main figure."],
        label="si_windows",
        intro=("Do the analysis windows or the mesoscope's slow sampling explain the BWM vs "
               "mesoscope differences? The same BWM units are re-analysed (i) with the mesoscope "
               "windows ($[0, 0.3]$\\,s after stimulus onset, $[-0.3, 0]$\\,s before movement onset, "
               "$[0, 0.3]$\\,s after feedback) and (ii) additionally frame-sampled: spikes are binned "
               "into 0.205\\,s frames (the mesoscope frame period) and the resulting rate trace is "
               "read out exactly like a mesoscope trace (linear interpolation between frame "
               "centres, mean over the window)."))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. layers
# ---------------------------------------------------------------------------
def layers_control(features_list=(("bwm", ""), ("bwm_w03", "_fr"))):
    targets = ["stim", "choice", "feedback", "wheel"]
    meso = pd.read_parquet(OUT / "layers_meso.pqt")
    w = NATURE_DOUBLE_COL_IN
    fig, axs = plt.subplots(len(features_list), 4, figsize=(w, 1.9 * len(features_list)),
                            squeeze=False)
    for r, (m, f) in enumerate(features_list):
        d = pd.read_parquet(OUT / f"layers_{m}{f}.pqt")
        for c, t in enumerate(targets):
            ax = axs[r, c]
            p = d[d.target == t].pivot_table(index="session", columns="group", values="score")
            paired = p[["superficial", "deep"]].dropna()
            for k, g in enumerate(("superficial", "deep")):
                ax.scatter(np.full(len(paired), k) + np.random.default_rng(k).uniform(-0.12, 0.12,
                           len(paired)), paired[g], s=3, color=COL[g], lw=0, alpha=0.7)
                ax.plot([k - 0.25, k + 0.25], [paired[g].mean()] * 2, color="k", lw=1)
            mv = meso[meso.target == t].score.mean()
            ax.axhline(mv, color=COL["meso"], lw=1, ls="--")
            ax.axhline(0.5 if t != "wheel" else 0, color="0.8", lw=0.5, zorder=0)
            stat = wilcoxon(paired.superficial, paired.deep) if len(paired) > 5 else None
            ptxt = f"p = {stat.pvalue:.2g}" if stat else ""
            ax.set_title(f"{TARGET_LABEL[t].capitalize()}\nn = {len(paired)} sessions, {ptxt}",
                         fontsize=6)
            ax.set_xticks([0, 1], ["L1--2/3", "L5--6"], fontsize=6)
            ax.set_xlim(-0.6, 1.6)
            if c == 0:
                ax.set_ylabel(("BWM windows" if f == "" else "0.3 s, frame-sampled")
                              + "\nbal. acc. / $R^2$", fontsize=6)
            style_axes(ax, categorical_x=True)
            SUMMARY.append(dict(control="layers", version=m + f, target=t, sessions=len(paired),
                                superficial=paired.superficial.mean(), deep=paired.deep.mean(),
                                meso=mv, p=stat.pvalue if stat else np.nan))
        axs[r, 0].text(-0.5, 1.15, "ab"[r], transform=axs[r, 0].transAxes, fontsize=8,
                       fontweight="bold")
    fig.tight_layout()
    save_with_caption(
        fig, OUT / "si_layers",
        ["BWM units with the BWM windows: decoding from 10 superficial (L1--2/3) vs 10 deep "
         "(L5--6) units of the same session (dots: sessions with $\\geq$10 units in both groups; "
         "bars: means; Wilcoxon signed-rank test). Dashed line: mesoscope, 10 neurons per session.",
         "Same with the 0.3\\,s windows and frame sampling, the BWM version closest to how the "
         "mesoscope measures activity."],
        label="si_layers",
        intro=("Recording depth. Mesoscope neurons lie in L1--2/3 (by region label), while most BWM "
               "isocortex units lie in L5--6. Here decoders use 10 randomly drawn neurons of one "
               "layer group per session, ignoring spatial bins (mean of 10 draws), so both groups "
               "come from the same sessions."))
    plt.close(fig)


def main(parts):
    tun = {m: load_tuning(m) for m in MODS}
    have_w03 = (OUT / "tuning_bwm_w03.pqt").exists()
    if "tuning" in parts:
        tuning_sessions(tun["meso"])
        t_w03 = load_tuning("bwm_w03") if have_w03 else None
        _, _, bin_eff = tuning_variance(tun["meso"], tun["bwm"], t_w03)
        tables = {"Mesoscope": tun["meso"], "BWM, BWM windows": tun["bwm"]}
        if have_w03:
            tables["BWM, 0.3 s windows"] = t_w03
            tables["BWM, 0.3 s, frame-sampled"] = load_tuning("bwm_w03", "_fr")
        co_tuning(tables)
        tuning_corrected(bin_eff)
        tuning_strength(tun)
    if "bins" in parts:
        large_bins(tun)
    if "windows" in parts:
        windows_control()
    if "layers" in parts:
        layers_control()
    path = OUT / "controls_summary.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    new = pd.DataFrame(SUMMARY)
    if len(old) and "control" in old:
        old = old[~old.control.isin(new.control.unique())] if len(new) else old
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
    print(new.to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["tuning", "bins", "windows", "layers"])
