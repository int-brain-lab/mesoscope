"""Figures + summary numbers.

tracking_qc.pdf        ROICaT tracking QC per mouse (qc.py output)
chronic_stability.pdf  cross-day reliability metrics (stability.py output),
                       every metric against its even/odd within-day baseline
summary.txt
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import ttest_1samp

OUT = Path(__file__).resolve().parent / "output"
BINS = [0, 2, 7, 14, 30, 1000]
BIN_LAB = ["1-2", "3-7", "8-14", "15-30", ">30"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 7,
                     "pdf.fonttype": 42})


def _ticks(a, x=True, y=True):
    if x:
        a.xaxis.set_major_locator(MaxNLocator(3))
    if y:
        a.yaxis.set_major_locator(MaxNLocator(3))


def normalised(p):
    """Every metric as the fraction of its within-day (even/odd) baseline kept
    across days: 1 = as reproducible as within a day, 0 = chance."""
    out = pd.DataFrame(index=p.index)
    out["profile"] = p.stability
    out["activity level"] = p.act_x / p.act_w
    for v in ("stim", "reward"):   # the choice map is too unreliable within a day (r~0.14)
        out[f"{v} map"] = p[f"sel_{v}_x"] / p[f"sel_{v}_w"]
    for v in ("stim", "choice"):
        ch = p[f"{v}_sig"] / 2           # same-sign significant by chance
        out[f"{v} sensitivity"] = (p[f"{v}_kept_x"] - ch) / (p[f"{v}_kept_w"] - ch)
    # decoders, chance from label shuffles; the choice decoder is not normalised:
    # its within-day accuracy is often near chance, so the per-pair ratio diverges
    sh = p["dec_stim_shuf"]
    out["stim decoder"] = (p["dec_stim_x"] - sh) / (p["dec_stim_w"] - sh)
    out = out[["profile", "activity level", "stim map", "reward map", "stim sensitivity",
               "stim decoder", "choice sensitivity"]]
    return out


def qc_figure(qc, sess):
    subjects = sorted(qc.subject.unique())
    t = qc[(qc.uid != "") & qc.qc_pass]
    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, len(subjects))
    for k, s in enumerate(subjects):
        g = t[t.subject == s]
        S = sorted(g.session.unique())
        U = {x: set(g.uid[g.session == x]) for x in S}
        n_all = qc[qc.subject == s].groupby("session").size()
        M = np.array([[len(U[a] & U[b]) / n_all[a] for b in S] for a in S])
        a = fig.add_subplot(gs[0, k])
        im = a.imshow(M, vmin=0, vmax=1, cmap="viridis")
        a.set_title(f"{s} ({len(S)} sessions)", fontsize=7)
        lab = [x[5:10] for x in S]
        step = max(1, len(S) // 4)
        a.set_xticks(range(0, len(S), step)), a.set_xticklabels(lab[::step], rotation=90, fontsize=5)
        a.set_yticks(range(0, len(S), step)), a.set_yticklabels(lab[::step], fontsize=5)
        if k == 0:
            a.set_ylabel("a  fraction of row-session\nneurons tracked in column session")
    fig.colorbar(im, ax=fig.axes[-1], shrink=0.7)
    # b deviation vs nearest-neighbour distance
    a = fig.add_subplot(gs[1, 0:2])
    bins = np.logspace(-1.5, 1.6, 50)
    d = qc[qc.dev_px.notna()]
    a.hist(d.dev_px, bins, histtype="step", color="k", density=True,
           label="tracked ROI: distance to\nits consensus position")
    a.hist(d.nn_px / 2, bins, histtype="step", color="C1", density=True,
           label="half the distance to\nthe nearest neuron ROI")
    a.set_xscale("log")
    a.set(xlabel="pixels (~2 µm)", ylabel="density",
          title=f"b  spatial QC: {100 * d.qc_pass.mean():.1f}% of (neuron, session) pass")
    a.legend(frameon=False, fontsize=6)
    a.yaxis.set_major_locator(MaxNLocator(3))
    # c, d neurons tracked in >= k sessions
    a = fig.add_subplot(gs[1, 2:4])
    a2 = fig.add_subplot(gs[1, 4:6])
    for k, s in enumerate(subjects):
        g = t[t.subject == s]
        ns = g.session.nunique()
        cnt = g.groupby("uid").session.nunique()
        kk = np.arange(1, ns + 1)
        surv = np.array([(cnt >= x).sum() for x in kk])
        a.plot(kk, surv, "-", color=f"C{k}", label=s)
        a2.plot(kk / ns, surv / sess[sess.subject == s].n_neurons.median(), "-", color=f"C{k}", label=s)
    a.set(yscale="log", xlabel="number of sessions k", ylabel="neurons tracked (QC) in ≥ k sessions",
          title="c  neurons tracked across sessions")
    a2.set(xlabel="fraction of the mouse's sessions", ylabel="÷ median neurons per session",
           title="d  same, normalised")
    a.legend(frameon=False, fontsize=6)
    _ticks(a, y=False), _ticks(a2)
    fig.savefig(OUT / "tracking_qc.pdf")
    fig.savefig(OUT / "tracking_qc.png", dpi=150)
    plt.close(fig)


NATURE_DOUBLE_IN = 183 / 25.4     # Nature double-column width
NATURE_RC = {"font.family": "sans-serif",
             "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
             "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6, "xtick.labelsize": 5,
             "ytick.labelsize": 5, "legend.fontsize": 5, "lines.markersize": 2.5,
             "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5}


def panel_letters(fig, axes, letters="abcdefgh"):
    """Bold 8 pt panel letters at the top left of each panel's bounding box
    (axes plus labels and title)."""
    fig.canvas.draw()
    r, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    for a, l in zip(axes, letters):
        bb = a.get_tightbbox(r).transformed(inv)
        fig.text(bb.x0, bb.y1, l, fontsize=8, fontweight="bold", ha="left", va="top")


def main_figure(pairs, neurons):
    with plt.rc_context(NATURE_RC):
        return _main_figure(pairs, neurons)


# one colour per metric/variable, used in every panel; dotted = same-day
# (even vs odd) baseline, thin grey solid = chance
MCOL = {"profile": "k", "act": "C5", "stim": "C2", "choice": "C1", "reward": "C4"}
BASE = dict(ls=":", lw=1.2)
CHANCE = dict(color="0.6", lw=.6, ls="-")
TCOL = {"1-2 days": "C0", ">14 days": "C3"}           # time-interval colours (c, h)
MOUSE_MK = ["o", "s", "^", "v", "D", "P"]
# mouse colours chosen to avoid the metric and time colours above
MOUSE_COL = ["#e377c2", "#bcbd22", "#17becf", "#e7ba52", "#393b79", "#a55194"]


def _main_figure(pairs, neurons):
    from matplotlib.lines import Line2D
    subjects = sorted(pairs.subject.unique())
    shade = dict(zip(subjects, MOUSE_COL))
    mk = dict(zip(subjects, MOUSE_MK))
    p = pairs.assign(dtbin=pd.cut(pairs.dt, BINS, labels=BIN_LAB))
    x = np.arange(len(BIN_LAB))

    def by_bin(series):   # subject means per bin -> (bins, subjects)
        return series.groupby([p.subject, p.dtbin], observed=True).mean().unstack(0).reindex(BIN_LAB)

    def style_legend(a, **kw):   # explains dotted baseline / grey chance
        h = [Line2D([], [], color="k", **BASE), Line2D([], [], **CHANCE)]
        return a.legend(h, ["same day, even vs odd", "chance"], frameon=False, fontsize=5, **kw)

    fig, ax = plt.subplots(2, 4, figsize=(NATURE_DOUBLE_IN, 4.3), constrained_layout=True)
    # a profile similarity
    a = ax[0, 0]
    m = by_bin(p.rx)
    for s in subjects:
        a.plot(x, m[s], "-", color=shade[s], lw=.7)
    a.plot(x, by_bin(p.rw).mean(axis=1), color="k", label="same day, even vs odd", **BASE)
    a.plot(x, m.mean(axis=1), "-o", color="k", lw=1.6, ms=2.5, label="other day, same neuron")
    a.plot(x, by_bin(p.rnn).mean(axis=1), "--", color="k", lw=1, label="other day, nearest neuron")
    a.plot(x, by_bin(p.r0).mean(axis=1), label="other day, random neuron", **CHANCE)
    a.set(xticks=x, xticklabels=BIN_LAB, xlabel="days apart", ylabel="profile correlation",
          ylim=(0, .6), title="response profile")
    a.legend(frameon=False, fontsize=5, loc="upper right")
    a.yaxis.set_major_locator(MaxNLocator(3))
    # b stability index per mouse
    a = ax[0, 1]
    m = by_bin(p.stability)
    for s in subjects:
        a.plot(x, m[s], "-", marker=mk[s], color=shade[s], ms=2.5, lw=.7, label=s)
    a.plot(x, m.mean(axis=1), "k-", lw=1.6)
    a.axhline(1, color="0.6", **BASE), a.axhline(0, **CHANCE)
    a.set(xticks=x, xticklabels=BIN_LAB, xlabel="days apart", ylim=(-.05, 1.1),
          ylabel="stability index", title="profile stability index")
    a.legend(frameon=False, fontsize=5, ncol=3, loc="lower left")
    a.yaxis.set_major_locator(MaxNLocator(3))
    # c per neuron
    a = ax[0, 2]
    edges = np.linspace(0, 1, 11)
    for (lo, hi), lab in [((0, 2), "1-2 days"), ((14, 1000), ">14 days")]:
        q = neurons[(neurons.dt > lo) & (neurons.dt <= hi)]
        med = q.groupby(pd.cut(q.rw, edges), observed=False).rx.median()
        a.plot(0.5 * (edges[1:] + edges[:-1]), med.values, "o-", color=TCOL[lab], ms=2.5, label=lab)
    a.plot([0, 1], [0, 1], color="k", label="same-day ceiling", **BASE)
    a.set(xlabel="even/odd reliability", ylabel="across-day r (median)",
          title="per neuron")
    a.legend(frameon=False, fontsize=5)
    _ticks(a)
    # d activity level and selectivity maps (raw)
    a = ax[0, 3]
    for k, lab in [("act", "activity level"), ("reward", "reward d'"),
                   ("stim", "stimulus d'"), ("choice", "choice d'")]:
        col = "act" if k == "act" else f"sel_{k}"
        a.plot(x, by_bin(p[f"{col}_w"]).mean(axis=1), color=MCOL[k], **BASE)
        a.plot(x, by_bin(p[f"{col}_x"]).mean(axis=1), "-o", color=MCOL[k], ms=2.5, lw=1.2, label=lab)
    a.axhline(0, **CHANCE)
    a.set(xticks=x, xticklabels=BIN_LAB, xlabel="days apart", ylabel="correlation across neurons",
          ylim=(-0.05, 1.05), title="maps across neurons")
    a.add_artist(a.legend(frameon=False, fontsize=5, ncol=2, loc="center", bbox_to_anchor=(0.5, 0.41)))
    style_legend(a, loc="upper right", bbox_to_anchor=(1, 0.9))
    a.yaxis.set_major_locator(MaxNLocator(3))
    # e sensitivity persistence (raw)
    a = ax[1, 0]
    for v in ("stim", "choice"):
        a.plot(x, by_bin(p[f"{v}_kept_w"]).mean(axis=1), color=MCOL[v], **BASE)
        a.plot(x, by_bin(p[f"{v}_kept_x"]).mean(axis=1), "-o", color=MCOL[v], ms=2.5, lw=1.2,
               label=f"{v} ({100 * p[f'{v}_sig'].mean():.1f}% sig.)")
        a.plot(x, by_bin(p[f"{v}_sig"] / 2).mean(axis=1), **CHANCE)
    a.set(xticks=x, xticklabels=BIN_LAB, xlabel="days apart", ylim=(0, .3),
          ylabel="P(sig., same sign | sig.)",
          title="sensitivity persistence")
    a.add_artist(a.legend(frameon=False, fontsize=5, loc="upper right"))
    style_legend(a, loc="lower right", bbox_to_anchor=(1, 0.06))
    a.yaxis.set_major_locator(MaxNLocator(3))
    # f decoders (raw)
    a = ax[1, 1]
    for v in ("stim", "choice"):
        a.plot(x, by_bin(p[f"dec_{v}_w"]).mean(axis=1), color=MCOL[v], **BASE)
        a.plot(x, by_bin(p[f"dec_{v}_x"]).mean(axis=1), "-o", color=MCOL[v], ms=2.5, lw=1.2, label=v)
        a.plot(x, by_bin(p[f"dec_{v}_shuf"]).mean(axis=1), **CHANCE)
    a.set(xticks=x, xticklabels=BIN_LAB, xlabel="days apart", ylim=(0.45, .75),
          ylabel="balanced accuracy", title="decoders")
    a.add_artist(a.legend(frameon=False, fontsize=5, loc="upper right"))
    style_legend(a, loc="lower left")
    a.yaxis.set_major_locator(MaxNLocator(3))
    # g all metrics normalised: colour = variable, marker = metric type
    a = ax[1, 2]
    N = normalised(p)
    var = {"profile": "profile", "activity level": "act", "stim map": "stim", "reward map": "reward",
           "stim sensitivity": "stim", "stim decoder": "stim", "choice sensitivity": "choice",
           "choice decoder": "choice"}
    for k in N.columns:
        mkr = "s" if "sensitivity" in k else ("^" if "decoder" in k else "o")
        a.plot(x, by_bin(N[k]).mean(axis=1), "-", marker=None if k == "profile" else mkr,
               color=MCOL[var[k]], ms=2.5, lw=2 if k == "profile" else 1, label=k)
    a.axhline(1, color="0.6", **BASE), a.axhline(0, **CHANCE)
    a.plot([], [], color="0.6", label="1 = same-day baseline", **BASE)
    a.plot([], [], label="0 = chance", **CHANCE)
    a.set(xticks=x, xticklabels=BIN_LAB, xlabel="days apart", ylim=(0, 1.1),
          ylabel="fraction of baseline kept", title="all metrics")
    a.legend(frameon=False, fontsize=4.5, ncol=2, loc="lower left", handlelength=1.8, columnspacing=0.8)
    a.yaxis.set_major_locator(MaxNLocator(3))
    # h normalised metrics per mouse at 1-2 vs >14 days
    a = ax[1, 3]
    short = N[p.dt <= 2].groupby(p.subject).mean()
    long_ = N[p.dt > 14].groupby(p.subject).mean()
    xx = np.arange(len(N.columns))
    YMAX = 1.2
    weak = np.array(["choice" in k for k in N.columns])     # baseline near chance: unreliable
    for off, tab, lab in ((-0.2, short, "1-2 days"), (0.2, long_, ">14 days")):
        bars = a.bar(xx + off, tab.mean(), 0.4, color=TCOL[lab], label=lab)
        for b_, w in zip(bars, weak):
            b_.set_alpha(0.35 if w else 1)
        for s_ in tab.index:
            v = tab.loc[s_].values
            a.plot(xx[v <= YMAX] + off, v[v <= YMAX], ".", color="k", ms=2)
            a.plot(xx[v > YMAX] + off, np.full((v > YMAX).sum(), YMAX - 0.03), "^", color="k", ms=2.5)
    a.set_xticks(xx), a.set_xticklabels(N.columns, rotation=60, ha="right", fontsize=5)
    for t, k in zip(a.get_xticklabels(), N.columns):
        t.set_color(MCOL[var[k]])
    a.axhline(1, color="0.6", **BASE)
    a.set(ylabel="fraction of baseline kept", title="per metric", ylim=(0, YMAX))
    a.legend(frameon=False, fontsize=5)
    a.yaxis.set_major_locator(MaxNLocator(3))
    panel_letters(fig, ax.ravel())
    fig.savefig(OUT / "chronic_stability.pdf")
    fig.savefig(OUT / "chronic_stability.png", dpi=150)
    plt.close(fig)
    return N, short, long_


def summary(qc, sess, pairs, N, short, long_):
    t = qc[(qc.uid != "") & qc.qc_pass]
    lines = [f"subjects {pairs.subject.nunique()}, sessions {len(sess)}, pairs {len(pairs)}",
             f"spatial QC pass {100 * qc.qc_pass[qc.dev_px.notna()].mean():.1f}%; median deviation "
             f"{qc.dev_px.median():.2f} px vs nearest-ROI distance {qc.nn_px.median():.2f} px", "",
             "neurons tracked with QC, per mouse:"]
    for s, g in t.groupby("subject"):
        ns = g.session.nunique()
        cnt = g.groupby("uid").session.nunique()
        lines.append(f"  {s}: {ns} sessions, {int(sess[sess.subject == s].n_neurons.median())} neurons/"
                     f"session; in >=2: {(cnt >= 2).sum()}, >=5: {(cnt >= 5).sum()}, >=half: "
                     f"{(cnt >= ns / 2).sum()}, >=80%: {(cnt >= 0.8 * ns).sum()}, all: {(cnt >= ns).sum()}")
    lines += ["", "fraction of even/odd baseline kept (mean ± sem over mice):"]
    for c in N.columns:
        sl = [np.polyfit(np.log10(q.dt), N.loc[q.index, c], 1)[0]
              for _, q in pairs.groupby("subject") if len(q) >= 10]
        lines.append(f"  {c:20s} 1-2 d {short[c].mean():.2f} ± {short[c].sem():.2f}   >14 d "
                     f"{long_[c].mean():.2f} ± {long_[c].sem():.2f}   slope/log10 day "
                     f"{np.mean(sl):+.3f} (t-test p={ttest_1samp(sl, 0).pvalue:.3f}, {len(sl)} mice)")
    p = pairs.assign(dtbin=pd.cut(pairs.dt, BINS, labels=BIN_LAB))
    cols = ["rw", "rx", "rnn", "r0", "stability", "act_w", "act_x", "stim_sig", "stim_kept_w",
            "stim_kept_x", "choice_sig", "choice_kept_w", "choice_kept_x", "dec_stim_w", "dec_stim_x",
            "dec_choice_w", "dec_choice_x"]
    tab = p.groupby(["subject", "dtbin"], observed=True)[cols].mean().groupby(
        level="dtbin", observed=True).mean()
    txt = "\n".join(lines) + "\n\nraw values (mean over mice):\n" + tab.round(3).T.to_string()
    (OUT / "summary.txt").write_text(txt)
    print(txt)


def drift_figure(d):
    with plt.rc_context(NATURE_RC):
        return _drift_figure(d)


def _drift_figure(d):
    """Stimulus decoder across days: transfer matrices, fixed vs recalibrated
    application, weight similarity, forward vs backward transfer."""
    from matplotlib.lines import Line2D
    d = d[d["var"] == "stim"].copy()
    d["adt"] = d.dt.abs()
    d["dtbin"] = pd.cut(d.adt, BINS, labels=BIN_LAB)
    x = np.arange(len(BIN_LAB))
    subjects = sorted(d.subject.unique())
    c = MCOL["stim"]

    def by_bin(q, col):   # mean over mice of per-mouse bin means -> (mean, sem, per-mouse table)
        t = q.groupby(["subject", "dtbin"], observed=True)[col].mean().unstack(0).reindex(BIN_LAB)
        return t.mean(axis=1), t.std(axis=1) / np.sqrt(t.notna().sum(axis=1)), t

    fig = plt.figure(figsize=(NATURE_DOUBLE_IN, 3.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15])
    top = gs[0].subgridspec(1, len(subjects) + 1, width_ratios=[1] * len(subjects) + [0.06])
    bot = gs[1].subgridspec(1, 3)
    axm = [fig.add_subplot(top[i]) for i in range(len(subjects))]
    # a transfer matrices (fraction of within-day accuracy kept, recalibrated)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")
    for a, s in zip(axm, subjects):
        q = d[d.subject == s]
        days = sorted(set(q.train) | set(q.test), key=lambda k: pd.Timestamp(k[:10]))
        idx = {k: i for i, k in enumerate(days)}
        M = np.full((len(days), len(days)), np.nan)
        M[q.train.map(idx).values, q.test.map(idx).values] = q.kept_own.values
        im = a.imshow(M, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        dnum = [(pd.Timestamp(k[:10]) - pd.Timestamp(days[0][:10])).days for k in days]
        t = np.unique(np.linspace(0, len(days) - 1, min(3, len(days))).round().astype(int))
        a.set_xticks(t), a.set_xticklabels([dnum[i] for i in t])
        a.set_yticks(t), a.set_yticklabels([dnum[i] for i in t])
        a.set_title(s)
        a.set_xlabel("test day")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(True)
    axm[0].set_ylabel("train day")
    cb = fig.colorbar(im, cax=fig.add_subplot(top[-1]), ticks=[0, 0.5, 1])
    cb.set_label("fraction of baseline kept")
    cb.outline.set_linewidth(0.5)
    # b recalibrated vs fixed
    a = fig.add_subplot(bot[0])
    for col, mk, fc, lab in (("kept_own", "o", c, "recalibrated (own z-score)"),
                             ("kept_fixed", "s", "white", "fixed (training z-score)")):
        m, e, _ = by_bin(d, col)
        a.errorbar(x, m, e, color=c, marker=mk, mfc=fc, ms=3, lw=1.2, capsize=0, label=lab)
    a.axhline(1, color="0.6", **BASE), a.axhline(0, **CHANCE)
    a.set(xticks=x, xticklabels=BIN_LAB, xlabel="days apart", ylabel="fraction of baseline kept",
          ylim=(-0.05, 1.1), title="stimulus decoder applied to another day")
    a.legend(frameon=False, fontsize=5, loc="lower left")
    a.yaxis.set_major_locator(MaxNLocator(3))
    # c weight similarity
    a = fig.add_subplot(bot[1])
    q = d[d.dt > 0]   # one row per unordered pair
    m, e, _ = by_bin(q, "cos_w")
    a.plot(x, m, color=c, **BASE)
    m, e, _ = by_bin(q, "cos_x")
    a.errorbar(x, m, e, color=c, marker="o", ms=3, lw=1.2, capsize=0)
    a.axhline(0, **CHANCE)
    a2 = a.twinx()
    m, e, _ = by_bin(q, "kept_cos")
    a2.errorbar(x, m, e, color="k", marker="D", ms=2.5, lw=0.8, capsize=0)
    a2.set_ylim(0, 1.1), a2.set_ylabel("across / same day")
    a2.spines["right"].set_visible(True), a2.yaxis.set_major_locator(MaxNLocator(3))
    a.set(xticks=x, xticklabels=BIN_LAB, xlabel="days apart", ylabel="cosine of decoder weights",
          title="decoder weights")
    a.set_ylim(bottom=-0.005)
    a.legend([Line2D([], [], color=c, **BASE), Line2D([], [], color=c, marker="o", ms=3),
              Line2D([], [], color="k", marker="D", ms=2.5, lw=.8), Line2D([], [], **CHANCE)],
             ["same day, even vs odd", "day a vs day b", "ratio (right axis)", "chance"],
             frameon=False, fontsize=5, loc="lower left")
    a.yaxis.set_major_locator(MaxNLocator(3))
    # d forward vs backward
    a = fig.add_subplot(bot[2])
    for sel, mk, fc, lab in ((d.dt > 0, "o", c, "trained earlier, tested later"),
                             (d.dt < 0, "o", "white", "trained later, tested earlier")):
        m, e, _ = by_bin(d[sel], "kept_own")
        a.errorbar(x, m, e, color=c, marker=mk, mfc=fc, ms=3, lw=1.2, capsize=0, label=lab)
    a.axhline(1, color="0.6", **BASE), a.axhline(0, **CHANCE)
    a.set(xticks=x, xticklabels=BIN_LAB, xlabel="days apart", ylabel="fraction of baseline kept",
          ylim=(-0.05, 1.1), title="direction in time")
    a.legend(frameon=False, fontsize=5, loc="lower left")
    a.yaxis.set_major_locator(MaxNLocator(3))
    panel_letters(fig, [axm[0], fig.axes[len(subjects) + 1], fig.axes[len(subjects) + 2],
                        fig.axes[len(subjects) + 4]])
    fig.savefig(OUT / "decoder_drift.pdf")
    fig.savefig(OUT / "decoder_drift.png", dpi=150)
    plt.close(fig)


def main():
    qc = pd.read_parquet(OUT / "qc_positions.parquet")
    sess = pd.read_csv(OUT / "sessions.csv")
    pairs = pd.read_csv(OUT / "pairs.csv")
    neurons = pd.read_parquet(OUT / "neurons.parquet")
    qc_figure(qc, sess)
    N, short, long_ = main_figure(pairs, neurons)
    summary(qc, sess, pairs, N, short, long_)
    if (OUT / "decoder_drift.parquet").exists():
        drift_figure(pd.read_parquet(OUT / "decoder_drift.parquet"))


if __name__ == "__main__":
    main()
