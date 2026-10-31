#!/usr/bin/env python3
"""Variance of all-neuron decoding across regions vs across sessions.

Panel a, one row per data set, scores per region:
- mesoscope (decode_region.py): one score per session and Beryl region, all neurons of the
  session in the region; shuffle control from decode_region.py --shuffle;
- BWM per probe (decode_region.py): one score per probe from all its isocortex units,
  region = Beryl region at the brain entry point; shuffle control as above;
- BWM decoding paper (BWM_PAPER, stage-2 tables): one score per session and region, with
  its pseudo-session p-value and median null; wheel velocity in place of trial-specific
  wheel speed.
Regions: >= MIN_PER_REGION mesoscope sessions and BWM probes.

Panels b, c (mesoscope only; BWM has about one probe per session, so session variance
cannot be separated there). Statistics per target, on the score with log(neuron count)
regressed out (all-neuron scores grow with the number of neurons):
- mixed model: score ~ log N + (1|region) + (1|animal) + (1|session), crossed random
  effects, REML (statsmodels); variance fractions and likelihood-ratio p-values
  (chi-square 0/1 mixture) per component;
- permutation (model-free): adjusted eta^2 of region within sessions (scores centred per
  session, region labels permuted within session) and of session within regions (scores
  centred per region, session labels permuted within region);
  adjusted eta^2 = (eta^2 - null mean) / (1 - null mean).

Usage: python region_variance.py
Writes output/v2/region_variance{_panels.pdf,.tex,.pdf} and output/v2/region_variance_stats.csv.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from controls2 import SUBJ
from decode_v2 import OUT_V2
from maps_v2 import CHANCE, COL, TARGETS, TITLE
from plotstyle import NATURE_DOUBLE_COL_IN, allen_color, save_with_caption, style_axes

MIN_PER_REGION = 3
N_PERM = 5000
MODS = ("meso", "bwm")
ROWS = ("meso", "bwm", "bwm_paper")
ROW_NAME = {"meso": "Mesoscope", "bwm": "BWM per probe", "bwm_paper": "BWM per region\n(decoding paper)"}
BWM_PAPER = Path.home() / "dmn" / "bwm_decoding"      # <variable>_stage2.pqt
PAPER_VAR = {"stim": "stimside", "choice": "choice", "feedback": "feedback", "wheel": "wheel-speed",
             "wheel_trial": "wheel-velocity"}
COMP = ("region", "animal", "session")
COMP_COL = {"region": "#8c2981", "animal": "#de4968", "session": "#fe9f6d", "residual": "0.85"}
rng = np.random.default_rng(0)


def load():
    m = pd.read_parquet(OUT_V2 / "decoding_meso_region.pqt")
    b = pd.read_parquet(OUT_V2 / "decoding_bwm_probe_region.pqt")
    f = OUT_V2 / "decoding_meso_region_shuf.pqt"      # decode_region.py --shuffle
    if f.exists():
        m = m.merge(pd.read_parquet(f), on=["session", "region", "target"], how="left")
    f = OUT_V2 / "decoding_bwm_probe_shuf.pqt"
    if f.exists():
        b = b.merge(pd.read_parquet(f), on=["pid", "target"], how="left")
    d = {"meso": m.assign(unit=m.session), "bwm": b.assign(unit=b.pid)}
    for k in MODS:
        d[k] = d[k].dropna(subset=["all", "region"]).assign(animal=lambda x: x.session.map(SUBJ),
                                                           logN=lambda x: np.log(x.n_neurons))
    count = {k: d[k].drop_duplicates(["unit", "region"]).groupby("region").size() for k in MODS}
    shared = sorted(set(count["meso"][count["meso"] >= MIN_PER_REGION].index)
                    & set(count["bwm"][count["bwm"] >= MIN_PER_REGION].index), key=allen_order)
    d["bwm_paper"] = pd.concat(
        [pd.read_parquet(BWM_PAPER / f"{v}_stage2.pqt").assign(target=t) for t, v in PAPER_VAR.items()]).rename(
        columns={"eid": "session", "subject": "animal", "score": "all", "n_units": "n_neurons",
                 "median-null": "null", "p-value": "p"}).dropna(subset=["all", "region"])
    d["bwm_paper"]["unit"] = d["bwm_paper"].session
    return {k: d[k][d[k].region.isin(shared)] for k in ROWS}, shared


_ORDER = None


def allen_order(acr):
    global _ORDER
    if _ORDER is None:
        import iblatlas
        t = pd.read_csv(Path(iblatlas.__file__).parent / "allen_structure_tree.csv")
        _ORDER = {a: i for i, a in enumerate(t.acronym)}
    return _ORDER.get(acr, 1e9)


def residualise(df):
    X = np.c_[np.ones(len(df)), df.logN.values]
    beta = np.linalg.lstsq(X, df["all"].values, rcond=None)[0]
    return df["all"].values - X @ beta


def mixed(df, comps=COMP):
    import statsmodels.formula.api as smf
    warnings.filterwarnings("ignore")
    df = df.assign(g=1)
    comps = [c for c in comps if df[c].nunique() > 1]

    def fit(cs):
        vc = {c: f"0 + C({c})" for c in cs}
        return smf.mixedlm("all ~ logN", df, groups="g", re_formula="0", vc_formula=vc).fit(reml=True, method="lbfgs")

    full = fit(comps)
    var = dict(zip(full.model.exog_vc.names, full.vcomp))
    var["residual"] = full.scale
    tot = sum(var.values())
    out = {f"frac_{c}": var.get(c, np.nan) / tot for c in COMP + ("residual",)}
    for c in comps:
        lr = max(2 * (full.llf - fit([x for x in comps if x != c]).llf), 0)
        out[f"p_lrt_{c}"] = 0.5 * stats.chi2.sf(lr, 1)
    out["slope_logN"] = full.fe_params["logN"]
    return out


def eta2(y, labels):
    """Between-group sum of squares / total, for each row of labels (n_perm, n)."""
    labels = np.atleast_2d(labels)
    out = np.empty(len(labels))
    tot = ((y - y.mean()) ** 2).sum()
    for i, lab in enumerate(labels):
        _, inv = np.unique(lab, return_inverse=True)
        s, n = np.bincount(inv, y), np.bincount(inv)
        out[i] = (s ** 2 / n).sum() - y.sum() ** 2 / len(y)
    return out / tot


def centred(y, by):
    return y - pd.Series(y).groupby(by).transform("mean").values


def perm_within(labels, block):
    """N_PERM permutations of labels within each block."""
    out = np.tile(labels, (N_PERM, 1))
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        if idx.size > 1:
            out[:, idx] = np.array([labels[rng.permutation(idx)] for _ in range(N_PERM)])
    return out


def perm_effect(y, labels, block):
    """Adjusted eta^2 of labels on y centred per block, and its permutation p."""
    keep = pd.Series(block).map(pd.Series(block).value_counts()).values > 1
    y, labels, block = y[keep], labels[keep], block[keep]
    if len(np.unique(labels)) < 2 or len(y) < 6:
        return np.nan, np.nan, 0
    yc = centred(y, block)
    obs = eta2(yc, labels)[0]
    null = eta2(yc, perm_within(labels, block))
    return (obs - null.mean()) / (1 - null.mean()), (1 + (null >= obs).sum()) / (1 + N_PERM), len(y)


def analyse(d):
    rows = []
    for k in ("meso",):
        for t in TARGETS:
            df = d[k][d[k].target == t].reset_index(drop=True)
            r = dict(modality=k, target=t, n=len(df), n_units=df.unit.nunique(), n_sessions=df.session.nunique(),
                     n_animals=df.animal.nunique(), rho_N=stats.spearmanr(df["all"], df.n_neurons)[0])
            r.update(mixed(df))
            y = residualise(df)
            reg, ses = df.region.values, df.session.values
            r["eta_region"], r["p_region"], r["n_region"] = perm_effect(y, reg, ses)
            multi = pd.Series(ses).map(pd.Series(ses).value_counts()).values > 1   # sessions with >= 2 regions
            r["eta_session"], r["p_session"], r["n_session"] = perm_effect(y[multi], ses[multi], reg[multi])
            rows.append(r)
            print(k, t, {a: round(b, 3) if isinstance(b, float) else b for a, b in r.items()}, flush=True)
    return pd.DataFrame(rows)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def short(t):
    return TITLE[t].replace("\n", " ").replace("Wheel speed, trial-specific", "Wheel, trial")


def figure(d, shared, st):
    nr = len(ROWS)
    fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 7.6))
    gs = fig.add_gridspec(nr + 2, 5, height_ratios=[1] * nr + [0.42, 0.95], hspace=0.28, wspace=0.35)
    xs = np.arange(len(shared))
    share = {}                                           # one y axis for balanced accuracy, one for R^2
    for r, k in enumerate(ROWS):
        for c, t in enumerate(TARGETS):
            grp = c > 2
            ax = fig.add_subplot(gs[r, c], sharey=share.get(grp))
            share.setdefault(grp, ax)
            df = d[k][d[k].target == t]
            base = [col for col in df.columns if col.startswith("shuf_")] or (["null"] if "null" in df else [])
            for i, reg in enumerate(shared):
                dr = df[df.region == reg]
                bv = dr[base].values.ravel() if base else np.array([])
                bv = bv[np.isfinite(bv)]
                if bv.size:                              # baseline: 5-95th percentile and median
                    lo, med, hi = np.percentile(bv, [5, 50, 95])
                    ax.fill_between([i - 0.38, i + 0.38], lo, hi, color="0.85", lw=0, zorder=0)
                    ax.plot([i - 0.38, i + 0.38], [med] * 2, color="0.55", lw=0.6, zorder=1)
                v = dr["all"].values
                jit = rng.uniform(-0.22, 0.22, len(v))
                col = allen_color(reg)
                face = np.where(dr.p.values < 0.05, col, "white") if "p" in dr else col
                ax.scatter(i + jit, v, s=4, facecolors=face, edgecolors=col if "p" in dr else "0.3",
                           lw=0.4 if "p" in dr else 0.2, zorder=2)
                if len(v):
                    ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="k", lw=0.9, zorder=3)
            ax.axhline(CHANCE[t], color="0.5", lw=0.5, ls="--", zorder=1)
            ax.set_xticks(xs, shared if r == nr - 1 else [], rotation=90, fontsize=4.5)
            ax.set_xlim(-0.6, len(shared) - 0.4)
            style_axes(ax, categorical_x=True)
            if k == "meso":
                s = st[st.target == t].iloc[0]
                ax.text(1.0, 1.01, f"region {s.frac_region:.0%} {stars(s.p_lrt_region)}", transform=ax.transAxes,
                        fontsize=4.5, va="bottom", ha="right")
            if k == "bwm_paper" and t == "wheel_trial":
                ax.text(1.0, 1.01, "wheel velocity", transform=ax.transAxes, fontsize=4.5, va="bottom", ha="right")
            if r == 0:
                ax.set_title(short(t).replace("Wheel, trial", "Wheel speed, trial") + (" ($R^2$)" if c > 2 else ""),
                             fontsize=6, pad=9)
            if c == 0:
                ax.set_ylabel(f"{ROW_NAME[k]}\nbalanced accuracy", fontsize=6)
            elif c != 3:
                ax.tick_params(labelleft=False)
            if r == 0 and c == 0:
                ax.text(-0.45, 1.12, "a", transform=ax.transAxes, fontsize=8, fontweight="bold")

    # b: mixed-model variance fractions (mesoscope)
    ax = fig.add_subplot(gs[nr + 1, 0:2])
    x0 = np.arange(len(TARGETS))
    s = st.set_index("target").loc[list(TARGETS)]
    bottom = np.zeros(len(TARGETS))
    for comp in COMP + ("residual",):
        v = s[f"frac_{comp}"].fillna(0).values
        ax.bar(x0, v, 0.7, bottom=bottom, color=COMP_COL[comp], label=comp, edgecolor="none")
        bottom += v
    for i, t in enumerate(TARGETS):
        ax.text(x0[i], 1.02, stars(s.loc[t, "p_lrt_region"]), ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(x0, [short(t) for t in TARGETS], rotation=30, ha="right", fontsize=5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("variance fraction")
    style_axes(ax, categorical_x=True)
    ax.legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.text(-0.18, 1.18, "b", transform=ax.transAxes, fontsize=8, fontweight="bold")

    # c: permutation, region vs session, adjusted eta^2 (mesoscope)
    ax = fig.add_subplot(gs[nr + 1, 3:5])
    for eff, mk, dx in (("region", "o", -0.1), ("session", "s", 0.1)):
        v, p = s[f"eta_{eff}"].values, s[f"p_{eff}"].values
        ax.scatter(x0 + dx, v, marker=mk, s=14, facecolors=np.where(p < 0.05, COL["meso"], "white"),
                   edgecolors=COL["meso"], lw=0.7, zorder=3, label=eff)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xticks(x0, [short(t) for t in TARGETS], rotation=30, ha="right", fontsize=5)
    ax.set_ylabel("adjusted $\\eta^2$")
    style_axes(ax, categorical_x=True)
    ax.legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.text(-0.18, 1.18, "c", transform=ax.transAxes, fontsize=8, fontweight="bold")
    return fig


def caption(d, shared, st):
    p = d["bwm"].drop_duplicates("pid")
    agree = (p.region == p.region_units).mean()
    n = {k: (d[k].unit.nunique(), d[k].session.nunique(), d[k].animal.nunique()) for k in ROWS}
    bp = d["bwm_paper"]
    sig = {t: (bp[bp.target == t].p < 0.05).mean() for t in TARGETS}
    pct = lambda col: ", ".join(f"{short(t).lower()} {100 * st.set_index('target').loc[t, col]:.0f}\\%"  # noqa: E731
                                for t in TARGETS)
    ses = ", ".join(f"{short(t).lower()} {100 * (st.set_index('target').loc[t, 'frac_session'] + st.set_index('target').loc[t, 'frac_animal']):.0f}\\%"
                    for t in TARGETS)
    intro = ("\\textbf{Variance of decoding across regions and sessions (all neurons).} "
             f"The {len(shared)} regions with $\\geq${MIN_PER_REGION} mesoscope sessions and BWM probes. "
             "Methods: \\emph{Region and session variance}.")
    caps = [
        "Scores per region (columns ordered as in the Allen structure tree; Allen colours; bar: median; dashed: "
        "chance; grey band and line: 5--95th percentile and median of the baseline). Top: mesoscope, one dot per "
        f"session and region from all its neurons there ({n['meso'][0]} sessions, {n['meso'][2]} animals); baseline: "
        "shuffle control. Middle: BWM, one dot per probe from all its isocortex units, placed in the region where it "
        f"enters the brain ({n['bwm'][0]} probes, {n['bwm'][2]} animals; the majority region of its superficial units "
        f"for {100 * agree:.0f}\\% of probes); baseline: shuffle control. Bottom: BWM decoding paper, one dot per "
        f"session and region from the region's units ({len(bp.drop_duplicates(['session', 'region']))} pairs, "
        f"{n['bwm_paper'][1]} sessions, {n['bwm_paper'][2]} animals); filled: $p<0.05$ against pseudo-sessions "
        f"({100 * min(sig.values()):.0f}--{100 * max(sig.values()):.0f}\\% of dots); baseline: median of the "
        "pseudo-session null; wheel velocity in place of trial-specific wheel speed. Y axes shared within balanced "
        "accuracy and within $R^2$. Inset, top row: region share of variance from \\textbf{b}.",
        "Mesoscope variance fractions from a mixed model, score $\\sim$ log $N$ + (1$|$region) + (1$|$animal) + "
        "(1$|$session); stars: likelihood-ratio test of the region component. Region: " + pct("frac_region")
        + "; session plus animal: " + ses + ".",
        "The same question as \\textbf{b}, do scores differ more between regions or between sessions, but without a "
        "statistical model, using only reshuffling: adjusted $\\eta^2$ of region (circles, within sessions) and of "
        "session (squares, within regions), 0 = chance. Filled: $p<0.05$.",
    ]
    return intro, caps


def main():
    d, shared = load()
    print("shared regions:", shared)
    st = analyse(d)
    st.to_csv(OUT_V2 / "region_variance_stats.csv", index=False)
    fig = figure(d, shared, st)
    intro, caps = caption(d, shared, st)
    save_with_caption(fig, OUT_V2 / "region_variance", caps, "region_variance", intro=intro)
    plt.close(fig)
    print(st.round(3).to_string())


if __name__ == "__main__":
    main()
