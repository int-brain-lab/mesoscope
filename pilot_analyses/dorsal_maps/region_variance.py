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
The region cohort is fixed to the original v7 comparison.

Panels b, c (mesoscope only; BWM has about one probe per session, so session variance
cannot be separated there). Statistics per target, on the score with log(neuron count)
regressed out (all-neuron scores grow with the number of neurons):
- mixed model: score ~ log N + (1|region) + (1|animal) + (1|session), crossed random
  effects, REML (statsmodels); variance fractions and likelihood-ratio p-values
  (chi-square 0/1 mixture) per component;
- region test: partial R^2 for adding region after session and log neuron count, with
  reduced-model residuals permuted within sessions and Holm adjustment across targets;
- session permutation: adjusted eta^2 of session within regions (scores
  centred per region, session labels permuted within region; includes animal differences), and
  of session within mouse (scores centred per region x mouse, session labels permuted within
  region x mouse; no animal differences);
  adjusted eta^2 = (eta^2 - null mean) / (1 - null mean);
- animal bootstrap: N_BOOT resamples of animals with replacement (a repeated animal and its
  sessions count as new ones), each refitted with reml.py; 95% intervals of the region and
  session + animal shares, and the fraction of resamples with session + animal > region.
  Outputs are written to output/v2/region_variance_boot_auc.csv;
- per mouse (panel d): score ~ log N + (1|region) + (1|session) within each mouse with
  >= MIN_MOUSE_SESSIONS sessions (reml.py); session share minus region share;
- region x mouse: the pooled model with an extra (1|region:animal) term (reml.py), columns
  ra_* in the stats csv (reported in the text, not plotted);
- panel e: session x region score matrices (log N regressed out) of three example mice with
  >= 10 sessions and >= 7 regions: the most region-dominated, the most session-dominated and
  the most balanced by panel d, averaged over HEAT_TARGETS.

Usage: python region_variance.py
Writes output/v2/region_variance{_panels.pdf,.tex,.pdf} and output/v2/region_variance_stats.csv.
"""
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from controls2 import SUBJ
from decode_v2 import OUT_V2
from maps_v2 import CHANCE, COL, TARGETS, TITLE, load_matched_decoding
from plotstyle import NATURE_DOUBLE_COL_IN, allen_color, save_with_caption, style_axes

REGIONS = ("MOp", "MOs", "SSp-bfd", "SSp-ll", "SSp-m", "SSp-ul", "SSp-tr",
           "VISam", "VISl", "VISp", "VISpm", "RSPagl", "RSPd", "VISa", "VISrl")
N_PERM = 5000
N_REGION_PERM = 9999
N_BOOT = 200
MIN_MOUSE_SESSIONS = 4
HEAT_TARGETS = ("stim", "feedback")                      # panel e
MODS = ("meso", "bwm")
ROWS = ("meso", "bwm", "bwm_paper")
ROW_NAME = {"meso": "Mesoscope", "bwm": "BWM per probe", "bwm_paper": "BWM per region\n(decoding paper)"}
BWM_PAPER = Path.home() / "dmn" / "bwm_decoding"      # <variable>_stage2.pqt
PAPER_REFERENCE = Path(__file__).resolve().parent / "reference" / "bwm_published_region_row.pdf"
PAPER_VAR = {"stim": "stimside", "choice": "choice", "feedback": "feedback", "wheel": "wheel-speed",
             "wheel_trial": "wheel-velocity"}
COMP = ("region", "animal", "session")
COMP_COL = {"region": "#8c2981", "animal": "#de4968", "session": "#fe9f6d", "residual": "0.85"}
rng = np.random.default_rng(0)


def load():
    matched, _ = load_matched_decoding()
    shared = list(REGIONS)
    d = {}
    for modality, stem, shuffle_stem, unit, keys in (
        ("meso", "meso_region", "meso_region", "session", ["target", "session", "region"]),
        ("bwm", "bwm_probe_region", "bwm_probe", "pid", ["target", "pid"]),
    ):
        scores = pd.read_parquet(OUT_V2 / f"decoding_{stem}_auc.pqt")
        scores = scores.loc[np.isfinite(scores["all"]) & scores.region.isin(shared)]
        ids = ["target", "session"] if modality == "meso" else ["target", "session", "pid"]
        cohort = matched[modality]
        scores = scores.merge(cohort[ids].drop_duplicates(), on=ids, validate="many_to_one")
        if modality == "bwm":
            map_scores = cohort[ids + ["all"]].rename(columns={"all": "map_score"})
            scores = scores.merge(map_scores, on=ids, validate="one_to_one")
            np.testing.assert_allclose(scores["all"], scores.map_score, rtol=0, atol=1e-12)
        shuffled = pd.read_parquet(OUT_V2 / f"decoding_{shuffle_stem}_shuf_auc.pqt")
        scores = scores.merge(shuffled, on=keys, validate="one_to_one", how="left", indicator=True)
        if scores.empty or not scores["_merge"].eq("both").all():
            raise ValueError(f"Missing {modality} region scores or trial-shuffle rows for the matched cohort")
        scores = scores.drop(columns="_merge").assign(unit=scores[unit], animal=scores.session.map(SUBJ),
                                                       logN=np.log(scores.n_neurons))
        if scores.animal.isna().any() or not np.isfinite(scores.logN).all():
            raise ValueError(f"Missing animal or invalid neuron count in {modality} scores")
        d[modality] = scores

    files = {t: BWM_PAPER / f"{v}_stage2.pqt" for t, v in PAPER_VAR.items()}
    if all(path.is_file() for path in files.values()):
        paper = pd.concat([pd.read_parquet(path).assign(target=t) for t, path in files.items()])
        paper = paper.rename(columns={"eid": "session", "subject": "animal", "score": "all",
                                      "n_units": "n_neurons", "median-null": "null", "p-value": "p"})
        paper = paper.dropna(subset=["all"]).loc[lambda x: x.region.isin(shared)]
        d["bwm_paper"] = paper.assign(unit=paper.session)
    else:
        if not PAPER_REFERENCE.is_file():
            raise FileNotFoundError(PAPER_REFERENCE)
        d["bwm_paper"] = d["bwm"].iloc[:0]
    return d, shared


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


def permute_residuals(residual, sessions, rng, n_draws):
    """Permute each session's residuals independently in every draw."""
    draws = np.empty((n_draws, len(residual)))
    for block in sessions:
        draws[:, block] = rng.permuted(np.broadcast_to(residual[block], (n_draws, len(block))), axis=1)
    return draws


def region_permutation(df, n_perm=N_REGION_PERM, seed=0):
    """Partial R? for Beryl region after session effects and log neuron count."""
    df = df.loc[df.groupby("session").session.transform("size") >= 2]
    df = df.sort_values(["session", "region"]).reset_index(drop=True)
    result = dict(region_partial_r2=np.nan, p_region=np.nan, n_region=len(df),
                  region_sessions=df.session.nunique(), region_mice=df.animal.nunique(),
                  region_labels=df.region.nunique(), region_permutations=n_perm)
    if result["region_labels"] < 2:
        return result

    # Within-session demeaning projects out the fixed session intercepts.
    y = centred(df["all"].to_numpy(), df.session.to_numpy())
    log_n = centred(df.logN.to_numpy(), df.session.to_numpy())
    z = pd.get_dummies(df.region, dtype=float)
    z = (z - z.groupby(df.session).transform("mean")).to_numpy()
    if np.linalg.norm(log_n) > 1e-12:
        log_n /= np.linalg.norm(log_n)
    residual = y - log_n * (log_n @ y)
    z -= np.outer(log_n, log_n @ z)
    u, singular, _ = np.linalg.svd(z, full_matrices=False)
    basis = u[:, singular > np.finfo(float).eps * max(z.shape) * singular.max()]
    if not basis.shape[1] or np.linalg.norm(residual) <= 1e-12:
        return result

    def partial_r2(values):
        return ((values @ basis) ** 2).sum(axis=-1) / (values ** 2).sum(axis=-1)

    observed = partial_r2(residual)
    blocks = list(df.groupby("session", sort=False).indices.values())
    generator = np.random.default_rng(seed)
    exceedances = 0
    for start in range(0, n_perm, 256):
        permuted = permute_residuals(residual, blocks, generator, min(256, n_perm - start))
        # Adding back the reduced-model fit and residualizing the shuffled response
        # is equivalent to removing the log-N projection from the shuffled residuals.
        permuted -= np.outer(permuted @ log_n, log_n)
        exceedances += np.count_nonzero(partial_r2(permuted) >= observed - 1e-12)
    return {**result, "region_partial_r2": observed, "p_region": (1 + exceedances) / (1 + n_perm)}


def adjust_region_pvalues(scores):
    """Holm per target and Bonferroni min-p for the prespecified five-target family."""
    from statsmodels.stats.multitest import multipletests

    p = scores.p_region
    adjusted = multipletests(p.fillna(1).to_numpy(), method="holm")[1]
    return scores.assign(p_region_holm=np.where(p.notna(), adjusted, np.nan),
                         p_region_omnibus=min(1, len(TARGETS) * p.min()) if p.notna().any() else np.nan)


def boot_shares(df):
    """Animal-level bootstrap of the region and session + animal shares (reml.py)."""
    import reml
    boot_rng = np.random.default_rng(1)
    animals = df.animal.unique()
    reg, sa = [], []
    for _ in range(N_BOOT):
        pick = boot_rng.choice(animals, len(animals))
        parts = [df[df.animal == a].assign(animal=f"{a}_{i}", session=lambda x, i=i: x.session + f"_{i}")
                 for i, a in enumerate(pick)]
        try:
            m = reml.components(pd.concat(parts, ignore_index=True), COMP)
        except RuntimeError:  # non positive-definite V when a variance collapses; skip this resample
            continue
        reg.append(m["frac_region"])
        sa.append(m.get("frac_session", 0) + m.get("frac_animal", 0))
    reg, sa = np.array(reg), np.array(sa)
    return dict(region_lo=np.percentile(reg, 2.5), region_hi=np.percentile(reg, 97.5),
                ses_anim_lo=np.percentile(sa, 2.5), ses_anim_hi=np.percentile(sa, 97.5),
                boot_frac_pos=np.mean(sa > reg), n_boot=len(reg))


def bootstrap(d):
    f = OUT_V2 / "region_variance_boot_auc.csv"
    rows = []
    for t in TARGETS:
        df = d["meso"][d["meso"].target == t].reset_index(drop=True)
        rows.append(dict(target=t, **boot_shares(df)))
        print("bootstrap", t, rows[-1], flush=True)
    b = pd.DataFrame(rows)
    b.to_csv(f, index=False)
    return b


def per_mouse(d):
    """Session and region shares within each mouse (no animal component)."""
    import reml
    rows = []
    for t in TARGETS:
        df = d["meso"][d["meso"].target == t]
        for a, g in df.groupby("animal"):
            if g.session.nunique() < MIN_MOUSE_SESSIONS or g.region.nunique() < 2:
                continue
            m = reml.components(g.reset_index(drop=True), ("region", "session"))
            rows.append(dict(target=t, animal=a, n=len(g), sessions=g.session.nunique(),
                             frac_region=m.get("frac_region", 0), frac_session=m.get("frac_session", 0)))
    return pd.DataFrame(rows)


def analyse(d):
    rows = []
    for k in ("meso",):
        for t in TARGETS:
            df = d[k][d[k].target == t].reset_index(drop=True)
            r = dict(modality=k, target=t, n=len(df), n_units=df.unit.nunique(), n_sessions=df.session.nunique(),
                     n_animals=df.animal.nunique(), rho_N=stats.spearmanr(df["all"], df.n_neurons)[0])
            r.update(mixed(df))
            import reml                                                      # + region x mouse term (text only)
            ra = reml.components(df.assign(region_animal=df.region + "|" + df.animal),
                                 COMP + ("region_animal",), lrt=("region_animal",))
            r.update({f"ra_{a}": b for a, b in ra.items() if a.startswith(("frac", "p_lrt"))})
            y = residualise(df)
            reg, ses = df.region.values, df.session.values
            r.update(region_permutation(df))
            multi = pd.Series(ses).map(pd.Series(ses).value_counts()).values > 1   # sessions with >= 2 regions
            r["eta_session"], r["p_session"], r["n_session"] = perm_effect(y[multi], ses[multi], reg[multi])
            reg_mouse = reg + "|" + df.animal.values                     # session within mouse: no animal term
            r["eta_session_mouse"], r["p_session_mouse"], r["n_session_mouse"] = perm_effect(y, ses, reg_mouse)
            rows.append(r)
            print(k, t, {a: round(b, 3) if isinstance(b, float) else b for a, b in r.items()}, flush=True)
    return adjust_region_pvalues(pd.DataFrame(rows))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def short(t):
    return TITLE[t].replace("\n", " ").replace("Wheel speed, trial-specific", "Wheel, trial")


def complete_rectangle(valid, min_regions=5):
    """Keep a complete rectangle with at least min_regions columns.

    Minimize total deletions, preferring session deletions when tied.
    """
    n_rows, n_cols = valid.shape
    best = None
    for n_removed in range(n_cols - min_regions + 1):
        if best is not None and n_removed > best[0]:
            break
        for removed in combinations(range(n_cols), n_removed):
            cols = np.ones(n_cols, dtype=bool)
            cols[list(removed)] = False
            # With these columns fixed, every complete row can be kept.
            rows = valid[:, cols].all(axis=1)
            n_kept = rows.sum()
            if not n_kept:
                continue
            key = (n_removed + n_rows - n_kept, n_removed)
            if best is None or key < best:
                best, keep_rows, keep_cols = key, rows, cols
    if best is None:
        raise ValueError(f"No complete illustrative matrix with at least {min_regions} regions.")
    return keep_rows, keep_cols


def plot_region_scores(ax, df, shared, t, rng):
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


def figure(d, shared, st, pm):
    nr = len(ROWS)
    fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 9.4))
    gs = fig.add_gridspec(nr + 4, 5, height_ratios=[1] * nr + [0.42, 1.25, 0.45, 0.95], hspace=0.28,
                          wspace=0.35)
    xs = np.arange(len(shared))
    share = {}                                           # group y axes by metric
    for r, k in enumerate(ROWS):
        for c, t in enumerate(TARGETS):
            metric = "balanced accuracy" if k == "bwm_paper" else "ROC-AUC"
            grp = "R2" if c > 2 else metric
            ax = fig.add_subplot(gs[r, c], sharey=share.get(grp))
            share.setdefault(grp, ax)
            df = d[k][d[k].target == t]
            if k == "bwm_paper" and d[k].empty:
                ax.set_visible(False)
                continue
            plot_region_scores(ax, df, shared, t, rng)
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
                ax.set_ylabel(f"{ROW_NAME[k]}\n{metric}", fontsize=6)
            elif c != 3:
                ax.tick_params(labelleft=False)
            if r == 0 and c == 0:
                ax.text(-0.45, 1.12, "a", transform=ax.transAxes, fontsize=8, fontweight="bold")

    stats_row = gs[nr + 3, :].subgridspec(1, 5, width_ratios=[1, 0.45, 1, 0.7, 1], wspace=0.35)
    # c: mixed-model variance fractions (mesoscope)
    ax = fig.add_subplot(stats_row[0, 0])
    x0 = np.arange(len(TARGETS))
    s = st.set_index("target").loc[list(TARGETS)]
    bottom = np.zeros(len(TARGETS))
    for comp in COMP + ("residual",):
        v = s[f"frac_{comp}"].fillna(0).values
        ax.bar(x0, v, 0.7, bottom=bottom, color=COMP_COL[comp], label=comp, edgecolor="none")
        bottom += v
    reg, sa = s.frac_region.values, s.frac_session.fillna(0).values + s.frac_animal.fillna(0).values
    ax.errorbar(x0 - 0.12, reg, yerr=[reg - s.region_lo, s.region_hi - reg], fmt="none", ecolor="k", elinewidth=0.6,
                capsize=1.5, capthick=0.6, zorder=3)
    ax.errorbar(x0 + 0.12, reg + sa, yerr=[sa - s.ses_anim_lo, s.ses_anim_hi - sa], fmt="none", ecolor="k",
                elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
    for i, t in enumerate(TARGETS):
        ax.text(x0[i], 1.02, f"{100 * s.loc[t, 'boot_frac_pos']:.0f}%", ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(x0, [short(t) for t in TARGETS], rotation=30, ha="right", fontsize=5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("variance fraction")
    style_axes(ax, categorical_x=True)
    ax.legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.text(-0.3, 1.12, "c", transform=ax.transAxes, fontsize=8, fontweight="bold")

    # d: region and session permutations use different statistics and axes.
    tests = stats_row[0, 2:4].subgridspec(1, 2, wspace=0.7)
    ax = fig.add_subplot(tests[0, 0])
    region, p = s.region_partial_r2, s.p_region_holm
    ax.scatter(x0, region, s=14, facecolors=np.where(p < 0.05, COL["meso"], "white"),
               edgecolors=COL["meso"], lw=0.7)
    for x, value, adjusted in zip(x0, region, p):
        if np.isfinite(adjusted):
            label = f"{adjusted:.1g}" if adjusted >= 0.001 else f"{adjusted:.0e}".replace("e-0", "e-")
            ax.annotate(label, (x, value), xytext=(0, 4), textcoords="offset points",
                        fontsize=4, ha="center")
    ax.set_title("Region (Holm $p$)", fontsize=6)
    ax.set_ylim(0, max(0.1, 1.24 * region.max()))
    ax.set_ylabel("partial $R^2$")
    ax.text(-0.3, 1.12, "d", transform=ax.transAxes, fontsize=8, fontweight="bold")
    session_ax = fig.add_subplot(tests[0, 1])
    for eff, marker, dx, label in (("session", "s", -0.10, "All mice"),
                                   ("session_mouse", "^", 0.10, "Within mouse")):
        v, p = s[f"eta_{eff}"], s[f"p_{eff}"]
        session_ax.scatter(x0 + dx, v, marker=marker, s=14,
                           facecolors=np.where(p < 0.05, COL["meso"], "white"),
                           edgecolors=COL["meso"], lw=0.7, label=label)
    session_ax.axhline(0, color="0.5", lw=0.5)
    session_ax.set_ylim(top=1.45 * np.nanmax(s[["eta_session", "eta_session_mouse"]].values))
    session_ax.set_title("Session", fontsize=6)
    session_ax.set_ylabel("adjusted $\\eta^2$")
    session_ax.legend(frameon=False, fontsize=4, loc="upper center")
    for axis in (ax, session_ax):
        axis.set_xticks(x0, [short(t) for t in TARGETS], rotation=45, ha="right", fontsize=4.5)
        style_axes(axis, categorical_x=True)

    # e: session minus region share within each mouse (mesoscope)
    ax = fig.add_subplot(stats_row[0, 4])
    jit = np.random.default_rng(2)
    for i, t in enumerate(TARGETS):
        v = (pm[pm.target == t].frac_session - pm[pm.target == t].frac_region).values
        ax.scatter(i + jit.uniform(-0.15, 0.15, len(v)), v, s=6, color=COL["meso"], alpha=0.7, lw=0, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=1, zorder=4)
        ax.text(i, 1.02, f"{(v > 0).sum()}/{len(v)}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=4.5)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xticks(x0, [short(t) for t in TARGETS], rotation=30, ha="right", fontsize=5)
    ax.set_ylabel("session $-$ region share")
    ax.yaxis.set_label_position("right")
    style_axes(ax, categorical_x=True)
    ax.text(-0.3, 1.12, "e", transform=ax.transAxes, fontsize=8, fontweight="bold")

    # b: session x region matrices of three example mice
    mice, kinds = example_mice(d, pm)
    sub = gs[nr + 1, :].subgridspec(1, 9, width_ratios=[1, 1, 1, 0.07, 0.45, 1, 1, 1, 0.07], wspace=0.25)
    df_all = d["meso"].assign(y=np.nan)
    for t in HEAT_TARGETS:
        idx = df_all.target == t
        df_all.loc[idx, "y"] = residualise(df_all[idx].reset_index(drop=True))
    for g, t in enumerate(HEAT_TARGETS):
        dt = df_all[df_all.target == t]
        lo, hi = np.nanpercentile(dt[dt.animal.isin(mice)].y, [5, 95])
        for j, (a, kind) in enumerate(zip(mice, kinds)):
            ax = fig.add_subplot(sub[0, 5 * g + j])
            m = dt[dt.animal == a].pivot_table(index="session", columns="region", values="y")
            m = m[sorted(m.columns, key=allen_order)]
            m = m.loc[m.mean(axis=1).sort_values(ascending=False).index]
            keep_rows, keep_cols = complete_rectangle(np.isfinite(m.values))
            m = m.iloc[keep_rows, keep_cols]
            im = ax.imshow(m.values, aspect="auto", cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks(range(m.shape[1]), m.columns, rotation=90, fontsize=3.5)
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            dv = pm[(pm.animal == a) & (pm.target == t)]
            dv = (dv.frac_session - dv.frac_region).iloc[0]
            ax.set_title(f"{short(t)}\n{a} ({kind})\nsession $-$ region {dv:+.2f}", fontsize=4.5, pad=2)
            if g == 0 and j == 0:
                ax.set_ylabel("sessions,\nsorted by mean", fontsize=4.5)
            if g == 0 and j == 0:
                ax.text(-0.45, 1.25, "b", transform=ax.transAxes, fontsize=8, fontweight="bold")
        cax = fig.add_subplot(sub[0, 5 * g + 3])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=4, length=1.5)
        cb.outline.set_visible(False)
        cb.set_label("score, log $N$ removed", fontsize=4.5)
    return fig


def example_mice(d, pm):
    """Region-dominated, balanced and session-dominated mouse (panel d, mean over HEAT_TARGETS)."""
    m = d["meso"][d["meso"].target == HEAT_TARGETS[0]]
    big = m.groupby("animal").agg(ses=("session", "nunique"), reg=("region", "nunique"))
    big = big[(big.ses >= 10) & (big.reg >= 7)].index
    q = pm[pm.target.isin(HEAT_TARGETS) & pm.animal.isin(big)]
    q = (q.frac_session - q.frac_region).groupby(q.animal).mean()
    return [q.idxmin(), (q - q.median()).abs().idxmin(), q.idxmax()], ["region-dominated", "balanced",
                                                                       "session-dominated"]


def caption(d, shared, st, pm):
    n = {k: (d[k].unit.nunique(), d[k].session.nunique(), d[k].animal.nunique()) for k in ROWS}
    intro = ("\\textbf{Variance of decoding across regions and sessions (all neurons).} "
             f"The {len(shared)} regions in the original v7 comparison; models, "
             "tests and baselines in Methods, \\emph{Region and session variance}.")
    caps = [
        "Scores per region (Allen order and colours; bar: median; dashed: chance; grey band and line: 5--95th "
        f"percentile and median of the baseline). Rows: mesoscope, one dot per session and region ({n['meso'][0]} "
        f"sessions, {n['meso'][2]} mice); BWM, one dot per probe at its entry region ({n['bwm'][0]} probes); BWM "
        "decoding paper, one dot per session and region, filled if $p<0.05$ (published balanced accuracy and "
        "wheel velocity in the last column). Binary scores in the first two rows use ROC-AUC. "
        "Inset: region share from \\textbf{c}.",
        "Session $\\times$ region scores (log $N$ removed) of a region-dominated, a balanced and a "
        "session-dominated mouse. For illustration, each matrix retains at least five regions "
        "with valid scores in every displayed session. Among equally small sets of omissions, "
        "the display keeps more regions. Horizontal stripes show session effects "
        "and vertical stripes show region effects.",
        "Mesoscope variance shares from the mixed model; error bars: 95\\% intervals over resampled mice; numbers: "
        "fraction of resamples with session plus animal above region.",
        "Region test (left): partial $R^2$ after accounting for session and log neuron count, with Holm-adjusted "
        f"$p$-values. The five-target omnibus permutation $p$-value is {st.p_region_omnibus.iloc[0]:.3g}. "
        "Session tests (right): adjusted $\\eta^2$, with sessions across all mice (squares) and within each mouse "
        "(triangles). Filled markers indicate $p<0.05$ using Holm-adjusted $p$ for the region test.",
        f"Session minus region share, fitted per mouse ({pm.animal.nunique()} mice with $\\geq${MIN_MOUSE_SESSIONS} "
        "sessions); bar: median; numbers: mice with a larger session share.",
    ]
    return intro, caps


def add_paper_reference(panel):
    """Place the archived published row into the reserved third row of panel a."""
    import pikepdf

    with pikepdf.open(PAPER_REFERENCE) as paper, pikepdf.open(panel, allow_overwriting_input=True) as current:
        page = current.pages[0]
        # The top two rows have identical subplot dimensions and anchor the reference.
        rects = [tuple(map(float, i.operands)) for i in pikepdf.parse_content_stream(page)
                 if str(i.operator) == "re"]
        axes = [r for r in rects if np.allclose(r[2:], (62.81619, 69.23739), atol=0.001, rtol=0)]
        left = min(r[0] for r in axes)
        y_top, y_middle = sorted({r[1] for r in axes if r[0] == left}, reverse=True)
        # The first axis in the archived crop starts at (52.159, 29.121) points.
        dx = left - 52.158984375
        dy = 2 * y_middle - y_top - 29.1213369388
        box = list(map(float, page.MediaBox))
        box[0] = min(box[0], dx)
        box[2] = max(box[2], dx + float(paper.pages[0].MediaBox[2]))
        page.MediaBox = pikepdf.Array(box)
        name = pikepdf.Name("/PublishedBwm")
        page.add_resource(current.copy_foreign(paper.pages[0].as_form_xobject()), pikepdf.Name.XObject, name=name)
        page.contents_add(current.make_stream(f"q 1 0 0 1 {dx:.6f} {dy:.6f} cm {name} Do Q\\n".encode()))
        current.save(panel)


def render(d, shared, st, pm):
    import shutil
    import subprocess

    fig = figure(d, shared, st, pm)
    intro, caps = caption(d, shared, st, pm)
    archive = d["bwm_paper"].empty
    if archive:
        intro += " The published BWM row is reproduced from the archived v7 vector panel."
    panel, tex, _ = save_with_caption(fig, OUT_V2 / "region_variance", caps, "region_variance",
                                      intro=intro, compile_pdf=not archive)
    if archive:
        add_paper_reference(panel)
        if shutil.which("tectonic"):
            subprocess.run(["tectonic", tex.name], cwd=tex.parent, check=True)
    plt.close(fig)


def main():
    d, shared = load()
    st = analyse(d).merge(bootstrap(d), on="target", validate="one_to_one")
    st.to_csv(OUT_V2 / "region_variance_stats.csv", index=False)
    pm = per_mouse(d)
    pm.to_csv(OUT_V2 / "region_variance_per_mouse.csv", index=False)
    render(d, shared, st, pm)
    print(st[["target", "region_partial_r2", "p_region", "p_region_holm", "p_region_omnibus",
              "region_sessions", "region_mice", "region_labels"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
