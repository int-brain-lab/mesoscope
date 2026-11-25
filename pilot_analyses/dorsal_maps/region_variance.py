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

Panels b--e (mesoscope only; BWM has about one probe per session, so session variance
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
- per mouse (panel c): score ~ log N + (1|region) + (1|session) within each mouse with
  >= MIN_MOUSE_SESSIONS sessions (reml.py); region share minus session share.
- region x mouse: the pooled model with an extra (1|region:animal) term (reml.py), columns
  ra_* in the stats csv (reported in the text, not plotted);
- panel b: session x region score matrices (log N regressed out) of three example mice with
  >= 10 sessions and >= 7 regions: the most region-dominated, the most session-dominated and
  the most balanced by panel c, averaged over HEAT_TARGETS.

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
from maps_v2 import CHANCE, TARGETS, TITLE, load_matched_decoding
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
COMP_COL = {"region": "#8c2981", "animal": "#de4968", "session": "#fe9f6d", "residual": "#c9ded6"}
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


def mixed(df, comps=COMP, *, adjust_log_n=True):
    import statsmodels.formula.api as smf
    warnings.filterwarnings("ignore")
    df = df.assign(g=1)
    comps = [c for c in comps if df[c].nunique() > 1]
    formula = "all ~ logN" if adjust_log_n else "all ~ 1"

    def fit(cs):
        vc = {c: f"0 + C({c})" for c in cs}
        return smf.mixedlm(formula, df, groups="g", re_formula="0", vc_formula=vc).fit(reml=True, method="lbfgs")

    full = fit(comps)
    var = dict(zip(full.model.exog_vc.names, full.vcomp))
    var["residual"] = full.scale
    tot = sum(var.values())
    out = {f"frac_{c}": var.get(c, np.nan) / tot for c in COMP + ("residual",)}
    for c in comps:
        lr = max(2 * (full.llf - fit([x for x in comps if x != c]).llf), 0)
        out[f"p_lrt_{c}"] = 0.5 * stats.chi2.sf(lr, 1)
    if adjust_log_n:
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
    result = dict(region_partial_r2=np.nan, region_null_lo=np.nan, region_null_hi=np.nan,
                  p_region=np.nan, n_region=len(df),
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
    if n_perm == 0:
        return {**result, "region_partial_r2": observed}
    blocks = list(df.groupby("session", sort=False).indices.values())
    generator = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for start in range(0, n_perm, 256):
        permuted = permute_residuals(residual, blocks, generator, min(256, n_perm - start))
        # Adding back the reduced-model fit and residualizing the shuffled response
        # is equivalent to removing the log-N projection from the shuffled residuals.
        permuted -= np.outer(permuted @ log_n, log_n)
        null[start:start + len(permuted)] = partial_r2(permuted)
    lo, hi = np.quantile(null, [0.025, 0.975])
    return {**result, "region_partial_r2": observed, "region_null_lo": lo, "region_null_hi": hi,
            "p_region": (1 + np.count_nonzero(null >= observed - 1e-12)) / (1 + n_perm)}


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


def figure(d, shared, st, pm, example_tests):
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

    mice, kinds = example_mice(d, pm)
    symbols = dict(zip(mice, (("o", "#8456a1"), ("s", "#3d719f"), ("^", "#b95b18"))))
    target_labels = [short(t) for t in TARGETS]
    x0 = np.arange(len(TARGETS))
    target_x = dict(zip(TARGETS, x0))
    offsets = dict(zip(sorted(pm.animal.unique()), np.linspace(-0.22, 0.22, pm.animal.nunique())))
    stats = st.set_index("target").loc[list(TARGETS)]
    stats_row = gs[nr + 3, :].subgridspec(1, 3, width_ratios=[1.05, 0.83, 1.65], wspace=0.4)
    ax_c = fig.add_subplot(stats_row[0, 0])
    ax_d = fig.add_subplot(stats_row[0, 1])
    tests = stats_row[0, 2].subgridspec(1, 2, width_ratios=[0.77, 1], wspace=0.1)
    ax_e = fig.add_subplot(tests[0, 0])
    ax_examples = fig.add_subplot(tests[0, 1], sharey=ax_e)

    # c: one labeled marker per mouse, linked to the examples in b.
    ax_c.axhline(0, color="0.4", lw=0.6, zorder=0)
    for mouse, (marker, color) in symbols.items():
        rows = pm.loc[pm.animal == mouse]
        ax_c.scatter(rows.target.map(target_x) + offsets[mouse], rows.frac_region - rows.frac_session,
                     s=18, marker=marker, color=color, edgecolors="white", linewidths=0.3,
                     label=mouse, zorder=4)
    other = pm.loc[~pm.animal.isin(symbols)]
    ax_c.scatter(other.target.map(target_x) + other.animal.map(offsets), other.frac_region - other.frac_session,
                 s=10, marker="o", color="0.3", linewidths=0, label="Other mice", zorder=2)
    ax_c.set_title("Within each mouse", fontsize=6, pad=14)
    ax_c.set_ylim(-1.21, 1.21)
    ax_c.text(0.02, 0.95, "More region variance", transform=ax_c.transAxes, fontsize=4.5, va="top")
    ax_c.text(0.02, 0.05, "More session variance", transform=ax_c.transAxes, fontsize=4.5, va="bottom")
    ax_c.set_ylabel("region $-$ session share")
    ax_c.legend(frameon=False, ncol=4, fontsize=4.2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                columnspacing=0.55, handletextpad=0.15, handlelength=0.9, borderaxespad=0.1)

    # d: fractions of total variance in the pooled mixed model.
    bottom = np.zeros(len(TARGETS))
    for comp in COMP + ("residual",):
        v = stats[f"frac_{comp}"].fillna(0).to_numpy()
        ax_d.bar(x0, v, 0.72, bottom=bottom, color=COMP_COL[comp], label=comp.capitalize(), edgecolor="none")
        bottom += v
    for x, share in zip(x0, stats.frac_region):
        ax_d.text(x, share / 2, f"{share:.0%}", color="white", ha="center", va="center", fontsize=4.5)
    ax_d.set_ylim(0, 1)
    ax_d.set_ylabel("share of total variance")
    ax_d.legend(frameon=False, ncol=2, fontsize=4.5, loc="lower center", bbox_to_anchor=(0.5, 1.01))

    # e: region effect after removing session and log neuron count.
    def null_band(axis, x, rows, width, label=None):
        axis.bar(x, rows.region_null_hi - rows.region_null_lo, width=width,
                 bottom=rows.region_null_lo, color="0.75", linewidth=0, label=label)

    null_band(ax_e, x0, stats, 0.65, "95% null")
    ax_e.scatter(x0, stats.region_partial_r2, marker="D", s=16, color="black",
                 edgecolors="white", linewidths=0.4, zorder=5, label="Observed")
    ax_e.set_title(f"Pooled\nOmnibus $p$ = {stats.p_region_omnibus.iloc[0]:.3g}", fontsize=6)
    ax_e.legend(frameon=False, fontsize=4.2, loc="upper left", borderaxespad=0.25)
    ax_e.set_ylim(0, 1)
    ax_e.set_ylabel("region partial $R^2$")

    for mouse, offset in zip(mice, (-0.26, 0, 0.26)):
        rows = example_tests.loc[example_tests.animal == mouse].set_index("target").reindex(TARGETS)
        marker, color = symbols[mouse]
        x = x0 + offset
        null_band(ax_examples, x, rows, 0.24)
        ax_examples.scatter(x, rows.region_partial_r2, marker=marker, s=14, color=color,
                            edgecolors="white", linewidths=0.3, zorder=5)
    ax_examples.set_title("Example mice\nSeparate fits", fontsize=6)
    ax_examples.tick_params(axis="y", left=False, labelleft=False)
    ax_examples.spines["left"].set_visible(False)
    for axis in (ax_c, ax_d, ax_e, ax_examples):
        axis.set_xlim(-0.5, len(TARGETS) - 0.5)
        axis.set_xticks(x0, target_labels, rotation=45, ha="right", fontsize=4.2)
        style_axes(axis, categorical_x=True)
    for label, axis in zip("cde", (ax_c, ax_d, ax_e)):
        axis.text(-0.28, 1.12, label, transform=axis.transAxes, fontsize=8, fontweight="bold")

    # b: session x region matrices of three example mice
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
            dv = (dv.frac_region - dv.frac_session).iloc[0]
            ax.set_title(f"{short(t)}\n{a} ({kind})\nregion $-$ session {dv:+.2f}", fontsize=4.5, pad=2)
            marker, color = symbols[a]
            ax.scatter(0.02, 1.166, transform=ax.transAxes, marker=marker, color=color, s=16,
                       linewidths=0, clip_on=False)
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
    """Region-dominated, balanced and session-dominated mouse (panel c, mean over HEAT_TARGETS)."""
    m = d["meso"][d["meso"].target == HEAT_TARGETS[0]]
    big = m.groupby("animal").agg(ses=("session", "nunique"), reg=("region", "nunique"))
    big = big[(big.ses >= 10) & (big.reg >= 7)].index
    q = pm[pm.target.isin(HEAT_TARGETS) & pm.animal.isin(big)]
    q = (q.frac_region - q.frac_session).groupby(q.animal).mean()
    return [q.idxmax(), (q - q.median()).abs().idxmin(), q.idxmin()], ["region-dominated", "balanced",
                                                                       "session-dominated"]


def example_region_tests(d, pm):
    """Exploratory blocked region tests for the examples selected for panel b."""
    mice, _ = example_mice(d, pm)
    meso = d["meso"]
    return pd.DataFrame([
        dict(animal=mouse, target=target,
             **region_permutation(meso.loc[(meso.animal == mouse) & (meso.target == target)]))
        for mouse in mice for target in TARGETS
    ])


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
        "Inset: region share from \\textbf{d}.",
        "Session $\\times$ region scores (log $N$ removed) of a region-dominated, a balanced and a "
        "session-dominated mouse. For illustration, each matrix retains at least five regions "
        "with valid scores in every displayed session. Among equally small sets of omissions, "
        "the display keeps more regions. Horizontal stripes show session effects "
        "and vertical stripes show region effects.",
        f"Region minus session share from separate mixed-model fits to each mouse "
        f"({pm.animal.nunique()} mice with at least {MIN_MOUSE_SESSIONS} sessions). "
        "Positive values indicate more region variance. The three example mice use the same coloured "
        r"markers as in \textbf{b}. The remaining mice are shown as filled dark grey circles.",
        "Shares of total variance from the pooled mesoscope mixed model. Labels give the percentage "
        "attributed to region.",
        "Fraction of the remaining variance explained by region (partial $R^2$) after accounting for session "
        "and log neuron count. Black diamonds show the pooled effects (left). Coloured points show separate "
        r"fits for the three mice in \textbf{b} (right). "
        "Grey bands span the central 95\\% of the 9,999 within-session permutations for each fit. "
        f"The pooled omnibus $p$-value across five targets is {st.p_region_omnibus.iloc[0]:.3g}. "
        "The individual tests are exploratory because the mice were selected as examples.",
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

    examples = example_region_tests(d, pm)
    fig = figure(d, shared, st, pm, examples)
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
    examples.to_csv(OUT_V2 / "region_variance_example_tests.csv", index=False)


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
