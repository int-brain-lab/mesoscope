#!/usr/bin/env python3
"""Second round of supplementary controls (comments on the second version).

1. strict: maps restricted to bins with >= MIN_SES sessions from >= MIN_ANIMALS
   animals, isoluminant colour map, opacity = number of animals
   (si_strict_tuning, si_strict_decoding_n10, si_strict_decoding_all,
   si_strict_hex1mm_decoding).
2. hotspot: is the mesoscope stimulus-side hotspot in SSp-ul / lateral MOp
   driven by single animals? (si_stim_hotspot)
3. spatial: are the bin-wise meso-vs-BWM Spearman correlations significant
   once spatial autocorrelation is accounted for (variogram-matched
   surrogate maps, Burt et al. 2020), and how reliable is each modality's
   map across animals (split halves)? (si_spatial_null, si_spatial_null.csv)
4. selection: decoding from the 10 most selective neurons (selected within
   the training folds) or the 10 most active neurons (needs selection.py)
   (si_selection, si_strict_decoding_top).
5. contrast: stimulus side from high-contrast trials only (needs
   selection.py and tuning.py --min-contrast 0.25) (si_high_contrast).

Usage: python controls2.py [strict] [hotspot] [spatial] [selection] [contrast]
"""
import sys

import colorcet as cc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import Normalize
from scipy.stats import spearmanr

from common import ALPHA, HEX_SIZE, MIN_NEURONS, OUT, VARIABLES, bin_neurons, hex_center, hex_unkey
from controls import HEX_LARGE, SUMMARY, VAR_TITLE, load_tuning
from maps import MODS, ROW_NAME, TARGET_LABEL, hexmap, joint_norm, map_grid, outline, style_map
from plotstyle import NATURE_DOUBLE_COL_IN, save_with_caption, style_axes

MIN_SES, MIN_ANIMALS = 3, 2
ISO = cc.cm.isolum
TARGETS = ("stim", "choice", "feedback", "wheel")
SUBJ = pd.read_csv(OUT / "session_subjects.csv").set_index("session").subject
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HOT = dict(ml=(1.8, 3.2), ap=(0.0, 1.2))  # mesoscope stimulus hotspot (bin centres, mm)
SPATIAL = []


def opacity(n_animals):
    return np.minimum(np.asarray(n_animals, float), 4) / 4


def strict_bins(d, value):
    """Per-bin mean of `value` over (session, bin) rows, with session and
    animal counts; only bins passing the inclusion rule."""
    d = d.assign(subject=d.session.map(SUBJ))
    g = d.dropna(subset=[value]).groupby("hex").agg(v=(value, "mean"), ns=("session", "nunique"),
                                                    na=("subject", "nunique"))
    return g[(g.ns >= MIN_SES) & (g.na >= MIN_ANIMALS)]


def strict_tuning_bins(t, v, keys):
    """Pooled fraction tuned per bin over sessions contributing >= MIN_NEURONS
    testable neurons; inclusion rule on those sessions."""
    ok = t[f"p_{v}"].notna().values
    d = pd.DataFrame({"hex": keys[ok], "session": t.session.values[ok],
                      "sig": (t[f"p_{v}"].values[ok] < ALPHA)})
    n = d.groupby(["hex", "session"]).sig.agg(["sum", "size"])
    n = n[n["size"] >= MIN_NEURONS].reset_index()
    n["subject"] = n.session.map(SUBJ)
    g = n.groupby("hex").agg(s=("sum", "sum"), n=("size", "sum"), ns=("session", "nunique"),
                             na=("subject", "nunique"))
    g["v"] = g.s / g.n
    return g[(g.ns >= MIN_SES) & (g.na >= MIN_ANIMALS)]


def strict_intro(size=HEX_SIZE):
    return (f"Strict inclusion: only bins ({2 * size:g}\\,mm vertex to vertex, hemispheres folded) "
            f"with $\\geq${MIN_SES} sessions from $\\geq${MIN_ANIMALS} animals, each session "
            f"contributing $\\geq${MIN_NEURONS} neurons. Isoluminant colour map; opacity encodes "
            f"the number of animals (2: 50\\,\\%, 3: 75\\,\\%, $\\geq$4: 100\\,\\%), so bins backed "
            f"by few animals fade towards white. Mesoscope top row, BWM bottom row.")


def strict_decoding(dec, ver, stem, size=HEX_SIZE, extra=""):
    per = {}
    for m in MODS:
        cols = []
        for t in TARGETS:
            g = strict_bins(dec[m][dec[m].target == t], ver)
            cols.append((g.index.values, g.v.values, opacity(g.na)))
        per[m] = cols
    binary = joint_norm([per[m][i][1] for m in MODS for i in range(3)], lo=0.5)
    norms = [binary] * 3 + [joint_norm([per[m][3][1] for m in MODS], lo=0)]
    who = {"n10": f"{MIN_NEURONS} randomly drawn neurons (mean of 3 draws)",
           "all": "all neurons of the session in the bin",
           "top": f"the {MIN_NEURONS} most selective neurons of the session in the bin, selected "
                  f"on the training trials of each cross-validation fold"}[ver]
    map_grid(per, [TARGET_LABEL[t].capitalize() for t in TARGETS],
             ["balanced accuracy"] * 3 + ["$R^2$"], norms, stem,
             ["Stimulus side.", "Choice.", "Feedback.", "Wheel speed ($R^2$)."], stem,
             intro=strict_intro(size) + f" Decoding from {who}, averaged across sessions; otherwise "
                                        f"as in the main decoding figures. The binary variables "
                                        f"share one colour scale." + extra,
             size=size, cmap=ISO)
    for m in MODS:
        for i, t in enumerate(TARGETS):
            SUMMARY.append(dict(control=f"{stem}", modality=m, target=t,
                                session_bins=len(per[m][i][0]), mean=np.mean(per[m][i][1])))
    return per


def strict(tun, dec, dec_large):
    keys = {m: bin_neurons(tun[m][["x", "y", "z"]].values) for m in MODS}
    per = {}
    for m in MODS:
        per[m] = []
        for v in VARIABLES:
            g = strict_tuning_bins(tun[m], v, keys[m])
            per[m].append((g.index.values, g.v.values, opacity(g.na)))
    norms = [joint_norm([per[m][i][1] for m in MODS], lo=0) for i in range(3)]
    map_grid(per, [VAR_TITLE[v] for v in VARIABLES], ["fraction significant"] * 3, norms,
             "si_strict_tuning",
             ["Fraction of testable neurons tuned to stimulus side ($p<0.05$), pooled over the "
              "included sessions.", "Same for choice.", "Same for feedback."],
             "si_strict_tuning", intro=strict_intro(), cmap=ISO)
    strict_decoding(dec, "n10", "si_strict_decoding_n10")
    strict_decoding(dec, "all", "si_strict_decoding_all",
                    extra=" Mesoscope bins hold many more neurons than BWM bins, so the rows are not "
                          "comparable in level.")
    strict_decoding(dec_large, "n10", "si_strict_hex1mm_decoding", size=HEX_LARGE)


# ---------------------------------------------------------------------------
# 2. stimulus hotspot
# ---------------------------------------------------------------------------
def in_hot(keys):
    x, y = hex_center(*hex_unkey(np.asarray(keys)))
    return (x >= HOT["ml"][0]) & (x <= HOT["ml"][1]) & (y >= HOT["ap"][0]) & (y <= HOT["ap"][1])


def hotspot(dec, tun):
    d = dec["meso"][dec["meso"].target == "stim"].copy()
    d["subject"] = d.session.map(SUBJ)
    d["hot"] = in_hot(d.hex.values)
    animals = sorted(d.subject.unique())
    fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 4.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.45, wspace=0.75)

    # a: map with hotspot outline
    ax = fig.add_subplot(gs[0, 0])
    outline(ax)
    g = d.groupby("hex")["all"].mean()
    norm = Normalize(0.5, np.percentile(g, 99))
    hexmap(ax, g.index, g.values, norm)
    ax.add_patch(plt.Rectangle((HOT["ml"][0], HOT["ap"][0]), np.diff(HOT["ml"])[0],
                               np.diff(HOT["ap"])[0], fill=False, ec="c", lw=0.8, zorder=4))
    style_map(ax, 1, 0, 2)
    ax.set_title("All animals", fontsize=7)

    # b: per animal, number of sessions in / out of the hotspot (map coverage)
    # c: per animal mean score in vs outside the hotspot
    ax = fig.add_subplot(gs[0, 1:3])
    s = d.groupby(["subject", "hot"])["all"].mean().unstack()
    ns = d[d.hot].groupby("subject").session.nunique().reindex(animals).fillna(0).astype(int)
    nall = d.groupby("subject").session.nunique().reindex(animals)
    xs = np.arange(len(animals))
    ax.scatter(xs, s[False].reindex(animals), s=10, color="0.6", label="outside", zorder=3)
    ax.scatter(xs, s[True].reindex(animals), s=12, color="c", label="hotspot", zorder=3)
    for i, a in enumerate(animals):
        ax.plot([i, i], [s[False].get(a, np.nan), s[True].get(a, np.nan)], color="0.8", lw=0.6)
        ax.text(i, 0.505, f"{ns[a]}/{nall[a]}", ha="center", fontsize=4.5, rotation=90, va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(animals, rotation=90, fontsize=5)
    ax.set_ylabel("stim. side bal. acc.\n(all neurons)", fontsize=6)
    ax.set_ylim(0.5, None)
    ax.legend(frameon=False, fontsize=5, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # d: within-session contrast (hotspot - outside), sessions with both
    ax = fig.add_subplot(gs[0, 3])
    w = d.groupby(["session", "hot"])["all"].mean().unstack().dropna()
    w["subject"] = w.index.map(SUBJ)
    w["diff"] = w[True] - w[False]
    for i, a in enumerate(animals):
        v = w.loc[w.subject == a, "diff"]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.2, 0.2, len(v)), v,
                   s=5, color="c" if len(v) else "0.7", lw=0)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(animals, rotation=90, fontsize=5)
    ax.set_ylabel("hotspot $-$ outside\n(within session)", fontsize=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # e-h: leave-one-animal-out maps for the animals with most hotspot sessions
    top = ns.sort_values(ascending=False).index[:4]
    for c, a in enumerate(top):
        ax = fig.add_subplot(gs[1, c])
        outline(ax)
        g = d[d.subject != a].groupby("hex")["all"].mean()
        hexmap(ax, g.index, g.values, norm)
        ax.add_patch(plt.Rectangle((HOT["ml"][0], HOT["ap"][0]), np.diff(HOT["ml"])[0],
                                   np.diff(HOT["ap"])[0], fill=False, ec="c", lw=0.8, zorder=4))
        style_map(ax, 1, c, 2)
        ax.set_title(f"without {a}", fontsize=7)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"), cax=ax.inset_axes([1.08, 0.25, 0.05, 0.6]))
    cb.set_label("bal. acc.", fontsize=5)
    cb.ax.tick_params(labelsize=5)
    letters = "abcdefg"
    for i, ax in enumerate(fig.axes[:3]):
        ax.text(-0.3 if i == 0 else -0.15, 1.08, letters[i], transform=ax.transAxes, fontsize=8,
                fontweight="bold")
    fig.axes[3].text(-0.3, 1.08, "d", transform=fig.axes[3].transAxes, fontsize=8, fontweight="bold")

    # numbers for the caption
    hot_ses = d[d.hot].groupby("subject").session.nunique().sort_values(ascending=False)
    share = hot_ses / hot_ses.sum()
    rest = d[~d.subject.isin(top[:2])]
    r = rest.groupby(["session", "hot"])["all"].mean().unstack().dropna()
    loo = {a: d[d.subject != a].groupby("hot")["all"].mean() for a in top}
    full = d.groupby("hot")["all"].mean()
    t = tun["meso"]
    t = t[t.p_stim.notna()]
    th = in_hot(t.hex.values)
    frac = lambda m: (t.p_stim[m] < ALPHA).mean()  # noqa: E731
    sub = t.session.map(SUBJ)
    nt = ~sub.isin(top[:2]).values
    SUMMARY.extend([dict(control="hotspot", modality="meso", target="stim", measure=k, mean=v) for k, v in {
        "dec_all_hot": full[True], "dec_all_out": full[False],
        **{f"dec_all_hot_without_{a}": loo[a][True] for a in top},
        **{f"dec_all_out_without_{a}": loo[a][False] for a in top},
        "within_session_diff_mean": w["diff"].mean(), "within_session_n": len(w),
        "within_session_diff_without_top2": (r[True] - r[False]).mean(),
        "frac_tuned_hot": frac(th), "frac_tuned_out": frac(~th),
        "frac_tuned_hot_without_top2": frac(th & nt), "frac_tuned_out_without_top2": frac(~th & nt),
    }.items()])
    top_txt = ", ".join(f"{a} ({hot_ses[a]} sessions)" for a in hot_ses.index[:2])
    save_with_caption(
        fig, OUT / "si_stim_hotspot",
        [f"Mesoscope stimulus-side decoding (all neurons of a session in a bin, mean over sessions; "
         f"0.4\\,mm bins) with the hotspot region (cyan; bin centres ML {HOT['ml'][0]:g}--"
         f"{HOT['ml'][1]:g}\\,mm, AP {HOT['ap'][0]:g}--{HOT['ap'][1]:g}\\,mm; mostly SSp-ul and MOp).",
         "Per animal: mean score of the session $\\times$ bin decoders inside (cyan) and outside "
         "(grey) the hotspot. Numbers: sessions with hotspot bins / all sessions of that animal.",
         "Within-session contrast: for sessions with bins both inside and outside the hotspot, "
         "mean score inside minus outside.",
         "Maps leaving out one animal (the four animals with most hotspot sessions; same colour "
         "scale as a)."],
        label="si_stim_hotspot",
        intro=(f"Is the mesoscope stimulus-side hotspot in SSp-ul / lateral MOp driven by single "
               f"animals? {len(hot_ses)} of {len(animals)} mice have sessions there, but "
               f"{share.iloc[:2].sum() * 100:.0f}\\,\\% of the hotspot sessions come from two mice, "
               f"{top_txt}. Mean score inside vs outside: {full[True]:.3f} vs {full[False]:.3f}; "
               f"without {top[0]}: {loo[top[0]][True]:.3f} vs {loo[top[0]][False]:.3f}; without "
               f"{top[1]}: {loo[top[1]][True]:.3f} vs {loo[top[1]][False]:.3f}. Within sessions, "
               f"the hotspot bins score {w['diff'].mean():+.3f} above the other bins on average "
               f"({len(w)} sessions; {(r[True] - r[False]).mean():+.3f} without the two mice). "
               f"Fraction of neurons tuned to stimulus side inside vs outside: {frac(th):.3f} vs "
               f"{frac(~th):.3f}; without the two mice {frac(th & nt):.3f} vs {frac(~th & nt):.3f}."))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. spatial null and split-half reliability
# ---------------------------------------------------------------------------
def _variogram(X, D, h, bw, pairs):
    """Smoothed variogram of each row of X (S, n) at distances h."""
    i, j = pairs
    v = 0.5 * (X[:, i] - X[:, j]) ** 2                          # (S, P)
    d = D[i, j]
    w = torch.exp(-0.5 * ((2.68 * (h[:, None] - d[None])) / bw) ** 2)  # (nh, P)
    return v @ w.T / w.sum(1)                                   # (S, nh)


def smash(xy, x, n_surr=1000, seed=0):
    """Variogram-matched surrogate maps (Burt et al. 2020, BrainSMASH
    'Base' algorithm): random permutations of x, smoothed with an
    exponential kernel over the k nearest bins (k chosen per surrogate to
    best match the empirical variogram), rescaled and with white noise
    added. xy (n, 2) bin centres; x (n,). Returns (n_surr, n) tensor."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    xy = torch.as_tensor(xy, dtype=torch.float64, device=DEV)
    x = torch.as_tensor(x, dtype=torch.float64, device=DEV)
    n = len(x)
    D = torch.cdist(xy, xy)
    iu = torch.triu_indices(n, n, 1, device=DEV)
    d = D[iu[0], iu[1]]
    cut = torch.quantile(d, 0.25)
    keep = d <= cut
    pairs = (iu[0][keep], iu[1][keep])
    h = torch.linspace(float(d[d > 0].min()), float(cut), 25, device=DEV, dtype=torch.float64)
    bw = 3 * (h[1] - h[0])
    target = _variogram(x[None], D, h, bw, pairs)[0]
    order = D.argsort(1)
    ks = np.unique(np.clip((np.linspace(0.1, 0.9, 9) * n).astype(int), 3, n - 1))
    perm = torch.stack([x[torch.randperm(n, generator=g, device=DEV)] for _ in range(n_surr)])
    best = torch.full((n_surr,), np.inf, dtype=torch.float64, device=DEV)
    out = torch.empty_like(perm)
    for k in ks:
        nb = order[:, :k]
        dk = torch.gather(D, 1, nb)
        w = torch.exp(-dk / dk[:, -1:].clamp(min=1e-9))
        W = torch.zeros(n, n, dtype=torch.float64, device=DEV).scatter_(1, nb, w)
        W = W / W.sum(1, keepdim=True)
        sm = perm @ W.T                                         # (S, n)
        gs = _variogram(sm, D, h, bw, pairs)                    # (S, nh)
        A = torch.stack([gs, torch.ones_like(gs)], 2)           # (S, nh, 2)
        coef = torch.linalg.lstsq(A, target.expand(n_surr, -1)[..., None]).solution[..., 0]
        sse = ((A @ coef[..., None])[..., 0] - target).pow(2).sum(1)
        better = sse < best
        noise = torch.randn(n_surr, n, generator=g, device=DEV, dtype=torch.float64)
        cand = coef[:, :1].abs().sqrt() * sm + coef[:, 1:].abs().sqrt() * noise
        out = torch.where(better[:, None], cand, out)
        best = torch.where(better, sse, best)
    return out


def _rank(X):
    return X.argsort(-1).argsort(-1).to(torch.float64)


def spearman_rows(S, y):
    """Spearman r of every row of S (m, n) with y (n,)."""
    a = _rank(S)
    b = _rank(torch.as_tensor(y, dtype=torch.float64, device=DEV)[None])
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))


def spatial_test(name, keys, x_bwm, y_meso, size, n_surr=1000):
    """Observed Spearman r between the BWM and meso maps over common bins,
    naive p and spatial-null p (surrogates of either map; reported: the
    larger of the two)."""
    xy = np.c_[hex_center(*hex_unkey(np.asarray(keys)), size)]
    r, p_naive = spearmanr(x_bwm, y_meso)
    nulls = []
    for src, other, seed in ((x_bwm, y_meso, 0), (y_meso, x_bwm, 1)):
        S = smash(xy, src, n_surr, seed)
        nulls.append(spearman_rows(S, other).cpu().numpy())
    ps = [(np.sum(np.abs(nl) >= abs(r)) + 1) / (len(nl) + 1) for nl in nulls]
    null = np.concatenate(nulls)
    row = dict(comparison=name, n_bins=len(keys), r=r, p_naive=p_naive, p_spatial=max(ps),
               p_spatial_bwm_surr=ps[0], p_spatial_meso_surr=ps[1],
               null_lo=np.percentile(null, 2.5), null_hi=np.percentile(null, 97.5))
    SPATIAL.append(row)
    return row


def split_half(d, value, n_split=200, seed=0):
    """Animal split-half reliability of a per-bin mean map: Spearman r between
    maps from two random halves of the animals over bins present in both,
    Spearman-Brown corrected to the full data. Returns (median, 2.5, 97.5 %)."""
    rng = np.random.default_rng(seed)
    d = d.dropna(subset=[value]).assign(subject=d.session.map(SUBJ))
    animals = d.subject.unique()
    rs = []
    for _ in range(n_split):
        half = rng.permutation(animals)[: len(animals) // 2]
        m = d.subject.isin(half)
        a = d[m].groupby("hex")[value].mean()
        b = d[~m].groupby("hex")[value].mean()
        j = a.index.intersection(b.index)
        if len(j) >= 5:
            r = spearmanr(a[j], b[j])[0]
            rs.append(2 * r / (1 + r) if r > -1 else np.nan)
    rs = np.asarray(rs)
    return np.nanmedian(rs), np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5)


def spatial(tun, dec, dec_large, sel=None):
    comps = []  # (label, per-bin BWM series, per-bin meso series, size, (bwm rows, meso rows, value))
    for ver in (["n10", "all"] + (["top"] if sel is not None else [])):
        for strict_ in (False, True):
            if strict_ and ver == "all":
                continue
            for t in TARGETS:
                src = sel if ver == "top" else dec
                sub = {m: src[m][src[m].target == t] for m in MODS}
                if ver == "top":
                    sub = {m: sub[m][sub[m].version == "top"].rename(columns={"score": "top"})
                           for m in MODS}
                if strict_:
                    g = {m: strict_bins(sub[m], ver).v for m in MODS}
                else:
                    g = {m: sub[m].groupby("hex")[ver].mean().dropna() for m in MODS}
                lab = {"n10": "10 random", "all": "all neurons", "top": "10 most selective"}[ver]
                comps.append((f"{TARGET_LABEL[t]}, {lab}" + (", strict" if strict_ else ""),
                              g["bwm"], g["meso"], HEX_SIZE, (sub["bwm"], sub["meso"], ver), t, ver,
                              strict_))
    for t in TARGETS:
        sub = {m: dec_large[m][dec_large[m].target == t] for m in MODS}
        g = {m: sub[m].groupby("hex")["n10"].mean().dropna() for m in MODS}
        comps.append((f"{TARGET_LABEL[t]}, 10 random, 1 mm", g["bwm"], g["meso"], HEX_LARGE,
                      (sub["bwm"], sub["meso"], "n10"), t, "n10_1mm", False))
    keys = {m: bin_neurons(tun[m][["x", "y", "z"]].values) for m in MODS}
    for v in VARIABLES:
        g = {}
        rows = {}
        for m in MODS:
            ok = tun[m][f"p_{v}"].notna().values
            s = (tun[m][f"p_{v}"][ok] < ALPHA).groupby(keys[m][ok])
            g[m] = s.mean()[s.count() >= MIN_NEURONS]
            per = pd.DataFrame({"hex": keys[m][ok], "session": tun[m].session.values[ok],
                                "sig": (tun[m][f"p_{v}"].values[ok] < ALPHA).astype(float)})
            c = per.groupby(["session", "hex"]).sig.agg(["mean", "size"])
            rows[m] = c[c["size"] >= MIN_NEURONS].reset_index().rename(columns={"mean": "frac"})
        comps.append((f"{TARGET_LABEL[v]}, fraction tuned", g["bwm"], g["meso"], HEX_SIZE,
                      (rows["bwm"], rows["meso"], "frac"), v, "tuning", False))
    rel = []
    for name, gb, gm, size, (db, dm, val), t, ver, strict_ in comps:
        j = gb.index.intersection(gm.index)
        row = spatial_test(name, j.values, gb[j].values, gm[j].values, size)
        if not strict_:
            rb, rm = split_half(db, val), split_half(dm, val)
            row.update(rel_bwm=rb[0], rel_bwm_lo=rb[1], rel_bwm_hi=rb[2],
                       rel_meso=rm[0], rel_meso_lo=rm[1], rel_meso_hi=rm[2])
        row.update(target=t, version=ver, strict=strict_)
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    res = pd.DataFrame(SPATIAL)
    res.to_csv(OUT / "si_spatial_null.csv", index=False)
    spatial_figure(res)
    return res


def spatial_figure(res):
    fig, axs = plt.subplots(1, 2, figsize=(NATURE_DOUBLE_COL_IN, 3.9),
                            gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.9))
    res = res.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(res))
    ax = axs[0]
    ax.hlines(y, res.null_lo, res.null_hi, color="0.75", lw=2.2)
    sig = res.p_spatial < 0.05
    ax.scatter(res.r[~sig], y[~sig], s=9, color="k", facecolor="w", lw=0.6, zorder=3)
    ax.scatter(res.r[sig], y[sig], s=9, color="k", zorder=3)
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(res.comparison, fontsize=4.6)
    ax.set_xlabel("Spearman $r_s$, BWM vs mesoscope map")
    ax.tick_params(axis="x", labelsize=5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(-1.1, 1.01, "a", transform=ax.transAxes, fontsize=8, fontweight="bold")
    ax = axs[1]
    r = res.dropna(subset=["rel_meso"]).reset_index(drop=True)
    yy = np.arange(len(r))
    for off, m, col in ((0.18, "meso", plt.get_cmap("magma")(0.6)), (-0.18, "bwm", "0.3")):
        ax.hlines(yy + off, r[f"rel_{m}_lo"], r[f"rel_{m}_hi"], color=col, lw=0.8)
        ax.scatter(r[f"rel_{m}"], yy + off, s=7, color=col, label=ROW_NAME[m], zorder=3)
    ax.scatter(r.r, yy, s=10, marker="x", color="c", lw=0.8, label="BWM vs meso", zorder=4)
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_yticks(yy)
    ax.set_yticklabels(r.comparison, fontsize=4.6)
    ax.set_xlabel("map reliability across animals")
    ax.tick_params(axis="x", labelsize=5)
    ax.legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(0, 1.13), ncol=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(-1.25, 1.01, "b", transform=ax.transAxes, fontsize=8, fontweight="bold")
    n_sig = int((res.p_spatial < 0.05).sum())
    save_with_caption(
        fig, OUT / "si_spatial_null",
        ["Spearman correlation between the BWM and the mesoscope map across the bins covered by "
         "both (dots; filled: spatial $p<0.05$), with the 95\\,\\% range of the spatial null "
         "(grey bars). Null: 1000 surrogate maps that keep the spatial autocorrelation "
         "(variogram) of the BWM map and 1000 of the mesoscope map (Burt et al. 2020: randomly "
         "permuted values smoothed over the $k$ nearest bins, $k$ and scale chosen to match the "
         "empirical variogram), each correlated with the other modality's map; the reported $p$ "
         "is the larger of the two. Rows: decoding from 10 random, all, or the 10 most selective "
         "neurons, with the original (0.4\\,mm), strict or 1\\,mm bins; fraction tuned.",
         "Reliability of each modality's map across animals: animals split into two random "
         "halves (200 splits), per-bin maps of each half correlated over bins present in both "
         "(Spearman), Spearman--Brown corrected to the full data (median and 95\\,\\% range). "
         "Crosses: the BWM vs mesoscope correlation from a. A cross-modality correlation cannot "
         "be expected to exceed roughly the geometric mean of the two reliabilities."],
        label="si_spatial_null",
        intro=(f"Are the bin-wise correlations between the modalities significant? Neighbouring bins "
               f"are not independent (both maps are spatially smooth), so the naive $p$-values of "
               f"the scatter plots are too small. {n_sig} of {len(res)} comparisons reach spatial "
               f"$p<0.05$ (table in output/si\\_spatial\\_null.csv)."))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. neuron selection
# ---------------------------------------------------------------------------
def selection(sel):
    from scipy.stats import sem
    pool_path = OUT / "selection_meso_pool24.pqt"
    pool = pd.read_parquet(pool_path) if pool_path.exists() else None
    vers = [("n10", "10 random"), ("active", "10 most active"), ("top", "10 most selective")]
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, 4, figsize=(NATURE_DOUBLE_COL_IN, 2.0))
    for c, (ax, t) in enumerate(zip(axs, TARGETS)):
        bars = []  # (x, data, colour, hatch, label)
        for j, (v, lab) in enumerate(vers):
            bars.append((j, sel["meso"], v, cmap(0.2 + 0.3 * j), None, lab))
        if pool is not None:
            for j, (v, lab) in enumerate(vers[1:], 1):
                bars.append((2.2 + j, pool, v, cmap(0.2 + 0.3 * j), "/////", f"{lab[3:]}, pool of 24"))
        x0 = bars[-1][0] + 1.6
        for j, (v, lab) in enumerate(vers):
            bars.append((x0 + j, sel["bwm"], v, cmap(0.2 + 0.3 * j), None, lab))
        seen = set()
        for x, df, v, col, hatch, lab in bars:
            d = df[(df.target == t) & (df.version == v)].score.dropna()
            name = ("pool24_" if df is pool else "") + v
            ax.bar(x, d.mean(), yerr=sem(d), color=col, width=0.85, hatch=hatch, ec="w" if hatch else None,
                   lw=0, error_kw=dict(lw=0.6),
                   label=lab if (c == 0 and (lab, hatch) not in seen) else None)
            seen.add((lab, hatch))
            mod = "bwm" if df is sel["bwm"] else "meso"
            SUMMARY.append(dict(control="selection", modality=mod, target=t, measure=name,
                                session_bins=len(d), mean=d.mean()))
        meso_c = 1 if pool is None else 2.1
        ax.set_xticks([meso_c, x0 + 1])
        ax.set_xticklabels(["Meso", "BWM"])
        chance = 0 if t == "wheel" else 0.5
        ax.set_ylim(chance if t == "wheel" else 0.45, None)
        if t != "wheel":
            ax.axhline(0.5, color="0.6", lw=0.5)
        ax.set_title(TARGET_LABEL[t].capitalize())
        ax.set_ylabel("$R^2$" if t == "wheel" else "balanced accuracy" if c == 0 else "")
        style_axes(ax, categorical_x=True)
        ax.text(-0.35 if c == 0 else -0.25, 1.08, "abcd"[c], transform=ax.transAxes, fontsize=8,
                fontweight="bold")
    fig.legend(frameon=False, fontsize=5.5, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    m = lambda mod, t, v: sel[mod][(sel[mod].target == t) & (sel[mod].version == v)].score.mean()  # noqa: E731
    pm = lambda t, v: pool[(pool.target == t) & (pool.version == v)].score.mean()  # noqa: E731
    pool_txt = "" if pool is None else (
        f" Mesoscope bins hold more neurons per session than BWM bins (median 140 vs 24), which "
        f"gives the selection more to choose from; hatched bars: the same selection from 24 randomly "
        f"drawn neurons of the bin (pool size matched to the BWM median). With the matched pool, "
        f"mesoscope scores are {pm('stim', 'top'):.3f} / {pm('choice', 'top'):.3f} / "
        f"{pm('feedback', 'top'):.3f} and $R^2$ {pm('wheel', 'top'):.3f} (stimulus side / choice / "
        f"feedback / wheel; most selective).")
    save_with_caption(
        fig, OUT / "si_selection",
        ["Stimulus side.", "Choice.", "Feedback.", "Wheel speed ($R^2$)."],
        label="si_selection",
        intro=(f"Neuron selection. Decoding per session $\\times$ bin (0.4\\,mm bins, $\\geq$"
               f"{MIN_NEURONS} neurons) from {MIN_NEURONS} neurons: randomly drawn (mean of 3 draws; "
               f"as in the main figures), the {MIN_NEURONS} most active (highest fraction of non-zero "
               f"window values; labels not used; a stand-in for ROI quality selection), or the "
               f"{MIN_NEURONS} most selective for the decoded variable ($|$AUC$-0.5|$ for the binary "
               f"variables, $|r|$ with wheel speed), selected separately in each outer "
               f"cross-validation fold from its training trials only, so the test trials do not "
               f"influence the selection (checked: shuffled labels give chance scores). Bars: mean "
               f"$\\pm$ s.e.m.\\ over session $\\times$ bin pairs. Selecting the most selective "
               f"neurons raises mesoscope choice decoding from {m('meso', 'choice', 'n10'):.3f} to "
               f"{m('meso', 'choice', 'top'):.3f} (BWM {m('bwm', 'choice', 'n10'):.3f} $\\to$ "
               f"{m('bwm', 'choice', 'top'):.3f}), feedback from {m('meso', 'feedback', 'n10'):.3f} "
               f"to {m('meso', 'feedback', 'top'):.3f} (BWM {m('bwm', 'feedback', 'n10'):.3f} "
               f"$\\to$ {m('bwm', 'feedback', 'top'):.3f}) and wheel $R^2$ from "
               f"{m('meso', 'wheel', 'n10'):.3f} to {m('meso', 'wheel', 'top'):.3f} (BWM "
               f"{m('bwm', 'wheel', 'n10'):.3f} $\\to$ {m('bwm', 'wheel', 'top'):.3f})." + pool_txt))
    plt.close(fig)
    dec_top = {m: sel[m][sel[m].version == "top"].rename(columns={"score": "top"}) for m in MODS}
    strict_decoding(dec_top, "top", "si_strict_decoding_top")


# ---------------------------------------------------------------------------
# 5. high-contrast stimulus side
# ---------------------------------------------------------------------------
def vis_bins(t, br):
    """Bins where most neurons carry a visual-area (VIS*) label."""
    vis = pd.Series(br.id2acronym(t.atlas_id.values, mapping="Beryl")).str.startswith("VIS").values
    f = pd.Series(vis).groupby(t.hex.values).mean()
    return set(f.index[f > 0.5])


def contrast(tun, sel, dec):
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    tc = {m: load_tuning(m, "_c25") for m in MODS}
    per = {}
    cols = [("tun", "Fraction tuned"), ("hc_all", "All neurons"), ("hc_n10", "10 random"),
            ("hc_top", "10 most selective")]
    for m in MODS:
        per[m] = []
        g = strict_tuning_bins(tc[m], "stim", tc[m].hex.values)
        per[m].append((g.index.values, g.v.values, opacity(g.na)))
        for v, _ in cols[1:]:
            d = sel[m][(sel[m].target == "stim") & (sel[m].version == v)].rename(columns={"score": v})
            g = strict_bins(d, v)
            per[m].append((g.index.values, g.v.values, opacity(g.na)))
    norms = [joint_norm([per[m][0][1] for m in MODS], lo=0)]
    dec_norm = joint_norm([per[m][i][1] for m in MODS for i in (1, 2, 3)], lo=0.5)
    norms += [dec_norm] * 3
    # VIS vs rest, fraction tuned, all contrasts vs high contrast
    stats = {}
    for m in MODS:
        for lab, t in (("all", tun[m]), ("hc", tc[m])):
            t = t[t.p_stim.notna()]
            vis = pd.Series(br.id2acronym(t.atlas_id.values, mapping="Beryl")).str.startswith("VIS").values
            stats[m, lab] = ((t.p_stim[vis] < ALPHA).mean(), (t.p_stim[~vis] < ALPHA).mean())
            SUMMARY.append(dict(control="contrast_vis", modality=m, target="stim", measure=f"{lab}_vis",
                                mean=stats[m, lab][0]))
            SUMMARY.append(dict(control="contrast_vis", modality=m, target="stim", measure=f"{lab}_rest",
                                mean=stats[m, lab][1]))
        for v in ("n10", "hc_n10", "top", "hc_top", "hc_all"):
            d = sel[m][(sel[m].target == "stim") & (sel[m].version == v)].score
            SUMMARY.append(dict(control="contrast_dec", modality=m, target="stim", measure=v,
                                session_bins=d.notna().sum(), mean=d.mean()))
    dv = {}
    for m in MODS:
        vb = vis_bins(tun[m], br)
        a = dec[m][dec[m].target == "stim"]
        h = sel[m][(sel[m].target == "stim") & (sel[m].version == "hc_all")]
        for lab, d, col in (("all", a, "all"), ("hc", h, "score")):
            iv = d.hex.isin(vb)
            dv[m, lab] = (d.loc[iv, col].mean(), d.loc[~iv, col].mean())
            SUMMARY.append(dict(control="contrast_vis_dec", modality=m, target="stim", measure=f"{lab}_vis",
                                session_bins=int(iv.sum()), mean=dv[m, lab][0]))
            SUMMARY.append(dict(control="contrast_vis_dec", modality=m, target="stim", measure=f"{lab}_rest",
                                session_bins=int((~iv).sum()), mean=dv[m, lab][1]))
    f = lambda m, l: f"{stats[m, l][0]:.3f} vs {stats[m, l][1]:.3f}"  # noqa: E731
    fd = lambda m, l: f"{dv[m, l][0]:.3f} vs {dv[m, l][1]:.3f}"  # noqa: E731
    map_grid(per, [c[1] for c in cols], ["fraction significant"] + ["balanced accuracy"] * 3, norms,
             "si_high_contrast",
             ["Fraction of neurons tuned to stimulus side, tested on high-contrast trials only.",
              "Stimulus side decoded from all neurons of the session in the bin, high-contrast "
              "trials only.", f"Same from {MIN_NEURONS} randomly drawn neurons (mean of 3 draws).",
              f"Same from the {MIN_NEURONS} most selective neurons (selected within the training "
              f"folds)."],
             "si_high_contrast",
             intro=(strict_intro() + " Stimulus side using only trials with contrast $\\geq$25\\,\\% "
                    "(the two highest contrasts, about 40\\,\\% of trials); 0\\,\\% and low-contrast "
                    "stimuli are unlikely to drive visual cortex. Fraction of neurons tuned in "
                    f"visual areas (VIS*) vs elsewhere: mesoscope {f('meso', 'all')} with all "
                    f"contrasts, {f('meso', 'hc')} with high contrasts; BWM {f('bwm', 'all')} and "
                    f"{f('bwm', 'hc')} (with 40\\,\\% of the trials the test has less power). "
                    f"Decoding from all neurons of a session in visual bins (most neurons VIS*) vs "
                    f"other bins: mesoscope {fd('meso', 'all')} with all contrasts, {fd('meso', 'hc')} "
                    f"with high contrasts; BWM {fd('bwm', 'all')} and {fd('bwm', 'hc')}."),
             cmap=ISO)


def main(parts):
    tun = {m: load_tuning(m) for m in MODS}
    dec = {m: pd.read_parquet(OUT / f"decoding_{m}.pqt") for m in MODS}
    dec_large = {m: pd.read_parquet(OUT / f"decoding_{m}_hex{HEX_LARGE:g}.pqt") for m in MODS}
    have_sel = all((OUT / f"selection_{m}.pqt").exists() for m in MODS)
    sel = {m: pd.read_parquet(OUT / f"selection_{m}.pqt") for m in MODS} if have_sel else None
    if "strict" in parts:
        strict(tun, dec, dec_large)
    if "hotspot" in parts:
        hotspot(dec, tun)
    if "spatial" in parts:
        spatial(tun, dec, dec_large, sel)
    if "selection" in parts and have_sel:
        selection(sel)
    if "contrast" in parts and have_sel:
        contrast(tun, sel, dec)
    path = OUT / "controls2_summary.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    new = pd.DataFrame(SUMMARY)
    if len(old) and len(new):
        old = old[~old.control.isin(new.control.unique())]
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
    print(new.to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["strict", "hotspot", "spatial", "selection", "contrast"])
