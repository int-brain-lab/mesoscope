#!/usr/bin/env python3
"""Why is wheel speed decoded so much worse from mesoscope than from BWM?

Rasters of the wheel-bin features (6 bins of 0.2 s from -0.2 to 1.0 s
around first movement, concatenated over trials) with neurons sorted by
their correlation with wheel speed, for one typical session per modality,
plus per-session statistics over all sessions:
- per-neuron Pearson r with wheel speed over all bins, split into a
  within-trial part (trial means removed) and an across-trial part (trial
  means only); 95th percentile per session, i.e. the best neurons;
- the trial-averaged time course of the 10 % most correlated neurons;
- lagged correlations: neural bin b + L vs wheel bin b (wheel bins 1-4).

Usage: python wheel_raster.py   (writes output/wheel_stats.pqt,
output/wheel_profiles.pqt and output/si_wheel_raster.pdf)
"""
import warnings
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import OUT, WHEEL_EDGES
from gpu_decode import DEVICE
from maps import MODS, ROW_NAME
from plotstyle import NATURE_DOUBLE_COL_IN, panel_label, save_with_caption, style_axes
from sessions import load_session, session_files

N_TRIALS_SHOWN = 40
N_SHOWN = 80        # most wheel-correlated neurons shown in the rasters
TOP_FRAC = 0.1
LAGS = (-1, 0, 1)
COL = {"meso": "tab:green", "bwm": "tab:purple"}


def corr(A, b):
    """Pearson r of every row of A (n, M) with b (M,)."""
    A = A - A.mean(1, keepdim=True)
    b = b - b.mean()
    return (A @ b) / (A.norm(dim=1) * b.norm()).clamp(min=1e-12)


def session_stats(s):
    X = torch.as_tensor(s["X_wheel"].astype(np.float32), device=DEVICE).double()
    y = torch.as_tensor(s["wheel_y"], device=DEVICE).double()
    X = X[X.reshape(len(X), -1).std(1) > 0]
    n, T, nb = X.shape
    if n < 10:
        return None
    r = corr(X.reshape(n, -1), y.reshape(-1))
    rw = corr((X - X.mean(2, keepdim=True)).reshape(n, -1), (y - y.mean(1, keepdim=True)).reshape(-1))
    ra = corr(X.mean(2), y.mean(1))
    lag = {L: corr(X[:, :, 1 + L:5 + L].reshape(n, -1), y[:, 1:5].reshape(-1)) for L in LAGS}
    top = torch.argsort(-r)[:max(5, int(n * TOP_FRAC))]
    q = lambda v: float(torch.quantile(v.float(), 0.95))  # noqa: E731
    stats = dict(n=n, n_trials=T, zero=float((X == 0).double().mean()), r_95=q(r), rw_95=q(rw),
                 ra_95=q(ra), **{f"lag{L}_95": q(v) for L, v in lag.items()},
                 peak=int(torch.argmax(X[top].mean((0, 1)))),
                 wheel_within=float(((y - y.mean(1, keepdim=True)) ** 2).mean() / y.var(unbiased=False)))
    return stats, X[top].mean((0, 1)).cpu().numpy(), y.mean(0).cpu().numpy()


def compute():
    warnings.filterwarnings("ignore")
    rows, prof = [], []
    for m in MODS:
        groups = list(session_files(m).items())
        with ThreadPoolExecutor(2) as ex:
            for (sid, _), s in zip(groups, ex.map(lambda g: load_session(g[1]), groups)):
                out = session_stats(s)
                if out is None:
                    continue
                st, p, w = out
                rows.append(dict(modality=m, session=sid, **st))
                prof += [dict(modality=m, session=sid, bin=b, neural=p[b], wheel=w[b]) for b in range(len(p))]
        print(m, sum(r["modality"] == m for r in rows), flush=True)
    stats, prof = pd.DataFrame(rows), pd.DataFrame(prof)
    stats.to_parquet(OUT / "wheel_stats.pqt")
    prof.to_parquet(OUT / "wheel_profiles.pqt")
    return stats, prof


def typical_session(m, stats, min_neurons=30):
    """Session whose median wheel R^2 (all neurons, over bins) is closest to the
    median over sessions, among sessions with >= min_neurons neurons."""
    d = pd.read_parquet(OUT / f"decoding_{m}.pqt")
    r2 = d[d.target == "wheel"].groupby("session")["all"].median()
    ok = stats[(stats.modality == m) & (stats.n >= min_neurons)].session
    r2 = r2[r2.index.isin(ok)]
    return (r2 - r2.median()).abs().idxmin(), r2.median()


def raster(fig, gs, m, sid):
    s = load_session(session_files(m)[sid])
    X = s["X_wheel"].astype(np.float64)
    X = X[X.reshape(len(X), -1).std(1) > 0]
    y = s["wheel_y"]
    n, T, nb = X.shape
    Xf, yf = X.reshape(n, -1), y.reshape(-1)
    Z = (Xf - Xf.mean(1, keepdims=True)) / Xf.std(1, keepdims=True)
    r = Z @ ((yf - yf.mean()) / yf.std()) / len(yf)
    order = np.argsort(-r)
    Z, r = Z[order], r[order]
    ns = min(n, N_SHOWN)
    Zs = Z[:ns, :N_TRIALS_SHOWN * nb]
    sub = gs.subgridspec(2, 2, height_ratios=(1, 3.2), width_ratios=(8, 1), hspace=0.08, wspace=0.08)
    aw = fig.add_subplot(sub[0, 0])
    t = np.arange(N_TRIALS_SHOWN * nb)
    aw.plot(t, yf[:t.size], color="k", lw=0.6)
    aw.set_xlim(-0.5, t.size - 0.5)
    aw.set_ylabel("wheel (rad/s)", fontsize=6)
    aw.set_xticks([])
    style_axes(aw, max_yticks=2)
    aw.spines["bottom"].set_visible(False)
    ar = fig.add_subplot(sub[1, 0], sharex=aw)
    ar.imshow(Zs, aspect="auto", cmap="Greys", vmin=0, vmax=np.quantile(Zs, 0.98),
              interpolation="nearest", extent=(-0.5, t.size - 0.5, ns, 0))
    for b in range(nb, t.size, nb):
        ar.axvline(b - 0.5, color="tab:orange", lw=0.3, alpha=0.35)
    ar.set_xlabel(f"wheel bin (trials 1-{N_TRIALS_SHOWN}, {nb} bins each)")
    ar.set_ylabel(f"top {ns} neurons\n(sorted by $r$)", fontsize=6)
    style_axes(ar)
    rr = fig.add_subplot(sub[1, 1])
    rr.plot(r, 100 * (np.arange(n) + 0.5) / n, color=COL[m], lw=0.8)
    rr.axvline(0, color="0.6", lw=0.4)
    rr.axhline(100 * ns / n, color="tab:orange", lw=0.6, ls=":")
    rr.set_ylim(100, 0)
    rr.set_xlabel("$r$ with wheel")
    rr.set_ylabel("rank (% of all)", fontsize=6)
    rr.yaxis.set_label_position("right")
    rr.yaxis.tick_right()
    style_axes(rr, max_xticks=2, max_yticks=2)
    rr.spines["left"].set_visible(False)
    rr.spines["right"].set_visible(True)
    aw.set_title(f"{ROW_NAME[m]}: {n} neurons, {T} trials", loc="left")
    return aw, n, r


def figure(stats, prof):
    sids = {m: typical_session(m, stats) for m in MODS}
    fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 6.6))
    gs = fig.add_gridspec(3, 3, height_ratios=(1.25, 1.25, 1), hspace=0.55, wspace=0.45)
    info = {}
    for i, m in enumerate(MODS):
        aw, n, r = raster(fig, gs[i, :], m, sids[m][0])
        panel_label(aw, "ab"[i], x=-0.09, y=1.15)
        info[m] = (n, np.quantile(r, 0.95))

    # c: trial-averaged time course of the most correlated neurons vs wheel speed
    ax = fig.add_subplot(gs[2, 0])
    centres = (WHEEL_EDGES[:-1] + WHEEL_EDGES[1:]) / 2
    norm = lambda v: (v - v.min()) / (v.max() - v.min())  # noqa: E731
    for m in MODS:
        p = prof[prof.modality == m]
        P = np.stack([norm(g.neural.to_numpy()) for _, g in p.groupby("session")])
        ax.plot(centres, np.median(P, 0), color=COL[m], label=ROW_NAME[m])
        ax.fill_between(centres, *np.quantile(P, [0.25, 0.75], 0), color=COL[m], alpha=0.2, lw=0)
    W = np.stack([norm(g.wheel.to_numpy()) for _, g in prof.groupby("session")])
    ax.plot(centres, np.median(W, 0), color="k", ls="--", label="wheel speed")
    ax.set_xlabel("time from first movement (s)")
    ax.set_ylabel("normalised mean")
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    style_axes(ax)
    panel_label(ax, "c")

    # d: best-neuron correlations, total / within-trial / across-trial
    ax = fig.add_subplot(gs[2, 1])
    cols = [("r_95", "all"), ("rw_95", "within\ntrial"), ("ra_95", "across\ntrials")]
    for j, (c, _) in enumerate(cols):
        for k, m in enumerate(MODS):
            v = stats.loc[stats.modality == m, c]
            ax.boxplot(v, positions=[j + (k - 0.5) * 0.35], widths=0.3, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=COL[m], alpha=0.5, lw=0.5),
                       medianprops=dict(color="k", lw=0.8), whiskerprops=dict(lw=0.5),
                       capprops=dict(lw=0.5))
    ax.set_xticks(range(len(cols)), [l for _, l in cols])
    ax.set_ylabel("95th percentile of $r$")
    style_axes(ax, categorical_x=True)
    panel_label(ax, "d")

    # e: lagged correlations
    ax = fig.add_subplot(gs[2, 2])
    for m in MODS:
        v = stats[stats.modality == m][[f"lag{L}_95" for L in LAGS]].to_numpy()
        ax.errorbar(np.array(LAGS) * 0.2, np.median(v, 0),
                    yerr=np.abs(np.quantile(v, [0.25, 0.75], 0) - np.median(v, 0)),
                    color=COL[m], marker="o", ms=3, lw=0.8, capsize=1.5)
    ax.set_xticks(np.array(LAGS) * 0.2)
    ax.set_xlabel("neural lag (s)")
    ax.set_ylabel("95th percentile of $r$")
    style_axes(ax, categorical_x=True)
    panel_label(ax, "e")

    med = stats.groupby("modality").median(numeric_only=True)
    pk = stats.groupby("modality").peak.apply(lambda v: (v == 2).mean())
    lab = {m: f"{ROW_NAME[m].lower() if m == 'meso' else ROW_NAME[m]}" for m in MODS}
    f2 = lambda m, c: f"{med.loc[m, c]:.2f}"  # noqa: E731
    captions = [
        f"Typical {lab['meso']} session ({sids['meso'][0][:8]}, median wheel $R^2$ over bins "
        f"{sids['meso'][1]:.2f}, the median over sessions): wheel speed (top) and z-scored activity "
        f"of the {N_SHOWN} neurons (of {info['meso'][0]}) most correlated with wheel speed in the 6 wheel bins (0.2\\,s, $-0.2$ to 1\\,s around first "
        f"movement) of the first {N_TRIALS_SHOWN} trials, concatenated (orange lines: trial "
        f"boundaries; the bins are not contiguous in time across trials). Neurons sorted by their "
        f"correlation $r$ with wheel speed over all bins of all trials. Right: $r$ of all neurons "
        f"against their rank (dotted: the {N_SHOWN} shown).",
        f"Same for a typical BWM session ({sids['bwm'][0][:8]}, median $R^2$ {sids['bwm'][1]:.2f}; "
        f"sessions with $\\geq$30 neurons).",
        f"Mean activity of the {TOP_FRAC:.0%} most wheel-correlated neurons of each session "
        f"(normalised to [0, 1]) and wheel speed, per wheel bin; median and interquartile range "
        f"over sessions. The mesoscope peaks one bin (0.2\\,s) later than wheel speed in "
        f"{pk['meso']:.0%} of sessions (BWM {pk['bwm']:.0%}) and decays more slowly.".replace("%", "\\%"),
        f"Correlation of the best neurons of a session (95th percentile of $r$ over neurons) with "
        f"wheel speed over all bins, within trials (trial means removed) and across trials (trial "
        f"means only); one value per session. Medians: all {f2('meso', 'r_95')} (mesoscope) vs "
        f"{f2('bwm', 'r_95')} (BWM), within {f2('meso', 'rw_95')} vs {f2('bwm', 'rw_95')}, across "
        f"{f2('meso', 'ra_95')} vs {f2('bwm', 'ra_95')}.",
        f"Same (all bins) with neural activity of bin $b+L$ against wheel speed of bin $b$ (wheel "
        f"bins 0 to 0.8\\,s); median and interquartile range over sessions. Mesoscope "
        f"{f2('meso', 'lag0_95')} at lag 0 vs {f2('meso', 'lag1_95')} at +0.2\\,s; BWM "
        f"{f2('bwm', 'lag0_95')} vs {f2('bwm', 'lag1_95')}.",
    ]
    intro = ("Why is wheel speed decoded worse from the mesoscope? Most of the wheel-speed variance "
             f"is the time course within a trial ({med.loc['meso', 'wheel_within']:.0%} in mesoscope, "
             f"{med.loc['bwm', 'wheel_within']:.0%} in BWM sessions; medians)".replace("%", "\\%") +
             ", and the mesoscope follows it poorly: its best "
             "neurons track trial-to-trial differences in speed almost as well as BWM neurons, but "
             "their activity lags and smears the movement.")
    save_with_caption(fig, OUT / "si_wheel_raster", captions, "si_wheel_raster", intro=intro)


if __name__ == "__main__":
    have = (OUT / "wheel_stats.pqt").exists() and (OUT / "wheel_profiles.pqt").exists()
    st, pr = ((pd.read_parquet(OUT / "wheel_stats.pqt"), pd.read_parquet(OUT / "wheel_profiles.pqt"))
              if have else compute())
    figure(st, pr)
