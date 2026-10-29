#!/usr/bin/env python3
"""Example rasters of the wheel-decoding features.

For three sessions per modality (the best two and the worst by the
session's median wheel-speed R^2 over bins, all neurons per bin, revised
decoder, from output/variants.pqt): wheel speed in the 6 wheel bins
(0.2 s, -0.2 to 1.0 s around first movement) of consecutive trials,
concatenated, above the z-scored activity of the neurons most positively
correlated with wheel speed (sorted by r); next to it, for the same neurons,
the trial-averaged activity in the 6 bins and r against neuron rank.

Usage: python rasters.py   -> output/fig1_rasters.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wd_common import (BIN_CENTRES, COL, DM, MODS, NAME, NATURE_DOUBLE_COL_IN, OUT, load_session,
                       panel_label, save_with_caption, session_files, style_axes, wheel_corr)

N_TRIALS = 30
N_SHOWN = 60
RANKS = (0, 1, -1)                     # best, second best, worst
RANK_NAME = {0: "best", 1: "2nd best", -1: "worst"}


def pick_sessions(m, min_neurons=30):
    """Best two and worst sessions (RANKS) by the session median wheel R^2 over 0.4 mm
    bins (all neurons; revised decoder with lags and leave-one-trial-out
    penalty, corrected wheel targets; variants.py)."""
    d = pd.read_parquet(OUT / "variants.pqt")
    w = d[(d.modality == m) & (d.variant == "lag") & d.session.isin(session_files(m))]
    g = w.groupby("session").agg(r2=("all", "median"), n=("n_neurons", "sum"))
    g = g[g.n >= min_neurons].sort_values("r2")
    g = g.sort_values("r2", ascending=False)
    return [(g.index[k], g.r2.iloc[k], len(g)) for k in RANKS]


def zscore_rows(A):
    sd = A.std(1, keepdims=True)
    return (A - A.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1)


def session_row(fig, gs, m, sid, r2, q, letter, rx=None):
    s = load_session(session_files(m)[sid])
    X, y = s["X_wheel"].astype(np.float64), s["wheel_y"]
    keep = X.reshape(len(X), -1).std(1) > 0
    X = X[keep]
    n, T, nb = X.shape
    # neurons ranked on the trials NOT shown (no selection bias in the shown trials)
    r = wheel_corr(X[:, N_TRIALS:], y[N_TRIALS:])
    order = np.argsort(-r)
    Z = zscore_rows(X.reshape(n, -1))[order]
    ns = min(n, N_SHOWN)
    t = np.arange(N_TRIALS * nb)
    pop = Z[:ns, :t.size].mean(0)
    r_pop = np.corrcoef(pop, y[:N_TRIALS].reshape(-1))[0, 1]
    sub = gs.subgridspec(2, 3, height_ratios=(1, 3), width_ratios=(7, 1.1, 1.1), hspace=0.06, wspace=0.12)
    aw = fig.add_subplot(sub[0, 0])
    aw.plot(t, y.reshape(-1)[:t.size], color="k", lw=0.6)
    yw = y.reshape(-1)[:t.size]
    aw.plot(t, yw.mean() + (pop - pop.mean()) / pop.std() * yw.std(), color=COL[m], lw=0.6, alpha=0.9)
    aw.set_xlim(-0.5, t.size - 0.5)
    aw.set_xticks([])
    aw.set_ylabel("rad/s", fontsize=5)
    style_axes(aw, max_yticks=2)
    aw.set_yticks([np.ceil(yw.max())])
    aw.spines["bottom"].set_visible(False)
    aw.set_title(f"{NAME[m]}, {RANK_NAME[q]} session {sid[:8]}: median $R^2$ {r2:.2f}, {n} neurons, "
                 f"{T} trials; top-{ns} mean vs wheel $r$ = {r_pop:.2f}", loc="left", fontsize=5.5)
    panel_label(aw, letter, x=-0.06, y=1.1)
    ar = fig.add_subplot(sub[1, 0], sharex=aw)
    S = Z[:ns, :t.size]
    ar.imshow(S, aspect="auto", cmap="Greys", vmin=0, vmax=max(np.quantile(S, 0.98), 1e-6),
              interpolation="nearest", extent=(-0.5, t.size - 0.5, ns, 0))
    for b in range(nb, t.size, nb):
        ar.axvline(b - 0.5, color="tab:orange", lw=0.3, alpha=0.35)
    ar.set_ylabel(f"top {ns} by $r$", fontsize=5)
    ar.set_yticks([ns])
    ar.set_xlabel(f"wheel bin (trials 1-{N_TRIALS}, 6 bins each)" if letter in "cf" else "", fontsize=5)
    style_axes(ar, max_yticks=2)
    # trial-averaged activity of the same top neurons, sorted by r
    at = fig.add_subplot(sub[1, 1], sharey=ar)
    P = zscore_rows(X.mean(1))[order[:ns]]
    at.imshow(P, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest",
              extent=(-0.5, nb - 0.5, ns, 0))
    at.set_xticks([0, nb - 1], [f"{BIN_CENTRES[0]:.1f}", f"{BIN_CENTRES[-1]:.1f}"], fontsize=5)
    at.tick_params(labelleft=False, left=False)
    at.set_xlabel("s from move" if letter in "cf" else "", fontsize=5)
    atw = fig.add_subplot(sub[0, 1], sharex=at)
    atw.plot(np.arange(nb), y.mean(0), color="k", lw=0.8)
    atw.axis("off")
    rr = fig.add_subplot(sub[1, 2], sharey=ar, sharex=rx)  # r axis shared across rows
    rr.plot(r[order[:ns]], np.arange(ns) + 0.5, color=COL[m], lw=0.8)
    rr.axvline(0, color="0.6", lw=0.4)
    rr.tick_params(labelsize=5, labelleft=False)
    rr.set_xlabel("$r$ with wheel" if letter in "cf" else "", fontsize=5)
    style_axes(rr, max_xticks=2, max_yticks=2)
    return dict(m=m, sid=sid, r2=r2, n=n, r95=np.nanquantile(r, 0.95), rmax=np.nanmax(r),
                frac_pos=np.nanmean(r > 0.1), r_pop=r_pop), rr


def main():
    fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 9.0))
    gs = fig.add_gridspec(6, 1, hspace=0.75)
    info = []
    n_ses_of = {}
    k = 0
    rx = None
    for m in MODS:
        for (sid, r2, n_ses), q in zip(pick_sessions(m), RANKS):
            row, rx = session_row(fig, gs[k], m, sid, r2, q, "abcdef"[k], rx)
            info.append(row)
            n_ses_of[m] = n_ses
            k += 1
    I = pd.DataFrame(info)
    I.to_csv(OUT / "fig1_sessions.csv", index=False)
    d = lambda i: I.iloc[i]  # noqa: E731
    caps = []
    for i in range(6):
        x = d(i)
        caps.append(f"{NAME[x.m]} session {x.sid[:8]} ({RANK_NAME[RANKS[i % 3]]} of {n_ses_of[x.m]} sessions by the "
                    f"session median wheel-speed $R^2$): 95th percentile of $r$ over neurons "
                    f"{x.r95:.2f}, maximum {x.rmax:.2f}, {x.frac_pos:.0%} of neurons with $r>0.1$; mean of the "
                    f"top {min(x.n, N_SHOWN)} neurons vs wheel speed in the shown trials $r$ = {x.r_pop:.2f}."
                    .replace("%", "\\%"))
    intro = (f"Neural correlates of wheel speed. Each row: wheel speed (top; rad/s) in the 6 wheel bins "
             f"(0.2\\,s, $-0.2$ to 1\\,s around first movement) of the first {N_TRIALS} trials, "
             f"concatenated (orange lines: trial boundaries; bins are not contiguous across trials; coloured: "
             f"mean z-scored activity of the neurons below, rescaled), above the z-scored activity of the "
             f"{N_SHOWN} neurons most positively correlated with wheel speed (sorted by $r$, highest on top). "
             f"Neurons are ranked by $r$ over all bins of the remaining trials, so the shown trials are "
             f"held out from the selection. Middle: trial-averaged "
             f"activity of the same {N_SHOWN} neurons per bin (z-scored per neuron, same order), with the "
             f"mean wheel speed above. Right: their $r$ with wheel speed over the remaining trials, same rows. Sessions at the "
             f"best two and the worst of each modality by the session median $R^2$ over bins (all neurons per 0.4\\,mm bin, revised "
             f"decoder with lagged features; sessions with $\\geq$30 neurons).")
    save_with_caption(fig, OUT / "fig1_rasters", caps, "wd2_rasters", intro=intro)


if __name__ == "__main__":
    main()
