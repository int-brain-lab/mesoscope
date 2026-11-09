"""Example tracked neurons: footprints, rasters, PSTHs and the metrics.

Examples are chosen from output/trends.parquet (trends.py): neurons of SP058
tracked (spatial QC) across many sessions and stimulus-selective overall
(mean contra-ipsi d' in the 0-0.3 s stimulus window > 0.2):
  stable: strong selectivity, no trend over days (p_lose, p_gain > 0.3)
  losing: significant decline of the sign-aligned d' with recording day
          (session-shuffle permutation test, p < 0.025)

Rasters show up to N_COLS sessions spread over each neuron's tracked
sessions; the d' panel shows all of them (even/odd halves, the fitted trend).

fetch(): per (session, FOV), the deconvolved traces of the example ROIs (with
per-ROI times), trials, the FOV mean image and the example ROI masks, cached to
output/examples/<subject>_<session>_FOV<fov>.npz; signal files downloaded here
are deleted afterwards.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
EX = OUT / "examples"
sys.path.insert(0, str(HERE.parent / "dorsal_maps"))
SUBJECT = "SP058"
STABLE = ["ded75c53", "9a5f5dbf", "850e9eda"]   # last: no trend, large day-to-day swings
LOSING = ["b81244f9", "0ce07373"]
N_COLS = 8
T_GRID = np.arange(-0.5, 1.0 + 1e-9, 0.05)
CROP = 14   # half-width of the footprint crop (px)
WIN = ["stim", "pre-move", "feedback"] + [f"m{t:+.1f}" for t in np.arange(-0.2, 1.0, 0.2)]
TYPES = ["rew. L", "rew. R", "unrew."]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 7,
                     "pdf.fonttype": 42})


def _table(uids):
    """One row per (session, example uid) with the session-file row, ROI index,
    pixel position, QC values and full-session d' / p."""
    n = pd.read_parquet(OUT / "neuron_sessions.parquet")
    n = n[n.subject == SUBJECT]
    full = {u: n.uid[n.uid.str.startswith(u)].iloc[0] for u in uids}
    q = pd.read_parquet(OUT / "qc_positions.parquet")
    q = q[q.subject == SUBJECT][["session", "row", "roi", "y", "x", "dev_px", "nn_px"]]
    t = n[n.uid.isin(full.values())].merge(q, on=["session", "row"])
    t["short"] = t.uid.str[:8]
    t["day"] = (pd.to_datetime(t.session.str[:10]) - pd.Timestamp(n.session.min()[:10])).dt.days
    return t


def _shown_sessions(tu):
    s = sorted(tu.session)
    return [s[i] for i in np.unique(np.linspace(0, len(s) - 1, min(N_COLS, len(s))).round().astype(int))]


def _cache(session, fov):
    return EX / f"{SUBJECT}_{session}_FOV{fov:02d}.npz"


def fetch(uids=STABLE + LOSING):
    from extract_meso import frame_times_for, task_collection
    from one.api import ONE
    import sparse  # noqa: F401  (ONE needs it to read .sparse_npz)
    one = ONE()
    EX.mkdir(parents=True, exist_ok=True)
    t = _table(uids)
    need = {}
    for u in uids:
        tu = t[t.short == u]
        for s in _shown_sessions(tu):
            r = tu[tu.session == s].iloc[0]
            need.setdefault((s, int(r.fov)), []).append(r)
    for (s, fov), rs in sorted(need.items()):
        out = _cache(s, fov)
        want = [r.short for r in rs]
        if out.exists() and set(want) <= set(np.load(out, allow_pickle=True)["uid"]):
            continue
        eid = one.path2eid(f"{SUBJECT}/{s[:10]}/{s[11:]}")
        col = f"alf/FOV_{fov:02d}"
        roi = np.array([int(r.roi) for r in rs])
        sess = Path(one.eid2path(eid))
        before = set(sess.rglob("mpci.ROIActivity*.npy"))
        try:
            path = one.load_dataset(eid, "mpci.ROIActivityDeconvolved.npy", collection=col,
                                    download_only=True)
            sig = np.load(path, mmap_mode="r")
            tr = np.asarray(sig[:, roi], np.float32).T
            ft = frame_times_for(one, eid, col, sig.shape[0])
            sp = one.load_dataset(eid, "mpciROIs.stackPos.npy", collection=col)
            ts = one.load_dataset(eid, "mpciStack.timeshift.npy", collection=col)
            img = one.load_dataset(eid, "mpciMeanImage.images.npy", collection=col)
            masks = one.load_dataset(eid, "mpciROIs.masks.sparse_npz", collection=col)
            trials = one.load_object(eid, "trials", collection=task_collection(one, eid))
            np.savez_compressed(out, uid=np.array(want), roi=roi, traces=tr, frame_times=ft,
                                offsets=ts[sp[roi, len(ts.shape)]], image=np.asarray(img, np.float32),
                                masks=np.stack([np.asarray(masks[r].todense()) > 0 for r in roi]),
                                **{k: np.asarray(trials[k], float) for k in
                                   ("stimOn_times", "contrastLeft", "contrastRight", "choice",
                                    "feedbackType", "firstMovement_times")})
            print(s, f"FOV_{fov:02d}", want, flush=True)
        finally:
            for f in set(sess.rglob("mpci.ROIActivity*.npy")) - before:
                f.unlink(missing_ok=True)


def _aligned(z, i, contra_left):
    """(trials, T_GRID) activity of ROI i around stimulus onset, the contra
    flag and contrast of each trial (trials with a choice)."""
    t0 = z["stimOn_times"]
    ok = np.isfinite(t0) & (z["choice"] != 0)
    left = np.isfinite(z["contrastLeft"])
    contrast = np.where(left, np.nan_to_num(z["contrastLeft"]), np.nan_to_num(z["contrastRight"]))[ok]
    A = np.interp(t0[ok, None] + T_GRID[None], z["frame_times"] + z["offsets"][i], z["traces"][i])
    return A, left[ok] == contra_left, contrast


def _session_file(s):
    return OUT / "sessions" / SUBJECT / f"{s}.npz"


def _profile(s, row):
    """Even/odd 27-value profile of one row of a session file (as stability.py)."""
    z = np.load(_session_file(s))
    X, ch, rw = z["X"][row], z["choice"], z["reward"]
    odd = np.arange(len(ch)) % 2 == 1
    types = [(ch == -1) & (rw == 1), (ch == 1) & (rw == 1), rw == 0]
    return np.stack([np.concatenate([X[t & h].mean(0) for t in types]) for h in (~odd, odd)])


def _dprime_halves(s, row, contra_left):
    z = np.load(_session_file(s))
    x = z["X"][row, :, 0]
    contra = (z["stim_left"] == 1) == contra_left
    odd = np.arange(len(x)) % 2 == 1
    return [(x[h & contra].mean() - x[h & ~contra].mean())
            / (np.sqrt(0.5 * (x[h & contra].var() + x[h & ~contra].var())) + 1e-6) for h in (~odd, odd)]


def _contra_left(tu):
    """tu: rows of one neuron indexed by session."""
    s = tu.index[0]
    return np.load(_session_file(s))["xyz"][int(tu.loc[s, "row"]), 0] > 0   # right hemisphere -> contra = left


def raster_figure(uids, title, fname):
    t = _table(uids)
    trends = pd.read_parquet(OUT / "trends.parquet").set_index("uid")
    fig = plt.figure(figsize=(1.1 * N_COLS + 2.6, 2.5 * len(uids)))
    gs = fig.add_gridspec(3 * len(uids), N_COLS + 2, height_ratios=[1, 1.6, 0.9] * len(uids),
                          width_ratios=[1] * N_COLS + [0.25, 2.4], hspace=0.45, wspace=0.25)
    for k, u in enumerate(uids):
        tu = t[t.short == u].sort_values("day").set_index("session")
        cl = _contra_left(tu)
        shown = _shown_sessions(tu.reset_index())
        data = {}
        for s in shown:
            f = _cache(s, int(tu.loc[s, "fov"]))
            if f.exists():
                z = np.load(f, allow_pickle=True)
                if u in list(z["uid"]):
                    data[s] = (z, list(z["uid"]).index(u))
        vmax = np.percentile(np.concatenate([_aligned(z, i, cl)[0].ravel() for z, i in data.values()]), 99)
        pmax, ax_p = 0, []
        for j, s in enumerate(shown):
            a_img, a_r, a_p = (fig.add_subplot(gs[3 * k + m, j]) for m in range(3))
            ax_p.append(a_p)
            for a in (a_img, a_r):
                a.set_xticks([]), a.set_yticks([])
            a_img.set_title(f"day {tu.loc[s, 'day']}", fontsize=6, pad=2)
            if s not in data:
                for a in (a_img, a_r, a_p):
                    a.axis("off")
                continue
            z, i = data[s]
            r = tu.loc[s]
            y0, x0 = int(r.y), int(r.x)
            sl = (slice(max(y0 - CROP, 0), y0 + CROP), slice(max(x0 - CROP, 0), x0 + CROP))
            crop = z["image"][sl]
            a_img.imshow(crop, cmap="gray", vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99.5),
                         aspect=2.52 / 1.68)
            a_img.contour(z["masks"][i][sl], levels=[0.5], colors="C1", linewidths=0.6)
            A, contra, contrast = _aligned(z, i, cl)
            order = np.lexsort((-contrast, ~contra))
            a_r.imshow(A[order], aspect="auto", cmap="Greys", vmin=0, vmax=vmax,
                       extent=[T_GRID[0], T_GRID[-1], len(A), 0], interpolation="nearest")
            a_r.axhline(contra.sum(), color="C3", lw=0.6)
            a_r.axvline(0, color="C0", lw=0.5, ls=":")
            for m, c in ((contra, "C3"), (~contra, "0.5")):
                mu, se = A[m].mean(0), A[m].std(0) / np.sqrt(m.sum())
                a_p.plot(T_GRID, mu, color=c, lw=0.8)
                a_p.fill_between(T_GRID, mu - se, mu + se, color=c, alpha=0.25, lw=0)
                pmax = max(pmax, (mu + se).max())
            a_p.axvline(0, color="C0", lw=0.5, ls=":")
            a_p.set_title(f"d'={r.d_stim:+.2f}", fontsize=5.5, pad=1)
            a_p.set_xticks([0, 1] if k == len(uids) - 1 else [])
            if j == 0:
                a_r.set_ylabel(f"{u}\ntrials\ncontra | ipsi", fontsize=6)
                a_p.yaxis.set_major_locator(MaxNLocator(2))
            else:
                a_p.set_yticks([])
        for a in ax_p:
            if a.axison:
                a.set_ylim(0, pmax * 1.05)
        if k == len(uids) - 1:
            ax_p[0].set_xlabel("time from stimulus (s)")
        # d' over all tracked sessions: full session, even and odd halves, trend
        a = fig.add_subplot(gs[3 * k:3 * k + 3, N_COLS + 1])
        ev, od = zip(*[_dprime_halves(s, int(tu.loc[s, "row"]), cl) for s in tu.index])
        days = tu.day.values
        a.plot(days, tu.d_stim, "k-o", ms=2.5, lw=0.8, label="all trials")
        a.plot(days - 0.3, ev, "v", color="C0", ms=2.2, label="even trials")
        a.plot(days + 0.3, od, "^", color="C1", ms=2.2, label="odd trials")
        tr = trends.loc[tu.uid.iloc[0]]
        sgn = tr["sign"]
        b = np.polyfit(days, tu.d_stim * sgn, 1)
        a.plot(days, sgn * np.polyval(b, days), "-", color="C3" if tr.loses else "0.5", lw=1.5,
               label="trend")
        a.axhline(0, color="grey", lw=0.6)
        a.set_ylabel("stimulus d' (contra - ipsi)", fontsize=6)
        a.xaxis.set_major_locator(MaxNLocator(3)), a.yaxis.set_major_locator(MaxNLocator(3))
        a.text(0.02, 0.02 if sgn > 0 else 0.98,
               f"trend {tr.slope_per_10d * sgn:+.2f} / 10 days, p(lose)={tr.p_lose:.3f}\n"
               f"significant (p<0.01) in {tr.frac_sig:.0%} of sessions\n"
               f"position dev {tu.dev_px.median():.1f} px, nearest ROI {tu.nn_px.median():.1f} px",
               transform=a.transAxes, fontsize=5, va="bottom" if sgn > 0 else "top")
        if k == 0:
            a.legend(frameon=False, fontsize=5, loc="upper right", ncol=2)
        if k == len(uids) - 1:
            a.set_xlabel("day")
    fig.suptitle(title, fontsize=9)
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def metric_figure(uids=(STABLE[1], LOSING[0]), fname="examples_metrics"):
    """The profile metric on two example neurons: even vs odd trials of the
    first session (r_w) and even trials of the first vs odd trials of the last
    session (r_x)."""
    t = _table(list(uids))
    fig, axes = plt.subplots(len(uids), 2, figsize=(7, 1.8 * len(uids)), constrained_layout=True,
                             sharex=True)
    for k, u in enumerate(uids):
        tu = t[t.short == u].sort_values("day")
        ra, rb = tu.iloc[0], tu.iloc[-1]
        Pa, Pb = _profile(ra.session, int(ra.row)), _profile(rb.session, int(rb.row))
        for j, (B, lab, c) in enumerate([(Pa[1], "odd trials, same day", "C0"),
                                         (Pb[1], f"odd trials, {rb.day - ra.day} days later", "C1")]):
            a = axes[k, j]
            a.plot(Pa[0], "k-", lw=1, label=f"even trials, day {ra.day}")
            a.plot(B, "-", color=c, lw=1, label=lab)
            for x in (9, 18):
                a.axvline(x - 0.5, color="grey", lw=0.5)
            a.set_title(f"{u}  {'r_w' if j == 0 else 'r_x'} = {np.corrcoef(Pa[0], B)[0, 1]:.2f}",
                        fontsize=7)
            a.legend(frameon=False, fontsize=5.5)
            a.yaxis.set_major_locator(MaxNLocator(3))
            if j == 0:
                a.set_ylabel("deconvolved activity")
    for a in axes[-1]:
        a.set_xticks(np.arange(27))
        a.set_xticklabels(WIN * 3, rotation=90, fontsize=5)
        for i, tl in enumerate(TYPES):
            a.text(4 + 9 * i, -0.42, tl, transform=a.get_xaxis_transform(), ha="center", fontsize=6)
    fig.savefig(OUT / f"{fname}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def trend_figure(fname="sensitivity_trends"):
    """Per mouse: fraction of stimulus-selective neurons that lose / gain
    selectivity over days, against chance (2.5% each)."""
    tr = pd.read_parquet(OUT / "trends.parquet")
    s = tr[tr.selective]
    g = s.groupby("subject").agg(n=("uid", "size"), loses=("loses", "mean"), gains=("gains", "mean"))
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.2), constrained_layout=True)
    x = np.arange(len(g))
    ax[0].bar(x - 0.2, g.loses, 0.4, color="C3", label="loses")
    ax[0].bar(x + 0.2, g.gains, 0.4, color="C2", label="gains")
    ax[0].axhline(0.025, color="k", ls=":", lw=0.8, label="chance")
    ax[0].set_xticks(x), ax[0].set_xticklabels([f"{i}\nn={n}" for i, n in g.n.items()], fontsize=6)
    ax[0].set(ylabel="fraction of selective neurons", title="a  significant trend in stimulus d'")
    ax[0].legend(frameon=False, fontsize=6)
    ax[0].yaxis.set_major_locator(MaxNLocator(3))
    a = ax[1]
    a.hist(s.slope_per_10d, np.linspace(-0.4, 0.4, 41), color="0.4")
    a.axvline(0, color="k", lw=0.6)
    a.set(xlabel="trend of sign-aligned d' (per 10 days)", ylabel="neurons",
          title=f"b  all selective neurons (n={len(s)})")
    a.xaxis.set_major_locator(MaxNLocator(3)), a.yaxis.set_major_locator(MaxNLocator(3))
    fig.savefig(OUT / f"{fname}.pdf")
    fig.savefig(OUT / f"{fname}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fetch()
    raster_figure(STABLE, f"{SUBJECT}: stimulus-selective neurons without a trend over days "
                  "(bottom: no trend, but large day-to-day swings)",
                  "examples_stable")
    raster_figure(LOSING, f"{SUBJECT}: stimulus-selective neurons that lose selectivity over days",
                  "examples_losing")
    metric_figure()
    trend_figure()
