#!/usr/bin/env python3
"""Face-video prediction of shared neural variance during the passive protocol.

Passive-epoch equivalent of `regional_behavior_prediction.py`: same
SVCA and video-PC regression (`stringer19_svca_prediction`), 1,000 neurons
per region-session, native frame rate, 128 components, 16 face-video motion
PCs; the window is the first `DURATION` s of imaged passive protocol, as in
`passive_regional_dimensionality.py`. Block and choice do not exist outside
the task, and extracted wheel data cover the passive epoch in only 2 of the
sessions, so the face video is the only predictor.
"""
from pathlib import Path
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from one.api import ONE

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "pilot_analyses")]
from meso_loader import load_mesoscope_session
from stringer19_region_comparison import N_TOTAL, select_region_neurons
from stringer19_svca_prediction import (
    _split_neurons_checkerboard, _train_test_blocks, _truncated_svd,
    _compute_motion_energy_pcs, _resample_nearest, _predictor_var_explained)
from passive_regional_dimensionality import OUT, PASSIVE_SESSIONS, DURATION, passive_periods

N_SVCS, N_VIDEO_PCS = 128, 16
# 367aff30 and 77141718 (MOs) have no camera data; replaced by the next SP076
# sessions with >= 1,000 MOs neurons, video and >= DURATION s imaged passive.
SESSIONS = {r: list(e) for r, e in PASSIVE_SESSIONS.items()}
SESSIONS["MOs"] = ["a4afea1e-ad72-433d-9498-95ddc54252fe", "66a5f0a1-828d-422e-9181-9432c3f094b2",
                   "d094fdec-fc93-48d4-892c-4b0dd17c553d", "c14efa12-5931-490b-b445-3552044a9c54",
                   "fc97a20e-2c1a-427d-93b9-0a4bc1c41ba7"]
CACHE_DIR = OUT / "passive_behavior_session_results"
VIDEO_CACHE = Path(ONE().cache_dir) / "meso" / "svca_video_pcs"


def compute_session(one, eid, region):
    session = load_mesoscope_session(eid, one=one)
    tab = passive_periods(one, eid)
    times = np.asarray(session.roi_times[0], dtype=float)
    t0 = times[times >= tab.loc["start", "passiveProtocol"]][0]
    t_last = times[times <= tab.loc["stop", "passiveProtocol"]][-1]
    if t_last - t0 < DURATION:
        raise ValueError(f"only {t_last - t0:.0f} s passive imaged")
    t1 = t0 + DURATION
    idx = select_region_neurons(session, region, n_total=N_TOTAL)
    assert len(idx) == N_TOTAL

    i0, i1 = np.searchsorted(times, [t0, t1])
    times_w = times[i0:i1]
    signal = session.roi_signal[idx, i0:i1].astype(np.float64)
    ia, ib = _split_neurons_checkerboard(session.xyz[idx])
    F = signal[ia] - signal[ia].mean(1, keepdims=True)
    G = signal[ib] - signal[ib].mean(1, keepdims=True)
    tr, te = _train_test_blocks(times_w, 72., 2.)
    k = min(N_SVCS, len(ia), len(ib), tr.sum(), te.sum())
    u, _, vt = _truncated_svd(F[:, tr] @ G[:, tr].T / tr.sum(), k)
    p_tr, q_tr, p_te, q_te = u.T @ F[:, tr], vt @ G[:, tr], u.T @ F[:, te], vt @ G[:, te]
    s_hat = np.mean(p_te * q_te, 1)
    s_tot = .5 * (np.mean(p_te ** 2, 1) + np.mean(q_te ** 2, 1))

    vfile = VIDEO_CACHE / f"{eid}_left_passive_pcs{N_VIDEO_PCS}_t{t0:.0f}-{t1:.0f}.npz"
    if vfile.exists():
        with np.load(vfile) as z:
            pcs, pc_times = z["pcs"], z["times"]
    else:
        video = one.load_dataset(eid, "_iblrig_leftCamera.raw.mp4", collection="raw_video_data",
                                 download_only=True)
        cam_times = one.load_dataset(eid, "_ibl_leftCamera.times.npy")
        pcs, pc_times = _compute_motion_energy_pcs(Path(video), cam_times, t0, t1,
                                                   n_pcs=N_VIDEO_PCS, verbose=False)
        np.savez(vfile, pcs=pcs, times=pc_times)
    x = _resample_nearest(pc_times, pcs, times_w).T
    video = _predictor_var_explained(x, p_tr, q_tr, p_te, q_te, tr, te, s_hat, s_tot)
    return dict(reliable_frac=s_hat / s_tot, video_var_explained=video, window=(t0, t1),
                n_neurons_a=len(ia), n_neurons_b=len(ib), n_train=int(tr.sum()), n_test=int(te.sum()))


def compute(regions=None):
    one = ONE()
    CACHE_DIR.mkdir(exist_ok=True)
    out = {}
    for region, eids in SESSIONS.items():
        if regions is not None and region not in regions:
            continue
        res = []
        for eid in eids:
            f = CACHE_DIR / f"{region}_{eid}.npz"
            if not f.exists():
                print(f"[{region}] {eid}", flush=True)
                np.savez(f, **compute_session(one, eid, region))
            with np.load(f) as z:
                res.append({k: z[k] for k in z.files})
        out[region] = res
    return out


def plot(data):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5, "legend.fontsize": 5, "axes.linewidth": .5,
        "lines.linewidth": .8, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(183 / 25.4, 48 / 25.4), sharey=True)
    specs = [("Maximum explainable", "reliable_frac", "#777777"),
             ("Face-video PCs", "video_var_explained", "#0072B2")]
    upper = []
    for letter, region, ax in zip("abcd", ["MOp", "MOs", "VISp", "SSp-bfd"], axes):
        for label, key, color in specs:
            values = 100 * np.clip(np.stack([r[key] for r in data[region]]), 0, None)
            rank = np.arange(1, values.shape[1] + 1)
            mean = values.mean(0)
            sem = values.std(0, ddof=1) / np.sqrt(len(values))
            ax.plot(rank, mean, color=color, label=label)
            ax.fill_between(rank, np.maximum(0, mean - sem), mean + sem, color=color, alpha=.16, lw=0)
            ax.plot(rank, values.T, color=color, lw=.3, alpha=.5)
            upper.append(np.nanmax(values))
        ax.set_xscale("log"); ax.set_xlim(.9, N_SVCS * 1.05)
        ax.set_xticks([1, 10, 100]); ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(which="minor", width=.4, length=1.5)
        ax.set_xlabel("SVC dimension")
        ax.text(.04, .96, region, transform=ax.transAxes, va="top", ha="left", fontsize=6)
        ax.text(-.18, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
    ymax = max(10, 10 * np.ceil(max(upper) / 10))
    axes[0].set_ylim(0, ymax); axes[0].set_yticks([0, ymax / 2, ymax])
    axes[0].set_ylabel("Variance explained (%)")
    axes[-1].legend(frameon=False, loc="upper right", handlelength=1.6)
    fig.subplots_adjust(left=.06, right=.995, bottom=.25, top=.91, wspace=.12)
    fig.savefig(OUT / "passive_behavior_prediction_panels.pdf")
    plt.close(fig)


def main():
    if sys.argv[1:2] == ["--cache-only"]:
        compute(regions=sys.argv[2:])
        return
    plot(compute())


if __name__ == "__main__":
    main()
