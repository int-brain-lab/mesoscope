"""Reproduce Stringer et al. 2019 (Science) Fig. 1F-H for an IBL mesoscope session.

Fig. 1F-H shows, for one example recording: (F) behavioral time series
(running speed, pupil area, whisking) plus the first PC of population
activity, all on a shared time axis with trial-event markers; (G) the raster
of all recorded neurons, sorted by their first-PC weight; (H) the same
raster, sorted by Rastermap (a 1D manifold embedding).

`plot_stringer_figure` adapts this to IBL mesoscope data: wheel speed and
whisker-pad motion energy stand in for running speed and whisking (IBL
sessions don't have a pupil-area trace), and trial events (stimulus onset,
first movement, reward delivery) are drawn as vertical lines instead of the
original stimulus-block markers.

PC1 and the Rastermap ordering are both computed on the *full* session (as
in the paper); the displayed window is a shorter, readable slice of it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy.stats import zscore
from sklearn.decomposition import TruncatedSVD
from rastermap import Rastermap

from one.api import ONE
from brainbox.behavior.wheel import interpolate_position, velocity_filtered

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meso_loader import load_mesoscope_session, MesoscopeSession  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "stringer19"

_RASTERMAP_DEFAULTS = dict(n_PCs=100, n_clusters=30, locality=0.75, time_lag_window=5, bin_size=1)


def _compute_pc1(signal_z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """First PC of population activity: time series + per-neuron weights.

    `signal_z` is (n_neurons, n_time), already z-scored per neuron.
    """
    svd = TruncatedSVD(n_components=1, random_state=0)
    pc1_time = svd.fit_transform(signal_z.T)[:, 0]  # (n_time,)
    pc1_weights = svd.components_[0]  # (n_neurons,)
    return pc1_time, pc1_weights


def _compute_rastermap_order(roi_signal: np.ndarray, **rastermap_kwargs) -> np.ndarray:
    params = {**_RASTERMAP_DEFAULTS, **rastermap_kwargs}
    model = Rastermap(**params).fit(roi_signal)
    return model.isort


def _load_whisker_motion_energy(eid: str, one: ONE, camera: str = "left"):
    try:
        me = one.load_dataset(eid, f"{camera}Camera.ROIMotionEnergy.npy")
        t = one.load_dataset(eid, f"_ibl_{camera}Camera.times.npy")
    except Exception:
        other = "right" if camera == "left" else "left"
        me = one.load_dataset(eid, f"{other}Camera.ROIMotionEnergy.npy")
        t = one.load_dataset(eid, f"_ibl_{other}Camera.times.npy")
    n = min(len(me), len(t))
    return np.asarray(t[:n], dtype=float), np.asarray(me[:n], dtype=float)


def _load_wheel_speed(eid: str, one: ONE, freq: float = 250.0):
    try:
        wheel = one.load_object(eid, "wheel")
    except Exception:
        wheel = one.load_object(eid, "wheel", collection="alf/task_00")
    pos, w_times = interpolate_position(wheel["timestamps"], wheel["position"], freq=freq)
    speed, _ = velocity_filtered(pos, freq)
    return np.asarray(w_times, dtype=float), np.abs(np.asarray(speed, dtype=float))


def _trial_event_times(eid: str, one: ONE):
    trials = one.load_object(eid, "trials")
    stim_on = np.asarray(trials["stimOn_times"], dtype=float)
    motion_on = np.asarray(trials["firstMovement_times"], dtype=float)
    reward = np.asarray(trials["feedback_times"][trials["feedbackType"] == 1], dtype=float)
    return stim_on[np.isfinite(stim_on)], motion_on[np.isfinite(motion_on)], reward[np.isfinite(reward)]


def _normalize(a: np.ndarray) -> np.ndarray:
    a = a - np.nanmin(a)
    peak = np.nanmax(a)
    return a / peak if peak > 0 else a


def plot_stringer_figure(
    eid: str,
    one: Optional[ONE] = None,
    session: Optional[MesoscopeSession] = None,
    window: Optional[Tuple[float, float]] = None,
    window_duration: float = 30.0,
    camera: str = "left",
    rastermap_kwargs: Optional[dict] = None,
    save: bool = True,
    out_dir: Optional[Path] = None,
):
    """Reproduce Stringer et al. 2019 Fig. 1F-H for one mesoscope session.

    Parameters
    ----------
    eid : str
    one : ONE, optional
    session : MesoscopeSession, optional
        Pass an already-loaded session (via `meso_loader.load_mesoscope_session`)
        to avoid reloading.
    window : (float, float), optional
        Display window in seconds (session time). If omitted, a
        `window_duration`-second window is picked starting a third of the
        way into the recording (avoids the sometimes-atypical session start).
    window_duration : float, default 30.0
        Length of the auto-picked display window, in seconds. Ignored if
        `window` is given. PC1 and the Rastermap ordering are always
        computed on the full session regardless of this window.
    camera : {'left', 'right'}, default 'left'
        Which camera's ROI motion energy to use as the whisker-pad signal
        (falls back to the other camera if unavailable).
    rastermap_kwargs : dict, optional
        Overrides for the Rastermap call (defaults match `meso.embed_meso`).
    save : bool, default True
        Save the figure as a PNG under `out_dir`.
    out_dir : Path, optional
        Defaults to `<repo_root>/stringer19/`.

    Returns
    -------
    (fig, info) where info holds `pc1_time`, `pc1_weights`, `pc_order`,
    `rastermap_order`, and the resolved `window`.
    """
    one = one if one is not None else ONE()
    session = session if session is not None else load_mesoscope_session(eid, one=one)

    signal_z = np.nan_to_num(zscore(session.roi_signal, axis=1))
    pc1_time, pc1_weights = _compute_pc1(signal_z)
    pc_order = np.argsort(pc1_weights)
    rm_order = _compute_rastermap_order(session.roi_signal, **(rastermap_kwargs or {}))

    times = np.asarray(session.roi_times[0], dtype=float)

    w_times, whisker_me = _load_whisker_motion_energy(eid, one, camera=camera)
    wheel_times, wheel_speed = _load_wheel_speed(eid, one)
    stim_on, motion_on, reward = _trial_event_times(eid, one)

    if window is not None:
        t0, t1 = window
    else:
        t0 = times[0] + (times[-1] - times[0]) / 3
        t1 = min(t0 + window_duration, times[-1])

    def _clip(t, a):
        m = (t >= t0) & (t <= t1)
        return t[m], a[m]

    t_pc, pc1_win = _clip(times, pc1_time)
    t_whisk, whisk_win = _clip(w_times, whisker_me)
    t_wheel, wheel_win = _clip(wheel_times, wheel_speed)
    stim_win = stim_on[(stim_on >= t0) & (stim_on <= t1)]
    motion_win = motion_on[(motion_on >= t0) & (motion_on <= t1)]
    reward_win = reward[(reward >= t0) & (reward <= t1)]

    idx0, idx1 = np.searchsorted(times, [t0, t1])

    # Per-neuron percentile scaling (20th-99th, computed on the full session so
    # it doesn't depend on which window is displayed) for a readable raster --
    # same convention as meso.plot_raster, needed because deconvolved-trace
    # magnitude varies a lot across neurons.
    p20 = np.percentile(session.roi_signal, 20, axis=1, keepdims=True)
    p99 = np.percentile(session.roi_signal, 99, axis=1, keepdims=True)
    signal_scaled = (session.roi_signal - p20) / np.clip(p99 - p20, 1e-6, None)

    sig_pc = signal_scaled[pc_order][:, idx0:idx1]
    sig_rm = signal_scaled[rm_order][:, idx0:idx1]

    fig = plt.figure(figsize=(12, 11))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 2], hspace=0.15)
    ax_beh = fig.add_subplot(gs[0])
    ax_pc = fig.add_subplot(gs[1], sharex=ax_beh)
    ax_rm = fig.add_subplot(gs[2], sharex=ax_beh)

    ax_beh.plot(t_wheel, _normalize(wheel_win), color="tab:green", lw=0.9, label="wheel speed")
    ax_beh.plot(t_whisk, _normalize(whisk_win), color="tab:orange", lw=0.9, label="whisker ME")
    ax_beh.plot(t_pc, _normalize(pc1_win), color="magenta", lw=0.9, ls="--", label="PC1")
    for t_ in stim_win:
        ax_beh.axvline(t_, color="tab:blue", lw=0.6, alpha=0.5, zorder=0)
    for t_ in motion_win:
        ax_beh.axvline(t_, color="black", lw=0.6, alpha=0.5, zorder=0)
    for t_ in reward_win:
        ax_beh.axvline(t_, color="darkred", lw=0.6, alpha=0.5, zorder=0)

    event_handles = [
        Line2D([0], [0], color="tab:blue", lw=1.5, alpha=0.7, label="stim on"),
        Line2D([0], [0], color="black", lw=1.5, alpha=0.7, label="motion on"),
        Line2D([0], [0], color="darkred", lw=1.5, alpha=0.7, label="reward"),
    ]
    handles, labels = ax_beh.get_legend_handles_labels()
    ax_beh.legend(
        handles=handles + event_handles,
        loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles) + len(event_handles), fontsize=8, frameon=False,
    )
    ax_beh.set_ylabel("normalized")
    ax_beh.set_xlim(t0, t1)
    fig.suptitle(f"{eid}  —  window [{t0:.0f}, {t1:.0f}] s, {session.roi_signal.shape[0]} neurons", y=0.99)
    ax_beh.tick_params(labelbottom=False)

    def _imshow(ax, sig, ylabel):
        # sig is already scaled to ~[0, 1] per neuron (20th-99th percentile).
        # Deconvolved traces are sparse (~85% exactly zero), so most nonzero
        # bins sit well below the neuron's own 99th percentile; vmax=0.5
        # (rather than 1) keeps those visible instead of washing them out,
        # at the cost of clipping the largest transients to solid black.
        ax.imshow(
            sig, aspect="auto", cmap="gray_r", vmin=0, vmax=0.5,
            extent=[t0, t1, 0, sig.shape[0]], origin="lower",
        )
        ax.set_ylabel(ylabel)

    _imshow(ax_pc, sig_pc, "neurons\n(sorted by PC1 weight)")
    ax_pc.tick_params(labelbottom=False)
    _imshow(ax_rm, sig_rm, "neurons\n(sorted by Rastermap)")
    ax_rm.set_xlabel("time (s)")

    for ax in (ax_beh, ax_pc, ax_rm):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if save:
        out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / f"{eid}_stringer19_fig1fgh.png"
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        print("Saved:", fpath)

    info = dict(
        pc1_time=pc1_time,
        pc1_weights=pc1_weights,
        pc_order=pc_order,
        rastermap_order=rm_order,
        window=(t0, t1),
    )
    return fig, info


if __name__ == "__main__":
    eid_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if eid_arg is None:
        raise SystemExit("usage: python stringer19_figure.py <eid>")
    plot_stringer_figure(eid_arg)
