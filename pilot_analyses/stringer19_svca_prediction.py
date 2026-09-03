"""Reproduce Stringer et al. 2019 (Science) Fig. 2E for an IBL mesoscope
session: how well facial-video motion-energy PCs, versus simple behavioral
traces, predict shared (cross-population) neural variance.

Method (shared variance component analysis, SVCA; see the paper's Methods,
"Shared variance component analysis" and "Predicting neural activity from
behavioral variables"): the recorded neurons are split into two spatially
segregated sets, and the recording into training and test time points.
Training data is used to find the dimensions along which the two neuron
sets' activity maximally covaries ("shared variance components", SVCs).
Projecting held-out test data onto these dimensions gives, for each SVC,
a reliable-variance fraction (how much of the two sets' variance in that
dimension is genuinely shared, i.e. not independent per-set noise) -- the
gray "max explainable" curve. Behavioral predictors are then regressed
against each set's SVC time series independently (on training data), and
the fraction of that reliable variance recovered by the *cross-set*
covariance of the residuals gives the "% variance explained" blue/green
curves.

Fig. 2E's blue curve uses ~1000-dim facial motion-energy PCA (from the
whole face video); the green curve uses running + pupil area + whisking.
IBL sessions don't have pupil or a comparable running measure, so here the
green curve uses wheel speed + whisker-pad motion energy instead, and the
blue curve uses a modest number of motion-energy PCs computed directly
from the raw video (see `_compute_motion_energy_pcs`) rather than ~1000,
since this is a pilot reproduction, not a full re-analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

from one.api import ONE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meso_loader import load_mesoscope_session, MesoscopeSession  # noqa: E402
from stringer19_figure import _load_wheel_speed, _load_whisker_motion_energy  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "stringer19"


def _split_neurons_spatial(ml_coord: np.ndarray, n_strips: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Interleaved-strip split (paper's method): avoids any neuron in set A
    having its closest neighbor (same-ish position, different depth -- a
    possible out-of-focus-fluorescence confound) in set B.
    """
    edges = np.linspace(ml_coord.min(), ml_coord.max(), n_strips + 1)
    strip_idx = np.clip(np.digitize(ml_coord, edges[1:-1]), 0, n_strips - 1)
    is_a = (strip_idx % 2) == 0
    return np.where(is_a)[0], np.where(~is_a)[0]


def _train_test_blocks(times: np.ndarray, block_duration: float) -> Tuple[np.ndarray, np.ndarray]:
    """Alternating contiguous time blocks -> (train_mask, test_mask)."""
    block_idx = ((times - times[0]) // block_duration).astype(int)
    train_mask = (block_idx % 2) == 0
    return train_mask, ~train_mask


def _resample_nearest(times_src: np.ndarray, values_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(times_src, times_dst), 1, len(times_src) - 1)
    left, right = times_src[idx - 1], times_src[idx]
    idx_final = np.where((times_dst - left) < (right - times_dst), idx - 1, idx)
    return values_src[idx_final]


def _compute_motion_energy_pcs(
    video_path: Path,
    cam_times: np.ndarray,
    t0: float,
    t1: float,
    n_pcs: int = 16,
    resize: Tuple[int, int] = (60, 45),
) -> Tuple[np.ndarray, np.ndarray]:
    """Motion-energy PCs of the raw video within [t0, t1].

    Reads frames sequentially (fast on a local file; avoids per-frame
    seeking), downsizes and grayscales each, takes the absolute
    frame-to-frame difference ("motion energy", as in Fig. 2B), then PCA's
    the resulting movie. Returns (pcs, pc_times): `pcs` is
    (n_frames - 1, n_pcs), `pc_times` are the later frame's camera
    timestamp for each motion-energy sample.
    """
    idx0, idx1 = np.searchsorted(cam_times, [t0, t1])
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)

    prev, me_frames, me_times = None, [], []
    for i in range(idx0, idx1):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, resize, interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            me_frames.append(np.abs(small - prev).ravel())
            me_times.append(cam_times[i])
        prev = small
    cap.release()

    if len(me_frames) < n_pcs + 1:
        raise RuntimeError(f"Only decoded {len(me_frames)} video frames in [{t0}, {t1}] -- too few for PCA")

    X = np.stack(me_frames, axis=0)
    X -= X.mean(axis=0, keepdims=True)
    svd = TruncatedSVD(n_components=n_pcs, random_state=0)
    pcs = svd.fit_transform(X)
    return pcs, np.asarray(me_times)


def _regress_and_residualize(proj_train: np.ndarray, proj_test: np.ndarray, x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Unregularized least-squares prediction of each row of `proj` from
    `x` (fit on train, applied to test); returns the test-set residual.
    """
    coef, *_ = np.linalg.lstsq(x_train.T, proj_train.T, rcond=None)  # (n_predictors, K)
    pred_test = coef.T @ x_test
    return proj_test - pred_test


def compute_svca_prediction(
    eid: str,
    one: Optional[ONE] = None,
    session: Optional[MesoscopeSession] = None,
    window: Optional[Tuple[float, float]] = None,
    window_duration: float = 300.0,
    n_svcs: int = 128,
    n_strips: int = 16,
    block_duration: float = 20.0,
    n_video_pcs: int = 16,
    video_resize: Tuple[int, int] = (60, 45),
    camera: str = "left",
    seed: int = 0,
) -> dict:
    """Compute Fig. 2E's three curves for one session (no plotting).

    Parameters
    ----------
    eid : str
    one : ONE, optional
    session : MesoscopeSession, optional
    window : (float, float), optional
        Analysis window in session seconds. If omitted, `window_duration`
        seconds starting a third of the way into the recording are used.
    window_duration : float, default 300.0
        Length of the auto-picked window. SVCA and the video-PC decoding
        are both restricted to this window -- the full ~1h recording (and
        its ~10^5-frame video) is impractical to decode/regress in a pilot
        script; 300s keeps video decoding and the SVD steps to well under
        a minute while still giving ~1,500 neural timepoints.
    n_svcs : int, default 128
        Number of shared variance components (matches the paper's default).
        Capped automatically by the smaller neuron set / time-block sizes.
    n_strips : int, default 16
        Number of alternating spatial strips (by ML coordinate) used to
        assign neurons to the two SVCA sets (matches the paper's method).
    block_duration : float, default 20.0
        Length of each alternating train/test time block, in seconds.
    n_video_pcs : int, default 16
        Number of motion-energy PCs computed from the raw video (the paper
        uses ~1000; 16 is a pragmatic choice for a single-session pilot
        script -- see module docstring).
    video_resize : (int, int), default (60, 45)
        Frame size (width, height) the video is downsized to before
        computing motion energy and its PCA.
    camera : {'left', 'right'}, default 'left'
    seed : int, default 0
        RNG seed (only used if `n_svcs` truncation ties need breaking; kept
        for reproducibility / future extensions).

    Returns
    -------
    dict with `reliable_frac`, `video_var_explained`, `behav_var_explained`
    (each length `n_svcs`, sorted by SVC rank), plus `window`, `n_neurons_a`,
    `n_neurons_b`, `n_train`, `n_test`, `n_video_frames`.
    """
    one = one if one is not None else ONE()
    session = session if session is not None else load_mesoscope_session(eid, one=one)

    times = np.asarray(session.roi_times[0], dtype=float)
    if window is not None:
        t0, t1 = window
    else:
        t0 = times[0] + (times[-1] - times[0]) / 3
        t1 = min(t0 + window_duration, times[-1])

    idx0, idx1 = np.searchsorted(times, [t0, t1])
    times_w = times[idx0:idx1]
    signal_w = session.roi_signal[:, idx0:idx1].astype(np.float64)

    idx_a, idx_b = _split_neurons_spatial(session.xyz[:, 0], n_strips=n_strips)
    F = signal_w[idx_a] - signal_w[idx_a].mean(axis=1, keepdims=True)
    G = signal_w[idx_b] - signal_w[idx_b].mean(axis=1, keepdims=True)

    train_mask, test_mask = _train_test_blocks(times_w, block_duration)
    F_train, F_test = F[:, train_mask], F[:, test_mask]
    G_train, G_test = G[:, train_mask], G[:, test_mask]

    k = min(n_svcs, len(idx_a), len(idx_b), train_mask.sum(), test_mask.sum())
    cov_train = F_train @ G_train.T / train_mask.sum()
    u, _, vt = randomized_svd(cov_train, n_components=k, random_state=seed)

    proj1_train, proj2_train = u.T @ F_train, vt @ G_train
    proj1_test, proj2_test = u.T @ F_test, vt @ G_test

    s_hat = np.mean(proj1_test * proj2_test, axis=1)
    s_tot = 0.5 * (np.mean(proj1_test ** 2, axis=1) + np.mean(proj2_test ** 2, axis=1))
    reliable_frac = s_hat / s_tot

    # --- behavior (wheel speed + whisker ME) ---
    wheel_times, wheel_speed = _load_wheel_speed(eid, one)
    whisk_times, whisk_me = _load_whisker_motion_energy(eid, one, camera=camera)
    wheel_w = _resample_nearest(wheel_times, wheel_speed, times_w)
    whisk_w = _resample_nearest(whisk_times, whisk_me, times_w)
    x_behav = np.stack([wheel_w, whisk_w], axis=0)
    x_behav = (x_behav - x_behav[:, train_mask].mean(axis=1, keepdims=True)) / x_behav[:, train_mask].std(axis=1, keepdims=True)

    resid1_b = _regress_and_residualize(proj1_train, proj1_test, x_behav[:, train_mask], x_behav[:, test_mask])
    resid2_b = _regress_and_residualize(proj2_train, proj2_test, x_behav[:, train_mask], x_behav[:, test_mask])
    s_res_behav = np.mean(resid1_b * resid2_b, axis=1)
    behav_var_explained = (s_hat - s_res_behav) / s_tot

    # --- video motion-energy PCs ---
    video_path = one.load_dataset(eid, f"_iblrig_{camera}Camera.raw.mp4", collection="raw_video_data", download_only=True)
    cam_times = one.load_dataset(eid, f"_ibl_{camera}Camera.times.npy")
    video_pcs, video_pc_times = _compute_motion_energy_pcs(
        Path(video_path), cam_times, t0, t1, n_pcs=n_video_pcs, resize=video_resize
    )
    video_pcs_w = _resample_nearest(video_pc_times, video_pcs, times_w)  # (T, n_video_pcs)
    x_video = video_pcs_w.T
    x_video = (x_video - x_video[:, train_mask].mean(axis=1, keepdims=True)) / x_video[:, train_mask].std(axis=1, keepdims=True)

    resid1_v = _regress_and_residualize(proj1_train, proj1_test, x_video[:, train_mask], x_video[:, test_mask])
    resid2_v = _regress_and_residualize(proj2_train, proj2_test, x_video[:, train_mask], x_video[:, test_mask])
    s_res_video = np.mean(resid1_v * resid2_v, axis=1)
    video_var_explained = (s_hat - s_res_video) / s_tot

    return dict(
        reliable_frac=reliable_frac,
        video_var_explained=video_var_explained,
        behav_var_explained=behav_var_explained,
        window=(t0, t1),
        n_neurons_a=len(idx_a),
        n_neurons_b=len(idx_b),
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        n_video_frames=video_pcs.shape[0],
    )


def plot_svca_behavior_prediction(
    eid: str,
    one: Optional[ONE] = None,
    session: Optional[MesoscopeSession] = None,
    save: bool = True,
    out_dir: Optional[Path] = None,
    **kwargs,
):
    """Reproduce Stringer et al. 2019 Fig. 2E for one mesoscope session.

    Extra keyword arguments are passed through to `compute_svca_prediction`.
    Returns `(fig, result)`.
    """
    one = one if one is not None else ONE()
    session = session if session is not None else load_mesoscope_session(eid, one=one)
    result = compute_svca_prediction(eid, one=one, session=session, **kwargs)

    k = len(result["reliable_frac"])
    rank = np.arange(1, k + 1)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(rank, 100 * np.clip(result["reliable_frac"], 0, None), color="gray", lw=1.5, label="max explainable")
    ax.plot(rank, 100 * np.clip(result["video_var_explained"], 0, None), color="tab:blue", lw=1.5, label="left video PCs")
    ax.plot(rank, 100 * np.clip(result["behav_var_explained"], 0, None), color="tab:green", lw=1.5, label="wheel + whisker ME")
    ax.set_xscale("log")
    ax.set_xlabel("SVC dimension")
    ax.set_ylabel("% variance explained")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    t0, t1 = result["window"]
    ax.set_title(
        f"{eid}\nwindow [{t0:.0f}, {t1:.0f}] s  "
        f"({result['n_neurons_a']}+{result['n_neurons_b']} neurons)",
        fontsize=10,
    )
    fig.tight_layout()

    if save:
        out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / f"{eid}_stringer19_fig2e.png"
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        print("Saved:", fpath)

    return fig, result


if __name__ == "__main__":
    eid_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if eid_arg is None:
        raise SystemExit("usage: python stringer19_svca_prediction.py <eid>")
    plot_svca_behavior_prediction(eid_arg)
