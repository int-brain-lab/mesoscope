"""Reproduce Stringer et al. 2019 (Science) Fig. 2E for an IBL mesoscope
session, on the FULL recording: how well facial-video motion-energy PCs,
versus simple behavioral traces, predict shared (cross-population) neural
variance.

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
green curve uses wheel speed + whisker-pad motion energy instead.

Efficiency for large matrices
------------------------------
Two things scale with the FULL session (~1h, ~30k neural timepoints, and a
~320k-frame video) rather than a short window: the neuron-side covariance
SVD, and the video motion-energy PCA. Both bottleneck on the same
operation -- a truncated SVD -- so `_truncated_svd` (used by both) picks
the fastest way to get it:

- **GPU, when available** (checked once via `torch.cuda.is_available()`):
  `torch.svd_lowrank`, a randomized low-rank SVD -- the same algorithm
  family as `sklearn`'s, just running its matmuls on the GPU. This machine
  has one, and it matters: benchmarked on this script's actual matrix
  sizes, it's ~200x faster than CPU for the neuron-side ~3,500x3,500
  covariance (k=128: 2.9s CPU vs 0.014s GPU warm) and ~40-140x faster for a
  single video segment's ~18,000x2,700 motion-energy matrix (k=200: 6.8s
  CPU vs 0.05-0.18s GPU). This is also what MouseLand/critical_init's
  `SVCA()` does (`torch.linalg.eigh(cov @ cov.T)` on a torch tensor) --
  same hardware target, different algorithm: `eigh` on the explicit Gram
  matrix vs. `svd_lowrank`'s randomized projections, which avoids ever
  forming that O(n^2) matrix and so stays cheap even if `cov` weren't
  already a modest size.
- **CPU fallback**: `sklearn.utils.extmath.randomized_svd` when no CUDA
  device is present -- the same randomized algorithm, just on the CPU.

For the video, GPU-accelerating the SVD step still leaves video *decoding*
(cv2/ffmpeg, CPU-bound, unrelated to the SVD) as the slower part overall --
but the SVD no longer adds meaningfully to it. Decoding the whole video at
once would need `n_frames x n_pixels` ~= 320,000 x 2,700 (~3.5 GB as
float32) in memory simultaneously; `_compute_motion_energy_pcs` instead
uses the paper's own two-stage segmented-SVD trick (Methods, "Behavioral
video acquisition and processing") regardless of GPU availability, since
it bounds *memory* (one segment at a time), not compute: split the video
into `video_segment_duration`-second segments; SVD each segment on its own
and keep only its top `n_seg_pcs` *spatial* components scaled by their
singular values (the paper's U_i = M_i @ V_i); concatenate those
per-segment spatial summaries and SVD that much smaller matrix to get a
session-wide spatial basis; then make a second decoding pass, projecting
each segment's frames onto that shared basis to get the final components'
full-session time course. Peak memory is one segment, not the whole video,
at the cost of decoding the video twice.

Neuron-set splitting also follows MouseLand/critical_init's `SVCA()` more
closely than our earlier single-axis version: a true 2D spatial
checkerboard (both ML and AP coordinates, not just ML strips), which more
robustly keeps any two spatially adjacent neurons (a possible out-of-focus
-fluorescence confound) in different sets.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

from one.api import ONE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meso_loader import load_mesoscope_session, MesoscopeSession  # noqa: E402
from stringer19_figure import _load_wheel_speed, _load_whisker_motion_energy  # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "stringer19"

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _truncated_svd(x: np.ndarray, k: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k SVD (u, s, vt), GPU-accelerated via `torch.svd_lowrank` when a
    CUDA device is available (measured ~40-200x faster than `sklearn`'s
    CPU-only randomized SVD for this script's matrix sizes -- see module
    docstring), falling back to `sklearn.utils.extmath.randomized_svd` on
    CPU otherwise. Both are randomized low-rank algorithms with the same
    asymptotic cost; this only changes which hardware does the matmuls.

    Returns (u, s, vt) in `sklearn.utils.extmath.randomized_svd`'s
    convention: `u` is (n_samples, k), `s` is (k,), `vt` is (k, n_features)
    -- `vt` alone matches `TruncatedSVD.components_` for callers that only
    need the feature-space (e.g. spatial) directions.
    """
    k = min(k, min(x.shape) - 1)
    if _DEVICE == "cuda":
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).cuda()
        torch.manual_seed(seed)
        u, s, v = torch.svd_lowrank(xt, q=k, niter=4)
        return u.cpu().numpy(), s.cpu().numpy(), v.T.cpu().numpy()
    return randomized_svd(x, n_components=k, random_state=seed)


def _split_neurons_checkerboard(xyz: np.ndarray, spacing_um: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
    """2D spatial checkerboard split (ML x AP), as in the paper's "16
    nonoverlapping strips" method and MouseLand/critical_init's `SVCA()`
    (its `dx`/`dy` checkerboard-by-both-axes logic) -- more robust against
    any single-axis spatial confound than a one-axis strip split.
    """
    dx = (xyz[:, 0] % (2 * spacing_um) < spacing_um).astype(int)
    dy = (xyz[:, 1] % (2 * spacing_um) < spacing_um).astype(int)
    is_a = (dx + dy) % 2 == 0
    return np.where(is_a)[0], np.where(~is_a)[0]


def _train_test_blocks(times: np.ndarray, block_duration: float, pad_duration: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Alternating contiguous time blocks -> (train_mask, test_mask), with
    an optional gap dropped at each block boundary (a lightweight version
    of MouseLand/critical_init's `split_traintest` padding, which excludes
    a few timepoints around each chunk to limit train/test leakage from
    slow autocorrelation).
    """
    block_idx = ((times - times[0]) // block_duration).astype(int)
    train_mask = (block_idx % 2) == 0
    test_mask = ~train_mask
    if pad_duration > 0:
        t_in_block = (times - times[0]) % block_duration
        near_edge = (t_in_block < pad_duration) | (t_in_block > block_duration - pad_duration)
        train_mask &= ~near_edge
        test_mask &= ~near_edge
    return train_mask, test_mask


def _resample_nearest(times_src: np.ndarray, values_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(times_src, times_dst), 1, len(times_src) - 1)
    left, right = times_src[idx - 1], times_src[idx]
    idx_final = np.where((times_dst - left) < (right - times_dst), idx - 1, idx)
    return values_src[idx_final]


def _video_segments(t0: float, t1: float, segment_duration: float) -> List[Tuple[float, float]]:
    edges = np.arange(t0, t1, segment_duration)
    edges = np.append(edges, t1)
    return list(zip(edges[:-1], edges[1:]))


def _decode_motion_energy_segment(
    video_path: Path, cam_times: np.ndarray, s0: float, s1: float, resize: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Decode one segment's motion-energy frames sequentially (fast on a
    local file; avoids per-frame seeking). Returns (frames, times) with
    `frames` of shape (n_frames_in_segment - 1, prod(resize)), or
    (None, None) if the segment had fewer than 2 frames.
    """
    idx0, idx1 = np.searchsorted(cam_times, [s0, s1])
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

    if len(me_frames) < 2:
        return None, None
    return np.stack(me_frames, axis=0), np.asarray(me_times)


def _compute_motion_energy_pcs(
    video_path: Path,
    cam_times: np.ndarray,
    t0: float,
    t1: float,
    n_pcs: int = 16,
    n_seg_pcs: int = 200,
    video_segment_duration: float = 300.0,
    resize: Tuple[int, int] = (60, 45),
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Motion-energy PCs of the raw video across [t0, t1], via the paper's
    two-stage segmented SVD (see module docstring): per-segment SVD (stage
    1) bounds peak memory to one segment; a second SVD (stage 2) merges
    segments' spatial summaries into one session-wide spatial basis; a
    second decoding pass (stage 3) projects the full video onto it.

    Returns (pcs, pc_times): `pcs` is (n_frames - n_segments, n_pcs) --
    each segment loses one frame to differencing -- `pc_times` are the
    corresponding later-frame camera timestamps, in time order.
    """
    segments = _video_segments(t0, t1, video_segment_duration)

    # --- stage 1: per-segment spatial summaries (bounded memory) ---
    spatial_chunks = []
    for i, (s0, s1) in enumerate(segments):
        me, _ = _decode_motion_energy_segment(video_path, cam_times, s0, s1, resize)
        if me is None:
            continue
        me -= me.mean(axis=0, keepdims=True)
        _, singular_values, components = _truncated_svd(me, n_seg_pcs, seed=seed)
        # paper's U_i = M_i @ V_i: spatial components scaled by singular value
        spatial_chunks.append(components.T * singular_values[None, :])
        if verbose:
            print(f"  [video SVD, {_DEVICE}] segment {i + 1}/{len(segments)} ({me.shape[0]} frames) -> {len(singular_values)} components")
        del me

    if not spatial_chunks:
        raise RuntimeError(f"No usable video segments decoded in [{t0}, {t1}]")

    merged = np.concatenate(spatial_chunks, axis=1)  # (n_pixels, n_seg_pcs * n_segments)
    del spatial_chunks

    # --- stage 2: merge into one session-wide spatial basis ---
    _, _, spatial_basis = _truncated_svd(merged.T, n_pcs, seed=seed)  # (n_pcs, n_pixels)
    del merged

    # --- stage 3: second pass, project the full video onto the shared basis ---
    pcs_chunks, time_chunks = [], []
    for i, (s0, s1) in enumerate(segments):
        me, me_t = _decode_motion_energy_segment(video_path, cam_times, s0, s1, resize)
        if me is None:
            continue
        pcs_chunks.append(me @ spatial_basis.T)
        time_chunks.append(me_t)
        if verbose:
            print(f"  [video proj] segment {i + 1}/{len(segments)} projected")
        del me

    pcs = np.concatenate(pcs_chunks, axis=0)
    pc_times = np.concatenate(time_chunks, axis=0)
    return pcs, pc_times


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
    n_svcs: int = 128,
    spacing_um: float = 60.0,
    block_duration: float = 72.0,
    pad_duration: float = 2.0,
    n_video_pcs: int = 16,
    n_seg_pcs: int = 200,
    video_segment_duration: float = 300.0,
    video_resize: Tuple[int, int] = (60, 45),
    camera: str = "left",
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Compute Fig. 2E's three curves for one session (no plotting).

    Parameters
    ----------
    eid : str
    one : ONE, optional
    session : MesoscopeSession, optional
    window : (float, float), optional
        Analysis window in session seconds. Defaults to the *entire*
        session -- see the module docstring for how the video-PCA step
        stays tractable at full length. Pass a shorter window for a quick
        test run.
    n_svcs : int, default 128
        Number of shared variance components (matches the paper's default).
        Capped automatically by the smaller neuron set / time-block sizes.
    spacing_um : float, default 60.0
        Checkerboard square size (microns) used to assign neurons to the
        two SVCA sets -- matches the paper's "16 nonoverlapping strips of
        width 60 um", generalized to 2D as in MouseLand/critical_init's
        `SVCA()`.
    block_duration : float, default 72.0
        Length of each alternating train/test time block, in seconds --
        matches the paper's stated value ("alternating periods of 72 s").
    pad_duration : float, default 2.0
        Seconds excluded from both sides of each block boundary, reducing
        train/test leakage from slow autocorrelation (a lightweight version
        of MouseLand/critical_init's `split_traintest` padding).
    n_video_pcs : int, default 16
        Number of final motion-energy PCs (the paper uses ~1000; 16 is a
        pragmatic choice for a pilot script -- see module docstring).
    n_seg_pcs : int, default 200
        Per-segment component count in the video PCA's first SVD stage
        (matches the paper's own choice of 200).
    video_segment_duration : float, default 300.0
        Length of each video-PCA segment, in seconds.
    video_resize : (int, int), default (60, 45)
        Frame size (width, height) the video is downsized to before
        computing motion energy and its PCA.
    camera : {'left', 'right'}, default 'left'
    seed : int, default 0
        RNG seed for the SVD steps (reproducible by default).
    verbose : bool, default True
        Print progress through the video-PCA segments (the slowest step).

    Returns
    -------
    dict with `reliable_frac`, `video_var_explained`, `behav_var_explained`
    (each length `n_svcs`, sorted by SVC rank), plus `window`, `n_neurons_a`,
    `n_neurons_b`, `n_train`, `n_test`, `n_video_frames`.
    """
    one = one if one is not None else ONE()
    session = session if session is not None else load_mesoscope_session(eid, one=one)

    times = np.asarray(session.roi_times[0], dtype=float)
    t0, t1 = window if window is not None else (times[0], times[-1])

    idx0, idx1 = np.searchsorted(times, [t0, t1])
    times_w = times[idx0:idx1]
    signal_w = session.roi_signal[:, idx0:idx1].astype(np.float64)

    idx_a, idx_b = _split_neurons_checkerboard(session.xyz, spacing_um=spacing_um)
    F = signal_w[idx_a] - signal_w[idx_a].mean(axis=1, keepdims=True)
    G = signal_w[idx_b] - signal_w[idx_b].mean(axis=1, keepdims=True)

    train_mask, test_mask = _train_test_blocks(times_w, block_duration, pad_duration)
    F_train, F_test = F[:, train_mask], F[:, test_mask]
    G_train, G_test = G[:, train_mask], G[:, test_mask]

    k = min(n_svcs, len(idx_a), len(idx_b), train_mask.sum(), test_mask.sum())
    cov_train = F_train @ G_train.T / train_mask.sum()
    u, _, vt = _truncated_svd(cov_train, k, seed=seed)

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

    # --- video motion-energy PCs (segmented SVD, full session) ---
    video_path = one.load_dataset(eid, f"_iblrig_{camera}Camera.raw.mp4", collection="raw_video_data", download_only=True)
    cam_times = one.load_dataset(eid, f"_ibl_{camera}Camera.times.npy")
    video_pcs, video_pc_times = _compute_motion_energy_pcs(
        Path(video_path), cam_times, t0, t1,
        n_pcs=n_video_pcs, n_seg_pcs=n_seg_pcs, video_segment_duration=video_segment_duration,
        resize=video_resize, seed=seed, verbose=verbose,
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
        f"{eid}\nfull session [{t0:.0f}, {t1:.0f}] s  "
        f"({result['n_neurons_a']}+{result['n_neurons_b']} neurons)",
        fontsize=10,
    )
    fig.tight_layout()

    if save:
        out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / f"{eid}_stringer19_fig2e_full.png"
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        print("Saved:", fpath)

    return fig, result


if __name__ == "__main__":
    eid_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if eid_arg is None:
        raise SystemExit("usage: python stringer19_svca_prediction.py <eid>")
    plot_svca_behavior_prediction(eid_arg)
