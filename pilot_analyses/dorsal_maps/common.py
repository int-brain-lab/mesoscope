"""Shared pieces for the meso-vs-BWM dorsal cortex maps.

Everything that must be *identical* across the two modalities lives here,
so meso and BWM go through the same code path once their per-session
features are extracted:

- trial selection and labels (BWM `load_trials_and_mask` criteria)
- hexagonal ML-AP spatial bins
- the single-neuron tuning test (vectorised re-implementation of
  Yanliang Shi's condition-combined, block-shuffled Mann-Whitney U from
  `brainwidemap.single_cell_stats`)
- regularised decoders (L2 logistic regression for stim / choice /
  feedback; ridge regression for wheel speed)
- the top-view atlas outline used under every map.

Coordinates are IBL/CCF bregma-centred, in mm: x = ML (+ right),
y = AP (+ anterior), z = DV (+ up).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"  # figures, tables, results (not tracked)

# Per-session feature files are large (~10 GB in total), keep them next to the
# ONE cache rather than in the repo.
FEATURE_ROOT = Path.home() / "Downloads" / "ONE" / "alyx.internationalbrainlab.org" / "dorsal_maps"

VARIABLES = ("stim", "choice", "feedback")

# Analysis windows (s) per variable: (alignment event, start, stop).
# BWM windows are the ones used for the BWM single-cell tests. Meso frames are
# ~0.2 s apart, so the meso windows are widened to 0.3 s so that each contains
# at least one sample of the deconvolved trace (values are read off the
# linearly interpolated per-ROI trace, so window means are well defined).
WINDOWS = {
    "bwm": {"stim": ("stimOn_times", 0.0, 0.1),
            "choice": ("firstMovement_times", -0.1, 0.0),
            "feedback": ("feedback_times", 0.0, 0.2)},
    "meso": {"stim": ("stimOn_times", 0.0, 0.3),
             "choice": ("firstMovement_times", -0.3, 0.0),
             "feedback": ("feedback_times", 0.0, 0.3)},
}
# Wheel-speed decoding: 0.2 s bins (one meso frame) from -0.2 to 1.0 s
# around first movement, for both modalities.
WHEEL_EDGES = np.arange(-0.2, 1.0 + 1e-9, 0.2)

MIN_NEURONS = 10  # neurons per (session, bin) to count a session / decode
ALPHA = 0.05      # significance level for the tuning test
N_SHUF = 3000     # shuffles for the tuning test, as in the BWM code
HEX_SIZE = 0.2    # hexagon centre-to-vertex distance, mm


# ---------------------------------------------------------------------------
# trials
# ---------------------------------------------------------------------------
def trial_mask(trials, min_rt=0.08, max_rt=2.0):
    """BWM trial selection (`brainwidemap.load_trials_and_mask` defaults):
    finite event times, a choice was made, 0.08 s <= RT <= 2 s."""
    keys = ("stimOn_times", "firstMovement_times", "feedback_times", "choice",
            "probabilityLeft", "feedbackType")
    ok = np.ones(len(trials["stimOn_times"]), bool)
    for k in keys:
        ok &= np.isfinite(np.asarray(trials[k], float))
    ok &= np.asarray(trials["choice"]) != 0
    rt = np.asarray(trials["firstMovement_times"]) - np.asarray(trials["stimOn_times"])
    with np.errstate(invalid="ignore"):
        ok &= (rt >= min_rt) & (rt <= max_rt)
    return ok


def trial_labels(trials, mask):
    """Per-trial arrays used by the tests and decoders (masked trials only)."""
    cl = np.asarray(trials["contrastLeft"], float)[mask]
    cr = np.asarray(trials["contrastRight"], float)[mask]
    return dict(
        c_L=np.nan_to_num(cl), c_R=np.nan_to_num(cr),
        stim_left=np.isfinite(cl).astype(int),       # stimulus side incl. 0 % contrast
        choice=np.asarray(trials["choice"])[mask].astype(int),  # +1 / -1 as in trials.choice
        reward=(np.asarray(trials["feedbackType"])[mask] == 1).astype(int),
        block=np.asarray(trials["probabilityLeft"], float)[mask],
    )


def task_intervals(trials, mask, modality):
    """(n_trials, 3 + n_wheel_bins, 2) array of [start, stop] times: the
    stim/choice/feedback windows followed by the wheel bins."""
    out = []
    for v in VARIABLES:
        ev, a, b = WINDOWS[modality][v]
        t = np.asarray(trials[ev], float)[mask]
        out.append(np.c_[t + a, t + b])
    t = np.asarray(trials["firstMovement_times"], float)[mask]
    for a, b in zip(WHEEL_EDGES[:-1], WHEEL_EDGES[1:]):
        out.append(np.c_[t + a, t + b])
    return np.stack(out, axis=1)


MAX_WHEEL_STEP = 100.0  # rad between consecutive wheel samples; larger steps are encoder glitches


def remove_wheel_jumps(wheel_pos, max_step=MAX_WHEEL_STEP):
    """Wheel position with impossible jumps between consecutive samples
    removed. Some mesoscope sessions have encoder counter overflows: the
    position jumps by exactly +-6588397.3 rad for one or more samples, which
    would give bins with ~3e7 rad/s."""
    d = np.diff(np.asarray(wheel_pos, float))
    d[np.abs(d) > max_step] = 0
    return np.r_[wheel_pos[0], wheel_pos[0] + np.cumsum(d)]


def wheel_speed(wheel_times, wheel_pos, intervals):
    """Mean absolute wheel velocity (rad/s) in each interval, from the
    cumulative absolute displacement of the 1 kHz-interpolated position
    (encoder glitches removed first, see remove_wheel_jumps)."""
    from brainbox.behavior.wheel import interpolate_position
    pos, t = interpolate_position(wheel_times, remove_wheel_jumps(wheel_pos), freq=1000)
    cum = np.r_[0, np.cumsum(np.abs(np.diff(pos)))]
    a, b = intervals[..., 0], intervals[..., 1]
    return (np.interp(b, t, cum) - np.interp(a, t, cum)) / (b - a)


# ---------------------------------------------------------------------------
# neural features
# ---------------------------------------------------------------------------
def interp_window_means(signal, frame_times, offsets, intervals, n_sub=8, chunk=64):
    """Mean of each ROI's linearly interpolated trace in each interval.

    signal: (n_rois, T); frame_times: (T,); offsets: (n_rois,) per-ROI time
    offset (ROI time = frame_times + offset); intervals: (..., 2).
    Returns (n_rois, *intervals.shape[:-1]) float32.
    """
    shp = intervals.shape[:-1]
    iv = intervals.reshape(-1, 2)
    frac = (np.arange(n_sub) + 0.5) / n_sub
    q = (iv[:, :1] + (iv[:, 1:] - iv[:, :1]) * frac).ravel()  # (K * n_sub,)
    T = frame_times.size
    out = np.empty((signal.shape[0], iv.shape[0]), np.float32)
    for i0 in range(0, signal.shape[0], chunk):
        s = signal[i0:i0 + chunk]
        tq = q[None, :] - offsets[i0:i0 + chunk, None]
        idx = np.clip(np.searchsorted(frame_times, tq), 1, T - 1)
        t_lo, t_hi = frame_times[idx - 1], frame_times[idx]
        w = np.clip((tq - t_lo) / (t_hi - t_lo), 0, 1)
        rows = np.arange(s.shape[0])[:, None]
        v = s[rows, idx - 1] * (1 - w) + s[rows, idx] * w
        out[i0:i0 + chunk] = v.reshape(s.shape[0], -1, n_sub).mean(-1)
    return out.reshape(signal.shape[0], *shp)


def spike_window_rates(spike_times, spike_clusters, cluster_ids, intervals):
    """Firing rate (Hz) of each cluster in each interval -> (n_clusters, ...)."""
    shp = intervals.shape[:-1]
    iv = intervals.reshape(-1, 2)
    order = np.argsort(spike_clusters, kind="stable")
    st, sc = spike_times[order], spike_clusters[order]
    out = np.zeros((len(cluster_ids), iv.shape[0]), np.float32)
    dur = iv[:, 1] - iv[:, 0]
    for k, c in enumerate(cluster_ids):
        lo = np.searchsorted(sc, c, "left")
        hi = np.searchsorted(sc, c, "right")
        t = np.sort(st[lo:hi])
        out[k] = (np.searchsorted(t, iv[:, 1]) - np.searchsorted(t, iv[:, 0])) / dur
    return out.reshape(len(cluster_ids), *shp)


# ---------------------------------------------------------------------------
# spatial bins
# ---------------------------------------------------------------------------
_SQ3 = np.sqrt(3.0)


def hex_index(x, y, size=HEX_SIZE):
    """Axial (q, r) index of the pointy-top hexagon containing each point."""
    qf = (_SQ3 / 3 * x - y / 3) / size
    rf = (2 / 3 * y) / size
    xf, zf = qf, rf
    yf = -xf - zf
    rx, ry, rz = np.round(xf), np.round(yf), np.round(zf)
    dx, dy, dz = np.abs(rx - xf), np.abs(ry - yf), np.abs(rz - zf)
    fix_x = (dx > dy) & (dx > dz)
    fix_z = ~fix_x & (dz >= dy)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(fix_z, -rx - ry, rz)
    return rx.astype(int), rz.astype(int)


def hex_center(q, r, size=HEX_SIZE):
    return size * _SQ3 * (q + r / 2), size * 1.5 * r


def hex_key(q, r):
    return np.asarray(q) * 10000 + np.asarray(r)  # single int per bin


def hex_unkey(k):
    k = np.asarray(k)
    r = (k + 5000) % 10000 - 5000
    return (k - r) // 10000, r


def bin_neurons(xyz_mm, fold=True, size=HEX_SIZE):
    """Hex bin key per neuron; with fold=True both hemispheres are mapped onto
    the right one (x -> |x|)."""
    x = np.abs(xyz_mm[:, 0]) if fold else xyz_mm[:, 0]
    return hex_key(*hex_index(x, xyz_mm[:, 1], size))


def isocortex_mask(atlas_ids, regions):
    return regions.id2acronym(atlas_ids, mapping="Cosmos") == "Isocortex"


# ---------------------------------------------------------------------------
# single-neuron tuning test
# ---------------------------------------------------------------------------
def _strata(lab, variable):
    """Condition strata (list of (x_mask, y_mask)) exactly as in the
    brainwidemap single_cell_stats example scripts."""
    blk, cL, cR, ch, rw = lab["block"], lab["c_L"], lab["c_R"], lab["choice"], lab["reward"]
    # BWM uses the biased blocks only. Sessions without biased blocks (all
    # trials at probabilityLeft 0.5, a third of the mesoscope sessions) would
    # get no test at all, so there the single 0.5 block is used instead.
    levels = (0.8, 0.2) if np.isin(blk, (0.8, 0.2)).any() else (0.5,)
    out = []
    for b in levels:
        if variable == "stim":      # left vs right stimulus, within block x choice
            for c in (1, -1):
                m = (blk == b) & (ch == c)
                out.append((m & (cL > 0), m & (cR > 0)))
        elif variable == "choice":  # choice +1 vs -1, within block x stim side
            for side in (cL, cR):
                m = (blk == b) & (side > 0)
                out.append((m & (ch == 1), m & (ch == -1)))
        elif variable == "feedback":  # rewarded vs not, within block x stim side
            for side in (cL, cR):
                m = (blk == b) & (side > 0)
                out.append((m & (rw == 1), m & (rw == 0)))
    return out


def block_ids(block):
    """Consecutive-block index per trial (s_block in the BWM code)."""
    return np.r_[0, np.cumsum(np.abs(np.diff(block)) > 0)]


def tuning_pvalues(rates, lab, variable, n_shuf=N_SHUF, seed=0, device=None, return_stat=False):
    """p-value per neuron for `variable` (stim / choice / feedback).

    Vectorised version of `get_{stim,choice,feedback}_time_shuffle` from
    brainwidemap.single_cell_stats: Mann-Whitney U statistics
    min(U_x, U_y) are summed over the 4 condition strata and normalised
    by sum(nx*ny); the null comes from permuting the x/y labels within
    each consecutive block (only blocks with trials of both kinds are
    permuted; U_x and U_y are shuffled independently, as in the original),
    and p = rank of the observed statistic among the n_shuf + 1 values /
    (n_shuf + 1) (small = tuned). Only difference to the original: the
    same permutations are shared by all neurons of a session.

    rates: (n_trials, n_neurons).
    With return_stat=True also returns the tuning strength: the mean of the
    shuffled statistics minus the observed one. The statistic is
    0.5 - |AUC - 0.5| pooled over strata, so this is the separation of the
    two conditions (|AUC - 0.5|) in excess of its chance level, which grows
    with fewer trials. 0 = chance; at most 0.5.
    """
    import torch
    from scipy.stats import rankdata
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    sb = block_ids(lab["block"])
    n = rates.shape[1]
    num = torch.zeros((n_shuf + 1, n), dtype=torch.float64, device=device)
    den = 0.0
    for xm, ym in _strata(lab, variable):
        ix, iy = np.flatnonzero(xm), np.flatnonzero(ym)
        nx, ny = ix.size, iy.size
        if nx == 0 or ny == 0:
            continue
        idx = np.r_[ix, iy]
        R = torch.as_tensor(rankdata(rates[idx], axis=0), dtype=torch.float64, device=device)
        lab0 = np.r_[np.ones(nx), np.zeros(ny)]
        b = sb[idx]
        blocks = np.intersect1d(sb[ix], sb[iy])

        def shuffled_labels():
            L = np.tile(lab0, (n_shuf + 1, 1))
            for blk in blocks:
                cols = np.flatnonzero(b == blk)
                L[1:, cols] = lab0[rng.permuted(np.tile(cols, (n_shuf, 1)), axis=1)]
            return torch.as_tensor(L, dtype=torch.float64, device=device)

        total = m * (m + 1) / 2 if (m := nx + ny) else 0
        # U_x and U_y from independent shuffles, as in the original code
        Ux = shuffled_labels() @ R - nx * (nx + 1) / 2
        Uy = total - shuffled_labels() @ R - ny * (ny + 1) / 2
        num += torch.minimum(Ux, Uy)
        den += nx * ny
    if den == 0:
        nan = np.full(n, np.nan)
        return (nan, nan) if return_stat else nan
    cp = num / den
    obs = cp[0]
    less = (cp[1:] < obs).sum(0)
    eq = (cp[1:] == obs).sum(0)
    rank = less + (eq + 2) / 2  # average rank of the observed value
    p = (rank / (n_shuf + 1)).cpu().numpy()
    return (p, (cp[1:].mean(0) - obs).cpu().numpy()) if return_stat else p


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------
def decode_binary(X, y, seed=0, n_folds=5):
    """Cross-validated balanced accuracy of an L2 logistic regression whose
    regularisation strength is chosen by an inner 3-fold CV."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    if np.bincount(y).min() < n_folds:
        return np.nan
    X = X[:, X.std(0) > 0]
    if X.shape[1] == 0:
        return np.nan
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(
            Cs=np.logspace(-4, 1, 6), cv=3, class_weight="balanced",
            scoring="balanced_accuracy", max_iter=1000))
        pred[te] = clf.fit(X[tr], y[tr]).predict(X[te])
    return balanced_accuracy_score(y, pred)


def decode_wheel(X, y, seed=0, n_folds=5):
    """Cross-validated R^2 of ridge regression of wheel speed.
    X: (n_trials, n_bins, n_neurons), y: (n_trials, n_bins); folds split by trial."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    nt, nb, nn = X.shape
    Xf, yf = X.reshape(nt * nb, nn), y.reshape(-1)
    keep = Xf.std(0) > 0
    if keep.sum() == 0:
        return np.nan
    Xf = Xf[:, keep]
    trial = np.repeat(np.arange(nt), nb)
    pred = np.empty_like(yf)
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(np.arange(nt)):
        itr, ite = np.isin(trial, tr), np.isin(trial, te)
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 7)))
        pred[ite] = m.fit(Xf[itr], yf[itr]).predict(Xf[ite])
    return r2_score(yf, pred)


# ---------------------------------------------------------------------------
# atlas outline
# ---------------------------------------------------------------------------
def top_boundaries(atlas=None, mapping="Beryl"):
    """Boolean top-view boundary image of `mapping` regions and its extent in
    mm ([x0, x1, y0, y1]) for imshow(origin='lower')."""
    from iblatlas.atlas import AllenAtlas
    ba = atlas or AllenAtlas()
    ba.compute_surface()
    ix, iy = np.meshgrid(np.arange(ba.bc.nx), np.arange(ba.bc.ny))
    iz = ba.bc.z2i(ba.top)
    inds = ba._lookup_inds(np.stack((ix, iy, iz), axis=-1))
    regions = ba._get_mapping(mapping=mapping)[ba.label.flat[inds]]
    bnd = ba.compute_boundaries(regions)
    bnd = np.where(np.isfinite(ba.top), bnd, 0)
    xs, ys = ba.bc.xscale * 1e3, ba.bc.yscale * 1e3
    return bnd, xs, ys
