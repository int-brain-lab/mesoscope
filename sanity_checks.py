"""Lightweight sanity checks for meso_loader.py.

These are smoke tests to run against a real session after loading it,
to catch alignment/loader bugs early -- not a full test suite and not a
scientific claim about any specific brain region.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import wilcoxon
from one.api import ONE

from meso_loader import MesoscopeSession, load_mesoscope_session


def _window_population_mean(session: MesoscopeSession, times: np.ndarray, lo: float, hi: float) -> Optional[float]:
    i0 = np.searchsorted(times, lo, side="left")
    i1 = np.searchsorted(times, hi, side="left")
    if i1 <= i0:
        return None
    return float(session.roi_signal[:, i0:i1].mean())


def check_motion_onset_response(
    eid: str,
    one: Optional[ONE] = None,
    session: Optional[MesoscopeSession] = None,
    n_trials: int = 20,
    baseline_window: tuple[float, float] = (-0.3, 0.0),
    response_window: tuple[float, float] = (0.0, 0.3),
    seed: Optional[int] = 0,
) -> dict:
    """Check that population activity rises after movement onset.

    For a random sample of trials, compares mean population ROI activity
    in a window right before `firstMovement_times` (baseline) to a window
    right after (response). Movement onset is reliably accompanied by a
    mesoscope-wide rise in activity (motor/arousal signals), so a
    correctly loaded and time-aligned session should show
    response > baseline for a clear majority of the sampled trials.

    Uses `session.roi_times[0]` as a shared approximate time axis (each
    ROI's true offset is sub-frame, negligible over a ~0.3 s window) --
    same approximation used elsewhere in this repo (see
    `meso.compute_sparseness`).

    Parameters
    ----------
    eid : str
    one : ONE, optional
    session : MesoscopeSession, optional
        Pass an already-loaded session to avoid reloading.
    n_trials : int, default 20
        Number of trials to randomly sample (capped by trials available).
    baseline_window, response_window : (float, float)
        Seconds relative to `firstMovement_times`.
    seed : int, optional
        RNG seed for trial sampling; None for nondeterministic.

    Returns
    -------
    dict with keys: eid, n_trials_used, trial_indices, baseline_means,
    response_means, frac_increased, mean_diff, wilcoxon_stat, wilcoxon_p, passed
    """
    one = one if one is not None else ONE()
    session = session if session is not None else load_mesoscope_session(eid, one=one)

    trials = one.load_object(eid, "trials")
    move_times = np.asarray(trials["firstMovement_times"], dtype=float)
    valid = np.isfinite(move_times)
    move_times = move_times[valid]
    trial_idx = np.where(valid)[0]
    if move_times.size == 0:
        raise ValueError(f"No trials with finite firstMovement_times for {eid}")

    rng = np.random.default_rng(seed)
    n_pick = min(n_trials, move_times.size)
    picked = rng.choice(move_times.size, size=n_pick, replace=False)

    times = np.asarray(session.roi_times[0], dtype=float)

    baseline_means, response_means, used_trials = [], [], []
    for i in picked:
        t0 = move_times[i]
        b = _window_population_mean(session, times, t0 + baseline_window[0], t0 + baseline_window[1])
        r = _window_population_mean(session, times, t0 + response_window[0], t0 + response_window[1])
        if b is None or r is None:
            continue
        baseline_means.append(b)
        response_means.append(r)
        used_trials.append(int(trial_idx[i]))

    baseline_means = np.array(baseline_means)
    response_means = np.array(response_means)
    n_used = baseline_means.size
    if n_used < 5:
        raise ValueError(f"Only {n_used} usable trials for {eid}; need >=5 for a meaningful check")

    diffs = response_means - baseline_means
    frac_increased = float((diffs > 0).mean())
    stat, p = wilcoxon(response_means, baseline_means)

    result = dict(
        eid=eid,
        n_trials_used=n_used,
        trial_indices=used_trials,
        baseline_means=baseline_means,
        response_means=response_means,
        frac_increased=frac_increased,
        mean_diff=float(diffs.mean()),
        wilcoxon_stat=float(stat),
        wilcoxon_p=float(p),
        passed=bool(frac_increased > 0.5 and p < 0.05 and diffs.mean() > 0),
    )

    print(
        f"[motion-onset check] {eid}: {n_used} trials, "
        f"{frac_increased:.0%} increased, mean diff={result['mean_diff']:.4g}, "
        f"wilcoxon p={p:.3g} -> {'PASS' if result['passed'] else 'FAIL'}"
    )
    return result


if __name__ == "__main__":
    import sys

    eid = sys.argv[1] if len(sys.argv) > 1 else None
    if eid is None:
        raise SystemExit("usage: python sanity_checks.py <eid>")
    check_motion_onset_response(eid)
