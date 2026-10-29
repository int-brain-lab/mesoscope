#!/usr/bin/env python3
"""Which mesoscope trials fall outside the imaging?

extract_meso interpolates each ROI's trace at the window times; before the
first or after the last frame it repeats the edge value, and across gaps in
mpci.times it bridges linearly, so trials there carry no real activity.
For every meso session: the frame times of each FOV (alf/FOV_xx/mpci.times)
and the trial windows actually used (common.task_intervals on the
trial_mask trials); a trial is covered when every window (task windows and
wheel bins), widened by MAX_GAP on each side, lies within the imaging and
contains no inter-frame gap longer than MAX_GAP, for all FOVs.

Usage: python imaging_coverage.py -> output/imaging_coverage.csv and
FEATURE_ROOT/meso_trial_coverage/<eid>.npy (bool per feature-file trial)
"""
import numpy as np
import pandas as pd

from common import FEATURE_ROOT, OUT, task_intervals, trial_mask
from sessions import load_session, session_files

MAX_GAP = 1.0  # s; frame period is ~0.2 s


def covered(frame_times, iv):
    """iv (n_trials, n_windows, 2) -> bool (n_trials,)"""
    t = np.asarray(frame_times, float)
    lo, hi = iv[..., 0] - MAX_GAP, iv[..., 1] + MAX_GAP
    ok = (lo >= t[0]) & (hi <= t[-1])
    gaps = np.flatnonzero(np.diff(t) > MAX_GAP)
    for g in gaps:  # a window overlapping a gap is not covered
        ok &= ~((iv[..., 0] < t[g + 1]) & (iv[..., 1] > t[g]))
    return ok.all(-1)


def main():
    import sys
    from one.api import ONE
    sys.path.insert(0, str(OUT.parents[1] / "wheel_decoding2"))
    from dorsal_maps_tasks import task_collection_for
    one = ONE()
    dst = FEATURE_ROOT / "meso_trial_coverage"
    dst.mkdir(exist_ok=True)
    rows = []
    for sid, f in session_files("meso").items():
        s = load_session(f)
        col = task_collection_for(one, sid)
        trials = one.load_object(sid, "trials", collection=col)
        iv = task_intervals(trials, trial_mask(trials), "meso")
        if len(iv) != len(s["choice"]) or len(iv) == 0:
            rows.append(dict(session=sid, n_trials=len(s["choice"]), mismatch=True))
            continue
        fovs = sorted({d.split("/")[1] for d in one.list_datasets(sid, collection="alf/FOV*")
                       if d.endswith("mpci.times.npy")})
        ok = np.ones(len(iv), bool)
        t0, t1, gmax = [], [], 0.0
        for fov in fovs:
            t = one.load_dataset(sid, "mpci.times.npy", collection=f"alf/{fov}")
            ok &= covered(t, iv)
            t0.append(t[0]); t1.append(t[-1]); gmax = max(gmax, float(np.diff(t).max()))
        np.save(dst / f"{sid}.npy", ok)
        rows.append(dict(session=sid, n_trials=len(ok), n_uncovered=int((~ok).sum()), frac_uncovered=1 - ok.mean(),
                         first_trial=float(iv[0, 0, 0]), last_trial=float(iv[-1, -1, 1]), imaging_start=max(t0),
                         imaging_end=min(t1), max_frame_gap=gmax, mismatch=False))
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "imaging_coverage.csv", index=False)
    bad = d[d.n_uncovered > 0]
    print(f"{len(bad)} of {len(d)} sessions have uncovered trials; "
          f"{int(d.n_uncovered.sum())} of {int(d.n_trials.sum())} trials")


if __name__ == "__main__":
    main()
