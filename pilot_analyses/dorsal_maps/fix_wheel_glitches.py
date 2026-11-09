#!/usr/bin/env python3
"""Recompute the wheel-speed target (wheel_y) of feature files whose wheel
data contain encoder glitches, with the corrected common.wheel_speed.

16 of 153 mesoscope sessions have wheel positions that jump by exactly
+-6588397.3 rad (counter overflow) for single samples; the old wheel_speed
turned these into bins of ~3e7 rad/s (36 bins), which dominate the variance
of wheel speed in those sessions and make their R^2 meaningless. Only
wheel_y is rewritten; neural features are unchanged. Files are patched when
any stored bin exceeds 100 rad/s (no BWM file does; maximum 11 rad/s).

Usage: python fix_wheel_glitches.py [modality ...]   (default: meso)
"""
import sys

import numpy as np

from common import FEATURE_ROOT, task_intervals, trial_mask, wheel_speed


def main(modalities):
    from one.api import ONE
    from extract_meso import task_collection
    one = ONE()
    for m in modalities:
        for f in sorted((FEATURE_ROOT / m).glob("*.npz")):
            z = dict(np.load(f))
            if z["wheel_y"].size == 0 or z["wheel_y"].max() <= 100:
                continue
            eid = str(z["eid"])
            col = task_collection(one, eid) if m == "meso" else "alf"
            trials = one.load_object(eid, "trials", collection=col)
            mask = trial_mask(trials)
            iv = task_intervals(trials, mask, "meso" if m == "meso" else "bwm")[:, 3:]
            w = one.load_object(eid, "wheel", collection=col)
            wy = wheel_speed(w["timestamps"], w["position"], iv).astype(np.float32)
            assert wy.shape == z["wheel_y"].shape, (f, wy.shape, z["wheel_y"].shape)
            ok = z["wheel_y"] <= 100
            r = np.corrcoef(wy[ok], z["wheel_y"][ok])[0, 1]
            print(f"{m} {eid}: {int((~ok).sum())} glitch bins, old max {z['wheel_y'].max():.3g}, new max "
                  f"{wy.max():.2f}; r with old on clean bins {r:.4f}", flush=True)
            z["wheel_y"] = wy
            np.savez_compressed(f, **z)


if __name__ == "__main__":
    main(sys.argv[1:] or ["meso"])
