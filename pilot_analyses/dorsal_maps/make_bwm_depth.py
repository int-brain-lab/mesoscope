#!/usr/bin/env python3
"""Depth-limited BWM features: keep only units within MAX_DEPTH_UM of the
pial surface, the depth range the mesoscope could plausibly image.

The mesoscope images 174-318 um below the surface (5th-95th percentile of
the 24 sessions with histology-registered ROI positions, median 257 um;
sessions with only estimated positions are projected onto the surface and
carry no depth). The BWM "voxel" below each hexagon is taken twice as deep,
0-640 um, to keep enough units while excluding deep layers 5-6. Depth = Allen
top surface at the unit's ML-AP minus its DV.

Writes FEATURE_ROOT/bwm_d640/<pid>.npz (same format as bwm/, units filtered;
probes without such units are skipped), so every script can use modality
"bwm_d640".

Usage: python make_bwm_depth.py
"""
import numpy as np

from common import FEATURE_ROOT

MAX_DEPTH_UM = 640
UNIT_KEYS = ("xyz", "atlas_id", "X_task", "X_wheel")


def depth_below_surface(xyz_mm, ba):
    x, y, z = (np.asarray(xyz_mm, float) / 1e3).T
    ix = np.clip(ba.bc.x2i(x, mode="clip"), 0, ba.bc.nx - 1)
    iy = np.clip(ba.bc.y2i(y, mode="clip"), 0, ba.bc.ny - 1)
    return (ba.top[iy, ix] - z) * 1e6


def main(src="bwm", dst=f"bwm_d{MAX_DEPTH_UM}"):
    from iblatlas.atlas import AllenAtlas
    ba = AllenAtlas()
    ba.compute_surface()
    out = FEATURE_ROOT / dst
    out.mkdir(exist_ok=True)
    n_in = n_out = n_files = 0
    for f in sorted((FEATURE_ROOT / src).glob("*.npz")):
        z = dict(np.load(f))
        d = depth_below_surface(z["xyz"], ba)
        keep = d <= MAX_DEPTH_UM
        n_in += len(keep)
        if not keep.any():
            continue
        for k in UNIT_KEYS:
            z[k] = z[k][keep]
        z["depth_um"] = d[keep].astype(np.float32)
        np.savez_compressed(out / f.name, **z)
        n_out += keep.sum()
        n_files += 1
    print(f"{dst}: kept {n_out} of {n_in} units ({n_out / n_in:.1%}) on {n_files} probes")


if __name__ == "__main__":
    main()
