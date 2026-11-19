#!/usr/bin/env python3
"""Cache BWM features for units within 640 µm of local pia.

The default uses the IBL Allen CCF streamline lookup and writes bwm_pial640.
For historical reproducibility, --vertical uses the former dorsal DV offset
and writes bwm_d640.

Usage: python make_bwm_depth.py [--streamlines | --vertical]
"""
import numpy as np

from common import FEATURE_ROOT

MAX_DEPTH_UM = 640
STREAMLINE_MODALITY = f"bwm_pial{MAX_DEPTH_UM}"
UNIT_KEYS = ("xyz", "atlas_id", "X_task", "X_wheel")


def vertical_depth_um(xyz_mm, ba):
    x, y, z = (np.asarray(xyz_mm, float) / 1e3).T
    ix = np.clip(ba.bc.x2i(x, mode="clip"), 0, ba.bc.nx - 1)
    iy = np.clip(ba.bc.y2i(y, mode="clip"), 0, ba.bc.ny - 1)
    return (ba.top[iy, ix] - z) * 1e6


def streamline_depth_um(xyz_mm):
    """Local pial depth using IBL's 25-micron Allen CCF cortical streamline lookup."""
    from iblatlas.streamlines.utils import xyz_to_depth

    return xyz_to_depth(np.asarray(xyz_mm, dtype=float) / 1e3, per=False, res_um=25)


def main(src="bwm", dst=None, *, streamlines=True):
    if dst is None:
        dst = STREAMLINE_MODALITY if streamlines else f"bwm_d{MAX_DEPTH_UM}"
    files = sorted((FEATURE_ROOT / src).glob("*.npz"))
    if not files:
        raise ValueError(f"No BWM features in {FEATURE_ROOT / src}")
    positions = []
    for f in files:
        with np.load(f) as z:
            positions.append(z["xyz"])
    xyz = np.concatenate(positions)
    if streamlines:
        depth = streamline_depth_um(xyz)
    else:
        from iblatlas.atlas import AllenAtlas

        ba = AllenAtlas()
        ba.compute_surface()
        depth = vertical_depth_um(xyz, ba)
    by_probe = np.split(depth, np.cumsum([len(x) for x in positions])[:-1])
    out = FEATURE_ROOT / dst
    out.mkdir(exist_ok=True)
    n_out = n_files = 0
    for f, d in zip(files, by_probe):
        keep = d <= MAX_DEPTH_UM
        target = out / f.name
        if not keep.any():
            target.unlink(missing_ok=True)
            continue
        with np.load(f) as source:
            z = {k: source[k][keep] if k in UNIT_KEYS else source[k] for k in source.files}
        z["depth_um"] = d[keep].astype(np.float32)
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **z)
        tmp.replace(target)
        n_out += keep.sum()
        n_files += 1
    print(f"{dst}: kept {n_out} of {len(depth)} units ({n_out / len(depth):.1%}) on {n_files} probes")
    if streamlines:
        print(f"No streamline depth for {(~np.isfinite(depth)).sum()} units.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    method = parser.add_mutually_exclusive_group()
    method.add_argument("--streamlines", dest="streamlines", action="store_true",
                        help="Use local pial depth from IBL cortical streamlines (default).")
    method.add_argument("--vertical", dest="streamlines", action="store_false",
                        help="Reproduce the historical dorsal vertical cutoff.")
    parser.set_defaults(streamlines=True)
    main(streamlines=parser.parse_args().streamlines)
