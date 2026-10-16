"""Shared pieces for wheel_decoding2: paths, imports from dorsal_maps, depth
below the pial surface, and per-neuron wheel correlations."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DM = HERE.parent / "dorsal_maps"
REPO = HERE.parents[1]
for p in (str(REPO), str(DM)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import FEATURE_ROOT, WHEEL_EDGES  # noqa: E402,F401
from plotstyle import (NATURE_DOUBLE_COL_IN, NATURE_SINGLE_COL_IN, panel_label,  # noqa: E402,F401
                       save_with_caption, style_axes)
from sessions import load_session, session_files  # noqa: E402,F401

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
MODS = ("meso", "bwm")
NAME = {"meso": "Mesoscope", "bwm": "BWM (Neuropixels)"}
COL = {"meso": "tab:green", "bwm": "tab:purple"}
BIN_CENTRES = (WHEEL_EDGES[:-1] + WHEEL_EDGES[1:]) / 2

_ATLAS = None


def atlas():
    global _ATLAS
    if _ATLAS is None:
        from iblatlas.atlas import AllenAtlas
        _ATLAS = AllenAtlas()
        _ATLAS.compute_surface()
    return _ATLAS


def depth_below_surface(xyz_mm):
    """Depth (um) of each point below the pial surface (Allen top surface at
    the same ML-AP); NaN outside the brain."""
    ba = atlas()
    x, y, z = (np.asarray(xyz_mm, float) / 1e3).T
    ix = np.clip(ba.bc.x2i(x, mode="clip"), 0, ba.bc.nx - 1)
    iy = np.clip(ba.bc.y2i(y, mode="clip"), 0, ba.bc.ny - 1)
    return (ba.top[iy, ix] - z) * 1e6


def wheel_corr(X, y):
    """Pearson r of every neuron with wheel speed over all wheel bins.
    X (n, T, nb), y (T, nb) -> (n,), NaN for constant neurons."""
    Xf = X.reshape(len(X), -1).astype(np.float64)
    yf = np.asarray(y, np.float64).reshape(-1)
    Xc = Xf - Xf.mean(1, keepdims=True)
    yc = yf - yf.mean()
    den = np.linalg.norm(Xc, axis=1) * np.linalg.norm(yc)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, Xc @ yc / den, np.nan)
