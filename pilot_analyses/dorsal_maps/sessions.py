"""Iterate over extracted sessions (meso: one file per session; BWM: one file
per insertion, merged per session since probes of a session share trials)."""
from collections import defaultdict

import numpy as np

from common import FEATURE_ROOT

LABEL_KEYS = ("c_L", "c_R", "stim_left", "choice", "reward", "block")


def session_files(modality):
    files = sorted((FEATURE_ROOT / modality).glob("*.npz"))
    if modality == "meso":
        return {f.stem: [f] for f in files}
    groups = defaultdict(list)
    for f in files:
        with np.load(f) as z:
            groups[str(z["eid"])].append(f)
    return dict(groups)


def load_session(files):
    """Dict with xyz, atlas_id, X_task (n, T, 3), X_wheel (n, T, 6),
    wheel_y (T, 6) and the trial labels; neurons of all files concatenated."""
    zs = [dict(np.load(f)) for f in files]
    T = {len(z["choice"]) for z in zs}
    assert len(T) == 1, f"trial count differs across probes of one session: {files}"
    out = {k: zs[0][k] for k in LABEL_KEYS + ("wheel_y",)}
    for k in ("xyz", "atlas_id", "X_task", "X_wheel"):
        out[k] = np.concatenate([z[k] for z in zs])
    return out
