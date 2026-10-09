#!/usr/bin/env python3
"""Single-neuron tuning p-values (stim / choice / feedback) for every
extracted neuron of both modalities, using the same test
(common.tuning_pvalues) on each modality's window rates.

Writes output/tuning_<modality>.pqt: one row per neuron with session,
x/y/z (mm), atlas_id and p_stim / p_choice / p_feedback.
"""
import sys

import numpy as np
import pandas as pd

from common import OUT, VARIABLES, tuning_pvalues
from sessions import load_session, session_files


def run(modality):
    rows = []
    groups = session_files(modality)
    for i, (sid, files) in enumerate(groups.items()):
        s = load_session(files)
        df = pd.DataFrame({"session": sid, "x": s["xyz"][:, 0], "y": s["xyz"][:, 1],
                           "z": s["xyz"][:, 2], "atlas_id": s["atlas_id"]})
        for j, v in enumerate(VARIABLES):
            df[f"p_{v}"] = tuning_pvalues(s["X_task"][:, :, j].T, s, v, seed=i)
        rows.append(df)
        print(modality, i + 1, len(groups), sid, len(df), flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT / f"tuning_{modality}.pqt")
    return out


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["meso", "bwm"]):
        run(m)
