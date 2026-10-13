#!/usr/bin/env python3
"""Single-neuron tuning p-values (stim / choice / feedback) for every
extracted neuron of both modalities, using the same test
(common.tuning_pvalues) on each modality's window rates.

Writes output/tuning_<modality>.pqt: one row per neuron with session,
x/y/z (mm), atlas_id, n_trials, p_<variable> and the tuning strength
u_<variable> (separation of the two conditions above its shuffle level,
see common.tuning_pvalues).
"""
import numpy as np
import pandas as pd

from common import OUT, VARIABLES, tuning_pvalues
from sessions import load_session, session_files


def run(modality, features=""):
    rows = []
    groups = session_files(modality)
    for i, (sid, files) in enumerate(groups.items()):
        s = load_session(files, features)
        df = pd.DataFrame({"session": sid, "x": s["xyz"][:, 0], "y": s["xyz"][:, 1],
                           "z": s["xyz"][:, 2], "atlas_id": s["atlas_id"],
                           "n_trials": len(s["choice"])})
        for j, v in enumerate(VARIABLES):
            df[f"p_{v}"], df[f"u_{v}"] = tuning_pvalues(s["X_task"][:, :, j].T, s, v, seed=i,
                                                        return_stat=True)
        rows.append(df)
        print(modality, i + 1, len(groups), sid, len(df), flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT / f"tuning_{modality}{features}.pqt")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("modality", nargs="*", default=["meso", "bwm"])
    ap.add_argument("--features", default="", help='feature suffix, e.g. "_fr"')
    a = ap.parse_args()
    for m in a.modality:
        run(m, a.features)
