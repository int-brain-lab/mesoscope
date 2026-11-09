#!/usr/bin/env python3
"""Single-neuron tuning p-values (stim / choice / feedback) for every
extracted neuron of both modalities, using the same test
(common.tuning_pvalues) on each modality's window rates.

Writes output/tuning_<modality>.pqt: one row per neuron with session,
x/y/z (mm), atlas_id, n_trials, p_<variable> and the tuning strength
u_<variable> (separation of the two conditions above its shuffle level,
see common.tuning_pvalues).

With --min-contrast c only stimulus side is tested, on the trials with
contrast >= c (output tuning_<modality>_c<c*100>.pqt).
"""
import numpy as np
import pandas as pd

from common import OUT, VARIABLES, tuning_pvalues
from sessions import load_session, session_files


def run(modality, features="", min_contrast=0.0):
    rows = []
    variables = ("stim",) if min_contrast else VARIABLES
    groups = session_files(modality)
    for i, (sid, files) in enumerate(groups.items()):
        s = load_session(files, features)
        df = pd.DataFrame({"session": sid, "x": s["xyz"][:, 0], "y": s["xyz"][:, 1],
                           "z": s["xyz"][:, 2], "atlas_id": s["atlas_id"],
                           "n_trials": len(s["choice"])})
        keep = np.maximum(s["c_L"], s["c_R"]) >= min_contrast
        lab = {k: s[k][keep] for k in ("c_L", "c_R", "choice", "reward", "block")}
        df["n_trials_test"] = keep.sum()
        for v in variables:
            j = VARIABLES.index(v)
            df[f"p_{v}"], df[f"u_{v}"] = tuning_pvalues(s["X_task"][:, keep, j].T, lab, v, seed=i,
                                                        return_stat=True)
        rows.append(df)
        print(modality, i + 1, len(groups), sid, len(df), flush=True)
    out = pd.concat(rows, ignore_index=True)
    tag = f"_c{round(min_contrast * 100)}" if min_contrast else ""
    out.to_parquet(OUT / f"tuning_{modality}{features}{tag}.pqt")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("modality", nargs="*", default=["meso", "bwm"])
    ap.add_argument("--features", default="", help='feature suffix, e.g. "_fr"')
    ap.add_argument("--min-contrast", type=float, default=0.0,
                    help="test stimulus side only, on trials with contrast >= this")
    a = ap.parse_args()
    for m in a.modality:
        run(m, a.features, a.min_contrast)
