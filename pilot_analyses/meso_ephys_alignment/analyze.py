#!/usr/bin/env python3
"""Alignment of task coding directions with the top spontaneous (grey-screen) PCs.

Implements the method fixed in HYPOTHESIS.md. Per session, region (>= N_NEURONS
neurons with non-zero grey-screen variance) and draw of N_NEURONS neurons:
z-score with grey-screen statistics, PCA of the grey screen, coding direction d per
task variable, alignment A(k) = ||U_k^T d||^2 / ||d||^2 and the same for N_SHIFTS
circularly shifted label sequences (noise null). Writes output/alignment.pqt
(one row per session x region x variable, draw-averaged) and output/kcurves.npz.
"""
import numpy as np
import pandas as pd
import torch
from common import (OUT, VARIABLES, N_NEURONS, N_DRAWS, K, K_MAX, N_SHIFTS, SUBJECT)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_CLASS = 20
EXCLUDE = {"root", "void", "nan", "None", ""}


def variable_sets(t):
    """Per variable: (trial index, binary label) in chronological order."""
    allt = np.arange(len(t))
    stim = np.where(t.contrast.values >= .25)[0]
    pl = t.probabilityLeft.values.astype(float)
    prior = np.where(np.isclose(pl, .2, atol=1e-3) | np.isclose(pl, .8, atol=1e-3))[0]
    sets = {"stim": (stim, t.stim_right.values[stim]),
            "choice": (allt, t.choice_right.values),
            "feedback": (allt, t.reward.values),
            "prior": (prior, np.isclose(pl[prior], .2, atol=1e-3).astype(int))}
    # a variable needs >= MIN_CLASS trials in each class (meso sessions here have no biased blocks)
    return {v: s for v, s in sets.items() if min(s[1].sum(), len(s[1]) - s[1].sum()) >= MIN_CLASS}


def shifted_weights(label, rng):
    """(1 + N_SHIFTS, T) weights giving mean(class 1) - mean(class 0); row 0 = real labels."""
    T = len(label)
    shifts = np.r_[0, rng.integers(int(.1 * T), int(.9 * T) + 1, N_SHIFTS)]
    L = np.stack([np.roll(label, s) for s in shifts]).astype(np.float32)
    n1 = L.sum(1, keepdims=True)
    return L / n1 - (1 - L) / (T - n1)


def alignment(grey, win, sets, rng):
    """Draw-wise A(k) for real and shifted labels: dict var -> (draws, 1+S, K_MAX), ||d||^2 (draws, 1+S)."""
    n = grey.shape[1]
    weights = {v: torch.from_numpy(shifted_weights(lab, rng)).to(DEV) for v, (_, lab) in sets.items()}
    out = {v: ([], []) for v in sets}
    for _ in range(N_DRAWS):
        idx = rng.choice(n, N_NEURONS, replace=False)
        g = torch.from_numpy(grey[:, idx]).to(DEV)
        mu, sd = g.mean(0), g.std(0)
        _, _, vt = torch.linalg.svd((g - mu) / sd, full_matrices=False)
        U = vt[:K_MAX].T                                                 # (N, K_MAX)
        for v, (tri, _) in sets.items():
            vi = VARIABLES.index(v)
            x = (torch.from_numpy(win[tri][:, vi][:, idx]).to(DEV) - mu) / sd   # (T, N)
            D = weights[v] @ x                                           # (1+S, N)
            tot = (D ** 2).sum(1)
            A = torch.cumsum((D @ U) ** 2, 1) / tot[:, None]             # (1+S, K_MAX)
            out[v][0].append(A.cpu().numpy())
            out[v][1].append(tot.cpu().numpy())
    return {v: (np.stack(a), np.stack(p)) for v, (a, p) in out.items()}


def main():
    rng = np.random.default_rng(0)
    rows, curves = [], {}
    for f in sorted((OUT / "features").glob("*.npz")):
        modality, eid = f.stem.split("_", 1)
        z = np.load(f, allow_pickle=True)
        grey, win, regions, classes = z["grey"].astype(np.float32), z["win"].astype(np.float32), z["regions"].astype(str), z["classes"].astype(str)
        t = pd.DataFrame(z["trials"])
        sets = variable_sets(t)
        live = grey.std(0) > 0
        for region in np.unique(regions):
            m = (regions == region) & live
            if region in EXCLUDE or m.sum() < N_NEURONS:
                continue
            res = alignment(grey[:, m], win[:, :, m], sets, rng)
            cls = classes[regions == region][0]
            for v, (A, P) in res.items():
                a_real = A[:, 0, K - 1].mean()
                a_null = A[:, 1:, K - 1].mean()
                coding_pct = (P[:, 1:] < P[:, :1]).mean(1)        # percentile of real ||d||^2 per draw
                rows.append(dict(modality=modality, eid=eid[:8], subject=SUBJECT[eid[:8]], region=region,
                                 cls=cls, n_available=int(m.sum()), variable=v, n_trials=len(sets[v][0]),
                                 A=a_real, A_null=a_null, R=a_real / a_null,
                                 null_sd=A[:, 1:, K - 1].std(1).mean(),
                                 coding_pct=np.median(coding_pct),
                                 coding=bool(np.median(coding_pct) > .95)))
                curves[(modality, eid[:8], region, v)] = (A[:, 0].mean(0), A[:, 1:].mean((0, 1)))
            print(f"{modality} {eid[:8]} {region:8s} {cls:7s} n={m.sum():5d} | " + "  ".join(
                f"{r['variable']}: R={r['R']:.2f}{'*' if r['coding'] else ' '}" for r in rows[-len(res):]), flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "alignment.pqt")
    keys = list(curves)
    np.savez(OUT / "kcurves.npz", keys=np.array(keys, dtype=object),
             real=np.stack([curves[k][0] for k in keys]), null=np.stack([curves[k][1] for k in keys]))


if __name__ == "__main__":
    main()
