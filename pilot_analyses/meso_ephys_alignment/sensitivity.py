#!/usr/bin/env python3
"""Sensitivity checks added in review round 1 (not pre-registered).

1. BWM grey-screen PCs from 0.1-s bins instead of 1-s bins (closer to the 0.1-0.2 s
   task windows).
2. BWM units with a grey-screen rate below 1 Hz excluded (their z-scores are inflated
   by a small grey-screen s.d.).
3. Meso with N = 200 neurons per draw (regions with >= 200 neurons), to use more of
   the meso population.
Writes output/sensitivity.pqt with the same columns as alignment.pqt plus `check`.
"""
import numpy as np
import pandas as pd
import analyze as an
from common import OUT, K, SUBJECT
from extract import bwm_units, grey_interval
from one.api import ONE


def run(features, check, n_neurons=None, keep=None):
    rng = np.random.default_rng(0)
    rows = []
    for (modality, eid), (grey, win, regions, classes, t) in features.items():
        sets = an.variable_sets(t)
        live = grey.std(0) > 0
        if keep is not None:
            live &= keep[(modality, eid)]
        n_req = n_neurons or an.N_NEURONS
        for region in np.unique(regions):
            m = (regions == region) & live
            if region in an.EXCLUDE or m.sum() < n_req:
                continue
            an.N_NEURONS = n_req
            res = an.alignment(grey[:, m], win[:, :, m], sets, rng)
            for v, (A, P) in res.items():
                a_real, a_null = A[:, 0, K - 1].mean(), A[:, 1:, K - 1].mean()
                pct = np.median((P[:, 1:] < P[:, :1]).mean(1))
                rows.append(dict(check=check, modality=modality, eid=eid[:8], subject=SUBJECT[eid[:8]],
                                 region=region, cls=classes[regions == region][0], variable=v,
                                 A=a_real, A_null=a_null, R=a_real / a_null, coding_pct=pct, coding=pct > .95))
    an.N_NEURONS = 30
    return rows


def load(modality_filter=None):
    feats = {}
    for f in sorted((OUT / "features").glob("*.npz")):
        modality, eid = f.stem.split("_", 1)
        if modality_filter and modality != modality_filter:
            continue
        z = np.load(f, allow_pickle=True)
        feats[(modality, eid)] = (z["grey"].astype(np.float32), z["win"].astype(np.float32),
                                  z["regions"].astype(str), z["classes"].astype(str), pd.DataFrame(z["trials"]))
    return feats


def main():
    rows = []
    bwm = load("bwm")
    # 1. 0.1-s grey bins
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
    fine = {}
    for (modality, eid), (grey, win, regions, classes, t) in bwm.items():
        trains, _ = bwm_units(eid)
        g0, g1 = grey_interval(one, eid)
        edges = np.arange(g0, g1 + 1e-9, .1)
        g = (np.stack([np.histogram(s, edges)[0] for s in trains]).T / .1).astype(np.float32)
        fine[(modality, eid)] = (g, win, regions, classes, t)
    rows += run(fine, "bwm_grey_0.1s")
    # 2. grey rate >= 1 Hz
    keep = {k: v[0].mean(0) >= 1. for k, v in bwm.items()}
    rows += run(bwm, "bwm_rate>=1Hz", keep=keep)
    # 3. meso N = 200
    rows += run(load("meso"), "meso_N200", n_neurons=200)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "sensitivity.pqt")
    d = df[df.cls != "other"]
    print(d.groupby(["check", "variable", "cls"]).R.median().round(2).unstack("cls"))
    print(d[d.coding].groupby(["check", "variable", "cls"]).R.median().round(2).unstack("cls"))




def n_sweep():
    """Meso R as a function of N (regions with >= 200 neurons, so the same regions at every N)."""
    feats = load("meso")
    rows = []
    for n in (30, 50, 100, 200):
        for (modality, eid), (grey, win, regions, classes, t) in feats.items():
            big = np.isin(regions, [r for r in np.unique(regions) if (regions == r).sum() >= 200])
            rows += [dict(r, N=n) for r in run({(modality, eid): (grey, win, np.where(big, regions, "void"),
                                                                    classes, t)}, f"meso_N{n}", n_neurons=n)]
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "n_sweep.pqt")
    d = df[df.cls != "other"]
    print(d.groupby(["variable", "cls", "N"]).R.median().round(2).unstack("N"))
    print(d.groupby(["variable", "cls", "N"]).A_null.median().round(3).unstack("N"))


if __name__ == "__main__":
    import sys
    n_sweep() if "--n-sweep" in sys.argv else main()
