#!/usr/bin/env python3
"""Decode BWM activity per probe and attach its insertion coordinates.

Use all eligible isocortex units or ten reproducible draws of 20 units on
each probe. Coordinates are Alyx trajectory entry points, preferring aligned
histology, then histology, then micro-manipulator records.

Usage: python decode_probe.py [bwm bwm_d640 bwm_pial640]
Writes output/v2/decoding_<modality>_probe_auc.pqt.
"""
import argparse
import warnings
import zlib

import numpy as np
import pandas as pd

from common import FEATURE_ROOT
from decode_v2 import BINARY, MIN_TRIALS, N_DRAWS, N_PICK, OUT_V2, decode_sets
from make_bwm_depth import STREAMLINE_MODALITY
from sessions import load_session

PROVENANCE = ("Ephys aligned histology track", "Histology track", "Micro-manipulator")


def entry_points(pids):
    from one.api import ONE
    one = ONE(base_url="https://alyx.internationalbrainlab.org")
    rows = {}
    for prov in PROVENANCE:
        todo = [p for p in pids if p not in rows]
        for i in range(0, len(todo), 50):
            for t in one.alyx.rest("trajectories", "list", provenance=prov,
                                   django=f"probe_insertion__in,{todo[i:i + 50]}"):
                rows.setdefault(t["probe_insertion"], dict(entry_x=t["x"] / 1e3, entry_y=t["y"] / 1e3,
                                                           provenance=prov))
    return pd.DataFrame.from_dict(rows, orient="index",
                                  columns=["entry_x", "entry_y", "provenance"])


def probe_output(modality, *, legacy=False):
    suffix = "" if legacy else "_auc"
    return OUT_V2 / f"decoding_{modality}_probe{suffix}.pqt"


def run(modality="bwm", *, cached=None):
    warnings.filterwarnings("ignore")
    files = sorted((FEATURE_ROOT / modality).glob("*.npz"))
    res = []
    for i, f in enumerate(files):
        s = load_session([f])
        n = len(s["xyz"])
        if n < N_PICK or len(s["choice"]) < MIN_TRIALS:
            continue
        rng = np.random.default_rng(zlib.crc32(f.stem.encode()))
        units = np.arange(n)
        sets = [units] + [rng.choice(units, N_PICK, replace=False) for _ in range(N_DRAWS)]
        scores = decode_sets(s, sets, wheel=cached is None)
        with np.load(f) as z:
            eid = str(z["eid"])
        res.extend(dict(pid=f.stem, session=eid, n_neurons=n, n_trials=len(s["choice"]),
                        target=t, all=v[0], n20=np.nanmean(v[1:])) for t, v in scores.items())
        print(i + 1, len(files), f.stem, n, flush=True)
    if not res:
        raise ValueError(f"No eligible probe features for {modality} in {FEATURE_ROOT / modality}")
    d = pd.DataFrame(res)
    if cached is None:
        ep = entry_points(d.pid.unique().tolist())
    else:
        ep = cached.drop_duplicates("pid").set_index("pid")[["entry_x", "entry_y", "provenance"]]
        d = pd.concat([d, cached[~cached.target.isin(BINARY)].drop(columns=ep.columns)],
                      ignore_index=True)
    d = d.join(ep, on="pid")
    if d.entry_x.isna().all():
        raise ValueError(f"No Alyx insertion positions returned for {modality}")
    OUT_V2.mkdir(parents=True, exist_ok=True)
    d.to_parquet(probe_output(modality))
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modality", nargs="*", default=["bwm"])
    args = parser.parse_args()
    for modality in args.modality:
        if modality not in ("bwm", "bwm_d640", STREAMLINE_MODALITY):
            parser.error(f"unsupported BWM modality: {modality}")
        d = run(modality)
        print(d.groupby("target")[["all", "n20"]].describe())
        print(d.drop_duplicates("pid").provenance.value_counts(dropna=False))
