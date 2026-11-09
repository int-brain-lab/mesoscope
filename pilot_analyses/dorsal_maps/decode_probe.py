#!/usr/bin/env python3
"""BWM decoding per probe instead of per bin, for the extra row of v2_decoding_all.

Per insertion (one feature file of FEATURE_ROOT/bwm): all isocortex units of the
probe at all depths, whatever bin they fall in, decoded with the same decoders as
decode_v2 (stimulus side, choice, feedback, wheel speed, trial-specific wheel
speed); probes need >= N_PICK units and the session >= MIN_TRIALS trials.
The probe is placed at its brain entry point: x, y of the Alyx trajectory
(provenance 'Ephys aligned histology track', falling back to 'Histology track'
and 'Micro-manipulator'), um from bregma.

Usage: python decode_probe.py
Writes output/v2/decoding_bwm_probe.pqt (pid, session, entry_x, entry_y [mm], provenance,
n_neurons, n_trials, target, all).
"""
import warnings

import numpy as np
import pandas as pd

from common import FEATURE_ROOT
from decode_v2 import MIN_TRIALS, N_PICK, OUT_V2, decode_sets
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
    return pd.DataFrame.from_dict(rows, orient="index")


def run():
    warnings.filterwarnings("ignore")
    files = sorted((FEATURE_ROOT / "bwm").glob("*.npz"))
    res = []
    for i, f in enumerate(files):
        s = load_session([f])
        n = len(s["xyz"])
        if n < N_PICK or len(s["choice"]) < MIN_TRIALS:
            continue
        sc = decode_sets(s, [np.arange(n)])
        eid = str(np.load(f)["eid"])
        res += [dict(pid=f.stem, session=eid, n_neurons=n, n_trials=len(s["choice"]), target=t, all=v[0])
                for t, v in sc.items()]
        print(i + 1, len(files), f.stem, n, flush=True)
    d = pd.DataFrame(res)
    ep = entry_points(d.pid.unique().tolist())
    d = d.join(ep, on="pid")
    d.to_parquet(OUT_V2 / "decoding_bwm_probe.pqt")
    return d


if __name__ == "__main__":
    d = run()
    print(d.groupby("target")["all"].describe())
    print(d.drop_duplicates("pid").provenance.value_counts(dropna=False))
