#!/usr/bin/env python3
"""Control: does recording depth explain the BWM vs mesoscope differences?

Mesoscope neurons sit in layers 1-2/3; most BWM isocortex units are in
layers 5-6. Within each session, decode every target from MIN_NEURONS
randomly drawn neurons of one layer group (N_DRAWS draws, scores averaged),
ignoring spatial bins, so groups are compared within the same sessions.
Layer groups come from the Allen layer of each neuron's region label:
superficial = L1 + L2/3, L4, deep = L5 + L6a + L6b. Mesoscope sessions are
decoded the same way from all their neurons ("all").

Usage: python layers.py bwm [bwm_w03 [--features _fr]] meso
Writes output/layers_<modality><features>.pqt (session, group, n_neurons,
target, score).
"""
import argparse
import re
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from common import MIN_NEURONS, OUT
from decode import MIN_TRIALS, TARGETS, decode_sets
from sessions import load_session, session_files

N_DRAWS = 10
GROUPS = {"superficial": ("1", "2/3"), "L4": ("4",), "deep": ("5", "6a", "6b")}


def layer_of(atlas_ids):
    from iblatlas.regions import BrainRegions
    acr = BrainRegions().id2acronym(atlas_ids)
    return np.array([re.sub(r"^[A-Za-z-]+?(?=\d|$)", "", a) for a in acr])


def group_indices(s, modality):
    if modality == "meso":
        return {"all": np.arange(len(s["xyz"]))}
    lay = layer_of(s["atlas_id"])
    return {g: np.flatnonzero(np.isin(lay, ls)) for g, ls in GROUPS.items()}


def run(modality, features=""):
    warnings.filterwarnings("ignore")
    groups = list(session_files(modality).items())
    rows = []
    with ThreadPoolExecutor(2) as pool:
        loaded = pool.map(lambda g: load_session(g[1], features), groups)
        for i, ((sid, _), s) in enumerate(zip(groups, loaded)):
            if len(s["choice"]) < MIN_TRIALS:
                continue
            rng = np.random.default_rng(zlib.crc32(sid.encode()))
            names, sets = [], []
            for g, idx in group_indices(s, modality).items():
                if idx.size >= MIN_NEURONS:
                    names.append((g, idx.size))
                    sets += [rng.choice(idx, MIN_NEURONS, replace=False) for _ in range(N_DRAWS)]
            if not sets:
                continue
            sc = decode_sets(s, sets)
            for j, (g, n) in enumerate(names):
                for t in TARGETS:
                    rows.append(dict(session=sid, group=g, n_neurons=n, target=t,
                                     score=np.nanmean(sc[t][j * N_DRAWS:(j + 1) * N_DRAWS])))
            print(modality, features, i + 1, len(groups), sid, [g for g, _ in names], flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(OUT / f"layers_{modality}{features}.pqt")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modality", nargs="+")
    ap.add_argument("--features", default="")
    a = ap.parse_args()
    for m in a.modality:
        run(m, a.features)
