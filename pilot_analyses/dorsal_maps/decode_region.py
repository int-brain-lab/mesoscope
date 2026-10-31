#!/usr/bin/env python3
"""Decoding per region instead of per bin, for the region / session variance figure
(region_variance.py).

Mesoscope: per session and Beryl region, all isocortex neurons of the session in
that region (>= N_PICK), hemispheres pooled, same decoders and trial handling as
decode_v2 (coverage, excluded sessions).
BWM: the full-probe decodes of decode_probe.py; each probe gets the Beryl region
at its brain entry point (histology-aligned trajectory), read off the atlas
ENTRY_DEPTH_UM below the cortical surface. The majority Beryl region of the
probe's isocortex units within 640 um is stored as a check.

Shuffle control (--shuffle): the same decoders with the trials of all labels and of wheel
speed permuted jointly (neural data untouched), N_SHUFFLE permutations per set; columns
shuf_0.. in output/v2/decoding_{meso_region,bwm_probe}_shuf.pqt.

Usage: python decode_region.py [--shuffle [--meso-only]]
Writes output/v2/decoding_meso_region.pqt and output/v2/decoding_bwm_probe_region.pqt.
"""
import warnings
import zlib

import numpy as np
import pandas as pd

from decode_v2 import (MIN_TRIALS, N_PICK, OUT_V2, apply_coverage, decode_sets,
                       excluded_sessions)
from sessions import load_session, session_files

ENTRY_DEPTH_UM = 300
N_SHUFFLE = 1
LABELS = ("stim_left", "choice", "reward", "wheel_y")


def decode_shuffled(s, sets, seed):
    """{target: (n_sets, N_SHUFFLE)} scores with trials of labels and wheel speed permuted."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_SHUFFLE):
        perm = rng.permutation(len(s["choice"]))
        out.append(decode_sets(dict(s, **{k: s[k][perm] for k in LABELS}), sets))
    return {t: np.stack([o[t] for o in out], axis=1) for t in out[0]}


def shuffled_rows(sc, keys):
    return [dict(**key, target=t, **{f"shuf_{i}": v for i, v in enumerate(sc[t][j])})
            for j, key in enumerate(keys) for t in sc]


def meso(shuffle=False):
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    warnings.filterwarnings("ignore")
    excl = excluded_sessions()
    rows = []
    groups = [(k, v) for k, v in session_files("meso").items() if k not in excl]
    for i, (sid, files) in enumerate(groups):
        s = apply_coverage(sid, load_session(files), "meso")
        if len(s["choice"]) < MIN_TRIALS:
            continue
        reg = br.id2acronym(s["atlas_id"], mapping="Beryl")
        regions = [r for r in np.unique(reg) if (reg == r).sum() >= N_PICK]
        if not regions:
            continue
        sets = [np.flatnonzero(reg == r) for r in regions]
        if shuffle:
            rows += shuffled_rows(decode_shuffled(s, sets, zlib.crc32(sid.encode())),
                                  [dict(session=sid, region=r) for r in regions])
            print("meso shuffle", i + 1, len(groups), sid, flush=True)
            continue
        sc = decode_sets(s, sets)
        rows += [dict(session=sid, region=r, n_neurons=len(idx), n_trials=len(s["choice"]), target=t, all=sc[t][j])
                 for j, (r, idx) in enumerate(zip(regions, sets)) for t in sc]
        print("meso", i + 1, len(groups), sid, regions, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT_V2 / f"decoding_meso_region{'_shuf' if shuffle else ''}.pqt")
    return d


def bwm_shuffle():
    from common import FEATURE_ROOT
    warnings.filterwarnings("ignore")
    pids = pd.read_parquet(OUT_V2 / "decoding_bwm_probe.pqt").pid.unique()
    rows = []
    for i, pid in enumerate(pids):
        s = load_session([FEATURE_ROOT / "bwm" / f"{pid}.npz"])
        rows += shuffled_rows(decode_shuffled(s, [np.arange(len(s["xyz"]))], zlib.crc32(pid.encode())),
                              [dict(pid=pid)])
        print("bwm shuffle", i + 1, len(pids), pid, flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT_V2 / "decoding_bwm_probe_shuf.pqt")
    return d


def bwm():
    from iblatlas.atlas import AllenAtlas
    from common import FEATURE_ROOT
    from make_bwm_depth import MAX_DEPTH_UM, depth_below_surface
    ba = AllenAtlas()
    ba.compute_surface()
    d = pd.read_parquet(OUT_V2 / "decoding_bwm_probe.pqt")
    p = d.drop_duplicates("pid").set_index("pid")
    x, y = p.entry_x.values / 1e3, p.entry_y.values / 1e3            # m
    top = ba.top[ba.bc.y2i(y), ba.bc.x2i(x)]                          # m
    ids = ba.get_labels(np.c_[x, y, top - ENTRY_DEPTH_UM * 1e-6], mapping="Beryl")
    p["region"] = ba.regions.id2acronym(ids)
    major = {}
    for pid in p.index:
        z = np.load(FEATURE_ROOT / "bwm" / f"{pid}.npz")
        keep = depth_below_surface(z["xyz"], ba) <= MAX_DEPTH_UM
        acr = ba.regions.id2acronym(z["atlas_id"][keep], mapping="Beryl")
        major[pid] = pd.Series(acr).mode().iloc[0] if keep.any() else None
    p["region_units"] = pd.Series(major)
    from one.api import ONE
    one = ONE(base_url="https://alyx.internationalbrainlab.org")
    pids = list(p.index)
    p["probe_name"] = pd.Series({i["id"]: i["name"] for k in range(0, len(pids), 50)
                                 for i in one.alyx.rest("insertions", "list", django=f"id__in,{pids[k:k + 50]}")})
    d = d.join(p[["region", "region_units", "probe_name"]], on="pid")
    d.to_parquet(OUT_V2 / "decoding_bwm_probe_region.pqt")
    return d


if __name__ == "__main__":
    import sys
    if "--shuffle" in sys.argv:
        if "--meso-only" not in sys.argv:
            bwm_shuffle()
        meso(shuffle=True)
        sys.exit()
    b = bwm().drop_duplicates("pid")
    print(b.region.value_counts().head(20))
    print("entry region == majority superficial-unit region:", (b.region == b.region_units).mean())
    m = meso()
    print(m.drop_duplicates(["session", "region"]).region.value_counts())
