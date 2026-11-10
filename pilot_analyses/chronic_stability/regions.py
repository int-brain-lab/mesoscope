"""Is the response profile more stable in some cortical regions than others?

The profile stability index of stability.py (the most robust metric: every
tracked neuron counts, no significance threshold, scale-free, chance- and
baseline-normalised) computed separately per region. Regions are the Beryl
regions of the ROI centre in the Allen CCF; a neuron counts for a region only
if both sessions of the pair put it there. Auditory areas (AUDp, AUDpo, AUDd)
are pooled as AUD.

Per session pair (a, b) and region R, on the tracked neurons of R:
  r_w, r_x as in stability.py (even/odd within a day; even a vs odd b)
  r_0reg: across days, partner a random other neuron of the same region and
      FOV (region-matched shuffle, as rho_region_shuffle in pairwise_correl.py)
  r_0any: across days, partner a random other neuron of the same FOV
      (any region; the chance level of stability.py)
  stability = (med r_x - med r_0reg) / (med r_w - med r_0reg)
The region-matched chance removes what neurons of a region share (a typical
regional profile), so the index measures only neuron-specific stability.

Output: output/region_pairs.csv (one row per pair and region),
output/region_positions.parquet (CCF position and region of each tracked UID,
first session, for the map).
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stability import OUT, _date, _rowcorr, session_summary  # noqa: E402

REGIONS = ["VISp", "VISa", "VISrl", "VISal", "SSp-bfd", "AUD"]
POOL = {"AUDp": "AUD", "AUDpo": "AUD", "AUDd": "AUD"}
MIN_REG = 30          # tracked neurons of the region per session pair


def _partner(groups, rng):
    """Random different member of the same group (or -1)."""
    out = np.full(groups.size, -1)
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        if i.size > 1:
            out[i] = i[(np.arange(i.size) + rng.integers(1, i.size)) % i.size]
    return out


def _rx(Pa, Pb, j=None):
    if j is None:
        return 0.5 * (_rowcorr(Pa[0], Pb[1]) + _rowcorr(Pa[1], Pb[0]))
    v = j >= 0
    return 0.5 * (_rowcorr(Pa[0][v], Pb[1][j[v]]) + _rowcorr(Pa[1][v], Pb[0][j[v]]))


def run(seed=0):
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    rng = np.random.default_rng(seed)
    qc = pd.read_parquet(OUT / "qc_positions.parquet")
    rows = []
    for sdir in sorted((OUT / "sessions").iterdir()):
        S = {}
        for f in sorted(sdir.glob("*.npz")):
            q = qc[(qc.subject == sdir.name) & (qc.session == f.stem)].sort_values("row")
            s = session_summary(f, qc_ok=q.qc_pass.values)
            if s is None:
                continue
            acr = br.id2acronym(np.load(f)["atlas_id"][s["row"]], mapping="Beryl")
            s["region"] = np.array([POOL.get(a, a) for a in acr])
            S[f.stem] = s
        for ka, kb in combinations(sorted(S, key=_date), 2):
            a, b = S[ka], S[kb]
            _, ia, ib = np.intersect1d(a["uid"], b["uid"], return_indices=True)
            same = a["region"][ia] == b["region"][ib]
            ia, ib = ia[same], ib[same]
            Pa, Pb = a["prof"][:, ia], b["prof"][:, ib]
            reg, fov = a["region"][ia], a["fov"][ia]
            j_any = _partner(fov, rng)                                   # same FOV, any region
            j_reg = _partner(np.char.add(fov.astype(str), reg), rng)     # same FOV and region
            rw_a, rw_b = _rowcorr(Pa[0], Pa[1]), _rowcorr(Pb[0], Pb[1])
            rx = _rx(Pa, Pb)
            r0a = np.full(rx.size, np.nan)
            r0r = np.full(rx.size, np.nan)
            r0a[j_any >= 0] = _rx(Pa, Pb, j_any)
            r0r[j_reg >= 0] = _rx(Pa, Pb, j_reg)
            for R in REGIONS:
                m = reg == R
                if m.sum() < MIN_REG:
                    continue
                row = dict(subject=sdir.name, a=ka, b=kb, dt=(_date(kb) - _date(ka)).days,
                           region=R, n=int(m.sum()),
                           rw=0.5 * (np.median(rw_a[m]) + np.median(rw_b[m])),
                           rx=np.median(rx[m]), r0reg=np.nanmedian(r0r[m]),
                           r0any=np.nanmedian(r0a[m]))
                for k in ("reg", "any"):
                    row[f"stab_{k}"] = (row["rx"] - row[f"r0{k}"]) / (row["rw"] - row[f"r0{k}"])
                rows.append(row)
        print(sdir.name, len(S), "sessions", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "region_pairs.csv", index=False)
    return d


def positions():
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    out = []
    for sdir in sorted((OUT / "sessions").iterdir()):
        for f in sorted(sdir.glob("*.npz")):
            z = np.load(f)
            ok = z["uid"] != ""
            acr = br.id2acronym(z["atlas_id"][ok], mapping="Beryl")
            out.append(pd.DataFrame(dict(subject=sdir.name, uid=z["uid"][ok], ml=z["xyz"][ok, 0],
                                         ap=z["xyz"][ok, 1], region=[POOL.get(a, a) for a in acr])))
    p = pd.concat(out).drop_duplicates(["subject", "uid"])
    p.to_parquet(OUT / "region_positions.parquet")
    return p


if __name__ == "__main__":
    positions()
    run()
