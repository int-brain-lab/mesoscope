#!/usr/bin/env python3
"""Scan canonical sessions for regional neuron counts and passive coverage.

Only small per-FOV metadata are downloaded (ROI types, region ids, frame
times), not activity. Neuron and region definitions match
`meso_loader.load_mesoscope_session` (mpciROITypes neurons; histology
region ids, falling back to the pipeline estimate) and
`select_region_neurons` (acronym prefix match).
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from one.api import ONE
from iblatlas.atlas import AllenAtlas

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REGIONS = ("VISp", "SSp-bfd", "MOp", "MOs")
OUT = HERE / "output"  # generated figures, caches and tables (not tracked)
OUT.mkdir(exist_ok=True)
SCAN_CSV = OUT / "region_session_scan.csv"


def scan(one, atlas, eid):
    ds = one.list_datasets(eid)
    fovs = sorted({d.split("/")[1] for d in ds if d.startswith("alf/FOV_")})
    counts = dict.fromkeys(REGIONS, 0)
    fov_times = []
    for fov in fovs:
        names = {d.split("/")[-1] for d in ds if d.startswith(f"alf/{fov}/")}
        key = next((k for k in ("mpciROIs.brainLocationIds_ccf_2017.npy",
                                "mpciROIs.brainLocationIds_ccf_2017_estimate.npy") if k in names), None)
        if key is None or "mpciROIs.mpciROITypes.npy" not in names:
            continue
        is_neuron = one.load_dataset(eid, "mpciROIs.mpciROITypes.npy", collection=f"alf/{fov}").astype(bool)
        labels = atlas.regions.id2acronym(one.load_dataset(eid, key, collection=f"alf/{fov}"))[is_neuron]
        for r in REGIONS:
            counts[r] += int(np.char.startswith(labels.astype(str), r).sum())
        if not fov_times:  # frame times are shared across FOVs
            fov_times.append(one.load_dataset(eid, "mpci.times.npy", collection=f"alf/{fov}"))
    times = fov_times[0] if fov_times else np.array([np.nan])
    row = dict(eid=eid, **counts, imaging_end=float(np.nanmax(times)))
    p = [d for d in ds if "passivePeriods" in d]
    row["passive_imaged_s"] = 0.
    if p and fov_times:
        tab = one.load_dataset(eid, p[0]).set_index("Unnamed: 0")
        a, b = tab.loc["start", "passiveProtocol"], tab.loc["stop", "passiveProtocol"]
        inside = times[(times >= a) & (times <= b)]
        row["passive_imaged_s"] = float(inside[-1] - inside[0]) if inside.size > 1 else 0.
    row["has_video"] = any("leftCamera.raw.mp4" in d for d in ds)
    row["has_motion_energy"] = any("leftCamera.ROIMotionEnergy" in d for d in ds)
    row["has_trials"] = any("trials.table" in d for d in ds)
    return row


def scan_path(path):
    one, atlas = ONE(), AllenAtlas()
    subject, date, number = path.split("/")
    try:
        eid = str(one.search(subject=subject, date=date, number=int(number))[0])
        return dict(path=path, subject=subject, **scan(one, atlas, eid))
    except Exception as e:
        print(path, "FAILED", repr(e)[:200], flush=True)
        return None


def main():
    paths = [s.strip() for s in (REPO / "canonical_sessions.txt").read_text().split(",") if s.strip()]
    rows = pd.read_csv(SCAN_CSV).to_dict("records") if SCAN_CSV.exists() else []
    todo = [p for p in paths if p not in {r["path"] for r in rows}]
    with ProcessPoolExecutor(6) as pool:
        for row in pool.map(scan_path, todo):
            if row is not None:
                print(row, flush=True)
                rows.append(row)
                pd.DataFrame(rows).to_csv(SCAN_CSV, index=False)


if __name__ == "__main__":
    main()
