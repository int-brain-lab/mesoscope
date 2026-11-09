"""Tracking QC: is a ROICaT-tracked neuron spatially the same cell on every day?

The ROICaT mirror gives only cluster UIDs, not ROICaT's own match-quality
scores, so the check is done here from ROI pixel positions
(`mpciROIs.stackPos`, (y, x) in the 512 x 512 FOV):

1. Per mouse, every pair of (session, FOV) sharing >= 20 UIDs is
   registered (FOV numbers can differ: ROICaT follows neurons across a
   renumbering of the FOVs) by an affine transform fitted on the shared UIDs, robustly
   (refit after dropping matches > 3 MAD off).
2. The deviation of a (UID, session) is the median of its residuals over
   all such pairs (pixels).
3. The deviation is compared to the ROI's nearest-neighbour distance among all
   neuron ROIs of that session and FOV. A (UID, session) passes when
   deviation < NN distance / 2, i.e. the ROI is closer to the consensus
   position than the midpoint to its closest neighbour: the match cannot be
   a swap with that neighbour.

Output: output/qc_positions.parquet, one row per (subject, session, row of
the session file): uid, fov, roi, y, x, nn_px, dev_px, qc_pass.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import _keep_mask  # noqa: E402

OUT = HERE / "output"


def session_positions(one, regions, f):
    """ROI index and (y, x) of every row of a session file (rebuilds the row
    selection and checks it against the stored UIDs)."""
    z = np.load(f, allow_pickle=True)
    eid = str(z["eid"])
    fovs = np.unique(z["fov"])
    xyz_key = ("mpciROIs.mlapdv.npy" if all(
        "mpciROIs.mlapdv.npy" in {d.split("/")[-1] for d in one.list_datasets(eid, collection=f"alf/FOV_{v:02d}")}
        for v in fovs) else "mpciROIs.mlapdv_estimate.npy")
    rows = []
    for v in fovs:
        col = f"alf/FOV_{v:02d}"
        names = {d.split("/")[-1] for d in one.list_datasets(eid, collection=col)}
        keep, _ = _keep_mask(one, eid, col, names, xyz_key, regions)
        sp = one.load_dataset(eid, "mpciROIs.stackPos.npy", collection=col)[:, :2].astype(float)
        neuron = one.load_dataset(eid, "mpciROIs.mpciROITypes.npy", collection=col).astype(bool)
        nn = cKDTree(sp[neuron]).query(sp, k=2)[0][:, 1]   # NN among neuron ROIs (excl. self)
        roi = np.flatnonzero(keep)
        if roi.size != (z["fov"] == v).sum():
            raise ValueError(f"{f.stem} FOV_{v:02d}: {roi.size} ROIs vs {(z['fov'] == v).sum()} rows")
        rows.append(pd.DataFrame(dict(fov=v, roi=roi, y=sp[roi, 0], x=sp[roi, 1], nn_px=nn[roi])))
    df = pd.concat(rows, ignore_index=True)
    df["uid"] = z["uid"]
    df["row"] = np.arange(len(df))
    return df


def _affine(src, dst, n_iter=3):
    """Robust least-squares affine dst ~ [src, 1] @ M."""
    A = np.c_[src, np.ones(len(src))]
    ok = np.ones(len(src), bool)
    for _ in range(n_iter):
        M = np.linalg.lstsq(A[ok], dst[ok], rcond=None)[0]
        r = np.linalg.norm(A @ M - dst, axis=1)
        mad = np.median(np.abs(r[ok] - np.median(r[ok]))) + 1e-9
        ok = r < np.median(r[ok]) + 3 * 1.4826 * mad
    return M


def deviations(df, min_shared=20):
    """Add dev_px for one subject's position table: for every pair of
    (session, FOV) sharing >= min_shared UIDs, a robust affine maps one onto the
    other; the residual of each shared UID is its pairwise deviation, and
    dev_px is the median over its pairs. (UID, session) entries without such a
    pair get NaN and do not count as tracked."""
    df = df.copy()
    df["dev_px"] = np.nan
    t = df[df.uid != ""].assign(idx=lambda d: d.index)
    res = []
    # FOV numbers can change between sessions (ROICaT follows neurons across a
    # renumbering), so every (session, FOV) pair sharing UIDs is registered
    per = {k: h.set_index("uid") for k, h in t.groupby(["session", "fov"])}
    keys = sorted(per)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            if ka[0] == kb[0]:          # same session
                continue
            shared = per[ka].index.intersection(per[kb].index)
            if len(shared) < min_shared:
                continue
            A, B = per[ka].loc[shared, ["y", "x"]].values, per[kb].loc[shared, ["y", "x"]].values
            r = np.linalg.norm(np.c_[A, np.ones(len(A))] @ _affine(A, B) - B, axis=1)
            res.append(pd.DataFrame(dict(idx=per[ka].loc[shared, "idx"].values, r=r)))
            res.append(pd.DataFrame(dict(idx=per[kb].loc[shared, "idx"].values, r=r)))
    if res:
        med = pd.concat(res).groupby("idx").r.median()
        df.loc[med.index, "dev_px"] = med.values
    df["qc_pass"] = df.dev_px < df.nn_px / 2
    return df


def run(reuse_positions=True):
    """reuse_positions: recompute only the deviations from an existing table."""
    from iblatlas.regions import BrainRegions
    from one.api import ONE
    old = OUT / "qc_positions.parquet"
    old = pd.read_parquet(old) if (reuse_positions and old.exists()) else None
    if old is None:
        one, regions = ONE(), BrainRegions()
    out = []
    for sdir in sorted((OUT / "sessions").iterdir()):
        if old is not None:
            pos = [old[old.subject == sdir.name].drop(columns=["dev_px", "qc_pass"])]
        else:
            pos = [session_positions(one, regions, f).assign(subject=sdir.name, session=f.stem)
                   for f in sorted(sdir.glob("*.npz"))]
        d = deviations(pd.concat(pos, ignore_index=True))
        print(sdir.name, f"tracked rows {np.mean(d.uid != ''):.2f}, "
              f"QC pass {d.qc_pass[d.dev_px.notna()].mean():.2f}, "
              f"median dev {d.dev_px.median():.2f} px, median NN {d.nn_px.median():.2f} px", flush=True)
        out.append(d)
    out = pd.concat(out, ignore_index=True)
    out.to_parquet(OUT / "qc_positions.parquet")
    return out


if __name__ == "__main__":
    run()
