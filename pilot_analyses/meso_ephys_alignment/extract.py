#!/usr/bin/env python3
"""Per-session features for the spontaneous-vs-task alignment pilot.

For every neuron: grey-screen activity in 1-s bins (spontaneousActivity block
of passivePeriods) and per-trial window means for the four task variables of
HYPOTHESIS.md. BWM spikes are read from the cached passingSpikes tables (good
units only); meso activity comes from meso_loader (deconvolved, neurons only).
Writes output/features/<modality>_<eid>.npz.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from one.api import ONE
from common import (OUT, BWM_SESSIONS, MESO_SESSIONS, WINDOWS, GREY_BIN, region_class,
                    meso_area, beryl)

FEAT = OUT / "features"
FEAT.mkdir(exist_ok=True)
ONE_ROOT = Path.home() / "Downloads" / "ONE"
OPENALYX = ONE_ROOT / "openalyx.internationalbrainlab.org"
MESO_REPO = Path(__file__).resolve().parents[2]  # mesoscope repo root


def trial_table(tr):
    """Masked trials (choice made, finite event times, 0.08 <= RT <= 2 s)."""
    t = pd.DataFrame({k: np.asarray(tr[k], float) for k in
                      ("stimOn_times", "firstMovement_times", "feedback_times", "choice",
                       "probabilityLeft", "feedbackType", "contrastLeft", "contrastRight")})
    rt = t.firstMovement_times - t.stimOn_times
    ok = t[["stimOn_times", "firstMovement_times", "feedback_times", "choice",
            "probabilityLeft", "feedbackType"]].notna().all(1) & (t.choice != 0) & rt.between(.08, 2.)
    t = t[ok].reset_index(drop=True)
    t["contrast"] = t.contrastLeft.fillna(0) + t.contrastRight.fillna(0)
    t["stim_right"] = t.contrastRight.notna().astype(int)
    t["choice_right"] = (t.choice == -1).astype(int)   # CCW wheel turn = right stimulus side
    t["reward"] = (t.feedbackType == 1).astype(int)
    return t


def windows_for(t, modality):
    """(n_trials, n_vars, 2) window bounds in the order of WINDOWS."""
    return np.stack([np.c_[t[ev].values + a, t[ev].values + b]
                     for ev, a, b in WINDOWS[modality].values()], axis=1)


def grey_interval(one, eid):
    ds = [d for d in one.list_datasets(eid) if "passivePeriods" in d]
    tab = one.load_dataset(eid, ds[0]).set_index("Unnamed: 0")
    return float(tab.loc["start", "spontaneousActivity"]), float(tab.loc["stop", "spontaneousActivity"])


# ---------------------------------------------------------------- BWM
def bwm_units(eid):
    import brainwidemap
    q = pd.read_csv(Path(brainwidemap.__file__).parent / "fixtures" / "2023_12_bwm_release.csv")
    q = q[q.eid == eid]
    cl = pd.read_parquet(ONE_ROOT / "alyx.internationalbrainlab.org" / "bwm_tables" / "clusters.pqt",
                         columns=["pid", "cluster_id", "label", "acronym"])
    trains, acr = [], []
    for _, p in q.iterrows():
        f = list(OPENALYX.glob(f"*/Subjects/{p.subject}/{p.date}/*/alf/{p.probe_name}/"
                               "pykilosort/*/passingSpikes.table.pqt"))
        assert len(f) == 1, (eid, p.probe_name, f)
        s = pd.read_parquet(f[0], columns=["times", "clusters"])
        c = cl[(cl.pid == p.pid) & (cl.label == 1)].set_index("cluster_id")
        order = np.argsort(s.clusters.values, kind="stable")
        times, clu = s.times.values[order], s.clusters.values[order]
        lo = np.searchsorted(clu, c.index.values, "left")
        hi = np.searchsorted(clu, c.index.values, "right")
        for cid, a, b in zip(c.index.values, lo, hi):
            trains.append(times[a:b])
            acr.append(c.loc[cid, "acronym"])
    return trains, beryl(np.array(acr))


def bwm_features(eid, one):
    trains, regions = bwm_units(eid)
    tr = pd.read_parquet(ONE_ROOT / "bwm_tables" / "trials.pqt")
    t = trial_table(tr[tr.eid == eid].to_dict("list"))
    g0, g1 = grey_interval(one, eid)
    edges = np.arange(g0, g1 + 1e-9, GREY_BIN)
    grey = np.stack([np.histogram(s, edges)[0] for s in trains]).T / GREY_BIN      # (bins, n)
    iv = windows_for(t, "bwm")
    flat = iv.reshape(-1, 2)
    win = np.stack([np.searchsorted(s, flat[:, 1]) - np.searchsorted(s, flat[:, 0]) for s in trains], -1)
    win = win.reshape(iv.shape[0], iv.shape[1], -1) / (iv[..., 1:] - iv[..., :1])   # (trials, vars, n)
    return grey, win.astype(np.float32), regions, t


# ---------------------------------------------------------------- meso
def meso_features(eid, one):
    sys.path.insert(0, str(MESO_REPO))
    from meso_loader import load_mesoscope_session
    s = load_mesoscope_session(eid, one=one)
    cols = sorted({d.rsplit("/", 1)[0] for d in one.list_datasets(eid) if "_ibl_trials." in d})
    t = trial_table(one.load_object(eid, "trials", collection=cols[0]))
    ft = s.roi_times[0].astype(np.float64)
    x = s.roi_signal                                            # (n, frames)
    g0, g1 = grey_interval(one, eid)
    assert ft[0] <= g0 and ft[-1] >= g1, "grey screen not fully imaged"
    edges = np.arange(g0, g1 + 1e-9, GREY_BIN)
    which = np.digitize(ft, edges) - 1
    keep = (which >= 0) & (which < len(edges) - 1)
    xk, wk = x[:, keep], which[keep]
    assert np.isin(np.arange(len(edges) - 1), wk).all(), "grey-screen bin without frames"
    grey = np.stack([xk[:, wk == b].mean(1) for b in range(len(edges) - 1)])
    iv = windows_for(t, "meso")
    covered = (iv[..., 0] >= ft[0]).all(1) & (iv[..., 1] <= ft[-1]).all(1)
    # no frame gap (> 1 s) inside or around any window, where interpolation would be meaningless
    gap_lo, gap_hi = ft[:-1][np.diff(ft) > 1.], ft[1:][np.diff(ft) > 1.]
    for a, b in zip(gap_lo, gap_hi):
        covered &= ~((iv[..., 1] > a) & (iv[..., 0] < b)).any(1)
    # no trial window overlapping the grey screen
    covered &= ~((iv[..., 1] > g0) & (iv[..., 0] < g1)).any(1)
    t, iv = t[covered].reset_index(drop=True), iv[covered]
    # window means of the linearly interpolated trace (11 samples per window), as in dorsal_maps
    pts = iv[..., :1] + (iv[..., 1:] - iv[..., :1]) * np.linspace(0, 1, 11)       # (trials, vars, 11)
    idx = np.clip(np.searchsorted(ft, pts) - 1, 0, len(ft) - 2)
    w = (pts - ft[idx]) / (ft[idx + 1] - ft[idx])
    win = (x[:, idx] * (1 - w) + x[:, idx + 1] * w).mean(-1)                      # (n, trials, vars)
    regions = np.array([meso_area(r) for r in s.region_labels])
    return grey.astype(np.float32), np.moveaxis(win, 0, -1).astype(np.float32), regions, t


def main():
    one_bwm = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
    one_meso = ONE()
    jobs = [("bwm", e, one_bwm, bwm_features) for e in BWM_SESSIONS] + \
           [("meso", e, one_meso, meso_features) for e in MESO_SESSIONS]
    for modality, eid, one, fn in jobs:
        out = FEAT / f"{modality}_{eid}.npz"
        if out.exists():
            continue
        grey, win, regions, t = fn(eid, one)
        np.savez(out, grey=grey, win=win, regions=regions,
                 classes=np.array([region_class(r) for r in regions]),
                 trials=t.to_records(index=False))
        print(f"{modality} {eid[:8]}: grey {grey.shape}, trials {len(t)}, neurons {len(regions)}", flush=True)


if __name__ == "__main__":
    main()
