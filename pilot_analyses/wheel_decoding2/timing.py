#!/usr/bin/env python3
"""Timing check at full temporal resolution: lagged correlation of neural
activity with wheel speed over whole sessions.

Mesoscope: deconvolved traces at each ROI's own sample times (frame times +
per-ROI scan offset, meso_loader). BWM: spike counts of good isocortex units
in 50 ms bins. Wheel speed: |velocity| of the 1 kHz-interpolated wheel
position (brainbox velocity_filtered). For lags L from -1.5 to +1.5 s,
r_i(L) = corr(x_i(t), speed(t - L)); positive L = neural activity follows the
wheel. Neurons are selected on the first half of the session (top 5 % by the
maximum r over lags) and the lag curve is measured on the second half, so the
selection does not bias the curve.

Usage: python timing.py  -> output/timing.pqt (lag curves), output/timing_summary.csv
"""
import warnings

import numpy as np
import pandas as pd
import torch

from wd_common import FEATURE_ROOT, OUT

LAGS = np.round(np.arange(-1.5, 1.5001, 0.05), 3)
TOP = 0.05
SESSIONS = {"meso": ["5d1d0861-bf08-4491-9e3a-eb9da09c3099", "bba507a8-6383-4729-83dc-8d5d53a03f7c"],
            "bwm": ["ee8b36de-779f-4dea-901f-e0141c95722b", "c557324b-b95d-414c-888f-6ee1329a2329"]}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FS = 1000


def wheel_speed_trace(one, eid, collection=None):
    from brainbox.behavior.wheel import interpolate_position, velocity_filtered
    kw = dict(collection=collection) if collection else {}
    w = one.load_object(eid, "wheel", **kw)
    pos, t = interpolate_position(w["timestamps"], w["position"], freq=FS)
    vel, _ = velocity_filtered(pos, FS)
    return t, np.abs(vel)


def lag_curves(X, times, wt, ws):
    """X (n, T) signal sampled at times (n, T) [or (T,)]; wheel speed ws on
    the regular grid wt. Returns r (n, n_lags) for the first and second half."""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
    tt = torch.as_tensor(np.broadcast_to(times, X.shape).copy(), dtype=torch.float64, device=DEV)
    w = torch.as_tensor(ws, dtype=torch.float32, device=DEV)
    T = X.shape[1]
    halves = [slice(0, T // 2), slice(T // 2, T)]
    out = np.zeros((2, len(X), len(LAGS)))
    for j, L in enumerate(LAGS):
        idx = torch.round((tt - L - wt[0]) * FS).long().clamp(0, len(ws) - 1)
        W = w[idx]
        for h, sl in enumerate(halves):
            a, b = Xt[:, sl], W[:, sl]
            a = a - a.mean(1, keepdim=True)
            b = b - b.mean(1, keepdim=True)
            out[h, :, j] = ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-12)).cpu().numpy()
    return out


def summarise(m, eid, r, rows, curves):
    ok = np.isfinite(r).all((0, 2))
    r = r[:, ok]
    sel = np.argsort(-r[0].max(1))[:max(5, int(TOP * r.shape[1]))]
    curve = r[1, sel].mean(0)
    allc = r[1].mean(0)
    peak_lag = LAGS[np.argmax(r[1, sel], 1)]
    rows.append(dict(modality=m, session=eid, n=r.shape[1], n_sel=len(sel), peak_lag_pop=LAGS[np.argmax(curve)],
                     peak_r_pop=curve.max(), r_at_0=curve[np.argmin(np.abs(LAGS))],
                     peak_lag_median_neuron=np.median(peak_lag), peak_lag_all=LAGS[np.argmax(allc)]))
    curves += [dict(modality=m, session=eid, lag=L, top=c, all=a) for L, c, a in zip(LAGS, curve, allc)]
    print(rows[-1], flush=True)


def main():
    warnings.filterwarnings("ignore")
    from one.api import ONE
    from meso_loader import load_mesoscope_session
    from dorsal_maps_tasks import task_collection_for
    one = ONE()
    rows, curves = [], []
    for eid in SESSIONS["meso"]:
        s = load_mesoscope_session(eid, one=one)
        wt, ws = wheel_speed_trace(one, eid, task_collection_for(one, eid))
        r = lag_curves(s.roi_signal, s.roi_times, wt, ws)
        summarise("meso", eid, r, rows, curves)
    from brainwidemap import load_good_units
    from iblatlas.regions import BrainRegions
    from common import isocortex_mask
    br = BrainRegions()
    for eid in SESSIONS["bwm"]:
        pids = [str(np.load(f)["pid"]) if "pid" in np.load(f).files else f.stem
                for f in (FEATURE_ROOT / "bwm").glob("*.npz") if str(np.load(f)["eid"]) == eid]
        wt, ws = wheel_speed_trace(one, eid)
        X = []
        for pid in pids:
            spikes, clusters = load_good_units(one, pid)
            # spikes["clusters"] indexes the rows of the good-unit table (0..n-1)
            iso = np.flatnonzero(isocortex_mask(clusters["atlas_id"].values, br))
            st, sc = spikes["times"], spikes["clusters"]
            edges = np.arange(st.min(), st.max(), 0.05)
            for c in iso:
                X.append(np.histogram(st[sc == c], edges)[0].astype(np.float32))
            centres = edges[:-1] + 0.025
        r = lag_curves(np.stack(X), centres, wt, ws)
        summarise("bwm", eid, r, rows, curves)
    pd.DataFrame(curves).to_parquet(OUT / "timing.pqt")
    pd.DataFrame(rows).to_csv(OUT / "timing_summary.csv", index=False)


if __name__ == "__main__":
    main()
