#!/usr/bin/env python3
"""Checks of the wheel-decoding pipeline.

1. Decoder: ridge.decode with LOO alpha reproduces dorsal_maps
   gpu_decode.decode_wheel and the scikit-learn reference (common.decode_wheel)
   on real neuron sets; a synthetic neuron (wheel speed + Gaussian noise at a
   given SNR) placed in a 10-neuron set is recovered at the expected
   R^2 = SNR / (1 + SNR).
2. Target: the per-bin wheel speed of the feature files (mean absolute
   velocity from the cumulative absolute displacement, common.wheel_speed)
   against the mean of brainbox's low-pass filtered |velocity| in the same
   bins, recomputed from the raw wheel data (encoder overflow jumps removed
   from both, common.remove_wheel_jumps).

Usage: python checks.py -> output/checks_decoder.csv, output/checks_synthetic.csv,
output/checks_target.csv
"""
import warnings

import numpy as np
import pandas as pd

import ridge
from wd_common import OUT, load_session, session_files

SESSIONS = {"meso": ["5d1d0861-bf08-4491-9e3a-eb9da09c3099", "bba507a8-6383-4729-83dc-8d5d53a03f7c",
                     "89ed3048-56c1-429e-9750-622d9845b793"],
            "bwm": ["ee8b36de-779f-4dea-901f-e0141c95722b", "c557324b-b95d-414c-888f-6ee1329a2329",
                    "28741f91-c837-4147-939e-918d38d849f2"]}
SNRS = (0.1, 0.25, 0.5, 1, 2, 4)


def decoder_checks():
    from common import MIN_NEURONS, bin_neurons, decode_wheel as sk_decode
    from gpu_decode import decode_wheel as gpu_decode
    rows, syn = [], []
    rng = np.random.default_rng(0)
    for m, sids in SESSIONS.items():
        for sid in sids:
            s = load_session(session_files(m)[sid])
            F = s["X_wheel"].astype(np.float64).transpose(1, 2, 0)
            y = s["wheel_y"]
            keys = bin_neurons(s["xyz"])
            sets = [np.flatnonzero(keys == k) for k in np.unique(keys) if (keys == k).sum() >= MIN_NEURONS][:6]
            sets += [rng.choice(F.shape[2], MIN_NEURONS, replace=False) for _ in range(4)]
            a, b = gpu_decode(F, sets, y), ridge.decode(F, sets, y, inner="loo")
            c = np.array([sk_decode(F[:, :, st], y) for st in sets])
            rows.append(dict(modality=m, session=sid, n_sets=len(sets), max_diff_gpu=np.abs(a - b).max(),
                             max_diff_sklearn=np.abs(c - b).max()))
            for snr in SNRS:
                for rep in range(5):
                    st = rng.choice(F.shape[2], MIN_NEURONS, replace=False)
                    Fs = F.copy()
                    Fs[:, :, st[0]] = y + rng.normal(0, np.sqrt(y.var() / snr), y.shape)
                    base = ridge.decode(F, [st], y, inner="trial")[0]
                    got = ridge.decode(Fs, [st], y, inner="trial")[0]
                    syn.append(dict(modality=m, session=sid, snr=snr, rep=rep, expected=snr / (1 + snr),
                                    r2=got, r2_without=base))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "checks_decoder.csv", index=False)
    pd.DataFrame(syn).to_csv(OUT / "checks_synthetic.csv", index=False)


def target_checks():
    from one.api import ONE
    from brainbox.behavior.wheel import interpolate_position, velocity_filtered
    from common import remove_wheel_jumps, task_intervals, trial_mask
    from dorsal_maps_tasks import task_collection_for
    one = ONE()
    rows = []
    for m, sids in SESSIONS.items():
        for sid in sids:
            s = load_session(session_files(m)[sid])
            col = task_collection_for(one, sid) if m == "meso" else "alf"
            trials = one.load_object(sid, "trials", collection=col)
            iv = task_intervals(trials, trial_mask(trials), m)[:, 3:]
            w = one.load_object(sid, "wheel", collection=col)
            pos, t = interpolate_position(w["timestamps"], remove_wheel_jumps(w["position"]), freq=1000)
            vel, _ = velocity_filtered(pos, 1000)
            cum = np.r_[0, np.cumsum(np.abs(vel))] / 1000
            ref = (np.interp(iv[..., 1], t, cum[1:]) - np.interp(iv[..., 0], t, cum[1:])) / (iv[..., 1] - iv[..., 0])
            y = s["wheel_y"]
            ok = ref.shape == y.shape
            rows.append(dict(modality=m, session=sid, same_shape=ok,
                             r=np.corrcoef(ref.ravel(), y.ravel())[0, 1] if ok else np.nan,
                             ratio_mean=(y.mean() / ref.mean()) if ok else np.nan))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "checks_target.csv", index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    decoder_checks()
    target_checks()
