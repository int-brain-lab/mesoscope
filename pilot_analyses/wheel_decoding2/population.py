#!/usr/bin/env python3
"""How much wheel-speed signal is in the best neurons of a session, and how
common are they? (Quantifies the rasters of fig1 over all sessions.)

Per session: neurons are ranked by their correlation with wheel speed on the
even trials; on the odd trials (held out) we report
- frac_pos: fraction of neurons with r > 0.1 on BOTH halves (replicable
  positive wheel correlates);
- r_top<k>: r of the mean z-scored activity of the k best-ranked neurons with
  wheel speed, k = 10 and 60;
- r_top10_trial: the same against trial-specific speed (wheel speed minus its
  per-bin mean over the even trials);
- the same r_top10 with the session subsampled to N_MATCH neurons (mean over
  N_DRAWS draws; every included session has at least that many), so sessions
  with thousands of mesoscope neurons and tens of BWM
  neurons are compared at equal neuron counts.
Sessions with >= MIN_NEURONS neurons and >= MIN_TRIALS trials.

Usage: python population.py -> output/population.pqt
"""
import numpy as np
import pandas as pd

from wd_common import OUT, load_session, session_files, wheel_corr

N_MATCH = 30
N_DRAWS = 20
MIN_NEURONS = 30
MIN_TRIALS = 50


def top_r(X, y, r_sel, k, target=None):
    """Held-out r of the mean of the k neurons with the highest r_sel."""
    idx = np.argsort(-np.nan_to_num(r_sel, nan=-9))[:k]
    Z = X[idx].reshape(len(idx), -1)
    sd = Z.std(1, keepdims=True)
    Z = (Z - Z.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1)
    t = (y if target is None else target).reshape(-1)
    return np.corrcoef(Z.mean(0), t)[0, 1]


def session_stats(s, rng):
    X, y = s["X_wheel"].astype(np.float64), np.asarray(s["wheel_y"], np.float64)
    X = X[X.reshape(len(X), -1).std(1) > 0]
    n = len(X)
    ev, od = np.arange(0, X.shape[1], 2), np.arange(1, X.shape[1], 2)
    r_ev, r_od = wheel_corr(X[:, ev], y[ev]), wheel_corr(X[:, od], y[od])
    Xo, yo = X[:, od], y[od]
    resid = yo - y[ev].mean(0)
    out = dict(n_neurons=n, n_trials=X.shape[1], frac_pos=np.mean((r_ev > 0.1) & (r_od > 0.1)),
               r_top10=top_r(Xo, yo, r_ev, 10), r_top60=top_r(Xo, yo, r_ev, 60),
               r_top10_trial=top_r(Xo, yo, r_ev, 10, resid))
    if n >= N_MATCH:
        d = [rng.choice(n, N_MATCH, replace=False) for _ in range(N_DRAWS)]
        out["r_top10_match"] = np.mean([top_r(Xo[i], yo, r_ev[i], 10) for i in d])
        out["r_top10_trial_match"] = np.mean([top_r(Xo[i], yo, r_ev[i], 10, resid) for i in d])
    return out


def main():
    rows = []
    rng = np.random.default_rng(0)
    for m in ("meso", "bwm", "bwm_d640"):
        for sid, f in session_files(m).items():
            s = load_session(f)
            Xf = s["X_wheel"].astype(np.float64).reshape(len(s["X_wheel"]), -1)
            if len(s["choice"]) < MIN_TRIALS or (Xf.std(1) > 0).sum() < MIN_NEURONS:
                continue
            rows.append(dict(modality=m, session=sid, **session_stats(s, rng)))
        print(m, len(rows), flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "population.pqt")
    print(d.groupby("modality").median(numeric_only=True).round(3).T)


if __name__ == "__main__":
    main()
