"""Do individual neurons gain or lose stimulus sensitivity over days?

Per tracked neuron (QC-passing sessions only, >= MIN_SESSIONS sessions) that
is stimulus-selective overall (mean contra-ipsi d' across its sessions, sign
aligned, > MIN_D): slope of the sign-aligned d' against recording day. Null:
the same slope after shuffling which day each session's d' belongs to
(N_PERM permutations per neuron). A neuron loses (gains) sensitivity when its
slope is below the 2.5% (above the 97.5%) quantile of its null.
Expected by chance: 2.5% per tail.

Output: output/trends.parquet (one row per neuron) and printed summary.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "output"
MIN_SESSIONS = 6
MIN_D = 0.2
N_PERM = 2000


def run(seed=0):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(seed)
    n = pd.read_parquet(OUT / "neuron_sessions.parquet")
    n["day"] = pd.to_datetime(n.session.str[:10])
    rows = []
    for subj, q in n.groupby("subject"):
        q = q.assign(day=(q.day - q.day.min()).dt.days)
        for uid, h in q.groupby("uid"):
            if len(h) < MIN_SESSIONS:
                continue
            rows.append((subj, uid, h.fov.iloc[0], h.day.values, h.d_stim.values, h.p_stim.values))
    out = []
    # batch neurons with the same number of sessions
    by_len = {}
    for r in rows:
        by_len.setdefault(len(r[3]), []).append(r)
    for L, rs in by_len.items():
        day = torch.tensor(np.stack([r[3] for r in rs]), dtype=torch.float32, device=dev)   # (n, L)
        d = torch.tensor(np.stack([r[4] for r in rs]), dtype=torch.float32, device=dev)
        sign = torch.sign(d.mean(1, keepdim=True))
        y = d * sign                                                                     # sign-aligned
        dc = day - day.mean(1, keepdim=True)
        slope = ((y - y.mean(1, keepdim=True)) * dc).sum(1) / (dc * dc).sum(1)
        perm = torch.argsort(torch.rand(N_PERM, len(rs), L, generator=g, device=dev), dim=2)
        yp = torch.gather(y.expand(N_PERM, -1, -1), 2, perm)
        sp = ((yp - yp.mean(2, keepdim=True)) * dc).sum(2) / (dc * dc).sum(1)            # (N_PERM, n)
        p_lo = ((sp <= slope).float().mean(0)).cpu().numpy()
        p_hi = ((sp >= slope).float().mean(0)).cpu().numpy()
        for i, r in enumerate(rs):
            out.append(dict(subject=r[0], uid=r[1], fov=r[2], n_sessions=L, span_days=int(np.ptp(r[3])),
                            mean_d=float(y[i].mean()), sign=float(sign[i]),
                            slope_per_10d=10 * float(slope[i]), p_lose=p_lo[i], p_gain=p_hi[i],
                            frac_sig=float(np.mean(r[5] < 0.01))))
    t = pd.DataFrame(out)
    t["selective"] = t.mean_d > MIN_D
    t["loses"] = t.p_lose < 0.025
    t["gains"] = t.p_gain < 0.025
    t.to_parquet(OUT / "trends.parquet")
    s = t[t.selective]
    print(f"neurons tracked in >= {MIN_SESSIONS} QC sessions: {len(t)}; stimulus-selective "
          f"(mean |d'| > {MIN_D}): {len(s)}")
    print(s.groupby("subject").agg(n=("uid", "size"), sessions=("n_sessions", "median"),
                                   span=("span_days", "median"), loses=("loses", "mean"),
                                   gains=("gains", "mean"),
                                   slope_per_10d=("slope_per_10d", "median")).round(3).to_string())
    print(f"all selective: loses {s.loses.mean():.3f}, gains {s.gains.mean():.3f} (chance 0.025 each); "
          f"non-selective: loses {t[~t.selective].loses.mean():.3f}, gains {t[~t.selective].gains.mean():.3f}")
    return t


if __name__ == "__main__":
    run()
