#!/usr/bin/env python3
"""Generate the session tables (LaTeX) for the regional SVCA report.

Neuron counts come from region_session_scan.csv (same neuron and region
definitions as meso_loader); checkerboard halves, frame counts and frame
periods are recomputed/read from the analyses' own inputs and caches.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "pilot_analyses"), str(HERE)]
from one.api import ONE
from meso_loader import load_mesoscope_session
from stringer19_region_comparison import TIER1_SESSIONS, N_TOTAL, WINDOW, select_region_neurons
from stringer19_svca_prediction import _split_neurons_checkerboard
from regional_behavior_prediction import OUT, SESSIONS as BEHAV_SESSIONS
from passive_behavior_prediction import SESSIONS as PASSIVE_BEHAV_SESSIONS

scan = pd.read_csv(OUT / "region_session_scan.csv").set_index("eid")
one = ONE()


def last_trial_end(eid):
    tr = one.load_object(eid, "trials")
    iv = np.asarray(tr["intervals"]) if "intervals" in tr else np.c_[tr["intervals_0"], tr["intervals_1"]]
    return float(np.nanmax(iv[:, 1]))


def tier1_table():
    rows = []
    for region, eids in TIER1_SESSIONS.items():
        for eid in eids:
            s = load_mesoscope_session(eid, one=one)
            idx = select_region_neurons(s, region, n_total=N_TOTAL)
            a, b = _split_neurons_checkerboard(s.xyz[idx])
            t = s.roi_times[0]
            m = (t >= WINDOW[0]) & (t < WINDOW[1])
            post = max(0., WINDOW[1] - last_trial_end(eid))
            rows.append((region, scan.loc[eid, "path"], eid[:8], int(scan.loc[eid, region]),
                         len(a), len(b), float(np.median(np.diff(t[m]))), post))
    return rows


def behav_table():
    rows = []
    for region, entries in BEHAV_SESSIONS.items():
        for _, eid in entries:
            f = OUT / "regional_behavior_session_results" / f"{region}_{eid}_wheel_only.npz"
            with np.load(f) as z:
                n_a, n_b = int(z["n_neurons_a"]), int(z["n_neurons_b"])
                n_frames = int(z["n_train"]) + int(z["n_test"])
            rows.append((region, scan.loc[eid, "path"], eid[:8], int(scan.loc[eid, region]),
                         n_a, n_b, n_frames))
    return rows


def passive_table():
    """One row per region-session of either passive analysis."""
    recs = np.load(OUT / "passive_dimensionality_results.npz", allow_pickle=True)["records"]
    dim = {(r["region"], r["eid"]): r for r in recs}
    keys = list(dim) + [(reg, e) for reg, es in PASSIVE_BEHAV_SESSIONS.items() for e in es
                        if (reg, e) not in dim]
    order = {"VISp": 0, "SSp-bfd": 1, "MOp": 2, "MOs": 3}
    rows = []
    for region, eid in sorted(keys, key=lambda k: order[k[0]]):
        in_behav = eid in PASSIVE_BEHAV_SESSIONS[region]
        split = dim[(region, eid)]["passive_split"] if (region, eid) in dim else None
        if split is None:
            with np.load(OUT / "passive_behavior_session_results" / f"{region}_{eid}.npz") as z:
                split = (int(z["n_neurons_a"]), int(z["n_neurons_b"]))
        spont = f"{dim[(region, eid)]['spont_s']:.0f}" if (region, eid) in dim else "--"
        use = ("D" if (region, eid) in dim else "") + ("B" if in_behav else "")
        rows.append((region, scan.loc[eid, "path"], eid[:8], int(scan.loc[eid, region]),
                     int(split[0]), int(split[1]), spont, use))
    return rows


PASSIVE_CAPTION = (r"\textbf{Sessions of the passive-protocol analyses.} Columns as in "
                   r"Table~\ref{tab:dim}; Spont.: seconds of spontaneous activity (grey screen) "
                   r"within the 730-s passive window (--: not in the dimensionality set); Used: D, "
                   r"passive dimensionality (Fig.~\ref{fig:pdim}); B, passive behavioural "
                   r"prediction (Fig.~\ref{fig:pbehav}).")


DIM_CAPTION = (r"\textbf{Sessions of the dimensionality analysis.} $N_{\mathrm{reg}}$: all neurons of "
               r"the region in the session, before subsampling to $N = 1{,}000$. $N_A/N_B$: sizes of "
               r"the two checkerboard halves. Frame: median native frame period (s). Post-task: "
               r"seconds of the 100--3,100~s window after the last trial ended.")
BEHAV_CAPTION = (r"\textbf{Sessions of the behavioural prediction analysis.} Columns as in "
                 r"Table~\ref{tab:dim}; $T$: number of native imaging frames in the 300-s window "
                 r"(train and test, after padding removal).")


def longtable(cols, caption, label, header, body):
    return (f"\\begin{{longtable}}{{@{{}}{cols}@{{}}}}\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n"
            f"\\toprule\n{header} \\\\\n\\midrule\n\\endfirsthead\n"
            f"\\toprule\n{header} \\\\\n\\midrule\n\\endhead\n"
            f"{body}\\bottomrule\n\\end{{longtable}}\n")


def main():
    t1 = tier1_table()
    body = "".join(f"{r[0]} & {r[1]} & \\texttt{{{r[2]}}} & {r[3]:,} & {r[4]}/{r[5]} & "
                   f"{r[6]:.3f} & {r[7]:.0f} \\\\\n" for r in t1)
    (OUT / "table_dimensionality.tex").write_text(longtable(
        "llrrrrr", DIM_CAPTION, "tab:dim",
        r"Region & Session & eid & $N_{\mathrm{reg}}$ & $N_A/N_B$ & Frame (s) & Post-task (s)", body))
    b = behav_table()
    body = "".join(f"{r[0]} & {r[1]} & \\texttt{{{r[2]}}} & {r[3]:,} & {r[4]}/{r[5]} & {r[6]} \\\\\n"
                   for r in b)
    (OUT / "table_behavior.tex").write_text(longtable(
        "llrrrr", BEHAV_CAPTION, "tab:behav",
        r"Region & Session & eid & $N_{\mathrm{reg}}$ & $N_A/N_B$ & $T$", body))
    pr = passive_table()
    body = "".join(f"{r[0]} & {r[1]} & \\texttt{{{r[2]}}} & {r[3]:,} & {r[4]}/{r[5]} & {r[6]} & {r[7]} \\\\\n"
                   for r in pr)
    (OUT / "table_passive.tex").write_text(longtable(
        "llrrrrl", PASSIVE_CAPTION, "tab:passive",
        r"Region & Session & eid & $N_{\mathrm{reg}}$ & $N_A/N_B$ & Spont.\ (s) & Used", body))
    # summary numbers quoted in the text
    t1 = np.array([(r[4], r[5], r[6]) for r in t1])
    b = np.array([(r[4], r[5], r[6]) for r in b])
    print("tier1 halves", t1[:, :2].min(), t1[:, :2].max(), "frame period", t1[:, 2].min(), t1[:, 2].max())
    print("behav halves", b[:, :2].min(), b[:, :2].max(), "frames", b[:, 2].min(), b[:, 2].max())


if __name__ == "__main__":
    main()
