#!/usr/bin/env python3
"""Regional SVCA dimensionality restricted to the passive protocol.

Same analysis as `stringer19_region_comparison.run_tier1_dimensionality`
(1,000 neurons per region-session, 1.25-s bins, power-law fit over ranks
10-100, 50% effective dimensionality), but the window is the first
`DURATION` s of imaged passive protocol (alf/task_0*/passivePeriods). As a
duration-matched control, the same analysis is run on a `DURATION`-s task
window starting at 100 s (the start of the original 3,000-s window), so
passive-vs-task differences are not a trivial consequence of the ~7x
shorter recording. Sessions with less than `DURATION` s of imaged passive
protocol are skipped. Movement is quantified in both windows as the mean
whisker-pad motion energy of the left camera.
"""
from pathlib import Path
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from one.api import ONE

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "output"  # generated figures, caches and tables (not tracked)
OUT.mkdir(exist_ok=True)
sys.path[:0] = [str(REPO), str(REPO / "pilot_analyses")]
from meso_loader import load_mesoscope_session
from stringer19_region_comparison import (
    REGION_COLORS, N_TOTAL, BIN_SECONDS,
    select_region_neurons, fit_powerlaw_exp, effective_dimensionality)
from stringer19_svca_prediction import compute_reliable_variance

# 730 s keeps 15/17 passive sessions (the two dropped VISp sessions have 415
# and 518 s imaged); at 400 s ~30% of ranks 10-100 were <= 0 and alpha was
# dominated by noise in both epochs.
DURATION = 730.
TASK_START = 100.
FIT_RANGE = np.arange(9, 100)  # ranks 10-100
CACHE = OUT / "passive_dimensionality_results.npz"

# Five sessions per region with >= N_TOTAL region neurons and >= DURATION s of
# imaged passive protocol (region_session_scan.csv). The 15 sessions of the
# task-epoch analysis that qualify are kept; replacements were picked from
# animals not yet represented where possible. MOs exists in SP076 only;
# d094fdec is also a MOp session (disjoint neurons).
PASSIVE_SESSIONS = {
    "VISp": ["29063845-ba7a-4dc1-a4ab-61d285365fa9", "76f74094-99bb-467f-a89a-f20e15656048",
             "97334cc3-059f-4fb9-8413-3fe0288a7144", "a5754e0c-ec0b-4218-96fe-46b9e92e14f1",
             "90029e53-b014-4bb1-98e6-15878e8a25ad"],
    "SSp-bfd": ["0377646d-a970-44f8-806a-712c4214e0ce", "bd8565ba-35fe-4c85-93b6-54be4f2b4ccc",
                "98e9c22f-1006-4bb4-b573-6e4676c5f61d", "aac478d1-7dc3-4acc-b965-c350c73e3610",
                "461efb38-9195-45de-bd06-a5ba4e32c8a1"],
    "MOp": ["c14efa12-5931-490b-b445-3552044a9c54", "52ad9bfb-583a-496c-8d2a-5b8dd20c6af2",
            "d094fdec-fc93-48d4-892c-4b0dd17c553d", "fa10ff03-be1f-4b67-9e1e-87bf6979c967",
            "4ee6a6f3-92d3-4980-8ebf-fc4376b68d11"],
    "MOs": ["a4afea1e-ad72-433d-9498-95ddc54252fe", "367aff30-254d-4ea9-ac09-b25feace139a",
            "77141718-8120-4931-bec2-f59ca47f7603", "66a5f0a1-828d-422e-9181-9432c3f094b2",
            "d094fdec-fc93-48d4-892c-4b0dd17c553d"],
}


def passive_periods(one, eid):
    ds = [d for d in one.list_datasets(eid) if "passivePeriods" in d]
    if not ds:
        return None
    return one.load_dataset(eid, ds[0]).set_index("Unnamed: 0")


def motion_energy(one, eid):
    """Left-camera whisker-pad motion energy and its timestamps."""
    try:
        me = one.load_dataset(eid, "leftCamera.ROIMotionEnergy.npy", collection="alf")
        ts = one.load_dataset(eid, "_ibl_leftCamera.times.npy", collection="alf")
    except Exception:
        return None
    n = min(len(me), len(ts))
    return ts[:n], me[:n]


def mean_in(me, t0, t1):
    if me is None or me[0][0] > t0 or me[0][-1] < t1:
        return np.nan
    m = (me[0] >= t0) & (me[0] <= t1)
    return float(np.nanmean(me[1][m]))


def analyse(session, idx, window):
    res = compute_reliable_variance(session, window=window, bin_seconds=BIN_SECONDS,
                                    neuron_subset=idx)
    r = res["reliable_frac"]
    alpha, _, _ = fit_powerlaw_exp(r, FIT_RANGE[FIT_RANGE < len(r)])
    return r, alpha, effective_dimensionality(r, .5), (res["n_neurons_a"], res["n_neurons_b"])


def compute():
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=True) as z:
            return z["records"].tolist()
    one = ONE()
    records = []
    for region, eids in PASSIVE_SESSIONS.items():
        for eid in eids:
            tab = passive_periods(one, eid)
            if tab is None:
                print(f"[{region}] {eid}: no passive protocol, skipped")
                continue
            session = load_mesoscope_session(eid, one=one)
            times = session.roi_times[0]
            p0 = tab.loc["start", "passiveProtocol"]
            t_first = times[times >= p0][0]
            t_last = times[times <= tab.loc["stop", "passiveProtocol"]][-1]
            if t_last - t_first < DURATION:
                print(f"[{region}] {eid}: only {t_last - t_first:.0f} s passive imaged, skipped")
                continue
            passive = (t_first, t_first + DURATION)
            task = (TASK_START, TASK_START + DURATION)
            spont_end = tab.loc["stop", "spontaneousActivity"]
            idx = select_region_neurons(session, region, n_total=N_TOTAL)
            me = motion_energy(one, eid)
            assert len(idx) == N_TOTAL
            rec = dict(region=region, eid=eid, n_neurons=len(idx),
                       spont_s=float(np.clip(spont_end - passive[0], 0, DURATION)))
            for name, window in (("passive", passive), ("task", task)):
                r, alpha, dim50, split = analyse(session, idx, window)
                rec.update({f"{name}_reliable_frac": r, f"{name}_alpha": alpha,
                            f"{name}_split": split,
                            f"{name}_dim50": dim50,
                            f"{name}_me": mean_in(me, *window)})
            rec["me_ratio"] = rec["passive_me"] / rec["task_me"]
            print(f"[{region}] {eid[:8]}: passive a={rec['passive_alpha']:.2f} "
                  f"d50={rec['passive_dim50']} | task a={rec['task_alpha']:.2f} "
                  f"d50={rec['task_dim50']} | ME passive/task={rec['me_ratio']:.2f} "
                  f"| spont {rec['spont_s']:.0f}s")
            records.append(rec)
    np.savez(CACHE, records=np.array(records, dtype=object))
    return records


def strip(ax, records, key, regions, scale=1.):
    for i, region in enumerate(regions):
        color = REGION_COLORS[region]
        rs = [r for r in records if r["region"] == region]
        off = np.linspace(-.06, .06, len(rs))
        task = scale * np.array([r[f"task_{key}"] for r in rs])
        pas = scale * np.array([r[f"passive_{key}"] for r in rs])
        for o, a, b in zip(off, task, pas):
            ax.plot([i - .17 + o, i + .17 + o], [a, b], color=color, lw=.3, alpha=.6)
        ax.scatter(i - .17 + off, task, s=8, facecolor="white", edgecolor=color, lw=.6, zorder=3)
        ax.scatter(i + .17 + off, pas, s=8, color=color, zorder=3)
        for x, v in ((i - .17, task), (i + .17, pas)):
            ax.plot([x - .1, x + .1], [np.nanmean(v)] * 2, color="black", lw=1, zorder=4)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=25, ha="right")
    ax.set_xlim(-.6, len(regions) - .4)


def three_ticks(ax, values, step):
    lo = step * np.floor(np.nanmin(values) / step)
    hi = step * np.ceil(np.nanmax(values) / step)
    ax.set_ylim(lo, hi)
    ax.set_yticks([lo, (lo + hi) / 2, hi])


def main():
    records = compute()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5, "legend.fontsize": 5.2, "axes.linewidth": .5,
        "lines.linewidth": .8, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(183 / 25.4, 53 / 25.4),
                             gridspec_kw={"width_ratios": [1.35, 1, 1, 1]})
    regions = list(REGION_COLORS)

    ax = axes[0]
    for region in regions:
        rs = np.stack([r["passive_reliable_frac"] for r in records if r["region"] == region])
        rank = np.arange(1, rs.shape[1] + 1)
        mean, sem = rs.mean(0), rs.std(0, ddof=1) / np.sqrt(len(rs))
        c = REGION_COLORS[region]
        ax.plot(rank, 100 * mean, color=c, label=f"{region}, n={len(rs)}")
        ax.fill_between(rank, 100 * (mean - sem), 100 * (mean + sem), color=c, alpha=.2, lw=0)
    ax.set_xscale("log"); ax.set_xlim(1, len(rank)); ax.set_xticks([1, 10, 100])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("SVC dimension")
    ymax = 10 * np.ceil(100 * max(r["passive_reliable_frac"].max() for r in records) / 10)
    ax.set_ylim(0, ymax); ax.set_yticks([0, ymax / 2, ymax])
    ax.set_ylabel("Reliable variance, passive (%)")
    ax.legend(frameon=False, handlelength=1.6, labelspacing=.25)

    strip(axes[1], records, "alpha", regions)
    three_ticks(axes[1], [r[f"{k}_alpha"] for r in records for k in ("task", "passive")], .1)
    axes[1].set_ylabel(r"Power-law exponent, $\alpha$")
    strip(axes[2], records, "dim50", regions)
    three_ticks(axes[2], [r[f"{k}_dim50"] for r in records for k in ("task", "passive")], 10)
    axes[2].set_ylabel("Dimensions explaining 50%")
    ax = axes[3]
    for i, region in enumerate(regions):
        v = np.array([r["me_ratio"] for r in records if r["region"] == region])
        ax.scatter(i + np.linspace(-.08, .08, len(v)), v, s=8, color=REGION_COLORS[region], zorder=3)
        ax.plot([i - .15, i + .15], [np.nanmean(v)] * 2, color="black", lw=1, zorder=4)
    ax.axhline(1, color="gray", lw=.5, ls="--")
    ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, rotation=25, ha="right")
    ax.set_xlim(-.6, len(regions) - .4)
    ax.set_ylim(0, 1.5); ax.set_yticks([0, .75, 1.5])
    ax.set_ylabel("Whisker-pad motion,\npassive / task")
    n_me = int(np.sum(np.isfinite([r["me_ratio"] for r in records])))
    ax.text(.98, .04, f"n={n_me} sessions\nwith motion energy", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5)
    axes[1].scatter([], [], s=8, facecolor="white", edgecolor="k", lw=.6, label="Task")
    axes[1].scatter([], [], s=8, color="k", label="Passive")
    axes[1].legend(frameon=False, handletextpad=.1, ncol=2, loc="lower center",
                   bbox_to_anchor=(.5, .97), columnspacing=.8)

    for label, ax in zip("abcd", axes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-.2, 1.04, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
    fig.subplots_adjust(left=.06, right=.995, bottom=.25, top=.92, wspace=.45)
    fig.savefig(OUT / "passive_regional_dimensionality_panels.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
