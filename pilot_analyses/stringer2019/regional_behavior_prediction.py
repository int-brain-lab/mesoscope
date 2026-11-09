#!/usr/bin/env python3
"""Regional SVCA prediction averaged across matched sessions."""
from pathlib import Path
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np
from one.api import ONE

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "output"  # generated figures, caches and tables (not tracked)
OUT.mkdir(exist_ok=True)
sys.path[:0] = [str(REPO), str(REPO / "pilot_analyses")]
from meso_loader import load_mesoscope_session
from stringer19_region_comparison import select_region_neurons
from stringer19_svca_prediction import compute_svca_prediction

# Eight sessions per region (>= N_NEURONS region neurons, video and trials;
# region_session_scan.csv): the original three, plus sessions from animals not
# yet represented where possible. Motor recordings exist only in SP075/SP076
# (+ SP081 for MOp), and MOs only in SP076; this is reported in the caption.
# Some sessions contribute to two regions with disjoint neurons.
SESSIONS = {
    "MOp": [("SP076", "16d4e507-d20f-4808-9584-fab050643077"),
            ("SP075", "fa10ff03-be1f-4b67-9e1e-87bf6979c967"),
            ("SP075", "52ad9bfb-583a-496c-8d2a-5b8dd20c6af2"),
            ("SP081", "4ee6a6f3-92d3-4980-8ebf-fc4376b68d11"),
            ("SP076", "c14efa12-5931-490b-b445-3552044a9c54"),
            ("SP076", "d094fdec-fc93-48d4-892c-4b0dd17c553d"),
            ("SP081", "33fefd36-2fae-4ed6-ab5a-e2ca3a693d87"),
            ("SP075", "a6108970-6556-4742-a6bd-61e2bfcf5626")],
    "MOs": [("SP076", "16d4e507-d20f-4808-9584-fab050643077"),
            ("SP076", "a4afea1e-ad72-433d-9498-95ddc54252fe"),
            ("SP076", "29220169-a779-4519-b131-de403fe87507"),
            ("SP076", "66a5f0a1-828d-422e-9181-9432c3f094b2"),
            ("SP076", "af256bb6-4302-442f-a7ed-780ad92a2413"),
            ("SP076", "fc97a20e-2c1a-427d-93b9-0a4bc1c41ba7"),
            ("SP076", "c14efa12-5931-490b-b445-3552044a9c54"),
            ("SP076", "d094fdec-fc93-48d4-892c-4b0dd17c553d")],
    "VISp": [("SP044", "97334cc3-059f-4fb9-8413-3fe0288a7144"),
             ("SP058", "460a4c0f-f3f3-49a4-b0da-ddcb44322cbe"),
             ("SP061", "dd45de54-6b3c-443a-949d-e74954fc33b2"),
             ("SP072", "29063845-ba7a-4dc1-a4ab-61d285365fa9"),
             ("SP060", "be5909b6-0df7-415c-bd54-1e84eaadf591"),
             ("SP037", "1552bad3-36a9-481e-be69-ffd2b6be0a87"),
             ("SP067", "a5754e0c-ec0b-4218-96fe-46b9e92e14f1"),
             ("SP054", "198bc994-6a9f-4e18-a058-5918bb0d6ceb")],
    "SSp-bfd": [("SP054", "aac478d1-7dc3-4acc-b965-c350c73e3610"),
                ("SP061", "dd45de54-6b3c-443a-949d-e74954fc33b2"),
                ("SP067", "98e9c22f-1006-4bb4-b573-6e4676c5f61d"),
                ("SP072", "0377646d-a970-44f8-806a-712c4214e0ce"),
                ("SP058", "f13a1f77-8462-4aea-a4f3-c2ccc0ecc346"),
                ("SP075", "7714458c-690f-46e1-82cf-89bf7a0112ae"),
                ("SP066", "98675fde-0dbf-4ad9-a11a-0a7f74bb527a"),
                ("SP081", "461efb38-9195-45de-bd06-a5ba4e32c8a1")],
}
WINDOW, N_NEURONS, N_SVCS = (1730., 2030.), 1000, 128
KEYS = ("reliable_frac", "video_var_explained", "behav_var_explained",
        "block_var_explained", "choice_var_explained")


def compute(regions=None):
    one = ONE()
    out = {}
    cache_dir = OUT / "regional_behavior_session_results"
    cache_dir.mkdir(exist_ok=True)
    for region, entries in SESSIONS.items():
        if regions is not None and region not in regions:
            continue
        out[region] = {key: [] for key in KEYS}
        for subject, eid in entries:
            cache = cache_dir / f"{region}_{eid}_wheel_only.npz"
            if cache.exists():
                with np.load(cache) as z:
                    result = {key: z[key] for key in z.files}
            else:
                print(f"[{region}; {subject}] {eid}")
                session = load_mesoscope_session(eid, one=one)
                idx = select_region_neurons(session, region, n_total=N_NEURONS, seed=0)
                assert len(idx) == N_NEURONS
                result = compute_svca_prediction(
                    eid, one=one, session=session, window=WINDOW, bin_seconds=None,
                    neuron_subset=idx, n_svcs=N_SVCS, use_video_cache=True,
                    use_whisker=False, verbose=True,
                )
                np.savez(cache, **result)
            for key in KEYS:
                out[region][key].append(np.asarray(result[key]))
        out[region] = {key: np.stack(value) for key, value in out[region].items()}
    return out


def plot(data, show_sessions=False):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5, "legend.fontsize": 5, "axes.linewidth": .5,
        "lines.linewidth": .8, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(183 / 25.4, 48 / 25.4), sharey=True,
                             gridspec_kw={"wspace": .12})
    specs = [("Maximum explainable", "reliable_frac", "#777777"),
             ("Left-camera video PCs", "video_var_explained", "#0072B2"),
             ("Wheel speed", "behav_var_explained", "#009E73"),
             ("Block", "block_var_explained", "#CC79A7"),
             ("Choice", "choice_var_explained", "#D55E00")]
    upper = []
    rank = np.arange(1, N_SVCS + 1)
    for label_letter, region, ax in zip("abcd", SESSIONS, axes):
        for label, key, color in specs:
            values = 100 * np.clip(data[region][key], 0, None)
            mean = values.mean(0)
            sem = values.std(0, ddof=1) / np.sqrt(values.shape[0])
            ax.plot(rank, mean, color=color, label=label)
            ax.fill_between(rank, np.maximum(0, mean-sem), mean+sem,
                            color=color, alpha=.16, linewidth=0)
            upper.append(np.nanmax(mean + sem))
            if show_sessions:
                # One fine line per session: the SEM band's raw material.
                ax.plot(rank, values.T, color=color, lw=.3, alpha=.5)
                upper.append(np.nanmax(values))
        ax.set_xscale("log"); ax.set_xlim(.9, N_SVCS * 1.05)
        ax.set_xticks([1, 10, 100]); ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(which="minor", width=.4, length=1.5)
        ax.set_xlabel("SVC dimension")
        ax.text(.04, .96, region, transform=ax.transAxes, va="top", ha="left", fontsize=6)
        ax.text(-.18, 1.04, label_letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
    ymax = max(10, 10 * np.ceil(max(upper) / 10))
    axes[0].set_ylim(0, ymax); axes[0].set_yticks([0, ymax/2, ymax])
    axes[0].set_ylabel("Variance explained (%)")
    axes[-1].legend(frameon=False, loc="upper right", handlelength=1.6)
    fig.subplots_adjust(left=.06, right=.995, bottom=.25, top=.91, wspace=.12)
    suffix = "_sessions" if show_sessions else ""
    fig.savefig(OUT / f"regional_behavior_prediction{suffix}_panels.pdf")
    plt.close(fig)


def main():
    if sys.argv[1:2] == ["--cache-only"]:  # fill the per-session cache for some regions
        compute(regions=sys.argv[2:])
        return
    data = compute()
    plot(data)
    plot(data, show_sessions=True)


if __name__ == "__main__":
    main()
