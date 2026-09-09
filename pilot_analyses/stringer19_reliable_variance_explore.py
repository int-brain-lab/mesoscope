"""Exploratory follow-up to stringer19_svca_prediction.py: why is our
"max explainable" (reliable variance) curve at SVC 1 ~60%, when Stringer et
al. 2019's Fig. 1L / Fig. 2E examples are close to 100%?

Three candidate explanations, each checked here against real data:

1. **Temporal binning.** The paper bins neural activity to 1.2-1.3s before
   running SVCA; this mesoscope's native frame period is ~0.18s. Coarser
   bins average down independent per-timepoint noise relative to genuinely
   shared signal, which should inflate the reliable-variance estimate.
   Checked via `compute_reliable_variance(..., bin_seconds=1.25)`.
2. **Area composition.** The paper's example recordings are single-area V1;
   our example session (eid 16d4e507-...) spans four functionally distinct
   areas (MOp/MOs/RSPd/SSp-ll). Mixing heterogeneous cortical areas into
   each SVCA neuron-set could dilute the dominant shared component that a
   homogeneous, retinotopically-organized V1 population would show.
   Checked against a session that's ~95% VISp (SP058/2024-07-31/001).
3. **Behavioral state.** The paper's recordings are pure spontaneous
   activity (no task, no stimuli); our sessions run an active behavioral
   task throughout. Checked by restricting the VISp session to its
   `passiveProtocol` window (no task, replayed/receptive-field-mapping
   stimuli only) instead of the full active-task session.

Results are saved as one comparison figure plus the raw curves (.npz) to
`stringer19/exploratory_reliable_variance/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from one.api import ONE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meso_loader import load_mesoscope_session  # noqa: E402
from stringer19_svca_prediction import compute_reliable_variance  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "stringer19" / "exploratory_reliable_variance"

EXAMPLE_EID = "16d4e507-d20f-4808-9584-fab050643077"  # MOp/MOs/RSPd/SSp-ll, no passive protocol
VISP_EID = "9e67c687-856d-4b94-938c-83c5775a1fff"      # SP058/2024-07-31/001, ~95% VISp, has passive protocol
PASSIVE_WINDOW = (3527.993667, 3962.072000)             # passivePeriods.intervalsTable 'passiveProtocol' row


def run(save: bool = True) -> dict:
    one = ONE()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading example session (MOp/MOs/RSPd/SSp-ll)...")
    example_session = load_mesoscope_session(EXAMPLE_EID, one=one)
    print("Loading VISp session (SP058/2024-07-31/001)...")
    visp_session = load_mesoscope_session(VISP_EID, one=one)

    curves = {}

    print("[1/5] example, full session, native resolution")
    curves["example: MOp/MOs/SSp, full, native"] = compute_reliable_variance(example_session)

    print("[2/5] example, full session, 1.25s bins")
    curves["example: MOp/MOs/SSp, full, 1.25s bins"] = compute_reliable_variance(example_session, bin_seconds=1.25)

    print("[3/5] VISp, full session, native resolution")
    curves["VISp (~95%), full, native"] = compute_reliable_variance(visp_session)

    print("[4/5] VISp, full session, 1.25s bins")
    curves["VISp (~95%), full, 1.25s bins"] = compute_reliable_variance(visp_session, bin_seconds=1.25)

    print("[5/5] VISp, passive-protocol window only, native resolution")
    curves["VISp (~95%), passive only, native"] = compute_reliable_variance(
        visp_session, window=PASSIVE_WINDOW, block_duration=36.0, pad_duration=1.0
    )

    for name, res in curves.items():
        print(f"  {name}: SVC1={100*res['reliable_frac'][0]:.1f}%  n_train/test={res['n_train']}/{res['n_test']}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = ["gray", "black", "tab:blue", "navy", "tab:orange"]
    for (name, res), color in zip(curves.items(), colors):
        r = res["reliable_frac"]
        rank = np.arange(1, len(r) + 1)
        ax.plot(rank, 100 * np.clip(r, 0, None), color=color, lw=1.5, label=f"{name} (n={res['n_train']}+{res['n_test']})")

    ax.set_xscale("log")
    ax.set_xlabel("SVC dimension")
    ax.set_ylabel("% reliable variance (\"max explainable\")")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Why is SVC1 reliable variance ~60% and not ~100%?\nbinning vs. area vs. active-task state", fontsize=10)
    fig.tight_layout()

    if save:
        fig.savefig(OUT_DIR / "reliable_variance_comparison.png", dpi=200, bbox_inches="tight")
        print("Saved:", OUT_DIR / "reliable_variance_comparison.png")
        np.savez(
            OUT_DIR / "reliable_variance_comparison.npz",
            **{name.replace(" ", "_").replace(":", "").replace("/", "-"): res["reliable_frac"] for name, res in curves.items()},
        )
        print("Saved:", OUT_DIR / "reliable_variance_comparison.npz")

    return curves


if __name__ == "__main__":
    run()
