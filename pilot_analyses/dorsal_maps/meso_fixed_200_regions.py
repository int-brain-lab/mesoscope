"""Figure 4a-style mesoscope region plots for a fixed 200-neuron decoder."""
import zlib

from iblatlas.regions import BrainRegions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decode_region import decode_shuffled, shuffled_rows
from decode_v2 import BINARY, OUT_V2, TARGETS
from maps_v2 import TITLE
from meso_fixed_n_pilot import STEM as PILOT_STEM, matched_pairs, sample_session
from plotstyle import NATURE_DOUBLE_COL_IN, save_with_caption, style_axes
from region_variance import REGIONS, mixed, plot_region_scores, stars
from sessions import session_files

STEM = "meso_fixed_200_regions"
SIZE = 200
KEYS = ["session", "region", "target"]


def controls(pairs):
    files = session_files("meso")
    brain_regions = BrainRegions()
    checkpoints = OUT_V2 / f"_{STEM}_shuffles"
    checkpoints.mkdir(exist_ok=True)
    frames = []
    for i, (sid, group) in enumerate(pairs.groupby("session"), 1):
        path = checkpoints / f"{sid}.pqt"
        cached = path.exists()
        if cached:
            shuffled = pd.read_parquet(path)
        else:
            session, pools = sample_session(sid, files[sid], group, brain_regions)
            scores = decode_shuffled(session, [pool[:SIZE] for pool in pools], zlib.crc32(sid.encode()))
            shuffled = pd.DataFrame(shuffled_rows(scores, group[["session", "region"]].to_dict("records")))
        expected = pd.MultiIndex.from_product([[sid], group.region, TARGETS], names=KEYS)
        actual = pd.MultiIndex.from_frame(shuffled[KEYS])
        if actual.has_duplicates or set(expected) != set(actual) or not np.isfinite(shuffled.shuf_0).all():
            raise ValueError(f"Missing, invalid or mismatched shuffled scores for {sid}")
        if not cached:
            tmp = path.with_suffix(".tmp")
            shuffled.to_parquet(tmp, index=False)
            tmp.replace(path)
        frames.append(shuffled)
        print(f"{i}/{pairs.session.nunique()} {sid}: {len(group)} pairs", flush=True)
    return pd.concat(frames, ignore_index=True)


def render(scores):
    fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 1.85))
    grid = fig.add_gridspec(1, len(TARGETS), wspace=0.35)
    rng = np.random.default_rng(0)
    shared_y = {}
    stats = []
    for c, target in enumerate(TARGETS):
        metric = "ROC-AUC" if target in BINARY else "$R^2$"
        ax = fig.add_subplot(grid[0, c], sharey=shared_y.get(metric))
        shared_y.setdefault(metric, ax)
        rows = scores.loc[scores.target == target]
        fit = mixed(rows, adjust_log_n=False)
        stats.append(dict(target=target, decoder_neurons=SIZE, n_pairs=len(rows), **fit))
        plot_region_scores(ax, rows, REGIONS, target, rng)
        ax.set_xticks(np.arange(len(REGIONS)), REGIONS, rotation=90, fontsize=4.5)
        ax.set_xlim(-0.6, len(REGIONS) - 0.4)
        ax.set_title(TITLE[target] + (f" ({metric})" if target not in BINARY else ""), fontsize=6, pad=9)
        ax.text(1, 1.01, f"region {fit['frac_region']:.0%} {stars(fit['p_lrt_region'])}",
                transform=ax.transAxes, fontsize=4.5, va="bottom", ha="right")
        if c == 0:
            ax.set_ylabel("Mesoscope\n" + metric, fontsize=6)
        elif c != 3:
            ax.tick_params(labelleft=False)
        style_axes(ax, categorical_x=True)
    intro = (
        f"Mesoscope decoding from {SIZE} randomly sampled neurons per session--region pair. "
        r"The layout follows \figref{region_variance}a. "
        f"Each dot shows one held-out score from {scores.session.nunique()} sessions and "
        f"{scores.animal.nunique()} mice, with {len(scores) // len(TARGETS):,} matched pairs for every target. "
        "The 200-neuron pilot uses a single seeded draw per pair. Black bars are region medians, and dashed "
        "lines mark chance. Grey bands and lines are the 5th--95th percentiles and medians of the regional "
        "nulls, with one joint trial-label shuffle per session using the same 200-neuron subsets. Above each "
        "panel, the region share and likelihood-ratio significance come from the crossed region, session "
        "and animal mixed model. Neuron count is fixed. "
        r"Stars indicate $p<0.05$, $p<0.01$ and $p<0.001$."
    )
    save_with_caption(fig, OUT_V2 / STEM, [], STEM, intro=intro, compile_pdf=False)
    plt.close(fig)
    pd.DataFrame(stats).to_csv(OUT_V2 / f"{STEM}_variance.csv", index=False)


def main():
    scores = pd.read_parquet(OUT_V2 / f"{PILOT_STEM}_scores.pqt")
    scores = scores.loc[scores.decoder_neurons == SIZE].rename(columns={"score": "all"})
    pairs = matched_pairs()
    expected = pairs[["session", "region"]].merge(pd.DataFrame({"target": TARGETS}), how="cross")
    if (scores.duplicated(KEYS).any() or
            set(pd.MultiIndex.from_frame(scores[KEYS])) != set(pd.MultiIndex.from_frame(expected[KEYS])) or
            not np.isfinite(scores["all"]).all()):
        raise ValueError("The 200-neuron pilot scores do not match the Figure 4a cohort")
    shuffled = controls(pairs)
    render(scores.merge(shuffled, on=KEYS, validate="one_to_one"))


if __name__ == "__main__":
    main()
