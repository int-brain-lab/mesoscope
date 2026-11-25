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
from plotstyle import NATURE_DOUBLE_COL_IN, panel_label, save_with_caption, style_axes
from region_variance import REGIONS, load, mixed, plot_region_scores, stars
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


def render(scores, all_scores, all_stats):
    fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.03))
    grid = fig.add_gridspec(2, len(TARGETS), wspace=0.35, hspace=0.28)
    shared_y = {}
    stats = []
    all_stats = all_stats.set_index("target")
    for row, (data, count_label) in enumerate(((all_scores, "All neurons"), (scores, f"{SIZE} neurons"))):
        rng = np.random.default_rng(0)
        for c, target in enumerate(TARGETS):
            metric = "ROC-AUC" if target in BINARY else "$R^2$"
            ax = fig.add_subplot(grid[row, c], sharey=shared_y.get(metric))
            shared_y.setdefault(metric, ax)
            rows = data.loc[data.target == target]
            if row == 0:
                fit = all_stats.loc[target]
            else:
                fit = mixed(rows, adjust_log_n=False)
                stats.append(dict(target=target, decoder_neurons=SIZE, n_pairs=len(rows), **fit))
            plot_region_scores(ax, rows, REGIONS, target, rng)
            ax.set_xticks(np.arange(len(REGIONS)), REGIONS if row == 1 else [], rotation=90, fontsize=4.5)
            ax.set_xlim(-0.6, len(REGIONS) - 0.4)
            if row == 0:
                ax.set_title(TITLE[target] + (f" ({metric})" if target not in BINARY else ""), fontsize=6, pad=9)
            ax.text(1, 1.01, f"region {fit['frac_region']:.0%} {stars(fit['p_lrt_region'])}",
                    transform=ax.transAxes, fontsize=4.5, va="bottom", ha="right")
            if c == 0:
                ax.set_ylabel(count_label + "\n" + metric, fontsize=6)
                panel_label(ax, "ab"[row], x=-0.32)
            elif c != 3:
                ax.tick_params(labelleft=False)
            style_axes(ax, categorical_x=True)
    intro = ("All-neuron and fixed-size mesoscope decoding by Beryl region. Black bars are region medians. "
             "Dashed lines mark chance, and grey bands and lines show the 5th--95th percentiles and medians "
             "of the trial-shuffle nulls. Both rows use the same score axes. Insets give the region share "
             "from the crossed region, session and animal model. "
             r"Stars indicate likelihood-ratio $p<0.05$, $p<0.01$ and $p<0.001$.")
    captions = [
        r"All-neuron mesoscope row from \figref{region_variance}a, including its original trial-shuffle "
        f"baselines and mixed-model estimates with log neuron count. It uses "
        f"{all_scores.session.nunique()} sessions and {all_scores.groupby('target').size().min()}--"
        f"{all_scores.groupby('target').size().max()} session--region pairs per target.",
        f"Decoding from {SIZE} randomly sampled neurons per pair, using one seeded draw from each of "
        f"{len(scores) // len(TARGETS):,} matched pairs across {scores.session.nunique()} sessions. "
        "The null uses one joint trial-label shuffle per session with the same sampled neurons. "
        "The mixed model has a fixed neuron count.",
    ]
    save_with_caption(fig, OUT_V2 / STEM, captions, STEM, intro=intro, compile_pdf=False)
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
    all_scores = load()[0]["meso"]
    all_stats = pd.read_csv(OUT_V2 / "region_variance_stats.csv")
    render(scores.merge(shuffled, on=KEYS, validate="one_to_one"), all_scores, all_stats)


if __name__ == "__main__":
    main()
