"""Mesoscope decoding versus atlas region volume and contributing neuron count."""
from itertools import product

from iblatlas.atlas import AllenAtlas
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from decode_v2 import BINARY, OUT_V2
from maps_v2 import TARGETS, TITLE
from plotstyle import NATURE_DOUBLE_COL_IN, allen_color, panel_label, save_with_caption, style_axes
from region_variance import load

STEM = "meso_size_scaling"
N_PERM = 9999
MIN_RESTRICTED_NEURONS = 300


def atlas_volumes(regions):
    atlas = AllenAtlas(res_um=25)
    atlas.compute_regions_volume()
    mapping = atlas.regions.mappings["Beryl"]
    volumes = np.bincount(mapping, weights=atlas.regions.volume, minlength=mapping.size)
    by_region = pd.Series(volumes, index=atlas.regions.acronym)
    bilateral = by_region.loc[~by_region.index.duplicated()].reindex(regions)
    if bilateral.isna().any() or (bilateral <= 0).any():
        raise ValueError("Missing atlas volume for a Figure 4 region")
    return bilateral.rename("volume_mm3")


def volume_pvalue(log_x, y):
    """Two-sided correlation test with volumes exchangeable across region labels."""
    x_centered, y_centered = log_x - log_x.mean(), y - y.mean()
    observed = abs(x_centered @ y_centered)
    rng = np.random.default_rng(0)
    permuted = rng.permuted(np.broadcast_to(x_centered, (N_PERM, len(x_centered))), axis=1)
    return (1 + np.count_nonzero(np.abs(permuted @ y_centered) >= observed - 1e-12)) / (N_PERM + 1)


def neuron_pvalue(log_x, y, animals):
    """Two-sided wild sign test of the slope score, clustered by mouse."""
    score = (log_x - log_x.mean()) * (y - y.mean())
    clusters = pd.Series(score).groupby(np.asarray(animals)).sum().to_numpy()
    signs = np.array(list(product((-1, 1), repeat=len(clusters))))
    return np.mean(np.abs(signs @ clusters) >= abs(clusters.sum()) - 1e-12)


def scatter(axis, x, y, p, colors, marker_size, alpha, *, lower_right=False):
    log_x = np.log10(x)
    r = np.corrcoef(log_x, y)[0, 1]
    slope, intercept = np.polyfit(log_x, y, 1)
    extent = np.array([x.min(), x.max()])
    axis.scatter(x, y, s=marker_size, color=colors, alpha=alpha, linewidths=0, rasterized=len(x) > 100)
    axis.plot(extent, intercept + slope * np.log10(extent), color="#202020", lw=0.8)
    axis.set_xscale("log")
    label_x, label_y = (0.95, 0.04) if lower_right else (0.05, 0.98)
    axis.text(label_x, label_y, f"$r$ = {r:+.2f}\n$p$ = {p:.2g}", transform=axis.transAxes,
              ha="right" if lower_right else "left", va="bottom" if lower_right else "top",
              fontsize=5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.83, pad=1))
    style_axes(axis, categorical_x=True)
    axis.xaxis.set_minor_locator(NullLocator())
    return r


def main():
    cohorts, regions = load()
    meso = cohorts["meso"]
    volumes = atlas_volumes(regions)
    fig, axes = plt.subplots(3, len(TARGETS), figsize=(NATURE_DOUBLE_COL_IN, 4.25),
                             sharex="row", sharey="col", gridspec_kw=dict(hspace=0.42, wspace=0.5))
    results = []
    for col, target in enumerate(TARGETS):
        scores = meso.loc[meso.target == target]
        restricted = scores.loc[scores.n_neurons >= MIN_RESTRICTED_NEURONS]
        regional_scores = (scores.groupby("region")["all"].median(), restricted.groupby("region")["all"].mean())
        for row, (regional, cohort) in enumerate(zip(regional_scores, ("all_pairs", "n_ge_300"))):
            points = volumes.to_frame().join(regional.rename("score"), how="inner")
            x, y = points.volume_mm3.to_numpy(), points.score.to_numpy()
            p = volume_pvalue(np.log10(x), y)
            r = scatter(axes[row, col], x, y, p, [allen_color(region) for region in points.index], 12, 0.88,
                        lower_right=target == "feedback")
            results.append(dict(target=target, predictor="atlas_volume_mm3", cohort=cohort,
                                n=len(x), n_mice=np.nan, r=r, p=p))
            if row == col == 0:
                for region, offset, align in (("VISam", (3, -4), "left"), ("VISp", (-3, 5), "right"),
                                              ("MOs", (3, 5), "left")):
                    point = points.loc[region]
                    axes[row, col].annotate(region, (point.volume_mm3, point.score), xytext=offset,
                                            textcoords="offset points", ha=align, va="center", fontsize=4.5)

        x, y = scores.n_neurons.to_numpy(), scores["all"].to_numpy()
        p = neuron_pvalue(np.log10(x), y, scores.animal)
        r = scatter(axes[2, col], x, y, p, scores.region.map(allen_color).tolist(), 4, 0.29,
                    lower_right=target == "feedback")
        results.append(dict(target=target, predictor="decoder_neurons", cohort="all_pairs", n=len(x),
                            n_mice=scores.animal.nunique(), r=r, p=p))

        metric = "ROC-AUC" if target in BINARY else "$R^2$"
        axes[0, col].set_title(f"{TITLE[target]}\n{metric}", fontsize=6)
        for row, ticks in ((0, [1, 3, 10]), (1, [1, 3, 10]), (2, [20, 100, 500, 2000])):
            ax = axes[row, col]
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.tick_params(labelbottom=True, labelsize=5)
        for row in (0, 1):
            axes[row, col].set_xlabel("Atlas volume (mm$^3$)", fontsize=5.5)
        axes[2, col].set_xlabel("Decoder neurons", fontsize=5.5)
    row_labels = ("Median decoding score\nAll session-region pairs",
                  "Mean decoding score\nAt least 300 neurons",
                  "Decoding score")
    for row, (letter, label) in enumerate(zip("abc", row_labels)):
        axes[row, 0].set_ylabel(label, fontsize=5.5)
        panel_label(axes[row, 0], letter, x=-0.3, y=1.06)

    stats = pd.DataFrame(results)
    stats["p_holm"] = multipletests(stats.p, method="holm")[1]
    stats.to_csv(OUT_V2 / f"{STEM}.csv", index=False)
    captions = [
        "Bilateral Beryl region volume from the 25-micron Allen CCF atlas versus the median score across "
        "sessions in that region (one dot per region, 15 regions). Scores are the mesoscope all-neuron "
        r"decoding scores from \figref{region_variance}a. Two-sided $p$-values use 9,999 shuffles of volumes "
        "across regions. Dots in all panels use Allen atlas colours for their Beryl regions.",
        f"Atlas volume versus the mean score per region, using only session--region pairs with at least "
        f"{MIN_RESTRICTED_NEURONS} decoder neurons. Each region with an eligible pair contributes one dot. "
        "The same volume permutation test is used.",
        "Number of neurons used by the decoder versus score, with one dot per session and region. "
        "The two-sided slope test flips the signs of the null score contributions by mouse, keeping each "
        "mouse's observations together (12 mice). All rows use log-scaled horizontal axes. Lines are "
        "least-squares fits and $r$ is Pearson correlation with the log predictor. Displayed $p$-values "
        "are unadjusted. Holm-adjusted values across all 15 tests are in the accompanying CSV.",
    ]
    save_with_caption(fig, OUT_V2 / STEM, captions, STEM, intro="Mesoscope decoding versus region volume and neuron count.",
                      compile_pdf=False)
    plt.close(fig)
    print(stats.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
