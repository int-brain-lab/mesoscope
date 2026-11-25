"""Plot VISa tuning and fixed-size decoding by registered cortical layer.

Run decode_bwm_depth.py with and without --meso-reference to generate the
decoder caches, then run python depth_profile.py to regenerate Figure 5.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions
from matplotlib.ticker import MaxNLocator

from common import ALPHA, OUT, VARIABLES
from decode_bwm_depth import LAYERS, MESO_SCORES, N_DRAWS, N_UNITS, REGION, SCORES, layer_units, unit_layers
from decode_v2 import OUT_V2, TARGETS, apply_coverage
from plotstyle import NATURE_DOUBLE_COL_IN, panel_label, save_with_caption, style_axes
from sessions import load_session


CATEGORIES = np.arange(len(LAYERS))
PANEL_TARGETS = VARIABLES + ("wheel",)
TITLE = {"stim": "Stimulus side", "choice": "Choice", "feedback": "Feedback",
         "wheel": "Wheel speed"}
WHEEL_CACHE = OUT_V2 / "report_v7_VISa_layer_wheel.pqt"
N_BOOTSTRAP = 10_000


def wheel_frac(X, y):
    """Per neuron: r > 0.1 with wheel speed on both even and odd trials."""
    def corr(Xs, ys):
        A = Xs.reshape(len(Xs), -1)
        A = A - A.mean(1, keepdims=True)
        b = ys.reshape(-1) - ys.mean()
        den = np.linalg.norm(A, axis=1) * np.linalg.norm(b)
        return np.where(den > 0, A @ b / np.where(den > 0, den, 1), np.nan)
    ev, od = slice(0, None, 2), slice(1, None, 2)
    return (corr(X[:, ev], y[ev]) > 0.1) & (corr(X[:, od], y[od]) > 0.1)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return c - h, c + h


def collect_wheel(cohorts):
    rows = []
    for modality, (groups, units) in cohorts.items():
        for sid, pool in units.groupby("session", sort=False):
            session = load_session(groups[sid])
            wheel = {
                "choice": session["choice"], "wheel_y": session["wheel_y"],
                "X_wheel": session["X_wheel"][pool.unit.to_numpy()],
            }
            wheel = apply_coverage(sid, wheel, modality)
            X = wheel["X_wheel"].astype(np.float64)
            y = wheel["wheel_y"].astype(np.float64)
            ok = X.reshape(len(X), -1).std(1) > 0
            if ok.any():
                rows.append(pool.loc[ok].assign(modality=modality, wheel=wheel_frac(X[ok], y)))
    return pd.concat(rows, ignore_index=True)


def tuning_flags(modality, units, wheel):
    columns = ["session", "atlas_id", *(f"p_{v}" for v in VARIABLES)]
    tun = pd.read_parquet(OUT / f"tuning_{modality}.pqt", columns=columns,
                          filters=[("session", "in", units.session.unique().tolist())])
    tun["unit"] = tun.groupby("session", sort=False).cumcount()
    tun = units.merge(tun, on=["session", "unit"], how="left", validate="one_to_one")
    if tun.atlas_id.isna().any() or not unit_layers(tun.atlas_id.to_numpy(), BrainRegions(), modality).eq(tun.layer).all():
        raise ValueError(f"Tuning cache does not match the {modality} VISa unit layers.")
    for v in VARIABLES:
        p = tun[f"p_{v}"]
        tun[v] = np.where(p.notna(), p.lt(ALPHA), np.nan)
    tun = tun.merge(wheel[["session", "unit", "wheel"]], on=["session", "unit"],
                    how="left", validate="one_to_one")
    tun["wheel"] = tun.wheel.astype(float)
    return tun


def mouse_interval(values, animals):
    """Estimate a pooled mean and a percentile interval by resampling mice."""
    valid = values.notna()
    mice = values[valid].groupby(animals[valid], sort=True).agg(["sum", "count"])
    total, n = mice["sum"].to_numpy(), mice["count"].to_numpy()
    draws = np.random.default_rng(0).integers(len(mice), size=(N_BOOTSTRAP, len(mice)))
    boot = total[draws].sum(axis=1) / n[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return values.mean(), lo, hi, len(mice)


def plot_meso(ax, mean, lo, hi):
    x = CATEGORIES[0]
    ax.vlines(x, lo, hi, color="tab:green", lw=1, zorder=4)
    ax.plot([x, x], [lo, hi], "_", color="tab:green", ms=4, zorder=4)
    ax.plot(x, mean, "D", color="tab:green", ms=3.5, zorder=5)


def save_summary(rows, filename):
    summary = pd.concat(rows, ignore_index=True)
    meso = summary.filter(regex="^meso_").where(summary.layer.eq(LAYERS[0]), axis=0)
    counts = meso.filter(regex="^meso_n").columns
    meso[counts] = meso[counts].astype("Int64")
    summary[meso.columns] = meso
    summary.to_csv(OUT_V2 / filename, index=False, float_format="%.6f")
    return summary


def plot_tuning(axes, bwm, meso, animals):
    grouped = bwm.groupby("layer")
    stats, maxima = [], []
    for ax, v in zip(axes, PANEL_TARGETS):
        k = grouped[v].sum().reindex(LAYERS, fill_value=0).to_numpy(dtype=float)
        n = grouped[v].count().reindex(LAYERS, fill_value=0).to_numpy()
        good = n > 0
        lo, hi = np.full(len(n), np.nan), np.full(len(n), np.nan)
        lo[good], hi[good] = wilson(k[good], n[good])
        fraction = np.divide(k, n, out=np.full(len(n), np.nan), where=good)
        ax.fill_between(CATEGORIES, lo, hi, color="tab:purple", alpha=0.25, lw=0)
        ax.plot(CATEGORIES, fraction, "o-", color="tab:purple", ms=3, lw=0.8)
        mean, meso_lo, meso_hi, n_mice = mouse_interval(meso[v], meso.session.map(animals))
        plot_meso(ax, mean, meso_lo, meso_hi)
        maxima.append(max(hi[good].max(initial=0), mean, meso_hi))
        stats.append(pd.DataFrame({
            "target": v, "layer": LAYERS, "bwm_n": n, "bwm_fraction": fraction,
            "wilson_lo": lo, "wilson_hi": hi, "meso_fraction": mean,
            "meso_ci_lo": meso_lo, "meso_ci_hi": meso_hi,
            "meso_n": meso[v].count(), "meso_n_mice": n_mice,
        }))
    for ax, high in zip(axes, [max(maxima[:3])] * 3 + maxima[3:]):
        ax.set_ylim(0, min(1, 1.12 * high))
    return save_summary(stats, "si_depth_profile.csv")


def plot_decoding(axes, bwm, meso, animals):
    grouped = bwm.groupby("layer")[list(TARGETS)]
    means, sem, counts = (grouped.mean().reindex(LAYERS), grouped.sem().reindex(LAYERS),
                          grouped.count().reindex(LAYERS, fill_value=0))
    groups = meso.session.map(animals)
    ref = pd.DataFrame({v: mouse_interval(meso[v], groups) for v in TARGETS},
                       index=["meso_mean_score", "meso_ci_lo", "meso_ci_hi", "meso_n_mice"])
    stats = []
    for v in TARGETS:
        stats.append(pd.DataFrame({
            "target": v, "layer": LAYERS, "n_sessions": counts[v], "n_units": N_UNITS,
            "n_draws": N_DRAWS, "mean_score": means[v], "sem": sem[v],
            "meso_n_sessions": meso[v].count(),
        }).assign(**ref[v].to_dict()))
    save_summary(stats, "si_depth_decoding.csv")
    for ax, v in zip(axes, PANEL_TARGETS):
        chance = 0 if v == "wheel" else 0.5
        ax.axhline(chance, color="0.5", lw=0.6, ls="--")
        lo, hi = means[v] - sem[v], means[v] + sem[v]
        ax.fill_between(CATEGORIES, lo, hi, color="tab:purple", alpha=0.25, lw=0)
        ax.plot(CATEGORIES, means[v], "o-", color="tab:purple", ms=3, lw=0.8)
        plot_meso(ax, *ref[v].iloc[:3])
    for ax_group, variables, chance in ((axes[:3], VARIABLES, 0.5), (axes[3:], ("wheel",), 0)):
        values = means[list(variables)]
        error = sem[list(variables)].fillna(0)
        low = min(chance, (values - error).min().min(), ref.loc["meso_ci_lo", list(variables)].min())
        high = max(chance, (values + error).max().max(), ref.loc["meso_ci_hi", list(variables)].max())
        for ax in ax_group:
            ax.set_ylim(low, high + 0.1 * (high - low))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
            if low == chance:
                ax.spines["bottom"].set_visible(False)
    return counts[list(PANEL_TARGETS)]


def format_row(axes, letters):
    for ax, v, letter in zip(axes, PANEL_TARGETS, letters):
        ax.set_title(TITLE[v], fontsize=7)
        ax.set_xlim(-0.32, len(LAYERS) - 0.68)
        style_axes(ax, categorical_x=True)
        ax.set_xticks(CATEGORIES, LAYERS)
        panel_label(ax, letter, x=-0.10, y=1.08)


def main():
    bwm, meso = pd.read_parquet(SCORES), pd.read_parquet(MESO_SCORES)
    for frame, args in ((bwm, ""), (meso, " --meso-reference")):
        if set(frame.n_units) != {N_UNITS} or set(frame.n_draws) != {N_DRAWS}:
            raise ValueError(f"Decoder settings changed. Run python decode_bwm_depth.py{args}.")
    cohorts = {m: layer_units(m) for m in ("bwm", "meso")}
    if WHEEL_CACHE.exists():
        wheel = pd.read_parquet(WHEEL_CACHE)
    else:
        wheel = collect_wheel(cohorts)
        wheel.to_parquet(WHEEL_CACHE, index=False)
    bwm_t, meso_t = (tuning_flags(m, cohort[1], wheel.loc[wheel.modality == m]) for m, cohort in cohorts.items())
    animals = pd.read_csv(OUT / "session_subjects.csv").set_index("session").subject
    if pd.concat([meso.session, meso_t.session]).map(animals).isna().any():
        raise ValueError("Missing mesoscope mouse labels for the confidence intervals.")

    fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.9), layout="constrained")
    grid = fig.add_gridspec(4, 4, height_ratios=(0.13, 1, 0.13, 1))
    for row, title in ((0, "Single-unit tuning"), (2, f"Decoding with {N_UNITS} units")):
        heading = fig.add_subplot(grid[row, :])
        heading.set_axis_off()
        heading.text(0.5, 0.5, title, ha="center", va="center", fontsize=8)
    a = [fig.add_subplot(grid[1, col]) for col in range(4)]
    b = [fig.add_subplot(grid[3, col], sharex=a[col]) for col in range(4)]
    format_row(a, "abcd")
    format_row(b, "efgh")
    tuning = plot_tuning(a, bwm_t, meso_t, animals)
    counts = plot_decoding(b, bwm, meso, animals)
    a[0].set_ylabel("Fraction of units")
    b[0].set_ylabel("ROC-AUC")
    b[3].set_ylabel("$R^2$")
    fig.supxlabel("Cortical layer", fontsize=7)

    tuning_n = ", ".join(map(str, tuning.loc[tuning.target.eq("stim"), "bwm_n"]))
    if counts.eq(counts.iloc[:, 0], axis=0).all().all():
        support = "BWM session counts from L2/3 to L6 were " + ", ".join(map(str, counts.iloc[:, 0])) + "."
    else:
        support = " ".join(f"{TITLE[v]} included " + ", ".join(map(str, counts[v]))
                           + " BWM sessions from L2/3 to L6." for v in PANEL_TARGETS)
    meso_t_mice = meso_t.session.map(animals).nunique()
    meso_d_mice = meso.session.map(animals).nunique()
    caps = [
        f"Fraction of BWM VISa units tuned to stimulus side ($p<{ALPHA}$). "
        f"Purple bands show 95\\,\\% Wilson intervals. BWM unit counts from L2/3 to L6 were {tuning_n}. "
        f"The mesoscope tuning sample included {meso_t.session.nunique()} sessions from {meso_t_mice} mice.",
        "Choice.", "Feedback.",
        "Replicable wheel-speed correlation ($r>0.1$ in both even and odd trials).",
        f"Mean held-out stimulus-side ROC-AUC after {N_DRAWS} random draws of {N_UNITS} units per eligible "
        "session and layer, sampled without replacement within each draw. We used identical folds "
        "across layers for each target within a session and weighted sessions equally. "
        f"Purple bands show the standard error across BWM sessions. {support} "
        f"The mesoscope decoding sample included {meso.session.nunique()} sessions from {meso_d_mice} mice. "
        "Dashed lines show chance.",
        "Choice ROC-AUC.", "Feedback ROC-AUC.", "Wheel-speed $R^2$.",
    ]
    save_with_caption(fig, OUT_V2 / "si_depth_profile", caps, "v2_depth_profile",
                      intro=f"Tuning and decoding in Beryl {REGION} by cortical layer. "
                            "BWM layers use the registered Allen CCF labels and pool L6a with L6b. "
                            "Mesoscope ROIs are assigned to L2/3 for this comparison (green diamonds). "
                            "Vertical whiskers show 95\\,\\% confidence intervals from resampling mice.")
    plt.close(fig)


if __name__ == "__main__":
    main()
