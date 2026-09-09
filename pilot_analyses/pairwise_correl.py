# %%
from pathlib import Path
from typing import Literal, Dict, List
from tqdm import tqdm
from itertools import combinations
from copy import copy
from datetime import datetime

import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt
import seaborn as sns
from one.api import ONE
from trial_type_definitions import event_definitions_biasedCW, event_definitions_trainingCW, add_info_to_trials_table
from scipy.stats import linregress

import chronic_data_loader as data_loader

one = ONE()

BASE_FOLDER = Path("/mnt/s0/Data/Subjects")


def parse_cannonicals_sessions_file(path: Path) -> pd.DataFrame:
    """parses the text file of canonical sessions into a
    DataFrame for easier groupby selection of sessions"""

    session_paths = []
    with open(path, "r") as fH:
        lines = fH.readlines()
    for line in lines:
        if line.startswith("#") or line == "\n":
            continue
        else:
            line = line.strip().replace("\\", "/")
            session_paths.append(line)

    eids = [str(one.path2eid(path)) for path in session_paths]
    df = pd.DataFrame(one.alyx.rest("sessions", "list", django=f"id__in,{eids}"))
    return df.drop(columns=["number", "lab", "projects", "url"]).set_index("id")


def load_behavior_data(eid: str) -> pd.DataFrame:
    """helper to load the trials table belonging to an eid and adding the relevant columns
    necessary for creating the response feature vectors"""
    trials_table_file = list((BASE_FOLDER / one.eid2path(eid).session_path_short()).rglob("*trials.table*"))
    assert len(trials_table_file) == 1
    trials_df = pd.read_parquet(trials_table_file[0])
    trials_df = add_info_to_trials_table(trials_df)
    return trials_df


def extract_event_based_responses(
    fov_data: nap.TsdFrame,
    trials_df: pd.DataFrame,
    event_definitions: dict,
) -> pd.DataFrame:
    """computes the feature vectors for all neurons / session"""
    stim_avgs = {}
    for event, definition in event_definitions.items():
        _trials_df = trials_df.query(definition["query"])
        timestamps = _trials_df[definition["align_event"]].values
        tensor = nap.compute_perievent_continuous(
            fov_data,
            nap.Ts(timestamps),
            minmax=definition["window"],
        )
        # tensor.shape is timepoints, trials, cells
        # time and trial average
        if timestamps.shape[0] > 10:  # exclude trial types with too little trials
            stim_avgs[event] = np.average(tensor, axis=(0, 1))
        else:
            nans = np.ones(tensor.shape[-1])
            nans[:] = np.nan
            stim_avgs[event] = nans

    return pd.DataFrame(stim_avgs)


def process_imaging_data(imaging_data: nap.TsdFrame) -> nap.TsdFrame:
    """ """
    a, b = np.percentile(imaging_data, (20, 99))
    imaging_data = (imaging_data - a) / (b - a)
    return imaging_data


# %%
# selecting the subject
# subject = "SP060"
subject = "SP058"
# subject = "SP072"
selection = "all cells"

# session_type selection: biased or training
session_type = "biased"
# session_type = "training"
save_plots = False

if session_type == "biased":
    event_definitions = event_definitions_biasedCW
elif session_type == "training":
    event_definitions = event_definitions_trainingCW

sessions_df = data_loader.parse_canonicals_sessions_file(Path(__file__).parent / "canonical_sessions_mod.txt")
sessions_df_ = sessions_df.loc[sessions_df["task_protocol"].str.contains(session_type)]
eids = sessions_df_.query("subject == @subject")["id"].values

# center_eid = "20ebc2b9-5b4c-42cd-8e4b-65ddb427b7ff"  # cosyne distance
# add odd even trials as internal control

# %% using the newer loaders
session_FOVs = [data_loader.get_session_FOVs(eid) for eid in eids]
common_session_FOVs = sorted(list(set.intersection(*[set(s) for s in session_FOVs])))
imaging_data = data_loader.load_chronic_imaging(eids, FOVs=common_session_FOVs, processing_fn=process_imaging_data)


# %% extract all responses
responses = {}
responses_cv = {}
for eid in tqdm(eids):
    fovs = data_loader.get_session_FOVs(eid)
    trials_df = load_behavior_data(eid)

    # these are now stacked along FOVs
    event_responses = []
    metadata = []
    for fov in fovs:
        event_responses.append(extract_event_based_responses(imaging_data[eid][fov], trials_df, event_definitions))
        metadata.append(imaging_data[eid][fov].metadata)
    event_responses = pd.concat(event_responses)
    metadata = pd.concat(metadata)

    responses[eid] = (event_responses, metadata)  # FIXME this isn't great, leads to hard to read syntax later on

    # this is for the cross validation
    trials_df_even = trials_df.iloc[::2]
    trials_df_odd = trials_df.iloc[1::2]

    event_responses_odd = []
    event_responses_even = []
    for fov in fovs:
        event_responses_odd.append(extract_event_based_responses(imaging_data[eid][fov], trials_df_odd, event_definitions))
        event_responses_even.append(extract_event_based_responses(imaging_data[eid][fov], trials_df_even, event_definitions))

    responses_cv[eid] = {}
    responses_cv[eid]["odd"] = pd.concat(event_responses_odd)
    responses_cv[eid]["even"] = pd.concat(event_responses_even)


# %%
def get_common_roicat_UCIDs(eids: list[str], responses: dict):
    common_roicat_UCIDs = np.array(list(set.intersection(*[set(responses[eid][1]["roicat_UCID"].values) for eid in eids])))
    common_roicat_UCIDs = common_roicat_UCIDs[~pd.isna(common_roicat_UCIDs)]
    common_roicat_UCIDs = common_roicat_UCIDs[~(common_roicat_UCIDs == "nan")]

    # do the quality control here
    refined_ids = []
    for eid in eids:
        subset = responses[eid][1].set_index("roicat_UCID").loc[common_roicat_UCIDs]
        refined_ids.append(set(subset.query("iscell > 0.5 & cluster_silhouette > 0.2").index))

    return np.array(list(set.intersection(*refined_ids)))


# %% correlations - unit feature vectors, pairwise, by eid
# a df with all the pairwise eids combinations and their time delta
eid_combos = list(combinations(eids, 2))
eid_combos = pd.DataFrame(eid_combos, columns=["eid_a", "eid_b"])
for i, row in eid_combos.iterrows():
    eid_a = row["eid_a"]
    eid_b = row["eid_b"]
    t1 = datetime.fromisoformat(sessions_df.query("id == @eid_a")["start_time"].values[0])
    t2 = datetime.fromisoformat(sessions_df.query("id == @eid_b")["start_time"].values[0])
    eid_combos.loc[i, "dt"] = (t2 - t1).days

event_order = np.sort(list(event_definitions.keys()))

for i, row in eid_combos.iterrows():
    eid_combos.loc[i, "n_commmon"] = len(get_common_roicat_UCIDs([row["eid_a"], row["eid_b"]], responses))

# %% extraction run for all
from scipy.stats import spearmanr


def metric(x, y):
    return spearmanr(x, y)[0]


results = []
for i, row in eid_combos.iterrows():
    common_roicat_UCIDs = get_common_roicat_UCIDs([row["eid_a"], row["eid_b"]], responses)

    # subselect cells
    response_pair = []
    for eid in [row["eid_a"], row["eid_b"]]:
        response_pair.append(responses[eid][0].set_index(responses[eid][1]["roicat_UCID"]).loc[common_roicat_UCIDs])

    rhos = np.zeros(common_roicat_UCIDs.shape[0])
    rhos_feature_shuffle = np.zeros(common_roicat_UCIDs.shape[0])
    sel = ["fback1", "fback0", "choiceL", "choiceR"]
    for i, roicat_ucid in enumerate(common_roicat_UCIDs):
        a = response_pair[0].loc[roicat_ucid]
        b = response_pair[1].loc[roicat_ucid]
        # a = a.loc[sel]
        # b = b.loc[sel]
        valid_ix = ~np.logical_or(pd.isna(a), pd.isna(b))
        a_values = a.loc[valid_ix].values
        b_values = b.loc[valid_ix].values
        rhos[i] = metric(a_values, b_values)

        # feature shuffle
        np.random.shuffle(a_values)
        rhos_feature_shuffle[i] = metric(a_values, b_values)

    # neuron shuffle
    rhos_neuron_shuffle = np.zeros(common_roicat_UCIDs.shape[0])
    common_roicat_UCIDs_shuffle = copy(common_roicat_UCIDs)
    np.random.shuffle(common_roicat_UCIDs_shuffle)
    for i, (roicat_ucid_a, roicat_ucid_b) in enumerate(
        zip(common_roicat_UCIDs, common_roicat_UCIDs_shuffle),
    ):
        a = response_pair[0].loc[roicat_ucid_a]
        b = response_pair[1].loc[roicat_ucid_b]
        valid_ix = ~np.logical_or(pd.isna(a), pd.isna(b))
        a_values = a.loc[valid_ix].values
        b_values = b.loc[valid_ix].values
        rhos_neuron_shuffle[i] = metric(a_values, b_values)

    # neuron within brain region shuffle
    rhos_region_shuffle = np.zeros(common_roicat_UCIDs.shape[0])

    # get brain regions for roicat_UCID
    # is this ugly expression
    brain_regions = responses[row["eid_a"]][1].set_index("roicat_UCID").loc[common_roicat_UCIDs, "region_labels"].values
    q = pd.DataFrame(zip(common_roicat_UCIDs, brain_regions), columns=["ucid", "brain_region"]).sort_values("brain_region")
    roicat_UCIDs = q["ucid"].values
    roicat_UCIDs_region_shuffle = []
    for brain_region, group in q.groupby("brain_region"):
        ucids = group["ucid"].values
        np.random.shuffle(ucids)
        roicat_UCIDs_region_shuffle.append(ucids)
    roicat_UCIDs_region_shuffle = np.concatenate(roicat_UCIDs_region_shuffle)

    for i, (roicat_ucid_a, roicat_ucid_b) in enumerate(
        zip(roicat_UCIDs, roicat_UCIDs_region_shuffle),
    ):
        a = response_pair[0].loc[roicat_ucid_a]
        b = response_pair[1].loc[roicat_ucid_b]
        valid_ix = ~np.logical_or(pd.isna(a), pd.isna(b))
        a_values = a.loc[valid_ix].values
        b_values = b.loc[valid_ix].values
        rhos_region_shuffle[i] = metric(a_values, b_values)

    # storing the result for easier plotting
    result_df = pd.DataFrame(columns=["eid_a", "eid_b", "dt", "roicat_UCID", "p", "brain_region"])
    result_df["roicat_UCID"] = common_roicat_UCIDs
    result_df["rho"] = rhos
    result_df["rho_feature_shuffle"] = rhos_feature_shuffle
    result_df["rho_neuron_shuffle"] = rhos_neuron_shuffle
    result_df["rho_region_shuffle"] = rhos_region_shuffle
    result_df["brain_region"] = (
        responses[row["eid_a"]][1].set_index("roicat_UCID").loc[common_roicat_UCIDs, "region_labels"].values
    )
    result_df["eid_a"] = row["eid_a"]
    result_df["eid_b"] = row["eid_b"]
    result_df["dt"] = row["dt"]
    results.append(result_df)

results = pd.concat(results, axis=0)
results["dt"] = results["dt"].astype("int")

if selection == "all cells":
    results_ = copy(results)
if selection == "positively correl only":
    # subselection to only positively correlated
    selection_ids = results.query("dt == 1 & rho > 0")["roicat_UCID"].values
    results_ = results.loc[results["roicat_UCID"].isin(selection_ids)]

# %% the CV self validation
intra_session_rhos = {}
for eid in eids:
    metadata = responses[eid][1]

    roicat_UCIDs = metadata.query("iscell > .5 & cluster_silhouette > .2")["roicat_UCID"]
    roicat_UCIDs = roicat_UCIDs[~pd.isna(roicat_UCIDs)]
    roicat_UCIDs = roicat_UCIDs[~(roicat_UCIDs == "nan")]

    rhos = np.zeros(roicat_UCIDs.shape[0])
    for i, roicat_ucid in enumerate(roicat_UCIDs):
        a = responses_cv[eid]["odd"].set_index(metadata["roicat_UCID"]).loc[roicat_ucid]
        b = responses_cv[eid]["even"].set_index(metadata["roicat_UCID"]).loc[roicat_ucid]
        valid_ix = ~np.logical_or(pd.isna(a), pd.isna(b))
        a_values = a.loc[valid_ix].values
        b_values = b.loc[valid_ix].values
        rhos[i] = metric(a_values, b_values)

    intra_session_rhos[eid] = rhos

# %% PLOT PREP
plots_folder = Path(__file__).parent / "local" / "plots"

hue_colors = dict(
    zip(
        ["rho", "rho_feature_shuffle", "rho_neuron_shuffle", "rho_region_shuffle"],
        [
            "#5854CD",
            "#CFCFCF",
            "#979797",
            "#626262",
        ],
    )
)
# %% plot some feature vectors
data = response_pair[0].values.flatten()
data = data[~pd.isna(data)]
vmin, vmax = np.percentile(data, (5, 99))
axes = sns.heatmap(response_pair[0].T, vmin=vmin, vmax=vmax, cmap="viridis")
axes.set_xticklabels("")
axes.set_yticks(0.5 + np.arange(len(response_pair[0].columns)))
axes.set_yticklabels(response_pair[0].columns)
axes.set_xticks([])

# %% the upper bound calculation
bins = np.linspace(-1, 1, 75)
fig, axes = plt.subplots()
eid_colors = dict(zip(eids, sns.color_palette("husl", n_colors=len(eids))))
for eid in eids:
    axes.hist(intra_session_rhos[eid], bins=bins, alpha=0.5, color=eid_colors[eid], density=True)
    # axes.axvline(np.median(intra_session_rhos[eid]), lw=1, color="k", alpha=0.6)
axes.set_xlabel("spearmans ρ")
axes.set_ylabel("density")
axes.set_title(f"intra session correlation (odd/even)\n{subject} - {session_type}")
sns.despine(fig)

# %% histograms
bins = np.linspace(-1, 1, 75)
dts = np.sort(results["dt"].unique())
dt_colors = dict(zip(dts, sns.color_palette("viridis", n_colors=dts.shape[0])))

fig, axes = plt.subplots()
axes.hist(
    np.concatenate([intra_session_rhos[eid] for eid in eids]),
    bins=bins,
    color="gray",
    histtype="step",
    alpha=0.8,
    density=True,
    label="cv",
)

for dt in dts:
    axes.hist(
        results.query(f"dt == {dt}")["rho"].values, bins=bins, color=dt_colors[dt], alpha=0.7, density=True, label=f"dt={dt}"
    )

sns.despine(fig)
axes.legend()
axes.set_ylabel("density")
axes.set_xlabel("spearmans ρ")
axes.set_title(f"session to session correlations\nall neurons {subject} - {session_type}")
if save_plots:
    fig.savefig(plots_folder / f"histograms - {subject} - {session_type} - {selection}.png")

# %%
fig, axes = plt.subplots()
for eid in eids:
    date = sessions_df.set_index("id").loc[eid]["start_time"]
    axes.bar(x=datetime.fromisoformat(date), height=np.average(intra_session_rhos[eid]), color="gray")

# %%
results_m = pd.melt(
    results_[["rho", "rho_feature_shuffle", "rho_neuron_shuffle", "rho_region_shuffle", "dt", "brain_region"]],
    id_vars=["dt", "brain_region"],
    value_vars=["rho", "rho_feature_shuffle", "rho_neuron_shuffle", "rho_region_shuffle"],
)


# %% All cells combined bar plot
def calc_linear_model(x, y, nx=100, nm=2000):
    """cacluates linear regression on data points x and y
    calculates confidence intervals by stochastically sampling the model"""
    linreg = linregress(x, y)
    xm = np.linspace(x[0], x[-1], nx)
    Y = np.zeros((nx, nm))
    ms = linreg.slope + np.random.randn(nm) * linreg.stderr
    bs = linreg.intercept + np.random.randn(nm) * linreg.intercept_stderr

    for i in range(nm):
        Y[:, i] = bs[i] + xm * ms[i]
    return linreg, xm, Y


# for all cells
results_m_median = results_m[["dt", "variable", "value"]].groupby(["dt", "variable"]).apply(np.median).reset_index()

fig, axes = plt.subplots()
for var, group in results_m_median.groupby("variable"):
    x = group["dt"].values
    y = group[0].values
    axes.plot(x, y, ".", color=hue_colors[var])

    linreg, xm, Y = calc_linear_model(x, y)
    # add line
    axes.plot(x, linreg.intercept + x * linreg.slope, lw=2, color=hue_colors[var], label=var)
    # add shading
    axes.fill_between(xm, *np.percentile(Y, (5, 95), axis=1), alpha=0.4, color=hue_colors[var], linewidth=0)
    if var == "rho":
        axes.set_title(f"{subject} - {session_type} - all cells\nslope:{linreg.slope:.2e} - p:{linreg.pvalue:.2e}")

axes.legend()
sns.despine(fig)
axes.set_xlabel("time between sessions (days)")
axes.set_ylabel("spearmans ρ")

intra_sess_rho_median = np.median([np.median(intra_session_rhos[eid]) for eid in eids])
axes.axhline(intra_sess_rho_median, linestyle=":", label="intra session xval", color="k")

if save_plots:
    fig.savefig(plots_folder / f"line plot - {subject} - {session_type} - {selection}.png")

# %% resolved by brain region
n = 500  # minimum this number of neurons per group
counts = results_m["brain_region"].value_counts()
brain_regions = counts[counts > n].index
region_colors = dict(zip(brain_regions, sns.color_palette("husl", n_colors=brain_regions.shape[0])))

res = (
    results_m[["dt", "brain_region", "variable", "value"]]
    .groupby(["dt", "brain_region", "variable"])
    .apply(np.median)
    .reset_index()
)

for brain_region in brain_regions:
    res_ = res.groupby("brain_region").get_group(brain_region)
    fig, axes = plt.subplots()
    for var, group in res_.groupby("variable"):
        x = group["dt"].values
        y = group[0].values
        color = region_colors[brain_region] if var == "rho" else hue_colors[var]
        axes.plot(x, y, ".", color=color)

        linreg, xm, Y = calc_linear_model(x, y)
        # add line
        axes.plot(x, linreg.intercept + x * linreg.slope, lw=2, color=color, label=var)
        # add shading
        axes.fill_between(xm, *np.percentile(Y, (5, 95), axis=1), alpha=0.4, color=color, linewidth=0)
        if var == "rho":
            axes.set_title(f"{subject} - {session_type} - {brain_region}\nslope:{linreg.slope:.2e} - p:{linreg.pvalue:.2e}")

    axes.legend()
    sns.despine(fig)
    axes.set_xlabel("time between sessions (days)")
    axes.set_ylabel("spearmans ρ")
    if save_plots:
        fig.savefig(plots_folder / f"line plot {subject} - {session_type} - {brain_region.replace('/', '-')}.png")


# %% barplot of slopes per brain region
slopes_df = []
for brain_region in brain_regions:
    res_ = res.groupby(["brain_region", "variable"]).get_group((brain_region, "rho"))
    x = res_["dt"].values
    y = res_[0].values
    linreg, xm, Y = calc_linear_model(x, y)
    slopes_df.append(dict(slope=linreg.slope, p=linreg.pvalue, err=linreg.stderr, brain_region=brain_region))
slopes_df = pd.DataFrame(slopes_df)
slopes_df = slopes_df.sort_values("slope").reset_index()

fig, axes = plt.subplots()

for i, row in slopes_df.iterrows():
    color = "royalblue" if row["p"] < 0.05 else "grey"
    axes.bar(height=row["slope"], x=i, color=color)
    axes.plot([i, i], [row["slope"] - row["err"], row["slope"] + row["err"]], lw=1, color="k")
    axes.axvline(i, lw=0.5, linestyle=":", color="k")

axes.set_title(f"slopes per brain region\n{subject} - {session_type}")
axes.set_xticks(np.arange(slopes_df.shape[0]))
axes.set_xticklabels(slopes_df["brain_region"], rotation=90, ha="center")
axes.set_ylabel("slope")
sns.despine(fig)
fig.tight_layout()
if save_plots:
    fig.savefig(plots_folder / f"slopes barplot {subject} - {session_type} - {brain_region.replace('/', '-')}.png")


# %% leftover from schonoover repdrift
# from numpy import linalg
# from itertools import combinations

# def angle(a: np.ndarray, b: np.ndarray):
#     # calculates the angle between to vectors (as 1d arrays)
#     return np.arccos(np.dot(a, b) / (linalg.norm(a) * linalg.norm(b)))

# def frobenius(A, B):
#     # frobenius norm of vectors or matrices
#     # faster than spatial.distance.euclidean(A.flatten(),B.flatten() ) :)
#     return np.sqrt(np.sum((A - B) ** 2))

# def angle_eval(response: np.ndarray):
#     # all pairwise angles between vectors in neural response space
#     # -> each dimension is the response magnitude of a neuron to a given stimulus
#     # dimensionality of response: neurons x stimuli

#     # returns: n-1 stim x n_pairwise combos
#     _, n_stims = response.shape
#     combos = []
#     for i in range(n_stims):
#         ix = list(range(n_stims))
#         ix.remove(i)
#         combos.append(list(combinations(ix, 2)))

#     return np.array([[angle(response[:, i], response[:, j]) for (i, j) in combos[k]] for k in range(len(combos))])

# common_roicat_UCIDs = get_common_roicat_UCIDs(eids, responses)
# for i, row in eid_combos.iterrows():
#     # subselect cells
#     response_pair = []
#     for eid in [row["eid_a"], row["eid_b"]]:
#         response_pair.append(responses[eid][0].set_index(responses[eid][1]["roicat_UCID"]).loc[common_roicat_UCIDs])

#     a = response_pair[0].loc[common_roicat_UCIDs]
#     b = response_pair[1].loc[common_roicat_UCIDs]
#     valid_ix = ~np.logical_or(pd.isna(a).values, pd.isna(b).values)
#     a_values = a.values[:, np.any(valid_ix, axis=0)]
#     b_values = b.values[:, np.any(valid_ix, axis=0)]

#     dist = frobenius(angle_eval(a_values), angle_eval(b_values))

#     eid_combos.loc[i, "dist"] = dist

# max_day = eid_combos["dt"].max() + 1
# # sns.barplot(data=results_, y="value", x="dt", hue="variable", palette=hue_colors,  ax=axes)
# sns.barplot(
#     data=eid_combos,
#     x="dt",
#     y="dist",
#     order=np.arange(1, max_day),
# )
