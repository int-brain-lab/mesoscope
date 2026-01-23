# %%
from pathlib import Path
from tqdm import tqdm
import pickle
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

from iblatlas.atlas import AllenAtlas

one = ONE()

base_folder = Path("/mnt/s0/Data/Subjects")


# %% get all SP058 eids
def parse_cannonicals_sessions_file(path: Path) -> pd.DataFrame:
    """parses the text file of canonical sessions into a
    DataFrame for easier groupby selection of sessions"""

    with open(path, "r") as fH:
        canonical_sessions = [line.strip() for line in fH.readlines() if line != "\n"]
    session_paths = [sess.replace("\\", "/") for sess in canonical_sessions]
    eids = [str(one.path2eid(path)) for path in session_paths]
    df = pd.DataFrame(one.alyx.rest("sessions", "list", django=f"id__in,{eids}"))
    return df.drop(columns=["number", "lab", "projects", "url"]).set_index("id")


# %% new data loading approach
def load_imaging_data(eid: str, fov: str, deconvolved: bool = True) -> nap.TsdFrame:
    session_path = base_folder / one.eid2path(eid).session_path_short()
    fov_collection = f"alf/{fov}"

    # our uuids
    fov_uuids = pd.read_csv(session_path / fov_collection / "mpciROIs.uuids.csv")

    # roicat UCIDs
    roicat_UCIDs = pd.read_csv(
        session_path / f"{fov_collection}/mpciROIs.clusterUIDs.csv",
        skip_blank_lines=False,
        header=None,
        names=["roicat_UCID"],
    )
    roi_info = pd.concat([fov_uuids, roicat_UCIDs], axis=1)

    # other suite2p info: cell classifier
    roi_info["iscell"] = np.load(session_path / fov_collection / "mpciROIs.cellClassifier.npy")

    # brain region estimate
    if (session_path / fov_collection / "mpciROIs.brainLocationIds_ccf_2017.npy").exists():
        roi_info["region_ids"] = np.load(session_path / fov_collection / "mpciROIs.brainLocationIds_ccf_2017.npy")
    else:
        roi_info["region_ids"] = np.load(session_path / fov_collection / "mpciROIs.brainLocationIds_ccf_2017_estimate.npy")

    atlas = AllenAtlas()
    roi_info["region_labels"] = atlas.regions.id2acronym(roi_info["region_ids"].values)
    roi_info["fov"] = fov

    # roicat data
    chronic_folder = base_folder / subject / "Chronic"
    roicat_metrics = ["cluster_intra_means", "cluster_intra_maxs", "cluster_intra_mins", "cluster_silhouette"]

    # get roicat data
    file = next(chronic_folder.glob(f"*{fov}*tracking.results.pkl"))
    with open(file, "rb") as fH:
        roicat_output = pickle.load(fH)

    cluster_uids = pd.read_csv(
        chronic_folder / f"{fov}.clusterUIDs_all.csv",
        header=None,
        names=["roicat_UCID"],
    )
    roicat_df = pd.DataFrame({metric: roicat_output["quality_metrics"][metric] for metric in roicat_metrics})

    # set index to cluster labels
    roicat_df.index = np.array(roicat_output["quality_metrics"]["cluster_labels_unique"], dtype="int32")
    # merge with cluster_uids
    roicat_df["roicat_UCID"] = cluster_uids

    # combine roi_info with roicat metrics
    roi_info = pd.merge(roi_info, roicat_df, on="roicat_UCID", how="left")

    # fluorescence data
    dataset = "mpci.ROIActivityDeconvolved.npy" if deconvolved else "mpci.ROIActivityF.npy"

    # load all traces (suite2p) for one FOV
    suite2p_data_fov = np.load(session_path / fov_collection / dataset)
    times = np.load(session_path / fov_collection / "mpci.times.npy")
    assert suite2p_data_fov.shape[1] == roi_info.shape[0]
    # as a pynapple object, merged with the metadata
    return nap.TsdFrame(t=times, d=suite2p_data_fov, metadata=roi_info)


# load behavior
def load_behavior_data(eid: str) -> pd.DataFrame:
    """helper to load the trials table belonging to an eid and adding the relevant columns
    necessary for creating the response feature vectors"""
    trials_table_file = list((base_folder / one.eid2path(eid).session_path_short()).rglob("*trials.table*"))
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


def get_fovs(eid: str) -> np.ndarray:
    """helper to get all FOVs for a session"""
    session_path = base_folder / one.eid2path(eid).session_path_short()
    # infer fov collections
    fov_folders = list(session_path.rglob("*alf/FOV_??*"))
    return np.sort([folder.parts[-1] for folder in fov_folders])


# %%
# selecting the subject
subject = "SP058"
# session_type selection: biased or training
session_type = "biased"
# session_type = "training"

if session_type == "biased":
    event_definitions = event_definitions_biasedCW
elif session_type == "training":
    event_definitions = event_definitions_trainingCW

sessions_df = parse_cannonicals_sessions_file(Path(__file__).parent / "canonical_sessions.txt")
sessions_df = sessions_df.groupby("subject").get_group(subject).sort_values("start_time")

sessions_df = sessions_df.loc[sessions_df["task_protocol"].str.contains(session_type)]  # <- here selection
eids = sessions_df.index.values

# center_eid = "20ebc2b9-5b4c-42cd-8e4b-65ddb427b7ff"  # cosyne distance
# add odd even trials as internal control

# %% load all imaging data
imaging_data = {}
for eid in tqdm(eids):
    imaging_data[eid] = {}
    fovs = get_fovs(eid)
    for fov in fovs:
        data = load_imaging_data(eid, fov)
        imaging_data[eid][fov] = process_imaging_data(data)

# %% extract all responses
responses = {}
for eid in tqdm(eids):
    fovs = get_fovs(eid)
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

# %% correlations - unit feature vectors, pairwise, by eid
# a df with all the pairwise eids combinations and their time delta
eid_combos = list(combinations(eids, 2))
eid_combos = pd.DataFrame(eid_combos, columns=["eid_a", "eid_b"])
for i, row in eid_combos.iterrows():
    t1 = datetime.fromisoformat(sessions_df.loc[row["eid_a"], "start_time"])
    t2 = datetime.fromisoformat(sessions_df.loc[row["eid_b"], "start_time"])
    eid_combos.loc[i, "dt"] = (t2 - t1).days

event_order = np.sort(list(event_definitions.keys()))


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
        # refined_ids.append(set(subset.query("iscell > 0.5 & cluster_silhouette > 0.2").index))

    return np.array(list(set.intersection(*refined_ids)))


# %% extraction run for all
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
        rhos[i] = np.corrcoef(a_values, b_values)[0, 1]

        # feature shuffle
        np.random.shuffle(a_values)
        rhos_feature_shuffle[i] = np.corrcoef(a_values, b_values)[0, 1]

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
        rhos_neuron_shuffle[i] = np.corrcoef(a_values, b_values)[0, 1]

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
        rhos_region_shuffle[i] = np.corrcoef(a_values, b_values)[0, 1]

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


# %% All cells combined
plots_folder = Path(__file__).parent / "local" / "plots"

results_ = pd.melt(
    results[["rho", "rho_feature_shuffle", "rho_neuron_shuffle", "rho_region_shuffle", "dt", "brain_region"]],
    id_vars=["dt", "brain_region"],
    value_vars=["rho", "rho_feature_shuffle", "rho_neuron_shuffle", "rho_region_shuffle"],
)

hue_colors = dict(
    zip(
        ["rho", "rho_feature_shuffle", "rho_neuron_shuffle", "rho_region_shuffle"],
        [
            "#5854CD",
            "#B6B6B6",
            "#7B7B7B",
            "#626262",
        ],
    )
)

fig, axes = plt.subplots()
max_day = results["dt"].max() + 1
sns.barplot(data=results_, y="value", x="dt", hue="variable", palette=hue_colors, order=np.arange(1, max_day), ax=axes)
axes.axhline(0, linestyle=":", lw=2, alpha=0.5, color="k")
axes.set_xlabel("time between sessions (days)")
axes.set_ylabel("spearmans ρ")
xs = results_.groupby("variable").get_group("rho")["dt"]
ys = results_.groupby("variable").get_group("rho")["value"]
linreg = linregress(xs, ys)
plt.gca().set_title(f"{subject} - {session_type} - all cells\nslope:{linreg.slope:.2e} - p:{linreg.pvalue:.2e}")

sns.despine(fig)
fig.savefig(plots_folder / f"{subject} - {session_type} - {'all cells'}.png")


# %%
n = 500
counts = results_["brain_region"].value_counts()
brain_regions = counts[counts > n].index

for brain_region in brain_regions:
    res = results_.groupby("brain_region").get_group(brain_region)

    region_colors = dict(zip(brain_regions, sns.color_palette("husl", n_colors=brain_regions.shape[0])))

    hue_colors = dict(
        zip(
            ["rho", "rho_feature_shuffle", "rho_neuron_shuffle", "rho_region_shuffle"],
            [
                region_colors[brain_region],
                "#B6B6B6",
                "#7B7B7B",
                "#626262",
            ],
        )
    )

    fig, axes = plt.subplots()
    max_day = results["dt"].max() + 1
    sns.barplot(data=res, y="value", x="dt", hue="variable", palette=hue_colors, order=np.arange(1, max_day), ax=axes)
    axes.axhline(0, linestyle=":", lw=2, alpha=0.5, color="k")
    axes.set_xlabel("time between sessions (days)")
    axes.set_ylabel("spearmans ρ")
    xs = res.groupby("variable").get_group("rho")["dt"]
    ys = res.groupby("variable").get_group("rho")["value"]
    linreg = linregress(xs, ys)
    plt.gca().set_title(f"{subject} - {session_type} - {brain_region}\nslope:{linreg.slope:.2e} - p:{linreg.pvalue:.2e}")

    sns.despine(fig)
    fig.savefig(plots_folder / f"{subject} - {session_type} - {brain_region.replace('/', '-')}.png")

# %%
