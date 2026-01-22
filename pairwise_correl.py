# %%
from pathlib import Path
from tqdm import tqdm
import pickle
from itertools import combinations
from datetime import datetime
import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt
import seaborn as sns
from one.api import ONE
from trial_type_definitions import event_definitions_biasedCW, event_definitions_trainingCW, add_info_to_trials_table

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
def load_imaging_data(eid: str, fov: str, deconvolved: bool = True):
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
def load_behavior_data(eid):
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
    # normalize
    a, b = np.percentile(imaging_data, (20, 99))
    imaging_data = (imaging_data - a) / (b - a)
    return imaging_data


def get_fovs(eid):
    session_path = base_folder / one.eid2path(eid).session_path_short()
    # infer fov collections
    fov_folders = list(session_path.rglob("*alf/FOV_??*"))
    return np.sort([folder.parts[-1] for folder in fov_folders])


# %%
# selecting the subject
subject = "SP058"
# session_type selection: biased or training
session_type = "biased"

if session_type == "biased":
    event_definitions = event_definitions_biasedCW
elif session_type == "training":
    event_definitions = event_definitions_trainingCW

sessions_df = parse_cannonicals_sessions_file(Path(__file__).parent / "canonical_sessions.txt")
sessions_df = sessions_df.groupby("subject").get_group(subject).sort_values("start_time")

sessions_df = sessions_df.loc[sessions_df["task_protocol"].str.contains(session_type)]  # <- here selection
eids = sessions_df.index.values

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
    responses[eid] = (event_responses, metadata)

# %% correlations - unit feature vectors, pairwise, by eid
# pairwise eids combinations
eid_combos = list(combinations(eids, 2))
eid_combos = pd.DataFrame(eid_combos, columns=["eid_a", "eid_b"])
for i, row in eid_combos.iterrows():
    t1 = datetime.fromisoformat(sessions_df.loc[row["eid_a"], "start_time"])
    t2 = datetime.fromisoformat(sessions_df.loc[row["eid_b"], "start_time"])
    eid_combos.loc[i, "dt"] = (t2 - t1).days

event_order = np.sort(list(event_definitions.keys()))

# %% extract
results = []
for i, row in eid_combos.iterrows():
    common_roicat_UCIDs = np.array(
        list(
            set.intersection(
                set(responses[row["eid_a"]][1]["roicat_UCID"].values),
                set(responses[row["eid_b"]][1]["roicat_UCID"].values),
            )
        )
    )
    # filter out nan
    common_roicat_UCIDs = common_roicat_UCIDs[~pd.isna(common_roicat_UCIDs)]
    common_roicat_UCIDs = common_roicat_UCIDs[~(common_roicat_UCIDs == "nan")]

    # subselect cells
    response_pair = []
    for eid in [row["eid_a"], row["eid_b"]]:
        criteria = [
            responses[eid][1]["iscell"] > 0.5,
            responses[eid][1]["roicat_UCID"].isin(common_roicat_UCIDs),
            responses[eid][1]["cluster_silhouette"] > 0.2,
        ]

        ix = np.prod(pd.concat(criteria, axis=1).values, axis=1).astype("bool")
        index = responses[eid][1].loc[ix]["roicat_UCID"]
        response_pair.append(responses[eid][0].loc[ix].set_index(index))

    # we need to filter here again as it happens that iscell criterion fails in only one of the
    # two sessions
    common_roicat_UCIDs = np.array(
        list(
            set.intersection(
                set(response_pair[0].index.values),
                set(response_pair[1].index.values),
            )
        )
    )

    rhos = np.zeros(common_roicat_UCIDs.shape[0])
    sel = ["fback1", "fback0", "choiceL", "choiceR"]
    for i, roicat_ucid in enumerate(common_roicat_UCIDs):
        a = response_pair[0].loc[roicat_ucid]
        b = response_pair[1].loc[roicat_ucid]
        # a = a.loc[sel]
        # b = b.loc[sel]
        valid_ix = ~np.logical_or(pd.isna(a), pd.isna(b))
        rhos[i] = np.corrcoef(a.loc[valid_ix].values, b.loc[valid_ix].values)[0, 1]

    # storing the result for easier plotting
    result_df = pd.DataFrame(columns=["eid_a", "eid_b", "dt", "roicat_UCID", "p", "brain_region"])
    result_df["roicat_UCID"] = common_roicat_UCIDs
    result_df["rho"] = rhos
    result_df["brain_region"] = (
        responses[row["eid_a"]][1].set_index("roicat_UCID").loc[common_roicat_UCIDs, "region_labels"].values
    )
    result_df["eid_a"] = row["eid_a"]
    result_df["eid_b"] = row["eid_b"]
    result_df["dt"] = row["dt"]
    results.append(result_df)

results = pd.concat(results, axis=0)


# %% overall
from scipy.stats import linregress

fig, axes = plt.subplots()
axes.scatter(x=results["dt"], y=results["rho"], alpha=0.01)
xs = []
ys = []
for dt, group in results.groupby("dt"):
    xs.append(dt)
    ys.append(group["rho"].mean())

axes.plot(xs, ys, ".", markersize=10, color="k")
axes.axhline(0, linestyle=":", lw=2, alpha=0.5, color="k")
axes.set_xlabel("time between sessions (days)")
axes.set_ylabel("spearmans ρ")

linreg = linregress(xs, ys)
plt.gca().set_title(f"{subject} - {session_type} - slope:{linreg.slope:.5f}, pval:{linreg.pvalue:.2f}")
sns.despine(fig)

# %% brain region resolved
# filter into brain regions that have actual numbers
n = 500
counts = results["brain_region"].value_counts()
brain_regions = counts[counts > n].index

region_colors = dict(zip(brain_regions, sns.color_palette("husl", n_colors=brain_regions.shape[0])))

for region in brain_regions:
    fig, axes = plt.subplots()
    results_ = results.groupby("brain_region").get_group(region)
    axes.scatter(x=results_["dt"], y=results_["rho"], alpha=0.1, color=region_colors[region])
    xs = []
    ys = []
    for dt, group in results_.groupby("dt"):
        xs.append(dt)
        ys.append(group["rho"].mean())

    axes.plot(xs, ys, ".", markersize=10, color="k")
    axes.axhline(0, linestyle=":", lw=2, alpha=0.5, color="k")
    axes.set_xlabel("time between sessions (days)")
    axes.set_ylabel("spearmans ρ")

    linreg = linregress(xs, ys)
    plt.gca().set_title(f"{subject} - {session_type} - {region} - slope:{linreg.slope:.5f}, pval:{linreg.pvalue:.2f}")
    sns.despine(fig)

# %%
