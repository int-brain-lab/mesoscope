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
from copy import copy
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


def get_roi_info(eids):
    """for the given eids, gather all the information about ROIs in one dataframe"""

    atlas = AllenAtlas()
    roi_info = []
    for eid in tqdm(eids):
        session_path = base_folder / one.eid2path(eid).session_path_short()

        # infer fov collections
        fov_folders = list(session_path.glob("*alf/FOV*"))
        fov_collections = ["/".join(str(folder).split("/")[-2:]) for folder in fov_folders]

        roi_info_session = []
        for fov_collection in fov_collections:
            fov = fov_collection.split("/")[1]
            # the session level uuids
            fov_uuids = pd.read_csv(session_path / fov_collection / "mpciROIs.uuids.csv")

            # uuids that are constant across sessions - ROIcat output
            # this can be replaced with a regular one. call once the data is available
            roicat_uuids = pd.read_csv(
                session_path / f"{fov_collection}/mpciROIs.clusterUIDs.csv",
                skip_blank_lines=False,
                header=None,
                names=["roicat_UCID"],
            )
            df = pd.concat([fov_uuids, roicat_uuids], axis=1)

            # other suite2p info: cell classifier
            df["iscell"] = np.load(session_path / fov_collection / "mpciROIs.cellClassifier.npy")

            # brain region
            if (session_path / fov_collection / "mpciROIs.brainLocationIds_ccf_2017.npy").exists():
                df["region_ids"] = np.load(session_path / fov_collection / "mpciROIs.brainLocationIds_ccf_2017.npy")
            else:
                df["region_ids"] = np.load(session_path / fov_collection / "mpciROIs.brainLocationIds_ccf_2017_estimate.npy")
            df["region_labels"] = atlas.regions.id2acronym(df["region_ids"].values)

            df["fov"] = fov
            roi_info_session.append(df)

        df = pd.concat(roi_info_session, axis=0)
        df["eid"] = eid
        roi_info.append(df)

    roi_info = pd.concat(roi_info, axis=0)
    return roi_info.reset_index()


def get_data_all_fovs(
    eid: str,
    roi_info: pd.DataFrame,
    deconvolved=True,
):
    """for a given eid, get all the imaging traces, combining all fovs
    note: curation is realized by dropping rois in roi_info"""

    dataset = "mpci.ROIActivityDeconvolved.npy" if deconvolved else "mpci.ROIActivityF.npy"
    roi_info_session = roi_info.groupby("eid").get_group(eid)

    suite2p_data = []

    # ensure iteration over fovs in order
    fovs = np.sort(roi_info_session["fov"].unique())
    for fov in fovs:
        roi_info_fov = roi_info_session.groupby("fov").get_group(fov)
        # load
        suite2p_data_fov = np.load(base_folder / one.eid2path(eid).session_path_short() / f"alf/{fov}/{dataset}")
        # curation
        ix = roi_info_fov["index"].values
        suite2p_data_fov = suite2p_data_fov[:, ix]
        # store
        suite2p_data.append(suite2p_data_fov)

    session_data = np.concatenate(suite2p_data, axis=1)
    # roi_info_session = pd.concat(roi_info_session, axis=0)

    # should be the same for all fovs, so we can just pick any
    # TODO verify!!
    times = np.load(base_folder / one.eid2path(eid).session_path_short() / f"alf/{fov}/mpci.times.npy")
    return nap.TsdFrame(
        t=times,
        d=session_data,
    )


def extract_data(
    eids: list[str],
    roi_info: pd.DataFrame,
    events_definitions: dict,
) -> dict[str]:
    """for all eids, load the imaging data, extract the responses defined by the
    event definitions, return a dict[eid] with keys 'response' and 'session_info'"""
    extracted_data = {}
    for eid in eids:
        # load imaging data
        session_data = get_data_all_fovs(eid, roi_info, deconvolved=True)

        # normalize
        a, b = np.percentile(session_data, (20, 99))
        session_data = (session_data - a) / (b - a)

        # load behavior
        trials_table_file = list((base_folder / one.eid2path(eid).session_path_short()).rglob("*trials.table*"))
        assert len(trials_table_file) == 1
        trials_df = pd.read_parquet(trials_table_file[0])
        trials_df = add_info_to_trials_table(trials_df)

        stim_avgs = {}
        for event, definition in events_definitions.items():
            _trials_df = trials_df.query(definition["query"])
            timestamps = _trials_df[definition["align_event"]].values

            # timestamps = timestamps[~pd.isna(timestamps)]  # there are trials with choices but no firstMovement_times ??

            tensor = nap.compute_perievent_continuous(session_data, nap.Ts(timestamps), minmax=definition["window"])
            # tensor.shape is timepoints, trials, cells
            # time and trial average
            if timestamps.shape[0] > 10:  # exclude trial types with too little trials
                stim_avgs[event] = np.average(tensor, axis=(0, 1))
            else:
                nans = np.ones(tensor.shape[-1])
                nans[:] = np.nan
                stim_avgs[event] = nans

        extracted_data[eid] = stim_avgs
    return extracted_data


def get_common_roicat_UCIDs(eids, roi_info):
    """helper to get all common roicat UCIDs for a set of eids"""
    all_ucids = [set(roi_info.groupby("eid").get_group(eid)["roicat_UCID"].values) for eid in eids]
    # make sure that '' or np.nan is not part of this list
    common_ucids = np.array(list(set.intersection(*all_ucids)))
    nan_ix = np.logical_or(pd.isna(common_ucids), common_ucids == "nan")
    return common_ucids[~nan_ix]


def get_data_indices(common_roicat_UCIDs, roi_info_session):
    # ensure that roi_info_session has the same order of fovs (numerically ascending)
    # as in the extraction loop
    fovs = np.sort(roi_info_session["fov"].unique())
    roi_info_session = pd.concat([roi_info_session.groupby("fov").get_group(fov) for fov in fovs])

    # drop the untracked cells
    roi_info_session = roi_info_session.loc[~pd.isna(roi_info_session["roicat_UCID"])]
    # get the indices
    return roi_info_session.set_index("roicat_UCID").loc[common_roicat_UCIDs]["index"].values


# %% definitions and run

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
roi_info = get_roi_info(eids)

# the curation step
roi_info = roi_info.loc[roi_info["iscell"] > 0.5]
# TODO add back in the curation by roicat metrics

# %% load data
extracted_data = extract_data(eids, roi_info, event_definitions)

# %% eids and their combinations
eid_combos = list(combinations(eids, 2))
eid_combos = pd.DataFrame(eid_combos, columns=["eid_a", "eid_b"])
for i, row in eid_combos.iterrows():
    t1 = datetime.fromisoformat(sessions_df.loc[row["eid_a"], "start_time"])
    t2 = datetime.fromisoformat(sessions_df.loc[row["eid_b"], "start_time"])
    eid_combos.loc[i, "dt"] = (t2 - t1).days


# %% pairwise correlatoins
event_order = np.sort(list(event_definitions.keys()))
# selection = ["fback1", "choiceL", "choiceR"]
# event_order = ["fback1", "choiceL", "choiceR"]

results = []
for i, row in eid_combos.iterrows():
    eid_a = row["eid_a"]
    eid_b = row["eid_b"]

    # common roicat UCIDs
    common_roicat_UCIDs = get_common_roicat_UCIDs([eid_a, eid_b], roi_info)

    # get the matching indices for roicat UCIDs to index into the extracted data
    indices_a = get_data_indices(common_roicat_UCIDs, roi_info.groupby("eid").get_group(eid_a))
    indices_b = get_data_indices(common_roicat_UCIDs, roi_info.groupby("eid").get_group(eid_b))
    assert indices_a.shape == indices_b.shape

    session_avgs_a = extracted_data[eid_a]
    session_avgs_b = extracted_data[eid_b]

    a_mat = np.concatenate([session_avgs_a[s][:, np.newaxis] for s in event_order], axis=1)[indices_a, :]
    b_mat = np.concatenate([session_avgs_b[s][:, np.newaxis] for s in event_order], axis=1)[indices_b, :]

    ps = np.zeros(a_mat.shape[0])
    for i in range(a_mat.shape[0]):
        valid_ix = ~np.logical_or(pd.isna(a_mat[i, :]), pd.isna(b_mat[i, :]))
        ps[i] = np.corrcoef(a_mat[i, valid_ix], b_mat[i, valid_ix])[0, 1]

    # storing the result for easier plotting
    result_df = pd.DataFrame(columns=["eid_a", "eid_b", "dt", "roicat_UCID", "p", "brain_region"])
    result_df["roicat_UCID"] = common_roicat_UCIDs
    result_df["p"] = ps
    result_df["brain_region"] = (
        roi_info.groupby("eid").get_group(eid_a).set_index("roicat_UCID").loc[common_roicat_UCIDs, "region_labels"].values
    )
    result_df["eid_a"] = eid_a
    result_df["eid_b"] = eid_b
    result_df["dt"] = row["dt"]
    results.append(result_df)

results = pd.concat(results, axis=0)


# %% overall
fig, axes = plt.subplots()
sns.boxplot(data=results, y="p", x="dt", ax=axes)
plt.gca().set_title(f"{subject} - {session_type} - pairwise session correlations")
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
    sns.boxplot(data=results_, y="p", x="dt", ax=axes, color=region_colors[region])
    plt.gca().set_title(f"{subject} - {session_type} - {region}")
    axes.set_title(region)
    sns.despine(fig)


# %%
