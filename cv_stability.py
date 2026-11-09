# %%
import numpy as np
import pynapple as nap
from one.api import ONE
import matplotlib.pyplot as plt
import chronic_data_loader as data_loader
import pandas as pd
from trial_type_definitions import event_definitions_biasedCW, event_definitions_trainingCW, add_info_to_trials_table
from pathlib import Path

one = ONE()


# %%
def normalize(imaging_data: nap.TsdFrame) -> nap.TsdFrame:
    a, b = np.percentile(imaging_data.d, (20, 99), axis=0)
    d = (imaging_data - a[np.newaxis, :]) / (b[np.newaxis, :] - a[np.newaxis, :])
    return nap.TsdFrame(t=imaging_data.t, d=d, metadata=imaging_data.metadata)


# move to plotting helpers
def plot_TsdFrame(data: nap.TsdFrame):
    _, axes = plt.subplots()
    kwargs = dict(zip(["vmin", "vmax"], np.percentile(data.d[~pd.isna(data)], (5, 99))))
    extent = (data.t[0], data.t[-1], data.d.shape[-1], 0)
    axes.matshow(data.d.T, **kwargs, extent=extent)
    axes.set_aspect("auto")
    axes.set_xlabel("time (s)")
    axes.set_ylabel("ROI")
    axes.xaxis.set_ticks_position("bottom")
    return axes


# move to data loader
# TODO has localization specific code
def load_behavior_data(eid: str) -> pd.DataFrame:
    """helper to load the trials table belonging to an eid and adding the relevant columns
    necessary for creating the response feature vectors"""
    trials_table_file = list((BASE_FOLDER / one.eid2path(eid).session_path_short()).rglob("*trials.table*"))
    assert len(trials_table_file) == 1
    trials_df = pd.read_parquet(trials_table_file[0])
    trials_df = add_info_to_trials_table(trials_df)
    return trials_df


# mote to analysis helpers
def extract_event_based_responses_for_fov(
    fov_data: nap.TsdFrame,
    trials_df: pd.DataFrame,
    event_definitions: dict,
    mode='average',
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
            avg = np.average(tensor, axis=(0, 1))
            stim_avgs[event] = np.std(np.average(tensor, axis=0)) / avg
        else:
            nans = np.ones(tensor.shape[-1])
            nans[:] = np.nan
            stim_avgs[event] = nans

    return pd.DataFrame(stim_avgs)


def extract_event_based_responses_for_session(
    session_data: dict,
    trials_df: pd.DataFrame,
    event_definitions: dict,
):
    # these are now stacked along FOVs
    event_responses = []
    metadata = []
    fovs = sorted(list(session_data.keys()))
    for fov in fovs:
        event_responses.append(extract_event_based_responses_for_fov(session_data[fov], trials_df, event_definitions))
        metadata.append(session_data[fov].metadata)
    event_responses = pd.concat(event_responses)
    metadata = pd.concat(metadata)

    return event_responses, metadata


# %%

eid = "20ebc2b9-5b4c-42cd-8e4b-65ddb427b7ff"
session_data = data_loader.load_imaging_session(eid, one=one, processing_fn=normalize)

BASE_FOLDER = Path("/mnt/s0/Data/Subjects")
trials_df = load_behavior_data(eid)
event_responses, metadata = extract_event_based_responses_for_session(session_data, trials_df, event_definitions_biasedCW)

# %% 
fig, axes = plt.subplots()
axes.scatter(metadata['cluster_silhouette'].values, event_responses['blockL'].values,s=1)
axes.set_ylim(0,10)

# %% extract all responses
# responses = {}
# responses_cv = {}
# for eid in tqdm(eids):
    # fovs = data_loader.get_session_FOVs(eid)
#     trials_df = load_behavior_data(eid)

#     # these are now stacked along FOVs
#     event_responses = []
#     metadata = []
#     for fov in fovs:
#         event_responses.append(extract_event_based_responses(imaging_data[eid][fov], trials_df, event_definitions))
#         metadata.append(imaging_data[eid][fov].metadata)
#     event_responses = pd.concat(event_responses)
#     metadata = pd.concat(metadata)

#     responses[eid] = (event_responses, metadata)  # FIXME this isn't great, leads to hard to read syntax later on

#     # this is for the cross validation
#     trials_df_even = trials_df.iloc[::2]
#     trials_df_odd = trials_df.iloc[1::2]

#     event_responses_odd = []
#     event_responses_even = []
#     for fov in fovs:
#         event_responses_odd.append(extract_event_based_responses(imaging_data[eid][fov], trials_df_odd, event_definitions))
#         event_responses_even.append(extract_event_based_responses(imaging_data[eid][fov], trials_df_even, event_definitions))

#     responses_cv[eid] = {}
#     responses_cv[eid]["odd"] = pd.concat(event_responses_odd)
#     responses_cv[eid]["even"] = pd.concat(event_responses_even)
