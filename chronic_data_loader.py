# %%
from pathlib import Path
from typing import Optional, Literal, Dict, List
from tqdm import tqdm
import pickle
import re
import logging

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import pynapple as nap
from one.api import ONE

from iblatlas.atlas import AllenAtlas

BASE_FOLDER = Path("/mnt/s0/Data/Subjects")

_logger = logging.getLogger(__name__)


def parse_canonicals_sessions_file(
    path: Path,
    one: Optional[ONE] = None,
    load: bool = True,
) -> pd.DataFrame:
    """parses the text file of canonical sessions into a
    DataFrame for easier groupby selection of sessions"""

    filepath = path.with_suffix(".csv")
    if filepath.exists() and load is True:
        return pd.read_csv(filepath)

    one = ONE() if one is None else one
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
    df = df.drop(columns=["number", "lab", "projects", "url"]).set_index("id")
    df = df.sort_values("start_time").reset_index()
    df.to_csv(filepath, index=False)
    return df


def get_chronic_fov_mapping(
    subject: str,
    load: bool = True,
    one: Optional[ONE] = None,
):
    """establishes the mapping between scanimage (equivalent to suite2p) FOVs and
    roicat FOVs, on a session by session basis. Computes once and reloads if file
    is found on disk"""

    one = ONE() if one is None else one
    chronic_folder = BASE_FOLDER / subject / "Chronic"
    filepath = chronic_folder / "chronic_fov_map.csv"
    if filepath.exists() and load is True:
        return pd.read_csv(filepath)

    chronic_fov_map = []
    roicat_rundata_files = sorted(list(chronic_folder.glob("*tracking.rundata.pkl")))

    for roicat_rundata_file in tqdm(roicat_rundata_files):
        with open(roicat_rundata_file, "rb") as fH:
            roicat_rundata = pickle.load(fH)
        # the roicat fov is baked into the filepath of the rundata file
        roicat_fov = re.search(r"(FOV_\d{2})", str(roicat_rundata_file)).group(1)
        # insite that file are the suite2p stat filepaths that formed the basis
        # of this roi - from those we can extract the corresponding scanimage fov
        # because at least here scanimage fov == suite2p fov
        suite2p_files = roicat_rundata["data"]["paths_stat"]
        for i, file_ in enumerate(suite2p_files):
            file_ = file_.replace("\\", "/")  # WARNING this is sensitive to where roicat was
            pattern = r"Subjects/([^/]+)/(\d{4}-\d{2}-\d{2})/(\d{3})/alf/(FOV_\d{2})"
            subject_extracted, date, number, fov = re.search(pattern, str(file_)).groups()
            assert subject == subject_extracted
            eid = one.path2eid(f"{subject}/{date}/{number}")
            chronic_fov_map.append(
                dict(
                    session_index=i,
                    eid=str(eid),
                    date=date,
                    number=str(number),
                    subject=subject,
                    roicat_fov=roicat_fov,
                    scanimage_fov=fov,
                )
            )

    chronic_fov_map = pd.DataFrame(chronic_fov_map).reset_index()
    chronic_fov_map.to_csv(filepath, index=False)
    return chronic_fov_map


def load_imaging_FOV(
    eid: str,
    fov: str,  # scanimage fov
    dataset: str = "mpci.ROIActivityDeconvolved.npy",
    processing_fn: Optional[callable] = None,
    # sort: bool = False,
    one: Optional[ONE] = None,
) -> nap.TsdFrame:
    """loads data from a single field of view (scanimage / suite2p) into a pynapple
    TsdFrame. Metadata from suite2p and roicat is stored as a dataframe in
    nap.TsdFrame.metadata

    qc_query: a string that can be passed to metadata.query(), drop all non matching
    cells. Example: load_imaging_FOV(eid, 'FOV_00', qc_query="cluster_silhouette > .2")
    will subset the dataset to only those cells.

    WARNING using this for suite2p metrics as such can lead to inconsistent cells across
    days, use ... instead
    """

    one = ONE() if one is None else one
    session_path = BASE_FOLDER / one.eid2path(eid).session_path_short()
    subject = one.eid2ref(eid)["subject"]
    chronic_folder = BASE_FOLDER / subject / "Chronic"
    fov_collection = f"alf/{fov}"

    # establish mapping between scanimage and roicat FOVs
    FOV_map = get_chronic_fov_mapping(subject, load=True, one=one)
    row = FOV_map.groupby(["eid", "scanimage_fov"]).get_group((eid, fov))
    assert row.shape[0] == 1
    roicat_fov = row["roicat_fov"].values[0]
    session_index = row["session_index"].values[0]

    # get the corresponding roicat results file and read it
    roicat_results_file = next(chronic_folder.glob(f"*{roicat_fov}.ROICaT.tracking.results.pkl"))
    with open(roicat_results_file, "rb") as fH:
        roicat_results = pickle.load(fH)
    # the indices of roicat results -> individual FOV in session (suite2p output)
    ix = roicat_results["clusters"]["labels_bySession"][session_index]

    # get the roicat cluster ucids
    file = next(chronic_folder.glob(f"*{roicat_fov}.clusterUIDs_all.csv"))
    roicat_ucids_all = pd.read_csv(
        file,
        header=None,
        names=["roicat_UCID"],
    )
    # back to the session / fov oder
    roicat_ucids_all.loc[-1, "roicat_UCID"] = np.nan
    roicat_ucids_fov = roicat_ucids_all.loc[ix].reset_index(drop=True)

    # combine with our uuids
    ibl_uuids_fov = pd.read_csv(session_path / fov_collection / "mpciROIs.uuids.csv")
    roi_info = pd.concat([ibl_uuids_fov, roicat_ucids_fov], axis=1)

    # loading other suite2p info: cell classifier
    roi_info["iscell"] = np.load(session_path / fov_collection / "mpciROIs.cellClassifier.npy")

    # brain region estimate
    if (session_path / fov_collection / "mpciROIs.brainLocationIds_ccf_2017.npy").exists():
        roi_info["region_ids"] = np.load(session_path / fov_collection / "mpciROIs.brainLocationIds_ccf_2017.npy")
    else:
        roi_info["region_ids"] = np.load(session_path / fov_collection / "mpciROIs.brainLocationIds_ccf_2017_estimate.npy")

    atlas = AllenAtlas()
    roi_info["region_labels"] = atlas.regions.id2acronym(roi_info["region_ids"].values)
    roi_info["fov"] = fov

    # getting the roicat QC metrics
    file = next(chronic_folder.glob(f"*{roicat_fov}*tracking.results.pkl"))
    with open(file, "rb") as fH:
        roicat_output = pickle.load(fH)
    roicat_metrics = ["cluster_intra_means", "cluster_intra_maxs", "cluster_intra_mins", "cluster_silhouette"]
    roicat_qc_df = pd.DataFrame({metric: roicat_output["quality_metrics"][metric] for metric in roicat_metrics})
    roicat_qc_df.index = np.array(roicat_output["quality_metrics"]["cluster_labels_unique"], dtype="int32")

    # merge with cluster_uids / combine roi_info with roicat metrics
    roicat_qc_df["roicat_UCID"] = roicat_ucids_all
    roi_info = pd.merge(roi_info, roicat_qc_df, on="roicat_UCID", how="left")

    # fluorescence data : load all traces (suite2p) for one FOV
    suite2p_data_fov = np.load(session_path / fov_collection / dataset)
    times = np.load(session_path / fov_collection / "mpci.times.npy")
    assert suite2p_data_fov.shape[1] == roi_info.shape[0]

    # as a pynapple object, merged with the metadata
    fov_data = nap.TsdFrame(t=times, d=suite2p_data_fov, metadata=roi_info)

    # apply optional processing_fning
    if processing_fn is not None:
        fov_data = processing_fn(fov_data)

    return fov_data


def qc_imaging_data_by_query(
    imaging_data: nap.TsdFrame,
    qc_query: str,
) -> nap.TsdFrame:
    metadata_ = imaging_data.metadata.query(qc_query)
    ix = metadata_.index.values
    return nap.TsdFrame(
        t=imaging_data.t,
        d=imaging_data.d[:, ix],
        metadata=metadata_,
    )


def qc_imaging_data_by_ids(
    imaging_data: nap.TsdFrame,
    roicat_UCIDs: List[str],
) -> nap.TsdFrame:
    ix = imaging_data.metadata["roicat_UCID"].isin(roicat_UCIDs)
    return nap.TsdFrame(
        t=imaging_data.t,
        d=imaging_data.d[:, ix],
        metadata=imaging_data.metadata.loc[ix],
    )


def qc_chronic_data_by_query(
    chronic_data: List[nap.TsdFrame],
    qc_query: str,
) -> List[nap.TsdFrame]:
    roicat_UCIDs = []
    for data in chronic_data:
        ucids = data.metadata.query(qc_query)["roicat_UCID"]
        # drop nans and 'nan'
        ucids = ucids[~pd.isna(ucids)]
        ucids = ucids[~(ucids == "nan")]
        roicat_UCIDs.append(ucids)

    common_UCIDs = list(set.intersection(*[set(ucids) for ucids in roicat_UCIDs]))
    return qc_chronic_data_by_ids(chronic_data, common_UCIDs)


def qc_chronic_data_by_ids(
    chronic_data: List[nap.TsdFrame],
    roicat_UCIDs: List[str],
) -> List[nap.TsdFrame]:
    return [qc_imaging_data_by_ids(data, roicat_UCIDs) for data in chronic_data]


def load_imaging_session(
    eid: str,
    FOVs: Optional[list[str]] = None,  # if None, infer
    one: Optional[ONE] = None,
    **kwargs,
) -> List[nap.TsdFrame]:
    """ """
    one = ONE() if one is None else one
    FOVs = FOVs if FOVs is not None else get_session_FOVs(eid)
    session_data = {}
    for fov in tqdm(FOVs):
        _logger.info(f"loading FOV {fov} for {eid}")
        session_data[fov] = load_imaging_FOV(eid, fov, **kwargs)
    return session_data


def load_chronic_imaging(
    eids: list[str],
    FOVs: Optional[list[str]] = None,  # roicat FOVs
    one: Optional[ONE] = None,
    **kwargs,
) -> Dict[str, Dict[str, nap.TsdFrame]]:
    one = ONE() if one is None else one
    subject = one.eid2ref(eids[0])["subject"]
    # FOVs = FOVs if FOVs is not None else get_chronic_FOVs(subject, one=one)
    chronic_data = {}
    for eid in tqdm(eids, desc=f"loading chronic imaging for {subject}"):
        _logger.info(f"loading imaging data for {eid}")
        chronic_data[eid] = load_imaging_session(eid, FOVs, one=one, **kwargs)
    return chronic_data


def get_session_FOVs(
    eid: str,
    one: Optional[ONE] = None,
) -> np.ndarray:
    """helper to get all FOVs for a session"""
    one = ONE() if one is None else one
    session_path = BASE_FOLDER / one.eid2path(eid).session_path_short()
    # infer fov collections
    fov_folders = list(session_path.rglob("*alf/FOV_??*"))
    return sorted([folder.parts[-1] for folder in fov_folders])


def get_common_ROIs(
    eids: List[str],
    one: Optional[ONE] = None,
) -> List:
    """return scanimage ROI names for all rois that are common in all sessions"""
    one = ONE() if one is None else one
    session_FOVs = [get_session_FOVs(eid, one) for eid in eids]
    return sorted(list(set.intersection(*[set(s) for s in session_FOVs])))


def get_chronic_FOVs(
    subject: str,
    one: Optional[ONE] = None,
) -> List:
    """get the names of the roicat FOVs"""
    one = ONE() if one is None else one
    # subject = one.eid2ref(eids[0])
    chronic_folder = BASE_FOLDER / subject / "Chronic"
    files = list(chronic_folder.glob("*tracking.results.pkl"))
    pattern = r"FOV_(\d{2})"
    roicat_FOVs = sorted(["FOV_" + re.search(pattern, str(f)).group(1) for f in files])
    return roicat_FOVs


def interpolate_to_common_time_base(
    session_data: Dict[str, nap.TsdFrame],
    temporal_align: Optional[str] = None,
) -> nap.TsdFrame:
    """combines data from different FOVs in a single nap.TsdFrame - requires
    interpolation to a common time base (can be taken from a single FOV,
    or the average is calculated)"""

    # get the common time base
    if temporal_align is not None:
        assert temporal_align.startswith("FOV")
        t_align = session_data[temporal_align].t
    else:
        ts = [fov_data.t for fov, fov_data in session_data.items()]
        t_align = np.average(np.stack(ts), axis=0)

    # for each fov:
    session_data_interp = {}
    for fov, fov_data in session_data.items():
        # interpolate
        data_interp = interp1d(
            fov_data.t,
            fov_data.d,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )(t_align)
        # and store
        session_data_interp[fov] = nap.TsdFrame(
            t=t_align,
            d=data_interp,
            metadata=fov_data.metadata,
        )
    return session_data_interp


def stack_fov_data(
    session_data: Dict[str, nap.TsdFrame],
    order: Optional[List[str]] = None,
) -> nap.TsdFrame:
    order = order if order is not None else sorted(list(session_data.keys()))
    stacked_metadata = pd.concatenate([session_data[o].metadata for o in order], axis=0)
    stacked_metadata = stacked_metadata.reset_index(names="fov_index")
    return nap.TsdFrame(
        t=session_data[order[0]].t,
        d=np.concatenate([session_data[o].d for o in order], axis=1),
        metadata=stacked_metadata,
    )


def interpolate_and_stack(
    session_data: Dict[str, nap.TsdFrame],
    temporal_align: Optional[str] = None,
    order: Optional[List[str]] = None,
) -> nap.TsdFrame:
    """user facing convenience function"""
    data_interp = interpolate_to_common_time_base(
        session_data,
        temporal_align=temporal_align,
    )
    return stack_fov_data(data_interp, order=order)


# this doesn't belong here
# def get_common_roicat_UCIDs(eids: List[str], responses: dict):
#     common_roicat_UCIDs = np.array(list(set.intersection(*[set(responses[eid][1]["roicat_UCID"].values) for eid in eids])))
#     common_roicat_UCIDs = common_roicat_UCIDs[~pd.isna(common_roicat_UCIDs)]
#     common_roicat_UCIDs = common_roicat_UCIDs[~(common_roicat_UCIDs == "nan")]

#     # do the quality control here
#     refined_ids = []
#     for eid in eids:
#         subset = responses[eid][1].set_index("roicat_UCID").loc[common_roicat_UCIDs]
#         refined_ids.append(set(subset.query("iscell > 0.5 & cluster_silhouette > 0.2").index))

#     return np.array(list(set.intersection(*refined_ids)))


# %% the beginning of testing
# logging.basicConfig()

# selecting the subject
subject = "SP058"
subject = "SP072"

# session_type selection: biased or training
session_type = "biased"

sessions_df = parse_canonicals_sessions_file(Path(__file__).parent / "canonical_sessions_mod.txt")
sessions_df_ = sessions_df.loc[sessions_df["task_protocol"].str.contains(session_type)]
eids = sessions_df_.query("subject == @subject")["id"].values

# %%
data = load_chronic_imaging(eids, FOVs=["FOV_00", "FOV_01"])

# %%
session_data = load_imaging_session(eids[0], ["FOV_00", "FOV_01"])

# %%
session_data = interpolate_to_common_time_base(session_data)
order = sorted(list(session_data.keys()))
stacked_data = np.concatenate([session_data[o].d for o in order], axis=1)
stacked_metadata = pd.concat([session_data[o].metadata for o in order], axis=0)
stacked_metadata = stacked_metadata.reset_index(names="fov_index")

# %%
qc_chronic_data_by_query(data, "iscell > .5")
# %% loader validation - picking a FOV and roicat tracekd cell that is present in all
fov = "FOV_07"
common_roicat_UCIDs = np.array(list(set.intersection(*[set(data[eid][fov].metadata["roicat_UCID"].values) for eid in eids])))
common_roicat_UCIDs = common_roicat_UCIDs[~pd.isna(common_roicat_UCIDs)]
common_roicat_UCIDs = common_roicat_UCIDs[~(common_roicat_UCIDs == "nan")]

roicat_id = common_roicat_UCIDs[10]

# following that cell over the sessions
ixs = []
for eid in eids:
    ixs.append(data[eid][fov].metadata[data[eid][fov].metadata["roicat_UCID"] == roicat_id].index[0])

# sanity check, is the neuron in the same location in each image
# across sessions?

BASE_FOLDER = Path("/mnt/s0/Data/Subjects")
chronic_folder = BASE_FOLDER / subject / "Chronic"

one = ONE()

for i, eid in enumerate(eids):
    session_folder = BASE_FOLDER / one.eid2path(eid).session_path_short()
    fov_folder = session_folder / "alf" / fov
    stat_path = fov_folder / "_suite2p_ROIData.raw" / "stat.npy"
    stat = np.load(stat_path, allow_pickle=True)
    print(f"x:{stat[ixs[i]]['xpix'].mean():.2f}, y:{stat[ixs[i]]['ypix'].mean():.2f}")


# %% print the location of the ROI for each session
fov_locations = []
for eid in eids:
    session_folder = BASE_FOLDER / one.eid2path(eid).session_path_short()
    fovs = get_session_FOVs(eid)
    for fov in fovs:
        fov_folder = session_folder / "alf" / fov
        mlapdv = np.load(fov_folder / "mpciMeanImage.mlapdv_estimate.npy")[0, 0]
        fov_locations.append(dict(eid=eid, fov=fov, ml=mlapdv[0], ap=mlapdv[1], dv=mlapdv[2]))
fov_locations = pd.DataFrame(fov_locations)

# %% check if all the ROIs are in the same locations on all days
for fov in fov_locations["fov"].unique():
    assert np.unique(fov_locations.query(f'fov == "{fov}"')[["ml", "ap", "dv"]].values, axis=0).shape[0] == 1
