from __future__ import annotations
from one.api import ONE
from iblutil.util import Bunch
from iblatlas.atlas import AllenAtlas
import numpy as np
import matplotlib.pyplot as plt
from collections  import Counter, OrderedDict
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba, hsv_to_rgb, to_hex
import gc
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from scipy.stats import zscore
from matplotlib import gridspec
import brainbox.io.one as bb_one
from brainbox.io.one import SessionLoader
from brainbox.behavior.wheel import interpolate_position
from brainbox.behavior.wheel import velocity_filtered
from importlib import import_module
import time, os, re 
from scipy.signal import hilbert
from sklearn.cluster import KMeans
from rastermap import Rastermap
import sys
from iblatlas.regions import BrainRegions
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.colors import to_rgba
import matplotlib as mpl
import requests
from pathlib import PurePosixPath
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
from PIL import ImageDraw, ImageFont, PngImagePlugin
import datoviz as dv
from itertools import combinations


sys.path.insert(0, str(Path.home() / "Dropbox" / "scripts" / "IBL"))

from dmn_bwm import get_allen_info
from meso_chronic import get_cluster_uids_neuronal_by_fov, match_tracked_indices_pair_per_fov


plt.ion()
one = ONE()

pth_meso = Path(one.cache_dir, 'meso')
pth_meso.mkdir(parents=True, exist_ok=True)


atlas = AllenAtlas()
br = BrainRegions()

adf = atlas.regions.to_df()
region_colors_dict = {
    row['acronym']: f"{row['hexcolor']}"  
    for _, row in adf.iterrows()}

_,pal = get_allen_info()


tts__ = [
        'inter_trial', 
        'blockL', 
        'blockR', 
        'quiescence', 
        'block_change_s', 
        'stimLbLcL', 
        'stimLbRcL', 
        'stimRbRcR', 
        'stimRbLcR', 
        'mistake_s', 
        'motor_init', 
        'block_change_m', 
        'sLbLchoiceL', 
        'sLbRchoiceL', 
        'sRbRchoiceR', 
        'sRbLchoiceR', 
        'mistake_m', 
        'choiceL', 
        'choiceR', 
        'fback1', 
        'fback0']
     
peth_ila = [
    r"$\mathrm{rest}$",
    r"$\mathrm{L_b}$",
    r"$\mathrm{R_b}$",
    r"$\mathrm{quies}$",
    r"$\mathrm{change_b, s}$",
    r"$\mathrm{L_sL_cL_b, s}$",
    r"$\mathrm{L_sL_cR_b, s}$",
    r"$\mathrm{R_sR_cR_b, s}$",
    r"$\mathrm{R_sR_cL_b, s}$",
    r"$\mathrm{mistake, s}$",
    r"$\mathrm{m}$",
    r"$\mathrm{change_b, m}$",
    r"$\mathrm{L_sL_cL_b, m}$",
    r"$\mathrm{L_sL_cR_b, m}$",
    r"$\mathrm{R_sR_cR_b, m}$",
    r"$\mathrm{R_sR_cL_b, m}$",
    r"$\mathrm{mistake, m}$",
    r"$\mathrm{L_{move}}$",
    r"$\mathrm{R_{move}}$",
    r"$\mathrm{feedbk1}$",
    r"$\mathrm{feedbk0}$"
]

peth_dictm = dict(zip(tts__, peth_ila))


def deep_in_block(trials, pleft, depth=3):

    '''
    get mask for trials object of pleft trials that are 
    "depth" trials into the block
    '''
    
    # pleft trial indices 
    ar = np.arange(len(trials['stimOn_times']))[trials['probabilityLeft'] == pleft]
    
    # pleft trial indices shifted by depth earlier 
    ar_shift = ar - depth
    
    # trial indices where shifted ones are in block
    ar_ = ar[trials['probabilityLeft'][ar_shift] == pleft]

    # transform into mask for all trials
    bool_array = np.full(len(trials['stimOn_times']), False, dtype=bool)
    bool_array[ar_] = True
    
    return bool_array

def first_three_after_block_switch(trials):
    pl = trials['probabilityLeft']
    n = len(pl)

    # indices where block changes (start of new block)
    bs = np.where(np.diff(pl) != 0)[0] + 1

    mask = np.zeros(n, dtype=bool)

    # mark block_start, block_start+1, block_start+2
    for b in bs:
        mask[b : min(b+3, n)] = True

    return mask


def lz76_complexity(s: str) -> int:
    """
    Fast LZ76 parser for a binary string.
    Returns the number of parsed phrases (c).
    """
    n = len(s)
    i = 0
    c = 0
    dictionary = set()

    while i < n:
        k = 1
        # extend substring as long as it exists
        while i + k <= n and s[i:i+k] in dictionary:
            k += 1
        dictionary.add(s[i:i+k])
        c += 1
        i += k

    return c

    
def lzs_pci(x: np.ndarray, rng: np.random.Generator) -> float:
    """
    PCI-style Lempel-Ziv complexity as used in Casali et al. (2013).
    - detrend, z-score
    - Hilbert envelope
    - threshold at its mean
    - binarize to string
    - compute LZ(s) / LZ(shuffled(s))

    Parameters
    ----------
    x : 1D array
    rng : np.random.Generator for reproducible shuffle

    Returns
    -------
    float
        LZ complexity normalized by shuffled surrogate.
    """
    x = np.asarray(x, float)

    # Hilbert envelope
    env = np.abs(hilbert(x))
    th = env.mean()

    # binary string
    s = (env > th).astype(np.uint8)
    s_str = ''.join('1' if b else '0' for b in s)

    # shuffle surrogate (same 0/1 counts)
    M = np.array(list(s_str))
    rng.shuffle(M)
    w_str = ''.join(M.tolist())

    # LZ complexity
    c_s = lz76_complexity(s_str)
    c_w = lz76_complexity(w_str)

    if c_w == 0:
        return 0.0
    return c_s / c_w


def save_trial_cuts_meso(
    eid: str,
    filter_neurons: bool = True,
    require_all: bool = True,
    out_dir: str | Path | None = None,
    *,
    restrict: None | np.ndarray | dict[str, np.ndarray] = None,
    restrict_uids: None | dict[str, np.ndarray] = None,
    pair_tag: str | None = None,
):
    """
    Cut mesoscope ROI traces into per-trial windows.

    Change vs previous version:
      - If a window spans >1 time bin (n_frames > 1), we save the *average over frames*
        so each trial cut is shape (n_trials, N, 1). If n_frames == 1, keep that 1 bin.

    Window semantics:
      window_start = t0 - pre
      window_end   = t0 + post   (post can be negative)
    Inclusive endpoints in frame-index space (after rounding).
    """
    if out_dir is None:
        out_dir = Path(pth_meso, "trial_cuts", str(eid))
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trials for eid:", eid)
    trials = one.load_object(eid, "trials")
    mask = np.full(len(trials["stimOn_times"]), True, dtype=bool)  # placeholder

    # --- build tts ---
    idcs = [0] + list(
        np.where((trials["stimOn_times"][1:] - trials["intervals"][:, -1][:-1]) > 1.15)[0] + 1
    )
    mask_iti = [True if i in idcs else False for i in range(len(trials["stimOn_times"]))]

    tts = {
        "inter_trial": ["stimOn_times", np.bitwise_and.reduce([mask, mask_iti]), [1.15, -1]],
        "blockL": ["stimOn_times", np.bitwise_and.reduce([mask, trials["probabilityLeft"] == 0.8]), [0.4, -0.1]],
        "blockR": ["stimOn_times", np.bitwise_and.reduce([mask, trials["probabilityLeft"] == 0.2]), [0.4, -0.1]],
        "quiescence": ["stimOn_times", mask, [0.4, -0.1]],
        "block_change_s": ["stimOn_times", np.bitwise_and(mask, first_three_after_block_switch(trials)), [0, 0.15]],

        "stimLbLcL": ["stimOn_times",
            np.bitwise_and.reduce([mask,
                ~np.isnan(trials["contrastLeft"]),
                trials["probabilityLeft"] == 0.8,
                deep_in_block(trials, 0.8),
                trials["choice"] == 1
            ]),
            [0, 0.15]
        ],
        "stimLbRcL": ["stimOn_times",
            np.bitwise_and.reduce([mask,
                ~np.isnan(trials["contrastLeft"]),
                trials["probabilityLeft"] == 0.2,
                deep_in_block(trials, 0.2),
                trials["choice"] == 1
            ]),
            [0, 0.15]
        ],
        "stimRbRcR": ["stimOn_times",
            np.bitwise_and.reduce([mask,
                ~np.isnan(trials["contrastRight"]),
                trials["probabilityLeft"] == 0.2,
                deep_in_block(trials, 0.2),
                trials["choice"] == -1
            ]),
            [0, 0.15]
        ],
        "stimRbLcR": ["stimOn_times",
            np.bitwise_and.reduce([mask,
                ~np.isnan(trials["contrastRight"]),
                trials["probabilityLeft"] == 0.8,
                deep_in_block(trials, 0.8),
                trials["choice"] == -1
            ]),
            [0, 0.15]
        ],

        "mistake_s": ["stimOn_times", np.bitwise_and.reduce([mask, trials["feedbackType"] == -1]), [0, 0.15]],

        "motor_init": ["firstMovement_times", mask, [0.15, 0]],
        "block_change_m": ["firstMovement_times", np.bitwise_and(mask, first_three_after_block_switch(trials)), [0.15, 0]],

        "sLbLchoiceL": ["firstMovement_times",
            np.bitwise_and.reduce([mask,
                ~np.isnan(trials["contrastLeft"]),
                trials["probabilityLeft"] == 0.8,
                trials["choice"] == 1
            ]),
            [0.15, 0]
        ],
        "sLbRchoiceL": ["firstMovement_times",
            np.bitwise_and.reduce([mask,
                ~np.isnan(trials["contrastLeft"]),
                trials["probabilityLeft"] == 0.2,
                trials["choice"] == 1
            ]),
            [0.15, 0]
        ],
        "sRbRchoiceR": ["firstMovement_times",
            np.bitwise_and.reduce([mask,
                ~np.isnan(trials["contrastRight"]),
                trials["probabilityLeft"] == 0.2,
                trials["choice"] == -1
            ]),
            [0.15, 0]
        ],
        "sRbLchoiceR": ["firstMovement_times",
            np.bitwise_and.reduce([mask,
                ~np.isnan(trials["contrastRight"]),
                trials["probabilityLeft"] == 0.8,
                trials["choice"] == -1
            ]),
            [0.15, 0]
        ],

        "mistake_m": ["firstMovement_times", np.bitwise_and.reduce([mask, trials["feedbackType"] == -1]), [0.15, 0]],

        "choiceL": ["firstMovement_times", np.bitwise_and.reduce([mask, trials["choice"] == 1]), [0, 0.15]],
        "choiceR": ["firstMovement_times", np.bitwise_and.reduce([mask, trials["choice"] == -1]), [0, 0.15]],

        "fback1": ["feedback_times", np.bitwise_and.reduce([mask, trials["feedbackType"] == 1]), [0, 0.3]],
        "fback0": ["feedback_times", np.bitwise_and.reduce([mask, trials["feedbackType"] == -1]), [0, 0.3]],
    }

    trial_names = list(tts.keys())

    # ----------------- helpers -----------------
    def _cut_one_peth_to_memmap(
        roi_signal: np.ndarray,          # (N, T)
        frame_times: np.ndarray,         # (T,)
        offsets_s: np.ndarray,           # (N,)
        event_times: np.ndarray,         # (n_trials,)
        pre: float,
        post: float,
        out_path: Path,
    ) -> tuple[int, int, dict]:
        """
        Cut an inclusive continuous-time window [t0-pre, t0+post] and store one value per trial/ROI:
        - if n_frames == 1: store that single frame
        - if n_frames  > 1: store nanmean over frames (per trial, per ROI)

        Indexing:
        - event frame index uses nearest frame in frame_times (via searchsorted + neighbor check)
        - window edges use floor/ceil in units of dt to avoid missing the intended interval

        Returns (n_trials, 1, info_dict).
        """
        pre = float(pre)
        post = float(post)
        if not np.isfinite(pre) or not np.isfinite(post):
            raise ValueError("Non-finite pre/post.")

        frame_times = np.asarray(frame_times, dtype=float)
        if frame_times.ndim != 1 or frame_times.size < 2:
            raise ValueError("frame_times must be 1D with >= 2 elements.")

        dts = np.diff(frame_times)
        dt = float(np.median(dts))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Bad frame_times dt.")

        # Window in event-relative time
        start_rel = -pre
        end_rel = post
        if end_rel <= start_rel:
            raise ValueError(f"Invalid window: [{start_rel}, {end_rel}] relative to event.")

        # Use floor/ceil so we cover the intended continuous interval
        idx0_rel = int(np.floor(start_rel / dt))
        idx1_rel = int(np.ceil(end_rel / dt))
        n_frames = idx1_rel - idx0_rel + 1
        if n_frames <= 0:
            raise ValueError("n_frames <= 0 after floor/ceil; check pre/post/dt.")

        N, T = roi_signal.shape
        n_trials = int(event_times.size)

        # Convert ROI offsets to frames; still an approximation.
        # (If you want exact handling, you must index with (frame_times + offset_n) per ROI.)
        offset_frames = np.rint(offsets_s / dt).astype(np.int32)  # (N,)

        # Precompute relative frame offsets
        frame_offsets = (np.arange(n_frames, dtype=np.int32) + idx0_rel)[None, :]  # (1, n_frames)

        mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_trials, N, 1))

        ft0 = frame_times[0]
        ftN = frame_times[-1]

        def nearest_frame_index(t: float) -> int:
            """Nearest index in frame_times to time t."""
            j = int(np.searchsorted(frame_times, t, side="left"))
            if j <= 0:
                return 0
            if j >= frame_times.size:
                return frame_times.size - 1
            # pick closer of j-1 and j
            return j if (frame_times[j] - t) < (t - frame_times[j - 1]) else (j - 1)

        for i, t0 in enumerate(np.asarray(event_times, dtype=float)):
            event_idx = nearest_frame_index(t0)

            # indices per ROI, per frame in window
            idx = (event_idx + frame_offsets) - offset_frames[:, None]  # (N, n_frames)

            valid = (idx >= 0) & (idx < T)
            idx_clip = np.clip(idx, 0, T - 1)

            vals = np.take_along_axis(roi_signal, idx_clip, axis=1).astype(np.float32, copy=False)
            vals[~valid] = np.nan

            if n_frames == 1:
                mm[i, :, 0] = vals[:, 0]
            else:
                mm[i, :, 0] = np.nanmean(vals, axis=1)

            if (i + 1) % 100 == 0 or (i + 1) == n_trials:
                print(f"    wrote {i+1}/{n_trials} trials to {out_path.name}")

        del mm

        # Report realized relative bounds in seconds (given dt-based indexing)
        realized_start = idx0_rel * dt
        realized_end = idx1_rel * dt

        info = dict(
            dt=dt,
            idx0_rel=idx0_rel,
            idx1_rel=idx1_rel,
            n_frames=n_frames,
            intended=(start_rel, end_rel),
            realized=(realized_start, realized_end),
            # Worst-case boundary overshoot relative to intended interval:
            overshoot_start=(start_rel - realized_start),  # >= 0 typically
            overshoot_end=(realized_end - end_rel),        # >= 0 typically
        )
        return n_trials, 1, info


    saved_meta_paths: list[Path] = []

    # ----------------- per-FOV loop -----------------
    objects = ["mpci", "mpciROIs", "mpciROITypes", "mpciStack"]
    fov_folders = one.list_collections(eid, collection="alf/FOV_*")
    fov_folders = sorted(fov_folders, key=lambda s: int(s[-2:]))

    for fov_collection in fov_folders:
        fov = fov_collection.split("/")[-1]
        print("\n=== FOV:", fov, "===")

        ROI = {}
        for obj in objects:
            try:
                ROI[obj] = one.load_object(eid, obj, collection=fov_collection)
            except Exception:
                ROI[obj] = {}

        key = (
            "brainLocationsIds_ccf_2017"
            if "brainLocationsIds_ccf_2017" in ROI["mpciROIs"]
            else "brainLocationIds_ccf_2017_estimate"
        )

        region_ids = ROI["mpciROIs"][key]
        region_labels_all = atlas.regions.id2acronym(region_ids)
        region_colors_all = np.array(
            [region_colors_dict.get(acr, "#808080") for acr in region_labels_all], dtype=object
        )

        frame_times = np.asarray(ROI["mpci"]["times"], dtype=float)  # (T,)
        if frame_times.size < 2:
            raise ValueError(f"{eid} {fov}: frame_times too short.")

        roi_xyz = np.asarray(ROI["mpciROIs"]["stackPos"])
        timeshift = np.asarray(ROI["mpciStack"]["timeshift"], dtype=float)
        roi_offsets = timeshift[roi_xyz[:, len(timeshift.shape)]].astype(float)  # (N,)

        if "ROIActivityDeconvolved" in ROI["mpci"]:
            roi_signal_all = ROI["mpci"]["ROIActivityDeconvolved"].T
        else:
            roi_signal_all = ROI["mpci"]["ROIActivityF"].T
        roi_signal_all = roi_signal_all.astype(np.float32, copy=False)  # (N, T)

        neuron_mask = ROI["mpciROIs"]["mpciROITypes"].astype(bool)
        print(f"  neuron ROIs: {int(neuron_mask.sum())}/{len(neuron_mask)}")

        if filter_neurons:
            roi_signal = roi_signal_all[neuron_mask]
            roi_offsets_use = roi_offsets[neuron_mask]
            region_labels = region_labels_all[neuron_mask]
            region_colors = region_colors_all[neuron_mask]
            xyz = np.asarray(ROI["mpciROIs"]["mlapdv_estimate"], dtype=float)[neuron_mask]
        else:
            roi_signal = roi_signal_all
            roi_offsets_use = roi_offsets
            region_labels = region_labels_all
            region_colors = region_colors_all
            xyz = np.asarray(ROI["mpciROIs"]["mlapdv_estimate"], dtype=float)

        # --- optional restriction (indices into current roi_signal for this FOV) ---
        restrict_used = None
        restrict_uids_used = None
        if restrict is not None:
            if isinstance(restrict, dict):
                restrict_used = np.asarray(restrict.get(fov, np.array([], dtype=int)), dtype=int)
            else:
                restrict_used = np.asarray(restrict, dtype=int)

            if restrict_uids is not None and isinstance(restrict_uids, dict):
                restrict_uids_used = np.asarray(restrict_uids.get(fov, np.array([], dtype=object)), dtype=object)

            if restrict_used.size == 0:
                print(f"  restrict: 0 ROIs in {fov} -> skipping FOV")
                del ROI, roi_signal_all, roi_signal, roi_offsets, roi_offsets_use
                del region_ids, region_labels_all, region_labels, region_colors_all, region_colors
                del xyz, frame_times, timeshift, roi_xyz, neuron_mask
                gc.collect()
                continue

            mn = int(restrict_used.min())
            mx = int(restrict_used.max())
            N0 = int(roi_signal.shape[0])
            if mn < 0 or mx >= N0:
                raise IndexError(
                    f"{eid} {fov}: restrict indices out of bounds (min={mn}, max={mx}, N={N0})."
                )

            roi_signal = roi_signal[restrict_used]
            roi_offsets_use = roi_offsets_use[restrict_used]
            region_labels = np.asarray(region_labels)[restrict_used]
            region_colors = np.asarray(region_colors)[restrict_used]
            xyz = np.asarray(xyz)[restrict_used]
            print(f"  restrict: keeping {restrict_used.size}/{N0} ROIs")

        dt = float(np.median(np.diff(frame_times)))
        print("  roi_signal:", roi_signal.shape, "frame_times:", frame_times.shape, "dt:", dt)

        peth_files: dict[str, str] = {}
        tls: dict[str, int] = {}
        peth_shapes: dict[str, tuple[int, int, int]] = {}

        for keyname in trial_names:
            align_col, trial_mask, (pre, post) = tts[keyname]

            events_all = trials[align_col][np.bitwise_and.reduce([mask, trial_mask])]
            events_all = np.asarray(events_all, dtype=float)
            events_all = events_all[np.isfinite(events_all)]
            tls[keyname] = int(events_all.size)

            out_path = out_dir / f"{fov}_{keyname}_filter_{filter_neurons}.npy"

            if tls[keyname] == 0:
                if require_all:
                    raise ValueError(f"Missing PETH '{keyname}' for eid={eid}, {fov} (0 trials).")

                # placeholder (always (1,N,1) now)
                mm = np.lib.format.open_memmap(
                    out_path, mode="w+", dtype=np.float32, shape=(1, roi_signal.shape[0], 1)
                )
                mm[:] = np.nan
                del mm

                peth_files[keyname] = out_path.name
                peth_shapes[keyname] = (1, int(roi_signal.shape[0]), 1)
                print(f"  {keyname}: 0 trials -> placeholder {out_path.name} shape={peth_shapes[keyname]}")
                continue

            print(f"  {keyname}: {tls[keyname]} trials -> writing {out_path.name}")
            n_trials, n_frames, _ = _cut_one_peth_to_memmap(
                roi_signal=roi_signal,
                frame_times=frame_times,
                offsets_s=roi_offsets_use,
                event_times=events_all,
                pre=pre,
                post=post,
                out_path=out_path,
            )
            peth_files[keyname] = out_path.name
            peth_shapes[keyname] = (int(n_trials), int(roi_signal.shape[0]), int(n_frames))  # n_frames==1

        meta = {
            "eid": str(eid),
            "fov": str(fov),
            "filter_neurons": bool(filter_neurons),
            "pair_tag": pair_tag,
            "restrict_used": restrict_used,
            "restrict_uids_used": restrict_uids_used,
            "trial_names": trial_names,
            "tls": tls,
            "peth_files": peth_files,
            "peth_shapes": peth_shapes,
            "region_labels": region_labels,
            "region_colors": region_colors,
            "xyz": xyz.astype(np.float32, copy=False),
            "roi_offsets_s": roi_offsets_use.astype(np.float32, copy=False),
            "frame_times": frame_times.astype(np.float32, copy=False),
            "dt": float(dt),
            "window_semantics": "start=t0-pre, end=t0+post, inclusive endpoints after rounding to frames; stored as mean over frames -> 1 bin",
        }

        meta_path = out_dir / f"{eid}_{fov}_meta_filter_{filter_neurons}.npy"
        np.save(meta_path, meta, allow_pickle=True)
        saved_meta_paths.append(meta_path)
        print("Saved meta:", meta_path.name)

        # free memory aggressively before next FOV
        del ROI, roi_signal_all, roi_signal, roi_offsets, roi_offsets_use
        del region_ids, region_labels_all, region_labels, region_colors_all, region_colors
        del xyz, frame_times, timeshift, roi_xyz, neuron_mask
        gc.collect()

    return saved_meta_paths


#################
'''
stack per session pairs, using chronic tracking of neurons
'''
#################


# =============================================================================
# 4) End-to-end: per subject -> all EID pairs -> restrict-by-chronic -> save cuts
# =============================================================================

def _eid_date(one, eid: str) -> str:
    try:
        meta = one.alyx.rest("sessions", "read", id=eid)
        return str(meta["start_time"])[:10]
    except Exception:
        return "9999-99-99"


def _canonical_sessions_subject_map(
    one: ONE | None = None,
    *,
    rerun: bool = False,
) -> Dict[str, List[str]]:
    """
    Return {subject: [eid, ...]} for canonical mesoscope sessions.

    Uses a local cache under ONE cache_dir/meso to avoid repeated HTTP + Alyx calls.
    """
    if one is None:
        one = ONE()

    pth_meso = Path(one.cache_dir, "meso")
    pth_meso.mkdir(parents=True, exist_ok=True)
    cache_path = pth_meso / "canonical_sessions_subject_map.npy"

    if cache_path.is_file() and not rerun:
        return np.load(cache_path, allow_pickle=True).item()

    # IMPORTANT: use RAW github content, not the HTML blob page
    url = "https://raw.githubusercontent.com/int-brain-lab/mesoscope/main/canonical_sessions.txt"
    txt = requests.get(url, timeout=30).text

    out: Dict[str, List[str]] = {}
    seen: set[str] = set()

    for line in txt.splitlines():
        line = line.strip()
        if (not line) or line.startswith("#"):
            continue

        # Windows -> POSIX
        line = line.replace("\\", "/")

        eid = one.path2eid(PurePosixPath(line))
        if eid is None:
            continue
        eid = str(eid)

        if eid in seen:
            continue
        seen.add(eid)

        ses = one.alyx.rest("sessions", "read", id=eid)
        subject = str(ses["subject"])

        out.setdefault(subject, []).append(eid)

    np.save(cache_path, out, allow_pickle=True)
    return out


def compute_and_save_trial_cuts_for_all_subject_pairs(
    one,
    *,
    roicat_root: str | Path = Path.home() / "chronic_csv",
    server_root: str | Path | None = None,
    trial_cuts_root: str | Path = Path.home() / "meso_trial_cuts_pairs",
    filter_neurons: bool = True,
    require_all: bool = True,
    pair_mode: str = "all",  # "all" | "consecutive"
    skip_if_no_shared: bool = True,
) -> Dict[str, Dict[str, object]]:
    """
    For each subject from canonical_sessions:
      - build all session pairs (or consecutive pairs)
      - compute chronically tracked neurons per pair PER FOV
      - run save_trial_cuts_meso for each eid, restricted per FOV to the shared neurons

    Saves unambiguously under:
      trial_cuts_root/<subject>/<eidA>__<eidB>/<eidX>/...

    Returns a compact index dict describing what was saved.
    """
    trial_cuts_root = Path(trial_cuts_root)
    trial_cuts_root.mkdir(parents=True, exist_ok=True)

    subj2eids = _canonical_sessions_subject_map(one)

    saved_index: Dict[str, Dict[str, object]] = {}

    for subject, eids in sorted(subj2eids.items()):
        eids = [e for e in eids if e and str(e).lower() != "none"]
        if len(eids) < 2:
            continue

        # chronological
        eids_sorted = sorted(eids, key=lambda e: (_eid_date(one, e), e))

        if pair_mode == "consecutive":
            pairs = [(eids_sorted[i], eids_sorted[i + 1]) for i in range(len(eids_sorted) - 1)]
        elif pair_mode == "all":
            pairs = list(combinations(eids_sorted, 2))
        else:
            raise ValueError("pair_mode must be 'all' or 'consecutive'")

        subj_out: Dict[str, object] = {"pairs": {}}

        for eid_a, eid_b in pairs:
            pair_tag = f"{eid_a}__{eid_b}"
            pair_dir = trial_cuts_root / subject / pair_tag
            (pair_dir / eid_a).mkdir(parents=True, exist_ok=True)
            (pair_dir / eid_b).mkdir(parents=True, exist_ok=True)

            fov_match = match_tracked_indices_pair_per_fov(
                one, eid_a, eid_b,
                roicat_root=roicat_root, server_root=server_root,
                filter_neurons=filter_neurons,
            )

            # Build per-eid restrict dicts (FOV-local), plus UID traceability
            restrict_a: Dict[str, np.ndarray] = {}
            restrict_b: Dict[str, np.ndarray] = {}
            uids_by_fov: Dict[str, np.ndarray] = {}

            total_shared = 0
            for fov, m in fov_match.items():
                restrict_a[fov] = m.idx_a
                restrict_b[fov] = m.idx_b
                uids_by_fov[fov] = m.shared_uids
                total_shared += int(m.shared_uids.size)

            if skip_if_no_shared and total_shared == 0:
                continue

            # Save restricted cuts for each eid
            meta_a = save_trial_cuts_meso(
                eid_a,
                filter_neurons=filter_neurons,
                require_all=require_all,
                out_dir=(pair_dir / eid_a),
                restrict=restrict_a,
                restrict_uids=uids_by_fov,
                pair_tag=pair_tag,
            )
            meta_b = save_trial_cuts_meso(
                eid_b,
                filter_neurons=filter_neurons,
                require_all=require_all,
                out_dir=(pair_dir / eid_b),
                restrict=restrict_b,
                restrict_uids=uids_by_fov,
                pair_tag=pair_tag,
            )

            subj_out["pairs"][pair_tag] = {
                "eid_a": eid_a,
                "eid_b": eid_b,
                "pair_dir": str(pair_dir),
                "total_shared_uids_sum_over_fov": int(total_shared),
                "meta_paths_a": [str(p) for p in meta_a],
                "meta_paths_b": [str(p) for p in meta_b],
            }

        saved_index[subject] = subj_out

    return saved_index


# =============================================================================
# 5) For one subject + one eid pair:
#    trial-average, stack neurons (same order across eids), and correlate feature vectors
# =============================================================================

def _load_meta_dict(p: Path) -> dict:
    obj = np.load(p, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.flat[0]
    return obj


def _avg_trials(X_mnt: np.ndarray, *, min_trials: int) -> Optional[np.ndarray]:
    """Return (N,T) average over all trials, or None if insufficient."""
    if X_mnt.ndim != 3:
        raise ValueError(f"Expected (M,N,T), got {X_mnt.shape}")
    if X_mnt.shape[0] < min_trials:
        return None
    return np.nanmean(X_mnt, axis=0).astype(np.float32, copy=False)


def _zscore_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def stack_pair_and_correlate(
    *,
    pair_dir: str | Path,
    eid_a: str,
    eid_b: str,
    filter_neurons: bool = True,
    min_trials: int = 10,
    cut_to_min: bool = True,
    zscore_before_corr: bool = True,
    out_name: str = "pair_stack_and_corr.npy",
) -> dict:
    """
    Reads the *restricted* trial-cuts saved under:
      pair_dir/<eid>/...

    For each eid:
      - per FOV: load each ttype memmap, trial-average -> (N,T)
      - (optional) truncate each ttype to min length shared across the *two eids* for that FOV+ttype
      - concat ttypes -> (N,L)
    Then:
      - stack FOV blocks in deterministic order (FOV_00, FOV_01, ...)
      - ensure the two stacks have identical shape and aligned neuron order
      - compute per-neuron Pearson r between the two feature vectors

    Saves a dict to pair_dir/out_name and returns it.
    """
    pair_dir = Path(pair_dir)
    dir_a = pair_dir / eid_a
    dir_b = pair_dir / eid_b
    if (not dir_a.is_dir()) or (not dir_b.is_dir()):
        raise FileNotFoundError(f"Missing eid subdirs in {pair_dir}: {dir_a} / {dir_b}")

    metas_a = sorted(dir_a.glob(f"{eid_a}_FOV_*_meta_filter_*.npy"))
    metas_b = sorted(dir_b.glob(f"{eid_b}_FOV_*_meta_filter_*.npy"))
    if not metas_a or not metas_b:
        raise RuntimeError("No meta files found for one or both eids in pair_dir.")

    # index metas by fov
    ma = { _load_meta_dict(p)["fov"]: _load_meta_dict(p) for p in metas_a }
    mb = { _load_meta_dict(p)["fov"]: _load_meta_dict(p) for p in metas_b }

    fovs = sorted(set(ma.keys()) & set(mb.keys()))
    if not fovs:
        raise RuntimeError("No overlapping FOVs between the two eids in this pair_dir.")

    # trial types must match; take from first overlapping FOV
    ref_ttypes = list(ma[fovs[0]].get("trial_names", []))
    if not ref_ttypes:
        raise RuntimeError("Missing trial_names in meta.")
    if list(mb[fovs[0]].get("trial_names", [])) != ref_ttypes:
        raise RuntimeError("trial_names mismatch between eids.")

    blocks_a: list[np.ndarray] = []
    blocks_b: list[np.ndarray] = []
    uids_blocks: list[np.ndarray] = []
    fov_blocks: list[np.ndarray] = []

    for fov in fovs:
        meta_a = ma[fov]
        meta_b = mb[fov]

        if bool(meta_a.get("filter_neurons")) != bool(filter_neurons):
            continue
        if bool(meta_b.get("filter_neurons")) != bool(filter_neurons):
            continue

        # For traceability / alignment checks (optional, but recommended)
        uids_a = np.asarray(meta_a.get("restrict_uids_used", []), dtype=object)
        uids_b = np.asarray(meta_b.get("restrict_uids_used", []), dtype=object)
        if uids_a.size and uids_b.size and (uids_a.shape != uids_b.shape or np.any(uids_a != uids_b)):
            raise ValueError(f"{fov}: restrict_uids_used mismatch between eids; ordering not aligned.")

        segs_a: list[np.ndarray] = []
        segs_b: list[np.ndarray] = []

        # Determine truncation lengths per ttype (shared across both eids) if cut_to_min
        if cut_to_min:
            min_len: Dict[str, int] = {}
            for t in ref_ttypes:
                Ta = int(meta_a["peth_shapes"][t][2])
                Tb = int(meta_b["peth_shapes"][t][2])
                min_len[t] = min(Ta, Tb)

        for t in ref_ttypes:
            Xa = np.load(dir_a / meta_a["peth_files"][t], mmap_mode="r")
            Xb = np.load(dir_b / meta_b["peth_files"][t], mmap_mode="r")

            Aa = _avg_trials(Xa, min_trials=min_trials)
            Ab = _avg_trials(Xb, min_trials=min_trials)
            if Aa is None or Ab is None:
                # skip this FOV entirely if insufficient trials in any segment
                segs_a = []
                segs_b = []
                break

            if cut_to_min:
                Tt = int(min_len[t])
                Aa = Aa[:, :Tt]
                Ab = Ab[:, :Tt]
            else:
                # require exact match across eids
                if Aa.shape[1] != Ab.shape[1]:
                    segs_a = []
                    segs_b = []
                    break

            if Aa.shape[0] != Ab.shape[0]:
                raise ValueError(f"{fov}/{t}: N mismatch after restriction: {Aa.shape[0]} != {Ab.shape[0]}")

            segs_a.append(Aa)
            segs_b.append(Ab)

            del Xa, Xb

        if not segs_a:
            continue

        Pa = np.concatenate(segs_a, axis=1).astype(np.float32, copy=False)  # (N,L)
        Pb = np.concatenate(segs_b, axis=1).astype(np.float32, copy=False)

        blocks_a.append(Pa)
        blocks_b.append(Pb)

        if uids_a.size:
            uids_blocks.append(uids_a)
        else:
            # fall back to placeholders if uids are absent
            uids_blocks.append(np.array([f"{fov}:roi{i:06d}" for i in range(Pa.shape[0])], dtype=object))

        fov_blocks.append(np.array([fov] * Pa.shape[0], dtype=object))

        del segs_a, segs_b, Pa, Pb
        gc.collect()

    if not blocks_a:
        raise RuntimeError("No FOV blocks survived min_trials / shape checks.")

    A = np.concatenate(blocks_a, axis=0)
    B = np.concatenate(blocks_b, axis=0)
    if A.shape != B.shape:
        raise RuntimeError(f"Pair stacks shape mismatch: {A.shape} vs {B.shape}")

    if zscore_before_corr:
        Ause = _zscore_rows(A)
        Buse = _zscore_rows(B)
    else:
        Ause = A
        Buse = B

    # per-row cosine similarity mapped to [0,1]
    mask = np.isfinite(Ause) & np.isfinite(Buse)
    A0 = np.where(mask, Ause, 0.0).astype(np.float32, copy=False)
    B0 = np.where(mask, Buse, 0.0).astype(np.float32, copy=False)

    num = np.sum(A0 * B0, axis=1)
    den = np.sqrt(np.sum(A0 * A0, axis=1) * np.sum(B0 * B0, axis=1))
    den = np.where(den == 0, np.nan, den)

    cos = num / den
    r = (0.5 * (cos + 1.0)).astype(np.float32)  # in [0,1]

    uids_all = np.concatenate(uids_blocks, axis=0)
    fov_all = np.concatenate(fov_blocks, axis=0)

    out = {
        "pair_dir": str(pair_dir),
        "eid_a": eid_a,
        "eid_b": eid_b,
        "filter_neurons": bool(filter_neurons),
        "min_trials": int(min_trials),
        "cut_to_min": bool(cut_to_min),
        "zscore_before_corr": bool(zscore_before_corr),
        "ttypes": ref_ttypes,
        "A": A.astype(np.float32, copy=False),
        "B": B.astype(np.float32, copy=False),
        "uids": uids_all,
        "FOV": fov_all,
        "corr_per_neuron": r,
    }

    out_path = pair_dir / out_name
    np.save(out_path, out, allow_pickle=True)
    return out


def chronic_pair_feature_corr_subject(
    subject: str,
    *,
    one: "ONE | None" = None,
    trial_cuts_root: "str | Path | None" = None,
    out_dir: "str | Path | None" = None,
    filter_neurons: bool = True,
    min_trials: int = 10,
    pair_mode: str = "all",          # "all" | "consecutive"
    cut_to_min: bool = True,         # kept for cache key compatibility (not used internally here)
    zscore_before_corr: bool = True, # kept for API compatibility; MUST be True (see below)
    rerun: bool = False,
    roicat_root: "str | Path" = Path.home() / "chronic_csv",
    server_root: "str | Path | None" = None,
    per_reg: bool = True,            # kept for API compatibility (not used internally here)
    include_cv_baseline: bool = True,
    cv_shuf_seed: int = 0,           # kept for API compatibility (not used internally here)
) -> "pd.DataFrame":
    """
    IMPORTANT CHANGE:
      - The per-ttype feature columns (*_a, *_b) are now ALWAYS the z-scored feature vectors
        (row-wise z-score across trial-types for each neuron), for BOTH between-session and CV rows.
      - Therefore, zscore_before_corr MUST be True; otherwise the stored features would be inconsistent
        with the user's requirement ("store z-scored only").
    """
    import gc
    import time
    from itertools import combinations

    import numpy as np
    import pandas as pd

    if one is None:
        one = ONE()

    # Enforce invariant requested by user
    if not bool(zscore_before_corr):
        raise ValueError("zscore_before_corr must be True: this function now stores ONLY z-scored features (*_a/*_b).")

    # ---------- roots ----------
    if trial_cuts_root is None:
        trial_cuts_root = Path(pth_meso, "trial_cuts_pairs")
    else:
        trial_cuts_root = Path(trial_cuts_root)
    trial_cuts_root.mkdir(parents=True, exist_ok=True)

    if out_dir is None:
        out_dir = Path(pth_meso, "chronic_pair_corr")
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- cache paths ----------
    cache_stem = (
        f"chronic_corr_subject={subject}"
        f"_filter={int(filter_neurons)}"
        f"_mintrials={int(min_trials)}"
        f"_pairs={pair_mode}"
        f"_cutmin={int(cut_to_min)}"
        f"_z=1"  # forced
    )
    cache_base_path = out_dir / f"{cache_stem}.parquet"
    cache_cv_path = out_dir / f"{cache_stem}_cv=1.parquet"

    # ---------- helpers ----------
    def _cv_eids_present(df: "pd.DataFrame") -> set[str]:
        if df is None or df.empty:
            return set()
        if ("pair_type" not in df.columns) or ("eid_a" not in df.columns):
            return set()
        m = (df["pair_type"].astype(str) == "within_session")
        if not np.any(m):
            return set()
        return set(df.loc[m, "eid_a"].astype(str).unique())

    def _expected_eids_for_cv_from_df(df: "pd.DataFrame") -> set[str]:
        if df is None or df.empty:
            return set()
        ea = set(df.get("eid_a", pd.Series([], dtype=object)).astype(str).unique())
        eb = set(df.get("eid_b", pd.Series([], dtype=object)).astype(str).unique())
        return {e for e in (ea | eb) if e and e.lower() != "none"}

    def _load_meta_dict(p: Path) -> dict:
        obj = np.load(p, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.shape == ():
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.flat[0]
        return obj

    def _avg_trials_1bin(X_mnt: np.ndarray) -> np.ndarray | None:
        """Expect (M,N,1). Return (N,) trial-mean, or None if M < min_trials."""
        if X_mnt.ndim != 3 or X_mnt.shape[2] != 1:
            raise ValueError(f"Expected (M,N,1), got {X_mnt.shape}")
        if X_mnt.shape[0] < min_trials:
            return None
        return np.nanmean(X_mnt[:, :, 0], axis=0).astype(np.float32, copy=False)

    def _zscore_rows(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        return (X - mu) / sd

    def _rowwise_cosine_sim01(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        mask = np.isfinite(A) & np.isfinite(B)
        A0 = np.where(mask, A, 0.0).astype(np.float32, copy=False)
        B0 = np.where(mask, B, 0.0).astype(np.float32, copy=False)
        num = np.sum(A0 * B0, axis=1)
        den = np.sqrt(np.sum(A0 * A0, axis=1) * np.sum(B0 * B0, axis=1))
        den = np.where(den == 0, np.nan, den)
        cos = num / den
        return (0.5 * (cos + 1.0)).astype(np.float32)

    def _compute_cv_rows_for_eid(eid: str) -> list[dict]:
        print(f"[chronic CV] computing within-session baseline for {eid}")

        def _as1d_obj(x) -> np.ndarray:
            if x is None:
                return np.empty(0, dtype=object)
            if isinstance(x, (str, bytes)):
                return np.array([x], dtype=object)
            arr = np.asarray(x, dtype=object)
            if arr.ndim == 0:
                return np.array([arr.item()], dtype=object)
            return arr.ravel().astype(object, copy=False)

        def _as_xyz(x) -> np.ndarray:
            if x is None:
                return np.empty((0, 3), dtype=np.float32)
            arr = np.asarray(x, dtype=np.float32)
            if arr.ndim == 1:
                if arr.size >= 3:
                    arr = arr.reshape(1, -1)
                else:
                    return np.empty((0, 3), dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 3:
                return np.empty((0, 3), dtype=np.float32)
            return arr[:, :3]

        single_eid_cuts_dir = Path(pth_meso) / "trial_cuts" / eid

        if (not single_eid_cuts_dir.is_dir()) or (not any(single_eid_cuts_dir.glob("*meta_filter_*.npy"))):
            print(f"[chronic CV]   building trial cuts for {eid}")
            save_trial_cuts_meso(
                eid,
                filter_neurons=filter_neurons,
                require_all=False,
                out_dir=single_eid_cuts_dir,
            )

        metas = sorted(single_eid_cuts_dir.glob(f"{eid}_FOV_*_meta_filter_{filter_neurons}.npy"))
        if not metas:
            print(f"[chronic CV]   no valid metas found for {eid}")
            return []

        cv_rows: list[dict] = []

        for mp in metas:
            meta = _load_meta_dict(mp)
            if bool(meta.get("filter_neurons")) != bool(filter_neurons):
                continue

            fov = str(meta.get("fov", ""))
            ttypes = list(meta.get("trial_names", []))
            if not ttypes:
                continue

            uids = _as1d_obj(meta.get("restrict_uids_used", None))
            area = _as1d_obj(meta.get("region_labels", None))
            xyz = _as_xyz(meta.get("xyz", None))

            if uids.size == 0:
                N0 = int(max(area.size, xyz.shape[0]))
                if N0 <= 0:
                    continue
                uids = np.array([f"{fov}:roi{i:06d}" for i in range(N0)], dtype=object)

            N = int(uids.size)

            if area.size == 0:
                area = np.array([""] * N, dtype=object)
            if area.size != N:
                print(f"[chronic CV]   {eid} {fov}: area size {area.size} != uids size {N} -> skip FOV")
                continue

            if xyz.shape[0] == 0:
                print(f"[chronic CV]   {eid} {fov}: missing xyz -> skip FOV")
                continue
            if xyz.shape[0] != N:
                print(f"[chronic CV]   {eid} {fov}: xyz rows {xyz.shape[0]} != uids size {N} -> skip FOV")
                continue

            feats_even: list[np.ndarray] = []
            feats_odd: list[np.ndarray] = []

            for t in ttypes:
                fp = single_eid_cuts_dir / meta["peth_files"][t]
                if not fp.is_file():
                    feats_even = []
                    feats_odd = []
                    break

                X = np.load(fp, mmap_mode="r")
                if X.ndim != 3 or X.shape[2] != 1:
                    feats_even = []
                    feats_odd = []
                    break

                M = int(X.shape[0])
                if M < min_trials:
                    feats_even = []
                    feats_odd = []
                    break

                X_even = X[0::2, :, 0]
                X_odd = X[1::2, :, 0]
                if (X_even.shape[0] == 0) or (X_odd.shape[0] == 0):
                    feats_even = []
                    feats_odd = []
                    break

                a_even = np.nanmean(X_even, axis=0).astype(np.float32, copy=False).ravel()
                a_odd = np.nanmean(X_odd, axis=0).astype(np.float32, copy=False).ravel()

                if a_even.size != N or a_odd.size != N:
                    print(
                        f"[chronic CV]   {eid} {fov} {t}: feature size mismatch "
                        f"(even {a_even.size}, odd {a_odd.size}, N {N}) -> skip FOV"
                    )
                    feats_even = []
                    feats_odd = []
                    break

                feats_even.append(a_even)
                feats_odd.append(a_odd)

            if not feats_even:
                continue

            A_even_raw = np.stack(feats_even, axis=1).astype(np.float32, copy=False)  # (N,P)
            A_odd_raw = np.stack(feats_odd, axis=1).astype(np.float32, copy=False)

            # ALWAYS store and compute on z-scored features
            A_even = _zscore_rows(A_even_raw)
            A_odd = _zscore_rows(A_odd_raw)

            r = _rowwise_cosine_sim01(A_even, A_odd)

            for i in range(N):
                d = {
                    "subject": subject,
                    "pair_tag": f"{eid}__CV",
                    "pair_type": "within_session",
                    "eid_a": eid,
                    "eid_b": eid,
                    "dt_days": 0.0,
                    "FOV": fov,
                    "uid": uids[i],
                    "area_a": area[i],
                    "area_b": area[i],
                    "x_a": float(xyz[i, 0]),
                    "y_a": float(xyz[i, 1]),
                    "z_a": float(xyz[i, 2]),
                    "x_b": float(xyz[i, 0]),
                    "y_b": float(xyz[i, 1]),
                    "z_b": float(xyz[i, 2]),
                    "corr": float(r[i]) if np.isfinite(r[i]) else np.nan,
                }
                # Store ONLY z-scored features
                for j, t in enumerate(ttypes):
                    d[f"{t}_a"] = float(A_even[i, j]) if np.isfinite(A_even[i, j]) else np.nan
                    d[f"{t}_b"] = float(A_odd[i, j]) if np.isfinite(A_odd[i, j]) else np.nan
                cv_rows.append(d)

            gc.collect()

        print(f"[chronic CV]   {eid}: generated {len(cv_rows)} CV rows")
        return cv_rows

    # ---------- cache fast paths ----------
    if not rerun:
        if include_cv_baseline and cache_cv_path.is_file():
            df_cached = pd.read_parquet(cache_cv_path)
            expected = _expected_eids_for_cv_from_df(df_cached)
            present = _cv_eids_present(df_cached)
            if expected.issubset(present):
                print(f"[chronic] using cached CV dataframe: {cache_cv_path}")
                return df_cached

            missing = sorted(expected - present)
            print(
                f"[chronic CV] cv cache incomplete: have {len(present)}/{len(expected)} eids; "
                f"adding {len(missing)} missing"
            )
            new_rows: list[dict] = []
            for i, eid in enumerate(missing, start=1):
                print(f"[chronic CV] ({i}/{len(missing)}) eid {eid}")
                new_rows.extend(_compute_cv_rows_for_eid(eid))

            if new_rows:
                df_new = pd.DataFrame(new_rows)
                df_cached = pd.concat([df_cached, df_new], ignore_index=True)
                df_cached.sort_values(["pair_tag", "FOV", "uid"], inplace=True, kind="mergesort", ignore_index=True)
                df_cached.to_parquet(cache_cv_path, index=False)
                print(f"[chronic CV] updated CV cache saved: {cache_cv_path}")

            return df_cached

        if include_cv_baseline and cache_base_path.is_file():
            df_base = pd.read_parquet(cache_base_path)
            expected = _expected_eids_for_cv_from_df(df_base)
            present = _cv_eids_present(df_base)
            missing = sorted(expected - present)

            print(f"[chronic CV] base cache found; computing CV for {len(missing)} sessions only")
            new_rows: list[dict] = []
            for i, eid in enumerate(missing, start=1):
                print(f"[chronic CV] ({i}/{len(missing)}) eid {eid}")
                new_rows.extend(_compute_cv_rows_for_eid(eid))

            df_cv = pd.concat([df_base, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else df_base.copy()
            df_cv.sort_values(["pair_tag", "FOV", "uid"], inplace=True, kind="mergesort", ignore_index=True)
            df_cv.to_parquet(cache_cv_path, index=False)
            print(f"[chronic CV] CV cache saved: {cache_cv_path}")
            return df_cv

        if (not include_cv_baseline) and cache_base_path.is_file():
            print(f"[chronic] using cached base dataframe: {cache_base_path}")
            return pd.read_parquet(cache_base_path)

    # ---------- get EIDs for subject ----------
    subj2eids = _canonical_sessions_subject_map(one)
    eids = [str(e) for e in subj2eids.get(subject, [])]
    eids = [e for e in eids if e and e.lower() != "none"]
    if len(eids) < 2:
        raise ValueError(f"{subject}: need >=2 canonical sessions, got {len(eids)}.")

    eids_sorted = sorted(eids, key=lambda e: (_eid_date(one, e), e))

    if pair_mode == "consecutive":
        pairs = [(eids_sorted[i], eids_sorted[i + 1]) for i in range(len(eids_sorted) - 1)]
    elif pair_mode == "all":
        pairs = list(combinations(eids_sorted, 2))
    else:
        raise ValueError("pair_mode must be 'all' or 'consecutive'.")

    # ---------- pair-cuts helper ----------
    def _ensure_pair_cuts(eid_a: str, eid_b: str) -> "Path | None":
        from meso_chronic import match_tracked_indices_pair_per_fov

        pair_tag = f"{eid_a}__{eid_b}"
        pair_dir = trial_cuts_root / subject / pair_tag
        dir_a = pair_dir / eid_a
        dir_b = pair_dir / eid_b
        dir_a.mkdir(parents=True, exist_ok=True)
        dir_b.mkdir(parents=True, exist_ok=True)

        have_a = any(dir_a.glob(f"{eid_a}_FOV_*_meta_filter_*.npy"))
        have_b = any(dir_b.glob(f"{eid_b}_FOV_*_meta_filter_*.npy"))
        if have_a and have_b:
            return pair_dir

        fov_match = match_tracked_indices_pair_per_fov(
            one,
            eid_a,
            eid_b,
            roicat_root=roicat_root,
            server_root=server_root,
            filter_neurons=filter_neurons,
        )

        restrict_a: dict[str, np.ndarray] = {}
        restrict_b: dict[str, np.ndarray] = {}
        uids_by_fov: dict[str, np.ndarray] = {}

        total_shared = 0
        for fov, m in fov_match.items():
            restrict_a[fov] = m.idx_a
            restrict_b[fov] = m.idx_b
            uids_by_fov[fov] = m.shared_uids
            total_shared += int(m.shared_uids.size)

        if total_shared == 0:
            return None

        save_trial_cuts_meso(
            eid_a,
            filter_neurons=filter_neurons,
            require_all=False,
            out_dir=dir_a,
            restrict=restrict_a,
            restrict_uids=uids_by_fov,
            pair_tag=pair_tag,
        )
        save_trial_cuts_meso(
            eid_b,
            filter_neurons=filter_neurons,
            require_all=False,
            out_dir=dir_b,
            restrict=restrict_b,
            restrict_uids=uids_by_fov,
            pair_tag=pair_tag,
        )
        return pair_dir

    # ---------- main loop: between-session ----------
    rows_between: list[dict] = []
    unique_eids: set[str] = set()

    print(f"[chronic] subject={subject} | pairs={len(pairs)} | filter={int(filter_neurons)} | min_trials={min_trials}")
    for ip, (eid_a, eid_b) in enumerate(pairs, start=1):
        t_pair = time.time()
        pair_tag = f"{eid_a}__{eid_b}"
        print(f"[chronic] ({ip}/{len(pairs)}) pair {pair_tag}")

        unique_eids.add(eid_a)
        unique_eids.add(eid_b)

        pair_dir = _ensure_pair_cuts(eid_a, eid_b)
        if pair_dir is None:
            print(f"[chronic]   {pair_tag}: no shared neurons -> skip")
            continue

        dir_a = pair_dir / eid_a
        dir_b = pair_dir / eid_b

        metas_a = sorted(dir_a.glob(f"{eid_a}_FOV_*_meta_filter_*.npy"))
        metas_b = sorted(dir_b.glob(f"{eid_b}_FOV_*_meta_filter_*.npy"))
        if (not metas_a) or (not metas_b):
            print(f"[chronic]   {pair_tag}: missing metas -> skip")
            continue

        ma = {(_load_meta_dict(p)["fov"]): _load_meta_dict(p) for p in metas_a}
        mb = {(_load_meta_dict(p)["fov"]): _load_meta_dict(p) for p in metas_b}

        fovs = sorted(set(ma.keys()) & set(mb.keys()))
        if not fovs:
            print(f"[chronic]   {pair_tag}: no overlapping FOVs -> skip")
            continue

        ref_ttypes = list(ma[fovs[0]].get("trial_names", []))
        if not ref_ttypes:
            print(f"[chronic]   {pair_tag}: empty trial_names -> skip")
            continue
        if list(mb[fovs[0]].get("trial_names", [])) != ref_ttypes:
            raise ValueError(f"{pair_tag}: trial_names mismatch between eids (meta inconsistency).")

        n_rows_before = len(rows_between)

        for fov in fovs:
            metaA = ma[fov]
            metaB = mb[fov]

            if bool(metaA.get("filter_neurons")) != bool(filter_neurons):
                continue
            if bool(metaB.get("filter_neurons")) != bool(filter_neurons):
                continue

            uidsA = np.asarray(metaA.get("restrict_uids_used", []), dtype=object)
            uidsB = np.asarray(metaB.get("restrict_uids_used", []), dtype=object)
            if uidsA.size == 0 or uidsB.size == 0:
                continue
            if uidsA.shape != uidsB.shape or np.any(uidsA != uidsB):
                raise ValueError(f"{pair_tag} {fov}: restrict_uids_used mismatch between eids.")

            areaA = np.asarray(metaA.get("region_labels", []), dtype=object)
            areaB = np.asarray(metaB.get("region_labels", []), dtype=object)
            xyzA = np.asarray(metaA.get("xyz", []), dtype=np.float32)[:, :3]
            xyzB = np.asarray(metaB.get("xyz", []), dtype=np.float32)[:, :3]

            if areaA.shape[0] != uidsA.shape[0] or areaB.shape[0] != uidsA.shape[0]:
                raise ValueError(f"{pair_tag} {fov}: N mismatch between uids and region_labels.")
            if xyzA.shape[0] != uidsA.shape[0] or xyzB.shape[0] != uidsA.shape[0]:
                raise ValueError(f"{pair_tag} {fov}: N mismatch between uids and xyz.")

            featsA: list[np.ndarray] = []
            featsB: list[np.ndarray] = []

            for t in ref_ttypes:
                fpA = dir_a / metaA["peth_files"][t]
                fpB = dir_b / metaB["peth_files"][t]

                XA = np.load(fpA, mmap_mode="r")
                XB = np.load(fpB, mmap_mode="r")

                a = _avg_trials_1bin(XA)
                b = _avg_trials_1bin(XB)
                if a is None or b is None:
                    featsA = []
                    featsB = []
                    break
                if a.shape[0] != uidsA.shape[0] or b.shape[0] != uidsA.shape[0]:
                    raise ValueError(f"{pair_tag} {fov} {t}: N mismatch in averaged features.")

                featsA.append(a)
                featsB.append(b)

            if not featsA:
                continue

            A_raw = np.stack(featsA, axis=1).astype(np.float32, copy=False)  # (N,P)
            B_raw = np.stack(featsB, axis=1).astype(np.float32, copy=False)

            # ALWAYS store and compute on z-scored features
            Ause = _zscore_rows(A_raw)
            Buse = _zscore_rows(B_raw)

            r = _rowwise_cosine_sim01(Ause, Buse)

            for i in range(uidsA.shape[0]):
                d = {
                    "subject": subject,
                    "pair_tag": pair_tag,
                    "pair_type": "between_session",
                    "eid_a": eid_a,
                    "eid_b": eid_b,
                    "dt_days": np.nan,
                    "FOV": fov,
                    "uid": uidsA[i],
                    "area_a": areaA[i],
                    "area_b": areaB[i],
                    "x_a": float(xyzA[i, 0]),
                    "y_a": float(xyzA[i, 1]),
                    "z_a": float(xyzA[i, 2]),
                    "x_b": float(xyzB[i, 0]),
                    "y_b": float(xyzB[i, 1]),
                    "z_b": float(xyzB[i, 2]),
                    "corr": float(r[i]) if np.isfinite(r[i]) else np.nan,
                }
                # Store ONLY z-scored features
                for j, t in enumerate(ref_ttypes):
                    d[f"{t}_a"] = float(Ause[i, j]) if np.isfinite(Ause[i, j]) else np.nan
                    d[f"{t}_b"] = float(Buse[i, j]) if np.isfinite(Buse[i, j]) else np.nan
                rows_between.append(d)

        gc.collect()
        print(
            f"[chronic] ({ip}/{len(pairs)}) pair {pair_tag} "
            f"added {len(rows_between) - n_rows_before} rows in {time.time() - t_pair:.1f}s"
        )

    if not rows_between:
        raise RuntimeError(f"{subject}: no between-session rows produced (no usable pairs/FOVs after filtering).")

    df_between = pd.DataFrame(rows_between)
    df_between.sort_values(["pair_tag", "FOV", "uid"], inplace=True, kind="mergesort", ignore_index=True)
    df_between.to_parquet(cache_base_path, index=False)
    print(f"\n[chronic] Saved between-session dataframe to {cache_base_path}")
    print(f"[chronic]   Between-session rows: {len(df_between)}")

    if not include_cv_baseline:
        return df_between

    print(f"\n[chronic CV] Computing within-session baselines for {len(unique_eids)} sessions")
    cv_rows_all: list[dict] = []
    for i, eid in enumerate(sorted(unique_eids), start=1):
        try:
            print(f"[chronic CV] ({i}/{len(unique_eids)}) eid {eid}")
            cv_rows_all.extend(_compute_cv_rows_for_eid(eid))
        except Exception as e:
            print(f"[chronic CV] Failed for {eid}: {type(e).__name__}: {e}")
            continue

    df_cv = pd.concat([df_between, pd.DataFrame(cv_rows_all)], ignore_index=True) if cv_rows_all else df_between.copy()
    df_cv.sort_values(["pair_tag", "FOV", "uid"], inplace=True, kind="mergesort", ignore_index=True)
    df_cv.to_parquet(cache_cv_path, index=False)
    print(f"\n[chronic] Saved dataframe (with CV) to {cache_cv_path}")
    print(f"[chronic]   Total rows: {len(df_cv)}")
    print(f"[chronic]   Within-session CV rows: {(df_cv['pair_type'].astype(str) == 'within_session').sum()}")

    return df_cv


def classify_session_protocol(one: ONE, eid: str) -> str:
    """
    Classify an IBL session by task protocol.

    Returns one of:
        - "training"
        - "biased"
        - "other"

    Logic:
      - Uses Alyx 'task_protocol' primarily
      - Falls back to 'type' if needed
    """
    ses = one.alyx.rest("sessions", "read", id=str(eid))

    tp = (ses.get("task_protocol") or "").lower()
    st = (ses.get("type") or "").lower()

    # training / habituation
    if any(k in tp for k in ["training", "habituation"]):
        return "training"

    # biased choiceworld (trained task)
    if "biased" in tp or "biasedchoiceworld" in tp or "biased_choiceworld" in tp:
        return "biased"

    # fallback: sometimes encoded in type
    if "training" in st:
        return "training"
    if "biased" in st:
        return "biased"

    return "other"


def filter_chronic_df_by_session(
    df: pd.DataFrame,
    *,
    one: ONE,
    sess: str = "all",   # "all" | "training" | "biased"
) -> pd.DataFrame:
    """
    Filter chronic_pair_feature_corr_subject dataframe by session type.

    sess:
        - "all": no filtering
        - "training": keep rows where involved eids are training sessions
        - "biased": keep rows where involved eids are biasedChoiceWorld sessions
    """
    if sess == "all":
        return df

    if sess not in ("training", "biased"):
        raise ValueError("sess must be 'all', 'training', or 'biased'")

    # collect unique eids present
    eids = pd.unique(
        pd.concat([df["eid_a"], df["eid_b"]], ignore_index=True)
        .astype(str)
    )
    eids = [e for e in eids if e and e.lower() != "none"]

    # classify once
    eid2cls = {eid: classify_session_protocol(one, eid) for eid in eids}

    if sess == "training":
        keep = {eid for eid, c in eid2cls.items() if c == "training"}
    else:  # sess == "biased"
        keep = {eid for eid, c in eid2cls.items() if c == "biased"}

    # row-level filtering
    def _row_ok(row):
        ea = str(row["eid_a"])
        eb = str(row["eid_b"])
        # CV rows: ea == eb
        return (ea in keep) and (eb in keep)

    mask = df.apply(_row_ok, axis=1)
    return df.loc[mask].reset_index(drop=True)


###################
'''
plotting pairwise correlations
'''
###################

def _eid_date_from_path(one: ONE, eid: str) -> pd.Timestamp:
    """
    Fast local extraction of session date from ONE eid path.
    Returns pandas.Timestamp (date only).
    """
    p = Path(one.eid2path(eid))
    # expect .../<subject>/<YYYY-MM-DD>/<NNN>
    for part in p.parts:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
            return pd.to_datetime(part)
    raise ValueError(f"Could not parse date from eid path: {p}")


def plot_chronic_corr_vs_time_subject(
    subject: str = "SP058",
    eid_c: str | None = "20ebc2b9-5b4c-42cd-8e4b-65ddb427b7ff",
    *,
    one: "ONE | None" = None,
    pair_mode: str = "all",   # "consecutive" | "all"
    rerun: bool = False,
    per_reg: bool = False,
    reg_col: str = "area_a",          # "area_a" | "area_b"
    agg: str = "mean",                # "mean" | "median"
    min_neurons: int = 50,
    shuf: bool = True,
    shuf_seed: int = 0,
    ax: "plt.Axes | None" = None,
    title: str | None = None,
    filter_neurons: bool = True,
    include_cv_baseline: bool = True,
    sess: str = "all",   # "all" | "training" | "biased"
) -> "tuple[pd.DataFrame, plt.Axes]":


    if one is None:
        one = ONE()

    if agg not in ("mean", "median"):
        raise ValueError("agg must be 'mean' or 'median'")
    if per_reg and reg_col not in ("area_a", "area_b"):
        raise ValueError("reg_col must be 'area_a' or 'area_b'")
    if per_reg and not shuf:
        raise ValueError("per_reg=True requires shuf=True (data − shuffle is plotted).")

    df = chronic_pair_feature_corr_subject(
        subject,
        one=one,
        pair_mode=pair_mode,
        rerun=rerun,
        filter_neurons=filter_neurons,
        include_cv_baseline=include_cv_baseline,
        zscore_before_corr=True,  # enforced by chronic function anyway
    )
    if df is None or df.empty:
        raise ValueError(f"Empty df for subject={subject}")

    df = filter_chronic_df_by_session(df, one=one, sess=sess)

    def _get_feature_pairs(dfin: pd.DataFrame) -> tuple[list[str], list[str]]:
        cols_a = [c for c in dfin.columns if c.endswith("_a")]
        exclude_bases = {
            "eid", "area", "subject", "pair_tag", "FOV", "uid",
            "x", "y", "z", "dt_days", "pair_type",
        }
        ca_list, cb_list = [], []
        for ca in cols_a:
            base = ca[:-2]
            if base in exclude_bases:
                continue
            cb = base + "_b"
            if cb not in dfin.columns:
                continue
            a_num = pd.to_numeric(dfin[ca], errors="coerce")
            b_num = pd.to_numeric(dfin[cb], errors="coerce")
            if np.isfinite(a_num.to_numpy()).any() and np.isfinite(b_num.to_numpy()).any():
                ca_list.append(ca)
                cb_list.append(cb)
        if not ca_list:
            raise ValueError("No numeric feature pairs found for shuffle control.")
        return ca_list, cb_list

    def _rowwise_cos_sim01(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A, float)
        B = np.asarray(B, float)
        mask = np.isfinite(A) & np.isfinite(B)
        A0 = np.where(mask, A, 0.0)
        B0 = np.where(mask, B, 0.0)
        num = np.sum(A0 * B0, axis=1)
        den = np.sqrt(np.sum(A0 * A0, axis=1) * np.sum(B0 * B0, axis=1))
        den = np.where(den == 0, np.nan, den)
        return 0.5 * (num / den + 1.0)

    def _nansem(x: np.ndarray) -> float:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size <= 1:
            return np.nan
        return float(np.std(x, ddof=1) / np.sqrt(x.size))

    def _stable_u32_seed(*parts: object) -> int:
        s = "|".join(str(p) for p in parts)
        return abs(hash(s)) % (2**32 - 1)

    # eid → date mapping
    eids = pd.unique(pd.concat([df["eid_a"], df["eid_b"]], ignore_index=True).astype(str))
    eids = [e for e in eids if e and e.lower() != "none"]
    eid2t = {str(eid): _eid_date_from_path(one, str(eid)) for eid in eids}

    # dt_days=0 baseline from within-session CV rows + shuffle baseline for CV
    cv0_row = None
    if include_cv_baseline and ("pair_type" in df.columns):
        m_cv = (df["pair_type"].astype(str) == "within_session")
        if np.any(m_cv):
            df_cv0 = df.loc[m_cv].copy()
            df_cv0["eid_a"] = df_cv0["eid_a"].astype(str)

            dcv = df_cv0[["eid_a", "corr"]].copy()
            dcv["corr"] = pd.to_numeric(dcv["corr"], errors="coerce")
            per_sess = dcv.groupby("eid_a")["corr"].mean()
            per_sess = per_sess[np.isfinite(per_sess.to_numpy())]

            if per_sess.size > 0:
                cv0_row = dict(
                    dt_days=0.0,
                    corr_avg=float(per_sess.mean()),
                    corr_sem=_nansem(per_sess.to_numpy()),
                    corr_shuf_avg=np.nan,
                    _n_sessions=int(per_sess.size),
                )

                if shuf:
                    ca_list, cb_list = _get_feature_pairs(df_cv0)
                    shuf_means = []
                    for eid, dd in df_cv0.groupby("eid_a", sort=False):
                        if dd.shape[0] < min_neurons:
                            continue
                        A = dd[ca_list].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=False)
                        B = dd[cb_list].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=False)
                        n = A.shape[0]
                        rng = np.random.default_rng(_stable_u32_seed("cv", subject, eid, int(shuf_seed)))
                        perm = rng.permutation(n)
                        sim = _rowwise_cos_sim01(A, B[perm, :])
                        shuf_means.append(float(np.nanmean(sim)))
                    if len(shuf_means) > 0:
                        cv0_row["corr_shuf_avg"] = float(np.nanmean(np.asarray(shuf_means, float)))

    # use between-session rows only for dt>0 curve
    d_between = df[df["pair_type"].astype(str) == "between_session"].copy() if ("pair_type" in df.columns) else df.copy()
    if d_between.empty:
        raise ValueError("No between-session rows available for plotting.")

    # compute dt_days and choose grouping
    if eid_c is None:
        d0 = d_between.copy()
        ta = d0["eid_a"].astype(str).map(eid2t)
        tb = d0["eid_b"].astype(str).map(eid2t)
        ok = np.isfinite((tb - ta).dt.total_seconds())
        d0 = d0.loc[ok].copy()
        d0["dt_days"] = np.abs((tb.loc[ok] - ta.loc[ok]).dt.total_seconds()) / (3600 * 24)
        group_cols = ["eid_a", "eid_b", "dt_days", "pair_tag"]
        x_label = "Days between sessions"
        mode_label = "all pairs"
    else:
        if str(eid_c) not in eid2t:
            raise ValueError(f"eid_c not resolvable: {eid_c}")
        t_c = eid2t[str(eid_c)]
        d0 = d_between[(d_between["eid_a"] == eid_c) | (d_between["eid_b"] == eid_c)].copy()
        if d0.empty:
            raise ValueError(f"No rows for center eid_c={eid_c}")
        d0["eid_other"] = np.where(d0["eid_a"] == eid_c, d0["eid_b"], d0["eid_a"])
        d0["t_other"] = d0["eid_other"].astype(str).map(eid2t)
        ok = np.isfinite((d0["t_other"] - t_c).dt.total_seconds())
        d0 = d0.loc[ok].copy()
        d0["dt_days"] = (d0["t_other"] - t_c).dt.total_seconds() / (3600 * 24)
        group_cols = ["eid_other", "dt_days", "pair_tag"]
        x_label = "Days relative to center eid"
        mode_label = "centered"

    if per_reg:
        group_cols = group_cols + [reg_col]

    # aggregate observed per group
    g = d0.groupby(group_cols, dropna=False)
    corr_avg = g["corr"].mean() if agg == "mean" else g["corr"].median()
    corr_sem = g["corr"].apply(lambda s: _nansem(pd.to_numeric(s, errors="coerce").to_numpy()))
    n_neu = g["corr"].size()

    out = (
        pd.concat([corr_avg.rename("corr_avg"), corr_sem.rename("corr_sem"), n_neu.rename("n_neurons")], axis=1)
        .reset_index()
        .query("n_neurons >= @min_neurons")
        .sort_values("dt_days", kind="mergesort")
        .reset_index(drop=True)
    )

    # shuffle baseline (now consistent because *_a/*_b are z-scored everywhere)
    if shuf:
        ca_list, cb_list = _get_feature_pairs(d0)

        def _group_shuf_mean(dd: pd.DataFrame) -> float:
            if dd.shape[0] < min_neurons:
                return np.nan
            A = dd[ca_list].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=False)
            B = dd[cb_list].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=False)
            n = A.shape[0]
            tag = str(dd["pair_tag"].iloc[0]) if ("pair_tag" in dd.columns and dd["pair_tag"].notna().any()) else "group"
            rng = np.random.default_rng(_stable_u32_seed("between", subject, tag, int(shuf_seed)))
            ia = rng.permutation(n)
            ib = rng.permutation(n)
            sim = _rowwise_cos_sim01(A[ia], B[ib])
            return float(np.nanmean(sim))

        shuf_series = g.apply(_group_shuf_mean)
        out = out.merge(shuf_series.rename("corr_shuf_avg").reset_index(), on=group_cols, how="left")

    # collapse by dt_days when eid_c is None, then inject dt=0 CV point
    out_plot = out
    if eid_c is None:
        if not per_reg:
            out_plot = (
                out.groupby("dt_days", dropna=False)
                .agg(
                    corr_avg=("corr_avg", "mean"),
                    corr_sem=("corr_avg", _nansem),
                    corr_shuf_avg=("corr_shuf_avg", "mean") if shuf else ("corr_avg", lambda x: np.nan),
                )
                .reset_index()
                .sort_values("dt_days")
                .reset_index(drop=True)
            )
            if cv0_row is not None:
                add = pd.DataFrame([{
                    "dt_days": 0.0,
                    "corr_avg": cv0_row["corr_avg"],
                    "corr_sem": cv0_row["corr_sem"],
                    "corr_shuf_avg": cv0_row.get("corr_shuf_avg", np.nan),
                }])
                out_plot = (
                    pd.concat([add, out_plot], ignore_index=True)
                    .drop_duplicates(subset=["dt_days"], keep="first")
                    .sort_values("dt_days")
                    .reset_index(drop=True)
                )
        else:
            out_plot = (
                out.groupby(["dt_days", reg_col], dropna=False)
                .agg(
                    corr_avg=("corr_avg", "mean"),
                    corr_shuf_avg=("corr_shuf_avg", "mean") if shuf else ("corr_avg", lambda x: np.nan),
                )
                .reset_index()
                .sort_values(["dt_days", reg_col])
                .reset_index(drop=True)
            )

    # plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    if eid_c is None and per_reg:
        out_plot["delta"] = out_plot["corr_avg"] - out_plot["corr_shuf_avg"]
        for reg, dd in out_plot.groupby(reg_col, sort=False):
            ax.plot(dd["dt_days"], dd["delta"], marker="o", lw=1.5, label=str(reg))
        gg = out_plot.groupby("dt_days")["delta"].mean().reset_index()
        ax.plot(gg["dt_days"], gg["delta"], marker="o", lw=2.5, color="k", label="all neurons")
        ax.axhline(0.0, color="k", lw=0.75, alpha=0.4)
        ax.set_ylabel("Δ cosine similarity (data − shuffle)")
        ax.legend(frameon=False)
    else:
        x = out_plot["dt_days"].to_numpy()
        y = out_plot["corr_avg"].to_numpy()
        ax.plot(x, y, marker="o", lw=1.5)

        if shuf and "corr_shuf_avg" in out_plot.columns:
            ys = out_plot["corr_shuf_avg"].to_numpy()
            ax.plot(x, ys, lw=1.0, alpha=0.6)
            for xi, ysi, ydi in zip(x, ys, y):
                if np.isfinite(ysi) and np.isfinite(ydi):
                    ax.plot([xi, xi], [ysi, ydi], color="k", lw=0.8, alpha=0.4)

        ax.set_ylabel(f"{agg} cosine similarity")

    if eid_c is not None:
        ax.axvline(0.0, color="k", lw=0.75, alpha=0.5)

    ax.set_xlabel(x_label)

    xmin, xmax = ax.get_xlim()
    if np.isfinite(xmin) and np.isfinite(xmax):
        ticks = np.arange(np.floor(xmin), np.ceil(xmax) + 1)
        if len(ticks) <= 50:  # avoid too many ticks
            ax.set_xticks(ticks)

    if title is None:
        title = (
            f"{subject}: cosine similarity vs time "
            f"({mode_label}, {sess})"
        )
        if per_reg:
            title += " | per_reg"

    ax.set_title(title)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    return out_plot, ax


def _session_behavior_metrics(one: ONE, eid: str) -> dict:
    """
    Compute compact behavioral/session-level metrics from `trials`.

    Returns
    -------
    dict with keys:
        performance, n_trials_total, n_trials_scored,
        n_correct, n_incorrect, n_no_choice, median_rt_s
    """
    trials = one.load_object(eid, "trials")

    fb = np.asarray(trials.get("feedbackType", []), dtype=float)
    stim = np.asarray(trials.get("stimOn_times", []), dtype=float)
    first = np.asarray(trials.get("firstMovement_times", []), dtype=float)

    n_trials_total = int(stim.size) if stim.size > 0 else int(fb.size)

    valid_fb = np.isfinite(fb) & np.isin(fb, (-1, 1))
    n_trials_scored = int(np.sum(valid_fb))
    n_correct = int(np.sum(fb[valid_fb] == 1)) if n_trials_scored > 0 else 0
    n_incorrect = int(np.sum(fb[valid_fb] == -1)) if n_trials_scored > 0 else 0
    n_no_choice = int(max(n_trials_total - n_trials_scored, 0))

    performance = float(n_correct / n_trials_scored) if n_trials_scored > 0 else np.nan

    median_rt_s = np.nan
    if stim.size > 0 and first.size > 0:
        n = min(int(stim.size), int(first.size))
        rt = first[:n] - stim[:n]
        ok = np.isfinite(rt) & (rt > 0) & (rt < 10)
        if np.any(ok):
            median_rt_s = float(np.nanmedian(rt[ok]))

    return {
        "performance": performance,
        "n_trials_total": n_trials_total,
        "n_trials_scored": n_trials_scored,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_no_choice": n_no_choice,
        "median_rt_s": median_rt_s,
    }


def plot_session_even_odd_similarity_vs_performance_subject(
    subject: str = "SP058",
    *,
    one: "ONE | None" = None,
    filter_neurons: bool = True,
    min_trials: int = 10,
    rerun_canonical: bool = False,
    rerun_trial_cuts: bool = False,
    rerun_stacks: bool = False,
    rerun_group: bool = False,
    nclus_rm: int = 20,
    shuf: bool = True,
    shuf_seed: int = 0,
    out_dir: str | Path | None = None,
    save_outputs: bool = True,
    fail_fast: bool = False,
    verbose: bool = True,
):
    """
    For each session of one subject, compute within-session neural stability
    as cosine similarity between even- and odd-trial averaged activity patterns,
    then relate it to behavioral performance.

    IMPORTANT
    ---------
    - This is independent of chronic tracking across sessions.
    - Processing is session-local (single_eid mode), using all available ROIs
      after optional filtering.

    Returns
    -------
    df : pd.DataFrame
        One row per session.
    fig : matplotlib.figure.Figure
        Two-panel summary figure.
    paths : dict
        Saved output paths (csv/parquet/png/svg).
    """
    if one is None:
        one = ONE()

    subj2eids = _canonical_sessions_subject_map(one=one, rerun=rerun_canonical)
    eids = [str(e) for e in subj2eids.get(subject, []) if e and str(e).lower() != "none"]
    if len(eids) == 0:
        raise ValueError(f"No canonical sessions found for subject={subject}.")

    # Stable chronological order
    eids_sorted = sorted(eids, key=lambda e: (_eid_date(one, e), e))

    trial_cuts_dir = Path(pth_meso, "trial_cuts")
    stack_dir = Path(pth_meso)

    if out_dir is None:
        out_dir = Path(pth_meso, "session_cv_similarity")
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _sim01(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = np.asarray(A, float)
        B = np.asarray(B, float)
        num = np.sum(A * B, axis=1)
        den = np.sqrt(np.sum(A**2, axis=1) * np.sum(B**2, axis=1))
        out = np.full_like(num, np.nan, float)
        good = den > 0
        out[good] = num[good] / den[good]
        return 0.5 * (out + 1.0)

    def _seed_u32(*parts: object) -> int:
        s = "|".join(str(p) for p in parts)
        return abs(hash(s)) % (2**32 - 1)

    rows = []
    failures = []

    for i, eid in enumerate(eids_sorted, start=1):
        try:
            if verbose:
                print(f"\n[session {i}/{len(eids_sorted)}] {eid}")

            single_cuts_dir = trial_cuts_dir / eid
            has_metas = single_cuts_dir.is_dir() and any(
                single_cuts_dir.glob(f"{eid}_FOV_*_meta_filter_{filter_neurons}.npy")
            )

            if rerun_trial_cuts or (not has_metas):
                if verbose:
                    print("  building trial cuts...")
                save_trial_cuts_meso(
                    eid,
                    filter_neurons=filter_neurons,
                    require_all=False,
                    out_dir=single_cuts_dir,
                )

            stack_all = stack_dir / f"stack_all_{eid}_filter{filter_neurons}.npy"
            stack_even = stack_dir / f"stack_even_{eid}_filter{filter_neurons}.npy"
            stack_odd = stack_dir / f"stack_odd_{eid}_filter{filter_neurons}.npy"

            if rerun_stacks or (not stack_all.is_file()) or (not stack_even.is_file()) or (not stack_odd.is_file()):
                if verbose:
                    print("  building single-session all/even/odd stacks...")
                stack_trial_cuts_meso(
                    trial_cuts_dir=trial_cuts_dir,
                    filter_neurons=filter_neurons,
                    min_trials=min_trials,
                    single_eid=eid,
                )

            for p in (stack_all, stack_even, stack_odd):
                if not p.is_file():
                    raise FileNotFoundError(f"Expected stack file missing: {p}")

            r = regional_group_meso(
                mapping="cv_sim",
                stack_dir=stack_dir,
                filter_neurons=filter_neurons,
                cv=True,
                nclus=100,
                nclus_rm=nclus_rm,
                rerun=rerun_group,
                single_eid=eid,
            )

            sim = np.asarray(r.get("cv_sim", []), dtype=float)
            finite = np.isfinite(sim)
            if sim.size == 0 or (not np.any(finite)):
                raise RuntimeError("cv_sim is empty/non-finite for this session.")

            sim_mean = float(np.nanmean(sim[finite]))
            sim_sem = float(np.nanstd(sim[finite], ddof=1) / np.sqrt(np.sum(finite))) if np.sum(finite) > 1 else np.nan
            n_neurons = int(np.sum(finite))

            sim_shuf_mean = np.nan
            sim_delta = np.nan
            if shuf:
                Xe = np.asarray(r["concat_z_even"], dtype=float)
                Xo = np.asarray(r["concat_z_odd"], dtype=float)
                if Xe.shape != Xo.shape:
                    raise ValueError(f"concat_z_even/odd mismatch: {Xe.shape} vs {Xo.shape}")

                rng = np.random.default_rng(_seed_u32(subject, eid, int(shuf_seed)))
                keys = rng.random(Xo.shape)
                perm = np.argsort(keys, axis=1)
                Xo_sh = np.take_along_axis(Xo, perm, axis=1)

                sim_sh = _sim01(Xe, Xo_sh)
                sim_shuf_mean = float(np.nanmean(sim_sh))
                sim_delta = sim_mean - sim_shuf_mean

            beh = _session_behavior_metrics(one, eid)
            date = _eid_date_from_path(one, eid)

            rows.append(
                {
                    "subject": subject,
                    "eid": str(eid),
                    "session_date": pd.to_datetime(date),
                    "cv_sim_mean": sim_mean,
                    "cv_sim_sem": sim_sem,
                    "cv_sim_shuf_mean": sim_shuf_mean,
                    "cv_sim_delta": sim_delta,
                    "n_neurons": n_neurons,
                    **beh,
                }
            )

            if verbose:
                ptxt = beh["performance"]
                if np.isfinite(ptxt):
                    print(f"  done: n_neurons={n_neurons}, cv_sim={sim_mean:.4f}, performance={ptxt:.4f}")
                else:
                    print(f"  done: n_neurons={n_neurons}, cv_sim={sim_mean:.4f}, performance=nan")

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            failures.append((eid, msg))
            print(f"[skip] {eid}: {msg}")
            if fail_fast:
                raise

    if len(rows) == 0:
        fail_txt = "\n".join([f"- {eid}: {err}" for eid, err in failures])
        raise RuntimeError(f"No sessions produced usable results for {subject}.\n{fail_txt}")

    df = pd.DataFrame(rows).sort_values("session_date").reset_index(drop=True)
    df["session_number"] = np.arange(1, len(df) + 1, dtype=int)

    cols_first = [
        "subject", "session_number", "session_date", "eid",
        "performance", "cv_sim_mean", "cv_sim_sem", "cv_sim_shuf_mean", "cv_sim_delta",
        "n_neurons", "n_trials_total", "n_trials_scored", "n_correct", "n_incorrect", "n_no_choice", "median_rt_s",
    ]
    cols_first = [c for c in cols_first if c in df.columns]
    df = df[cols_first + [c for c in df.columns if c not in cols_first]]

    # ---------- plotting ----------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: progression over sessions
    x = df["session_number"].to_numpy(int)
    y_cv = df["cv_sim_mean"].to_numpy(float)
    y_perf = df["performance"].to_numpy(float)

    ax0.plot(x, y_cv, marker="o", lw=1.8, color="tab:blue", label="CV similarity")
    ax0.set_xlabel("Session number")
    ax0.set_ylabel("Even/Odd cosine similarity [0,1]", color="tab:blue")
    ax0.tick_params(axis="y", labelcolor="tab:blue")
    ax0.set_ylim(0, 1)

    ax0b = ax0.twinx()
    ax0b.plot(x, y_perf, marker="s", lw=1.5, color="tab:orange", label="Performance")
    ax0b.set_ylabel("Performance (feedbackType==1 fraction)", color="tab:orange")
    ax0b.tick_params(axis="y", labelcolor="tab:orange")
    ax0b.set_ylim(0, 1)

    # Combined legend
    h0, l0 = ax0.get_legend_handles_labels()
    h1, l1 = ax0b.get_legend_handles_labels()
    ax0.legend(h0 + h1, l0 + l1, frameon=False, loc="lower right")

    # Panel B: performance vs stability
    m = np.isfinite(y_perf) & np.isfinite(y_cv)
    ax1.scatter(y_perf[m], y_cv[m], s=45, alpha=0.85, color="k")

    # annotate by session number for traceability
    for xi, yi, si in zip(y_perf[m], y_cv[m], x[m]):
        ax1.text(xi, yi, str(int(si)), fontsize=8, ha="left", va="bottom", alpha=0.75)

    r_val = np.nan
    if np.sum(m) >= 2:
        r_val = float(np.corrcoef(y_perf[m], y_cv[m])[0, 1])
        xp = y_perf[m]
        yp = y_cv[m]
        if np.nanstd(xp) > 0:
            coef = np.polyfit(xp, yp, deg=1)
            xx = np.linspace(np.nanmin(xp), np.nanmax(xp), 100)
            yy = coef[0] * xx + coef[1]
            ax1.plot(xx, yy, lw=1.5, color="tab:green", alpha=0.9)

    ax1.set_xlabel("Performance")
    ax1.set_ylabel("Even/Odd cosine similarity")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ttl = f"{subject}: session neural stability vs behavior"
    if np.isfinite(r_val):
        ttl += f"\nPearson r={r_val:.3f} (n={int(np.sum(m))} sessions)"
    ax1.set_title(ttl)

    for ax_ in (ax0, ax1):
        ax_.spines["top"].set_visible(False)
        ax_.spines["right"].set_visible(False)

    fig.tight_layout()

    # ---------- save ----------
    stem = f"{subject}_session_even_odd_similarity_vs_performance"
    paths = {
        "out_dir": str(out_dir),
        "df_csv": str(out_dir / f"{stem}.csv"),
        "df_parquet": str(out_dir / f"{stem}.parquet"),
        "plot_png": str(out_dir / f"{stem}.png"),
        "plot_svg": str(out_dir / f"{stem}.svg"),
        "n_sessions": int(df.shape[0]),
        "n_failures": int(len(failures)),
        "failures": failures,
    }

    if save_outputs:
        df.to_csv(paths["df_csv"], index=False)
        try:
            df.to_parquet(paths["df_parquet"], index=False)
        except Exception:
            # optional dependency might be missing; CSV is always saved
            pass

        fig.savefig(paths["plot_png"], dpi=180, bbox_inches="tight")
        fig.savefig(paths["plot_svg"], bbox_inches="tight")

        if verbose:
            print(f"\n[saved] dataframe: {paths['df_csv']}")
            print(f"[saved] plot: {paths['plot_png']}")
            if failures:
                print(f"[note] skipped {len(failures)} sessions")

    return df, fig, paths


#################
'''
stack all sessions (or single)
'''
#################




def run_all(eids):
    """
    Run save_trial_cuts_meso() for all eids in canonical sessions.
    """
     
    if eids is None:
        a = _canonical_sessions_subject_map(one)
        eids = np.concatenate([a[k] for k in a])

    print(f"Processing {len(eids)} eids...")

    for i, eid in enumerate(eids):
        print(f"\n=== [{i+1}/{len(eids)}] eid: {eid} ===")
        try:
            save_trial_cuts_meso(eid, filter_neurons=True)
        except Exception as e:
            print(f"[error] eid={eid}: {type(e).__name__}: {e}")




def stack_trial_cuts_meso(
    trial_cuts_dir: str | Path | None = None,
    filter_neurons: bool = True,
    min_trials: int = 10,
    single_eid: str | list[str] | None = None,  # NEW
):
    """
    Build stacked ROI feature matrices from per-EID trial-cut PETH files.

    Changes vs previous version
    ---------------------------
    - single_eid:
        * None / [] -> process all eids
        * str or [str] -> process only that eid and include eid in output filenames
    - Removed all cut_to_min / length-matching logic.
    - For each PETH type (ttype), after averaging across trials (all/even/odd),
      average across the time axis so each ttype contributes exactly 1 value per ROI.
      Resulting feature length L == number of ttypes.

    NEW POLICY
    ----------
    - EID-level min_trials enforcement:
        If *any* PETH type for an EID has M < min_trials in *any* FOV/meta entry,
        skip the whole EID (all its entries).
        If single_eid is set, print the violating PETH(s) and return early (no files saved).
    """
    start_time = time.time()

    # ---------------- paths ----------------
    if trial_cuts_dir is None:
        trial_cuts_dir = Path(pth_meso, "trial_cuts")
    else:
        trial_cuts_dir = Path(trial_cuts_dir)

    if not trial_cuts_dir.is_dir():
        raise FileNotFoundError(f"trial_cuts_dir not found: {trial_cuts_dir}")

    out_dir = trial_cuts_dir.parent

    # normalize single_eid
    if single_eid is None:
        single_eids: list[str] = []
    elif isinstance(single_eid, str):
        single_eids = [single_eid]
    else:
        single_eids = list(single_eid)

    single_mode = (len(single_eids) == 1)
    # tr_nums only has a clear meaning for a single session (single eid)
    allow_tr_nums = single_mode

    eid_tag = f"_{single_eids[0]}" if single_mode else ""
    out_all = out_dir / f"stack_all{eid_tag}_filter{filter_neurons}.npy"
    out_even = out_dir / f"stack_even{eid_tag}_filter{filter_neurons}.npy"
    out_odd = out_dir / f"stack_odd{eid_tag}_filter{filter_neurons}.npy"

    # ---------------- helpers ----------------
    def _zscore_rows(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        return (X - mu) / sd

    def _avg_trials_mode(X_mnt: np.ndarray, mode: str) -> np.ndarray | None:
        """
        X_mnt: (M, N, T)
        returns: (N, T) mean across selected trials, or None if insufficient trials
        """
        if X_mnt.ndim != 3:
            raise ValueError(f"Expected (M,N,T), got {X_mnt.shape}")
        M = int(X_mnt.shape[0])
        if M < int(min_trials):
            return None
        if mode == "all":
            sel = slice(None)
        elif mode == "even":
            sel = slice(0, None, 2)
        elif mode == "odd":
            sel = slice(1, None, 2)
        else:
            raise ValueError(mode)

        Xs = X_mnt[sel, :, :]
        if Xs.shape[0] == 0:
            return None
        return np.nanmean(Xs, axis=0).astype(np.float32, copy=False)  # (N,T)

    def _load_meta(p: Path) -> dict:
        obj = np.load(p, allow_pickle=True)
        if isinstance(obj, np.ndarray):
            if obj.shape == ():
                return obj.item()
            return obj.flat[0]
        return obj

    def _reduce_time_to_scalar(A: np.ndarray) -> np.ndarray | None:
        """
        A: (N, T)
        returns: (N, 1) scalar per ROI (mean across time); None if T < 1
        """
        if A.ndim != 2:
            raise ValueError(f"Expected (N,T), got {A.shape}")
        if A.shape[1] < 1:
            return None
        return np.nanmean(A, axis=1, keepdims=True).astype(np.float32, copy=False)  # (N,1)

    # ---------------- discover EIDs ----------------
    eid_dirs = sorted([p for p in trial_cuts_dir.iterdir() if p.is_dir()])
    if not eid_dirs:
        raise RuntimeError(f"No eid subfolders found in {trial_cuts_dir}")

    if single_mode:
        eid_dirs = [p for p in eid_dirs if p.name == single_eids[0]]
        if not eid_dirs:
            raise RuntimeError(f"single_eid={single_eids[0]!r} not found in {trial_cuts_dir}")

    # ---------------- pass 1: find valid entries ----------------
    ref_ttypes = None
    entries: list[tuple[str, str, Path]] = []  # (eid, fov, meta_path)

    for eid_path in eid_dirs:
        eid = eid_path.name
        metas = sorted(eid_path.glob("*meta_filter_*.npy"))

        for mp in metas:
            try:
                meta = _load_meta(mp)
            except Exception as e:
                print(f"[skip] {eid}/{mp.name}: cannot load meta ({type(e).__name__}: {e})")
                continue

            if bool(meta.get("filter_neurons", None)) != bool(filter_neurons):
                continue

            ttypes = list(meta.get("trial_names", []))
            if not ttypes:
                print(f"[skip] {eid}/{mp.name}: missing trial_names")
                continue

            if ref_ttypes is None:
                ref_ttypes = ttypes
            else:
                if ttypes != ref_ttypes:
                    print(f"[skip] {eid}/{mp.name}: ttypes mismatch vs reference")
                    continue

            if "peth_files" not in meta:
                print(f"[skip] {eid}/{mp.name}: missing peth_files")
                continue

            ok = True
            for t in ref_ttypes:
                if t not in meta["peth_files"]:
                    print(f"[skip] {eid}/{mp.name}: missing peth_files entry for {t}")
                    ok = False
                    break
            if not ok:
                continue

            entries.append((eid, meta.get("fov", ""), mp))

    if ref_ttypes is None or not entries:
        raise RuntimeError(
            f"No valid meta files found for filter_neurons={filter_neurons} in {trial_cuts_dir}."
        )

    L = int(len(ref_ttypes))
    print(f"[info] combining {len(entries)} FOV entries; L={L} (1 scalar per {len(ref_ttypes)} ttypes)")

    # ---------------- group entries by eid ----------------
    from collections import defaultdict
    entries_by_eid: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
    for eid, fov, mp in entries:
        entries_by_eid[eid].append((eid, fov, mp))

    # ---------------- pass 2: EID-level min_trials screening, then build stacks ----------------
    blocks_all, blocks_even, blocks_odd = [], [], []
    ids_blocks, xyz_blocks, eid_blocks, fov_blocks = [], [], [], []

    # session-level trial counts (only meaningful in single_mode)
    tr_nums_all = {t: None for t in ref_ttypes}
    tr_nums_even = {t: None for t in ref_ttypes}
    tr_nums_odd = {t: None for t in ref_ttypes}

    for eid, eid_entries in entries_by_eid.items():
        # --------- EID-level min_trials check (fail => skip entire eid) ---------
        viol: dict[str, int] = {}  # ttype -> minimal M observed (for reporting)
        for _eid, fov, mp in eid_entries:
            meta = _load_meta(mp)
            eid_path = trial_cuts_dir / eid
            for t in ref_ttypes:
                fp = eid_path / meta["peth_files"][t]
                try:
                    X = np.load(fp, mmap_mode="r")
                except Exception:
                    # loading failures already handled later; here just mark as violation-like
                    viol[t] = min(viol.get(t, 10**9), -1)
                    continue
                if X.ndim != 3:
                    viol[t] = min(viol.get(t, 10**9), -1)
                    continue
                M = int(X.shape[0])
                if M < int(min_trials):
                    viol[t] = min(viol.get(t, 10**9), M)
                del X

        if viol:
            # If single eid requested, report and return empty (no saving)
            if single_mode:
                for t, m in sorted(viol.items()):
                    if m >= 0:
                        print(f"[min_trials fail] eid={eid} ttype={t}: M={m} < min_trials={min_trials}")
                    else:
                        print(f"[min_trials fail] eid={eid} ttype={t}: cannot load/invalid shape")
                return

            # Otherwise skip this eid entirely
            bad_list = ", ".join(
                f"{t}(M={m})" if m >= 0 else f"{t}(invalid)"
                for t, m in sorted(viol.items())
            )
            print(f"[skip eid] {eid}: min_trials violated for {bad_list}")
            continue

        # --------- build stacks for this eid (all its FOV/meta entries) ---------
        for _eid, fov, mp in eid_entries:
            eid_path = trial_cuts_dir / eid
            meta = _load_meta(mp)

            ids = np.asarray(meta.get("region_labels", []), dtype=object)
            xyz = np.asarray(meta.get("xyz", []), dtype=np.float32)

            if ids.size == 0 or xyz.size == 0:
                print(f"[skip] {eid}/{fov}: missing region_labels or xyz")
                continue
            if xyz.ndim != 2 or xyz.shape[1] < 3:
                print(f"[skip] {eid}/{fov}: xyz has unexpected shape {xyz.shape}")
                continue
            xyz = xyz[:, :3]

            segs_all, segs_even, segs_odd = [], [], []
            ok = True

            for t in ref_ttypes:
                fp = eid_path / meta["peth_files"][t]
                try:
                    X = np.load(fp, mmap_mode="r")  # (M,N,T)
                except Exception as e:
                    print(f"[skip] {eid}/{fov}: cannot load {t} ({fp.name}) ({type(e).__name__}: {e})")
                    ok = False
                    break

                if X.ndim != 3:
                    print(f"[skip] {eid}/{fov}: {t} has unexpected ndim {X.ndim} shape {X.shape}")
                    ok = False
                    break

                if X.shape[1] != ids.shape[0]:
                    print(f"[skip] {eid}/{fov}: N mismatch for {t}: {X.shape[1]} != {ids.shape[0]}")
                    ok = False
                    break

                # session-level trial counts: set once (they are consistent by the EID-level screen)
                if allow_tr_nums:
                    M = int(X.shape[0])
                    n_all = M
                    n_even = (M + 1) // 2
                    n_odd = M // 2
                    if tr_nums_all[t] is None:
                        tr_nums_all[t] = n_all
                        tr_nums_even[t] = n_even
                        tr_nums_odd[t] = n_odd

                A_all = _avg_trials_mode(X, "all")
                A_even = _avg_trials_mode(X, "even")
                A_odd = _avg_trials_mode(X, "odd")

                # This should never happen now (we pre-screened), but keep as a guard.
                if A_all is None or A_even is None or A_odd is None:
                    print(f"[skip] {eid}/{fov}: insufficient trials for {t} (min_trials={min_trials}) [unexpected after prescreen]")
                    ok = False
                    break

                S_all = _reduce_time_to_scalar(A_all)
                S_even = _reduce_time_to_scalar(A_even)
                S_odd = _reduce_time_to_scalar(A_odd)

                if S_all is None or S_even is None or S_odd is None:
                    print(f"[skip] {eid}/{fov}: {t} has T<1 after averaging (unexpected)")
                    ok = False
                    break

                segs_all.append(S_all)   # (N,1)
                segs_even.append(S_even)
                segs_odd.append(S_odd)

                del X

            if not ok:
                continue

            P_all = np.concatenate(segs_all, axis=1).astype(np.float32, copy=False)   # (N,L)
            P_even = np.concatenate(segs_even, axis=1).astype(np.float32, copy=False)
            P_odd = np.concatenate(segs_odd, axis=1).astype(np.float32, copy=False)

            blocks_all.append(P_all)
            blocks_even.append(P_even)
            blocks_odd.append(P_odd)

            ids_blocks.append(ids)
            xyz_blocks.append(xyz)
            eid_blocks.append(np.array([eid] * ids.shape[0], dtype=object))
            fov_blocks.append(np.array([fov] * ids.shape[0], dtype=object))

            del segs_all, segs_even, segs_odd, P_all, P_even, P_odd
            gc.collect()

    if not blocks_all:
        raise RuntimeError("No FOV blocks aggregated (all were skipped).")

    concat_all = np.concatenate(blocks_all, axis=0)
    concat_even = np.concatenate(blocks_even, axis=0)
    concat_odd = np.concatenate(blocks_odd, axis=0)

    ids_all = np.concatenate(ids_blocks, axis=0)
    xyz_all = np.concatenate(xyz_blocks, axis=0)
    eid_all = np.concatenate(eid_blocks, axis=0)
    fov_all = np.concatenate(fov_blocks, axis=0)

    good = (
        (~np.isnan(concat_all).any(axis=1))
        & (~np.isnan(concat_even).any(axis=1))
        & (~np.isnan(concat_odd).any(axis=1))
        & np.any(concat_all, axis=1)
    )

    ids_all = ids_all[good]
    xyz_all = xyz_all[good]
    eid_all = eid_all[good]
    fov_all = fov_all[good]

    concat_all = concat_all[good]
    concat_even = concat_even[good]
    concat_odd = concat_odd[good]

    print(f"[info] merged: {concat_all.shape[0]} ROIs kept; feature length {concat_all.shape[1]}")

    def _finalize_stack(concat: np.ndarray) -> dict:
        r = {}
        r["ids"] = ids_all
        r["xyz"] = xyz_all.astype(np.float32, copy=False)
        r["eid"] = eid_all
        r["FOV"] = fov_all
        r["ttypes"] = list(ref_ttypes)
        r["len"] = {t: 1 for t in ref_ttypes}  # 1 scalar per ttype

        r["concat"] = concat.astype(np.float32, copy=False)
        r["fr"] = np.array([np.mean(x) for x in r["concat"]], dtype=np.float32)
        r["concat_z"] = _zscore_rows(r["concat"])

        rng = np.random.default_rng(0)
        N = r["concat_z"].shape[0]
        lz_vals = np.zeros(N, float)
        for i in range(N):
            lz_vals[i] = lzs_pci(r["concat_z"][i], rng)
        r["lz"] = lz_vals

        if allow_tr_nums:
            r["tr_nums"] = {
                "all": {k: int(v) for k, v in tr_nums_all.items() if v is not None},
                "even": {k: int(v) for k, v in tr_nums_even.items() if v is not None},
                "odd": {k: int(v) for k, v in tr_nums_odd.items() if v is not None},
            }

        return r

    r_all = _finalize_stack(concat_all)
    r_even = _finalize_stack(concat_even)
    r_odd = _finalize_stack(concat_odd)

    np.save(out_all, r_all, allow_pickle=True)
    np.save(out_even, r_even, allow_pickle=True)
    np.save(out_odd, r_odd, allow_pickle=True)

    print(f"saved: {out_all}")
    print(f"saved: {out_even}")
    print(f"saved: {out_odd}")
    print(f"Function 'stack_trial_cuts_meso' executed in: {time.time() - start_time:.4f} s")




def _load_dict(p: Path) -> dict:
    obj = np.load(p, allow_pickle=True)
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            return obj.item()
        return obj.flat[0]
    return obj


def regional_group_meso(
    mapping: str,
    *,
    stack_dir: str | Path | None = None,
    filter_neurons: bool = True,
    cv: bool = False,
    nclus: int = 100,                 # KMeans n_clusters (ONLY affects mapping='kmeans')
    nclus_rm: int = 100,              # Rastermap n_clusters (RM cache + isort for ALL mappings)
    grid_upsample: int = 0,
    locality: float = 0.75,
    time_lag_window: int = 5,
    symmetric: bool = False,
    rerun: bool = False,
    cache_dir: str | Path | None = None,
    single_eid: str | list[str] | None = None,   # NEW
):
    """
    Mesoscope analogue of `regional_group(...)`, using stack files created by `stack_trial_cuts_meso()`.

    Mapping : one of ['Beryl','Cosmos','rm','lz','fr','cv_sim','kmeans','PCA'].

    Updates
    -------
    - single_eid:
        * None / [] -> load standard stack_all/stack_even/stack_odd
        * str or [str] (length 1) -> load stack_*_{eid}_filter*.npy
    - If cv=True:
        * attach r['concat_z_even'] and r['concat_z_odd'] as before
        * ALSO save concat_z_even into RM and KMeans caches (do not save non-zscored 'concat')
    """
    mapping = str(mapping)
    nclus = int(nclus)
    nclus_rm = int(nclus_rm)

    # ---------- paths ----------
    if stack_dir is None:
        try:
            stack_dir = Path(pth_meso)
        except NameError:
            raise NameError("stack_dir is None and pth_meso is not in scope. Pass stack_dir explicitly.")
    else:
        stack_dir = Path(stack_dir)

    if cache_dir is None:
        cache_dir = stack_dir / "res"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # normalize single_eid
    if single_eid is None:
        single_eids: list[str] = []
    elif isinstance(single_eid, str):
        single_eids = [single_eid]
    else:
        single_eids = list(single_eid)

    if len(single_eids) > 1:
        raise ValueError("regional_group_meso: single_eid must be None/[] or a single eid (str or [str]).")

    eid_tag = f"_{single_eids[0]}" if len(single_eids) == 1 else ""

    stack_all_path = stack_dir / f"stack_all{eid_tag}_filter{bool(filter_neurons)}.npy"
    stack_even_path = stack_dir / f"stack_even{eid_tag}_filter{bool(filter_neurons)}.npy"
    stack_odd_path = stack_dir / f"stack_odd{eid_tag}_filter{bool(filter_neurons)}.npy"

    if not stack_all_path.is_file():
        raise FileNotFoundError(f"Missing meso stack: {stack_all_path}")

    r = _load_dict(stack_all_path)

    # enforce ordered 'len'
    if "ttypes" in r and "len" in r:
        r["len"] = OrderedDict((k, int(r["len"][k])) for k in r["ttypes"])

    # ---------- attach cv blocks ----------
    _need_cv_blocks = bool(cv) or (mapping == "cv_sim")

    if _need_cv_blocks:
        if not stack_even_path.is_file() or not stack_odd_path.is_file():
            raise FileNotFoundError(
                f"Need even/odd stacks (cv=True or mapping='cv_sim') but missing:\n"
                f"  {stack_even_path}\n  {stack_odd_path}"
            )
        r_even = _load_dict(stack_even_path)
        r_odd = _load_dict(stack_odd_path)

        # stacks may no longer contain 'concat' (new policy), so use concat_z if present, else fall back to concat
        def _get_feat_for_cv(d: dict) -> np.ndarray:
            if "concat_z" in d:
                return np.asarray(d["concat_z"], dtype=np.float32)
            if "concat" in d:
                X = np.asarray(d["concat"], dtype=np.float32)
                mu = np.nanmean(X, axis=1, keepdims=True)
                sd = np.nanstd(X, axis=1, keepdims=True)
                sd = np.where(sd == 0, 1.0, sd)
                return (X - mu) / sd
            raise KeyError("cv stacks must contain 'concat_z' (preferred) or legacy 'concat'.")

        # ensure all-stack feature exists (used for N consistency checks)
        if "concat_z" in r:
            X_all = np.asarray(r["concat_z"], dtype=np.float32)
        elif "concat" in r:
            X = np.asarray(r["concat"], dtype=np.float32)
            mu = np.nanmean(X, axis=1, keepdims=True)
            sd = np.nanstd(X, axis=1, keepdims=True)
            sd = np.where(sd == 0, 1.0, sd)
            X_all = (X - mu) / sd
            r["concat_z"] = X_all
        else:
            raise KeyError("Saved all-stack lacks 'concat_z' (preferred) or legacy 'concat'.")

        X_even = _get_feat_for_cv(r_even)
        X_odd = _get_feat_for_cv(r_odd)

        N = int(X_all.shape[0])
        if int(X_even.shape[0]) != N or int(X_odd.shape[0]) != N:
            raise ValueError("even/odd stacks do not match N of all-stack (ordering mismatch).")

        r["concat_z_even"] = X_even
        r["concat_z_odd"] = X_odd

    # ---------- common bookkeeping ----------
    if "xyz" not in r:
        raise KeyError("Saved stack lacks 'xyz'.")

    feat_key = "concat_z"
    if feat_key not in r:
        # allow legacy stacks
        if "concat" in r:
            X = np.asarray(r["concat"], dtype=np.float32)
            mu = np.nanmean(X, axis=1, keepdims=True)
            sd = np.nanstd(X, axis=1, keepdims=True)
            sd = np.where(sd == 0, 1.0, sd)
            r["concat_z"] = (X - mu) / sd
        else:
            raise KeyError(f"Saved stack lacks '{feat_key}' (or legacy 'concat').")

    n_rows = int(np.asarray(r[feat_key]).shape[0])

    r["_order_signature"] = (
        "|".join(f"{k}:{int(r['len'][k])}" for k in r.get("ttypes", []))
        + f"|shape:{tuple(np.asarray(r[feat_key]).shape)}"
    )

    # ---------- cache paths ----------
    def _cache_path(kind: str) -> Path:
        # include single-eid tag in cache names so per-eid caches do not collide
        eid_part = f"_eid{single_eids[0]}" if len(single_eids) == 1 else ""

        if kind == "rm":
            base = f"meso_rm{eid_part}_filter{bool(filter_neurons)}_cv{bool(cv)}_nclusrm{nclus_rm}"
            return cache_dir / (base + ".npy")

        if kind == "kmeans":
            base = f"meso_kmeans{eid_part}_filter{bool(filter_neurons)}_cv{bool(cv)}_n{nclus}_nclusrm{nclus_rm}"
            return cache_dir / (base + ".npy")

        if kind == "pca":
            base = f"meso_pca{eid_part}_filter{bool(filter_neurons)}_cv{bool(cv)}"
            return cache_dir / (base + ".npy")

        raise ValueError(kind)

    # ---------- helpers ----------
    def _try_load_dict(p: Path) -> dict | None:
        if rerun or (not p.is_file()):
            return None
        try:
            d = np.load(p, allow_pickle=True).flat[0]
            return d if isinstance(d, dict) else None
        except Exception:
            return None

    def _tab20_color_map_for_labels(labels: np.ndarray) -> tuple[np.ndarray, dict]:
        labels = np.asarray(labels, dtype=int).reshape(-1)
        unique_sorted = np.sort(np.unique(labels))
        cmap = mpl.colormaps["tab20"]
        u_to_idx = {u: (i % 20) for i, u in enumerate(unique_sorted)}
        color_map = {u: cmap(u_to_idx[u]) for u in unique_sorted}
        cols = np.array([color_map[int(c)] for c in labels], dtype=np.float32)
        return cols, color_map

    def _load_rm_cache(p: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
        cached = _try_load_dict(p)
        if cached is None:
            return None, None
        try:
            if not (
                cached.get("order_sig") == r["_order_signature"]
                and cached.get("nclus_rm") == nclus_rm
                and "rm_labels" in cached
                and "isort" in cached
            ):
                return None, None

            labels = np.asarray(cached["rm_labels"], dtype=int).reshape(-1)
            isort = np.asarray(cached["isort"], dtype=int).reshape(-1)
            if labels.shape[0] != n_rows or isort.shape[0] != n_rows:
                return None, None
            return labels, isort
        except Exception:
            return None, None

    def _compute_and_save_rm(p: Path) -> tuple[np.ndarray, np.ndarray]:
        feat_fit = "concat_z_even" if cv else feat_key
        if feat_fit not in r:
            raise KeyError(f"Feature '{feat_fit}' missing; need it for Rastermap fit.")

        model = Rastermap(
            n_PCs=200,
            n_clusters=nclus_rm,
            grid_upsample=grid_upsample,
            locality=locality,
            time_lag_window=time_lag_window,
            bin_size=1,
            symmetric=symmetric,
        ).fit(np.asarray(r[feat_fit]))

        labels = np.asarray(model.embedding_clust, dtype=int)
        if labels.ndim > 1:
            labels = labels[:, 0]
        labels = labels.reshape(-1)

        isort = np.asarray(model.isort, dtype=int).reshape(-1)

        if labels.shape[0] != n_rows or isort.shape[0] != n_rows:
            raise ValueError("Rastermap outputs do not match all-stack length.")

        payload = {
            "rm_labels": labels,
            "isort": isort,
            "order_sig": r["_order_signature"],
            "nclus_rm": nclus_rm,
        }
        # NEW: if cv=True, save concat_z_even into cache (but do not save any non-zscored 'concat')
        if cv:
            payload["concat_z_even"] = np.asarray(r["concat_z_even"], dtype=np.float32)

        np.save(p, payload, allow_pickle=True)
        return labels, isort

    def _ensure_rm_isort_attached() -> None:
        """Always attach r['isort'] from RM cache; compute+save if missing/invalid."""
        rm_cache_path = _cache_path("rm")
        labels_rm, isort_rm = _load_rm_cache(rm_cache_path)
        if labels_rm is None or isort_rm is None:
            labels_rm, isort_rm = _compute_and_save_rm(rm_cache_path)
        r["isort"] = isort_rm  # do not overwrite r['acs']/r['cols'] here

    # ---------- mapping branches ----------
    if mapping == "rm":
        rm_cache_path = _cache_path("rm")
        labels, isort = _load_rm_cache(rm_cache_path)
        if labels is None or isort is None:
            labels, isort = _compute_and_save_rm(rm_cache_path)

        cols, color_map = _tab20_color_map_for_labels(labels)
        r["els"] = [Line2D([0], [0], color=color_map[u], lw=4, label=f"{u}") for u in np.sort(np.unique(labels))]
        r["acs"] = labels
        r["cols"] = cols
        r["isort"] = isort

    elif mapping == "kmeans":
        km_cache_path = _cache_path("kmeans")
        cached = _try_load_dict(km_cache_path)

        feat_fit = "concat_z_even" if cv else feat_key
        if feat_fit not in r:
            raise KeyError(f"Feature '{feat_fit}' missing; need it for kmeans fit.")

        clusters = None
        if cached is not None:
            try:
                if (
                    cached.get("order_sig") == r["_order_signature"]
                    and cached.get("feat_fit") == feat_fit
                    and cached.get("nclus") == nclus
                    and "kmeans_labels" in cached
                ):
                    clusters = np.asarray(cached["kmeans_labels"], dtype=int).reshape(-1)
                    if clusters.shape[0] != n_rows:
                        clusters = None
            except Exception:
                clusters = None

        if clusters is None:
            km = KMeans(n_clusters=nclus, random_state=0)
            km.fit(np.asarray(r[feat_fit]))
            clusters = km.predict(np.asarray(r[feat_key])).astype(int).reshape(-1)

            if clusters.shape[0] != n_rows:
                raise ValueError("KMeans labels do not match all-stack length.")

            payload = {
                "kmeans_labels": clusters,
                "order_sig": r["_order_signature"],
                "feat_fit": feat_fit,
                "nclus": nclus,
            }
            # NEW: if cv=True, save concat_z_even into cache (but do not save any non-zscored 'concat')
            if cv:
                payload["concat_z_even"] = np.asarray(r["concat_z_even"], dtype=np.float32)

            np.save(km_cache_path, payload, allow_pickle=True)

        cols, color_map = _tab20_color_map_for_labels(clusters)
        r["els"] = [Line2D([0], [0], color=color_map[u], lw=4, label=f"{u + 1}") for u in np.sort(np.unique(clusters))]
        r["acs"] = clusters
        r["cols"] = cols

        _ensure_rm_isort_attached()

    elif mapping == "PCA":
        from sklearn.decomposition import PCA

        pca_cache_path = _cache_path("pca")
        cached = _try_load_dict(pca_cache_path)

        X = np.asarray(r[feat_key], dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"{feat_key} must be 2D (cells × features); got {X.shape}.")

        scores = None
        evr = None
        if cached is not None:
            try:
                if cached.get("order_sig") == r["_order_signature"] and cached.get("feat_key") == feat_key and "scores" in cached:
                    scores = np.asarray(cached["scores"], dtype=np.float32)
                    if scores.shape != (n_rows, 3):
                        scores = None
                    evr = cached.get("explained_variance_ratio", None)
            except Exception:
                scores = None
                evr = None

        if scores is None:
            pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
            scores = pca.fit_transform(X).astype(np.float32, copy=False)
            evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float32)

            np.save(
                pca_cache_path,
                {
                    "scores": scores,
                    "order_sig": r["_order_signature"],
                    "feat_key": feat_key,
                    "explained_variance_ratio": evr,
                },
                allow_pickle=True,
            )

        lo = np.nanpercentile(scores, 1.0, axis=0)
        hi = np.nanpercentile(scores, 99.0, axis=0)
        denom = np.where((hi - lo) > 0, (hi - lo), 1.0)
        rgb = np.clip((scores - lo) / denom, 0.0, 1.0).astype(np.float32, copy=False)
        rgba = np.concatenate([rgb, np.ones((n_rows, 1), dtype=np.float32)], axis=1)

        r["acs"] = np.zeros((n_rows,), dtype=int)
        r["cols"] = rgba
        r["els"] = []
        r["pca3"] = scores
        r["pca_explained_variance_ratio"] = evr

        _ensure_rm_isort_attached()

    elif mapping == "cv_sim":
        # Requires even/odd features; ensured above via _need_cv_blocks
        if ("concat_z_even" not in r) or ("concat_z_odd" not in r):
            raise KeyError("mapping='cv_sim' requires r['concat_z_even'] and r['concat_z_odd'].")

        Xe = np.asarray(r["concat_z_even"], dtype=np.float32)
        Xo = np.asarray(r["concat_z_odd"], dtype=np.float32)

        if Xe.shape != Xo.shape or Xe.ndim != 2:
            raise ValueError(f"concat_z_even/odd must be same 2D shape; got {Xe.shape} vs {Xo.shape}.")

        # cosine similarity per neuron (row)
        # sim_i = <Xe_i, Xo_i> / (||Xe_i|| * ||Xo_i||)
        num = np.nansum(Xe * Xo, axis=1)
        den = np.sqrt(np.nansum(Xe * Xe, axis=1) * np.nansum(Xo * Xo, axis=1))
        den = np.where(den == 0, np.nan, den)
        sim = (num / den).astype(np.float32)

        # store like 'fr'/'lz'
        r["cv_sim"] = sim
        r["acs"] = np.asarray(r.get("ids", []), dtype=object)

        # colorize (cosine similarity nominally in [-1, 1])
        norm = Normalize(vmin=-1.0, vmax=1.0)
        cmap = cm.get_cmap("magma")
        r["cols"] = cmap(norm(sim))

        _ensure_rm_isort_attached()


    elif mapping == "fr":
        if "fr" not in r:
            raise KeyError("Saved stack lacks 'fr'.")
        scaled = np.asarray(r["fr"], dtype=float) ** 0.1
        norm = Normalize(vmin=float(np.nanmin(scaled)), vmax=float(np.nanmax(scaled)))
        cmap = cm.get_cmap("magma")
        r["acs"] = np.asarray(r.get("ids", []), dtype=object)
        r["cols"] = cmap(norm(scaled))

        _ensure_rm_isort_attached()

    elif mapping == "lz":
        if "lz" not in r:
            raise KeyError("Saved stack lacks 'lz'.")
        scaled = np.asarray(r["lz"], dtype=float) ** 0.1
        norm = Normalize(vmin=float(np.nanmin(scaled)), vmax=float(np.nanmax(scaled)))
        cmap = cm.get_cmap("cividis")
        r["acs"] = np.asarray(r.get("ids", []), dtype=object)
        r["cols"] = cmap(norm(scaled))

        _ensure_rm_isort_attached()

    elif mapping in ("Beryl", "Cosmos"):
        acs_in = np.asarray(r.get("ids", []), dtype=object)

        if br is not None:
            ids_num = br.acronym2id(acs_in)
            acs = np.asarray(br.id2acronym(ids_num, mapping=mapping), dtype=object)
        else:
            acs = acs_in

        if pal is None:
            raise ValueError("For mapping in {'Beryl','Cosmos'} you must have palette dict `pal` (acronym->color).")

        r["acs"] = acs
        r["cols"] = np.array([pal.get(a, (0.5, 0.5, 0.5, 1.0)) for a in acs], dtype=object)

        regsC = Counter(acs)
        r["els"] = [
            Line2D([0], [0], color=pal.get(reg, (0.5, 0.5, 0.5, 1.0)), lw=4, label=f"{reg} {regsC[reg]}")
            for reg in regsC
        ]

        _ensure_rm_isort_attached()

    else:
        raise ValueError("mapping must be one of ['Beryl','Cosmos','rm','lz','fr','cv_sim','kmeans','PCA'].")


    # Always attach Beryl acronyms as r['Beryl'] for convenience
    acs_in = np.asarray(r.get("ids", []), dtype=object)
    if br is not None:
        ids_num = br.acronym2id(acs_in)
        r["Beryl"] = np.asarray(br.id2acronym(ids_num, mapping="Beryl"), dtype=object)
    else:
        r["Beryl"] = acs_in

    # Final guarantee: r['isort'] exists even if a future mapping branch forgets to call _ensure_rm_isort_attached()
    if "isort" not in r:
        _ensure_rm_isort_attached()

    return r






######################################################
###########  plotting
######################################################


def _override_cols_tab20(r, *, key_labels: str = "acs", key_cols: str = "cols") -> None:
    """
    In-place override of r[key_cols] with repeating tab20 colors keyed by r[key_labels] categories.
    Produces RGBA float32 in [0,1].
    """
    if key_labels not in r:
        raise KeyError(f"r lacks '{key_labels}'")
    labels = np.asarray(r[key_labels], dtype=object)

    # stable category order by first occurrence
    _, first_idx = np.unique(labels, return_index=True)
    cats = labels[np.sort(first_idx)]

    cmap = mpl.colormaps["tab20"]
    cat2rgba = {cat: np.array(cmap(i % 20), dtype=np.float32) for i, cat in enumerate(cats)}
    r[key_cols] = np.stack([cat2rgba[l] for l in labels], axis=0).astype(np.float32, copy=False)


def scatter_x_vs_cv_sim_meso(
    *,
    stack_dir=None,
    filter_neurons: bool = True,
    mapping: str = "Beryl",
    cv: bool = True,
    nclus: int = 20,
    nclus_rm: int = 20,
    rerun: bool = False,
    single_eid: str | None = None, # '20ebc2b9-5b4c-42cd-8e4b-65ddb427b7ff'
    xvar: str = "fr",
    jitter: float = 0.15,
    shuf: bool = True,
    shuf_seed: int = 0,
    ax=None,
):
    """
    Scatter plot:
      y = per-neuron CV similarity = 0.5*(cosine(concat_z_even, concat_z_odd) + 1)

    If shuf=True:
      - compute per-neuron shuffle control by independently permuting each row of X_odd
      - plot shuffle points in grey, real points in black (same x)
    """
    # --- load grouped data ---
    r = regional_group_meso(
        mapping=mapping,
        stack_dir=stack_dir,
        filter_neurons=filter_neurons,
        cv=cv,
        nclus=nclus,
        nclus_rm=nclus_rm,
        rerun=rerun,
        single_eid=single_eid,
    )

    # --- required keys ---
    for k in ("concat_z_even", "concat_z_odd"):
        if k not in r:
            raise KeyError(f"Required key '{k}' missing from result dict.")
    if xvar in ("fr", "lz") and xvar not in r:
        raise KeyError(f"xvar='{xvar}' requires r['{xvar}'].")

    # --- x variable ---
    xticks = None
    xticklabels = None

    if xvar == "fr":
        x = np.asarray(r["fr"], float)
        xlabel = "firing rate (fr)"

    elif xvar == "lz":
        x = np.asarray(r["lz"], float)
        xlabel = "LZ / PCI complexity"

    elif xvar == "acs":
        if "acs" not in r:
            raise KeyError("xvar='acs' requires r['acs'].")

        acs_arr = np.asarray(r["acs"])

        # numeric acs → use directly
        if np.issubdtype(acs_arr.dtype, np.number):
            x = acs_arr.astype(float)
            xticks = np.unique(x)
            xticklabels = [str(int(t)) for t in xticks]

        # categorical / string acs → canonical mapping
        else:
            uniq_sorted = np.sort(np.unique(acs_arr.astype(str)))
            acs_to_idx = {lab: i for i, lab in enumerate(uniq_sorted)}
            x = np.array([acs_to_idx[str(a)] for a in acs_arr], dtype=float)

            xticks = np.arange(len(uniq_sorted))
            xticklabels = uniq_sorted.tolist()

        # jitter x positions only (do not jitter tick positions)
        if jitter and jitter > 0:
            rng = np.random.default_rng(shuf_seed)
            x = x + rng.uniform(-jitter, jitter, size=x.shape)

        xlabel = "acs"

    else:
        raise ValueError(f"Unsupported xvar='{xvar}'")

    # --- matrices ---
    X_even = np.asarray(r["concat_z_even"], float)
    X_odd  = np.asarray(r["concat_z_odd"], float)
    if X_even.shape != X_odd.shape:
        raise ValueError(f"even/odd shape mismatch: {X_even.shape} vs {X_odd.shape}")

    # --- cosine similarity mapped to [0,1] ---
    def _sim01(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        num = np.sum(A * B, axis=1)
        den = np.sqrt(np.sum(A**2, axis=1) * np.sum(B**2, axis=1))
        out = np.full_like(num, np.nan, float)
        good = den > 0
        out[good] = num[good] / den[good]
        return 0.5 * (out + 1.0)

    sim = _sim01(X_even, X_odd)
    mean_sim = float(np.nanmean(sim))

    # --- shuffle control: independently permute each row of X_odd ---
    sim_shuf = None
    mean_sim_shuf = None
    if shuf:
        rng = np.random.default_rng(shuf_seed)

        # per-row random permutation using random keys + argsort
        keys = rng.random(X_odd.shape)
        perm = np.argsort(keys, axis=1)
        X_odd_shuf = np.take_along_axis(X_odd, perm, axis=1)

        sim_shuf = _sim01(X_even, X_odd_shuf)
        mean_sim_shuf = float(np.nanmean(sim_shuf))

    # --- plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4))
    else:
        fig = ax.figure

    # REAL data first (black, underneath)
    ax.scatter(
        x,
        sim,
        s=6,
        alpha=0.55,
        linewidths=0,
        color="k",
        zorder=1,
        label="real" if shuf else None,
    )

    # SHUFFLE on top (grey)
    if shuf and sim_shuf is not None:
        ax.scatter(
            x,
            sim_shuf,
            s=6,
            alpha=0.25,
            linewidths=0,
            color="0.6",
            zorder=2,
            label="shuffled",
        )

    if xvar == "acs" and xticks is not None and xticklabels is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=90)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CV similarity (0.5·(cos+1))")
    ax.set_ylim(-0.05, 1.05)

    # reference: cos=0 -> sim=0.5
    ax.axhline(0.5, color="k", lw=0.8, alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if shuf:
        ax.legend(frameon=False)

    # --- title ---
    title = f"CV similarity vs {xvar} | mean={mean_sim:.3f}"
    if shuf and mean_sim_shuf is not None:
        title += f"\nshuf={mean_sim_shuf:.3f} | Δ={mean_sim - mean_sim_shuf:+.3f}"
    if single_eid is not None:
        title += f" | eid={single_eid[:3]}..."
    ax.set_title(title)

    plt.tight_layout()





def plot_rastermap_meso(
    *,
    stack_dir: str | Path | None = None,
    filter_neurons: bool = True,
    feat: str = "concat_z",
    mapping: str = "rm",
    cv: bool = True,
    sort_method: str = "rastermap",   # 'rastermap' or 'acs'
    nclus: int = 100,
    nclus_rm: int = 20,
    rerun: bool = False,
    bounds: bool = True,
    bg: bool = False,
    bg_bright: float = 0.99,
    vmax: float = 2.0,
    interp: str = "none",             # default for single values per PETH
    img_only: bool = False,
    clabels: str | int = "all",
    fps: float = 8,
    out_dir: str | Path | None = None,
    fname_prefix: str = "meso_rastermap",
    grid_upsample: int = 0,
    locality: float = 0.75,
    time_lag_window: int = 5,
    symmetric: bool = False,
    clsfig: bool = False,
    peth_dict: dict = peth_dictm,
    exa: bool = False,
    single_eid: str | None = None,
):
    """
    Simplified mesoscope raster plot.

    If exa=True, also opens a new figure with plot_cluster_mean_PETHs_meso(r, ...).

    Descriptor (used for window title + filename if saving):
      single_eid, mapping, cv, sort_method, nclus, nclus_rm
    """
    # --- get grouped result dict ---
    r = regional_group_meso(
        mapping=mapping,
        stack_dir=stack_dir,
        filter_neurons=filter_neurons,
        cv=cv,
        nclus=nclus,
        nclus_rm=nclus_rm,
        grid_upsample=grid_upsample,
        locality=locality,
        time_lag_window=time_lag_window,
        symmetric=symmetric,
        rerun=rerun,
        single_eid=single_eid,
    )

    if exa:
        plt.ion()
        plot_cluster_mean_PETHs_meso(r, feat=feat, peth_dict=peth_dict)

    if feat not in r:
        raise KeyError(f"feat='{feat}' not in result dict. Available keys: {list(r.keys())[:30]}...")

    X = np.asarray(r[feat])
    if X.ndim != 2:
        raise ValueError(f"Expected r[{feat}] to be 2D, got shape {X.shape}")

    # --- descriptor ---
    nclus_rm_eff = int(nclus) if nclus_rm is None else int(nclus_rm)
    eid_desc = str(single_eid) if single_eid is not None else "all"
    descriptor = (
        f"eid={eid_desc[:3]}...|feat={feat}|mapping={mapping}|cv={cv}|sort={sort_method}|nclus={int(nclus)}|nclus_rm={nclus_rm_eff}"
    )

    # --- sorting ---
    if sort_method == "rastermap":
        if "isort" not in r:
            raise KeyError("sort_method='rastermap' requires r['isort'] (use mapping='rm' or 'kmeans').")
        isort = np.asarray(r["isort"], dtype=int).reshape(-1)

    elif sort_method == "acs":
        if "acs" not in r:
            raise KeyError("sort_method='acs' requires r['acs'].")
        acs_arr = np.asarray(r["acs"])
        try:
            isort = np.argsort(acs_arr, kind="stable")
        except TypeError:
            isort = np.argsort(acs_arr.astype(str), kind="stable")

    else:
        raise ValueError("sort_method must be 'rastermap' or 'acs'.")

    data = X[isort]
    row_colors = np.asarray(r.get("cols"))[isort] if "cols" in r else None
    clus_sorted = np.asarray(r.get("acs"))[isort] if "acs" in r else None

    n_rows, n_cols = data.shape

    # --- figure ---
    plt.ion()
    if clsfig:
        plt.ioff()

    fig, ax = plt.subplots(figsize=(6, 8))

    # clip + normalize to 0..1
    vmin = 0.0
    vmax = float(vmax)
    data_clipped = np.clip(data, vmin, vmax)
    gray_scaled = (data_clipped - vmin) / max(vmax - vmin, 1e-12)

    extent = (0, n_cols, n_rows, 0)  # origin='upper' in data coords

    if bg and (row_colors is not None):
        rgba_bg = np.array([to_rgba(c) for c in row_colors], dtype=np.float32)
        rgba_bg = np.broadcast_to(rgba_bg[:, None, :], (n_rows, n_cols, 4)).copy()
        rgba_bg[..., :3] = rgba_bg[..., :3] * bg_bright + (1.0 - bg_bright)

        alpha_overlay = gray_scaled
        rgba_bg[..., 0] *= (1.0 - alpha_overlay)
        rgba_bg[..., 1] *= (1.0 - alpha_overlay)
        rgba_bg[..., 2] *= (1.0 - alpha_overlay)
        rgba_bg[..., 3] = 1.0

        ax.imshow(
            rgba_bg,
            aspect="auto",
            interpolation=interp,
            extent=extent,
            origin="upper",
        )
        del rgba_bg
        gc.collect()
    else:
        rgba_overlay = np.zeros((n_rows, n_cols, 4), dtype=np.float32)
        inv_gray = 1.0 - gray_scaled  # high activity -> bright
        rgba_overlay[..., :3] = inv_gray[..., None]
        rgba_overlay[..., 3] = 1.0

        ax.imshow(
            rgba_overlay,
            aspect="auto",
            interpolation=interp,
            extent=extent,
            origin="upper",
        )
        del rgba_overlay
        gc.collect()

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)

    # --- horizontal boundaries + right labels ---
    if bounds and (clus_sorted is not None):
        rc = np.asarray(clus_sorted)
        boundaries = np.where(rc[1:] != rc[:-1])[0] + 0.5

        for y in boundaries:
            ax.axhline(y, color="k", linewidth=0.6, zorder=5)

        trans_right = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        edges = np.concatenate(([0.5], boundaries, [n_rows - 0.5]))
        n_segments = len(edges) - 1

        if clabels == "all":
            label_idxs = np.arange(n_segments)
        elif isinstance(clabels, int):
            if clabels < 1:
                label_idxs = np.array([], dtype=int)
            elif clabels == 1:
                label_idxs = np.array([0], dtype=int)
            else:
                label_idxs = np.linspace(0, n_segments - 1, clabels, dtype=int)
        else:
            raise ValueError("clabels must be 'all' or a positive int")

        fontsize = float(np.clip(300 / max(int(nclus), 1), 3, 8))
        for i in label_idxs:
            y0, y1 = edges[i], edges[i + 1]
            mid_y = 0.5 * (y0 + y1)
            row_idx = int(np.clip(np.floor(mid_y), 0, n_rows - 1))
            ax.text(
                1.01,
                mid_y,
                str(clus_sorted[row_idx]),
                transform=trans_right,
                va="center",
                ha="left",
                fontsize=fontsize,
                color="k",
                clip_on=False,
            )

    if "len" in r and isinstance(r["len"], dict) and len(r["len"]) > 0:
        ordered_segments = list(r["len"].keys())

        labels = (peth_dict if peth_dict is not None else r.get("peth_dict", None))
        if labels is None:
            labels = {k: k for k in ordered_segments}

        # one scalar per PETH -> one column
        n_seg = min(len(ordered_segments), n_cols)

        # IMPORTANT: shift ticks by +0.5 so they describe each column correctly
        xticks = np.arange(n_seg) + 0.5
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [labels.get(seg, seg) for seg in ordered_segments[:n_seg]],
            rotation=90,
        )

        # move ticks/labels to top only
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(
            axis="x",
            which="both",
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
        )

        fontsize = float(np.clip(180 / max(n_seg, 1), 6, 12))
        for lab in ax.get_xticklabels():
            lab.set_fontsize(fontsize)

        ax.set_xlabel("")
    else:
        # fallback (should not happen if stacks are well-formed)
        if fps is not None and float(fps) > 0:
            step = max(int(round(1.0 * float(fps))), 1)
            x_ticks = np.arange(0, n_cols + 1, step)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{int(t / float(fps))}" for t in x_ticks])
            ax.set_xlabel("time [sec]")
        else:
            ax.set_xlabel("PETH index")

    ax.set_ylabel("cells")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(False)

    if img_only:
        ax.axis("off")

    plt.tight_layout()

    # --- window title ---
    try:
        fig.canvas.manager.set_window_title(descriptor)
    except Exception:
        pass

    # --- save ---
    out_path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{fname_prefix}_{descriptor}".replace("|", "_").replace("=", "_") + ".png"
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")

    if clsfig:
        plt.close(fig)


def plot_cluster_mean_PETHs_meso(
    r: dict,
    *,
    feat: str = "concat_z",
    extraclus=None,
    axx=None,
    alone: bool = True,
    mapping: str | None = None,
    vmax: float | None = None,
    vmin: float | None = None,
    interp: str = "none",
    top_labels: bool = True,
    top_label_fs: float = 10.0,
    peth_dict: dict = peth_dictm,
):
    """
    Plot cluster-mean PETH values for mesoscope data.

    IMPORTANT:
    - Each PETH type contributes exactly ONE scalar value (1 column).
    - Visualization is done with imshow (1 × n_peth row per cluster, or overlay rows).

    Modes
    -----
    - overlay mode (extraclus non-empty): one axis, multiple rows (one per requested cluster)
    - multi-panel mode (extraclus empty): one axis per cluster (each axis shows a 1×n_peth image)
    """
    if feat not in r:
        raise KeyError(f"Feature '{feat}' not in result dict.")
    if "acs" not in r or "cols" not in r:
        raise KeyError("Result dict must contain 'acs' and 'cols'.")
    if "len" not in r or not isinstance(r["len"], dict) or len(r["len"]) == 0:
        raise KeyError("r['len'] (segment lengths) missing or empty.")

    ordered_segments = list(r["len"].keys())
    if peth_dict is None:
        peth_dict = {k: k for k in ordered_segments}

    # normalize extraclus
    if extraclus is None:
        extraclus = []
    if not isinstance(extraclus, (list, tuple, np.ndarray)):
        raise TypeError("extraclus must be a list/tuple/array of integers (or empty).")
    extraclus = [int(x) for x in extraclus]

    X = np.asarray(r[feat])          # (N_cells, N_peth)
    acs = np.asarray(r["acs"])
    n_peth = int(X.shape[1])

    # tick/label centers on columns
    x_centers = np.arange(n_peth) + 0.5
    xticklabels = [peth_dict.get(seg, seg) for seg in ordered_segments[:n_peth]]

    clu_vals = np.array(sorted(np.unique(acs)))
    n_clu = len(clu_vals)
    if n_clu > 50 and len(extraclus) == 0:
        print("too many (>50) plots!")
        return

    # common extent so ticks at 0.5..n-0.5 align with imshow pixel centers
    extent = (0, n_peth, 0, 1)  # x in [0,n_peth], y in [0,1] for single-row images

    def _set_top_peth_ticks(ax):
        if not top_labels:
            return
        ax.set_xticks(x_centers)
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=top_label_fs)
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(axis="x", which="both", top=True, labeltop=True, bottom=False, labelbottom=False)

    # ================= overlay mode =================
    if len(extraclus) > 0:
        valid = set(int(c) for c in clu_vals.tolist())
        bad = [c for c in extraclus if c not in valid]
        if bad:
            raise ValueError(f"extraclus contains invalid cluster IDs {bad}. Valid: {sorted(valid)}")

        # build (K, n_peth) matrix
        rows = []
        for clu in extraclus:
            idx = np.where(acs == clu)[0]
            if idx.size == 0:
                rows.append(np.full(n_peth, np.nan, dtype=float))
            else:
                rows.append(np.nanmean(X[idx, :], axis=0).astype(float, copy=False))
        M = np.vstack(rows)  # (K, n_peth)

        if axx is None:
            fg, ax = plt.subplots(figsize=(6, max(1.5, 0.35 * len(extraclus) + 1.0)))
        else:
            ax = axx if not isinstance(axx, (list, np.ndarray)) else axx[0]
            fg = ax.figure

        # choose vmin/vmax if not provided
        vmin_eff = np.nanmin(M) if vmin is None else float(vmin)
        vmax_eff = np.nanmax(M) if vmax is None else float(vmax)

        ax.imshow(
            M,
            aspect="auto",
            interpolation=interp,
            extent=(0, n_peth, len(extraclus), 0),  # origin='upper' via extent
            vmin=vmin_eff,
            vmax=vmax_eff,
        )

        _set_top_peth_ticks(ax)

        ax.set_yticks(np.arange(len(extraclus)) + 0.5)
        ax.set_yticklabels([str(c) for c in extraclus])
        ax.set_ylabel("cluster")
        ax.set_xlabel("")
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        if alone:
            plt.tight_layout()

        try:
            fg.canvas.manager.set_window_title(
                f"Avg | {feat} | {mapping if mapping is not None else 'meso'} | overlay={extraclus}"
            )
        except Exception:
            pass

        return fg, ax

    # ================= multi-panel mode =================
    if axx is None:
        fg, axx = plt.subplots(nrows=n_clu, sharex=True, sharey=False, figsize=(6, max(2.0, 0.28 * n_clu + 1.5)))
    else:
        fg = plt.gcf()

    if not isinstance(axx, (list, np.ndarray)):
        axx = [axx]
    if len(axx) != n_clu:
        raise ValueError(f"Expected {n_clu} axes, got {len(axx)}.")

    # global vmin/vmax across all clusters for comparability (unless user pins it)
    if vmin is None or vmax is None:
        # compute per-cluster means then global min/max
        Ms = []
        for clu in clu_vals:
            idx = np.where(acs == clu)[0]
            Ms.append(np.nanmean(X[idx, :], axis=0) if idx.size else np.full(n_peth, np.nan))
        M_all = np.vstack(Ms)
        vmin_eff = np.nanmin(M_all) if vmin is None else float(vmin)
        vmax_eff = np.nanmax(M_all) if vmax is None else float(vmax)
        del M_all
    else:
        vmin_eff = float(vmin)
        vmax_eff = float(vmax)

    for k, clu in enumerate(clu_vals):
        idx = np.where(acs == clu)[0]
        row = (np.nanmean(X[idx, :], axis=0) if idx.size else np.full(n_peth, np.nan)).astype(float, copy=False)

        ax = axx[k]
        ax.imshow(
            row[None, :],                 # (1, n_peth)
            aspect="auto",
            interpolation=interp,
            extent=extent,
            vmin=vmin_eff,
            vmax=vmax_eff,
        )

        ax.set_yticks([])

        # y label: cluster id
        ax.set_ylabel(str(clu), rotation=0, labelpad=10)

        # top labels only on first axis
        if k == 0:
            _set_top_peth_ticks(ax)
        else:
            ax.tick_params(axis="x", which="both", top=False, labeltop=False, bottom=False, labelbottom=False)

        # remove clutter
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if k != (n_clu - 1):
            ax.spines["bottom"].set_visible(False)

        ax.set_xlim(0, n_peth)
        ax.set_ylim(0, 1)

    # no bottom xlabel (x labels are on top of first axis)
    axx[-1].set_xlabel("")

    if alone:
        plt.tight_layout()

    try:
        fg.canvas.manager.set_window_title(
            f"Avg | {feat} | {mapping if mapping is not None else 'meso'} | nclus={n_clu}"
        )
    except Exception:
        pass

    plt.show()




def plot_xy_meso(
    mapping: str = "kmeans",
    *,
    # ---- shared loading/behavior ----
    tab20_colors: bool = False,
    nclus: int = 100,
    cv: bool = False,
    xcol: int = 0,
    ycol: int = 1,
    to_mm: bool = True,
    stack_dir=None,
    cache_dir=None,
    filter_neurons: bool = True,
    background: str = "white",
    # ---- datoviz (always runs) ----
    dv_point_size: float = 3.0,
    dv_width: int = 1200,
    dv_height: int = 900,
    # ---- optional PNG save (datashader) ----
    save_png: bool = False,                 # NEW: optional static save
    savepath: str | Path | None = None,     # default handled below (only if save_png)
    dpi: int | None = None,
    png_width: int = 3000,
    png_height: int = 2500,
    how: str = "eq_hist",
    png_point_size: int = 3,                # spread px
    x_range=None,
    y_range=None,
    title: str | None = None,
    title_color=(0, 0, 0),
    title_xy=(20, 20),
    title_fontsize: int = 28,
    png_min_alpha: int = 80,          # 0..255 (higher = more visible)
    png_span: float | None = None, 
    cv_sim_threshold: float | None = None,
):
    """
    Mesoscope XY visualization with:
      1) Datoviz interactive scatter ALWAYS shown.
      2) Optional static PNG save via Datashader (save_png=True).

    Data loading and preprocessing are shared between both outputs:
      - single call to regional_group_meso(...)
      - optional _override_cols_tab20(r)
      - same x/y extraction and scaling
      - same colors taken from r['cols'] (RGBA floats in [0,1])

    Default PNG save location (if save_png and savepath is None):
        pth_meso / 'figs' / f'xy_{mapping}_nclus{nclus}_cv{cv}.png'
    """

    legend: bool = True,
    legend_max_items: int = 20,        # for categorical mappings
    legend_xy: tuple[int, int] = (40, 80),
    legend_swatch: int = 18,
    legend_linegap: int = 6,
    legend_bg_alpha: int = 160,   


    # ---------- load once ----------
    r = regional_group_meso(
        mapping,
        stack_dir=stack_dir,
        cache_dir=cache_dir,
        filter_neurons=filter_neurons,
        cv=cv,
        nclus=nclus,
    )



    for k in ("xyz", "acs", "cols"):
        if k not in r:
            raise KeyError(f"regional_group_meso output must contain key '{k}'.")



    if tab20_colors:
        _override_cols_tab20(r)



    xyz = np.asarray(r["xyz"])
    scale = 1000.0 if to_mm else 1.0

    # Datoviz build you used requires float64 for axes.normalize
    x64 = (xyz[:, xcol] / scale).astype(np.float64, copy=False)
    y64 = (xyz[:, ycol] / scale).astype(np.float64, copy=False)

    # Keep float32 copies too (useful for datashader / dataframe; and cheaper)
    x32 = x64.astype(np.float32, copy=False)
    y32 = y64.astype(np.float32, copy=False)

    labels = np.asarray(r["acs"])
    cols = np.asarray(r["cols"], dtype=np.float32)

     # ---------- optional cv_sim alpha masking ----------
    if cv_sim_threshold is not None:
        if "cv_sim" not in r:
            raise ValueError(
                "cv_sim_threshold was set, but r does not contain 'cv_sim'. "
                "Use mapping='cv_sim' or compute cv_sim upstream."
            )

        sim = np.asarray(r["cv_sim"], dtype=float)
        if sim.shape[0] != cols.shape[0]:
            raise ValueError("cv_sim length does not match number of neurons.")

        # set alpha = 0 for low-cv neurons
        mask = np.isfinite(sim) & (sim > float(cv_sim_threshold))
        cols = cols.copy()
        cols[~mask, 3] = 0.0

        # write back
        r["cols"] = cols   


    if cols.ndim != 2 or cols.shape[1] != 4:
        raise ValueError("r['cols'] must be (N,4) RGBA floats.")

    # RGBA uint8 for Datoviz
    color_u8 = np.clip(np.round(cols * 255.0), 0, 255).astype(np.uint8, copy=False)

    N = int(x64.shape[0])

    if title is None:
        title = f"mapping={mapping}   nclus={int(nclus)}   cv={bool(cv)}   N={N}"

    # Common bounds for both outputs (unless overridden for PNG)
    xlim = (float(np.nanmin(x64)), float(np.nanmax(x64)))
    ylim = (float(np.nanmin(y64)), float(np.nanmax(y64)))

    # =========================
    # 1) Datoviz interactive (ALWAYS)
    # =========================
    try:
        import datoviz as dv
    except Exception as e:
        raise ImportError(
            "Datoviz is required for plot_xy_meso (datoviz always runs here). "
            "Install with: pip install datoviz"
        ) from e

    app = dv.App(background=str(background))
    fig = app.figure(int(dv_width), int(dv_height))
    panel = fig.panel()
    axes = panel.axes(xlim, ylim)

    pos_ndc = axes.normalize(x64, y64)  # float64 input required in this build

    size = np.full(N, float(dv_point_size), dtype=np.float32)

    visual = app.point(
        position=pos_ndc,
        color=color_u8,
        size=size,
    )
    panel.add(visual)

    # =========================
    # 2) Optional PNG save (Datashader)
    # =========================
    if save_png:
        import pandas as pd
        import datashader as ds
        import datashader.transfer_functions as tf
        from PIL import ImageDraw, ImageFont
        from PIL.PngImagePlugin import PngInfo

        # ---------- default save path ----------
        if savepath is None:
            try:
                base = Path(pth_meso) / "figs"
            except NameError:
                raise NameError(
                    "pth_meso is not defined. Either define pth_meso or pass savepath explicitly."
                )
            base.mkdir(parents=True, exist_ok=True)
            savepath = base / f"xy_{mapping}_nclus{int(nclus)}_cv{bool(cv)}.png"
        else:
            savepath = Path(savepath)

        if x_range is None:
            x_range = (float(np.nanmin(x32)), float(np.nanmax(x32)))
        if y_range is None:
            y_range = (float(np.nanmin(y32)), float(np.nanmax(y32)))

        # ---------- color key (first occurrence per label; stable w.r.t. original order) ----------
        s = pd.Series(labels)
        first_idx = s.groupby(s, sort=False).head(1).index.to_numpy()
        uniq = s.iloc[first_idx].to_numpy()

        color_key = {}
        for idx, lab in zip(first_idx, uniq):
            rgba8 = np.clip(np.round(cols[int(idx)] * 255.0), 0, 255).astype(np.uint8)
            color_key[lab] = "#{:02x}{:02x}{:02x}".format(*rgba8[:3])

        df = pd.DataFrame(
            dict(
                x=x32,
                y=y32,
                label=pd.Categorical(labels, categories=pd.Index(uniq)),
            )
        )

        # ---------- rasterize ----------
        cvs = ds.Canvas(
            plot_width=int(png_width),
            plot_height=int(png_height),
            x_range=x_range,
            y_range=y_range,
        )

        agg = cvs.points(df, "x", "y", agg=ds.count_cat("label"))
        img = tf.shade( agg,
                        color_key=color_key,
                        how=str(how),
                        min_alpha=int(png_min_alpha),
                        span=png_span,          # None means datashader chooses span
                    )

        if int(png_point_size) > 1:
            img = tf.spread(img, px=int(png_point_size))

        if background is not None:
            img = tf.set_background(img, background)

        # ---------- title annotation via PIL ----------
        pil = img.to_pil()
        draw = ImageDraw.Draw(pil)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", int(title_fontsize))
        except Exception:
            font = ImageFont.load_default()

        draw.text(tuple(title_xy), str(title), fill=title_color, font=font)

        if legend:

            def _scalar(x, name: str):
                # Accept scalar or 1-tuple/list; reject other iterables
                if isinstance(x, (list, tuple)):
                    if len(x) != 1:
                        raise ValueError(f"{name} must be scalar or 1-tuple; got {x!r}")
                    x = x[0]
                return x

            # normalize legend params (robust to accidental 1-tuples)
            legend_xy = _scalar(legend_xy, "legend_xy")
            legend_max_items = int(_scalar(legend_max_items, "legend_max_items"))
            legend_swatch = int(_scalar(legend_swatch, "legend_swatch"))
            legend_linegap = int(_scalar(legend_linegap, "legend_linegap"))
            legend_bg_alpha = int(_scalar(legend_bg_alpha, "legend_bg_alpha"))

            # normalize legend_xy to (x,y)
            if isinstance(legend_xy, (list, tuple)) and len(legend_xy) == 1 and isinstance(legend_xy[0], (list, tuple)):
                legend_xy = legend_xy[0]
            if not (isinstance(legend_xy, (list, tuple)) and len(legend_xy) == 2):
                raise ValueError(f"legend_xy must be a 2-tuple (x,y); got {legend_xy!r}")
            lx, ly = int(legend_xy[0]), int(legend_xy[1])

            # ensure RGBA image so we can draw semi-transparent legend background
            pil = pil.convert("RGBA")
            draw = ImageDraw.Draw(pil, "RGBA")

            # legend_xy can be (x, y) or ((x, y),); normalize
            if isinstance(legend_xy, (list, tuple)) and len(legend_xy) == 1 and isinstance(legend_xy[0], (list, tuple)):
                legend_xy = legend_xy[0]
            if not (isinstance(legend_xy, (list, tuple)) and len(legend_xy) == 2):
                raise ValueError(f"legend_xy must be a 2-tuple (x,y); got {legend_xy!r}")
            lx, ly = int(legend_xy[0]), int(legend_xy[1])

            # semi-transparent background box sized after we know content
            def _draw_bg_box(x0, y0, x1, y1):
                pad = 10
                draw.rectangle(
                    [x0 - pad, y0 - pad, x1 + pad, y1 + pad],
                    fill=(255, 255, 255, int(legend_bg_alpha)),
                    outline=(0, 0, 0, 200),
                    width=2,
                )

            # Scalar legend for fr/lz/cv_sim: draw a simple vertical color bar with min/max
            if mapping in ("fr", "lz", "cv_sim"):
                if mapping == "cv_sim":
                    vals = np.asarray(r.get("cv_sim", []), dtype=float)
                    vmin, vmax = -1.0, 1.0
                    cmap_for_bar = cm.get_cmap("magma")
                    label_lo, label_hi = f"{vmin:.2f}", f"{vmax:.2f}"
                    legend_title = "cv_sim"
                elif mapping == "fr":
                    vals = np.asarray(r.get("fr", []), dtype=float)
                    # you used fr**0.1 for coloring; keep legend consistent with what you colorized
                    vals = np.where(np.isfinite(vals), vals ** 0.1, np.nan)
                    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                    cmap_for_bar = cm.get_cmap("magma")
                    label_lo, label_hi = f"{vmin:.3g}", f"{vmax:.3g}"
                    legend_title = "fr**0.1"
                else:  # lz
                    vals = np.asarray(r.get("lz", []), dtype=float)
                    vals = np.where(np.isfinite(vals), vals ** 0.1, np.nan)
                    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                    cmap_for_bar = cm.get_cmap("cividis")
                    label_lo, label_hi = f"{vmin:.3g}", f"{vmax:.3g}"
                    legend_title = "lz**0.1"

                bar_w = 22
                bar_h = 180
                # background box bounds
                text_w = 140
                x0, y0 = lx, ly
                x1, y1 = lx + bar_w + 12 + text_w, ly + bar_h + 30
                _draw_bg_box(x0, y0, x1, y1)

                # title
                draw.text((lx, ly), legend_title, fill=(0, 0, 0, 255), font=font)

                # draw bar (top=vmax bright)
                bx = lx
                by = ly + 26
                for j in range(bar_h):
                    t = 1.0 - (j / max(1, bar_h - 1))
                    rgba = np.array(cmap_for_bar(t)) * 255.0
                    rgba = np.clip(np.round(rgba), 0, 255).astype(np.uint8)
                    draw.line([(bx, by + j), (bx + bar_w, by + j)], fill=tuple(rgba.tolist()), width=1)

                # min/max labels
                draw.text((bx + bar_w + 10, by - 6), label_hi, fill=(0, 0, 0, 255), font=font)
                draw.text((bx + bar_w + 10, by + bar_h - 18), label_lo, fill=(0, 0, 0, 255), font=font)

            else:
                # Categorical legend (kmeans/rm/Beryl/Cosmos/etc.): show top-N labels by count
                lab = labels
                c = Counter(lab.tolist())
                # normalize legend_max_items (must be scalar int)
                lmi = legend_max_items
                if isinstance(lmi, (list, tuple)):
                    if len(lmi) != 1:
                        raise ValueError(f"legend_max_items must be an int or 1-tuple; got {lmi!r}")
                    lmi = lmi[0]

                try:
                    lmi = int(lmi)
                except Exception:
                    raise ValueError(f"legend_max_items must be convertible to int; got {legend_max_items!r}")

                top = c.most_common(lmi)

                # measure approximate box size
                line_h = legend_swatch + legend_linegap
                box_h = 30 + line_h * len(top)
                box_w = 320
                x0, y0 = lx, ly
                x1, y1 = lx + box_w, ly + box_h
                _draw_bg_box(x0, y0, x1, y1)

                draw.text((lx, ly), f"Legend (top {len(top)})", fill=(0, 0, 0, 255), font=font)

                yy = ly + 26
                for lab_i, cnt_i in top:
                    # color_key already maps label -> "#RRGGBB" from first occurrence
                    hexcol = color_key.get(lab_i, "#808080")
                    rr = int(hexcol[1:3], 16)
                    gg = int(hexcol[3:5], 16)
                    bb = int(hexcol[5:7], 16)

                    # swatch
                    draw.rectangle(
                        [lx, yy, lx + int(legend_swatch), yy + int(legend_swatch)],
                        fill=(rr, gg, bb, 255),
                        outline=(0, 0, 0, 200),
                        width=1,
                    )
                    # text
                    draw.text(
                        (lx + int(legend_swatch) + 10, yy - 2),
                        f"{lab_i}  (n={cnt_i})",
                        fill=(0, 0, 0, 255),
                        font=font,
                    )
                    yy += line_h


        # ---------- save ----------
        if dpi is None:
            pil.save(savepath, format="PNG", optimize=True)
        else:
            meta = PngInfo()
            meta.add_text("dpi", str(int(dpi)))
            pil.save(
                savepath,
                format="PNG",
                optimize=True,
                dpi=(int(dpi), int(dpi)),
                pnginfo=meta,
            )

        print(f"Saved PNG → {savepath}")

    # Run Datoviz last so the window appears and stays interactive;
    # PNG work (if any) is done before run().
    app.run()
    app.destroy()



def plot_cv_sim_hist_meso(
    *,
    stack_dir=None,
    cache_dir=None,
    filter_neurons: bool = True,
    single_eid=None,
    rerun: bool = False,
    bins: int = 80,
    range_=(-1.0, 1.0),
    density: bool = False,
    title: str | None = None,
):
    """
    Load mapping='cv_sim' from regional_group_meso and plot histogram of per-neuron cv cosine similarity.
    """
    r = regional_group_meso(
        "cv_sim",
        stack_dir=stack_dir,
        cache_dir=cache_dir,
        filter_neurons=filter_neurons,
        cv=False,              # not required for cv_sim (you made cv blocks load for mapping=='cv_sim')
        single_eid=single_eid,
        rerun=rerun,
    )

    sim = np.asarray(r["cv_sim"], dtype=float)
    sim = sim[np.isfinite(sim)]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(sim, bins=int(bins), range=range_, density=bool(density))
    ax.set_xlabel("cv_sim = cosine(concat_z_even, concat_z_odd)")
    ax.set_ylabel("Density" if density else "Count")
    if title is None:
        title = f"cv_sim histogram (N={sim.size})"
    ax.set_title(title)
    ax.set_xlim(range_)
    fig.tight_layout()

