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

sys.path.insert(0, str(Path.home() / "Dropbox" / "scripts" / "IBL"))

from dmn_bwm import get_allen_info

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


def get_canonical_sessions(one: ONE | None = None):
    """
    Download canonical mesoscope session paths, flip backslashes to slashes,
    and map to eids via ONE.path2eid.

    Returns
    -------
    eids : list[str]
        Unique eids in file order (None filtered out).
    """
    if one is None:
        one = ONE()

    url = "https://raw.githubusercontent.com/int-brain-lab/mesoscope/main/canonical_sessions.txt"
    txt = requests.get(url, timeout=30).text

    eids = []
    seen = set()

    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # flip Windows separators -> POSIX
        line = line.replace("\\", "/")

        eid = str(one.path2eid(PurePosixPath(line)))
        if eid is None:
            continue

        if eid not in seen:
            seen.add(eid)
            eids.append(eid)

    return eids


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
):
    """
    Cut mesoscope ROI traces into per-trial windows (no averaging, no rastermap, no binning).

    IMPORTANT window semantics (matches your examples):
      Given [pre, post] and event time t0:
        window_start = t0 - pre
        window_end   = t0 + post
      where post can be negative.
      Examples:
        [0.15, 0.0]  -> [t0-0.15, t0]
        [0.4, -0.1]  -> [t0-0.4,  t0-0.1]

    The cut is *inclusive* of both endpoints in frame-index space:
      frames from nearest-frame(window_start) ... nearest-frame(window_end), inclusive.
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
    ) -> tuple[int, int]:
        """
        Inclusive endpoint cutting at native frame rate.

        window_start = t0 - pre
        window_end   = t0 + post

        Frames are selected by converting window_start/window_end to frame indices using rounding,
        then taking idx_start..idx_end inclusive.

        Writes shape (n_trials, N, n_frames).
        """
        pre = float(pre)
        post = float(post)

        if not np.isfinite(pre) or not np.isfinite(post):
            raise ValueError("Non-finite pre/post.")

        dt = float(np.median(np.diff(frame_times)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Bad frame_times dt.")

        frame0 = float(frame_times[0])

        # base (ROI-independent) inclusive frame length, derived from the offsets in seconds
        # n_frames = round((post - (-pre))/dt) + 1 = round((pre + post)/dt) + 1 when post>=-pre, but post can be negative
        # safer: compute from times:
        #   start_time = t0 - pre
        #   end_time   = t0 + post
        # length depends only on (pre, post) for fixed dt.
        # We'll compute using a dummy t0=0 in frame space:
        start_rel = -pre
        end_rel = post
        if end_rel <= start_rel:
            raise ValueError(f"Invalid window: [{start_rel}, {end_rel}] relative to event (pre={pre}, post={post}).")

        idx0_rel = int(np.round(start_rel / dt))
        idx1_rel = int(np.round(end_rel / dt))
        n_frames = idx1_rel - idx0_rel + 1
        if n_frames <= 0:
            raise ValueError(f"n_frames <= 0 for window [{start_rel},{end_rel}] and dt={dt}.")

        N, T = roi_signal.shape
        n_trials = int(event_times.size)

        # per-ROI shift in frames (same convention as before)
        offset_frames = np.round(offsets_s / dt).astype(np.int32)  # (N,)

        # within-window offsets in frames, inclusive
        frame_offsets = (np.arange(n_frames, dtype=np.int32) + idx0_rel)[None, :]  # (1, n_frames)

        mm = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.float32, shape=(n_trials, N, n_frames)
        )

        for i, t0 in enumerate(event_times.astype(float, copy=False)):
            # event frame index (base, ROI-independent)
            event_idx = int(np.round((t0 - frame0) / dt))

            # for each ROI, apply time shift by subtracting offset_frames
            # idx = event_idx + frame_offsets - offset_frames
            idx = (event_idx + frame_offsets) - offset_frames[:, None]  # (N, n_frames)

            valid = (idx >= 0) & (idx < T)
            idx_clip = np.clip(idx, 0, T - 1)

            vals = np.take_along_axis(roi_signal, idx_clip, axis=1).astype(np.float32, copy=False)
            vals[~valid] = np.nan

            mm[i, :, :] = vals

            if (i + 1) % 100 == 0 or (i + 1) == n_trials:
                print(f"    wrote {i+1}/{n_trials} trials to {out_path.name}")

        del mm
        return n_trials, n_frames

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

        dt = float(np.median(np.diff(frame_times)))
        print("  roi_signal:", roi_signal.shape, "frame_times:", frame_times.shape, "dt:", dt)
        print("  regions:", Counter(region_labels))

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

                # placeholder with correct inclusive-frame length
                pre_f = float(pre)
                post_f = float(post)
                dt_f = dt

                start_rel = -pre_f
                end_rel = post_f
                if end_rel <= start_rel:
                    raise ValueError(f"{eid} {fov} {keyname}: invalid window [{start_rel},{end_rel}].")

                idx0_rel = int(np.round(start_rel / dt_f))
                idx1_rel = int(np.round(end_rel / dt_f))
                n_frames = max(idx1_rel - idx0_rel + 1, 1)

                mm = np.lib.format.open_memmap(
                    out_path, mode="w+", dtype=np.float32, shape=(1, roi_signal.shape[0], n_frames)
                )
                mm[:] = np.nan
                del mm

                peth_files[keyname] = out_path.name
                peth_shapes[keyname] = (1, int(roi_signal.shape[0]), int(n_frames))
                print(f"  {keyname}: 0 trials -> placeholder {out_path.name} shape={peth_shapes[keyname]}")
                continue

            print(f"  {keyname}: {tls[keyname]} trials -> writing {out_path.name}")
            n_trials, n_frames = _cut_one_peth_to_memmap(
                roi_signal=roi_signal,
                frame_times=frame_times,
                offsets_s=roi_offsets_use,
                event_times=events_all,
                pre=pre,
                post=post,
                out_path=out_path,
            )
            peth_files[keyname] = out_path.name
            peth_shapes[keyname] = (int(n_trials), int(roi_signal.shape[0]), int(n_frames))

        meta = {
            "eid": str(eid),
            "fov": str(fov),
            "filter_neurons": bool(filter_neurons),
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
            "window_semantics": "start=t0-pre, end=t0+post, inclusive endpoints after rounding to frames",
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


def run_all(eids):
    """
    Run save_trial_cuts_meso() for all eids in canonical sessions.
    """
     
    if eids is None:
        eids_all = get_canonical_sessions(one=one)
        eids = eids_all

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
    cut_to_min: bool = True,   # NEW
):
    """
    If cut_to_min=True:
      - include entries even if some PETH segments have fewer time bins than others
      - determine min T per ttype across all usable entries
      - truncate every segment to that min before concatenation (so all match)

    If cut_to_min=False:
      - keep old behavior: require exact match to reference lengths (mismatch entries skipped)
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
    out_all = out_dir / f"stack_all_filter{filter_neurons}.npy"
    out_even = out_dir / f"stack_even_filter{filter_neurons}.npy"
    out_odd = out_dir / f"stack_odd_filter{filter_neurons}.npy"

    # ---------------- helpers ----------------
    def _zscore_rows(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        return (X - mu) / sd

    def _avg_trials_mode(X_mnt: np.ndarray, mode: str) -> np.ndarray | None:
        if X_mnt.ndim != 3:
            raise ValueError(f"Expected (M,N,T), got {X_mnt.shape}")
        M = X_mnt.shape[0]
        if M < min_trials:
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

    # ---------------- discover EIDs ----------------
    eid_dirs = sorted([p for p in trial_cuts_dir.iterdir() if p.is_dir()])
    if not eid_dirs:
        raise RuntimeError(f"No eid subfolders found in {trial_cuts_dir}")

    # ---------------- pass 1: find entries + (optionally) min lengths ----------------
    ref_ttypes = None
    entries: list[tuple[str, str, Path]] = []

    # track min T per ttype across entries (only used when cut_to_min=True)
    min_len: dict[str, int] | None = {} if cut_to_min else None

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
                if cut_to_min:
                    for t in ref_ttypes:
                        if "peth_shapes" not in meta or t not in meta["peth_shapes"]:
                            print(f"[skip] {eid}/{mp.name}: missing peth_shapes for {t}")
                            ref_ttypes = None
                            break
                        min_len[t] = int(meta["peth_shapes"][t][2])
                if ref_ttypes is None:
                    continue
            else:
                if ttypes != ref_ttypes:
                    print(f"[skip] {eid}/{mp.name}: ttypes mismatch vs reference")
                    continue

            # require peth_shapes for speed/consistency
            if "peth_shapes" not in meta:
                print(f"[skip] {eid}/{mp.name}: missing peth_shapes")
                continue

            if cut_to_min:
                ok = True
                for t in ref_ttypes:
                    if t not in meta["peth_shapes"]:
                        print(f"[skip] {eid}/{mp.name}: missing peth_shapes for {t}")
                        ok = False
                        break
                    Tt = int(meta["peth_shapes"][t][2])
                    # update min
                    if Tt < min_len[t]:
                        min_len[t] = Tt
                if not ok:
                    continue
                entries.append((eid, meta["fov"], mp))

            else:
                # old behavior: establish ref_len from first entry, then require exact match
                # (kept close to your current logic)
                if "ref_len" not in locals():
                    ref_len = {t: int(meta["peth_shapes"][t][2]) for t in ref_ttypes}
                ok = True
                for t in ref_ttypes:
                    if t not in meta["peth_shapes"]:
                        print(f"[skip] {eid}/{mp.name}: missing peth_shapes for {t}")
                        ok = False
                        break
                    Tt = int(meta["peth_shapes"][t][2])
                    if Tt != ref_len[t]:
                        print(f"[skip] {eid}/{mp.name}: length mismatch for {t}: {Tt} != {ref_len[t]}")
                        ok = False
                        break
                if not ok:
                    continue
                entries.append((eid, meta["fov"], mp))

    if ref_ttypes is None or not entries:
        raise RuntimeError(
            f"No valid meta files found for filter_neurons={filter_neurons} in {trial_cuts_dir}."
        )

    if cut_to_min:
        ref_len = dict(min_len)  # redefine effective lengths as minima
        print("[info] cut_to_min=True; using per-ttype min lengths:")
        for t in ref_ttypes:
            print(f"  {t}: {ref_len[t]}")
    else:
        print("[info] cut_to_min=False; requiring exact per-ttype lengths (skip mismatches).")

    L = int(sum(ref_len[t] for t in ref_ttypes))
    print(f"[info] combining {len(entries)} FOV entries; L={L} (sum over {len(ref_ttypes)} ttypes)")

    # ---------------- pass 2: load, average, truncate (if needed), concat ----------------
    blocks_all, blocks_even, blocks_odd = [], [], []
    ids_blocks, xyz_blocks, eid_blocks, fov_blocks = [], [], [], []

    for eid, fov, mp in entries:
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
            X = np.load(fp, mmap_mode="r")  # (M,N,T_local)

            if X.shape[1] != ids.shape[0]:
                print(f"[skip] {eid}/{fov}: N mismatch for {t}: {X.shape[1]} != {ids.shape[0]}")
                ok = False
                break

            A_all = _avg_trials_mode(X, "all")
            A_even = _avg_trials_mode(X, "even")
            A_odd = _avg_trials_mode(X, "odd")

            if A_all is None or A_even is None or A_odd is None:
                print(f"[skip] {eid}/{fov}: insufficient trials for {t} (min_trials={min_trials})")
                ok = False
                break

            # enforce length handling
            T_target = int(ref_len[t])
            T_local = int(A_all.shape[1])

            if cut_to_min:
                # truncate to common minimum (or if somehow smaller than min, skip)
                if T_local < T_target:
                    print(f"[skip] {eid}/{fov}: {t} has T={T_local} < min_T={T_target} (unexpected)")
                    ok = False
                    break
                A_all = A_all[:, :T_target]
                A_even = A_even[:, :T_target]
                A_odd = A_odd[:, :T_target]
            else:
                # old behavior: require exact match
                if T_local != T_target:
                    print(f"[skip] {eid}/{fov}: T mismatch for {t}: {T_local} != {T_target}")
                    ok = False
                    break

            segs_all.append(A_all)
            segs_even.append(A_even)
            segs_odd.append(A_odd)

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
        r["len"] = {t: int(ref_len[t]) for t in ref_ttypes}

        r["concat"] = concat.astype(np.float32, copy=False)
        r["fr"] = np.array([np.mean(x) for x in r["concat"]], dtype=np.float32)
        r["concat_z"] = _zscore_rows(r["concat"])

        rng = np.random.default_rng(0)
        N = r["concat_z"].shape[0]
        lz_vals = np.zeros(N, float)
        for i in range(N):
            lz_vals[i] = lzs_pci(r["concat_z"][i], rng)
        r["lz"] = lz_vals
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
    nclus: int = 100,
    nclus_rm: int | None = None,
    grid_upsample: int = 0,
    locality: float = 0.75,
    time_lag_window: int = 5,
    symmetric: bool = False,
    rerun: bool = False,
    cache_dir: str | Path | None = None,
):
    """
    Mesoscope analogue of `regional_group(...)`, using stack files created by `stack_trial_cuts_meso()`.

    Inputs
    ------
    mapping : one of ['Beryl','Cosmos','rm','lz','fr','kmeans'].
        - 'Beryl'/'Cosmos': remap region acronyms via `br` (recommended).
        - 'rm'            : Rastermap cluster labels + isort (cached).
        - 'kmeans'        : KMeans labels (cached) + also attaches Rastermap isort (cached).
        - 'fr'/'lz'       : continuous colormaps over r['fr'] or r['lz'].

    cv : if True
        Loads:
          - stack_all_filter{filter}.npy   -> used for returned `r['concat']` / `r['concat_z']`
          - stack_even_filter{filter}.npy  -> used as TRAIN features for rm/kmeans
          - stack_odd_filter{filter}.npy   -> saved/attached as `r['concat_odd']` (and z-scored)
        and attaches:
          r['concat_even'], r['concat_odd'], r['concat_z_even'], r['concat_z_odd'].
        For fitting rm/kmeans, uses concat_z_even (train). Labels are always for ALL rows (using concat_z).

    Notes
    -----
    - Assumes the meso stack dict has at least keys:
        'ids','xyz','eid','FOV','ttypes','len','concat','concat_z','fr','lz'
      as produced by your `stack_trial_cuts_meso`.
    - If your `r['ids']` are already Beryl acronyms and you do not want remapping,
      you can pass mapping='Beryl' with br=None, and it will just echo ids.
    """
    mapping = str(mapping)

    # ---------- paths ----------
    if stack_dir is None:
        # same root as your save_trial_cuts_meso() defaults: <one.cache_dir>/meso/trial_cuts
        # stack_trial_cuts_meso saves stacks in trial_cuts_dir.parent == <one.cache_dir>/meso
        # so here stack_dir should be <one.cache_dir>/meso
        try:
            # expects you have `pth_meso` in scope (as in your code)
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

    stack_all_path = stack_dir / f"stack_all_filter{bool(filter_neurons)}.npy"
    stack_even_path = stack_dir / f"stack_even_filter{bool(filter_neurons)}.npy"
    stack_odd_path = stack_dir / f"stack_odd_filter{bool(filter_neurons)}.npy"

    if not stack_all_path.is_file():
        raise FileNotFoundError(f"Missing meso stack: {stack_all_path}")



    r = _load_dict(stack_all_path)

    # enforce ordered 'len'
    if "ttypes" in r and "len" in r:
        r["len"] = OrderedDict((k, int(r["len"][k])) for k in r["ttypes"])

    # ---------- attach cv blocks ----------
    if cv:
        if not stack_even_path.is_file() or not stack_odd_path.is_file():
            raise FileNotFoundError(
                f"cv=True but missing even/odd stacks:\n  {stack_even_path}\n  {stack_odd_path}"
            )
        r_even = _load_dict(stack_even_path)
        r_odd = _load_dict(stack_odd_path)

        # minimal sanity: same ordering & same N
        N = int(r["concat"].shape[0])
        if int(r_even["concat"].shape[0]) != N or int(r_odd["concat"].shape[0]) != N:
            raise ValueError("even/odd stacks do not match N of all-stack (ordering mismatch).")

        # z-score rows (consistent with your stack_trial_cuts_meso semantics)
        def _zscore_rows(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=np.float32)
            mu = np.nanmean(X, axis=1, keepdims=True)
            sd = np.nanstd(X, axis=1, keepdims=True)
            sd = np.where(sd == 0, 1.0, sd)
            return (X - mu) / sd

        r["concat_z_even"] = _zscore_rows(np.asarray(r_even["concat"], dtype=np.float32))
        r["concat_z_odd"] = _zscore_rows(np.asarray(r_odd["concat"], dtype=np.float32))

    # ---------- common bookkeeping ----------
    if "xyz" not in r:
        raise KeyError("Saved stack lacks 'xyz'.")
    r["nums"] = np.arange(r["xyz"].shape[0], dtype=int)

    feat_key = "concat_z"
    if feat_key not in r:
        raise KeyError(f"Saved stack lacks '{feat_key}'.")

    # order signature: stable across caches for this stack ordering
    r["_order_signature"] = (
        "|".join(f"{k}:{int(r['len'][k])}" for k in r.get("ttypes", []))
        + f"|shape:{tuple(r[feat_key].shape)}"
    )

    nclus_rm_eff = int(nclus) if nclus_rm is None else int(nclus_rm)

    def _cache_path(kind: str) -> Path:
        # include only what matters for that cache
        if kind == "rm":
            base = f"meso_rm_filter{bool(filter_neurons)}_cv{bool(cv)}_n{int(nclus_rm_eff)}"
            return cache_dir / (base + ".npy")
        if kind == "kmeans":
            base = f"meso_kmeans_filter{bool(filter_neurons)}_cv{bool(cv)}_n{int(nclus)}"
            return cache_dir / (base + ".npy")
        raise ValueError(kind)

    def _try_load_cache(p: Path) -> dict | None:
        if (not rerun) and p.is_file():
            try:
                d = np.load(p, allow_pickle=True).flat[0]
                return d if isinstance(d, dict) else None
            except Exception:
                return None
        return None

    def _tab20_repeat(labels: np.ndarray) -> np.ndarray:
        cmap = mpl.colormaps["tab20"]
        idx = (labels.astype(int) % 20)
        return cmap(idx)

    # ---------- mapping ----------
    if mapping == "rm":
        rm_cache_path = _cache_path("rm")
        cached = _try_load_cache(rm_cache_path)

        labels = None
        isort = None
        if cached is not None and (
            cached.get("order_sig") == r["_order_signature"]
            and cached.get("nclus_rm") == int(nclus_rm_eff)
            and "rm_labels" in cached
            and "isort" in cached
        ):
            labels = np.asarray(cached["rm_labels"], dtype=int).reshape(-1)
            isort = np.asarray(cached["isort"], dtype=int).reshape(-1)
            if labels.shape[0] != r[feat_key].shape[0] or isort.shape[0] != r[feat_key].shape[0]:
                labels, isort = None, None

        if labels is None or isort is None:
            feat_fit = "concat_z_even" if cv else feat_key
            if feat_fit not in r:
                raise KeyError(f"Feature '{feat_fit}' missing; need it for rm fit.")

            model = Rastermap(
                n_PCs=200,
                n_clusters=int(nclus_rm_eff),
                grid_upsample=grid_upsample,
                locality=locality,
                time_lag_window=time_lag_window,
                bin_size=1,
                symmetric=symmetric,
            ).fit(r[feat_fit])

            labels = np.asarray(model.embedding_clust, dtype=int)
            if labels.ndim > 1:
                labels = labels[:, 0]
            isort = np.asarray(model.isort, dtype=int).reshape(-1)

            if labels.shape[0] != r[feat_key].shape[0] or isort.shape[0] != r[feat_key].shape[0]:
                raise ValueError("Rastermap outputs do not match all-stack length.")

            np.save(
                rm_cache_path,
                {
                    "rm_labels": labels,
                    "isort": isort,
                    "order_sig": r["_order_signature"],
                    "nclus_rm": int(nclus_rm_eff),
                },
                allow_pickle=True,
            )

        cols = _tab20_repeat(labels)
        regs = np.unique(labels)
        color_map = {reg: cols[labels == reg][0] for reg in regs}
        r["els"] = [Line2D([0], [0], color=color_map[reg], lw=4, label=f"{reg}") for reg in regs]
        r["acs"] = labels
        r["cols"] = cols
        r["isort"] = isort

    elif mapping == "kmeans":
        km_cache_path = _cache_path("kmeans")
        cached = _try_load_cache(km_cache_path)

        clusters = None
        feat_fit = "concat_z_even" if cv else feat_key
        if feat_fit not in r:
            raise KeyError(f"Feature '{feat_fit}' missing; need it for kmeans fit.")

        if cached is not None and (
            cached.get("order_sig") == r["_order_signature"]
            and cached.get("feat_fit") == feat_fit
            and cached.get("nclus") == int(nclus)
            and "kmeans_labels" in cached
        ):
            clusters = np.asarray(cached["kmeans_labels"], dtype=int).reshape(-1)
            if clusters.shape[0] != r[feat_key].shape[0]:
                clusters = None

        if clusters is None:
            km = KMeans(n_clusters=int(nclus), random_state=0)
            km.fit(r[feat_fit])
            clusters = km.predict(r[feat_key]).astype(int)

            if clusters.shape[0] != r[feat_key].shape[0]:
                raise ValueError("KMeans labels do not match all-stack length.")

            np.save(
                km_cache_path,
                {
                    "kmeans_labels": clusters,
                    "order_sig": r["_order_signature"],
                    "feat_fit": feat_fit,
                    "nclus": int(nclus),
                },
                allow_pickle=True,
            )

        cols = _tab20_repeat(clusters)
        regs = np.unique(clusters)
        color_map = {reg: cols[clusters == reg][0] for reg in regs}
        r["els"] = [Line2D([0], [0], color=color_map[reg], lw=4, label=f"{reg + 1}") for reg in regs]
        r["acs"] = clusters
        r["cols"] = cols

        # attach Rastermap ordering too (cached in the rm cache, computed if missing)
        rm_cache_path = _cache_path("rm")
        cached_rm = _try_load_cache(rm_cache_path)
        isort = None
        if cached_rm is not None and (
            cached_rm.get("order_sig") == r["_order_signature"]
            and cached_rm.get("nclus_rm") == int(nclus_rm_eff)
            and "isort" in cached_rm
        ):
            isort = np.asarray(cached_rm["isort"], dtype=int).reshape(-1)
            if isort.shape[0] != r[feat_key].shape[0]:
                isort = None

        if isort is None:
            model = Rastermap(
                n_PCs=200,
                n_clusters=int(nclus_rm_eff),
                grid_upsample=grid_upsample,
                locality=locality,
                time_lag_window=time_lag_window,
                bin_size=1,
                symmetric=symmetric,
            ).fit(r[feat_fit])

            isort = np.asarray(model.isort, dtype=int).reshape(-1)
            labels_rm = np.asarray(model.embedding_clust, dtype=int)
            if labels_rm.ndim > 1:
                labels_rm = labels_rm[:, 0]

            np.save(
                rm_cache_path,
                {
                    "rm_labels": labels_rm,
                    "isort": isort,
                    "order_sig": r["_order_signature"],
                    "nclus_rm": int(nclus_rm_eff),
                },
                allow_pickle=True,
            )

        r["isort"] = isort

    elif mapping == "fr":
        if "fr" not in r:
            raise KeyError("Saved stack lacks 'fr'.")
        scaled = np.asarray(r["fr"], dtype=float) ** 0.1
        norm = Normalize(vmin=float(np.nanmin(scaled)), vmax=float(np.nanmax(scaled)))
        cmap = cm.get_cmap("magma")
        r["acs"] = np.asarray(r.get("ids", []), dtype=object)
        r["cols"] = cmap(norm(scaled))

    elif mapping == "lz":
        if "lz" not in r:
            raise KeyError("Saved stack lacks 'lz'.")
        scaled = np.asarray(r["lz"], dtype=float) ** 0.1
        norm = Normalize(vmin=float(np.nanmin(scaled)), vmax=float(np.nanmax(scaled)))
        cmap = cm.get_cmap("cividis")
        r["acs"] = np.asarray(r.get("ids", []), dtype=object)
        r["cols"] = cmap(norm(scaled))

    elif mapping in ("Beryl", "Cosmos"):
        acs_in = np.asarray(r.get("ids", []), dtype=object)

        # If `br` is provided, remap acronyms via id space.
        if br is not None:
            # acronym -> id -> acronym(mapping=...)
            ids_num = br.acronym2id(acs_in)
            acs = np.asarray(br.id2acronym(ids_num, mapping=mapping), dtype=object)
        else:
            # fall back: just echo ids (common if ids already Beryl acronyms)
            acs = acs_in

        if pal is None:
            raise ValueError("For mapping in {'Beryl','Cosmos'} you must pass a palette dict `pal` (acronym->color).")

        r["acs"] = acs
        # unknown regions -> gray
        r["cols"] = np.array([pal.get(a, (0.5, 0.5, 0.5, 1.0)) for a in acs], dtype=object)

        regsC = Counter(acs)
        r["els"] = [
            Line2D([0], [0], color=pal.get(reg, (0.5, 0.5, 0.5, 1.0)), lw=4, label=f"{reg} {regsC[reg]}")
            for reg in regsC
        ]

    else:
        raise ValueError("mapping must be one of ['Beryl','Cosmos','rm','lz','fr','kmeans'].")

    acs_in = np.asarray(r.get("ids", []), dtype=object)

    # If `br` is provided, remap acronyms via id space.
    if br is not None:
        # acronym -> id -> acronym(mapping=...)
        ids_num = br.acronym2id(acs_in)
        r['Beryl'] = np.asarray(br.id2acronym(ids_num, mapping='Beryl'), dtype=object)
    else:
        # fall back: just echo ids (common if ids already Beryl acronyms)
        r['Beryl'] = acs_in
     

    return r


def plot_rastermap_meso(
    *,
    stack_dir: str | Path | None = None,
    filter_neurons: bool = True,
    feat: str = "concat_z",
    mapping: str = "rm",
    cv: bool = True,
    sort_method: str = "rastermap",   # 'rastermap' or 'acs'
    nclus: int = 100,
    nclus_rm: int | None = None,
    rerun: bool = False,
    bounds: bool = True,
    bg: bool = False,
    bg_bright: float = 0.99,
    vmax: float = 2.0,
    interp: str = "antialiased",
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
    exa: bool = False,                 # NEW
):
    """
    Simplified mesoscope raster plot.

    If exa=True, also opens a new figure with plot_cluster_mean_PETHs_meso(r, ...).

    Adds a descriptor string composed of:
      mapping, cv, sort_method, nclus, nclus_rm
    used for window title and (if saving) filename.
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
    )

    if exa:
        # separate figure
        plt.ion()
        plot_cluster_mean_PETHs_meso(r)

    if feat not in r:
        raise KeyError(f"feat='{feat}' not in result dict. Available keys: {list(r.keys())[:30]}...")

    X = np.asarray(r[feat])
    if X.ndim != 2:
        raise ValueError(f"Expected r[{feat}] to be 2D, got shape {X.shape}")

    # --- descriptor ---
    nclus_rm_eff = int(nclus) if nclus_rm is None else int(nclus_rm)
    descriptor = f"mapping={mapping}|cv={cv}|sort={sort_method}|nclus={int(nclus)}|nclus_rm={nclus_rm_eff}"

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

    # avoid half-pixel strip issues
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

    # --- vertical segment boundaries + top labels (text), centered & boundary-shifted ---
    if "len" in r and isinstance(r["len"], dict) and len(r["len"]) > 0:
        ordered_segments = list(r["len"].keys())

        labels = (peth_dict if peth_dict is not None else r.get("peth_dict", None))
        if labels is None:
            labels = {k: k for k in ordered_segments}

        trans_top = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)

        h = 0
        for seg in ordered_segments:
            seg_len = int(r["len"][seg])
            end_bin = h + seg_len
            if end_bin > n_cols:
                break

            # boundary BETWEEN samples: between (end_bin-1) and (end_bin)
            xline = end_bin - 0.5
            if 0 <= xline <= (n_cols - 1):
                ax.axvline(xline, linestyle=":", linewidth=1, color="grey", zorder=6)

            if not img_only:
                # center of bins [h, ..., end_bin-1]
                center = h + (seg_len - 1) / 2.0
                ax.text(
                    center,
                    1.02,
                    labels.get(seg, seg),
                    rotation=90,
                    fontsize=10,
                    ha="center",
                    va="bottom",
                    transform=trans_top,
                    clip_on=False,
                )

            h = end_bin


    # --- x-axis ticks in seconds ---
    if fps is not None and float(fps) > 0:
        step = max(int(round(1.0 * float(fps))), 1)  # 1-second ticks
        x_ticks = np.arange(0, n_cols + 1, step)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{int(t / float(fps))}" for t in x_ticks])
        ax.set_xlabel("time [sec]")
    else:
        ax.set_xlabel("time [frames]")

    ax.set_ylabel("cells")
    ax.spines["top"].set_visible(False)
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
    fps: float = 8.0,
    extraclus=None,
    axx=None,
    alone: bool = True,
    mapping: str | None = None,
    vline_style: str = ":",
    vline_lw: float = 1.0,
    vline_color: str = "grey",
    top_labels: bool = True,
    top_label_fs: float = 10.0,
    # pretty segment labels (LaTeX) for TOP labels only
    peth_dict: dict = peth_dictm,
):
    if feat not in r:
        raise KeyError(f"Feature '{feat}' not in result dict.")
    if "acs" not in r or "cols" not in r:
        raise KeyError("Result dict must contain 'acs' and 'cols'.")
    if "len" not in r or not isinstance(r["len"], dict) or len(r["len"]) == 0:
        raise KeyError("r['len'] (segment lengths) missing or empty.")

    ordered_segments = list(r["len"].keys())
    if peth_dict is None:
        peth_dict = r.get("peth_dict", None)
    if peth_dict is None:
        peth_dict = {k: k for k in ordered_segments}

    # normalize extraclus
    if extraclus is None:
        extraclus = []
    if not isinstance(extraclus, (list, tuple, np.ndarray)):
        raise TypeError("extraclus must be a list/tuple/array of integers (or empty).")
    extraclus = [int(x) for x in extraclus]

    X = np.asarray(r[feat])
    acs = np.asarray(r["acs"])
    cols = np.asarray(r["cols"])

    clu_vals = np.array(sorted(np.unique(acs)))
    n_clu = len(clu_vals)
    if n_clu > 50 and len(extraclus) == 0:
        print("too many (>50) line plots!")
        return

    n_bins = int(X.shape[1])

    # time axis
    fps = float(fps) if fps is not None else 0.0
    if fps > 0:
        xx = np.arange(n_bins) / fps
        to_x = lambda b: b / fps
        xlabel = "time [sec]"
    else:
        xx = np.arange(n_bins)
        to_x = lambda b: b
        xlabel = "time [frames]"

    seg_lengths = [int(r["len"][s]) for s in ordered_segments]
    if sum(seg_lengths) != n_bins:
        print(f"[warn] sum(r['len'])={sum(seg_lengths)} != n_bins={n_bins}")

    def _draw_segments(ax, *, labels_on_top: bool):
        h = 0
        if labels_on_top and top_labels:
            trans_top = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)

        for s in ordered_segments:
            seg_len = int(r["len"][s])
            xv_bins = h + seg_len
            if xv_bins > n_bins:
                break

            # draw boundary BETWEEN samples: between (xv_bins-1) and (xv_bins)
            # boundary
            xline_bins = xv_bins - 0.5
            ax.axvline(
                to_x(xline_bins),
                linestyle=vline_style,
                linewidth=vline_lw,
                color=vline_color,
                zorder=0,
            )

            # centered top label
            if labels_on_top and top_labels:
                seg_center = h + (seg_len - 1) / 2.0
                ax.text(
                    to_x(seg_center),
                    1.02,
                    peth_dict.get(s, s),
                    rotation=90,
                    fontsize=top_label_fs,
                    ha="center",
                    va="bottom",
                    transform=trans_top,
                    clip_on=False,
                )
            h += seg_len


    # ---------------- overlay mode ----------------
    if len(extraclus) > 0:
        valid = set(int(c) for c in clu_vals.tolist())
        bad = [c for c in extraclus if c not in valid]
        if bad:
            raise ValueError(f"extraclus contains invalid cluster IDs {bad}. Valid: {sorted(valid)}")

        if axx is None:
            fg, ax = plt.subplots(figsize=(6, 3))
        else:
            ax = axx if not isinstance(axx, (list, np.ndarray)) else axx[0]
            fg = ax.figure

        for clu in extraclus:
            idx = np.where(acs == clu)[0]
            if idx.size == 0:
                continue
            yy = np.nanmean(X[idx, :], axis=0)
            col = cols[idx[0]]
            ax.step(xx, yy, where="mid", color=col, linewidth=1.5, label=str(clu))

        _draw_segments(ax, labels_on_top=True)

        ax.set_xlim(0, to_x(n_bins))
        ax.set_xlabel(xlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="cluster", frameon=False, loc="best")

        if alone:
            plt.tight_layout()

        try:
            fg.canvas.manager.set_window_title(
                f"Avg | {feat} | {mapping if mapping is not None else 'meso'} | overlay={extraclus}"
            )
        except Exception:
            pass

        return fg, ax

    # ---------------- multi-panel mode ----------------
    if axx is None:
        fg, axx = plt.subplots(nrows=n_clu, sharex=True, sharey=False, figsize=(6, 10))
    else:
        fg = plt.gcf()

    if not isinstance(axx, (list, np.ndarray)):
        axx = [axx]
    if len(axx) != n_clu:
        raise ValueError(f"Expected {n_clu} axes, got {len(axx)}.")

    for k, clu in enumerate(clu_vals):
        idx = np.where(acs == clu)[0]

        axx[k].spines["top"].set_visible(False)
        axx[k].spines["right"].set_visible(False)
        axx[k].spines["left"].set_visible(False)
        axx[k].tick_params(left=False, labelleft=False)

        yy = np.nanmean(X[idx, :], axis=0) if idx.size else np.full(n_bins, np.nan)
        col = cols[idx[0]] if idx.size else (0, 0, 0, 1)
        axx[k].step(xx, yy, where="mid", color=col, linewidth=1.5)

        # y label: cluster / acs category (as before)
        axx[k].set_ylabel(str(clu), rotation=0, labelpad=10)

        if k != (n_clu - 1):
            axx[k].spines["bottom"].set_visible(False)
            axx[k].tick_params(bottom=False, labelbottom=False)
        else:
            axx[k].spines["bottom"].set_visible(True)
            axx[k].tick_params(bottom=True, labelbottom=True)

        # boundaries on every axis; TOP labels only on first axis (like your ephys version)
        _draw_segments(axx[k], labels_on_top=(k == 0))
        axx[k].set_xlim(0, to_x(n_bins))

    axx[-1].set_xlabel(xlabel)

    if alone:
        plt.tight_layout()

    try:
        fg.canvas.manager.set_window_title(
            f"Avg | {feat} | {mapping if mapping is not None else 'meso'} | nclus={n_clu}"
        )
    except Exception:
        pass

    plt.show()


def plot_xy_datashader(
    mapping: str = "kmeans",
    nclus: int = 20,
    feat: str | None = None,
    cv: bool = False,
    agg: str = "count",
    width: int = 900,
    height: int = 900,
    xlim=None,
    ylim=None,
    background: str = "white",
    alpha: int = 255,
    color_key: dict | None = None,
    cmap=None,
    how: str = "eq_hist",
    export_png: str | None = None,
):
    """
    Fast 2D rendering of many points using datashader from a regional_group-style dict.

    Expects r to contain at least:
      r['xyz'] : (N,3) or (N,>=2) array (x,y, z...)
    And optionally (depending on mapping):
      r['acs']   : (N,) cluster labels (ints or strings)
      r['ids']   : (N,) region labels (strings)
      r['cols']  : (N,) per-point colors (hex strings or RGBA tuples)
      r[feat]    : (N,) numeric values for continuous shading

    Parameters
    ----------
    mapping:
      - "none" / "count": density image (counts)
      - "acs": categorical by r['acs']
      - "ids": categorical by r['ids'] (or region labels)
      - "cols": categorical using r['acs'] (or r['ids']) but colored by r['cols']
      - "feat": continuous color by r[feat] (requires feat)
    agg:
      - "count" (default), "mean", "sum", "max", "min" (used only for continuous when mapping='feat')
    how:
      - datashader transfer function for normalization: "eq_hist", "log", "cbrt", or None

    Returns
    -------
    img : datashader.transfer_functions.Image (PIL-like)
    """

    r = regional_group_meso(
            mapping=mapping,               # keep same semantics as your pipeline
            cv=cv,
            nclus=nclus,
        )

    xyz = np.asarray(r["xyz"])
    if xyz.ndim != 2 or xyz.shape[1] < 2:
        raise ValueError(f"r['xyz'] must be (N,>=2), got {xyz.shape}")

    x = xyz[:, 0].astype(np.float32, copy=False)
    y = xyz[:, 1].astype(np.float32, copy=False)

    df = pd.DataFrame({"x": x, "y": y})

    # ---- choose category / value columns ----
    mode = mapping.lower()

    cat_col = None
    val_col = None

    if mode in ("none", "count"):
        pass

    elif mode == "acs":
        if "acs" not in r:
            raise KeyError("mapping='acs' requires r['acs']")
        df["cat"] = pd.Categorical(np.asarray(r["acs"]).astype(str))
        cat_col = "cat"

    elif mode == "ids":
        if "ids" not in r:
            raise KeyError("mapping='ids' requires r['ids']")
        df["cat"] = pd.Categorical(np.asarray(r["ids"]).astype(str))
        cat_col = "cat"

    elif mode == "cols":
        # use labels from acs if present else ids; colors come from r['cols']
        if "cols" not in r:
            raise KeyError("mapping='cols' requires r['cols']")
        if "acs" in r:
            labels = np.asarray(r["acs"]).astype(str)
        elif "ids" in r:
            labels = np.asarray(r["ids"]).astype(str)
        else:
            raise KeyError("mapping='cols' requires r['acs'] or r['ids'] to define categories")

        cols = np.asarray(r["cols"])
        if cols.shape[0] != labels.shape[0]:
            raise ValueError(f"r['cols'] length {cols.shape[0]} != labels length {labels.shape[0]}")

        df["cat"] = pd.Categorical(labels)
        cat_col = "cat"

        # build color_key from first occurrence per category if not provided
        if color_key is None:
            color_key = {}
            for lab, col in zip(labels, cols):
                if lab not in color_key:
                    # accept hex strings; if RGBA tuples, convert to hex-ish via matplotlib if available
                    if isinstance(col, str):
                        color_key[lab] = col
                    else:
                        try:
                            from matplotlib.colors import to_hex
                            color_key[lab] = to_hex(col)
                        except Exception:
                            # fall back: datashader may accept tuples
                            color_key[lab] = col

    elif mode == "feat":
        if feat is None:
            raise ValueError("mapping='feat' requires feat='name_in_r'")
        if feat not in r:
            raise KeyError(f"feat='{feat}' not in r")
        v = np.asarray(r[feat]).astype(np.float32, copy=False)
        if v.ndim != 1 or v.shape[0] != df.shape[0]:
            raise ValueError(f"r[{feat}] must be (N,), got {v.shape}")
        df["v"] = v
        val_col = "v"

    else:
        raise ValueError("mapping must be one of: 'count','acs','ids','cols','feat'")

    # ---- canvas ----
    cvs = ds.Canvas(
        plot_width=int(width),
        plot_height=int(height),
        x_range=xlim,
        y_range=ylim,
    )

    # ---- aggregate ----
    if cat_col is not None:
        agg_img = cvs.points(df, "x", "y", ds.count_cat(cat_col))
        img = tf.shade(agg_img, color_key=color_key, how=how, min_alpha=0, alpha=alpha)
        img = tf.set_background(img, background)
    else:
        if val_col is None:
            agg_img = cvs.points(df, "x", "y", ds.count())
            img = tf.shade(agg_img, how=how, cmap=cmap, alpha=alpha)
            img = tf.set_background(img, background)
        else:
            # continuous aggregation
            agg_fun = agg.lower()
            if agg_fun == "mean":
                red = ds.mean(val_col)
            elif agg_fun == "sum":
                red = ds.sum(val_col)
            elif agg_fun == "max":
                red = ds.max(val_col)
            elif agg_fun == "min":
                red = ds.min(val_col)
            elif agg_fun == "count":
                red = ds.count()
            else:
                raise ValueError("agg must be one of: count, mean, sum, max, min")
            agg_img = cvs.points(df, "x", "y", red)
            img = tf.shade(agg_img, how=how, cmap=cmap, alpha=alpha)
            img = tf.set_background(img, background)

    if export_png is not None:
        img.to_pil().save(export_png)




def plot_xy_meso_png(
    mapping: str = "kmeans",
    *,
    nclus: int = 100,
    cv: bool = False,
    savepath: str | Path | None = None,   # default handled below
    dpi: int | None = None,
    xcol: int = 0,
    ycol: int = 1,
    to_mm: bool = True,
    width: int = 3000,
    height: int = 2500,
    background: str = "white",
    how: str = "eq_hist",
    point_size: int = 1,
    x_range=None,
    y_range=None,
    stack_dir=None,
    cache_dir=None,
    filter_neurons: bool = True,
    title: str | None = None,
    title_color=(0, 0, 0),
    title_xy=(20, 20),
    title_fontsize: int = 28,
):
    """
    Save a static PNG of mesoscope XY locations rasterized with Datashader.

    Default save location:
        pth_meso / 'figs' / f'xy_{mapping}_nclus{nclus}_cv{cv}.png'
    """

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


    # ---------- load ----------
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

    xyz = np.asarray(r["xyz"])
    scale = 1000.0 if to_mm else 1.0
    x = (xyz[:, xcol] / scale).astype(np.float32, copy=False)
    y = (xyz[:, ycol] / scale).astype(np.float32, copy=False)

    labels = np.asarray(r["acs"])
    cols = np.asarray(r["cols"], dtype=np.float32)
    if cols.shape[1] != 4:
        raise ValueError("r['cols'] must be (N,4) RGBA.")

    N = int(x.shape[0])

    if title is None:
        title = f"mapping={mapping}   nclus={nclus}   cv={cv}   N={N}"

    if x_range is None:
        x_range = (float(np.nanmin(x)), float(np.nanmax(x)))
    if y_range is None:
        y_range = (float(np.nanmin(y)), float(np.nanmax(y)))

    # ---------- color key (fast, first occurrence per label) ----------
    s = pd.Series(labels)
    first_idx = s.groupby(s, sort=False).head(1).index.to_numpy()
    uniq = s.iloc[first_idx].to_numpy()

    color_key = {}
    for idx, lab in zip(first_idx, uniq):
        rgba8 = np.clip(np.round(cols[int(idx)] * 255), 0, 255).astype(np.uint8)
        color_key[lab] = "#{:02x}{:02x}{:02x}".format(*rgba8[:3])

    df = pd.DataFrame(
        dict(
            x=x,
            y=y,
            label=pd.Categorical(labels, categories=pd.Index(uniq)),
        )
    )

    # ---------- rasterize ----------
    cvs = ds.Canvas(
        plot_width=int(width),
        plot_height=int(height),
        x_range=x_range,
        y_range=y_range,
    )

    agg = cvs.points(df, "x", "y", agg=ds.count_cat("label"))
    img = tf.shade(agg, color_key=color_key, how=how)

    # point size (spread)
    if point_size > 1:
        img = tf.spread(img, px=int(point_size))

    if background is not None:
        img = tf.set_background(img, background)

    # ---------- title annotation via PIL ----------
    pil = img.to_pil()
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", title_fontsize)
    except Exception:
        font = ImageFont.load_default()

    draw.text(tuple(title_xy), title, fill=title_color, font=font)

    # ---------- save ----------
    if dpi is None:
        pil.save(savepath, format="PNG", optimize=True)
    else:
        meta = PngImagePlugin.PngInfo()
        meta.add_text("dpi", str(int(dpi)))
        pil.save(
            savepath,
            format="PNG",
            optimize=True,
            dpi=(int(dpi), int(dpi)),
            pnginfo=meta,
        )

    print(f"Saved PNG → {savepath}")



import numpy as np
import datoviz as dv


def plot_xy_datoviz(
    mapping: str = "kmeans",
    *,
    nclus: int = 100,
    cv: bool = False,
    xcol: int = 0,
    ycol: int = 1,
    to_mm: bool = True,
    stack_dir=None,
    cache_dir=None,
    filter_neurons: bool = True,
    point_size: float = 3.0,
    width: int = 1200,
    height: int = 900,
    background: str = "white",
):
    r = regional_group_meso(
        mapping,
        stack_dir=stack_dir,
        cache_dir=cache_dir,
        filter_neurons=filter_neurons,
        cv=cv,
        nclus=nclus,
    )

    xyz = np.asarray(r["xyz"], dtype=np.float32)
    scale = 1000.0 if to_mm else 1.0

    # IMPORTANT: axes.normalize requires float64 in this datoviz build
    x = (xyz[:, xcol] / scale).astype(np.float64, copy=False)
    y = (xyz[:, ycol] / scale).astype(np.float64, copy=False)

    col = np.asarray(r["cols"], dtype=np.float32)
    if col.ndim != 2 or col.shape[1] != 4:
        raise ValueError("r['cols'] must be RGBA (N,4).")

    color = np.clip(np.round(col * 255.0), 0, 255).astype(np.uint8, copy=False)
    color[:, 3] = 255  # opaque

    n = x.shape[0]
    size = np.full(n, float(point_size), dtype=np.float32)

    xlim = (float(np.nanmin(x)), float(np.nanmax(x)))
    ylim = (float(np.nanmin(y)), float(np.nanmax(y)))

    app = dv.App(background=background)
    fig = app.figure(int(width), int(height))
    panel = fig.panel()
    axes = panel.axes(xlim, ylim)

    pos_ndc = axes.normalize(x, y)  # float64 input required

    visual = app.point(
        position=pos_ndc,
        color=color,
        size=size,
    )
    panel.add(visual)

    app.run()
    app.destroy()


