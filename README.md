# Mesoscope: data loading, chronic tracking, and decoding

This repository contains utilities for loading mesoscope sessions from the IBL **ONE** backend, caching per‑session data, plotting Rastermap‑sorted activity, and aligning the **same tracked neurons** across days to enable cross‑session analyses and decoding.

> You need a working ONE setup and access to the relevant datasets.

---

## Environment and dependencies

Typical Python stack used by these scripts:

- `one.api` (ONE client) and `brainbox` (IBL helpers)
- `iblatlas` (Allen CCF 2017 region info)
- `numpy`, `matplotlib`, `scipy`
- `scikit-learn` (decoding)
- `rastermap` (embedding / ordering)
- `pathlib`, `dataclasses`

Example install:

```bash
pip install one-api iblutil iblatlas brainbox rastermap numpy matplotlib scipy scikit-learn
```

After installing, make sure you can log in and query:

```python
from one.api import ONE
one = ONE()
one.search()  # should return without auth errors
```

---

## Cache and directory layout

By default, `meso.py` writes per‑session caches under:

```
<ONE.cache_dir>/meso/
└── data/
    └── <eid>.npy      # cached dictionary for that session
```

Figures created by plotting functions are also saved under this tree.

---

## What `embed_meso` does (in `meso.py`)

For a mesoscope experiment ID (`eid`), `embed_meso`:

1. Enumerates `alf/FOV_*` collections and loads:
   - `mpci`, `mpciROIs`, `mpciROITypes`, `mpciStack`
2. Builds per‑ROI metadata:
   - Region acronyms (Allen CCF 2017) and corresponding colors
   - Per‑ROI time bases from `mpci.times` and `mpciStack.timeshift`
   - Deconvolved ROI signals when available (falls back to fluorescence)
   - Masks to keep only neuron ROIs (`mpciROITypes`)
3. Stacks ROIs across FOVs and computes a Rastermap ordering (`isort`)
4. Saves a dict with keys such as:
   - `roi_signal` (N×T), `roi_times`, `region_labels`, `region_colors`, `isort`, `xyz`

### Quick usage

```python
from meso import load_or_embed, plot_raster

eid = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
rr = load_or_embed(eid)           # loads cached dict or computes & caches
plot_raster(eid, bg="regions")    # Rastermap-sorted raster with region background
```

---

## Chronic tracking across days (`meso_chronic.py`)

- Functions to **mirror** `mpciROIs.clusterUIDs.csv` files locally (ROICaT tree) and
- Align tracked neurons across sessions: `match_tracked_indices_across_sessions(...)`

With the mirrored CSVs available, you can produce a mapping from each EID to indices that select the **same** cells, enabling cross‑day comparisons on a fixed population.

---

## Decoding and neuron‑dropping curves (`meso_decode.py`) — brief

**Goal.** Test whether the neural code becomes more **efficient** over sessions by asking: *How many shared neurons are required to decode a behavioral variable (e.g., choice) to near‑ceiling accuracy?*

**Idea.**
1. **Align the same cells** across sessions using `match_tracked_indices_across_sessions(...)`.
2. **Extract trial features**: for each trial, average activity in a short window around an event
   (default `firstMovement_times`, window `[0.0, 0.15]` s for choice for example; can do stimulus and fedback decoding also.).
3. **Decode with cross‑validation**: for each neuron count `k`, sample many random subsets, fit
   a linear logistic decoder (AUC/accuracy), and record performance. Also build a **label‑shuffled
   baseline** (repeat permutations many times) to visualize chance levels.
4. **Summarize** with **neuron‑dropping curves (NDCs)** and report \(k^*\): the smallest `k`
   whose mean score is within Δ (e.g., 0.02 AUC) of that session’s ceiling. A decreasing \(k^*\)
   over time suggests increasing coding efficiency.

# Decoding & Neuron-Dropping Curves (`meso_decode.py`)

This module quantifies how many shared neurons are required to decode behavioral variables and how decoding accuracy scales with population size.  
It computes **neuron-dropping curves (NDCs)** by subsampling neurons and cross-validating a linear decoder.  
By default, **label-shuffled baselines** are computed to visualize chance performance.

---

## Minimal example

```python
from pathlib import Path
from meso_decode import neuron_dropping_curves_cached

# Example session EIDs (replace with your own)
eids = [
    "38bc2e7c-d160-4887-86f6-4368bfd58c5f",
    "74ffa405-3e23-47d9-972b-11bea1c3c2f6",
    "1322edbf-5c42-4b9a-aecd-7ddaf4f44387",
    "20ebc2b9-5b4c-42cd-8e4b-65ddb427b7ff",
]

# Decode up to 3 behavioral variables
out = neuron_dropping_curves_cached(
    eids=eids,
    targets=("choice", "feedback", "stimulus"),  # up to 3
    cache_dir=Path.home() / "ndc_cache",
    event="firstMovement_times",
    win=(0.0, 0.15),                 # seconds relative to event
    ks=(8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    R=50,                            # resamples per k
    metric="auc",                    # or "accuracy"
    cv_splits=5,                     # stratified K-fold CV
    class_balance="downsample",      # balances trial classes
    scaling=True,                    # per-neuron z-score across trials
    equalize_trials=True,            # match trial counts across sessions
    ceiling_mode="maxk",             # session ceiling = max k
    # Shuffled baseline control (enabled by default):
    n_label_shuffles=100,
    shuffle_seed=0,
)
```
The function plots session‑wise NDCs (overlay) and prints a compact table with \(k^*\) and the per‑session ceiling. True curves should lie clearly above the shuffled band; a **leftward shift** of the true curves (or decreasing \(k^*\)) across days indicates improved coding efficiency.

---

## Notes

- Scripts prefer deconvolved activity; they fall back to fluorescence where necessary.
- Trial classes are balanced by default for decoding; you can disable or change the strategy.
- Time bases are per ROI; plotting treats the first ROI’s timeline as the session reference.

---

## License

MIT (see `LICENSE` if present).
