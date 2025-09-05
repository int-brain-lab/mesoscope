# Mesoscope data loading and raster plotting

This README documents the **data-loading pipeline** and the **raster plot** utilities in the accompanying script, `meso.py`. Here are two functions to get you started:

1. `embed_meso(eid)` + `load_or_embed(eid)`: loading, preprocessing, and caching mesoscope ROI activity and metadata from the IBL/ONE backend.
2. `plot_raster(eid, ...)`: producing a Rastermap-sorted ROI activity raster with optional region-color background and an overlaid wheel-velocity trace.

> The code assumes a working [ONE API](https://int-brain-lab.github.io/ONE/) configuration and access credentials for IBL data.

## Environment and dependencies

Python packages used by the relevant functions:

- `one.api` (ONE client) and `brainbox` (IBL helpers)
- `iblatlas` for region IDs and colors (Allen CCF 2017)
- `numpy`, `matplotlib`
- `rastermap` (for low‑dimensional embedding/sorting of ROIs)
- `scipy` (only indirectly used elsewhere in the file)
- `pathlib`

Minimal install (conda-like environment):

```bash
pip install one-api iblutil iblatlas ibl-neuropixel-brainbound \
            brainbox rastermap numpy matplotlib scipy
```

> **Note:** IBL libraries are typically installed from the IBL org GitHub or dedicated wheels. Make sure your ONE profile is configured (e.g., `one.set_base_url`, `one.login`) and you can `one.search()` successfully before running the script.


---

## Directory layout and cache

On import, the script creates a root cache under your ONE cache directory:

```
<ONE.cache_dir>/meso/
├── data/              # per-eid cached numpy dictionaries
└── *                  # figures saved by plotting functions
```

The function `embed_meso(eid)` writes a cache file:

```
<ONE.cache_dir>/meso/data/<eid>.npy
```

This `.npy` contains a Python dict (see below). Subsequent calls use `load_or_embed(eid)` to avoid re-downloading and re-embedding.


---

## What does `embed_meso` load and compute?

**Input:** an experiment ID (`eid`, UUID string) for a mesoscope session.

**Steps:**

1. **Enumerate FOVs** for the session under `alf/FOV_*` and load per‑FOV objects:
   - `mpci`, `mpciROIs`, `mpciROITypes`, `mpciStack`

2. **Extract ROI metadata and signals per FOV:**
   - **Region IDs → acronyms:** using Allen Atlas 2017 IDs from `mpciROIs` (key fallback logic distinguishes between exact and estimated IDs). Convert to acronyms with `AllenAtlas().regions.id2acronym`.
   - **Region hexcolors:** taken from `iblatlas.regions` table for visualization.
   - **Frame times and per‑ROI time shifts:** using `mpci.times` and `mpciStack.timeshift` together with each ROI’s `stackPos` to compute the **per‑ROI aligned time base** (`roi_times`).
   - **Signals:** use `mpci.ROIActivityDeconvolved` (transposed to shape `(n_rois, n_time)`). A boolean mask from `mpciROITypes` filters **neuronal** ROIs only.

3. **Stack across FOVs:** vertically concatenate ROIs across all FOVs for a session; **time base is shared** across FOVs.

4. **Compute Rastermap ordering:** fit a `Rastermap` model on the ROI matrix to derive `isort` (row order). Parameters in the script:
   - `n_PCs=100, n_clusters=30, locality=0.75, time_lag_window=5, bin_size=1`

5. **Persist to disk:** save a dict with signals, times, region labels/colors, the sort index, and ROI coordinates (`xyz`).


---

## Cached data structure (`.npy` file)

The saved dictionary has the following keys (all `np.ndarray` unless noted):

- `roi_signal`: shape `(n_rois, n_time)`, deconvolved activity, neuron‑only, stacked across FOVs.
- `roi_times`: shape `(n_timepoints, n_rois)` *or* `(n_rois, n_time)` depending on the original arrangement; the script builds a tiled per‑ROI time base with ROI‑specific shifts and later uses `roi_times[0]` as the **global time axis**. For plotting, the code expects `roi_times` such that `roi_times[0]` is a 1D vector of timestamps (length `n_time`).
- `region_labels`: shape `(n_rois,)`, Allen acronyms (e.g., `VISp`, `VISl`, …).
- `region_colors`: shape `(n_rois,)`, hex colors per ROI (from Allen table at load time; plotting later remaps to higher‑contrast colors).
- `isort`: shape `(n_rois,)`, integer permutation for Rastermap ordering.
- `xyz`: shape `(n_rois, 3)`, estimated anatomical coordinates (μm).

> **Important:** The raster plot uses `roi_signal[isort]` and `roi_times[0]` as the x‑axis. If you change the time‑base construction, adjust the plot accordingly.


---

## Fast path: `load_or_embed`

```python
def load_or_embed(eid):
    fpath = Path(pth_meso, 'data', f"{eid}.npy")
    if fpath.exists():
        rr = np.load(fpath, allow_pickle=True).item()
    else:
        rr = embed_meso(eid)
    return rr
```

- **Behavior:** If the cache exists, return it. Otherwise, run `embed_meso(eid)` (which computes and saves the cache), then return the in‑memory dict.
- **Tip:** Use this in any downstream analysis to avoid redundant downloads and embeddings.


---

## Raster plotting: `plot_raster`

This function renders a Rastermap‑sorted ROI activity raster for a given `eid`. It can optionally overlay a **wheel‑velocity trace** and a **background** that encodes either ROI region identity or mean firing rate.

### Inputs and options

```python
plot_raster(
    eid,
    bg='regions',        # {'regions', 'firing_rate'} or None
    alpha_bg=0.3,        # background opacity
    alpha_data=0.5,      # activity heatmap opacity
    interp='none',       # imshow interpolation
    restr=True,          # restrict to a 1-minute excerpt
    rsort=True,          # apply Rastermap ordering (isort)
    scaling=True         # per-ROI robust scaling
)
```

- **`rsort`**: when `True`, the function applies the saved Rastermap permutation to `roi_signal`, `region_labels`, and `region_colors` so rows are ordered by the embedding.
- **`scaling`**: robustly rescales each ROI to `~[0,1]` using its 20th and 99th percentiles. This improves legibility and comparability across ROIs.
- **`interp`**: passed to `matplotlib.pyplot.imshow`; `"none"` keeps a pixelated raster (recommended for discrete time bins).
- **`alpha_*`**: control transparency of foreground data and background panels.

### Signal scaling

When `scaling=True`:

```python
p20 = np.percentile(rr['roi_signal'], 20, axis=1, keepdims=True)
p99 = np.percentile(rr['roi_signal'], 99, axis=1, keepdims=True)
rr['roi_signal'] = (rr['roi_signal'] - p20) / (p99 - p20)
```

This preserves intra‑ROI dynamics while dampening outliers. The raster colormap is grayscale (`'gray_r'`) with `vmin = min` and `vmax = 0.1` to emphasize low‑to‑moderate activity fluctuations.

### Time restriction (`restr=True`)

To standardize figures and reduce rendering time, the function optionally plots a **1‑minute excerpt** starting at the **end of the first third** of the recording:

1. Compute sampling rate from the `roi_times` extent.
2. Convert 60 seconds to number of bins.
3. Slice all arrays in the unified time window.

Disable by setting `restr=False` to plot the entire session.


### Background layers

Two optional backgrounds:

- **`bg='regions'`** (default): Each ROI row receives a solid color strip (underlay) encoding its **region label**. For plot readability, the script maps unique regions to a **distinct, high‑saturation palette** (rather than default Allen colors). A legend lists regions and counts.
- **`bg='firing_rate'`**: Underlay represents the **mean activity** per ROI (row). The scalar mean is repeated across time to form a background image; a colorbar quantifies values.

Set `bg=None` to omit backgrounds entirely.


### Wheel velocity overlay

- Loads `wheel` from ONE (`one.load_object(eid, 'wheel')`).
- Uses `brainbox.behavior.wheel.interpolate_position` to upsample position and `velocity_filtered` to compute a smooth velocity at **250 Hz**.
- Restricts the wheel time series to the plot time window and renders it in a compact top axis aligned with the raster’s x‑axis.

If the session lacks wheel data, this call will raise; see **Troubleshooting** for fallbacks.


### Output image

- The figure is saved automatically to the meso cache folder with a descriptive filename containing `eid`, the set of **unique regions**, and the raster array shape (rows × columns).
- DPI is 300 for manuscript‑quality export.

---

## Typical usage

```python
from meso import plot_raster, load_or_embed

eid = "71e53fd1-38f2-49bb-93a1-3c826fbe7c13"  # example

# Ensure the cache exists (or build it)
rr = load_or_embed(eid)

# Make a region-annotated raster with wheel overlay
plot_raster(
    eid,
    bg='regions',
    alpha_bg=0.25,
    alpha_data=0.6,
    rsort=True,
    scaling=True,
    restr=True
)
```

To switch the background to mean-activity:
```python
plot_raster(eid, bg='firing_rate')
```

To plot the **entire session**:
```python
plot_raster(eid, restr=False)
```


---

## Performance notes and pitfalls

- **Memory footprint:** Mesoscope sessions can have many thousands of ROIs and long recordings. Use `restr=True` for exploratory figures. For full‑length plots, prefer vector‑free formats (PNG over PDF) and consider downsampling in time.
- **Rastermap parameters:** The defaults (`n_PCs=100`, `n_clusters=30`, `locality=0.75`, `time_lag_window=5`, `bin_size=1`) are a compromise between structure discovery and speed. Tuning them affects `isort`; cache will need regeneration if you change them.
- **Region colors:** At load time, the script records Allen hex colors per ROI. For plotting, it **remaps** regions to a distinct palette via `load_distinct_bright_colors` to increase perceptual separability across many visual areas.
- **Time base:** The script constructs a per‑ROI shifted time base and **uses `roi_times[0]`** as a common x‑axis. Ensure this remains valid for your data; if time shifts vary across ROIs, you may prefer to resample onto a global time grid before saving.
- **Wheel data:** Some sessions may lack `wheel` or have irregular timestamps. Guard code or `try/except` wrappers can improve robustness.


---


## Acknowledgements

- International Brain Laboratory mesoscope task force members, lead by Samuel Picard.
