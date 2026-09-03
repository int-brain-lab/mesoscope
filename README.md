# Mesoscope: loading, chronic tracking, and decoding

Utilities for working with IBL mesoscope sessions: fast per-session embedding/caching, cross-day neuron alignment from ROICaT CSVs, and decoding analyses (neuron-dropping curves; cross-session generalization).

> You need an IBL ONE setup and access to mesoscope datasets.

---

## Quick install

Tested with Python ≥3.10.

```bash
# conda (recommended)
conda create -n iblenv python=3.10 -y
conda activate iblenv

# core deps
pip install one-api iblutil iblatlas brainbox rastermap numpy scipy matplotlib scikit-learn

# optional: progress bars, plotting niceties
pip install tqdm
```

Check ONE auth:

```python
from one.api import ONE
one = ONE()
one.search()  # should return without auth errors
```

---

## Repo layout

- `meso_loader.py` — clean, dependency-light loader for basic per-session ROI data (signal, per-ROI time, region, location); **start here**
- `sanity_checks.py` — smoke test for a loaded session: does population activity rise after movement onset
- `meso.py` — load or embed a mesoscope session into a cached dict (Rastermap ordering, ROI signals/metadata)
- `meso_chronic.py` — mirror `mpciROIs.clusterUIDs.csv` (ROICaT) and align tracked neurons across days
- `meso_decode.py` — trial-window feature extraction, neuron-dropping curves (NDCs), and cross-session decoding

---

## Caching and paths

By default files are cached under:

```
<ONE.cache_dir>/meso/
└── data/
    └── <eid>.npy         # per-session dict (signals, times, regions, isort, etc.)
└── decoding/
    ├── <prefix>_<target>.pkl         # group NDC payloads
    └── pair_summaries/<subject>/
        ├── res/<prefix>_<target>.pkl # pairwise payloads
        └── imgs/<prefix>.png         # saved figures
```

`<prefix>` is derived from the EID list (date-sorted), `<target> ∈ {choice, feedback, stimulus, block}`.

---

## Basic session loading (clean loader, start here)

For most analyses you just need per-ROI signal, per-ROI corrected time, brain region, and 3D location for a session — without pulling in matplotlib/rastermap. Use `meso_loader.py`:

```python
from meso_loader import load_mesoscope_session

sess = load_mesoscope_session(eid)   # creates a default ONE() if none is given

sess.roi_signal      # (n_rois, n_timepoints) float32
sess.roi_times        # (n_rois, n_timepoints) float32, per-ROI time-corrected
sess.region_labels     # (n_rois,) Allen acronyms
sess.region_ids         # (n_rois,) Allen CCF structure ids
sess.xyz                 # (n_rois, 3) mm, MLAPDV estimate
sess.fov                  # (n_rois,) originating FOV, e.g. 'FOV_00'
sess.signal_type           # 'deconvolved' or 'fluorescence'
```

Every FOV in the session is loaded and stacked into these flat, ROI-indexed arrays. Results are cached to `<ONE.cache_dir>/meso/basic_data/<eid>_filter_<filter_neurons>.npz`; pass `rerun=True` to force a fresh load, or `use_cache=False` to skip caching entirely.

### Default loading decisions

These are the defaults `load_mesoscope_session` makes for you, and why:

- **Neuron-only ROIs by default** (`filter_neurons=True`). Non-neuronal / artifact ROIs (`mpciROITypes == 0`) are dropped before anything is stacked, so `roi_signal.shape[0]` is directly usable as a neuron count. Pass `filter_neurons=False` to keep every segmented ROI.
- **Deconvolved signal preferred over raw fluorescence.** If `mpci.ROIActivityDeconvolved` exists it's used; otherwise the loader falls back to `mpci.ROIActivityF` and records which one it used in `sess.signal_type`. Deconvolved traces are closer to spiking activity and are what this repo's decoding/embedding code expects.
- **Histology-registered brain regions preferred over the pipeline estimate.** `mpciROIs.brainLocationIds_ccf_2017` (final, histology-based) is used when present; the loader falls back to `mpciROIs.brainLocationIds_ccf_2017_estimate` (available before histology is done) otherwise.
- **Per-ROI corrected timestamps, not one shared clock.** Mesoscope FOVs are scanned with a small, ROI-dependent sub-frame time offset (`mpciStack.timeshift`); `roi_times` bakes this in per-ROI rather than giving every ROI the same frame-time vector. If you need a single shared time axis for speed (e.g. windowed aggregation), `roi_times[0]` is a good approximation — the offsets are sub-frame and negligible over typical event windows (same approximation used in `meso.compute_sparseness`; see `sanity_checks.py` for a check that this holds up in practice).
- **All FOVs are loaded and stacked, session-wide** — there's no partial-FOV mode. If you only need one FOV, load it directly with `one.load_object(eid, ..., collection='alf/FOV_XX')`.
- **A signal-type mismatch across FOVs raises**, rather than silently mixing deconvolved and raw-fluorescence traces in one array — that's a strong signal the session's suite2p/deconvolution run is inconsistent and worth checking before use.
- **Caching is on by default and keyed by `(eid, filter_neurons)`.** The cache directory (`meso/basic_data/`, plain `.npz`) is deliberately separate from the Rastermap-based cache in `meso.py` (`meso/data/`, pickled `.npy`) — the two loaders store different schemas and shouldn't be mixed up.

### Sanity-checking a loaded session

`sanity_checks.py` runs a quick physiological smoke test: population activity should rise shortly after movement onset. It samples random trials, compares mean population activity just before vs. just after each trial's `firstMovement_times`, and reports the fraction of trials that increased plus a paired Wilcoxon signed-rank test.

```python
from sanity_checks import check_motion_onset_response

result = check_motion_onset_response(eid, n_trials=20)
# [motion-onset check] <eid>: 20 trials, 90% increased, mean diff=3.57, wilcoxon p=1.9e-05 -> PASS
```

This isn't a scientific claim about any specific brain region — it's a fast way to catch a broken time alignment or a bad signal fallback after touching the loader.

---

## Load or embed a session (Rastermap + region metadata)

```python
from meso import load_or_embed, plot_raster

eid = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
rr = load_or_embed(eid)     # loads cached dict or computes & caches
plot_raster(eid, bg="regions")
```

The cached dict includes at least:
- `roi_signal` (N×T, float; deconvolved if available, else fluorescence)
- `roi_times` (per-FOV)
- `isort` (Rastermap row order)
- `region_labels`, `region_colors`
- `xyz`, ROI metadata

---

## Cross-day alignment (tracked same neurons)

```python
from pathlib import Path
from one.api import ONE
from meso_chronic import match_tracked_indices_across_sessions

one = ONE()
eids = ["…", "…", "…"]  # chronological
idx_map = match_tracked_indices_across_sessions(one, eids[0], eids[1:], roicat_root=Path.home()/"chronic_csv")
# idx_map[eid] gives integer indices selecting the shared cells in each session
```

Mirror (or point to) your `mpciROIs.clusterUIDs.csv` tree under `~/chronic_csv/ROICaT/<subject>/<date>/…`.

---

## Decoding I: Neuron-dropping curves (NDCs)

**Goal:** quantify how decoding performance scales with #neurons and summarize k* (smallest k within Δ of ceiling).

```python
from pathlib import Path
from meso_decode import neuron_dropping_curves_cached

eids = ["…", "…", "…"]  # ≥2 sessions
out = neuron_dropping_curves_cached(
    eids=eids,
    targets=("choice","feedback","stimulus","block"),
    cache_dir=Path.home()/ "FlatIron/meso/decoding",
    ks=(8,16,32,64,128,256,512,1024,2048),
    R=50,
    metric="auc",
    cv_splits=5,
    n_label_shuffles=100,   # builds a shuffle band
    equalize_trials=True,   # same #trials across sessions per target
)
# If cache exists for the derived prefix, it plots from cache only.
```

**What features are used?** For each trial, mean ROI activity in a short window around an event (per target):
- choice: `firstMovement_times`, window `(-0.1, 0.0)` s
- feedback: `feedback_times`, `(0.0, 0.20)` s
- stimulus: `stimOn_times`, `(0.0, 0.10)` s
- block (context): `stimOn_times`, `(-0.40, -0.10)` s (pre-stimulus)

Trials are z-scored per neuron across trials and downsampled to balance classes.

---

## Decoding II: Train on first session, test on the rest

```python
from meso_decode import run_cross_session_train_first_test_rest, plot_cross_session_train_first_test_rest

payload = run_cross_session_train_first_test_rest(
    subject="SP072",
    min_sessions=10,
    min_k_shared=1000,            # require at least this many globally shared neurons
    targets=("choice","feedback","stimulus","block"),
    dimreduce="pca", dim_k=512,   # PCA is capped by training fold size; see notes
    n_shuffle=200,
)
plot_cross_session_train_first_test_rest(payload)
```

Pipeline:
1. Choose a date-ordered subset with robust shared-neuron count
2. Build trial-window features using the **same** shared indices for all sessions
3. Tune a linear decoder on the first session (grid over `C`, optional elastic-net)
4. Report cross-validated train accuracy and test accuracy per later session, with a permutation control

---

## Notes & troubleshooting

- **PCA safety cap:** During inner CV on the training session, the PCA component count is automatically clipped to `≤ min(n_features, min_training_fold_size−1)` to avoid `ValueError: n_components must be ≤ …`
- **“no finite event times” / empty windows:** happens if the chosen event vector contains NaNs after trial filtering; verify `get_win_times(eid)` and the target’s event field
- **Low trial counts after filtering:** class balancing and removal of NaN rows can reduce `n`. Inspect `Counter(trials['choice'])` vs. the post-filter sizes
- **Cache location:** override with the `cache_dir` argument if you don’t want to use `<ONE.cache_dir>/meso/decoding`

---

## Citation

If you use these tools, please cite the IBL Brain-Wide Map dataset and mesoscope methods papers as appropriate, along with this repository.

---

## License

MIT (unless otherwise stated in file headers)
