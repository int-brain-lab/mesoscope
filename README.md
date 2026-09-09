# Mesoscope: basic session loading

A clean, dependency-light loader for IBL mesoscope sessions, plus the curated list of sessions ("canonical sessions") this repo's analyses are run on.

> You need an IBL ONE setup and access to mesoscope datasets.

Further exploratory/WIP analyses (Rastermap embedding, cross-day tracking, decoding, ...) live in `pilot_analyses/` and are not covered by this README — read their own docstrings.

---

## Quick install

Tested with Python ≥3.10.

```bash
conda create -n iblenv python=3.10 -y
conda activate iblenv
pip install one-api iblutil iblatlas numpy scipy
```

Check ONE auth:

```python
from one.api import ONE
one = ONE()
one.search()  # should return without auth errors
```

---

## Basic session loading

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
- **Deconvolved signal preferred over raw fluorescence.** If `mpci.ROIActivityDeconvolved` exists it's used; otherwise the loader falls back to `mpci.ROIActivityF` and records which one it used in `sess.signal_type`. Deconvolved traces are closer to spiking activity.
- **Histology-registered brain regions preferred over the pipeline estimate.** `mpciROIs.brainLocationIds_ccf_2017` (final, histology-based) is used when present; the loader falls back to `mpciROIs.brainLocationIds_ccf_2017_estimate` (available before histology is done) otherwise.
- **Per-ROI corrected timestamps, not one shared clock.** Mesoscope FOVs are scanned with a small, ROI-dependent sub-frame time offset (`mpciStack.timeshift`); `roi_times` bakes this in per-ROI rather than giving every ROI the same frame-time vector. If you need a single shared time axis for speed (e.g. windowed aggregation), `roi_times[0]` is a good approximation — the offsets are sub-frame and negligible over typical event windows (see `sanity_checks.py` for a check that this holds up in practice).
- **All FOVs are loaded and stacked, session-wide** — there's no partial-FOV mode. If you only need one FOV, load it directly with `one.load_object(eid, ..., collection='alf/FOV_XX')`.
- **Only the specific datasets needed are downloaded** (via `one.load_dataset` per file), not whole ALF objects — `one.load_object(eid, 'mpci', ...)` would also pull `mpci.ROIActivityF.npy` and `mpci.ROINeuropilActivityF.npy` (hundreds of MB each) even when the deconvolved trace is used, and `one.load_object(eid, 'mpciROIs', ...)` would pull the ROI mask arrays we never use.
- **A signal-type mismatch across FOVs raises**, rather than silently mixing deconvolved and raw-fluorescence traces in one array — that's a strong signal the session's suite2p/deconvolution run is inconsistent and worth checking before use.
- **Caching is on by default and keyed by `(eid, filter_neurons)`**, stored as plain `.npz` under `meso/basic_data/`.

### Sanity-checking a loaded session

`sanity_checks.py` runs a quick physiological smoke test: population activity should rise shortly after movement onset. It samples random trials, compares mean population activity just before vs. just after each trial's `firstMovement_times`, and reports the fraction of trials that increased plus a paired Wilcoxon signed-rank test.

```python
from sanity_checks import check_motion_onset_response

result = check_motion_onset_response(eid, n_trials=20)
# [motion-onset check] <eid>: 20 trials, 90% increased, mean diff=3.57, wilcoxon p=1.9e-05 -> PASS
```

This isn't a scientific claim about any specific brain region — it's a fast way to catch a broken time alignment or a bad signal fallback after touching the loader.

---

## Canonical sessions

`canonical_sessions.txt` is the curated list of mesoscope sessions used by this repo's analyses: a single line of comma-separated `subject/date/number` session paths (165 sessions as of writing, spanning subjects SP037–SP081).

```python
from one.api import ONE

one = ONE()
session_paths = [p.strip() for p in open("canonical_sessions.txt").read().split(",") if p.strip()]
eids = [one.path2eid(p) for p in session_paths]
```

Not every canonical session has fully processed mesoscope data registered yet — `load_mesoscope_session` raises `FileNotFoundError` (or `ValueError` if no FOVs are found) for a session whose suite2p/ALF extraction hasn't completed, so wrap batch loading in a `try`/`except` when iterating over the full list.

---

## License

MIT (unless otherwise stated in file headers)
