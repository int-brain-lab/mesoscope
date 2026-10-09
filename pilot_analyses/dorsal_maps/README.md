# Dorsal cortex maps: mesoscope vs BWM

Side-by-side dorsal cortex maps (mesoscope top row, BWM Neuropixels bottom
row). Each isocortex neuron goes into one ML-AP hexagonal bin (0.4 mm vertex
to vertex). This is a clean rebuild of `drawing.svg`; the original code was
not found.

| figure (`output/`) | content |
|---|---|
| `dorsal_counts` | neurons per bin; sessions with >= 10 neurons per bin |
| `dorsal_decoding_n10` / `_all` (+ `_scatter`: per-bin BWM vs meso) | decoding of stim side, choice, feedback (balanced accuracy) and wheel speed (R^2) per (session, bin), averaged over sessions; 10 random neurons / all neurons |
| `dorsal_tuning` / `_unfolded` | fraction of neurons significantly tuned to stim side, choice, feedback; hemispheres folded / as recorded (`_unfolded` is the `drawing.svg` replica) |

Each figure comes as `<name>.pdf` (panels plus LaTeX caption, compiled with
tectonic), `<name>_panels.pdf` and `<name>.tex`. Headline numbers are in
`output/dorsal_maps_summary.csv`.

## Pipeline

Run inside the `iblenv` conda environment, in this order:

```
python extract_meso.py   # canonical sessions -> FEATURE_ROOT/meso/<eid>.npz  (downloads; memory-mapped, ROI chunks)
python extract_bwm.py    # BWM insertions with isocortex units -> FEATURE_ROOT/bwm/<pid>.npz
python tuning.py         # output/tuning_<modality>.pqt, one row per neuron (GPU)
python decode.py         # output/decoding_<modality>.pqt, one row per session x bin x target (GPU)
python maps.py           # figures + summary
```

`FEATURE_ROOT` is `~/Downloads/ONE/alyx.internationalbrainlab.org/dorsal_maps`.
Signal and spike files downloaded by the extraction are deleted afterwards
to save disk space.

- `common.py`: windows, trial selection, hex binning, the tuning test and the
  sklearn reference decoders.
- `gpu_decode.py`: batched torch decoders. They match the sklearn references
  at tight solver tolerance.
- `sessions.py`: loads the per-session feature files.

## Choices and caveats

- **Data.** Meso uses 153 of the 165 sessions in `canonical_sessions.txt`.
  The other 12 failed: 9 (SP072, SP075, SP076) have no region or position
  datasets on Alyx, and 3 have `mpci.times` / signal length mismatches.
  BWM uses the 2023_12 release (`bwm_query`): 351 insertions in 288
  sessions, good units only.
- **Region labels.** Neurons are restricted to Cosmos Isocortex. Meso
  labels are the latest on Alyx (histology `mpciROIs.brainLocationIds_ccf_2017`
  when present, else `_estimate`). Only 25 of 165 sessions have histology
  labels; see `output/meso_region_sources.csv`.
- **Windows.** BWM stim [0, 0.1] s after stimulus onset, choice [-0.1, 0] s
  before movement onset, feedback [0, 0.2] s. Meso windows are 0.3 s long
  (frame period about 0.2 s), read from the linearly interpolated
  deconvolved trace with per-ROI time offsets.
- **Tuning.** `mpciROIs.taskTunedP` is not on Alyx and Yanliang's BWM
  p-values are not in the aggregates. Both modalities therefore use a GPU
  re-implementation of `brainwidemap` `get_{stim,choice,feedback}_time_shuffle`.
  On 60 neurons it agreed with the original on p < 0.05 in 97-100% of
  cases.
- **Decoding.** Sessions need at least 50 trials and at least 10 neurons in
  a bin. Use the "all neurons" maps with care: meso bins hold hundreds of
  neurons, so their scores are not comparable to BWM. The `n10` version
  makes the two rows comparable.
