# Dorsal cortex maps: mesoscope vs BWM

Side-by-side dorsal cortex maps (mesoscope top row, BWM Neuropixels bottom
row). Each isocortex neuron goes into one ML-AP hexagonal bin (0.4 mm vertex
to vertex). This is a clean rebuild of `drawing.svg`; the original code was
not found.

| figure (`output/`) | content |
|---|---|
| `dorsal_windows` | analysis windows per variable and modality |
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

# controls (Supplementary Information)
python extract_bwm.py w03                 # BWM with mesoscope windows + frame-sampled features
python tuning.py bwm_w03 && python tuning.py bwm_w03 --features _fr
python decode.py bwm_w03 && python decode.py bwm_w03 --features _fr
python decode.py bwm meso --hex 0.5 --n10-only   # 1 mm bins
python layers.py bwm meso bwm_w03 && python layers.py bwm_w03 --features _fr
python controls.py                        # si_* figures, output/controls_summary.csv

# second round of controls
python selection.py meso bwm              # top-10 / most-active / high-contrast decoding
python selection.py meso --pool 24        # pool-size control
python tuning.py meso bwm --min-contrast 0.25
python controls2.py                       # si_strict_*, si_stim_hotspot, si_spatial_null,
                                          # si_selection, si_high_contrast; output/controls2_summary.csv
python build_report.py                    # report/: LaTeX report (v6), compiled with tectonic
python build_report_v7.py                 # report_v7/: current LaTeX report (v7), compiled with tectonic
```

`report_v7/main.tex` is the single source of the current report: figures are
cited with `\ref`, and `build_report_v7.py` copies the ten figure panels and their
captions (from `output/`, `output/v2/` and `../wheel_decoding2/output/`) into
`report_v7/figures/`. The shared web page renders the compiled `main.pdf`.

`report/` holds the LaTeX report: `main.tex` (text), one folder per main
figure in `report/figures/`, all supplementary figures in `report/si/`.
Figure panels and captions are copied from `output/` by `build_report.py`,
so captions are edited in `maps.py` / `controls.py` / `controls2.py`.
`controls2.py` needs `output/session_subjects.csv` (session -> subject, from
`one.eid2path`).

`FEATURE_ROOT` is `~/Downloads/ONE/alyx.internationalbrainlab.org/dorsal_maps`.
Signal and spike files downloaded by the extraction are deleted afterwards
to save disk space.

- `common.py`: windows, trial selection, hex binning, the tuning test and the
  sklearn reference decoders.
- `gpu_decode.py`: batched torch decoders. They match the sklearn references
  at tight solver tolerance.
- `sessions.py`: loads the per-session feature files.
- `selection.py`: decoding from the 10 most selective neurons of a bin. They
  are picked on the training trials of each outer fold, so held-out trials
  never influence the selection; with shuffled labels the scores are at
  chance. Fold-specific neuron sets are supported by `gpu_decode`.
- `controls2.py`: spatial null for map correlations. Variogram-matched
  surrogate maps (Burt et al. 2020, BrainSMASH "Base" algorithm, in torch).
  On independent smooth random maps over the real bin layout it gives 6%
  false positives at p < 0.05, against 47% for the naive Spearman p.

## Revision v2 (default from 2026-10-15)

`output/v2/` holds the revised maps; they replace the 0.4 mm maps above as the default.

- **1 mm hexagons.** Centre-to-vertex 0.5 mm (`decode_v2.HEX_V2`).
- **Depth-limited BWM (`bwm_pial640`).** Use isocortex units within 640 µm of the pia by the Allen CCF streamline lookup. This keeps 6,125 of 12,890 units in the report cache. BWM at all depths is the reference row.
- **Revised wheel decoding.** See `../wheel_decoding2/README.md`.
  - The wheel targets had encoder overflow jumps in 16 meso sessions; these are fixed (`fix_wheel_glitches.py`, `common.remove_wheel_jumps`).
  - The ridge penalty is chosen by exact leave-one-trial-out error.
  - Features: neural activity of the previous, same and next wheel bin.
  - New target: trial-specific wheel speed (per-bin mean of the training trials removed).
- **All-neuron binary decoding.** For sets with more neurons than trials, it fits in the row space of the trials (`gpu_decode.decode_binary(dual=True)`); this is exact and gives identical scores.
- **20 random neurons per bin, mean over 10 draws** (`decode_v2.N_PICK`, `N_DRAWS`); bins need >= 20 neurons. The first run with 10 neurons x 3 draws is kept in `output/v2/n10_3draws`.
- **Excluded meso sessions** (`decode_v2.excluded_sessions`). Two sessions whose task and imaging clocks are offset (`CLOCK_OFFSET`: bd92d5ef -146 s, 38438b7d -338 s; population-wheel cross-correlation peaks there and everything decodes at chance). Trials outside the imaging period are dropped and sessions with < 50 % coverage excluded (`imaging_coverage.py` -> `FEATURE_ROOT/meso_trial_coverage`).
- **Map display.** Coverage and tuning maps use the ColorCET CET-I3 palette (install
  with `pip install colorcet`). On decoding maps, opacity encodes the number
  of animals per bin, reaching full opacity at four animals. The meso vs
  `bwm_pial640` per-bin correlations are tested against a variogram-matched
  spatial null (`controls2.spatial_test`).

```
python fix_wheel_glitches.py              # patch wheel_y of affected feature files (backup kept)
python make_bwm_depth.py                  # FEATURE_ROOT/bwm_pial640/<pid>.npz
python imaging_coverage.py                # trials covered by imaging, per meso session
python decode_v2.py meso bwm_pial640 bwm     # output/v2/decoding_<modality>_auc.pqt
python maps_v2.py                         # v2_counts, v2_decoding_n20/_all, v2_tuning, v2_summary
python decode_bwm_depth.py                # refit 10-unit BWM decoders by registered VISa layer
python decode_bwm_depth.py --meso-reference  # refit 10-ROI mesoscope VISa decoders
python depth_profile.py                   # si_depth_profile: VISa tuning and decoding by layer
```

## v7 decoding figures

Figures 1 and 2 use 0.5 mm mesoscope hexagons (vertex to vertex) and BWM
dots at the Alyx insertion coordinates, with separate rows for superficial
and all-depth BWM units. The depth-limited row uses the IBL 25 µm Allen
CCF streamline lookup for local pial depth. BWM dots use 80% opacity. Each variable has its
own colour scale shared across rows from chance (0.5 for ROC-AUC and 0
for wheel $R^2$) to the 99th percentile of its plotted scores. Values below
chance use the lowest colour. Binary scores are ROC-AUC calculated from
pooled held-out logistic decision scores. The existing inner tuning criterion
remains balanced accuracy. Use `--cmap NAME` to change the Matplotlib colormap
(default `cool`). The random-20, all-neuron and best-ten maps use the same mesoscope
session--hexagon pairs and BWM insertions for each target, based on valid
scores in all three analyses. Rendering any map requires the score caches
for all three analyses.
The headline table and supplementary bin-wise comparisons use 1 mm bins
with the same superficial BWM selection.

Once the feature files and mesoscope trial coverage masks are present, run:

```sh
python decode_v2.py meso --hex 0.25  # centre-to-vertex radius in mm
python make_bwm_depth.py  # writes bwm_pial640
python decode_probe.py bwm bwm_pial640
python decode_top10.py
python maps_v2.py --report-decoding --cmap plasma
python build_report_v7.py --decoding-only
```

The report uses `iblatlas.streamlines.utils.xyz_to_depth` for local pial
depth throughout. Use `python make_bwm_depth.py --vertical` to reproduce the
historical `bwm_d640` cache. Generate individual bin-wise supplement plots
with `python maps_v2.py --supplement counts` (or `tuning` or `summary`).
The best-ten decoder ranks units inside each outer training fold.
If any score cache changes, regenerate all three decoding figures so
they continue to use a matched cohort.

To regenerate the supplementary depth and insertion histograms from cached
features, run `python depth_coverage.py` followed by
`python build_report_v7.py --figure depth_coverage`.

Figure 5 compares Beryl VISa across the registered Allen CCF layers L2/3, L4,
L5, and L6. Mesoscope VISa ROIs are assigned to L2/3. Run
`python decode_bwm_depth.py` to refit the BWM decoders from cached features
and `python decode_bwm_depth.py --meso-reference` to fit the mesoscope
decoders. Then run `python depth_profile.py` and
`python build_report_v7.py --figure depth`.

## Figure 4 regional analysis

Run `python region_variance.py` followed by
`python build_report_v7.py --figure region_variance`. The regional and
trial-shuffle tables from `decode_region.py` are checked against the matching
cohort from Figures 1--3. The stats CSV includes the blocked region permutation
test, Holm corrections and the five-target omnibus result. The build refits
the variance-component bootstrap from the matched scores.

The published BWM reference uses the stage-2 tables when available. The
fallback is `reference/bwm_published_region_row.pdf`, a vector crop of the
published row in commit `3475913`, with the original balanced accuracy and
wheel velocity. The fallback uses `pikepdf` to place the reference in panel a.

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
- **Unbiased sessions.** 55 meso sessions have only 0.5 blocks. The BWM test
  uses the 0.8 / 0.2 blocks only, so these sessions are tested within their
  0.5 block (earlier versions left them untested and counted them as untuned).
- **Tuning.** `mpciROIs.taskTunedP` is not on Alyx and Yanliang's BWM
  p-values are not in the aggregates. Both modalities therefore use a GPU
  re-implementation of `brainwidemap` `get_{stim,choice,feedback}_time_shuffle`.
  On 60 neurons it agreed with the original on p < 0.05 in 97-100% of
  cases.
- **Decoding.** Sessions need at least 50 trials and at least 10 neurons in
  a bin. Use the "all neurons" maps with care: meso bins hold hundreds of
  neurons, so their scores are not comparable to BWM. The `n10` version
  makes the two rows comparable.
