# Do tracked mesoscope neurons keep their response profile across days?

Clean re-do of the chronic analyses in `../pairwise_correl.py` and
`../meso_concat.py::chronic_pair_feature_corr_subject`.

* `data.py`: ROICaT cluster UIDs (FlatIron ROICaT mirror, cached in
  `~/chronic_csv/ROICaT`) attached to the per-trial window features of
  `../dorsal_maps/extract_meso.py`. The ROI rows of the cached features are
  rebuilt and checked (row count and region id of every row). Sessions whose
  histology changed after the dorsal_maps extraction (all SP058) are
  re-extracted from the deconvolved traces. Output: `output/sessions/<subject>/*.npz`.
* `qc.py`: spatial tracking QC from ROI pixel positions (pairwise affine
  registration of sessions, deviation vs nearest-neighbour distance).
  Output: `output/qc_positions.parquet`.
* `stability.py`: reliability metrics, all against the even/odd-trial
  within-session baseline (profile, activity level, selectivity maps,
  sensitivity persistence, decoders; see docstring). Output: `output/pairs.csv`,
  `sessions.csv`, `neurons.parquet`, `neuron_sessions.parquet`.
* `trends.py`: per-neuron trend of stimulus selectivity over days (permutation test).
* `decoder_drift.py`: stimulus/choice decoders across days: transfer matrix (per direction),
  recalibrated vs fixed z-scoring, weight cosine (`decoder_drift.parquet`).
* `figure.py`: `tracking_qc.pdf`, `chronic_stability.pdf`, `decoder_drift.pdf`, `summary.txt`.
* `report/report.tex`: LaTeX report (compile with `tectonic report.tex`).
* `examples.py`: example neurons (footprints, rasters, d' over days),
  `examples_stable.pdf`, `examples_losing.pdf`, `examples_metrics.pdf`,
  `sensitivity_trends.pdf`. Downloads the traces of the example FOVs only.

Every figure has a LaTeX caption (`.tex`) next to it. Run in order with the `iblenv`
environment: `data.py, qc.py, stability.py, decoder_drift.py, trends.py, figure.py, examples.py`.
