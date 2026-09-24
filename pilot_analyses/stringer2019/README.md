# Regional shared-variance analyses (Stringer et al. 2019 extension)

Scripts that generate the report `report/main.pdf`: SVCA dimensionality and
behavioural prediction of shared neural variance in VISp, SSp-bfd, MOp and MOs,
during the task and during the passive protocol, with 1,000 neurons per
region-session. The report's Methods section describes the data, the session
selection, the matching and the maths.

Method reference: C. Stringer et al., *Spontaneous behaviors drive
multidimensional, brainwide activity*, Science 364, eaav7893 (2019),
doi:10.1126/science.aav7893.

Shared code lives one level up: `meso_loader.py` (repo root),
`pilot_analyses/stringer19_svca_prediction.py` (SVCA, video PCs, predictor
regression) and `pilot_analyses/stringer19_region_comparison.py` (region neuron
selection, power-law fit, effective dimensionality).

## Running

Use the `iblenv` conda environment with ONE access to the mesoscope data. A CUDA
GPU is used for the SVDs when available. All outputs (figures, per-session
caches, tables) go to `output/`, which is not tracked.

| Step | Script | Output | Report |
|---|---|---|---|
| 1 | `region_session_scan.py` | `region_session_scan.csv`: neurons per region, passive coverage, video/trials per canonical session | session selection, tables |
| 2 | `regional_dimensionality.py` | `regional_dimensionality_panels.pdf` | Fig. 1 |
| 3 | `regional_behavior_prediction.py` | `regional_behavior_prediction{,_sessions}_panels.pdf` | Fig. 2 |
| 4 | `passive_regional_dimensionality.py` | `passive_regional_dimensionality_panels.pdf` | Fig. 3 |
| 5 | `passive_behavior_prediction.py` | `passive_behavior_prediction_panels.pdf` | Fig. 4 |
| 6 | `make_tables.py` | `table_{dimensionality,behavior,passive}.tex` | Tables 2-4 |
| 7 | `cd report && tectonic main.tex` | `main.pdf` | |

```bash
python region_session_scan.py
python regional_dimensionality.py
python regional_behavior_prediction.py
python passive_regional_dimensionality.py
python passive_behavior_prediction.py
python make_tables.py
cd report && tectonic main.tex
```

Steps 3 and 5 decode the raw face video of every session, which is slow and
download-bound. Both cache per-session results in `output/`, and each can be split
across processes by region, e.g.
`python regional_behavior_prediction.py --cache-only VISp`; a final run without
arguments then only plots. Face-video PCs are also cached under
`<ONE cache>/meso/svca_video_pcs/`.

The sessions of each analysis are listed explicitly in the scripts
(`TIER1_SESSIONS` in `stringer19_region_comparison.py`, `SESSIONS` in
`regional_behavior_prediction.py` and `passive_behavior_prediction.py`,
`PASSIVE_SESSIONS` in `passive_regional_dimensionality.py`), chosen from the scan
as described in the report.
