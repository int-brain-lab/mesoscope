# wheel_decoding2: can the wheel-speed decoding be trusted?

This folder revisits the wheel-speed decoding of `../dorsal_maps`, which uses ridge regression from the neurons of one spatial bin.
The question is why mesoscope (meso) decoding is so much weaker than Neuropixels (BWM) decoding.

Features are taken from the dorsal_maps feature files:
- 6 wheel bins of 0.2 s, from −0.2 to 1.0 s around first movement;
- mean deconvolved activity (meso) or spike counts of isocortex units (BWM) per bin;
- target: mean wheel speed per bin.

## Findings

1. **Encoder glitch (bug, fixed).** 16 of 153 meso sessions had wheel-position jumps of exactly ±6588397 rad, from encoder counter overflow. This gave bins at ~3×10⁷ rad/s, and R² in these sessions was meaningless. No BWM session is affected.
   - `dorsal_maps/common.remove_wheel_jumps` drops steps larger than 100 rad.
   - `dorsal_maps/fix_wheel_glitches.py` patched the 16 feature files; clean bins are unchanged.
2. **The decoder and target are correct** (`checks.py`, fig2a):
   - It matches the dorsal_maps GPU decoder to 2×10⁻¹⁵ and scikit-learn to 10⁻⁷.
   - A synthetic neuron (wheel speed plus noise) is recovered at the expected R² = SNR/(1+SNR).
   - The per-bin speed agrees with brainbox's filtered speed (r ≥ 0.9999).
3. **Timing** (`timing.py`, fig2b): meso deconvolved activity peaks 0.15–0.25 s *after* wheel speed, while BWM spikes peak 0.1 s *before* it. Adding the previous and next bin as features (lags) raises R² in both modalities.
4. **What R² measures** (fig2c): predicting each bin by its mean over the training trials, with no neural data, gives R² ≈ 0.47 (meso) and 0.44 (BWM). The decoders score below that, so their R² mostly reflects the stereotyped speed profile after movement onset. The trial-specific target (speed minus the per-bin mean) isolates trial-to-trial differences.
5. **Decoder variants** (`variants.py`, fig2d–f, 0.4 mm bins, mean R²):

   | Variant | meso n10 | BWM n10 | meso all | BWM all |
   |---|---|---|---|---|
   | current | 0.015 | 0.144 | 0.095 | 0.249 |
   | + LOTO alpha | 0.016 | 0.144 | 0.096 | 0.249 |
   | + lags | 0.026 | 0.218 | 0.131 | 0.324 |
   | trial-specific + lags | 0.007 | 0.046 | 0.040 | 0.085 |
   | shuffled null | −0.001 | 0.000 | −0.001 | −0.001 |

6. **Why meso is weaker: wheel correlates are rare, not absent** (`rasters.py`, `population.py`; fig1, fig3).
   - Neurons were ranked on held-out trials.
   - **Fraction of wheel-correlated neurons:** 3.5% of meso neurons correlate with wheel speed at r > 0.1 in both halves of a session, vs 22% of BWM isocortex units.
   - **Best 10 of a whole session:** their mean tracks wheel speed in held-out trials as well in meso (r = 0.50, thousands of neurons) as in BWM (0.47).
   - **Random set of 30 neurons:** the best 10 reach only 0.15 in meso vs 0.37 in BWM.
   - **Consequence:** a spatial bin with 10 random neurons rarely contains a wheel neuron in meso.

## Scripts

| Script | Output |
|---|---|
| `wd_common.py` | Shared paths, depth below the pial surface, per-neuron wheel r |
| `ridge.py` | Batched GPU ridge: LOO, exact leave-one-trial-out or trial-CV alpha; lags; residual target; shuffle null; primal/dual solver |
| `checks.py` | `checks_decoder.csv`, `checks_synthetic.csv`, `checks_target.csv` |
| `timing.py` | `timing.pqt`, `timing_summary.csv` |
| `variants.py` | `variants.pqt`, `variants_sessions.pqt` |
| `population.py` | `population.pqt` |
| `rasters.py` | `fig1_rasters.pdf` |
| `fig_checks.py` | `fig2_checks.pdf` |
| `fig_population.py` | `fig3_population.pdf` |
| `dorsal_maps_tasks.py` | Task collection lookup for meso sessions |

The revised decoder (LOTO alpha, lags b−1..b+1, both targets) is used by `../dorsal_maps/decode_v2.py`.
That script uses 1 mm hexagons and adds a BWM set restricted to units within 640 µm of the pial surface (`make_bwm_depth.py`).

Order of runs:
1. `python checks.py`
2. `python timing.py`
3. `python variants.py`
4. `python population.py`
5. `python rasters.py`
6. `python fig_checks.py`
7. `python fig_population.py`

Environment: `iblenv`, with a GPU for the ridge decoding.
