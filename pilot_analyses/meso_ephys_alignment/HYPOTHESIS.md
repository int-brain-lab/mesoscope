# Pre-registered hypothesis (written 2026-10-23, before any analysis)

## Question

Do task signals use the same population patterns as spontaneous activity? We
compare IBL mesoscope (calcium, thousands of layer-2/3 neurons in dorsal cortex) and
brain-wide-map Neuropixels recordings (spikes, all depths, subcortex) in the same
task, each with a 600-s grey screen (passive protocol, spontaneous activity).

Background: Stringer et al. (2019, Science) found that in mouse V1, stimulus responses
were largely orthogonal to the dominant dimensions of spontaneous, behaviour-related
activity. Whether this holds for other task variables and outside sensory cortex has
not been mapped.

## Hypothesis

**H.** The alignment of task coding with spontaneous activity follows the
sensory-to-motor hierarchy:
- in **sensory** regions, task-variable coding directions are **not** preferentially
  aligned with the top spontaneous dimensions (alignment at or below the noise null);
- in **motor/action** regions, the coding directions, especially for choice, lie
  **preferentially** in the top spontaneous dimensions (above the noise null);
- the ordering (motor > sensory) is the **same in both modalities**.

## Region classes (fixed in advance)

- **Sensory:** VIS* (all visual areas), AUD*, SSp-* , SSs, LGd, LP, PO, VPM, VPL, VPLpc,
  MGv, MGd, SCs.
- **Motor/action:** MOp, MOs, CP, ACB, GPe, SNr, SCm, MRN, GRN, IRN, PRNr, PRNc, ZI,
  VAL, VM, PAG.
- Every other region with enough neurons (e.g. RSP, hippocampus, APN) is reported as
  *other* and is not part of the test.

BWM uses Beryl areas. Meso uses Allen areas with layers pooled. Region labels in meso
are mostly pre-histology estimates.

## Data (pilot)

- **BWM:** 5c0c560e (NR_0028), 6274dda8 (CSH_ZAD_025), 2bdf206a (UCLA037). Each has
  600 s of grey screen and both region classes with >= 30 good units.
- **Meso:** 90029e53 (SP054, visual and RSP), fa10ff03 (SP075, SSp and MOp/MOs),
  a4afea1e (SP076, MOs/MOp). Each has 600 s of grey screen, fully imaged.

## Method (fixed in advance)

- **Neurons:** regions with >= 30 neurons. Each analysis uses N = 30 randomly drawn
  neurons (20 draws, averaged), so both modalities and all regions have the same
  dimensionality.
- **Normalisation:** each neuron is z-scored with the mean and s.d. of its grey-screen
  activity in 1-s bins. Task-window activity uses the same transform.
- **Spontaneous subspace:** PCA of the z-scored grey-screen activity (1-s bins,
  600 bins). U = top k = 3 PCs (primary); k = 1..10 is shown as a curve.
- **Task coding direction** d = mean(z | class A) - mean(z | class B), from per-trial
  window means. BWM trial mask: choice made, 0.08 s <= RT <= 2 s. Windows as in
  dorsal_maps (BWM / meso, meso widened to 0.3 s for the frame rate):
  - *stimulus side* (contrast >= 25%): stimOn + [0, 0.1] / [0, 0.3] s;
  - *choice*: firstMovement + [-0.1, 0] / [-0.3, 0] s;
  - *feedback* (reward vs error): feedback + [0, 0.2] / [0, 0.3] s;
  - *block prior* (pL 0.8 vs 0.2): stimOn + [-0.4, -0.1] s, both modalities.
- **Alignment** A = ||U^T d||^2 / ||d||^2. Isotropic chance is k/N = 0.1.
- **Noise null:** 500 circular shifts of the label sequence (shift >= 10% of trials;
  this keeps block autocorrelation). This gives A_null, the alignment of a "coding"
  direction made only of noise with the data's covariance. Main statistic:
  **R = A / mean(A_null)**. R > 1 means more aligned than noise, R < 1 more
  orthogonal than noise.
- **Coding reliability:** a region-variable pair is interpreted only if ||d||^2
  exceeds the 95th percentile of its null (actual coding exists).

## What counts as promising (decides whether to expand)

1. For **choice**, the median R of motor regions > the median R of sensory regions
   **in both modalities**.
2. The difference has the same sign in at least 2 of 3 sessions per modality that
   contain both classes or, where sessions hold only one class, across sessions.
3. Sensory R for **stimulus side** is <= 1 (replicates Stringer et al. in direction).

If 1 fails in either modality the hypothesis is not supported, and we do not expand.
Outcomes 2 and 3 qualify the strength of the evidence. With 3 + 3 sessions this is a
descriptive pilot; no p-value will be treated as confirmatory.

## Known limitations, stated in advance

- Stimulus side and choice are correlated at >= 25% contrast (mostly correct trials).
- The grey screen contains movement. The spontaneous dimensions are partly
  behaviour-driven, which is part of the question, not a confound.
- Calcium and spikes differ in time resolution; the 1-s grey-screen bins and the
  matched N reduce but do not remove this.
