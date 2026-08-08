# Random Observation Specification

Status: `FROZEN_COMPONENT_DEFINITION`  
Actual estimator execution: `NOT STARTED`

## Observation identity

There are 7,200 independent simulated observations. Each is uniquely defined
by material, true thickness, contamination scenario and trial number. E0, E1
and E3 receive the same realized observation for a given `observation_id`.

The registry stores, before execution:

- material and material-source identity;
- true thickness;
- scenario and every scenario parameter;
- master seed, encoded Stage 28 scenario ID and trial ID;
- complete NumPy `SeedSequence` entropy `[master_seed, stage28_scenario_id,
  trial_id]`;
- required bit generator (`PCG64`);
- wavelength grid and expected method identities.

The random spectrum is reconstructed deterministically rather than stored as a
large pre-generated matrix.

## Mathematical observation model

For wavelength index j, the observed spectrum is

```text
y_j = G * R_j(d, material) + O + B_j + N_j + I_j
```

where:

- `R_j` is the clean single-layer reflectance at registered true thickness;
- `G = 1 + Normal(0, gain_std)` is one scalar multiplicative calibration
  factor shared by all wavelengths of the observation;
- `O = Normal(0, offset_std)` is one scalar additive offset;
- `N_j` are independent `Normal(0, gaussian_std)` values;
- `B_j` is a smooth independently generated baseline drift;
- `I_j` is zero except at sampled impulsive-outlier positions.

No clipping, renormalization or post-generation correction is applied.

## Smooth baseline

Let x map the wavelength grid linearly to [-1, 1]. Four coefficients are drawn:

```text
c ~ Normal([0.15, 0.55, 0.30, 0.18],
           [0.05, 0.08, 0.08, 0.04])
phase ~ Uniform(-0.5, 0.5)

raw(x) = c0 + c1*x + c2*x^2 + c3*sin(1.35*pi*x + phase)
```

The mean is removed and the curve is scaled so its maximum absolute value is
the registered `baseline_amplitude`. This drift is intentionally not exactly
contained in the fitted quadratic Legendre nuisance basis.

## Impulsive contamination

The number of outliers is `round(outlier_ratio * 361)`. Positions are sampled
uniformly without replacement and sorted. Each sign is sampled independently
from {-1, +1}. Each absolute magnitude is

```text
outlier_magnitude * Uniform(0.75, 1.25)
```

Thus the registered counts are 14 outliers for the impulsive scenario and 22
for mixed contamination.

## Scenario parameters

| Scenario | Gaussian SD | Outlier ratio | Outlier magnitude | Baseline amplitude | Gain SD | Offset SD |
|---|---:|---:|---:|---:|---:|---:|
| gaussian | 0.0008 | 0 | 0 | 0 | 0.005 | 0.0002 |
| impulsive | 0.0004 | 0.04 | 0.012 | 0 | 0.005 | 0 |
| baseline_drift | 0.0004 | 0 | 0 | 0.0035 | 0.005 | 0 |
| mixed | 0.0008 | 0.06 | 0.015 | 0.0035 | 0.008 | 0.0003 |

## Seed construction and independence

```text
master_seed = 20260726
stage28_scenario_id = 280000
                    + 1000 * material_index
                    + 100 * thickness_index
                    + scenario_index
trial_id = 0,...,199
SeedSequence entropy = [master_seed, stage28_scenario_id, trial_id]
bit generator = PCG64
```

Different material/thickness/scenario cells use distinct streams. Method rows
are paired because they reuse one reconstructed observation, not because they
generate separate streams with similar seeds.

## Execution-time observation audit

The cloud runner writes one audit row per observation with:

- clean and observed spectrum SHA-256;
- Gaussian-noise, baseline, outlier-index and outlier-value SHA-256;
- realized gain and offset;
- Gaussian RMS, baseline maximum absolute value and outlier count;
- outlier indices as JSON;
- minimum and maximum observed reflectance;
- the three independently recomputed adaptive scales and their equality flag.

This audit allows reviewers to verify what was randomized without treating
wavelength samples or estimator rows as independent observations.
