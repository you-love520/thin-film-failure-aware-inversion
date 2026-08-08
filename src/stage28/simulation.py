from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    gaussian_std: float = 0.0
    outlier_ratio: float = 0.0
    outlier_magnitude: float = 0.0
    baseline_amplitude: float = 0.0
    gain_std: float = 0.0
    offset_std: float = 0.0


SCENARIOS = {
    "clean": Scenario("clean"),
    "gaussian": Scenario("gaussian", gaussian_std=0.0008, gain_std=0.005, offset_std=0.0002),
    "impulsive": Scenario(
        "impulsive", gaussian_std=0.0004, outlier_ratio=0.04, outlier_magnitude=0.012, gain_std=0.005
    ),
    "baseline_drift": Scenario(
        "baseline_drift", gaussian_std=0.0004, baseline_amplitude=0.0035, gain_std=0.005
    ),
    "mixed": Scenario(
        "mixed",
        gaussian_std=0.0008,
        outlier_ratio=0.06,
        outlier_magnitude=0.015,
        baseline_amplitude=0.0035,
        gain_std=0.008,
        offset_std=0.0003,
    ),
}


@dataclass
class Observation:
    values: np.ndarray
    gaussian_noise: np.ndarray
    outlier_indices: np.ndarray
    outlier_values: np.ndarray
    baseline: np.ndarray
    gain: float
    offset: float


def normalized_axis(wavelength_nm: np.ndarray) -> np.ndarray:
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    return 2.0 * (wavelength_nm - wavelength_nm.min()) / np.ptp(wavelength_nm) - 1.0


def independent_smooth_baseline(wavelength_nm: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Generate smooth drift that is not exactly contained in an affine/quadratic basis."""
    x = normalized_axis(wavelength_nm)
    coefficients = rng.normal([0.15, 0.55, 0.30, 0.18], [0.05, 0.08, 0.08, 0.04])
    raw = (
        coefficients[0]
        + coefficients[1] * x
        + coefficients[2] * x**2
        + coefficients[3] * np.sin(1.35 * np.pi * x + rng.uniform(-0.5, 0.5))
    )
    raw -= np.mean(raw)
    normalizer = max(float(np.max(np.abs(raw))), 1e-12)
    return amplitude * raw / normalizer


def generate_observation(
    clean_spectrum: np.ndarray,
    wavelength_nm: np.ndarray,
    scenario: Scenario,
    rng: np.random.Generator,
) -> Observation:
    clean_spectrum = np.asarray(clean_spectrum, dtype=float)
    gain = float(1.0 + rng.normal(0.0, scenario.gain_std))
    offset = float(rng.normal(0.0, scenario.offset_std))
    gaussian = rng.normal(0.0, scenario.gaussian_std, clean_spectrum.size)
    baseline = independent_smooth_baseline(wavelength_nm, scenario.baseline_amplitude, rng)

    outlier_count = int(round(scenario.outlier_ratio * clean_spectrum.size))
    if outlier_count:
        indices = np.sort(rng.choice(clean_spectrum.size, outlier_count, replace=False))
        signs = rng.choice(np.array([-1.0, 1.0]), outlier_count)
        magnitudes = scenario.outlier_magnitude * rng.uniform(0.75, 1.25, outlier_count)
        outlier_values = signs * magnitudes
    else:
        indices = np.zeros(0, dtype=int)
        outlier_values = np.zeros(0, dtype=float)

    values = gain * clean_spectrum + offset + baseline + gaussian
    values = values.copy()
    values[indices] += outlier_values
    return Observation(values, gaussian, indices, outlier_values, baseline, gain, offset)


def paired_rng(master_seed: int, scenario_id: int, trial_id: int) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence([master_seed, scenario_id, trial_id])
    return np.random.default_rng(seed_sequence)
