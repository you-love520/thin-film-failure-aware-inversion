from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LossSpec:
    name: str
    tuning: float


LOSS_SPECS = {
    "L2": LossSpec("L2", 1.0),
    "Huber": LossSpec("Huber", 1.345),
    "Cauchy": LossSpec("Cauchy", 2.385),
    "Tukey": LossSpec("Tukey", 4.685),
    "Welsch": LossSpec("Welsch", 2.985),
}


def mad_scale(residual: np.ndarray, floor: float = 1e-7) -> float:
    residual = np.asarray(residual, dtype=float)
    median = float(np.median(residual))
    value = 1.4826 * float(np.median(np.abs(residual - median)))
    if not np.isfinite(value) or value < floor:
        rms = float(np.sqrt(np.mean((residual - median) ** 2)))
        value = max(rms, floor)
    return value


def rho(u: np.ndarray, spec: LossSpec) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    c = spec.tuning
    if spec.name == "L2":
        return 0.5 * u**2
    if spec.name == "Huber":
        a = np.abs(u)
        return np.where(a <= c, 0.5 * u**2, c * a - 0.5 * c**2)
    if spec.name == "Cauchy":
        return 0.5 * c**2 * np.log1p((u / c) ** 2)
    if spec.name == "Tukey":
        z = u / c
        inside = (c**2 / 6.0) * (1.0 - (1.0 - z**2) ** 3)
        return np.where(np.abs(u) <= c, inside, c**2 / 6.0)
    if spec.name == "Welsch":
        return 0.5 * c**2 * (1.0 - np.exp(-((u / c) ** 2)))
    raise ValueError(spec.name)


def weight(u: np.ndarray, spec: LossSpec) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    c = spec.tuning
    a = np.abs(u)
    if spec.name == "L2":
        return np.ones_like(u)
    if spec.name == "Huber":
        return np.where(a <= c, 1.0, c / np.maximum(a, 1e-15))
    if spec.name == "Cauchy":
        return 1.0 / (1.0 + (u / c) ** 2)
    if spec.name == "Tukey":
        z2 = (u / c) ** 2
        return np.where(a <= c, (1.0 - z2) ** 2, 0.0)
    if spec.name == "Welsch":
        return np.exp(-((u / c) ** 2))
    raise ValueError(spec.name)
