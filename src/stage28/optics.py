from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


MALITSON_B = np.array([0.6961663, 0.4079426, 0.8974794], dtype=float)
MALITSON_C_UM2 = np.array([0.0684043**2, 0.1162414**2, 9.896161**2], dtype=float)

N_BK7_B = np.array([1.03961212, 0.231792344, 1.01046945], dtype=float)
N_BK7_C_UM2 = np.array([0.00600069867, 0.0200179144, 103.560653], dtype=float)


def sellmeier_index(wavelength_nm: np.ndarray | float, b: np.ndarray, c_um2: np.ndarray) -> np.ndarray:
    """Return refractive index from a three-term Sellmeier equation.

    The vacuum wavelength is converted from nm to µm. Coefficients C are in µm².
    """
    wavelength_um = np.asarray(wavelength_nm, dtype=float) / 1000.0
    lam2 = wavelength_um**2
    n2 = 1.0 + np.sum(b[:, None] * lam2.ravel()[None, :] / (lam2.ravel()[None, :] - c_um2[:, None]), axis=0)
    result = np.sqrt(n2).reshape(wavelength_um.shape)
    return result


def fused_silica_index(wavelength_nm: np.ndarray | float) -> np.ndarray:
    return sellmeier_index(wavelength_nm, MALITSON_B, MALITSON_C_UM2)


def n_bk7_index(wavelength_nm: np.ndarray | float) -> np.ndarray:
    return sellmeier_index(wavelength_nm, N_BK7_B, N_BK7_C_UM2)


def _forward_cosine(n0: complex | np.ndarray, nj: complex | np.ndarray, angle0_rad: float) -> np.ndarray:
    sin_j = np.asarray(n0, dtype=complex) * np.sin(angle0_rad) / np.asarray(nj, dtype=complex)
    cos_j = np.sqrt(1.0 - sin_j**2 + 0j)
    flip = (np.real(cos_j) < 0) | ((np.abs(np.real(cos_j)) < 1e-14) & (np.imag(cos_j) < 0))
    return np.where(flip, -cos_j, cos_j)


def optical_admittance(n: complex | np.ndarray, cos_theta: np.ndarray, polarization: str) -> np.ndarray:
    n = np.asarray(n, dtype=complex)
    if polarization == "s":
        return n * cos_theta
    if polarization == "p":
        return n / cos_theta
    raise ValueError("polarization must be 's' or 'p'")


def fresnel_amplitude(
    n_i: complex | np.ndarray,
    n_j: complex | np.ndarray,
    angle0_deg: float,
    n_ambient: complex | np.ndarray,
    polarization: str,
) -> np.ndarray:
    """Interface reflection amplitude under an admittance convention."""
    theta0 = np.deg2rad(angle0_deg)
    cos_i = _forward_cosine(n_ambient, n_i, theta0)
    cos_j = _forward_cosine(n_ambient, n_j, theta0)
    eta_i = optical_admittance(n_i, cos_i, polarization)
    eta_j = optical_admittance(n_j, cos_j, polarization)
    return (eta_i - eta_j) / (eta_i + eta_j)


def airy_single_layer_reflectance(
    wavelength_nm: np.ndarray,
    thickness_nm: float,
    n_ambient: complex | np.ndarray,
    n_film: complex | np.ndarray,
    n_substrate: complex | np.ndarray,
    angle_deg: float = 0.0,
    polarization: str = "s",
) -> np.ndarray:
    """Coherent reflectance of one film using the Airy amplitude formula."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    theta0 = np.deg2rad(angle_deg)
    cos_film = _forward_cosine(n_ambient, n_film, theta0)
    r01 = fresnel_amplitude(n_ambient, n_film, angle_deg, n_ambient, polarization)
    r12 = fresnel_amplitude(n_film, n_substrate, angle_deg, n_ambient, polarization)
    delta = 2.0 * np.pi * np.asarray(n_film, dtype=complex) * cos_film * thickness_nm / wavelength_nm
    phase = np.exp(2j * delta)
    amplitude = (r01 + r12 * phase) / (1.0 + r01 * r12 * phase)
    return np.abs(amplitude) ** 2


def tmm_single_layer_reflectance(
    wavelength_nm: np.ndarray,
    thickness_nm: float,
    n_ambient: complex | np.ndarray,
    n_film: complex | np.ndarray,
    n_substrate: complex | np.ndarray,
    angle_deg: float = 0.0,
    polarization: str = "s",
) -> np.ndarray:
    """Coherent characteristic-matrix reflectance of one film."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    theta0 = np.deg2rad(angle_deg)
    cos0 = _forward_cosine(n_ambient, n_ambient, theta0)
    cos1 = _forward_cosine(n_ambient, n_film, theta0)
    cos2 = _forward_cosine(n_ambient, n_substrate, theta0)
    eta0 = optical_admittance(n_ambient, cos0, polarization)
    eta1 = optical_admittance(n_film, cos1, polarization)
    eta2 = optical_admittance(n_substrate, cos2, polarization)
    delta = 2.0 * np.pi * np.asarray(n_film, dtype=complex) * cos1 * thickness_nm / wavelength_nm

    a = np.cos(delta)
    b = 1j * np.sin(delta) / eta1
    c = 1j * eta1 * np.sin(delta)
    d = np.cos(delta)
    numerator = eta0 * a + eta0 * eta2 * b - c - eta2 * d
    denominator = eta0 * a + eta0 * eta2 * b + c + eta2 * d
    return np.abs(numerator / denominator) ** 2


def unpolarized_reflectance(
    model: str,
    wavelength_nm: np.ndarray,
    thickness_nm: float,
    n_ambient: complex | np.ndarray,
    n_film: complex | np.ndarray,
    n_substrate: complex | np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    function = {
        "airy": airy_single_layer_reflectance,
        "tmm": tmm_single_layer_reflectance,
    }[model]
    rs = function(wavelength_nm, thickness_nm, n_ambient, n_film, n_substrate, angle_deg, "s")
    rp = function(wavelength_nm, thickness_nm, n_ambient, n_film, n_substrate, angle_deg, "p")
    return 0.5 * (rs + rp)


@dataclass(frozen=True)
class PhysicalModelConfig:
    angle_deg: float = 8.0
    spectral_blur_sigma_nm: float = 0.6
    model: str = "tmm"
    dispersion: bool = True
    constant_film_index: float = 1.46
    constant_substrate_index: float = 1.5168
    film_index_scale: float = 1.0
    substrate_index_scale: float = 1.0


def physical_spectrum(
    wavelength_nm: np.ndarray,
    thickness_nm: float,
    config: PhysicalModelConfig,
) -> np.ndarray:
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    if config.dispersion:
        n_film = fused_silica_index(wavelength_nm)
        n_substrate = n_bk7_index(wavelength_nm)
    else:
        n_film = np.full_like(wavelength_nm, config.constant_film_index, dtype=float)
        n_substrate = np.full_like(wavelength_nm, config.constant_substrate_index, dtype=float)
    n_film = np.asarray(n_film) * config.film_index_scale
    n_substrate = np.asarray(n_substrate) * config.substrate_index_scale

    reflectance = unpolarized_reflectance(
        config.model,
        wavelength_nm,
        thickness_nm,
        1.0,
        n_film,
        n_substrate,
        config.angle_deg,
    )
    if config.spectral_blur_sigma_nm > 0:
        spacing_nm = float(np.median(np.diff(wavelength_nm)))
        reflectance = gaussian_filter1d(reflectance, config.spectral_blur_sigma_nm / spacing_nm, mode="nearest")
    return np.asarray(reflectance, dtype=float)
