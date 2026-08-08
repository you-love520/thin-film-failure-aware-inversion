from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar


EPS = 1e-12


@dataclass(frozen=True)
class E4Result:
    observation_id: str
    e3_thickness_nm: float
    e4_candidate_thickness_nm: float
    e4_final_thickness_nm: float
    triggered: bool
    accepted: bool
    fallback_reason: str
    objective_anchor: float
    objective_candidate: float
    objective_relative_improvement: float
    baseline_gain_gb: float
    baseline_amplitude_ratio_ab: float
    first_order_projection_qc: float
    second_order_leakage: float
    identifiability_e3: float
    identifiability_e4: float
    identifiability_retention_ratio: float
    consensus_median: float
    consensus_mad: float
    trust_radius_nm: float
    profile_curvature: float
    drift_effective_rank: int
    drift_condition_number: float
    drift_active_columns: int
    second_order_protection_active: bool
    inner_converged: bool
    inner_iterations: int
    outer_nfev: int
    runtime_e4_seconds: float
    e4_coefficients_json: str


def normalized_axis(wavelength_nm: np.ndarray) -> np.ndarray:
    wavelength = np.asarray(wavelength_nm, dtype=float)
    return 2.0 * (wavelength - wavelength.min()) / np.ptp(wavelength) - 1.0


def legendre_b2(wavelength_nm: np.ndarray) -> np.ndarray:
    x = normalized_axis(wavelength_nm)
    return np.polynomial.legendre.legvander(x, 2)


def tukey_rho(u: np.ndarray, c: float = 4.685) -> np.ndarray:
    value = np.asarray(u, dtype=float)
    out = np.full_like(value, c * c / 6.0)
    mask = np.abs(value) <= c
    z = value[mask] / c
    out[mask] = (c * c / 6.0) * (1.0 - (1.0 - z * z) ** 3)
    return out


def tukey_weight(u: np.ndarray, c: float = 4.685) -> np.ndarray:
    value = np.asarray(u, dtype=float)
    out = np.zeros_like(value)
    mask = np.abs(value) <= c
    z = value[mask] / c
    out[mask] = (1.0 - z * z) ** 2
    return out


def finite_difference(
    spectrum,
    d: float,
    step_nm: float,
    bounds: tuple[float, float],
    order: int,
) -> np.ndarray:
    lower, upper = map(float, bounds)
    step = float(step_nm)
    left = max(lower, d - step)
    right = min(upper, d + step)
    center = float(d)
    if order == 1:
        if right > center and left < center:
            return (spectrum(right) - spectrum(left)) / (right - left)
        if right > center:
            return (spectrum(right) - spectrum(center)) / (right - center)
        if left < center:
            return (spectrum(center) - spectrum(left)) / (center - left)
        raise ValueError("cannot compute first derivative inside bounds")
    if order == 2:
        if right > center and left < center:
            h1 = center - left
            h2 = right - center
            if abs(h1 - h2) <= 1e-12:
                return (spectrum(right) - 2.0 * spectrum(center) + spectrum(left)) / (h1 * h1)
            y0, y1, y2 = spectrum(left), spectrum(center), spectrum(right)
            return 2.0 * (
                y0 / (h1 * (h1 + h2))
                - y1 / (h1 * h2)
                + y2 / (h2 * (h1 + h2))
            )
        return np.zeros_like(spectrum(center), dtype=float)
    raise ValueError("order must be 1 or 2")


def binary_tree_nodes(depth: int) -> list[tuple[int, int]]:
    return [(level, index) for level in range(int(depth) + 1) for index in range(2**level)]


def smooth_window(x: np.ndarray, center: float, half_width: float, overlap_fraction: float) -> np.ndarray:
    expanded = half_width * (1.0 + float(overlap_fraction))
    distance = np.abs(x - center)
    core = max(half_width * (1.0 - float(overlap_fraction)), 0.0)
    window = np.zeros_like(x, dtype=float)
    window[distance <= core] = 1.0
    taper = (distance > core) & (distance < expanded)
    if expanded > core:
        t = (distance[taper] - core) / (expanded - core)
        window[taper] = 0.5 * (1.0 + np.cos(np.pi * t))
    return window


def residual_tree_dictionary(
    wavelength_nm: np.ndarray,
    depth: int,
    overlap_fraction: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x = normalized_axis(wavelength_nm)
    columns: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for level, index in binary_tree_nodes(depth):
        width = 2.0 / (2**level)
        left = -1.0 + index * width
        right = left + width
        center = 0.5 * (left + right)
        half = 0.5 * width
        w = smooth_window(x, center, half, overlap_fraction)
        if np.max(w) <= 0.0:
            continue
        xi = np.clip((x - center) / max(half, EPS), -1.0, 1.0)
        atoms = (
            ("local_constant", w),
            ("local_linear", w * xi),
            ("local_quadratic", w * (xi * xi - np.average(xi * xi, weights=np.maximum(w, EPS)))),
        )
        for basis_name, atom in atoms:
            columns.append(np.asarray(atom, dtype=float))
            metadata.append({"level": level, "node": index, "basis": basis_name})
    return np.column_stack(columns), metadata


def weighted_residualize(matrix: np.ndarray, against: np.ndarray, weights: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    a = np.asarray(against, dtype=float)
    w = np.sqrt(np.clip(np.asarray(weights, dtype=float), 0.0, None))[:, None]
    if a.size == 0 or a.shape[1] == 0:
        return m.copy()
    coefficients, *_ = np.linalg.lstsq(w * a, w * m, rcond=None)
    return m - a @ coefficients


def weighted_rank_condition(matrix: np.ndarray, weights: np.ndarray, tolerance: float) -> tuple[int, float]:
    if matrix.size == 0 or matrix.shape[1] == 0:
        return 0, float("inf")
    weighted = np.sqrt(np.clip(weights, 0.0, None))[:, None] * matrix
    singular = np.linalg.svd(weighted, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return 0, float("inf")
    rank = int(np.sum(singular > singular[0] * float(tolerance)))
    if rank <= 0:
        return 0, float("inf")
    condition = float(singular[0] / max(singular[rank - 1], EPS))
    return rank, condition


def build_physical_drift_space(
    wavelength_nm: np.ndarray,
    spectrum,
    d3: float,
    g3: float,
    weights: np.ndarray,
    config: dict[str, Any],
    bounds: tuple[float, float],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    params = config["residual_tree"]
    deriv = config["derivatives"]
    projection = config["physical_projection"]
    b2 = legendre_b2(wavelength_nm)
    f3 = np.asarray(spectrum(float(d3)), dtype=float)
    fp = finite_difference(spectrum, d3, deriv["first_derivative_step_nm"], bounds, 1)
    fpp = finite_difference(spectrum, d3, deriv["second_derivative_step_nm"], bounds, 2)
    t1 = np.column_stack([f3, b2, float(g3) * fp])
    sraw, meta = residual_tree_dictionary(
        wavelength_nm,
        int(params["tree_depth"]),
        float(params["window_overlap_fraction"]),
    )
    s1 = weighted_residualize(sraw, t1, weights)
    h2 = float(g3) * fpp
    h2_perp = weighted_residualize(h2[:, None], t1, weights)[:, 0]
    h2_norm = float(h2_perp @ (weights * h2_perp))
    second_active = bool(np.isfinite(h2_norm) and h2_norm > float(deriv["second_order_rank_tolerance"]))
    if second_active:
        pi2_s1 = h2_perp[:, None] @ ((h2_perp @ (weights[:, None] * s1)) / max(h2_norm, EPS))[None, :]
        sphys = s1 - float(projection["eta2"]) * pi2_s1
    else:
        sphys = s1

    weighted_norms = np.sqrt(np.sum((np.sqrt(np.clip(weights, 0.0, None))[:, None] * sphys) ** 2, axis=0))
    keep = weighted_norms > float(params["rank_tolerance"])
    sphys = sphys[:, keep]
    kept_meta = [item for item, use in zip(meta, keep, strict=True) if bool(use)]
    norms = weighted_norms[keep]
    if sphys.shape[1]:
        sphys = sphys / np.maximum(norms, EPS)
    rank, condition = weighted_rank_condition(sphys, weights, float(params["rank_tolerance"]))
    numerator = np.linalg.norm(t1.T @ (weights[:, None] * sphys), ord="fro") if sphys.shape[1] else 0.0
    denominator = (
        np.linalg.norm(np.sqrt(np.clip(weights, 0.0, None))[:, None] * t1, ord="fro")
        * np.linalg.norm(np.sqrt(np.clip(weights, 0.0, None))[:, None] * sphys, ord="fro")
        + EPS
    )
    qc = float(numerator / denominator)
    return sphys, kept_meta, {
        "f3": f3,
        "fp": fp,
        "fpp": fpp,
        "t1": t1,
        "b2": b2,
        "h2_perp": h2_perp,
        "second_order_protection_active": second_active,
        "drift_effective_rank": rank,
        "drift_condition_number": condition,
        "first_order_projection_qc": qc,
    }


def tree_penalty_matrix(meta: list[dict[str, Any]], config: dict[str, Any]) -> np.ndarray:
    if not meta:
        return np.zeros((0, 0), dtype=float)
    weights = config["residual_tree"]["level_weights"]
    rows: list[np.ndarray] = []
    for idx, item in enumerate(meta):
        row = np.zeros(len(meta), dtype=float)
        level = int(item["level"])
        row[idx] = float(weights[min(level, len(weights) - 1)])
        rows.append(row)
    lookup = {(int(item["level"]), int(item["node"]), str(item["basis"])): i for i, item in enumerate(meta)}
    for idx, item in enumerate(meta):
        level = int(item["level"])
        if level == 0:
            continue
        parent = (level - 1, int(item["node"]) // 2, str(item["basis"]))
        parent_idx = lookup.get(parent)
        if parent_idx is not None:
            row = np.zeros(len(meta), dtype=float)
            row[idx] = 1.0
            row[parent_idx] = -1.0
            rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, len(meta)), dtype=float)


def fit_inner(
    observed: np.ndarray,
    model: np.ndarray,
    b2: np.ndarray,
    sphys: np.ndarray,
    scale: float,
    e3_coefficients: np.ndarray,
    weights_seed: np.ndarray,
    meta: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[np.ndarray, float, bool, int, np.ndarray]:
    opt = config["optimization"]
    x = np.column_stack([model, b2, sphys])
    p = x.shape[1]
    e3_full = np.zeros(p, dtype=float)
    e3_full[: min(e3_coefficients.size, 4)] = e3_coefficients[: min(e3_coefficients.size, 4)]
    lower = np.full(p, -np.inf, dtype=float)
    upper = np.full(p, np.inf, dtype=float)
    lower[0], upper[0] = 0.0, 2.0
    coeff = e3_full.copy()
    tree_l = tree_penalty_matrix(meta, config)
    last = coeff.copy()
    converged = False
    iterations = 0
    for iterations in range(1, int(opt["inner_maxiter"]) + 1):
        residual = observed - x @ coeff
        robust_w = tukey_weight(residual / max(scale, EPS))
        robust_w = np.clip(robust_w, 0.0, 1.0)
        aw = np.sqrt(np.maximum(robust_w, 0.0)) / max(scale, EPS)
        a_blocks = [aw[:, None] * x]
        b_blocks = [aw * observed]
        anchor = np.zeros((4, p), dtype=float)
        anchor[:, :4] = np.eye(4)
        a_blocks.append(np.sqrt(float(opt["lambda_eta"])) * anchor)
        b_blocks.append(np.sqrt(float(opt["lambda_eta"])) * e3_full[:4])
        if sphys.shape[1] and tree_l.size:
            tree = np.zeros((tree_l.shape[0], p), dtype=float)
            tree[:, 4:] = tree_l
            a_blocks.append(np.sqrt(float(opt["lambda_tree"])) * tree)
            b_blocks.append(np.zeros(tree.shape[0], dtype=float))
        a_aug = np.vstack(a_blocks)
        b_aug = np.concatenate(b_blocks)
        solved = lsq_linear(a_aug, b_aug, bounds=(lower, upper), lsmr_tol="auto", max_iter=200)
        if not solved.success and not np.all(np.isfinite(solved.x)):
            break
        coeff = np.asarray(solved.x, dtype=float)
        delta = np.linalg.norm(coeff - last) / (np.linalg.norm(last) + EPS)
        last = coeff.copy()
        if delta <= float(opt["inner_tolerance"]):
            converged = bool(solved.success)
            break
    residual = observed - x @ coeff
    objective = float(np.mean(tukey_rho(residual / max(scale, EPS))))
    if p >= 4:
        objective += float(opt["lambda_eta"]) * float(np.sum((coeff[:4] - e3_full[:4]) ** 2))
    if sphys.shape[1] and tree_l.size:
        objective += float(opt["lambda_tree"]) * float(np.sum((tree_l @ coeff[4:]) ** 2))
    return coeff, objective, converged, iterations, residual


def residualized_energy_ratio(vector: np.ndarray, against: np.ndarray, weights: np.ndarray) -> float:
    residual = weighted_residualize(vector[:, None], against, weights)[:, 0]
    top = float(residual @ (weights * residual))
    bottom = float(vector @ (weights * vector)) + EPS
    return float(top / bottom)


def compute_consensus(
    observed: np.ndarray,
    pred3: np.ndarray,
    pred4: np.ndarray,
    scale: float,
    config: dict[str, Any],
) -> tuple[float, float]:
    count = int(config["consensus"]["window_count"])
    overlap = float(config["consensus"]["window_overlap_fraction"])
    n = observed.size
    deltas: list[float] = []
    for k in range(count):
        left = int(max(0, np.floor(k * n / count - overlap * n / count)))
        right = int(min(n, np.ceil((k + 1) * n / count + overlap * n / count)))
        if right <= left:
            continue
        j3 = float(np.mean(tukey_rho((observed[left:right] - pred3[left:right]) / max(scale, EPS))))
        j4 = float(np.mean(tukey_rho((observed[left:right] - pred4[left:right]) / max(scale, EPS))))
        deltas.append((j3 - j4) / max(abs(j3), EPS))
    if not deltas:
        return 0.0, float("inf")
    values = np.asarray(deltas, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, mad


def refine_e4_horp(
    *,
    observation_id: str,
    observed: np.ndarray,
    wavelength_nm: np.ndarray,
    spectrum,
    e3_thickness_nm: float,
    e3_coefficients: np.ndarray,
    e3_scale: float,
    e3_objective: float,
    thickness_bounds: tuple[float, float],
    config: dict[str, Any],
) -> E4Result:
    started = perf_counter()
    d3 = float(e3_thickness_nm)
    scale = max(float(e3_scale), EPS)
    if not np.isfinite(d3) or not np.isfinite(scale) or e3_coefficients.size < 4:
        return E4Result(observation_id, d3, d3, d3, False, False, "invalid_e3_anchor", float("nan"), float("nan"), 0.0, 0.0, 0.0, float("nan"), float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), 0, float("inf"), 0, False, False, 0, 0, perf_counter() - started, "[]")
    g3 = float(e3_coefficients[0])
    b2 = legendre_b2(wavelength_nm)
    f3 = np.asarray(spectrum(d3), dtype=float)
    pred3 = g3 * f3 + b2 @ e3_coefficients[1:4]
    r3 = observed - pred3
    weights = tukey_weight(r3 / scale)
    sphys, meta, geom = build_physical_drift_space(
        wavelength_nm,
        spectrum,
        d3,
        g3,
        weights,
        config,
        thickness_bounds,
    )
    rank = int(geom["drift_effective_rank"])
    if rank <= 0:
        return E4Result(observation_id, d3, d3, d3, False, False, "drift_subspace_rank_deficient", float(e3_objective), float(e3_objective), 0.0, 0.0, 0.0, float(geom["first_order_projection_qc"]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), rank, float(geom["drift_condition_number"]), int(sphys.shape[1]), bool(geom["second_order_protection_active"]), False, 0, 0, perf_counter() - started, "[]")
    trust = config["trust_region"]
    hcurv = float(trust["curvature_step_nm"])
    try:
        left = max(float(thickness_bounds[0]), d3 - hcurv)
        right = min(float(thickness_bounds[1]), d3 + hcurv)
        j_left = float(np.mean(tukey_rho((observed - (g3 * spectrum(left) + b2 @ e3_coefficients[1:4])) / scale)))
        j_mid = float(np.mean(tukey_rho(r3 / scale)))
        j_right = float(np.mean(tukey_rho((observed - (g3 * spectrum(right) + b2 @ e3_coefficients[1:4])) / scale)))
        curvature = max((j_left - 2.0 * j_mid + j_right) / max(hcurv * hcurv, EPS), 0.0)
        kbar = (float(trust["trust_radius_max_nm"]) ** 2) * curvature / max(abs(j_mid), EPS)
        radius = float(trust["trust_radius_max_nm"]) / np.sqrt(1.0 + kbar)
        radius = float(np.clip(radius, float(trust["trust_radius_min_nm"]), float(trust["trust_radius_max_nm"])))
    except Exception:
        curvature = float("nan")
        radius = float(trust["curvature_fallback_radius_nm"])
    low = max(float(thickness_bounds[0]), d3 - radius)
    high = min(float(thickness_bounds[1]), d3 + radius)

    coeff_anchor, obj_anchor, ok_anchor, it_anchor, residual_anchor = fit_inner(
        observed, f3, b2, sphys, scale, e3_coefficients[:4], weights, meta, config
    )
    coeff_base, obj_base, _, _, _ = fit_inner(
        observed, f3, b2, np.zeros((observed.size, 0)), scale, e3_coefficients[:4], weights, [], config
    )
    gb = (obj_base - obj_anchor) / max(abs(obj_base), EPS)
    drift_anchor = sphys @ coeff_anchor[4:]
    ab = float(
        np.sqrt(np.sum(weights * drift_anchor * drift_anchor))
        / (np.sqrt(np.sum(weights * r3 * r3)) + EPS)
    )
    t_eta = np.column_stack([f3, b2])
    gd = g3 * geom["fp"]
    id3 = residualized_energy_ratio(gd, t_eta, weights)
    id4 = residualized_energy_ratio(gd, np.column_stack([t_eta, sphys]), weights)
    ri = id4 / (id3 + EPS)
    h2 = np.asarray(geom["h2_perp"], dtype=float)
    denom = float((h2 @ (weights * h2)) * (drift_anchor @ (weights * drift_anchor)) + EPS)
    leak2 = float(((h2 @ (weights * drift_anchor)) ** 2) / denom)

    gates = config["gates"]
    triggered = bool(
        gb >= float(gates["tau_trigger_gain"])
        and ab <= float(gates["tau_drift_amplitude"])
        and ri >= float(gates["tau_identifiability_retention"])
        and leak2 <= float(gates["tau_second_order_leakage"])
        and float(geom["first_order_projection_qc"]) <= float(config["physical_projection"]["first_order_projection_tolerance"])
    )
    if not triggered or high <= low:
        reason = "trigger_gate_failed" if high > low else "empty_trust_region"
        return E4Result(observation_id, d3, d3, d3, triggered, False, reason, obj_anchor, obj_anchor, 0.0, gb, ab, float(geom["first_order_projection_qc"]), leak2, id3, id4, ri, 0.0, float("inf"), radius, curvature, rank, float(geom["drift_condition_number"]), int(sphys.shape[1]), bool(geom["second_order_protection_active"]), ok_anchor, it_anchor, 0, perf_counter() - started, json.dumps(coeff_anchor.tolist()))

    cache: dict[float, tuple[np.ndarray, float, bool, int, np.ndarray]] = {}

    def profile(d: float) -> float:
        key = round(float(d), 9)
        if key not in cache:
            cache[key] = fit_inner(
                observed,
                np.asarray(spectrum(float(d)), dtype=float),
                b2,
                sphys,
                scale,
                e3_coefficients[:4],
                weights,
                meta,
                config,
            )
        return cache[key][1]

    optimized = minimize_scalar(
        profile,
        bounds=(low, high),
        method="bounded",
        options={"xatol": float(config["optimization"]["outer_xatol_nm"]), "maxiter": int(config["optimization"]["outer_max_nfev"])},
    )
    d4 = float(optimized.x) if optimized.success and np.isfinite(optimized.x) else d3
    coeff_candidate, obj_candidate, ok_candidate, it_candidate, residual_candidate = cache.get(
        round(d4, 9),
        fit_inner(observed, np.asarray(spectrum(d4), dtype=float), b2, sphys, scale, e3_coefficients[:4], weights, meta, config),
    )
    improvement = (obj_anchor - obj_candidate) / max(abs(obj_anchor), EPS)
    pred4 = observed - residual_candidate
    consensus_median, consensus_mad = compute_consensus(observed, pred3, pred4, scale, config)
    distance = abs(d4 - d3) / max(radius, EPS)
    inside = d4 > low + float(trust["boundary_tolerance_nm"]) and d4 < high - float(trust["boundary_tolerance_nm"])
    accepted = bool(
        np.isfinite(d4)
        and inside
        and ok_candidate
        and 0.0 <= coeff_candidate[0] <= 2.0
        and improvement >= float(gates["tau_accept_gain"])
        and ri >= float(gates["tau_identifiability_retention"])
        and leak2 <= float(gates["tau_second_order_leakage"])
        and consensus_median >= float(gates["tau_consensus"])
    )
    if accepted:
        final = d4
        reason = "accepted"
    else:
        final = d3
        reason = "accept_gate_failed"
        if not inside:
            reason = "candidate_on_trust_boundary"
        elif not ok_candidate:
            reason = "inner_solver_not_converged"
        elif improvement < float(gates["tau_accept_gain"]):
            reason = "insufficient_objective_gain"
        elif consensus_median < float(gates["tau_consensus"]):
            reason = "cross_window_consensus_failed"
    _ = distance
    return E4Result(
        observation_id=observation_id,
        e3_thickness_nm=d3,
        e4_candidate_thickness_nm=d4,
        e4_final_thickness_nm=final,
        triggered=triggered,
        accepted=accepted,
        fallback_reason=reason,
        objective_anchor=float(obj_anchor),
        objective_candidate=float(obj_candidate),
        objective_relative_improvement=float(improvement),
        baseline_gain_gb=float(gb),
        baseline_amplitude_ratio_ab=float(ab),
        first_order_projection_qc=float(geom["first_order_projection_qc"]),
        second_order_leakage=float(leak2),
        identifiability_e3=float(id3),
        identifiability_e4=float(id4),
        identifiability_retention_ratio=float(ri),
        consensus_median=float(consensus_median),
        consensus_mad=float(consensus_mad),
        trust_radius_nm=float(radius),
        profile_curvature=float(curvature),
        drift_effective_rank=rank,
        drift_condition_number=float(geom["drift_condition_number"]),
        drift_active_columns=int(sphys.shape[1]),
        second_order_protection_active=bool(geom["second_order_protection_active"]),
        inner_converged=bool(ok_candidate),
        inner_iterations=int(max(it_anchor, it_candidate)),
        outer_nfev=int(getattr(optimized, "nfev", 0)),
        runtime_e4_seconds=float(perf_counter() - started),
        e4_coefficients_json=json.dumps([float(v) for v in coeff_candidate]),
    )
