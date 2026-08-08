from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar

from .robust import LOSS_SPECS, LossSpec, mad_scale, rho, weight


SpectrumFunction = Callable[[float], np.ndarray]


@dataclass
class LinearFit:
    coefficients: np.ndarray
    residual: np.ndarray
    objective: float
    converged: bool
    iterations: int
    selected_start: str
    starts_attempted: int
    converged_starts: int
    start_objectives: tuple[float, ...]


@dataclass
class FitResult:
    thickness_nm: float
    objective: float
    coefficients: np.ndarray
    residual_scale: float
    converged: bool
    boundary_hit: bool
    flat_objective: bool
    gain_lower_hit: bool
    gain_upper_hit: bool
    evaluations: int
    irls_iterations: int
    runtime_ms: float
    status: str
    selected_start: str
    starts_attempted: int
    converged_starts: int
    candidate_basins: int
    start_objectives: tuple[float, ...]

    @property
    def gain(self) -> float:
        return float(self.coefficients[0])


def normalized_axis(wavelength_nm: np.ndarray) -> np.ndarray:
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    return 2.0 * (wavelength_nm - wavelength_nm.min()) / np.ptp(wavelength_nm) - 1.0


def baseline_matrix(wavelength_nm: np.ndarray, order: int) -> np.ndarray:
    """Legendre baseline basis, including an intercept for order >= 0."""
    if order < 0:
        return np.zeros((np.asarray(wavelength_nm).size, 0), dtype=float)
    return np.polynomial.legendre.legvander(normalized_axis(wavelength_nm), order)


def complete_nuisance_design(model: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return np.column_stack([np.asarray(model, dtype=float), np.asarray(baseline, dtype=float)])


def _coefficient_bounds(column_count: int, gain_bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(column_count, -np.inf, dtype=float)
    upper = np.full(column_count, np.inf, dtype=float)
    lower[0], upper[0] = gain_bounds
    return lower, upper


def _bounded_l2(
    design: np.ndarray,
    target: np.ndarray,
    gain_bounds: tuple[float, float],
    sqrt_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    x = design
    y = target
    if sqrt_weight is not None:
        x = x * sqrt_weight[:, None]
        y = y * sqrt_weight
    result = lsq_linear(
        x,
        y,
        bounds=_coefficient_bounds(design.shape[1], gain_bounds),
        method="trf",
        tol=1e-12,
        lsmr_tol="auto",
        max_iter=200,
    )
    coefficients = np.asarray(result.x, dtype=float)
    return coefficients, target - design @ coefficients, bool(result.success)


def _seed_coefficients(
    design: np.ndarray,
    target: np.ndarray,
    gain: float,
    gain_bounds: tuple[float, float],
) -> np.ndarray:
    gain = float(np.clip(gain, *gain_bounds))
    model = design[:, 0]
    baseline = design[:, 1:]
    if baseline.shape[1]:
        beta, *_ = np.linalg.lstsq(baseline, target - gain * model, rcond=None)
        return np.concatenate([[gain], np.asarray(beta, dtype=float)])
    return np.array([gain], dtype=float)


def constrained_linear_fit(
    design: np.ndarray,
    target: np.ndarray,
    spec: LossSpec,
    scale: float,
    gain_bounds: tuple[float, float],
    gain_starts: tuple[float, ...],
    max_iterations: int,
    coefficient_tolerance: float,
    objective_tolerance: float,
) -> LinearFit:
    """Fit the same complete nuisance design for every loss.

    L2 uses the unique bounded least-squares solution. Robust losses use the
    same deterministic gain starts and retain the best finite converged
    solution. If no start converges, the lowest finite objective is retained
    and marked as failed instead of being silently discarded.
    """
    target = np.asarray(target, dtype=float)
    if spec.name == "L2":
        coefficients, residual, ok = _bounded_l2(design, target, gain_bounds)
        objective = float(np.sum(rho(residual / scale, spec)))
        return LinearFit(
            coefficients,
            residual,
            objective,
            ok and np.isfinite(objective),
            1,
            "bounded_l2",
            1,
            int(ok),
            (objective,),
        )

    candidates: list[LinearFit] = []
    for start_gain in gain_starts:
        coefficients = _seed_coefficients(design, target, start_gain, gain_bounds)
        previous_objective = float(np.sum(rho((target - design @ coefficients) / scale, spec)))
        converged = False
        iteration = 0
        for iteration in range(1, max_iterations + 1):
            residual = target - design @ coefficients
            weights = np.clip(weight(residual / scale, spec), 1e-12, 1.0)
            updated, _, solver_ok = _bounded_l2(design, target, gain_bounds, np.sqrt(weights))
            updated_residual = target - design @ updated
            updated_objective = float(np.sum(rho(updated_residual / scale, spec)))
            coefficient_change = float(np.linalg.norm(updated - coefficients)) / max(
                1.0, float(np.linalg.norm(coefficients))
            )
            objective_change = abs(updated_objective - previous_objective) / max(
                1.0, abs(previous_objective)
            )
            coefficients = updated
            previous_objective = updated_objective
            if solver_ok and coefficient_change <= coefficient_tolerance and objective_change <= objective_tolerance:
                converged = True
                break
        residual = target - design @ coefficients
        objective = float(np.sum(rho(residual / scale, spec)))
        candidates.append(
            LinearFit(
                coefficients=coefficients,
                residual=residual,
                objective=objective,
                converged=converged and np.isfinite(objective),
                iterations=iteration,
                selected_start=f"gain={start_gain:g}",
                starts_attempted=len(gain_starts),
                converged_starts=0,
                start_objectives=(),
            )
        )

    finite = [candidate for candidate in candidates if np.isfinite(candidate.objective)]
    converged_candidates = [candidate for candidate in finite if candidate.converged]
    pool = converged_candidates or finite
    if not pool:
        nan = np.full(design.shape[1], np.nan)
        return LinearFit(
            nan,
            np.full_like(target, np.nan),
            float("inf"),
            False,
            max_iterations,
            "none",
            len(gain_starts),
            0,
            tuple(float("inf") for _ in gain_starts),
        )
    selected = min(pool, key=lambda candidate: candidate.objective)
    objectives = tuple(float(candidate.objective) for candidate in candidates)
    selected.converged_starts = len(converged_candidates)
    selected.start_objectives = objectives
    return selected


class ConstrainedThicknessEstimator:
    def __init__(
        self,
        wavelength_nm: np.ndarray,
        spectrum_function: SpectrumFunction,
        thickness_bounds: tuple[float, float],
        gain_bounds: tuple[float, float],
        coarse_step_nm: float,
        local_tolerance_nm: float,
        gain_starts: tuple[float, ...],
        max_irls_iterations: int,
        coefficient_tolerance: float,
        objective_tolerance: float,
        candidate_basin_count: int,
    ) -> None:
        self.wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        self.spectrum_function = spectrum_function
        self.thickness_bounds = tuple(map(float, thickness_bounds))
        self.gain_bounds = tuple(map(float, gain_bounds))
        self.coarse_step_nm = float(coarse_step_nm)
        self.local_tolerance_nm = float(local_tolerance_nm)
        self.gain_starts = tuple(map(float, gain_starts))
        self.max_irls_iterations = int(max_irls_iterations)
        self.coefficient_tolerance = float(coefficient_tolerance)
        self.objective_tolerance = float(objective_tolerance)
        self.candidate_basin_count = int(candidate_basin_count)
        self.grid = np.arange(
            self.thickness_bounds[0],
            self.thickness_bounds[1] + 0.5 * self.coarse_step_nm,
            self.coarse_step_nm,
        )

    def _linear(
        self,
        observed: np.ndarray,
        d: float,
        baseline: np.ndarray,
        spec: LossSpec,
        scale: float,
        gain_starts_override: tuple[float, ...] | None = None,
    ) -> LinearFit:
        design = complete_nuisance_design(self.spectrum_function(float(d)), baseline)
        return constrained_linear_fit(
            design,
            observed,
            spec,
            scale,
            self.gain_bounds,
            self.gain_starts if gain_starts_override is None else gain_starts_override,
            self.max_irls_iterations,
            self.coefficient_tolerance,
            self.objective_tolerance,
        )

    def _separated_candidates(self, values: np.ndarray, count: int) -> list[int]:
        order = np.argsort(np.where(np.isfinite(values), values, np.inf))
        selected: list[int] = []
        for index in order:
            if not np.isfinite(values[index]):
                continue
            if all(abs(int(index) - previous) > 1 for previous in selected):
                selected.append(int(index))
            if len(selected) == count:
                break
        return selected or [int(np.nanargmin(values))]

    def _finalize(
        self,
        d: float,
        fit: LinearFit,
        scale: float,
        start: float,
        evaluations: int,
        candidate_basins: int,
        outer_ok: bool,
        flat: bool,
    ) -> FitResult:
        edge_tolerance = max(1e-5, 0.01 * self.coarse_step_nm)
        thickness_boundary = (
            d <= self.thickness_bounds[0] + edge_tolerance
            or d >= self.thickness_bounds[1] - edge_tolerance
        )
        gain = float(fit.coefficients[0]) if fit.coefficients.size else float("nan")
        gain_tolerance = 1e-7 * max(1.0, self.gain_bounds[1] - self.gain_bounds[0])
        gain_lower_hit = np.isfinite(gain) and gain <= self.gain_bounds[0] + gain_tolerance
        gain_upper_hit = np.isfinite(gain) and gain >= self.gain_bounds[1] - gain_tolerance
        converged = bool(
            outer_ok
            and fit.converged
            and np.isfinite(fit.objective)
            and not thickness_boundary
            and not flat
        )
        if converged:
            status = "ok"
        elif thickness_boundary:
            status = "thickness_boundary"
        elif flat:
            status = "flat_objective"
        elif not fit.converged:
            status = "inner_nonconvergence"
        else:
            status = "outer_nonconvergence"
        return FitResult(
            thickness_nm=float(d),
            objective=float(fit.objective),
            coefficients=np.asarray(fit.coefficients, dtype=float),
            residual_scale=float(scale),
            converged=converged,
            boundary_hit=thickness_boundary,
            flat_objective=flat,
            gain_lower_hit=bool(gain_lower_hit),
            gain_upper_hit=bool(gain_upper_hit),
            evaluations=int(evaluations),
            irls_iterations=int(fit.iterations),
            runtime_ms=1000.0 * (perf_counter() - start),
            status=status,
            selected_start=fit.selected_start,
            starts_attempted=int(fit.starts_attempted),
            converged_starts=int(fit.converged_starts),
            candidate_basins=int(candidate_basins),
            start_objectives=tuple(fit.start_objectives),
        )

    def fit_l2_profile(self, observed: np.ndarray, baseline_order: int, scale: float = 1.0) -> FitResult:
        start = perf_counter()
        observed = np.asarray(observed, dtype=float)
        baseline = baseline_matrix(self.wavelength_nm, baseline_order)
        spec = LOSS_SPECS["L2"]
        cache: dict[float, LinearFit] = {}

        def fit_at(d: float) -> LinearFit:
            key = float(d)
            if key not in cache:
                cache[key] = self._linear(observed, key, baseline, spec, scale)
            return cache[key]

        grid_values = np.array([fit_at(float(d)).objective for d in self.grid])
        candidate_indices = self._separated_candidates(grid_values, self.candidate_basin_count)
        refined: list[tuple[float, LinearFit, bool]] = []
        for index in candidate_indices:
            lower = max(self.thickness_bounds[0], self.grid[index] - 1.1 * self.coarse_step_nm)
            upper = min(self.thickness_bounds[1], self.grid[index] + 1.1 * self.coarse_step_nm)
            optimized = minimize_scalar(
                lambda d: fit_at(float(d)).objective,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": self.local_tolerance_nm},
            )
            d = float(optimized.x)
            refined.append((d, fit_at(d), bool(optimized.success)))
        converged = [item for item in refined if item[1].converged and np.isfinite(item[1].objective)]
        finite = [item for item in refined if np.isfinite(item[1].objective)]
        pool = converged or finite
        if not pool:
            d = float(self.grid[candidate_indices[0]])
            fit = fit_at(d)
            outer_ok = False
        else:
            d, fit, outer_ok = min(pool, key=lambda item: item[1].objective)
        candidate_values = np.array([item[1].objective for item in refined], dtype=float)
        finite_values = np.sort(candidate_values[np.isfinite(candidate_values)])
        flat = finite_values.size < 2 or abs(finite_values[1] - finite_values[0]) <= 1e-12 * max(
            1.0, abs(finite_values[0])
        )
        return self._finalize(
            d,
            fit,
            scale,
            start,
            len(cache),
            len(candidate_indices),
            outer_ok,
            flat,
        )

    def fit_one_step_score(
        self,
        observed: np.ndarray,
        loss_name: str,
        baseline_order: int,
        scale: float,
    ) -> FitResult:
        """Fair projection score: full-design bounded L2 nuisance fit, robust score."""
        start = perf_counter()
        observed = np.asarray(observed, dtype=float)
        baseline = baseline_matrix(self.wavelength_nm, baseline_order)
        score_spec = LOSS_SPECS[loss_name]
        cache: dict[float, LinearFit] = {}

        def fit_at(d: float) -> LinearFit:
            key = float(d)
            if key not in cache:
                l2 = self._linear(observed, key, baseline, LOSS_SPECS["L2"], scale)
                objective = float(np.sum(rho(l2.residual / scale, score_spec)))
                cache[key] = LinearFit(
                    l2.coefficients,
                    l2.residual,
                    objective,
                    l2.converged,
                    l2.iterations,
                    "full_design_bounded_l2",
                    1,
                    int(l2.converged),
                    (objective,),
                )
            return cache[key]

        grid_values = np.array([fit_at(float(d)).objective for d in self.grid])
        candidate_indices = self._separated_candidates(grid_values, self.candidate_basin_count)
        refined: list[tuple[float, LinearFit, bool]] = []
        for index in candidate_indices:
            lower = max(self.thickness_bounds[0], self.grid[index] - 1.1 * self.coarse_step_nm)
            upper = min(self.thickness_bounds[1], self.grid[index] + 1.1 * self.coarse_step_nm)
            optimized = minimize_scalar(
                lambda d: fit_at(float(d)).objective,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": self.local_tolerance_nm},
            )
            d = float(optimized.x)
            refined.append((d, fit_at(d), bool(optimized.success)))
        converged = [item for item in refined if item[1].converged and np.isfinite(item[1].objective)]
        finite = [item for item in refined if np.isfinite(item[1].objective)]
        pool = converged or finite
        if not pool:
            d = float(self.grid[candidate_indices[0]])
            fit = fit_at(d)
            outer_ok = False
        else:
            d, fit, outer_ok = min(pool, key=lambda item: item[1].objective)
        candidate_values = np.array([item[1].objective for item in refined], dtype=float)
        finite_values = np.sort(candidate_values[np.isfinite(candidate_values)])
        flat = finite_values.size < 2 or abs(finite_values[1] - finite_values[0]) <= 1e-12 * max(
            1.0, abs(finite_values[0])
        )
        return self._finalize(
            d,
            fit,
            scale,
            start,
            len(cache),
            len(candidate_indices),
            outer_ok,
            flat,
        )

    def fit_robust_profile(
        self,
        observed: np.ndarray,
        loss_name: str,
        baseline_order: int,
        scale: float,
    ) -> FitResult:
        """Multibasin thickness search with deterministic multistart robust IRLS."""
        start = perf_counter()
        observed = np.asarray(observed, dtype=float)
        baseline = baseline_matrix(self.wavelength_nm, baseline_order)
        spec = LOSS_SPECS[loss_name]

        # Candidate basins are defined by the fair one-step score, so candidate
        # generation does not privilege robust nuisance re-estimation.
        one_step_cache: dict[float, float] = {}
        local_cache: dict[float, LinearFit] = {}
        validation_cache: dict[float, LinearFit] = {}

        def one_step_score(d: float) -> float:
            key = float(d)
            if key not in one_step_cache:
                l2 = self._linear(observed, key, baseline, LOSS_SPECS["L2"], scale)
                one_step_cache[key] = float(np.sum(rho(l2.residual / scale, spec)))
            return one_step_cache[key]

        def local_at(d: float) -> LinearFit:
            key = float(d)
            if key not in local_cache:
                # The deterministic constrained-L2 start traces each candidate
                # basin cheaply. Full gain multistart is then used to validate
                # every refined basin before a solution is selected.
                local_cache[key] = self._linear(
                    observed,
                    key,
                    baseline,
                    spec,
                    scale,
                    gain_starts_override=(1.0,),
                )
            return local_cache[key]

        def validate_at(d: float) -> LinearFit:
            key = float(d)
            if key not in validation_cache:
                validation_cache[key] = self._linear(observed, key, baseline, spec, scale)
            return validation_cache[key]

        grid_scores = np.array([one_step_score(float(d)) for d in self.grid])
        candidate_indices = self._separated_candidates(grid_scores, self.candidate_basin_count)
        refined: list[tuple[float, LinearFit, bool]] = []
        for index in candidate_indices:
            lower = max(self.thickness_bounds[0], self.grid[index] - 1.1 * self.coarse_step_nm)
            upper = min(self.thickness_bounds[1], self.grid[index] + 1.1 * self.coarse_step_nm)
            optimized = minimize_scalar(
                lambda d: local_at(float(d)).objective,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": self.local_tolerance_nm},
            )
            d = float(optimized.x)
            refined.append((d, validate_at(d), bool(optimized.success)))

        converged = [item for item in refined if item[1].converged and np.isfinite(item[1].objective)]
        finite = [item for item in refined if np.isfinite(item[1].objective)]
        pool = converged or finite
        if not pool:
            d = float(self.grid[candidate_indices[0]])
            fit = validate_at(d)
            outer_ok = False
        else:
            d, fit, outer_ok = min(pool, key=lambda item: item[1].objective)
        candidate_values = np.array([item[1].objective for item in refined], dtype=float)
        finite_values = np.sort(candidate_values[np.isfinite(candidate_values)])
        flat = finite_values.size < 2 or abs(finite_values[1] - finite_values[0]) <= 1e-12 * max(
            1.0, abs(finite_values[0])
        )
        evaluations = len(one_step_cache) + len(local_cache) + len(validation_cache)
        return self._finalize(
            d,
            fit,
            scale,
            start,
            evaluations,
            len(candidate_indices),
            outer_ok,
            flat,
        )

    def estimate_adaptive_scale(
        self,
        observed: np.ndarray,
        baseline_order: int,
        floor: float,
    ) -> tuple[float, FitResult]:
        preliminary = self.fit_l2_profile(observed, baseline_order, scale=1.0)
        model = self.spectrum_function(preliminary.thickness_nm)
        design = complete_nuisance_design(model, baseline_matrix(self.wavelength_nm, baseline_order))
        coefficients, residual, _ = _bounded_l2(design, np.asarray(observed, dtype=float), self.gain_bounds)
        _ = coefficients
        return mad_scale(residual, floor=floor), preliminary
