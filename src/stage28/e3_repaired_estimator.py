from __future__ import annotations

from dataclasses import dataclass, fields
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from path_b_scientific_reconstruction_v2.src.estimators import (
    ConstrainedThicknessEstimator,
    FitResult,
    LinearFit,
    baseline_matrix,
)
from path_b_scientific_reconstruction_v2.src.robust import LOSS_SPECS


@dataclass(frozen=True)
class CandidateDiagnostic:
    candidate_id: str
    coarse_index: int
    coarse_thickness_nm: float
    interval_lower_nm: float
    interval_upper_nm: float
    refined_thickness_nm: float
    objective: float
    outer_success: bool
    inner_converged: bool
    status: str
    deduplication_status: str
    selected_start: str
    starts_attempted: int
    converged_starts: int
    start_objectives: tuple[float, ...]
    representative_candidate_id: str
    eligible_for_selection: bool


@dataclass
class E3FitResult(FitResult):
    candidate_diagnostics: tuple[CandidateDiagnostic, ...] = ()
    coarse_candidate_indices: tuple[int, ...] = ()
    coarse_grid_objectives: tuple[float, ...] = ()


@dataclass
class _CandidateAttempt:
    candidate_id: str
    coarse_index: int
    coarse_thickness_nm: float
    interval_lower_nm: float
    interval_upper_nm: float
    refined_thickness_nm: float
    fit: LinearFit
    outer_success: bool
    representative_candidate_id: str = ""
    eligible_for_selection: bool = False
    status: str = "unclassified"


def _comparison_tolerance(a: float, b: float, relative_tolerance: float) -> float:
    return relative_tolerance * max(1.0, abs(float(a)), abs(float(b)))


def _nearly_equal(a: float, b: float, relative_tolerance: float) -> bool:
    return abs(float(a) - float(b)) <= _comparison_tolerance(a, b, relative_tolerance)


def discover_discrete_local_minima(
    values: Sequence[float],
    *,
    relative_tolerance: float = 1e-12,
) -> list[int]:
    """Return deterministic representatives of finite local-minimum plateaus."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be nonnegative")

    runs: list[list[int]] = []
    index = 0
    while index < array.size:
        if not np.isfinite(array[index]):
            index += 1
            continue
        run = [index]
        anchor_value = float(array[index])
        while (
            run[-1] + 1 < array.size
            and np.isfinite(array[run[-1] + 1])
            and _nearly_equal(anchor_value, array[run[-1] + 1], relative_tolerance)
        ):
            run.append(run[-1] + 1)
        runs.append(run)
        index = run[-1] + 1

    selected: list[int] = []
    for run in runs:
        run_values = array[run]
        minimum = float(np.min(run_values))
        midpoint = 0.5 * (run[0] + run[-1])
        tied = [i for i in run if _nearly_equal(array[i], minimum, relative_tolerance)]
        representative = min(tied, key=lambda i: (abs(i - midpoint), i))
        representative_value = float(array[representative])

        left = run[0] - 1
        right = run[-1] + 1
        neighbors = []
        if left >= 0 and np.isfinite(array[left]):
            neighbors.append(float(array[left]))
        if right < array.size and np.isfinite(array[right]):
            neighbors.append(float(array[right]))
        if not neighbors:
            selected.append(representative)
            continue

        no_greater = all(representative_value <= neighbor for neighbor in neighbors)
        strictly_lower = any(
            representative_value
            < neighbor - _comparison_tolerance(representative_value, neighbor, relative_tolerance)
            for neighbor in neighbors
        )
        boundary_run = run[0] == 0 or run[-1] == array.size - 1
        if no_greater and (strictly_lower or boundary_run):
            selected.append(representative)
    return selected


def build_refinement_intervals(
    grid: Sequence[float],
    candidate_indices: Sequence[int],
) -> list[tuple[float, float]]:
    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("grid must be a strictly increasing one-dimensional array")
    indices = [int(i) for i in candidate_indices]
    if indices != sorted(set(indices)):
        raise ValueError("candidate indices must be sorted and unique")
    if any(i < 0 or i >= values.size for i in indices):
        raise ValueError("candidate index outside grid")

    intervals = [
        [float(values[max(0, i - 1)]), float(values[min(values.size - 1, i + 1)])]
        for i in indices
    ]
    for position in range(len(indices) - 1):
        if intervals[position][1] > intervals[position + 1][0]:
            midpoint = 0.5 * (float(values[indices[position]]) + float(values[indices[position + 1]]))
            intervals[position][1] = midpoint
            intervals[position + 1][0] = midpoint
    return [(lower, upper) for lower, upper in intervals]


def deduplicate_refined_candidates(
    thicknesses_nm: Sequence[float],
    objectives: Sequence[float],
    preliminary_success: Sequence[bool],
    source_indices: Sequence[int],
    *,
    tolerance_nm: float,
) -> list[int]:
    """Map each candidate to the representative candidate position."""
    count = len(thicknesses_nm)
    if not (len(objectives) == len(preliminary_success) == len(source_indices) == count):
        raise ValueError("candidate fields must have identical lengths")
    if tolerance_nm < 0:
        raise ValueError("tolerance_nm must be nonnegative")
    order = sorted(range(count), key=lambda i: (float(thicknesses_nm[i]), int(source_indices[i])))
    groups: list[list[int]] = []
    for position in order:
        if not groups or abs(float(thicknesses_nm[position]) - float(thicknesses_nm[groups[-1][-1]])) > tolerance_nm:
            groups.append([position])
        else:
            groups[-1].append(position)

    representative = list(range(count))
    for group in groups:
        best = min(
            group,
            key=lambda i: (
                not bool(preliminary_success[i]),
                not np.isfinite(float(objectives[i])),
                float(objectives[i]) if np.isfinite(float(objectives[i])) else float("inf"),
                int(source_indices[i]),
            ),
        )
        for position in group:
            representative[position] = best
    return representative


def _profile_is_flat(values: np.ndarray, relative_tolerance: float) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return False
    low = float(np.min(finite))
    high = float(np.max(finite))
    return high - low <= relative_tolerance * max(1.0, abs(low), abs(high))


class RepairedConstrainedThicknessEstimator(ConstrainedThicknessEstimator):
    """E3 implementation with candidate discovery on E3's own robust profile."""

    def __init__(
        self,
        *args,
        local_minimum_relative_tolerance: float = 1e-12,
        refined_candidate_deduplication_nm: float = 1e-4,
        flat_objective_relative_tolerance: float = 1e-12,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.local_minimum_relative_tolerance = float(local_minimum_relative_tolerance)
        self.refined_candidate_deduplication_nm = float(refined_candidate_deduplication_nm)
        self.flat_objective_relative_tolerance = float(flat_objective_relative_tolerance)

    def _as_e3_result(
        self,
        result: FitResult,
        diagnostics: Sequence[CandidateDiagnostic],
        indices: Sequence[int],
        grid_objectives: Sequence[float],
        *,
        status_override: str | None = None,
    ) -> E3FitResult:
        payload = {field.name: getattr(result, field.name) for field in fields(FitResult)}
        if status_override is not None:
            payload["status"] = status_override
            payload["converged"] = False
        return E3FitResult(
            **payload,
            candidate_diagnostics=tuple(diagnostics),
            coarse_candidate_indices=tuple(int(i) for i in indices),
            coarse_grid_objectives=tuple(float(value) for value in grid_objectives),
        )

    def fit_robust_profile(
        self,
        observed: np.ndarray,
        loss_name: str,
        baseline_order: int,
        scale: float,
    ) -> E3FitResult:
        start = perf_counter()
        observed = np.asarray(observed, dtype=float)
        baseline = baseline_matrix(self.wavelength_nm, baseline_order)
        spec = LOSS_SPECS[loss_name]
        cache: dict[float, LinearFit] = {}

        def fit_at(thickness_nm: float) -> LinearFit:
            key = float(thickness_nm)
            if key not in cache:
                cache[key] = self._linear(observed, key, baseline, spec, scale)
            return cache[key]

        grid_objectives = np.array([fit_at(float(d)).objective for d in self.grid], dtype=float)
        candidate_indices = discover_discrete_local_minima(
            grid_objectives,
            relative_tolerance=self.local_minimum_relative_tolerance,
        )
        flat = _profile_is_flat(grid_objectives, self.flat_objective_relative_tolerance)

        if not candidate_indices:
            coefficients = np.full(baseline.shape[1] + 1, np.nan, dtype=float)
            failed_fit = LinearFit(
                coefficients=coefficients,
                residual=np.full_like(observed, np.nan),
                objective=float("inf"),
                converged=False,
                iterations=0,
                selected_start="none",
                starts_attempted=len(self.gain_starts),
                converged_starts=0,
                start_objectives=tuple(float("inf") for _ in self.gain_starts),
            )
            base = self._finalize(
                float("nan"), failed_fit, scale, start, len(cache), 0, False, flat
            )
            return self._as_e3_result(
                base,
                (),
                (),
                grid_objectives,
                status_override="no_finite_e3_candidate",
            )

        intervals = build_refinement_intervals(self.grid, candidate_indices)
        attempts: list[_CandidateAttempt] = []
        for ordinal, (index, interval) in enumerate(zip(candidate_indices, intervals, strict=True)):
            lower, upper = interval
            optimized = minimize_scalar(
                lambda d: fit_at(float(d)).objective,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": self.local_tolerance_nm},
            )
            trial_thicknesses = [float(optimized.x), float(lower), float(upper), float(self.grid[index])]
            unique_thicknesses = list(dict.fromkeys(trial_thicknesses))
            finite_trials = [
                (d, fit_at(d)) for d in unique_thicknesses if np.isfinite(fit_at(d).objective)
            ]
            if finite_trials:
                refined_thickness, fit = min(finite_trials, key=lambda item: (item[1].objective, item[0]))
            else:
                refined_thickness = float(self.grid[index])
                fit = fit_at(refined_thickness)
            attempts.append(
                _CandidateAttempt(
                    candidate_id=f"candidate_{ordinal:03d}",
                    coarse_index=int(index),
                    coarse_thickness_nm=float(self.grid[index]),
                    interval_lower_nm=float(lower),
                    interval_upper_nm=float(upper),
                    refined_thickness_nm=float(refined_thickness),
                    fit=fit,
                    outer_success=bool(optimized.success),
                )
            )

        edge_tolerance = max(1e-5, 0.01 * self.coarse_step_nm)

        def preliminary_success(attempt: _CandidateAttempt) -> bool:
            boundary = (
                attempt.refined_thickness_nm <= self.thickness_bounds[0] + edge_tolerance
                or attempt.refined_thickness_nm >= self.thickness_bounds[1] - edge_tolerance
            )
            return bool(
                attempt.outer_success
                and attempt.fit.converged
                and np.isfinite(attempt.fit.objective)
                and not boundary
                and not flat
            )

        representatives = deduplicate_refined_candidates(
            [attempt.refined_thickness_nm for attempt in attempts],
            [attempt.fit.objective for attempt in attempts],
            [preliminary_success(attempt) for attempt in attempts],
            [attempt.coarse_index for attempt in attempts],
            tolerance_nm=self.refined_candidate_deduplication_nm,
        )
        for position, representative in enumerate(representatives):
            attempts[position].representative_candidate_id = attempts[representative].candidate_id

        unique_positions = [position for position, representative in enumerate(representatives) if position == representative]
        for position, attempt in enumerate(attempts):
            boundary = (
                attempt.refined_thickness_nm <= self.thickness_bounds[0] + edge_tolerance
                or attempt.refined_thickness_nm >= self.thickness_bounds[1] - edge_tolerance
            )
            if not np.isfinite(attempt.fit.objective):
                attempt.status = "nonfinite_objective"
            elif not attempt.outer_success:
                attempt.status = "outer_nonconvergence"
            elif not attempt.fit.converged:
                attempt.status = "inner_nonconvergence"
            elif boundary:
                attempt.status = "thickness_boundary"
            elif flat:
                attempt.status = "flat_objective"
            else:
                attempt.status = "ok"
                attempt.eligible_for_selection = position == representatives[position]

        eligible = [position for position in unique_positions if attempts[position].eligible_for_selection]
        finite = [position for position in unique_positions if np.isfinite(attempts[position].fit.objective)]
        pool = eligible or finite
        if not pool:
            selected_position = unique_positions[0]
            outer_ok = False
        else:
            selected_position = min(
                pool,
                key=lambda position: (
                    attempts[position].fit.objective,
                    attempts[position].coarse_index,
                ),
            )
            outer_ok = attempts[selected_position].outer_success
        selected = attempts[selected_position]

        diagnostics = tuple(
            CandidateDiagnostic(
                candidate_id=attempt.candidate_id,
                coarse_index=attempt.coarse_index,
                coarse_thickness_nm=attempt.coarse_thickness_nm,
                interval_lower_nm=attempt.interval_lower_nm,
                interval_upper_nm=attempt.interval_upper_nm,
                refined_thickness_nm=attempt.refined_thickness_nm,
                objective=float(attempt.fit.objective),
                outer_success=attempt.outer_success,
                inner_converged=bool(attempt.fit.converged),
                status=attempt.status,
                deduplication_status=(
                    "representative" if attempt.candidate_id == attempt.representative_candidate_id else "deduplicated"
                ),
                selected_start=attempt.fit.selected_start,
                starts_attempted=int(attempt.fit.starts_attempted),
                converged_starts=int(attempt.fit.converged_starts),
                start_objectives=tuple(float(value) for value in attempt.fit.start_objectives),
                representative_candidate_id=attempt.representative_candidate_id,
                eligible_for_selection=attempt.eligible_for_selection,
            )
            for attempt in attempts
        )
        base = self._finalize(
            selected.refined_thickness_nm,
            selected.fit,
            scale,
            start,
            len(cache),
            len(unique_positions),
            outer_ok,
            flat,
        )
        if not eligible:
            base.converged = False
            base.status = selected.status
        return self._as_e3_result(base, diagnostics, candidate_indices, grid_objectives)
