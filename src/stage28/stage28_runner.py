from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_ROOT = PACKAGE_ROOT / "authority_snapshot" / "02_estimator_implementation"
EXECUTION_ROOT = PACKAGE_ROOT / "authority_snapshot" / "05_execution_scripts"
sys.path.insert(0, str(IMPLEMENTATION_ROOT))
sys.path.insert(0, str(EXECUTION_ROOT))

from path_b_scientific_reconstruction_v2.src import run_reconstruction as rr  # noqa: E402
from e3_implementation_freeze.src.factory import make_repaired_estimator  # noqa: E402
from v3_unit_runner import e3_fields  # noqa: E402
from material_adapter import make_material_spectrum  # noqa: E402


E0 = "E0_constrained_L2_profile"
E1 = "E1_full_design_one_step_score"
E3 = "E3_constrained_multistart_robust_profile"
METHOD_ORDER = {E0: 0, E1: 1, E3: 2}
PROTOCOL_VERSION = "stage28-mm-v1.0"


def array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@lru_cache(maxsize=3)
def estimator_pair(material_id: str):
    spectrum = make_material_spectrum(material_id)
    base = rr.make_estimator(spectrum)
    repaired = make_repaired_estimator(rr.WAVELENGTH_NM, spectrum, rr.PROTOCOL)
    return base, repaired, spectrum


def method_row(
    registry: dict[str, Any],
    strategy: str,
    loss: str,
    scale_mode: str,
    result: Any,
    base: Any,
) -> dict[str, Any]:
    row = rr.result_row(
        suite="stage28_multimaterial_thickness_generalization",
        observation_id=str(registry["observation_id"]),
        scenario=str(registry["scenario"]),
        trial=int(registry["trial"]),
        model_tier=f"material_{registry['material_id']}_matched",
        strategy=strategy,
        loss=loss,
        baseline_order=2,
        result=result,
        estimator=base,
        true_thickness_nm=float(registry["true_thickness_nm"]),
        scale_mode=scale_mode,
        perturbation_factor="material_system",
        perturbation_level=float(registry["material_index"]),
    )
    row.update(
        {
            "stage28_protocol_version": PROTOCOL_VERSION,
            "observation_ordinal": int(registry["observation_ordinal"]),
            "condition_id": str(registry["condition_id"]),
            "material_id": str(registry["material_id"]),
            "film_material": str(registry["film_material"]),
            "substrate_material": str(registry["substrate_material"]),
            "material_source_id": str(registry["material_source_id"]),
            "method_order": METHOD_ORDER[strategy],
            "failure_code": "" if bool(row["converged"]) else str(row["status"]),
            "failure_stage": "none" if bool(row["converged"]) else (
                "inner" if str(row["status"]).startswith("inner") else "outer_or_candidate"
            ),
            "gain_bound_hit": bool(row["gain_lower_hit"] or row["gain_upper_hit"]),
            "fallback_used": bool(strategy == E3 and not row["converged"] and np.isfinite(row["objective"])),
        }
    )
    row.update(e3_fields(result))
    return row


def execute_observation(registry: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    material_id = str(registry["material_id"])
    base, repaired, spectrum = estimator_pair(material_id)
    true_thickness_nm = float(registry["true_thickness_nm"])
    clean = np.asarray(spectrum(true_thickness_nm), dtype=float)
    entropy = json.loads(str(registry["rng_entropy_json"]))
    expected_entropy = [
        int(registry["master_seed"]),
        int(registry["stage28_scenario_id"]),
        int(registry["trial"]),
    ]
    if entropy != expected_entropy:
        raise ValueError("Observation registry entropy mismatch")
    seed_sequence = np.random.SeedSequence(entropy)
    rng = np.random.default_rng(seed_sequence)
    if type(rng.bit_generator).__name__ != "PCG64":
        raise RuntimeError("Frozen bit generator must be PCG64")
    observation = rr.generate_observation(
        clean,
        rr.WAVELENGTH_NM,
        rr.SCENARIOS[str(registry["scenario"])],
        rng,
    )
    observed = np.asarray(observation.values, dtype=float)
    if observed.shape != rr.WAVELENGTH_NM.shape or not np.isfinite(observed).all():
        raise ValueError("Invalid reconstructed observation")

    order = 2
    scale_e0, l2_e0 = base.estimate_adaptive_scale(observed, order, floor=float(rr.PROTOCOL["robust_scale"]["floor"]))
    scale_e1, _ = base.estimate_adaptive_scale(observed, order, floor=float(rr.PROTOCOL["robust_scale"]["floor"]))
    e1_result = base.fit_one_step_score(observed, "Tukey", order, scale_e1)
    scale_e3, _ = base.estimate_adaptive_scale(observed, order, floor=float(rr.PROTOCOL["robust_scale"]["floor"]))
    e3_result = repaired.fit_robust_profile(observed, "Tukey", order, scale_e3)
    scales_equal = bool(scale_e0 == scale_e1 == scale_e3)
    if not scales_equal:
        raise RuntimeError("Deterministically recomputed adaptive scales differ")

    result_rows = [
        method_row(registry, E0, "L2", "not_applicable", l2_e0, base),
        method_row(registry, E1, "Tukey", "adaptive_mad", e1_result, base),
        method_row(registry, E3, "Tukey", "adaptive_mad", e3_result, base),
    ]
    observation_audit = {
        "stage28_protocol_version": PROTOCOL_VERSION,
        "observation_ordinal": int(registry["observation_ordinal"]),
        "observation_id": str(registry["observation_id"]),
        "condition_id": str(registry["condition_id"]),
        "material_id": material_id,
        "true_thickness_nm": true_thickness_nm,
        "scenario": str(registry["scenario"]),
        "trial": int(registry["trial"]),
        "rng_entropy_json": json.dumps(entropy, separators=(",", ":")),
        "bit_generator": type(rng.bit_generator).__name__,
        "clean_spectrum_sha256": array_sha256(clean, "<f8"),
        "observed_spectrum_sha256": array_sha256(observed, "<f8"),
        "gaussian_noise_sha256": array_sha256(observation.gaussian_noise, "<f8"),
        "baseline_sha256": array_sha256(observation.baseline, "<f8"),
        "outlier_indices_sha256": array_sha256(observation.outlier_indices, "<i8"),
        "outlier_values_sha256": array_sha256(observation.outlier_values, "<f8"),
        "outlier_indices_json": json.dumps([int(value) for value in observation.outlier_indices]),
        "outlier_count": int(len(observation.outlier_indices)),
        "realized_gain": float(observation.gain),
        "realized_offset": float(observation.offset),
        "gaussian_rms": float(np.sqrt(np.mean(np.square(observation.gaussian_noise)))),
        "baseline_max_abs": float(np.max(np.abs(observation.baseline))),
        "observed_min": float(np.min(observed)),
        "observed_max": float(np.max(observed)),
        "adaptive_scale_e0": float(scale_e0),
        "adaptive_scale_e1": float(scale_e1),
        "adaptive_scale_e3": float(scale_e3),
        "adaptive_scales_exactly_equal": scales_equal,
    }
    return observation_audit, result_rows


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_progress(
    output_root: Path,
    *,
    status: str,
    completed: int,
    total: int,
    started: float,
    active_futures: int,
    completed_shards: int,
    total_shards: int,
) -> None:
    elapsed = max(perf_counter() - started, 0.0)
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    remaining = max(total - completed, 0)
    eta_seconds = remaining / rate if rate > 0.0 else None
    payload = {
        "schema": "stage28-persistent-progress/1.0",
        "status": status,
        "completed_observations": int(completed),
        "total_observations": int(total),
        "fraction_complete": float(completed / total if total else 0.0),
        "completed_shards": int(completed_shards),
        "total_shards": int(total_shards),
        "active_futures": int(active_futures),
        "elapsed_seconds": float(elapsed),
        "observations_per_second": float(rate),
        "eta_seconds": None if eta_seconds is None else float(eta_seconds),
    }
    atomic_json(payload, output_root / "STAGE28_PROGRESS.json")


def finalize(output_root: Path, expected_observations: int) -> dict[str, Any]:
    shard_root = output_root / "shards"
    result_paths = sorted(shard_root.glob("results_*.parquet"))
    observation_paths = sorted(shard_root.glob("observations_*.parquet"))
    results = pd.concat([pd.read_parquet(path) for path in result_paths], ignore_index=True)
    observations = pd.concat([pd.read_parquet(path) for path in observation_paths], ignore_index=True)
    results = results.sort_values(["observation_ordinal", "method_order"], kind="stable").reset_index(drop=True)
    observations = observations.sort_values("observation_ordinal", kind="stable").reset_index(drop=True)
    if len(observations) != expected_observations or observations["observation_id"].nunique() != expected_observations:
        raise RuntimeError("Observation finalization cardinality failure")
    if len(results) != expected_observations * 3:
        raise RuntimeError("Result finalization cardinality failure")
    if not (results.groupby("observation_id")["strategy"].nunique() == 3).all():
        raise RuntimeError("Incomplete method pairing")
    atomic_parquet(results, output_root / "MASTER_STAGE28_RESULTS.parquet")
    atomic_csv(results, output_root / "MASTER_STAGE28_RESULTS.csv")
    atomic_parquet(observations, output_root / "MASTER_STAGE28_OBSERVATION_AUDIT.parquet")
    atomic_csv(observations, output_root / "MASTER_STAGE28_OBSERVATION_AUDIT.csv")
    report = {
        "schema": "stage28-execution-report/1.0",
        "status": "PASS",
        "observations": len(observations),
        "result_rows": len(results),
        "strategy_counts": results["strategy"].value_counts().sort_index().to_dict(),
        "material_counts": observations["material_id"].value_counts().sort_index().to_dict(),
        "scenario_counts": observations["scenario"].value_counts().sort_index().to_dict(),
        "all_adaptive_scales_equal": bool(observations["adaptive_scales_exactly_equal"].all()),
    }
    path = output_root / "STAGE28_EXECUTION_REPORT.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen Stage 28 observations with authoritative E0/E1/E3.")
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=7200)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--in-flight-multiplier", type=int, default=4)
    args = parser.parse_args()
    if args.package_root.resolve() != PACKAGE_ROOT.resolve():
        raise ValueError("This frozen runner must execute from its packaged root")
    if (
        args.workers < 1
        or args.start < 0
        or args.stop > 7200
        or args.start >= args.stop
        or args.chunk_size < 1
        or args.in_flight_multiplier < 1
    ):
        raise ValueError("Invalid execution range or worker configuration")
    registry_path = PACKAGE_ROOT / "stage28" / "registry" / "OBSERVATION_REGISTRY_7200.csv"
    registry = pd.read_csv(registry_path)
    selected = registry[(registry["observation_ordinal"] >= args.start) & (registry["observation_ordinal"] < args.stop)]
    selected = selected.sort_values("observation_ordinal", kind="stable")
    if len(selected) != args.stop - args.start:
        raise RuntimeError("Selected registry interval is incomplete")
    output_root = args.output_root.resolve()
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    shard_specs: list[tuple[int, int, Path, Path]] = []
    ordinal_to_shard: dict[int, tuple[int, int]] = {}
    completed = 0
    completed_shards = 0
    pending_payloads: list[dict[str, Any]] = []
    for chunk_start in range(args.start, args.stop, args.chunk_size):
        chunk_stop = min(chunk_start + args.chunk_size, args.stop)
        result_path = shard_root / f"results_{chunk_start:06d}_{chunk_stop:06d}.parquet"
        observation_path = shard_root / f"observations_{chunk_start:06d}_{chunk_stop:06d}.parquet"
        shard_specs.append((chunk_start, chunk_stop, result_path, observation_path))
        if result_path.is_file() and observation_path.is_file():
            completed += chunk_stop - chunk_start
            completed_shards += 1
            print(f"SKIP {chunk_start}:{chunk_stop} complete shard", flush=True)
            continue
        chunk = selected[
            (selected["observation_ordinal"] >= chunk_start)
            & (selected["observation_ordinal"] < chunk_stop)
        ]
        for payload in chunk.to_dict(orient="records"):
            ordinal = int(payload["observation_ordinal"])
            ordinal_to_shard[ordinal] = (chunk_start, chunk_stop)
            pending_payloads.append(payload)
    total = args.stop - args.start
    total_shards = len(shard_specs)
    buffers: dict[tuple[int, int], list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {}
    spec_by_key = {(start, stop): (result_path, observation_path) for start, stop, result_path, observation_path in shard_specs}
    write_progress(
        output_root,
        status="RUNNING",
        completed=completed,
        total=total,
        started=started,
        active_futures=0,
        completed_shards=completed_shards,
        total_shards=total_shards,
    )
    max_in_flight = args.workers * args.in_flight_multiplier
    payload_iterator = iter(pending_payloads)
    last_progress_write = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures: dict[Any, dict[str, Any]] = {}

        def fill_queue() -> None:
            while len(futures) < max_in_flight:
                try:
                    payload = next(payload_iterator)
                except StopIteration:
                    break
                futures[pool.submit(execute_observation, payload)] = payload

        fill_queue()
        while futures:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            shard_written = False
            for future in done:
                payload = futures.pop(future)
                item = future.result()
                ordinal = int(payload["observation_ordinal"])
                key = ordinal_to_shard[ordinal]
                buffer = buffers.setdefault(key, [])
                buffer.append(item)
                completed += 1
                expected = key[1] - key[0]
                if len(buffer) == expected:
                    buffer.sort(key=lambda value: int(value[0]["observation_ordinal"]))
                    observation_rows = [value[0] for value in buffer]
                    result_rows = [row for value in buffer for row in value[1]]
                    result_path, observation_path = spec_by_key[key]
                    atomic_parquet(pd.DataFrame(result_rows), result_path)
                    atomic_parquet(pd.DataFrame(observation_rows), observation_path)
                    del buffers[key]
                    completed_shards += 1
                    shard_written = True
                    elapsed = perf_counter() - started
                    print(
                        f"DONE {key[0]}:{key[1]}; observations={completed}/{total}; "
                        f"shards={completed_shards}/{total_shards}; elapsed={elapsed:.1f}s",
                        flush=True,
                    )
            fill_queue()
            now = perf_counter()
            if shard_written or now - last_progress_write >= 5.0:
                write_progress(
                    output_root,
                    status="RUNNING",
                    completed=completed,
                    total=total,
                    started=started,
                    active_futures=len(futures),
                    completed_shards=completed_shards,
                    total_shards=total_shards,
                )
                last_progress_write = now
    if buffers:
        raise RuntimeError("Incomplete in-memory shard buffers after execution")
    if args.start == 0 and args.stop == 7200:
        write_progress(
            output_root,
            status="FINALIZING",
            completed=completed,
            total=total,
            started=started,
            active_futures=0,
            completed_shards=completed_shards,
            total_shards=total_shards,
        )
        report = finalize(output_root, 7200)
        write_progress(
            output_root,
            status="PASS",
            completed=completed,
            total=total,
            started=started,
            active_futures=0,
            completed_shards=completed_shards,
            total_shards=total_shards,
        )
        print(json.dumps(report, indent=2), flush=True)
    else:
        print("Partial interval complete; final consolidation not performed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
