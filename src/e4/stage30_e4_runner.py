from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
STAGE28_ROOT = PACKAGE_ROOT / "stage28_package"
STAGE28_SRC = STAGE28_ROOT / "stage28" / "src"
IMPLEMENTATION_ROOT = STAGE28_ROOT / "authority_snapshot" / "02_estimator_implementation"
sys.path.insert(0, str(STAGE28_SRC))
sys.path.insert(0, str(IMPLEMENTATION_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from material_adapter import make_material_spectrum  # noqa: E402
from path_b_scientific_reconstruction_v2.src import run_reconstruction as rr  # noqa: E402
from e4_horp import refine_e4_horp  # noqa: E402


E3 = "E3_constrained_multistart_robust_profile"


def array_sha256(values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def try_atomic_parquet(frame: pd.DataFrame, path: Path) -> bool:
    try:
        atomic_parquet(frame, path)
        return True
    except ImportError:
        return False


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def read_stage28_zip(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("MASTER_STAGE28_RESULTS.csv") as handle:
            results = pd.read_csv(handle)
        with zf.open("MASTER_STAGE28_OBSERVATION_AUDIT.csv") as handle:
            audit = pd.read_csv(handle)
    return results, audit


def load_inputs(stage28_zip: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    registry = pd.read_csv(STAGE28_ROOT / "stage28" / "registry" / "OBSERVATION_REGISTRY_7200.csv")
    results, audit = read_stage28_zip(stage28_zip)
    config = json.loads((PACKAGE_ROOT / "stage30" / "config" / "E4_PARAMETERS.json").read_text(encoding="utf-8"))
    e3_rows = results[results["strategy"] == E3].copy()
    required = {
        "observation_id",
        "estimate_nm",
        "gain",
        "linear_coefficients_json",
        "residual_scale",
        "objective",
        "converged",
        "status",
    }
    missing = sorted(required - set(e3_rows.columns))
    if missing:
        raise RuntimeError(f"Stage 28 E3 rows missing required fields: {missing}")
    if len(registry) != 7200 or len(audit) != 7200 or len(e3_rows) != 7200:
        raise RuntimeError("Stage 30 requires 7200 registry, audit and E3 rows")
    return registry, audit, e3_rows, config


def reconstruct_observation(registry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Any]:
    spectrum = make_material_spectrum(str(registry["material_id"]))
    clean = np.asarray(spectrum(float(registry["true_thickness_nm"])), dtype=float)
    entropy = json.loads(str(registry["rng_entropy_json"]))
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    observation = rr.generate_observation(clean, rr.WAVELENGTH_NM, rr.SCENARIOS[str(registry["scenario"])], rng)
    return clean, np.asarray(observation.values, dtype=float), observation


def execute_one(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = payload["registry"]
    audit = payload["audit"]
    e3 = payload["e3"]
    config = payload["config"]
    clean, observed, observation = reconstruct_observation(registry)
    checks = {
        "clean_spectrum_sha256_match": array_sha256(clean, "<f8") == str(audit["clean_spectrum_sha256"]),
        "observed_spectrum_sha256_match": array_sha256(observed, "<f8") == str(audit["observed_spectrum_sha256"]),
        "gaussian_noise_sha256_match": array_sha256(observation.gaussian_noise, "<f8") == str(audit["gaussian_noise_sha256"]),
        "baseline_sha256_match": array_sha256(observation.baseline, "<f8") == str(audit["baseline_sha256"]),
        "outlier_indices_sha256_match": array_sha256(observation.outlier_indices, "<i8") == str(audit["outlier_indices_sha256"]),
        "outlier_values_sha256_match": array_sha256(observation.outlier_values, "<f8") == str(audit["outlier_values_sha256"]),
    }
    if not all(checks.values()):
        result = {
            "observation_id": str(registry["observation_id"]),
            "material_id": str(registry["material_id"]),
            "scenario": str(registry["scenario"]),
            "true_thickness_nm": float(registry["true_thickness_nm"]),
            "e3_thickness_nm": float(e3["estimate_nm"]),
            "e4_candidate_thickness_nm": float(e3["estimate_nm"]),
            "e4_final_thickness_nm": float(e3["estimate_nm"]),
            "triggered": False,
            "accepted": False,
            "fallback_reason": "stage28_observation_reconstruction_hash_mismatch",
            "absolute_error_e3_nm": abs(float(e3["estimate_nm"]) - float(registry["true_thickness_nm"])),
            "absolute_error_e4_nm": abs(float(e3["estimate_nm"]) - float(registry["true_thickness_nm"])),
        }
        return result, {"observation_id": str(registry["observation_id"]), **checks}
    coefficients = np.asarray(json.loads(str(e3["linear_coefficients_json"])), dtype=float)
    spectrum = make_material_spectrum(str(registry["material_id"]))
    e4 = refine_e4_horp(
        observation_id=str(registry["observation_id"]),
        observed=observed,
        wavelength_nm=rr.WAVELENGTH_NM,
        spectrum=spectrum,
        e3_thickness_nm=float(e3["estimate_nm"]),
        e3_coefficients=coefficients,
        e3_scale=float(e3["residual_scale"]),
        e3_objective=float(e3["objective"]),
        thickness_bounds=tuple(rr.PROTOCOL["thickness_bounds_nm"]),
        config=config,
    )
    row = asdict(e4)
    truth = float(registry["true_thickness_nm"])
    row.update(
        {
            "material_id": str(registry["material_id"]),
            "film_material": str(registry["film_material"]),
            "substrate_material": str(registry["substrate_material"]),
            "condition_id": str(registry["condition_id"]),
            "scenario": str(registry["scenario"]),
            "trial": int(registry["trial"]),
            "true_thickness_nm": truth,
            "stage28_e3_status": str(e3["status"]),
            "stage28_e3_converged": bool(e3["converged"]),
            "absolute_error_e3_nm": abs(float(e3["estimate_nm"]) - truth),
            "absolute_error_e4_nm": abs(float(e4.e4_final_thickness_nm) - truth),
            "signed_error_e4_nm": float(e4.e4_final_thickness_nm) - truth,
        }
    )
    return row, {"observation_id": str(registry["observation_id"]), **checks}


def write_progress(output_root: Path, status: str, done: int, total: int, started: float, active: int) -> None:
    elapsed = max(perf_counter() - started, 0.0)
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else None
    atomic_json(
        {
            "schema": "stage30-e4-progress/1.0",
            "status": status,
            "completed_observations": int(done),
            "total_observations": int(total),
            "fraction_complete": float(done / total if total else 0.0),
            "active_futures": int(active),
            "elapsed_seconds": float(elapsed),
            "observations_per_second": float(rate),
            "eta_seconds": None if eta is None else float(eta),
        },
        output_root / "STAGE30_E4_PROGRESS.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 30 E4-HORP on Stage 28 observations.")
    parser.add_argument("--stage28-results-zip", type=Path, default=PACKAGE_ROOT / "inputs" / "STAGE28_RESULTS_HANDOFF.zip")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=7200)
    parser.add_argument("--in-flight-multiplier", type=int, default=4)
    args = parser.parse_args()
    registry, audit, e3_rows, config = load_inputs(args.stage28_results_zip)
    registry = registry.sort_values("observation_ordinal", kind="stable").reset_index(drop=True)
    audit_map = audit.set_index("observation_id").to_dict(orient="index")
    e3_map = e3_rows.set_index("observation_id").to_dict(orient="index")
    selected = registry[(registry["observation_ordinal"] >= args.start) & (registry["observation_ordinal"] < args.stop)]
    output_root = args.output_root.resolve()
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    total = len(selected)
    started = perf_counter()
    payloads = []
    for row in selected.to_dict(orient="records"):
        obs_id = str(row["observation_id"])
        payloads.append({"registry": row, "audit": audit_map[obs_id], "e3": e3_map[obs_id], "config": config})
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    done_count = 0
    write_progress(output_root, "RUNNING", done_count, total, started, 0)
    iterator = iter(payloads)
    max_in_flight = max(1, int(args.workers) * int(args.in_flight_multiplier))
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures: dict[Any, int] = {}

        def fill() -> None:
            while len(futures) < max_in_flight:
                try:
                    payload = next(iterator)
                except StopIteration:
                    break
                futures[pool.submit(execute_one, payload)] = int(payload["registry"]["observation_ordinal"])

        fill()
        last_write = perf_counter()
        while futures:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                ordinal = futures.pop(future)
                row, audit_row = future.result()
                row["observation_ordinal"] = ordinal
                audit_row["observation_ordinal"] = ordinal
                rows.append(row)
                audits.append(audit_row)
                done_count += 1
            fill()
            if perf_counter() - last_write >= 5.0:
                write_progress(output_root, "RUNNING", done_count, total, started, len(futures))
                last_write = perf_counter()
                print(f"Stage30 E4 progress {done_count}/{total}", flush=True)
    result = pd.DataFrame(rows).sort_values("observation_ordinal", kind="stable")
    audit_frame = pd.DataFrame(audits).sort_values("observation_ordinal", kind="stable")
    parquet_results_written = try_atomic_parquet(result, output_root / "MASTER_STAGE30_E4_RESULTS.parquet")
    atomic_csv(result, output_root / "MASTER_STAGE30_E4_RESULTS.csv")
    parquet_audit_written = try_atomic_parquet(audit_frame, output_root / "MASTER_STAGE30_E4_AUDIT.parquet")
    atomic_csv(audit_frame, output_root / "MASTER_STAGE30_E4_AUDIT.csv")
    report = {
        "schema": "stage30-e4-execution-report/1.0",
        "status": "PASS" if len(result) == total and bool(audit_frame.drop(columns=["observation_id", "observation_ordinal"]).all().all()) else "FAIL",
        "observations": int(len(result)),
        "triggered": int(result["triggered"].sum()) if "triggered" in result else 0,
        "accepted": int(result["accepted"].sum()) if "accepted" in result else 0,
        "fallback_counts": result["fallback_reason"].value_counts().sort_index().to_dict() if "fallback_reason" in result else {},
        "mean_absolute_error_e3_nm": float(result["absolute_error_e3_nm"].mean()),
        "mean_absolute_error_e4_nm": float(result["absolute_error_e4_nm"].mean()),
        "parquet_results_written": bool(parquet_results_written),
        "parquet_audit_written": bool(parquet_audit_written),
    }
    atomic_json(report, output_root / "STAGE30_E4_EXECUTION_REPORT.json")
    write_progress(output_root, report["status"], done_count, total, started, 0)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
