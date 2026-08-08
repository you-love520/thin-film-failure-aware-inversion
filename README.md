# Failure-Mechanism-Aware Thin-Film Thickness Inversion

Reproducibility release candidate for the manuscript **“Failure-Mechanism-Aware Robust Profiling with Selective Structured-Residual Refinement for Thin-Film Thickness Inversion.”**

## Scope

The repository contains the estimator source code, fixed configurations, stored result tables, read-only statistics, and figure/table generation material supporting the manuscript. The scientific results are frozen: this release package is intended to reproduce reported summaries and figures, not to introduce a new analysis branch.

## Estimator hierarchy

- **E0**: constrained L2 nuisance profile.
- **E1**: one-step Tukey score after L2 nuisance fitting.
- **E3**: Tukey robust nuisance profiling with deterministic multibasin search.
- **E4-HORP**: E3-anchored local structured-residual refinement with physical sensitivity protection and selective V2/Fine-q75 acceptance.

## Repository layout

```text
src/stage28/        E0/E1/E3 forward-model, simulation, and estimator implementation
src/e4/             E4-HORP implementation
configs/            Fixed protocol and E4 parameter files
data/processed/     Stored Stage28/development/independent-validation result tables
data/external/      Optical-constant tables and external-data provenance notes
analysis/           Read-only statistics and figure generation scripts
figures/            Vector manuscript/supplement figure masters
tables/             Main and supplementary table source CSVs
docs/               Frozen scientific manuscript snapshot plus current formatting-source/supplement snapshots
environment/        Locked Python dependencies
provenance/         SHA-256 manifest
```

## Environment

The frozen Python dependencies are listed in `environment/requirements-lock.txt` (NumPy 2.5.1, SciPy 1.18.0, pandas 2.2.2, pyarrow 17.0.0, matplotlib 3.9.1, psutil 7.2.2, polars 1.42.1).

## Verifying the frozen release

A portable verification script is provided:

```bash
python analysis/verify_release.py
sha256sum -c provenance/SHA256SUMS.txt
```

The verification uses released result/table artifacts and does **not** rerun E0/E1/E3/E4. It checks the 21,600 selected-material estimator records represented by Table S5, the retained boundary event, development-gate counts, independent-validation counts, and principal Stage28 contrasts. The other analysis scripts are archived provenance copies; see `ANALYSIS_SCRIPT_NOTE.md` for path-portability details.

## External data

The Sheffield measurements are not redistributed here. See `data/external/SHEFFIELD_SOURCE.md` for the primary article and dataset DOI. Material optical-property provenance is recorded in the manuscript, Supplementary Information, and `configs/MATERIAL_SOURCES.json`.

## Important interpretation boundaries

- No global-optimality certificate is claimed.
- E3 is not a universal winner; smooth baseline drift is a documented failure regime.
- E4 is a selective post-anchor refinement, not a universal replacement for E3.
- Independent-seed validation is not external experimental validation.
- Numerical completion does not imply correct physical-basin recovery.
- The reference implementation is not presented as a real-time system.

## Public release

This folder is a release candidate. Before public deposition, choose an explicit software/data license, create the archival GitHub/Zenodo record, and replace repository/DOI placeholders in the manuscript. See `PUBLIC_RELEASE_STEPS.md`.
