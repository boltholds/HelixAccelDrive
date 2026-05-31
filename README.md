# HelixAccel Phase 1 — Reproducible Benchmark Skeleton

This repository is the Phase 1 implementation slice for HelixAccel.

Goal: build a reproducible measurement loop for a standard scRNA-seq Scanpy CPU pipeline before GPU optimization, planner logic, and biological validation gates are added.

Phase 1 pipeline:

```text
load → QC/filtering → normalize → log1p → HVG → scale → PCA → KNN/neighbors → Leiden → UMAP → marker detection
```

The first working target is PBMC 3k as a sanity/correctness benchmark. PBMC 68k is the next benchmark once this runner is stable.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

GPU telemetry is optional:

```bash
pip install -e ".[gpu,dev]"
```

## Commands

Profile dataset:

```bash
helix profile --dataset pbmc3k
```

Run Scanpy CPU baseline:

```bash
helix run --dataset pbmc3k --backend scanpy-cpu --config configs/default.yaml
```

Artifacts:

```text
runs/*.json       structured benchmark history
reports/*.md      human-readable benchmark report
```
