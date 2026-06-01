# Phase 3 — RAPIDS GPU Baseline

Goal: add a first GPU baseline for PBMC68k on local RTX 4070 Ti and capture runtime, RAM, VRAM, CPU↔GPU transfer steps, and artifacts for later biological validation.

This phase is not the final planner and not a full cloud hardware comparison. It is a fair first comparison against the Phase 2 Scanpy CPU reference.

## Important environment note

RAPIDS is usually not supported well in native Windows Python. Use WSL2/Linux or a Linux GPU machine with a RAPIDS-compatible conda/mamba environment.

Recommended separate environment:

```bash
conda create -n helixaccel-gpu python=3.11 -y
conda activate helixaccel-gpu
# Install RAPIDS according to the official selector for your CUDA/driver version.
# Then install this project in editable mode:
pip install -e ".[gpu]"
```

Keep the existing CPU environment intact. Do not break the Phase 2 Scanpy CPU baseline environment.

## Execution strategy

The Phase 3 backend is intentionally hybrid:

CPU:
- qc_filter
- normalize
- log1p
- hvg

GPU / RAPIDS:
- transfer_to_gpu
- rapids_scale
- rapids_pca
- rapids_neighbors
- rapids_leiden
- rapids_umap

CPU:
- transfer_to_cpu
- marker_detection

This matches the MVP assumption that the main GPU candidate is the core compute group: PCA + KNN/neighbors + Leiden + UMAP.

## Run

```bash
helix run --dataset pbmc68k --backend rapids-gpu --config configs/pbmc68k_gpu.yaml
```

If your raw file is a specific `.h5ad`:

```bash
helix run \
  --dataset pbmc68k \
  --backend rapids-gpu \
  --path data/raw/pbmc68k/your_file.h5ad \
  --cache-path data/cache/pbmc68k_raw.h5ad \
  --config configs/pbmc68k_gpu.yaml
```

## Compare against Phase 2 CPU baseline

```bash
helix compare \
  --baseline runs/20260531T220546Z_pbmc68k_scanpy-cpu_5871998a.json \
  --candidate runs/<rapids_run_id>.json
```

The comparison is runtime/VRAM only. Biological validation is Phase 4.

## Expected outputs

- `runs/<run_id>_pbmc68k_rapids-gpu.json`
- `reports/<run_id>_pbmc68k_rapids-gpu.md`
- `artifacts/<run_id>/result.h5ad`
- `artifacts/<run_id>/clusters.csv`
- `artifacts/<run_id>/markers.csv`
- optional comparison markdown report

## Definition of Done

Phase 3 is done if:

- RAPIDS backend is registered as `rapids-gpu`.
- PBMC68k GPU run succeeds or fails with a clear environment/OOM error.
- VRAM fields are present in the JSON record when pynvml is available.
- Transfer steps are measured.
- Markdown report includes GPU/transfer observations.
- Artifacts are saved for successful runs.
- A preliminary CPU vs GPU comparison can be generated.

A VRAM/OOM failure on RTX 4070 Ti is still useful planner signal: this local GPU tier may be memory-risky for the workload and A10G/A100 should be evaluated next.
