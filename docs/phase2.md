# Phase 2 — PBMC68k Scanpy CPU Baseline

Phase 2 turns the Phase 1 benchmark skeleton into the first meaningful CPU reference run.

## Goal

Run the canonical Scanpy CPU pipeline on PBMC68k and produce a structured baseline:

- dataset profile;
- pipeline parameters;
- runtime per step;
- RSS before / after / delta / peak per step;
- sparse/dense and dtype transitions;
- `result.h5ad`, `clusters.csv`, `markers.csv` artifacts;
- JSON benchmark history;
- Markdown report with bottlenecks and memory observations.

This run becomes the reference for later RAPIDS/GPU comparison, biological validation gates, cost model, and planner v0.

## Dataset preparation

Put PBMC68k data in one of the supported formats:

```text
data/raw/pbmc68k/                    # 10x mtx directory
# or
data/raw/pbmc68k_filtered_gene_bc_matrices.h5
# or
data/raw/pbmc68k.h5ad
```

Then set the path in `configs/pbmc68k_cpu.yaml`, or pass it explicitly with `--path`.

## Profile dataset

```bash
helix profile --dataset pbmc68k --config configs/pbmc68k_cpu.yaml
```

or:

```bash
helix profile --dataset pbmc68k --path data/raw/pbmc68k --cache-path data/cache/pbmc68k_raw.h5ad
```

## Run CPU baseline

```bash
helix run --dataset pbmc68k --backend scanpy-cpu --config configs/pbmc68k_cpu.yaml
```

or:

```bash
helix run \
  --dataset pbmc68k \
  --backend scanpy-cpu \
  --path data/raw/pbmc68k \
  --cache-path data/cache/pbmc68k_raw.h5ad
```

## Expected outputs

```text
runs/<run_id>.json
reports/<run_id>.md
artifacts/<run_id>/result.h5ad
artifacts/<run_id>/clusters.csv
artifacts/<run_id>/markers.csv
```

## Definition of Done

Phase 2 is done when:

- PBMC68k loads successfully;
- Scanpy CPU pipeline completes end-to-end;
- per-step runtime and memory metrics are recorded;
- matrix transitions are visible in the report;
- CPU reference artifacts are saved;
- bottleneck summary is generated.

## Out of scope

- RAPIDS/GPU backend;
- VRAM profiling;
- ARI/NMI and marker overlap comparison;
- cost model;
- planner v0;
- FAISS / approximate KNN experiments.

## PBMC68k QC note

Some public PBMC68k `.h5ad` files have a low average number of non-zero genes per cell. For those files, `min_genes: 200` can remove every cell and produce an empty `(0, 0)` AnnData object. The Phase 2 config therefore sets `min_genes: 0` for the first CPU baseline. This keeps the baseline focused on performance measurement. QC thresholds can be tightened later once the science lead confirms the exact canonical preprocessing policy.
