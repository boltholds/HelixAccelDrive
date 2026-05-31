# Phase 1 — Reproducible Benchmark Skeleton

Phase 1 builds the measurement foundation for HelixAccel.

The goal is not optimization yet. The goal is to create a reliable benchmark skeleton:

```text
dataset loader → dataset profiler → Scanpy CPU pipeline → step-level metrics → JSON run record → Markdown report
```

This phase creates the basis for later RAPIDS comparison, biological validation, cost model, and planner v0.

## Required output

A CLI command:

```bash
helix run --dataset pbmc3k --backend scanpy-cpu --pipeline canonical-scrna-v0
```

It should produce:

```text
runs/<run_id>.json
reports/<run_id>.md
```

## Measured metrics

For each pipeline step:

- runtime
- RSS before / after
- sampled peak RAM
- matrix shape before / after
- matrix format before / after
- nnz before / after
- success/failure
- error message if failed

## Not in Phase 1

- RAPIDS GPU execution
- planner v0
- cost model
- biological validation gates
- PBMC 68k as required benchmark
- KNN/PCA optimization experiments


## Fix pass before Phase 2

Before moving to PBMC68k CPU baseline, the Phase 1 skeleton was updated to make run records usable as benchmark history:

1. `rss_after_mb` is now correctly captured after the memory sampler exits.
2. Every step includes `rss_delta_mb`.
3. Run records include exact `pipeline_params`.
4. Successful runs can save baseline artifacts for later validation: `.h5ad`, Leiden clusters, and marker tables.
5. Reports include artifact paths, so Phase 4 can reuse CPU reference outputs for ARI/NMI and marker overlap.

