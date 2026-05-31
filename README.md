# HelixAccelDrive
Benchmark-driven query planner for scRNA-seq pipeline


Vision: query planner for biology.
MVP: one scRNA-seq workflow, PBMC 68k, CPU + one GPU, validation gates, benchmark history, planner v0.
Not a benchmark notebook.
Not a full cloud optimizer.
First executable slice of the planner


Required execution:
- local CPU
- local RTX 4070 Ti

Architecture support:
- CPU
- T4
- A10G
- A100
- H100



Infrastructure assumptions for MVP

The first development and benchmark environment is a local workstation with NVIDIA RTX 4070 Ti 12 GB VRAM.

The local GPU is used to build and validate the first execution loop:
dataset profiling → Scanpy CPU baseline → RAPIDS GPU baseline → metrics collection → biological validation → planner v0 recommendation.

The MVP architecture must include a hardware profile registry for future cloud tiers: CPU, T4, A10G, A100, H100.

However, the first executable MVP is not required to benchmark all hardware tiers. Real execution is limited to local CPU + local RTX 4070 Ti unless cloud access is provided.

Cost model v1 is normalized compute-only estimate:

cost = runtime_seconds × hardware_price_per_hour / 3600

For local RTX 4070 Ti, price_per_hour is set to 0.00 for development runs. For future cloud estimates, initial reference prices can be stored for A10G/A100/H100 profiles and updated before real cloud benchmarking.

The first investor/demo run can be performed locally. If stronger hardware is needed, the same runner should be portable to cloud GPU instances.

