from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str | None = None
    raw_path: Path | None = None
    cache_path: Path | None = None
    force_reload: bool = False


class PipelineConfig(BaseModel):
    name: str = "canonical_scrna_v0"
    min_genes: int = 200
    min_cells: int = 3
    target_sum: float = 10_000
    n_top_genes: int = 2_000
    n_pcs: int = 50
    n_neighbors: int = 15
    leiden_resolution: float = 1.0
    marker_method: str = "t-test"
    random_state: int = 0


class StorageConfig(BaseModel):
    runs_dir: Path = Path("runs")
    reports_dir: Path = Path("reports")
    artifacts_dir: Path = Path("artifacts")
    save_h5ad: bool = True
    save_clusters: bool = True
    save_markers: bool = True


class HardwareConfig(BaseModel):
    profile: str = "local_cpu_with_rtx4070ti"
    local_gpu_name: str | None = "RTX 4070 Ti"
    local_gpu_vram_gb: int | None = 12


class SafetyConfig(BaseModel):
    warn_dense_gb: float = 4.0
    fail_dense_gb: float | None = 20.0
    max_runtime_minutes: int | None = 120


class AppConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


def load_config(path: str | Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)
