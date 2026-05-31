from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import scanpy as sc
from pydantic import BaseModel, Field

from helixaccel.config.settings import PipelineConfig
from helixaccel.datasets.profiler import DatasetProfile
from helixaccel.metrics.hardware import collect_hardware_info
from helixaccel.metrics.runtime import StepMeasurement


class RunRecord(BaseModel):
    run_id: str
    created_at: str
    dataset: str
    backend: str
    pipeline: str
    pipeline_params: dict[str, Any]
    dataset_profile: dict[str, Any]
    hardware: dict[str, Any]
    software: dict[str, Any]
    artifact_paths: dict[str, Any] = Field(default_factory=dict)
    steps: list[dict[str, Any]]
    total_runtime_sec: float
    success: bool
    error: str | None = None
    notes: list[str] = Field(default_factory=list)


def build_run_record(
    dataset: str,
    backend: str,
    pipeline: str,
    pipeline_params: PipelineConfig,
    dataset_profile: DatasetProfile,
    steps: list[StepMeasurement],
    success: bool,
    error: str | None = None,
    artifact_paths: dict[str, Any] | None = None,
) -> RunRecord:
    now = datetime.now(UTC)
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{dataset}_{backend}_{uuid4().hex[:8]}"
    return RunRecord(
        run_id=run_id,
        created_at=now.isoformat(),
        dataset=dataset,
        backend=backend,
        pipeline=pipeline,
        pipeline_params=pipeline_params.model_dump(mode="json"),
        dataset_profile=dataset_profile.to_dict(),
        hardware=collect_hardware_info(),
        software={"scanpy": sc.__version__},
        artifact_paths=artifact_paths or {},
        steps=[step.to_dict() for step in steps],
        total_runtime_sec=round(sum(step.runtime_sec for step in steps), 6),
        success=success,
        error=error,
    )
