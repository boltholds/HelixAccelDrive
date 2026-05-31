from __future__ import annotations

from anndata import AnnData

from helixaccel.backends.base import PipelineBackend
from helixaccel.config.settings import PipelineConfig
from helixaccel.metrics.runtime import StepMeasurement
from helixaccel.pipeline.scanpy_pipeline import run_scanpy_pipeline


class ScanpyCPUBackend(PipelineBackend):
    name = "scanpy-cpu"

    def run(self, adata: AnnData, config: PipelineConfig) -> tuple[AnnData, list[StepMeasurement]]:
        return run_scanpy_pipeline(adata, config)
