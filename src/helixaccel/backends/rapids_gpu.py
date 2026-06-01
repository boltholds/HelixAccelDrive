from __future__ import annotations

from anndata import AnnData

from helixaccel.backends.base import PipelineBackend
from helixaccel.config.settings import PipelineConfig
from helixaccel.metrics.runtime import StepMeasurement
from helixaccel.pipeline.rapids_pipeline import run_rapids_pipeline


class RapidsGPUBackend(PipelineBackend):
    name = "rapids-gpu"

    def run(self, adata: AnnData, config: PipelineConfig) -> tuple[AnnData, list[StepMeasurement]]:
        return run_rapids_pipeline(adata, config)
