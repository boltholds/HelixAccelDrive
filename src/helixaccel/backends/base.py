from __future__ import annotations

from abc import ABC, abstractmethod

from anndata import AnnData

from helixaccel.config.settings import PipelineConfig
from helixaccel.metrics.runtime import StepMeasurement


class PipelineBackend(ABC):
    name: str

    @abstractmethod
    def run(self, adata: AnnData, config: PipelineConfig) -> tuple[AnnData, list[StepMeasurement]]:
        raise NotImplementedError
