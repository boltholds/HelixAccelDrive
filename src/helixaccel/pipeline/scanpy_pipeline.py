from __future__ import annotations

from collections.abc import Callable

from anndata import AnnData

from helixaccel.config.settings import PipelineConfig
from helixaccel.metrics.runtime import StepMeasurement, measure_step
from helixaccel.pipeline import steps

PipelineFn = Callable[[AnnData, PipelineConfig], AnnData]

CANONICAL_SCANPY_STEPS: list[tuple[str, PipelineFn]] = [
    ("qc_filter", steps.step_qc_filter),
    ("normalize", steps.step_normalize),
    ("log1p", steps.step_log1p),
    ("hvg", steps.step_hvg),
    ("scale", steps.step_scale),
    ("pca", steps.step_pca),
    ("neighbors", steps.step_neighbors),
    ("leiden", steps.step_leiden),
    ("umap", steps.step_umap),
    ("marker_detection", steps.step_markers),
]


def run_scanpy_pipeline(adata: AnnData, config: PipelineConfig) -> tuple[AnnData, list[StepMeasurement]]:
    measurements: list[StepMeasurement] = []
    for name, fn in CANONICAL_SCANPY_STEPS:
        with measure_step(name, adata) as holder:
            adata = fn(adata, config)
        measurements.extend(holder)
    return adata, measurements
