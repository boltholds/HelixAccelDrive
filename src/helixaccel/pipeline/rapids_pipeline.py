from __future__ import annotations

from typing import Callable

import scanpy as sc
from anndata import AnnData

from helixaccel.config.settings import PipelineConfig
from helixaccel.metrics.runtime import StepMeasurement, measure_step
from helixaccel.pipeline.steps import (
    _assert_non_empty,
    step_hvg,
    step_log1p,
    step_markers,
    step_normalize,
    step_qc_filter,
)


def _import_rapids():
    try:
        import rapids_singlecell as rsc  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "RAPIDS backend requires rapids-singlecell. "
            "Recommended environment: WSL2/Linux + conda/mamba + Python 3.11/3.12 + RAPIDS. "
            f"Original import error: {exc!r}"
        ) from exc
    return rsc


def _sync_gpu() -> None:
    try:
        import cupy as cp  # type: ignore

        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def _call_first(candidates: list[Callable[[], None]], step_name: str) -> None:
    errors: list[str] = []
    for candidate in candidates:
        try:
            candidate()
            _sync_gpu()
            return
        except AttributeError as exc:
            errors.append(repr(exc))
        except TypeError as exc:
            errors.append(repr(exc))
    raise RuntimeError(f"No compatible RAPIDS call found for {step_name}. Tried candidates: {errors}")


def step_transfer_to_gpu(adata: AnnData, config: PipelineConfig) -> AnnData:
    rsc = _import_rapids()
    _assert_non_empty(adata, "transfer_to_gpu/input")
    if hasattr(rsc, "get") and hasattr(rsc.get, "anndata_to_GPU"):
        rsc.get.anndata_to_GPU(adata)
    elif hasattr(rsc, "get") and hasattr(rsc.get, "anndata_to_gpu"):
        rsc.get.anndata_to_gpu(adata)
    else:
        raise RuntimeError("rapids_singlecell.get.anndata_to_GPU is not available in this environment.")
    adata.uns["_helixaccel_device"] = "gpu"
    _sync_gpu()
    return adata


def step_transfer_to_cpu(adata: AnnData, config: PipelineConfig) -> AnnData:
    rsc = _import_rapids()
    _assert_non_empty(adata, "transfer_to_cpu/input")
    if hasattr(rsc, "get") and hasattr(rsc.get, "anndata_to_CPU"):
        rsc.get.anndata_to_CPU(adata)
    elif hasattr(rsc, "get") and hasattr(rsc.get, "anndata_to_cpu"):
        rsc.get.anndata_to_cpu(adata)
    else:
        raise RuntimeError("rapids_singlecell.get.anndata_to_CPU is not available in this environment.")
    adata.uns["_helixaccel_device"] = "cpu"
    return adata


def step_rapids_scale(adata: AnnData, config: PipelineConfig) -> AnnData:
    rsc = _import_rapids()
    _assert_non_empty(adata, "rapids_scale/input")
    if hasattr(rsc, "pp") and hasattr(rsc.pp, "scale"):
        rsc.pp.scale(adata, max_value=10)
        _sync_gpu()
        return adata
    raise RuntimeError("rapids_singlecell.pp.scale is not available.")


def step_rapids_pca(adata: AnnData, config: PipelineConfig) -> AnnData:
    rsc = _import_rapids()
    _assert_non_empty(adata, "rapids_pca/input")
    max_comps = max(1, min(config.n_pcs, adata.n_obs - 1, adata.n_vars - 1))
    _call_first(
        [
            lambda: rsc.pp.pca(adata, n_comps=max_comps, random_state=config.random_state),
            lambda: rsc.tl.pca(adata, n_comps=max_comps, random_state=config.random_state),
            lambda: rsc.pp.pca(adata, n_comps=max_comps),
        ],
        "rapids_pca",
    )
    return adata


def step_rapids_neighbors(adata: AnnData, config: PipelineConfig) -> AnnData:
    rsc = _import_rapids()
    _assert_non_empty(adata, "rapids_neighbors/input")
    n_pcs = min(config.n_pcs, adata.obsm.get("X_pca", []).shape[1]) if "X_pca" in adata.obsm else config.n_pcs
    _call_first(
        [
            lambda: rsc.pp.neighbors(adata, n_neighbors=config.n_neighbors, n_pcs=n_pcs),
            lambda: rsc.pp.neighbors(adata, n_neighbors=config.n_neighbors),
        ],
        "rapids_neighbors",
    )
    return adata


def step_rapids_leiden(adata: AnnData, config: PipelineConfig) -> AnnData:
    rsc = _import_rapids()
    _assert_non_empty(adata, "rapids_leiden/input")
    _call_first(
        [
            lambda: rsc.tl.leiden(adata, resolution=config.leiden_resolution, key_added="leiden"),
            lambda: rsc.tl.leiden(adata, resolution=config.leiden_resolution),
        ],
        "rapids_leiden",
    )
    if "leiden" not in adata.obs and "clusters" in adata.obs:
        adata.obs["leiden"] = adata.obs["clusters"]
    return adata


def step_rapids_umap(adata: AnnData, config: PipelineConfig) -> AnnData:
    rsc = _import_rapids()
    _assert_non_empty(adata, "rapids_umap/input")
    _call_first(
        [
            lambda: rsc.tl.umap(adata, random_state=config.random_state),
            lambda: rsc.tl.umap(adata),
        ],
        "rapids_umap",
    )
    return adata


RAPIDS_HYBRID_STEPS: list[tuple[str, Callable[[AnnData, PipelineConfig], AnnData]]] = [
    ("qc_filter", step_qc_filter),
    ("normalize", step_normalize),
    ("log1p", step_log1p),
    ("hvg", step_hvg),
    ("transfer_to_gpu", step_transfer_to_gpu),
    ("rapids_scale", step_rapids_scale),
    ("rapids_pca", step_rapids_pca),
    ("rapids_neighbors", step_rapids_neighbors),
    ("rapids_leiden", step_rapids_leiden),
    ("rapids_umap", step_rapids_umap),
    ("transfer_to_cpu", step_transfer_to_cpu),
    ("marker_detection", step_markers),
]


def run_rapids_pipeline(adata: AnnData, config: PipelineConfig) -> tuple[AnnData, list[StepMeasurement]]:
    """Run Phase 3 hybrid RAPIDS baseline.

    The first GPU baseline intentionally keeps preprocessing and marker detection
    on CPU, while moving the core compute block to RAPIDS:
    transfer_to_gpu → scale → PCA → neighbors/KNN → Leiden → UMAP → transfer_to_cpu.
    """

    measurements: list[StepMeasurement] = []
    for name, fn in RAPIDS_HYBRID_STEPS:
        with measure_step(name, adata, record_gpu=True) as holder:
            adata = fn(adata, config)
        measurements.extend(holder)
    return adata, measurements
