from __future__ import annotations

import scanpy as sc
from anndata import AnnData

from helixaccel.config.settings import PipelineConfig


def _assert_non_empty(adata: AnnData, step_name: str) -> None:
    """Fail early with a useful message when QC/config removes all data."""
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(
            f"Step '{step_name}' produced an empty AnnData object "
            f"with shape=({adata.n_obs}, {adata.n_vars}). "
            "For PBMC68k-like sparse datasets, lower/disable QC thresholds "
            "such as pipeline.min_genes or pipeline.min_cells, then rerun."
        )


def step_qc_filter(adata: AnnData, config: PipelineConfig) -> AnnData:
    """Apply lightweight QC filters.

    Thresholds <= 0 are treated as disabled. This is useful for public PBMC68k
    files that may already be filtered or have low detected genes per barcode.
    """
    if config.min_genes and config.min_genes > 0:
        sc.pp.filter_cells(adata, min_genes=config.min_genes)
        _assert_non_empty(adata, "qc_filter/filter_cells")

    if config.min_cells and config.min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=config.min_cells)
        _assert_non_empty(adata, "qc_filter/filter_genes")

    return adata


def step_normalize(adata: AnnData, config: PipelineConfig) -> AnnData:
    _assert_non_empty(adata, "normalize/input")
    sc.pp.normalize_total(adata, target_sum=config.target_sum)
    return adata


def step_log1p(adata: AnnData, config: PipelineConfig) -> AnnData:
    _assert_non_empty(adata, "log1p/input")
    sc.pp.log1p(adata)
    return adata


def step_hvg(adata: AnnData, config: PipelineConfig) -> AnnData:
    _assert_non_empty(adata, "hvg/input")
    n_top_genes = min(config.n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata._inplace_subset_var(adata.var["highly_variable"].to_numpy())
    _assert_non_empty(adata, "hvg/output")
    return adata


def step_scale(adata: AnnData, config: PipelineConfig) -> AnnData:
    _assert_non_empty(adata, "scale/input")
    sc.pp.scale(adata, max_value=10)
    return adata


def step_pca(adata: AnnData, config: PipelineConfig) -> AnnData:
    _assert_non_empty(adata, "pca/input")
    max_comps = max(1, min(config.n_pcs, adata.n_obs - 1, adata.n_vars - 1))
    sc.tl.pca(adata, svd_solver="arpack", n_comps=max_comps, random_state=config.random_state)
    return adata


def step_neighbors(adata: AnnData, config: PipelineConfig) -> AnnData:
    _assert_non_empty(adata, "neighbors/input")
    n_pcs = min(config.n_pcs, adata.obsm.get("X_pca", []).shape[1]) if "X_pca" in adata.obsm else config.n_pcs
    sc.pp.neighbors(adata, n_neighbors=config.n_neighbors, n_pcs=n_pcs)
    return adata


def step_leiden(adata: AnnData, config: PipelineConfig) -> AnnData:
    _assert_non_empty(adata, "leiden/input")
    sc.tl.leiden(adata, resolution=config.leiden_resolution, key_added="leiden")
    return adata


def step_umap(adata: AnnData, config: PipelineConfig) -> AnnData:
    _assert_non_empty(adata, "umap/input")
    sc.tl.umap(adata, random_state=config.random_state)
    return adata


def step_markers(adata: AnnData, config: PipelineConfig) -> AnnData:
    _assert_non_empty(adata, "marker_detection/input")
    sc.tl.rank_genes_groups(adata, groupby="leiden", method=config.marker_method)
    return adata
