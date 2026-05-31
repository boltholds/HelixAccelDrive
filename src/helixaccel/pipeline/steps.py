from __future__ import annotations

import scanpy as sc
from anndata import AnnData

from helixaccel.config.settings import PipelineConfig


def step_qc_filter(adata: AnnData, config: PipelineConfig) -> AnnData:
    sc.pp.filter_cells(adata, min_genes=config.min_genes)
    sc.pp.filter_genes(adata, min_cells=config.min_cells)
    return adata


def step_normalize(adata: AnnData, config: PipelineConfig) -> AnnData:
    sc.pp.normalize_total(adata, target_sum=config.target_sum)
    return adata


def step_log1p(adata: AnnData, config: PipelineConfig) -> AnnData:
    sc.pp.log1p(adata)
    return adata


def step_hvg(adata: AnnData, config: PipelineConfig) -> AnnData:
    sc.pp.highly_variable_genes(adata, n_top_genes=config.n_top_genes)
    adata._inplace_subset_var(adata.var["highly_variable"].to_numpy())
    return adata


def step_scale(adata: AnnData, config: PipelineConfig) -> AnnData:
    sc.pp.scale(adata, max_value=10)
    return adata


def step_pca(adata: AnnData, config: PipelineConfig) -> AnnData:
    sc.tl.pca(adata, svd_solver="arpack", n_comps=config.n_pcs)
    return adata


def step_neighbors(adata: AnnData, config: PipelineConfig) -> AnnData:
    sc.pp.neighbors(adata, n_neighbors=config.n_neighbors, n_pcs=config.n_pcs)
    return adata


def step_leiden(adata: AnnData, config: PipelineConfig) -> AnnData:
    sc.tl.leiden(adata, resolution=config.leiden_resolution, key_added="leiden")
    return adata


def step_umap(adata: AnnData, config: PipelineConfig) -> AnnData:
    sc.tl.umap(adata)
    return adata


def step_markers(adata: AnnData, config: PipelineConfig) -> AnnData:
    sc.tl.rank_genes_groups(adata, groupby="leiden", method=config.marker_method)
    return adata
