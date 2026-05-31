from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from anndata import AnnData


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def save_run_artifacts(
    adata: AnnData,
    run_id: str,
    artifacts_dir: Path,
    *,
    save_h5ad: bool = True,
    save_clusters: bool = True,
    save_markers: bool = True,
) -> dict[str, Any]:
    """Persist lightweight output artifacts for later validation phases.

    Phase 4 will need baseline labels and markers to compute ARI/NMI and marker
    preservation. Saving paths already in Phase 1 keeps the run record useful as
    benchmark history rather than just a timing log.
    """

    run_dir = artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {"directory": _relative_path(run_dir)}

    if save_h5ad:
        h5ad_path = run_dir / "result.h5ad"
        adata.write_h5ad(h5ad_path)
        paths["h5ad"] = _relative_path(h5ad_path)

    if save_clusters and "leiden" in adata.obs:
        clusters_path = run_dir / "clusters.csv"
        adata.obs[["leiden"]].to_csv(clusters_path)
        paths["clusters_csv"] = _relative_path(clusters_path)

    if save_markers and "rank_genes_groups" in adata.uns:
        markers_path = run_dir / "markers.csv"
        markers = _rank_genes_groups_to_frame(adata)
        markers.to_csv(markers_path, index=False)
        paths["markers_csv"] = _relative_path(markers_path)

    return paths


def _rank_genes_groups_to_frame(adata: AnnData) -> pd.DataFrame:
    result = adata.uns["rank_genes_groups"]
    names = result.get("names")
    scores = result.get("scores")
    pvals_adj = result.get("pvals_adj")
    logfoldchanges = result.get("logfoldchanges")

    rows: list[dict[str, Any]] = []
    if names is None:
        return pd.DataFrame(rows)

    for group in names.dtype.names or []:
        group_names = names[group]
        group_scores = scores[group] if scores is not None else [None] * len(group_names)
        group_pvals_adj = pvals_adj[group] if pvals_adj is not None else [None] * len(group_names)
        group_logfc = (
            logfoldchanges[group] if logfoldchanges is not None else [None] * len(group_names)
        )
        for rank, gene in enumerate(group_names, start=1):
            rows.append(
                {
                    "cluster": group,
                    "rank": rank,
                    "gene": str(gene),
                    "score": _safe_float(group_scores[rank - 1]),
                    "pvals_adj": _safe_float(group_pvals_adj[rank - 1]),
                    "logfoldchange": _safe_float(group_logfc[rank - 1]),
                }
            )
    return pd.DataFrame(rows)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
