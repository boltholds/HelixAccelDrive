from __future__ import annotations

from pathlib import Path

import scanpy as sc
from anndata import AnnData


def load_dataset(name: str, path: str | None = None) -> AnnData:
    normalized = name.lower().replace("-", "_")

    if path:
        local_path = Path(path)
        if not local_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {local_path}")
        if local_path.suffix.lower() == ".h5ad":
            return sc.read_h5ad(local_path)
        return sc.read_10x_mtx(local_path, var_names="gene_symbols", cache=True)

    if normalized == "pbmc3k":
        return sc.datasets.pbmc3k()

    if normalized in {"pbmc68k", "pbmc_68k"}:
        raise ValueError(
            "PBMC 68k is intentionally not auto-downloaded in Phase 1. "
            "Provide --path to a local 10x directory or .h5ad file."
        )

    raise ValueError(f"Unknown dataset: {name}. Supported in Phase 1: pbmc3k, local .h5ad/10x path")
