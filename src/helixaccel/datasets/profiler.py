from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from anndata import AnnData
from scipy import sparse


@dataclass(frozen=True)
class DatasetProfile:
    dataset: str
    n_cells: int
    n_genes: int
    nnz: int | None
    sparsity: float | None
    matrix_format: str
    dtype: str
    estimated_dense_gb: float
    estimated_sparse_gb: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matrix_format(x: Any) -> str:
    if sparse.isspmatrix_csr(x):
        return "csr"
    if sparse.isspmatrix_csc(x):
        return "csc"
    if sparse.isspmatrix_coo(x):
        return "coo"
    if sparse.issparse(x):
        return "sparse"
    return "dense"


def _dtype_size(dtype: Any) -> int:
    try:
        return int(np.dtype(dtype).itemsize)
    except TypeError:
        return 4


def profile_adata(dataset: str, adata: AnnData) -> DatasetProfile:
    x = adata.X
    n_cells, n_genes = adata.shape
    dtype = getattr(x, "dtype", np.float32)
    itemsize = _dtype_size(dtype)
    estimated_dense_gb = n_cells * n_genes * itemsize / (1024**3)

    if sparse.issparse(x):
        nnz = int(x.nnz)
        sparsity = 1.0 - (nnz / float(n_cells * n_genes))
        estimated_sparse_gb = (x.data.nbytes + x.indices.nbytes + x.indptr.nbytes) / (1024**3)
    else:
        nnz = int(np.count_nonzero(x))
        sparsity = 1.0 - (nnz / float(n_cells * n_genes))
        estimated_sparse_gb = None

    return DatasetProfile(
        dataset=dataset,
        n_cells=n_cells,
        n_genes=n_genes,
        nnz=nnz,
        sparsity=round(float(sparsity), 6),
        matrix_format=_matrix_format(x),
        dtype=str(dtype),
        estimated_dense_gb=round(float(estimated_dense_gb), 6),
        estimated_sparse_gb=None if estimated_sparse_gb is None else round(float(estimated_sparse_gb), 6),
    )


def matrix_snapshot(adata: AnnData) -> dict[str, Any]:
    x = adata.X
    n_cells, n_genes = adata.shape
    if sparse.issparse(x):
        nnz = int(x.nnz)
    else:
        nnz = int(np.count_nonzero(x))
    return {
        "shape": [int(n_cells), int(n_genes)],
        "matrix_format": _matrix_format(x),
        "dtype": str(getattr(x, "dtype", "unknown")),
        "nnz": nnz,
    }
