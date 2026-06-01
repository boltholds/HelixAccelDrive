from __future__ import annotations

from dataclasses import asdict, dataclass
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


def _is_cupy_array(x: Any) -> bool:
    return x.__class__.__module__.startswith("cupy")


def _is_cupy_sparse(x: Any) -> bool:
    module = x.__class__.__module__
    return module.startswith("cupyx.scipy.sparse")


def _matrix_format(x: Any) -> str:
    if sparse.isspmatrix_csr(x):
        return "csr"
    if sparse.isspmatrix_csc(x):
        return "csc"
    if sparse.isspmatrix_coo(x):
        return "coo"
    if sparse.issparse(x):
        return "sparse"
    if _is_cupy_sparse(x):
        name = x.__class__.__name__.lower()
        if "csr" in name:
            return "gpu_csr"
        if "csc" in name:
            return "gpu_csc"
        if "coo" in name:
            return "gpu_coo"
        return "gpu_sparse"
    if _is_cupy_array(x):
        return "gpu_dense"
    return "dense"


def _dtype_size(dtype: Any) -> int:
    try:
        return int(np.dtype(dtype).itemsize)
    except TypeError:
        return 4


def _safe_nnz(x: Any) -> int | None:
    if sparse.issparse(x):
        return int(x.nnz)
    if _is_cupy_sparse(x):
        return int(getattr(x, "nnz"))
    if _is_cupy_array(x):
        try:
            import cupy as cp  # type: ignore

            return int(cp.count_nonzero(x).get())
        except Exception:
            return None
    try:
        return int(np.count_nonzero(x))
    except Exception:
        return None


def _safe_sparse_gb(x: Any) -> float | None:
    if sparse.issparse(x):
        return float(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes) / (1024**3)
    if _is_cupy_sparse(x):
        try:
            return float(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes) / (1024**3)
        except Exception:
            return None
    return None


def profile_adata(dataset: str, adata: AnnData) -> DatasetProfile:
    x = adata.X
    n_cells, n_genes = adata.shape
    dtype = getattr(x, "dtype", np.float32)
    itemsize = _dtype_size(dtype)
    estimated_dense_gb = n_cells * n_genes * itemsize / (1024**3)

    nnz = _safe_nnz(x)
    sparsity = None if nnz is None else 1.0 - (nnz / float(n_cells * n_genes))
    estimated_sparse_gb = _safe_sparse_gb(x)

    return DatasetProfile(
        dataset=dataset,
        n_cells=n_cells,
        n_genes=n_genes,
        nnz=nnz,
        sparsity=None if sparsity is None else round(float(sparsity), 6),
        matrix_format=_matrix_format(x),
        dtype=str(dtype),
        estimated_dense_gb=round(float(estimated_dense_gb), 6),
        estimated_sparse_gb=None if estimated_sparse_gb is None else round(float(estimated_sparse_gb), 6),
    )


def matrix_snapshot(adata: AnnData) -> dict[str, Any]:
    x = adata.X
    n_cells, n_genes = adata.shape
    nnz = _safe_nnz(x)
    return {
        "shape": [int(n_cells), int(n_genes)],
        "matrix_format": _matrix_format(x),
        "dtype": str(getattr(x, "dtype", "unknown")),
        "nnz": nnz,
    }
