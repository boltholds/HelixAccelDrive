from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy import sparse

from helixaccel.datasets.profiler import profile_adata


def test_profile_sparse_adata() -> None:
    x = sparse.csr_matrix(np.array([[1, 0, 2], [0, 0, 3]], dtype=np.float32))
    adata = AnnData(X=x)

    profile = profile_adata("toy", adata)

    assert profile.dataset == "toy"
    assert profile.n_cells == 2
    assert profile.n_genes == 3
    assert profile.nnz == 3
    assert profile.matrix_format == "csr"
    assert profile.estimated_dense_gb >= 0
