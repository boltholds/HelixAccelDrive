from __future__ import annotations

from pathlib import Path

import scanpy as sc
from anndata import AnnData


H5_SUFFIXES = {".h5", ".hdf5"}
TENX_MTX_FILES = {"matrix.mtx", "matrix.mtx.gz"}
TENX_FEATURE_FILES = {"features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"}
TENX_BARCODE_FILES = {"barcodes.tsv", "barcodes.tsv.gz"}


def _find_single_file(directory: Path, suffixes: set[str]) -> Path | None:
    """Return a single matching file from a directory, or None.

    This supports the common case where the user places one downloaded .h5ad
    or 10x .h5 file inside data/raw/pbmc68k while config points to the folder.
    """
    matches = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    if len(matches) == 1:
        return matches[0]
    return None


def _is_10x_mtx_directory(directory: Path) -> bool:
    names = {p.name for p in directory.iterdir() if p.is_file()}
    return bool(names & TENX_MTX_FILES) and bool(names & TENX_FEATURE_FILES) and bool(names & TENX_BARCODE_FILES)


def _read_local_dataset(local_path: Path) -> AnnData:
    """Read a local dataset from .h5ad, 10x .h5, or 10x mtx directory.

    If a directory is provided, the loader first looks for a single .h5ad file,
    then a single 10x .h5 file, and only then treats the directory as a 10x
    Matrix Market folder. This avoids trying to read matrix.mtx.gz when the user
    actually uploaded an .h5ad into data/raw/pbmc68k.
    """
    if not local_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {local_path}")

    suffix = local_path.suffix.lower()
    if suffix == ".h5ad":
        return sc.read_h5ad(local_path)
    if suffix in H5_SUFFIXES:
        return sc.read_10x_h5(local_path)

    if local_path.is_dir():
        h5ad_file = _find_single_file(local_path, {".h5ad"})
        if h5ad_file is not None:
            return sc.read_h5ad(h5ad_file)

        h5_file = _find_single_file(local_path, H5_SUFFIXES)
        if h5_file is not None:
            return sc.read_10x_h5(h5_file)

        if _is_10x_mtx_directory(local_path):
            return sc.read_10x_mtx(local_path, var_names="gene_symbols", cache=True)

        files = ", ".join(sorted(p.name for p in local_path.iterdir() if p.is_file())[:20])
        raise FileNotFoundError(
            f"Could not detect dataset format in directory: {local_path}. "
            "Expected one .h5ad file, one 10x .h5 file, or a 10x mtx directory "
            "with matrix.mtx(.gz), features/genes.tsv(.gz), and barcodes.tsv(.gz). "
            f"Found files: {files or '<none>'}"
        )

    raise ValueError(
        f"Unsupported dataset path: {local_path}. "
        "Expected .h5ad, 10x .h5, or 10x matrix directory."
    )


def load_dataset(
    name: str,
    path: str | Path | None = None,
    cache_path: str | Path | None = None,
    force_reload: bool = False,
) -> AnnData:
    """Load a supported dataset.

    Phase 2 adds PBMC68k support through a local raw path plus optional cache.
    The code intentionally does not hard-code external download URLs; the same
    runner should work with public 10x files or a partner-lab dataset provided
    as .h5ad / 10x .h5 / 10x mtx directory.
    """
    normalized = name.lower().replace("-", "_")
    cache = Path(cache_path) if cache_path else None

    if cache and cache.exists() and not force_reload:
        return sc.read_h5ad(cache)

    if path:
        adata = _read_local_dataset(Path(path))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(cache)
        return adata

    if normalized == "pbmc3k":
        adata = sc.datasets.pbmc3k()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(cache)
        return adata

    if normalized in {"pbmc68k", "pbmc_68k"}:
        raise ValueError(
            "PBMC68k requires a local raw path or existing cache. "
            "Use --path data/raw/pbmc68k or set dataset.raw_path/cache_path in config. "
            "Supported formats: .h5ad, 10x .h5, or 10x mtx directory."
        )

    raise ValueError(
        f"Unknown dataset: {name}. Supported: pbmc3k, pbmc68k with local path/cache, or local path."
    )
