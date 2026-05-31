SUPPORTED_DATASETS = {
    "pbmc3k": {
        "role": "sanity/correctness baseline",
        "source": "scanpy.datasets.pbmc3k",
        "phase": "phase1",
    },
    "pbmc68k": {
        "role": "meaningful performance benchmark",
        "source": "10x Genomics public PBMC 68k; provide local 10x directory/.h5/.h5ad path",
        "phase": "phase2",
    },
}
