from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from helixaccel.backends import BACKENDS
from helixaccel.config.settings import load_config
from helixaccel.datasets.loaders import load_dataset
from helixaccel.datasets.profiler import profile_adata
from helixaccel.reports.markdown import save_markdown_report
from helixaccel.storage.artifacts import save_run_artifacts
from helixaccel.storage.json_store import save_run_record
from helixaccel.storage.run_record import build_run_record

app = typer.Typer(help="HelixAccel Phase 2 benchmark CLI")
console = Console()


def _resolve_dataset_io(
    dataset: str,
    config_path: str | None,
    path: str | None,
    cache_path: str | None,
    force_reload: bool,
):
    app_config = load_config(config_path)
    raw_path = Path(path) if path else app_config.dataset.raw_path
    cache = Path(cache_path) if cache_path else app_config.dataset.cache_path
    effective_dataset = dataset or app_config.dataset.name
    if not effective_dataset:
        raise typer.BadParameter("Dataset must be provided by --dataset or dataset.name in config.")
    effective_force_reload = force_reload or app_config.dataset.force_reload
    return app_config, effective_dataset, raw_path, cache, effective_force_reload


@app.command()
def profile(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name, e.g. pbmc3k/pbmc68k"),
    config: Optional[str] = typer.Option("configs/default.yaml", "--config", "-c"),
    path: Optional[str] = typer.Option(None, "--path", help="Optional local .h5ad, 10x .h5, or 10x directory"),
    cache_path: Optional[str] = typer.Option(None, "--cache-path", help="Optional .h5ad cache path"),
    force_reload: bool = typer.Option(False, "--force-reload", help="Ignore cache and reload from raw path"),
) -> None:
    """Load a dataset and print its matrix profile."""
    app_config, effective_dataset, raw_path, cache, effective_force_reload = _resolve_dataset_io(
        dataset or "", config, path, cache_path, force_reload
    )
    adata = load_dataset(effective_dataset, path=raw_path, cache_path=cache, force_reload=effective_force_reload)
    dataset_profile = profile_adata(effective_dataset, adata)

    table = Table(title=f"Dataset profile: {effective_dataset}")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in dataset_profile.to_dict().items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def run(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name, e.g. pbmc3k/pbmc68k"),
    backend: str = typer.Option("scanpy-cpu", "--backend", "-b", help="Backend name"),
    config: Optional[str] = typer.Option("configs/default.yaml", "--config", "-c"),
    path: Optional[str] = typer.Option(None, "--path", help="Optional local .h5ad, 10x .h5, or 10x directory"),
    cache_path: Optional[str] = typer.Option(None, "--cache-path", help="Optional .h5ad cache path"),
    force_reload: bool = typer.Option(False, "--force-reload", help="Ignore cache and reload from raw path"),
) -> None:
    """Run canonical scRNA-seq pipeline and save JSON + Markdown report."""
    app_config, effective_dataset, raw_path, cache, effective_force_reload = _resolve_dataset_io(
        dataset or "", config, path, cache_path, force_reload
    )
    if backend not in BACKENDS:
        allowed = ", ".join(sorted(BACKENDS))
        raise typer.BadParameter(f"Unsupported backend: {backend}. Allowed: {allowed}")

    console.print(f"[bold]Loading dataset[/bold]: {effective_dataset}")
    if raw_path:
        console.print(f"Raw path: {raw_path}")
    if cache:
        console.print(f"Cache path: {cache}")
    adata = load_dataset(effective_dataset, path=raw_path, cache_path=cache, force_reload=effective_force_reload)
    dataset_profile = profile_adata(effective_dataset, adata)

    dense_gb = dataset_profile.estimated_dense_gb
    if dense_gb >= app_config.safety.warn_dense_gb:
        console.print(
            f"[yellow]Warning:[/yellow] dense representation estimate is {dense_gb:.3f} GB "
            f"before filtering/HVG. Watch scale/PCA memory."
        )
    if app_config.safety.fail_dense_gb is not None and dense_gb >= app_config.safety.fail_dense_gb:
        raise typer.BadParameter(
            f"Dense estimate {dense_gb:.3f} GB exceeds safety.fail_dense_gb={app_config.safety.fail_dense_gb}. "
            "Lower config threshold only if this is intentional."
        )

    console.print(f"[bold]Running backend[/bold]: {backend}")
    backend_instance = BACKENDS[backend]()

    steps = []
    success = True
    error = None
    try:
        result_adata, steps = backend_instance.run(adata, app_config.pipeline)
    except Exception as exc:  # keep failed run record if possible
        success = False
        error = repr(exc)
        console.print(f"[red]Run failed:[/red] {error}")
        if not steps:
            raise

    result_adata = locals().get("result_adata", adata)
    record = build_run_record(
        dataset=effective_dataset,
        backend=backend,
        pipeline=app_config.pipeline.name,
        pipeline_params=app_config.pipeline,
        dataset_profile=dataset_profile,
        steps=steps,
        success=success,
        error=error,
    )

    if success:
        record.artifact_paths = save_run_artifacts(
            result_adata,
            record.run_id,
            app_config.storage.artifacts_dir,
            save_h5ad=app_config.storage.save_h5ad,
            save_clusters=app_config.storage.save_clusters,
            save_markers=app_config.storage.save_markers,
        )

    run_path = save_run_record(record, app_config.storage.runs_dir)
    report_path = save_markdown_report(record, app_config.storage.reports_dir)

    console.print("[green]Run completed[/green]")
    console.print(f"Total runtime: {record.total_runtime_sec:.3f} sec")
    console.print(f"Run record: {run_path}")
    console.print(f"Report: {report_path}")
