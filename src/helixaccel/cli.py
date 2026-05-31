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

app = typer.Typer(help="HelixAccel Phase 1 benchmark CLI")
console = Console()


@app.command()
def profile(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name, e.g. pbmc3k"),
    path: Optional[str] = typer.Option(None, "--path", help="Optional local .h5ad or 10x directory"),
) -> None:
    """Load a dataset and print its matrix profile."""
    adata = load_dataset(dataset, path=path)
    dataset_profile = profile_adata(dataset, adata)

    table = Table(title=f"Dataset profile: {dataset}")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in dataset_profile.to_dict().items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def run(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name, e.g. pbmc3k"),
    backend: str = typer.Option("scanpy-cpu", "--backend", "-b", help="Backend name"),
    config: Optional[str] = typer.Option("configs/default.yaml", "--config", "-c"),
    path: Optional[str] = typer.Option(None, "--path", help="Optional local .h5ad or 10x directory"),
) -> None:
    """Run canonical scRNA-seq pipeline and save JSON + Markdown report."""
    app_config = load_config(config)
    if backend not in BACKENDS:
        allowed = ", ".join(sorted(BACKENDS))
        raise typer.BadParameter(f"Unsupported backend: {backend}. Allowed: {allowed}")

    console.print(f"[bold]Loading dataset[/bold]: {dataset}")
    adata = load_dataset(dataset, path=path)
    dataset_profile = profile_adata(dataset, adata)

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
        dataset=dataset,
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
