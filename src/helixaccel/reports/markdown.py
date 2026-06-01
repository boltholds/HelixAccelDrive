from __future__ import annotations

from pathlib import Path

from helixaccel.storage.run_record import RunRecord


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _shape(value: dict) -> str:
    return str(value.get("shape"))


def _matrix_label(value: dict) -> str:
    return f"{value.get('matrix_format')} {value.get('dtype')} {_shape(value)}"


def _transition(before: dict, after: dict) -> str | None:
    changes: list[str] = []
    if before.get("matrix_format") != after.get("matrix_format"):
        changes.append(f"format: {before.get('matrix_format')}→{after.get('matrix_format')}")
    if before.get("dtype") != after.get("dtype"):
        changes.append(f"dtype: {before.get('dtype')}→{after.get('dtype')}")
    if before.get("shape") != after.get("shape"):
        changes.append(f"shape: {before.get('shape')}→{after.get('shape')}")
    if before.get("nnz") != after.get("nnz"):
        changes.append(f"nnz: {before.get('nnz')}→{after.get('nnz')}")
    return "; ".join(changes) if changes else None


def _step_by_name(record: RunRecord) -> dict[str, dict]:
    return {step["name"]: step for step in record.steps}


def _group_summary(record: RunRecord) -> list[dict[str, object]]:
    groups = [
        (
            "Preprocessing",
            ["qc_filter", "normalize", "log1p", "hvg", "scale"],
            "Load/QC/normalization/HVG/scale. Mostly CPU-friendly; watch sparse→dense transition during scale.",
        ),
        (
            "Core compute",
            ["pca", "neighbors", "leiden", "umap", "rapids_scale", "rapids_pca", "rapids_neighbors", "rapids_leiden", "rapids_umap"],
            "Main candidate for GPU/RAPIDS comparison: dimensionality reduction, KNN graph, clustering, embedding.",
        ),
        (
            "Transfer",
            ["transfer_to_gpu", "transfer_to_cpu"],
            "CPU↔GPU movement. Planner must account for these costs before claiming per-step routing gains.",
        ),
        (
            "Marker detection",
            ["marker_detection"],
            "Downstream biological interpretation / validation artifact. May remain CPU initially.",
        ),
    ]
    by_name = _step_by_name(record)
    total = record.total_runtime_sec or 0.0
    out: list[dict[str, object]] = []
    for group_name, names, note in groups:
        steps = [by_name[name] for name in names if name in by_name]
        runtime = sum(float(step.get("runtime_sec", 0.0)) for step in steps)
        peak_ram = max((float(step.get("peak_rss_mb", 0.0)) for step in steps), default=0.0)
        largest_delta = max((float(step.get("rss_delta_mb", 0.0)) for step in steps), default=0.0)
        share = (runtime / total * 100.0) if total else 0.0
        out.append(
            {
                "group": group_name,
                "steps": names,
                "runtime_sec": runtime,
                "share_pct": share,
                "peak_ram_mb": peak_ram,
                "largest_rss_delta_mb": largest_delta,
                "note": note,
            }
        )
    return out


def render_markdown_report(record: RunRecord) -> str:
    profile = record.dataset_profile
    sorted_steps = sorted(record.steps, key=lambda step: step["runtime_sec"], reverse=True)
    top = sorted_steps[:3]

    transitions = []
    for step in record.steps:
        transition = _transition(step["matrix_before"], step["matrix_after"])
        if transition:
            transitions.append((step, transition))

    lines: list[str] = []
    lines.append(f"# HelixAccel Benchmark Report — {record.run_id}")
    lines.append("")
    lines.append(f"Dataset: `{record.dataset}`")
    lines.append(f"Backend: `{record.backend}`")
    lines.append(f"Pipeline: `{record.pipeline}`")
    lines.append(f"Success: `{record.success}`")
    lines.append(f"Total runtime: `{record.total_runtime_sec:.3f} sec`")
    lines.append("")

    lines.append("## Pipeline parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---:|")
    for key, value in record.pipeline_params.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.append("")

    lines.append("## Dataset profile")
    lines.append("")
    lines.append(f"- Cells: `{profile.get('n_cells')}`")
    lines.append(f"- Genes: `{profile.get('n_genes')}`")
    lines.append(f"- NNZ: `{profile.get('nnz')}`")
    lines.append(f"- Sparsity: `{profile.get('sparsity')}`")
    lines.append(f"- Matrix format: `{profile.get('matrix_format')}`")
    lines.append(f"- Dtype: `{profile.get('dtype')}`")
    lines.append(f"- Dense estimate GB: `{profile.get('estimated_dense_gb')}`")
    lines.append(f"- Sparse estimate GB: `{profile.get('estimated_sparse_gb')}`")
    lines.append("")

    lines.append("## Runtime by step")
    lines.append("")
    lines.append(
        "| Step | Runtime sec | RSS before MB | RSS after MB | ΔRSS MB | Peak RAM MB | VRAM before MB | VRAM after MB | ΔVRAM MB | Peak VRAM MB | Matrix before | Matrix after | Success |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for step in record.steps:
        before = step["matrix_before"]
        after = step["matrix_after"]
        lines.append(
            "| {name} | {runtime} | {rss_before} | {rss_after} | {rss_delta} | {ram} | {vram_before} | {vram_after} | {vram_delta} | {vram_peak} | {before_label} | {after_label} | {success} |".format(
                name=step["name"],
                runtime=_fmt(step["runtime_sec"]),
                rss_before=_fmt(step.get("rss_before_mb")),
                rss_after=_fmt(step.get("rss_after_mb")),
                rss_delta=_fmt(step.get("rss_delta_mb")),
                ram=_fmt(step["peak_rss_mb"]),
                vram_before=_fmt(step.get("vram_before_mb")),
                vram_after=_fmt(step.get("vram_after_mb")),
                vram_delta=_fmt(step.get("vram_delta_mb")),
                vram_peak=_fmt(step.get("peak_vram_mb")),
                before_label=_matrix_label(before),
                after_label=_matrix_label(after),
                success=step["success"],
            )
        )
    lines.append("")

    lines.append("## Matrix transitions")
    lines.append("")
    if transitions:
        lines.append("| Step | Transition |")
        lines.append("|---|---|")
        for step, transition in transitions:
            lines.append(f"| `{step['name']}` | `{transition}` |")
    else:
        lines.append("No matrix format / dtype / shape / nnz transitions detected.")
    lines.append("")


    lines.append("## Core compute group summary")
    lines.append("")
    lines.append("| Group | Steps | Runtime sec | Share of total | Peak RAM MB | Largest ΔRSS MB | Planner note |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for group in _group_summary(record):
        steps = ", ".join(f"`{name}`" for name in group["steps"] if name in _step_by_name(record))
        lines.append(
            "| {group_name} | {steps} | {runtime:.3f} | {share:.1f}% | {peak:.3f} | {delta:.3f} | {note} |".format(
                group_name=group["group"],
                steps=steps,
                runtime=group["runtime_sec"],
                share=group["share_pct"],
                peak=group["peak_ram_mb"],
                delta=group["largest_rss_delta_mb"],
                note=group["note"],
            )
        )
    lines.append("")

    lines.append("## Bottleneck summary")
    lines.append("")
    for idx, step in enumerate(top, start=1):
        share = (step["runtime_sec"] / record.total_runtime_sec * 100.0) if record.total_runtime_sec else 0.0
        lines.append(f"{idx}. `{step['name']}` — `{step['runtime_sec']:.3f} sec` ({share:.1f}% of total)")
    lines.append("")

    lines.append("## GPU / transfer observations")
    lines.append("")
    gpu_steps = [step for step in record.steps if step.get("peak_vram_mb") is not None]
    if gpu_steps:
        max_vram_step = max(gpu_steps, key=lambda step: step.get("peak_vram_mb") or 0)
        lines.append(
            f"Highest sampled VRAM: `{_fmt(max_vram_step.get('peak_vram_mb'))} MB` during `{max_vram_step['name']}`."
        )
    else:
        lines.append("No VRAM metrics were captured. Install/use pynvml in an NVIDIA environment for GPU profiling.")
    transfer_steps = [step for step in record.steps if step["name"] in {"transfer_to_gpu", "transfer_to_cpu"}]
    if transfer_steps:
        lines.append("")
        lines.append("Transfer step runtime:")
        for step in transfer_steps:
            lines.append(f"- `{step['name']}`: `{step['runtime_sec']:.3f} sec`")
    lines.append("")

    lines.append("## Memory observations")
    lines.append("")
    max_mem_step = max(record.steps, key=lambda step: step["peak_rss_mb"], default=None)
    max_delta_step = max(record.steps, key=lambda step: step.get("rss_delta_mb", 0), default=None)
    if max_mem_step:
        lines.append(
            f"Highest sampled RAM: `{max_mem_step['peak_rss_mb']:.3f} MB` during `{max_mem_step['name']}`."
        )
    if max_delta_step:
        lines.append(
            f"Largest RSS increase: `{max_delta_step.get('rss_delta_mb', 0):.3f} MB` during `{max_delta_step['name']}`."
        )
    if transitions:
        dense_steps = [step for step, transition in transitions if "format:" in transition and "dense" in transition]
        if dense_steps:
            names = ", ".join(f"`{step['name']}`" for step in dense_steps)
            lines.append(f"Sparse/dense transition candidate(s): {names}.")
    lines.append("")

    if record.artifact_paths:
        lines.append("## Artifacts")
        lines.append("")
        for key, value in record.artifact_paths.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    if record.backend == "rapids-gpu":
        lines.append("## Phase 3 note")
        lines.append("")
        lines.append(
            "This report is generated by the Phase 3 RAPIDS GPU baseline runner. "
            "Biological validation gates, cost model, and planner v0 are intentionally out of scope here. "
            "The output artifacts are intended to be compared against the Phase 2 CPU reference in Phase 4."
        )
    else:
        lines.append("## Phase 2 note")
        lines.append("")
        lines.append(
            "This report is generated by the Phase 2 CPU baseline runner. "
            "GPU execution, biological validation gates, cost model, and planner v0 are intentionally out of scope here. "
            "The output artifacts are intended to become the CPU reference for later GPU comparison and validation."
        )
    lines.append("")
    return "\n".join(lines)


def save_markdown_report(record: RunRecord, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{record.run_id}.md"
    path.write_text(render_markdown_report(record), encoding="utf-8")
    return path
