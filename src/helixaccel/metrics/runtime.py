from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from helixaccel.datasets.profiler import matrix_snapshot
from helixaccel.metrics.memory import MemorySampler
from helixaccel.metrics.gpu import GpuMemorySampler


@dataclass
class StepMeasurement:
    name: str
    runtime_sec: float
    rss_before_mb: float
    rss_after_mb: float
    peak_rss_mb: float
    matrix_before: dict
    matrix_after: dict
    success: bool
    error: str | None = None
    peak_vram_mb: float | None = None
    vram_before_mb: float | None = None
    vram_after_mb: float | None = None
    gpu_metrics_available: bool = False
    gpu_metrics_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "runtime_sec": self.runtime_sec,
            "rss_before_mb": self.rss_before_mb,
            "rss_after_mb": self.rss_after_mb,
            "rss_delta_mb": round(self.rss_after_mb - self.rss_before_mb, 3),
            "peak_rss_mb": self.peak_rss_mb,
            "vram_before_mb": self.vram_before_mb,
            "vram_after_mb": self.vram_after_mb,
            "vram_delta_mb": None if self.vram_before_mb is None or self.vram_after_mb is None else round(self.vram_after_mb - self.vram_before_mb, 3),
            "peak_vram_mb": self.peak_vram_mb,
            "gpu_metrics_available": self.gpu_metrics_available,
            "gpu_metrics_error": self.gpu_metrics_error,
            "matrix_before": self.matrix_before,
            "matrix_after": self.matrix_after,
            "success": self.success,
            "error": self.error,
        }


@contextmanager
def measure_step(name: str, adata, record_gpu: bool = False) -> Iterator[list[StepMeasurement]]:  # type: ignore[no-untyped-def]
    """Measure one pipeline step.

    The sampler is controlled manually rather than via ``with MemorySampler()``.
    This is important because a ``contextmanager`` function runs its ``finally``
    block before nested context managers exit. Calling ``memory.result()`` before
    ``MemorySampler.__exit__`` caused ``rss_after_mb`` to remain 0.0.
    """

    holder: list[StepMeasurement] = []
    before = matrix_snapshot(adata)
    start = time.perf_counter()
    memory = MemorySampler()
    gpu_memory = GpuMemorySampler() if record_gpu else None
    memory.__enter__()
    if gpu_memory is not None:
        gpu_memory.__enter__()
    success = False
    error: str | None = None

    try:
        yield holder
        success = True
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        # Stop samplers before reading final memory values.
        if gpu_memory is not None:
            gpu_memory.__exit__(None, None, None)
        memory.__exit__(None, None, None)
        elapsed = round(time.perf_counter() - start, 6)
        after = matrix_snapshot(adata)
        mem = memory.result()
        gpu_mem = gpu_memory.result() if gpu_memory is not None else None
        holder.append(
            StepMeasurement(
                name=name,
                runtime_sec=elapsed,
                rss_before_mb=mem.rss_before_mb,
                rss_after_mb=mem.rss_after_mb,
                peak_rss_mb=mem.peak_rss_mb,
                matrix_before=before,
                matrix_after=after,
                success=success,
                error=error,
                vram_before_mb=None if gpu_mem is None else gpu_mem.vram_before_mb,
                vram_after_mb=None if gpu_mem is None else gpu_mem.vram_after_mb,
                peak_vram_mb=None if gpu_mem is None else gpu_mem.peak_vram_mb,
                gpu_metrics_available=False if gpu_mem is None else gpu_mem.available,
                gpu_metrics_error=None if gpu_mem is None else gpu_mem.error,
            )
        )
