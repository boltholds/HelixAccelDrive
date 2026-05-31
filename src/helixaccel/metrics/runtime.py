from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from helixaccel.datasets.profiler import matrix_snapshot
from helixaccel.metrics.memory import MemorySampler


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

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "runtime_sec": self.runtime_sec,
            "rss_before_mb": self.rss_before_mb,
            "rss_after_mb": self.rss_after_mb,
            "rss_delta_mb": round(self.rss_after_mb - self.rss_before_mb, 3),
            "peak_rss_mb": self.peak_rss_mb,
            "peak_vram_mb": self.peak_vram_mb,
            "matrix_before": self.matrix_before,
            "matrix_after": self.matrix_after,
            "success": self.success,
            "error": self.error,
        }


@contextmanager
def measure_step(name: str, adata) -> Iterator[list[StepMeasurement]]:  # type: ignore[no-untyped-def]
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
    memory.__enter__()
    success = False
    error: str | None = None

    try:
        yield holder
        success = True
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        # Stop memory sampling before reading final RSS.
        memory.__exit__(None, None, None)
        elapsed = round(time.perf_counter() - start, 6)
        after = matrix_snapshot(adata)
        mem = memory.result()
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
            )
        )
