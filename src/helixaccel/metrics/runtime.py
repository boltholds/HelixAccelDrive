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
            "peak_rss_mb": self.peak_rss_mb,
            "peak_vram_mb": self.peak_vram_mb,
            "matrix_before": self.matrix_before,
            "matrix_after": self.matrix_after,
            "success": self.success,
            "error": self.error,
        }


@contextmanager
def measure_step(name: str, adata) -> Iterator[list[StepMeasurement]]:  # type: ignore[no-untyped-def]
    holder: list[StepMeasurement] = []
    before = matrix_snapshot(adata)
    start = time.perf_counter()
    with MemorySampler() as memory:
        try:
            yield holder
            success = True
            error = None
        except Exception as exc:
            success = False
            error = repr(exc)
            raise
        finally:
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
