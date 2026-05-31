from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import psutil


@dataclass
class MemorySample:
    rss_before_mb: float
    rss_after_mb: float
    peak_rss_mb: float


class MemorySampler:
    def __init__(self, interval_sec: float = 0.05) -> None:
        self.interval_sec = interval_sec
        self.process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.rss_before_mb = 0.0
        self.rss_after_mb = 0.0
        self.peak_rss_mb = 0.0

    def __enter__(self) -> "MemorySampler":
        self.rss_before_mb = self._rss_mb()
        self.peak_rss_mb = self.rss_before_mb
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.rss_after_mb = self._rss_mb()
        self.peak_rss_mb = max(self.peak_rss_mb, self.rss_after_mb)

    def _rss_mb(self) -> float:
        return self.process.memory_info().rss / (1024**2)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())
            time.sleep(self.interval_sec)

    def result(self) -> MemorySample:
        return MemorySample(
            rss_before_mb=round(self.rss_before_mb, 3),
            rss_after_mb=round(self.rss_after_mb, 3),
            peak_rss_mb=round(self.peak_rss_mb, 3),
        )
