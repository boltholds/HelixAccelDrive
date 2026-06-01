from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class GpuMemorySample:
    vram_before_mb: float | None
    vram_after_mb: float | None
    peak_vram_mb: float | None
    available: bool
    error: str | None = None


class GpuMemorySampler:
    """Best-effort NVIDIA VRAM sampler.

    It uses pynvml when available. If pynvml / NVIDIA driver is unavailable,
    all values are returned as ``None`` and the benchmark still runs.
    """

    def __init__(self, device_index: int = 0, interval_sec: float = 0.05) -> None:
        self.device_index = device_index
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        self.available = False
        self.error: str | None = None
        self.vram_before_mb: float | None = None
        self.vram_after_mb: float | None = None
        self.peak_vram_mb: float | None = None

    def __enter__(self) -> "GpuMemorySampler":
        try:
            import pynvml  # type: ignore

            self._nvml = pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.available = True
            self.vram_before_mb = self._used_mb()
            self.peak_vram_mb = self.vram_before_mb
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        except Exception as exc:  # noqa: BLE001 - benchmark should not fail without NVML
            self.available = False
            self.error = repr(exc)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if self.available:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1)
            self.vram_after_mb = self._used_mb()
            if self.peak_vram_mb is None:
                self.peak_vram_mb = self.vram_after_mb
            elif self.vram_after_mb is not None:
                self.peak_vram_mb = max(self.peak_vram_mb, self.vram_after_mb)
            try:
                self._nvml.nvmlShutdown()  # type: ignore[union-attr]
            except Exception:
                pass

    def _used_mb(self) -> float:
        info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)  # type: ignore[union-attr]
        return round(float(info.used) / (1024**2), 3)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            used = self._used_mb()
            if self.peak_vram_mb is None:
                self.peak_vram_mb = used
            else:
                self.peak_vram_mb = max(self.peak_vram_mb, used)
            time.sleep(self.interval_sec)

    def result(self) -> GpuMemorySample:
        return GpuMemorySample(
            vram_before_mb=self.vram_before_mb,
            vram_after_mb=self.vram_after_mb,
            peak_vram_mb=self.peak_vram_mb,
            available=self.available,
            error=self.error,
        )
