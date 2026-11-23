"""Server-side telemetry for the Whisper benchmark: CPU, GPU and loop lag."""

import asyncio
import threading
import time
from collections import deque
from typing import Any

import psutil
from loguru import logger

try:
    import pynvml
except ImportError:  # pragma: no cover - nvidia-ml-py is a project dependency
    pynvml = None


class SystemSampler:
    """Sample CPU, memory and GPU on a background thread.

    Sampling on a thread keeps readings coming while the event loop is busy,
    which is exactly when they matter.
    """

    def __init__(self, *, interval: float = 0.5, gpu_enabled: bool = True):
        self._interval = interval
        self._gpu_enabled = gpu_enabled
        self._process = psutil.Process()
        self._cpu_count = psutil.cpu_count() or 1
        self._gpu_handles: list[Any] = []
        self._latest: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._gpu_enabled:
            self._gpu_handles = self._init_gpus()
        # The first cpu_percent call only sets the baseline.
        psutil.cpu_percent(None)
        self._process.cpu_percent(None)
        self._latest = self._sample()
        self._thread = threading.Thread(
            target=self._run, name="bench-telemetry", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._gpu_handles and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def _init_gpus(self) -> list[Any]:
        if pynvml is None:
            return []
        try:
            pynvml.nvmlInit()
            return [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
        except pynvml.NVMLError as exc:
            logger.warning("GPU telemetry unavailable: {}", exc)
            return []

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            sample = self._sample()
            with self._lock:
                self._latest = sample

    def _sample(self) -> dict[str, Any]:
        memory = self._process.memory_info()
        sample: dict[str, Any] = {
            "cpu_percent": psutil.cpu_percent(None),
            # Normalized to the whole machine, like cpu_percent.
            "process_cpu_percent": self._process.cpu_percent(None) / self._cpu_count,
            "process_rss_mb": memory.rss / 2**20,
            "gpus": [],
        }
        for index, handle in enumerate(self._gpu_handles):
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            except pynvml.NVMLError:
                continue
            gpu = {
                "index": index,
                "utilization_percent": float(utilization.gpu),
                "memory_used_mb": gpu_memory.used / 2**20,
                "memory_total_mb": gpu_memory.total / 2**20,
            }
            # Laptop GPUs throttle when hot; clocks and temperature show it.
            for key, read in (
                ("temperature_c", lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
                ("sm_clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
                ("power_w", lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000),
            ):
                try:
                    gpu[key] = float(read())
                except pynvml.NVMLError:
                    pass
            sample["gpus"].append(gpu)
        return sample


class LoopLagMonitor:
    """Measure how late the event loop wakes a short sleep.

    A transcription that blocks the loop stalls audio for every session, so
    this is reported next to the queue state.
    """

    def __init__(self, *, interval: float = 0.02, window: float = 1.0):
        self._interval = interval
        self._window = window
        self._samples: deque[tuple[float, float]] = deque()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def max_lag_ms(self) -> float:
        """The worst lag in the last ``window`` seconds, in milliseconds."""
        cutoff = time.monotonic() - self._window
        return max((lag for at, lag in self._samples if at >= cutoff), default=0.0) * 1000

    async def _run(self) -> None:
        while True:
            started = time.monotonic()
            await asyncio.sleep(self._interval)
            now = time.monotonic()
            self._samples.append((now, max(0.0, now - started - self._interval)))
            cutoff = now - self._window
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()
