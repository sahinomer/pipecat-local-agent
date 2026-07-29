"""Prometheus metrics and local hardware collectors."""

from __future__ import annotations

import re
import threading
from typing import Any
from urllib.parse import urlsplit

import psutil
from loguru import logger
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

try:
    import pynvml
except ImportError:  # pragma: no cover - dependency is installed in normal runs.
    pynvml = None


_PROCESSOR_INSTANCE_SUFFIX = re.compile(r"#\d+$")
_DURATION_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)
_CONNECTION_DURATION_BUCKETS = (
    1.0,
    5.0,
    15.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1800.0,
    3600.0,
)
def normalize_processor_name(name: str) -> str:
    """Remove Pipecat's per-instance suffix to keep label cardinality bounded."""

    return _PROCESSOR_INSTANCE_SUFFIX.sub("", name)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class LocalSystemCollector:
    """Collect host, agent-process, and optional NVIDIA GPU metrics."""

    def __init__(self, *, gpu_enabled: bool = True):
        self._process = psutil.Process()
        self._gpu_enabled = gpu_enabled
        self._nvml_initialized = False
        self._nvml_initialization_attempted = False
        self._nvml_lock = threading.Lock()

    def set_gpu_enabled(self, enabled: bool) -> None:
        self._gpu_enabled = enabled

    def _initialize_nvml(self) -> bool:
        if not self._gpu_enabled or pynvml is None:
            return False
        if self._nvml_initialized:
            return True

        with self._nvml_lock:
            if self._nvml_initialized:
                return True
            if self._nvml_initialization_attempted:
                return False

            self._nvml_initialization_attempted = True
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError as exc:
                logger.info("NVIDIA GPU metrics are unavailable: {}", exc)
                return False

            self._nvml_initialized = True
            return True

    def collect(self):
        host_cpu = CounterMetricFamily(
            "local_agent_host_cpu_seconds",
            "Cumulative host CPU time.",
            labels=["mode"],
        )
        cpu_times = psutil.cpu_times()
        for mode in cpu_times._fields:
            host_cpu.add_metric([mode], float(getattr(cpu_times, mode)))
        yield host_cpu

        logical_cpu_count = GaugeMetricFamily(
            "local_agent_host_logical_cpu_count",
            "Number of logical CPUs visible to the agent.",
        )
        logical_cpu_count.add_metric([], float(psutil.cpu_count() or 1))
        yield logical_cpu_count

        virtual_memory = psutil.virtual_memory()
        host_memory = GaugeMetricFamily(
            "local_agent_host_memory_bytes",
            "Host physical memory by state.",
            labels=["state"],
        )
        host_memory.add_metric(["total"], float(virtual_memory.total))
        host_memory.add_metric(["available"], float(virtual_memory.available))
        host_memory.add_metric(["used"], float(virtual_memory.used))
        yield host_memory

        process_cpu = CounterMetricFamily(
            "local_agent_process_cpu_seconds",
            "Cumulative CPU time used by the agent process.",
            labels=["mode"],
        )
        process_times = self._process.cpu_times()
        process_cpu.add_metric(["user"], float(process_times.user))
        process_cpu.add_metric(["system"], float(process_times.system))
        yield process_cpu

        process_memory_info = self._process.memory_info()
        process_memory = GaugeMetricFamily(
            "local_agent_process_memory_bytes",
            "Memory used by the agent process.",
            labels=["state"],
        )
        process_memory.add_metric(["resident"], float(process_memory_info.rss))
        process_memory.add_metric(["virtual"], float(process_memory_info.vms))
        yield process_memory

        process_threads = GaugeMetricFamily(
            "local_agent_process_threads",
            "Number of threads in the agent process.",
        )
        process_threads.add_metric([], float(self._process.num_threads()))
        yield process_threads

        process_start = GaugeMetricFamily(
            "local_agent_process_start_time_seconds",
            "Agent process start time as Unix epoch seconds.",
        )
        process_start.add_metric([], float(self._process.create_time()))
        yield process_start

        gpu_available = GaugeMetricFamily(
            "local_agent_gpu_collector_available",
            "Whether NVIDIA NVML GPU metrics are available.",
        )
        gpu_count = GaugeMetricFamily(
            "local_agent_gpu_count",
            "Number of NVIDIA GPUs visible to the agent.",
        )

        if not self._initialize_nvml():
            gpu_available.add_metric([], 0.0)
            gpu_count.add_metric([], 0.0)
            yield gpu_available
            yield gpu_count
            return

        gpu_available.add_metric([], 1.0)
        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as exc:
            logger.warning("Unable to query NVIDIA GPU count: {}", exc)
            gpu_available = GaugeMetricFamily(
                "local_agent_gpu_collector_available",
                "Whether NVIDIA NVML GPU metrics are available.",
            )
            gpu_count = GaugeMetricFamily(
                "local_agent_gpu_count",
                "Number of NVIDIA GPUs visible to the agent.",
            )
            gpu_available.add_metric([], 0.0)
            gpu_count.add_metric([], 0.0)
            yield gpu_available
            yield gpu_count
            return
        gpu_count.add_metric([], float(device_count))
        yield gpu_available
        yield gpu_count

        utilization = GaugeMetricFamily(
            "local_agent_gpu_utilization_ratio",
            "Whole-GPU utilization as a ratio from zero to one.",
            labels=["index", "uuid", "name"],
        )
        memory = GaugeMetricFamily(
            "local_agent_gpu_memory_bytes",
            "Whole-GPU memory by state.",
            labels=["index", "uuid", "name", "state"],
        )
        temperature = GaugeMetricFamily(
            "local_agent_gpu_temperature_celsius",
            "GPU temperature in degrees Celsius.",
            labels=["index", "uuid", "name"],
        )
        power = GaugeMetricFamily(
            "local_agent_gpu_power_watts",
            "GPU board power usage in watts.",
            labels=["index", "uuid", "name"],
        )

        for index in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                labels = [
                    str(index),
                    _text(pynvml.nvmlDeviceGetUUID(handle)),
                    _text(pynvml.nvmlDeviceGetName(handle)),
                ]

                gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization.add_metric(labels, float(gpu_utilization.gpu) / 100.0)

                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory.add_metric([*labels, "total"], float(gpu_memory.total))
                memory.add_metric([*labels, "used"], float(gpu_memory.used))
                memory.add_metric([*labels, "free"], float(gpu_memory.free))

                try:
                    value = pynvml.nvmlDeviceGetTemperature(
                        handle,
                        pynvml.NVML_TEMPERATURE_GPU,
                    )
                    temperature.add_metric(labels, float(value))
                except pynvml.NVMLError:
                    pass

                try:
                    value = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power.add_metric(labels, float(value) / 1000.0)
                except pynvml.NVMLError:
                    pass
            except pynvml.NVMLError as exc:
                logger.warning("Unable to collect NVIDIA GPU {} metrics: {}", index, exc)

        yield utilization
        yield memory
        yield temperature
        yield power


class AgentMetrics:
    """All application-owned Prometheus instruments."""

    def __init__(
        self,
        *,
        registry: CollectorRegistry | None = None,
        collect_system: bool = True,
        gpu_enabled: bool = True,
    ):
        self.registry = (
            registry
            if registry is not None
            else CollectorRegistry(auto_describe=True)
        )

        self.connections_active = Gauge(
            "local_agent_connections_active",
            "Current accepted WebSocket connections.",
            registry=self.registry,
        )
        self.connections_total = Counter(
            "local_agent_connections",
            "Accepted WebSocket connections.",
            registry=self.registry,
        )
        self.connection_terminations_total = Counter(
            "local_agent_connection_terminations",
            "Connection terminations by outcome.",
            labelnames=["outcome"],
            registry=self.registry,
        )
        self.connection_duration_seconds = Histogram(
            "local_agent_connection_duration_seconds",
            "Accepted WebSocket connection duration.",
            buckets=_CONNECTION_DURATION_BUCKETS,
            registry=self.registry,
        )

        self.pipeline_duration_seconds = Histogram(
            "local_agent_pipeline_duration_seconds",
            "Pipecat pipeline timing observations.",
            labelnames=["processor", "model", "metric"],
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.llm_tokens_total = Counter(
            "local_agent_llm_tokens",
            "LLM tokens reported by Pipecat.",
            labelnames=["processor", "model", "type"],
            registry=self.registry,
        )
        self.tts_characters_total = Counter(
            "local_agent_tts_characters",
            "TTS characters reported by Pipecat.",
            labelnames=["processor", "model"],
            registry=self.registry,
        )
        self.turn_predictions_total = Counter(
            "local_agent_turn_predictions",
            "Turn-completion predictions.",
            labelnames=["processor", "result"],
            registry=self.registry,
        )
        self.pipeline_errors_total = Counter(
            "local_agent_pipeline_errors",
            "Pipecat error frames by source processor.",
            labelnames=["processor"],
            registry=self.registry,
        )
        self.log_records_sent_total = Counter(
            "local_agent_log_records_sent",
            "Log records accepted by Loki.",
            registry=self.registry,
        )
        self.log_records_dropped_total = Counter(
            "local_agent_log_records_dropped",
            "Log records dropped before Loki accepted them.",
            labelnames=["reason"],
            registry=self.registry,
        )
        self.loki_available = Gauge(
            "local_agent_loki_available",
            "Whether the most recent Loki delivery attempt succeeded.",
            registry=self.registry,
        )

        self._system_collector = None
        if collect_system:
            self._system_collector = LocalSystemCollector(gpu_enabled=gpu_enabled)
            self.registry.register(self._system_collector)

    def set_gpu_enabled(self, enabled: bool) -> None:
        if self._system_collector is not None:
            self._system_collector.set_gpu_enabled(enabled)

    def observe_pipeline_duration(
        self,
        *,
        processor: str,
        model: str | None,
        metric: str,
        value: float,
    ) -> None:
        self.pipeline_duration_seconds.labels(
            processor=normalize_processor_name(processor),
            model=model or "",
            metric=metric,
        ).observe(value)

class MetricsHttpServer:
    """Lifecycle wrapper for prometheus_client's threaded HTTP server."""

    def __init__(
        self,
        *,
        url: str,
        registry: CollectorRegistry,
    ):
        parsed = urlsplit(url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Metrics URL contains an invalid port") from exc
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Metrics URL must be an HTTP URL with an explicit host and port"
            )

        self._host = parsed.hostname
        self._port = port
        self._registry = registry
        self._server = None
        self._thread = None

    def start(self) -> None:
        if self._server is not None:
            return
        self._server, self._thread = start_http_server(
            port=self._port,
            addr=self._host,
            registry=self._registry,
        )
        bound_port = self._server.server_port
        logger.info(
            "Prometheus metrics listening on http://{}:{}/metrics",
            self._host,
            bound_port,
        )

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None


METRICS = AgentMetrics()
