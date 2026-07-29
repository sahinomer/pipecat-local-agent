"""Non-blocking direct delivery of structured Loguru records to Loki."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

import aiohttp

from observability.metrics import METRICS, AgentMetrics


@dataclass(frozen=True)
class _LokiLogEntry:
    timestamp_ns: str
    level: str
    line: str
    metadata: dict[str, str]


class LokiLogSink:
    """Batch Loguru records directly to Loki without persistent buffering.

    Logging calls only enqueue an in-memory record. A background task performs
    HTTP delivery, and records are dropped if the queue is full or Loki rejects
    a batch. No processor or service needs to know that this sink exists.
    """

    def __init__(
        self,
        *,
        base_url: str,
        metrics: AgentMetrics = METRICS,
        service_name: str = "pipecat-local-agent",
        queue_capacity: int = 5000,
        batch_size: int = 100,
        flush_interval: float = 0.5,
        request_timeout: float = 2.0,
    ):
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be greater than zero")
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        if flush_interval <= 0:
            raise ValueError("flush_interval must be greater than zero")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")

        self._push_url = f"{base_url.rstrip('/')}/loki/api/v1/push"
        self._metrics = metrics
        self._service_name = service_name
        self._queue_capacity = queue_capacity
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._request_timeout = request_timeout

        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[_LokiLogEntry | None] | None = None
        self._session: aiohttp.ClientSession | None = None
        self._worker: asyncio.Task | None = None
        self._accepting = False
        self._last_warning_time = 0.0
        self._warning_lock = threading.Lock()

    async def start(self) -> None:
        """Start the in-memory queue and Loki delivery task."""

        if self._worker is not None:
            return

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._queue_capacity)
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._request_timeout)
        )
        self._accepting = True
        self._worker = self._loop.create_task(
            self._run(),
            name="loki-log-delivery",
        )

    async def stop(self) -> None:
        """Flush queued records once, then close the HTTP client."""

        if self._worker is None or self._queue is None:
            return

        self._accepting = False
        await asyncio.sleep(0)
        await self._queue.put(None)
        await self._worker

        if self._session is not None:
            await self._session.close()

        self._worker = None
        self._session = None
        self._queue = None
        self._loop = None

    def __call__(self, message) -> None:
        """Receive a Loguru message without blocking the caller."""

        entry = self._entry_from_record(message.record)
        loop = self._loop
        if not self._accepting or loop is None:
            self._record_drop("not_running", 1)
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is loop:
            self._enqueue(entry)
            return

        try:
            loop.call_soon_threadsafe(self._enqueue, entry)
        except RuntimeError:
            self._record_drop("event_loop_closed", 1)

    def _enqueue(self, entry: _LokiLogEntry) -> None:
        queue = self._queue
        if not self._accepting or queue is None:
            self._record_drop("not_running", 1)
            return

        try:
            queue.put_nowait(entry)
        except asyncio.QueueFull:
            self._record_drop("queue_full", 1)

    async def _run(self) -> None:
        queue = self._queue
        if queue is None:
            return

        while True:
            first = await queue.get()
            if first is None:
                return

            batch = [first]
            stop_after_batch = False
            deadline = asyncio.get_running_loop().time() + self._flush_interval

            while len(batch) < self._batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    entry = await asyncio.wait_for(
                        queue.get(),
                        timeout=remaining,
                    )
                except TimeoutError:
                    break

                if entry is None:
                    stop_after_batch = True
                    break
                batch.append(entry)

            await self._send(batch)
            if stop_after_batch:
                return

    async def _send(self, batch: list[_LokiLogEntry]) -> None:
        session = self._session
        if session is None:
            self._record_drop("not_running", len(batch))
            return

        streams: dict[str, list[list[Any]]] = {}
        for entry in batch:
            streams.setdefault(entry.level, []).append(
                [entry.timestamp_ns, entry.line, entry.metadata]
            )

        payload = {
            "streams": [
                {
                    "stream": {
                        "service_name": self._service_name,
                        "level": level,
                    },
                    "values": values,
                }
                for level, values in streams.items()
            ]
        }

        try:
            async with session.post(self._push_url, json=payload) as response:
                if not 200 <= response.status < 300:
                    response_body = await response.text()
                    raise RuntimeError(
                        f"Loki returned HTTP {response.status}: "
                        f"{response_body[:200]}"
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
            self._metrics.loki_available.set(0)
            self._record_drop("delivery_failed", len(batch))
            self._warn_delivery_failure(exc, len(batch))
            return

        self._metrics.loki_available.set(1)
        self._metrics.log_records_sent_total.inc(len(batch))

    def _record_drop(self, reason: str, count: int) -> None:
        self._metrics.log_records_dropped_total.labels(reason=reason).inc(count)

    def _warn_delivery_failure(self, exc: Exception, count: int) -> None:
        now = time.monotonic()
        with self._warning_lock:
            if now - self._last_warning_time < 30:
                return
            self._last_warning_time = now
        print(
            f"Loki log delivery failed; dropped {count} record(s): {exc}",
            file=sys.stderr,
        )

    def _entry_from_record(self, record: dict[str, Any]) -> _LokiLogEntry:
        timestamp = record["time"]
        level = record["level"].name
        extra = {
            key: LokiLogSink._string_value(value)
            for key, value in record["extra"].items()
        }
        session_id = extra.get("session_id", "system")

        exception = record["exception"]
        exception_text = None
        if exception is not None:
            exception_text = "".join(
                traceback.format_exception(
                    exception.type,
                    exception.value,
                    exception.traceback,
                )
            )

        document = {
            **extra,
            "timestamp": timestamp.isoformat(),
            "level": level,
            "service": self._service_name,
            "session_id": session_id,
            "logger": record["name"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "exception": exception_text,
        }
        metadata = {
            "session_id": session_id,
            "module": str(record["module"]),
        }

        return _LokiLogEntry(
            timestamp_ns=str(int(timestamp.timestamp() * 1_000_000_000)),
            level=level,
            line=json.dumps(
                document,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            metadata=metadata,
        )

    @staticmethod
    def _string_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return json.dumps(value, ensure_ascii=False, default=str)
