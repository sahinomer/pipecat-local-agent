"""Process-wide Whisper engines shared by every STT session.

An engine owns one loaded faster-whisper model and runs transcriptions for
any number of pipelines. It holds no per-session state: everything that
belongs to a session (audio, language, hotwords, initial prompt) is passed
with each request, so concurrent sessions cannot affect each other's
results.
"""

import asyncio
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from loguru import logger
from pipecat.utils.models import load_shared_model


class WhisperEngine(Protocol):
    """Transcribes audio for many sessions with one loaded model."""

    @property
    def model_name(self) -> str:
        """The model the engine serves, as named in its configuration."""
        ...

    @property
    def supported_languages(self) -> Sequence[str] | None:
        """Language codes the model can transcribe, or ``None`` if unknown."""
        ...

    async def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
        hotwords: str | None = None,
        initial_prompt: str | None = None,
    ) -> list[Segment]:
        """Transcribe mono float32 audio at 16 kHz."""
        ...


@dataclass(frozen=True)
class WhisperEngineConfig:
    """Everything that decides which model is loaded and how it runs.

    Two configs that compare equal share one loaded engine.

    Parameters:
        model: A faster-whisper model size, Hugging Face repo, or local path.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"``.
        compute_type: CTranslate2 compute type, e.g. ``"float16"`` or ``"int8"``.
        num_workers: Transcriptions that may run at once. Each worker is a
            separate model replica inside CTranslate2, so memory use grows
            with it. ``1`` runs requests one at a time in arrival order.
        cpu_threads: Threads per worker on CPU; ``0`` keeps the library default.
    """

    model: str
    device: str = "auto"
    compute_type: str = "default"
    num_workers: int = 1
    cpu_threads: int = 0

    def __post_init__(self):
        if not str(self.model).strip():
            raise ValueError("Whisper model cannot be empty")
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if self.cpu_threads < 0:
            raise ValueError("cpu_threads cannot be negative")


class SharedWhisperEngine:
    """Run transcriptions from every session on one shared model.

    Requests go to a dedicated thread pool with ``num_workers`` threads, the
    same number of workers CTranslate2 is given, so at most that many
    transcriptions run at once and the rest wait in arrival order. A request
    whose caller is cancelled while it is still waiting is dropped without
    reaching the model; one that is already running finishes and its result
    is discarded.

    Normal code gets an engine with :meth:`load`, which loads the model off
    the event loop once per process and hands the same engine to every later
    caller with an equal config. The constructor accepts any object with
    faster-whisper's ``transcribe`` signature, which tests use to inject a
    fake model.
    """

    def __init__(
        self,
        model: Any,
        *,
        model_name: str,
        num_workers: int = 1,
    ):
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")

        self._model = model
        self._model_name = str(model_name)
        self._num_workers = num_workers
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="whisper-engine",
        )
        self._stats_lock = threading.Lock()
        self._queued = 0
        self._running = 0
        self._closed = False

    @classmethod
    async def load(cls, config: WhisperEngineConfig) -> "SharedWhisperEngine":
        """Return the process-wide engine for ``config``, loading it on first use."""
        return await load_shared_model(
            ("pipecat-local-agent", "whisper-engine", config),
            lambda: cls.from_config(config),
        )

    @classmethod
    def from_config(cls, config: WhisperEngineConfig) -> "SharedWhisperEngine":
        """Load a new, unshared engine. Blocks while the model loads."""
        logger.info(
            "Loading shared Whisper model {} (device={}, compute_type={}, workers={})",
            config.model,
            config.device,
            config.compute_type,
            config.num_workers,
        )
        model = WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
            cpu_threads=config.cpu_threads,
            num_workers=config.num_workers,
        )
        logger.info("Loaded shared Whisper model {}", config.model)
        return cls(model, model_name=config.model, num_workers=config.num_workers)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supported_languages(self) -> Sequence[str] | None:
        return getattr(self._model, "supported_languages", None)

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def queued(self) -> int:
        """Requests waiting for a free worker."""
        with self._stats_lock:
            return self._queued

    @property
    def running(self) -> int:
        """Requests the model is transcribing right now."""
        with self._stats_lock:
            return self._running

    @property
    def closed(self) -> bool:
        return self._closed

    async def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
        hotwords: str | None = None,
        initial_prompt: str | None = None,
    ) -> list[Segment]:
        """Transcribe mono float32 audio at 16 kHz on a worker thread.

        Raises:
            RuntimeError: If the engine has been closed.
        """
        if self._closed:
            raise RuntimeError("Whisper engine is closed")

        with self._stats_lock:
            self._queued += 1
        try:
            future = self._executor.submit(
                self._transcribe_blocking,
                audio,
                language,
                hotwords,
                initial_prompt,
            )
        except RuntimeError:
            # The executor was shut down between the closed check and submit.
            with self._stats_lock:
                self._queued -= 1
            raise RuntimeError("Whisper engine is closed") from None
        future.add_done_callback(self._forget_if_cancelled)

        # Cancelling the awaiting task cancels ``future`` too; a request that
        # has not started yet is then removed from the queue.
        return await asyncio.wrap_future(future)

    def close(self) -> None:
        """Refuse new requests and fail the ones still waiting.

        Waiting requests raise ``RuntimeError`` rather than being cancelled, so
        a session sees an ordinary transcription error instead of a
        ``CancelledError`` that would stop its tasks. Requests already running
        finish. The engine stays in the process-wide registry, so close it
        only when the process is shutting down.
        """
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=False)

    def _forget_if_cancelled(self, future: Future) -> None:
        if future.cancelled():
            with self._stats_lock:
                self._queued -= 1

    def _transcribe_blocking(
        self,
        audio: np.ndarray,
        language: str | None,
        hotwords: str | None,
        initial_prompt: str | None,
    ) -> list[Segment]:
        with self._stats_lock:
            self._queued -= 1
            self._running += 1
        try:
            if self._closed:
                raise RuntimeError("Whisper engine is closed")
            # Only per-request arguments are passed. ``chunk_length`` must never
            # be passed: faster-whisper stores it on the shared feature
            # extractor, which would change it for every other session.
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                hotwords=hotwords,
                initial_prompt=initial_prompt,
            )
            # ``segments`` is lazy and decodes while it is consumed, so consume
            # it here on the worker thread, not on the event loop.
            return list(segments)
        finally:
            with self._stats_lock:
                self._running -= 1
