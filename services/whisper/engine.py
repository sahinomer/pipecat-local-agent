"""Process-wide Whisper engines shared by every STT session.

An engine owns one loaded faster-whisper model and runs transcriptions for
any number of pipelines. It holds no per-session state: everything that
belongs to a session (audio, language, hotwords, initial prompt) is passed
with each request, so concurrent sessions cannot affect each other's
results.
"""

import asyncio
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from loguru import logger
from pipecat.utils.models import load_shared_model


@dataclass(frozen=True)
class WhisperTranscription:
    """One request's result and where its time went.

    Parameters:
        segments: The decoded segments, fully consumed.
        queue_wait_secs: Time from submission until a worker picked it up.
        inference_secs: Time the worker spent on it, including feature
            extraction, decoding and, for a batched engine, batch waits.
        batch_wait_secs: Batched engine only: time spent waiting for a batch
            slot on the GPU, summed over the request's decode calls.
        batch_size: Batched engine only: the largest batch the request's
            decode calls ran in; ``1`` when it ran alone.
    """

    segments: list[Segment]
    queue_wait_secs: float
    inference_secs: float
    batch_wait_secs: float = 0.0
    batch_size: int = 1


@dataclass(frozen=True)
class WhisperEngineStats:
    """A point-in-time view of an engine's load and totals.

    The ``batch*`` fields stay at their defaults for an unbatched engine.
    """

    workers: int
    queued: int
    running: int
    completed: int
    failed: int
    cancelled: int
    max_batch_size: int = 1
    batch_pending: int = 0
    batches: int = 0
    batched_items: int = 0
    last_batch_size: int = 0


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
    ) -> WhisperTranscription:
        """Transcribe mono float32 audio at 16 kHz."""
        ...

    def stats(self) -> WhisperEngineStats:
        """The engine's current load and totals."""
        ...


@dataclass(frozen=True)
class WhisperEngineConfig:
    """Everything that decides which model is loaded and how it runs.

    Two configs that compare equal share one loaded engine.

    Parameters:
        model: A faster-whisper model size, Hugging Face repo, or local path.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"``.
        compute_type: CTranslate2 compute type, e.g. ``"float16"`` or ``"int8"``.
        num_workers: Unbatched engine: transcriptions that may run at once.
            Each worker is a separate model replica inside CTranslate2, so
            memory use grows with it. A batched engine needs exactly one.
        cpu_threads: Threads per worker on CPU; ``0`` keeps the library default.
        beam_size: Beam search width; ``1`` is greedy decoding. faster-whisper
            and Pipecat use 5.
        max_batch_size: Decode calls the GPU may run together. ``1`` selects
            the unbatched :class:`SharedWhisperEngine`.
        max_batch_wait_ms: Batched engine: how long a batch may wait for more
            requests before it runs. ``0`` runs whatever is waiting as soon as
            the GPU is free, which adds no latency when load is light.
    """

    model: str
    device: str = "auto"
    compute_type: str = "default"
    num_workers: int = 1
    cpu_threads: int = 0
    beam_size: int = 5
    max_batch_size: int = 1
    max_batch_wait_ms: float = 0.0

    def __post_init__(self):
        if not str(self.model).strip():
            raise ValueError("Whisper model cannot be empty")
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if self.cpu_threads < 0:
            raise ValueError("cpu_threads cannot be negative")
        if self.beam_size < 1:
            raise ValueError("beam_size must be at least 1")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if self.max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms cannot be negative")
        if self.max_batch_size > 1 and self.num_workers != 1:
            raise ValueError("a batched engine runs on exactly one model worker")


@dataclass
class RequestContext:
    """State of one in-flight request, shared with the thread running it.

    ``cancelled`` is set when the caller stops waiting, so a batched engine
    can drop the request's remaining decode calls. The batch fields are
    filled in by the batch scheduler.
    """

    cancelled: threading.Event = field(default_factory=threading.Event)
    batch_wait_secs: float = 0.0
    batch_size: int = 1


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
        beam_size: int = 5,
    ):
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if beam_size < 1:
            raise ValueError("beam_size must be at least 1")

        self._model = model
        self._model_name = str(model_name)
        self._num_workers = num_workers
        self._beam_size = beam_size
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="whisper-engine",
        )
        self._stats_lock = threading.Lock()
        self._queued = 0
        self._running = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._closed = False

    @classmethod
    async def load(cls, config: WhisperEngineConfig) -> "SharedWhisperEngine":
        """Return the process-wide engine for ``config``, loading it on first use."""
        return await load_shared_model(
            ("pipecat-local-agent", cls.__name__, config),
            lambda: cls.from_config(config),
        )

    @classmethod
    def from_config(cls, config: WhisperEngineConfig) -> "SharedWhisperEngine":
        """Load a new, unshared engine. Blocks while the model loads."""
        logger.info(
            "Loading Whisper model {} for {} (device={}, compute_type={}, {})",
            config.model,
            cls.__name__,
            config.device,
            config.compute_type,
            cls._describe(config),
        )
        model = WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
            cpu_threads=config.cpu_threads,
            num_workers=config.num_workers,
        )
        logger.info("Loaded Whisper model {}", config.model)
        return cls._from_model(model, config)

    @classmethod
    def _describe(cls, config: WhisperEngineConfig) -> str:
        return f"workers={config.num_workers}, beam_size={config.beam_size}"

    @classmethod
    def _from_model(cls, model: Any, config: WhisperEngineConfig) -> "SharedWhisperEngine":
        return cls(
            model,
            model_name=config.model,
            num_workers=config.num_workers,
            beam_size=config.beam_size,
        )

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
    def beam_size(self) -> int:
        return self._beam_size

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

    def stats(self) -> WhisperEngineStats:
        with self._stats_lock:
            return WhisperEngineStats(
                workers=self._num_workers,
                queued=self._queued,
                running=self._running,
                completed=self._completed,
                failed=self._failed,
                cancelled=self._cancelled,
            )

    async def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
        hotwords: str | None = None,
        initial_prompt: str | None = None,
    ) -> WhisperTranscription:
        """Transcribe mono float32 audio at 16 kHz on a worker thread.

        Raises:
            RuntimeError: If the engine has been closed.
        """
        if self._closed:
            raise RuntimeError("Whisper engine is closed")

        context = RequestContext()
        with self._stats_lock:
            self._queued += 1
        try:
            future = self._executor.submit(
                self._transcribe_blocking,
                audio,
                language,
                hotwords,
                initial_prompt,
                time.perf_counter(),
                context,
            )
        except RuntimeError:
            # The executor was shut down between the closed check and submit.
            with self._stats_lock:
                self._queued -= 1
            raise RuntimeError("Whisper engine is closed") from None
        future.add_done_callback(self._forget_if_cancelled)

        # Cancelling the awaiting task cancels ``future`` too; a request that
        # has not started yet is then removed from the queue. One that has
        # started is told to stop through its context.
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            context.cancelled.set()
            raise

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
                self._cancelled += 1

    def _decode(
        self,
        audio: np.ndarray,
        language: str | None,
        hotwords: str | None,
        initial_prompt: str | None,
        context: RequestContext,
    ) -> list[Segment]:
        # Only per-request arguments are passed. ``chunk_length`` must never
        # be passed: faster-whisper stores it on the shared feature
        # extractor, which would change it for every other session.
        segments, _info = self._model.transcribe(
            audio,
            language=language,
            hotwords=hotwords,
            initial_prompt=initial_prompt,
            beam_size=self._beam_size,
        )
        # ``segments`` is lazy and decodes while it is consumed, so consume
        # it here on the worker thread, not on the event loop.
        return list(segments)

    def _transcribe_blocking(
        self,
        audio: np.ndarray,
        language: str | None,
        hotwords: str | None,
        initial_prompt: str | None,
        submitted_at: float,
        context: RequestContext,
    ) -> WhisperTranscription:
        started_at = time.perf_counter()
        with self._stats_lock:
            self._queued -= 1
            self._running += 1
        succeeded = False
        try:
            if self._closed:
                raise RuntimeError("Whisper engine is closed")
            segments = self._decode(audio, language, hotwords, initial_prompt, context)
            result = WhisperTranscription(
                segments=segments,
                queue_wait_secs=started_at - submitted_at,
                inference_secs=time.perf_counter() - started_at,
                batch_wait_secs=context.batch_wait_secs,
                batch_size=context.batch_size,
            )
            succeeded = True
            return result
        finally:
            with self._stats_lock:
                self._running -= 1
                if succeeded:
                    self._completed += 1
                elif context.cancelled.is_set():
                    self._cancelled += 1
                else:
                    self._failed += 1


async def load_whisper_engine(config: WhisperEngineConfig) -> SharedWhisperEngine:
    """The process-wide engine for ``config``: batched when ``max_batch_size > 1``."""
    if config.max_batch_size > 1:
        from .batching import BatchedWhisperEngine

        return await BatchedWhisperEngine.load(config)
    return await SharedWhisperEngine.load(config)
