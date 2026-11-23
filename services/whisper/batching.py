"""Dynamic batching for the shared Whisper engine.

Each request still runs faster-whisper's own ``transcribe`` on a thread, so
its decoding rules stay exactly as they are: temperature fallback, the
no-speech check, continuing into a new window when the model stops early,
hotwords and the initial prompt. Only the GPU work is batched. The engine's
model gets two substitutions:

- ``WhisperModel.encode`` returns a :class:`DeferredFeatures` handle that
  just holds the mel features.
- ``WhisperModel.model``, the CTranslate2 model, becomes a
  :class:`BatchingModel`. Its ``generate`` hands deferred calls to a
  :class:`BatchScheduler` and blocks until the result is back.

The scheduler runs on one thread. Whenever the GPU is free it takes the
oldest waiting call, adds up to ``max_batch_size - 1`` more calls with the
same decoding options in arrival order, encodes their features in one call
and generates for all of them in one call. Every call gets its own prompt
and its own result back, so requests from different sessions share GPU
work but never state.
"""

import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from typing import Any

import ctranslate2
import numpy as np
from loguru import logger

from .engine import RequestContext, SharedWhisperEngine, WhisperEngineConfig, WhisperEngineStats

_request = threading.local()


class RequestCancelled(Exception):
    """Raised in a request's thread when its caller stopped waiting."""


@dataclass(frozen=True)
class DeferredFeatures:
    """Mel features whose encoding is left to the batch scheduler.

    Parameters:
        features: ``[batch, n_mels, frames]`` float32 features.
    """

    features: np.ndarray


def _freeze(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


@dataclass
class _DecodeCall:
    features: np.ndarray
    prompts: list
    kwargs: dict[str, Any]
    key: tuple
    context: RequestContext | None
    enqueued_at: float = field(default_factory=time.perf_counter)
    future: Future = field(default_factory=Future)

    @property
    def rows(self) -> int:
        return len(self.prompts)


class BatchScheduler:
    """Run decode calls from many threads as GPU batches.

    Calls are grouped only with calls that have identical decoding options
    and feature shapes, because one CTranslate2 ``generate`` applies one set
    of options to the whole batch. A temperature-fallback retry therefore
    runs with other retries, or alone.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int,
        max_wait_secs: float = 0.0,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        self._model = model
        self._max_batch_size = max_batch_size
        self._max_wait_secs = max_wait_secs
        self._pending: deque[_DecodeCall] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._batches = 0
        self._batched_items = 0
        self._last_batch_size = 0
        self._thread = threading.Thread(
            target=self._run, name="whisper-batch-scheduler", daemon=True
        )
        self._thread.start()

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "batch_pending": len(self._pending),
                "batches": self._batches,
                "batched_items": self._batched_items,
                "last_batch_size": self._last_batch_size,
            }

    def submit(self, features: np.ndarray, prompts: list, kwargs: dict[str, Any]) -> list:
        """Queue one decode call and block until its results are ready.

        Raises:
            RequestCancelled: If the calling request was cancelled.
            RuntimeError: If the scheduler is closed.
        """
        context: RequestContext | None = getattr(_request, "context", None)
        if context is not None and context.cancelled.is_set():
            raise RequestCancelled()
        if features.shape[0] != len(prompts):
            raise ValueError("features and prompts must have the same batch size")
        call = _DecodeCall(
            features=features,
            prompts=list(prompts),
            kwargs=dict(kwargs),
            key=(features.shape[1:], features.dtype.str, _freeze(kwargs)),
            context=context,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("Whisper engine is closed")
            self._pending.append(call)
            self._condition.notify()
        return call.future.result()

    def close(self) -> None:
        """Stop the scheduler and fail every call still waiting."""
        with self._condition:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            self._condition.notify_all()
        for call in pending:
            if not call.future.done():
                call.future.set_exception(RuntimeError("Whisper engine is closed"))

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                self._wait_for_more()
                batch = self._take_batch()
            if batch:
                self._execute(batch)

    def _wait_for_more(self) -> None:
        """Optionally hold a short batch back until it fills or times out."""
        if self._max_wait_secs <= 0:
            return
        deadline = self._pending[0].enqueued_at + self._max_wait_secs
        while not self._closed:
            key = self._pending[0].key
            rows = sum(c.rows for c in self._pending if c.key == key)
            remaining = deadline - time.perf_counter()
            if rows >= self._max_batch_size or remaining <= 0:
                return
            self._condition.wait(remaining)

    def _take_batch(self) -> list[_DecodeCall]:
        """The oldest call plus compatible calls, in arrival order."""
        batch: list[_DecodeCall] = []
        rows = 0
        key = None
        kept: deque[_DecodeCall] = deque()
        while self._pending:
            call = self._pending.popleft()
            if call.context is not None and call.context.cancelled.is_set():
                call.future.set_exception(RequestCancelled())
                continue
            if key is None:
                key = call.key
            if call.key == key and (not batch or rows + call.rows <= self._max_batch_size):
                batch.append(call)
                rows += call.rows
            else:
                kept.append(call)
        self._pending = kept
        return batch

    def _execute(self, batch: list[_DecodeCall]) -> None:
        started_at = time.perf_counter()
        size = sum(call.rows for call in batch)
        for call in batch:
            if call.context is not None:
                call.context.batch_wait_secs += started_at - call.enqueued_at
                call.context.batch_size = max(call.context.batch_size, size)
        try:
            results = self._generate(batch)
        except Exception as exc:
            if len(batch) == 1:
                batch[0].future.set_exception(exc)
            else:
                # Find the call that broke the batch: run each one alone, so
                # a bad request can't fail everyone it was batched with.
                logger.warning(
                    "Whisper batch of {} failed ({}); retrying its calls one by one",
                    size,
                    exc,
                )
                for call in batch:
                    self._execute([call])
            return

        offset = 0
        for call in batch:
            call.future.set_result(results[offset : offset + call.rows])
            offset += call.rows
        with self._condition:
            self._batches += 1
            self._batched_items += size
            self._last_batch_size = size

    def _generate(self, batch: list[_DecodeCall]) -> list:
        features = np.concatenate([call.features for call in batch])
        prompts = [prompt for call in batch for prompt in call.prompts]
        encoder_output = self._model.encode(
            ctranslate2.StorageView.from_array(np.ascontiguousarray(features)),
            to_cpu=False,
        )
        return list(self._model.generate(encoder_output, prompts, **batch[0].kwargs))


class BatchingModel:
    """Stands in for faster-whisper's CTranslate2 model.

    ``generate`` on deferred features goes through the scheduler. Calls that
    need the encoder output itself (language detection, word alignment)
    encode the deferred features on the spot. Everything else is passed to
    the real model.
    """

    def __init__(self, model: Any, scheduler: BatchScheduler):
        self._model = model
        self._scheduler = scheduler

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def generate(self, encoder_output: Any, prompts: list, **kwargs) -> list:
        if isinstance(encoder_output, DeferredFeatures):
            return self._scheduler.submit(encoder_output.features, prompts, kwargs)
        return self._model.generate(encoder_output, prompts, **kwargs)

    def detect_language(self, encoder_output: Any, *args, **kwargs):
        return self._model.detect_language(self._materialize(encoder_output), *args, **kwargs)

    def align(self, encoder_output: Any, *args, **kwargs):
        return self._model.align(self._materialize(encoder_output), *args, **kwargs)

    def _materialize(self, encoder_output: Any) -> Any:
        if not isinstance(encoder_output, DeferredFeatures):
            return encoder_output
        return self._model.encode(
            ctranslate2.StorageView.from_array(np.ascontiguousarray(encoder_output.features)),
            to_cpu=False,
        )


def _deferred_encode(features: np.ndarray) -> DeferredFeatures:
    """Replacement for ``WhisperModel.encode``: keep the features for batching."""
    if features.ndim == 2:
        features = np.expand_dims(features, 0)
    return DeferredFeatures(np.ascontiguousarray(features, dtype=np.float32))


class BatchedWhisperEngine(SharedWhisperEngine):
    """A shared engine that batches decode calls across sessions.

    Up to ``request_threads`` requests run faster-whisper at once; their
    decode calls meet in the scheduler and run on the GPU in batches of up
    to ``max_batch_size``. The default of two threads per batch slot lets the
    next batch gather while one runs.

    The engine takes over ``model``: it replaces ``model.encode`` and
    ``model.model``, so the model must not be used by anything else.
    """

    def __init__(
        self,
        model: Any,
        *,
        model_name: str,
        max_batch_size: int = 8,
        max_batch_wait_ms: float = 0.0,
        beam_size: int = 5,
        request_threads: int | None = None,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms cannot be negative")
        super().__init__(
            model,
            model_name=model_name,
            num_workers=request_threads or 2 * max_batch_size,
            beam_size=beam_size,
        )
        self._scheduler = BatchScheduler(
            model.model,
            max_batch_size=max_batch_size,
            max_wait_secs=max_batch_wait_ms / 1000,
        )
        model.model = BatchingModel(model.model, self._scheduler)
        model.encode = _deferred_encode

    @classmethod
    def from_config(cls, config: WhisperEngineConfig) -> "BatchedWhisperEngine":
        if config.num_workers != 1:
            raise ValueError("a batched engine runs on exactly one model worker")
        return super().from_config(config)

    @classmethod
    def _describe(cls, config: WhisperEngineConfig) -> str:
        return (
            f"max_batch_size={config.max_batch_size}, "
            f"max_batch_wait_ms={config.max_batch_wait_ms:g}, beam_size={config.beam_size}"
        )

    @classmethod
    def _from_model(cls, model: Any, config: WhisperEngineConfig) -> "BatchedWhisperEngine":
        return cls(
            model,
            model_name=config.model,
            max_batch_size=config.max_batch_size,
            max_batch_wait_ms=config.max_batch_wait_ms,
            beam_size=config.beam_size,
        )

    @property
    def max_batch_size(self) -> int:
        return self._scheduler.max_batch_size

    def stats(self) -> WhisperEngineStats:
        return replace(
            super().stats(),
            max_batch_size=self._scheduler.max_batch_size,
            **self._scheduler.stats(),
        )

    def close(self) -> None:
        super().close()
        self._scheduler.close()

    def _decode(self, audio, language, hotwords, initial_prompt, context):
        _request.context = context
        try:
            return super()._decode(audio, language, hotwords, initial_prompt, context)
        finally:
            _request.context = None


__all__: list[str] = [
    "BatchScheduler",
    "BatchedWhisperEngine",
    "BatchingModel",
    "DeferredFeatures",
    "RequestCancelled",
]
