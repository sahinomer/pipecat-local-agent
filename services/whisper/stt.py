"""Pipecat Whisper STT service backed by a process-wide shared engine."""

from collections.abc import AsyncGenerator
from typing import Any, cast

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import NOT_GIVEN, assert_given, is_given

from .engine import WhisperEngine, WhisperTranscription


class SharedWhisperSTTService(WhisperSTTService):
    """Whisper STT that transcribes on a shared engine instead of its own model.

    Pipecat's ``WhisperSTTService`` loads a model for every instance, so every
    pipeline holds its own copy. This subclass keeps the upstream segmenting,
    settings, filtering and frame behavior, but sends each segment to an
    injected :class:`WhisperEngine` that every session shares.

    All per-session state stays on this instance (settings, audio buffer,
    user id, metrics) and is passed to the engine with each request. The
    engine's model is fixed: ``settings.model`` must match it, and a runtime
    model change is reported as an error and ignored.
    """

    def __init__(
        self,
        *,
        engine: WhisperEngine,
        settings: WhisperSTTService.Settings | None = None,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            engine: The shared engine that runs transcriptions.
            settings: Runtime-updatable settings. ``model`` may be omitted; it
                defaults to the engine's model and must match it if given.
            **kwargs: Additional arguments passed to ``SegmentedSTTService``.

        Raises:
            ValueError: If ``settings.model`` differs from the engine's model,
                or the model cannot transcribe the configured language.
        """
        # Set before the base initializer, which calls ``_load``.
        self._engine = engine
        self._last_transcription: WhisperTranscription | None = None

        effective_settings = self.Settings(model=engine.model_name)
        if settings is not None:
            if is_given(settings.model) and str(settings.model) != engine.model_name:
                raise ValueError(
                    f"settings.model {str(settings.model)!r} does not match the "
                    f"shared Whisper engine model {engine.model_name!r}"
                )
            effective_settings.apply_update(settings)

        super().__init__(settings=effective_settings, **kwargs)

    @property
    def engine(self) -> WhisperEngine:
        return self._engine

    @property
    def last_transcription(self) -> WhisperTranscription | None:
        """The engine result of the most recent segment, or ``None`` if it failed.

        Segments are transcribed one at a time per service, so this always
        belongs to the segment whose frames were pushed last.
        """
        return self._last_transcription

    def _load(self):
        """Use the shared engine instead of loading a model for this instance."""
        self._model = None
        unsupported = self._unsupported_language()
        if unsupported:
            raise ValueError(unsupported)

    def _unsupported_language(self) -> str | None:
        """Describe why the engine's model can't transcribe the current language."""
        supported = self._engine.supported_languages
        language = self._settings.language
        if not supported or not is_given(language) or language is None:
            return None
        if language in supported:
            return None
        return (
            f"Whisper model '{self._engine.model_name}' cannot transcribe "
            f"'{language}' — it supports only {sorted(supported)}. Use a "
            "multilingual model (e.g. 'large-v3-turbo')."
        )

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, refusing to switch away from the engine's model."""
        if is_given(delta.model) and str(delta.model) != self._engine.model_name:
            await self.push_error(
                f"{self}: cannot switch Whisper model to {str(delta.model)!r}; "
                f"the shared engine serves {self._engine.model_name!r}"
            )
            delta.model = NOT_GIVEN
        return await super()._update_settings(delta)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe one 16 kHz 16-bit PCM segment on the shared engine.

        Args:
            audio: Raw 16-bit PCM audio at 16 kHz.

        Yields:
            Frame: A TranscriptionFrame with the transcribed text, or an
                ErrorFrame if transcription fails.
        """
        # Read settings once, so a settings change mid-request can't mix values.
        language = cast("Language | None", assert_given(self._settings.language))
        hotwords = assert_given(self._settings.hotwords)
        initial_prompt = assert_given(self._settings.initial_prompt)
        no_speech_prob_threshold = assert_given(self._settings.no_speech_prob)

        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        result: WhisperTranscription | None = None
        error: Exception | None = None
        self._last_transcription = None
        await self.start_processing_metrics()
        try:
            result = await self._engine.transcribe(
                audio_float,
                language=language,
                hotwords=hotwords,
                initial_prompt=initial_prompt,
            )
        except Exception as exc:
            error = exc
        finally:
            await self.stop_processing_metrics()

        if error is not None or result is None:
            yield ErrorFrame(error=f"Whisper transcription failed: {error}", exception=error)
            return
        self._last_transcription = result

        text = ""
        for segment in result.segments:
            if (
                no_speech_prob_threshold is not None
                and segment.no_speech_prob < no_speech_prob_threshold
            ):
                text += f"{segment.text} "

        if text:
            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result={
                    "queue_wait_secs": result.queue_wait_secs,
                    "inference_secs": result.inference_secs,
                },
            )
