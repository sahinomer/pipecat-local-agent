"""Benchmark STT services: Whisper plus per-segment reports for the client."""

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from pipecat.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    Frame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.whisper.stt import WhisperSTTService

from services.whisper import SharedWhisperSTTService, WhisperEngine

from . import protocol

BYTES_PER_SECOND = protocol.SAMPLE_RATE * 2


@dataclass
class BenchState:
    """Server-wide counters shown by ``/stats``.

    Only touched from the event loop, so it needs no lock.
    """

    engine_mode: str
    model: str
    language: str
    beam_size: int = 5
    engine: WhisperEngine | None = None
    sessions_active: int = 0
    sessions_total: int = 0
    models_loaded: int = 0
    segments_in_flight: int = 0
    segments_completed: int = 0
    segments_failed: int = 0
    session_setup_secs: list[float] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        engine: dict[str, Any] = {
            "mode": self.engine_mode,
            "model": self.model,
            "language": self.language,
            "beam_size": self.beam_size,
        }
        if self.engine is not None:
            stats = self.engine.stats()
            engine.update(
                workers=stats.workers,
                queued=stats.queued,
                running=stats.running,
                completed=stats.completed,
                failed=stats.failed,
                cancelled=stats.cancelled,
                max_batch_size=stats.max_batch_size,
                batch_pending=stats.batch_pending,
                batches=stats.batches,
                batched_items=stats.batched_items,
                last_batch_size=stats.last_batch_size,
            )
        else:
            # Each session owns a model; there is no shared queue.
            engine.update(models_loaded=self.models_loaded)
        return {
            "engine": engine,
            "sessions": {"active": self.sessions_active, "total": self.sessions_total},
            "segments": {
                "in_flight": self.segments_in_flight,
                "completed": self.segments_completed,
                "failed": self.segments_failed,
            },
        }


class SegmentReportingMixin:
    """Report VAD events and every transcribed segment to the client.

    Mixed into a Pipecat ``SegmentedSTTService`` subclass. It relies on the
    base class numbering segments as VAD starts them (``_segment_count``)
    and transcribing them one at a time in that order
    (``_transcribing_segment``).
    """

    def __init__(self, *, bench_state: BenchState, **kwargs):
        self._bench_state = bench_state
        self._bench_stopped_at: dict[int, float] = {}
        self._bench_audio_bytes = 0
        self._bench_positions: dict[int, list[float | None]] = {}
        self._bench_flush_bytes: int | None = None
        self._bench_audio_done = False
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputTransportMessageFrame):
            message = frame.message if isinstance(frame.message, dict) else {}
            if message.get("type") == protocol.FLUSH:
                self._bench_flush_bytes = int(message.get("audio_bytes") or 0)
                await self._bench_check_audio_done()

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        self._bench_audio_bytes += len(frame.audio)
        await super().process_audio_frame(frame, direction)
        await self._bench_check_audio_done()

    async def _bench_check_audio_done(self) -> None:
        # The flush message can overtake queued audio, so wait for both.
        if (
            self._bench_audio_done
            or self._bench_flush_bytes is None
            or self._bench_audio_bytes < self._bench_flush_bytes
        ):
            return
        self._bench_audio_done = True
        await self._send_bench_message(
            {
                "type": protocol.AUDIO_DONE,
                "audio_secs": self._bench_audio_bytes / BYTES_PER_SECOND,
                "segments": self._segment_count,
            }
        )

    def _bench_position(self, vad_delay_secs: float) -> float:
        """Where in the session's stream the speech change happened.

        VAD confirms a start or stop ``vad_delay_secs`` after it happens, and
        the STT stage has received the stream up to the confirming frame.
        """
        return max(0.0, self._bench_audio_bytes / BYTES_PER_SECOND - vad_delay_secs)

    async def _handle_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        await super()._handle_user_started_speaking(frame)
        self._bench_positions[self._segment_count] = [
            self._bench_position(frame.start_secs),
            None,
        ]
        await self._send_bench_message(
            {"type": protocol.VAD, "event": "started", "segment": self._segment_count}
        )

    async def _handle_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        segment = self._segment_count
        self._bench_stopped_at[segment] = time.perf_counter()
        self._bench_positions.setdefault(segment, [None, None])[1] = self._bench_position(
            frame.stop_secs
        )
        await super()._handle_user_stopped_speaking(frame)
        await self._send_bench_message(
            {"type": protocol.VAD, "event": "stopped", "segment": segment}
        )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        segment = self._transcribing_segment
        stopped_at = self._bench_stopped_at.pop(segment, None)
        start_position, end_position = self._bench_positions.pop(segment, [None, None])
        started_at = time.perf_counter()
        texts: list[str] = []
        error: str | None = None
        state = self._bench_state
        state.segments_in_flight += 1
        try:
            try:
                async for frame in super().run_stt(audio):
                    if isinstance(frame, TranscriptionFrame):
                        texts.append(frame.text.strip())
                    elif isinstance(frame, ErrorFrame):
                        error = frame.error
                    yield frame
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                yield ErrorFrame(error=f"Whisper transcription failed: {error}", exception=exc)
        finally:
            state.segments_in_flight -= 1

        if error is None:
            state.segments_completed += 1
        else:
            state.segments_failed += 1

        finished_at = time.perf_counter()
        stt_secs = finished_at - started_at
        queue_wait_secs, inference_secs, batch_wait_secs, batch_size = self._bench_timings(
            stt_secs
        )
        yield self._bench_frame(
            {
                "type": protocol.SEGMENT,
                "segment": segment,
                "text": " ".join(t for t in texts if t),
                "audio_secs": len(audio) / BYTES_PER_SECOND,
                "latency_secs": None if stopped_at is None else finished_at - stopped_at,
                "stt_secs": stt_secs,
                "queue_wait_secs": queue_wait_secs,
                "inference_secs": inference_secs,
                "batch_wait_secs": batch_wait_secs,
                "batch_size": batch_size,
                "start_audio_secs": start_position,
                "end_audio_secs": end_position,
                "error": error,
            }
        )

    def _bench_timings(
        self, stt_secs: float
    ) -> tuple[float | None, float | None, float | None, int | None]:
        """Queue wait, inference, batch wait and batch size of the last segment."""
        return None, stt_secs, None, None

    def _bench_frame(self, message: dict[str, Any]) -> OutputTransportMessageUrgentFrame:
        return OutputTransportMessageUrgentFrame(message=message)

    async def _send_bench_message(self, message: dict[str, Any]) -> None:
        await self.push_frame(self._bench_frame(message))


class BenchSharedWhisperSTTService(SegmentReportingMixin, SharedWhisperSTTService):
    """The shared or batched engine service, reporting where each segment's time went."""

    def _bench_timings(self, stt_secs: float):
        result = self.last_transcription
        if result is None:
            return None, None, None, None
        return (
            result.queue_wait_secs,
            result.inference_secs,
            result.batch_wait_secs,
            result.batch_size,
        )


class BenchPerSessionWhisperSTTService(SegmentReportingMixin, WhisperSTTService):
    """Pipecat's stock service: one model per connection, loaded on creation.

    There is no queue, so the whole transcription counts as inference time.
    """
