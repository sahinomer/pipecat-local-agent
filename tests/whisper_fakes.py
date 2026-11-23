"""Test doubles and helpers for the shared Whisper engine tests."""

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

SAMPLE_RATE = 16000
CHUNK_SAMPLES = SAMPLE_RATE // 50  # 20 ms, like a transport audio frame


def audio_marker(audio: np.ndarray) -> int:
    """The constant sample value a fixture segment was built from.

    Fixtures fill speech with one int16 value per session; padding and
    pre-roll are silence, so the most common non-zero value identifies whose
    audio a request carried.
    """
    samples = np.round(np.asarray(audio, dtype=np.float64) * 32768.0).astype(np.int64)
    non_zero = samples[samples != 0]
    if non_zero.size == 0:
        return 0
    values, counts = np.unique(non_zero, return_counts=True)
    return int(values[np.argmax(counts)])


def marker_audio(marker: int, *, samples: int = SAMPLE_RATE // 2) -> np.ndarray:
    """Float32 audio at 16 kHz whose every sample is ``marker`` as int16."""
    return np.full(samples, marker / 32768.0, dtype=np.float32)


def speech_frames(marker: int, *, chunks: int = 10) -> list[Frame]:
    """One VAD-bounded utterance of constant-valued 16 kHz PCM."""
    pcm = np.full(CHUNK_SAMPLES, marker, dtype="<i2").tobytes()
    return [
        VADUserStartedSpeakingFrame(),
        *(
            InputAudioRawFrame(audio=pcm, sample_rate=SAMPLE_RATE, num_channels=1)
            for _ in range(chunks)
        ),
        VADUserStoppedSpeakingFrame(),
    ]


class FakeWhisperModel:
    """Stands in for ``faster_whisper.WhisperModel``.

    Like the real model, ``transcribe`` returns a lazy generator and the work
    happens while it is consumed. Each segment's text is
    ``"<language>:<marker>"``, so a result that carries another session's
    audio or language is visible in the text.

    Args:
        gate: When given, decoding waits until it is set.
        delay: Seconds each decode takes, to make requests overlap.
        fail_markers: Markers whose decode raises.
        supported_languages: Reported like faster-whisper does.
    """

    def __init__(
        self,
        *,
        gate: threading.Event | None = None,
        delay: float = 0.0,
        fail_markers: frozenset[int] = frozenset(),
        supported_languages: list[str] | None = None,
    ):
        self.gate = gate
        self.delay = delay
        self.fail_markers = fail_markers
        self.supported_languages = supported_languages or ["en", "tr"]
        self.calls: list[dict] = []
        self.decoded_markers: list[int] = []
        self.decode_threads: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def transcribe(self, audio, *, language=None, hotwords=None, initial_prompt=None):
        marker = audio_marker(audio)
        with self._lock:
            self.calls.append(
                {
                    "marker": marker,
                    "language": language,
                    "hotwords": hotwords,
                    "initial_prompt": initial_prompt,
                    "dtype": audio.dtype,
                }
            )
        return self._decode(marker, language), None

    def _decode(self, marker, language):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.decode_threads.append(threading.current_thread().name)
        try:
            if self.gate is not None and not self.gate.wait(timeout=5):
                raise TimeoutError("test gate was never opened")
            if self.delay:
                time.sleep(self.delay)
            if marker in self.fail_markers:
                raise RuntimeError(f"decode failed for marker {marker}")
            with self._lock:
                self.decoded_markers.append(marker)
            yield SimpleNamespace(text=f"{language}:{marker}", no_speech_prob=0.0)
        finally:
            with self._lock:
                self.active -= 1


async def wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.005):
    """Poll ``predicate`` from the event loop until it is true."""

    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met before timeout")
        await asyncio.sleep(interval)
