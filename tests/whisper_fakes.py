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

    def transcribe(self, audio, *, language=None, hotwords=None, initial_prompt=None, beam_size=5):
        marker = audio_marker(audio)
        with self._lock:
            self.calls.append(
                {
                    "marker": marker,
                    "language": language,
                    "hotwords": hotwords,
                    "initial_prompt": initial_prompt,
                    "beam_size": beam_size,
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


class FakeCT2Whisper:
    """Stands in for the CTranslate2 Whisper model under faster-whisper.

    ``generate`` reads each row's marker back from its features and its
    language from its prompt, so a result that lands on the wrong request
    is visible. Every call is recorded with its batch size and options.

    Args:
        gate: When given, the first ``generate`` waits until it is set, so
            later requests pile up behind it.
        delay: Seconds each ``generate`` takes.
        poison_markers: Markers whose presence in a batch makes it fail.
    """

    def __init__(
        self,
        *,
        gate: threading.Event | None = None,
        delay: float = 0.0,
        poison_markers: frozenset[int] = frozenset(),
    ):
        self.gate = gate
        self.delay = delay
        self.poison_markers = poison_markers
        self.encode_batches: list[int] = []
        self.generate_calls: list[dict] = []
        self.is_multilingual = True
        self.device = "cpu"
        self.device_index = [0]
        self._lock = threading.Lock()
        self._gate_used = False

    def encode(self, features, to_cpu=False):
        array = np.asarray(features)
        with self._lock:
            self.encode_batches.append(array.shape[0])
        return SimpleNamespace(markers=[int(round(float(row[0, 0]))) for row in array])

    def generate(self, encoder_output, prompts, **kwargs):
        with self._lock:
            first = not self._gate_used
            self._gate_used = True
        if first and self.gate is not None and not self.gate.wait(timeout=5):
            raise TimeoutError("test gate was never opened")
        if self.delay:
            time.sleep(self.delay)
        markers = encoder_output.markers
        with self._lock:
            self.generate_calls.append(
                {"rows": len(prompts), "markers": list(markers), "kwargs": dict(kwargs)}
            )
        if self.poison_markers.intersection(markers):
            raise RuntimeError(f"batch with poison marker {sorted(self.poison_markers)}")
        return [
            SimpleNamespace(
                sequences_ids=[[prompt[0], marker]],
                scores=[-0.1],
                no_speech_prob=0.0,
            )
            for prompt, marker in zip(prompts, markers)
        ]


class FakeFasterWhisper:
    """Stands in for ``faster_whisper.WhisperModel`` on top of a CTranslate2 fake.

    Like the real one, ``transcribe`` returns a lazy generator that calls
    ``self.encode`` and then ``self.model.generate`` once per decode, so a
    batched engine can substitute both. Requests whose marker is in
    ``fallback_markers`` decode a second time with sampling options, the way
    faster-whisper retries at a higher temperature.
    """

    def __init__(self, ct2: FakeCT2Whisper, *, fallback_markers: frozenset[int] = frozenset()):
        self.model = ct2
        self.fallback_markers = fallback_markers
        self.supported_languages = ["en", "tr"]

    def encode(self, features):
        import ctranslate2

        return self.model.encode(ctranslate2.StorageView.from_array(features[None]))

    def transcribe(self, audio, *, language=None, hotwords=None, initial_prompt=None, beam_size=5):
        marker = audio_marker(audio)
        features = np.full((4, 8), marker, dtype=np.float32)
        prompt = [language, hotwords, initial_prompt]

        def decode():
            encoder_output = self.encode(features)
            result = self.model.generate(
                encoder_output, [prompt], beam_size=beam_size, return_scores=True
            )[0]
            if marker in self.fallback_markers:
                result = self.model.generate(
                    encoder_output,
                    [prompt],
                    beam_size=1,
                    sampling_temperature=0.2,
                    return_scores=True,
                )[0]
            decoded_language, decoded_marker = result.sequences_ids[0]
            yield SimpleNamespace(
                text=f"{decoded_language}:{decoded_marker}",
                no_speech_prob=result.no_speech_prob,
            )

        return decode(), None
