"""Tests for the dynamically batched Whisper engine.

The fakes sit one layer below faster-whisper: ``FakeFasterWhisper`` calls
``encode`` and ``model.generate`` the way the real ``WhisperModel`` does, and
``FakeCT2Whisper`` records the batches the GPU would run. Each result
carries the language from its own prompt and the marker from its own
features, so a result delivered to the wrong request shows up in the text.
"""

import asyncio
import os
import threading
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from services.whisper import (
    BatchedWhisperEngine,
    SharedWhisperEngine,
    SharedWhisperSTTService,
    WhisperEngineConfig,
    load_whisper_engine,
)
from services.whisper.batching import BatchingModel, DeferredFeatures
from whisper_fakes import (
    FakeCT2Whisper,
    FakeFasterWhisper,
    marker_audio,
    speech_frames,
    wait_until,
)


def _engine(ct2: FakeCT2Whisper, *, max_batch_size=4, fallback_markers=frozenset(), **kwargs):
    model = FakeFasterWhisper(ct2, fallback_markers=fallback_markers)
    return BatchedWhisperEngine(model, model_name="fake", max_batch_size=max_batch_size, **kwargs)


def _texts(results) -> list[str]:
    return [r.segments[0].text for r in results]


class BatchedEngineConfigTests(unittest.TestCase):
    def test_batching_needs_one_model_worker(self):
        with self.assertRaisesRegex(ValueError, "exactly one model worker"):
            WhisperEngineConfig(model="small", max_batch_size=4, num_workers=2)
        for kwargs in ({"beam_size": 0}, {"max_batch_size": 0}, {"max_batch_wait_ms": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                WhisperEngineConfig(model="small", **kwargs)

    def test_the_engine_takes_over_encode_and_the_ctranslate2_model(self):
        ct2 = FakeCT2Whisper()
        engine = _engine(ct2)
        self.addCleanup(engine.close)
        model = engine._model

        self.assertIsInstance(model.model, BatchingModel)
        self.assertIsInstance(model.encode(np.zeros((4, 8), np.float32)), DeferredFeatures)
        # Attributes faster-whisper reads still come from the real model.
        self.assertTrue(model.model.is_multilingual)


class BatchedWhisperEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_lone_request_runs_at_once_in_a_batch_of_one(self):
        ct2 = FakeCT2Whisper()
        engine = _engine(ct2)
        self.addCleanup(engine.close)

        result = await engine.transcribe(marker_audio(7), language="tr", hotwords="hw")

        self.assertEqual(result.segments[0].text, "tr:7")
        self.assertEqual(result.batch_size, 1)
        self.assertLess(result.batch_wait_secs, 0.5)
        self.assertEqual(ct2.encode_batches, [1])
        self.assertEqual(ct2.generate_calls[0]["kwargs"]["beam_size"], 5)

    async def test_requests_that_arrive_while_the_gpu_is_busy_share_batches(self):
        gate = threading.Event()
        ct2 = FakeCT2Whisper(gate=gate)
        # Enough request threads for all nine to wait in the scheduler at once.
        engine = _engine(ct2, max_batch_size=4, request_threads=9)
        self.addCleanup(engine.close)

        languages = ["en", "tr"]
        first = asyncio.create_task(engine.transcribe(marker_audio(1), language="en"))
        await wait_until(lambda: engine.stats().running == 1)
        rest = [
            asyncio.create_task(engine.transcribe(marker_audio(m), language=languages[m % 2]))
            for m in range(2, 10)
        ]
        await wait_until(lambda: engine.stats().batch_pending == 8)
        gate.set()
        results = [await first, *await asyncio.gather(*rest)]

        self.assertEqual(
            _texts(results),
            ["en:1"] + [f"{languages[m % 2]}:{m}" for m in range(2, 10)],
        )
        self.assertEqual([c["rows"] for c in ct2.generate_calls], [1, 4, 4])
        self.assertEqual(ct2.encode_batches, [1, 4, 4])
        self.assertEqual([r.batch_size for r in results], [1] + [4] * 8)
        self.assertTrue(all(r.batch_wait_secs > 0 for r in results[1:]))
        stats = engine.stats()
        self.assertEqual((stats.batches, stats.batched_items, stats.max_batch_size), (3, 9, 4))

    async def test_calls_with_different_decoding_options_never_share_a_batch(self):
        gate = threading.Event()
        ct2 = FakeCT2Whisper(gate=gate)
        engine = _engine(ct2, max_batch_size=8, fallback_markers=frozenset({3, 5}))
        self.addCleanup(engine.close)

        first = asyncio.create_task(engine.transcribe(marker_audio(1), language="en"))
        await wait_until(lambda: engine.stats().running == 1)
        rest = [
            asyncio.create_task(engine.transcribe(marker_audio(m), language="en"))
            for m in range(2, 7)
        ]
        await wait_until(lambda: engine.stats().batch_pending == 5)
        gate.set()
        results = [await first, *await asyncio.gather(*rest)]

        self.assertEqual(_texts(results), [f"en:{m}" for m in range(1, 7)])
        for call in ct2.generate_calls:
            sampled = "sampling_temperature" in call["kwargs"]
            # Retries at a temperature run only with other retries.
            self.assertEqual(sampled, set(call["markers"]) <= {3, 5}, call)
        retried = [m for c in ct2.generate_calls if "sampling_temperature" in c["kwargs"] for m in c["markers"]]
        self.assertEqual(sorted(retried), [3, 5])

    async def test_a_bad_request_fails_alone_and_its_batch_still_completes(self):
        gate = threading.Event()
        ct2 = FakeCT2Whisper(gate=gate, poison_markers=frozenset({4}))
        engine = _engine(ct2, max_batch_size=4)
        self.addCleanup(engine.close)

        first = asyncio.create_task(engine.transcribe(marker_audio(1), language="en"))
        await wait_until(lambda: engine.stats().running == 1)
        rest = [
            asyncio.create_task(engine.transcribe(marker_audio(m), language="en"))
            for m in (2, 3, 4, 5)
        ]
        await wait_until(lambda: engine.stats().batch_pending == 4)
        gate.set()
        results = await asyncio.gather(first, *rest, return_exceptions=True)

        self.assertEqual(results[0].segments[0].text, "en:1")
        self.assertEqual([r.segments[0].text for r in results[1:3]], ["en:2", "en:3"])
        self.assertIsInstance(results[3], RuntimeError)
        self.assertEqual(results[4].segments[0].text, "en:5")
        stats = engine.stats()
        self.assertEqual((stats.completed, stats.failed), (4, 1))

    async def test_a_cancelled_request_is_dropped_before_its_batch_runs(self):
        gate = threading.Event()
        ct2 = FakeCT2Whisper(gate=gate)
        engine = _engine(ct2, max_batch_size=4)
        self.addCleanup(engine.close)

        first = asyncio.create_task(engine.transcribe(marker_audio(1), language="en"))
        await wait_until(lambda: engine.stats().running == 1)
        kept = asyncio.create_task(engine.transcribe(marker_audio(2), language="en"))
        dropped = asyncio.create_task(engine.transcribe(marker_audio(3), language="en"))
        await wait_until(lambda: engine.stats().batch_pending == 2)

        dropped.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await dropped
        gate.set()

        self.assertEqual((await first).segments[0].text, "en:1")
        self.assertEqual((await kept).segments[0].text, "en:2")
        await wait_until(lambda: engine.stats().running == 0)
        self.assertNotIn(3, [m for c in ct2.generate_calls for m in c["markers"]])
        self.assertEqual(engine.stats().cancelled, 1)

    async def test_close_fails_calls_waiting_for_a_batch(self):
        gate = threading.Event()
        ct2 = FakeCT2Whisper(gate=gate)
        engine = _engine(ct2, max_batch_size=4)

        first = asyncio.create_task(engine.transcribe(marker_audio(1), language="en"))
        await wait_until(lambda: engine.stats().running == 1)
        waiting = asyncio.create_task(engine.transcribe(marker_audio(2), language="en"))
        await wait_until(lambda: engine.stats().batch_pending == 1)

        engine.close()
        gate.set()

        self.assertEqual((await first).segments[0].text, "en:1")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await waiting

    async def test_a_batch_wait_lets_requests_arriving_close_together_share_a_batch(self):
        ct2 = FakeCT2Whisper()
        engine = _engine(ct2, max_batch_size=4, max_batch_wait_ms=300)
        self.addCleanup(engine.close)

        async def later(marker, delay):
            await asyncio.sleep(delay)
            return await engine.transcribe(marker_audio(marker), language="en")

        results = await asyncio.gather(later(1, 0), later(2, 0.03), later(3, 0.06))

        self.assertEqual(_texts(results), ["en:1", "en:2", "en:3"])
        self.assertEqual([c["rows"] for c in ct2.generate_calls], [3])

    async def test_load_picks_the_batched_engine_for_batch_sizes_above_one(self):
        name = f"fake-{uuid.uuid4().hex}"
        with patch(
            "services.whisper.engine.WhisperModel",
            side_effect=lambda *a, **k: FakeFasterWhisper(FakeCT2Whisper()),
        ):
            batched = await load_whisper_engine(
                WhisperEngineConfig(model=name, max_batch_size=8, beam_size=1)
            )
            unbatched = await load_whisper_engine(WhisperEngineConfig(model=name))
        self.addCleanup(batched.close)
        self.addCleanup(unbatched.close)

        self.assertIsInstance(batched, BatchedWhisperEngine)
        self.assertEqual((batched.max_batch_size, batched.beam_size), (8, 1))
        self.assertIs(type(unbatched), SharedWhisperEngine)


class BatchedTranscriptPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_sessions_get_only_their_own_transcripts(self):
        from test_whisper_stt import _transcribe
        from pipecat.tests.utils import SleepFrame
        from pipecat.transcriptions.language import Language

        ct2 = FakeCT2Whisper(delay=0.03)
        engine = _engine(ct2, max_batch_size=4)
        self.addCleanup(engine.close)
        sessions = [
            (Language.EN if i % 2 else Language.TR, (1000 * i + 1, 1000 * i + 2))
            for i in range(1, 7)
        ]

        results = await asyncio.gather(
            *(
                _transcribe(
                    SharedWhisperSTTService(
                        engine=engine,
                        settings=SharedWhisperSTTService.Settings(language=language),
                    ),
                    [*speech_frames(markers[0]), SleepFrame(sleep=0.01), *speech_frames(markers[1])],
                )
                for language, markers in sessions
            )
        )

        for (language, markers), (transcripts, errors) in zip(sessions, results):
            with self.subTest(markers=markers):
                self.assertEqual(
                    transcripts, [f"{language.value}:{markers[0]}", f"{language.value}:{markers[1]}"]
                )
                self.assertEqual(errors, [])
        self.assertGreater(max(c["rows"] for c in ct2.generate_calls), 1)


@unittest.skipUnless(
    os.getenv("WHISPER_MODEL_TESTS"),
    "set WHISPER_MODEL_TESTS=1 to run against a real faster-whisper model",
)
class BatchedRealModelTests(unittest.IsolatedAsyncioTestCase):
    """Compare batched and unbatched transcripts of the same audio.

    Uses up to 24 files from ``benchmarks/whisper/audio`` (or
    ``WHISPER_TEST_AUDIO``), cut into segments of 2-20 s.
    """

    async def test_batched_transcripts_match_unbatched_ones(self):
        from benchmarks.whisper.audio import find_audio_files, load_clip
        from benchmarks.whisper.report import word_error_rate

        audio_dir = os.getenv("WHISPER_TEST_AUDIO", "benchmarks/whisper/audio")
        try:
            paths = find_audio_files([audio_dir])[:24]
        except FileNotFoundError:
            self.skipTest(f"no audio in {audio_dir}")
        rng = np.random.default_rng(0)
        segments = []
        for path, seconds in zip(paths, [2, 3, 5, 8, 12, 20] * 4):
            pcm = np.frombuffer(load_clip(path).pcm, dtype="<i2").astype(np.float32) / 32768
            length = min(len(pcm), seconds * 16000)
            start = int(rng.integers(0, len(pcm) - length + 1))
            segments.append(pcm[start : start + length])

        model = os.getenv("WHISPER_TEST_MODEL", "small")
        unbatched = SharedWhisperEngine.from_config(WhisperEngineConfig(model=model))
        batched = BatchedWhisperEngine.from_config(
            WhisperEngineConfig(model=model, max_batch_size=8)
        )
        self.addCleanup(unbatched.close)
        self.addCleanup(batched.close)

        expected = [await unbatched.transcribe(s, language="en") for s in segments]
        actual = await asyncio.gather(*(batched.transcribe(s, language="en") for s in segments))

        def text(result):
            return " ".join(s.text.strip() for s in result.segments)

        rates = [word_error_rate(text(e), text(a)) for e, a in zip(expected, actual)]
        self.assertGreater(max(a.batch_size for a in actual), 1)
        # Batched GPU math can differ in the last bits; allow rare word changes.
        self.assertLessEqual(sum(rates) / len(rates), 0.02, rates)
        self.assertLessEqual(sum(1 for r in rates if r > 0), len(rates) // 4, rates)


if __name__ == "__main__":
    unittest.main()
