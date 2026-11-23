import asyncio
import threading
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from services.whisper import SharedWhisperEngine, WhisperEngineConfig
from whisper_fakes import FakeWhisperModel, marker_audio, wait_until


def _engine(model: FakeWhisperModel, *, num_workers: int = 1) -> SharedWhisperEngine:
    return SharedWhisperEngine(model, model_name="fake", num_workers=num_workers)


class WhisperEngineConfigTests(unittest.TestCase):
    def test_invalid_values_are_rejected(self):
        for kwargs, message in (
            ({"model": " "}, "cannot be empty"),
            ({"model": "small", "num_workers": 0}, "num_workers"),
            ({"model": "small", "cpu_threads": -1}, "cpu_threads"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    WhisperEngineConfig(**kwargs)

    def test_equal_configs_are_interchangeable_keys(self):
        self.assertEqual(
            hash(WhisperEngineConfig(model="small")),
            hash(WhisperEngineConfig(model="small", device="auto")),
        )


class SharedWhisperEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_arguments_reach_the_model_and_segments_are_listed(self):
        model = FakeWhisperModel()
        engine = _engine(model)
        self.addCleanup(engine.close)

        result = await engine.transcribe(
            marker_audio(123),
            language="tr",
            hotwords="pipecat",
            initial_prompt="previous turn",
        )

        self.assertIsInstance(result.segments, list)
        self.assertEqual([s.text for s in result.segments], ["tr:123"])
        self.assertGreaterEqual(result.queue_wait_secs, 0)
        self.assertGreaterEqual(result.inference_secs, 0)
        self.assertEqual(
            model.calls,
            [
                {
                    "marker": 123,
                    "language": "tr",
                    "hotwords": "pipecat",
                    "initial_prompt": "previous turn",
                    "beam_size": 5,
                    "dtype": np.float32,
                }
            ],
        )

    async def test_decoding_runs_on_an_engine_worker_thread(self):
        model = FakeWhisperModel()
        engine = _engine(model)
        self.addCleanup(engine.close)

        await engine.transcribe(marker_audio(1), language="en")

        self.assertEqual(len(model.decode_threads), 1)
        self.assertTrue(model.decode_threads[0].startswith("whisper-engine"))
        self.assertNotEqual(model.decode_threads[0], threading.current_thread().name)

    async def test_one_worker_runs_requests_one_at_a_time_in_arrival_order(self):
        model = FakeWhisperModel(delay=0.02)
        engine = _engine(model, num_workers=1)
        self.addCleanup(engine.close)

        results = await asyncio.gather(
            *(engine.transcribe(marker_audio(m), language="en") for m in (1, 2, 3, 4))
        )

        self.assertEqual([r.segments[0].text for r in results], ["en:1", "en:2", "en:3", "en:4"])
        self.assertEqual(model.max_active, 1)
        self.assertEqual(model.decoded_markers, [1, 2, 3, 4])

    async def test_workers_bound_how_many_requests_run_at_once(self):
        gate = threading.Event()
        model = FakeWhisperModel(gate=gate)
        engine = _engine(model, num_workers=2)
        self.addCleanup(engine.close)

        tasks = [
            asyncio.create_task(engine.transcribe(marker_audio(m), language="en"))
            for m in (1, 2, 3, 4)
        ]
        await wait_until(lambda: engine.running == 2 and engine.queued == 2)
        self.assertEqual(model.active, 2)

        gate.set()
        results = await asyncio.gather(*tasks)

        self.assertEqual(sorted(r.segments[0].text for r in results), ["en:1", "en:2", "en:3", "en:4"])
        self.assertEqual(model.max_active, 2)
        self.assertEqual((engine.queued, engine.running), (0, 0))

    async def test_cancelled_waiting_request_never_reaches_the_model(self):
        gate = threading.Event()
        model = FakeWhisperModel(gate=gate)
        engine = _engine(model, num_workers=1)
        self.addCleanup(engine.close)

        running = asyncio.create_task(engine.transcribe(marker_audio(1), language="en"))
        await wait_until(lambda: engine.running == 1)
        waiting = asyncio.create_task(engine.transcribe(marker_audio(2), language="en"))
        await wait_until(lambda: engine.queued == 1)

        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertEqual(engine.queued, 0)

        gate.set()
        self.assertEqual((await running).segments[0].text, "en:1")
        self.assertEqual([call["marker"] for call in model.calls], [1])
        self.assertEqual((engine.queued, engine.running), (0, 0))

    async def test_cancelling_a_running_request_leaves_others_untouched(self):
        gate = threading.Event()
        model = FakeWhisperModel(gate=gate)
        engine = _engine(model, num_workers=2)
        self.addCleanup(engine.close)

        cancelled = asyncio.create_task(engine.transcribe(marker_audio(1), language="en"))
        kept = asyncio.create_task(engine.transcribe(marker_audio(2), language="tr"))
        await wait_until(lambda: engine.running == 2)

        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        gate.set()

        self.assertEqual((await kept).segments[0].text, "tr:2")
        # The cancelled request was already decoding, so it finishes in the
        # background and its result is discarded.
        await wait_until(lambda: engine.running == 0)
        self.assertEqual(sorted(model.decoded_markers), [1, 2])

    async def test_a_failing_request_does_not_affect_concurrent_ones(self):
        model = FakeWhisperModel(delay=0.01, fail_markers=frozenset({2}))
        engine = _engine(model, num_workers=2)
        self.addCleanup(engine.close)

        results = await asyncio.gather(
            *(engine.transcribe(marker_audio(m), language="en") for m in (1, 2, 3)),
            return_exceptions=True,
        )

        self.assertEqual(results[0].segments[0].text, "en:1")
        self.assertIsInstance(results[1], RuntimeError)
        self.assertIn("marker 2", str(results[1]))
        self.assertEqual(results[2].segments[0].text, "en:3")
        self.assertEqual((engine.queued, engine.running), (0, 0))

    async def test_close_fails_waiting_requests_and_rejects_new_ones(self):
        gate = threading.Event()
        model = FakeWhisperModel(gate=gate)
        engine = _engine(model, num_workers=1)

        running = asyncio.create_task(engine.transcribe(marker_audio(1), language="en"))
        await wait_until(lambda: engine.running == 1)
        waiting = asyncio.create_task(engine.transcribe(marker_audio(2), language="en"))
        await wait_until(lambda: engine.queued == 1)

        engine.close()
        gate.set()

        self.assertEqual((await running).segments[0].text, "en:1")
        # A waiting request fails like any transcription error, not with a
        # CancelledError that would stop the session's task.
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await waiting
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await engine.transcribe(marker_audio(3), language="en")
        self.assertEqual([call["marker"] for call in model.calls], [1])
        self.assertTrue(engine.closed)

    async def test_stats_count_completed_failed_and_cancelled_requests(self):
        gate = threading.Event()
        model = FakeWhisperModel(gate=gate, fail_markers=frozenset({2}))
        engine = _engine(model, num_workers=1)
        self.addCleanup(engine.close)

        first = asyncio.create_task(engine.transcribe(marker_audio(1), language="en"))
        await wait_until(lambda: engine.running == 1)
        failing = asyncio.create_task(engine.transcribe(marker_audio(2), language="en"))
        dropped = asyncio.create_task(engine.transcribe(marker_audio(3), language="en"))
        await wait_until(lambda: engine.queued == 2)
        self.assertEqual(engine.stats().queued, 2)

        dropped.cancel()
        gate.set()
        await first
        with self.assertRaises(RuntimeError):
            await failing
        with self.assertRaises(asyncio.CancelledError):
            await dropped

        stats = engine.stats()
        self.assertEqual(
            (stats.workers, stats.queued, stats.running),
            (1, 0, 0),
        )
        self.assertEqual(
            (stats.completed, stats.failed, stats.cancelled),
            (1, 1, 1),
        )

    async def test_load_shares_one_engine_per_config(self):
        # The process-wide registry outlives each test, so use unique names.
        name = f"fake-{uuid.uuid4().hex}"
        config = WhisperEngineConfig(model=name, device="cpu", compute_type="int8")

        with patch(
            "services.whisper.engine.WhisperModel",
            side_effect=lambda *a, **k: FakeWhisperModel(),
        ) as whisper_model:
            first, second = await asyncio.gather(
                SharedWhisperEngine.load(config),
                SharedWhisperEngine.load(WhisperEngineConfig(model=name, device="cpu", compute_type="int8")),
            )
            other = await SharedWhisperEngine.load(
                WhisperEngineConfig(model=name, device="cpu", compute_type="int8", num_workers=2)
            )
        self.addCleanup(first.close)
        self.addCleanup(other.close)

        self.assertIs(first, second)
        self.assertIsNot(first, other)
        self.assertEqual(whisper_model.call_count, 2)
        whisper_model.assert_any_call(
            name, device="cpu", compute_type="int8", cpu_threads=0, num_workers=1
        )
        self.assertEqual(first.model_name, name)
        self.assertEqual(other.num_workers, 2)


if __name__ == "__main__":
    unittest.main()
