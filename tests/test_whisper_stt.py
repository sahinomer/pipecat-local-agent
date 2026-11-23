"""Whisper -> transcript pipeline tests for the shared Whisper engine.

Each test runs ``SharedWhisperSTTService`` alone in a Pipecat pipeline, feeds
it VAD-bounded 16 kHz audio, and checks the TranscriptionFrames that come out.
The fake model writes each request's language and audio marker into the
transcript, so a session that received another session's audio or settings
shows up as a wrong transcript.
"""

import asyncio
import os
import time
import unittest

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

from services.whisper import (
    SharedWhisperEngine,
    SharedWhisperSTTService,
    WhisperEngineConfig,
)
from whisper_fakes import SAMPLE_RATE, FakeWhisperModel, speech_frames


def _engine(model: FakeWhisperModel | None = None, *, num_workers: int = 1):
    return SharedWhisperEngine(
        model or FakeWhisperModel(),
        model_name="fake-model",
        num_workers=num_workers,
    )


def _stt(engine, language: Language = Language.EN, **settings):
    return SharedWhisperSTTService(
        engine=engine,
        settings=SharedWhisperSTTService.Settings(language=language, **settings),
    )


async def _transcribe(stt, frames: list[Frame]) -> tuple[list[str], list[ErrorFrame]]:
    """Run ``frames`` through a pipeline holding only ``stt``."""
    down, up = await run_test(stt, frames_to_send=frames)
    transcripts = [f.text.strip() for f in down if isinstance(f, TranscriptionFrame)]
    errors = [f for f in (*down, *up) if isinstance(f, ErrorFrame)]
    return transcripts, errors


class SharedWhisperSTTServiceSetupTests(unittest.TestCase):
    def test_model_setting_defaults_to_the_engine_model(self):
        stt = _stt(_engine())

        self.assertEqual(stt._settings.model, "fake-model")

    def test_a_model_setting_that_differs_from_the_engine_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            _stt(_engine(), model="other-model")

    def test_a_language_the_engine_model_cannot_transcribe_is_rejected(self):
        engine = _engine(FakeWhisperModel(supported_languages=["en"]))

        with self.assertRaisesRegex(ValueError, "cannot transcribe 'tr'"):
            _stt(engine, Language.TR)

    def test_services_do_not_load_their_own_model(self):
        stt = _stt(_engine())

        self.assertIsNone(stt._model)


class SharedWhisperTranscriptPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_speech_segment_becomes_a_transcription(self):
        model = FakeWhisperModel()
        engine = _engine(model)
        self.addCleanup(engine.close)
        stt = _stt(engine, Language.TR, hotwords="pipecat", initial_prompt="hello")

        transcripts, errors = await _transcribe(stt, speech_frames(1200))

        self.assertEqual(transcripts, ["tr:1200"])
        self.assertEqual(errors, [])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["hotwords"], "pipecat")
        self.assertEqual(model.calls[0]["initial_prompt"], "hello")

    async def test_concurrent_sessions_get_only_their_own_transcripts(self):
        # Two workers plus a decode delay make the sessions' requests overlap
        # and queue behind each other.
        model = FakeWhisperModel(delay=0.03)
        engine = _engine(model, num_workers=2)
        self.addCleanup(engine.close)

        sessions = [
            (Language.EN, (1001, 1002)),
            (Language.TR, (2001, 2002)),
            (Language.EN, (3001, 3002)),
            (Language.TR, (4001, 4002)),
        ]
        runs = []
        for language, markers in sessions:
            frames = [
                *speech_frames(markers[0]),
                SleepFrame(sleep=0.01),
                *speech_frames(markers[1]),
            ]
            runs.append(_transcribe(_stt(engine, language), frames))

        results = await asyncio.gather(*runs)

        for (language, markers), (transcripts, errors) in zip(sessions, results):
            with self.subTest(language=language, markers=markers):
                self.assertEqual(
                    transcripts,
                    [f"{language.value}:{markers[0]}", f"{language.value}:{markers[1]}"],
                )
                self.assertEqual(errors, [])
        self.assertEqual(len(model.calls), 8)
        self.assertGreater(model.max_active, 1)

    async def test_language_update_changes_only_that_session(self):
        engine = _engine(FakeWhisperModel(delay=0.01), num_workers=2)
        self.addCleanup(engine.close)
        switched = _stt(engine, Language.EN)
        unchanged = _stt(engine, Language.EN)

        results = await asyncio.gather(
            _transcribe(
                switched,
                [
                    STTUpdateSettingsFrame(
                        delta=SharedWhisperSTTService.Settings(language=Language.TR)
                    ),
                    # Settings frames are queued behind system frames such as
                    # audio and VAD; let the update land before speech starts.
                    SleepFrame(sleep=0.05),
                    *speech_frames(10),
                ],
            ),
            _transcribe(unchanged, [SleepFrame(sleep=0.05), *speech_frames(20)]),
        )

        self.assertEqual(results[0][0], ["tr:10"])
        self.assertEqual(results[1][0], ["en:20"])

    async def test_runtime_model_change_is_refused_and_transcription_continues(self):
        engine = _engine()
        self.addCleanup(engine.close)
        stt = _stt(engine)

        transcripts, errors = await _transcribe(
            stt,
            [
                STTUpdateSettingsFrame(
                    delta=SharedWhisperSTTService.Settings(model="large-v3")
                ),
                SleepFrame(sleep=0.05),
                *speech_frames(7),
            ],
        )

        self.assertEqual(transcripts, ["en:7"])
        self.assertEqual(len(errors), 1)
        self.assertIn("cannot switch Whisper model", errors[0].error)
        self.assertEqual(stt._settings.model, "fake-model")

    async def test_engine_failure_reports_an_error_and_the_session_continues(self):
        engine = _engine(FakeWhisperModel(fail_markers=frozenset({5})))
        self.addCleanup(engine.close)
        stt = _stt(engine)

        transcripts, errors = await _transcribe(
            stt,
            [*speech_frames(5), SleepFrame(sleep=0.01), *speech_frames(6)],
        )

        self.assertEqual(transcripts, ["en:6"])
        self.assertEqual(len(errors), 1)
        self.assertIn("Whisper transcription failed", errors[0].error)
        self.assertIn("marker 5", errors[0].error)

    async def test_failure_in_one_session_does_not_reach_another(self):
        engine = _engine(FakeWhisperModel(delay=0.01, fail_markers=frozenset({5})), num_workers=2)
        self.addCleanup(engine.close)

        failing, healthy = await asyncio.gather(
            _transcribe(_stt(engine), speech_frames(5)),
            _transcribe(_stt(engine), speech_frames(6)),
        )

        self.assertEqual(failing[0], [])
        self.assertEqual(len(failing[1]), 1)
        self.assertEqual(healthy, (["en:6"], []))


@unittest.skipUnless(
    os.getenv("WHISPER_MODEL_TESTS"),
    "set WHISPER_MODEL_TESTS=1 to run against a real faster-whisper model",
)
class SharedWhisperRealModelTests(unittest.IsolatedAsyncioTestCase):
    """Runs the real model. ``WHISPER_TEST_MODEL`` picks it (default ``tiny``)."""

    async def asyncSetUp(self):
        # IsolatedAsyncioTestCase runs its loop in asyncio debug mode, whose
        # bookkeeping alone stalls Pipecat pipelines for 0.1-0.3 s even with a
        # fake model. Measure loop lag the way production runs the loop.
        asyncio.get_running_loop().set_debug(False)

    async def test_concurrent_sessions_share_one_model_without_blocking_the_loop(self):
        config = WhisperEngineConfig(
            model=os.getenv("WHISPER_TEST_MODEL", "tiny"),
            num_workers=int(os.getenv("WHISPER_TEST_WORKERS", "1")),
        )
        engine = await SharedWhisperEngine.load(config)
        self.assertIs(engine, await SharedWhisperEngine.load(config))

        # Three seconds of low-level noise per session: enough work for the
        # model to run, without needing a speech fixture.
        noise = os.urandom(SAMPLE_RATE * 2 * 3)
        quiet = bytes(b & 0x0F if i % 2 == 0 else 0 for i, b in enumerate(noise))
        chunk = SAMPLE_RATE * 2 // 50

        def frames():
            return [
                VADUserStartedSpeakingFrame(),
                *(
                    InputAudioRawFrame(
                        audio=quiet[i : i + chunk], sample_rate=SAMPLE_RATE, num_channels=1
                    )
                    for i in range(0, len(quiet), chunk)
                ),
                VADUserStoppedSpeakingFrame(),
            ]

        max_lag = 0.0
        stop = asyncio.Event()

        async def heartbeat():
            nonlocal max_lag
            interval = 0.01
            while not stop.is_set():
                started = time.perf_counter()
                await asyncio.sleep(interval)
                max_lag = max(max_lag, time.perf_counter() - started - interval)

        beat = asyncio.create_task(heartbeat())
        results = await asyncio.gather(
            *(_transcribe(_stt(engine), frames()) for _ in range(3))
        )
        stop.set()
        await beat

        for _transcripts, errors in results:
            self.assertEqual(errors, [])
        self.assertLess(max_lag, 0.1, f"event loop stalled for {max_lag:.3f}s")


if __name__ == "__main__":
    unittest.main()
