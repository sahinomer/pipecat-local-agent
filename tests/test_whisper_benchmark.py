"""Tests for the Whisper load-test tool in benchmarks/whisper.

The end-to-end tests run the benchmark server in-process with a fake engine
and an energy-based VAD in place of Silero, then drive it with the real
client session code over a real WebSocket.
"""

import asyncio
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
import numpy as np
import soundfile
import uvicorn
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import InputAudioRawFrame, OutputTransportMessageUrgentFrame
from pipecat.serializers.protobuf import ProtobufFrameSerializer

from benchmarks.whisper import protocol, server
from benchmarks.whisper.audio import (
    AudioClip,
    build_playlist,
    find_audio_files,
    load_clip,
    split_chunks,
)
from benchmarks.whisper.matrix import capacity
from benchmarks.whisper.matrix import parse_args as matrix_args
from benchmarks.whisper.client import Options, load_baseline, parse_args, run_session
from benchmarks.whisper.report import (
    PlaylistItem,
    SegmentResult,
    SessionResult,
    Thresholds,
    percentile,
    summarize_step,
    word_error_rate,
)
from services.whisper import BatchedWhisperEngine, SharedWhisperEngine
from whisper_fakes import FakeCT2Whisper, FakeFasterWhisper, FakeWhisperModel


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_audio_is_what_the_server_serializer_reads(self):
        pcm = np.arange(320, dtype="<i2").tobytes()

        frame = await ProtobufFrameSerializer().deserialize(protocol.encode_audio(pcm))

        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.audio, pcm)
        self.assertEqual((frame.sample_rate, frame.num_channels), (16000, 1))

    async def test_server_messages_decode_on_the_client(self):
        message = {"type": protocol.SEGMENT, "segment": 3, "text": "merhaba dünya"}

        data = await ProtobufFrameSerializer().serialize(
            OutputTransportMessageUrgentFrame(message=message)
        )

        self.assertEqual(protocol.decode_message(data), message)
        self.assertEqual(protocol.decode_message(protocol.encode_message(message)), message)

    def test_non_message_frames_decode_to_none(self):
        self.assertIsNone(protocol.decode_message(protocol.encode_audio(b"\x00\x00")))


class AudioTests(unittest.TestCase):
    def test_files_are_mixed_to_mono_resampled_and_get_their_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            stereo = np.zeros((8000, 2), dtype=np.float32)
            stereo[:, 0] = 0.5
            soundfile.write(root / "nested" / "clip.wav", stereo, 8000)
            # Windows tools often write a byte-order mark.
            (root / "nested" / "clip.txt").write_text("﻿hello there\n", encoding="utf-8")
            (root / "notes.md").write_text("not audio", encoding="utf-8")

            paths = find_audio_files([tmp])
            clip = load_clip(paths[0])

        self.assertEqual([p.name for p in paths], ["clip.wav"])
        self.assertAlmostEqual(clip.duration_secs, 1.0, places=2)
        samples = np.frombuffer(clip.pcm, dtype="<i2")
        self.assertAlmostEqual(float(np.median(samples)) / 32767, 0.25, places=2)
        self.assertEqual(clip.reference, "hello there")

    def test_missing_inputs_are_reported(self):
        with self.assertRaises(FileNotFoundError):
            find_audio_files(["does-not-exist.wav"])

    def test_chunks_cover_the_audio(self):
        chunks = split_chunks(bytes(16000 * 2 + 100), chunk_ms=20)

        self.assertEqual(len(chunks[0]), 640)
        self.assertEqual(sum(len(c) for c in chunks), 16000 * 2 + 100)


def _clip(name: str, seconds: float) -> AudioClip:
    return AudioClip(name=name, path=Path(name), pcm=bytes(int(16000 * seconds) * 2))


class PlaylistTests(unittest.TestCase):
    def test_files_fill_the_duration_whole_and_in_rotation(self):
        clips = [_clip("a", 30), _clip("b", 50), _clip("c", 20), _clip("d", 40)]

        pcm, items = build_playlist(clips, 1, duration_secs=100, gap_secs=1)

        # b (50) + c (20) fit, d (40) would overflow, a (30) does not fit either.
        self.assertEqual([i.clip for i in items], ["b", "c"])
        self.assertEqual([i.offset_secs for i in items], [0.0, 51.0])
        self.assertEqual(len(pcm), int(16000 * 71) * 2)

    def test_a_first_file_longer_than_the_duration_still_plays(self):
        clips = [_clip("long", 180), _clip("short", 10)]

        _pcm, items = build_playlist(clips, 0, duration_secs=120, gap_secs=1)

        self.assertEqual([i.clip for i in items], ["long"])

    def test_segments_are_matched_to_the_file_they_started_in(self):
        session = SessionResult(
            session="s1",
            clip="a +1",
            playlist=[PlaylistItem("a", 0.0, 10.0), PlaylistItem("b", 11.0, 5.0)],
            segments=[
                SegmentResult(1, "first", 3.0, 0.1, 0.1, 0.0, 0.1, None, start_audio_secs=0.4),
                SegmentResult(2, "second", 3.0, 0.1, 0.1, 0.0, 0.1, None, start_audio_secs=6.0),
                # Estimated slightly before b's offset, but inside the gap.
                SegmentResult(3, "third", 3.0, 0.1, 0.1, 0.0, 0.1, None, start_audio_secs=10.7),
            ],
        )

        self.assertEqual(session.clip_transcripts(), [("a", "first second"), ("b", "third")])

    def test_drift_and_reference_wer_are_per_file(self):
        session = SessionResult(
            session="s1",
            clip="a +1",
            state="done",
            playlist=[PlaylistItem("a", 0.0, 10.0), PlaylistItem("b", 11.0, 5.0)],
            segments=[
                SegmentResult(1, "hello there", 3.0, 0.2, 0.1, 0.0, 0.1, None, start_audio_secs=1.0),
                SegmentResult(2, "wrong words", 3.0, 0.2, 0.1, 0.0, 0.1, None, start_audio_secs=12.0),
            ],
        )

        summary = summarize_step(
            [session], [], baseline={"a": "hello there", "b": "good bye"}, references={"a": "hello there"}
        )

        self.assertEqual(summary["drift_wer_max"], 1.0)
        self.assertEqual(summary["drift_wer_mean"], 0.5)
        self.assertEqual(summary["reference_wer_mean"], 0.0)


def _segment(latency: float, *, text="hi", error=None, queue=0.0, inference=0.1):
    return SegmentResult(
        segment=1,
        text=text,
        audio_secs=1.0,
        latency_secs=latency,
        stt_secs=queue + inference,
        queue_wait_secs=queue,
        inference_secs=inference,
        error=error,
    )


class ReportTests(unittest.TestCase):
    def test_word_error_rate(self):
        self.assertEqual(word_error_rate("Hello, world!", "hello world"), 0.0)
        self.assertAlmostEqual(word_error_rate("a b c d", "a x c"), 0.5)
        self.assertEqual(word_error_rate("", ""), 0.0)
        self.assertEqual(word_error_rate("", "extra"), 1.0)
        # casefold is not locale-aware: Turkish dotted/dotless I stay distinct.
        self.assertEqual(word_error_rate("Şeker geldi", "şeker geldi"), 0.0)

    def test_percentile_interpolates(self):
        self.assertIsNone(percentile([], 50))
        self.assertEqual(percentile([3.0], 95), 3.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)

    def test_a_healthy_step_passes(self):
        session = SessionResult(session="s1", clip="a.wav", state="done", segments=[_segment(0.3)])

        summary = summarize_step([session], [], baseline={"a.wav": "hi"})

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["drift_wer_max"], 0.0)

    def test_failures_slow_latency_and_drift_fail_a_step(self):
        sessions = [
            SessionResult(session="s1", clip="a.wav", state="done", segments=[_segment(2.0, text="bye")]),
            SessionResult(session="s2", clip="a.wav", state="failed", failure="connect: TimeoutError"),
            SessionResult(
                session="s3",
                clip="a.wav",
                state="done",
                segments=[_segment(0.2, text="", error="RuntimeError: boom")],
            ),
        ]

        summary = summarize_step(
            sessions,
            [],
            baseline={"a.wav": "hi"},
            thresholds=Thresholds(max_p95_latency_secs=1.0, max_wer_drift=0.5),
        )

        self.assertFalse(summary["passed"])
        reasons = " | ".join(summary["reasons"])
        self.assertIn("1 session(s) failed", reasons)
        self.assertIn("1 segment error(s)", reasons)
        self.assertIn("p95 latency 2.00s", reasons)
        self.assertIn("transcript drift 1.00", reasons)

    def test_peaks_come_from_stats_samples(self):
        samples = [
            {
                "engine": {"queued": 2, "running": 1},
                "segments": {"in_flight": 3},
                "system": {
                    "cpu_percent": 40.0,
                    "process_cpu_percent": 10.0,
                    "gpus": [{"utilization_percent": 70.0, "memory_used_mb": 2000.0}],
                },
                "loop_lag_ms": 12.0,
            },
            {
                "engine": {"queued": 5, "running": 1},
                "segments": {"in_flight": 6},
                "system": {"cpu_percent": 20.0, "process_cpu_percent": 5.0, "gpus": []},
                "loop_lag_ms": 30.0,
            },
        ]

        summary = summarize_step([], samples)

        self.assertEqual(summary["peak_queued"], 5)
        self.assertEqual(summary["peak_in_flight"], 6)
        self.assertEqual(summary["peak_gpu_percent"], 70.0)
        self.assertEqual(summary["peak_gpu_memory_mb"], 2000.0)
        self.assertEqual(summary["peak_loop_lag_ms"], 30.0)
        self.assertIsNone(summary["peak_models_loaded"])


class MatrixTests(unittest.TestCase):
    def test_capacity_is_the_largest_passing_step(self):
        def step(n, passed, p95):
            return {
                "sessions_requested": n,
                "summary": {
                    "passed": passed,
                    "reasons": [] if passed else [f"p95 latency {p95:.2f}s > 1.00s"],
                    "latency_p95": p95,
                    "peak_gpu_memory_mb": 2000.0 + n,
                    "reference_wer_mean": 0.15,
                    "drift_wer_max": 0.0,
                    "batch_size_mean": 1.5,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "variant.json"
            report.write_text(
                json.dumps({"steps": [step(1, True, 0.3), step(8, True, 0.8), step(12, False, 1.4)]}),
                encoding="utf-8",
            )
            result = capacity(report)

        self.assertEqual(result["max_sessions_passing"], 8)
        self.assertEqual(result["first_failing_sessions"], 12)
        self.assertEqual(result["latency_p95_at_max"], 0.8)
        self.assertEqual(result["gpu_memory_mb_at_max"], 2008.0)

    def test_variants_are_validated(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            matrix_args(["--variants", "nope"])
        args = matrix_args(["--variants", "shared-w1,batched-beam5-b4", "--ramp", "1,4"])
        self.assertEqual((args.variant_names, args.steps), (["shared-w1", "batched-beam5-b4"], [1, 4]))


class ClientOptionTests(unittest.TestCase):
    def test_baseline_options_are_validated(self):
        for argv, message in (
            (["a.wav", "--baseline-only"], "requires --baseline"),
            (["a.wav", "--baseline", "--baseline-only", "--ramp", "1,2"], "cannot be combined"),
            (["a.wav", "--baseline", "--baseline-speed", "0"], "must be positive"),
            (["a.wav", "--max-files", "0"], "at least 1"),
            (["a.wav", "--duration", "0"], "must be positive"),
        ):
            with self.subTest(argv=argv):
                with patch("sys.stderr"), self.assertRaises(SystemExit):
                    parse_args(argv)
        args = parse_args(["a.wav", "--baseline", "--baseline-speed", "4", "--ramp", "1,8"])
        self.assertEqual((args.baseline_speed, args.steps), (4.0, [1, 8]))

    def test_baseline_transcripts_load_from_an_earlier_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                '{"baseline": {"transcripts": {"a.wav": "hello"}}}', encoding="utf-8"
            )
            empty = Path(tmp) / "empty.json"
            empty.write_text('{"baseline": null}', encoding="utf-8")

            self.assertEqual(load_baseline(report), {"a.wav": "hello"})
            with self.assertRaises(SystemExit):
                load_baseline(empty)


class EnergyVADAnalyzer(VADAnalyzer):
    """Treats any loud 10 ms window as speech, so tests need no real voices."""

    def __init__(self):
        super().__init__(params=VADParams(min_volume=0.0))

    def num_frames_required(self) -> int:
        return int(self.sample_rate * 0.01)

    def voice_confidence(self, buffer: bytes) -> float:
        samples = np.frombuffer(buffer, dtype="<i2")
        return 1.0 if samples.size and float(np.abs(samples).mean()) > 500 else 0.0


def _utterances(*markers: int) -> bytes:
    """Silence, then each marker as 0.6 s of constant audio followed by silence."""
    silence = np.zeros(int(16000 * 0.4), dtype="<i2")
    parts = [silence]
    for marker in markers:
        parts += [np.full(int(16000 * 0.6), marker, dtype="<i2"), silence]
    return np.concatenate(parts).tobytes()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class BenchmarkServerEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def _serve(self, *server_args: str):
        port = _free_port()
        args = server.parse_args(
            ["--model", "fake-model", "--language", "en", "--no-gpu-stats", "--port", str(port), *server_args]
        )
        uv_server = uvicorn.Server(
            uvicorn.Config(server.create_app(args), host="127.0.0.1", port=port, log_level="warning")
        )
        task = asyncio.create_task(uv_server.serve())

        async def shutdown():
            uv_server.should_exit = True
            await asyncio.wait_for(task, timeout=10)

        self.addAsyncCleanup(shutdown)
        while not uv_server.started:
            if task.done():
                task.result()
            await asyncio.sleep(0.02)
        base = f"http://127.0.0.1:{port}"
        return Options(
            ws_url=base.replace("http", "ws") + "/ws",
            stats_url=base + "/stats",
            chunk_ms=20,
            # Faster than real time, so the server works through a backlog.
            speed=10.0,
            tail_silence_secs=0.6,
            stagger_secs=0.0,
            connect_timeout=5.0,
            ready_timeout=10.0,
            drain_timeout=10.0,
        )

    async def test_concurrent_sessions_get_their_own_segments_and_timings(self):
        engine = SharedWhisperEngine(
            FakeWhisperModel(delay=0.02), model_name="fake-model", num_workers=2
        )
        self.addCleanup(engine.close)
        with (
            patch.object(server, "load_whisper_engine", AsyncMock(return_value=engine)),
            patch.object(server, "SileroVADAnalyzer", EnergyVADAnalyzer),
        ):
            opts = await self._serve("--engine", "shared")
            clips = [
                AudioClip(name=f"c{i}.wav", path=Path(f"c{i}.wav"), pcm=_utterances(1000 * i + 1, 1000 * i + 2))
                for i in (1, 2, 3)
            ]
            sessions = [SessionResult(session=f"s{i}", clip=c.name) for i, c in enumerate(clips)]

            async with aiohttp.ClientSession() as http:
                await asyncio.gather(
                    *(run_session(http, opts, c.pcm, s) for c, s in zip(clips, sessions))
                )
                async with http.get(opts.stats_url) as response:
                    stats = await response.json()

        for i, session in zip((1, 2, 3), sessions):
            with self.subTest(session=session.session):
                self.assertEqual(session.state, "done", session.failure)
                self.assertEqual((session.vad_started, session.vad_stopped), (2, 2))
                self.assertEqual(session.transcript, f"en:{1000 * i + 1} en:{1000 * i + 2}")
                self.assertIsNotNone(session.setup_secs)
                for segment in session.segments:
                    self.assertIsNone(segment.error)
                    self.assertIsNotNone(segment.latency_secs)
                    self.assertIsNotNone(segment.queue_wait_secs)
                    self.assertIsNotNone(segment.inference_secs)
        self.assertEqual(stats["engine"]["mode"], "shared")
        self.assertEqual(stats["sessions"]["total"], 3)
        # Six segments plus the warm-up request at startup.
        self.assertEqual(stats["engine"]["completed"], 7)
        self.assertEqual(stats["segments"], {"in_flight": 0, "completed": 6, "failed": 0})

    async def test_batched_engine_reports_batch_sizes_per_segment(self):
        engine = BatchedWhisperEngine(
            FakeFasterWhisper(FakeCT2Whisper(delay=0.05)),
            model_name="fake-model",
            max_batch_size=4,
        )
        self.addCleanup(engine.close)
        with (
            patch.object(server, "load_whisper_engine", AsyncMock(return_value=engine)),
            patch.object(server, "SileroVADAnalyzer", EnergyVADAnalyzer),
        ):
            opts = await self._serve("--engine", "batched")
            clips = [
                AudioClip(name=f"c{i}.wav", path=Path(f"c{i}.wav"), pcm=_utterances(1000 * i + 7))
                for i in range(1, 7)
            ]
            sessions = [SessionResult(session=f"s{i}", clip=c.name) for i, c in enumerate(clips)]
            async with aiohttp.ClientSession() as http:
                await asyncio.gather(
                    *(run_session(http, opts, c.pcm, s) for c, s in zip(clips, sessions))
                )
                async with http.get(opts.stats_url) as response:
                    stats = await response.json()

        for i, session in zip(range(1, 7), sessions):
            self.assertEqual(session.state, "done", session.failure)
            self.assertEqual(session.transcript, f"en:{1000 * i + 7}")
            (segment,) = session.segments
            self.assertGreaterEqual(segment.batch_size, 1)
            self.assertIsNotNone(segment.batch_wait_secs)
        self.assertEqual(stats["engine"]["mode"], "batched")
        self.assertEqual(stats["engine"]["max_batch_size"], 4)
        self.assertGreater(max(s.segments[0].batch_size for s in sessions), 1)

    async def test_a_session_streaming_several_files_reports_each_files_transcript(self):
        engine = SharedWhisperEngine(FakeWhisperModel(), model_name="fake-model")
        self.addCleanup(engine.close)
        with (
            patch.object(server, "load_whisper_engine", AsyncMock(return_value=engine)),
            patch.object(server, "SileroVADAnalyzer", EnergyVADAnalyzer),
        ):
            opts = await self._serve("--engine", "shared")
            clips = [
                AudioClip(name="a.wav", path=Path("a.wav"), pcm=_utterances(2001, 2002)),
                AudioClip(name="b.wav", path=Path("b.wav"), pcm=_utterances(3001)),
            ]
            pcm, playlist = build_playlist(clips, 0, duration_secs=60, gap_secs=1.0)
            session = SessionResult(session="s1", clip="a.wav +1", playlist=playlist)
            async with aiohttp.ClientSession() as http:
                await run_session(http, opts, pcm, session)

        self.assertEqual(session.state, "done", session.failure)
        self.assertEqual(
            session.clip_transcripts(),
            [("a.wav", "en:2001 en:2002"), ("b.wav", "en:3001")],
        )
        for segment in session.segments:
            self.assertIsNotNone(segment.start_audio_secs)
            self.assertGreater(segment.end_audio_secs, segment.start_audio_secs)

    async def test_session_setup_failure_reaches_the_client(self):
        with patch.object(
            server,
            "BenchPerSessionWhisperSTTService",
            side_effect=RuntimeError("CUDA failed with error out of memory"),
        ):
            opts = await self._serve("--engine", "per-session")
            session = SessionResult(session="s1", clip="c.wav")
            clip = AudioClip(name="c.wav", path=Path("c.wav"), pcm=_utterances(5))

            async with aiohttp.ClientSession() as http:
                await run_session(http, opts, clip.pcm, session)
                async with http.get(opts.stats_url) as response:
                    stats = await response.json()

        self.assertEqual(session.state, "failed")
        self.assertEqual(
            session.failure, "server: RuntimeError: CUDA failed with error out of memory"
        )
        self.assertEqual(stats["engine"]["models_loaded"], 0)
        self.assertEqual(stats["sessions"]["active"], 0)


if __name__ == "__main__":
    unittest.main()
