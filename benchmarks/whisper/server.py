"""Focused speech-to-text server for load-testing Whisper.

Each WebSocket connection runs a minimal Pipecat pipeline:

    transport input -> Silero VAD -> Whisper STT -> transport output

There is no LLM or TTS, so only speech-to-text is measured. Transport,
serializer and VAD match the agent server, so audio is segmented the same
way a real session would segment it.

Run from the repository root:

    uv run python -m benchmarks.whisper.server --engine shared
    uv run python -m benchmarks.whisper.server --engine batched --max-batch-size 4
"""

import argparse
import asyncio
import sys
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import OutputTransportMessageUrgentFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

from config import _resolve_language, config
from services.whisper import WhisperEngineConfig, load_whisper_engine

from . import protocol
from .stt import BenchPerSessionWhisperSTTService, BenchSharedWhisperSTTService, BenchState
from .telemetry import LoopLagMonitor, SystemSampler

ENGINE_MODES = ("shared", "batched", "per-session")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speech-to-text-only Pipecat server for Whisper load tests."
    )
    parser.add_argument(
        "--engine",
        choices=ENGINE_MODES,
        default="shared",
        help="shared: one model for all sessions, one transcription at a time per "
        "worker; batched: one model, decode calls batched across sessions; "
        "per-session: Pipecat's WhisperSTTService, one model per connection "
        "(the agent's current behavior)",
    )
    parser.add_argument("--model", default=config.whisper_model.value)
    parser.add_argument(
        "--language",
        default=config.agent_language.value,
        help="Pipecat language code, e.g. en or tr",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="shared engine only: transcriptions that may run at once",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="shared and batched engines; 1 is greedy. per-session always uses 5",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="batched engine only: decode calls the GPU runs together",
    )
    parser.add_argument(
        "--max-batch-wait-ms",
        type=float,
        default=0.0,
        help="batched engine only: how long a short batch may wait for more requests",
    )
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-gpu-stats", action="store_true")
    return parser.parse_args(argv)


def create_app(args: argparse.Namespace) -> FastAPI:
    language: Language = _resolve_language(args.language)
    state = BenchState(
        engine_mode=args.engine,
        model=args.model,
        language=language.value,
        beam_size=5 if args.engine == "per-session" else args.beam_size,
    )
    sampler = SystemSampler(gpu_enabled=not args.no_gpu_stats)
    loop_lag = LoopLagMonitor()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        sampler.start()
        loop_lag.start()
        engine = None
        if args.engine in ("shared", "batched"):
            batched = args.engine == "batched"
            engine = await load_whisper_engine(
                WhisperEngineConfig(
                    model=args.model,
                    device=args.device,
                    compute_type=args.compute_type,
                    num_workers=1 if batched else args.workers,
                    cpu_threads=args.cpu_threads,
                    beam_size=args.beam_size,
                    # A batch size of 1 would select the unbatched engine.
                    max_batch_size=max(2, args.max_batch_size) if batched else 1,
                    max_batch_wait_ms=args.max_batch_wait_ms if batched else 0.0,
                )
            )
            # Warm up so the first session doesn't pay for CUDA initialization.
            await engine.transcribe(
                np.zeros(protocol.SAMPLE_RATE, dtype=np.float32),
                language=language.value,
            )
            state.engine = engine
        logger.info(
            "Benchmark server ready: engine={} model={} language={} beam_size={}{}",
            args.engine,
            args.model,
            language.value,
            state.beam_size,
            f" max_batch_size={args.max_batch_size}" if args.engine == "batched" else "",
        )
        try:
            yield
        finally:
            if engine is not None:
                engine.close()
            await loop_lag.stop()
            sampler.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/stats")
    async def stats():
        return {
            "time": time.time(),
            **state.snapshot(),
            "system": sampler.latest(),
            "loop_lag_ms": loop_lag.max_lag_ms(),
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, session: str | None = None):
        session_id = session or uuid.uuid4().hex[:8]
        with logger.contextualize(session_id=session_id):
            await run_session(websocket, session_id)

    def create_stt():
        settings_cls = BenchSharedWhisperSTTService.Settings
        if args.engine in ("shared", "batched"):
            return BenchSharedWhisperSTTService(
                engine=state.engine,
                bench_state=state,
                settings=settings_cls(language=language),
                audio_passthrough=False,
            )
        # Like the agent server: the model loads here, on the event loop.
        return BenchPerSessionWhisperSTTService(
            bench_state=state,
            device=args.device,
            compute_type=args.compute_type,
            settings=settings_cls(model=args.model, language=language),
            audio_passthrough=False,
        )

    async def run_session(websocket: WebSocket, session_id: str):
        await websocket.accept()
        accepted_at = time.perf_counter()
        state.sessions_active += 1
        state.sessions_total += 1
        owns_model = False
        logger.info("Session started")
        try:
            transport = FastAPIWebsocketTransport(
                websocket=websocket,
                params=FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_out_enabled=False,
                    add_wav_header=False,
                    serializer=ProtobufFrameSerializer(),
                ),
            )
            stt = create_stt()
            if args.engine == "per-session":
                owns_model = True
                state.models_loaded += 1

            pipeline = Pipeline(
                [
                    transport.input(),
                    VADProcessor(vad_analyzer=SileroVADAnalyzer()),
                    stt,
                    transport.output(),
                ]
            )
            worker = PipelineWorker(
                pipeline,
                params=PipelineParams(enable_metrics=False),
                enable_rtvi=False,
                cancel_on_idle_timeout=False,
            )

            @worker.event_handler("on_pipeline_started")
            async def on_pipeline_started(_worker, _frame):
                setup_secs = time.perf_counter() - accepted_at
                state.session_setup_secs.append(setup_secs)
                await worker.queue_frame(
                    OutputTransportMessageUrgentFrame(
                        message={
                            "type": protocol.READY,
                            "session": session_id,
                            "engine": args.engine,
                            "model": args.model,
                            "language": language.value,
                            "setup_secs": setup_secs,
                        }
                    )
                )

            @transport.event_handler("on_client_disconnected")
            async def on_client_disconnected(_transport, _client):
                await worker.cancel()

            runner = WorkerRunner(handle_sigint=False)
            await runner.add_workers(worker)
            await runner.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Session failed: {}", exc)
            # Tell the client why before the connection closes.
            try:
                await websocket.send_bytes(
                    protocol.encode_message(
                        {"type": protocol.ERROR, "error": f"{type(exc).__name__}: {exc}"}
                    )
                )
            except Exception:
                pass
        finally:
            state.sessions_active -= 1
            if owns_model:
                state.models_loaded -= 1
            logger.info("Session finished")

    return app


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger.remove()
    logger.configure(extra={"session_id": "server"})
    logger.add(
        sys.stderr,
        level=args.log_level.upper(),
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
            "<cyan>{extra[session_id]}</cyan> | {message}"
        ),
    )
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
