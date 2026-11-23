import asyncio
import sys
import time
import uuid
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI, WebSocket
from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.xtts.tts import XTTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

# Custom Imports
from config import config
from observability import (
    METRICS,
    LokiLogSink,
    MetricsHttpServer,
    PrometheusMetricsObserver,
)
from prompts import load_prompt
from services import (
    ChatterboxTTSService,
    VLLMService,
    Wav2ArkitBlendshapeProcessor,
)

SYSTEM_PROMPT = load_prompt(
    "system.md",
    agent_language=config.agent_language_name,
)
INITIAL_PROMPT = load_prompt(
    "startup.md",
    agent_language=config.agent_language_name,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    metrics_server = None
    loki_sink = None
    loki_handler_id = None

    logger.remove()
    logger.configure(extra={"session_id": "system"})
    logger.add(
        sys.stderr,
        level=config.observability.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "session=<cyan>{extra[session_id]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
        diagnose=False,
    )

    if config.observability.enabled:
        loki_sink = LokiLogSink(base_url=config.observability.loki_url)
        await loki_sink.start()
        loki_handler_id = logger.add(
            loki_sink,
            level=config.observability.log_level,
            diagnose=False,
            catch=True,
        )

        METRICS.set_gpu_enabled(config.observability.gpu_metrics_enabled)
        metrics_server = MetricsHttpServer(
            url=config.observability.metrics_url,
            registry=METRICS.registry,
        )
        metrics_server.start()

    try:
        yield
    finally:
        if metrics_server is not None:
            metrics_server.stop()
        if loki_handler_id is not None:
            logger.remove(loki_handler_id)
        if loki_sink is not None:
            await loki_sink.stop()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = uuid.uuid4().hex
    with logger.contextualize(session_id=session_id):
        await run_agent_session(websocket, session_id=session_id)


async def run_agent_session(websocket: WebSocket, *, session_id: str):
    await websocket.accept()
    connection_started = time.perf_counter()
    connection_outcome = "completed"
    METRICS.connections_active.inc()
    METRICS.connections_total.inc()
    logger.info("Agent session started")

    try:
        # 1. Transport with Custom Serializer
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,  # PCM for low latency
                serializer=ProtobufFrameSerializer(),
            )
        )

        # 2. Services
        async with aiohttp.ClientSession() as session:
            stt = WhisperSTTService(
                settings=WhisperSTTService.Settings(
                    model=config.whisper_model,
                    language=config.agent_language,
                )
            )

            if config.llm.service == "vllm":
                llm = VLLMService(
                    base_url=config.llm.base_url,
                    api_key=config.llm.api_key,
                    settings=VLLMService.Settings(
                        model=config.llm.model,
                        system_instruction=SYSTEM_PROMPT,
                    ),
                )
            else:
                llm = OLLamaLLMService(
                    base_url=config.llm.base_url,
                    settings=OLLamaLLMService.Settings(
                        model=config.llm.model,
                        system_instruction=SYSTEM_PROMPT,
                    ),
                )

            if config.tts.model.startswith("chatterbox"):
                tts = ChatterboxTTSService(
                    aiohttp_session=session,
                    base_url=config.tts.base_url,
                    settings=ChatterboxTTSService.Settings(
                        model=config.tts.model,
                        voice=config.tts.voice,
                        language=config.agent_language,
                    ),
                )
            else:
                tts = XTTSService(
                    aiohttp_session=session,
                    base_url=config.tts.base_url,
                    settings=XTTSService.Settings(
                        voice=config.tts.voice,
                        language=config.agent_language,
                    ),
                )

            blendshapes = Wav2ArkitBlendshapeProcessor()

            context = LLMContext()
            user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
                context,
                user_params=LLMUserAggregatorParams(
                    vad_analyzer=SileroVADAnalyzer(),
                    user_turn_strategies=UserTurnStrategies(
                        stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
                    ),
                ),
            )

            rtvi = RTVIProcessor()

            pipeline = Pipeline(
                [
                    transport.input(),  # Transport user input
                    rtvi,  # RTVI processor
                    stt,
                    user_aggregator,  # User responses
                    llm,  # LLM
                    tts,  # TTS
                    blendshapes,  # Pass-through audio + wav2arkit blendshapes
                    transport.output(),  # Transport bot output
                    assistant_aggregator,  # Assistant spoken responses
                ]
            )

            observers = [RTVIObserver(rtvi)]
            if config.observability.enabled:
                observers.append(PrometheusMetricsObserver())

            worker = PipelineWorker(
                pipeline,
                app_resources={"session_id": session_id},
                conversation_id=session_id,
                params=PipelineParams(
                    enable_metrics=True,
                    enable_usage_metrics=True,
                    send_initial_empty_metrics=False,
                ),
                observers=observers,
            )

            @transport.event_handler("on_client_connected")
            async def on_client_connected(transport, client):
                logger.info(f"Client connected")
                context.add_message(
                    {"role": "developer", "content": INITIAL_PROMPT}
                )
                await worker.queue_frames([LLMRunFrame()])

            @transport.event_handler("on_client_disconnected")
            async def on_client_disconnected(transport, client):
                nonlocal connection_outcome
                connection_outcome = "client_disconnected"
                logger.info(f"Client disconnected")
                await worker.cancel()

            runner = WorkerRunner()
            await runner.add_workers(worker)
            await runner.run()

    except asyncio.CancelledError:
        if connection_outcome != "client_disconnected":
            connection_outcome = "cancelled"
        raise
    except Exception as e:
        connection_outcome = "error"
        logger.exception("Error in agent session: {}", e)
    finally:
        connection_duration = time.perf_counter() - connection_started
        METRICS.connections_active.dec()
        METRICS.connection_terminations_total.labels(
            outcome=connection_outcome
        ).inc()
        METRICS.connection_duration_seconds.observe(connection_duration)
        logger.info(
            "Agent session finished",
            outcome=connection_outcome,
            duration_seconds=round(connection_duration, 3),
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
