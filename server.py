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
from processor import Wav2ArkitBlendshapeProcessor
from prompts import load_prompt

SYSTEM_PROMPT = load_prompt(
    "system.md",
    agent_language=config.agent_language_name,
)
INITIAL_PROMPT = load_prompt(
    "startup.md",
    agent_language=config.agent_language_name,
)

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

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

            llm = OLLamaLLMService(
                base_url=config.ollama_base_url,
                settings=OLLamaLLMService.Settings(
                    model=config.ollama_model,
                    system_instruction=SYSTEM_PROMPT,
                )
            )

            tts = XTTSService(
                aiohttp_session=session,
                base_url=config.xtts_base_url,
                settings=XTTSService.Settings(
                    voice=config.xtts_voice,
                    language=config.agent_language,
                )
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

            worker = PipelineWorker(
                pipeline,
                params=PipelineParams(
                    enable_metrics=True,
                    enable_usage_metrics=True,
                ),
                observers=[RTVIObserver(rtvi)],
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
                logger.info(f"Client disconnected")
                await worker.cancel()

            runner = WorkerRunner()
            await runner.add_workers(worker)
            await runner.run()

    except Exception as e:
        logger.error(f"Error in websocket session: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
