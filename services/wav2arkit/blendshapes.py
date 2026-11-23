"""Pipeline processor that turns outgoing TTS audio into ARKit blendshapes."""

import asyncio
import math
from typing import Optional

import numpy as np
from loguru import logger
from onnxruntime import InferenceSession
from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from scipy.signal import resample_poly

from .model import MODEL_SAMPLE_RATE, BlendshapeModel, Wav2ArkitBlendshapeModel


class Wav2ArkitBlendshapeProcessor(FrameProcessor):
    """Generate ARKit blendshape coefficients from outgoing TTS audio.

    Downstream ``TTSAudioRawFrame`` instances are forwarded unchanged. A mono,
    16 kHz float32 copy is prepared for wav2arkit inference, and the resulting
    coefficients are emitted to the client in an RTVI server message. Other
    frames pass through unchanged.
    """

    def __init__(
        self,
        *,
        model: Optional[BlendshapeModel] = None,
        session: Optional[InferenceSession] = None,
    ):
        super().__init__()

        if model is not None and session is not None:
            raise ValueError("model and session cannot both be provided")
        self._model = (
            model if model is not None
            else Wav2ArkitBlendshapeModel(session=session)
        )

    def can_generate_metrics(self) -> bool:
        return True

    @staticmethod
    def _audio_input(frame: TTSAudioRawFrame) -> np.ndarray:
        if frame.sample_rate <= 0:
            raise ValueError("audio frame sample_rate must be greater than zero")
        if frame.num_channels <= 0:
            raise ValueError("audio frame num_channels must be greater than zero")

        pcm = np.frombuffer(frame.audio, dtype="<i2")
        if pcm.size % frame.num_channels:
            raise ValueError("audio frame does not contain complete PCM samples")

        audio = pcm.reshape(-1, frame.num_channels).astype(np.float32)
        if frame.num_channels > 1:
            audio = audio.mean(axis=1)
        else:
            audio = audio[:, 0]
        audio /= 32768.0

        if frame.sample_rate != MODEL_SAMPLE_RATE and audio.size:
            divisor = math.gcd(frame.sample_rate, MODEL_SAMPLE_RATE)
            audio = resample_poly(
                audio,
                MODEL_SAMPLE_RATE // divisor,
                frame.sample_rate // divisor,
            ).astype(np.float32, copy=False)

        return audio.reshape(1, -1)

    def _infer_blendshapes(self, frame: TTSAudioRawFrame) -> list[list[float]]:
        audio_input = self._audio_input(frame)
        return self._model.predict(audio_input)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Playback always receives the exact frame produced by the TTS service.
        await self.push_frame(frame, direction)

        if direction is not FrameDirection.DOWNSTREAM or not isinstance(
                frame, TTSAudioRawFrame
        ):
            return

        logger.debug(
            "{}: audio frame bytes={} sample_rate={} channels={}",
            self,
            len(frame.audio),
            frame.sample_rate,
            frame.num_channels,
        )

        await self.start_processing_metrics()

        try:
            blendshapes = await asyncio.to_thread(self._infer_blendshapes, frame)
            await self.push_frame(
                RTVIServerMessageFrame(
                    data={
                        "type": "blendshapes",
                        "data": blendshapes,
                        "pts": frame.pts,
                    }
                ),
                direction,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.push_error(
                "Wav2Arkit processing failed",
                exception=exc,
            )
        finally:
            await self.stop_processing_metrics()
