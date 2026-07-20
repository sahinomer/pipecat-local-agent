import io
import wave
from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts


class ChatterboxTTSService(TTSService):
    """Pipecat adapter for devnen/Chatterbox-TTS-Server's OpenAI API."""

    Settings = TTSSettings
    _settings: TTSSettings

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8004",
        aiohttp_session: aiohttp.ClientSession,
        voice: str | None = None,
        model: str = "chatterbox",
        language: Language | str = Language.EN,
        sample_rate: int | None = None,
        settings: TTSSettings | None = None,
        **kwargs,
    ):
        default_settings = self.Settings(
            model=model,
            voice=voice,
            language=language,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._speech_url = f"{base_url.rstrip('/')}/v1/audio/speech"
        self._aiohttp_session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str:
        return language.value

    @staticmethod
    def _decode_wav(data: bytes) -> tuple[bytes, int]:
        try:
            with wave.open(io.BytesIO(data), "rb") as wav_file:
                if wav_file.getcomptype() != "NONE":
                    raise ValueError(
                        f"unsupported WAV compression: {wav_file.getcomptype()}"
                    )
                if wav_file.getsampwidth() != 2:
                    raise ValueError(
                        "Chatterbox must return 16-bit PCM WAV audio; "
                        f"received {wav_file.getsampwidth() * 8}-bit audio"
                    )
                if wav_file.getnchannels() != 1:
                    raise ValueError(
                        "Chatterbox must return mono WAV audio; "
                        f"received {wav_file.getnchannels()} channels"
                    )

                sample_rate = wav_file.getframerate()
                pcm = wav_file.readframes(wav_file.getnframes())
        except wave.Error as error:
            raise ValueError(f"invalid WAV response from Chatterbox: {error}") from error

        if sample_rate <= 0 or not pcm:
            raise ValueError("Chatterbox returned an empty or invalid WAV response")

        return pcm, sample_rate

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        voice = self._settings.voice
        if not isinstance(voice, str) or not voice:
            yield ErrorFrame(error="Chatterbox voice filename must be specified")
            return

        model = self._settings.model
        if not isinstance(model, str) or not model:
            yield ErrorFrame(error="Chatterbox model must be specified")
            return

        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "language": self._settings.language,
            "response_format": "wav",
        }

        try:
            async with self._aiohttp_session.post(
                self._speech_url, json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"{self}: Chatterbox error (status: {response.status}, "
                        f"error: {error_text})"
                    )
                    yield ErrorFrame(
                        error=f"Chatterbox error (status: {response.status}, "
                        f"error: {error_text})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                wav_data = bytearray()
                received_first_chunk = False
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    if not received_first_chunk:
                        await self.stop_ttfb_metrics()
                        received_first_chunk = True
                    wav_data.extend(chunk)

                if not wav_data:
                    yield ErrorFrame(error="Chatterbox returned an empty response")
                    return

                pcm, source_sample_rate = self._decode_wav(bytes(wav_data))
                if source_sample_rate != self.sample_rate:
                    pcm = await self._resampler.resample(
                        pcm,
                        source_sample_rate,
                        self.sample_rate,
                    )

                for offset in range(0, len(pcm), self.chunk_size):
                    yield TTSAudioRawFrame(
                        pcm[offset : offset + self.chunk_size],
                        self.sample_rate,
                        1,
                        context_id=context_id,
                    )
        except (aiohttp.ClientError, ValueError) as error:
            logger.error(f"{self}: Chatterbox request failed: {error}")
            yield ErrorFrame(error=f"Chatterbox request failed: {error}")
        except Exception as error:
            logger.exception(f"{self}: unexpected Chatterbox exception: {error}")
            yield ErrorFrame(error=f"Chatterbox request failed: {error}")
