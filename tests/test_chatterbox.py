import io
import struct
import unittest
import wave
from unittest.mock import AsyncMock

from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.transcriptions.language import Language

from services import ChatterboxTTSService


def _wav_bytes(sample_rate: int = 24000, samples: int = 100) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{samples}h", *range(samples)))
    return buffer.getvalue()


class _FakeContent:
    def __init__(self, data: bytes):
        self._data = data

    async def iter_any(self):
        midpoint = len(self._data) // 2
        yield self._data[:midpoint]
        yield self._data[midpoint:]


class _FakeResponse:
    status = 200

    def __init__(self, data: bytes):
        self.content = _FakeContent(data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeSession:
    def __init__(self, data: bytes):
        self._data = data
        self.url = None
        self.payload = None

    def post(self, url: str, *, json: dict):
        self.url = url
        self.payload = json
        return _FakeResponse(self._data)


class ChatterboxTTSServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_wav_response_becomes_context_audio(self):
        session = _FakeSession(_wav_bytes())
        service = ChatterboxTTSService(
            base_url="http://chatterbox:8004/",
            aiohttp_session=session,
            sample_rate=24000,
            settings=ChatterboxTTSService.Settings(
                model="chatterbox-multilingual",
                voice="Emily.wav",
                language=Language.EN,
            ),
        )
        # Pipecat normally sets this from StartFrame before calling run_tts.
        service._sample_rate = 24000
        service.start_tts_usage_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()

        frames = [frame async for frame in service.run_tts("Hello", "context-1")]

        self.assertEqual(session.url, "http://chatterbox:8004/v1/audio/speech")
        self.assertEqual(
            session.payload,
            {
                "model": "chatterbox-multilingual",
                "input": "Hello",
                "voice": "Emily.wav",
                "language": "en",
                "response_format": "wav",
            },
        )
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], TTSAudioRawFrame)
        self.assertEqual(frames[0].context_id, "context-1")
        self.assertEqual(frames[0].sample_rate, 24000)
        self.assertEqual(frames[0].num_channels, 1)
        self.assertEqual(len(frames[0].audio), 200)


if __name__ == "__main__":
    unittest.main()
