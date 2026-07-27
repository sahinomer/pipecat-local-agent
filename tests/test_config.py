import os
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values
from pipecat.services.whisper.stt import Model
from pipecat.transcriptions.language import Language

from config import AppConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppConfigTests(unittest.TestCase):
    def test_env_example_parses_to_documented_defaults(self):
        environment = {
            key: value
            for key, value in dotenv_values(
                PROJECT_ROOT / ".env.example"
            ).items()
            if value is not None
        }

        with patch.dict(os.environ, environment, clear=True):
            config = AppConfig.from_env()

        self.assertEqual(config.agent_language, Language.EN)
        self.assertEqual(config.agent_language_name, "English")
        self.assertEqual(config.whisper_model, Model.SMALL)
        self.assertEqual(config.llm.service, "ollama")
        self.assertEqual(config.llm.base_url, "http://localhost:11434/v1")
        self.assertEqual(config.llm.model, "gpt-oss:20b")
        self.assertEqual(config.llm.api_key, "ollama")
        self.assertEqual(config.tts.model, "xtts")
        self.assertEqual(config.tts.base_url, "http://localhost:8000")
        self.assertEqual(config.tts.voice, "Claribel Dervla")

    def test_vllm_and_chatterbox_configuration(self):
        environment = {
            "AGENT_LANGUAGE": "tr",
            "WHISPER_MODEL_SIZE": "medium",
            "LLM_SERVICE": "vllm",
            "LLM_BASE_URL": "http://vllm:8000/v1",
            "LLM_MODEL": "served-model",
            "LLM_API_KEY": "secret",
            "TTS_MODEL": "chatterbox-multilingual",
            "TTS_BASE_URL": "http://chatterbox:8004",
            "TTS_VOICE": "Turkish.wav",
        }

        with patch.dict(os.environ, environment, clear=True):
            config = AppConfig.from_env()

        self.assertEqual(config.agent_language, Language.TR)
        self.assertEqual(config.agent_language_name, "Turkish")
        self.assertEqual(config.whisper_model, Model.MEDIUM)
        self.assertEqual(config.llm.service, "vllm")
        self.assertEqual(config.llm.base_url, "http://vllm:8000/v1")
        self.assertEqual(config.llm.model, "served-model")
        self.assertEqual(config.llm.api_key, "secret")
        self.assertEqual(config.tts.model, "chatterbox-multilingual")
        self.assertEqual(config.tts.base_url, "http://chatterbox:8004")
        self.assertEqual(config.tts.voice, "Turkish.wav")


if __name__ == "__main__":
    unittest.main()
