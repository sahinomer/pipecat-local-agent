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
        self.assertTrue(config.observability.enabled)
        self.assertEqual(config.observability.metrics_host, "127.0.0.1")
        self.assertEqual(config.observability.metrics_port, 8001)
        self.assertTrue(config.observability.gpu_metrics_enabled)

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

    def test_observability_configuration(self):
        environment = {
            "OBSERVABILITY_ENABLED": "false",
            "METRICS_HOST": "0.0.0.0",
            "METRICS_PORT": "9101",
            "GPU_METRICS_ENABLED": "off",
        }

        with patch.dict(os.environ, environment, clear=True):
            config = AppConfig.from_env()

        self.assertFalse(config.observability.enabled)
        self.assertEqual(config.observability.metrics_host, "0.0.0.0")
        self.assertEqual(config.observability.metrics_port, 9101)
        self.assertFalse(config.observability.gpu_metrics_enabled)

    def test_invalid_observability_configuration_is_rejected(self):
        with patch.dict(
            os.environ,
            {"OBSERVABILITY_ENABLED": "sometimes"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "OBSERVABILITY_ENABLED"):
                AppConfig.from_env()

        with patch.dict(
            os.environ,
            {"METRICS_PORT": "70000"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "METRICS_PORT"):
                AppConfig.from_env()


if __name__ == "__main__":
    unittest.main()
