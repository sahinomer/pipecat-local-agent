import os
import unittest
from unittest.mock import patch

from config import AppConfig
from services import VLLMService


class VLLMServiceTests(unittest.TestCase):
    def test_service_uses_vllm_openai_defaults(self):
        service = VLLMService(
            settings=VLLMService.Settings(model="served-model")
        )

        self.assertEqual(service._settings.model, "served-model")
        self.assertFalse(service.supports_developer_role)
        self.assertEqual(str(service._client.base_url), "http://localhost:8000/v1/")
        self.assertEqual(service._client.api_key, "vllm")
        self.assertEqual(
            service._settings.extra["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        request_params = service.build_chat_completion_params({"messages": []})
        self.assertEqual(
            request_params["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_reasoning_can_be_enabled_in_settings(self):
        service = VLLMService(
            settings=VLLMService.Settings(
                model="served-model",
                extra={
                    "extra_body": {
                        "chat_template_kwargs": {
                            "enable_thinking": True,
                        }
                    }
                },
            )
        )

        self.assertEqual(
            service._settings.extra["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": True}},
        )

    def test_settings_take_precedence_over_legacy_model_argument(self):
        with patch.object(
            VLLMService, "_warn_init_param_moved_to_settings"
        ) as warn:
            service = VLLMService(
                model="legacy-model",
                settings=VLLMService.Settings(model="settings-model"),
            )

        warn.assert_called_once_with("model", "model")
        self.assertEqual(service._settings.model, "settings-model")


class LLMConfigTests(unittest.TestCase):
    def test_vllm_configuration(self):
        environment = {
            "LLM_SERVICE": "vllm",
            "LLM_BASE_URL": "http://vllm:8000/v1",
            "LLM_MODEL": "served-model",
            "LLM_API_KEY": "secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = AppConfig.from_env()

        self.assertEqual(config.llm.service, "vllm")
        self.assertEqual(config.llm.base_url, "http://vllm:8000/v1")
        self.assertEqual(config.llm.model, "served-model")
        self.assertEqual(config.llm.api_key, "secret")

    def test_ollama_configuration_uses_llm_variables(self):
        environment = {
            "LLM_SERVICE": "ollama",
            "LLM_BASE_URL": "http://ollama:11434/v1",
            "LLM_MODEL": "configured-model",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = AppConfig.from_env()

        self.assertEqual(config.llm.service, "ollama")
        self.assertEqual(config.llm.base_url, "http://ollama:11434/v1")
        self.assertEqual(config.llm.model, "configured-model")
        self.assertEqual(config.llm.api_key, "ollama")

    def test_legacy_ollama_variables_raise_migration_error(self):
        environment = {
            "OLLAMA_BASE_URL": "http://ollama:11434/v1",
            "OLLAMA_MODEL": "legacy-model",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(
                ValueError,
                "Configure Ollama with LLM_SERVICE=ollama",
            ):
                AppConfig.from_env()



if __name__ == "__main__":
    unittest.main()
