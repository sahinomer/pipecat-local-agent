"""vLLM service implementation for Pipecat's OpenAI-compatible LLM API."""

from dataclasses import dataclass

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class VLLMSettings(BaseOpenAILLMService.Settings):
    """Runtime-updatable settings for :class:`VLLMService`."""

    pass


class VLLMService(OpenAILLMService):
    """Connect Pipecat to a vLLM OpenAI-compatible server."""

    # Whether a model accepts the newer developer role depends on its chat
    # template. Converting it keeps this service compatible with more models.
    supports_developer_role = False

    Settings = VLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "vllm",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize a vLLM LLM service.

        Args:
            model: Model name exposed by the vLLM server. Prefer passing this
                through ``settings=VLLMService.Settings(model=...)``.
            base_url: OpenAI-compatible vLLM endpoint.
            api_key: API key configured on the vLLM server. The default is a
                non-secret placeholder suitable for a server without auth.
            settings: Runtime-updatable settings. Values here take precedence
                over the deprecated ``model`` argument.
            **kwargs: Additional arguments passed to ``OpenAILLMService``.
        """
        default_settings = self.Settings(
            model="default",
            extra={
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    }
                }
            },
        )

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            base_url=base_url,
            api_key=api_key,
            settings=default_settings,
            **kwargs,
        )

    def create_client(self, base_url=None, **kwargs):
        """Create an OpenAI-compatible client configured for vLLM."""
        logger.debug(f"Creating vLLM client with API {base_url}")
        return super().create_client(base_url=base_url, **kwargs)
