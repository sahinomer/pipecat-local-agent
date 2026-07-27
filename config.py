import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from babel import Locale, UnknownLocaleError
from dotenv import load_dotenv
from pipecat.services.whisper.stt import Model
from pipecat.transcriptions.language import Language


load_dotenv(Path(__file__).with_name(".env"))


def _resolve_language(value: str) -> Language:
    normalized_value = value.strip().replace("_", "-")
    enum_name = normalized_value.replace("-", "_").upper()

    if enum_name in Language.__members__:
        return Language[enum_name]

    for language in Language:
        if language.value.casefold() == normalized_value.casefold():
            return language

    raise ValueError(
        f"Unsupported AGENT_LANGUAGE value: {value!r}. "
        "Use a Pipecat language code such as 'en', 'en-US', or 'tr'."
    )


def _language_name(language: Language) -> str:
    try:
        locale = Locale.parse(language.value, sep="-")
        return locale.get_language_name("en")
    except (UnknownLocaleError, ValueError):
        return language.name.replace("_", " ").title()


def _resolve_whisper_model(value: str) -> Model:
    normalized_value = value.strip()
    enum_name = normalized_value.replace("-", "_").upper()

    if enum_name in Model.__members__:
        return Model[enum_name]

    for model in Model:
        if model.value.casefold() == normalized_value.casefold():
            return model

    raise ValueError(
        f"Unsupported WHISPER_MODEL_SIZE value: {value!r}. "
        "Use a Pipecat Whisper model such as 'small' or 'medium'."
    )


def _env_value(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return value


TTSModel = Literal[
    "xtts",
    "chatterbox",
    "chatterbox-multilingual",
    "chatterbox-turbo",
]
LLMService = Literal["ollama", "vllm"]


def _resolve_llm_service(value: str) -> LLMService:
    service = value.strip().casefold()
    if service not in {"ollama", "vllm"}:
        raise ValueError("LLM_SERVICE must be one of: ollama, vllm")
    return cast(LLMService, service)


def _resolve_tts_model(value: str) -> TTSModel:
    model = value.strip().casefold()
    supported_models = {
        "xtts",
        "chatterbox",
        "chatterbox-multilingual",
        "chatterbox-turbo",
    }
    if model not in supported_models:
        raise ValueError(
            "TTS_MODEL must be one of: " + ", ".join(sorted(supported_models))
        )
    return cast(TTSModel, model)


@dataclass(frozen=True)
class TTSConfig:
    model: TTSModel
    base_url: str
    voice: str


@dataclass(frozen=True)
class LLMConfig:
    service: LLMService
    base_url: str
    model: str
    api_key: str


@dataclass(frozen=True)
class AppConfig:
    agent_language: Language
    agent_language_name: str
    whisper_model: Model
    llm: LLMConfig
    tts: TTSConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        legacy_ollama_variables = {
            "OLLAMA_BASE_URL",
            "OLLAMA_MODEL",
        }.intersection(os.environ)
        if legacy_ollama_variables:
            variables = ", ".join(sorted(legacy_ollama_variables))
            raise ValueError(
                f"{variables} are no longer supported. Configure Ollama with "
                "LLM_SERVICE=ollama, LLM_BASE_URL, and LLM_MODEL instead."
            )

        language = _resolve_language(os.getenv("AGENT_LANGUAGE", "en"))
        llm_service = _resolve_llm_service(os.getenv("LLM_SERVICE", "ollama"))
        tts_model = _resolve_tts_model(os.getenv("TTS_MODEL", "xtts"))
        if llm_service == "ollama":
            default_llm_base_url = "http://localhost:11434/v1"
            default_llm_model = "gpt-oss:20b"
            default_llm_api_key = "ollama"
        else:
            default_llm_base_url = "http://localhost:8000/v1"
            default_llm_model = "default"
            default_llm_api_key = "vllm"

        if tts_model == "xtts":
            default_tts_base_url = "http://localhost:8000"
            default_tts_voice = "Claribel Dervla"
        else:
            default_tts_base_url = "http://localhost:8004"
            default_tts_voice = "Emily.wav"

        return cls(
            agent_language=language,
            agent_language_name=_language_name(language),
            whisper_model=_resolve_whisper_model(
                _env_value("WHISPER_MODEL_SIZE", "small")
            ),
            llm=LLMConfig(
                service=llm_service,
                base_url=_env_value("LLM_BASE_URL", default_llm_base_url),
                model=_env_value("LLM_MODEL", default_llm_model),
                api_key=_env_value("LLM_API_KEY", default_llm_api_key),
            ),
            tts=TTSConfig(
                model=tts_model,
                base_url=_env_value("TTS_BASE_URL", default_tts_base_url),
                voice=_env_value("TTS_VOICE", default_tts_voice),
            ),
        )


config = AppConfig.from_env()
