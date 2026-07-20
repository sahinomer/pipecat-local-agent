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
class AppConfig:
    agent_language: Language
    agent_language_name: str
    whisper_model: Model
    ollama_base_url: str
    ollama_model: str
    tts: TTSConfig

    @classmethod
    def from_env(cls) -> "AppConfig":
        language = _resolve_language(os.getenv("AGENT_LANGUAGE", "en"))
        tts_model = _resolve_tts_model(os.getenv("TTS_MODEL", "xtts"))
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
            ollama_base_url=_env_value(
                "OLLAMA_BASE_URL", "http://localhost:11434/v1"
            ),
            ollama_model=_env_value("OLLAMA_MODEL", "gpt-oss:20b"),
            tts=TTSConfig(
                model=tts_model,
                base_url=_env_value("TTS_BASE_URL", default_tts_base_url),
                voice=_env_value("TTS_VOICE", default_tts_voice),
            ),
        )


config = AppConfig.from_env()
