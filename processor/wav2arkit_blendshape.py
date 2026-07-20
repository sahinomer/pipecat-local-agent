import asyncio
import hashlib
import math
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional, Protocol

import numpy as np
import onnxruntime as ort
from loguru import logger
from onnxruntime import InferenceSession
from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from scipy.signal import resample_poly

MODEL_SAMPLE_RATE = 16000
MODEL_CACHE_DIR = Path.cwd() / ".cache" / "wav2arkit"


@dataclass(frozen=True)
class ModelArtifact:
    path: Path
    url: str
    sha256: str


MODEL_ARTIFACTS = (
    ModelArtifact(
        path=MODEL_CACHE_DIR / "wav2arkit_cpu.onnx",
        url=(
            "https://huggingface.co/myned-ai/wav2arkit_cpu/resolve/main/"
            "wav2arkit_cpu.onnx?download=true"
        ),
        sha256=(
            "cdecbfad3915dd20b2f0718942d0b8894b2ee11edcc5a9a9da45d29a46af2ed9"
        ),
    ),
    ModelArtifact(
        path=MODEL_CACHE_DIR / "wav2arkit_cpu.onnx.data",
        url=(
            "https://huggingface.co/myned-ai/wav2arkit_cpu/resolve/main/"
            "wav2arkit_cpu.onnx.data?download=true"
        ),
        sha256=(
            "c0f0364673c6e50be126b193e2b56809c16ac6bee4805aea9b8251ce53429bf8"
        ),
    ),
)


class BlendshapeModel(Protocol):
    def predict(self, audio_input: np.ndarray) -> list[list[float]]: ...


class Wav2ArkitBlendshapeModel:
    """Load and share the wav2arkit CPU ONNX model.

    Model reference:
        https://huggingface.co/myned-ai/wav2arkit_cpu

    The first instance downloads missing model artifacts and creates the ONNX
    session. Later instances reuse that session, so creating a processor for
    every connection does not reload the roughly 400 MB of model weights.

    ``session`` is retained as a test/compatibility seam. Normal application
    code should construct the model without arguments and let it manage its
    own files and session.
    """

    _shared_session: ClassVar[Optional[InferenceSession]] = None
    _load_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, *, session: Optional[InferenceSession] = None):
        self._session: InferenceSession = session if session is not None else self._get_shared_session()

    @classmethod
    def _get_shared_session(cls) -> InferenceSession:
        if cls._shared_session is None:
            with cls._load_lock:
                if cls._shared_session is None:
                    model_path = cls._ensure_model_files()
                    logger.info("Loading wav2arkit model from {}", model_path)
                    cls._shared_session = ort.InferenceSession(
                        str(model_path), providers=["CPUExecutionProvider"]
                    )
        return cls._shared_session

    @classmethod
    def _ensure_model_files(cls) -> Path:
        for artifact in MODEL_ARTIFACTS:
            artifact.path.parent.mkdir(parents=True, exist_ok=True)
            if not cls._has_expected_hash(artifact):
                cls._download_file(artifact)

        return MODEL_ARTIFACTS[0].path

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _has_expected_hash(cls, artifact: ModelArtifact) -> bool:
        if not artifact.path.is_file():
            return False

        actual_hash = cls._sha256(artifact.path)
        if actual_hash != artifact.sha256:
            logger.warning(
                "Hash mismatch for {}; expected {}, got {}. Re-downloading.",
                artifact.path,
                artifact.sha256,
                actual_hash,
            )
            return False
        return True

    @classmethod
    def _download_file(cls, artifact: ModelArtifact) -> None:
        temporary_path = artifact.path.with_name(f"{artifact.path.name}.part")
        logger.info("Downloading wav2arkit model artifact to {}", artifact.path)
        try:
            urllib.request.urlretrieve(artifact.url, temporary_path)
            actual_hash = cls._sha256(temporary_path)
            if actual_hash != artifact.sha256:
                raise RuntimeError(
                    f"hash mismatch for downloaded model artifact "
                    f"{artifact.path}: expected {artifact.sha256}, got {actual_hash}"
                )
            temporary_path.replace(artifact.path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def predict(self, audio_input: np.ndarray) -> list[list[float]]:
        output = np.asarray(
            self._session.run(None, {"audio_waveform": audio_input})[0]
        )

        # wav2arkit returns [batch, animation_frames, blendshape_coefficients].
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2:
            raise ValueError(
                f"unexpected wav2arkit output shape: {tuple(output.shape)}"
            )

        return output.astype(float).tolist()


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
        self._model = model or Wav2ArkitBlendshapeModel(session=session)

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

        try:
            blendshapes = await asyncio.to_thread(self._infer_blendshapes, frame)
        except Exception:
            logger.exception("{}: wav2arkit inference failed", self)
            return

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
