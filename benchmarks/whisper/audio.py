"""Load benchmark audio files as 16 kHz mono 16-bit PCM."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
from scipy.signal import resample_poly

from .protocol import SAMPLE_RATE

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".opus")


@dataclass(frozen=True)
class AudioClip:
    """One decoded input file.

    Parameters:
        name: File name, used to label sessions and match baselines.
        path: Where the file came from.
        pcm: 16 kHz mono little-endian int16 samples.
        reference: Transcript from a sibling ``.txt`` file, if there is one.
    """

    name: str
    path: Path
    pcm: bytes
    reference: str | None = None

    @property
    def duration_secs(self) -> float:
        return len(self.pcm) / (SAMPLE_RATE * 2)


def find_audio_files(inputs: list[str]) -> list[Path]:
    """Expand files and directories into a sorted, de-duplicated file list.

    Directories are searched recursively for known audio extensions.

    Raises:
        FileNotFoundError: If an input does not exist or nothing was found.
    """
    found: dict[Path, None] = {}
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.suffix.lower() in AUDIO_EXTENSIONS and candidate.is_file():
                    found[candidate.resolve()] = None
        elif path.is_file():
            found[path.resolve()] = None
        else:
            raise FileNotFoundError(f"audio input not found: {raw}")
    if not found:
        raise FileNotFoundError("no audio files found in: " + ", ".join(inputs))
    return list(found)


def load_clip(path: Path) -> AudioClip:
    """Decode ``path``, mix to mono and resample to 16 kHz."""
    data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if sample_rate != SAMPLE_RATE and audio.size:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sample_rate // divisor)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    reference_path = path.with_suffix(".txt")
    reference = None
    if reference_path.is_file():
        reference = reference_path.read_text(encoding="utf-8-sig").strip() or None
    return AudioClip(name=path.name, path=path, pcm=pcm, reference=reference)


def split_chunks(pcm: bytes, *, chunk_ms: int) -> list[bytes]:
    """Split PCM into transport-sized chunks; the last one may be shorter."""
    size = SAMPLE_RATE * 2 * chunk_ms // 1000
    if size <= 0:
        raise ValueError("chunk_ms is too small")
    return [pcm[i : i + size] for i in range(0, len(pcm), size)]
