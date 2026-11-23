"""Wire format between the Whisper benchmark server and client.

The server uses Pipecat's ``ProtobufFrameSerializer``, the same one the agent
uses in production. The client speaks the same protobuf schema directly:
it sends ``AudioRawFrame`` messages and JSON ``MessageFrame`` messages, and
receives JSON ``MessageFrame`` messages, so it needs no Pipecat pipeline of
its own.

Client -> server messages:

``bench.flush``
    Sent after the last audio frame. ``{audio_bytes}`` is the total PCM the
    client sent. The server answers with ``bench.audio_done`` once its STT
    stage has received that much audio.

Server -> client messages:

``bench.ready``
    The session's pipeline is running; the client may start streaming.
    ``{session, engine, model, language, setup_secs}``
``bench.vad``
    Voice activity changed. ``{event: "started" | "stopped", segment}``
``bench.segment``
    One VAD segment was transcribed. Sent for every stopped segment, also
    when the transcript is empty or transcription failed. ``{segment, text,
    audio_secs, latency_secs, stt_secs, queue_wait_secs, inference_secs,
    batch_wait_secs, batch_size, error}``. The batch fields are set only by
    the batched engine. Timings are measured on the server; ``latency_secs`` runs from
    the VAD stop decision to the moment the report is sent.
``bench.audio_done``
    The STT stage has received all audio named by ``bench.flush``. VAD runs
    before STT, so every segment in that audio has started and, unless the
    speaker was still talking, stopped. ``{audio_secs, segments}``
``bench.error``
    The session failed and the server is closing it. ``{error}``
"""

import json
from typing import Any

from pipecat.frames.protobufs import frames_pb2

SAMPLE_RATE = 16000

READY = "bench.ready"
VAD = "bench.vad"
SEGMENT = "bench.segment"
AUDIO_DONE = "bench.audio_done"
ERROR = "bench.error"
FLUSH = "bench.flush"


def encode_audio(pcm: bytes, *, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap 16-bit mono PCM in a protobuf audio frame."""
    frame = frames_pb2.Frame()
    frame.audio.audio = pcm
    frame.audio.sample_rate = sample_rate
    frame.audio.num_channels = 1
    return frame.SerializeToString()


def encode_message(message: dict[str, Any]) -> bytes:
    """Wrap a JSON message the way ``ProtobufFrameSerializer`` does."""
    frame = frames_pb2.Frame()
    frame.message.data = json.dumps(message)
    return frame.SerializeToString()


def decode_message(data: bytes) -> dict[str, Any] | None:
    """Return the JSON payload of a protobuf message frame, or ``None``."""
    frame = frames_pb2.Frame.FromString(data)
    if frame.WhichOneof("frame") != "message":
        return None
    payload = json.loads(frame.message.data)
    return payload if isinstance(payload, dict) else None
