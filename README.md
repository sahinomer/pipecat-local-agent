# Pipecat Local Agent

Pipecat Local Agent is a small, independent, local-first voice-agent demo built with [Pipecat](https://github.com/pipecat-ai/pipecat). Pipecat provides the pipeline, transport, turn handling, and service integrations; this repository connects those pieces to local speech-to-text, language-model, and text-to-speech services, with real-time facial blendshapes as its first extension.

## Current capabilities

- Streams microphone input and synthesized speech over a Pipecat WebSocket transport.
- Uses faster-whisper for local speech-to-text, Ollama for the language model, and an XTTS Streaming Server for text-to-speech.
- Uses Pipecat's bundled Silero VAD and Local Smart Turn model for voice activity and end-of-turn handling.
- Starts the assistant automatically when a browser client connects.
- Shows user and assistant transcripts in the browser client.
- Generates ARKit-style blendshape coefficients from TTS audio and animates a user-supplied GLB avatar in real time.
- Applies one configured language to the prompts, speech-to-text, and text-to-speech services.

## How it works

```text
browser microphone
       |
       | PCM audio over WebSocket (Pipecat protobuf)
       v
Pipecat transport -> RTVI -> STT -> context + turn handling -> LLM -> TTS
                                                                       |
                                                                       v
                                                              blendshape processor
                                                                |           |
                                 unchanged TTS audio <----------+           +----> blendshape frames
                                          |                                                |
                                          v                                                v
                                   browser playback                                   3D frontend
```

The FastAPI backend exposes one WebSocket endpoint at `ws://localhost:7860/ws`. Pipecat carries audio, transcripts, lifecycle events, and custom blendshape messages between the backend and browser.

> **Note:** The default server binds only to `127.0.0.1` and does not implement authentication. Do not expose it directly to an untrusted network.

## Extensions

### wav2arkit blendshape processor

`Wav2ArkitBlendshapeProcessor` receives downstream `TTSAudioRawFrame` objects immediately after TTS. It forwards every audio frame unchanged, while preparing a mono 16 kHz float32 copy for the [wav2arkit CPU](https://huggingface.co/myned-ai/wav2arkit_cpu) ONNX model.

Inference produces batches of 52-value ARKit coefficient frames. The processor wraps each batch in an RTVI server message with `type`, `data`, and the source audio presentation timestamp, then sends it through the same Pipecat WebSocket transport.

The frontend receives these custom server messages and queues their coefficient frames to animate the avatar. Client-specific rendering and mapping details are kept in the [frontend guide](frontend/README.md).

## Requirements

Required for the backend:

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/) for the locked Python environment.
- A running [Ollama](https://github.com/ollama/ollama) server with the model named by `OLLAMA_MODEL`.
- A running [XTTS Streaming Server](https://github.com/coqui-ai/xtts-streaming-server) with the voice named by `XTTS_VOICE`.

Models and assets:

- Pull the configured Ollama model explicitly. The default is `gpt-oss:20b`.
- faster-whisper downloads the configured Whisper model on first use. The default is `small`.
- wav2arkit downloads two hash-checked ONNX artifacts on first connection and stores them in `.cache/wav2arkit` when the backend is launched from the repository root.
- Silero VAD and Local Smart Turn model files are bundled with the installed Pipecat package; they need no separate setup.
- No avatar model is included. Copy a compatible GLB model to `frontend/public/avatar.glb`; see the [frontend guide](frontend/README.md) for the expected morph targets and replacement instructions.

Hardware notes:

- A capable CPU and GPU are recommended for smooth, real-time, low-latency use.
- The browser demo needs microphone access, audio output, and WebGL support.

The browser client requires npm and a Vite-supported Node.js release: Node.js `20.19` or newer in the 20.x line, or Node.js `22.12` or newer.

## Configuration

Create `.env` from the checked-in example and change values to match your local services. Do not commit `.env`.

```sh
cp .env.example .env
```

| Variable | Purpose | Example default |
| --- | --- | --- |
| `AGENT_LANGUAGE` | BCP-47 language code used by prompts, STT, and TTS | `en` |
| `WHISPER_MODEL_SIZE` | Pipecat faster-whisper model name | `small` |
| `OLLAMA_BASE_URL` | Ollama's OpenAI-compatible endpoint | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Model served by Ollama | `gpt-oss:20b` |
| `XTTS_BASE_URL` | XTTS Streaming Server endpoint | `http://localhost:8000` |
| `XTTS_VOICE` | Voice exposed by the XTTS server | `Claribel Dervla` |

The frontend has a separate optional `VITE_BACKEND_URL`; it is documented in the [frontend guide](frontend/README.md).

## Running the project

Install the locked backend dependencies from the repository root:

```sh
uv sync --locked
```

Pull the default Ollama model, then ensure Ollama is running:

```sh
ollama pull gpt-oss:20b
ollama serve
```

If the Ollama desktop app or service is already running, there is no need to start a second server. If you changed `OLLAMA_MODEL`, pull that model instead.

In another terminal, start the XTTS server. Review the [Coqui model license](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt) before setting `COQUI_TOS_AGREED=1`:

```sh
docker run --gpus=all -e COQUI_TOS_AGREED=1 --rm -p 8000:80 ghcr.io/coqui-ai/xtts-streaming-server:latest-cuda121
```

Then start the backend from the repository root:

```sh
uv run python server.py
```

For browser-client installation and startup, follow [`frontend/README.md`](frontend/README.md).


## License

Code and documentation original to this repository are licensed under the [MIT License](LICENSE).
