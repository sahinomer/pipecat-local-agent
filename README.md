# Pipecat Local Agent

Pipecat Local Agent is a small, independent, local-first voice-agent demo built with [Pipecat](https://github.com/pipecat-ai/pipecat). Pipecat provides the pipeline, transport, turn handling, and service integrations; this repository connects those pieces to local speech-to-text, language-model, and text-to-speech services, with real-time facial blendshapes as its first extension.

## Current capabilities

- Streams microphone input and synthesized speech over a Pipecat WebSocket transport.
- Uses faster-whisper for local speech-to-text, selectable Ollama or vLLM for the language model, and a selectable XTTS or Chatterbox server for text-to-speech.
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
- A running [Ollama](https://github.com/ollama/ollama) or
  [vLLM](https://docs.vllm.ai/) server with the model named by `LLM_MODEL`.
- One supported TTS server:
  - [XTTS Streaming Server](https://github.com/coqui-ai/xtts-streaming-server); or
  - [Chatterbox TTS Server](https://github.com/devnen/Chatterbox-TTS-Server).

Models and assets:

- Make the configured `LLM_MODEL` available to the selected LLM server. The default Ollama model is `gpt-oss:20b`.
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

The example enables XTTS by default and includes a commented Chatterbox configuration. Keep only one TTS configuration block active.

| Variable | Purpose | Example default |
| --- | --- | --- |
| `AGENT_LANGUAGE` | BCP-47 language code used by prompts, STT, and TTS | `en` |
| `WHISPER_MODEL_SIZE` | Pipecat faster-whisper model name | `small` |
| `LLM_SERVICE` | LLM backend: `ollama` or `vllm` | `ollama` |
| `LLM_BASE_URL` | Selected backend's OpenAI-compatible endpoint | `http://localhost:11434/v1` |
| `LLM_MODEL` | Model exposed by the selected backend | `gpt-oss:20b` |
| `LLM_API_KEY` | vLLM API key, or a placeholder when vLLM auth is disabled | `ollama` |
| `TTS_MODEL` | TTS model: `xtts`, `chatterbox`, `chatterbox-multilingual`, or `chatterbox-turbo` | `xtts` |
| `TTS_BASE_URL` | Base URL for the selected TTS model | `http://localhost:8000` |
| `TTS_VOICE` | Voice name or filename for the selected TTS model | `Claribel Dervla` |
| `OBSERVABILITY_ENABLED` | Enable the Prometheus exporter and Pipecat metrics observer | `true` |
| `METRICS_HOST` | Address for the metrics-only HTTP server | `127.0.0.1` |
| `METRICS_PORT` | Port for the Prometheus `/metrics` endpoint | `8001` |
| `GPU_METRICS_ENABLED` | Collect whole-GPU metrics through NVIDIA NVML when available | `true` |

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

If the Ollama desktop app or service is already running, there is no need to start a second server. If you changed `LLM_MODEL`, pull that model instead.

To use an already-running vLLM OpenAI-compatible server instead, configure:

```env
LLM_SERVICE=vllm
LLM_BASE_URL=http://localhost:8000/v1
LLM_MODEL=your-model-name
LLM_API_KEY=vllm
```

`LLM_MODEL` must match the model name exposed by vLLM. If the server was
started with API-key authentication, set the same key in `LLM_API_KEY`;
otherwise, a non-empty placeholder such as `vllm` is sufficient for the
OpenAI client.

In another terminal, start the TTS backend selected by `TTS_MODEL`.

For XTTS, review the [Coqui model license](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt) before setting `COQUI_TOS_AGREED=1`:

```sh
docker run --gpus=all -e COQUI_TOS_AGREED=1 --rm -p 8000:80 ghcr.io/coqui-ai/xtts-streaming-server:latest-cuda121
```

For Chatterbox, follow the server's hardware-specific installation instructions. Its default Docker Compose setup can be started from a separate checkout:

```sh
git clone https://github.com/devnen/Chatterbox-TTS-Server.git
cd Chatterbox-TTS-Server
docker compose up -d --build
```

Then configure the active TTS model and choose a filename returned by `GET /v1/audio/voices`:

```env
TTS_MODEL=chatterbox
TTS_BASE_URL=http://localhost:8004
TTS_VOICE=Emily.wav
```

`TTS_MODEL` is also sent as the OpenAI-compatible `model` field. For languages other than English, use `TTS_MODEL=chatterbox-multilingual` and ensure the Chatterbox server has its Multilingual engine active. The current server implementation still controls its active engine server-side.

The Chatterbox OpenAI-compatible route returns a complete encoded audio file. The local adapter decodes its WAV response, resamples it when necessary, and emits raw PCM frames for Pipecat. The route does not provide token-level audio streaming; for short voice-agent utterances, audio begins after Chatterbox finishes synthesizing the utterance.

Then start the backend from the repository root:

```sh
uv run python server.py
```

For browser-client installation and startup, follow [`frontend/README.md`](frontend/README.md).

## Local observability

The agent exposes Prometheus metrics from a separate metrics-only HTTP server.
The application server remains bound to `127.0.0.1:7860`, while metrics use
the address and port configured by `METRICS_HOST` and `METRICS_PORT`.

The exporter includes:

- Active and total connections, duration, and termination outcomes.
- Whole-host CPU and RAM, agent-process CPU and RAM, and thread count.
- Whole-GPU utilization, memory, temperature, and power through NVIDIA NVML.
  GPU panels remain empty on systems without a supported NVIDIA driver.
- Pipecat processing, TTFB, TTFA, text aggregation, token, TTS usage, turn,
  and error metrics.
- Any processor that opts into Pipecat processing metrics, including
  `Wav2ArkitBlendshapeProcessor`, appears automatically under its processor
  name. Custom processors do not depend on the observability implementation.

System CPU/RAM metrics describe the machine and the Pipecat process. GPU
metrics describe each complete GPU. They do not attribute usage to individual
Ollama, vLLM, or TTS processes.

### Start Prometheus and Grafana

Prometheus in Docker Desktop reaches the host through
`host.docker.internal`. The metrics server must therefore listen beyond the
loopback interface. Set this in the local `.env` while using the containerized
dashboard:

```env
OBSERVABILITY_ENABLED=true
METRICS_HOST=0.0.0.0
METRICS_PORT=8001
GPU_METRICS_ENABLED=true
```

This exposes read-only system and agent metrics on the local network interface.
Do not publish port `8001` through a router or expose it to an untrusted
network. Keep `METRICS_HOST=127.0.0.1` when Prometheus runs directly on the
host or observability is not in use.

Start the backend, then verify the metrics endpoint:

```powershell
uv run python server.py
Invoke-WebRequest http://127.0.0.1:8001/metrics
```

Start the provisioned observability stack in another terminal:

```sh
docker compose -f compose.observability.yaml up -d
```

Open Grafana at `http://localhost:3000`, sign in with `admin` / `admin`, then
open **Dashboards → Pipecat Local Agent → Local Agent Performance**. Prometheus
is available at `http://localhost:9090`. Both services bind their published
ports to host loopback, and their data is retained in named Docker volumes.

Stop the stack without removing its stored data:

```sh
docker compose -f compose.observability.yaml down
```

### Optional native vLLM metrics

The dashboard always derives token throughput from Pipecat usage metrics.
When vLLM is selected, its native Prometheus endpoint provides more precise
generation throughput and time-per-output-token measurements.

Edit `observability/prometheus/targets/vllm.json` to point to the vLLM API
server:

```json
[
  {
    "targets": ["host.docker.internal:8000"],
    "labels": {
      "source": "vllm"
    }
  }
]
```

Use the actual vLLM port when it differs from `8000`. Prometheus reloads this
file automatically. Leave the file as `[]` for Ollama or when native vLLM
metrics are not required.


## License

Code and documentation original to this repository are licensed under the [MIT License](LICENSE).
