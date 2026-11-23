# Whisper load test

A speech-to-text-only server and a concurrent client for measuring how Whisper
behaves as sessions are added. They exist to compare Whisper engine modes
(the per-session model the agent uses today, the shared engine, and later
batching) without running the LLM, TTS, or browser.

```text
client.py ── N WebSockets ──> server.py
  audio files, streamed          transport -> Silero VAD -> Whisper STT -> transport
  in real time                   GET /stats: engine queue, CPU, GPU, event-loop lag
```

The server uses the agent's transport, serializer, and VAD, so audio is cut
into segments the same way a real session would cut it.

## Setup

Run everything from the repository root. The client's live view needs the
optional `bench` dependency group:

```sh
uv sync --group bench
```

Put test audio in `benchmarks/whisper/audio/` (ignored by git), or pass any
other files or directories. WAV, FLAC, OGG, MP3, and Opus files of any sample
rate and channel count are accepted; they are mixed to mono and resampled to
16 kHz before the run starts. Use several files with different speakers,
pacing, and pauses: the pauses decide how many segments VAD produces.

Optionally, put a reference transcript next to a file (`clip.wav` +
`clip.txt`) to get its word error rate.

## Run

Start the server in one terminal:

```sh
# One model shared by every session (services/whisper)
uv run python -m benchmarks.whisper.server --engine shared --model small

# Pipecat's WhisperSTTService: one model per connection, like the agent today
uv run python -m benchmarks.whisper.server --engine per-session --model small
```

| Option | Default | Meaning |
|---|---|---|
| `--engine` | `shared` | `shared` or `per-session` |
| `--model` | `WHISPER_MODEL_SIZE` | Any faster-whisper model name or path |
| `--language` | `AGENT_LANGUAGE` | Pipecat language code, e.g. `en`, `tr` |
| `--workers` | `1` | Shared engine: transcriptions that may run at once (each is a model replica) |
| `--device`, `--compute-type` | `auto`, `default` | Passed to faster-whisper |
| `--port` | `7870` | Different from the agent's 7860, so both can run |

Then run the client in another:

```sh
# Each file once, all at the same time
uv run --group bench python -m benchmarks.whisper.client benchmarks/whisper/audio

# Find the breaking point: a baseline per file, then 1, 2, 4, ... sessions
uv run --group bench python -m benchmarks.whisper.client benchmarks/whisper/audio \
    --baseline --ramp 1,2,4,8,16,32
```

| Option | Default | Meaning |
|---|---|---|
| `--sessions N` | one per file | Concurrent sessions; files are assigned round-robin |
| `--ramp 1,2,4` | | Run several session counts, one after another |
| `--baseline` | off | First transcribe each file alone, then report each session's drift from it |
| `--speed` | `1.0` | Streaming speed; 1.0 is real time like a microphone |
| `--stagger` | `0` | Seconds between session starts; 0 connects everyone at once |
| `--max-p95-latency` | `1.5` | Fail a step when p95 latency exceeds this |
| `--max-drift` | off | Fail a step when a transcript's WER against its baseline exceeds this |
| `--keep-going` | off | Continue the ramp after a failed step |
| `--no-live` | off | Plain output instead of the live table (automatic when not a terminal) |

Each run writes a JSON report to `benchmarks/whisper/results/` (ignored by
git) with every session's segments and every `/stats` sample.

## What is measured

Per segment, on the server:

- **latency**: from the moment VAD decides speech stopped to the moment the
  transcript is sent. This is what a user waits for.
- **queue wait** (shared engine): time the request waited for a free engine
  worker.
- **inference**: time the model spent on the segment. In per-session mode
  the whole transcription counts as inference, since there is no queue.

Latency can exceed queue wait plus inference. Pipecat transcribes a session's
segments one at a time, so a segment can also wait behind the same session's
previous segment before it reaches the engine.

Server-wide, sampled twice a second: engine queue and running requests (or
loaded models in per-session mode), segments in flight, active sessions,
system and server CPU, GPU utilization and memory, and event-loop lag. The
client's own CPU is shown too: if it is high, the client is the bottleneck.

**Drift** is the word error rate between a session's transcript and the same
file's transcript when it ran alone (`--baseline`). Sessions that interfere
with each other show up as drift. Small drift without interference is
possible: GPU floating-point results can differ slightly between runs.

A step **fails** when a session cannot connect or finish, a segment errors,
p95 latency exceeds `--max-p95-latency`, or, with `--max-drift`, a transcript
drifts too far.

## Reading the results

- Rising queue wait with flat inference time means the engine is saturated:
  requests arrive faster than one worker can serve them.
- Rising inference time with rising GPU memory, as in per-session mode, means
  models are competing for the GPU; out-of-memory errors follow.
- High loop lag stalls every session's audio. Session setup (creating Silero
  VAD and the pipeline) runs on the event loop, so connecting many sessions at
  once shows up here even when transcription does not. On Windows, about
  15 ms of lag is the timer resolution, not load.
- `--speed` above 1 sends audio faster than a person speaks. Use it to push
  harder, not to estimate how many real users fit.
