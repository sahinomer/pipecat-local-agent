# Whisper engine comparison: results

Two runs on 2026-11-23 with `benchmarks/whisper/matrix.py`, one with Whisper `small` and one with Whisper `medium`.

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB |
| Models | Whisper `small` and `medium`, float16, English, beam size 5 unless noted |
| Software | Pipecat 1.17.0, faster-whisper 1.2.1, CTranslate2 4.8.1, Windows 11 |
| Audio | 72 English dictation recordings, 61.4 minutes, with reference transcripts |

| Variant | Speech-to-text service | Runs |
|---|---|---|
| Pipecat, model per connection | Pipecat's `WhisperSTTService`: each connection loads its own model | small, medium |
| Shared, 1 worker | One model for all sessions; requests run one at a time | small |
| Batched, beam 5, max batch 4 | One model for all sessions; concurrent decode calls run as one GPU batch | small, medium |
| Batched, greedy, max batch 16 | As above, with beam size 1 | small |

## Method

- Each session streams recordings back to back for about 120 seconds, in real time, with 1 second of silence between recordings.
- Session starts are staggered by 0.25 seconds.
- Steps: 1, 4, 8, 12, 16 and 20 concurrent sessions for `small`; 1, 2, 3, 4 and 8 for `medium`. Each variant stops at its first failing step.
- A step passes when the 95th percentile latency is at most 1.0 second, with no failed sessions and no transcription errors.
- Latency is measured on the server, from the voice activity detector's end-of-speech decision to the transcript being sent.
- GPU memory is the peak for the whole device during the step, in GB of 1024 MB. 0.93 GB was in use before the server started.
- Every variant gets a fresh server. The GPU idles 20 seconds between steps and 60 seconds between variants.
- The server runs speech-to-text only. No LLM or TTS model was loaded or running on the GPU.

## Whisper small

### Concurrent sessions

| Variant | Most sessions passing | p95 latency there | GPU memory there | First failing step |
|---|---|---|---|---|
| Pipecat, model per connection | 8 | 0.74 s | 6.93 GB | 12 sessions: p95 425 s, 4 sessions timed out connecting |
| Shared, 1 worker | 1 | 0.53 s | 1.59 GB | 4 sessions: p95 1.05 s |
| Batched, beam 5, max batch 4 | 16 | 0.87 s | 1.92 GB | 20 sessions: p95 1.65 s |
| Batched, greedy, max batch 16 | 16 | 0.42 s | 1.90 GB | 20 sessions: p95 1.30 s |

### GPU memory

Peak GPU memory in GB, whole device.

| Sessions | Pipecat, model per connection | Shared, 1 worker | Batched, beam 5 | Batched, greedy |
|---|---|---|---|---|
| 1 | 1.78 | 1.59 | 1.59 | 1.90 |
| 4 | 4.47 | 1.81 | 1.81 | 1.90 |
| 8 | 6.93 | | 1.81 | 1.90 |
| 12 | 7.89 | | 1.89 | 1.90 |
| 16 | | | 1.92 | 1.90 |
| 20 | | | 1.93 | 2.71 |

With a model per connection, memory grows by 0.74 GB per session from 1 to 8 sessions. With one shared model, it grows by 0.34 GB from 1 to 20 sessions (beam 5).

### Latency

95th percentile latency in seconds, end of speech to transcript.

| Sessions | Pipecat, model per connection | Shared, 1 worker | Batched, beam 5 | Batched, greedy |
|---|---|---|---|---|
| 1 | 0.51 | 0.53 | 0.53 | 0.48 |
| 4 | 0.58 | 1.05 | 0.48 | 0.80 |
| 8 | 0.74 | | 0.72 | 0.41 |
| 12 | 425 | | 0.77 | 0.41 |
| 16 | | | 0.87 | 0.42 |
| 20 | | | 1.65 | 1.30 |

Pipecat, model per connection, at 12 sessions: peak GPU memory 7.89 of 8 GB, median inference time 13.3 s (0.24 s at 8 sessions), session setup 9.8 to 18.4 s, and 4 of 12 clients timed out after the 10 s connect limit. No transcription errors were reported. This run does not separate the effect of GPU memory from that of shared GPU compute.

Mean batch size in the batched variants was 1.00 to 1.57 (beam 5) and 1.00 to 1.25 (greedy).

### Transcripts

Each session's recordings depend only on its index, so the same step, session and recording means the same audio in every variant. WER is the word error rate between transcripts of the same audio.

| Compared | Recordings compared | Recordings that differ | Mean WER |
|---|---|---|---|
| Pipecat with itself, across steps | 40 | 4 | 0.003 |
| Shared, 1 worker vs Pipecat | 18 | 2 | 0.004 |
| Batched, beam 5 vs Pipecat | 70 | 5 | 0.002 |
| Batched, greedy vs Pipecat | 70 | 46 | 0.039 |

Mean WER against the human reference transcripts, over all steps each variant ran: Pipecat 0.164, shared 0.179, batched beam 5 0.140, batched greedy 0.139. Variants that stopped earlier cover fewer recordings.

## Whisper medium

### Concurrent sessions

| Variant | Most sessions passing | p95 latency there | GPU memory there | First failing step |
|---|---|---|---|---|
| Pipecat, model per connection | 2 | 0.72 s | 7.09 GB | 3 sessions: p95 71.3 s |
| Batched, beam 5, max batch 4 | 4 | 0.80 s | 3.15 GB | 8 sessions: p95 1.06 s |

### GPU memory and latency

Peak GPU memory in GB, whole device, and 95th percentile latency in seconds.

| Sessions | Pipecat memory | Pipecat p95 | Batched memory | Batched p95 |
|---|---|---|---|---|
| 1 | 3.21 | 0.80 | 2.89 | 0.86 |
| 2 | 7.09 | 0.72 | 2.89 | 0.73 |
| 3 | 7.86 | 71.3 | 3.13 | 0.77 |
| 4 | | | 3.15 | 0.80 |
| 8 | | | 3.47 | 1.06 |

With a model per connection, the second session added 3.88 GB. With one shared model, memory grows by 0.58 GB from 1 to 8 sessions.

Pipecat, model per connection, at 3 sessions: peak GPU memory 7.86 of 8 GB, median inference time 4.45 s (0.38 s at 2 sessions), session setup up to 12.5 s, and event-loop stalls up to 10.0 s. No sessions failed and no transcription errors were reported. This run does not separate the effect of GPU memory from that of shared GPU compute.

Mean batch size in the batched variant was 1.00 to 1.16. Session setup took at most 0.2 s.

### Transcripts

| Compared | Recordings compared | Recordings that differ | Mean WER |
|---|---|---|---|
| Pipecat with itself, across steps | 14 | 0 | 0.000 |
| Batched, beam 5 vs Pipecat | 25 | 0 | 0.000 |

Mean WER against the human reference transcripts, over all steps each variant ran: Pipecat 0.134, batched beam 5 0.141. The variants cover different numbers of recordings (25 and 65).

## Batch size and GPU memory

Largest batch the scheduler ran in each step, with the step's peak GPU memory in GB.

| Run | Variant | Sessions | Largest batch | GPU memory |
|---|---|---|---|---|
| small | Batched, beam 5, max batch 4 | 1 | 1 | 1.59 |
| small | Batched, beam 5, max batch 4 | 4 | 2 | 1.81 |
| small | Batched, beam 5, max batch 4 | 8 | 3 | 1.81 |
| small | Batched, beam 5, max batch 4 | 12 | 4 | 1.89 |
| small | Batched, beam 5, max batch 4 | 16 | 4 | 1.92 |
| small | Batched, beam 5, max batch 4 | 20 | 4 | 1.93 |
| small | Batched, greedy, max batch 16 | 1 to 16 | 1 to 4 | 1.90 |
| small | Batched, greedy, max batch 16 | 20 | 14 | 2.71 |
| medium | Batched, beam 5, max batch 4 | 1 to 2 | 1 | 2.89 |
| medium | Batched, beam 5, max batch 4 | 3 to 4 | 2 | 3.13 to 3.15 |
| medium | Batched, beam 5, max batch 4 | 8 | 4 | 3.47 |

In the batched variants, steps that reached the same largest batch had GPU memory within 0.04 GB of each other, whatever their number of sessions.

## Reproduce

```sh
uv run --group bench python -m benchmarks.whisper.matrix --model small
uv run --group bench python -m benchmarks.whisper.matrix --model medium \
    --variants pipecat-per-session,batched-beam5-b4 --ramp 1,2,3,4,8,12,16,20,24,28,32
```

Per-step data for every variant: `benchmarks/whisper/results/matrix-final/` and `benchmarks/whisper/results/matrix-medium/` (not committed).
