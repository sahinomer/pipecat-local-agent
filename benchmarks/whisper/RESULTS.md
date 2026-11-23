# Whisper engine comparison: results

Run on 2026-11-23 with `benchmarks/whisper/matrix.py`.

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU, 8 GB |
| Model | Whisper `small`, float16, English, beam size 5 unless noted |
| Software | Pipecat 1.17.0, faster-whisper 1.2.1, CTranslate2 4.8.1, Windows 11 |
| Audio | 72 English dictation recordings, 61.4 minutes, with reference transcripts |

| Variant | Speech-to-text service |
|---|---|
| Pipecat, model per connection | Pipecat's `WhisperSTTService`: each connection loads its own model |
| Shared, 1 worker | One model for all sessions; requests run one at a time |
| Batched, beam 5, max batch 4 | One model for all sessions; concurrent decode calls run as one GPU batch |
| Batched, greedy, max batch 16 | As above, with beam size 1 |

## Method

- Each session streams recordings back to back for about 120 seconds, in real time, with 1 second of silence between recordings.
- Session starts are staggered by 0.25 seconds.
- Steps of 1, 4, 8, 12, 16 and 20 concurrent sessions. Each variant stops at its first failing step.
- A step passes when the 95th percentile latency is at most 1.0 second, with no failed sessions and no transcription errors.
- Latency is measured on the server, from the voice activity detector's end-of-speech decision to the transcript being sent.
- GPU memory is the peak for the whole device during the step, in GB of 1024 MB. 0.93 GB was in use before the server started.
- Every variant gets a fresh server. The GPU idles 20 seconds between steps and 60 seconds between variants.

## Concurrent sessions

| Variant | Most sessions passing | p95 latency there | GPU memory there | First failing step |
|---|---|---|---|---|
| Pipecat, model per connection | 8 | 0.74 s | 6.93 GB | 12 sessions: GPU memory full, 4 sessions failed, p95 425 s |
| Shared, 1 worker | 1 | 0.53 s | 1.59 GB | 4 sessions: p95 1.05 s |
| Batched, beam 5, max batch 4 | 16 | 0.87 s | 1.92 GB | 20 sessions: p95 1.65 s |
| Batched, greedy, max batch 16 | 16 | 0.42 s | 1.90 GB | 20 sessions: p95 1.30 s |

## GPU memory

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

## Latency

95th percentile latency in seconds, end of speech to transcript.

| Sessions | Pipecat, model per connection | Shared, 1 worker | Batched, beam 5 | Batched, greedy |
|---|---|---|---|---|
| 1 | 0.51 | 0.53 | 0.53 | 0.48 |
| 4 | 0.58 | 1.05 | 0.48 | 0.80 |
| 8 | 0.74 | | 0.72 | 0.41 |
| 12 | 425 | | 0.77 | 0.41 |
| 16 | | | 0.87 | 0.42 |
| 20 | | | 1.65 | 1.30 |

Mean batch size in the batched variants was 1.00 to 1.57 (beam 5) and 1.00 to 1.25 (greedy).

## Transcripts

Each session's recordings depend only on its index, so the same step, session and recording means the same audio in every variant. WER is the word error rate between transcripts of the same audio.

| Compared | Recordings compared | Recordings that differ | Mean WER |
|---|---|---|---|
| Pipecat with itself, across steps | 40 | 4 | 0.003 |
| Shared, 1 worker vs Pipecat | 18 | 2 | 0.004 |
| Batched, beam 5 vs Pipecat | 70 | 5 | 0.002 |
| Batched, greedy vs Pipecat | 70 | 46 | 0.039 |

Mean WER against the human reference transcripts, over all steps each variant ran: Pipecat 0.164, shared 0.179, batched beam 5 0.140, batched greedy 0.139. Variants that stopped earlier cover fewer recordings.

## Reproduce

```sh
uv run --group bench python -m benchmarks.whisper.matrix --max-files 72
```

Per-step data for every variant: `benchmarks/whisper/results/matrix-final/` (not committed).
