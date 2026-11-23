"""Result types, statistics and pass/fail rules for Whisper load tests."""

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SegmentResult:
    """One ``bench.segment`` report, as the server measured it."""

    segment: int
    text: str
    audio_secs: float
    latency_secs: float | None
    stt_secs: float
    queue_wait_secs: float | None
    inference_secs: float | None
    error: str | None
    batch_wait_secs: float | None = None
    batch_size: int | None = None
    start_audio_secs: float | None = None
    end_audio_secs: float | None = None

    @classmethod
    def from_message(cls, message: dict[str, Any]) -> "SegmentResult":
        return cls(
            segment=int(message["segment"]),
            text=str(message.get("text") or ""),
            audio_secs=float(message.get("audio_secs") or 0.0),
            latency_secs=message.get("latency_secs"),
            stt_secs=float(message.get("stt_secs") or 0.0),
            queue_wait_secs=message.get("queue_wait_secs"),
            inference_secs=message.get("inference_secs"),
            error=message.get("error"),
            batch_wait_secs=message.get("batch_wait_secs"),
            batch_size=message.get("batch_size"),
            start_audio_secs=message.get("start_audio_secs"),
            end_audio_secs=message.get("end_audio_secs"),
        )


@dataclass(frozen=True)
class PlaylistItem:
    """One file inside a session's audio stream.

    Parameters:
        clip: File name.
        offset_secs: Where the file starts in the session's stream.
        duration_secs: The file's length.
    """

    clip: str
    offset_secs: float
    duration_secs: float


# A segment whose speech starts this soon before a file's offset still
# belongs to that file: VAD positions are estimates, and files are separated
# by a silence gap longer than this.
_CLIP_BOUNDARY_SLACK_SECS = 0.5


@dataclass
class SessionResult:
    """Everything one client session observed."""

    session: str
    clip: str
    state: str = "pending"
    setup_secs: float | None = None
    stream_secs: float | None = None
    drain_secs: float | None = None
    vad_started: int = 0
    vad_stopped: int = 0
    segments: list[SegmentResult] = field(default_factory=list)
    failure: str | None = None
    playlist: list[PlaylistItem] = field(default_factory=list)

    @property
    def transcript(self) -> str:
        return " ".join(s.text for s in sorted(self.segments, key=lambda s: s.segment) if s.text)

    def clip_transcripts(self) -> list[tuple[str, str]]:
        """The transcript of each file the session streamed, in order.

        Segments are matched to files by where their speech started in the
        stream. A session that streamed one file gets its whole transcript.
        """
        items = self.playlist or [PlaylistItem(self.clip, 0.0, 0.0)]
        if len(items) == 1:
            return [(items[0].clip, self.transcript)]
        texts: list[list[str]] = [[] for _ in items]
        for segment in sorted(self.segments, key=lambda s: s.segment):
            if not segment.text or segment.start_audio_secs is None:
                continue
            position = segment.start_audio_secs + _CLIP_BOUNDARY_SLACK_SECS
            index = max(
                (i for i, item in enumerate(items) if item.offset_secs <= position),
                default=0,
            )
            texts[index].append(segment.text)
        return [(item.clip, " ".join(t)) for item, t in zip(items, texts)]

    @property
    def segment_errors(self) -> list[str]:
        return [s.error for s in self.segments if s.error]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transcript"] = self.transcript
        data["clip_transcripts"] = [
            {"clip": clip, "transcript": text} for clip, text in self.clip_transcripts()
        ]
        return data


@dataclass(frozen=True)
class Thresholds:
    """When a load step counts as failed."""

    max_p95_latency_secs: float = 1.0
    max_wer_drift: float | None = None


def normalize_words(text: str) -> list[str]:
    """Lower-case words without punctuation, for word error rate.

    ``casefold`` is not locale-aware, so Turkish dotted and dotless I count as
    different letters.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance divided by the reference length.

    An empty reference gives 0.0 for an empty hypothesis and 1.0 otherwise.
    """
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_word != hyp_word),
            )
        previous = current
    return previous[-1] / len(ref)


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile; ``None`` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _peak(samples: list[dict[str, Any]], getter) -> float | None:
    values = []
    for sample in samples:
        try:
            value = getter(sample)
        except (KeyError, IndexError, TypeError):
            continue
        if value is not None:
            values.append(float(value))
    return max(values) if values else None


def gpu_peak(samples: list[dict[str, Any]], key: str) -> float | None:
    """The highest value of ``key`` across all GPUs and samples."""
    return _peak(
        samples,
        lambda s: max((g[key] for g in s["system"]["gpus"]), default=None),
    )


def _busy_clock_min(samples: list[dict[str, Any]]) -> float | None:
    """The lowest SM clock seen while a GPU was busy; a drop means throttling."""
    clocks = [
        gpu["sm_clock_mhz"]
        for s in samples
        for gpu in s.get("system", {}).get("gpus", [])
        if gpu.get("sm_clock_mhz") and gpu.get("utilization_percent", 0) >= 50
    ]
    return float(min(clocks)) if clocks else None


def _engine_delta(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [s["engine"][key] for s in samples if key in s.get("engine", {})]
    return values[-1] - values[0] if len(values) >= 2 else None


def summarize_step(
    sessions: list[SessionResult],
    stats_samples: list[dict[str, Any]],
    *,
    baseline: dict[str, str] | None = None,
    references: dict[str, str] | None = None,
    thresholds: Thresholds = Thresholds(),
    client_cpu_peak: float | None = None,
) -> dict[str, Any]:
    """Aggregate one load step and decide whether it passed."""
    segments = [seg for s in sessions for seg in s.segments]
    ok_segments = [seg for seg in segments if not seg.error]
    latencies = [seg.latency_secs for seg in ok_segments if seg.latency_secs is not None]
    queue_waits = [seg.queue_wait_secs for seg in ok_segments if seg.queue_wait_secs is not None]
    inference = [seg.inference_secs for seg in ok_segments if seg.inference_secs is not None]
    batch_waits = [seg.batch_wait_secs for seg in ok_segments if seg.batch_wait_secs is not None]
    batches = _engine_delta(stats_samples, "batches")
    batched_items = _engine_delta(stats_samples, "batched_items")
    rtf = [
        seg.inference_secs / seg.audio_secs
        for seg in ok_segments
        if seg.inference_secs is not None and seg.audio_secs > 0
    ]

    clip_texts = [pair for s in sessions if s.state == "done" for pair in s.clip_transcripts()]
    drift = [
        word_error_rate(baseline[clip], text)
        for clip, text in clip_texts
        if baseline and clip in baseline
    ]
    reference_wer = [
        word_error_rate(references[clip], text)
        for clip, text in clip_texts
        if references and clip in references
    ]

    failed_sessions = [s for s in sessions if s.failure]
    segment_errors = [e for s in sessions for e in s.segment_errors]
    p95_latency = percentile(latencies, 95)

    reasons = []
    if failed_sessions:
        reasons.append(f"{len(failed_sessions)} session(s) failed")
    if segment_errors:
        reasons.append(f"{len(segment_errors)} segment error(s)")
    if p95_latency is not None and p95_latency > thresholds.max_p95_latency_secs:
        reasons.append(
            f"p95 latency {p95_latency:.2f}s > {thresholds.max_p95_latency_secs:.2f}s"
        )
    if thresholds.max_wer_drift is not None and drift and max(drift) > thresholds.max_wer_drift:
        reasons.append(f"transcript drift {max(drift):.2f} > {thresholds.max_wer_drift:.2f}")

    return {
        "sessions": len(sessions),
        "sessions_failed": len(failed_sessions),
        "segments": len(segments),
        "segments_empty": sum(1 for seg in ok_segments if not seg.text),
        "segment_errors": len(segment_errors),
        "audio_secs": sum(seg.audio_secs for seg in segments),
        "latency_p50": percentile(latencies, 50),
        "latency_p95": p95_latency,
        "latency_p99": percentile(latencies, 99),
        "latency_max": max(latencies) if latencies else None,
        "queue_wait_p50": percentile(queue_waits, 50),
        "queue_wait_p95": percentile(queue_waits, 95),
        "queue_wait_max": max(queue_waits) if queue_waits else None,
        "inference_p50": percentile(inference, 50),
        "inference_p95": percentile(inference, 95),
        "batch_wait_p95": percentile(batch_waits, 95),
        "batch_size_mean": batched_items / batches if batches else None,
        "peak_batch_size": _peak(stats_samples, lambda s: s["engine"].get("last_batch_size")),
        "peak_batch_pending": _peak(stats_samples, lambda s: s["engine"].get("batch_pending")),
        "rtf_p50": percentile(rtf, 50),
        "setup_max": max((s.setup_secs for s in sessions if s.setup_secs is not None), default=None),
        "drift_wer_mean": sum(drift) / len(drift) if drift else None,
        "drift_wer_max": max(drift) if drift else None,
        "drift_sessions": sum(1 for d in drift if d > 0),
        "reference_wer_mean": sum(reference_wer) / len(reference_wer) if reference_wer else None,
        "peak_queued": _peak(stats_samples, lambda s: s["engine"].get("queued")),
        "peak_running": _peak(stats_samples, lambda s: s["engine"].get("running")),
        "peak_in_flight": _peak(stats_samples, lambda s: s["segments"]["in_flight"]),
        "peak_models_loaded": _peak(stats_samples, lambda s: s["engine"].get("models_loaded")),
        "peak_cpu_percent": _peak(stats_samples, lambda s: s["system"]["cpu_percent"]),
        "peak_server_cpu_percent": _peak(
            stats_samples, lambda s: s["system"]["process_cpu_percent"]
        ),
        "peak_gpu_percent": gpu_peak(stats_samples, "utilization_percent"),
        "peak_gpu_memory_mb": gpu_peak(stats_samples, "memory_used_mb"),
        "peak_gpu_temp_c": gpu_peak(stats_samples, "temperature_c"),
        "peak_gpu_power_w": gpu_peak(stats_samples, "power_w"),
        "gpu_sm_clock_min_busy_mhz": _busy_clock_min(stats_samples),
        "peak_loop_lag_ms": _peak(stats_samples, lambda s: s["loop_lag_ms"]),
        "peak_client_cpu_percent": client_cpu_peak,
        "passed": not reasons,
        "reasons": reasons,
    }
