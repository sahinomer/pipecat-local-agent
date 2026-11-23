"""Load client for the Whisper benchmark server.

Streams audio files over concurrent WebSocket sessions at real-time pace,
shows each session's transcript and the server's telemetry live, and writes
a JSON report. With ``--ramp`` it repeats the run at increasing session
counts to find where the server stops keeping up.

Run from the repository root, with the server already running:

    uv run --group bench python -m benchmarks.whisper.client benchmarks/whisper/audio --ramp 1,2,4,8 --baseline
"""

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

# Pipecat logs a banner when its protobuf module is imported; keep the
# client's output to its own.
from loguru import logger

logger.remove()

import aiohttp  # noqa: E402
import psutil  # noqa: E402

from . import protocol  # noqa: E402
from .audio import AudioClip, find_audio_files, load_clip, split_chunks  # noqa: E402
from .report import (  # noqa: E402
    SegmentResult,
    SessionResult,
    Thresholds,
    summarize_step,
    word_error_rate,
)

RESULTS_DIR = Path(__file__).parent / "results"
MAX_LIVE_ROWS = 30


@dataclass(frozen=True)
class Options:
    ws_url: str
    stats_url: str
    chunk_ms: int
    speed: float
    tail_silence_secs: float
    stagger_secs: float
    connect_timeout: float
    ready_timeout: float
    drain_timeout: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent speech-to-text load test against the benchmark server."
    )
    parser.add_argument("inputs", nargs="+", help="audio files or directories")
    parser.add_argument("--server", default="http://127.0.0.1:7870")
    sessions = parser.add_mutually_exclusive_group()
    sessions.add_argument(
        "--sessions",
        type=int,
        help="concurrent sessions; files are assigned round-robin (default: one per file)",
    )
    sessions.add_argument(
        "--ramp",
        help="comma-separated session counts run one after another, e.g. 1,2,4,8,16",
    )
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument(
        "--baseline",
        action="store_true",
        help="first transcribe each file alone, then measure transcript drift under load",
    )
    baseline.add_argument(
        "--baseline-from",
        type=Path,
        help="reuse the baseline transcripts saved in an earlier report",
    )
    parser.add_argument(
        "--baseline-speed",
        type=float,
        help="streaming speed for the solo baseline runs (default: --speed)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="run the --baseline and stop, without load steps",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 streams in real time")
    parser.add_argument("--stagger", type=float, default=0.0, help="seconds between session starts")
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument(
        "--tail-silence",
        type=float,
        default=1.0,
        help="seconds of silence sent after each file so VAD closes the last segment",
    )
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=180.0,
        help="seconds to wait for a session pipeline (per-session mode loads a model)",
    )
    parser.add_argument(
        "--drain-timeout",
        type=float,
        default=60.0,
        help="seconds to wait for outstanding segments after the audio ends",
    )
    parser.add_argument("--max-p95-latency", type=float, default=1.5)
    parser.add_argument(
        "--max-drift",
        type=float,
        default=None,
        help="fail a step when any transcript's WER against its baseline exceeds this",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="run every ramp step even after one fails",
    )
    parser.add_argument("--report", type=Path, help="JSON report path")
    parser.add_argument("--no-live", action="store_true", help="plain progress output")
    args = parser.parse_args(argv)

    if args.ramp:
        try:
            args.steps = [int(v) for v in args.ramp.split(",") if v.strip()]
        except ValueError:
            parser.error("--ramp must be comma-separated integers")
    else:
        args.steps = [args.sessions] if args.sessions else []
    if any(n < 1 for n in args.steps):
        parser.error("session counts must be at least 1")
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.baseline_speed is not None and args.baseline_speed <= 0:
        parser.error("--baseline-speed must be positive")
    if args.baseline_only and not args.baseline:
        parser.error("--baseline-only requires --baseline")
    if args.baseline_only and args.steps:
        parser.error("--baseline-only cannot be combined with --sessions or --ramp")
    return args


def load_baseline(path: Path) -> dict[str, str]:
    """Baseline transcripts from an earlier report, keyed by file name."""
    data = json.loads(path.read_text(encoding="utf-8"))
    transcripts = (data.get("baseline") or {}).get("transcripts")
    if not isinstance(transcripts, dict) or not transcripts:
        raise SystemExit(f"{path} has no baseline transcripts")
    return {str(k): str(v) for k, v in transcripts.items()}


# --------------------------------------------------------------------------
# Sessions


async def _wait_for(predicate, *, timeout: float, receiver: asyncio.Task, what: str):
    deadline = time.monotonic() + timeout
    while not predicate():
        if receiver.done():
            raise ConnectionError(f"server closed the connection while {what}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out after {timeout:.0f}s while {what}")
        await asyncio.sleep(0.02)


async def run_session(
    http: aiohttp.ClientSession,
    opts: Options,
    clip: AudioClip,
    result: SessionResult,
) -> None:
    """Stream ``clip`` over one connection and record what the server reports."""
    loop = asyncio.get_running_loop()
    result.state = "connecting"
    try:
        async with asyncio.timeout(opts.connect_timeout):
            ws = await http.ws_connect(
                f"{opts.ws_url}?session={result.session}",
                max_msg_size=0,
                autoping=True,
            )
    except Exception as exc:
        result.state = "failed"
        result.failure = f"connect: {type(exc).__name__}: {exc}"
        return

    ready = asyncio.Event()
    audio_done = asyncio.Event()

    def handle(message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == protocol.READY:
            result.setup_secs = message.get("setup_secs")
            ready.set()
        elif kind == protocol.VAD:
            if message.get("event") == "started":
                result.vad_started += 1
            else:
                result.vad_stopped += 1
        elif kind == protocol.SEGMENT:
            result.segments.append(SegmentResult.from_message(message))
        elif kind == protocol.AUDIO_DONE:
            audio_done.set()
        elif kind == protocol.ERROR:
            server_error = message.get("error") or "unknown error"
            result.failure = f"server: {server_error}"

    async def receive() -> None:
        async for message in ws:
            if message.type == aiohttp.WSMsgType.BINARY:
                payload = protocol.decode_message(message.data)
                if payload is not None:
                    handle(payload)
            elif message.type == aiohttp.WSMsgType.ERROR:
                break

    receiver = asyncio.create_task(receive())
    try:
        result.state = "setup"
        await _wait_for(
            ready.is_set,
            timeout=opts.ready_timeout,
            receiver=receiver,
            what="waiting for the session pipeline",
        )

        result.state = "streaming"
        chunk_secs = opts.chunk_ms / 1000
        chunks = split_chunks(clip.pcm, chunk_ms=opts.chunk_ms)
        silence = bytes(protocol.SAMPLE_RATE * 2 * opts.chunk_ms // 1000)
        chunks += [silence] * round(opts.tail_silence_secs / chunk_secs)
        sent_bytes = 0
        started = loop.time()
        for index, chunk in enumerate(chunks):
            if receiver.done():
                raise ConnectionError("server closed the connection while streaming")
            await ws.send_bytes(protocol.encode_audio(chunk))
            sent_bytes += len(chunk)
            delay = started + (index + 1) * chunk_secs / opts.speed - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
        result.stream_secs = loop.time() - started
        await ws.send_bytes(
            protocol.encode_message({"type": protocol.FLUSH, "audio_bytes": sent_bytes})
        )

        result.state = "draining"
        drain_started = loop.time()
        # The server may still be working through audio sent faster than
        # real time; only once it has all of it are the VAD counts final.
        await _wait_for(
            lambda: audio_done.is_set()
            and result.vad_started == result.vad_stopped
            and len(result.segments) >= result.vad_stopped,
            timeout=opts.drain_timeout,
            receiver=receiver,
            what=(
                "waiting for segments "
                f"({len(result.segments)}/{result.vad_stopped} reported)"
            ),
        )
        result.drain_secs = loop.time() - drain_started
        result.state = "done"
    except Exception as exc:
        result.state = "failed"
        # A reason the server sent beats the generic "connection closed".
        if not result.failure:
            result.failure = f"{type(exc).__name__}: {exc}"
    finally:
        receiver.cancel()
        try:
            await receiver
        except (asyncio.CancelledError, Exception):
            pass
        await ws.close()


class StatsPoller:
    """Poll ``/stats`` and keep every sample for the current step."""

    def __init__(self, http: aiohttp.ClientSession, url: str, *, interval: float = 0.5):
        self._http = http
        self._url = url
        self._interval = interval
        self._process = psutil.Process()
        self._cpu_count = psutil.cpu_count() or 1
        self._process.cpu_percent(None)
        self.samples: list[dict[str, Any]] = []
        self.latest: dict[str, Any] | None = None
        self.client_cpu_percent = 0.0
        self.client_cpu_peak = 0.0
        self.error: str | None = None

    def reset(self) -> None:
        self.samples = []
        self.client_cpu_peak = 0.0

    async def run(self) -> None:
        while True:
            try:
                async with self._http.get(
                    self._url, timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    sample = await response.json()
                self.samples.append(sample)
                self.latest = sample
                self.error = None
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self.client_cpu_percent = self._process.cpu_percent(None) / self._cpu_count
            self.client_cpu_peak = max(self.client_cpu_peak, self.client_cpu_percent)
            await asyncio.sleep(self._interval)


async def run_step(
    http: aiohttp.ClientSession,
    opts: Options,
    clips: list[AudioClip],
    count: int,
    poller: StatsPoller,
    view: "View",
    label: str,
) -> list[SessionResult]:
    sessions = [
        SessionResult(session=f"s{i + 1:03d}", clip=clips[i % len(clips)].name)
        for i in range(count)
    ]
    by_name = {clip.name: clip for clip in clips}

    async def start(index: int, result: SessionResult):
        if opts.stagger_secs:
            await asyncio.sleep(index * opts.stagger_secs)
        await run_session(http, opts, by_name[result.clip], result)

    poller.reset()
    view.begin_step(label, sessions)
    try:
        await asyncio.gather(*(start(i, s) for i, s in enumerate(sessions)))
    finally:
        view.end_step()
    return sessions


# --------------------------------------------------------------------------
# Output


def _fmt(value: float | None, unit: str = "", digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}{unit}"


def _stats_line(sample: dict[str, Any] | None, poller: StatsPoller) -> str:
    if sample is None:
        return f"server stats unavailable: {poller.error or 'waiting'}"
    engine = sample["engine"]
    system = sample["system"]
    parts = [f"engine {engine['mode']} {engine['model']} beam {engine.get('beam_size', 5)}"]
    if engine.get("max_batch_size", 1) > 1:
        parts.append(
            f"batch last {engine['last_batch_size']}/{engine['max_batch_size']} "
            f"waiting {engine['batch_pending']} | requests queued {engine['queued']}"
        )
    elif "workers" in engine:
        parts.append(
            f"workers {engine['workers']} | queued {engine['queued']} "
            f"running {engine['running']}"
        )
    else:
        parts.append(f"models loaded {engine.get('models_loaded', 0)}")
    parts.append(
        f"in-flight {sample['segments']['in_flight']} | sessions {sample['sessions']['active']}"
    )
    if system:
        parts.append(
            f"CPU {system.get('cpu_percent', 0):.0f}% "
            f"(server {system.get('process_cpu_percent', 0):.0f}%)"
        )
        for gpu in system.get("gpus", []):
            parts.append(
                f"GPU{gpu['index']} {gpu['utilization_percent']:.0f}% "
                f"{gpu['memory_used_mb'] / 1024:.1f}/{gpu['memory_total_mb'] / 1024:.1f} GB"
            )
    parts.append(f"loop lag {sample['loop_lag_ms']:.0f} ms")
    parts.append(f"client CPU {poller.client_cpu_percent:.0f}%")
    return " | ".join(parts)


class View:
    """Plain progress output; :class:`RichView` replaces it when available."""

    def __init__(self, poller: StatsPoller):
        self.poller = poller
        self.sessions: list[SessionResult] = []
        self._task: asyncio.Task | None = None

    def begin_step(self, label: str, sessions: list[SessionResult]) -> None:
        self.sessions = sessions
        print(f"\n== {label}: {len(sessions)} session(s) ==", flush=True)
        self._task = asyncio.get_running_loop().create_task(self._loop())

    def end_step(self) -> None:
        if self._task is not None:
            self._task.cancel()
        self.refresh()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(2)
            self.refresh()

    def refresh(self) -> None:
        states: dict[str, int] = {}
        for s in self.sessions:
            states[s.state] = states.get(s.state, 0) + 1
        segments = sum(len(s.segments) for s in self.sessions)
        print(
            f"  {dict(sorted(states.items()))} segments={segments} | "
            + _stats_line(self.poller.latest, self.poller),
            flush=True,
        )

    def print(self, renderable: Any) -> None:
        print(renderable, flush=True)

    def transcripts(self, sessions, baseline, references) -> None:
        for s in sessions:
            drift = _drift(s, baseline)
            line = f"  {s.session} {s.clip} [{s.state}] segs={len(s.segments)} drift={drift}"
            if s.failure:
                line += f" FAILURE={s.failure}"
            print(line)
            print(f"      {s.transcript or '(no transcript)'}")
        print("", flush=True)

    def summary(self, rows: list[tuple[str, dict[str, Any]]]) -> None:
        for label, summary in rows:
            verdict = "PASS" if summary["passed"] else "FAIL: " + "; ".join(summary["reasons"])
            print(
                f"{label}: sessions={summary['sessions']} segments={summary['segments']} "
                f"errors={summary['segment_errors']} "
                f"latency p50/p95/max={_fmt(summary['latency_p50'])}/"
                f"{_fmt(summary['latency_p95'])}/{_fmt(summary['latency_max'])}s "
                f"queue p95={_fmt(summary['queue_wait_p95'])}s "
                f"infer p50={_fmt(summary['inference_p50'])}s "
                f"batch mean={_fmt(summary.get('batch_size_mean'), '', 1)} "
                f"peak GPU={_fmt(summary['peak_gpu_percent'], '%', 0)} "
                f"{_fmt(summary['peak_gpu_memory_mb'], 'MB', 0)} "
                f"drift max={_fmt(summary['drift_wer_max'])} "
                f"ref WER={_fmt(summary['reference_wer_mean'])} -> {verdict}",
                flush=True,
            )


def _drift(session: SessionResult, baseline: dict[str, str] | None) -> str:
    if not baseline or session.clip not in baseline or session.state != "done":
        return "-"
    return f"{word_error_rate(baseline[session.clip], session.transcript):.2f}"


def make_view(poller: StatsPoller, plain: bool) -> View:
    if plain or not sys.stdout.isatty():
        return View(poller)
    try:
        from rich import box
        from rich.console import Console, Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        print("rich is not installed (uv run --group bench ...); using plain output")
        return View(poller)

    state_styles = {
        "done": "green",
        "failed": "bold red",
        "streaming": "cyan",
        "draining": "yellow",
        "setup": "magenta",
    }

    class RichView(View):
        def __init__(self, poller: StatsPoller):
            super().__init__(poller)
            self.console = Console()
            self._live: Live | None = None
            self._label = ""

        def begin_step(self, label, sessions):
            self.sessions = sessions
            self._label = label
            self._live = Live(
                self._render(), console=self.console, refresh_per_second=4, transient=False
            )
            self._live.start()
            self._task = asyncio.get_running_loop().create_task(self._loop())

        def end_step(self):
            if self._task is not None:
                self._task.cancel()
            if self._live is not None:
                self._live.update(self._render())
                self._live.stop()

        async def _loop(self):
            while True:
                await asyncio.sleep(0.25)
                if self._live is not None:
                    self._live.update(self._render())

        def _render(self):
            table = Table(
                title=f"{self._label}: {len(self.sessions)} session(s)",
                expand=True,
                title_justify="left",
            )
            table.add_column("session", no_wrap=True)
            table.add_column("file", no_wrap=True, max_width=24, overflow="ellipsis")
            table.add_column("state", no_wrap=True)
            table.add_column("segs", justify="right", no_wrap=True)
            table.add_column("last lat", justify="right", no_wrap=True)
            table.add_column("max lat", justify="right", no_wrap=True)
            table.add_column("transcript", ratio=1, no_wrap=True, overflow="ellipsis")
            for s in self.sessions[:MAX_LIVE_ROWS]:
                latencies = [x.latency_secs for x in s.segments if x.latency_secs is not None]
                text = s.failure or (s.segments[-1].text if s.segments else "")
                table.add_row(
                    s.session,
                    s.clip,
                    f"[{state_styles.get(s.state, '')}]{s.state}[/]",
                    f"{len(s.segments)}/{s.vad_stopped}",
                    _fmt(latencies[-1] if latencies else None, "s"),
                    _fmt(max(latencies) if latencies else None, "s"),
                    text,
                )
            hidden = len(self.sessions) - MAX_LIVE_ROWS
            if hidden > 0:
                done = sum(1 for s in self.sessions[MAX_LIVE_ROWS:] if s.state == "done")
                failed = sum(1 for s in self.sessions[MAX_LIVE_ROWS:] if s.state == "failed")
                table.add_row("...", f"{hidden} more", f"{done} done, {failed} failed", "", "", "", "")
            stats = Panel(_stats_line(self.poller.latest, self.poller), title="server")
            return Group(table, stats)

        def print(self, renderable):
            self.console.print(renderable)

        def transcripts(self, sessions, baseline, references):
            table = Table(title="transcripts", expand=True, title_justify="left", show_lines=True)
            table.add_column("session", no_wrap=True)
            table.add_column("file", no_wrap=True, max_width=24, overflow="ellipsis")
            table.add_column("segs", justify="right")
            table.add_column("drift", justify="right")
            table.add_column("ref WER", justify="right")
            table.add_column("transcript", ratio=1)
            for s in sessions:
                ref = "-"
                if references and s.clip in references and s.state == "done":
                    ref = f"{word_error_rate(references[s.clip], s.transcript):.2f}"
                body = s.transcript or "[dim](no transcript)[/]"
                if s.failure:
                    body = f"[bold red]{s.failure}[/]\n{s.transcript}"
                table.add_row(s.session, s.clip, str(len(s.segments)), _drift(s, baseline), ref, body)
            self.console.print(table)

        def summary(self, rows):
            table = Table(
                title=(
                    "summary - seconds unless noted; lat = VAD stop -> transcript, "
                    "q = engine queue wait, inf = inference, batch = mean batch size, "
                    "drift = max WER vs solo "
                    "baseline, refWER = mean WER vs .txt references"
                ),
                title_justify="left",
                box=box.SIMPLE_HEAD,
            )
            table.add_column("step", no_wrap=True, min_width=8)
            for name in (
                "sess", "segs", "err", "lat50", "lat95", "latmax", "q95",
                "inf50", "queued", "batch", "gpu%", "gpuMB", "cpu%", "lag ms", "drift", "refWER",
            ):
                table.add_column(name, justify="right", no_wrap=True)
            for label, s in rows:
                style = "green" if s["passed"] else "bold red"
                table.add_row(
                    f"[{style}]{label}[/]",
                    str(s["sessions"]),
                    str(s["segments"]),
                    str(s["segment_errors"] + s["sessions_failed"]),
                    _fmt(s["latency_p50"]),
                    _fmt(s["latency_p95"]),
                    _fmt(s["latency_max"]),
                    _fmt(s["queue_wait_p95"]),
                    _fmt(s["inference_p50"]),
                    _fmt(s["peak_queued"], "", 0),
                    _fmt(s.get("batch_size_mean"), "", 1),
                    _fmt(s["peak_gpu_percent"], "", 0),
                    _fmt(s["peak_gpu_memory_mb"], "", 0),
                    _fmt(s["peak_cpu_percent"], "", 0),
                    _fmt(s["peak_loop_lag_ms"], "", 0),
                    _fmt(s["drift_wer_max"]),
                    _fmt(s["reference_wer_mean"]),
                )
            self.console.print(table)
            for label, s in rows:
                if s["passed"]:
                    self.console.print(f"  {label}: [green]PASS[/]")
                else:
                    self.console.print(f"  {label}: [bold red]FAIL[/] " + "; ".join(s["reasons"]))

    return RichView(poller)


# --------------------------------------------------------------------------
# Main


def _report_path(args: argparse.Namespace, server: dict[str, Any]) -> Path:
    if args.report:
        return args.report
    engine = server.get("engine", {})
    model = re.sub(r"[^A-Za-z0-9._-]+", "_", str(engine.get("model", "model")))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return RESULTS_DIR / f"{stamp}-{engine.get('mode', 'engine')}-{model}.json"


async def run(args: argparse.Namespace) -> int:
    paths = find_audio_files(args.inputs)
    clips = [load_clip(path) for path in paths]
    names = [clip.name for clip in clips]
    if len(set(names)) != len(names):
        raise SystemExit("audio file names must be unique (they label sessions and baselines)")
    references = {clip.name: clip.reference for clip in clips if clip.reference}
    steps = [] if args.baseline_only else (args.steps or [len(clips)])
    baseline: dict[str, str] | None = None
    if args.baseline_from:
        baseline = load_baseline(args.baseline_from)
        missing = [name for name in names if name not in baseline]
        if missing:
            print(f"warning: {len(missing)} file(s) have no baseline transcript, e.g. {missing[0]}")

    server = args.server.rstrip("/")
    opts = Options(
        ws_url=re.sub(r"^http", "ws", server) + "/ws",
        stats_url=server + "/stats",
        chunk_ms=args.chunk_ms,
        speed=args.speed,
        tail_silence_secs=args.tail_silence,
        stagger_secs=args.stagger,
        connect_timeout=args.connect_timeout,
        ready_timeout=args.ready_timeout,
        drain_timeout=args.drain_timeout,
    )
    thresholds = Thresholds(max_p95_latency_secs=args.max_p95_latency, max_wer_drift=args.max_drift)

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as http:
        try:
            async with http.get(opts.stats_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                server_info = await response.json()
        except Exception as exc:
            print(f"cannot reach the benchmark server at {server}: {exc}", file=sys.stderr)
            return 2

        poller = StatsPoller(http, opts.stats_url)
        poll_task = asyncio.create_task(poller.run())
        view = make_view(poller, args.no_live)
        engine = server_info["engine"]
        view.print(
            f"server: engine={engine['mode']} model={engine['model']} "
            f"language={engine['language']} beam_size={engine.get('beam_size', 5)}"
            + (
                f" max_batch_size={engine['max_batch_size']}"
                if engine.get("max_batch_size", 1) > 1
                else (f" workers={engine['workers']}" if "workers" in engine else "")
            )
        )
        view.print(
            f"{len(clips)} file(s), {sum(c.duration_secs for c in clips):.1f}s of audio; "
            f"steps={steps} speed={args.speed}x"
        )

        report: dict[str, Any] = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "server": server_info,
            "options": {
                "inputs": [str(p) for p in paths],
                "steps": steps,
                "speed": args.speed,
                "stagger": args.stagger,
                "chunk_ms": args.chunk_ms,
                "tail_silence": args.tail_silence,
                "thresholds": vars(thresholds),
            },
            "clips": [
                {"name": c.name, "duration_secs": c.duration_secs, "reference": c.reference}
                for c in clips
            ],
            "baseline": (
                {"from": str(args.baseline_from), "transcripts": baseline} if baseline else None
            ),
            "steps": [],
        }
        report_path = _report_path(args, server_info)
        summaries: list[tuple[str, dict[str, Any]]] = []
        exit_code = 0

        def save_report() -> None:
            # Saved after every step, so a long run that dies keeps its data.
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        try:
            if args.baseline:
                baseline_speed = args.baseline_speed or args.speed
                baseline_opts = replace(opts, speed=baseline_speed)
                baseline = {}
                baseline_sessions: list[SessionResult] = []
                baseline_samples: list[dict[str, Any]] = []
                client_cpu_peak = 0.0
                for index, clip in enumerate(clips, start=1):
                    (session,) = await run_step(
                        http,
                        baseline_opts,
                        [clip],
                        1,
                        poller,
                        view,
                        f"baseline {index}/{len(clips)} {clip.name}",
                    )
                    baseline_sessions.append(session)
                    baseline_samples.extend(poller.samples)
                    client_cpu_peak = max(client_cpu_peak, poller.client_cpu_peak)
                    if session.state == "done":
                        baseline[clip.name] = session.transcript
                view.transcripts(baseline_sessions, None, references)
                # Solo runs never queue, so only failures fail the baseline.
                baseline_summary = summarize_step(
                    baseline_sessions,
                    baseline_samples,
                    references=references,
                    thresholds=replace(thresholds, max_p95_latency_secs=float("inf")),
                    client_cpu_peak=client_cpu_peak,
                )
                label = f"baseline x{baseline_speed:g}"
                summaries.append((label, baseline_summary))
                view.summary([(label, baseline_summary)])
                report["baseline"] = {
                    "speed": baseline_speed,
                    "summary": baseline_summary,
                    "transcripts": baseline,
                    "sessions": [s.to_dict() for s in baseline_sessions],
                }
                save_report()
                if not baseline_summary["passed"]:
                    exit_code = 1

            for count in steps:
                label = f"step {count}"
                sessions = await run_step(http, opts, clips, count, poller, view, label)
                summary = summarize_step(
                    sessions,
                    poller.samples,
                    baseline=baseline,
                    references=references,
                    thresholds=thresholds,
                    client_cpu_peak=poller.client_cpu_peak,
                )
                summaries.append((label, summary))
                report["steps"].append(
                    {
                        "sessions_requested": count,
                        "summary": summary,
                        "sessions": [s.to_dict() for s in sessions],
                        "stats_samples": poller.samples,
                    }
                )
                save_report()
                view.transcripts(sessions, baseline, references)
                view.summary([(label, summary)])
                if not summary["passed"]:
                    exit_code = 1
                    if not args.keep_going:
                        view.print("stopping: step failed (use --keep-going to continue)")
                        break
        finally:
            poll_task.cancel()
            if len(summaries) > 1:
                view.summary(summaries)
            save_report()
            view.print(f"report: {report_path}")
        return exit_code


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args(argv)
    try:
        sys.exit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
