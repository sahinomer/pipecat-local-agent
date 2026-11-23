"""Run the Whisper engine comparison: every variant, same client settings.

For each variant the runner starts a fresh benchmark server, runs the load
client against it, stops the server, and lets the GPU cool down. Results go
to one directory: a JSON report and a server log per variant, plus a
``summary.json`` with each variant's capacity.

Run from the repository root:

    uv run --group bench python -m benchmarks.whisper.matrix --dry-run
    uv run --group bench python -m benchmarks.whisper.matrix
    uv run --group bench python -m benchmarks.whisper.matrix --summarize <results dir>

Transcript agreement is measured against ``REFERENCE_VARIANT`` on identical
audio: the same step, session and file. Every session's playlist depends only
on its index, so ``s003`` streams the same files at every step and in every
variant. The reference variant compared with itself across steps gives the
noise floor: how much transcripts differ between runs with nothing changed.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import soundfile

from .audio import find_audio_files, plan_playlist
from .report import word_error_rate

RESULTS_DIR = Path(__file__).parent / "results"

# name -> server arguments
VARIANTS: dict[str, list[str]] = {
    "pipecat-per-session": ["--engine", "per-session"],
    "shared-w1": ["--engine", "shared", "--workers", "1"],
    "batched-beam5-b4": ["--engine", "batched", "--max-batch-size", "4", "--beam-size", "5"],
    "batched-greedy-b16": ["--engine", "batched", "--max-batch-size", "16", "--beam-size", "1"],
}

# Transcripts of every variant are compared with this one: Pipecat as shipped.
REFERENCE_VARIANT = "pipecat-per-session"

# Rough fixed costs, for the estimate only.
SERVER_START_SECS = 20.0  # model load and warm-up
STEP_OVERHEAD_SECS = 10.0  # connect, drain, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Whisper engine comparison.")
    parser.add_argument("inputs", nargs="*", default=["benchmarks/whisper/audio"])
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="comma-separated subset of: " + ", ".join(VARIANTS),
    )
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-files", type=int, default=72)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--clip-gap", type=float, default=1.0)
    parser.add_argument("--ramp", default="1,4,8,12,16,20,24,28,32")
    parser.add_argument("--stagger", type=float, default=0.25)
    parser.add_argument("--max-p95-latency", type=float, default=1.0)
    parser.add_argument("--cooldown", type=float, default=20.0, help="seconds between steps")
    parser.add_argument(
        "--variant-cooldown", type=float, default=60.0, help="seconds between variants"
    )
    parser.add_argument("--baseline-from", type=Path)
    parser.add_argument(
        "--keep-going", action="store_true", help="run every step even after a failure"
    )
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and estimate only")
    parser.add_argument(
        "--summarize",
        type=Path,
        metavar="DIR",
        help="recompute summary.json for an earlier run's results directory and exit",
    )
    args = parser.parse_args(argv)
    args.variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in args.variant_names if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {', '.join(unknown)}")
    args.steps = [int(v) for v in args.ramp.split(",") if v.strip()]
    return args


def step_secs(durations: list[float], sessions: int, args: argparse.Namespace) -> float:
    """Expected wall time of one step: staggered starts, the longest session, overhead."""
    longest = 0.0
    for i in range(sessions):
        plan = plan_playlist(
            durations, i % len(durations), duration_secs=args.duration, gap_secs=args.clip_gap
        )
        index, offset = plan[-1]
        longest = max(longest, offset + durations[index])
    tail_silence = 1.0
    return (sessions - 1) * args.stagger + longest + tail_silence + STEP_OVERHEAD_SECS


def print_plan(durations: list[float], args: argparse.Namespace) -> None:
    per_step = {n: step_secs(durations, n, args) for n in args.steps}
    print(f"{len(durations)} files, {sum(durations) / 60:.1f} min of audio")
    print(
        f"steps {args.steps}; {args.duration:g}s per session, {args.stagger:g}s stagger, "
        f"{args.cooldown:g}s cooldown between steps, {args.variant_cooldown:g}s between variants"
    )
    print("\nper step (sessions: minutes incl. overhead, segments ~):")
    segments_per_min = 1031 / 61.6  # measured on the dictation set
    for n, secs in per_step.items():
        print(f"  {n:3}: {secs / 60:4.1f} min, ~{n * args.duration / 60 * segments_per_min:5.0f} segments")

    def variant_secs(steps: list[int]) -> float:
        return (
            SERVER_START_SECS
            + sum(per_step[n] for n in steps)
            + args.cooldown * max(0, len(steps) - 1)
        )

    total_all = 0.0
    print("\nvariant                 all steps")
    for name in args.variant_names:
        secs = variant_secs(args.steps)
        total_all += secs
        print(f"  {name:22} {secs / 60:5.1f} min  ({len(args.steps)} steps)")
    total_all += args.variant_cooldown * max(0, len(args.variant_names) - 1)
    print(f"  {'total, every step':22} {total_all / 60:5.1f} min")


def wait_for_server(url: str, process: subprocess.Popen, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("server did not become ready")


def file_transcripts(report: dict) -> dict[tuple[int, str, int], tuple[str, str]]:
    """Every finished session's per-file transcripts.

    Keyed by ``(sessions in the step, session, position in its playlist)``,
    which identifies the exact audio that was streamed.
    """
    result = {}
    for step in report.get("steps", []):
        for session in step["sessions"]:
            if session["state"] != "done":
                continue
            for position, item in enumerate(session.get("clip_transcripts", [])):
                result[(step["sessions_requested"], session["session"], position)] = (
                    item["clip"],
                    item["transcript"],
                )
    return result


def _agreement_stats(pairs: list[tuple[str, str]]) -> dict:
    rates = [word_error_rate(expected, actual) for expected, actual in pairs]
    return {
        "files_compared": len(rates),
        "files_differing": sum(1 for r in rates if r > 0),
        "mean_wer": sum(rates) / len(rates) if rates else None,
        "max_wer": max(rates) if rates else None,
    }


def agreement(reference: dict, report: dict) -> dict:
    """How far ``report``'s transcripts are from ``reference``'s on identical audio."""
    expected = file_transcripts(reference)
    actual = file_transcripts(report)
    common = sorted(set(expected) & set(actual))
    return _agreement_stats([(expected[k][1], actual[k][1]) for k in common])


def noise_floor(report: dict) -> dict:
    """How much one variant's transcripts of the same audio differ between steps."""
    by_audio: dict[tuple[str, int], list[str]] = {}
    for (_step, session, position), (_clip, text) in sorted(file_transcripts(report).items()):
        by_audio.setdefault((session, position), []).append(text)
    return _agreement_stats(
        [(texts[0], other) for texts in by_audio.values() for other in texts[1:]]
    )


def capacity(report_path: Path) -> dict:
    """The largest passing step of a report, with its key numbers."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    steps = report.get("steps", [])
    passing = [s for s in steps if s["summary"]["passed"]]
    failed = next((s for s in steps if not s["summary"]["passed"]), None)
    best = passing[-1] if passing else None
    pick = lambda step, key: step["summary"].get(key) if step else None  # noqa: E731
    return {
        "max_sessions_passing": best["sessions_requested"] if best else 0,
        "first_failing_sessions": failed["sessions_requested"] if failed else None,
        "first_failure_reasons": failed["summary"]["reasons"] if failed else [],
        "latency_p95_at_max": pick(best, "latency_p95"),
        "gpu_memory_mb_at_max": pick(best, "peak_gpu_memory_mb"),
        "reference_wer_at_max": pick(best, "reference_wer_mean"),
        "batch_mean_at_max": pick(best, "batch_size_mean"),
        "report": str(report_path),
    }


def summarize(out: Path, variant_names: list[str]) -> dict[str, dict]:
    """Capacity of each variant, and its transcript agreement with the reference."""
    reports = {
        name: json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
        for name in variant_names
        if (out / f"{name}.json").exists()
    }
    summary = {name: capacity(out / f"{name}.json") for name in reports}
    reference = reports.get(REFERENCE_VARIANT)
    if reference is not None:
        summary[REFERENCE_VARIANT]["noise_floor"] = noise_floor(reference)
        for name, report in reports.items():
            if name != REFERENCE_VARIANT:
                summary[name]["agreement_vs_reference"] = agreement(reference, report)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(summary: dict[str, dict]) -> None:
    for name, result in summary.items():
        line = (
            f"  {name:22} passes up to {result['max_sessions_passing']:3} sessions "
            f"(p95 {result['latency_p95_at_max'] or 0:.2f}s, "
            f"GPU {result['gpu_memory_mb_at_max'] or 0:.0f} MB)"
        )
        compared = result.get("agreement_vs_reference") or result.get("noise_floor")
        if compared and compared["files_compared"]:
            label = "vs reference" if "agreement_vs_reference" in result else "noise floor"
            line += (
                f" | {label}: {compared['files_differing']}/{compared['files_compared']} "
                f"files differ, mean WER {compared['mean_wer']:.3f}"
            )
        print(line)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize:
        print_summary(summarize(args.summarize, args.variant_names))
        return 0
    paths = find_audio_files(args.inputs)[: args.max_files]
    durations = [soundfile.info(str(p)).duration for p in paths]
    print_plan(durations, args)
    if args.dry_run:
        return 0

    out = args.out or RESULTS_DIR / f"matrix-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    stats_url = f"http://127.0.0.1:{args.port}/stats"
    summary: dict[str, dict] = {}
    if args.baseline_from:
        print(
            "note: with --duration, drift against a solo baseline includes the effect "
            "of streaming files back to back; summary.json compares variants on "
            "identical audio instead"
        )

    for number, name in enumerate(args.variant_names):
        if number:
            print(f"\n== cooling down {args.variant_cooldown:g}s before {name} ==", flush=True)
            time.sleep(args.variant_cooldown)
        print(f"\n== {name} ==", flush=True)
        server_cmd = [
            sys.executable, "-m", "benchmarks.whisper.server",
            *VARIANTS[name],
            "--model", args.model,
            "--language", args.language,
            "--port", str(args.port),
        ]
        client_cmd = [
            sys.executable, "-m", "benchmarks.whisper.client",
            *args.inputs,
            "--server", f"http://127.0.0.1:{args.port}",
            "--max-files", str(args.max_files),
            "--duration", str(args.duration),
            "--clip-gap", str(args.clip_gap),
            "--ramp", ",".join(str(n) for n in args.steps),
            "--stagger", str(args.stagger),
            "--max-p95-latency", str(args.max_p95_latency),
            "--cooldown", str(args.cooldown),
            "--drain-timeout", "600",
            "--report", str(out / f"{name}.json"),
        ]
        if args.baseline_from:
            client_cmd += ["--baseline-from", str(args.baseline_from)]
        if args.keep_going:
            client_cmd.append("--keep-going")

        with open(out / f"{name}.server.log", "w", encoding="utf-8") as log:
            server = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_for_server(stats_url, server)
                subprocess.run(client_cmd, check=False)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
        summary = summarize(out, args.variant_names)

    print(f"\nresults: {out}")
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
