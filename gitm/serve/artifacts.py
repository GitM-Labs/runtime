"""The artifact set every vLLM capture writes, however the server was obtained.

``gitm capture serve`` and ``gitm capture attach`` differ only in who owns the
process — one launches it, the other adopts one already running. The evidence they
produce must not differ, or two traces of the same server stop being comparable and
the whole point of having both paths is lost. So the writing lives here, once:

    trace.jsonl            the merged CUPTI trace (written by capture() itself)
    kernel_breakdown.json  what the trace is made of, and whether it is readable
    run_manifest.json      how the trace was obtained, in enough detail to redo it
    serving_summary.json   what the server was doing during the window
    preflight.json         what was checked before anything was touched

``kernel_breakdown`` is written on every run, not only on the empty-trace path: a
trace with plenty of kernels in it can still be unusable — CUDA-graph replay loses
per-kernel attribution and shows up as an idle-looking GPU — and that is exactly the
failure that gets mistaken for a real result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CaptureResult:
    """What a capture produced, and whether it is worth anything."""

    out_dir: Path
    trace_path: Path
    n_events: int
    n_kernels: int
    status: str  # "ok" | "no_kernels" | "no_traffic"
    breakdown: Any = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return {
            "out_dir": str(self.out_dir),
            "trace_path": str(self.trace_path),
            "n_events": self.n_events,
            "n_kernels": self.n_kernels,
            "status": self.status,
            "warnings": self.warnings,
        }


def write_preflight(out_dir: Path, checks: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.json").write_text(
        json.dumps([asdict(c) for c in checks], indent=2)
    )


def write_capture_artifacts(
    out_dir: Path,
    *,
    trace,
    trace_path: Path,
    manifest: dict,
    serving_summary: dict | None = None,
    checks: list | None = None,
    had_traffic: bool = True,
) -> CaptureResult:
    """Write the common artifact set and classify the result.

    ``had_traffic`` is the caller's answer to "did anything actually run inside the
    window" — successful client requests for the launch path, completed server-side
    requests for the attach path. It is kept separate from the kernel count because
    the two failures need different fixes: no kernels means the collector never saw
    the process (injection, driver, shim), no traffic means it saw an idle server and
    the window was simply pointed at nothing.
    """
    from gitm.tracer.kernel_taxonomy import summarize_kernels

    out_dir.mkdir(parents=True, exist_ok=True)
    kernels = [e for e in trace.events if getattr(e, "kind", None) == "kernel"]

    breakdown = summarize_kernels(kernels, window_ns=trace.duration_ns)
    warnings = breakdown.warnings()
    (out_dir / "kernel_breakdown.json").write_text(
        json.dumps({**asdict(breakdown), "warnings": warnings}, indent=2)
    )

    if serving_summary is not None:
        (out_dir / "serving_summary.json").write_text(json.dumps(serving_summary, indent=2))

    if checks is not None:
        manifest = {**manifest, "preflight": [asdict(c) for c in checks]}
    manifest = {
        **manifest,
        "trace": {
            "path": str(trace_path),
            "run_id": trace.run_id,
            "source": trace.source,
            "device_count": trace.device_count,
            "events": len(trace.events),
            "kernel_records": len(kernels),
            "kernels": breakdown.n_kernels,
            "invalid_kernel_durations": breakdown.n_invalid_duration,
            "duration_ns": trace.duration_ns,
        },
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    if breakdown.n_kernels == 0:
        status = "no_kernels"
    elif not had_traffic:
        status = "no_traffic"
    else:
        status = "ok"

    return CaptureResult(
        out_dir=out_dir,
        trace_path=trace_path,
        n_events=len(trace.events),
        n_kernels=breakdown.n_kernels,
        status=status,
        breakdown=breakdown,
        warnings=warnings,
    )


def print_result(result: CaptureResult) -> None:
    """Operator-facing tail of a capture: what landed, and what to distrust."""
    from gitm.tracer.kernel_taxonomy import format_breakdown

    print(
        f"    trace: {result.n_events} events, {result.n_kernels} kernels "
        f"-> {result.trace_path}"
    )
    print()
    print(format_breakdown(result.breakdown))
    if result.warnings:
        print("\nTRACE COVERAGE WARNINGS")
        for w in result.warnings:
            print("  - " + w)
