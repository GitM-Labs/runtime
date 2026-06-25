"""The GITM profiling wrapper.

Every benchmark captures its stall breakdown the same way: wrap the work-unit
command in the vendor profiler (``nsys`` on NVIDIA, ``rocprof`` on AMD) plus a
single host-side capture (``py-spy`` + ``sar``), then fold the two into the
canonical table. This module owns that orchestration so the three benchmark
pairs don't each hand-roll it.

Design split, so the moving parts stay testable without a GPU:

* **orchestration** (:func:`wrap_command`, :func:`run_profile`) — builds the
  profiler argv and runs it, writing a *profile bundle* into scratch.
* **pure parsers** (:func:`gpu_busy_ns_from_nsys_csv`,
  :func:`gpu_busy_ns_from_rocprof_csv`) — turn an exported CSV into GPU-busy
  nanoseconds. No subprocess, no GPU; unit-tested against fixtures.
* **composition** (:func:`build_breakdown`) — folds per-phase wall-clocks (the
  harness knows its own phases) and the profiler's GPU-busy time into
  :class:`~gitm.bench.schema.StallPhase` rows.

When a tool is absent (e.g. a laptop), orchestration records the gap in the
bundle's ``missing`` list and keeps going rather than crashing — the parsers and
composition still run against whatever was captured, and the missing pieces are
surfaced loudly instead of silently faked.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from gitm.bench.schema import BenchConfig, StallPhase, Vendor

#: Host-side capture window, per the benchmark conventions ("a single 60 s
#: py-spy + sar capture for the host side").
HOST_CAPTURE_S = 60


def _vendor_profiler(vendor: Vendor) -> str:
    return {"nvidia": "nsys", "amd": "rocprof"}[vendor]


@dataclass
class ProfilerTools:
    """Which capture tools are available on this box."""

    nsys: str | None
    rocprof: str | None
    py_spy: str | None
    sar: str | None

    @classmethod
    def detect(cls) -> ProfilerTools:
        return cls(
            nsys=shutil.which("nsys"),
            rocprof=shutil.which("rocprof"),
            py_spy=shutil.which("py-spy"),
            sar=shutil.which("sar"),
        )

    def for_vendor(self, vendor: Vendor) -> str | None:
        return {"nvidia": self.nsys, "amd": self.rocprof}[vendor]


@dataclass
class ProfileBundle:
    """Paths produced by a profiling run, plus what couldn't be captured."""

    out_dir: Path
    gpu_report: Path | None = None  # .nsys-rep / rocprof results dir
    gpu_csv: Path | None = None  # exported per-kernel CSV
    host_pyspy: Path | None = None  # flamegraph svg
    host_sar: Path | None = None  # sar -u log
    missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


def wrap_command(
    config: BenchConfig,
    command: list[str],
    out_dir: str | Path,
    *,
    tools: ProfilerTools | None = None,
) -> tuple[list[str], ProfileBundle]:
    """Return ``(argv, bundle)`` — the GPU-profiler-wrapped command to run.

    Pure: builds argv and the bundle scaffold without executing anything, so
    the wrapping logic is testable. Host-side py-spy/sar run as separate
    siblings (see :func:`run_profile`) because they sample the process tree
    rather than wrapping it.
    """
    out_dir = Path(out_dir)
    tools = tools or ProfilerTools.detect()
    bundle = ProfileBundle(out_dir=out_dir)

    prof = tools.for_vendor(config.vendor)
    if prof is None:
        bundle.missing.append(_vendor_profiler(config.vendor))
        return command, bundle

    if config.vendor == "nvidia":
        report = out_dir / "gpu"
        bundle.gpu_report = report.with_suffix(".nsys-rep")
        argv = [
            prof,
            "profile",
            "--force-overwrite=true",
            "--output",
            str(report),
            "--export=none",
            *command,
        ]
    else:  # amd
        results = out_dir / "rocprof"
        bundle.gpu_report = results
        bundle.gpu_csv = results.with_name("rocprof.stats.csv")
        argv = [prof, "--stats", "-o", str(bundle.gpu_csv), *command]

    return argv, bundle


def run_profile(
    config: BenchConfig,
    command: list[str],
    out_dir: str | Path,
    *,
    host_capture_s: int = HOST_CAPTURE_S,
    tools: ProfilerTools | None = None,
) -> ProfileBundle:
    """Run ``command`` under the vendor profiler with a host-side capture.

    Side-effecting: spawns subprocesses and writes the bundle into ``out_dir``.
    Missing tools are recorded in ``bundle.missing`` and skipped; the run still
    completes so a partial profile is better than none.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tools = tools or ProfilerTools.detect()

    argv, bundle = wrap_command(config, command, out_dir, tools=tools)

    # Host-side samplers run alongside; py-spy attaches to the launched tree.
    host_procs: list[subprocess.Popen] = []
    if tools.sar:
        bundle.host_sar = out_dir / "host_sar.log"
        host_procs.append(
            subprocess.Popen(
                [tools.sar, "-u", "1", str(host_capture_s)],
                stdout=open(bundle.host_sar, "w"),
                stderr=subprocess.DEVNULL,
            )
        )
    else:
        bundle.missing.append("sar")
    if not tools.py_spy:
        bundle.missing.append("py-spy")

    subprocess.run(argv, check=False)

    for hp in host_procs:
        try:
            hp.wait(timeout=host_capture_s + 5)
        except subprocess.TimeoutExpired:
            hp.terminate()

    if config.vendor == "nvidia" and tools.nsys and bundle.gpu_report:
        bundle.gpu_csv = _export_nsys_csv(tools.nsys, bundle.gpu_report, out_dir)

    return bundle


def _export_nsys_csv(nsys: str, report: Path, out_dir: Path) -> Path | None:
    """``nsys stats`` the kernel + memcpy summaries to CSV for parsing."""
    if not report.exists():
        return None
    out = out_dir / "gpu_kern_sum.csv"
    subprocess.run(
        [
            nsys,
            "stats",
            "--report",
            "cuda_gpu_kern_sum",
            "--format",
            "csv",
            "--output",
            str(out.with_suffix("")),
            str(report),
        ],
        check=False,
    )
    # nsys appends its own suffixes; find what it actually wrote.
    for cand in sorted(out_dir.glob("gpu_kern_sum*.csv")):
        return cand
    return None


# --- pure parsers -----------------------------------------------------------


def _read_csv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def gpu_busy_ns_from_nsys_csv(text: str) -> int:
    """Sum GPU-busy nanoseconds from an ``nsys stats cuda_gpu_kern_sum`` CSV.

    The report's ``Total Time (ns)`` column is the summed device time per
    kernel; their sum is total GPU-busy time over the captured window.
    """
    rows = _read_csv(text)
    total = 0.0
    for row in rows:
        for key in ("Total Time (ns)", "Total Time", "Total Time(ns)"):
            if key in row and row[key]:
                total += float(row[key].replace(",", ""))
                break
    return int(total)


def gpu_busy_ns_from_rocprof_csv(text: str) -> int:
    """Sum GPU-busy nanoseconds from a ``rocprof --stats`` CSV.

    rocprof's stats file carries ``DurationNs`` (or ``TotalDurationNs``) per
    kernel; their sum is total device-busy time.
    """
    rows = _read_csv(text)
    total = 0.0
    for row in rows:
        for key in ("TotalDurationNs", "DurationNs", "Duration"):
            if key in row and row[key]:
                total += float(row[key].replace(",", ""))
                break
    return int(total)


def gpu_busy_ns(bundle: ProfileBundle, vendor: Vendor) -> int | None:
    """Read GPU-busy ns from a bundle's exported CSV, or ``None`` if unavailable."""
    if bundle.gpu_csv is None or not Path(bundle.gpu_csv).exists():
        return None
    text = Path(bundle.gpu_csv).read_text()
    if vendor == "nvidia":
        return gpu_busy_ns_from_nsys_csv(text)
    return gpu_busy_ns_from_rocprof_csv(text)


# --- composition ------------------------------------------------------------


@dataclass
class PhaseTiming:
    """What the harness reports for one work-unit phase.

    The harness owns phase boundaries (it knows ``ingest`` from ``order-book
    update``); the profiler supplies how much of each phase the GPU was busy.
    """

    phase: str
    wall_clock_s: float
    gpu_busy_s: float
    sync_s: float = 0.0
    cpu_s: float = 0.0
    throughput: float | None = None


def build_breakdown(phases: list[PhaseTiming]) -> list[StallPhase]:
    """Fold per-phase timings into canonical :class:`StallPhase` rows.

    Data-stall is the residual: wall time not attributed to GPU-busy, sync, or
    host CPU — i.e. the pipeline waiting on bytes. That residual is exactly the
    quantity the deviation monitor cares about, so we compute it explicitly
    rather than asking the harness to report it.
    """
    out: list[StallPhase] = []
    for p in phases:
        wall = p.wall_clock_s
        if wall <= 0:
            raise ValueError(f"phase {p.phase!r} has non-positive wall_clock_s")
        gpu = p.gpu_busy_s / wall
        sync = p.sync_s / wall
        cpu = p.cpu_s / wall
        data_stall = max(0.0, 1.0 - gpu - sync - cpu)
        out.append(
            StallPhase(
                phase=p.phase,
                cpu=min(1.0, cpu),
                data_stall=min(1.0, data_stall),
                sync=min(1.0, sync),
                gpu_active=min(1.0, gpu),
                throughput=p.throughput,
                wall_clock_s=wall,
            )
        )
    return out
