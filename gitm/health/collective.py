"""Vendor-neutral collective readiness before the Runtime loop.

Standalone NCCL (NVIDIA) / RCCL (AMD) AllReduce probe that runs *before*
``factory(cfg)`` so a broken fabric fails fast with diagnostics. Accepts
double-init: the workload will create its own process group later.

Hard-fail: AllReduce timeout/hang or wrong reduced sum (whole buffer, all ranks).
Soft-warn: busbw below 10% of the interconnect catalogue (never aborts on BW).

The probe is scoped to the workload's participating world size (e.g. vLLM
``tensor_parallel_size`` / ``GITM_VLLM_TP``). Workloads with ``world_size < 2``
skip — unused GPUs on a multi-GPU box must not block a single-GPU run.

Ranks are launched via ``python -m gitm.health.probe_worker`` (subprocess), not
``multiprocessing`` spawn, so embedded ``optimize()`` callers do not re-execute
their ``__main__`` module in every child.

Escape hatch: ``GITM_SKIP_COLLECTIVE_HEALTH=1``.

GPU-live smoke (multi-GPU box, TP>=2)::

    GITM_VLLM_TP=2 python -c "from gitm.health import run_collective_health; \\
        print(run_collective_health(world_size=2).to_dict())"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Soft floor: warn when measured busbw is below this fraction of catalogue BW.
_SOFT_FLOOR_FRAC = 0.10

# Fixed FP32 buffer for one AllReduce (4 MiB) — small enough for low overhead,
# large enough for a meaningful busbw sample.
_DEFAULT_NBYTES = 4 * 1024 * 1024

_DEFAULT_TIMEOUT_S = 60.0

_ATOL = 1e-3
_RTOL = 1e-5

# Workloads that may use NCCL/RCCL collectives when world_size >= 2.
_COLLECTIVE_WORKLOADS = frozenset({"vllm-decode"})


@dataclass
class Check:
    """Same shape as serve preflight checks; kept local to avoid a serve import."""

    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str


@dataclass
class HealthReport:
    """Collective health outcome + metrics for forensics."""

    vendor: str
    checks: list[Check] = field(default_factory=list)
    world_size: int = 0
    dtype: str | None = None
    nbytes: int | None = None
    elapsed_ms: float | None = None
    algbw_gbs: float | None = None
    busbw_gbs: float | None = None
    expected_sum: float | None = None
    observed_sum: float | None = None
    sku: str | None = None
    catalogue_bw_gbs: float | None = None
    skipped: bool = False

    @property
    def ok(self) -> bool:
        """True when no check failed (warns are fine)."""
        return all(c.status != "fail" for c in self.checks)

    def diagnostic(self) -> str:
        fails = [c for c in self.checks if c.status == "fail"]
        if not fails:
            return ""
        return "; ".join(f"{c.name}: {c.detail}" for c in fails)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "ok": self.ok,
            "skipped": self.skipped,
            "world_size": self.world_size,
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "elapsed_ms": self.elapsed_ms,
            "algbw_gbs": self.algbw_gbs,
            "busbw_gbs": self.busbw_gbs,
            "expected_sum": self.expected_sum,
            "observed_sum": self.observed_sum,
            "sku": self.sku,
            "catalogue_bw_gbs": self.catalogue_bw_gbs,
            "checks": [asdict(c) for c in self.checks],
        }


def expected_allreduce_sum(world_size: int, fill: float = 1.0) -> float:
    """Each rank contributes ``fill``; SUM AllReduce yields ``fill * world_size``."""
    return float(fill) * float(world_size)


def numerical_ok(
    observed: float,
    expected: float,
    *,
    atol: float = _ATOL,
    rtol: float = _RTOL,
) -> bool:
    """Tight absolute/relative check for the AllReduce correctness gate."""
    return abs(observed - expected) <= atol + rtol * abs(expected)


def algbw_busbw_gbs(nbytes: int, elapsed_s: float, world_size: int) -> tuple[float, float]:
    """nccl-tests-style algorithm and bus bandwidth in GB/s (decimal 1e9).

    For AllReduce: ``busbw = algbw * 2 * (n-1) / n``.
    """
    if elapsed_s <= 0 or nbytes <= 0 or world_size < 1:
        return 0.0, 0.0
    algbw = (nbytes / elapsed_s) / 1e9
    busbw = algbw * (2.0 * (world_size - 1) / world_size) if world_size > 1 else algbw
    return algbw, busbw


def soft_bw_check(
    busbw_gbs: float,
    catalogue_bw_bytes_s: float,
    *,
    frac: float = _SOFT_FLOOR_FRAC,
) -> Check:
    """Warn when busbw is far below catalogue; never fail. Skip if catalogue is 0."""
    if catalogue_bw_bytes_s <= 0:
        return Check(
            "collective_busbw",
            "pass",
            f"busbw={busbw_gbs:.3f} GB/s (no catalogue floor for SKU)",
        )
    floor_gbs = (catalogue_bw_bytes_s * frac) / 1e9
    catalogue_gbs = catalogue_bw_bytes_s / 1e9
    if busbw_gbs < floor_gbs:
        return Check(
            "collective_busbw",
            "warn",
            f"busbw={busbw_gbs:.3f} GB/s below soft floor "
            f"{floor_gbs:.3f} GB/s (10% of catalogue {catalogue_gbs:.3f} GB/s)",
        )
    return Check(
        "collective_busbw",
        "pass",
        f"busbw={busbw_gbs:.3f} GB/s >= soft floor {floor_gbs:.3f} GB/s "
        f"(catalogue {catalogue_gbs:.3f} GB/s)",
    )


def write_collective_health(out_dir: Path, report: HealthReport) -> Path:
    """Persist ``collective_health.json`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "collective_health.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path


def resolve_probe_world_size(*, workload: str, engine: Any | None = None) -> int:
    """Participating world size for the upcoming workload's collectives.

    Returns ``< 2`` when the workload has no multi-GPU collectives — the probe
    must skip so unused GPUs on a multi-GPU host cannot block a single-GPU run.
    """
    if engine is not None:
        for attr in ("tensor_parallel_size", "tp_size"):
            v = getattr(engine, attr, None)
            if v is not None:
                return max(1, int(v))
        kwargs = getattr(engine, "gitm_llm_kwargs", None) or {}
        if isinstance(kwargs, dict) and "tensor_parallel_size" in kwargs:
            return max(1, int(kwargs["tensor_parallel_size"]))

    if workload in _COLLECTIVE_WORKLOADS or workload.startswith("vllm"):
        tp = os.environ.get("GITM_VLLM_TP")
        if tp is not None and str(tp).strip():
            try:
                return max(1, int(tp))
            except ValueError:
                return 1
        return 1  # default vLLM path is TP=1

    # HFT / edge / openfold / etc. do not drive NCCL/RCCL collectives here.
    return 1


def _timeout_s() -> float:
    raw = os.environ.get("GITM_COLLECTIVE_HEALTH_TIMEOUT_S")
    if raw is None or not raw.strip():
        return _DEFAULT_TIMEOUT_S
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_TIMEOUT_S


def _env_skip() -> bool:
    return os.environ.get("GITM_SKIP_COLLECTIVE_HEALTH", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _skipped_report(vendor: str, world_size: int, reason: str) -> HealthReport:
    return HealthReport(
        vendor=vendor,
        world_size=world_size,
        skipped=True,
        checks=[
            Check(
                "collective_allreduce",
                "pass",
                f"world_size={world_size}, skipped ({reason})",
            )
        ],
    )


def _stop_procs(procs: list[subprocess.Popen[Any]]) -> None:
    """Terminate then kill any still-alive children."""
    alive = [p for p in procs if p.poll() is None]
    for p in alive:
        p.terminate()
    deadline = time.monotonic() + 5.0
    for p in alive:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass
    still = [p for p in procs if p.poll() is None]
    for p in still:
        p.kill()
    for p in still:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def run_torch_nccl_allreduce(
    world_size: int,
    *,
    nbytes: int = _DEFAULT_NBYTES,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Launch ``world_size`` probe_worker subprocesses; return rank-0 metrics.

    Uses ``python -m gitm.health.probe_worker`` so the caller's ``__main__`` is
    never re-imported. Raises ``TimeoutError`` / ``RuntimeError`` on failure.
    """
    if world_size < 2:
        raise ValueError("world_size must be >= 2 for collective probe")

    timeout = timeout_s if timeout_s is not None else _timeout_s()
    with tempfile.TemporaryDirectory(prefix="gitm-coll-health-") as tmp:
        init_file = os.path.join(tmp, "pg")
        result_path = os.path.join(tmp, "result.json")
        init_method = f"file://{init_file}"
        procs: list[subprocess.Popen[Any]] = []
        try:
            for rank in range(world_size):
                cmd = [
                    sys.executable,
                    "-m",
                    "gitm.health.probe_worker",
                    "--rank",
                    str(rank),
                    "--world-size",
                    str(world_size),
                    "--nbytes",
                    str(nbytes),
                    "--init-method",
                    init_method,
                    "--result-path",
                    result_path,
                    "--atol",
                    str(_ATOL),
                    "--rtol",
                    str(_RTOL),
                ]
                procs.append(
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            deadline = time.monotonic() + timeout
            for p in procs:
                remaining = max(0.1, deadline - time.monotonic())
                try:
                    p.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass
            if any(p.poll() is None for p in procs):
                _stop_procs(procs)
                raise TimeoutError(
                    f"collective AllReduce timed out after {timeout:.0f}s "
                    f"(world_size={world_size})"
                )
            bad = [p for p in procs if p.returncode not in (0,)]
            if bad:
                snippets = []
                for p in bad:
                    err = (p.stderr.read() if p.stderr else "") or ""
                    snippets.append(f"pid={p.pid} rc={p.returncode}: {err.strip()[:200]}")
                raise RuntimeError(
                    "collective AllReduce worker(s) failed: " + "; ".join(snippets)
                )
            if not Path(result_path).is_file():
                raise RuntimeError("collective AllReduce produced no result from rank 0")
            return json.loads(Path(result_path).read_text())
        finally:
            _stop_procs(procs)


def _build_probed_report(
    *,
    vendor: str,
    world_size: int,
    sku: str | None,
    metrics: dict[str, Any],
) -> HealthReport:
    from gitm.planner.context import interconnect_bw_for_sku

    expected = float(metrics["expected_sum"])
    observed = float(metrics["observed_sum"])
    observed_min = float(metrics.get("observed_min", observed))
    observed_max = float(metrics.get("observed_max", observed))
    all_ranks_ok = bool(metrics.get("all_ranks_ok", True))
    nbytes = int(metrics["nbytes"])
    elapsed_s = float(metrics["elapsed_s"])
    algbw, busbw = algbw_busbw_gbs(nbytes, elapsed_s, world_size)
    catalogue = interconnect_bw_for_sku(sku)

    checks: list[Check] = []
    values_ok = (
        all_ranks_ok
        and numerical_ok(observed, expected)
        and numerical_ok(observed_min, expected)
        and numerical_ok(observed_max, expected)
    )
    if values_ok:
        checks.append(
            Check(
                "collective_allreduce",
                "pass",
                f"AllReduce ok: observed={observed:.6g} "
                f"(min={observed_min:.6g} max={observed_max:.6g}) "
                f"expected={expected:.6g} in {elapsed_s * 1000:.2f} ms",
            )
        )
    else:
        checks.append(
            Check(
                "collective_allreduce",
                "fail",
                f"AllReduce numerical mismatch: observed={observed:.6g} "
                f"(min={observed_min:.6g} max={observed_max:.6g}) "
                f"expected={expected:.6g} all_ranks_ok={all_ranks_ok}",
            )
        )
    checks.append(soft_bw_check(busbw, catalogue))

    return HealthReport(
        vendor=vendor,
        checks=checks,
        world_size=world_size,
        dtype="float32",
        nbytes=nbytes,
        elapsed_ms=elapsed_s * 1000.0,
        algbw_gbs=algbw,
        busbw_gbs=busbw,
        expected_sum=expected,
        observed_sum=observed,
        sku=sku,
        catalogue_bw_gbs=(catalogue / 1e9) if catalogue > 0 else None,
        skipped=False,
    )


def run_collective_health(
    *,
    timeout_s: float | None = None,
    skip: bool = False,
    world_size: int | None = None,
) -> HealthReport:
    """Dispatch vendor probe for ``world_size`` participating ranks.

    ``world_size`` should be the workload's collective size (e.g. TP). When
    omitted, falls back to all visible devices (legacy); callers should pass
    :func:`resolve_probe_world_size` instead.
    """
    if skip or _env_skip():
        from gitm.tracer.injection import detect_vendor

        return _skipped_report(
            detect_vendor(),
            world_size or 0,
            "GITM_SKIP_COLLECTIVE_HEALTH or skip=True",
        )

    from gitm.tracer.injection import detect_vendor

    vendor = detect_vendor()
    if vendor == "amd":
        import gitm.health.amd as backend
    else:
        import gitm.health.nvidia as backend

    available = backend.device_count()
    if world_size is None:
        world_size = available
    world_size = int(world_size)

    if world_size < 2:
        return _skipped_report(
            vendor,
            world_size,
            "workload has no multi-GPU collectives",
        )
    if available < world_size:
        return HealthReport(
            vendor=vendor,
            world_size=world_size,
            sku=backend.detect_sku(),
            checks=[
                Check(
                    "collective_allreduce",
                    "fail",
                    f"need {world_size} GPUs for workload collectives, "
                    f"only {available} visible",
                )
            ],
        )

    sku = backend.detect_sku()
    try:
        metrics = run_torch_nccl_allreduce(
            world_size, timeout_s=timeout_s if timeout_s is not None else _timeout_s()
        )
    except TimeoutError as exc:
        return HealthReport(
            vendor=vendor,
            world_size=world_size,
            sku=sku,
            checks=[Check("collective_allreduce", "fail", str(exc))],
        )
    except Exception as exc:
        return HealthReport(
            vendor=vendor,
            world_size=world_size,
            sku=sku,
            checks=[
                Check(
                    "collective_allreduce",
                    "fail",
                    f"{type(exc).__name__}: {exc}",
                )
            ],
        )
    return _build_probed_report(
        vendor=vendor, world_size=world_size, sku=sku, metrics=metrics
    )
