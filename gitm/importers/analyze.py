"""``gitm analyze`` orchestration — ingest customer profiler dumps → headroom report."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gitm.importers._common import ImportError, ImportStats, atomic_write_text
from gitm.importers.detect import DetectedFormat, detect_format
from gitm.importers.node_rollup import NodeRollup, build_node_rollup
from gitm.optimizer.headroom import HeadroomReport, build_headroom, render_headroom_md
from gitm.optimizer.metrics import HardwarePeak, compute_metrics
from gitm.optimizer.preconditions import GateContext
from gitm.optimizer.qualification import QualificationResult, qualify
from gitm.planner.context import peak_for_sku
from gitm.tracer.capture import write_trace_jsonl
from gitm.tracer.schema import KernelEvent, Trace

# Default catalogue peak when SKU is unknown (A100 PCIe figures; called out in caveats).
_DEFAULT_PEAK = HardwarePeak(name="A100", peak_flops=312e12, peak_bw_bytes_s=1555e9)


@dataclass
class DeviceAnalysis:
    """One device's metrics + headroom within a multi-device workload file."""

    device_id: int
    headroom: HeadroomReport
    metrics: dict[str, Any]
    n_kernels: int
    internal_id: str  # workload:devN for internal artifacts only


@dataclass
class WorkloadAnalysis:
    path: str
    workload_id: str  # customer-facing, never :devN-suffixed
    source_format: str
    sku: str
    sku_known: bool
    capture_date: str
    captured_at_source: str
    devices: list[DeviceAnalysis]
    rollup: NodeRollup | None
    qualification: QualificationResult
    import_warnings: list[str] = field(default_factory=list)
    has_collective: bool = False
    gate_context: GateContext | None = None
    # Convenience: primary (first) device headroom for single-device callers/tests.
    headroom: HeadroomReport | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    n_kernels: int = 0
    internal_headroom_md: str = ""


@dataclass
class AnalyzeResult:
    workloads: list[WorkloadAnalysis] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    report_md: str = ""
    summary: dict[str, Any] = field(default_factory=dict)


def _collect_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    files.append(f)
        else:
            raise FileNotFoundError(f"path not found: {p}")
    seen: set[Path] = set()
    out: list[Path] = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(f)
    return out


def _import_one(
    path: Path,
    *,
    workload_id: str | None,
    device: int | None,
    strict: bool,
    run_id: str | None,
    sku: str | None,
) -> tuple[list[Trace], ImportStats]:
    det = detect_format(path)
    if det.format == DetectedFormat.UNKNOWN:
        raise ImportError(f"unrecognized format ({det.reason})")

    if det.format in (DetectedFormat.NSYS_SQLITE, DetectedFormat.NSYS_REP):
        from gitm.importers.nsys import import_nsys

        return import_nsys(
            path,
            workload_id=workload_id,
            device=device,
            strict=strict,
            run_id=run_id,
            sku=sku,
        )
    if det.format in (DetectedFormat.TORCH_JSON, DetectedFormat.TORCH_JSON_GZ):
        from gitm.importers.torch_trace import import_torch_trace

        return import_torch_trace(
            path,
            workload_id=workload_id,
            device=device,
            strict=strict,
            run_id=run_id,
            sku=sku,
            gzipped=det.format == DetectedFormat.TORCH_JSON_GZ,
        )
    raise ImportError(f"format {det.format} not supported for import")


def _resolve_peak(sku: str | None) -> tuple[HardwarePeak, bool]:
    if sku and sku != "unknown":
        peak = peak_for_sku(sku)
        if peak is not None:
            return peak, True
        return HardwarePeak(
            name=sku,
            peak_flops=_DEFAULT_PEAK.peak_flops,
            peak_bw_bytes_s=_DEFAULT_PEAK.peak_bw_bytes_s,
        ), False
    return _DEFAULT_PEAK, False


def _predicted_floor_s(trace: Trace, metrics_busy_fraction: float) -> float:
    observed = trace.duration_ns / 1e9
    return max(0.0, observed * metrics_busy_fraction)


def _capture_date(captured_at_ns: int) -> str:
    if captured_at_ns <= 0:
        return "unknown"
    try:
        ts = captured_at_ns / 1e9 if captured_at_ns > 10_000_000_000 else float(captured_at_ns)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (OverflowError, OSError, ValueError):
        return "unknown"


def _device_id_of(trace: Trace) -> int:
    for e in trace.events:
        if getattr(e, "kind", None) == "kernel":
            return e.device_id
    return trace.events[0].device_id if trace.events else 0


def analyze_paths(
    paths: list[str | Path],
    *,
    out: Path | None = None,
    sku: str | None = None,
    workload_id: str | None = None,
    device: int | None = None,
    json_out: Path | None = None,
    keep_traces: Path | None = None,
    strict: bool = False,
    run_id: str | None = None,
) -> AnalyzeResult:
    """Ingest every recognized profiler file under ``paths`` and build the report.

    All devices in each file are analyzed unless ``device`` filters to one.
    """
    path_list = [Path(p) for p in paths]
    files = _collect_inputs(path_list)

    if workload_id is not None:
        recognized = [f for f in files if detect_format(f).format != DetectedFormat.UNKNOWN]
        if len(recognized) != 1:
            raise SystemExit(
                "--workload-id applies only when exactly one recognized input file is provided"
            )

    result = AnalyzeResult()
    sections: list[dict[str, Any]] = []
    eng_sections: list[str] = []

    for fpath in files:
        det = detect_format(fpath)
        if det.format == DetectedFormat.UNKNOWN:
            if fpath in path_list or any(p.is_dir() for p in path_list):
                result.failures.append(
                    {"path": str(fpath), "error": f"unrecognized format ({det.reason})"}
                )
                if strict:
                    raise SystemExit(f"strict: unrecognized format for {fpath}: {det.reason}")
            continue

        try:
            recognized = [
                x for x in files if detect_format(x).format != DetectedFormat.UNKNOWN
            ]
            wl = workload_id if (workload_id and len(recognized) == 1) else None
            traces, stats = _import_one(
                fpath,
                workload_id=wl,
                device=device,
                strict=strict,
                run_id=run_id,
                sku=sku,
            )
            multi = len(stats.per_device_kernel_counts) > 1
            if multi:
                print(
                    f"{fpath.name}: multi-GPU kernel counts {stats.per_device_kernel_counts}; "
                    f"analyzing {len(traces)} device trace(s)"
                    + (f" (filter --device {device})" if device is not None else ""),
                    file=sys.stderr,
                )

            resolved_sku = sku or stats.sku or "unknown"
            peak, sku_known = _resolve_peak(resolved_sku)

            device_analyses: list[DeviceAnalysis] = []
            rollup_inputs: list[tuple[Trace, float, float]] = []
            eng_bits: list[str] = []
            # Qualification uses first device (import source gates commit=False for all).
            qual = qualify(traces[0])

            for tr in traces:
                dev_id = _device_id_of(tr)
                metrics = compute_metrics(tr, peak)
                floor_s = _predicted_floor_s(tr, metrics.busy_fraction)
                headroom = build_headroom(
                    tr,
                    predicted_floor_s=floor_s,
                    metrics=metrics,
                    workload=tr.workload_id,
                    sku=resolved_sku,
                )
                if not sku_known:
                    extra = (
                        f"SKU {resolved_sku!r} is not in the hardware catalogue; "
                        f"peak rates fall back to {_DEFAULT_PEAK.name} defaults."
                    )
                    if extra not in headroom.caveats:
                        headroom.caveats = list(headroom.caveats) + [extra]

                internal_id = f"{tr.workload_id}:dev{dev_id}"
                if keep_traces is not None:
                    keep_dir = Path(keep_traces)
                    keep_dir.mkdir(parents=True, exist_ok=True)
                    tpath = keep_dir / f"{internal_id}_{tr.run_id}.jsonl"
                    write_trace_jsonl(tpath, tr)
                    eng_bits.append(render_headroom_md(headroom))

                da = DeviceAnalysis(
                    device_id=dev_id,
                    headroom=headroom,
                    metrics={
                        "busy_fraction": metrics.busy_fraction,
                        "stall_fraction": metrics.stall_fraction,
                        "stall_breakdown": metrics.stall_breakdown,
                        "mbu": metrics.mbu,
                        "hfu": metrics.hfu,
                    },
                    n_kernels=metrics.n_kernels,
                    internal_id=internal_id,
                )
                device_analyses.append(da)
                rollup_inputs.append((tr, metrics.busy_fraction, headroom.ceiling_distance))

            rollup = build_node_rollup(rollup_inputs, multi_device_file=multi)
            # Free event payloads after rollup (rollup reads kernel intervals).
            for tr in traces:
                try:
                    object.__setattr__(tr, "events", [])
                except (AttributeError, TypeError):
                    pass
            # Merge rollup caveats into every device headroom (display once at workload level).
            for c in rollup.caveats:
                for da in device_analyses:
                    if c not in da.headroom.caveats:
                        da.headroom.caveats = list(da.headroom.caveats) + [c]

            gate = GateContext(
                workload=traces[0].workload_id,
                hardware=resolved_sku,
                num_gpus=max(len(device_analyses), 1),
                has_collective=rollup.has_collective,
                has_interconnect=rollup.has_collective,  # best-effort: collectives imply interconnect
            )

            primary = device_analyses[0]
            analysis = WorkloadAnalysis(
                path=str(fpath),
                workload_id=traces[0].workload_id,
                source_format=stats.format,
                sku=resolved_sku,
                sku_known=sku_known,
                capture_date=_capture_date(traces[0].captured_at_ns),
                captured_at_source=stats.captured_at_source,
                devices=device_analyses,
                rollup=rollup,
                qualification=qual,
                import_warnings=list(stats.warnings),
                has_collective=rollup.has_collective,
                gate_context=gate,
                headroom=primary.headroom,
                metrics=primary.metrics,
                n_kernels=sum(d.n_kernels for d in device_analyses),
                internal_headroom_md="\n".join(eng_bits),
            )
            result.workloads.append(analysis)
            sections.append(_section_ctx(analysis))
            eng_sections.extend(eng_bits)
        except ImportError as exc:
            result.failures.append({"path": str(fpath), "error": str(exc)})
            if strict:
                raise SystemExit(f"strict: {fpath}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 — per-file isolation for customer dumps
            result.failures.append({"path": str(fpath), "error": f"{type(exc).__name__}: {exc}"})
            if strict:
                raise SystemExit(f"strict: {fpath}: {exc}") from exc

    if not result.workloads and not result.failures:
        raise SystemExit("no input files found")
    if not result.workloads and result.failures:
        msgs = "; ".join(f"{f['path']}: {f['error']}" for f in result.failures)
        raise SystemExit(f"no workloads analyzed — {msgs}")

    report_md = render_customer_report(sections, failures=result.failures)
    if eng_sections and keep_traces is not None:
        report_md += "\n\n---\n\n# Internal headroom (engineering)\n\n"
        report_md += "\n".join(eng_sections)

    result.report_md = report_md
    result.summary = _build_summary(result)

    if out is not None:
        atomic_write_text(Path(out), report_md)
    if json_out is not None:
        atomic_write_text(Path(json_out), json.dumps(result.summary, indent=2) + "\n")

    return result


def _device_section(d: DeviceAnalysis) -> dict[str, Any]:
    h = d.headroom
    gap = h.gap_by_stall_class
    return {
        "device_id": d.device_id,
        "ceiling_distance": h.ceiling_distance,
        "ceiling_pct": f"{h.ceiling_distance * 100:.1f}",
        "observed_s": h.observed_s,
        "observed_ms": h.observed_s * 1e3,
        "predicted_floor_s": h.predicted_floor_s,
        "predicted_floor_ms": h.predicted_floor_s * 1e3,
        "gap_idle": gap.get("idle_stall", 0.0),
        "gap_memory": gap.get("memory_bound", 0.0),
        "gap_compute": gap.get("compute_bound", 0.0),
        "gap_idle_pct": f"{gap.get('idle_stall', 0.0) * 100:.1f}",
        "gap_memory_pct": f"{gap.get('memory_bound', 0.0) * 100:.1f}",
        "gap_compute_pct": f"{gap.get('compute_bound', 0.0) * 100:.1f}",
        "busy_fraction": h.busy_fraction,
        "mbu": h.mbu,
        "n_kernels": d.n_kernels,
        "already_optimized": h.already_optimized,
        "indicative_split": h.indicative_mem_compute_split,
    }


def _section_ctx(a: WorkloadAnalysis) -> dict[str, Any]:
    primary = a.devices[0].headroom
    r = a.rollup
    devices = [_device_section(d) for d in a.devices]
    comm_rows: list[dict[str, Any]] = []
    if r is not None:
        for cs in r.per_device_comm:
            comm_rows.append(
                {
                    "device_id": cs.device_id,
                    "comm_share_pct": f"{cs.comm_share_of_busy * 100:.1f}",
                    "exposed_pct": f"{cs.exposed_comm_share_of_wall * 100:.1f}",
                    "comm_share_of_busy": cs.comm_share_of_busy,
                    "exposed_comm_share_of_wall": cs.exposed_comm_share_of_wall,
                }
            )
    # Deduplicate caveats: primary + rollup, order-preserving.
    caveats: list[str] = []
    for caveat in primary.caveats:
        if caveat not in caveats:
            caveats.append(caveat)

    return {
        "workload_id": a.workload_id,  # never :devN
        "sku": a.sku,
        "capture_date": a.capture_date,
        "source_format": a.source_format,
        "path": a.path,
        "confidence": primary.confidence,
        "caveats": caveats,
        "qualification_diagnostic": a.qualification.diagnostic,
        "import_warnings": a.import_warnings,
        "n_devices": len(a.devices),
        "devices": devices,
        # Primary headline (single-device or device 0) kept for template simplicity.
        "ceiling_distance": primary.ceiling_distance,
        "ceiling_pct": f"{primary.ceiling_distance * 100:.1f}",
        "observed_ms": primary.observed_s * 1e3,
        "predicted_floor_ms": primary.predicted_floor_s * 1e3,
        "busy_fraction": primary.busy_fraction,
        "n_kernels": a.n_kernels,
        "already_optimized": primary.already_optimized,
        "indicative_split": primary.indicative_mem_compute_split,
        "gap_idle_pct": f"{primary.gap_by_stall_class.get('idle_stall', 0.0) * 100:.1f}",
        "gap_memory_pct": f"{primary.gap_by_stall_class.get('memory_bound', 0.0) * 100:.1f}",
        "gap_compute_pct": f"{primary.gap_by_stall_class.get('compute_bound', 0.0) * 100:.1f}",
        # Node rollup
        "has_rollup": r is not None and r.n_devices > 0,
        "multi_device": r is not None and r.n_devices > 1,
        "node_ceiling_pct": f"{(r.node_ceiling_distance if r else 0.0) * 100:.1f}",
        "node_ceiling_distance": r.node_ceiling_distance if r else 0.0,
        "skew": r.skew if r else 0.0,
        "skew_pct": f"{(r.skew if r else 0.0) * 100:.1f}",
        "has_straggler": r.has_straggler if r else False,
        "has_collective": r.has_collective if r else False,
        "comm_inconclusive": r.comm_inconclusive if r else False,
        "comm_rows": comm_rows,
        "total_exposed_comm_pct": f"{(r.total_exposed_comm_share if r else 0.0) * 100:.1f}",
        "device_busy_rows": (
            [
                {
                    "device_id": did,
                    "busy_pct": f"{bf * 100:.1f}",
                    "wall_ms": f"{(r.device_wall_s.get(did, 0.0) * 1e3):.3f}",
                    "ceiling_pct": f"{(r.device_ceiling.get(did, 0.0) * 100):.1f}",
                }
                for did, bf in sorted(r.device_busy.items())
            ]
            if r is not None
            else []
        ),
    }


def _build_summary(result: AnalyzeResult) -> dict[str, Any]:
    workloads = []
    for a in result.workloads:
        devices = []
        for d in a.devices:
            devices.append(
                {
                    "device_id": d.device_id,
                    "ceiling_distance": d.headroom.ceiling_distance,
                    "gap_by_stall_class": d.headroom.gap_by_stall_class,
                    "busy_fraction": d.headroom.busy_fraction,
                    "mbu": d.headroom.mbu,
                    "observed_s": d.headroom.observed_s,
                    "predicted_floor_s": d.headroom.predicted_floor_s,
                    "n_kernels": d.n_kernels,
                }
            )
        entry: dict[str, Any] = {
            "workload_id": a.workload_id,
            "path": a.path,
            "source_format": a.source_format,
            "sku": a.sku,
            "confidence": a.devices[0].headroom.confidence,
            "caveats": a.devices[0].headroom.caveats,
            "commit": a.qualification.commit,
            "qualification_diagnostic": a.qualification.diagnostic,
            "devices": devices,
            "has_collective": a.has_collective,
            # Back-compat single-device fields (primary device).
            "ceiling_distance": a.devices[0].headroom.ceiling_distance,
            "gap_by_stall_class": a.devices[0].headroom.gap_by_stall_class,
            "observed_s": a.devices[0].headroom.observed_s,
            "predicted_floor_s": a.devices[0].headroom.predicted_floor_s,
            "busy_fraction": a.devices[0].headroom.busy_fraction,
            "mbu": a.devices[0].headroom.mbu,
        }
        if a.rollup is not None:
            entry["node_rollup"] = a.rollup.to_dict()
        workloads.append(entry)
    return {
        "workloads": workloads,
        "failures": result.failures,
        "n_workloads": len(workloads),
        "n_failures": len(result.failures),
    }


def render_customer_report(
    sections: list[dict[str, Any]],
    *,
    failures: list[dict[str, str]] | None = None,
) -> str:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    tpl_dir = Path(__file__).resolve().parent.parent / "optimizer" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape([]),
        keep_trailing_newline=True,
    )
    tpl = env.get_template("headroom_customer.md.j2")
    return tpl.render(sections=sections, failures=failures or [])
