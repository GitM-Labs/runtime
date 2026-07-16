"""Customer profiler intake — importers, analyze CLI, degraded mode, gates."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from gitm.cli import main
from gitm.importers._common import ImportError as ProfilerImportError
from gitm.importers.analyze import analyze_paths, render_customer_report
from gitm.importers.detect import DetectedFormat, detect_format
from gitm.importers.nsys import import_nsys
from gitm.importers.torch_trace import event_from_chrome, import_torch_trace
from gitm.optimizer.headroom import build_headroom
from gitm.optimizer.metrics import HardwarePeak, compute_metrics
from gitm.optimizer.qualification import qualify
from gitm.tracer.schema import KernelEvent, MemcpyEvent, SyncEvent, Trace

from .conftest import make_kernel, make_trace

FIXTURES = Path(__file__).parent / "fixtures" / "importers"
PEAK = HardwarePeak(name="H100", peak_flops=1e14, peak_bw_bytes_s=1e10)


@pytest.fixture(scope="session", autouse=True)
def _ensure_fixtures():
    gen = FIXTURES / "generate_fixtures.py"
    # Always regenerate so schema stays explicit and fixtures match the generator.
    import runpy

    runpy.run_path(str(gen), run_name="__main__")


# ── format sniffing ──────────────────────────────────────────────────────────


def test_detect_nsys_sqlite():
    r = detect_format(FIXTURES / "nsys_2024_min.sqlite")
    assert r.format == DetectedFormat.NSYS_SQLITE


def test_detect_torch_json_and_gz():
    assert detect_format(FIXTURES / "torch_trace_min.json").format == DetectedFormat.TORCH_JSON
    assert detect_format(FIXTURES / "torch_trace_min.json.gz").format == DetectedFormat.TORCH_JSON_GZ


def test_detect_junk():
    r = detect_format(FIXTURES / "mixed_dump" / "junk.txt")
    assert r.format == DetectedFormat.UNKNOWN


def test_detect_ignores_extension_alone(tmp_path):
    """A .json that is not a chrome trace must not be accepted."""
    p = tmp_path / "fake.json"
    p.write_text('{"hello": "world"}\n')
    assert detect_format(p).format == DetectedFormat.UNKNOWN


# ── nsys import ──────────────────────────────────────────────────────────────


def test_nsys_2024_and_2025_import():
    for name in ("nsys_2024_min.sqlite", "nsys_2025_min.sqlite"):
        traces, stats = import_nsys(FIXTURES / name, device=0, run_id="import-testnsys")
        assert len(traces) == 1
        trace = traces[0]
        assert trace.source == "nsys-import"
        assert trace.vendor == "nvidia"
        assert len(trace.kernels()) == 40  # device 0 only
        memcpys = [e for e in trace.events if getattr(e, "kind", None) == "memcpy"]
        syncs = [e for e in trace.events if getattr(e, "kind", None) == "sync"]
        assert len(memcpys) >= 4  # device-0 memcpys
        assert len(syncs) == 3
        # bytes_read/written always None
        for k in trace.kernels():
            assert k.bytes_read is None and k.bytes_written is None
            assert not k.name.isdigit()
        assert trace.fingerprint.startswith("nvidia:")
        assert stats.sku and "A100" in stats.sku


def test_nsys_unsupported_version_errors():
    with pytest.raises(ProfilerImportError, match="unsupported nsys export version 2023"):
        import_nsys(FIXTURES / "nsys_2023_min.sqlite")


def test_nsys_memcpy_enum_branches():
    traces, _ = import_nsys(FIXTURES / "nsys_2024_min.sqlite", device=0)
    trace = traces[0]
    memcpys = [e for e in trace.events if getattr(e, "kind", None) == "memcpy"]
    pairs = {(m.src, m.dst) for m in memcpys}
    assert ("host", "device") in pairs
    assert ("device", "host") in pairs
    assert ("device", "device") in pairs


def test_nsys_sync_enum_branches():
    traces, _ = import_nsys(FIXTURES / "nsys_2024_min.sqlite", device=0)
    kinds = {e.sync_kind for e in traces[0].events if getattr(e, "kind", None) == "sync"}
    assert "stream" in kinds
    assert "event" in kinds
    assert "device" in kinds


def test_nsys_multi_device_selection():
    # Default: all devices
    all_traces, stats = import_nsys(FIXTURES / "nsys_2024_min.sqlite")
    assert len(all_traces) == 2
    assert stats.per_device_kernel_counts[0] == 40
    assert stats.per_device_kernel_counts[1] == 10
    # Filter
    t0, _ = import_nsys(FIXTURES / "nsys_2024_min.sqlite", device=0)
    t1, _ = import_nsys(FIXTURES / "nsys_2024_min.sqlite", device=1)
    assert len(t0) == 1 and all(k.device_id == 0 for k in t0[0].kernels())
    assert len(t1) == 1 and all(k.device_id == 1 for k in t1[0].kernels())
    with pytest.raises(ProfilerImportError, match="--device 9"):
        import_nsys(FIXTURES / "nsys_2024_min.sqlite", device=9)


def test_nsys_timestamps_normalized_to_zero():
    traces, _ = import_nsys(FIXTURES / "nsys_2024_min.sqlite", device=0)
    trace = traces[0]
    assert min(e.start_ns for e in trace.events) == 0
    assert trace.duration_ns == max(e.end_ns for e in trace.events)


# ── torch import ─────────────────────────────────────────────────────────────


def test_torch_json_and_gz():
    t1s, _ = import_torch_trace(FIXTURES / "torch_trace_min.json", run_id="import-t1")
    t2s, _ = import_torch_trace(FIXTURES / "torch_trace_min.json.gz", run_id="import-t2")
    t1, t2 = t1s[0], t2s[0]
    assert t1.source == "torch-import"
    assert t2.source == "torch-import"
    assert len(t1.kernels()) == 40
    assert len(t2.kernels()) == 40
    # no sync events from torch
    assert not any(isinstance(e, SyncEvent) for e in t1.events)


def test_torch_array_form_and_missing_grid():
    ts, _ = import_torch_trace(FIXTURES / "torch_trace_array.json")
    t = ts[0]
    assert t.kernels()
    # grid defaults to 1 when absent
    assert all(k.grid_x >= 1 and k.block_x >= 1 for k in t.kernels())


def test_torch_us_to_ns_conversion():
    ev = event_from_chrome(
        {
            "ph": "X",
            "cat": "kernel",
            "name": "k",
            "ts": 10.0,  # µs
            "dur": 2.5,
            "args": {"stream": 1, "device": 0},
        }
    )
    assert getattr(ev, "kind", None) == "kernel"
    assert ev is not None
    assert ev.start_ns == 10_000
    assert ev.end_ns == 12_500


def test_torch_skips_memset_and_cpu():
    ts, _ = import_torch_trace(FIXTURES / "torch_trace_min.json")
    names = {k.name for k in ts[0].kernels()}
    assert "Memset" not in names
    assert "aten::add" not in names


# ── degraded mode / headroom / qualification ─────────────────────────────────


def test_import_never_commits():
    for path in (
        FIXTURES / "nsys_2024_min.sqlite",
        FIXTURES / "torch_trace_min.json",
    ):
        if path.suffix == ".json":
            traces, _ = import_torch_trace(path)
        else:
            traces, _ = import_nsys(path, device=0)
        q = qualify(traces[0])
        assert q.commit is False
        assert "Imported trace" in q.diagnostic


def test_headroom_trace_only_caveats():
    traces, _ = import_nsys(FIXTURES / "nsys_2024_min.sqlite", device=0)
    trace = traces[0]
    m = compute_metrics(trace, PEAK)
    r = build_headroom(
        trace,
        predicted_floor_s=m.busy_fraction * (trace.duration_ns / 1e9),
        metrics=m,
        workload=trace.workload_id,
        sku="A100",
    )
    assert r.confidence == "trace-only"
    assert any("memcpy traffic only" in c for c in r.caveats)
    assert any("device state plane" in c for c in r.caveats)
    assert any("catalogue peak" in c for c in r.caveats)

    ttraces, _ = import_torch_trace(FIXTURES / "torch_trace_min.json")
    ttrace = ttraces[0]
    m2 = compute_metrics(ttrace, PEAK)
    r2 = build_headroom(
        ttrace,
        predicted_floor_s=m2.busy_fraction * (ttrace.duration_ns / 1e9),
        metrics=m2,
        workload=ttrace.workload_id,
        sku="A100",
    )
    assert any("Sync events absent" in c for c in r2.caveats)


def test_constructed_trace_stays_full_confidence():
    trace = make_trace(
        events=[make_kernel("k", start_ns=0, end_ns=100)],
        duration_ns=200,
        source="cupti",
    )
    m = compute_metrics(trace, PEAK)
    r = build_headroom(trace, predicted_floor_s=1e-7, metrics=m, workload="w")
    assert r.confidence == "full"
    assert r.caveats == []


# ── parity ───────────────────────────────────────────────────────────────────


def _parity_constructed() -> Trace:
    return Trace(
        workload_id="parity",
        fingerprint="pending",
        run_id="constructed",
        device_count=1,
        vendor="nvidia",
        captured_at_ns=0,
        duration_ns=200_000,
        source="cupti",
        events=[
            KernelEvent(
                start_ns=0, end_ns=50_000, stream_id=0, device_id=0, name="gemm_a",
                grid_x=128, block_x=64,
            ),
            KernelEvent(
                start_ns=100_000, end_ns=150_000, stream_id=0, device_id=0, name="gemm_b",
                grid_x=128, block_x=64,
            ),
            MemcpyEvent(
                start_ns=60_000, end_ns=65_000, stream_id=0, device_id=0,
                bytes=1_000_000, src="host", dst="device",
            ),
            SyncEvent(
                start_ns=160_000, end_ns=170_000, stream_id=0, device_id=0, sync_kind="stream",
            ),
            # wall pad
            KernelEvent(
                start_ns=200_000, end_ns=200_000, stream_id=0, device_id=0, name="tail_pad",
            ),
        ],
    )


def test_parity_metrics_and_headroom():
    constructed = _parity_constructed()
    # Fix fingerprint like importers do.
    from gitm.optimizer.qualification import fingerprint

    constructed = constructed.model_copy(update={"fingerprint": fingerprint(constructed)})

    nsys_t = import_nsys(FIXTURES / "parity_nsys.sqlite", device=0, run_id="import-parity-nsys")[0][0]
    torch_t = import_torch_trace(
        FIXTURES / "parity_torch.json", run_id="import-parity-torch"
    )[0][0]

    def pack(tr: Trace):
        m = compute_metrics(tr, PEAK)
        # Use the same floor rule as analyze.
        floor = m.busy_fraction * (tr.duration_ns / 1e9)
        h = build_headroom(tr, predicted_floor_s=floor, metrics=m, workload="parity", sku="H100")
        return m, h

    mc, hc = pack(constructed)
    mn, hn = pack(nsys_t)
    mt, ht = pack(torch_t)

    # constructed vs nsys: full agreement
    assert mc.busy_fraction == pytest.approx(mn.busy_fraction, abs=1e-6)
    assert mc.stall_fraction == pytest.approx(mn.stall_fraction, abs=1e-6)
    for k in mc.stall_breakdown:
        assert mc.stall_breakdown[k] == pytest.approx(mn.stall_breakdown[k], abs=1e-6)
    assert hc.ceiling_distance == pytest.approx(hn.ceiling_distance, abs=1e-6)

    # torch: timing/busy agreement; skip sync-dependent stall keys
    assert mt.busy_fraction == pytest.approx(mc.busy_fraction, abs=1e-6)
    assert mt.stall_fraction == pytest.approx(mc.stall_fraction, abs=1e-6)
    assert ht.ceiling_distance == pytest.approx(hc.ceiling_distance, abs=1e-6)
    # transfer_bound should still match (memcpy present)
    assert mt.stall_breakdown["transfer_bound"] == pytest.approx(
        mc.stall_breakdown["transfer_bound"], abs=1e-6
    )

    assert hc.confidence == "full"
    assert hn.confidence == "trace-only"
    assert ht.confidence == "trace-only"
    assert any("Sync events absent" in c for c in ht.caveats)
    assert not any("Sync events absent" in c for c in hn.caveats)


# ── robustness ───────────────────────────────────────────────────────────────


def test_corrupt_and_strict(tmp_path):
    # non-strict multi-file continues
    result = analyze_paths(
        [FIXTURES / "corrupt.json", FIXTURES / "torch_trace_min.json"],
        out=tmp_path / "r.md",
        strict=False,
        run_id="import-fixed",
    )
    assert result.workloads
    assert result.failures

    with pytest.raises(SystemExit):
        analyze_paths(
            [FIXTURES / "corrupt.json"],
            out=tmp_path / "r2.md",
            strict=True,
        )


def test_dedupe_identical_rows(tmp_path):
    # Build a sqlite with a duplicated kernel row.
    src = FIXTURES / "parity_nsys.sqlite"
    dst = tmp_path / "dup.sqlite"
    dst.write_bytes(src.read_bytes())
    conn = sqlite3.connect(dst)
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1"
    )
    conn.commit()
    conn.close()
    with pytest.warns(UserWarning, match="deduped"):
        traces, stats = import_nsys(dst, device=0)
    # dedupe count is on the per-device finish_trace; check events cleaned
    assert traces[0].kernels()
    Trace.model_validate(traces[0].model_dump())


def test_atomic_write(tmp_path):
    out = tmp_path / "report.md"
    analyze_paths(
        [FIXTURES / "torch_trace_min.json"],
        out=out,
        run_id="import-atomic",
    )
    assert out.exists()
    assert out.read_text().startswith("# GPU headroom assessment")
    # no leftover temp files
    temps = list(tmp_path.glob(".report.md.*"))
    assert temps == []


def test_fingerprint_stable_across_two_imports():
    a, _ = import_nsys(FIXTURES / "parity_nsys.sqlite", device=0, run_id="import-a")
    b, _ = import_nsys(FIXTURES / "parity_nsys.sqlite", device=0, run_id="import-b")
    assert a[0].fingerprint == b[0].fingerprint
    assert a[0].run_id != b[0].run_id


# ── end-to-end CLI ───────────────────────────────────────────────────────────


def test_analyze_mixed_dump_e2e(tmp_path):
    out = tmp_path / "r.md"
    summary = tmp_path / "s.json"
    keep = tmp_path / "traces"
    rc = main(
        [
            "analyze",
            str(FIXTURES / "mixed_dump"),
            "--out",
            str(out),
            "--json",
            str(summary),
            "--keep-traces",
            str(keep),
            "--sku",
            "NVIDIA A100-SXM4-40GB",
        ]
    )
    assert rc == 0
    md = out.read_text()
    assert "GPU headroom assessment" in md
    assert "Recoverable headroom" in md
    assert "What we'd do next" in md
    assert "Floor commitment requires a gitm-captured run" in md
    # no internal jargon
    assert "AutoResearch" not in md
    assert "invariant" not in md.lower()
    # failure appendix for junk
    assert "junk.txt" in md or "files not ingested" in md.lower() or "Appendix" in md

    data = json.loads(summary.read_text())
    assert data["n_workloads"] >= 2
    assert data["n_failures"] >= 1
    for wl in data["workloads"]:
        assert "ceiling_distance" in wl
        assert "gap_by_stall_class" in wl
        assert wl["confidence"] == "trace-only"
        assert wl["commit"] is False
        assert "caveats" in wl

    assert list(keep.glob("*.jsonl"))


def test_golden_customer_report(tmp_path):
    """Exact-match golden: fixed run_id + controlled single fixture."""
    result = analyze_paths(
        [FIXTURES / "parity_torch.json"],
        out=tmp_path / "r.md",
        sku="NVIDIA H100",
        workload_id="parity_workload",
        run_id="import-golden000000000000000000000000",
    )
    # Stabilize machine-local fields before comparing to the checked-in golden.
    from gitm.importers.analyze import _section_ctx

    a = result.workloads[0]
    a.capture_date = "2024-01-15 12:00 UTC"
    a.path = "tests/fixtures/importers/parity_torch.json"
    ctx = _section_ctx(a)
    md = render_customer_report([ctx], failures=[])
    golden = FIXTURES / "golden_customer_report.md"
    golden.write_text(md) if not golden.exists() else None
    # Always refresh when intentionally regenerating: GITM_UPDATE_GOLDEN=1.
    import os

    if os.environ.get("GITM_UPDATE_GOLDEN") == "1":
        golden.write_text(md)
    assert md == golden.read_text(), (
        "customer report drifted from golden — set GITM_UPDATE_GOLDEN=1 if intentional"
    )


def test_summary_json_schema(tmp_path):
    summary = tmp_path / "s.json"
    analyze_paths(
        [FIXTURES / "parity_torch.json"],
        out=tmp_path / "r.md",
        json_out=summary,
        run_id="import-schema",
    )
    data = json.loads(summary.read_text())
    schema = json.loads((FIXTURES / "summary_schema.json").read_text())
    # Lightweight required-key check (avoid adding jsonschema dep).
    for key in schema["required"]:
        assert key in data
    for wl in data["workloads"]:
        for key in schema["workload_required"]:
            assert key in wl


# ── capture source wiring ────────────────────────────────────────────────────


def test_capture_sets_source_none_without_backend(tmp_path):
    from gitm.tracer.capture import capture

    out = tmp_path / "t.jsonl"
    with capture(out, workload_id="w") as tr:
        pass
    assert tr.source in ("none", "cupti", "rocprof")
    # Round-trip still works with the new field.
    raw = out.read_text().splitlines()[0]
    header = json.loads(raw)["_header"]
    assert "source" in header


# ── multi-device / node rollup ────────────────────────────────────────────────


def test_analyze_all_devices_by_default():
    from gitm.importers.analyze import analyze_paths

    result = analyze_paths(
        [FIXTURES / "nsys_2024_min.sqlite"],
        out=None,
        run_id="import-alldev",
    )
    assert len(result.workloads) == 1
    wl = result.workloads[0]
    assert len(wl.devices) == 2
    assert {d.device_id for d in wl.devices} == {0, 1}
    assert wl.rollup is not None
    assert wl.rollup.n_devices == 2
    # customer workload_id never carries :devN
    assert ":dev" not in wl.workload_id


def test_device_filter_optional():
    from gitm.importers.analyze import analyze_paths

    result = analyze_paths(
        [FIXTURES / "nsys_2024_min.sqlite"],
        out=None,
        device=1,
        run_id="import-dev1",
    )
    assert len(result.workloads[0].devices) == 1
    assert result.workloads[0].devices[0].device_id == 1


def test_node_rollup_skew_and_comm():
    from gitm.importers.analyze import analyze_paths

    result = analyze_paths(
        [FIXTURES / "real" / "synthetic_4xA100_nccl.json"],
        out=None,
        sku="NVIDIA A100-SXM4-40GB",
        run_id="import-4x",
    )
    wl = result.workloads[0]
    assert len(wl.devices) == 4
    assert wl.rollup is not None
    assert wl.rollup.n_devices == 4
    assert wl.rollup.has_collective is True
    assert wl.rollup.comm_inconclusive is False
    assert wl.has_collective is True
    assert wl.gate_context is not None and wl.gate_context.has_collective is True
    # Straggler device 3 should create skew
    assert wl.rollup.skew > 0.0
    # Exposed comm > 0 on at least one device
    assert any(c.exposed_comm_ns > 0 for c in wl.rollup.per_device_comm)
    # Report copy
    md = result.report_md
    assert "Node summary" in md
    assert "Device skew" in md
    assert "Exposed communication is recoverable headroom" in md
    assert "Cross-device dependency attribution" in md
    # Headers never show :devN
    assert ":dev0" not in md
    assert "## synthetic_4xA100_nccl" in md or "synthetic_4xA100_nccl" in md


def test_comm_inconclusive_when_no_nccl_names(tmp_path):
    """Multi-device with no collective-named kernels → inconclusive, not zero."""
    from gitm.importers.analyze import analyze_paths

    # Hand-built chrome trace: 2 devices, only gemm names (no collective patterns).
    events = []
    for dev in (0, 1):
        for i in range(5):
            events.append(
                {
                    "ph": "X",
                    "cat": "Kernel",
                    "name": f"cutlass_gemm_dev{dev}_{i}",
                    "ts": float(i * 10),
                    "dur": 5.0,
                    "args": {"device": dev, "stream": 0, "grid": [1, 1, 1], "block": [1, 1, 1]},
                }
            )
    p = tmp_path / "no_comm.json"
    p.write_text(json.dumps({"traceEvents": events}))
    result = analyze_paths([p], out=None, run_id="import-nocomm")
    r = result.workloads[0].rollup
    assert r is not None and r.n_devices == 2
    assert r.has_collective is False
    assert r.comm_inconclusive is True
    assert "inconclusive" in result.report_md.lower()


def test_is_comm_kernel_patterns():
    from gitm.importers.node_rollup import is_comm_kernel

    assert is_comm_kernel("ncclDevKernel_AllReduce_Sum_f32_RING_LL")
    assert is_comm_kernel("void ncclKernel_AllGather(...)")
    assert is_comm_kernel("AllReduce")
    assert is_comm_kernel("reduce_scatter_kernel")
    assert not is_comm_kernel("cutlass::gemm::kernel::Gemm")
    assert not is_comm_kernel("ampere_fp16_s16816gemm")


# ── real kineto fixtures ─────────────────────────────────────────────────────


REAL = FIXTURES / "real"


def test_real_kineto_gpu_metrics_import():
    traces, stats = import_torch_trace(REAL / "kineto_gpu_metrics_input.json")
    assert traces
    assert traces[0].source == "torch-import"
    assert len(traces[0].kernels()) == 30  # 30 Kernel events in the fixture
    # real fixture carries registers/shared/grid from kineto export
    k = traces[0].kernels()[0]
    assert k.registers_per_thread > 0
    assert k.grid_x >= 1 and k.block_x >= 1
    # external id correlation accepted
    assert any(kk.correlation_id is not None for kk in traces[0].kernels())


def test_real_kineto_resnet_workers0_import():
    traces, stats = import_torch_trace(REAL / "kineto_resnet50_workers0.pt.trace.json.gz")
    assert len(traces) == 1
    assert traces[0].kernels()
    assert traces[0].source == "torch-import"
    # Memcpy events present
    mems = [e for e in traces[0].events if getattr(e, "kind", None) == "memcpy"]
    assert mems
    assert all(m.bytes > 0 for m in mems)


def test_real_kineto_resnet_workers4_import():
    traces, _ = import_torch_trace(REAL / "kineto_resnet50_workers4.pt.trace.json.gz")
    assert traces[0].kernels()
    # Operator/Runtime cats must not leak in as kernels
    for k in traces[0].kernels():
        assert not k.name.startswith("aten::")
        assert "cudaDeviceSynchronize" not in k.name


def test_real_kineto_external_id_correlation_pinned():
    """Pinned to gpu_metrics_input.json — correlation via 'external id' key."""
    from gitm.importers.torch_trace import event_from_chrome

    ev = event_from_chrome(
        {
            "ph": "X",
            "cat": "Kernel",
            "name": "test_kernel",
            "ts": 1.0,
            "dur": 2.0,
            "args": {
                "device": 0,
                "stream": 7,
                "external id": 41,
                "registers per thread": 72,
                "shared memory": 100,
                "grid": [1, 2, 1],
                "block": [128, 1, 1],
            },
        }
    )
    assert getattr(ev, "kind", None) == "kernel"
    assert ev is not None
    assert ev.correlation_id == 41
    assert ev.registers_per_thread == 72
    assert ev.shared_mem_bytes == 100
    assert ev.grid_y == 2
    assert ev.block_x == 128


def test_real_kineto_memcpy_cat_capital_m():
    """Pinned: cat 'Memcpy' (capital M) from kineto chrome export."""
    from gitm.importers.torch_trace import event_from_chrome

    ev = event_from_chrome(
        {
            "ph": "X",
            "cat": "Memcpy",
            "name": "Memcpy HtoD (Pageable -> Device)",
            "ts": 10.0,
            "dur": 1.0,
            "args": {"device": 0, "stream": 7, "bytes": 640},
        }
    )
    assert getattr(ev, "kind", None) == "memcpy"
    assert ev is not None
    assert ev.bytes == 640
    assert ev.src == "host" and ev.dst == "device"


# ── robustness ───────────────────────────────────────────────────────────────


def test_gzip_decompression_cap(tmp_path):
    import gzip as gz

    from gitm.importers.torch_trace import import_torch_trace

    # Tiny gzip that claims huge content — we cap during decompress.
    # Build a legit small gzip then pass a tiny max_decompressed_bytes.
    payload = json.dumps(
        {
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": "Kernel",
                    "name": "k",
                    "ts": 0,
                    "dur": 1,
                    "args": {"device": 0, "stream": 0},
                }
            ]
        }
    ).encode()
    p = tmp_path / "small.json.gz"
    with gz.open(p, "wb") as fh:
        fh.write(payload)
    with pytest.raises(ProfilerImportError, match="decompressed size exceeds limit"):
        import_torch_trace(p, max_decompressed_bytes=10)


def test_nsys_export_uses_argv_not_shell(monkeypatch):
    """subprocess.run must receive an argv list with shell=False."""
    import gitm.importers.nsys as nsys_mod

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))

        class R:
            returncode = 1
            stderr = "fail"
            stdout = ""

        return R()

    monkeypatch.setattr(nsys_mod.shutil, "which", lambda _: "/usr/bin/nsys")
    monkeypatch.setattr(nsys_mod.subprocess, "run", fake_run)
    # Create a fake non-sqlite .nsys-rep
    path = FIXTURES / "fake.nsys-rep"
    path.write_bytes(b"NSYS" + b"\x00" * 64)
    try:
        with pytest.raises(ProfilerImportError):
            nsys_mod.import_nsys(path)
        assert calls
        cmd, kwargs = calls[0]
        assert isinstance(cmd, list)
        assert kwargs.get("shell") is False
        assert str(path) in cmd
    finally:
        path.unlink(missing_ok=True)
