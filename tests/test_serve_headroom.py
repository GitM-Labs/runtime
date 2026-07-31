"""Tests for the serve-capture headroom report (scripts/serve_headroom.py).

The property worth protecting here is not the arithmetic — the modules it composes
own that — but the *gating*: headroom is derived from GPU-busy time, so a trace that
lost kernels reports a large recoverable fraction with full confidence. These tests
pin the behaviour that such a trace is flagged, and that a comparison against a
complete trace names the loss.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from gitm.tracer.capture import write_trace_jsonl
from gitm.tracer.schema import Trace

from .conftest import make_kernel

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("serve_headroom", REPO / "scripts" / "serve_headroom.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sh = _load()

SKU = "NVIDIA H100 80GB HBM3"


SLOT_NS = 4 * (20_000 + 3_000)  # one decode step's worth of wall time


def _write_capture(d: Path, kinds: tuple[str, ...], *, reps: int = 50) -> Path:
    """A capture dir holding one kernel per name in ``kinds``, repeated.

    The window is a fixed ``reps * SLOT_NS`` regardless of how many kinds are
    present, because that is what attribution loss actually looks like: the run
    takes just as long, the kernels simply are not in the trace. Deriving the
    duration from the events instead would shrink the window along with them and
    hide the gap this whole report is built to catch.
    """
    d.mkdir(parents=True, exist_ok=True)
    events = []
    for i in range(reps):
        t = i * SLOT_NS
        for name in kinds:
            events.append(make_kernel(name, start_ns=t, end_ns=t + 20_000, device_id=i % 2))
            t += 20_000 + 3_000
    trace = Trace(
        workload_id="vllm-serve", fingerprint="Qwen3.6-35B-A3B-FP8", run_id="r",
        device_count=2, vendor="nvidia", captured_at_ns=1, duration_ns=reps * SLOT_NS,
        source="cupti", events=events,
    )
    write_trace_jsonl(d / "trace.jsonl", trace)
    return d


FULL = ("fused_moe_kernel", "ampere_fp16_s16816gemm_tn",
        "ncclDevKernel_AllReduce_Sum_f16_RING_LL", "_ZN5flash24flash_fwd_splitkv_kernelI")
LOSSY = ("_ZN5flash24flash_fwd_splitkv_kernelI",)  # graphed work never attributed


def test_resolve_trace_accepts_a_dir_or_a_file(tmp_path):
    d = _write_capture(tmp_path / "cap", FULL)
    assert sh.resolve_trace(d) == d / "trace.jsonl"
    assert sh.resolve_trace(d / "trace.jsonl") == d / "trace.jsonl"


def test_full_trace_reports_headroom_without_coverage_warnings(tmp_path):
    d = _write_capture(tmp_path / "cap", FULL)
    a = sh.analyse(sh.load(d / "trace.jsonl"), SKU)

    assert a["breakdown"].warnings() == []
    assert 0.0 <= a["headroom"].ceiling_distance < 1.0
    assert {b.bucket for b in a["breakdown"].buckets} == {"moe", "gemm", "collective", "attention"}


def test_lossy_trace_still_produces_a_number_but_flags_it(tmp_path):
    """The failure mode this gate exists for: missing kernels are indistinguishable
    from idle, and idle is exactly what gets reported as recoverable."""
    d = _write_capture(tmp_path / "cap", LOSSY)
    a = sh.analyse(sh.load(d / "trace.jsonl"), SKU)

    # a large, entirely untrustworthy headline
    assert a["headroom"].ceiling_distance > 0.5
    warnings = a["breakdown"].warnings()
    assert any("moe" in w for w in warnings)
    assert any("GPU active" in w for w in warnings)


def test_report_puts_the_warning_above_the_number(tmp_path):
    d = _write_capture(tmp_path / "cap", LOSSY)
    trace = sh.load(d / "trace.jsonl")
    md = sh.render(sh.analyse(trace, SKU), serving=None, trace=trace)

    assert "Coverage warnings" in md
    assert md.index("Coverage warnings") < md.index("Ceiling distance")


def test_compare_names_the_buckets_the_lossy_run_lost(tmp_path):
    lossy = sh.analyse(sh.load(_write_capture(tmp_path / "l", LOSSY) / "trace.jsonl"), SKU)
    full = sh.analyse(sh.load(_write_capture(tmp_path / "f", FULL) / "trace.jsonl"), SKU)

    out = sh.compare(lossy, full, label_a="graphs", label_b="eager")
    assert "missing" in out
    for bucket in ("moe", "gemm", "collective"):
        assert bucket in out
    assert "use eager for any per-kernel claim" in out


def test_compare_is_quiet_when_both_runs_agree(tmp_path):
    a = sh.analyse(sh.load(_write_capture(tmp_path / "a", FULL) / "trace.jsonl"), SKU)
    b = sh.analyse(sh.load(_write_capture(tmp_path / "b", FULL) / "trace.jsonl"), SKU)

    assert "missing" not in sh.compare(a, b, label_a="graphs", label_b="eager")


def test_fp8_roofline_caveat_is_always_stated():
    """The catalogue peak is bf16 dense; this model is FP8. A reader must not take
    the compute-bound share as measured against the right ceiling."""
    caveats = sh.fp8_caveat(True, SKU)
    assert any("FP8" in c and "bf16" in c for c in caveats)
    assert any("not in the peak catalogue" in c for c in sh.fp8_caveat(False, "made-up-gpu"))


def test_empty_trace_refuses_to_report(tmp_path, capsys):
    d = tmp_path / "cap"
    d.mkdir()
    trace = Trace(workload_id="vllm-serve", fingerprint="f", run_id="r", device_count=0,
                  vendor="nvidia", captured_at_ns=1, duration_ns=1000, source="cupti", events=[])
    write_trace_jsonl(d / "trace.jsonl", trace)

    assert sh.main([str(d)]) == 1
    assert "no headroom claim" in capsys.readouterr().err
    assert not (d / "headroom.md").exists()


def test_serving_summary_is_folded_in_when_present(tmp_path):
    d = _write_capture(tmp_path / "cap", FULL)
    (d / "serving_summary.json").write_text(json.dumps({
        "ttft_p50_s": 0.31, "ttft_p95_s": 0.88, "tpot_p50_s": 0.021,
        "n_requests": 512, "n_failed_requests": 0, "goodput_rps": 41.2,
    }))

    assert sh.main([str(d)]) == 0
    md = (d / "headroom.md").read_text()
    assert "310 ms" in md and "21 ms" in md
    assert json.loads((d / "headroom.json").read_text())["headroom"]["workload"] == "vllm-serve"
