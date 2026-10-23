"""Applicability gate: AND of workload/dtype/hardware/kv-cache conditions."""

from __future__ import annotations

from gitm.kernels.spec import Applicability, InterventionSpec
from gitm.optimizer.preconditions import GateContext, applicable


def _spec(**app_kw) -> InterventionSpec:
    return InterventionSpec(
        name="lever",
        summary="test lever",
        knob="max_num_batched_tokens",
        value=4096,
        expected_delta_mean=0.05,
        expected_delta_lo=0.01,
        expected_delta_hi=0.09,
        source="unit-test",
        applicability=Applicability(**app_kw),
    )


def test_applies_when_all_conditions_met():
    spec = _spec(
        workloads=["vllm-decode"], requires_dtype=["fp16"], requires_hardware=["H100"]
    )
    ctx = GateContext(workload="vllm-decode", dtype="fp16", hardware="NVIDIA H100 80GB")
    ok, reason = applicable(spec, ctx)
    assert ok and reason == ""


def test_rejects_wrong_workload():
    ok, reason = applicable(_spec(workloads=["vllm-decode"]), GateContext(workload="edge"))
    assert not ok and "workload" in reason


def test_rejects_wrong_dtype():
    spec = _spec(workloads=["vllm-decode"], requires_dtype=["bf16"])
    ok, reason = applicable(spec, GateContext(workload="vllm-decode", dtype="fp16"))
    assert not ok and "dtype" in reason


def test_rejects_unknown_dtype_when_required():
    spec = _spec(workloads=["vllm-decode"], requires_dtype=["bf16"])
    ok, reason = applicable(spec, GateContext(workload="vllm-decode"))
    assert not ok and "unknown" in reason


def test_hardware_substring_match():
    spec = _spec(workloads=["vllm-decode"], requires_hardware=["A100", "H100"])
    ok, _ = applicable(spec, GateContext(workload="vllm-decode", hardware="NVIDIA A100-SXM4-40GB"))
    assert ok


def test_kv_cache_bounds():
    spec = _spec(workloads=["vllm-decode"], min_kv_cache_len=1024, max_kv_cache_len=8192)
    assert applicable(spec, GateContext(workload="vllm-decode", kv_cache_len=4096))[0]
    assert not applicable(spec, GateContext(workload="vllm-decode", kv_cache_len=512))[0]
    assert not applicable(spec, GateContext(workload="vllm-decode", kv_cache_len=9000))[0]


def test_catalogue_levers_listing_nvidia_parts_apply_on_h200_and_b200():
    """The hardware match is a substring test, so a list naming only A100/H100
    rejects every H200 run: kv_cache_dtype_fp8 never reached H200 job 99, and
    kept +11.9% in job 100 once it could."""
    from gitm.kernels.library import load_library

    by_name = {s.name: s for s in load_library(workload="vllm-decode")}
    names = ["kv_cache_dtype_fp8", "attention_backend_flashinfer",
             "tensor_parallel_size_2", "pipeline_parallel_size_2"]
    for sku in ("NVIDIA H200", "NVIDIA B200"):
        ctx = GateContext(workload="vllm-decode", dtype="bf16", hardware=sku)
        for name in names:
            ok, reason = applicable(by_name[name], ctx)
            assert ok or "Hardware" not in reason, (sku, name, reason)
