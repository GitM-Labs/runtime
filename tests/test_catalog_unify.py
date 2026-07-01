from __future__ import annotations

from gitm.kernels.library import load_library
from gitm.kernels.spec import Applicability, InterventionSpec
from gitm.optimizer.preconditions import GateContext, applicable


def _spec(**app_kw) -> InterventionSpec:
    return InterventionSpec(
        name="lever", summary="s", knob="k", value=1,
        expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="t", applicability=Applicability(**app_kw),
    )


def test_min_gpus_gate():
    spec = _spec(workloads=["vllm-decode"], min_gpus=2)
    assert not applicable(spec, GateContext(workload="vllm-decode", num_gpus=1))[0]
    assert applicable(spec, GateContext(workload="vllm-decode", num_gpus=4))[0]


def test_requires_collective_and_interconnect():
    spec = _spec(workloads=["vllm-decode"], requires_collective=True, requires_interconnect=True)
    ok_ctx = GateContext(workload="vllm-decode", num_gpus=2, has_collective=True,
                         has_interconnect=True)
    assert applicable(spec, ok_ctx)[0]
    no_ic = GateContext(workload="vllm-decode", num_gpus=2, has_collective=True)
    ok, reason = applicable(spec, no_ic)
    assert not ok and "interconnect" in reason


def test_unified_library_scopes_by_workload():
    vllm = {s.name for s in load_library(workload="vllm-decode")}
    edge = {s.name for s in load_library(workload="kitti")}
    hft = {s.name for s in load_library(workload="hft")}

    assert "edge_fp16_autocast" in edge
    assert "hft_top_of_book_fewer_scans" in hft
    # vLLM run must not see edge/hft levers (cross-workload bug stays fixed)
    assert "edge_fp16_autocast" not in vllm
    assert "hft_top_of_book_fewer_scans" not in vllm
    assert any(s.startswith("kv_cache") or "batched_tokens" in s for s in vllm)


def test_full_library_loads_and_validates():
    # every entry (including the new unified ones) validates against the schema
    specs = load_library()
    assert len(specs) >= 3
    names = {s.name for s in specs}
    assert {"edge_fp16_autocast", "hft_top_of_book_fewer_scans"} <= names
