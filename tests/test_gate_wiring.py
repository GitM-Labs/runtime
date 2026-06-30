from __future__ import annotations

from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate
from gitm.optimizer.preconditions import GateContext
from gitm.tracer.schema import Trace


def _trace() -> Trace:
    return Trace(
        workload_id="vllm-decode", fingerprint="fp", run_id="r", device_count=1,
        vendor="nvidia", captured_at_ns=0, duration_ns=1000, events=[],
    )


def _spec(name, workloads, *, tier="moderate") -> InterventionSpec:
    return InterventionSpec(
        name=name, summary="s", knob="k", value=1,
        expected_delta_mean=0.05, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="t", applicability=Applicability(workloads=workloads),
        safety=SafetyGate(tier=tier),
    )


def test_gate_rejects_cross_workload_lever_first():
    lib = [_spec("vllm_lever", ["vllm-decode"]), _spec("edge_lever", ["edge"])]
    ctx = GateContext(workload="vllm-decode", dtype="fp16")
    ranked = select_interventions(_trace(), lib, Policy(), top_n=5, ctx=ctx)

    by_name = {c.spec.name: c for c in ranked}
    assert by_name["edge_lever"].rejected_reason.startswith("not_applicable")
    assert by_name["vllm_lever"].rejected_reason is None


def test_without_ctx_no_applicability_filtering():
    lib = [_spec("edge_lever", ["edge"])]
    ranked = select_interventions(_trace(), lib, Policy(), top_n=5)
    # No ctx -> gate skipped; only safety prefilter applies (none here).
    assert ranked[0].rejected_reason is None


def test_load_library_filters_by_workload(tmp_path):
    import yaml

    from gitm.kernels.library import load_library

    p = tmp_path / "library.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "interventions": [
                    {**_spec("v", ["vllm-decode"]).model_dump()},
                    {**_spec("e", ["edge"]).model_dump()},
                ]
            }
        )
    )
    assert [s.name for s in load_library(p, workload="vllm-decode")] == ["v"]
    assert {s.name for s in load_library(p)} == {"v", "e"}  # unfiltered
