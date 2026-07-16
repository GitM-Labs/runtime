"""Candidate selection must not be decided by how a spec was *written*.

Regression cover for the bug that produced the first vllm-decode report. Its five
claims were exactly the five specs with an empty ``applies_to_kernels`` — because an
empty list meant "matches every kernel", so those specs scored their expected delta
against the WHOLE trace while a targeted lever like fp8 KV scored only against its own
kernels. Broad levers swept the top-N by construction and no structural lever could
ever be selected. The null result that followed ("scheduling knobs are noise") was an
artifact of the scoring function, not evidence about fp8.

Nothing in the suite caught it, which is how it survived to a published report.
"""

from __future__ import annotations

from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate
from gitm.optimizer.replay import predict_delta
from tests.conftest import make_kernel, make_trace


def _spec(name: str, kernels: list[str], expected: float = 0.05) -> InterventionSpec:
    return InterventionSpec(
        name=name,
        summary=name,
        knob=name,
        value=1,
        applies_to_kernels=kernels,
        expected_delta_mean=expected,
        expected_delta_lo=0.0,
        expected_delta_hi=expected * 2,
        source="test",
        applicability=Applicability(workloads=["vllm-decode"]),
        safety=SafetyGate(tier="low_risk"),
    )


def _decode_trace():
    """A decode window shaped like the real H100 capture: GPU busy ~30% of the time.

    300ns of kernels in a 1000ns window — 30ns of it attention, the kernel an fp8 KV
    lever actually touches.
    """
    return make_trace(
        duration_ns=1000,
        events=[
            make_kernel("nvjet_tst_224x64_gemm", start_ns=0, end_ns=270),
            make_kernel("flash::FlashAttnFwdSm90", start_ns=270, end_ns=300),
        ],
    )


# --------------------------------------------------------------------------- #
# predict_delta: which headroom does a spec get scored against?               #
# --------------------------------------------------------------------------- #
def test_scheduler_knob_is_scored_against_idle_time_not_the_whole_trace():
    """A batch-size knob makes no kernel faster — it fills the gaps.

    Under the old model an empty applies_to_kernels meant "matches everything", so
    this scored against all 300ns of kernel time. It should score against the 700ns
    the GPU spent idle, which is the headroom it can actually reclaim.
    """
    delta = predict_delta(_decode_trace(), _spec("max_num_seqs", [], expected=0.05))

    assert abs(delta - 0.7 * 0.05) < 1e-9


def test_kernel_attributable_spec_is_scored_against_only_its_own_kernels():
    delta = predict_delta(_decode_trace(), _spec("fp8", ["FlashAttnFwd"], expected=0.06))

    assert abs(delta - 0.03 * 0.06) < 1e-9  # 30ns of 1000ns


def test_a_spec_that_names_kernels_which_never_run_scores_zero():
    """The other half of the bug: `paged_attention` appears nowhere in a V1 trace."""
    assert predict_delta(_decode_trace(), _spec("fp8_old", ["paged_attention"])) == 0.0


# --------------------------------------------------------------------------- #
# select_interventions: the slate                                             #
# --------------------------------------------------------------------------- #
def test_broad_levers_cannot_sweep_the_whole_slate():
    """The exact shape of the failed report: 5 scheduler knobs + 1 structural lever.

    The scheduler knobs each out-score fp8 on their own axis (they act on 70% idle
    time; fp8 acts on 3% attention time). A flat top-5 hands them every slot and fp8
    is never measured — which is precisely what shipped.
    """
    library = [
        _spec("max_num_batched_tokens", []),
        _spec("max_num_seqs", []),
        _spec("swap_space", []),
        _spec("gpu_memory_utilization", []),
        _spec("scheduling_policy", []),
        _spec("kv_cache_dtype_fp8", ["FlashAttnFwd", "reshape_and_cache_flash"], 0.06),
    ]

    picked = [c.spec.name for c in select_interventions(_decode_trace(), library, Policy(), top_n=5)]

    assert "kv_cache_dtype_fp8" in picked
    assert picked[0] == "kv_cache_dtype_fp8"  # the starved class leads the slate


def test_slate_interleaves_the_two_mechanism_classes():
    library = [
        _spec("broad_a", []), _spec("broad_b", []), _spec("broad_c", []),
        _spec("targeted_a", ["FlashAttnFwd"], 0.06),
        _spec("targeted_b", ["nvjet_tst"], 0.05),
    ]

    picked = [c.spec.name for c in select_interventions(_decode_trace(), library, Policy(), top_n=4)]

    assert sum(n.startswith("targeted") for n in picked) == 2
    assert sum(n.startswith("broad") for n in picked) == 2


def test_a_class_with_nothing_left_does_not_waste_slots():
    """One targeted spec, many broad ones: the rest of the slate still fills up."""
    library = [_spec("targeted", ["FlashAttnFwd"], 0.06)] + [
        _spec(f"broad_{i}", []) for i in range(5)
    ]

    picked = select_interventions(_decode_trace(), library, Policy(), top_n=4)

    assert len(picked) == 4
    assert "targeted" in [c.spec.name for c in picked]
