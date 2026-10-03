"""Ranking levers from what previous runs measured, rather than from constants alone."""

from __future__ import annotations

from gitm.agents.policy import Policy, select_interventions
from gitm.kernels.spec import Applicability, InterventionSpec, SafetyGate
from gitm.optimizer.history import History, LeverRecord
from gitm.tracer.schema import KernelEvent, Trace

SKU = "AMD Instinct MI355X"
FP = "kimi-k2.5"


def _trace() -> Trace:
    """One kernel per lever's scope, so coverage is equal and only the effect
    estimate can move the ranking."""
    events = [
        KernelEvent(name="fused_moe_kernel", start_ns=0, end_ns=500, stream_id=7,
                    device_id=0, correlation_id=1),
        KernelEvent(name="void gemm_kernel", start_ns=500, end_ns=1000, stream_id=7,
                    device_id=0, correlation_id=2),
    ]
    return Trace(
        workload_id="vllm-decode", fingerprint="fp", run_id="r", device_count=1,
        vendor="amd", captured_at_ns=0, duration_ns=1000, events=events,
    )


def _spec(name, kernels, *, mean=0.05, kernel_time=False) -> InterventionSpec:
    return InterventionSpec(
        name=name, summary="s", knob=name, value=1,
        expected_delta_mean=mean, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="t", applies_to_kernels=kernels, recovers_kernel_time=kernel_time,
        applicability=Applicability(workloads=["vllm-decode"]),
        safety=SafetyGate(tier="moderate"),
    )


def _record(name, *, mean, wins=1, losses=0, gpu=SKU, fp=FP) -> LeverRecord:
    return LeverRecord(
        intervention_name=name, gpu_sku=gpu, fingerprint=fp, runs=1,
        attempts=wins + losses,
        wins=wins, losses=losses, inconclusive=0, mean_delta=mean,
        best_delta=mean, worst_delta=mean, last_run_id="r1",
    )


def _history(*records) -> History:
    return History(records={(r.intervention_name, r.gpu_sku, r.fingerprint): r
                            for r in records},
                   runs_read=1)


def _ranked(**kw):
    lib = [_spec("moe_lever", ["fused_moe_kernel"]), _spec("gemm_lever", ["gemm"])]
    kw.setdefault("fingerprint", FP)
    return select_interventions(_trace(), lib, kw.pop("policy", Policy()), top_n=5, **kw)


def test_history_is_ignored_until_the_policy_asks_for_it():
    """The flag defaults off, so merging this changes no ranking by itself —
    turning it on is its own decision."""
    h = _history(_record("moe_lever", mean=-0.30))

    off = {c.spec.name: c for c in _ranked(history=h, gpu_sku=SKU)}

    assert off["moe_lever"].delta_source == "prior"
    assert off["moe_lever"].predicted_delta > 0      # scored from the constant


def test_a_measured_delta_replaces_the_hand_authored_estimate():
    """Both levers cover the same share of the trace, so the only thing that can
    separate them is the effect estimate. The lever measured at -30% must fall
    below the one still scored from its prior."""
    h = _history(_record("moe_lever", mean=-0.30))
    policy = Policy(use_history=True)

    ranked = _ranked(policy=policy, history=h, gpu_sku=SKU)
    by_name = {c.spec.name: c for c in ranked}

    assert by_name["moe_lever"].delta_source == "measured"
    assert by_name["moe_lever"].predicted_delta < 0
    assert by_name["gemm_lever"].delta_source == "prior"
    assert ranked[0].spec.name == "gemm_lever"


def test_a_measured_win_outranks_an_unmeasured_lever():
    h = _history(_record("moe_lever", mean=0.49))
    ranked = _ranked(policy=Policy(use_history=True), history=h, gpu_sku=SKU)

    assert ranked[0].spec.name == "moe_lever"
    assert ranked[0].delta_source == "measured"


def test_a_conflicted_lever_is_demoted_but_never_removed():
    """Won twice and lost twice is not neutral — it behaved differently under
    conditions the record does not capture. It ranks below every clean candidate
    and still runs when nothing better is available."""
    h = _history(_record("moe_lever", mean=0.49, wins=2, losses=2))
    ranked = _ranked(policy=Policy(use_history=True), history=h, gpu_sku=SKU)
    by_name = {c.spec.name: c for c in ranked}

    assert by_name["moe_lever"].demoted is True
    # demoted despite the larger measured delta, which alone would rank it first
    assert by_name["moe_lever"].predicted_delta > by_name["gemm_lever"].predicted_delta
    assert ranked[0].spec.name == "gemm_lever"
    assert by_name["moe_lever"] in ranked      # still a candidate


def test_the_demotion_lifts_once_the_record_stops_disagreeing():
    """It describes the evidence, not the lever."""
    settled = _history(_record("moe_lever", mean=0.49, wins=3, losses=0))
    ranked = _ranked(policy=Policy(use_history=True), history=settled, gpu_sku=SKU)

    assert ranked[0].spec.name == "moe_lever"
    assert ranked[0].demoted is False


def test_a_record_from_another_box_is_not_evidence_about_this_one():
    h = _history(_record("moe_lever", mean=-0.30, gpu="NVIDIA H100 80GB HBM3"))
    ranked = _ranked(policy=Policy(use_history=True), history=h, gpu_sku=SKU)
    by_name = {c.spec.name: c for c in ranked}

    assert by_name["moe_lever"].delta_source == "prior"
    assert by_name["moe_lever"].predicted_delta > 0


def test_no_sku_means_no_substitution_rather_than_a_guess():
    """An unnamed box is the case the record's GPU key exists to protect against
    — scoring off whichever record happened to be there is the mistake."""
    h = _history(_record("moe_lever", mean=-0.30))
    ranked = _ranked(policy=Policy(use_history=True), history=h, gpu_sku=None)

    assert all(c.delta_source == "prior" for c in ranked)


def test_a_record_with_no_usable_delta_keeps_the_prior_and_the_demotion():
    """"Tried, and we have no number" is not "measured at zero" — the record
    still says the lever disagreed with itself, but carries nothing to rank on."""
    rec = LeverRecord(intervention_name="moe_lever", gpu_sku=SKU, runs=2, attempts=2,
                      fingerprint=FP, wins=1, losses=1, inconclusive=0, mean_delta=None,
                      best_delta=None, worst_delta=None, last_run_id="r1")
    ranked = _ranked(policy=Policy(use_history=True), history=_history(rec), gpu_sku=SKU)
    by_name = {c.spec.name: c for c in ranked}

    assert by_name["moe_lever"].delta_source == "prior"
    assert by_name["moe_lever"].predicted_delta > 0     # the constant still applies
    assert by_name["moe_lever"].demoted is True         # but the conflict still counts


def test_a_known_loser_never_outranks_an_uncertain_candidate(tmp_path=None):
    """The demotion orders levers that might help; it does not promote one that
    measured negative every time. Ranking a consistent -9% above an inconsistent
    +1% would spend the run on a result already in hand."""
    h = _history(
        _record("moe_lever", mean=0.02, wins=2, losses=2),   # conflicted, demoted
        _record("gemm_lever", mean=-0.09, wins=0, losses=3),  # a settled loser
    )
    ranked = _ranked(policy=Policy(use_history=True), history=h, gpu_sku=SKU)
    by_name = {c.spec.name: c for c in ranked}

    assert by_name["moe_lever"].demoted is True
    assert by_name["gemm_lever"].demoted is False
    assert by_name["gemm_lever"].predicted_delta < 0
    assert ranked[0].spec.name == "moe_lever"      # demoted, but still the better bet


def test_another_models_record_is_not_evidence_about_this_one():
    """A shared scratch holds runs from several checkpoints on one box. A lever
    measured on a sparse-MoE model says nothing about a dense one."""
    h = _history(_record("moe_lever", mean=-0.30, fp="glm-5.2"))
    ranked = _ranked(policy=Policy(use_history=True), history=h, gpu_sku=SKU)
    by_name = {c.spec.name: c for c in ranked}

    assert by_name["moe_lever"].delta_source == "prior"
    assert by_name["moe_lever"].predicted_delta > 0


def test_no_fingerprint_means_no_substitution():
    """Same reasoning as an unnamed GPU: without knowing which workload the
    record came from, the prior stands rather than a guess."""
    h = _history(_record("moe_lever", mean=-0.30))
    ranked = _ranked(policy=Policy(use_history=True), history=h, gpu_sku=SKU,
                     fingerprint=None)

    assert all(c.delta_source == "prior" for c in ranked)


# ── gating on where time is actually recoverable ─────────────────────────────


def test_lever_aimed_at_a_region_at_its_floor_is_not_a_candidate():
    """The point of the gate: coverage says the lever touches the trace, the
    residuals say the region it touches has nothing to give back."""
    specs = [_spec("fix_moe", ["moe_routed"], kernel_time=True),
             _spec("fix_gemm", ["gemm"], kernel_time=True)]
    ranked = select_interventions(
        _trace(), specs, Policy(), top_n=5,
        recoverable={"moe_routed": 0.0, "gemm": 0.004},
    )
    by = {c.spec.name: c for c in ranked}
    assert by["fix_moe"].rejected_reason is not None
    assert "no_recoverable_time" in by["fix_moe"].rejected_reason
    assert "moe_routed" in by["fix_moe"].rejected_reason
    assert by["fix_gemm"].rejected_reason is None
    # Rejected sorts last, so the one that can help is picked first.
    assert ranked[0].spec.name == "fix_gemm"


def test_no_recoverable_map_gates_nothing():
    """Every caller that does not pass one keeps exactly its old behaviour."""
    specs = [_spec("fix_moe", ["moe_routed"], kernel_time=True)]
    ranked = select_interventions(_trace(), specs, Policy(), top_n=5)
    assert ranked[0].rejected_reason is None


def test_whole_step_lever_is_never_gated_by_a_per_op_floor():
    """It reshapes the step rather than aiming at a region, so no per-op gap
    speaks to it — gating it on one would be reading the map backwards."""
    spec = _spec("cuda_graphs", [])
    spec = spec.model_copy(update={"whole_step": True})
    ranked = select_interventions(
        _trace(), [spec], Policy(), top_n=5, recoverable={"moe_routed": 0.0},
    )
    assert ranked[0].rejected_reason is None


def test_unjudgeable_and_absent_ops_are_kept():
    """An unanswered question is not a no. ``None`` means the gap could not be
    measured soundly; an op missing from the map was never classified at all."""
    specs = [_spec("unjudgeable", ["moe_routed"], kernel_time=True),
             _spec("absent", ["attn_prefill"], kernel_time=True)]
    ranked = select_interventions(
        _trace(), specs, Policy(), top_n=5, recoverable={"moe_routed": None},
    )
    assert all(c.rejected_reason is None for c in ranked)


def test_a_lever_is_kept_if_any_op_it_names_is_over_its_floor():
    """Dropping it would discard the one region it could still help."""
    spec = _spec("both", ["moe_routed", "gemm"], kernel_time=True)
    ranked = select_interventions(
        _trace(), [spec], Policy(), top_n=5,
        recoverable={"moe_routed": 0.0, "gemm": 0.004},
    )
    assert ranked[0].rejected_reason is None


def test_the_gate_runs_before_safety_reasons_but_does_not_mask_them():
    """A lever that is both unsafe and pointless reports one reason, and either
    way it is rejected — the ordering must not let one state hide the other."""
    spec = _spec("risky", ["moe_routed"], kernel_time=True)
    spec = spec.model_copy(update={"safety": SafetyGate(tier="high_risk")})
    ranked = select_interventions(
        _trace(), [spec], Policy(skip_high_risk=True), top_n=5,
        recoverable={"moe_routed": 0.0},
    )
    assert ranked[0].rejected_reason is not None


def test_a_lever_that_has_not_claimed_a_kernel_local_mechanism_is_never_gated():
    """``applies_to_kernels`` says which kernels a lever touches, not where its
    gain comes from. Five of the six real levers scoped to attention work through
    cache capacity, host swap or avoided recomputation, and none of them need
    attention to be above its floor to pay off."""
    spec = _spec("bigger_kv_cache", ["attn_score_value"])  # kernel_time defaults off
    ranked = select_interventions(
        _trace(), [spec], Policy(), top_n=5, recoverable={"attn_score_value": 0.0},
    )
    assert ranked[0].rejected_reason is None


def test_one_op_at_its_floor_beside_an_unjudged_one_is_partial_evidence():
    """``quantization_awq`` names five ops. Rejecting on the subset that happens
    to be in the map would drop it on evidence about one op while another was
    never measured at all."""
    spec = _spec("quantise", ["qkv_proj", "lm_head"], kernel_time=True)
    ranked = select_interventions(
        _trace(), [spec], Policy(), top_n=5,
        recoverable={"qkv_proj": 0.0},  # lm_head never classified
    )
    assert ranked[0].rejected_reason is None

    # With both measured at their floor, it is complete evidence and it goes.
    ranked = select_interventions(
        _trace(), [spec], Policy(), top_n=5,
        recoverable={"qkv_proj": 0.0, "lm_head": 0.0},
    )
    assert ranked[0].rejected_reason is not None


def test_the_real_catalogue_only_exposes_stated_mechanisms_to_the_gate():
    """Pins the audit behind the flag: a lever reaches the gate only by declaring
    that its gain is the slack between an op and its floor."""
    from gitm.kernels.library import load_library

    lib = load_library(workload="vllm-decode")
    gated = {s.name for s in lib if s.recovers_kernel_time}
    # Both are "same work, different kernel": if the op already runs at its
    # roofline floor, another implementation of it has no slack to take.
    assert gated == {"attention_backend_flashinfer", "moe_backend_deep_gemm"}
    # Every one of them is op-scoped: a whole-step lever could never be gated,
    # so declaring it there would be meaningless rather than merely unused.
    assert all(s.applies_to_kernels and not s.whole_step
               for s in lib if s.recovers_kernel_time)


def test_eplb_is_not_gated_on_a_per_op_floor():
    """"Cut stragglers" reads like kernel time and is not. A straggler rank runs
    *more* expert GEMMs, not slower ones, so each kernel sits at its floor while
    the step waits on that rank. Rank skew is measured separately
    (importers/node_rollup.py); the per-op gap cannot see a distribution
    problem, and gating on it would skip a lever that would have helped."""
    from gitm.kernels.library import load_library

    eplb = next(s for s in load_library(workload="vllm-decode")
                if s.name == "enable_eplb")
    assert eplb.applies_to_kernels == ["moe_routed"]  # still scoped there
    assert not eplb.recovers_kernel_time              # but not gated on it

    ranked = select_interventions(
        _trace(), [eplb], Policy(require_qualification_commit=True), top_n=5,
        recoverable={"moe_routed": 0.0},
    )
    assert ranked[0].rejected_reason is None
