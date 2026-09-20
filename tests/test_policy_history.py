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


def _spec(name, kernels, *, mean=0.05) -> InterventionSpec:
    return InterventionSpec(
        name=name, summary="s", knob=name, value=1,
        expected_delta_mean=mean, expected_delta_lo=0.0, expected_delta_hi=0.1,
        source="t", applies_to_kernels=kernels,
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
