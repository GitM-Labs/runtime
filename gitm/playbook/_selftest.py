"""The check that fails if the playbook schema or its match semantics break.

One runnable thing, ``python -m gitm.playbook --selftest``, and the same
functions are the pytest cases in ``tests/test_playbook.py`` — the check a reader
is told about and the check CI runs are the *same* check.

The regime coordinates below are the **real** ones measured off deliverable 1's
committed fixtures, so the distances asserted here describe the gap between two
real production traces rather than between two made-up ones. The *deltas* are
invented and every shipped example row says so.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from gitm.playbook.match import (
    DEFAULT_AXES,
    UNCALIBRATED_POLICY,
    AxisTolerance,
    MatchPolicy,
    MatchStatus,
    dispersion_distance,
    log2_ratio,
    lookup,
    regime_distance,
)
from gitm.playbook.schema import (
    EnvCapture,
    Evidence,
    Invalidation,
    MeasuredDelta,
    Playbook,
    PlaybookRow,
    Provenance,
    RowIdentity,
)
from gitm.traffic.regime import Regime, SourceKind

#: The shipped worked examples. Data, not package content — same rule as the
#: traffic fixtures. ``$GITM_PLAYBOOK_EXAMPLES`` overrides for an installed
#: checkout.
EXAMPLES = Path(
    os.environ.get(
        "GITM_PLAYBOOK_EXAMPLES",
        Path(__file__).resolve().parents[2] / "benchmarks" / "playbook" / "examples.json",
    )
)

EXAMPLE_ROWS = 6

# --- real regime coordinates, measured off the D1 fixtures -------------------
MOONCAKE = Regime(
    source_kind=SourceKind.PRODUCTION, trace="mooncake", requests=400,
    rate_rps=2.8368794326241136, io_ratio=39.09367234191124,
    input_p50=9075, input_p95=49904, output_p50=370, output_p95=662,
    burstiness=6.738120567375886,
)
BURSTGPT = Regime(
    source_kind=SourceKind.PRODUCTION, trace="burstgpt", requests=383,
    rate_rps=0.010276637419839545, io_ratio=1.8821556431490254,
    input_p50=353, input_p95=1638, output_p50=238, output_p95=841,
    burstiness=1.0106110910396904,
)

#: How far apart the two real traces are, on the default axes. Pinned because it
#: is the sanity check on the whole metric: if these two collapsed to a small
#: distance the axes would not be separating anything.
BURSTGPT_VS_MOONCAKE_LINF = 4.929  # limited by input_p95 (49904 vs 1638, ~30x)

H100 = "NVIDIA H100 80GB"
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
REV = "95a723d0"
ENV = EnvCapture(engine="vllm", engine_version="0.11.0")
T0 = datetime(2026, 9, 2, tzinfo=timezone.utc)


def _identity(regime: Regime, **kw) -> RowIdentity:
    base = dict(
        model=MODEL, model_revision=REV, gpu_sku=H100, env=ENV,
        regime=regime, knobs={"enable_prefix_caching": True},
    )
    base.update(kw)
    return RowIdentity(**base)


def _row(row_id: str, regime: Regime, *, tput: float = 10.0, verified: datetime | None = T0,
         **kw) -> PlaybookRow:
    """A *measured* row — the selftest needs selectable rows, which the shipped
    examples deliberately are not."""
    identity_kw = kw.pop("identity_kw", {})
    return PlaybookRow(
        row_id=row_id,
        identity=_identity(regime, **identity_kw),
        delta=MeasuredDelta(throughput_pct=tput, ttft_p99_ms=-5.0, itl_p99_ms=0.2, repeats=5),
        provenance=Provenance(
            trace_source=regime.trace, trace_sha256="0" * 64,
            regime_label=regime.label(), verified_at=verified,
        ),
        evidence=Evidence.MEASURED,
        **kw,
    )


def _calibrated(max_distance: float, axes: tuple[str, ...] = DEFAULT_AXES) -> MatchPolicy:
    return MatchPolicy(
        name="test-calibrated",
        axes=axes,
        tolerances={
            a: AxisTolerance(max_distance=max_distance, calibration="selftest fixture, not a real run")
            for a in axes
        },
    )


# --- the checks --------------------------------------------------------------

def check_log2_ratio_is_scale_free() -> None:
    """The three worked numbers from the design, plus the properties behind them."""
    assert log2_ratio(1024, 2048) == 1.0  # a 2x change is exactly 1
    assert abs(log2_ratio(1024, 1536) - 0.5849625007211562) < 1e-12
    assert log2_ratio(1024, 1024) == 0.0

    # scale-free: the same factor at any magnitude is the same distance. This is
    # why a raw difference would be the wrong metric — 100 vs 200 tokens and
    # 10,000 vs 20,000 are the same *kind* of mismatch.
    assert log2_ratio(100, 200) == log2_ratio(10_000, 20_000) == 1.0
    # symmetric
    assert log2_ratio(2048, 1024) == log2_ratio(1024, 2048)
    # zero is not a small number: one side zero is incomparable, both zero is equal
    assert math.isinf(log2_ratio(0, 256))
    assert log2_ratio(0, 0) == 0.0


def check_dispersion_distance_handles_flat_traces() -> None:
    """``D = 0`` is a real trace (perfectly paced), not an incomparable one."""
    assert dispersion_distance(0.0, 0.0) == 0.0
    assert dispersion_distance(0.0, 1.0) == 1.0  # flat vs Poisson, one unit
    # the two real traces are far apart on this axis, correctly
    d = dispersion_distance(BURSTGPT.burstiness, MOONCAKE.burstiness)
    assert 1.9 < d < 2.0, d
    # and a nearby dispersion is near
    assert dispersion_distance(5.0, MOONCAKE.burstiness) < 0.4
    # a raw log2 ratio would have blown up here; the shift is what prevents it
    assert math.isinf(log2_ratio(0.0, 1.0))


def check_linf_is_the_worst_axis() -> None:
    """A mean would let four close axes hide one that breaks the row."""
    d = regime_distance(BURSTGPT, MOONCAKE, UNCALIBRATED_POLICY)
    assert d.limiting_axis == "input_p95", d.per_axis
    assert abs(d.linf - BURSTGPT_VS_MOONCAKE_LINF) < 0.001, d.render()
    assert d.linf == max(d.per_axis.values())

    # The case L-inf exists for: identical on five axes, 8x off on the sixth.
    # The mean calls that a 0.5 mismatch; L-inf calls it a 3.0 mismatch. The
    # workload is a long-context one against a short-context row, and the mean
    # is the reading that would apply the row.
    long_ctx = MOONCAKE.model_copy(update={"input_p95": MOONCAKE.input_p95 * 8})
    one_axis = regime_distance(MOONCAKE, long_ctx, UNCALIBRATED_POLICY)
    assert one_axis.limiting_axis == "input_p95" and one_axis.linf == 3.0
    assert sum(one_axis.per_axis.values()) / len(one_axis.per_axis) == 0.5

    # a regime against itself is exactly zero on every axis
    same = regime_distance(MOONCAKE, MOONCAKE, UNCALIBRATED_POLICY)
    assert same.exact and set(same.per_axis.values()) == {0.0}


def check_rate_is_not_in_the_default_axes() -> None:
    """``rate_rps`` is excluded by decision, and the decision is material."""
    assert "rate_rps" not in DEFAULT_AXES
    assert "rate_rps" in regime_distance(
        BURSTGPT, MOONCAKE, MatchPolicy(name="with-rate", axes=(*DEFAULT_AXES, "rate_rps"))
    ).per_axis

    # Two regimes identical except for offered rate: distance 0 by default, and
    # a large distance the moment rate is included. If including it were a
    # no-op the decision would not need making.
    slower = MOONCAKE.model_copy(update={"rate_rps": MOONCAKE.rate_rps / 8})
    assert regime_distance(MOONCAKE, slower, UNCALIBRATED_POLICY).exact
    with_rate = MatchPolicy(name="with-rate", axes=(*DEFAULT_AXES, "rate_rps"))
    assert abs(regime_distance(MOONCAKE, slower, with_rate).linf - 3.0) < 1e-9


def check_tolerance_requires_calibration() -> None:
    """A threshold cannot enter a policy without the experiment that set it."""
    with pytest.raises(ValidationError, match="max_distance without calibration"):
        AxisTolerance(max_distance=1.0)
    with pytest.raises(ValidationError):
        AxisTolerance(max_distance=-0.5, calibration="whatever")
    ok = AxisTolerance(max_distance=1.0, calibration="prereg E4: sign flip at 1.4 on input_p95")
    assert ok.calibrated
    # and the shipped policy has none of them
    assert UNCALIBRATED_POLICY.uncalibrated_axes == DEFAULT_AXES
    assert not any(UNCALIBRATED_POLICY.tolerance(a).calibrated for a in DEFAULT_AXES)


def check_exact_gates_reject_before_distance() -> None:
    """Categorical mismatches are rejections, never a large distance."""
    book = Playbook(rows=[_row("r-mooncake", MOONCAKE)])
    for label, kw in [
        ("gpu_sku", {"gpu_sku": "NVIDIA A100 80GB"}),
        ("model", {"model": "meta-llama/Llama-4-70B"}),
        ("model_revision", {"model_revision": "deadbeef"}),
        ("engine_version", {"env": EnvCapture(engine="vllm", engine_version="0.12.0")}),
        ("knob set", {"knobs": {"max_num_seqs": 64}}),
    ]:
        res = lookup(book, _identity(MOONCAKE, **kw), UNCALIBRATED_POLICY)
        assert res.status is MatchStatus.NO_MATCH, (label, res.render())
        assert res.route_to_discovery
        assert label.split("_")[0] in res.rejected["r-mooncake"], (label, res.rejected)

    # source_kind: a scoreboard row is gated out of a production query even
    # though every numeric axis is identical.
    board = MOONCAKE.model_copy(update={"source_kind": SourceKind.SCOREBOARD})
    assert regime_distance(MOONCAKE, board, UNCALIBRATED_POLICY).exact
    res = lookup(Playbook(rows=[_row("r-board", board)]), _identity(MOONCAKE), UNCALIBRATED_POLICY)
    assert res.status is MatchStatus.NO_MATCH
    assert "scoreboard" in res.rejected["r-board"]

    # concurrency: exact by policy, and turning the gate off is a named edit
    capped = MOONCAKE.model_copy(update={"concurrency": 64})
    book_capped = Playbook(rows=[_row("r-capped", capped)])
    assert lookup(book_capped, _identity(MOONCAKE), UNCALIBRATED_POLICY).status is MatchStatus.NO_MATCH
    loose = MatchPolicy(name="ignore-concurrency", match_concurrency=False)
    assert lookup(book_capped, _identity(MOONCAKE), loose).status is MatchStatus.EXACT_REGIME


def check_uncalibrated_policy_routes_to_discovery() -> None:
    """The headline: a near miss is not a match until an axis is calibrated."""
    near = MOONCAKE.model_copy(update={"output_p50": 400})  # ~0.11 away, one axis
    res = lookup(Playbook(rows=[_row("r-near", near)]), _identity(MOONCAKE), UNCALIBRATED_POLICY)
    assert res.status is MatchStatus.UNCALIBRATED
    assert res.route_to_discovery and res.row is None
    assert "not yet calibrated" in res.reason
    assert "output_p50" in res.reason  # the reason names the axis, not just a number
    # the candidate is still reported — a miss has to be actionable
    assert [c.row.row_id for c in res.candidates] == ["r-near"]
    assert res.candidates[0].distance.limiting_axis == "output_p50"


def check_exact_regime_matches_without_any_calibration() -> None:
    """Distance 0 needs no threshold, so the schema is usable on day one."""
    res = lookup(Playbook(rows=[_row("r-exact", MOONCAKE)]), _identity(MOONCAKE), UNCALIBRATED_POLICY)
    assert res.status is MatchStatus.EXACT_REGIME
    assert not res.route_to_discovery
    assert res.row is not None and res.row.row_id == "r-exact"
    assert res.distance is not None and res.distance.exact


def check_calibrated_policy_matches_inside_and_rejects_outside() -> None:
    """Once an axis is calibrated, near means near — and far still means no."""
    near = MOONCAKE.model_copy(update={"output_p50": 400})  # log2(400/370) = 0.112
    far = MOONCAKE.model_copy(update={"output_p50": 1480})  # log2(1480/370) = 2.0
    book_near = Playbook(rows=[_row("r-near", near)])
    book_far = Playbook(rows=[_row("r-far", far)])
    policy = _calibrated(0.5)

    ok = lookup(book_near, _identity(MOONCAKE), policy)
    assert ok.status is MatchStatus.NEAR_REGIME and ok.row is not None
    assert not ok.route_to_discovery

    no = lookup(book_far, _identity(MOONCAKE), policy)
    assert no.status is MatchStatus.NO_MATCH and no.route_to_discovery
    assert "output_p50" in no.reason and "2.0" in no.reason

    # the two real traces are far outside any plausible tolerance
    assert lookup(
        Playbook(rows=[_row("r-burstgpt", BURSTGPT)]), _identity(MOONCAKE), _calibrated(2.0)
    ).status is MatchStatus.NO_MATCH


def check_precedence_exact_then_recent_then_conservative() -> None:
    """Three tie-breaks, in order, each demonstrated on its own."""
    near = MOONCAKE.model_copy(update={"output_p50": 400})
    policy = _calibrated(0.5)

    # 1. exact regime beats nearest regime, even with a smaller claimed delta
    res = lookup(
        Playbook(rows=[_row("near-big", near, tput=99.0), _row("exact-small", MOONCAKE, tput=1.0)]),
        _identity(MOONCAKE), policy,
    )
    assert res.row is not None and res.row.row_id == "exact-small"
    assert res.status is MatchStatus.EXACT_REGIME

    # 2. among equals, the most recently verified wins
    res = lookup(
        Playbook(rows=[
            _row("stale", MOONCAKE, verified=T0 - timedelta(days=30)),
            _row("fresh", MOONCAKE, verified=T0),
        ]),
        _identity(MOONCAKE), policy,
    )
    assert res.row is not None and res.row.row_id == "fresh"

    # 3. equally close and equally fresh: the smaller claim wins, because a
    #    wrong row inside the live window costs more than a missed one
    res = lookup(
        Playbook(rows=[_row("bold", MOONCAKE, tput=40.0), _row("modest", MOONCAKE, tput=3.0)]),
        _identity(MOONCAKE), policy,
    )
    assert res.row is not None and res.row.row_id == "modest"


def check_synthesized_prefixes_make_a_prefix_cache_delta_a_floor() -> None:
    """D1-11: BurstGPT has no prefix identity, so its reuse number is a bound.

    D1 synthesizes unique blocks per request, which invents no sharing the source
    never had. A prefix-cache knob measured there saw the *least* reuse possible,
    and the row has to say so or the floor gets quoted as the gain.
    """
    row = _row("r-synth", BURSTGPT)
    assert not row.delta_is_floor  # prefix_synthesized defaults False
    synth = row.model_copy(update={
        "provenance": row.provenance.model_copy(update={"prefix_synthesized": True})
    })
    assert synth.delta_is_floor
    assert "FLOOR" in synth.summary()

    # a non-prefix knob on the same synthesized trace is unaffected — the
    # synthesis only distorts what depends on reuse
    other = synth.model_copy(update={
        "identity": _identity(BURSTGPT, knobs={"max_num_seqs": 64})
    })
    assert not other.delta_is_floor

    # the marker is a substring, so a renamed flag still trips it
    renamed = synth.model_copy(update={
        "identity": _identity(BURSTGPT, knobs={"enable-prefix-caching-v2": True})
    })
    assert renamed.delta_is_floor


def check_extrapolated_rows_lose_ties_but_not_distance() -> None:
    """``/xenv`` is a tie-break below distance, and the ordering is asserted."""
    xenv = MOONCAKE.model_copy(update={"in_envelope": False})
    policy = _calibrated(1.0)

    # equal distance: the in-envelope row wins
    res = lookup(
        Playbook(rows=[_row("extrapolated", xenv), _row("measured", MOONCAKE)]),
        _identity(MOONCAKE), policy,
    )
    assert res.row is not None and res.row.row_id == "measured"

    # nearer-but-extrapolated still beats a far in-envelope row: distance is the
    # workload question and comes first (this is where we depart from todo.md)
    far = MOONCAKE.model_copy(update={"output_p50": 640})  # 0.79 away
    res = lookup(
        Playbook(rows=[_row("near-xenv", xenv), _row("far-inenv", far)]),
        _identity(MOONCAKE), policy,
    )
    assert res.row is not None and res.row.row_id == "near-xenv"


def check_replay_conditions_are_carried_into_provenance() -> None:
    """The 512-vs-16 finding is a field, so a wrong-block-size row is checkable."""
    row = _row("r-mooncake", MOONCAKE)
    assert row.provenance.replay_chunk_hash_size is None  # unrecorded, not assumed
    pinned = row.model_copy(update={
        "provenance": row.provenance.model_copy(
            update={"replay_chunk_hash_size": 512, "replay_self_timed": True}
        )
    })
    assert pinned.provenance.replay_chunk_hash_size == 512
    # and D2's stored verdict has somewhere to go without being recomputed
    assert pinned.delta.latency_blowout is None
    assert pinned.delta.model_copy(update={"latency_blowout": True}).latency_blowout is True


def check_invalidated_row_is_kept_and_never_selected() -> None:
    """Retirement is a field with a reason, not a deletion."""
    dead = _row("r-dead", MOONCAKE)
    dead = dead.model_copy(update={
        "invalidated": Invalidation(reason="vLLM 0.11 -> 0.12 scheduler rewrite", at=T0)
    })
    book = Playbook(rows=[dead])
    assert len(book.rows) == 1 and book.selectable() == []
    res = lookup(book, _identity(MOONCAKE), UNCALIBRATED_POLICY)
    assert res.status is MatchStatus.NO_MATCH
    assert "scheduler rewrite" in res.rejected["r-dead"]  # the reason survives the miss


def check_a_row_cannot_exist_without_its_evidence() -> None:
    """The type refuses rows that could not have come through the promotion rule."""
    with pytest.raises(ValidationError, match="knobs is required"):
        _identity(MOONCAKE, knobs={})
    with pytest.raises(ValidationError, match="single run has no variance"):
        MeasuredDelta(throughput_pct=8.0, ttft_p99_ms=-1.0, itl_p99_ms=0.0, repeats=1)
    with pytest.raises(ValidationError):  # trace_sha256 is required
        Provenance(trace_source="mooncake", trace_sha256="", regime_label="x")
    with pytest.raises(ValidationError):  # throughput alone cannot be promoted (D2 criterion 3)
        MeasuredDelta(throughput_pct=8.0, repeats=5)


def check_examples_ship_nothing_selectable() -> None:
    """Every worked example is labelled, and none of them can be applied."""
    book = Playbook.model_validate(json.loads(EXAMPLES.read_text(encoding="utf-8")))
    assert len(book.rows) == EXAMPLE_ROWS
    assert all(r.evidence is Evidence.ILLUSTRATIVE for r in book.rows)
    assert book.selectable() == []  # an example in a live file is the failure mode

    # round-trips byte-identically: the file is the schema, not a rendering of it
    assert json.loads(json.dumps(book.model_dump(mode="json"))) == json.loads(
        EXAMPLES.read_text(encoding="utf-8")
    )

    # the illustrative rows carry real regimes and real trace checksums
    by_id = {r.row_id: r for r in book.rows}
    assert by_id["ex1-prefix-cache-mooncake"].identity.regime.input_p50 == MOONCAKE.input_p50
    assert by_id["ex2-max-num-seqs-burstgpt"].identity.regime.input_p50 == BURSTGPT.input_p50
    assert all(len(r.provenance.trace_sha256) == 64 for r in book.rows)
    # the biggest claimed delta in the file is the scoreboard row, gated by equality
    assert max(book.rows, key=lambda r: r.delta.throughput_pct).row_id == (
        "ex5-scoreboard-not-production"
    )
    # exactly one example is a floor, and it is the prefix-cache knob on the
    # source with no prefix identity — both halves of D1-11 have an example
    floors = [r.row_id for r in book.rows if r.delta_is_floor]
    assert floors == ["ex6-prefix-cache-on-a-synthesized-trace"], floors
    assert by_id["ex2-max-num-seqs-burstgpt"].provenance.prefix_synthesized
    assert not by_id["ex2-max-num-seqs-burstgpt"].delta_is_floor  # synthesized != floor
    assert all(r.provenance.replay_chunk_hash_size == 512 for r in book.rows)

    # and lookup refuses all of them even for a perfectly matching query
    res = lookup(book, _identity(MOONCAKE), UNCALIBRATED_POLICY)
    assert res.status is MatchStatus.NO_MATCH
    assert all("never selectable" in w or "invalidated" in w for w in res.rejected.values())


def check_regime_is_imported_not_redeclared() -> None:
    """One coordinate system. A second Regime here would drift within a week."""
    import gitm.traffic.regime as d1

    assert RowIdentity.model_fields["regime"].annotation is d1.Regime
    # and every distance axis is a real field on it, by name
    from gitm.playbook.match import AXIS_METRICS

    assert set(AXIS_METRICS) <= set(d1.Regime.model_fields)


# --- the last mile: seam 3's records become a row ----------------------------
#: Deltas injected into the treatment arm's copy of the real result JSON. The
#: *runs* are real (the committed 0.28.0 result); the difference between the arms
#: is constructed, because a second real arm needs a knob and a GPU. What is
#: under test is the arithmetic and the refusals, not the numbers.
ARM_TPUT_GAIN = 1.10
ARM_TTFT_DELTA_MS = -40.0


def _runs(**bump):
    """N ``BenchRun`` records off the committed real result, optionally bumped."""
    from gitm.traffic._selftest import _real_run
    from gitm.traffic.results import join_result

    result, plan, reg, _ = _real_run()
    return result, plan, reg, (lambda n=2, **kw: [join_result({**result, **kw}, plan, reg)
                                                 for _ in range(n)])


def check_two_bench_runs_become_a_row() -> None:
    """The last mile. Seam 3 makes one arm; a row is the difference between two.

    Before this existed, nothing could populate a row end to end no matter how
    complete the join was — which is why every shipped example is illustrative.
    """
    from gitm.playbook.schema import PENDING_ADIT, row_from_runs

    result, _, reg, mk = _runs()
    base = mk(2)
    treat = mk(2,
               output_throughput=result["output_throughput"] * ARM_TPUT_GAIN,
               p99_ttft_ms=result["p99_ttft_ms"] + ARM_TTFT_DELTA_MS)

    row = row_from_runs(
        "r-from-real-runs", base, treat,
        model=MODEL, model_revision=REV, gpu_sku=H100, env=ENV,
        knobs={"enable_prefix_caching": True},
    )

    # the arithmetic
    assert row.delta.repeats == 2
    assert math.isclose(row.delta.throughput_pct, (ARM_TPUT_GAIN - 1) * 100, rel_tol=1e-9)
    assert math.isclose(row.delta.ttft_p99_ms, ARM_TTFT_DELTA_MS, rel_tol=1e-9)

    # provenance came off the runs, not off the caller — the whole point
    assert row.provenance.trace_sha256 == base[0].source.sha256
    assert len(row.provenance.trace_sha256) == 64
    assert row.provenance.regime_label == reg.label()
    assert row.provenance.replay_chunk_hash_size == 512
    assert row.provenance.replay_self_timed is True
    assert row.identity.regime == reg

    # measured, and therefore selectable — the first row in the repo that is
    assert row.evidence is Evidence.MEASURED and row.selectable

    # ...and it still says what it does not know
    assert row.provenance.config_capture == PENDING_ADIT
    assert row.provenance.promotion_rule.startswith(PENDING_ADIT)
    assert any("R1" in n for n in row.notes), row.notes
    assert row.delta.latency_blowout is None      # D2 owns the predicate
    assert row.delta.throughput_ci95_pct is None  # D2 owns the variance rule


def check_a_row_refuses_arms_that_are_not_one_experiment() -> None:
    """Every way a pair of runs is not a delta. Each is a wrong row prevented."""
    from gitm.playbook.schema import row_from_runs

    _, _, reg, mk = _runs()
    ident = dict(model=MODEL, model_revision=REV, gpu_sku=H100, env=ENV,
                 knobs={"enable_prefix_caching": True})

    def refused(base, treat, needle):
        with pytest.raises((ValueError, ValidationError)) as e:
            row_from_runs("r-bad", base, treat, **ident)
        assert needle in str(e.value), str(e.value)

    refused(mk(2), mk(1), "unequal arms")            # the interleave broke
    refused(mk(1), mk(1), "repeats=1")               # delegated to MeasuredDelta
    refused(mk(2), mk(2, failed=1), "promotable")    # a run with failures
    refused(mk(2), mk(2, completed=39), "promotable")  # did not replay the trace
    refused([], mk(2), "both arms are required")

    # two different workloads: same result JSON, a regime that labels differently
    from gitm.traffic.results import join_result
    result, plan, _, _ = _runs()
    other = reg.model_copy(update={"input_p50": 100, "input_p95": 200})
    assert other.label() != reg.label()
    refused(mk(2), [join_result(result, plan, other) for _ in range(2)],
            "did not run the same workload")


CHECKS = (
    check_log2_ratio_is_scale_free,
    check_dispersion_distance_handles_flat_traces,
    check_linf_is_the_worst_axis,
    check_rate_is_not_in_the_default_axes,
    check_tolerance_requires_calibration,
    check_exact_gates_reject_before_distance,
    check_uncalibrated_policy_routes_to_discovery,
    check_exact_regime_matches_without_any_calibration,
    check_calibrated_policy_matches_inside_and_rejects_outside,
    check_precedence_exact_then_recent_then_conservative,
    check_extrapolated_rows_lose_ties_but_not_distance,
    check_synthesized_prefixes_make_a_prefix_cache_delta_a_floor,
    check_replay_conditions_are_carried_into_provenance,
    check_invalidated_row_is_kept_and_never_selected,
    check_a_row_cannot_exist_without_its_evidence,
    check_examples_ship_nothing_selectable,
    check_regime_is_imported_not_redeclared,
    check_two_bench_runs_become_a_row,
    check_a_row_refuses_arms_that_are_not_one_experiment,
)


def run_all() -> int:
    for fn in CHECKS:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"selftest ok -- {len(CHECKS)} checks, 2 real regimes, 0 calibrated axes")
    return 0
