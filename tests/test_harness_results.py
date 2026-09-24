"""Reading a cluster experiment back into the record the loop ranks from.

The harness runs experiments across a fleet and hands the server to
``gitm capture serve``, which leaves ``serving_summary.json`` and
``run_manifest.json`` per arm. ``load_history`` reads
``runs/<id>/verification.json``, and only ``run_loop`` writes one — so every
result measured on the cluster was invisible to the ranking that reads history.
A loop that proposes experiments and cannot see their results is the failure the
history reader exists to prevent, one layer out.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from gitm.kernels.library import load_library
from gitm.optimizer.harness_results import (
    CaptureError,
    compare,
    fingerprint_of,
    knob_difference,
    read_capture,
    resolve_lever,
    write_comparison,
)
from gitm.optimizer.history import load_history, record_for
from gitm.optimizer.qualification import fingerprint as trace_fingerprint
from gitm.tracer.capture import write_trace_jsonl
from gitm.tracer.schema import KernelEvent, Trace

LIB = load_library(workload="vllm-decode")

BASE_ARGV = ["--tensor-parallel-size", "2"]
LOAD = {"requests": 512, "concurrency": 256, "input_tokens": 1024,
        "output_tokens": 256, "seed": 42}


def _trace(vendor="amd", n=12):
    events = [
        KernelEvent(name=f"k{i % 3}", start_ns=i * 100, end_ns=i * 100 + 90,
                    stream_id=7, device_id=0, correlation_id=i,
                    grid_x=i % 2 + 1, grid_y=1, grid_z=1,
                    block_x=128, block_y=1, block_z=1)
        for i in range(n)
    ]
    return Trace(workload_id="vllm-serve", fingerprint="", run_id="r", device_count=1,
                 vendor=vendor, captured_at_ns=0, duration_ns=10 ** 6, events=events)


def _arm(root, name, *, argv=None, rps=40.0, model="Kimi-K2.5", tracing="cupti",
         load=None, summary=None, manifest=None, trace=True):
    """One arm's directory in the shape `gitm capture serve` writes it."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if trace:
        write_trace_jsonl(d / "trace.jsonl", _trace())
    (d / "serving_summary.json").write_text(json.dumps(summary if summary is not None else {
        "mode": "drive", "tracing": tracing, "nvtx": False, "wall_s": 300.0,
        "client": {"latency_source": "client", "n_failed_requests": 0,
                   "n_requests": 512, "goodput_rps": rps, "window_s": 300.0},
    }))
    (d / "run_manifest.json").write_text(json.dumps(manifest if manifest is not None else {
        "workload_id": "vllm-serve", "capture_mode": "serve", "served_model": model,
        "serve_argv": BASE_ARGV if argv is None else argv,
        "load": LOAD if load is None else load,
    }))
    return d


# --------------------------------------------------------------------------- #
# the round trip                                                               #
# --------------------------------------------------------------------------- #
def test_a_cluster_result_reaches_the_record_the_loop_ranks_from(tmp_path):
    """The whole point: a pair of harness arms becomes a lever the ranking can
    see, keyed the same way a local run would be."""
    base = _arm(tmp_path, "tp2", rps=40.0)
    cand = _arm(tmp_path, "tp2-ep", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=59.6)

    write_comparison(read_capture(base), read_capture(cand), library=LIB,
                     out_dir=tmp_path / "runs" / "cluster-1",
                     gpu_sku="AMD Instinct MI355X", fingerprint="kimi-k2.5-mi355x")

    rec = record_for(load_history(tmp_path / "runs"), "enable_expert_parallel",
                     gpu_sku="AMD Instinct MI355X", fingerprint="kimi-k2.5-mi355x")
    assert rec is not None
    assert abs(rec.mean_delta - 0.49) < 1e-9
    assert (rec.wins, rec.losses) == (1, 0)


def test_a_measured_win_is_a_win_and_not_a_loss(tmp_path):
    """``kept`` maps to the verdict, and leaving it False because no rollback
    gate ran would record every cluster win as a loss — demoting the lever the
    result proves. A harness arm runs standalone, so there is nothing to roll
    back and the number is the whole question."""
    base = _arm(tmp_path, "b", rps=40.0)
    win = _arm(tmp_path, "w", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=59.6)
    loss = _arm(tmp_path, "l", argv=[*BASE_ARGV, "--enable-dbo"], rps=36.4)

    assert compare(read_capture(base), read_capture(win), library=LIB).kept is True
    assert compare(read_capture(base), read_capture(loss), library=LIB).kept is False


def test_a_change_inside_the_band_is_not_significant(tmp_path):
    base = _arm(tmp_path, "b", rps=40.0)
    noise = _arm(tmp_path, "n", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=40.4)

    rec = compare(read_capture(base), read_capture(noise), library=LIB)

    assert rec.significant is False
    assert rec.kept is False


# --------------------------------------------------------------------------- #
# arms that are not an A/B                                                     #
# --------------------------------------------------------------------------- #
def test_a_traced_arm_against_an_untraced_one_is_refused(tmp_path):
    """Tracing costs throughput. Comparing across arms would report the tracer's
    overhead as the lever's effect — and the harness's own default arm list makes
    this easy to do by accident."""
    base = _arm(tmp_path, "b", tracing="off", rps=44.0)
    cand = _arm(tmp_path, "c", argv=[*BASE_ARGV, "--enable-expert-parallel"],
                tracing="cupti", rps=40.0)

    with pytest.raises(CaptureError, match="not an A/B"):
        compare(read_capture(base), read_capture(cand), library=LIB)


def test_a_different_load_shape_is_refused(tmp_path):
    base = _arm(tmp_path, "b", rps=40.0)
    cand = _arm(tmp_path, "c", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=59.6,
                load={**LOAD, "concurrency": 32})

    with pytest.raises(CaptureError, match="not an A/B"):
        compare(read_capture(base), read_capture(cand), library=LIB)


def test_a_different_model_is_refused(tmp_path):
    base = _arm(tmp_path, "b", rps=40.0)
    cand = _arm(tmp_path, "c", argv=[*BASE_ARGV, "--enable-expert-parallel"],
                model="GLM-5.2", rps=59.6)

    with pytest.raises(CaptureError, match="not an A/B"):
        compare(read_capture(base), read_capture(cand), library=LIB)


def test_identical_flags_are_refused_rather_than_recorded_as_a_lever(tmp_path):
    """Two arms of the same config measure run-to-run scatter. Recording that as
    an intervention would put noise in the record under a lever's name."""
    base = _arm(tmp_path, "b", rps=40.0)
    same = _arm(tmp_path, "s", rps=41.0)

    with pytest.raises(CaptureError, match="no intervention"):
        compare(read_capture(base), read_capture(same), library=LIB)


# --------------------------------------------------------------------------- #
# reading a directory                                                          #
# --------------------------------------------------------------------------- #
def test_a_capture_with_no_throughput_raises_rather_than_half_reporting(tmp_path):
    """A comparison built from it would state a delta against nothing."""
    d = _arm(tmp_path, "b", summary={"mode": "drive", "tracing": "cupti",
                                     "client": {"n_failed_requests": 0}})

    with pytest.raises(CaptureError, match="no throughput"):
        read_capture(d)


def test_goodput_of_zero_is_a_measurement_not_a_missing_field(tmp_path):
    """A run that met no SLO really did achieve zero goodput. Falling back to
    raw request rate there would report throughput the run did not deliver."""
    d = _arm(tmp_path, "b", rps=0.0)

    assert read_capture(d).throughput == 0.0


def test_request_rate_is_used_only_when_goodput_is_absent(tmp_path):
    d = _arm(tmp_path, "b", summary={
        "mode": "drive", "tracing": "cupti", "wall_s": 256.0,
        "client": {"n_requests": 512, "window_s": 256.0}})

    assert read_capture(d).throughput == 2.0


def test_a_missing_directory_says_so(tmp_path):
    with pytest.raises(CaptureError, match="not a directory"):
        read_capture(tmp_path / "nope")


def test_malformed_json_names_the_file(tmp_path):
    d = _arm(tmp_path, "b")
    (d / "serving_summary.json").write_text("{ truncated")

    with pytest.raises(CaptureError, match="serving_summary.json"):
        read_capture(d)


# --------------------------------------------------------------------------- #
# which knob moved                                                             #
# --------------------------------------------------------------------------- #
def test_the_knob_is_the_flag_the_candidate_added(tmp_path):
    base = read_capture(_arm(tmp_path, "b"))
    cand = read_capture(_arm(tmp_path, "c", argv=[*BASE_ARGV, "--max-num-seqs", "512"]))

    assert knob_difference(base, cand) == {"--max-num-seqs": "512"}


def test_a_bare_switch_reads_as_true(tmp_path):
    base = read_capture(_arm(tmp_path, "b"))
    cand = read_capture(_arm(tmp_path, "c", argv=[*BASE_ARGV, "--enforce-eager"]))

    assert knob_difference(base, cand) == {"--enforce-eager": True}


def test_a_dropped_flag_is_reported_rather_than_ignored(tmp_path):
    """It used to be left out, which was worse than incomplete: the measured
    delta would be credited entirely to whatever the candidate *added*, while a
    removal had moved it too."""
    base = read_capture(_arm(tmp_path, "b", argv=[*BASE_ARGV, "--enforce-eager"]))
    cand = read_capture(_arm(tmp_path, "c", argv=BASE_ARGV))

    assert knob_difference(base, cand) == {"--enforce-eager": None}


def test_launch_settings_are_not_part_of_the_lever(tmp_path):
    """Real manifests carry --host and --port. Where a server binds says nothing
    about what it computes, and folding a port into the knob under test invents a
    lever no catalog entry can match."""
    base = read_capture(_arm(tmp_path, "b", argv=[*BASE_ARGV, "--port", "8000"]))
    cand = read_capture(_arm(tmp_path, "c", argv=[*BASE_ARGV, "--port", "8001",
                                                  "--enable-expert-parallel"]))

    assert knob_difference(base, cand) == {"--enable-expert-parallel": True}


# --------------------------------------------------------------------------- #
# the record has to be findable, which is the whole point                      #
# --------------------------------------------------------------------------- #
def test_the_lever_name_comes_from_the_catalog_not_from_the_flag(tmp_path):
    """Ranking looks a record up by ``spec.name``, and the names do not follow
    from the flags: ``--max-num-seqs`` is the lever ``max_num_seqs_dynamic``. A
    name invented from the flag is a record nothing will ever look up."""
    base = read_capture(_arm(tmp_path, "b"))
    cand = read_capture(_arm(tmp_path, "c", argv=[*BASE_ARGV, "--max-num-seqs", "256"]))

    assert compare(base, cand, library=LIB).intervention_name == "max_num_seqs_dynamic"


def test_a_valued_flag_must_carry_the_value_the_lever_names(tmp_path):
    """No catalog entry shares a knob with another, so matching on the knob
    alone resolves *every* setting of it to the one entry. ``max_num_seqs_dynamic``
    is the setting 256; an arm that ran 512 measured something else, and filing
    it under that name is a number the ranking will trust for a config nobody
    ran."""
    base = read_capture(_arm(tmp_path, "b"))
    cand = read_capture(_arm(tmp_path, "c", argv=[*BASE_ARGV, "--max-num-seqs", "512"]))

    with pytest.raises(CaptureError, match="no catalog entry"):
        compare(base, cand, library=LIB)


def test_resolution_is_by_knob_not_by_name():
    assert resolve_lever("--max-num-seqs", "256", LIB).name == "max_num_seqs_dynamic"
    assert resolve_lever("--enable-expert-parallel", True, LIB).name == "enable_expert_parallel"
    assert resolve_lever("--not-a-real-knob", 1, LIB) is None


def test_an_arm_is_never_credited_to_the_opposite_lever(tmp_path):
    """The only catalog entry on the ``enforce_eager`` knob is
    ``cuda_graphs_enable``, which sets it *false*. An arm that passes
    ``--enforce-eager`` ran the opposite intervention, so crediting it there
    would record a win for eager mode as evidence for disabling it — and the
    loop would rank the reverse of what was measured."""
    base = read_capture(_arm(tmp_path, "b"))
    cand = read_capture(_arm(tmp_path, "c", argv=[*BASE_ARGV, "--enforce-eager"], rps=59.6))

    assert resolve_lever("--enforce-eager", True, LIB) is None
    with pytest.raises(CaptureError, match="no catalog entry"):
        compare(base, cand, library=LIB)


def test_removing_a_boolean_flag_is_the_lever_that_turns_it_off(tmp_path):
    """``cuda_graphs_enable`` sets ``enforce_eager`` false, which a server
    expresses as the *absence* of ``--enforce-eager``. Refusing every removal
    would leave that lever unmeasurable through the harness."""
    base = read_capture(_arm(tmp_path, "b", argv=[*BASE_ARGV, "--enforce-eager"], rps=40.0))
    cand = read_capture(_arm(tmp_path, "c", argv=list(BASE_ARGV), rps=59.6))

    rec = compare(base, cand, library=LIB)

    assert rec.intervention_name == "cuda_graphs_enable"
    assert rec.value is False
    assert rec.kept is True


def test_removing_a_valued_flag_is_refused(tmp_path):
    """Dropping ``--max-num-seqs`` restores a server default this module does
    not know, so there is no lever whose value it realises."""
    base = read_capture(_arm(tmp_path, "b", argv=[*BASE_ARGV, "--max-num-seqs", "256"]))
    cand = read_capture(_arm(tmp_path, "c", argv=list(BASE_ARGV)))

    with pytest.raises(CaptureError, match="no catalog entry"):
        compare(base, cand, library=LIB)


def test_a_flag_with_no_catalog_entry_is_refused(tmp_path):
    """Recording it under an invented name would put a measurement in the record
    that the ranking can never find — present, and useless."""
    base = read_capture(_arm(tmp_path, "b"))
    cand = read_capture(_arm(tmp_path, "c", argv=[*BASE_ARGV, "--some-future-flag", "7"]))

    with pytest.raises(CaptureError, match="no catalog entry"):
        compare(base, cand, library=LIB)


def test_more_than_one_changed_flag_is_refused(tmp_path):
    base = read_capture(_arm(tmp_path, "b"))
    cand = read_capture(_arm(tmp_path, "c", argv=[*BASE_ARGV, "--enforce-eager",
                                                  "--enable-expert-parallel"]))

    with pytest.raises(CaptureError, match="cannot be credited to one lever"):
        compare(base, cand, library=LIB)


# --------------------------------------------------------------------------- #
# the fingerprint the loop actually filters on                                 #
# --------------------------------------------------------------------------- #
def test_the_fingerprint_matches_what_the_loop_computes(tmp_path):
    """The loop filters history on qualification.fingerprint(trace). A record
    filed under anything else — a model name, say — is written and then filtered
    straight back out, which is invisible rather than merely coarse.

    Streamed rather than loaded, because a real capture is millions of kernels;
    this pins that the two agree."""
    arm = _arm(tmp_path, "b")

    assert fingerprint_of(read_capture(arm)) == trace_fingerprint(_trace())


def test_a_synced_capture_uses_the_trace_beside_it(tmp_path):
    """Cluster captures are read after ``sync_results.sh`` has mirrored them, so
    the manifest holds the path the trace had on the machine that produced it.
    Following that path finds nothing — and the measurement never reaches the
    loop — or finds a *different* run's trace, which is worse: the result is
    filed against a workload it did not run."""
    stale = tmp_path / "elsewhere"
    stale.mkdir()
    write_trace_jsonl(stale / "trace.jsonl", _trace(vendor="nvidia"))

    arm = _arm(tmp_path, "b", manifest={
        "workload_id": "vllm-serve", "capture_mode": "serve",
        "served_model": "Kimi-K2.5", "serve_argv": BASE_ARGV, "load": LOAD,
        "trace": {"path": str(stale / "trace.jsonl")},
    })

    assert read_capture(arm).trace_path == arm / "trace.jsonl"
    assert fingerprint_of(read_capture(arm)) == trace_fingerprint(_trace())


def test_a_capture_read_in_place_still_honours_its_manifest(tmp_path):
    """The recorded path is not ignored, only outranked: a capture with no
    trace beside it falls back to what the manifest says."""
    away = tmp_path / "away"
    away.mkdir()
    write_trace_jsonl(away / "capture.jsonl", _trace())

    arm = _arm(tmp_path, "b", trace=False, manifest={
        "workload_id": "vllm-serve", "capture_mode": "serve",
        "served_model": "Kimi-K2.5", "serve_argv": BASE_ARGV, "load": LOAD,
        "trace": {"path": str(away / "capture.jsonl")},
    })

    assert fingerprint_of(read_capture(arm)) == trace_fingerprint(_trace())


def test_an_untraced_arm_cannot_be_fingerprinted(tmp_path):
    """`--no-trace` arms exist — they are the baseline of a tracing-overhead
    measurement. One cannot be filed against a workload the loop would know."""
    arm = _arm(tmp_path, "b", trace=False)

    with pytest.raises(CaptureError, match="no trace.jsonl"):
        fingerprint_of(read_capture(arm))


def test_the_written_record_carries_the_trace_fingerprint(tmp_path):
    base = _arm(tmp_path, "b")
    cand = _arm(tmp_path, "c", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=59.6)

    write_comparison(read_capture(base), read_capture(cand), library=LIB,
                     out_dir=tmp_path / "runs" / "c1", gpu_sku="MI355X")

    h = load_history(tmp_path / "runs")
    rec = record_for(h, "enable_expert_parallel", gpu_sku="MI355X",
                     fingerprint=trace_fingerprint(_trace()))
    assert rec is not None and rec.wins == 1


# --------------------------------------------------------------------------- #
# not clobbering what is already there                                         #
# --------------------------------------------------------------------------- #
def test_an_existing_export_is_never_overwritten(tmp_path):
    """The loop's own exports live under the same tree. A reused run id would
    replace whatever a directory records with this single comparison."""
    base = _arm(tmp_path, "b")
    cand = _arm(tmp_path, "c", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=59.6)
    out = tmp_path / "runs" / "c1"
    write_comparison(read_capture(base), read_capture(cand), library=LIB, out_dir=out)

    with pytest.raises(CaptureError, match="already exists"):
        write_comparison(read_capture(base), read_capture(cand), library=LIB, out_dir=out)


# --------------------------------------------------------------------------- #
# arms measured on different metrics                                           #
# --------------------------------------------------------------------------- #
def test_goodput_against_raw_rate_is_refused(tmp_path):
    """One arm SLO-qualified and the other not is a comparison of two different
    numbers, and can record a win that never happened."""
    base = _arm(tmp_path, "b", rps=40.0)
    cand = _arm(tmp_path, "c", argv=[*BASE_ARGV, "--enable-expert-parallel"], summary={
        "mode": "drive", "tracing": "cupti", "wall_s": 300.0,
        "client": {"n_requests": 512, "goodput_rps": None, "window_s": 300.0}})

    with pytest.raises(CaptureError, match="different metrics"):
        compare(read_capture(base), read_capture(cand), library=LIB)


def test_a_null_goodput_is_not_the_same_as_an_absent_one(tmp_path):
    """null is a present field carrying no number, which is what a capture writes
    when its window had no usable duration. It falls back to the raw rate, and
    the capture records that it did so."""
    d = _arm(tmp_path, "b", summary={
        "mode": "drive", "tracing": "cupti", "wall_s": 256.0,
        "client": {"n_requests": 512, "goodput_rps": None, "window_s": 256.0}})

    cap = read_capture(d)

    assert cap.throughput == 2.0
    assert cap.goodput is False


def test_the_export_does_not_claim_a_gate_that_never_ran(tmp_path):
    """`kept` and `metric` mean different things depending on `via`, and the
    protocol block is shared by both paths. A harness arm ran standalone on a
    cluster: there was no rollback gate behind it and no decode-throughput
    number in it, so a blanket description would claim a provenance and a unit
    that half these records do not have."""
    base = _arm(tmp_path, "b", rps=40.0)
    cand = _arm(tmp_path, "c", argv=[*BASE_ARGV, "--enable-expert-parallel"], rps=59.6)

    out = write_comparison(read_capture(base), read_capture(cand), library=LIB,
                           out_dir=tmp_path / "runs" / "cluster-1",
                           gpu_sku="AMD Instinct MI355X", fingerprint="kimi-k2.5-mi355x")
    protocol = json.loads(pathlib.Path(out).read_text())["protocol"]

    assert json.loads(pathlib.Path(out).read_text())["results"][0]["via"] == "harness"
    for field in ("kept", "metric"):
        assert "harness" in protocol[field], f"{field} does not describe harness records"
        assert protocol[field].startswith("per record"), f"{field} is stated as a blanket claim"
    assert "requests/sec" in protocol["metric"]
    assert "measured delta" in protocol["kept"]
