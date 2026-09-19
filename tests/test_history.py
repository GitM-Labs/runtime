"""Reading previous runs' verification exports back into a per-lever record."""

from __future__ import annotations

import json
import os

from gitm.optimizer.history import (
    load_history,
    record_for,
    render_history,
)


def _result(name, *, kept=True, significant=True, delta=0.05):
    return {
        "intervention_name": name,
        "summary": f"{name} summary",
        "knob": "some_knob",
        "value": 1,
        "source": "test",
        "baseline_tps": 100.0,
        "candidate_tps": 100.0 * (1.0 + delta),
        "speedup": 1.0 + delta,
        "delta": delta,
        "reps": 3,
        "significant": significant,
        "kept": kept,
        "via": "hot-swap",
    }


def _run(tmp_path, run_id, results, *, gpu_sku="NVIDIA H100 80GB", mtime=None, body=None):
    d = tmp_path / run_id
    d.mkdir(parents=True, exist_ok=True)
    if results is None and body is None:
        return d  # a run folder with no export at all
    path = d / "verification.json"
    if body is not None:
        path.write_text(body)
    else:
        path.write_text(json.dumps({
            "schema": 1,
            "provenance": {"run_id": run_id, "workload_id": "vllm-decode"},
            "environment": {"gpu_sku": gpu_sku, "driver_cuda": None, "torch_cuda": None},
            "protocol": {},
            "results": results,
        }))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return d


def test_two_wins_for_the_same_lever_accumulate(tmp_path):
    _run(tmp_path, "run-a", [_result("kv_cache_dtype_fp8", delta=0.04)])
    _run(tmp_path, "run-b", [_result("kv_cache_dtype_fp8", delta=0.06)])

    h = load_history(tmp_path)
    r = record_for(h, "kv_cache_dtype_fp8", gpu_sku="NVIDIA H100 80GB")

    assert h.runs_read == 2
    assert (r.runs, r.attempts, r.wins, r.losses) == (2, 2, 2, 0)
    assert r.mean_delta == 0.05
    assert (r.best_delta, r.worst_delta) == (0.06, 0.04)
    assert not r.conflicted


def test_a_win_and_a_loss_conflict_rather_than_cancelling(tmp_path):
    """Both counts survive. Averaging a disagreement into 'neutral' would present
    it as a measurement rather than as the open question it is."""
    _run(tmp_path, "run-a", [_result("enable_expert_parallel", kept=True, delta=0.20)])
    _run(tmp_path, "run-b", [_result("enable_expert_parallel", kept=False, delta=-0.18)])

    r = record_for(load_history(tmp_path), "enable_expert_parallel",
                   gpu_sku="NVIDIA H100 80GB")

    assert (r.wins, r.losses) == (1, 1)
    assert r.conflicted
    assert r.runs == 2


def test_kept_but_not_significant_is_inconclusive_not_a_win(tmp_path):
    _run(tmp_path, "run-a", [_result("async_scheduling", kept=True, significant=False)])

    r = record_for(load_history(tmp_path), "async_scheduling", gpu_sku="NVIDIA H100 80GB")

    assert (r.wins, r.losses, r.inconclusive) == (0, 0, 1)
    assert not r.conflicted


def test_the_same_lever_on_two_gpus_stays_two_records(tmp_path):
    """A result on one GPU says nothing about another."""
    _run(tmp_path, "run-a", [_result("cuda_graphs_enable")], gpu_sku="NVIDIA H100 80GB")
    _run(tmp_path, "run-b", [_result("cuda_graphs_enable")], gpu_sku="AMD MI355X")

    h = load_history(tmp_path)

    assert len(h) == 2
    assert record_for(h, "cuda_graphs_enable", gpu_sku="AMD MI355X").runs == 1
    assert record_for(h, "cuda_graphs_enable", gpu_sku="NVIDIA H100 80GB").runs == 1


def test_gpu_filter_excludes_other_boxes_and_says_why(tmp_path):
    _run(tmp_path, "run-a", [_result("cuda_graphs_enable")], gpu_sku="NVIDIA H100 80GB")
    _run(tmp_path, "run-b", [_result("cuda_graphs_enable")], gpu_sku="AMD MI355X")

    h = load_history(tmp_path, gpu_sku="AMD MI355X")

    assert h.runs_read == 1
    assert len(h) == 1
    assert h.filtered == 1
    assert "run-a" not in h.skipped


def test_a_run_with_no_export_is_counted_not_dropped(tmp_path):
    """A crashed run leaves the folder without an export. A reader that ignores it
    silently makes a history of 1 run look identical to a history of 10."""
    _run(tmp_path, "run-a", [_result("kv_cache_dtype_fp8")])
    _run(tmp_path, "run-crashed", None)

    h = load_history(tmp_path)

    assert h.runs_read == 1
    assert h.skipped["run-crashed"] == "no verification.json"


def test_a_malformed_export_is_skipped_rather_than_raising(tmp_path):
    _run(tmp_path, "run-a", [_result("kv_cache_dtype_fp8")])
    _run(tmp_path, "run-bad", None, body="{not json at all")

    h = load_history(tmp_path)

    assert h.runs_read == 1
    assert "unreadable" in h.skipped["run-bad"]


def test_a_lever_never_tried_is_none_not_a_zeroed_record(tmp_path):
    """'never measured' and 'measured at zero' point opposite ways for a caller
    deciding whether this lever is worth an experiment."""
    _run(tmp_path, "run-a", [_result("kv_cache_dtype_fp8")])

    h = load_history(tmp_path)

    assert record_for(h, "quantization_awq", gpu_sku="NVIDIA H100 80GB") is None


def test_last_run_id_follows_the_newest_export(tmp_path):
    _run(tmp_path, "run-old", [_result("swap_space_dynamic")], mtime=1_000_000)
    _run(tmp_path, "run-new", [_result("swap_space_dynamic")], mtime=2_000_000)

    r = record_for(load_history(tmp_path), "swap_space_dynamic",
                   gpu_sku="NVIDIA H100 80GB")

    assert r.last_run_id == "run-new"


def test_missing_runs_dir_is_empty_not_an_error(tmp_path):
    h = load_history(tmp_path / "nope")

    assert h.runs_read == 0
    assert len(h) == 0
    assert h.skipped


def test_render_names_the_levers_and_the_skips(tmp_path):
    _run(tmp_path, "run-a", [_result("enable_prefix_caching", delta=0.07)])
    _run(tmp_path, "run-b", [_result("enable_prefix_caching", kept=False, delta=-0.02)])
    _run(tmp_path, "run-crashed", None)

    out = render_history(load_history(tmp_path))

    assert "enable_prefix_caching" in out
    assert "CONFLICTED" in out
    assert "read 2 runs" in out and "skipped 1" in out


# --------------------------------------------------------------------------- #
# round-trip against the real writer                                          #
# --------------------------------------------------------------------------- #
# Every fixture above is hand-written to match verification_export's shape. That
# proves the reader is self-consistent, not that it agrees with the producer. The
# tests below drive the real build_record/write_verification path so a change to
# the export format fails here rather than silently emptying the history.


def _real_export(tmp_path, run_id, specs_and_abs, *, gpu_sku="NVIDIA H100 80GB"):
    from gitm.kernels.spec import InterventionSpec
    from gitm.optimizer.apply import ApplyResult, EngineABResult
    from gitm.optimizer.report import Provenance
    from gitm.optimizer.verification_export import build_record, write_verification

    records = []
    for name, knob, value, speedup, kept, significant in specs_and_abs:
        spec = InterventionSpec(
            name=name, summary=f"{name} summary", knob=knob, value=value,
            expected_delta_mean=0.08, expected_delta_lo=0.02, expected_delta_hi=0.14,
            source="https://docs.vllm.ai/example",
        )
        ab = EngineABResult(
            knob=knob, value=value, baseline_tps=100.0,
            candidate_tps=100.0 * speedup, speedup=speedup, kept=kept,
            via="hot-swap", baseline_std=1.0, candidate_std=1.0, reps=3,
            significant=significant,
        )
        records.append(
            build_record(spec, ab, ApplyResult(True, not kept, speedup - 1.0))
        )

    prov = Provenance(
        workload_id="vllm-decode", fingerprint="fp", run_id=run_id,
        git_sha="abc1234", gitm_version="0.1.13", started_at_ns=0, ended_at_ns=1,
    )
    d = tmp_path / run_id
    d.mkdir(parents=True, exist_ok=True)
    write_verification(records, prov, d / "verification.json", gpu_sku=gpu_sku)
    return d


def test_reads_an_export_written_by_the_real_writer(tmp_path):
    _real_export(tmp_path, "run-real", [
        ("kv_cache_dtype_fp8", "kv_cache_dtype", "fp8", 1.12, True, True),
        ("cuda_graphs_enable", "enforce_eager", False, 0.98, False, False),
    ])

    h = load_history(tmp_path)
    won = record_for(h, "kv_cache_dtype_fp8", gpu_sku="NVIDIA H100 80GB")
    lost = record_for(h, "cuda_graphs_enable", gpu_sku="NVIDIA H100 80GB")

    assert h.runs_read == 1
    assert (won.wins, won.losses) == (1, 0)
    assert abs(won.mean_delta - 0.12) < 1e-9
    assert (lost.wins, lost.losses) == (0, 1)
    assert abs(lost.mean_delta - (-0.02)) < 1e-9
    assert won.last_run_id == "run-real"


def test_gpu_sku_survives_the_real_writer(tmp_path):
    """write_verification puts the SKU under environment, which is where the
    reader keys its records from — if that moves, this fails."""
    _real_export(tmp_path, "run-mi", [
        ("enable_expert_parallel", "enable_expert_parallel", True, 1.49, True, True),
    ], gpu_sku="AMD MI355X")

    h = load_history(tmp_path, gpu_sku="AMD MI355X")

    assert h.runs_read == 1 and h.filtered == 0
    assert record_for(h, "enable_expert_parallel", gpu_sku="AMD MI355X").wins == 1


def test_a_real_export_with_no_results_is_not_mistaken_for_a_crash(tmp_path):
    _real_export(tmp_path, "run-empty", [])

    h = load_history(tmp_path)

    assert h.runs_read == 1
    assert h.skipped == {}
    assert len(h) == 0


def test_records_with_no_gpu_are_flagged_as_possibly_merged(tmp_path):
    """NVML cannot name an AMD part, so gpu_sku comes through as None and every
    such run keys together — results from two different boxes would merge
    silently. The table has to say so."""
    _run(tmp_path, "run-a", [_result("enable_expert_parallel")], gpu_sku=None)

    out = render_history(load_history(tmp_path))

    assert "WARNING" in out
    assert "GITM_GPU_SKU" in out


def test_no_warning_when_every_run_named_its_gpu(tmp_path):
    _run(tmp_path, "run-a", [_result("kv_cache_dtype_fp8")], gpu_sku="NVIDIA H100 80GB")

    assert "WARNING" not in render_history(load_history(tmp_path))


def test_two_ab_runs_in_one_folder_count_as_one_run_but_two_attempts(tmp_path):
    """Five A/Bs inside one run is far weaker evidence than five across five
    runs. A single counter cannot tell those apart, so the record carries both."""
    _run(tmp_path, "run-a", [_result("kv_cache_dtype_fp8", delta=0.04),
                             _result("kv_cache_dtype_fp8", delta=0.06)])

    r = record_for(load_history(tmp_path), "kv_cache_dtype_fp8",
                   gpu_sku="NVIDIA H100 80GB")

    assert r.runs == 1
    assert r.attempts == 2
    assert r.wins + r.losses + r.inconclusive == r.attempts


def test_the_same_lever_across_two_folders_counts_two_runs(tmp_path):
    _run(tmp_path, "run-a", [_result("kv_cache_dtype_fp8", delta=0.04)])
    _run(tmp_path, "run-b", [_result("kv_cache_dtype_fp8", delta=0.06)])

    r = record_for(load_history(tmp_path), "kv_cache_dtype_fp8",
                   gpu_sku="NVIDIA H100 80GB")

    assert (r.runs, r.attempts) == (2, 2)


def test_reading_the_same_directory_twice_does_not_accumulate(tmp_path):
    """load_history is read-only and builds a fresh tally each call; nothing is
    carried between calls."""
    _run(tmp_path, "run-a", [_result("kv_cache_dtype_fp8")])

    first = record_for(load_history(tmp_path), "kv_cache_dtype_fp8",
                       gpu_sku="NVIDIA H100 80GB")
    second = record_for(load_history(tmp_path), "kv_cache_dtype_fp8",
                        gpu_sku="NVIDIA H100 80GB")

    assert (first.runs, first.attempts) == (1, 1)
    assert (second.runs, second.attempts) == (1, 1)


def test_no_recorded_delta_is_none_not_zero(tmp_path):
    """Same distinction record_for makes: "we have no number" and "the number was
    zero" point opposite ways, and 0.0 for both reads as a measured no-op."""
    _run(tmp_path, "run-a", [{"intervention_name": "mystery_lever",
                              "kept": True, "significant": True}])

    r = record_for(load_history(tmp_path), "mystery_lever", gpu_sku="NVIDIA H100 80GB")

    assert r.attempts == 1          # the attempt still counts
    assert r.wins == 1              # and its verdict is still known
    assert r.mean_delta is None     # only the magnitude is missing
    assert r.best_delta is None and r.worst_delta is None


def test_a_measured_zero_is_not_the_same_as_no_measurement(tmp_path):
    _run(tmp_path, "run-a", [{"intervention_name": "flat_lever", "kept": True,
                              "significant": True, "delta": 0.0, "speedup": 1.0}])

    r = record_for(load_history(tmp_path), "flat_lever", gpu_sku="NVIDIA H100 80GB")

    assert r.mean_delta == 0.0      # not None — this one was measured


def test_the_table_never_cuts_a_sku_down_to_another_board(tmp_path):
    """A fixed-width gpu column rendered "AMD Instinct MI355X" as "MI355", and an
    H100 HBM3 and an HBM3e byte-identically. Two boxes that look like one box is
    the exact confusion keying on gpu_sku exists to prevent, so the column sizes
    to what it is showing and says so when it genuinely cannot fit."""
    _run(tmp_path, "run-a", [_result("kv_cache_dtype_fp8")],
         gpu_sku="NVIDIA H100 80GB HBM3")
    _run(tmp_path, "run-b", [_result("kv_cache_dtype_fp8")],
         gpu_sku="NVIDIA H100 80GB HBM3e")
    _run(tmp_path, "run-c", [_result("enable_expert_parallel")],
         gpu_sku="AMD Instinct MI355X")

    out = render_history(load_history(tmp_path))

    assert "AMD Instinct MI355X" in out
    assert "NVIDIA H100 80GB HBM3e" in out
    # the two H100s stay distinguishable on screen, not just in the data
    assert out.count("NVIDIA H100 80GB HBM3 ") >= 1


def test_a_value_too_long_for_its_column_is_marked_not_silently_cut(tmp_path):
    """A silent cut reads as the whole value. Past the cap the row has to admit
    it is showing a prefix."""
    long_name = "an_intervention_with_a_truly_unreasonable_name_" + "x" * 40
    _run(tmp_path, "run-a", [_result(long_name)], gpu_sku="AMD Instinct MI355X")

    out = render_history(load_history(tmp_path))

    assert long_name not in out
    assert "\u2026" in out


def test_one_damaged_export_does_not_take_the_readable_runs_with_it(tmp_path):
    """Valid JSON with the wrong shape inside — a string where an object belongs —
    reached an unguarded ``.get()`` and raised, so a single damaged export lost
    every sound run beside it. That is the opposite of what ``skipped`` is for."""
    _run(tmp_path, "good", [_result("kv_cache_dtype_fp8")], gpu_sku="AMD Instinct MI355X")
    _run(tmp_path, "bad-env", None, body=json.dumps({
        "provenance": {"run_id": "bad-env"}, "environment": "cuda",
        "results": [_result("enforce_eager")]}))
    _run(tmp_path, "bad-entry", None, body=json.dumps({
        "provenance": {"run_id": "bad-entry"},
        "environment": {"gpu_sku": "AMD Instinct MI355X"},
        "results": ["kv_cache_dtype_fp8"]}))

    h = load_history(tmp_path)

    assert h.runs_read == 1
    assert set(h.skipped) == {"bad-env", "bad-entry"}
    assert record_for(h, "kv_cache_dtype_fp8", gpu_sku="AMD Instinct MI355X").wins == 1


def test_a_malformed_provenance_still_reads(tmp_path):
    """Unlike the SKU, ``run_id`` already falls back to the directory name when
    provenance is absent. A bad one costs nothing, so it is not worth losing the
    run's measurements over."""
    _run(tmp_path, "run-x", None, body=json.dumps({
        "provenance": "not-an-object",
        "environment": {"gpu_sku": "AMD Instinct MI355X"},
        "results": [_result("kv_cache_dtype_fp8")]}))

    h = load_history(tmp_path)

    assert h.runs_read == 1 and not h.skipped
    rec = record_for(h, "kv_cache_dtype_fp8", gpu_sku="AMD Instinct MI355X")
    assert rec.last_run_id == "run-x"


def test_a_damaged_export_is_never_counted_as_filtered(tmp_path):
    """``filtered`` means a sound export measured on another box. Checking shape
    after the gpu filter would let a damaged file be counted as one, and a clean
    read of one box would look identical to a damaged history."""
    _run(tmp_path, "bad-env", None, body=json.dumps({
        "environment": "cuda", "results": [_result("kv_cache_dtype_fp8")]}))

    h = load_history(tmp_path, gpu_sku="AMD Instinct MI355X")

    assert h.filtered == 0
    assert h.skipped == {"bad-env": "malformed environment"}
