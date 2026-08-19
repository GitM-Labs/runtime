"""Roofline math against hand-computed A100 reference numbers.

The roofline prediction underpins every prediction in the runtime; any unit-
conversion or peak-rate bug here corrupts every downstream residual. These
tests pin the math against numbers that were derived by hand from the
A100-SXM4-80GB defaults in ``HardwareSpec`` so a silent regression on those
defaults — or on the formula itself — is loud.

Default A100 peak rates being asserted against:
    peak_flops_fp16  = 312e12        (312 TFLOPS, fp16/bf16)
    peak_flops_fp32  = 19.5e12       (19.5 TFLOPS)
    peak_mem_bw      = 2_039e9       (2,039 GB/s)
"""

from __future__ import annotations

import json

import pytest

from gitm.planner.registry import main
from gitm.planner.roofline import HardwareSpec, roofline

# ── memory-bound reference: 1 MiB over A100 HBM ──────────────────────────────


def test_roofline_a100_memory_bound_reference():
    """1 MiB transferred, ~no FLOPs, against A100 defaults.

    t_memory = 1_048_576 / 2_039e9 ≈ 5.14e-7 s.
    Bound must be 'memory'; t_pred == t_memory.
    """
    hw = HardwareSpec()  # A100-SXM4-80GB defaults
    bytes_moved = 1 << 20  # 1 MiB == 1,048,576 bytes
    pred = roofline("memcpy_ref", flops=0, bytes_moved=bytes_moved, hw=hw)

    expected_t_memory = bytes_moved / 2_039e9
    assert pred.t_memory_s == pytest.approx(expected_t_memory, rel=1e-9)
    assert pred.t_compute_s == pytest.approx(0.0)
    assert pred.bound == "memory"
    assert pred.t_pred_s == pytest.approx(expected_t_memory, rel=1e-9)


# ── compute-bound reference: 1 TFLOP @ fp16 ─────────────────────────────────


def test_roofline_a100_compute_bound_reference():
    """1 TFLOP at fp16, ~no bytes moved, against A100 defaults.

    t_compute = 1e12 / 312e12 ≈ 3.205e-3 s.
    Bound must be 'compute'; t_pred == t_compute.
    """
    hw = HardwareSpec()
    flops = 1e12  # 1 TFLOP
    pred = roofline("compute_ref", flops=flops, bytes_moved=0, hw=hw)

    expected_t_compute = flops / 312e12
    assert pred.t_compute_s == pytest.approx(expected_t_compute, rel=1e-9)
    assert pred.t_memory_s == pytest.approx(0.0)
    assert pred.bound == "compute"
    assert pred.t_pred_s == pytest.approx(expected_t_compute, rel=1e-9)


# ── dtype selection: fp16/bf16 share peak; fp32 takes the slower path ───────


def test_roofline_dtype_selects_fp16_peak_for_bf16():
    hw = HardwareSpec()
    fp16 = roofline("op", flops=1e12, bytes_moved=0, hw=hw, dtype="fp16")
    bf16 = roofline("op", flops=1e12, bytes_moved=0, hw=hw, dtype="bf16")
    assert fp16.t_compute_s == pytest.approx(bf16.t_compute_s, rel=1e-12)
    # Both must pick the fp16 peak (312e12), not the fp32 peak (19.5e12).
    assert fp16.t_compute_s == pytest.approx(1e12 / 312e12, rel=1e-9)


def test_roofline_dtype_fp32_uses_slower_peak():
    hw = HardwareSpec()
    fp32 = roofline("op", flops=1e12, bytes_moved=0, hw=hw, dtype="fp32")
    expected_t_compute = 1e12 / 19.5e12
    assert fp32.t_compute_s == pytest.approx(expected_t_compute, rel=1e-9)
    # And materially slower than the fp16 case (sanity).
    fp16 = roofline("op", flops=1e12, bytes_moved=0, hw=hw, dtype="fp16")
    assert fp32.t_compute_s > fp16.t_compute_s * 10


# ── bound label boundary: equal t_compute and t_memory → "compute" ───────────


def test_roofline_bound_label_at_equality_picks_compute():
    """Boundary case: when t_compute == t_memory, the implementation picks
    'compute' (the ``>=`` branch). Lock the tie-break behavior."""
    hw = HardwareSpec()
    # Choose flops, bytes so t_compute == t_memory exactly.
    bytes_moved = 1_000_000
    flops = bytes_moved * (312e12 / 2_039e9)  # makes t_compute == t_memory
    pred = roofline("tie", flops=flops, bytes_moved=bytes_moved, hw=hw)
    assert pred.t_compute_s == pytest.approx(pred.t_memory_s, rel=1e-9)
    assert pred.bound == "compute"


# ── degenerate inputs: zero peak rates do not raise ──────────────────────────


def test_roofline_zero_peak_rates_dont_divide_by_zero():
    """If a hardware spec defines a zero peak (e.g. a tier without that path),
    the roofline returns zeros for that dimension rather than raising."""
    hw = HardwareSpec(
        peak_flops_fp16_per_s=0.0,
        peak_flops_bf16_per_s=0.0,
        peak_flops_fp32_per_s=0.0,
        peak_mem_bw_bytes_per_s=0.0,
    )
    pred = roofline("op", flops=1e12, bytes_moved=1 << 20, hw=hw)
    assert pred.t_compute_s == 0.0
    assert pred.t_memory_s == 0.0
    assert pred.t_pred_s == 0.0

# ── ``gitm plan`` — rendering that roofline ───────────────────────────────────
#
# ``gitm plan`` — a predicted floor from a config file and a SKU name.
#
# Every other route to a predicted graph needs something running. This one is
# arithmetic over a checkpoint's declared shape, so the tests need no GPU and no
# server; what they check is that the command refuses in every case where it would
# otherwise emit a confident number about the wrong thing.



def test_list_names_each_entry_and_its_family(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "qwen3.6-35b-a3b" in out
    assert "[hybrid]" in out


def test_a_table_reports_the_floor_and_what_bounds_it(capsys):
    assert main(["qwen3.6-35b-a3b", "--gpu", "H200", "--batch", "8",
                 "--kv-len", "1024"]) == 0
    out = capsys.readouterr().out
    assert "moe_routed" in out
    assert "linattn_recurrent" in out
    assert "ridge" in out
    assert "floor" in out


def test_the_table_says_the_floor_is_not_a_target(capsys):
    """A predicted number sitting next to a measured one is read as a target
    unless it says otherwise, and then a 1.4x residual reads as a defect."""
    main(["qwen3.6-35b-a3b", "--gpu", "H200"])
    out = capsys.readouterr().out
    assert "not a target" in out
    assert "a lead, not a defect" in out


def test_fitted_fields_are_named_in_the_source_line(capsys):
    """The catalogue distinguishes transcribed values from fitted ones. If the
    command did not surface that, the distinction would exist only in a file
    nobody opens."""
    main(["qwen3.6-35b-a3b", "--gpu", "H200"])
    assert "conv_dim" in capsys.readouterr().out


def test_an_unknown_sku_warns_rather_than_silently_repricing(capsys):
    """The failure this guards: an unrecognised SKU resolves to the A100
    defaults, and 2.04 TB/s against an H200's 4.80 is a 2.4x error on every
    memory-bound node — which at decode is all of them."""
    main(["qwen3.6-35b-a3b", "--gpu", "RTX9090", "--batch", "1"])
    out = capsys.readouterr().out
    assert "not in the catalogue" in out
    assert "A100" in out


def test_a_known_sku_does_not_warn(capsys):
    main(["qwen3.6-35b-a3b", "--gpu", "H200"])
    assert "not in the catalogue" not in capsys.readouterr().out


def test_sweep_reports_where_the_model_stops_being_memory_bound(capsys):
    """The knee is the capacity-planning answer: below it, throughput is set by
    how efficiently bytes move, and no amount of extra FLOPs helps."""
    assert main(["qwen3.6-35b-a3b", "--gpu", "H200", "--sweep", "1,512",
                 "--kv-len", "1024"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    at_1 = next(ln for ln in lines if ln.split()[0] == "1")
    at_512 = next(ln for ln in lines if ln.split()[0] == "512")
    assert at_1.endswith("0/281")
    assert not at_512.endswith("0/281")


def test_json_output_carries_every_node(capsys):
    assert main(["qwen3.6-35b-a3b", "--gpu", "H200", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["family"] == "hybrid"
    assert len(payload["nodes"]) == 281
    assert {"t_compute_s", "t_memory_s", "bound"} <= set(payload["nodes"][0])


def test_a_config_json_path_is_accepted_and_marked_as_having_no_provenance(
    tmp_path, capsys
):
    """A checkpoint's config is authoritative for shape and silent about what
    the graph gets wrong. Saying so is the difference between a prediction and
    a prediction you can weigh."""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"text_config": {
        "hidden_size": 2048, "num_hidden_layers": 8, "vocab_size": 1000,
        "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
        "num_experts": 64, "num_experts_per_tok": 4, "moe_intermediate_size": 512,
        "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "full_attention_interval": 4,
    }}))
    assert main([str(cfg), "--gpu", "H200"]) == 0
    assert "no provenance" in capsys.readouterr().out


def test_an_unknown_model_lists_what_is_available(capsys):
    assert main(["no-such-model"]) == 2
    assert "Available entries" in capsys.readouterr().out


def test_a_dense_config_is_declined_rather_than_planned(tmp_path, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"hidden_size": 4096, "num_hidden_layers": 32}))
    assert main([str(cfg)]) == 2
    assert "dense family" in capsys.readouterr().out


def test_an_indivisible_sharding_is_reported_not_raised(capsys):
    """The graph refuses a split that would floor a whole path to zero work.
    The command must surface that as a message and an exit code, not a
    traceback."""
    assert main(["qwen3.6-35b-a3b", "--gpu", "H200", "--tp", "3"]) == 2
    assert "does not divide" in capsys.readouterr().out


def test_a_malformed_sweep_is_reported(capsys):
    assert main(["qwen3.6-35b-a3b", "--sweep", "1,two,3"]) == 2
    assert "comma-separated" in capsys.readouterr().out


def test_no_model_is_a_usage_error(capsys):
    with pytest.raises(SystemExit):
        main([])


# ── reachable through the top-level CLI ────────────────────────────────────


def test_the_subcommand_is_wired_into_gitm(capsys):
    from gitm.cli import main as cli_main

    assert cli_main(["plan", "--list"]) == 0
    assert "qwen3.6-35b-a3b" in capsys.readouterr().out


def test_flags_survive_the_cli_hand_off(capsys):
    """The dispatcher rebuilds an argv rather than passing the namespace, so a
    flag can be silently dropped there while the subcommand still works when
    invoked directly."""
    from gitm.cli import main as cli_main

    assert cli_main(["plan", "qwen3.6-35b-a3b", "--gpu", "H200", "--batch", "4",
                     "--kv-len", "2048", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hardware"] == "H200"
    assert payload["batch"] == {"batch": 4, "kv_cache_len": 2048}
