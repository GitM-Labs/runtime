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

import pytest

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
