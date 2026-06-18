"""Tests for ``gitm.optimizer.qualification.fingerprint``.

The qualification gate uses the fingerprint to decide whether a workload is
"the same" as one we've seen before — so any silent change in stability,
format, or sort-invariance would silently corrupt the gate's commit decision
and the cross-run learning loop it enables.

The contract under test:
1. Stable — same input → same digest, always.
2. Formatted — vendor-prefixed, sha256-truncated to 16 hex chars.
3. Sort-invariant — same set of kernels in any order → same digest.
4. Sensitive — different kernel name → different digest (no trivial collision).
"""

from __future__ import annotations

import re

from gitm.optimizer.qualification import fingerprint

from .conftest import make_kernel, make_trace

# ── stability ────────────────────────────────────────────────────────────────


def test_fingerprint_stable_across_repeated_calls():
    trace = make_trace(
        events=[
            make_kernel("paged_attention", grid=(128, 1, 1), block=(64, 1, 1)),
            make_kernel("attn_score_value", grid=(32, 1, 1), block=(128, 1, 1)),
        ],
        vendor="nvidia",
    )
    assert fingerprint(trace) == fingerprint(trace)


# ── format ───────────────────────────────────────────────────────────────────


def test_fingerprint_format_is_vendor_colon_hex16():
    trace = make_trace(
        events=[make_kernel("paged_attention")],
        vendor="nvidia",
    )
    fp = fingerprint(trace)
    assert re.match(r"^nvidia:[a-f0-9]{16}$", fp), f"unexpected format: {fp!r}"


def test_fingerprint_vendor_prefix_propagates():
    """Same kernels under a different vendor → different prefix, otherwise
    same digest body."""
    kernels = [make_kernel("k1"), make_kernel("k2")]
    nv = fingerprint(make_trace(events=kernels, vendor="nvidia"))
    amd = fingerprint(make_trace(events=kernels, vendor="amd"))
    assert nv.startswith("nvidia:")
    assert amd.startswith("amd:")
    assert nv.split(":", 1)[1] == amd.split(":", 1)[1]  # digest body identical


# ── sort invariance ──────────────────────────────────────────────────────────


def test_fingerprint_invariant_under_kernel_reorder():
    a = make_trace(
        events=[
            make_kernel("paged_attention", start_ns=0, end_ns=10),
            make_kernel("attn_score_value", start_ns=20, end_ns=30),
        ],
    )
    b = make_trace(
        events=[
            make_kernel("attn_score_value", start_ns=0, end_ns=10),
            make_kernel("paged_attention", start_ns=20, end_ns=30),
        ],
    )
    assert fingerprint(a) == fingerprint(b)


def test_fingerprint_invariant_under_timestamp_jitter():
    """Timestamps are not part of the fingerprint — two captures of the same
    workload at different wall-clock moments must hash identically."""
    a = make_trace(
        events=[make_kernel("paged_attention", start_ns=0, end_ns=10)],
        captured_at_ns=1_700_000_000_000_000_000,
        duration_ns=1_000_000,
    )
    b = make_trace(
        events=[make_kernel("paged_attention", start_ns=999_999, end_ns=999_999 + 10)],
        captured_at_ns=1_700_000_999_000_000_000,
        duration_ns=2_000_000,
    )
    assert fingerprint(a) == fingerprint(b)


# ── sensitivity ──────────────────────────────────────────────────────────────


def test_fingerprint_different_kernel_name_yields_different_digest():
    a = make_trace(events=[make_kernel("paged_attention")])
    b = make_trace(events=[make_kernel("flash_attention")])
    assert fingerprint(a) != fingerprint(b)


def test_fingerprint_different_grid_shape_yields_different_digest():
    """Grid shape is folded into the fingerprint via grid_x * grid_y * grid_z;
    different grid totals must produce different digests."""
    a = make_trace(events=[make_kernel("k", grid=(128, 1, 1))])
    b = make_trace(events=[make_kernel("k", grid=(64, 1, 1))])
    assert fingerprint(a) != fingerprint(b)


def test_fingerprint_empty_trace_still_well_formed():
    """No kernels → still vendor:hex16 format (just a digest of the empty
    summary). Qualification handles the no-kernels case separately."""
    fp = fingerprint(make_trace(events=[], vendor="nvidia"))
    assert re.match(r"^nvidia:[a-f0-9]{16}$", fp)
