"""The sweep in ``_classify_stalls`` against a frozen copy of the function it replaced."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from gitm.optimizer.metrics import _LAUNCH_LATENCY_NS, _classify_stalls, _merge_intervals


# Frozen with the function below: the per-gap clip the sweep replaced. Do not edit.
def _clip_union_ns(spans: list[tuple[int, int]], lo: int, hi: int) -> int:
    """Total length of ``spans`` clipped to [lo, hi), overlaps counted once."""
    clipped = [(max(s, lo), min(e, hi)) for s, e in spans if min(e, hi) > max(s, lo)]
    return sum(e - s for s, e in _merge_intervals(clipped))


def _frozen_classify_stalls(gaps, memcpys, syncs, wall_ns):
    # Copy of _classify_stalls as of f710117 (per gap, re-clip every record). Do not edit.
    ns = {"sync_wait": 0, "transfer_bound": 0, "launch_latency": 0, "idle": 0}
    if wall_ns <= 0:
        return {k: 0.0 for k in ns}
    if not memcpys and not syncs:
        for g0, g1 in gaps:
            residual = g1 - g0
            ns["launch_latency" if residual <= _LAUNCH_LATENCY_NS else "idle"] += residual
        return {k: v / wall_ns for k, v in ns.items()}
    cp = [(e.start_ns, e.end_ns) for e in memcpys]
    sy = [(e.start_ns, e.end_ns) for e in syncs]
    for g0, g1 in gaps:
        transfer = _clip_union_ns(cp, g0, g1)
        transfer_or_sync = _clip_union_ns(cp + sy, g0, g1)
        ns["transfer_bound"] += transfer
        ns["sync_wait"] += transfer_or_sync - transfer
        residual = (g1 - g0) - transfer_or_sync
        ns["launch_latency" if (g1 - g0) <= _LAUNCH_LATENCY_NS else "idle"] += residual
    return {k: v / wall_ns for k, v in ns.items()}


def _records(rng: random.Random, n: int, wall_ns: int) -> list[SimpleNamespace]:
    out = []
    for _ in range(n):
        start = rng.randrange(-5_000, wall_ns)
        # Mostly short, some long enough to span several gaps, some empty.
        length = rng.choice([0, rng.randrange(1, 4_000), rng.randrange(1, 400_000)])
        out.append(SimpleNamespace(start_ns=start, end_ns=start + length))
    return out


def _gaps(rng: random.Random, n: int, wall_ns: int) -> list[tuple[int, int]]:
    # Sorted and disjoint, as _idle_gaps produces them, on both sides of the 20 us line.
    cuts = sorted(rng.sample(range(wall_ns), 2 * n))
    return [(cuts[i], cuts[i + 1]) for i in range(0, 2 * n, 2)]


@pytest.mark.parametrize("seed", range(200))
def test_sweep_matches_frozen_copy(seed):
    rng = random.Random(seed)
    wall_ns = rng.choice([200_000, 2_000_000])
    gaps = _gaps(rng, rng.randrange(0, 40), wall_ns)
    memcpys = _records(rng, rng.randrange(0, 30), wall_ns)
    syncs = _records(rng, rng.randrange(0, 30), wall_ns)
    assert _classify_stalls(gaps, memcpys, syncs, wall_ns) == _frozen_classify_stalls(
        gaps, memcpys, syncs, wall_ns
    )


@pytest.mark.parametrize("seed", range(50))
def test_sweep_matches_frozen_copy_on_unsorted_overlapping_gaps(seed):
    # Not what _idle_gaps produces, but the signature allows it and the total is a sum.
    rng = random.Random(1000 + seed)
    wall_ns = 500_000
    gaps = []
    for _ in range(rng.randrange(1, 25)):
        g0 = rng.randrange(wall_ns)
        gaps.append((g0, g0 + rng.randrange(-50, 60_000)))
    memcpys = _records(rng, rng.randrange(0, 20), wall_ns)
    syncs = _records(rng, rng.randrange(1, 20), wall_ns)
    got = _classify_stalls(gaps, memcpys, syncs, wall_ns)
    assert got == _frozen_classify_stalls(gaps, memcpys, syncs, wall_ns)


def test_record_spanning_two_gaps_is_split_between_them():
    gaps = [(0, 10_000), (30_000, 100_000)]
    sync = [SimpleNamespace(start_ns=5_000, end_ns=40_000)]
    copy = [SimpleNamespace(start_ns=8_000, end_ns=32_000)]
    got = _classify_stalls(gaps, copy, sync, 100_000)
    assert got == _frozen_classify_stalls(gaps, copy, sync, 100_000)
    assert got["transfer_bound"] == pytest.approx(0.04)  # 2 us in the first gap, 2 us in the second
    assert got["sync_wait"] == pytest.approx(0.11)  # 3 us, then 8 us past the copy


def test_a_twenty_microsecond_gap_is_launch_scale_and_one_more_is_idle():
    gaps = [(0, 20_000), (100_000, 120_001)]
    got = _classify_stalls(gaps, [], [], 1_000_000)
    assert got["launch_latency"] == pytest.approx(0.02) and got["idle"] == pytest.approx(0.020001)
    sync = [SimpleNamespace(start_ns=0, end_ns=5_000)]
    got = _classify_stalls(gaps, [], sync, 1_000_000)
    assert got["launch_latency"] == pytest.approx(0.015) and got["sync_wait"] == pytest.approx(0.005)
    assert _classify_stalls(gaps, [], sync, 0) == dict.fromkeys(
        ("sync_wait", "transfer_bound", "launch_latency", "idle"), 0.0)


def test_timestamps_past_int64_still_classify():
    huge = 2**64 - 1
    got = _classify_stalls([(0, huge)], [], [SimpleNamespace(start_ns=0, end_ns=huge // 2)], huge)
    assert got["sync_wait"] == pytest.approx(0.5) and got["idle"] == pytest.approx(0.5)
