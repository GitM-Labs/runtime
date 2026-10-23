"""Synthetic captures with a known amount of time injected under each witness.

For testing :mod:`gitm.optimizer.witness` without hardware: kernels of about 1 ms
with a gap after each, and in a seeded share of the gaps a sync record, a memcpy
record, or an exposed collective kernel of seeded length. The generator returns
what it injected, so a test can check a price against the realised total.

    python -m gitm.optimizer.witness_fixtures demo OUT_DIR
    python -m gitm.optimizer.witness OUT_DIR

Nothing here resembles a real capture's timing; it exercises the arithmetic.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from gitm.tracer.capture import write_trace_jsonl
from gitm.tracer.schema import KernelEvent, MemcpyEvent, SyncEvent, Trace, TraceEvent

_MS = 1_000_000


@dataclass(frozen=True)
class Injection:
    """Per gap: with ``probability``, a record of ``mean_ns`` (uniform, half to 1.5x)."""

    probability: float = 0.0
    mean_ns: int = 0


@dataclass
class Fixture:
    events: list[TraceEvent]
    window_ns: int
    #: What this seed did inject, per row id, in ns.
    realised_ns: dict[str, int] = field(default_factory=dict)


def generate(
    seed: int,
    *,
    window_s: float = 25.0,
    kernel_ns: int = _MS,
    gap_ns: int = _MS,
    sync: Injection | None = None,
    memcpy: Injection | None = None,
    all_reduce: Injection | None = None,
    pid: int | None = 4242,
    device_id: int = 0,
) -> Fixture:
    """One partition of ``window_s`` seconds. Injected records never overlap each other.

    The sync record sits at the start of its gap, the memcpy after it, the
    collective kernel at the end, so ``gap_ns`` must hold 1.5x all three means.
    """
    sync, memcpy, all_reduce = (i or Injection() for i in (sync, memcpy, all_reduce))
    if 1.5 * (sync.mean_ns + memcpy.mean_ns + all_reduce.mean_ns) > gap_ns:
        raise ValueError("gap_ns is too small for the injected records")
    rng = random.Random(seed)
    events: list[TraceEvent] = []
    realised = {"10": 0, "11": 0, "22": 0}
    common = {"stream_id": 0, "device_id": device_id, "pid": pid}
    t, end = 0, int(window_s * 1e9)
    while t < end:
        k_end = t + rng.randrange(int(kernel_ns * 0.9), int(kernel_ns * 1.1) + 1)
        events.append(KernelEvent(start_ns=t, end_ns=k_end, name="gemm", **common))
        nxt = k_end + gap_ns
        if nxt >= end:  # the last kernel closes the window; no gap follows it
            break
        at = k_end
        for row, inj in (("10", sync), ("11", memcpy)):
            if inj.mean_ns and rng.random() < inj.probability:
                length = rng.randrange(inj.mean_ns // 2, inj.mean_ns * 3 // 2 + 1)
                if row == "10":
                    events.append(SyncEvent(start_ns=at, end_ns=at + length, sync_kind="stream",
                                            **common))
                else:
                    events.append(MemcpyEvent(start_ns=at, end_ns=at + length, bytes=1,
                                              src="host", dst="device", **common))
                realised[row] += length
                at += length
        if all_reduce.mean_ns and rng.random() < all_reduce.probability:
            length = rng.randrange(all_reduce.mean_ns // 2, all_reduce.mean_ns * 3 // 2 + 1)
            events.append(KernelEvent(start_ns=nxt - length, end_ns=nxt,
                                      name="ncclDevKernel_AllReduce_Sum_f32",
                                      **{**common, "stream_id": 1}))
            realised["22"] += length
        t = nxt
    window_ns = max(e.end_ns for e in events) - events[0].start_ns
    return Fixture(events=events, window_ns=window_ns, realised_ns=realised)


def write_run_dir(out_dir: Path, fixture: Fixture, dropped_records: int | None = None) -> Path:
    """Write the fixture as a capture directory the witness command can read.

    ``dropped_records`` is the count the header states; ``None`` states none.
    """
    if (out_dir / "trace.jsonl").exists() or (out_dir / "trace.jsonl").is_symlink():
        # Synthetic data must never replace a capture.
        raise FileExistsError(f"{out_dir} already holds trace.jsonl; choose an empty directory")
    trace = Trace(workload_id="witness-fixture", fingerprint="synthetic", run_id="demo",
                  device_count=1, vendor="nvidia", captured_at_ns=0,
                  duration_ns=fixture.window_ns, events=fixture.events,
                  dropped_records=dropped_records)
    write_trace_jsonl(out_dir / "trace.jsonl", trace)
    return out_dir


#: Kernels of one decode step in the varying-batch capture: name, and ns at batch 1.
_STEP = (("gemm_fp16", 600_000), ("attn_fused", 300_000), ("rms_norm", 60_000),
         ("silu_and_mul", 40_000))


def varying_batch(batch: int, seed: int = 0, *, window_s: float = 25.0) -> Fixture:
    """A clean capture at one batch size: no injected cause, only load.

    Kernel time grows with the batch, a padding kernel appears only at some batches
    (well under 0.5% of launches), and one launch in a hundred follows its
    predecessor by 10 us. A change of load must not look like a cause: on this
    capture no check may fire, at any batch.
    """
    rng = random.Random(seed * 1000 + batch)
    events: list[TraceEvent] = []
    common = {"stream_id": 0, "device_id": 0, "pid": 4242}
    t, launch, end = 0, 0, int(window_s * 1e9)
    while not events or events[-1].end_ns < end:  # the kernel span reaches window_s
        for name, base in _STEP:
            length = int(base * batch ** 0.5 * rng.uniform(0.9, 1.1))
            events.append(KernelEvent(start_ns=t, end_ns=t + length, name=name, grid_x=batch,
                                      **common))
            launch += 1
            # Mostly a host-side wait of about 1 ms; one launch-scale gap in a hundred.
            t += length + (10_000 if launch % 100 == 0 else rng.randrange(900_000, 1_100_000))
        if batch % 8 and launch % 4000 < len(_STEP):
            events.append(KernelEvent(start_ns=t, end_ns=t + 20_000, name="pad_to_batch",
                                      grid_x=batch, **common))
            t += 20_000 + _MS
    return Fixture(events=events, window_ns=max(e.end_ns for e in events))


def demo(out_dir: Path) -> Path:
    """A 120 s capture in which rows 10 and 11 fire; its header reports no dropped record."""
    fixture = generate(0, window_s=120.0, sync=Injection(0.5, 100_000),
                       memcpy=Injection(0.25, 100_000))
    return write_run_dir(out_dir, fixture, dropped_records=0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gitm.optimizer.witness_fixtures")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="Write a small synthetic capture directory.")
    d.add_argument("out_dir", type=Path)
    args = ap.parse_args(argv)
    try:
        out = demo(args.out_dir)
    except FileExistsError as exc:
        print(f"nothing written: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out}; now run: python -m gitm.optimizer.witness {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
