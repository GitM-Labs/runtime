"""vLLM scheduler-stats adapter — engine-level telemetry alongside the CUPTI trace.

CUPTI tells us what the *GPU* did (per-kernel timings, streams). It cannot tell
us what the *engine scheduler* did: how deep the waiting queue got, how full each
decode batch was, whether sequences were preempted and recomputed. Those are the
signals that explain *why* the kernels look the way they do — a half-empty decode
batch is launch-bound for a scheduling reason, not a kernel reason — so this
series feeds causal attribution (stream C).

Usage mirrors :func:`gitm.tracer.capture.capture`: wrap the same window.

    from gitm.tracer.capture import capture
    from gitm.tracer.vllm_stats import sample_scheduler_stats

    with capture(trace_path) as trace, sample_scheduler_stats(engine) as stats:
        run_decode()
    stats.summary()  # peak queue depth, mean batch occupancy, preemptions, ...

Everything is **duck-typed and best-effort**: vLLM moves scheduler internals
between releases, so each field is probed across several known attribute paths
and silently left ``None`` when unavailable. A read never raises into the
workload, and on a box without vLLM the sampler degrades to an empty series — the
honest "no engine stats" outcome, never fabricated numbers.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SchedulerSample:
    """One scheduler snapshot. Any field is ``None`` if the engine didn't expose it."""

    t_ns: int  # ns since sampling started
    num_running: int | None = None  # sequences decoding this step
    num_waiting: int | None = None  # queue depth
    num_swapped: int | None = None  # sequences swapped to CPU
    num_unfinished: int | None = None  # total in-flight requests
    preemptions_cumulative: int | None = None  # running total of preemptions
    gpu_cache_usage: float | None = None  # KV-cache blocks used / total, 0..1
    cpu_cache_usage: float | None = None
    batch_occupancy: float | None = None  # num_running / max_num_seqs, 0..1


@dataclass
class SchedulerStatsSummary:
    """Aggregates over the sampled window — the compact form attribution consumes."""

    n_samples: int
    duration_s: float
    peak_queue_depth: int | None
    mean_running: float | None
    peak_running: int | None
    mean_batch_occupancy: float | None
    total_preemptions: int | None  # delta of cumulative counter over the window
    peak_gpu_cache_usage: float | None
    peak_swapped: int | None


def _first_attr(obj: Any, *paths: str) -> Any:
    """Return the first dotted-path attribute on ``obj`` that resolves non-None.

    Duck-typing across vLLM versions: try ``engine.scheduler``,
    ``engine.engine.scheduler``, ``engine.llm_engine.scheduler`` in turn.
    """
    for path in paths:
        cur: Any = obj
        for attr in path.split("."):
            cur = getattr(cur, attr, None)
            if cur is None:
                break
        if cur is not None:
            return cur
    return None


def _len_or_none(x: Any) -> int | None:
    try:
        return len(x)
    except TypeError:
        return None


def _schedulers(engine: Any) -> list[Any]:
    """Resolve the engine's scheduler(s) as a list (vLLM may keep one per PP stage)."""
    sched = _first_attr(
        engine, "scheduler", "engine.scheduler", "llm_engine.scheduler"
    )
    if sched is None:
        return []
    return list(sched) if isinstance(sched, (list, tuple)) else [sched]


def _max_num_seqs(engine: Any) -> int | None:
    val = _first_attr(
        engine,
        "scheduler_config.max_num_seqs",
        "engine.scheduler_config.max_num_seqs",
        "llm_engine.scheduler_config.max_num_seqs",
        "vllm_config.scheduler_config.max_num_seqs",
    )
    return int(val) if isinstance(val, int) and val > 0 else None


def read_scheduler_stats(engine: Any, *, t_ns: int = 0) -> SchedulerSample | None:
    """One duck-typed snapshot of the engine scheduler, or ``None`` if unreadable.

    Reads queue/running/swapped depths off the scheduler's request deques, the
    cumulative preemption counter, and KV-cache usage off the block manager.
    Each is independent — a version that exposes some but not others yields a
    partial sample rather than nothing.
    """
    if engine is None:
        return None

    schedulers = _schedulers(engine)
    sample = SchedulerSample(t_ns=t_ns)
    saw_any = False

    if schedulers:
        # Sum each depth across schedulers, but only report a field if at least
        # one scheduler exposed it (a missing attr must not read as 0).
        totals: dict[str, int] = {}
        for sch in schedulers:
            for field_name, attr, is_len in (
                ("num_running", "running", True),
                ("num_waiting", "waiting", True),
                ("num_swapped", "swapped", True),
                ("preemptions_cumulative", "num_cumulative_preemption", False),
            ):
                raw = getattr(sch, attr, None)
                val = _len_or_none(raw) if is_len else (raw if isinstance(raw, int) else None)
                if val is not None:
                    totals[field_name] = totals.get(field_name, 0) + val
        for field_name, val in totals.items():
            setattr(sample, field_name, val)
            saw_any = True

        # KV-cache usage off the first scheduler's block manager (best-effort).
        usage = _gpu_cache_usage(schedulers[0])
        if usage is not None:
            sample.gpu_cache_usage = usage
            saw_any = True

    # Total unfinished — a stable public method on LLMEngine across versions.
    getter = _first_attr(
        engine,
        "get_num_unfinished_requests",
        "engine.get_num_unfinished_requests",
        "llm_engine.get_num_unfinished_requests",
    )
    if callable(getter):
        try:
            sample.num_unfinished = int(getter())
            saw_any = True
        except Exception:
            pass

    if sample.num_running is not None:
        max_seqs = _max_num_seqs(engine)
        if max_seqs:
            # num_running is summed across schedulers, so the capacity is
            # max_num_seqs per scheduler — divide by both, and clamp to [0,1] so a
            # transient over-count (or a partially-exposed config) can never make a
            # half-empty engine read as full and silently suppress under_filled.
            capacity = max_seqs * max(len(schedulers), 1)
            sample.batch_occupancy = min(1.0, sample.num_running / capacity)

    return sample if saw_any else None


def _gpu_cache_usage(scheduler: Any) -> float | None:
    """KV-cache block occupancy (0..1) off a scheduler's block manager, if exposed."""
    bm = getattr(scheduler, "block_manager", None)
    if bm is None:
        return None
    # Newer vLLM: get_num_free_gpu_blocks(); total via num_total_gpu_blocks.
    free_fn = getattr(bm, "get_num_free_gpu_blocks", None)
    total = getattr(bm, "num_total_gpu_blocks", None)
    if callable(free_fn) and isinstance(total, int) and total > 0:
        try:
            return max(0.0, 1.0 - free_fn() / total)
        except Exception:
            return None
    return None


class SchedulerStatsSampler:
    """Background sampler: snapshots the engine scheduler on a fixed interval.

    Runs a daemon thread so it never outlives the process. ``start``/``stop`` are
    idempotent; ``stop`` joins the thread so all samples are flushed before
    :meth:`summary` is called. Reads are best-effort — a read that raises is
    dropped, not propagated into the workload.
    """

    def __init__(self, engine: Any, *, interval_s: float = 0.05) -> None:
        self.engine = engine
        self.interval_s = max(interval_s, 1e-3)
        self.samples: list[SchedulerSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0_ns: int = 0

    def start(self) -> None:
        if self._thread is not None or self.engine is None:
            return
        self._t0_ns = time.perf_counter_ns()
        # Take one snapshot synchronously so even a decode shorter than one
        # sampling interval still yields a scheduler reading (no start/stop race).
        try:
            s0 = read_scheduler_stats(self.engine, t_ns=0)
            if s0 is not None:
                self.samples.append(s0)
        except Exception:
            pass
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gitm-vllm-stats", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                s = read_scheduler_stats(self.engine, t_ns=time.perf_counter_ns() - self._t0_ns)
                if s is not None:
                    self.samples.append(s)
            except Exception:
                pass  # best-effort; never let sampling crash the run
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def summary(self) -> SchedulerStatsSummary:
        # Snapshot first: stop() joins with a timeout, so in the pathological case
        # where the daemon thread is still alive, summarize must iterate a stable
        # copy rather than a list being appended to concurrently.
        return summarize(list(self.samples))

    def to_records(self) -> list[dict[str, Any]]:
        """Samples as plain dicts, ready for JSONL alongside the kernel trace."""
        return [asdict(s) for s in list(self.samples)]


def summarize(samples: list[SchedulerSample]) -> SchedulerStatsSummary:
    """Aggregate a sample series into the compact summary attribution consumes."""
    if not samples:
        return SchedulerStatsSummary(
            n_samples=0, duration_s=0.0, peak_queue_depth=None, mean_running=None,
            peak_running=None, mean_batch_occupancy=None, total_preemptions=None,
            peak_gpu_cache_usage=None, peak_swapped=None,
        )

    def _vals(attr: str) -> list[float]:
        return [getattr(s, attr) for s in samples if getattr(s, attr) is not None]

    waiting = _vals("num_waiting")
    running = _vals("num_running")
    occ = _vals("batch_occupancy")
    preempt = _vals("preemptions_cumulative")
    cache = _vals("gpu_cache_usage")
    swapped = _vals("num_swapped")
    duration_s = max(samples[-1].t_ns - samples[0].t_ns, 0) / 1e9

    return SchedulerStatsSummary(
        n_samples=len(samples),
        duration_s=duration_s,
        peak_queue_depth=int(max(waiting)) if waiting else None,
        mean_running=(sum(running) / len(running)) if running else None,
        peak_running=int(max(running)) if running else None,
        mean_batch_occupancy=(sum(occ) / len(occ)) if occ else None,
        # Preemptions over the window: max − min of the cumulative counter, which
        # is monotonic within a single engine/sampling window (the sampler runs
        # over one engine — the restart A/B happens later, after sampling stops).
        total_preemptions=int(max(preempt) - min(preempt)) if preempt else None,
        peak_gpu_cache_usage=max(cache) if cache else None,
        peak_swapped=int(max(swapped)) if swapped else None,
    )


@contextmanager
def sample_scheduler_stats(
    engine: Any, *, interval_s: float = 0.05
) -> Iterator[SchedulerStatsSampler]:
    """Sample the engine scheduler for the duration of the ``with`` block.

    A no-op (empty series) when ``engine`` is ``None`` — the loop can wrap every
    capture window in this unconditionally and only get stats when an engine is
    actually attached.
    """
    sampler = SchedulerStatsSampler(engine, interval_s=interval_s)
    sampler.start()
    try:
        yield sampler
    finally:
        sampler.stop()
