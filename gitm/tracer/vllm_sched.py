"""vLLM scheduler-stats adapter-the engine level signal alongside CUPTI.

The CUPTI /Kineto trace tells us what the GPU did, it can't tell us why the
batch was the size it was. vLLM scheduler told that:how many sequences are
running vs waiting, KV-cache occupancy, preemptions. Sampling it alongside
the decode traces lets attribution seperate "GPU-bound" from "scheduler-starved".
(e.g. A half-full batch because requests are queued behind KV-cache pressures).

This adapter duck-types the engine so it works across vLLM versions and in tests:
it reads a ``Stats``-like object if the engine exposes one, otherwise it
introspect ``engine scheduler``(running/waiting/swapped+KV usage). Sampling
is cheap(reads counter), so the loop can poll it at decode-step cadence.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SchedulerSample(BaseModel):
    """One point in time read of the vLLM scheduler state."""

    model_config = ConfigDict(extra="forbid")

    ts_ns: int
    num_running: int = 0
    num_waiting: int = 0
    num_swapped: int = 0
    num_preemptions: int = 0
    gpu_cache_usage: float = 0.0  # fraction [0, 1] of KV-cache blocks in use
    batch_size: int = 0  # sequences in the current decode batch

    @property
    def queue_depth(self) -> int:
        """Sequences not currently being decoded(waiting + swapped)."""
        return self.num_waiting + self.num_swapped

    @property
    def batch_occupancy(self) -> float:
        """Running fraction of all live sequences-low = scheduler-starved"""
        live = self.num_running + self.queue_depth
        return self.num_running / live if live else 0.0


def _first_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            return v() if callable(v) else v
    return default


def _len_or_int(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, int):
        return v
    try:
        return len(v)
    except TypeError:
        return 0


def sample_scheduler(engine: Any, *, ts_ns: int) -> SchedulerSample:
    """Read a :class:SchedulerSample from a vLLM engine (or a stand-in).

    Accepts, in order of preference:
      - an object with ``get_scheduler_stats()`` / ``get_stats()`` returning a
        mapping or stats object,
      - an engine exposing ``.scheduler`` with running/waiting/swapped,
      - a plain mapping of the field names.
    """
    if isinstance(engine, dict):
        src: Any = engine
    else:
        stats = _first_attr(engine, "get_scheduler_stats", "get_stats")
        src = stats if stats is not None else engine

    # Mapping form (already-collected stats or a test dict).
    if isinstance(src, dict):
        running = _len_or_int(src.get("num_running", src.get("running")))
        waiting = _len_or_int(src.get("num_waiting", src.get("waiting")))
        swapped = _len_or_int(src.get("num_swapped", src.get("swapped")))
        preempt = int(src.get("num_preemptions", 0) or 0)
        usage = float(src.get("gpu_cache_usage", src.get("gpu_cache_usage_sys", 0.0)) or 0.0)
        batch = _len_or_int(src.get("batch_size", running))
        return SchedulerSample(
            ts_ns=ts_ns,
            num_running=running,
            num_waiting=waiting,
            num_swapped=swapped,
            num_preemptions=preempt,
            gpu_cache_usage=usage,
            batch_size=batch or running,
        )

    # Object form: introspect a scheduler.
    scheduler = _first_attr(src, "scheduler", default=src)
    running = _len_or_int(_first_attr(scheduler, "running", "num_running"))
    waiting = _len_or_int(_first_attr(scheduler, "waiting", "num_waiting"))
    swapped = _len_or_int(_first_attr(scheduler, "swapped", "num_swapped"))
    preempt = int(_first_attr(scheduler, "num_preemptions", default=0) or 0)
    usage = float(_first_attr(scheduler, "gpu_cache_usage", default=0.0) or 0.0)
    return SchedulerSample(
        ts_ns=ts_ns,
        num_running=running,
        num_waiting=waiting,
        num_swapped=swapped,
        num_preemptions=preempt,
        gpu_cache_usage=usage,
        batch_size=running,
    )


class SchedulerStatsTracker(BaseModel):
    """Accumulates :class:SchedulerSample points over a decode run."""

    model_config = ConfigDict(extra="forbid")

    samples: list[SchedulerSample] = Field(default_factory=list)

    def record(self, engine: Any, *, ts_ns: int) -> SchedulerSample:
        s = sample_scheduler(engine, ts_ns=ts_ns)
        self.samples.append(s)
        return s

    def summarize(self) -> dict[str, Any]:
        """Aggregate the series into report-ready scheduler signals."""
        if not self.samples:
            return {"n_samples": 0}
        n = len(self.samples)
        return {
            "n_samples": n,
            "mean_batch_occupancy": sum(s.batch_occupancy for s in self.samples) / n,
            "mean_queue_depth": sum(s.queue_depth for s in self.samples) / n,
            "max_queue_depth": max(s.queue_depth for s in self.samples),
            "mean_gpu_cache_usage": sum(s.gpu_cache_usage for s in self.samples) / n,
            "total_preemptions": max(s.num_preemptions for s in self.samples) - min(s.num_preemptions for s in self.samples),
            "starved_fraction": sum(1 for s in self.samples if s.batch_occupancy < 0.5) / n,
        }
