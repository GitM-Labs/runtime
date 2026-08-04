"""vLLM's own Prometheus counters, turned into a per-window serving account.

When gitm drives the load it can time every request itself (see
:mod:`gitm.tracer.vllm_stats`). When it *attaches* to a server that is already
handling somebody else's traffic there is no client to time anything from, and
inventing one would change the very thing being measured. The server's counters are
then the only honest account of what happened inside the capture window.

Counters are monotonic, so a window is the difference between two snapshots. That
difference is what makes these numbers window-scoped rather than lifetime-scoped:
``vllm:e2e_request_latency_seconds_sum / _count`` read once gives the mean since the
process started — including the first requests after a cold start — which is not the
window and is not comparable between two captures of the same server.

Gauges (running/waiting queue depth) cannot be differenced; they are sampled on a
period through the window instead, because "was the engine ever queue-bound" is not
a question two endpoint reads can answer.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

# vLLM has shipped both ``vllm:foo`` and ``vllm_foo`` spellings depending on how the
# registry is configured. Families are looked up through every alias so a rename in
# the server does not silently turn every number into None.
_ALIASES: dict[str, tuple[str, ...]] = {
    "prompt_tokens": ("vllm:prompt_tokens_total", "vllm_prompt_tokens_total"),
    "generation_tokens": ("vllm:generation_tokens_total", "vllm_generation_tokens_total"),
    "requests_finished": ("vllm:request_success_total", "vllm_request_success_total"),
    "ttft_sum": ("vllm:time_to_first_token_seconds_sum", "vllm_time_to_first_token_seconds_sum"),
    "ttft_count": (
        "vllm:time_to_first_token_seconds_count",
        "vllm_time_to_first_token_seconds_count",
    ),
    "tpot_sum": (
        "vllm:time_per_output_token_seconds_sum",
        "vllm_time_per_output_token_seconds_sum",
    ),
    "tpot_count": (
        "vllm:time_per_output_token_seconds_count",
        "vllm_time_per_output_token_seconds_count",
    ),
    "e2e_sum": ("vllm:e2e_request_latency_seconds_sum", "vllm_e2e_request_latency_seconds_sum"),
    "e2e_count": (
        "vllm:e2e_request_latency_seconds_count",
        "vllm_e2e_request_latency_seconds_count",
    ),
    "running": ("vllm:num_requests_running", "vllm_num_requests_running"),
    "waiting": ("vllm:num_requests_waiting", "vllm_num_requests_waiting"),
    "kv_cache_usage": (
        "vllm:gpu_cache_usage_perc",
        "vllm_gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc",
    ),
}


def parse_prometheus(text: str) -> dict[str, float]:
    """Prometheus exposition text -> {family name: value summed over label sets}.

    Summing across label sets is right for every family used here: vLLM labels its
    metrics by ``model_name`` (one per server) and ``finished_reason`` (stop/length/
    abort), and a window's request count is all of those added up. Histogram buckets
    are skipped — the ``_sum``/``_count`` pair is what a mean needs, and keeping the
    buckets would double-count into the family totals.
    """
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition("{")
        if rest:
            _, _, tail = rest.partition("}")
            value_str = tail.strip()
        else:
            name, _, value_str = line.partition(" ")
        name = name.strip()
        if not name or name.endswith("_bucket"):
            continue
        try:
            value = float(value_str.split()[0])
        except (ValueError, IndexError):
            continue
        if value != value:  # NaN: an untouched histogram, not a zero
            continue
        out[name] = out.get(name, 0.0) + value
    return out


def _family(snapshot: dict[str, float], key: str) -> float | None:
    for name in _ALIASES[key]:
        if name in snapshot:
            return snapshot[name]
    return None


def _delta(before: dict[str, float], after: dict[str, float], key: str) -> float | None:
    a, b = _family(before, key), _family(after, key)
    if a is None or b is None:
        return None
    # A counter that went backwards means the server restarted mid-window; the
    # difference is meaningless and reporting it as a small positive number would be
    # worse than reporting nothing.
    return b - a if b >= a else None


def _mean(before, after, sum_key: str, count_key: str) -> float | None:
    ds = _delta(before, after, sum_key)
    dc = _delta(before, after, count_key)
    if ds is None or dc is None or dc <= 0:
        return None
    return ds / dc


def _percentile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    ordered = sorted(vals)
    idx = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[idx]


@dataclass
class ServerWindow:
    """What the server says it did between two snapshots."""

    latency_source: str = "server-prometheus"
    window_s: float | None = None
    requests_finished: float | None = None
    prompt_tokens: float | None = None
    generation_tokens: float | None = None
    output_tokens_per_s: float | None = None
    ttft_mean_s: float | None = None
    tpot_mean_s: float | None = None
    e2e_mean_s: float | None = None
    running_p50: float | None = None
    running_p95: float | None = None
    waiting_p50: float | None = None
    waiting_p95: float | None = None
    kv_cache_usage_p95: float | None = None
    n_samples: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def window_from_snapshots(
    before_text: str,
    after_text: str,
    *,
    window_s: float | None = None,
    samples: list[dict] | None = None,
) -> ServerWindow:
    """Difference two ``/metrics`` snapshots into a window-scoped summary."""
    before = parse_prometheus(before_text)
    after = parse_prometheus(after_text)

    w = ServerWindow(window_s=window_s)
    w.requests_finished = _delta(before, after, "requests_finished")
    w.prompt_tokens = _delta(before, after, "prompt_tokens")
    w.generation_tokens = _delta(before, after, "generation_tokens")
    w.ttft_mean_s = _mean(before, after, "ttft_sum", "ttft_count")
    w.tpot_mean_s = _mean(before, after, "tpot_sum", "tpot_count")
    w.e2e_mean_s = _mean(before, after, "e2e_sum", "e2e_count")
    if w.generation_tokens is not None and window_s:
        w.output_tokens_per_s = w.generation_tokens / window_s

    if samples:
        w.n_samples = len(samples)
        running = [s["running"] for s in samples if s.get("running") is not None]
        waiting = [s["waiting"] for s in samples if s.get("waiting") is not None]
        kv = [s["kv_cache_usage"] for s in samples if s.get("kv_cache_usage") is not None]
        w.running_p50, w.running_p95 = _percentile(running, 0.5), _percentile(running, 0.95)
        w.waiting_p50, w.waiting_p95 = _percentile(waiting, 0.5), _percentile(waiting, 0.95)
        w.kv_cache_usage_p95 = _percentile(kv, 0.95)

    if not before or not after:
        w.notes.append(
            "the /metrics endpoint was unreadable, so there is no server-side account "
            "of this window — the trace stands alone"
        )
    elif w.requests_finished is None:
        w.notes.append(
            "no vLLM request counters in /metrics: either this is not a vLLM server or "
            "it was started with --disable-log-stats"
        )
    elif w.requests_finished == 0:
        w.notes.append(
            "zero requests completed inside the window — the server was idle, so any "
            "kernels in this trace are background work, not serving work"
        )
    if w.running_p95 == 0 and w.n_samples:
        w.notes.append("the running-request gauge never left zero for the whole window")
    return w


def fetch_metrics(base_url: str, timeout: float = 15.0) -> str:
    """Raw ``/metrics`` text, or a commented error line — never an exception.

    A capture must not die because a scrape timed out: the trace is the artifact that
    costs GPU time to reproduce, and the metrics are context around it.
    """
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - any failure degrades to a note
        return f"# unavailable: {exc}\n"


def snapshot_metrics(base_url: str, dest: Path, timeout: float = 15.0) -> str:
    """Fetch and save the raw text, returning it as well.

    The raw file is kept alongside the parsed summary on purpose: vLLM's own TTFT and
    TPOT histograms are server-side truth, and when they disagree with a client-side
    summary the gap is the thing worth looking at. Keeping both makes it visible
    instead of arguable.
    """
    text = fetch_metrics(base_url, timeout)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    return text


class MetricsSampler:
    """Poll the gauges on a period for the length of the capture window.

    Queue depth is the difference between "the GPU was busy" and "the GPU was busy
    because requests were piling up", and it is invisible to before/after counter
    reads. Runs on a daemon thread and swallows every scrape error: a sampler that
    can take down a capture is worse than a sampler with holes in it.
    """

    def __init__(self, base_url: str, *, interval_s: float = 1.0) -> None:
        self.base_url = base_url
        self.interval_s = max(interval_s, 0.05)
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            text = fetch_metrics(self.base_url, timeout=min(self.interval_s * 4, 10.0))
            snap = parse_prometheus(text)
            if snap:
                self.samples.append(
                    {
                        "wall_s": time.time(),
                        "running": _family(snap, "running"),
                        "waiting": _family(snap, "waiting"),
                        "kv_cache_usage": _family(snap, "kv_cache_usage"),
                    }
                )
            self._stop.wait(self.interval_s)

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        return self.samples

    def write(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            for s in self.samples:
                fh.write(json.dumps(s))
                fh.write("\n")
