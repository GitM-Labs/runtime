"""Capture kernels from a vLLM server that is already running.

The launch path (:mod:`gitm.serve.vllm`) owns the server: it sets the injection
environment, waits out weight load and CUDA-graph capture, then drives its own load.
None of that is available for a server that is already up and serving. What is
available is the piece that matters — the collector inside that server is already
running, and the window it writes into is opened by a *file*, not by an API call. So
a short-lived gitm invocation can open a window on a long-lived server, watch it
handle real traffic, close the window, and merge the shards, without the server ever
knowing and without a single request being rerouted.

Two window shapes, and the difference is not cosmetic:

* **observe** (default) — arm, wait ``--duration`` while the server does whatever it
  was already doing, disarm. This is the only mode that measures production: the
  traffic mix, the batch shapes, and the queue depth are the real ones. Latency comes
  from vLLM's own counters (:mod:`gitm.serve.metrics`), because there is no client
  here to time anything and adding one would change what is being measured.
* **drive** (``--requests N``) — arm and issue synthetic load, same generator the
  launch path uses. Reproducible and comparable across runs, but it is now gitm's
  traffic mixed into somebody else's, so it is for a server you own.

Fail-open throughout: the window is disarmed on every exit path, nothing is written
into the server's process, and the server is never signalled. Detaching leaves a job
that cannot tell it was traced.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path

from gitm.serve import discover, metrics
from gitm.serve.artifacts import (
    CaptureResult,
    print_result,
    write_capture_artifacts,
    write_preflight,
)
from gitm.tracer import injection

WORKLOAD_ID = "vllm-attach"

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


@dataclass
class AttachOptions:
    """Everything the attach path can be told. Defaults describe the common case:
    find the one vLLM server on this box and watch it for 30 seconds."""

    pid: int | None = None
    port: int | None = None
    base_url: str | None = None
    out: Path | None = None
    duration_s: float = 30.0
    requests: int = 0
    concurrency: int = 64
    input_tokens: int = 1024
    output_tokens: int = 256
    seed: int = 42
    ignore_eos: bool = True
    request_timeout: float = 600.0
    metrics_interval: float = 1.0
    dry_run: bool = False
    proc: Path = discover.PROC

    @property
    def mode(self) -> str:
        return "drive" if self.requests > 0 else "observe"


def _port_from_cmdline(cmdline: list[str]) -> int:
    """The port a vLLM server is serving on, read off its own command line.

    Saves the operator from repeating a flag gitm can already see, and is right more
    often than assuming 8000 — a box with two servers has at most one of them there.
    """
    for i, a in enumerate(cmdline):
        if a == "--port" and i + 1 < len(cmdline) and cmdline[i + 1].isdigit():
            return int(cmdline[i + 1])
        if a.startswith("--port="):
            tail = a.split("=", 1)[1]
            if tail.isdigit():
                return int(tail)
    return 8000


def _base_url_for(target: discover.Target, opts: AttachOptions) -> str:
    if opts.base_url:
        return opts.base_url.rstrip("/")
    port = opts.port or _port_from_cmdline(target.cmdline)
    return f"http://127.0.0.1:{port}"


def attach_checks(target: discover.Target, base_url: str) -> list:
    """What has to hold before a window is worth opening.

    Deliberately not a copy of the launch preflight: GPU count, NVLink topology and
    serve flags are settled facts for a process that is already running — checking
    them here would produce warnings about a decision nobody can still change. What is
    checkable is whether *this* capture can bound and read a window.
    """
    from gitm.serve.vllm import Check

    checks: list = []

    checks.append(
        Check(
            "target",
            "pass" if target.traceable else "fail",
            f"PID {target.pid}: {target.reason}",
        )
    )
    if not target.traceable:
        return checks

    # The cohort: frontend + EngineCore + one worker per TP rank. A cohort of one is
    # the signature of a server whose engine processes were spawned before the
    # injection variable was exported — it traces cleanly and contains no model
    # kernels, which is the most expensive way to learn nothing.
    n = len(target.shard_pids)
    checks.append(
        Check(
            "collector-cohort",
            "pass" if n > 1 else "warn",
            f"{n} process(es) writing shards under {target.trace_out}"
            + (
                ""
                if n > 1
                else " — only one CUDA process is collecting, so the engine's kernels "
                "are probably not in it. Check the server was launched with the "
                "injection variable already exported."
            ),
        )
    )

    if injection.cupti_now():
        checks.append(Check("cupti-clock", "pass", "readable — the window can be bounded"))
    else:
        checks.append(
            Check(
                "cupti-clock",
                "fail",
                "cannot read the CUPTI clock, so the window cannot be bounded in the "
                "collector's time domain. The shards hold records from every earlier "
                "window too, and merging them unbounded would silently mix runs. "
                "Build the shim: python -m gitm.tracer._cupti.build",
            )
        )

    # Two armed windows on one server is not a partial failure, it is two traces that
    # both contain both windows' kernels.
    prior_out = os.environ.get(injection.ENV_OUT)
    try:
        os.environ[injection.ENV_OUT] = str(target.trace_out)
        arm_path = injection.arm_path()
        already_armed = arm_path.exists()
    finally:
        # Borrowed, not adopted: the real adoption happens only once preflight passes,
        # so a failed check never leaves this process pointed at another run's shards.
        if prior_out is None:
            os.environ.pop(injection.ENV_OUT, None)
        else:
            os.environ[injection.ENV_OUT] = prior_out
    checks.append(
        Check(
            "window",
            "fail" if already_armed else "pass",
            f"another capture window is already open on this server ({arm_path} exists) "
            "— wait for it to finish, or remove the marker if its owner died"
            if already_armed
            else "no window currently open",
        )
    )

    host = urllib.parse.urlparse(base_url).hostname or ""
    if host not in _LOCAL_HOSTS:
        checks.append(
            Check(
                "server-host",
                "fail",
                f"{base_url} is not on this box. The collector writes its shards to the "
                "server's local filesystem, so gitm must run on the same host.",
            )
        )
    else:
        checks.append(Check("server-host", "pass", base_url))

    text = metrics.fetch_metrics(base_url, timeout=10.0)
    if text.startswith("# unavailable"):
        checks.append(
            Check(
                "metrics",
                "warn",
                f"{base_url}/metrics unreadable ({text.strip()[:160]}) — the trace will "
                "have no server-side account of what ran during the window",
            )
        )
    else:
        checks.append(Check("metrics", "pass", f"{base_url}/metrics readable"))

    return checks


def _served_model(base_url: str) -> str:
    from gitm.serve.vllm import served_model_name

    return served_model_name(base_url, "unknown")


def attach_and_capture(opts: AttachOptions) -> tuple[int, CaptureResult | None]:
    """Open a capture window on a running vLLM server. Returns ``(exit_code, result)``.

    Exit codes match the launch path: 0 = usable trace, 1 = captured but unusable,
    2 = skipped because this target cannot be traced.
    """
    from gitm._paths import traces_dir
    from gitm.serve.vllm import print_checks

    target, how = discover.resolve_target(
        pid=opts.pid, port=opts.port, proc=opts.proc, base_url=opts.base_url
    )
    if target is None:
        print(f"==> no target: {how}")
        return 2, None

    print(f"==> target: {how}")
    print(f"    cmdline: {' '.join(target.cmdline[:8])}{' ...' if len(target.cmdline) > 8 else ''}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(opts.out) if opts.out else traces_dir() / f"vllm-attach-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    print(f"==> out dir: {out_dir}")

    base_url = _base_url_for(target, opts)
    print("==> preflight")
    checks = attach_checks(target, base_url)
    print_checks(checks)
    write_preflight(out_dir, checks)

    # Predict what this rank *should* be doing, from the server's own config, so the
    # measured trace has an honest floor to sit next to. Independent of traceability
    # and of the window — it reads the loaded model's shape, not the trace — so it
    # runs here, before the fail/dry-run returns, and never blocks a capture.
    _emit_predicted_graph(target, out_dir)

    if any(c.status == "fail" for c in checks):
        print("\ncannot attach — see the FAILs above.")
        if not target.traceable:
            print()
            print(target.remedy())
        return 2, None

    if opts.dry_run:
        print("\ndry run — target verified, no window opened, server untouched.")
        return 0, None

    # Adopt the server's collector: ENV_OUT is the rendezvous the injected library
    # already stats for the arm marker, and ENV_LIB is what makes capture() take the
    # merge path instead of starting an in-process CUPTI collector that would fight
    # the one already running. This process never initializes CUDA, so pointing the
    # driver's injection variable at the library here has no effect beyond that.
    os.environ[injection.ENV_LIB] = str(target.inject_lib)
    os.environ[injection.ENV_OUT] = str(target.trace_out)
    print(f"==> adopted collector at {target.trace_out}")

    model = _served_model(base_url)
    print(f"==> served model: {model}")

    sampler = metrics.MetricsSampler(base_url, interval_s=opts.metrics_interval)
    before = metrics.snapshot_metrics(base_url, out_dir / "metrics_before.txt")

    records: list = []
    failures = 0
    trace = None

    if opts.mode == "drive":
        print(
            f"==> capture window (drive): {opts.requests} requests @ concurrency "
            f"{opts.concurrency}, ~{opts.input_tokens} in / {opts.output_tokens} out"
        )
    else:
        print(
            f"==> capture window (observe): {opts.duration_s:.0f}s of the server's own "
            "traffic — no requests are issued by gitm"
        )

    from gitm.tracer import capture

    sampler.start()
    wall0 = time.time()
    try:
        with capture(trace_path, workload_id=WORKLOAD_ID, fingerprint=model) as trace:
            if opts.mode == "drive":
                from gitm.serve.vllm import build_prompts, drive_load

                prompts = build_prompts(opts.requests, opts.input_tokens, opts.seed)
                records, failures = drive_load(
                    base_url,
                    model,
                    prompts,
                    concurrency=opts.concurrency,
                    max_tokens=opts.output_tokens,
                    ignore_eos=opts.ignore_eos,
                    timeout_s=opts.request_timeout,
                )
            else:
                time.sleep(opts.duration_s)
    finally:
        wall = time.time() - wall0
        samples = sampler.stop()

    after = metrics.snapshot_metrics(base_url, out_dir / "metrics_after.txt")
    if samples:
        sampler.write(out_dir / "metrics_samples.jsonl")

    server_window = metrics.window_from_snapshots(
        before, after, window_s=wall, samples=samples
    )

    # Server-side truth is the summary for observe mode, where no client exists. In
    # drive mode both are kept: when the client's view and the server's histograms
    # disagree, the gap is the network and this process's own scheduling, and that is
    # worth seeing rather than averaging away.
    summary: dict = {"mode": opts.mode, "wall_s": wall, "server": server_window.to_dict()}
    if opts.mode == "drive":
        from gitm.tracer.vllm_stats import summarize_requests

        summary["client"] = {
            "latency_source": "client",
            "n_failed_requests": failures,
            **asdict(summarize_requests(records)),
        }

    # In observe mode the server's counters are the only witness to whether anything
    # was served. When they are unreadable (--disable-log-stats, or a scrape that
    # failed) the answer is *unknown*, not "no": failing the run there would throw away
    # a perfectly good trace over a missing metrics endpoint. The note in
    # server_window carries the caveat instead.
    had_traffic = (
        len(records) > 0
        if opts.mode == "drive"
        else server_window.requests_finished is None or server_window.requests_finished > 0
    )

    result = write_capture_artifacts(
        out_dir,
        trace=trace,
        trace_path=trace_path,
        checks=checks,
        serving_summary=summary,
        had_traffic=had_traffic,
        manifest={
            "workload_id": WORKLOAD_ID,
            "capture_mode": "attach",
            "window": opts.mode,
            "target": target.to_dict(),
            "base_url": base_url,
            "served_model": model,
            "load": (
                {
                    "requests": opts.requests,
                    "concurrency": opts.concurrency,
                    "input_tokens": opts.input_tokens,
                    "output_tokens": opts.output_tokens,
                    "ignore_eos": opts.ignore_eos,
                    "seed": opts.seed,
                }
                if opts.mode == "drive"
                else None
            ),
            "duration_s": opts.duration_s if opts.mode == "observe" else None,
        },
    )

    print(f"\n==> window closed after {wall:.1f}s — server left running (PID {target.pid})")
    if opts.mode == "drive":
        print(f"    client: {len(records)} ok / {failures} failed")
    if server_window.requests_finished is not None:
        print(
            f"    server: {server_window.requests_finished:.0f} requests, "
            f"{(server_window.generation_tokens or 0):.0f} output tokens"
            + (
                f", {server_window.output_tokens_per_s:.0f} tok/s"
                if server_window.output_tokens_per_s
                else ""
            )
        )
        if server_window.ttft_mean_s is not None:
            print(
                f"    server TTFT mean {server_window.ttft_mean_s * 1e3:.0f} ms   "
                f"TPOT mean {(server_window.tpot_mean_s or 0) * 1e3:.1f} ms"
            )
        if server_window.running_p50 is not None and server_window.waiting_p50 is not None:
            print(
                f"    queue depth p50/p95 running "
                f"{server_window.running_p50:.0f}/{server_window.running_p95:.0f}, "
                f"waiting {server_window.waiting_p50:.0f}/{server_window.waiting_p95:.0f}"
            )
    print_result(result)
    for note in server_window.notes:
        print("  - " + note)

    if result.status == "no_kernels":
        return 1, result
    if result.status == "no_traffic":
        # Not a crash and not a usable trace: an idle window is a real outcome that
        # should not be reported as a pass by automation.
        return 1, result
    return 0, result


def _emit_predicted_graph(target: discover.Target, out_dir: Path) -> None:
    """Write ``predicted_moe_graph.json`` — a per-rank floor from the live config.

    The first production consumer of :func:`predict_moe_graph`. On a config the
    planner cannot honestly predict (missing dominant terms, unpriceable dtype) it
    prints the named-key refusal and writes nothing — a defaulted DeepSeek prediction
    next to a real trace would be read as a measurement, which is the one outcome the
    gate exists to prevent. Never raises into the capture path.

    """
    from gitm.serve.model_config import LiveSpec, live_moe_spec

    try:
        resolved = live_moe_spec(target)
    except Exception as e:  # a prediction sidecar must never sink a real capture
        print(f"==> predicted graph: skipped ({type(e).__name__}: {e})")
        return

    if not isinstance(resolved, LiveSpec):
        print("==> predicted graph: skipped — live config not predictable:")
        for line in resolved.render().splitlines():
            print(f"    {line}")
        return

    from gitm.planner.context import build_planner_context, hardware_spec_for
    from gitm.planner.moe_graph import predict_moe_graph

    planner_ctx = build_planner_context()
    hw = hardware_spec_for(planner_ctx.peak)
    g = predict_moe_graph(resolved.spec, hw, resolved.batch, resolved.sharding)
    sh, spec = resolved.sharding, resolved.spec
    warnings = list(resolved.warnings)
    if planner_ctx.peak is None:
        warnings.append(
            f"GPU SKU {planner_ctx.sku or 'unknown'!r} is not in the hardware catalogue; "
            f"pricing uses fallback {hw.name!r}"
        )
    if g.has_unpriced_nodes:
        missing = []
        if g.has_unpriced_compute:
            missing.append("compute throughput")
        if g.has_unpriced_memory:
            missing.append("memory bandwidth")
        warnings.append(f"predicted nodes have unpriced {' and '.join(missing)}")
    if g.has_fallback_peaks:
        warnings.append("priced against fallback compute peaks; the ceiling is low")
    if g.has_fallback_bytes:
        warnings.append(
            "byte widths include an unknown-dtype bf16 fallback; the memory floor is approximate"
        )
    n_estimated = sum(1 for n in g.nodes if n.prediction.estimated)
    if n_estimated:
        warnings.append(f"{n_estimated} predicted node(s) use documented estimated cost models")

    payload = {
        "model_ref": resolved.model_ref,
        "config_source": str(resolved.source_path),
        "hardware": planner_ctx.sku,
        "hardware_pricing": hw.name,
        "hardware_is_fallback": planner_ctx.peak is None,
        "sharding": {"tp": sh.tp, "ep": sh.ep, "dp": sh.dp},
        "dtypes": {
            "weight": spec.weight_dtype,
            "expert": spec.expert_dtype,
            "kv": spec.kv_dtype,
            "act": spec.act_dtype,
        },
        "batch": {"batch": resolved.batch.batch, "kv_cache_len": resolved.batch.kv_cache_len},
        "applied_overrides": resolved.applied_overrides,
        "total_pred_s": g.total_pred_s,
        "has_unpriced_collectives": g.has_unpriced_collectives,
        "has_unpriced_nodes": g.has_unpriced_nodes,
        "has_unpriced_compute": g.has_unpriced_compute,
        "has_unpriced_memory": g.has_unpriced_memory,
        "has_fallback_peaks": g.has_fallback_peaks,
        "has_fallback_bytes": g.has_fallback_bytes,
        "warnings": warnings,
        "nodes": [
            {
                "op": n.op,
                "layer": n.layer,
                "t_pred_s": n.prediction.t_pred_s,
                "bound": n.prediction.bound,
                "dtype": n.prediction.dtype,
                "flops": n.prediction.flops,
                "bytes": n.prediction.bytes,
                "estimated": n.prediction.estimated,
                "bytes_are_fallback": n.prediction.bytes_are_fallback,
                "compute_is_unpriced": n.prediction.compute_is_unpriced,
                "memory_is_unpriced": n.prediction.memory_is_unpriced,
            }
            for n in g.nodes
        ],
    }
    (out_dir / "predicted_moe_graph.json").write_text(json.dumps(payload, indent=2))

    print(
        f"==> predicted graph: {resolved.model_ref} per-rank floor "
        f"{g.total_pred_s * 1e3:.2f} ms/step "
        f"(TP={sh.tp} EP={sh.ep} DP={sh.dp}, "
        f"w={spec.weight_dtype}/e={spec.expert_dtype}/kv={spec.kv_dtype})"
    )
    for warning in warnings:
        print(f"    - {warning}")


def describe_targets(proc: Path = discover.PROC) -> list[dict]:
    """Every vLLM server on this box with its traceability verdict — for ``--list``."""
    return [discover.classify(pid, proc).to_dict() for pid in discover.find_vllm_pids(proc)]


def print_targets(proc: Path = discover.PROC) -> int:
    targets = describe_targets(proc)
    if not targets:
        print("no vLLM server found in this box's process table.")
        return 1
    print(json.dumps(targets, indent=2))
    return 0
