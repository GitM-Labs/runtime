"""Traced ``vllm serve`` run — launch the OpenAI server, drive load, keep the CUPTI trace.

The existing vLLM path (``gitm run --workload vllm-decode``) builds an in-process
``LLM`` and calls ``generate``. An OpenAI *server* is a different shape: it is a
long-lived process that must be up, warm, and past CUDA-graph capture before any
measurement is meaningful, and the load has to arrive over HTTP. This script is that
shape, reusing the tracer unchanged:

  1. preflight — GPU count vs TP, NVLink topology, driver/vLLM CUDA majors, the
     injection library, the CUPTI clock, and the serve flags themselves (parsed by
     the installed vLLM's own parser, so a bad flag costs a second instead of the
     ~90s+ it takes to die after the weight load).
  2. launch ``vllm serve`` with ``CUDA_INJECTION64_PATH`` exported. The driver
     dlopens our collector into EngineCore and every TP worker; the in-process shim
     cannot see any of them. See gitm/tracer/injection.py.
  3. wait for ``/health``, then warm up — both OUTSIDE the capture window. Weight
     load, torch.compile and CUDA-graph capture are ~80s of kernels that would
     otherwise dominate the trace and make kernel-time coverage meaningless.
  4. arm the window (``capture()``), drive closed-loop streaming load, disarm.
  5. write the merged trace + a client-side serving summary + /metrics snapshots.

    python scripts/serve_capture.py                      # the pinned experiment below
    python scripts/serve_capture.py --dry-run            # preflight only, no GPU needed
    python scripts/serve_capture.py -- vllm serve OTHER/MODEL --tensor-parallel-size 2

Exit codes: 0 = trace captured, 1 = failed, 2 = skipped (preflight says this box
cannot run it).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

# The experiment this script was written for. Overridable by passing a full command
# after ``--``; kept here so the pinned run is reproducible without a shell history.
# --host/--port are appended by the script (it needs to know where to send load).
DEFAULT_SERVE_ARGV = [
    "vllm", "serve", "Qwen/Qwen3.6-35B-A3B-FP8",
    "--trust-remote-code",
    "--tensor-parallel-size", "2",
    "--gpu-memory-utilization", "0.95",
    "--max-num-batched-tokens", "8192",
    "--max-num-seqs", "256",
    "--max-model-len", "auto",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_xml",
    "--reasoning-parser", "qwen3",
    "--mm-encoder-tp-mode", "data",
]

WORKLOAD_ID = "vllm-serve"


# --------------------------------------------------------------------------- preflight


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str


def _run(cmd: list[str], timeout: float = 60.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _arg_of(argv: list[str], flag: str) -> str | None:
    """Value of ``--flag v`` or ``--flag=v`` in a serve argv."""
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def check_gpus(tp: int) -> list[Check]:
    rc, out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"])
    if rc != 0:
        return [Check("gpus", "fail", f"nvidia-smi unavailable: {out.strip()[:200]}")]
    gpus = [ln.strip() for ln in out.splitlines() if ln.strip()]
    status = "pass" if len(gpus) >= tp else "fail"
    return [Check("gpus", status, f"{len(gpus)} visible (need {tp} for TP={tp}): " + "; ".join(gpus))]


def check_nvlink(tp: int) -> list[Check]:
    """Warn — loudly — when the pod's GPUs are not NVLink-peered.

    TP=2 decode does an all-reduce every layer every step. On PCIe those all-reduces
    land in the critical path and the numbers describe the interconnect, not the
    model. Same command, same flags, a different conclusion — so this is worth
    catching before the run, not after.
    """
    if tp < 2:
        return []
    rc, out = _run(["nvidia-smi", "topo", "-m"])
    if rc != 0:
        return [Check("nvlink", "warn", "could not read `nvidia-smi topo -m`")]
    link = None
    for ln in out.splitlines():
        if ln.startswith("GPU0"):
            cells = ln.split()
            if len(cells) > 2:
                link = cells[2]  # GPU0 row, GPU1 column
            break
    if link is None:
        return [Check("nvlink", "warn", "could not parse the topology matrix")]
    if link.startswith("NV"):
        return [Check("nvlink", "pass", f"GPU0<->GPU1 = {link}")]
    return [Check("nvlink", "warn",
                  f"GPU0<->GPU1 = {link}, NOT NVLink. TP={tp} all-reduces will run over "
                  f"{link}; decode latency here is not comparable with an NVLink SKU.")]


def check_driver_stack() -> list[Check]:
    """Driver vs torch vs vLLM CUDA majors, via the module that already knows."""
    try:
        from gitm import cuda_env
    except Exception as exc:
        return [Check("cuda-stack", "warn", f"gitm.cuda_env unimportable: {exc}")]

    driver = cuda_env.driver_cuda()
    if driver is None:
        return [Check("cuda-stack", "fail", "no NVIDIA driver detected")]
    if cuda_env.stack_for(driver) is None:
        return [Check("cuda-stack", "fail",
                      f"driver supports only CUDA {driver[0]}.{driver[1]}; no pinned vLLM "
                      f"stack runs here. Redeploy with CUDA {max(cuda_env.SUPPORTED_STACKS)}.0+.")]
    problems = cuda_env.check()
    if problems:
        return [Check("cuda-stack", "fail", "\n".join(str(p) for p in problems))]
    return [Check("cuda-stack", "pass", f"driver CUDA {driver[0]}.{driver[1]}, stack consistent")]


def check_injection_lib() -> list[Check]:
    """The injection library must exist before the server starts — the driver reads
    CUDA_INJECTION64_PATH at CUDA init and never looks again."""
    from gitm.tracer import injection

    lib = injection.lib_path()
    if not lib.exists():
        rc, out = _run([sys.executable, "-m", "gitm.tracer._cupti.build"], timeout=600)
        if rc != 0 or not lib.exists():
            return [Check("injection-lib", "fail",
                          f"{lib} missing and the build failed:\n{out.strip()[-800:]}")]
    return [Check("injection-lib", "pass", str(lib))]


def check_cupti_clock() -> list[Check]:
    """Without the in-process shim the window cannot be bounded in the CUPTI clock
    domain, and the merged trace silently includes weight load and graph capture."""
    from gitm.tracer import injection

    return [Check("cupti-clock", "pass", "readable")] if injection.cupti_now() else [
        Check("cupti-clock", "warn",
              "cannot read the CUPTI clock — the trace will NOT be windowed and will "
              "include model load and CUDA-graph capture. Build the shim: "
              "python -m gitm.tracer._cupti.build")
    ]


_ARG_CHECK_SRC = r"""
import json, sys
argv = json.loads(sys.argv[1])
try:
    try:
        from vllm.utils import FlexibleArgumentParser
    except Exception:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser
except Exception as exc:
    print("cannot build vLLM's parser: %r" % (exc,))
    raise SystemExit(3)
parser = make_arg_parser(FlexibleArgumentParser())
# `vllm serve <model>` passes the model as a positional that the OpenAI arg parser
# itself does not declare (the serve subcommand adds it), so declare it here.
parser.add_argument("model_tag", nargs="?")
try:
    parser.parse_args(argv)
except SystemExit:
    raise SystemExit(4)
print("accepted by vLLM's own parser")
"""


def check_serve_args(serve_argv: list[str]) -> list[Check]:
    """Parse the serve flags with the installed vLLM's parser, without building anything.

    Runs in a subprocess so importing vLLM cannot leave state (or a CUDA context) in
    this process. A parser we cannot construct is a warn, not a fail: vLLM moves these
    modules between releases, and refusing to run because the *check* broke would be
    worse than letting the server reject the flags itself 90 seconds later.
    """
    args = serve_argv[2:] if serve_argv[:2] == ["vllm", "serve"] else serve_argv[1:]
    p = subprocess.run(
        [sys.executable, "-c", _ARG_CHECK_SRC, json.dumps(args)],
        capture_output=True, text=True, timeout=600,
    )
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    if p.returncode == 0:
        return [Check("serve-args", "pass", out.splitlines()[-1] if out else "accepted")]
    if p.returncode == 3:
        return [Check("serve-args", "warn", f"could not validate: {out[-400:]}")]
    return [Check("serve-args", "fail", f"vLLM rejects these flags:\n{out[-800:]}")]


def preflight(serve_argv: list[str], *, skip_args: bool = False) -> list[Check]:
    tp = int(_arg_of(serve_argv, "--tensor-parallel-size") or _arg_of(serve_argv, "-tp") or 1)
    checks: list[Check] = []
    checks += check_gpus(tp)
    checks += check_nvlink(tp)
    checks += check_driver_stack()
    checks += check_injection_lib()
    checks += check_cupti_clock()
    if not skip_args:
        checks += check_serve_args(serve_argv)
    return checks


def print_checks(checks: list[Check]) -> None:
    colour = {"pass": "\033[32mPASS\033[0m", "warn": "\033[33mWARN\033[0m", "fail": "\033[31mFAIL\033[0m"}
    for c in checks:
        head, *rest = c.detail.splitlines() or [""]
        print(f"  {colour[c.status]} {c.name}: {head}")
        for ln in rest:
            print(f"        {ln}")


# ------------------------------------------------------------------------- the server


def wait_healthy(base: str, proc: subprocess.Popen, timeout_s: float, log_path: Path) -> None:
    """Block until ``/health`` answers, or the server dies, or we run out of patience.

    A 35B FP8 checkpoint is a ~35GB download on a cold pod, so the default budget is
    generous; the server dying is detected immediately either way.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-2000:] if log_path.exists() else ""
            raise RuntimeError(f"vllm serve exited with {proc.returncode} before becoming "
                               f"healthy. Last log lines:\n{tail}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2.0)
    raise RuntimeError(f"server not healthy after {timeout_s:.0f}s — see {log_path}")


def served_model_name(base: str, fallback: str) -> str:
    """What the server calls the model. ``--served-model-name`` can rename it, and a
    completion request with the wrong id is a 404, not a slow path."""
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
            data = json.loads(r.read())
        return data["data"][0]["id"]
    except Exception:
        return fallback


def snapshot_metrics(base: str, dest: Path) -> None:
    """Save the raw Prometheus text. vLLM's own TTFT/TPOT histograms are server-side
    truth; the summary this script computes is the client's view, which includes the
    network and the client's own scheduling. Keeping both makes the gap visible
    instead of arguable."""
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=15) as r:
            dest.write_bytes(r.read())
    except Exception as exc:
        dest.write_text(f"# unavailable: {exc}\n")


def shutdown(proc: subprocess.Popen) -> None:
    """SIGINT the whole process group, then escalate.

    vLLM's server is a tree — EngineCore plus one worker per TP rank — and signalling
    only the parent leaves the workers holding GPU memory.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except OSError:
        proc.terminate()
    try:
        proc.wait(timeout=60)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        proc.kill()


# -------------------------------------------------------------------------- load gen


def build_prompts(n: int, approx_tokens: int, seed: int) -> list[str]:
    """Deterministic synthetic prompts of roughly ``approx_tokens`` tokens.

    Random words, not repeated text: a repeated string compresses into prefix-cache
    hits across requests, which would quietly turn a prefill+decode measurement into a
    cache-hit measurement. ~0.75 words/token is the usual English ratio; the exact
    count doesn't matter as long as it is stable across runs.
    """
    rng = random.Random(seed)
    vocab = [f"tok{i:04d}" for i in range(4096)]
    n_words = max(int(approx_tokens * 0.75), 8)
    return [" ".join(rng.choice(vocab) for _ in range(n_words)) for _ in range(n)]


def one_request(base: str, model: str, prompt: str, max_tokens: int,
                ignore_eos: bool, timeout_s: float):
    """Issue one streaming chat completion; return a RequestRecord, or None on error.

    Streaming is what makes TTFT observable at all — a non-streamed response only
    reveals when the *last* token landed. The first token can arrive as
    ``reasoning_content`` rather than ``content`` on this model (the run enables
    ``--reasoning-parser qwen3``), so both count: taking only ``content`` would
    mis-time TTFT by the entire reasoning block.
    """
    from gitm.tracer.vllm_stats import RequestRecord

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        # vLLM extension. Every request then decodes exactly max_tokens, so the decode
        # phase has a fixed shape instead of one set by where the model chose to stop.
        body["ignore_eos"] = True

    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    rec = RequestRecord()
    rec.arrival_wall_s = time.time()
    chunks = 0
    usage_tokens: int | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                    usage_tokens = int(usage["completion_tokens"])
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        now = time.time()
                        if rec.first_token_wall_s is None:
                            rec.first_token_wall_s = now
                        rec.finished_wall_s = now
                        chunks += 1
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    # usage is authoritative; the chunk count is the fallback when a build doesn't
    # emit it, and it undercounts whenever a chunk carries several tokens.
    rec.n_output_tokens = usage_tokens if usage_tokens is not None else chunks
    return rec


def drive_load(base: str, model: str, prompts: list[str], *, concurrency: int,
               max_tokens: int, ignore_eos: bool, timeout_s: float):
    """Closed-loop: ``concurrency`` workers, each sending the next prompt the moment
    its previous one finishes. That is what keeps the engine's running-batch full and
    makes ``--max-num-seqs 256`` the binding constraint rather than the client."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(
            lambda p: one_request(base, model, p, max_tokens, ignore_eos, timeout_s),
            prompts,
        ))
    return [r for r in results if r is not None], sum(1 for r in results if r is None)


# ------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    serve_argv = DEFAULT_SERVE_ARGV
    if "--" in argv:
        i = argv.index("--")
        argv, serve_argv = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser(
        description="Run vllm serve under the CUPTI injection tracer and capture a trace.",
        epilog="Pass a full serve command after `--` to override the pinned experiment.",
    )
    ap.add_argument("--out", default=None, help="output dir (default $GITM_SCRATCH/traces/vllm-serve-<ts>)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--requests", type=int, default=512, help="requests inside the window")
    ap.add_argument("--concurrency", type=int, default=256, help="in-flight requests (match --max-num-seqs)")
    ap.add_argument("--input-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=8, help="requests before arming the window")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ignore-eos", action="store_true", help="let the model stop early")
    ap.add_argument("--health-timeout", type=float, default=1800.0)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--dry-run", action="store_true", help="preflight only; never starts a server")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--keep-server", action="store_true", help="leave the server up after capture")
    args = ap.parse_args(argv)

    from gitm._paths import traces_dir

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else traces_dir() / f"vllm-serve-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"

    # Both variables must be set BEFORE the server starts CUDA — the driver reads
    # CUDA_INJECTION64_PATH once, at CUDA init, and it is inherited by every child.
    # This process needs them too: injection.active() is how capture() knows to merge
    # the children's shards instead of collecting in-process.
    from gitm.tracer import injection

    print(f"==> out dir: {out_dir}")
    print("==> preflight")
    checks = [] if args.skip_preflight else preflight(serve_argv, skip_args=False)
    print_checks(checks)
    if any(c.status == "fail" for c in checks):
        (out_dir / "preflight.json").write_text(json.dumps([asdict(c) for c in checks], indent=2))
        print("\npreflight failed — not starting the server. See the FAILs above.")
        return 2
    (out_dir / "preflight.json").write_text(json.dumps([asdict(c) for c in checks], indent=2))
    if args.dry_run:
        print("\ndry run — preflight only, nothing launched.")
        return 0

    os.environ.update(injection.run_env(trace_path))
    print(f"==> {injection.ENV_LIB}={os.environ[injection.ENV_LIB]}")

    base = f"http://{args.host}:{args.port}"
    cmd = list(serve_argv)
    if not _arg_of(cmd, "--port"):
        cmd += ["--port", str(args.port)]
    if not _arg_of(cmd, "--host"):
        cmd += ["--host", args.host]

    log_path = out_dir / "server.log"
    print("==> launching: " + " ".join(cmd))
    print(f"    server log -> {log_path}")
    log_fh = log_path.open("wb")
    # start_new_session: the server is a process tree (EngineCore + one worker per TP
    # rank); its own group is what makes a clean group-wide shutdown possible.
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)

    records: list = []
    failures = 0
    trace = None
    try:
        print(f"==> waiting for /health (up to {args.health_timeout:.0f}s: weight download, "
              f"torch.compile, CUDA-graph capture)")
        t0 = time.time()
        wait_healthy(base, proc, args.health_timeout, log_path)
        print(f"    healthy after {time.time() - t0:.1f}s")

        model = served_model_name(base, serve_argv[2] if len(serve_argv) > 2 else "model")
        print(f"==> served model: {model}")

        ignore_eos = not args.no_ignore_eos
        if args.warmup:
            print(f"==> warmup: {args.warmup} requests (outside the capture window)")
            drive_load(base, model, build_prompts(args.warmup, args.input_tokens, args.seed - 1),
                       concurrency=min(args.warmup, args.concurrency),
                       max_tokens=min(args.output_tokens, 32), ignore_eos=ignore_eos,
                       timeout_s=args.request_timeout)

        snapshot_metrics(base, out_dir / "metrics_before.txt")

        prompts = build_prompts(args.requests, args.input_tokens, args.seed)
        print(f"==> capture window: {args.requests} requests @ concurrency {args.concurrency}, "
              f"~{args.input_tokens} in / {args.output_tokens} out")

        from gitm.tracer import capture

        wall0 = time.time()
        with capture(trace_path, workload_id=WORKLOAD_ID, fingerprint=model) as trace:
            records, failures = drive_load(
                base, model, prompts,
                concurrency=args.concurrency, max_tokens=args.output_tokens,
                ignore_eos=ignore_eos, timeout_s=args.request_timeout,
            )
        wall = time.time() - wall0

        snapshot_metrics(base, out_dir / "metrics_after.txt")
    finally:
        if args.keep_server:
            print(f"==> leaving the server up (pid {proc.pid}) — kill it with: "
                  f"kill -INT -{proc.pid}")
        else:
            print("==> shutting the server down")
            shutdown(proc)
        log_fh.close()

    # ---- artifacts
    from gitm.tracer.vllm_stats import summarize_requests

    summary = summarize_requests(records)
    (out_dir / "serving_summary.json").write_text(json.dumps({
        # Client-side wall clock, NOT vLLM's RequestOutput.metrics: the server path
        # never hands those to an HTTP client. It therefore includes network and this
        # client's own scheduling. metrics_{before,after}.txt hold vLLM's own
        # histograms for the same window; prefer those when the two disagree.
        "latency_source": "client",
        "n_failed_requests": failures,
        "wall_s": wall,
        **asdict(summary),
    }, indent=2))

    kernels = [e for e in trace.events if getattr(e, "kind", None) == "kernel"]

    # What the trace is made of, and whether it can be read at all: GEMM/MoE shares,
    # truncated names, and GPU-active time. A trace with kernels in it can still be
    # unusable (graph-replay attribution loss shows up as an idle-looking GPU), so
    # this is written and printed on every run, not only on the empty-trace path.
    from gitm.tracer.kernel_taxonomy import format_breakdown, summarize_kernels

    breakdown = summarize_kernels(kernels, window_ns=trace.duration_ns)
    (out_dir / "kernel_breakdown.json").write_text(json.dumps({
        **asdict(breakdown),
        "warnings": breakdown.warnings(),
    }, indent=2))

    (out_dir / "run_manifest.json").write_text(json.dumps({
        "workload_id": WORKLOAD_ID,
        "serve_argv": cmd,
        "load": {
            "requests": args.requests, "concurrency": args.concurrency,
            "input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
            "ignore_eos": not args.no_ignore_eos, "seed": args.seed,
        },
        "trace": {
            "path": str(trace_path), "run_id": trace.run_id, "source": trace.source,
            "device_count": trace.device_count, "events": len(trace.events),
            "kernels": len(kernels), "duration_ns": trace.duration_ns,
        },
        "preflight": [asdict(c) for c in checks],
    }, indent=2))

    print(f"\n==> {len(records)} ok / {failures} failed in {wall:.1f}s")
    if summary.ttft_p50_s is not None:
        print(f"    TTFT p50/p95 {summary.ttft_p50_s * 1e3:.0f}/{summary.ttft_p95_s * 1e3:.0f} ms"
              f"   TPOT p50 {(summary.tpot_p50_s or 0) * 1e3:.1f} ms")
    print(f"    trace: {len(trace.events)} events, {len(kernels)} kernels -> {trace_path}")
    print()
    print(format_breakdown(breakdown))

    problems = breakdown.warnings()
    if problems:
        print("\nTRACE COVERAGE WARNINGS")
        for w in problems:
            print("  - " + w)

    if not kernels:
        return 1
    if not records:
        print("\nEvery request failed — see server.log.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
