"""Traced ``vllm serve`` run — launch the OpenAI server, drive load, keep the CUPTI trace.

This is the *launch* half of vLLM kernel capture; :mod:`gitm.serve.attach` is the
half that adopts a server someone else already started. Both end in the same
artifacts, because the only difference between them is who owns the process.

The in-process vLLM path (``gitm run --workload vllm-decode``) builds an ``LLM``
and calls ``generate``. An OpenAI *server* is a different shape: it is a long-lived
process that must be up, warm, and past CUDA-graph capture before any measurement is
meaningful, and the load has to arrive over HTTP. This module is that shape, reusing
the tracer unchanged:

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

    gitm capture serve                       # the pinned experiment below
    gitm capture serve --dry-run             # preflight only, no GPU needed
    gitm capture serve -- vllm serve OTHER/MODEL --tensor-parallel-size 2

``--keep-server`` leaves the process up afterwards, which is the intended handoff
into ``gitm capture attach``: one launch, many capture windows.

Exit codes: 0 = trace captured, 1 = failed, 2 = skipped (preflight says this box
cannot run it).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The experiment this script was written for. Overridable by passing a full command
# after ``--``; kept here so the pinned run is reproducible without a shell history.
# --host/--port are appended by the script (it needs to know where to send load).
# --tensor-parallel-size is deliberately absent: it defaults to however many GPUs
# this box actually exposes (see visible_devices), so the same command is correct on
# a 2x and a 4x pod. Pass it explicitly here or after `--` to pin it.
DEFAULT_SERVE_ARGV = [
    "vllm", "serve", "Qwen/Qwen3.6-35B-A3B-FP8",
    "--trust-remote-code",
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


@dataclass
class Devices:
    """The GPUs this run can actually use.

    ``indices`` are *physical* indices — the ones `nvidia-smi topo -m` labels its
    rows with — and are ``None`` when they cannot be resolved, which happens when
    ``CUDA_VISIBLE_DEVICES`` is set to UUIDs rather than ordinals. The count is
    still known in that case; only per-pair topology lookup is impossible.
    """

    indices: list[int] | None
    count: int
    source: str
    names: list[str] = field(default_factory=list)


def visible_devices() -> Devices:
    """How many GPUs this run gets, honouring ``CUDA_VISIBLE_DEVICES``.

    nvidia-smi reports every card in the box regardless of that variable, but vLLM
    only ever sees the masked subset — so sizing tensor parallelism off nvidia-smi
    alone would ask for 4 ranks on a box where CUDA exposes 2, and the failure lands
    deep inside distributed init rather than here.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        if all(p.isdigit() for p in parts):
            return Devices(indices=[int(p) for p in parts], count=len(parts),
                           source="CUDA_VISIBLE_DEVICES")
        # UUID form (GPU-xxxxxxxx): the count is trustworthy, the mapping to
        # topology rows is not.
        return Devices(indices=None, count=len(parts),
                       source="CUDA_VISIBLE_DEVICES (UUIDs)")

    rc, out = _run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"])
    if rc != 0:
        return Devices(indices=None, count=0, source="nvidia-smi unavailable")
    idx, names = [], []
    for ln in out.splitlines():
        if not ln.strip():
            continue
        head, _, name = ln.partition(",")
        if head.strip().isdigit():
            idx.append(int(head.strip()))
            names.append(name.strip())
    return Devices(indices=idx, count=len(idx), source="nvidia-smi", names=names)


def check_gpus(tp: int, devices: Devices) -> list[Check]:
    if devices.count == 0:
        return [Check("gpus", "fail", f"no GPUs visible ({devices.source})")]
    detail = f"{devices.count} visible via {devices.source}"
    if devices.names:
        detail += ": " + "; ".join(devices.names)
    if devices.count < tp:
        return [Check("gpus", "fail", f"{detail} — but TP={tp} needs {tp}")]
    if devices.count > tp:
        # Not an error: a deliberate TP smaller than the box is a legitimate
        # experiment. It is worth saying out loud, because idle GPUs in a
        # throughput number are the kind of thing nobody notices in a report.
        detail += f" — running TP={tp}, leaving {devices.count - tp} GPU(s) idle"
    return [Check("gpus", "pass", detail)]


def parse_topo(out: str) -> dict[int, list[str]]:
    """Parse ``nvidia-smi topo -m`` into {gpu_index: [link to GPU0, GPU1, ...]}.

    Row labels are ``GPU<n>``; the trailing columns (CPU Affinity, NUMA, …) are
    kept and simply never indexed, since callers only ever look up GPU columns.
    """
    rows: dict[int, list[str]] = {}
    for ln in out.splitlines():
        parts = ln.split()
        if parts and re.fullmatch(r"GPU\d+", parts[0]):
            rows[int(parts[0][3:])] = parts[1:]
    return rows


def check_nvlink(tp: int, devices: Devices | None = None) -> list[Check]:
    """Warn — loudly — when the ranks a collective spans are not NVLink-peered.

    Decode does an all-reduce across every TP rank, every layer, every step. On PCIe
    those land in the critical path and the numbers describe the interconnect rather
    than the model — same command, same flags, a different conclusion.

    Every *pair* among the first ``tp`` ranks matters, not just the first two: a
    collective is only as fast as its worst edge, and a box can be NVLink-peered
    within a pair while crossing PCIe or QPI between pairs. Checking GPU0<->GPU1 and
    calling a 4-GPU box healthy is exactly the mistake that makes a TP=4 result
    look inexplicably worse than TP=2.
    """
    if tp < 2:
        return []
    rc, out = _run(["nvidia-smi", "topo", "-m"])
    if rc != 0:
        return [Check("nvlink", "warn", "could not read `nvidia-smi topo -m`")]

    rows = parse_topo(out)

    # The ranks this run occupies, in physical index space — CUDA_VISIBLE_DEVICES=2,3
    # means the pairs that matter are (2,3), not (0,1).
    if devices is not None and devices.indices is None:
        return [Check("nvlink", "warn",
                      f"cannot map {devices.source} to topology rows; "
                      f"interconnect between the {devices.count} visible GPUs unverified")]
    phys = list(devices.indices)[:tp] if devices and devices.indices else list(range(tp))
    if len(rows) < len(phys) or any(p not in rows for p in phys):
        return [Check("nvlink", "warn",
                      f"topology matrix has {len(rows)} GPU rows, need {phys}")]

    weak: list[str] = []
    links: list[str] = []
    for a, i in enumerate(phys):
        for j in phys[a + 1:]:
            try:
                link = rows[i][j]
            except (KeyError, IndexError):
                return [Check("nvlink", "warn", "could not parse the topology matrix")]
            links.append(f"{i}<->{j}={link}")
            if not link.startswith("NV"):
                weak.append(f"GPU{i}<->GPU{j} = {link}")

    if not weak:
        kinds = sorted({x.split("=")[1] for x in links})
        return [Check("nvlink", "pass",
                      f"all {len(links)} pairs across {tp} ranks {phys} are NVLink "
                      f"({', '.join(kinds)})")]
    return [Check("nvlink", "warn",
                  f"{len(weak)} of {len(links)} rank pairs are NOT NVLink: {'; '.join(weak)}. "
                  f"TP={tp} collectives run at the speed of the worst edge, so these "
                  f"numbers are not comparable with a fully NVLink-peered SKU.")]


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
    unverified = cuda_env.unverified_cuda_components()
    if unverified:
        return [
            Check(
                "cuda-stack",
                "warn",
                f"driver CUDA {driver[0]}.{driver[1]} detected, but "
                f"{' and '.join(unverified)} could not be verified; "
                "stack compatibility is unknown",
            )
        ]
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


def resolve_dp(serve_argv: list[str]) -> int:
    """Data-parallel size from the serve command (1 when unset)."""
    raw = _arg_of(serve_argv, "--data-parallel-size") or _arg_of(serve_argv, "-dp")
    return int(raw) if raw and raw.isdigit() else 1


def resolve_tp(serve_argv: list[str], devices: Devices, override: int | None) -> int:
    """Tensor-parallel size for this run: explicit wins, else the box divided by DP.

    vLLM's world size is TP x DP, so "default to every visible GPU" is only right
    when DP=1. With ``--data-parallel-size 4`` on a 4-GPU box the correct TP is 1;
    filling in 4 would ask for 16 GPUs and die in distributed init. Falls back to 1
    rather than 0 so a CPU box reports "no GPUs" instead of a nonsense TP.
    """
    explicit = _arg_of(serve_argv, "--tensor-parallel-size") or _arg_of(serve_argv, "-tp")
    if explicit and explicit.isdigit():
        return int(explicit)
    if override:
        return override
    return max(devices.count // max(resolve_dp(serve_argv), 1), 1)


def check_world(tp: int, dp: int, devices: Devices) -> list[Check]:
    """vLLM needs exactly TP x DP GPUs. Catch the mismatch before distributed init.

    Asking for more ranks than exist fails somewhere deep in NCCL bootstrap with a
    message that does not mention either flag. Asking for fewer silently leaves GPUs
    idle, which is worse: the run succeeds and the throughput number is quietly wrong.
    """
    world = tp * dp
    if devices.count == 0:
        return []
    if world > devices.count:
        return [Check("world-size", "fail",
                      f"TP={tp} x DP={dp} needs {world} GPUs, only {devices.count} visible")]
    if world < devices.count:
        return [Check("world-size", "warn",
                      f"TP={tp} x DP={dp} uses {world} of {devices.count} GPUs — "
                      f"{devices.count - world} idle, so per-GPU throughput is not "
                      f"comparable with a run that uses the whole box")]
    return [Check("world-size", "pass", f"TP={tp} x DP={dp} = {world} GPUs")]


def preflight(serve_argv: list[str], devices: Devices, tp: int,
              *, skip_args: bool = False) -> list[Check]:
    checks: list[Check] = []
    checks += check_gpus(tp, devices)
    checks += check_world(tp, resolve_dp(serve_argv), devices)
    checks += check_nvlink(tp, devices)
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
    except Exception as exc:
        warnings.warn(
            f"served-model discovery failed at {base}/v1/models; using configured "
            f"fallback {fallback!r} ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback


def snapshot_metrics(base: str, dest: Path) -> str:
    """Save the raw Prometheus text and return it.

    Delegates to :mod:`gitm.serve.metrics` so the launch and attach paths produce
    byte-identical ``metrics_*.txt`` files and feed the same parser — two captures of
    one server have to be comparable regardless of which path took them.
    """
    from gitm.serve.metrics import snapshot_metrics as _snapshot

    return _snapshot(base, dest)


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
    rec.token_count_source = "usage" if usage_tokens is not None else "chunks"
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


def add_serve_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The launch path's flags. Shared verbatim between ``gitm capture serve`` and the
    standalone script so the two can never drift into different defaults."""
    ap.add_argument("--out", default=None,
                    help="output dir (default $GITM_SCRATCH/traces/vllm-serve-<ts>)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--requests", type=int, default=512, help="requests inside the window")
    ap.add_argument("--concurrency", type=int, default=256,
                    help="in-flight requests (match --max-num-seqs)")
    ap.add_argument("--input-tokens", type=int, default=1024)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=8, help="requests before arming the window")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ignore-eos", action="store_true", help="let the model stop early")
    ap.add_argument("--health-timeout", type=float, default=1800.0)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--tp", type=int, default=None,
                    help="tensor-parallel size (default: every visible GPU)")
    ap.add_argument("--dry-run", action="store_true", help="preflight only; never starts a server")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--keep-server", action="store_true",
                    help="leave the server up after capture — the handoff into "
                         "`gitm capture attach` for further windows")
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    serve_argv = DEFAULT_SERVE_ARGV
    if "--" in argv:
        i = argv.index("--")
        argv, serve_argv = argv[:i], argv[i + 1:]

    ap = add_serve_arguments(argparse.ArgumentParser(
        description="Run vllm serve under the CUPTI injection tracer and capture a trace.",
        epilog="Pass a full serve command after `--` to override the pinned experiment.",
    ))
    rc, _ = launch_and_capture(ap.parse_args(argv), serve_argv)
    return rc


def launch_and_capture(args, serve_argv: list[str] | None = None):
    """Launch ``vllm serve`` under the collector, capture a window, tear it down.

    Returns ``(exit_code, CaptureResult | None)``. The result is ``None`` when nothing
    was captured — a failed preflight or a dry run — so a caller can tell "no trace"
    from "a trace with nothing in it".
    """
    from gitm._paths import traces_dir
    from gitm.serve.artifacts import print_result, write_capture_artifacts, write_preflight

    serve_argv = list(serve_argv) if serve_argv else list(DEFAULT_SERVE_ARGV)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else traces_dir() / f"vllm-serve-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"

    # Both variables must be set BEFORE the server starts CUDA — the driver reads
    # CUDA_INJECTION64_PATH once, at CUDA init, and it is inherited by every child.
    # This process needs them too: injection.active() is how capture() knows to merge
    # the children's shards instead of collecting in-process.
    from gitm.tracer import injection

    devices = visible_devices()
    tp = resolve_tp(serve_argv, devices, args.tp)
    # Pin it into the command now, before preflight validates the flags, so what gets
    # checked is exactly what gets launched.
    serve_argv = list(serve_argv)
    if not (_arg_of(serve_argv, "--tensor-parallel-size") or _arg_of(serve_argv, "-tp")):
        serve_argv += ["--tensor-parallel-size", str(tp)]

    print(f"==> out dir: {out_dir}")
    print(f"==> {devices.count} GPU(s) via {devices.source} -> "
          f"TP={tp} x DP={resolve_dp(serve_argv)}")
    print("==> preflight")
    checks = [] if args.skip_preflight else preflight(serve_argv, devices, tp, skip_args=False)
    print_checks(checks)
    write_preflight(out_dir, checks)
    if any(c.status == "fail" for c in checks):
        print("\npreflight failed — not starting the server. See the FAILs above.")
        return 2, None
    if args.dry_run:
        print("\ndry run — preflight only, nothing launched.")
        return 0, None

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

        before_text = snapshot_metrics(base, out_dir / "metrics_before.txt")

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

        after_text = snapshot_metrics(base, out_dir / "metrics_after.txt")
    finally:
        if args.keep_server:
            print(f"==> leaving the server up (pid {proc.pid}) — kill it with: "
                  f"kill -INT -{proc.pid}")
        else:
            print("==> shutting the server down")
            shutdown(proc)
        log_fh.close()

    # ---- artifacts
    from gitm.serve.metrics import window_from_snapshots
    from gitm.tracer.vllm_stats import summarize_requests

    summary = summarize_requests(records)
    server_window = window_from_snapshots(before_text, after_text, window_s=wall)
    result = write_capture_artifacts(
        out_dir,
        trace=trace,
        trace_path=trace_path,
        checks=checks,
        had_traffic=bool(records),
        serving_summary={
            "mode": "drive",
            "wall_s": wall,
            # Client-side wall clock, NOT vLLM's RequestOutput.metrics: the server path
            # never hands those to an HTTP client. It therefore includes network and
            # this client's own scheduling. The `server` block below is vLLM's own
            # histograms over the same window; prefer those when the two disagree.
            "client": {
                "latency_source": "client",
                "n_failed_requests": failures,
                **asdict(summary),
            },
            "server": server_window.to_dict(),
        },
        manifest={
            "workload_id": WORKLOAD_ID,
            "capture_mode": "serve",
            "window": "drive",
            "serve_argv": cmd,
            "served_model": model,
            "base_url": base,
            "load": {
                "requests": args.requests, "concurrency": args.concurrency,
                "input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
                "ignore_eos": not args.no_ignore_eos, "seed": args.seed,
            },
        },
    )

    print(f"\n==> {len(records)} ok / {failures} failed in {wall:.1f}s")
    if summary.ttft_p50_s is not None:
        tpot = f"{summary.tpot_p50_s * 1e3:.1f} ms" if summary.tpot_p50_s is not None else "n/a"
        print(f"    TTFT p50/p95 {summary.ttft_p50_s * 1e3:.0f}/{summary.ttft_p95_s * 1e3:.0f} ms"
              f"   TPOT p50 {tpot}")
    for warning in summary.warnings:
        print(f"    WARN: {warning}")
    print_result(result)

    if result.status == "no_kernels":
        return 1, result
    if result.status == "no_traffic":
        print("\nEvery request failed — see server.log.")
        return 1, result
    return 0, result


if __name__ == "__main__":
    raise SystemExit(main())
