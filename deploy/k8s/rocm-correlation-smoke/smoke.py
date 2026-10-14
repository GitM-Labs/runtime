"""MI355X smoke for the ROCm correlation port (PR #164) — checklist items 1-4
of docs/rocm_correlation.md "On hardware". Runs inside the Job pod.

vLLM 0.29 defaults to its V2 model runner, which accepts
--enable-layerwise-nvtx-tracing and never registers the hooks (they exist only
in v1/worker/gpu_model_runner.py). Every phase pins the V1 runner; without it
no range is pushed and items 2 and 4 fail on zero markers.

Two serving phases of a small model, each a fresh `vllm serve` under
run_env(nvtx=True) + --enable-layerwise-nvtx-tracing:

  eager   --enforce-eager                                   -> items 1, 2
  graphs  mode 0 (no compile) + FULL_DECODE_ONLY graphs     -> items 1, 3, 4

Each phase arms a capture window around a handful of requests, stops the
server, and decodes the shards with the same path read_shards uses, keeping
the CorrelationReport. Exit 0 only if every item passes.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from gitm.distributed.correlate import UNNAMED_SHARE
from gitm.tracer import injection
from gitm.tracer._cupti_decode import decode_records_with_report

MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ROOT = Path(os.environ.get("SMOKE_OUT", "/scratch"))
# Each phase on its own GPU and port, so a server that has not fully exited
# cannot starve or block the next one.
PHASES = {
    "eager": ("0", 8000, ["--enforce-eager"]),
    "graphs": ("1", 8001, ["--compilation-config",
                           json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})]),
}
PROMPTS = [f"Write {n} sentences about GPUs." for n in range(1, 9)]


def wait_healthy(proc: subprocess.Popen, port: int, timeout_s: float = 1800) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"vllm exited early with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise SystemExit("vllm never became healthy")


def complete(port: int, prompt: str) -> int:
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 64,
                       "temperature": 0.7}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["usage"]["completion_tokens"]


def load(out: Path) -> tuple[list[dict], list[dict]]:
    """(meta records, decodable records), filtered exactly as read_shards does."""
    metas, records = [], []
    for shard in sorted(out.parent.glob(out.name + ".*")):
        if shard.suffix == ".arm":
            continue
        pid = injection._shard_pid(shard)
        for line in shard.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            rec["pid"] = pid
            if rec.get("kind") == "meta":
                metas.append(rec)
            elif rec.get("kind") in injection.CORRELATION_KINDS or isinstance(rec.get("start_ns"), int):
                records.append(rec)
    return metas, records


def run_phase(name: str, gpu: str, port: int, extra: list[str]) -> tuple[list[dict], list[dict]]:
    out = ROOT / name / "trace.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **injection.run_env(out, nvtx=True, vendor="amd"),
           "HIP_VISIBLE_DEVICES": gpu, "VLLM_USE_V2_MODEL_RUNNER": "0"}
    print(f"\n=== phase {name}: env " + json.dumps(
        {k: env[k] for k in ("ROCP_TOOL_LIBRARIES", "GITM_TRACE_NVTX", "LD_PRELOAD",
                             "VLLM_WORKER_MULTIPROC_METHOD", "VLLM_USE_V2_MODEL_RUNNER")
         if k in env}), flush=True)
    log = open(ROOT / f"vllm_{name}.log", "w")
    cmd = ["vllm", "serve", MODEL, "--port", str(port), "--max-model-len", "4096",
           "--enable-layerwise-nvtx-tracing", *extra]
    print("  " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        wait_healthy(proc, port)
        os.environ[injection.ENV_OUT] = str(out)
        injection.arm()
        with cf.ThreadPoolExecutor(4) as pool:
            toks = list(pool.map(lambda p: complete(port, p), PROMPTS))
        injection.settle()
        time.sleep(2)
        injection.disarm()
        print(f"  {len(toks)} requests, {sum(toks)} tokens generated", flush=True)
    finally:
        # The whole group: EngineCore is a child of the API server.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        log.close()
    time.sleep(2)
    return load(out)


def main() -> int:
    results: dict[str, bool] = {}
    items = {"eager": ["2 eager identity=range_id, no disagreement"],
             "graphs": ["3 graphs: no duplicate_ordinal/untracked_launch, every exec has capture_id",
                        "4 graph_unnamed / graph_layerless not reported"]}
    for name, (gpu, port, extra) in PHASES.items():
        try:
            metas, records = run_phase(name, gpu, port, extra)
        except BaseException as exc:  # noqa: BLE001 - one phase must not hide the other
            print(f"  phase {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
            for k in [f"1 direct_dispatch==1 ({name})", *items[name]]:
                results[k] = False
            continue
        _, rep = decode_records_with_report(records)
        kinds = {}
        for r in records:
            kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
        print(f"  records by kind: {kinds}")
        print(f"  meta: {[{k: m.get(k) for k in ('collector', 'direct_dispatch', 'identity', 'dropped_records', 'graph_stamp_overflow', 'graph_untracked_launch')} for m in metas]}")
        print(f"  report: kernels={rep.kernels} identity={dict(rep.identity)} "
              f"graph_kernels={rep.graph_kernels} graph_refused={dict(rep.graph_refused)} "
              f"graph_unnamed={rep.graph_unnamed} graph_layerless={rep.graph_layerless} "
              f"stamp_containment_disagree={rep.stamp_containment_disagree} "
              f"stamp_unresolved={rep.stamp_unresolved} "
              f"memcpys={rep.memcpys} labelled={rep.memcpys_labelled}")
        for p in rep.problems():
            print(f"  problem: {p}")
        # Zero markers means nothing pushed a range: say why up front, or
        # items 2 and 4 read as identity bugs in the collector.
        names = [r["name"] for r in records if r.get("kind") == "marker" and r.get("name")]
        print(f"  markers: {len(names)} pushed; sample: {[n[:100] for n in names[:3]]}")
        if not names:
            print("  problem: no rocTX range reached the collector")
            server_log = (ROOT / f"vllm_{name}.log").read_text(errors="replace")
            if "Using V2 Model Runner" in server_log:
                print("  problem: vLLM ran its V2 model runner, which never registers "
                      "the layerwise NVTX hooks")

        dd = [m.get("direct_dispatch") for m in metas if "direct_dispatch" in m]
        results[f"1 direct_dispatch==1 ({name})"] = bool(dd) and all(v == 1 for v in dd)
        if name == "eager":
            results["2 eager identity=range_id, no disagreement"] = (
                rep.kernels > 0 and rep.identity.get("range_id", 0) > 0
                and rep.identity.get("containment", 0) == 0
                and rep.stamp_containment_disagree == 0)
        else:
            execs = [r for r in records if r.get("kind") == "graph_exec"]
            print(f"  graph_exec: {len(execs)} records, "
                  f"{sum(1 for r in execs if r.get('capture_id'))} with capture_id")
            results["3 graphs: no duplicate_ordinal/untracked_launch, every exec has capture_id"] = (
                rep.graph_kernels > 0 and bool(execs)
                and not rep.graph_refused.get("duplicate_ordinal")
                and not rep.graph_refused.get("untracked_launch")
                and all(r.get("capture_id") for r in execs))
            results["4 graph_unnamed / graph_layerless not reported"] = (
                rep.graph_kernels > 0
                and rep.graph_unnamed < UNNAMED_SHARE * rep.graph_kernels
                and rep.graph_layerless < UNNAMED_SHARE * rep.graph_kernels)

    print("\n=== checklist (docs/rocm_correlation.md, On hardware)")
    for k, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
