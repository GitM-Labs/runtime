"""Serial fp8-KV A/B for vllm-decode — one engine per process, full GPU budget.

    python scripts/fp8_ab.py            # runs both legs, prints the delta
    python scripts/fp8_ab.py bf16 OUT   # (internal) single leg, used by the driver

Each leg runs in its **own process** and builds exactly **one** engine, so only one
engine is ever resident on the GPU. That removes the coexisting-engines constraint
that forced ``GITM_VLLM_GPU_MEM=0.45`` — the old flow measured the bf16 baseline,
then called ``gitm_restart_fn`` to build the fp8 candidate alongside the still-live
baseline, so both had to fit at once (0.45 + 0.45 < 1.0). Serially, each engine gets
the whole budget: **0.75 by default here**, and 0.9+ works too.

Run as a FILE, never ``python -c`` or a stdin heredoc — a leg may build its engine
under the ``spawn`` start method, which re-imports ``__main__`` in the child.

Env (all optional; anything you export wins):

    GITM_VLLM_GPU_MEM     GPU memory fraction per engine (default 0.75 here)
    GITM_VLLM_MODEL       model (default: ungated Llama-3-8B mirror)
    GITM_VLLM_PROMPTS     number of prompts (default 512)
    GITM_VLLM_MAX_TOKENS  output tokens per prompt (default 2048)
    GITM_VLLM_PROFILE     "torch" -> also capture a PyTorch/kineto trace per leg

To sweep the memory budget, just run it once per level:

    for m in 0.45 0.75 0.95; do GITM_VLLM_GPU_MEM=$m python scripts/fp8_ab.py; done

See docs/vllm_decode_runbook.md for the full procedure.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

#: leg name -> vLLM ``kv_cache_dtype``. "auto" is bf16 for this model.
LEGS = {"bf16": "auto", "fp8": "fp8"}

#: A serial A/B keeps only one engine resident, so the 0.45 cap the coexisting
#: two-engine design needed no longer applies. Export GITM_VLLM_GPU_MEM to override.
DEFAULT_GPU_MEM = "0.75"
os.environ.setdefault("GITM_VLLM_GPU_MEM", DEFAULT_GPU_MEM)

PROFILE_TORCH = os.environ.get("GITM_VLLM_PROFILE", "").lower() == "torch"
PROFILE_DIR = os.environ.get("VLLM_TORCH_PROFILER_DIR", "/root/.cache/gitm/torchprof")


def _run_leg(leg: str) -> float:
    """Build ONE engine in this leg's KV dtype and return decode throughput (tok/s)."""
    # Imported here so the driver process never pulls in torch/vLLM.
    from gitm.scheduler.loop import LoopConfig
    from gitm.workloads import get_factory, set_decode_run_defaults

    os.environ["GITM_VLLM_KV_DTYPE"] = LEGS[leg]
    if PROFILE_TORCH:
        # Per-leg trace dir, created before the engine (hence the child) is built.
        d = Path(PROFILE_DIR) / leg
        d.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_TORCH_PROFILER_DIR"] = str(d)

    env = set_decode_run_defaults()
    if PROFILE_TORCH:
        # torch/kineto and the CUPTI injection tracer both want the single CUPTI
        # subscriber slot; disable injection so the profiler wins.
        os.environ.pop("CUDA_INJECTION64_PATH", None)

    print(
        f"[{leg}] kv_cache_dtype={LEGS[leg]} "
        f"gpu_mem={os.environ['GITM_VLLM_GPU_MEM']} "
        f"model={env['GITM_VLLM_MODEL']} "
        f"load={env['GITM_VLLM_PROMPTS']}x{env['GITM_VLLM_MAX_TOKENS']}"
    )

    run = get_factory("vllm-decode")(LoopConfig(workload="vllm-decode"))
    eng = run.engine
    if PROFILE_TORCH:
        eng.start_profile()
    tps = eng.gitm_throughput_fn(eng)
    if PROFILE_TORCH:
        eng.stop_profile()
    return tps


def main(argv: list[str]) -> None:
    # --- child mode: run one leg, hand the number back via a file ---
    if len(argv) == 2 and argv[0] in LEGS:
        leg, out_path = argv
        tps = _run_leg(leg)
        Path(out_path).write_text(repr(tps))
        print(f"[{leg}] {tps:,.0f} tok/s")
        return

    # --- driver: one process per leg, so only one engine is ever resident ---
    gpu_mem = os.environ["GITM_VLLM_GPU_MEM"]
    tmp = Path(tempfile.mkdtemp(prefix="gitm-fp8ab-"))
    results: dict[str, float] = {}
    for leg in LEGS:
        out = tmp / f"{leg}.txt"
        print(f"\n{'=' * 72}")
        print(f"=== leg: {leg}  (kv_cache_dtype={LEGS[leg]}, gpu_mem={gpu_mem})")
        print(f"{'=' * 72}", flush=True)
        rc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), leg, str(out)]
        ).returncode
        if rc != 0 or not out.exists():
            raise SystemExit(f"leg {leg!r} failed (exit {rc}) — see output above")
        results[leg] = float(out.read_text())

    base, cand = results["bf16"], results["fp8"]
    print(f"\n{'=' * 72}")
    print(f"gpu_memory_utilization = {gpu_mem}   (serial A/B — one engine at a time)")
    print(f"baseline bf16 KV: {base:,.0f} tok/s")
    print(f"fp8 KV:           {cand:,.0f} tok/s   ({(cand / base - 1) * 100:+.1f}%)")
    if PROFILE_TORCH:
        print(f"\ntraces under {PROFILE_DIR}/ (bf16/ and fp8/) -> https://ui.perfetto.dev")


if __name__ == "__main__":
    main(sys.argv[1:])