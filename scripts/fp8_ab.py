"""Direct fp8-KV A/B for vllm-decode — the structural question, no optimizer loop.

    python scripts/fp8_ab.py

Builds one engine, measures bf16-KV decode throughput, rebuilds with
``kv_cache_dtype=fp8``, measures again, prints the delta. No scheduler candidates,
so nothing in vLLM's scheduler can deadlock the run (unlike the full loop).

Run as a FILE, never ``python -c`` or a stdin heredoc. Once CUDA is initialized in
the parent, vLLM builds the second (fp8) engine under the ``spawn`` start method,
and spawn re-imports ``__main__`` in the child — which only works if ``__main__``
is a real importable file guarded by ``if __name__ == "__main__"``. gitm also keeps
the parent CUDA-free so ``fork`` is used instead, but the guard is the portable
guarantee.

Reads the same GITM_VLLM_* environment as the workload factory. For a meaningful
result the KV cache must be saturated — set, before running:

    export GITM_VLLM_GPU_MEM=0.45          # both engines coexist during the A/B
    export GITM_VLLM_PROMPTS=512
    export GITM_VLLM_MAX_TOKENS=2048       # ~8x oversubscription of a ~151k KV cache
    export GITM_VLLM_MODEL=NousResearch/Meta-Llama-3-8B

See docs/vllm_decode_runbook.md for the full procedure.
"""

from __future__ import annotations

from gitm.scheduler.loop import LoopConfig
from gitm.workloads import get_factory, set_decode_run_defaults


def main() -> None:
    # Sets injection + KV-pressure env in-process, before any CUDA init, so this
    # runs with zero manual exports. Anything you did export still wins.
    env = set_decode_run_defaults()
    print("run env:", {k: env[k] for k in ("GITM_VLLM_MODEL", "GITM_VLLM_GPU_MEM",
                                            "GITM_VLLM_PROMPTS", "GITM_VLLM_MAX_TOKENS")})

    run = get_factory("vllm-decode")(LoopConfig(workload="vllm-decode"))
    eng = run.engine

    base = eng.gitm_throughput_fn(eng)
    print(f"baseline bf16 KV: {base:,.0f} tok/s")

    fp8 = eng.gitm_restart_fn(eng, "kv_cache_dtype", "fp8")
    # The probe measures whatever engine it is handed, so either engine's bound
    # copy works; use the original's so this holds even if a rebuilt engine ever
    # lacks the hook.
    cand = eng.gitm_throughput_fn(fp8)
    print(f"fp8 KV:           {cand:,.0f} tok/s   ({(cand / base - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
