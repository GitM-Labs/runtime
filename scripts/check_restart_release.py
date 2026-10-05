"""On-GPU check that restart-A/B engines actually release their GPUs.

    GITM_VLLM_TP=8 GITM_RESTART_MODE=parallel GITM_VLLM_GPU_MEM=0.4 python scripts/check_restart_release.py
    GITM_VLLM_TP=8 GITM_RESTART_MODE=serial                          python scripts/check_restart_release.py

Runs the real vllm-decode engine through LiveEngineApplicator with the outcome of
each A/B *forced* via min_keep_delta (the throughput is still really measured), so
every release path runs deterministically:

    1. forced KEEP     -> the replaced engine's processes must be gone (the leak fix)
    2. forced ROLLBACK -> the candidate's processes must be gone, and the survivor
                          must still decode (global teardown skipped while it lives)
    3. final shutdown  -> nothing of ours left on the GPUs

Prints rocm-smi after each step: VRAM should never exceed one engine's share in
serial mode, or two in parallel. Exits non-zero on any failed check or on a
"still running ... killed" warning (a worker outlived shutdown — the race the
fix guards against; it was handled, but you want to know it happened).

Run as a FILE (not python -c / heredoc): the pod builds engines under spawn.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import warnings

os.environ.setdefault("GITM_VLLM_PROMPTS", "8")
os.environ.setdefault("GITM_VLLM_MAX_TOKENS", "32")

KEEP, ROLLBACK = -1.0, 10.0  # min_keep_delta that every / no delta clears


def _spec(value: int):
    from gitm.kernels.spec import InterventionSpec

    return InterventionSpec.model_validate(dict(
        name=f"max_num_seqs={value}", summary="release check", knob="max_num_seqs",
        value=value, expected_delta_mean=0.0, expected_delta_lo=0.0,
        expected_delta_hi=0.0, source="check_restart_release"))


def _gpus(label: str) -> None:
    print(f"\n--- {label}", flush=True)
    smi = shutil.which("rocm-smi") or shutil.which("nvidia-smi")
    if smi and "rocm" in smi:
        subprocess.run([smi, "--showmemuse", "--showpids"])
    elif smi:
        subprocess.run([smi, "--query-gpu=index,memory.used", "--format=csv"])


def _alive(p) -> bool:
    import psutil

    try:
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _gone(engine, what: str) -> bool:
    import psutil

    procs = sorted(getattr(engine, "gitm_worker_pids", None) or ())
    alive = [pid for pid in procs if psutil.pid_exists(pid) and _alive(psutil.Process(pid))]
    ok = bool(procs) and not alive
    print(f"[{'PASS' if ok else 'FAIL'}] {what}: {len(procs)} processes tracked, alive={alive}")
    if not procs:
        print("       (none tracked: psutil missing, or vLLM ran the engine in-process)")
    return ok


def main() -> int:
    from gitm.optimizer.apply import LiveEngineApplicator, apply_intervention
    from gitm.scheduler.loop import LoopConfig
    from gitm.workloads import get_factory

    mode = os.environ.get("GITM_RESTART_MODE", "parallel")
    results: list[bool] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        base = get_factory("vllm-decode")(LoopConfig(workload="vllm-decode")).engine
        built: list = []

        def restart(old, knob_values):  # keep a handle on every candidate
            built.append(base.gitm_restart_fn(old, knob_values))
            return built[-1]

        app = LiveEngineApplicator(
            base, throughput_fn=base.gitm_throughput_fn, restart_fn=restart,
            baseline_restart_fn=base.gitm_baseline_restart_fn, restart_mode=mode)
        _gpus(f"baseline built (mode={mode})")

        res = apply_intervention(_spec(128), app, min_keep_delta=KEEP)
        results.append(res.applied and not res.rolled_back)
        print(f"[{'PASS' if results[-1] else 'FAIL'}] forced keep applied ({res.error or 'ok'})")
        results.append(_gone(base, "keep released the replaced baseline"))
        _gpus("after forced keep: expect ONE engine resident")

        res = apply_intervention(_spec(64), app, min_keep_delta=ROLLBACK)
        results.append(res.rolled_back)
        results.append(_gone(built[-1], "rollback released the candidate"))
        survivor_ok = app.engine.gitm_llm_kwargs.get("max_num_seqs") == 128
        tps = base.gitm_throughput_fn(app.engine)
        results.append(survivor_ok and tps > 0)
        print(f"[{'PASS' if results[-1] else 'FAIL'}] survivor is the kept config and "
              f"still decodes ({tps:,.0f} tok/s)")
        _gpus("after forced rollback: expect ONE engine resident")

        last = app.engine
        app._shutdown(last)
        results.append(_gone(last, "final shutdown released the last engine"))
        _gpus("after final shutdown: expect NO gitm processes")

    for w in caught:
        print(f"[WARN] {w.message}")
    killed = [w for w in caught if "killed" in str(w.message)]
    if killed:
        print("[FAIL] a worker outlived shutdown and had to be killed (see WARN above)")
    ok = all(results) and not killed
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'} (mode={mode})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
