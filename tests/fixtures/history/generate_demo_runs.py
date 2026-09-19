#!/usr/bin/env python3
"""A runs/ directory to point `gitm history` at, without needing a GPU.

    python tests/fixtures/history/generate_demo_runs.py /tmp/demo
    gitm history --scratch /tmp/demo

Every folder is what a finished `gitm run` leaves behind, written through the
real build_record / write_verification path rather than hand-authored JSON — so
if the export format moves, this breaks instead of quietly producing a history
the reader can no longer parse. Levers are the measured MI355X interventions
named in scripts/kimi_loop/gen_pods.py.

It covers each case the reader distinguishes, since they are what the record is
for: a lever that won twice, one that won and then lost (reported conflicted
rather than averaged), the same lever measured twice inside ONE run (so runs and
a/b differ), one lever on two different boxes (which must not merge), a run
whose SKU is missing as it is on any ROCm box with GITM_GPU_SKU unset, a run
that crashed before writing its export, and an export that is truncated.

Nothing here is a pytest fixture; it writes only to the directory given.
"""
import shutil
import sys
import uuid
from pathlib import Path

from gitm.kernels.spec import InterventionSpec
from gitm.optimizer.apply import ApplyResult, EngineABResult
from gitm.optimizer.report import Provenance
from gitm.optimizer.verification_export import build_record, write_verification

# run ids are uuid4().hex, exactly as scheduler/loop.py mints them
R1, R2, R3, R4, R5, R6, R7 = (uuid.uuid4().hex for _ in range(7))

OUT = Path(sys.argv[1])
if OUT.exists():
    shutil.rmtree(OUT)
RUNS = OUT / "runs"
RUNS.mkdir(parents=True)


def run(run_id, gpu_sku, attempts, *, git_sha="abc1234"):
    records = []
    for name, knob, value, speedup, kept, significant in attempts:
        spec = InterventionSpec(
            name=name, summary=f"{name}", knob=knob, value=value,
            expected_delta_mean=0.08, expected_delta_lo=0.02, expected_delta_hi=0.14,
            source="https://docs.vllm.ai/en/latest/configuration/engine_args.html",
        )
        ab = EngineABResult(
            knob=knob, value=value, baseline_tps=538.8,
            candidate_tps=538.8 * speedup, speedup=speedup, kept=kept,
            via="restart", baseline_std=4.1, candidate_std=4.4, reps=3,
            significant=significant,
        )
        records.append(build_record(spec, ab, ApplyResult(True, not kept, speedup - 1.0)))
    prov = Provenance(workload_id="vllm-decode", fingerprint="kimi-k2.5-mi355x",
                      run_id=run_id, git_sha=git_sha, gitm_version="0.1.13",
                      started_at_ns=0, ended_at_ns=1)
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    write_verification(records, prov, d / "verification.json", gpu_sku=gpu_sku)
    return d


MI = "AMD Instinct MI355X"
H100 = "NVIDIA H100 80GB HBM3"

# Run 1 — EP is the big measured win on MI355X (+49%); eager is a clear loss.
run(R1, MI, [
    ("enable_expert_parallel", "enable_expert_parallel", True,  1.49, True,  True),
    ("enforce_eager",          "enforce_eager",          True,  0.91, False, True),
    ("max_num_seqs_512",       "max_num_seqs",           512,   1.02, True,  False),
])

# Run 2 — EP wins again; rccl ring loses; the SAME lever twice in ONE run,
# which is why runs and a/b are counted separately.
run(R2, MI, [
    ("enable_expert_parallel", "enable_expert_parallel", True,  1.44, True,  True),
    ("rccl_algo_ring",         "NCCL_ALGO",              "Ring", 0.97, False, True),
    ("kv_cache_dtype_fp8",     "kv_cache_dtype",         "fp8", 1.08, True,  True),
    ("kv_cache_dtype_fp8",     "kv_cache_dtype",         "fp8", 1.11, True,  True),
])

# Run 3 — kv_cache_dtype_fp8 LOSES here: the record now disagrees with itself.
run(R3, MI, [
    ("kv_cache_dtype_fp8",     "kv_cache_dtype",         "fp8", 0.94, False, True),
    ("mla_triton_backend",     "attention_backend", "TRITON_MLA", 1.06, True, True),
])

# Run 4 — a different box. Same lever name, and it must NOT merge with MI355X.
run(R4, H100, [
    ("enable_expert_parallel", "enable_expert_parallel", True,  1.03, True,  False),
    ("kv_cache_dtype_fp8",     "kv_cache_dtype",         "fp8", 1.19, True,  True),
])

# Run 5 — ROCm box with GITM_GPU_SKU unset: _query_nvml is pynvml-only, so the
# SKU arrives as None. This is the case render_history warns about.
run(R5, None, [
    ("tp4_degree",             "tensor_parallel_size",   4,     1.21, True,  True),
])

# Run 6 — crashed before the export was written (measurement-only run).
(RUNS / R6).mkdir()

# Run 7 — the export exists but is truncated.
bad = RUNS / R7
bad.mkdir()
(bad / "verification.json").write_text('{"results": [{"intervention_name": "kv_cac')

print(f"wrote {len(list(RUNS.iterdir()))} run folders under {RUNS}")
