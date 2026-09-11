#!/usr/bin/env python3
"""Generate per-arm pod manifests for the parallel loop.

Each pod is the base kimi-loop pod with a unique name and its arm BAKED into
the initContainer's default arm.env, so it is born in its arm and never
reloads. Parallelism is across nodes, one node per pod; no two experiments
share a server, an arm, or a shard path.

The intervention pods (int-*) all run the `bench` role: the headline point
(rag c=64) under one configuration lever, directly comparable to the traced
baseline. Env-var levers (MoE backend, RCCL) ride as extra export lines in
arm.env, which the supervisor sources with `set -a` before launching vllm, so
they reach the server process. Shape levers (TP) sed the serve args and the
GPU request.

  python3 scripts/kimi_loop/gen_pods.py  ->  deploy/k8s/parallel/kimi-<slot>.yaml
"""

from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "deploy" / "k8s" / "mi355x-kimi-loop.yaml"
OUT = Path(__file__).resolve().parents[2] / "deploy" / "k8s" / "parallel"

TOOL = "/scratch/lib/libgitm_rocm_inject.so"


# slot -> dict(arm, rocp, nvtx, extra, preload, env=[...], tp=8, gpus=8)
def S(arm, extra="", *, rocp=TOOL, nvtx="0", preload="", env=(), tp=8, gpus=8):
    return dict(arm=arm, rocp=rocp, nvtx=nvtx, extra=extra, preload=preload,
               env=list(env), tp=tp, gpus=gpus)


SLOTS = {
    # ---- the original sweep/overhead/correlation fleet -----------------------
    "sweep-chat": S("A", rocp=""),
    "sweep-rag": S("A", rocp=""),
    "sweep-long": S("A", rocp=""),
    "traced": S("B"),
    "intervene": S("I", "--kv-cache-dtype fp8"),
    # ---- the nine measured interventions (bench role) ------------------------
    # Target: routed-expert GEMMs (dominant term)
    "int-ep": S("ep", "--enable-expert-parallel"),
    "int-moe-triton": S("moe-triton", env=["VLLM_ROCM_USE_AITER_MOE=0"]),
    # Target: MLA attention
    "int-mla-triton": S("mla-triton", "--attention-backend TRITON_MLA"),
    # Target: collectives
    "int-rccl-ring": S("rccl-ring", env=["NCCL_ALGO=Ring", "RCCL_DEBUG=INFO"]),
    # Full node required (National Compute forbids partial-node jobs), so it
    # holds 8 GPUs but drives only 4 — the TP-degree lever, 4 GPUs idle.
    "int-tp4": S("tp4", tp=4, gpus=8),
    # Target: launch-bound / host dispatch
    "int-eager": S("eager", "--enforce-eager"),
    # Target: operating point
    "int-kvhead": S("kvhead", "--gpu-memory-utilization 0.97"),
    "int-maxseqs": S("maxseqs", "--max-num-seqs 512"),
    "int-prefill": S("prefill", "--max-num-batched-tokens 16384"),
}

DEFAULT_ARM_BLOCK = """              cat > /scratch/arm.env <<'EOF'
              export GITM_ARM=B
              export ROCP_TOOL_LIBRARIES=/scratch/lib/libgitm_rocm_inject.so
              export GITM_TRACE_OUT=/scratch/trace/kimi.jsonl
              export GITM_TRACE_NVTX=0
              export GITM_EXTRA_VLLM_ARGS=
              export LD_PRELOAD=
              EOF"""


def arm_block(s) -> str:
    lines = [
        "              cat > /scratch/arm.env <<'EOF'",
        f"              export GITM_ARM={s['arm']}",
        f"              export ROCP_TOOL_LIBRARIES={s['rocp']}",
        "              export GITM_TRACE_OUT=/scratch/trace/kimi.jsonl",
        f"              export GITM_TRACE_NVTX={s['nvtx']}",
        f'              export GITM_EXTRA_VLLM_ARGS="{s["extra"]}"',
        f"              export LD_PRELOAD={s['preload']}",
    ]
    for e in s["env"]:
        lines.append(f"              export {e}")
    lines.append("              EOF")
    return "\n".join(lines)


def main() -> int:
    base = BASE.read_text()
    assert DEFAULT_ARM_BLOCK in base, "base manifest arm.env block changed; update generator"
    OUT.mkdir(parents=True, exist_ok=True)
    for slot, s in SLOTS.items():
        y = base
        y = y.replace("name: kimi-k25-loop", f"name: kimi-k25-{slot}")
        y = y.replace("app: kimi-loop", f"app: kimi-{slot}")
        y = y.replace("name: kimi-loop\n", f"name: kimi-{slot}\n")  # the Service
        y = y.replace(DEFAULT_ARM_BLOCK, arm_block(s))
        if s["tp"] != 8:
            y = y.replace("--tensor-parallel-size 8", f"--tensor-parallel-size {s['tp']}")
        if s["gpus"] != 8:
            y = y.replace("amd.com/gpu: 8", f"amd.com/gpu: {s['gpus']}")
        (OUT / f"kimi-{slot}.yaml").write_text(y)
        tag = f"arm {s['arm']}"
        if s["extra"]:
            tag += f" extra='{s['extra']}'"
        if s["env"]:
            tag += f" env={s['env']}"
        if s["tp"] != 8:
            tag += f" tp={s['tp']} gpus={s['gpus']}"
        print(f"kimi-{slot}.yaml: {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
