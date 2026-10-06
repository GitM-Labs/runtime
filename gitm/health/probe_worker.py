"""Guarded one-rank AllReduce probe entrypoint.

Launched as ``python -m gitm.health.probe_worker`` from the parent health
check so embedded ``optimize()`` callers never re-execute their ``__main__``
module (unlike ``multiprocessing`` spawn).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def run_rank(
    *,
    rank: int,
    world_size: int,
    nbytes: int,
    init_method: str,
    result_path: str,
    atol: float,
    rtol: float,
) -> int:
    """Run one NCCL/RCCL AllReduce rank; write rank-0 metrics JSON. Returns exit code."""
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        n_elem = nbytes // 4
        expected = float(world_size)
        buf = torch.ones(n_elem, device=f"cuda:{rank}", dtype=torch.float32)
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0

        # Whole-buffer check on every rank — a single index on rank 0 is not enough.
        local_ok = bool(
            torch.allclose(
                buf,
                torch.full_like(buf, expected),
                atol=atol,
                rtol=rtol,
            )
        )
        flag = torch.tensor(
            [1.0 if local_ok else 0.0], device=buf.device, dtype=torch.float32
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        all_ranks_ok = float(flag.item()) >= 1.0

        if rank == 0:
            payload = {
                "elapsed_s": elapsed_s,
                "observed_sum": float(buf.flatten()[0].item()),
                "observed_min": float(buf.min().item()),
                "observed_max": float(buf.max().item()),
                "expected_sum": expected,
                "all_ranks_ok": all_ranks_ok,
                "nbytes": nbytes,
            }
            Path(result_path).write_text(json.dumps(payload))
        return 0 if all_ranks_ok and local_ok else 2
    finally:
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="gitm collective health probe rank")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--nbytes", type=int, required=True)
    p.add_argument("--init-method", type=str, required=True)
    p.add_argument("--result-path", type=str, required=True)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-5)
    args = p.parse_args(argv)
    try:
        return run_rank(
            rank=args.rank,
            world_size=args.world_size,
            nbytes=args.nbytes,
            init_method=args.init_method,
            result_path=args.result_path,
            atol=args.atol,
            rtol=args.rtol,
        )
    except Exception as exc:
        print(f"probe_worker rank={args.rank} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
