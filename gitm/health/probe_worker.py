"""One torchrun rank for the GPU health check."""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path


def main() -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=45))
    try:
        expected = world_size * (world_size + 1) / 2
        values = torch.full((256,), rank + 1.0, device=f"cuda:{local_rank}")
        dist.all_reduce(values)
        ok = torch.allclose(values, torch.full_like(values, expected))

        all_ok = torch.tensor(int(ok), device=values.device)
        dist.all_reduce(all_ok, op=dist.ReduceOp.MIN)
        if rank == 0:
            Path(sys.argv[1]).write_text(json.dumps({"ok": bool(all_ok.item())}))
        return 0 if all_ok.item() else 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
