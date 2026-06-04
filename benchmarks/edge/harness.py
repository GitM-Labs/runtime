"""Edge/robotics baseline harness — 3D perception via OpenPCDet CenterPoint.

The work unit is one keyframe through
``voxelization -> 3D backbone -> BEV head -> NMS -> mAP accumulation``, run with
OpenPCDet's CenterPoint config on a PointPillars backbone (pinned commit +
config hash). The metric is ``frames_per_second`` over a warm window of frames
drawn from the combined ``manifest.jsonl`` (nuScenes + KITTI).

As with biotech, there is no CPU equivalent — OpenPCDet *is* the workload — so
this is framework-integration code for a GPU box. The per-frame inference sits
behind a ``Runner`` seam (``infer(frame) -> {...}``) so the scaffolding
(manifest iteration, warm-window timing, contract emission) is testable here
with an injected fake; the real runner is built by :func:`load_openpcdet_runner`.

Prints the one-line harness contract: ``metric_value`` = frames/sec, plus device
info and an optional mean-mAP auxiliary (regression sentinel, not a target).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

OPENPCDET_COMMIT = "v0.6.0"  # pinned; config hash pinned in datasets.md


class Frame(dict):
    """One manifest.jsonl row: scene_id, frame_id, lidar_path, gt_path, source."""


class Runner(Protocol):
    name: str

    def infer(self, frame: Frame, stage: Path) -> dict: ...


def load_openpcdet_runner(*, config_hash: str | None = None):
    """Build the real OpenPCDet CenterPoint runner. GPU-only."""
    try:
        import pcdet  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - framework absent on laptop
        raise RuntimeError(
            "OpenPCDet/torch not importable — the edge harness runs on a GPU box "
            "with OpenPCDet installed (its CUDA ops compiled). The dataset + "
            "reproducibility loop is exercised via the CPU smoke harness instead."
        ) from exc
    raise NotImplementedError(  # pragma: no cover
        "Wire OpenPCDet CenterPoint(PointPillars) construction here: load the "
        "pinned config + checkpoint, return a Runner whose infer() voxelizes the "
        "frame's lidar, runs the backbone+BEV head+NMS, and returns detections."
    )


def iter_frames(manifest: Path, *, warm: int) -> Iterator[Frame]:
    """Yield up to ``warm`` keyframes from manifest.jsonl (file order)."""
    n = 0
    with open(manifest) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield Frame(json.loads(line))
            n += 1
            if n >= warm:
                return


def run(stage: Path, *, warm: int, runner: Runner) -> dict:
    """Run the warm window of frames through ``runner`` and return the payload."""
    manifest = stage / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"missing {manifest} — build it with `make keyframes` (gitm.bench edge-manifest)"
        )

    n = 0
    maps: list[float] = []
    t0 = time.perf_counter()
    for frame in iter_frames(manifest, warm=warm):
        result = runner.infer(frame, stage)
        n += 1
        if "map" in result:
            maps.append(float(result["map"]))
    elapsed = max(time.perf_counter() - t0, 1e-9)

    if n == 0:
        raise RuntimeError(f"no frames in {manifest}")

    return {
        "metric_value": n / elapsed,  # frames per second
        "n_frames": n,
        "mean_map": (sum(maps) / len(maps)) if maps else None,
        "harness_commit": f"openpcdet-{OPENPCDET_COMMIT}",
    }


def _device_info() -> tuple[str, int]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), torch.cuda.device_count()
    except Exception:
        pass
    return "cpu", 0


def main(argv: list[str] | None = None, *, runner: Runner | None = None) -> int:
    p = argparse.ArgumentParser(description="Edge perception harness (OpenPCDet).")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--warm-frames", type=int, default=5000)
    p.add_argument("--stage", type=Path, default=None)
    args, _ = p.parse_known_args(argv)

    stage = args.stage or Path(os.environ.get("GITM_BENCH_STAGE", "."))
    runner = runner or load_openpcdet_runner()
    gpu_name, device_count = _device_info()

    payload = run(stage, warm=args.warm_frames, runner=runner)
    payload.update({"gpu_name": gpu_name, "device_count": device_count})

    print(f"[edge harness:{getattr(runner, 'name', '?')}] "
          f"{payload['n_frames']} frames, {payload['metric_value']:.1f} fps")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
