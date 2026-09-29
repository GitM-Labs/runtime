"""Audit a pinned safetensors index from shard headers without reading payloads.

Example (HTTP Range reads only header bytes):
  python scripts/audit_safetensors_headers.py \
    --index model.safetensors.index.json \
    --remote-base https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/<revision>/ \
    --expected gitm/planner/models/deepseek-v3.2.evidence.json

For a downloaded checkpoint, use --shards-dir instead of --remote-base.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

DTYPE_BYTES = {"F8_E4M3": 1, "BF16": 2, "F32": 4}


def _header(shard: str, shards_dir: Path | None, remote_base: str | None):
    if shards_dir is not None:
        with (shards_dir / shard).open("rb") as fh:
            prefix = fh.read(8)
            if len(prefix) != 8:
                raise ValueError(f"{shard}: short safetensors prefix")
            length = struct.unpack("<Q", prefix)[0]
            data = fh.read(length)
    else:
        url = remote_base.rstrip("/") + "/" + shard

        def get_range(start: int, end: int) -> bytes:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206:
                    raise ValueError(f"{shard}: server did not honor byte range")
                return response.read()

        prefix = get_range(0, 7)
        if len(prefix) != 8:
            raise ValueError(f"{shard}: short remote prefix")
        length = struct.unpack("<Q", prefix)[0]
        data = get_range(8, 7 + length)
    if len(data) != length:
        raise ValueError(f"{shard}: short header ({len(data)} != {length})")
    contents = json.loads(data)
    contents.pop("__metadata__", None)
    return shard, contents, {"shard": shard, "header_len": length,
                             "header_sha256": hashlib.sha256(data).hexdigest()}


def audit(index_path: Path, *, shards_dir: Path | None = None,
          remote_base: str | None = None, workers: int = 12) -> dict:
    if (shards_dir is None) == (remote_base is None):
        raise ValueError("choose exactly one of --shards-dir and --remote-base")
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda s: _header(s, shards_dir, remote_base), shards))
    classes: dict[str, dict[str, int]] = {}
    components: dict[str, dict[str, dict[str, int]]] = {}
    seen: set[str] = set()
    for shard, contents, _ in results:
        for name, tensor in contents.items():
            if name in seen or weight_map.get(name) != shard:
                raise ValueError(f"{name}: duplicate or index/shard mismatch")
            seen.add(name)
            count = 1
            for dim in tensor["shape"]:
                count *= dim
            dtype = tensor["dtype"]
            if dtype not in DTYPE_BYTES:
                raise ValueError(f"{name}: unhandled dtype {dtype}")
            size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
            if size != count * DTYPE_BYTES[dtype]:
                raise ValueError(f"{name}: shape/dtype and offset bytes disagree")
            kind = "scale" if name.endswith("weight_scale_inv") else (
                "weight" if name.endswith(".weight") else "other")
            row = classes.setdefault(f"{kind}/{dtype}", {"tensors": 0, "numel": 0, "bytes": 0})
            row["tensors"] += 1
            row["numel"] += count
            row["bytes"] += size
            if name == "model.embed_tokens.weight":
                component = "embedding"
            elif name == "lm_head.weight":
                component = "lm_head"
            elif ".self_attn.indexer." in name:
                component = "indexer"
            elif ".self_attn." in name:
                component = "attention"
            elif ".mlp.experts." in name:
                component = "routed_experts"
            elif ".mlp.shared_experts." in name:
                component = "shared_experts"
            elif ".mlp.gate.weight" in name:
                component = "router"
            elif any(s in name for s in (".mlp.gate_proj.", ".mlp.up_proj.",
                                         ".mlp.down_proj.")):
                component = "dense_ffn"
            else:
                component = "aux_norm_other"
            if name.startswith("model.layers.61."):
                component = "mtp/" + component
            part = components.setdefault(component, {}).setdefault(
                f"{kind}/{dtype}", {"tensors": 0, "numel": 0, "bytes": 0})
            part["tensors"] += 1
            part["numel"] += count
            part["bytes"] += size
    if seen != set(weight_map):
        raise ValueError(f"index has {len(set(weight_map) - seen)} tensors missing from headers")
    return {
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "index_total_size": index["metadata"]["total_size"],
        "tensors": len(seen), "shards": len(shards),
        "total_tensor_elements": sum(row["numel"] for row in classes.values()),
        "payload_bytes": sum(row["bytes"] for row in classes.values()),
        "classes": classes,
        "components": components,
        "shard_headers": [meta for _, _, meta in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shards-dir", type=Path)
    source.add_argument("--remote-base")
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    result = audit(args.index, shards_dir=args.shards_dir,
                   remote_base=args.remote_base, workers=args.workers)
    if args.expected:
        expected = json.loads(args.expected.read_text())
        for key in ("index_sha256", "index_total_size", "tensors", "shards",
                    "total_tensor_elements", "payload_bytes", "classes",
                    "components", "shard_headers"):
            if result[key] != expected[key]:
                raise SystemExit(f"audit mismatch: {key}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
