import hashlib, os, yaml
from pathlib import Path

data_root = Path(os.environ["GITM_DATA_ROOT"])
seeds = [42, 43, 44]
files = []

for seed in seeds:
    p = data_root / f"datasets/hft/hft_1b_seed{seed}/part-00000.parquet"
    sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
    size   = p.stat().st_size
    files.append({
        "path": str(p),
        "seed": seed,
        "sha256": sha256,
        "bytes": size,
        "rows": 1_000_000_000,
    })
    print(f"seed {seed}: {size} bytes, sha256={sha256[:16]}...")

manifest = {"datasets": files}
out = Path("benchmarks/hft/manifest.yaml")
out.write_text(yaml.dump(manifest, default_flow_style=False))
print(f"\nManifest written to {out}")
