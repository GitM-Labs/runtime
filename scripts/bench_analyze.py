#!/usr/bin/env python3
"""Benchmark ``gitm analyze`` on 5M-kernel synthetic inputs.

Generates:
  * chrome-trace JSON (torch/kineto shape)
  * nsys-style sqlite (CUPTI_ACTIVITY_KIND_KERNEL + StringIds)

Runs ``gitm analyze`` on each in a **fresh subprocess** (accurate peak RSS),
records wall time + peak RSS to evidence/perf.json.

Pass criteria: wall < 5 minutes AND peak RSS < 4 GiB, per format.

Usage:
    python scripts/bench_analyze.py [--kernels N] [--out evidence/perf.json]
    python scripts/bench_analyze.py --phase before   # label results as before
    python scripts/bench_analyze.py --phase after    # merge into after
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return int(usage.ru_maxrss)
    return int(usage.ru_maxrss) * 1024


def write_chrome_trace(path: Path, n_kernels: int) -> None:
    """Compact one-object-per-line chrome-trace JSON."""
    with path.open("w", encoding="utf-8") as fh:
        fh.write('{"schemaVersion":"1.0","traceEvents":[\n')
        t_us = 0.0
        for i in range(n_kernels):
            dev = i % 2
            is_nccl = (i % 200) == 0
            name = (
                "ncclDevKernel_AllReduce_Sum_f32_RING_LL"
                if is_nccl
                else f"cutlass_gemm_{i % 16}"
            )
            dur = 5.0 if not is_nccl else 20.0
            fh.write(
                f'{{"ph":"X","cat":"Kernel","name":"{name}",'
                f'"ts":{t_us:.3f},"dur":{dur:.3f},'
                f'"args":{{"device":{dev},"stream":{dev},'
                f'"correlation":{i},"grid":[128,1,1],"block":[64,1,1]}}}}'
            )
            fh.write(",\n" if i + 1 < n_kernels else "\n")
            t_us += dur + 0.5
        fh.write("]}\n")


def write_nsys_sqlite(path: Path, n_kernels: int) -> None:
    """Minimal nsys-export-shaped sqlite with n_kernels rows."""
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        names = [
            "ncclDevKernel_AllReduce_Sum_f32_RING_LL",
            *[f"cutlass_gemm_{i}" for i in range(16)],
        ]
        for i, n in enumerate(names):
            conn.execute("INSERT INTO StringIds(id, value) VALUES (?, ?)", (i, n))
        conn.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER,
                deviceId INTEGER, streamId INTEGER, correlationId INTEGER,
                demangledName INTEGER, shortName INTEGER,
                gridX INTEGER, gridY INTEGER, gridZ INTEGER,
                blockX INTEGER, blockY INTEGER, blockZ INTEGER,
                staticSharedMemory INTEGER, dynamicSharedMemory INTEGER,
                registersPerThread INTEGER
            )
            """
        )
        conn.execute(
            "CREATE TABLE MetaData (name TEXT, value TEXT)"
        )
        conn.execute(
            "INSERT INTO MetaData VALUES ('ExportVersion', '2024.5.1')"
        )
        conn.execute(
            "CREATE TABLE GITM_NSYS_META "
            "(version TEXT, session_start_ns INTEGER, deviceName TEXT, device_count INTEGER)"
        )
        conn.execute(
            "INSERT INTO GITM_NSYS_META VALUES ('2024.5.1', 0, 'NVIDIA A100', 2)"
        )
        # Batch inserts for speed.
        batch: list[tuple] = []
        t = 0
        batch_size = 50_000
        for i in range(n_kernels):
            dev = i % 2
            is_nccl = (i % 200) == 0
            name_id = 0 if is_nccl else 1 + (i % 16)
            dur = 20_000 if is_nccl else 5_000  # ns
            batch.append(
                (
                    t,
                    t + dur,
                    dev,
                    dev,
                    i,
                    name_id,
                    name_id,
                    128,
                    1,
                    1,
                    64,
                    1,
                    1,
                    0,
                    0,
                    32,
                )
            )
            t += dur + 500
            if len(batch) >= batch_size:
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    batch,
                )
                batch.clear()
        if batch:
            conn.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                batch,
            )
        conn.commit()
    finally:
        conn.close()


def _worker_payload() -> str:
    """Python source run in a child process to analyze one file."""
    return r"""
import json, resource, sys, time
from pathlib import Path

def peak_rss():
    u = resource.getrusage(resource.RUSAGE_SELF)
    return int(u.ru_maxrss) if sys.platform == "darwin" else int(u.ru_maxrss) * 1024

path = Path(sys.argv[1])
out = Path(sys.argv[2])
from gitm.importers.analyze import analyze_paths
t0 = time.perf_counter()
result = analyze_paths(
    [path],
    out=out,
    sku="NVIDIA A100",
    run_id="import-bench",
)
wall = time.perf_counter() - t0
rss = peak_rss()
n_dev = len(result.workloads[0].devices) if result.workloads else 0
n_k = sum(d.n_kernels for d in result.workloads[0].devices) if result.workloads else 0
print(json.dumps({
    "analyze_wall_s": round(wall, 3),
    "peak_rss_bytes": rss,
    "peak_rss_gib": round(rss / (1024 ** 3), 3),
    "n_workloads": result.summary.get("n_workloads", 0),
    "n_devices": n_dev,
    "n_kernels_imported": n_k,
    "passed": wall < float(sys.argv[3]) and rss < int(sys.argv[4]),
}))
"""


def run_analyze_subprocess(
    path: Path,
    *,
    report: Path,
    max_wall_s: float,
    max_rss_bytes: int,
) -> dict:
    """Fresh process → accurate peak RSS for one analyze run."""
    env = os.environ.copy()
    # Ensure package import works without editable install surprises.
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _worker_payload(),
            str(path),
            str(report),
            str(max_wall_s),
            str(max_rss_bytes),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        return {
            "analyze_wall_s": None,
            "peak_rss_bytes": None,
            "peak_rss_gib": None,
            "n_workloads": 0,
            "n_devices": 0,
            "n_kernels_imported": 0,
            "passed": False,
            "error": (proc.stderr or proc.stdout or f"exit {proc.returncode}")[-2000:],
            "returncode": proc.returncode,
        }
    # Last JSON line wins.
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {
            "analyze_wall_s": None,
            "peak_rss_bytes": None,
            "peak_rss_gib": None,
            "passed": False,
            "error": (proc.stderr or "no json output")[-2000:],
            "returncode": proc.returncode,
        }
    data = json.loads(lines[-1])
    data["returncode"] = proc.returncode
    if proc.stderr:
        data["stderr_tail"] = proc.stderr[-500:]
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernels", type=int, default=5_000_000)
    p.add_argument("--out", type=Path, default=Path("evidence/perf.json"))
    p.add_argument("--scratch", type=Path, default=None)
    p.add_argument("--max-wall-s", type=float, default=300.0)
    p.add_argument("--max-rss-bytes", type=int, default=4 * 1024**3)
    p.add_argument(
        "--phase",
        choices=("standalone", "before", "after"),
        default="standalone",
        help="standalone overwrites; before/after merge into that key",
    )
    p.add_argument(
        "--formats",
        default="json,sqlite",
        help="Comma list: json,sqlite",
    )
    args = p.parse_args(argv)

    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="gitm-bench-"))
    scratch.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    results: dict[str, dict] = {}
    all_passed = True

    if "json" in formats:
        jpath = scratch / f"bench_{args.kernels}.json"
        print(f"generating chrome-trace {args.kernels} kernels → {jpath}", flush=True)
        t0 = time.perf_counter()
        write_chrome_trace(jpath, args.kernels)
        gen_s = time.perf_counter() - t0
        size_b = jpath.stat().st_size
        print(f"  generated in {gen_s:.1f}s, {size_b / 1e6:.1f} MB", flush=True)
        print("  analyzing chrome-trace ...", flush=True)
        r = run_analyze_subprocess(
            jpath,
            report=scratch / "report_json.md",
            max_wall_s=args.max_wall_s,
            max_rss_bytes=args.max_rss_bytes,
        )
        r.update(
            {
                "format": "chrome-trace-json",
                "kernels": args.kernels,
                "input_bytes": size_b,
                "generate_wall_s": round(gen_s, 3),
                "max_wall_s": args.max_wall_s,
                "max_rss_bytes": args.max_rss_bytes,
                "platform": sys.platform,
                "python": sys.version.split()[0],
            }
        )
        # Recompute passed if worker omitted criteria.
        if r.get("analyze_wall_s") is not None and r.get("peak_rss_bytes") is not None:
            r["passed"] = (
                r["analyze_wall_s"] < args.max_wall_s
                and r["peak_rss_bytes"] < args.max_rss_bytes
            )
        results["chrome_trace_json"] = r
        all_passed = all_passed and bool(r.get("passed"))
        print(json.dumps(r, indent=2), flush=True)

    if "sqlite" in formats:
        spath = scratch / f"bench_{args.kernels}.sqlite"
        print(f"generating nsys sqlite {args.kernels} kernels → {spath}", flush=True)
        t0 = time.perf_counter()
        write_nsys_sqlite(spath, args.kernels)
        gen_s = time.perf_counter() - t0
        size_b = spath.stat().st_size
        print(f"  generated in {gen_s:.1f}s, {size_b / 1e6:.1f} MB", flush=True)
        print("  analyzing nsys sqlite ...", flush=True)
        r = run_analyze_subprocess(
            spath,
            report=scratch / "report_sqlite.md",
            max_wall_s=args.max_wall_s,
            max_rss_bytes=args.max_rss_bytes,
        )
        r.update(
            {
                "format": "nsys-sqlite",
                "kernels": args.kernels,
                "input_bytes": size_b,
                "generate_wall_s": round(gen_s, 3),
                "max_wall_s": args.max_wall_s,
                "max_rss_bytes": args.max_rss_bytes,
                "platform": sys.platform,
                "python": sys.version.split()[0],
            }
        )
        if r.get("analyze_wall_s") is not None and r.get("peak_rss_bytes") is not None:
            r["passed"] = (
                r["analyze_wall_s"] < args.max_wall_s
                and r["peak_rss_bytes"] < args.max_rss_bytes
            )
        results["nsys_sqlite"] = r
        all_passed = all_passed and bool(r.get("passed"))
        print(json.dumps(r, indent=2), flush=True)

    payload: dict
    if args.phase in ("before", "after"):
        existing: dict = {}
        if args.out.exists():
            try:
                existing = json.loads(args.out.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing[args.phase] = {
            "results": results,
            "all_passed": all_passed,
            "kernels": args.kernels,
            "max_wall_s": args.max_wall_s,
            "max_rss_bytes": args.max_rss_bytes,
        }
        # Convenience top-level for the done criteria.
        if args.phase == "after":
            existing["passed"] = all_passed
            existing["results"] = results
        payload = existing
    else:
        payload = {
            "results": results,
            "all_passed": all_passed,
            "passed": all_passed,
            "kernels": args.kernels,
            "max_wall_s": args.max_wall_s,
            "max_rss_bytes": args.max_rss_bytes,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, args.out)
    print(f"wrote {args.out}", flush=True)

    if not all_passed:
        print("FAIL: one or more formats missed wall/RSS targets", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
