#!/usr/bin/env python3
"""Generate checked-in importer fixtures. Schema assumptions are explicit here.

Run from repo root:
    python tests/fixtures/importers/generate_fixtures.py
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Shared synthetic workload: ~50 kernels / 2 streams, 5 memcpys, 3 syncs, 2 devices.
# Timestamps in nanoseconds (absolute); importers re-base to t=0.


def _kernel_rows(n: int = 50) -> list[tuple]:
    rows = []
    t = 1_000_000  # start at 1ms absolute
    for i in range(n):
        device = 0 if i < 40 else 1
        stream = i % 2
        dur = 10_000 + (i % 5) * 1_000  # 10–14 µs
        # demangledName / shortName are StringIds references
        demangled = 100 + (i % 8)  # cycle a few names
        short = 200 + (i % 8)
        rows.append(
            (
                t,  # start
                t + dur,  # end
                device,
                stream,
                1000 + i,  # correlationId
                demangled,
                short,
                128 + (i % 4),  # gridX
                1,
                1,
                64,
                1,
                1,
                4096 if i % 3 == 0 else 0,  # staticSharedMemory
                1024 if i % 5 == 0 else 0,  # dynamicSharedMemory
                32 + (i % 4) * 8,  # registersPerThread
            )
        )
        t += dur + (2_000 if i % 7 == 0 else 500)  # small gaps; occasional longer
    return rows


def _memcpy_rows() -> list[tuple]:
    # 5 memcpys with various CUPTI memory kinds / copy kinds.
    # (start, end, deviceId, streamId, correlationId, bytes, srcKind, dstKind, copyKind)
    base = 1_000_000 + 50_000
    return [
        (base, base + 5_000, 0, 0, 9001, 1_000_000, 1, 3, 1),  # pageable→device HTOD
        (base + 10_000, base + 12_000, 0, 1, 9002, 500_000, 3, 2, 2),  # device→pinned DTOH
        (base + 20_000, base + 21_000, 0, 0, 9003, 256_000, 3, 3, 8),  # DTOD
        (base + 30_000, base + 31_000, 1, 0, 9004, 128_000, 5, 3, 1),  # managed→device
        (base + 40_000, base + 42_000, 0, 0, 9005, 64_000, None, None, 1),  # copyKind only
    ]


def _sync_rows() -> list[tuple]:
    # (start, end, deviceId, streamId, correlationId, syncType)
    base = 1_000_000 + 100_000
    return [
        (base, base + 3_000, 0, 0, 9101, 3),  # STREAM_SYNCHRONIZE
        (base + 20_000, base + 21_000, 0, 1, 9102, 1),  # EVENT_SYNCHRONIZE
        (base + 40_000, base + 45_000, 0, 0, 9103, 4),  # CONTEXT_SYNCHRONIZE → device
    ]


_STRINGS = {
    # demangled names 100–107
    100: "void cutlass::gemm::kernel::Gemm<float>(float*)",
    101: "ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_tn",
    102: "void at::native::vectorized_elementwise_kernel<4>(int, float*)",
    103: "flash_attn_kernels::flash_fwd_kernel",
    104: "void cub::DeviceReduce::Sum(int*, int*)",
    105: "triton_poi_fused_add_0",
    106: "void cublasLt::gemm_kernel(half*)",
    107: "void ncclDevKernel_AllReduce(void*)",
    # short names 200–207
    200: "Gemm",
    201: "ampere_gemm",
    202: "elementwise",
    203: "flash_fwd",
    204: "DeviceReduce",
    205: "triton_poi",
    206: "cublasLt_gemm",
    207: "ncclAllReduce",
}


def _create_nsys_sqlite(path: Path, version: str) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        for sid, val in _STRINGS.items():
            conn.execute("INSERT INTO StringIds(id, value) VALUES (?, ?)", (sid, val))

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
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _kernel_rows(),
        )

        conn.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                start INTEGER, end INTEGER,
                deviceId INTEGER, streamId INTEGER, correlationId INTEGER,
                bytes INTEGER, srcKind INTEGER, dstKind INTEGER, copyKind INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?,?,?)",
            _memcpy_rows(),
        )

        conn.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (
                start INTEGER, end INTEGER,
                deviceId INTEGER, streamId INTEGER, correlationId INTEGER,
                syncType INTEGER
            )
            """
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (?,?,?,?,?,?)",
            _sync_rows(),
        )

        conn.execute(
            """
            CREATE TABLE TARGET_INFO_GPU (
                id INTEGER, name TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO TARGET_INFO_GPU VALUES (0, ?), (1, ?)",
            ("NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-40GB"),
        )

        # Version stamp — both MetaData spelling and GITM helper table.
        conn.execute("CREATE TABLE MetaData (name TEXT, value TEXT)")
        conn.execute(
            "INSERT INTO MetaData VALUES ('ExportVersion', ?)",
            (version,),
        )
        conn.execute(
            "CREATE TABLE GITM_NSYS_META (version TEXT, session_start_ns INTEGER, deviceName TEXT, device_count INTEGER)"
        )
        conn.execute(
            "INSERT INTO GITM_NSYS_META VALUES (?, ?, ?, ?)",
            (version, 1_700_000_000_000_000_000, "NVIDIA A100-SXM4-40GB", 2),
        )
        conn.commit()
    finally:
        conn.close()


def _torch_events(*, with_grid: bool = True) -> list[dict]:
    """Chrome-trace events mirroring the same synthetic shape (device 0 only)."""
    events: list[dict] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "args": {"name": "NVIDIA A100-SXM4-40GB"},
        }
    ]
    t_us = 1000.0  # absolute µs
    names = [
        "cutlass_gemm",
        "ampere_fp16_s16816gemm",
        "vectorized_elementwise",
        "flash_fwd_kernel",
        "cub_DeviceReduce",
        "triton_poi_fused_add",
        "cublasLt_gemm",
        "ncclAllReduce",
    ]
    for i in range(40):  # device 0 only for torch default
        dur_us = 10.0 + (i % 5)
        args: dict = {
            "stream": i % 2,
            "device": 0,
            "correlation": 1000 + i,
        }
        if with_grid:
            args["grid"] = [128 + (i % 4), 1, 1]
            args["block"] = [64, 1, 1]
            args["shared memory"] = 4096 if i % 3 == 0 else 0
            args["registers per thread"] = 32
        events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": names[i % len(names)],
                "ts": t_us,
                "dur": dur_us,
                "pid": 1,
                "tid": i % 2,
                "args": args,
            }
        )
        t_us += dur_us + (2.0 if i % 7 == 0 else 0.5)

    # memcpys
    for j, (dur, nbytes, name) in enumerate(
        [
            (5.0, 1_000_000, "Memcpy HtoD"),
            (2.0, 500_000, "Memcpy DtoH"),
            (1.0, 256_000, "Memcpy DtoD"),
            (1.0, 128_000, "Memcpy HtoD"),
            (2.0, 64_000, "Memcpy HtoD"),
        ]
    ):
        events.append(
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": name,
                "ts": 1000.0 + 50.0 + j * 10.0,
                "dur": dur,
                "pid": 1,
                "tid": 0,
                "args": {"stream": 0, "device": 0, "bytes": nbytes},
            }
        )
    # A memset that must be skipped.
    events.append(
        {
            "ph": "X",
            "cat": "gpu_memset",
            "name": "Memset",
            "ts": 2000.0,
            "dur": 1.0,
            "pid": 1,
            "tid": 0,
            "args": {"device": 0},
        }
    )
    # CPU op that must be skipped.
    events.append(
        {
            "ph": "X",
            "cat": "cpu_op",
            "name": "aten::add",
            "ts": 0.0,
            "dur": 100.0,
            "pid": 0,
            "tid": 0,
            "args": {},
        }
    )
    return events


def _write_torch_object(path: Path, events: list[dict]) -> None:
    path.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ms"}, indent=2))


def _write_torch_array(path: Path, events: list[dict]) -> None:
    path.write_text(json.dumps(events, indent=2))


def _write_torch_gz(path: Path, events: list[dict]) -> None:
    payload = json.dumps({"traceEvents": events}).encode("utf-8")
    with gzip.open(path, "wb") as fh:
        fh.write(payload)


def _parity_events() -> list[tuple]:
    """Exact shared event set for the parity test (device 0 only, no syncs in torch).

    Returns list of (kind, start_ns, end_ns, stream, name_or_bytes, extra).
    """
    # Two non-overlapping 50us kernels + one 5us memcpy in the gap + wall 200us.
    # Matches the spirit of tests/test_metrics.py.
    return [
        ("kernel", 0, 50_000, 0, "gemm_a", {"grid": (128, 1, 1), "block": (64, 1, 1)}),
        ("kernel", 100_000, 150_000, 0, "gemm_b", {"grid": (128, 1, 1), "block": (64, 1, 1)}),
        ("memcpy", 60_000, 65_000, 0, 1_000_000, {"src": "host", "dst": "device"}),
        ("sync", 160_000, 170_000, 0, "stream", {}),
    ]


def _create_parity_sqlite(path: Path) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO StringIds VALUES (1, 'gemm_a'), (2, 'gemm_b')")
        conn.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                correlationId INTEGER, demangledName INTEGER, shortName INTEGER,
                gridX INTEGER, gridY INTEGER, gridZ INTEGER,
                blockX INTEGER, blockY INTEGER, blockZ INTEGER,
                staticSharedMemory INTEGER, dynamicSharedMemory INTEGER,
                registersPerThread INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0,50000,0,0,1,1,1,128,1,1,64,1,1,0,0,32)"
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100000,150000,0,0,2,2,2,128,1,1,64,1,1,0,0,32)"
        )
        conn.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                correlationId INTEGER, bytes INTEGER, srcKind INTEGER, dstKind INTEGER, copyKind INTEGER
            )
            """
        )
        # pageable(1) → device(3)
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (60000,65000,0,0,3,1000000,1,3,1)"
        )
        conn.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                correlationId INTEGER, syncType INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (160000,170000,0,0,4,3)"
        )
        # Zero-duration marker so wall duration is 200 µs after re-base.
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
            "(200000,200000,0,0,99,1,1,1,1,1,1,1,1,0,0,0)"
        )
        conn.execute("CREATE TABLE MetaData (name TEXT, value TEXT)")
        conn.execute("INSERT INTO MetaData VALUES ('ExportVersion', '2024.5.1')")
        conn.execute(
            "CREATE TABLE GITM_NSYS_META (version TEXT, session_start_ns INTEGER, deviceName TEXT, device_count INTEGER)"
        )
        conn.execute(
            "INSERT INTO GITM_NSYS_META VALUES ('2024.5.1', 0, 'NVIDIA H100', 1)"
        )
        conn.execute("CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA H100')")
        conn.commit()
    finally:
        conn.close()


def _create_parity_torch(path: Path) -> None:
    # Same timing as nsys/constructed minus sync events (torch format lacks them).
    # Wall = 200 µs via a zero-duration tail marker at ts=200.
    events = [
        {
            "ph": "X",
            "cat": "kernel",
            "name": "gemm_a",
            "ts": 0.0,
            "dur": 50.0,
            "args": {"stream": 0, "device": 0, "grid": [128, 1, 1], "block": [64, 1, 1]},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "gemm_b",
            "ts": 100.0,
            "dur": 50.0,
            "args": {"stream": 0, "device": 0, "grid": [128, 1, 1], "block": [64, 1, 1]},
        },
        {
            "ph": "X",
            "cat": "gpu_memcpy",
            "name": "Memcpy HtoD",
            "ts": 60.0,
            "dur": 5.0,
            "args": {"stream": 0, "device": 0, "bytes": 1_000_000},
        },
        {
            "ph": "X",
            "cat": "kernel",
            "name": "tail_pad",
            "ts": 200.0,
            "dur": 0.0,
            "args": {"stream": 0, "device": 0, "grid": [1, 1, 1], "block": [1, 1, 1]},
        },
    ]
    path.write_text(json.dumps({"traceEvents": events}))


def _create_4x_a100_nccl(path: Path) -> None:
    """Synthetic 4-device chrome-trace with interleaved NCCL kernels.

    Event structure mirrors real kineto resnet samples (cat Kernel/Memcpy,
    args keys device/stream/correlation/grid/block/shared memory/registers
    per thread/bytes). Marked synthetic in real/SOURCES.md — no multi-GPU
    sample ships in kineto v0.4.0.
    """
    events: list[dict] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": 1,
            "args": {"name": "NVIDIA A100-SXM4-40GB"},
        }
    ]
    # Per-device compute + collective phases. Device 3 is a straggler (more idle).
    # Timeline in microseconds (chrome units).
    for dev in range(4):
        t = 0.0
        # 8 compute kernels
        for i in range(8):
            dur = 40.0 if dev < 3 else 25.0  # straggler does less work → more idle later
            events.append(
                {
                    "ph": "X",
                    "cat": "Kernel",
                    "name": f"void cutlass::gemm::kernel::Gemm<float>(float*) [dev{dev}]",
                    "ts": t,
                    "dur": dur,
                    "pid": 0,
                    "tid": f"stream {dev}",
                    "args": {
                        "device": dev,
                        "stream": dev,
                        "correlation": 1000 + dev * 100 + i,
                        "external id": 1000 + dev * 100 + i,
                        "grid": [128, 1, 1],
                        "block": [64, 1, 1],
                        "shared memory": 4096,
                        "registers per thread": 64,
                    },
                }
            )
            t += dur + 2.0
        # NCCL AllReduce — partially exposed (no overlap) on all devices
        nccl_start = t + (5.0 if dev < 3 else 30.0)  # straggler starts late
        events.append(
            {
                "ph": "X",
                "cat": "Kernel",
                "name": "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevComm*, ...)",
                "ts": nccl_start,
                "dur": 50.0,
                "pid": 0,
                "tid": f"stream {dev}",
                "args": {
                    "device": dev,
                    "stream": 0,
                    "correlation": 9000 + dev,
                    "external id": 9000 + dev,
                    "grid": [1, 1, 1],
                    "block": [512, 1, 1],
                    "shared memory": 0,
                    "registers per thread": 32,
                },
            }
        )
        # Overlapped compute during second half of NCCL on devices 0-2 only
        if dev < 3:
            events.append(
                {
                    "ph": "X",
                    "cat": "Kernel",
                    "name": "void at::native::vectorized_elementwise_kernel<4>(int, float*)",
                    "ts": nccl_start + 20.0,
                    "dur": 25.0,
                    "pid": 0,
                    "tid": f"stream {dev + 4}",
                    "args": {
                        "device": dev,
                        "stream": dev + 4,
                        "correlation": 9100 + dev,
                        "grid": [64, 1, 1],
                        "block": [128, 1, 1],
                        "shared memory": 0,
                        "registers per thread": 32,
                    },
                }
            )
        # pad wall to 500 us
        events.append(
            {
                "ph": "X",
                "cat": "Kernel",
                "name": "tail_pad",
                "ts": 500.0,
                "dur": 0.0,
                "pid": 0,
                "tid": f"stream {dev}",
                "args": {
                    "device": dev,
                    "stream": 0,
                    "correlation": 9990 + dev,
                    "grid": [1, 1, 1],
                    "block": [1, 1, 1],
                },
            }
        )
        # one HtoD memcpy per device
        events.append(
            {
                "ph": "X",
                "cat": "Memcpy",
                "name": "Memcpy HtoD (Pageable -> Device)",
                "ts": 10.0 + dev,
                "dur": 3.0,
                "pid": 0,
                "tid": f"stream {dev}",
                "args": {
                    "device": dev,
                    "stream": dev,
                    "correlation": 8000 + dev,
                    "bytes": 1_000_000,
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0",
                "deviceProperties": [{"name": "NVIDIA A100-SXM4-40GB"} for _ in range(4)],
                "traceEvents": events,
            }
        )
    )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _create_nsys_sqlite(ROOT / "nsys_2024_min.sqlite", "2024.5.1")
    _create_nsys_sqlite(ROOT / "nsys_2025_min.sqlite", "2025.1.0")
    # Unsupported version fixture for error-path tests.
    _create_nsys_sqlite(ROOT / "nsys_2023_min.sqlite", "2023.4.1")

    events_grid = _torch_events(with_grid=True)
    events_nogrids = _torch_events(with_grid=False)
    _write_torch_object(ROOT / "torch_trace_min.json", events_grid)
    _write_torch_array(ROOT / "torch_trace_array.json", events_nogrids)
    _write_torch_gz(ROOT / "torch_trace_min.json.gz", events_grid)

    _create_parity_sqlite(ROOT / "parity_nsys.sqlite")
    _create_parity_torch(ROOT / "parity_torch.json")

    # 4-device synthetic (also mirrored under real/ for SOURCES.md)
    _create_4x_a100_nccl(ROOT / "synthetic_4xA100_nccl.json")
    real_dir = ROOT / "real"
    real_dir.mkdir(exist_ok=True)
    _create_4x_a100_nccl(real_dir / "synthetic_4xA100_nccl.json")

    mixed = ROOT / "mixed_dump"
    mixed.mkdir(exist_ok=True)
    # Copy essentials into mixed_dump.
    import shutil

    for name in (
        "nsys_2024_min.sqlite",
        "torch_trace_min.json",
        "torch_trace_min.json.gz",
    ):
        shutil.copy(ROOT / name, mixed / name)
    (mixed / "junk.txt").write_text("this is not a profiler file\n")
    (mixed / "notes.md").write_text("# notes\n")

    # Corrupt JSON for error paths.
    (ROOT / "corrupt.json").write_text("{traceEvents: not-json")

    print(f"wrote fixtures under {ROOT}")


if __name__ == "__main__":
    main()
