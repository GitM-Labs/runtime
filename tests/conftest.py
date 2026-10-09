"""Shared pytest fixtures and tiny constructors for trace/kernel test data.

Kept deliberately small — reusable across the new roofline / tracer-jsonl /
fingerprint / report-snapshot test modules without pulling in the production
loop. Lives at the tests/ root so pytest picks it up automatically.
"""

from __future__ import annotations

import sys

import pytest

from gitm.tracer.schema import KernelEvent, MemcpyEvent, SyncEvent, Trace


@pytest.fixture
def without_vllm(monkeypatch):
    """vLLM is not importable for the duration of the test, wherever it runs.

    CI installs ``.[dev,bench]``, which has no vLLM, and the tests of the offline
    fallback were written against that box. Anywhere vLLM is installed — every
    machine that actually runs the loop — they failed, and the end-to-end one
    built a real engine on the GPU. A ``None`` entry in ``sys.modules`` makes
    ``import vllm`` (and any ``from vllm... import``) raise ``ImportError``, which
    is exactly what the fallback paths catch.
    """
    monkeypatch.setitem(sys.modules, "vllm", None)


def make_kernel(
    name: str = "k",
    *,
    start_ns: int = 0,
    end_ns: int = 100,
    stream_id: int = 0,
    device_id: int = 0,
    grid: tuple[int, int, int] = (1, 1, 1),
    block: tuple[int, int, int] = (1, 1, 1),
    correlation_id: int | None = None,
    bytes_read: int | None = None,
    bytes_written: int | None = None,
    shared_mem_bytes: int = 0,
    registers_per_thread: int = 0,
) -> KernelEvent:
    return KernelEvent(
        kind="kernel",
        name=name,
        start_ns=start_ns,
        end_ns=end_ns,
        stream_id=stream_id,
        device_id=device_id,
        grid_x=grid[0],
        grid_y=grid[1],
        grid_z=grid[2],
        block_x=block[0],
        block_y=block[1],
        block_z=block[2],
        shared_mem_bytes=shared_mem_bytes,
        registers_per_thread=registers_per_thread,
        correlation_id=correlation_id,
        bytes_read=bytes_read,
        bytes_written=bytes_written,
    )


def make_memcpy(
    *,
    bytes: int = 1024,
    src: str = "host",
    dst: str = "device",
    start_ns: int = 0,
    end_ns: int = 100,
    stream_id: int = 0,
    device_id: int = 0,
    correlation_id: int | None = None,
) -> MemcpyEvent:
    return MemcpyEvent(
        kind="memcpy",
        bytes=bytes,
        src=src,
        dst=dst,
        start_ns=start_ns,
        end_ns=end_ns,
        stream_id=stream_id,
        device_id=device_id,
        correlation_id=correlation_id,
    )


def make_sync(
    *,
    sync_kind: str = "stream",
    start_ns: int = 0,
    end_ns: int = 50,
    stream_id: int = 0,
    device_id: int = 0,
    correlation_id: int | None = None,
) -> SyncEvent:
    return SyncEvent(
        kind="sync",
        sync_kind=sync_kind,
        start_ns=start_ns,
        end_ns=end_ns,
        stream_id=stream_id,
        device_id=device_id,
        correlation_id=correlation_id,
    )


def make_trace(
    events: list | None = None,
    *,
    workload_id: str = "w",
    fingerprint: str = "f",
    run_id: str = "r",
    device_count: int = 1,
    vendor: str = "nvidia",
    captured_at_ns: int = 0,
    duration_ns: int = 1000,
    source: str = "cupti",
) -> Trace:
    return Trace(
        workload_id=workload_id,
        fingerprint=fingerprint,
        run_id=run_id,
        device_count=device_count,
        vendor=vendor,
        captured_at_ns=captured_at_ns,
        duration_ns=duration_ns,
        events=events or [],
        source=source,  # type: ignore[arg-type]
    )
