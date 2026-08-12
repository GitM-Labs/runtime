"""Tests for the CUPTI tracer path.

GPU-free: the native shim isn't built here, so we test (1) the pure decode layer
against synthetic record dicts, (2) the enum mappings, (3) graceful fallback
when the shim is absent, and (4) full backend wiring via a fake shim that
returns canned records — exactly the dict contract the C shim emits.
"""

from __future__ import annotations

import json

import pytest


def _kernel_rec(**over):
    rec = {
        "kind": "kernel", "name": "ampere_sgemm_128x64",
        "start_ns": 1000, "end_ns": 4000,
        "device_id": 0, "context_id": 1, "stream_id": 7, "correlation_id": 42,
        "grid": [128, 1, 1], "block": [64, 2, 1],
        "static_shared_mem": 2048, "dynamic_shared_mem": 1024,
        "registers_per_thread": 64,
    }
    rec.update(over)
    return rec


# --- decode: kernel ---------------------------------------------------------


def test_decode_kernel_fields():
    from gitm.tracer._cupti_decode import decode_kernel

    ev = decode_kernel(_kernel_rec())
    assert ev.kind == "kernel"
    assert ev.name == "ampere_sgemm_128x64"
    assert (ev.start_ns, ev.end_ns) == (1000, 4000)
    assert (ev.grid_x, ev.grid_y, ev.grid_z) == (128, 1, 1)
    assert (ev.block_x, ev.block_y, ev.block_z) == (64, 2, 1)
    assert ev.shared_mem_bytes == 2048 + 1024  # static + dynamic folded
    assert ev.registers_per_thread == 64
    assert ev.stream_id == 7
    assert ev.correlation_id == 42


def test_decode_kernel_anonymous_name():
    from gitm.tracer._cupti_decode import decode_kernel

    ev = decode_kernel(_kernel_rec(name=None))
    assert ev.name == "<anonymous>"


def test_decode_kernel_warns_when_shape_or_name_is_missing():
    from gitm.tracer._cupti_decode import decode_kernel

    with pytest.warns(RuntimeWarning, match="kernel grid dimensions unavailable"):
        decode_kernel(_kernel_rec(grid=None))
    with pytest.warns(RuntimeWarning, match="kernel name unavailable"):
        decode_kernel(_kernel_rec(name=None))


# --- decode: memcpy copy-kind mapping ---------------------------------------


@pytest.mark.parametrize(
    "copy_kind, src, dst",
    [
        (1, "host", "device"),    # HTOD
        (2, "device", "host"),    # DTOH
        (8, "device", "device"),  # DTOD
        (9, "host", "host"),      # HTOH
        (10, "device", "device"), # PTOP
        (0, "device", "device"),  # UNKNOWN default
    ],
)
def test_decode_memcpy_copy_kinds(copy_kind, src, dst):
    from gitm.tracer._cupti_decode import decode_memcpy

    ev = decode_memcpy({
        "kind": "memcpy", "copy_kind": copy_kind, "bytes": 1 << 20,
        "start_ns": 10, "end_ns": 20, "device_id": 0, "context_id": 1,
        "stream_id": 3, "correlation_id": 5,
    })
    assert ev.kind == "memcpy"
    assert ev.bytes == 1 << 20
    assert (ev.src, ev.dst) == (src, dst)


# --- decode: sync type mapping ----------------------------------------------


@pytest.mark.parametrize(
    "sync_type, kind",
    [(1, "event"), (2, "stream"), (3, "stream"), (4, "device"), (0, "device")],
)
def test_decode_sync_types(sync_type, kind):
    from gitm.tracer._cupti_decode import decode_sync

    ev = decode_sync({
        "kind": "sync", "sync_type": sync_type,
        "start_ns": 5, "end_ns": 9, "device_id": 0, "context_id": 1,
        "stream_id": 2, "correlation_id": 0,
    })
    assert ev.sync_kind == kind


def test_decode_unknown_cupti_enums_warn_before_using_safe_endpoint_defaults():
    from gitm.tracer._cupti_decode import decode_memcpy, decode_sync

    memcpy = {
        "kind": "memcpy", "copy_kind": 99, "bytes": 1,
        "start_ns": 10, "end_ns": 20, "device_id": 0, "stream_id": 0,
    }
    with pytest.warns(RuntimeWarning, match="unknown CUPTI memcpy kind 99"):
        decode_memcpy(memcpy)

    sync = {"kind": "sync", "sync_type": 99, "start_ns": 10, "end_ns": 20}
    with pytest.warns(RuntimeWarning, match="unknown CUPTI synchronization type 99"):
        decode_sync(sync)


# --- decode: batch ----------------------------------------------------------


def test_decode_records_sorts_and_drops_unknown():
    from gitm.tracer._cupti_decode import decode_records

    records = [
        _kernel_rec(start_ns=3000),
        {"kind": "memcpy", "copy_kind": 1, "bytes": 4, "start_ns": 1000,
         "end_ns": 1100, "device_id": 0, "context_id": 0, "stream_id": 0,
         "correlation_id": 1},
        {"kind": "marker", "start_ns": 0},  # unmodeled -> dropped
        {"kind": "sync", "sync_type": 3, "start_ns": 2000, "end_ns": 2050,
         "device_id": 0, "context_id": 0, "stream_id": 0, "correlation_id": 2},
    ]
    events = decode_records(records)
    assert [e.kind for e in events] == ["memcpy", "sync", "kernel"]  # by start_ns
    assert all(e.start_ns <= n.start_ns for e, n in zip(events, events[1:], strict=False))


def test_decode_record_returns_none_for_unmodeled():
    from gitm.tracer._cupti_decode import decode_record

    assert decode_record({"kind": "driver", "start_ns": 0}) is None


# --- graceful fallback ------------------------------------------------------


def test_load_shim_and_available_agree():
    from gitm.tracer._cupti import available, load_shim

    # On CI/laptop the extension is absent (None); on a GPU box where it's been
    # built it's a module. Either way the two helpers must agree.
    assert (load_shim() is None) == (not available())


def test_cupti_backend_raises_without_shim(monkeypatch):
    import gitm.tracer.cupti as cupti_mod

    # Force the absent case so this passes regardless of whether the shim is
    # built on this box.
    monkeypatch.setattr(cupti_mod, "load_shim", lambda: None)
    with pytest.raises(RuntimeError, match="CUPTI shim not built"):
        cupti_mod.CuptiBackend()


def test_capture_falls_back_to_noop_trace(tmp_path, monkeypatch):
    import importlib

    # Force no backend so we test the no-op path even on a GPU box.
    capture_mod = importlib.import_module("gitm.tracer.capture")
    monkeypatch.setattr(capture_mod, "_backend", lambda: None)

    out = tmp_path / "t.jsonl"
    with capture_mod.capture(out, workload_id="cupti-fallback"):
        pass
    lines = out.read_text().strip().splitlines()
    header = json.loads(lines[0])["_header"]
    assert header["vendor"] == "none"  # no backend -> no-op
    assert header["device_count"] == 0


@pytest.mark.parametrize(
    "counter, message",
    [
        (lambda: 0, "reported 0 devices"),
        (lambda: (_ for _ in ()).throw(RuntimeError("driver denied")), "driver denied"),
    ],
)
def test_capture_warns_when_active_backend_device_count_is_unavailable(counter, message):
    import importlib
    from types import SimpleNamespace

    capture_mod = importlib.import_module("gitm.tracer.capture")
    backend = SimpleNamespace(device_count=counter)

    with pytest.warns(RuntimeWarning, match=message):
        count = capture_mod._device_count(backend, injected=False)

    assert count == 0


# --- full backend wiring via a fake shim ------------------------------------


class _FakeShim:
    """Mimics the C shim's dict contract without a GPU."""

    def __init__(self, records):
        self._records = records
        self.started = False

    def device_count(self):
        return 2

    def start(self):
        self.started = True

    def stop(self):
        return self._records


def test_cupti_backend_end_to_end_with_fake_shim(monkeypatch):
    import gitm.tracer.cupti as cupti_mod

    fake = _FakeShim([
        _kernel_rec(start_ns=2000, name="k1"),
        {"kind": "memcpy", "copy_kind": 2, "bytes": 512, "start_ns": 500,
         "end_ns": 800, "device_id": 0, "context_id": 0, "stream_id": 1,
         "correlation_id": 9},
    ])
    monkeypatch.setattr(cupti_mod, "load_shim", lambda: fake)

    backend = cupti_mod.CuptiBackend()
    assert backend.vendor == "nvidia"
    assert backend.device_count() == 2

    backend.start()
    assert fake.started
    events = backend.stop()
    assert [e.kind for e in events] == ["memcpy", "kernel"]  # sorted by start_ns
    assert events[0].bytes == 512 and events[0].src == "device" and events[0].dst == "host"
    assert events[1].name == "k1"


def test_cupti_backend_drives_capture_with_fake_shim(monkeypatch, tmp_path):
    """capture() wired to a present CUPTI backend writes real kernel events."""
    import importlib

    import gitm.tracer.cupti as cupti_mod

    # `gitm.tracer.capture` the function shadows the submodule as a package
    # attribute (re-exported in __init__), so reach the module via import_module.
    capture_mod = importlib.import_module("gitm.tracer.capture")

    fake = _FakeShim([_kernel_rec(name="hot_kernel")])
    monkeypatch.setattr(cupti_mod, "load_shim", lambda: fake)
    monkeypatch.setattr(capture_mod, "_backend", lambda: cupti_mod.CuptiBackend())

    out = tmp_path / "trace.jsonl"
    with capture_mod.capture(out, workload_id="real") as trace:
        pass
    assert len(trace.events) == 1
    assert trace.events[0].name == "hot_kernel"

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2  # header + 1 kernel event
    assert json.loads(lines[1])["name"] == "hot_kernel"
