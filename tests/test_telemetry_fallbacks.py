from __future__ import annotations

import threading
import time

import pytest

from gitm.telemetry.collector import Collector, CollectorConfig
from gitm.telemetry.schema import Sample


class _BrokenBackend:
    def device_count(self):
        return 1

    def sample(self, _index, labels=None):
        raise RuntimeError("nvml read failed")

    def close(self):
        return None


def test_collector_surfaces_background_sample_failure_once():
    collector = Collector(CollectorConfig(interval_s=0.001, backends=[_BrokenBackend()]))

    with pytest.warns(RuntimeWarning, match="sample failed"):
        collector.start()
        time.sleep(0.02)
        collector.stop()

    assert len([d for d in collector.diagnostics if "sample failed" in d]) == 1


def test_collector_names_missing_backend_instead_of_looking_idle():
    with pytest.warns(RuntimeWarning, match="no live GPU telemetry backend"):
        collector = Collector(CollectorConfig(backends=[]))

    assert collector.diagnostics


class _PartialBackend:
    def device_count(self):
        return 1

    def sample(self, _index, labels=None):
        return Sample(
            ts_ns=1,
            node="n",
            gpu_uuid="g",
            gpu_index=0,
            vendor="nvidia",
            diagnostics=["clock throttle reasons unavailable (NVMLError: denied)"],
        )

    def close(self):
        return None


def test_collector_surfaces_partial_sample_field_failure():
    collector = Collector(CollectorConfig(interval_s=0.001, backends=[_PartialBackend()]))

    with pytest.warns(RuntimeWarning, match="clock throttle reasons unavailable"):
        collector.start()
        time.sleep(0.01)
        collector.stop()

    assert any("clock throttle reasons unavailable" in d for d in collector.diagnostics)


class _BrokenClose:
    def flush(self):
        raise OSError("disk full")

    def close(self):
        raise AssertionError("flush failure should propagate first")


def test_jsonl_sink_close_propagates_for_collector_diagnostic():
    from gitm.telemetry.sinks.jsonl import JsonlSink

    sink = JsonlSink.__new__(JsonlSink)
    sink._lock = threading.Lock()
    sink._fh = _BrokenClose()

    with pytest.raises(OSError, match="disk full"):
        sink.close()


def test_otlp_sink_close_propagates_for_collector_diagnostic():
    from gitm.telemetry.sinks.otlp import OtlpSink

    sink = OtlpSink.__new__(OtlpSink)
    sink._provider = type(
        "BrokenProvider", (), {"shutdown": lambda self: (_ for _ in ()).throw(RuntimeError("export"))}
    )()

    with pytest.raises(RuntimeError, match="export"):
        sink.close()
