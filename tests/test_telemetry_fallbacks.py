from __future__ import annotations

import time

import pytest

from gitm.telemetry.collector import Collector, CollectorConfig


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
