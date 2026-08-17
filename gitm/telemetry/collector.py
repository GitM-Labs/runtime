"""Collector — polls vendor backends at a configurable interval and emits to sinks.

One daemon thread per node. Polling cost is ~microseconds per GPU per sample.
"""

from __future__ import annotations

import threading
import time
import warnings
from dataclasses import dataclass, field

from gitm.telemetry.backends import Backend, discover_backends
from gitm.telemetry.schema import WorkloadLabels
from gitm.telemetry.sinks import Sink


@dataclass
class CollectorConfig:
    interval_s: float = 1.0  # 1 Hz default
    labels: WorkloadLabels | None = None
    backends: list[Backend] | None = None  # if None, discover_backends()
    sinks: list[Sink] = field(default_factory=list)


class Collector:
    """Background sampler. Use as a context manager.

    Example:
        cfg = CollectorConfig(sinks=[JsonlSink("samples.jsonl")])
        with Collector(cfg):
            run_workload()
    """

    def __init__(self, cfg: CollectorConfig) -> None:
        self._cfg = cfg
        self.diagnostics: list[str] = []
        self._diagnostic_keys: set[str] = set()
        discovery_diagnostics: list[str] = []
        self._backends: list[Backend] = (
            cfg.backends
            if cfg.backends is not None
            else discover_backends(diagnostics=discovery_diagnostics)
        )
        for i, diagnostic in enumerate(discovery_diagnostics):
            self._record_failure(f"backend-discovery:{i}", diagnostic)
        if not self._backends and not discovery_diagnostics:
            self._record_failure("backend-discovery", "no live GPU telemetry backend found")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="gitm-telemetry")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        for s in self._cfg.sinks:
            try:
                s.close()
            except Exception as exc:
                self._record_failure(type(s).__name__, f"sink close failed: {exc}")
        for b in self._backends:
            try:
                b.close()
            except Exception as exc:
                self._record_failure(type(b).__name__, f"backend close failed: {exc}")

    def _record_failure(self, key: str, detail: str) -> None:
        """Record and warn once per failing component/path."""
        if key in self._diagnostic_keys:
            return
        self._diagnostic_keys.add(key)
        message = f"telemetry degraded [{key}]: {detail}"
        self.diagnostics.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def __enter__(self) -> Collector:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            for backend in self._backends:
                for idx in range(backend.device_count()):
                    try:
                        sample = backend.sample(idx, labels=self._cfg.labels)
                    except Exception as exc:
                        self._record_failure(
                            f"sample:{type(backend).__name__}:{idx}",
                            f"sample failed ({type(exc).__name__}: {exc})",
                        )
                        continue
                    for diagnostic in sample.diagnostics:
                        self._record_failure(
                            f"sample-field:{type(backend).__name__}:{idx}:{diagnostic}",
                            diagnostic,
                        )
                    for sink in self._cfg.sinks:
                        try:
                            sink.emit(sample)
                        except Exception as exc:
                            self._record_failure(
                                f"sink:{type(sink).__name__}",
                                f"emit failed ({type(exc).__name__}: {exc})",
                            )
                            continue
            next_tick += self._cfg.interval_s
            sleep_for = max(0.0, next_tick - time.monotonic())
            if self._stop.wait(timeout=sleep_for):
                return
