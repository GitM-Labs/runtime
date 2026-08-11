"""OTLP push sink — wired up behind the ``otlp`` extra.

Requires ``pip install gitm[otlp]``.
"""

from __future__ import annotations

from gitm.telemetry.schema import Sample


class OtlpSink:
    """Push samples as OTLP metric data points to an OTel collector."""

    def __init__(self, endpoint: str) -> None:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # type: ignore[import-not-found]
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        exporter = OTLPMetricExporter(endpoint=endpoint)
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
        self._provider = MeterProvider(metric_readers=[reader])
        meter = self._provider.get_meter("gitm.telemetry")
        self._util = meter.create_gauge("gitm.gpu.util_pct")
        self._mem = meter.create_gauge("gitm.gpu.mem_used_bytes")
        self._power = meter.create_gauge("gitm.gpu.power_w")
        self._temp = meter.create_gauge("gitm.gpu.temp_c")

    def _attrs(self, s: Sample) -> dict[str, str]:
        return {
            "node": s.node,
            "gpu_uuid": s.gpu_uuid,
            "gpu_index": str(s.gpu_index),
            "vendor": s.vendor,
            "workload_id": s.labels.workload_id if s.labels else "unknown",
        }

    def emit(self, sample: Sample) -> None:
        attrs = self._attrs(sample)
        if sample.util_pct is not None:
            self._util.set(sample.util_pct, attributes=attrs)
        if sample.mem_used_bytes is not None:
            self._mem.set(sample.mem_used_bytes, attributes=attrs)
        if sample.power_w is not None:
            self._power.set(sample.power_w, attributes=attrs)
        if sample.temp_c is not None:
            self._temp.set(sample.temp_c, attributes=attrs)

    def close(self) -> None:
        # Collector.stop owns the fail-open boundary and turns this into a named
        # diagnostic. Hiding it here would make dropped final exports invisible.
        self._provider.shutdown()
