from __future__ import annotations

from gitm.optimizer.headroom_kernel_rank import gpu_headroom


def test_memory_only_telemetry_does_not_fabricate_compute_headroom():
    result = gpu_headroom(
        [{"mem_used_bytes": 4_000, "mem_total_bytes": 10_000}]
    )

    assert result.compute_headroom_pct is None
    assert result.mean_util_pct is None
    assert result.mem_free_at_peak_bytes == 6_000
    assert any("utilization telemetry is absent" in note for note in result.diagnostics)


def test_utilization_only_telemetry_does_not_fabricate_memory_headroom():
    result = gpu_headroom([{"util_pct": 65.0}])

    assert result.compute_headroom_pct == 35.0
    assert result.mem_free_at_peak_bytes is None
    assert result.mem_total_bytes is None
    assert any("memory headroom unavailable" in note for note in result.diagnostics)


def test_complete_headroom_sample_stays_clean():
    result = gpu_headroom(
        [{"util_pct": 65.0, "mem_used_bytes": 4_000, "mem_total_bytes": 10_000}]
    )

    assert result.compute_headroom_pct == 35.0
    assert result.mem_free_at_peak_bytes == 6_000
    assert result.diagnostics == []


def test_empty_headroom_input_is_explicitly_unavailable():
    result = gpu_headroom([])

    assert result.compute_headroom_pct is None
    assert result.mem_free_at_peak_bytes is None
    assert any("no samples" in note for note in result.diagnostics)
