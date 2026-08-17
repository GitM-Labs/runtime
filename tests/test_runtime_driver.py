from __future__ import annotations

from contextlib import contextmanager

import pytest


@pytest.mark.parametrize(
    ("summary", "workload", "expected"),
    [({"events": 10}, "hft", 10), ({"frames": 3}, "edge", 3)],
)
def test_work_units_require_the_named_positive_counter(summary, workload, expected):
    from gitm.runtime_driver import _work_units

    assert _work_units(summary, workload) == expected


@pytest.mark.parametrize(
    ("summary", "workload"),
    [({}, "hft"), ({"frames": 0}, "edge"), ({"events": -1}, "hft")],
)
def test_work_units_refuse_missing_or_empty_coverage(summary, workload):
    from gitm.runtime_driver import _work_units

    with pytest.raises(RuntimeError, match="work coverage unavailable"):
        _work_units(summary, workload)


def test_runtime_driver_refuses_empty_trace_instead_of_printing_pass(tmp_path, monkeypatch, capsys):
    from gitm import runtime_driver
    from gitm.tracer.schema import Trace

    monkeypatch.setattr(
        runtime_driver,
        "_load_hft",
        lambda *_args, **_kwargs: (
            lambda: {"events": 10, "vwap_buckets": 1},
            10,
            "cpu",
            "cpu",
            0,
        ),
    )

    @contextmanager
    def empty_capture(*_args, **_kwargs):
        yield Trace(
            workload_id="hft",
            fingerprint="f",
            run_id="r",
            device_count=0,
            vendor="none",
            captured_at_ns=0,
            duration_ns=1,
            source="none",
            events=[],
        )

    monkeypatch.setattr("gitm.tracer.capture", empty_capture)

    rc = runtime_driver.main(
        ["--workload", "hft", "--stage", str(tmp_path), "--outdir", str(tmp_path)]
    )

    output = capsys.readouterr().out
    assert rc == 3
    assert "runtime details unavailable" in output
    assert "all details measured" not in output
    report = (tmp_path / "hft_seed42_report.md").read_text()
    assert "NO DATA" in report
