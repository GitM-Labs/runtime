"""Tests for the shared benchmark systems layer (gitm.bench).

All GPU-free: the profiling subprocess orchestration is not exercised here, but
its pure parsers and composition are. Everything else (config, manifest,
baseline gates, edge manifest, results render, runner contract) runs fully.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BENCHMARKS = REPO / "benchmarks"


# --- config -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["hft", "biotech", "edge"])
def test_shipped_bench_tomls_parse(name: str):
    from gitm.bench.schema import BenchConfig

    cfg = BenchConfig.from_toml(BENCHMARKS / name / "bench.toml")
    assert cfg.name == name
    assert cfg.seeds and len(cfg.seeds) >= 1
    # expected_stall bands are ordered and in [0, 1]
    for band in (cfg.expected_stall.cpu, cfg.expected_stall.data_stall,
                 cfg.expected_stall.sync, cfg.expected_stall.gpu_active):
        assert 0.0 <= band.lo <= band.hi <= 1.0


def test_bench_config_rejects_unknown_keys(tmp_path: Path):
    from pydantic import ValidationError

    from gitm.bench.schema import BenchConfig

    bad = tmp_path / "bench.toml"
    bad.write_text(
        'name="x"\nvendor="nvidia"\nmetric="m"\nwarm_window_s=1\nseeds=[1]\n'
        'bogus_key=true\n'
        '[dataset]\nroot="x"\n'
        '[work_unit]\ncommand="true"\n'
        '[expected_stall]\ncpu={lo=0,hi=0.1}\ndata_stall={lo=0,hi=0.3}\n'
        'sync={lo=0,hi=0.2}\ngpu_active={lo=0,hi=0.8}\n'
    )
    with pytest.raises(ValidationError):
        BenchConfig.from_toml(bad)


# --- manifest ---------------------------------------------------------------


def _make_dataset(root: Path) -> None:
    (root / "sub").mkdir(parents=True)
    (root / "a.parquet").write_bytes(b"alpha-bytes")
    (root / "sub" / "b.parquet").write_bytes(b"beta-bytes-longer")


def test_manifest_build_and_verify_roundtrip(tmp_path: Path):
    from gitm.bench import manifest as m

    root = tmp_path / "hft_1b_seed42"
    _make_dataset(root)

    man = m.build_manifest(root, "hft")
    assert man["file_count"] == 2
    assert man["total_bytes"] == len(b"alpha-bytes") + len(b"beta-bytes-longer")
    # deterministic ordering by path
    assert [f["path"] for f in man["files"]] == ["a.parquet", "sub/b.parquet"]

    result = m.verify_manifest(man, root)
    assert result.ok, result.summary()
    assert result.checked == 2


def test_manifest_detects_mismatch_missing_and_extra(tmp_path: Path):
    from gitm.bench import manifest as m

    root = tmp_path / "ds"
    _make_dataset(root)
    man = m.build_manifest(root, "hft")

    # tamper: change content, delete a file, add an unexpected one
    (root / "a.parquet").write_bytes(b"DIFFERENT")
    (root / "sub" / "b.parquet").unlink()
    (root / "surprise.parquet").write_bytes(b"x")

    result = m.verify_manifest(man, root)
    assert not result.ok
    assert any("a.parquet" in s for s in result.mismatched)
    assert "sub/b.parquet" in result.missing
    assert "surprise.parquet" in result.extra


def test_manifest_write_load_and_digest(tmp_path: Path):
    from gitm.bench import manifest as m

    root = tmp_path / "ds"
    _make_dataset(root)
    out = tmp_path / "manifest.yaml"
    m.write_manifest(m.build_manifest(root, "hft"), out)

    loaded = m.load_manifest(out)
    assert loaded["benchmark"] == "hft"
    # digest is stable across reads
    assert m.manifest_digest(out) == m.manifest_digest(out)


# --- baseline gates ---------------------------------------------------------


def _run(seed: int, value: float, gpu: float = 0.7) -> object:
    from gitm.bench.schema import BaselineRun, StallPhase

    return BaselineRun(
        benchmark="hft",
        seed=seed,
        vendor="nvidia",
        metric="events_per_second",
        metric_value=value,
        warm_window_s=60,
        git_sha="abc1234",
        gitm_version="0.0.1",
        manifest_sha256="manifest123",
        gpu_name="A100",
        device_count=1,
        started_at_ns=1,
        ended_at_ns=2,
        stall_breakdown=[
            StallPhase(phase="all", cpu=0.03, data_stall=max(0.0, 1 - gpu - 0.03 - 0.05),
                       sync=0.05, gpu_active=gpu, throughput=value, wall_clock_s=60.0)
        ],
    )


def _hft_config():
    from gitm.bench.schema import BenchConfig

    return BenchConfig.from_toml(BENCHMARKS / "hft" / "bench.toml")


def test_baseline_passes_within_spread_and_target():
    from gitm.bench.baseline import aggregate

    cfg = _hft_config()
    runs = [_run(42, 25.0e6), _run(43, 25.2e6), _run(44, 25.1e6)]
    summary = aggregate(runs, cfg)
    assert summary.passed, summary.to_dict()
    assert summary.spread < 0.02
    assert abs(summary.recorded - 25.1e6) < 1e5


def test_baseline_fails_on_wide_spread():
    from gitm.bench.baseline import aggregate

    cfg = _hft_config()
    runs = [_run(42, 25.0e6), _run(43, 30.0e6), _run(44, 25.1e6)]
    summary = aggregate(runs, cfg)
    assert not summary.passed
    assert any(g.name == "spread" and not g.passed for g in summary.gates)


def test_baseline_fails_on_saturation():
    from gitm.bench.baseline import aggregate

    cfg = _hft_config()
    runs = [_run(42, 26e6, gpu=0.92), _run(43, 26e6, gpu=0.92), _run(44, 26e6, gpu=0.92)]
    summary = aggregate(runs, cfg)
    assert not summary.passed
    assert any(g.name == "saturation" and not g.passed for g in summary.gates)


def test_baseline_refuses_missing_gpu_activity_instead_of_substituting_zero():
    from gitm.bench.baseline import aggregate

    runs = [_run(42, 26e6), _run(43, 26e6), _run(44, 26e6)]
    for run in runs:
        run.stall_breakdown = []

    summary = aggregate(runs, _hft_config())

    assert summary.gpu_active_overall is None
    gate = next(g for g in summary.gates if g.name == "saturation")
    assert not gate.passed
    assert "coverage unavailable" in gate.detail


def test_baseline_refuses_cpu_or_unpinned_runs():
    from gitm.bench.baseline import aggregate

    runs = [_run(42, 26e6), _run(43, 26e6), _run(44, 26e6)]
    runs[0].gpu_name = "cpu"
    runs[1].manifest_sha256 = None

    summary = aggregate(runs, _hft_config())

    gate = next(g for g in summary.gates if g.name == "provenance")
    assert not gate.passed
    assert "GPU identity unavailable" in gate.detail
    assert "manifest digest unavailable" in gate.detail


def test_baseline_refuses_missing_or_reversed_timing_provenance():
    from gitm.bench.baseline import aggregate

    runs = [_run(42, 26e6), _run(43, 26e6), _run(44, 26e6)]
    runs[0].started_at_ns = 0
    runs[1].ended_at_ns = runs[1].started_at_ns

    summary = aggregate(runs, _hft_config())

    gate = next(g for g in summary.gates if g.name == "provenance")
    assert not gate.passed
    assert "start timestamp unavailable" in gate.detail
    assert "end timestamp does not follow start" in gate.detail


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_baseline_run_refuses_unusable_metric_values(value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _run(42, value)


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf")])
def test_bench_config_refuses_unusable_baseline_target(target):
    from pydantic import ValidationError

    cfg = _hft_config().model_dump()
    cfg["baseline_target"] = target
    with pytest.raises(ValidationError):
        type(_hft_config()).model_validate(cfg)


def test_baseline_fails_on_too_few_runs_and_below_target():
    from gitm.bench.baseline import aggregate

    cfg = _hft_config()
    summary = aggregate([_run(42, 10e6)], cfg)  # 1 run, below 25M target
    names = {g.name: g.passed for g in summary.gates}
    assert names["count"] is False
    assert names["target"] is False


# --- profile pure parsers ---------------------------------------------------


def test_nsys_csv_parser_sums_total_time():
    from gitm.bench.profile import gpu_busy_ns_from_nsys_csv

    csv = (
        "Time (%),Total Time (ns),Instances,Name\n"
        "60.0,\"1,000,000\",10,fooKernel\n"
        "40.0,500000,5,barKernel\n"
    )
    assert gpu_busy_ns_from_nsys_csv(csv) == 1_500_000


def test_rocprof_csv_parser_sums_duration():
    from gitm.bench.profile import gpu_busy_ns_from_rocprof_csv

    csv = "KernelName,Calls,TotalDurationNs\nfoo,10,800000\nbar,5,200000\n"
    assert gpu_busy_ns_from_rocprof_csv(csv) == 1_000_000


def test_build_breakdown_computes_data_stall_residual():
    from gitm.bench.profile import PhaseTiming, build_breakdown

    phases = [PhaseTiming(phase="ingest", wall_clock_s=100.0, gpu_busy_s=70.0,
                          sync_s=5.0, cpu_s=3.0, throughput=25e6)]
    rows = build_breakdown(phases)
    assert len(rows) == 1
    r = rows[0]
    assert r.gpu_active == pytest.approx(0.70)
    assert r.data_stall == pytest.approx(1 - 0.70 - 0.05 - 0.03)


def test_profiler_tools_detect_runs_without_error():
    from gitm.bench.profile import ProfilerTools

    tools = ProfilerTools.detect()
    # on a laptop these are typically all None; just confirm the call is clean
    assert hasattr(tools, "nsys")


def test_wrap_command_marks_missing_profiler():
    from gitm.bench.profile import ProfilerTools, wrap_command

    cfg = _hft_config()
    tools = ProfilerTools(nsys=None, rocprof=None, py_spy=None, sar=None)
    argv, bundle = wrap_command(cfg, ["echo", "hi"], "/tmp", tools=tools)
    assert argv == ["echo", "hi"]  # passes command through unwrapped
    assert "nsys" in bundle.missing


def test_profile_marks_failed_workload_command(tmp_path):
    import sys

    from gitm.bench.profile import ProfilerTools, run_profile

    bundle = run_profile(
        _hft_config(),
        [sys.executable, "-c", "raise SystemExit(7)"],
        tmp_path,
        tools=ProfilerTools(nsys=None, rocprof=None, py_spy=None, sar=None),
    )

    assert any("exit 7" in item for item in bundle.missing)
    assert not bundle.complete


def test_profile_launches_pyspy_and_records_its_artifact(monkeypatch, tmp_path):
    from gitm.bench import profile

    calls = []

    class Proc:
        def __init__(self, argv, **_kwargs):
            self.argv = argv
            self.pid = 4321
            self.returncode = None
            calls.append(argv)
            if argv[0] == "py-spy":
                output = Path(argv[argv.index("--output") + 1])
                output.write_text("<svg/>")

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(profile.subprocess, "Popen", Proc)
    bundle = profile.run_profile(
        _hft_config(),
        ["workload"],
        tmp_path,
        host_capture_s=1,
        tools=profile.ProfilerTools(nsys=None, rocprof=None, py_spy="py-spy", sar=None),
    )

    pyspy = next(call for call in calls if call[0] == "py-spy")
    assert pyspy[pyspy.index("--pid") + 1] == "4321"
    assert "--subprocesses" in pyspy
    assert bundle.host_pyspy == tmp_path / "host_flamegraph.svg"
    assert bundle.host_pyspy.exists()
    assert not any("py-spy" in item for item in bundle.missing)


def test_profile_surfaces_failed_pyspy_capture(monkeypatch, tmp_path):
    from gitm.bench import profile

    class Proc:
        def __init__(self, argv, **_kwargs):
            self.argv = argv
            self.pid = 4321
            self.returncode = None

        def wait(self, timeout=None):
            self.returncode = 9 if self.argv[0] == "py-spy" else 0
            return self.returncode

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(profile.subprocess, "Popen", Proc)
    bundle = profile.run_profile(
        _hft_config(),
        ["workload"],
        tmp_path,
        host_capture_s=1,
        tools=profile.ProfilerTools(nsys=None, rocprof=None, py_spy="py-spy", sar=None),
    )

    assert any("py-spy capture failed (exit 9)" in item for item in bundle.missing)
    assert not bundle.complete


def test_profile_cli_prints_every_bundle_artifact(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    from gitm.bench import cli
    from gitm.bench.profile import ProfileBundle

    bundle = ProfileBundle(
        out_dir=tmp_path,
        gpu_report=tmp_path / "gpu.nsys-rep",
        gpu_csv=tmp_path / "gpu.csv",
        host_pyspy=tmp_path / "host.svg",
        host_sar=tmp_path / "host.log",
    )
    monkeypatch.setattr("gitm.bench.schema.BenchConfig.from_toml", lambda _path: object())
    monkeypatch.setattr("gitm.bench.runner.build_command", lambda _cfg, _seed: ["workload"])
    monkeypatch.setattr("gitm.bench.profile.run_profile", lambda *_args, **_kwargs: bundle)

    rc = cli._cmd_profile(SimpleNamespace(config=tmp_path / "bench.toml", seed=1, out=tmp_path))
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["gpu_report"] == str(bundle.gpu_report)
    assert payload["gpu_csv"] == str(bundle.gpu_csv)
    assert payload["host_pyspy"] == str(bundle.host_pyspy)
    assert payload["host_sar"] == str(bundle.host_sar)


def test_breakdown_refuses_to_clamp_overlapping_timings():
    from gitm.bench.profile import PhaseTiming, build_breakdown

    with pytest.raises(ValueError, match="refusing to clamp"):
        build_breakdown(
            [PhaseTiming("overlap", wall_clock_s=1.0, gpu_busy_s=0.8, cpu_s=0.4)]
        )


# --- edge manifest ----------------------------------------------------------


def test_edge_manifest_builds_from_nuscenes_and_kitti(tmp_path: Path):
    from gitm.bench import edge_manifest as em

    edge = tmp_path / "edge"
    # minimal nuScenes metadata tables
    meta = edge / "nuscenes" / "v1.0-trainval"
    meta.mkdir(parents=True)
    (meta / "scene.json").write_text(json.dumps([{"token": "S1", "name": "scene-0001"}]))
    (meta / "sample.json").write_text(
        json.dumps([{"token": "SAMP1", "scene_token": "S1"}])
    )
    (meta / "sample_data.json").write_text(
        json.dumps([
            {"sample_token": "SAMP1", "is_key_frame": True,
             "filename": "samples/LIDAR_TOP/scene-0001__LIDAR_TOP__1.pcd.bin"},
            {"sample_token": "SAMP1", "is_key_frame": False,
             "filename": "sweeps/LIDAR_TOP/x.pcd.bin"},  # not a keyframe -> skipped
        ])
    )
    # minimal KITTI
    velo = edge / "kitti" / "training" / "velodyne"
    velo.mkdir(parents=True)
    (velo / "000000.bin").write_bytes(b"\x00")
    labels = edge / "kitti" / "training" / "label_2"
    labels.mkdir(parents=True)
    (labels / "000000.txt").write_text("Car 0 0 0\n")

    rows = em.build_manifest(edge)
    sources = sorted(r.source for r in rows)
    assert sources == ["kitti", "nuscenes"]

    nusc = next(r for r in rows if r.source == "nuscenes")
    assert nusc.scene_id == "scene-0001"
    assert nusc.lidar_path.startswith("samples/LIDAR_TOP/")
    assert nusc.gt_path == "SAMP1"

    kitti = next(r for r in rows if r.source == "kitti")
    assert kitti.frame_id == "000000"
    assert kitti.gt_path.endswith("label_2/000000.txt")

    out = tmp_path / "manifest.jsonl"
    em.write_manifest(rows, out)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])  # valid JSONL


def test_edge_manifest_skips_absent_datasets(tmp_path: Path):
    from gitm.bench import edge_manifest as em

    rows = em.build_manifest(tmp_path / "empty_edge")
    assert rows == []


# --- runner contract --------------------------------------------------------


def test_runner_parses_last_json_line():
    from gitm.bench.runner import _last_json_line

    stdout = (
        "loading shards...\n"
        '{"metric_value": 1.0}\n'  # earlier, overwritten
        "warm window done\n"
        '{"metric_value": 25300000.0, "gpu_name": "A100"}\n'
    )
    payload = _last_json_line(stdout)
    assert payload["metric_value"] == 25300000.0
    assert payload["gpu_name"] == "A100"


def test_runner_raises_without_metric_line():
    from gitm.bench.runner import _last_json_line

    with pytest.raises(ValueError):
        _last_json_line("no json here\n{\"other\": 1}\n")


def test_run_seed_end_to_end_with_echo_harness(tmp_path: Path, monkeypatch):
    """Drive run_seed against a fake harness that just echoes a metric line."""
    from gitm.bench.runner import run_seed, write_run
    from gitm.bench.schema import BenchConfig

    harness = tmp_path / "fake_harness.py"
    harness.write_text(
        "import json, sys\n"
        "print('warming up')\n"
        "print(json.dumps({'metric_value': 42.0, 'gpu_name': 'fake'}))\n"
    )
    cfg_path = tmp_path / "bench.toml"
    cfg_path.write_text(
        'name="hft"\nvendor="nvidia"\nmetric="events_per_second"\n'
        'warm_window_s=60\nseeds=[42]\n'
        '[dataset]\nroot="hft"\n'
        f'[work_unit]\ncommand="python {harness.as_posix()} --seed {{seed}}"\n'
        '[expected_stall]\ncpu={lo=0,hi=0.05}\ndata_stall={lo=0.1,hi=0.25}\n'
        'sync={lo=0.05,hi=0.15}\ngpu_active={lo=0.6,hi=0.8}\n'
    )
    cfg = BenchConfig.from_toml(cfg_path)
    run = run_seed(cfg, 42, config_dir=tmp_path)
    assert run.metric_value == 42.0
    assert run.gpu_name == "fake"
    assert run.git_sha  # provenance populated

    out = write_run(run, tmp_path / "out.json")
    reloaded = json.loads(out.read_text())
    assert reloaded["metric_value"] == 42.0


# --- results render ---------------------------------------------------------


def test_results_render_contains_table_and_gate_verdict():
    from gitm.bench.baseline import aggregate
    from gitm.bench.results import render_results, representative_breakdown

    cfg = _hft_config()
    runs = [_run(42, 25.0e6), _run(43, 25.2e6), _run(44, 25.1e6)]
    summary = aggregate(runs, cfg)
    md = render_results(
        summary,
        representative_breakdown(runs),
        gpu_active_ceiling=cfg.gpu_active_ceiling,
    )
    assert "SIGNED OFF" in md
    assert "Data-stall %" in md
    assert "events_per_second" in md


# --- cli smoke --------------------------------------------------------------


def test_cli_manifest_build_and_verify(tmp_path: Path, capsys):
    from gitm.bench.cli import main

    root = tmp_path / "ds"
    _make_dataset(root)
    out = tmp_path / "manifest.yaml"

    rc = main(["manifest", "build", "--root", str(root), "--benchmark", "hft",
               "--out", str(out)])
    assert rc == 0
    assert out.exists()

    rc = main(["manifest", "verify", "--manifest", str(out), "--root", str(root)])
    assert rc == 0
