"""Tests for ``gitm.optimizer.qualify_run`` — capture integrity, not floor commitment."""

from __future__ import annotations

import json
from pathlib import Path

from gitm.optimizer.deployment_spec import (
    DeploymentSpec,
    SignalContract,
    default_signal_contract,
)
from gitm.optimizer.qualify_run import main as qualify_main
from gitm.optimizer.qualify_run import qualify_run
from gitm.tracer.kernel_taxonomy import NAME_MAX

MODEL = "moonshotai/Kimi-K2.5"
REV = "ckpt-rev-1"
ENGINE_VER = "0.9.0"
ROCM_VER = "7.2.3"
DIGEST = "sha256:deadbeef"
TRAFFIC = "smoke"
WORKLOAD = "kimi-loop"


def _header(**kwargs):
    base = {
        "workload_id": WORKLOAD,
        "fingerprint": MODEL,
        "run_id": "run-test",
        "device_count": 1,
        "vendor": "amd",
        "captured_at_ns": 1_000,
        "duration_ns": 10_000,
        "source": "rocprof",
    }
    base.update(kwargs)
    return base


def _kernel(pid=100, device_id=0, start=1000, end=2000, name="attn_fwd", **extra):
    rec = {
        "kind": "kernel",
        "name": name,
        "start_ns": start,
        "end_ns": end,
        "stream_id": 0,
        "device_id": device_id,
        "pid": pid,
        "grid_x": 1,
        "grid_y": 1,
        "grid_z": 1,
        "block_x": 64,
        "block_y": 1,
        "block_z": 1,
    }
    rec.update(extra)
    return rec


def _write_trace(dirpath: Path, header: dict, events: list[dict], name="trace.jsonl") -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / name
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": header}) + "\n")
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return path


def _write_identity(art: Path, **overrides):
    """Observed deployment identity matching the default full spec."""
    data = {
        "checkpoint_revision": REV,
        "engine": "vllm",
        "vllm_version": ENGINE_VER,
        "rocm_version": ROCM_VER,
        "image_digest": DIGEST,
        "traffic_manifest_id": TRAFFIC,
        "tp": 1,
    }
    data.update(overrides)
    (art / "run_manifest.json").write_text(json.dumps(data), encoding="utf-8")
    (art / "MANIFEST").write_text(
        f"phase=e0 arm=C label={TRAFFIC} ts=2026-01-01T00:00:00Z "
        f"checkpoint_revision={data['checkpoint_revision']} "
        f"image_digest={data['image_digest']} engine=vllm tp={data['tp']}\n"
        f"rocm={data['rocm_version']}\n"
        f"vllm={data['vllm_version']}\n"
        f"traffic_manifest_id={data['traffic_manifest_id']}\n",
        encoding="utf-8",
    )


def _spec(**kwargs) -> DeploymentSpec:
    data = {
        "model_repository": MODEL,
        "checkpoint_revision": REV,
        "engine": "vllm",
        "engine_version": ENGINE_VER,
        "rocm_version": ROCM_VER,
        "image_digest": DIGEST,
        "topology": {"nodes": 1, "gpus": 1, "processes": 1, "ranks": 1, "tp": 1},
        "capture_backend": "rocprof-inject",
        "signal_contract_version": "v0",
        "workload_id": WORKLOAD,
        "traffic_manifest_id": TRAFFIC,
    }
    data.update(kwargs)
    if "topology" in kwargs and isinstance(kwargs["topology"], dict):
        data["topology"] = kwargs["topology"]
    return DeploymentSpec.model_validate(data)


def _check_map(result):
    return {c.name: c for c in result.checks}


def test_valid_run_qualified(tmp_path: Path):
    art = tmp_path / "cap"
    _write_trace(art, _header(), [_kernel()])
    _write_identity(art)
    result = qualify_run(art, _spec())
    assert result.verdict == "qualified", json.dumps(result.to_dict(), indent=2)
    assert result.deployment_fingerprint.startswith("deploy:")
    assert result.capture_fingerprint.startswith("capture:")
    assert result.signal_contract_version == "v0"
    assert result.qualify_run_revision.startswith("0.")


def test_dead_collector_empty_kernels(tmp_path: Path):
    art = tmp_path / "cap"
    _write_trace(art, _header(), [])  # header only — no kernels
    _write_identity(art)
    result = qualify_run(art, _spec())
    assert result.verdict == "not_established"
    assert _check_map(result)["kernel_events"].status == "unknown"


def test_dead_collector_missing_trace(tmp_path: Path):
    art = tmp_path / "empty_dir"
    art.mkdir()
    result = qualify_run(art, _spec())
    assert result.verdict == "not_established"
    assert _check_map(result)["merged_trace"].status == "unknown"


def test_missing_rank_vs_declared_topology(tmp_path: Path):
    art = tmp_path / "cap"
    _write_trace(art, _header(device_count=1), [_kernel(pid=1, device_id=0)])
    _write_identity(art)
    result = qualify_run(
        art,
        _spec(topology={"nodes": 1, "gpus": 8, "processes": 8, "ranks": 8, "tp": 8}),
    )
    assert result.verdict == "invalid_run"
    statuses = {c.name: c.status for c in result.checks}
    assert statuses["topology_processes"] == "fail"
    assert statuses["topology_gpus"] == "fail"
    assert statuses["topology_ranks"] == "fail"


def test_truncated_names(tmp_path: Path):
    art = tmp_path / "cap"
    long_name = "x" * NAME_MAX
    _write_trace(art, _header(), [_kernel(name=long_name)])
    _write_identity(art)
    result = qualify_run(art, _spec())
    assert result.verdict == "invalid_run"
    assert _check_map(result)["truncated_names"].status == "fail"


def test_truncated_shard(tmp_path: Path):
    art = tmp_path / "cap"
    art.mkdir(parents=True)
    path = art / "trace.jsonl"
    # Valid header + kernel, then EOF mid-line (no trailing newline on incomplete JSON).
    with path.open("wb") as fh:
        fh.write((json.dumps({"_header": _header()}) + "\n").encode())
        fh.write((json.dumps(_kernel()) + "\n").encode())
        fh.write(b'{"kind":"kernel","name":"cut_off"')
    _write_identity(art)
    result = qualify_run(art, _spec())
    assert result.verdict == "invalid_run"
    assert _check_map(result)["truncated_shard"].status == "fail"


def test_conflicting_engine_identity(tmp_path: Path):
    art = tmp_path / "cap"
    _write_trace(art, _header(), [_kernel()])
    _write_identity(art, vllm_version="0.9.0")
    (art / "MANIFEST").write_text(
        f"phase=e0 arm=C label={TRAFFIC} ts=2026-01-01T00:00:00Z "
        f"checkpoint_revision={REV} image_digest={DIGEST} engine=vllm tp=1\n"
        f"rocm={ROCM_VER}\n"
        "vllm=0.9.0\n"
        f"traffic_manifest_id={TRAFFIC}\n",
        encoding="utf-8",
    )
    result = qualify_run(
        art,
        _spec(engine_version="0.8.0"),
    )
    assert result.verdict == "invalid_run"
    assert _check_map(result)["engine_version"].status == "fail"
    assert _check_map(result)["rocm_version"].status == "pass"


def test_unavailable_required_evidence(tmp_path: Path):
    art = tmp_path / "cap"
    _write_trace(art, _header(), [_kernel()])
    # Identity sidecars omit checkpoint — declared revision is unobservable.
    (art / "run_manifest.json").write_text(
        json.dumps({
            "engine": "vllm",
            "vllm_version": ENGINE_VER,
            "rocm_version": ROCM_VER,
            "image_digest": DIGEST,
            "traffic_manifest_id": TRAFFIC,
            "tp": 1,
        }),
        encoding="utf-8",
    )
    result = qualify_run(art, _spec())
    assert result.verdict == "not_established"
    assert _check_map(result)["checkpoint_revision"].status == "unknown"


def test_conflicting_model_repository(tmp_path: Path):
    art = tmp_path / "cap"
    _write_trace(art, _header(fingerprint="other/model"), [_kernel()])
    _write_identity(art)
    result = qualify_run(art, _spec())
    assert result.verdict == "invalid_run"
    assert _check_map(result)["model_repository"].status == "fail"


def test_meta_dropped_records_fail(tmp_path: Path):
    art = tmp_path / "cap"
    events = [
        {"kind": "meta", "dropped_records": 3},
        _kernel(),
    ]
    _write_trace(art, _header(), events)
    _write_identity(art)
    result = qualify_run(art, _spec())
    assert result.verdict == "invalid_run"
    assert _check_map(result)["dropped_records"].status == "fail"


def test_required_source_amdsmi_absent(tmp_path: Path):
    art = tmp_path / "cap"
    _write_trace(art, _header(), [_kernel()])
    _write_identity(art)
    contract = SignalContract(required_sources=["merged_trace", "amdsmi"])
    result = qualify_run(art, _spec(), contract)
    assert result.verdict == "not_established"
    assert _check_map(result)["source:amdsmi"].status == "unknown"


def test_sampling_gap_and_clock_alignment(tmp_path: Path):
    art = tmp_path / "cap"
    _write_trace(art, _header(), [_kernel()])
    _write_identity(art)
    # Wall window + amd-smi with a large gap.
    manifest = json.loads((art / "run_manifest.json").read_text())
    manifest.update({
        "wall_start_ns": 1_000_000_000,
        "wall_end_ns": 5_000_000_000,
        "captured_at_is_wall": True,
    })
    (art / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (art / "amdsmi.jsonl").write_text(
        "\n".join([
            json.dumps({"ts_ns": 1_000_000_000, "gpu_index": 0}),
            json.dumps({"ts_ns": 2_000_000_000, "gpu_index": 0}),
            json.dumps({"ts_ns": 5_000_000_000, "gpu_index": 0}),  # 3s gap
        ]) + "\n",
        encoding="utf-8",
    )
    contract = SignalContract(required_sources=["merged_trace", "amdsmi"])
    result = qualify_run(art, _spec(), contract)
    assert result.verdict == "invalid_run"
    assert _check_map(result)["sampling_gaps"].status == "fail"
    assert _check_map(result)["clock_alignment"].status == "pass"


def test_cli_emits_json(tmp_path: Path, capsys):
    art = tmp_path / "cap"
    _write_trace(art, _header(), [_kernel()])
    _write_identity(art)
    spec_path = tmp_path / "deploy.json"
    spec_path.write_text(
        json.dumps({
            "model_repository": MODEL,
            "checkpoint_revision": REV,
            "engine": "vllm",
            "engine_version": ENGINE_VER,
            "rocm_version": ROCM_VER,
            "image_digest": DIGEST,
            "topology": {"nodes": 1, "gpus": 1, "processes": 1, "ranks": 1, "tp": 1},
            "capture_backend": "rocprof-inject",
            "signal_contract_version": "v0",
            "workload_id": WORKLOAD,
            "traffic_manifest_id": TRAFFIC,
        })
    )
    rc = qualify_main([
        "--artifacts", str(art),
        "--deployment-spec", str(spec_path),
        "--signal-contract-version", "v0",
    ])
    out_text = capsys.readouterr().out
    assert rc == 0, out_text
    out = json.loads(out_text)
    assert out["verdict"] == "qualified"


def test_cli_missing_deployment_spec(tmp_path: Path, capsys):
    art = tmp_path / "cap"
    art.mkdir()
    rc = qualify_main([
        "--artifacts", str(art),
        "--deployment-spec", str(tmp_path / "missing.json"),
    ])
    assert rc == 2
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "not_established"


def test_default_contract_version():
    c = default_signal_contract()
    assert c.version == "v0"
    assert c.capture_backend == "rocprof-inject"
    assert "merged_trace" in c.required_sources
