"""Fault-injection tests for deploy/h200/gitm-node-health + qualify_run wiring."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from gitm.optimizer.deployment_spec import (
    DeploymentSpec,
    SignalContract,
    default_h200_signal_contract,
)
from gitm.optimizer.qualify_run import qualify_run

ROOT = Path(__file__).resolve().parents[1]
HEALTH_PATH = ROOT / "deploy" / "h200" / "gitm-node-health"


def _load_script(name: str, path: Path):
    # deploy/h200 scripts have no .py suffix.
    from importlib.machinery import SourceFileLoader
    from importlib.util import module_from_spec, spec_from_loader

    loader = SourceFileLoader(name, str(path))
    spec = spec_from_loader(name, loader)
    assert spec and spec.loader
    mod = module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


health = _load_script("gitm_node_health", HEALTH_PATH)


def _healthy_snap(**overrides) -> dict:
    snap = {
        "ts": 1.0,
        "phase": "pre",
        "node": "gitmachine-2",
        "job_id": "42",
        "temps_c": [60.0] * 8,
        "throttle_reasons": ["Not Active"] * 8,
        "power": [["700", "700", "Not Active"] for _ in range(8)],
        "xid_lines": [],
        "xid_error": None,
        "shm": ["64G", "60G", "5%"],
        "nvlink": {
            "links": "GPU0: link 0 Active\n",
            "links_rc": 0,
            "topo": (
                "\tGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\n"
                "GPU0\tX\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tCPU Affinity\n"
                "GPU1\tNV18\tX\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\n"
                "GPU2\tNV18\tNV18\tX\tNV18\tNV18\tNV18\tNV18\tNV18\n"
                "GPU3\tNV18\tNV18\tNV18\tX\tNV18\tNV18\tNV18\tNV18\n"
                "GPU4\tNV18\tNV18\tNV18\tNV18\tX\tNV18\tNV18\tNV18\n"
                "GPU5\tNV18\tNV18\tNV18\tNV18\tNV18\tX\tNV18\tNV18\n"
                "GPU6\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tX\tNV18\n"
                "GPU7\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tX\n"
            ),
            "topo_rc": 0,
        },
        "vllm_alive": False,
    }
    snap.update(overrides)
    return snap


def _nh_contract() -> SignalContract:
    return SignalContract(
        capture_backend="cupti-inject",
        required_sources=["node_health"],
        require_kernel_events=False,
        require_named_kernels=False,
        require_pid_on_kernels=False,
        require_device_on_kernels=False,
        require_nonempty_header=False,
        require_capture_window=False,
        require_deployment_identity=False,
    )


def _spec() -> DeploymentSpec:
    return DeploymentSpec.model_validate(
        {
            "model_repository": "Qwen/Qwen3-0.6B",
            "topology": {"nodes": 1, "gpus": 8, "tp": 8},
            "capture_backend": "cupti-inject",
            "signal_contract_version": "v0",
        }
    )


def _write_window(art: Path, window: dict) -> Path:
    art.mkdir(parents=True, exist_ok=True)
    path = art / "node_health_window.json"
    path.write_text(json.dumps(window), encoding="utf-8")
    return path


def test_print_thresholds_matches_doc_keys():
    assert "thermal_max_c" in health.THRESHOLDS
    assert health.THRESHOLDS["thermal_max_c"] == 83.0
    assert health.THRESHOLDS["shm_min_size_g"] == 16


def test_dmesg_helper_filters_before_returning_and_is_user_scoped():
    setup = (ROOT / "deploy" / "h200" / "setup-node.sh").read_text()
    helper = setup.split(
        "cat > /usr/local/libexec/gitm-dmesg-xid <<'EOF'\n", 1
    )[1].split("\nEOF", 1)[0]
    assert "grep -i 'NVRM:[[:space:]]*Xid'" in helper
    assert '"$HEALTH_USER ALL=(root) NOPASSWD:' in setup
    assert "getent passwd | awk" not in setup
    assert "usermod -a -G gitm-health" not in setup


def test_throttled_node_caught(tmp_path: Path):
    # Per-field Active values (what nvidia-smi returns for bad reason fields).
    pre = _healthy_snap(
        phase="pre",
        throttle_reasons=["Active", "Not Active", "Not Active", "Not Active"] * 2,
        temps_c=[85.0] * 8,
    )
    post = _healthy_snap(
        phase="post",
        ts=2.0,
        throttle_reasons=["0x0000000000000020"] * 8,  # SW thermal bitmask
    )
    window = health.eval_window([pre, post])
    assert window["healthy"] is False
    assert "clocks" in window["failed_signals"] or "thermal" in window["failed_signals"]

    art = tmp_path / "run"
    _write_window(art, window)
    result = qualify_run(art, _spec(), _nh_contract())
    assert result.verdict == "invalid_run"
    nh = next(c for c in result.checks if c.name == "node_health_window")
    assert nh.status == "fail"
    assert "clocks" in nh.evidence.get("failed_signals", []) or "thermal" in nh.evidence.get(
        "failed_signals", []
    )


def test_clocks_bad_decodes_hex_and_active():
    assert health._clocks_bad("Active") is True
    assert health._clocks_bad("Not Active") is False
    assert health._clocks_bad("0x0000000000000020") is True  # SW thermal
    assert health._clocks_bad("0x0000000000000000") is False


def test_xid_mid_run_caught(tmp_path: Path):
    pre = _healthy_snap(phase="pre", xid_lines=[])
    # XID only in a mid-window sample (rolled out of dmesg before post).
    mid = _healthy_snap(
        phase="sample",
        ts=1.5,
        xid_lines=["[123] NVRM: Xid (PCI:0000:00): 79, pid=1, name=vllm"],
    )
    post = _healthy_snap(phase="post", ts=2.0, xid_lines=[])
    window = health.eval_window([pre, mid, post])
    assert window["healthy"] is False
    assert "dmesg_xid" in window["failed_signals"]

    art = tmp_path / "run"
    _write_window(art, window)
    result = qualify_run(art, _spec(), _nh_contract())
    assert result.verdict == "invalid_run"
    nh = next(c for c in result.checks if c.name == "node_health_window")
    assert nh.status == "fail"
    assert "dmesg_xid" in nh.evidence["failed_signals"]


def test_dead_vllm_caught(tmp_path: Path):
    pre = _healthy_snap(phase="pre")
    s1 = _healthy_snap(phase="sample", ts=1.5, vllm_alive=True)
    s2 = _healthy_snap(phase="sample", ts=1.6, vllm_alive=False)
    post = _healthy_snap(phase="post", ts=2.0, vllm_alive=False)
    window = health.eval_window([pre, s1, s2, post])
    assert window["healthy"] is False
    assert "vllm_liveness" in window["failed_signals"]

    art = tmp_path / "run"
    _write_window(art, window)
    result = qualify_run(art, _spec(), _nh_contract())
    assert result.verdict == "invalid_run"
    nh = next(c for c in result.checks if c.name == "node_health_window")
    assert "vllm_liveness" in nh.evidence["failed_signals"]


def test_vllm_alive_ignores_srun_wrapper(monkeypatch: pytest.MonkeyPatch):
    def fake_run(cmd, **kwargs):
        out = (
            "1234 srun --container-image=x bash -c 'vllm serve foo'\n"
            "5678 python3 -m vllm.entrypoints.openai.api_server\n"
        )
        return type("P", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(health, "RUN", fake_run)
    assert health._vllm_alive() is True

    def only_srun(cmd, **kwargs):
        out = "1234 srun --container-image=x bash -c 'vllm serve foo'\n"
        return type("P", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(health, "RUN", only_srun)
    assert health._vllm_alive() is False


def test_missing_post_is_unhealthy():
    pre = _healthy_snap(phase="pre")
    window = health.eval_window([pre])
    assert window["healthy"] is False
    assert "window_incomplete" in window["failed_signals"]


def test_nvlink_requires_status_output():
    snap = _healthy_snap(
        nvlink={"links": "", "links_rc": 1, "topo": "GPU0 X\n", "topo_rc": 0}
    )
    assert "nvlink" in health.eval_snapshot(snap)


def test_healthy_window_passes_qualify(tmp_path: Path):
    pre = _healthy_snap(phase="pre")
    post = _healthy_snap(phase="post", ts=2.0)
    sample = _healthy_snap(phase="sample", ts=1.5, vllm_alive=True)
    window = health.eval_window([pre, sample, post])
    assert window["healthy"] is True
    assert window["failed_signals"] == []

    art = tmp_path / "run"
    _write_window(art, window)
    result = qualify_run(art, _spec(), _nh_contract())
    # No merged_trace → not_established, but node_health_window must pass.
    nh = next(c for c in result.checks if c.name == "node_health_window")
    assert nh.status == "pass"
    assert result.verdict != "invalid_run"


@pytest.mark.parametrize("contents", [None, "{", "[]", "{}"])
def test_missing_or_malformed_node_health_is_invalid(tmp_path: Path, contents: str | None):
    art = tmp_path / "run"
    art.mkdir()
    if contents is not None:
        (art / "node_health_window.json").write_text(contents, encoding="utf-8")
    result = qualify_run(art, _spec(), _nh_contract())
    assert result.verdict == "invalid_run"


def test_h200_contract_requires_node_health():
    c = default_h200_signal_contract()
    assert "node_health" in c.required_sources
    assert c.capture_backend == "cupti-inject"


def test_gitm_submit_dry_run_wires_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_script("gitm_submit", ROOT / "deploy" / "h200" / "gitm-submit")
    args = Namespace(
        command=None,
        gitm_ref="main",
        capture_args="",
        vllm_args="",
        env=None,
        image=None,
        model="Qwen/Qwen3-0.6B",
        name="health-dry",
        partition="gpu",
        max_hours=0.25,
        results_dir="/var/lib/gitm/results",
        node=None,
        dry_run=True,
    )
    script = mod.build_script(args)
    assert "gitm-node-health" in script
    assert 'pre --job "$SLURM_JOB_ID"' in script
    assert 'post --job "$SLURM_JOB_ID"' in script
    assert "summarize" in script
    assert "gate --window" in script
    assert 'if [ ! -x "$HEALTH_BIN" ]; then' in script


def test_gitm_job_dry_run_wires_health():
    mod = _load_script("gitm_job", ROOT / "deploy" / "h200" / "gitm-job")
    cfg = {
        "PARTITION": "gpu",
        "RESULTS_DIR": "/var/lib/gitm/results",
        "HF_CACHE": "/var/lib/gitm/hf-cache",
        "DEFAULT_IMAGE": "vllm/vllm-openai:latest",
        "GRES": "gpu:8",
    }
    args = Namespace(
        command=None,
        gitm_ref="main",
        capture_args="",
        vllm_args="",
        env=None,
        image=None,
        model="Qwen/Qwen3-0.6B",
        name="health-dry",
        max_hours=0.25,
        node=None,
    )
    script = mod.build_script(args, cfg)
    assert "gitm-node-health" in script
    assert "pre --job" in script
    assert "post --job" in script
    assert 'if [ ! -x "$HEALTH_BIN" ]; then' in script


def test_gate_discards_and_requeues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    results = tmp_path / "results"
    out = results / "99"
    out.mkdir(parents=True)
    (out / "serving_summary.json").write_text("{}", encoding="utf-8")
    window = {
        "job_id": "99",
        "node": "gitmachine-1",
        "healthy": False,
        "failed_signals": ["dmesg_xid"],
    }
    wpath = out / "node_health_window.json"
    wpath.write_text(json.dumps(window), encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["scontrol", "show", "job"]:
            return type("P", (), {"returncode": 0, "stdout": "JobId=99 Restarts=0\n", "stderr": ""})()
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(health, "RUN", fake_run)
    rc = health.gate(
        Namespace(window=str(wpath), results=str(results), job="99")
    )
    assert rc == 1
    assert not out.exists()
    assert (results / "discarded" / "99").is_dir()
    assert any(c[:2] == ["scontrol", "requeue"] for c in calls)
