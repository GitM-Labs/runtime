"""Run-capture integrity gate: qualify_run.

Emits ``qualified`` / ``invalid_run`` / ``not_established`` for a run's artifacts
against a declared deployment spec and signal contract. This is *not*
``gitm.optimizer.qualification.qualify`` (commercial floor commitment).

    python -m gitm.optimizer.qualify_run --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from gitm.optimizer.deployment_spec import (
    PROVISIONAL_SIGNAL_CONTRACT_VERSION,
    DeploymentSpec,
    SignalContract,
    default_signal_contract,
    deployment_fingerprint,
    load_deployment_spec,
    load_signal_contract,
    parse_run_manifest,
)
from gitm.optimizer.replay import _load_trace_jsonl
from gitm.tracer.kernel_taxonomy import NAME_MAX
from gitm.tracer.schema import Trace

QUALIFY_RUN_REVISION = "0.2.0"

CheckStatus = Literal["pass", "fail", "unknown"]
Verdict = Literal["qualified", "invalid_run", "not_established"]

# Source name → relative path globs searched under artifacts (and parent run dir).
_SOURCE_GLOBS: dict[str, tuple[str, ...]] = {
    "merged_trace": ("**/trace.jsonl", "**/*capture*/**/*.jsonl", "**/*.jsonl"),
    "amdsmi": ("amdsmi.jsonl", "**/amdsmi*.jsonl", "telemetry/amdsmi.jsonl"),
    "metrics": ("**/metrics*.txt", "**/vllm_metrics*", "**/scrape_metrics*"),
}


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualifyRunResult:
    verdict: Verdict
    checks: list[CheckResult]
    deployment_fingerprint: str
    capture_fingerprint: str
    signal_contract_version: str
    qualify_run_revision: str = QUALIFY_RUN_REVISION

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "checks": [asdict(c) for c in self.checks],
            "deployment_fingerprint": self.deployment_fingerprint,
            "capture_fingerprint": self.capture_fingerprint,
            "signal_contract_version": self.signal_contract_version,
            "qualify_run_revision": self.qualify_run_revision,
        }


def _find_trace_jsonl(artifacts_dir: Path) -> Path | None:
    if artifacts_dir.is_file() and artifacts_dir.suffix == ".jsonl":
        return artifacts_dir
    candidates = sorted(artifacts_dir.rglob("*.jsonl"))
    # Prefer capture/trace naming; fall back to last jsonl (matches E0 habit).
    preferred = [
        p
        for p in candidates
        if (p.name == "trace.jsonl" or "capture" in p.parts)
        and "amdsmi" not in p.name.lower()
    ]
    pool = preferred or [
        p for p in candidates if "amdsmi" not in p.name.lower()
    ]
    return pool[-1] if pool else None


def _scan_jsonl(path: Path) -> dict[str, Any]:
    """Integrity scan without requiring every line to be a Trace event."""
    malformed = 0
    meta_drops = 0
    truncated = 0
    kernels = 0
    named = 0
    anonymous = 0
    invalid_ts = 0
    workers: dict[tuple[Any, Any], int] = {}
    pids: set[Any] = set()
    devices: set[Any] = set()
    ranks: set[Any] = set()
    nodes: set[Any] = set()
    shards: set[Any] = set()
    header: dict[str, Any] | None = None
    duplicate_keys = 0
    seen: set[tuple] = set()
    empty_file = True
    min_start: int | None = None
    max_end: int | None = None
    truncated_shard = False
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        # EOF mid-line is the truncated-shard failure mode.
        truncated_shard = True

    text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if not line.strip():
            continue
        empty_file = False
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(rec, dict):
            malformed += 1
            continue
        if "_header" in rec:
            header = rec["_header"] if isinstance(rec["_header"], dict) else {}
            continue
        kind = rec.get("kind")
        if kind == "meta":
            drops = rec.get("dropped_records")
            if isinstance(drops, int):
                meta_drops += drops
            if rec.get("truncated_shard") or rec.get("truncated"):
                truncated_shard = True
            continue
        if kind != "kernel":
            continue
        kernels += 1
        start, end = rec.get("start_ns"), rec.get("end_ns")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            invalid_ts += 1
        else:
            min_start = start if min_start is None else min(min_start, start)
            max_end = end if max_end is None else max(max_end, end)
        name = rec.get("name")
        if isinstance(name, str) and name and "unknown" not in name.lower():
            named += 1
        else:
            anonymous += 1
        if isinstance(name, str) and len(name.encode("utf-8")) >= NAME_MAX:
            truncated += 1
        pid, device = rec.get("pid"), rec.get("device_id")
        workers[(pid, device)] = workers.get((pid, device), 0) + 1
        if pid is not None:
            pids.add(pid)
        if device is not None:
            devices.add(device)
        if rec.get("rank") is not None:
            ranks.add(rec.get("rank"))
        if rec.get("node") is not None:
            nodes.add(rec.get("node"))
        if rec.get("shard") is not None:
            shards.add(rec.get("shard"))
        key = (pid, device, start, end, name)
        if key in seen:
            duplicate_keys += 1
        else:
            seen.add(key)

    # Trailing incomplete line (no final newline) counts as truncated + malformed.
    if truncated_shard and text and not text.endswith("\n"):
        last = text.rsplit("\n", 1)[-1]
        if last.strip():
            try:
                json.loads(last)
            except json.JSONDecodeError:
                malformed += 1

    return {
        "empty_file": empty_file,
        "header": header,
        "malformed_lines": malformed,
        "dropped_records": meta_drops,
        "truncated_names": truncated,
        "truncated_shard": truncated_shard,
        "kernels": kernels,
        "named_kernels": named,
        "anonymous_kernels": anonymous,
        "invalid_kernels": invalid_ts,
        "workers": [{"pid": p, "device": d, "kernels": n} for (p, d), n in workers.items()],
        "pids": sorted(pids, key=lambda x: (x is None, x)),
        "devices": sorted(devices, key=lambda x: (x is None, x)),
        "ranks": sorted(ranks, key=lambda x: (x is None, x)),
        "nodes": sorted(nodes, key=lambda x: (x is None, str(x))),
        "shards": sorted(shards, key=lambda x: (x is None, str(x))),
        "duplicate_event_keys": duplicate_keys,
        "missing_pid_kernels": sum(1 for w in workers if w[0] is None),
        "missing_device_kernels": sum(1 for w in workers if w[1] is None),
        "min_start_ns": min_start,
        "max_end_ns": max_end,
    }


def _capture_fingerprint(scan: dict[str, Any], trace: Trace | None) -> str:
    payload = {
        "run_id": (trace.run_id if trace else None) or (scan.get("header") or {}).get("run_id"),
        "source": (trace.source if trace else None) or (scan.get("header") or {}).get("source"),
        "device_count": (
            (trace.device_count if trace else None)
            or (scan.get("header") or {}).get("device_count")
        ),
        "pids": scan.get("pids"),
        "devices": scan.get("devices"),
        "dropped_records": scan.get("dropped_records"),
        "kernels": scan.get("kernels"),
        "truncated_shard": scan.get("truncated_shard"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]
    return f"capture:{digest}"


def _cmp(
    name: str,
    declared: Any,
    observed: Any,
    *,
    required: bool = True,
) -> CheckResult:
    if declared is None and observed is None:
        return CheckResult(
            name,
            "unknown" if required else "pass",
            {"declared": None, "observed": None, "note": "absent"},
        )
    if declared is None:
        return CheckResult(
            name,
            "unknown" if required else "pass",
            {"declared": None, "observed": observed, "note": "undeclared"},
        )
    if observed is None:
        return CheckResult(
            name,
            "unknown",
            {"declared": declared, "observed": None, "note": "unobservable"},
        )
    if str(declared) != str(observed):
        return CheckResult(
            name,
            "fail",
            {"declared": declared, "observed": observed},
        )
    return CheckResult(name, "pass", {"declared": declared, "observed": observed})


def _load_observed_manifest(artifacts_dir: Path) -> dict[str, Any]:
    for candidate in (
        artifacts_dir / "MANIFEST",
        artifacts_dir.parent / "MANIFEST",
        artifacts_dir / "run_manifest.json",
    ):
        if candidate.is_file():
            if candidate.suffix == ".json":
                return json.loads(candidate.read_text(encoding="utf-8"))
            return parse_run_manifest(candidate.read_text(encoding="utf-8"))
    return {}


def _serving_sidecar(artifacts_dir: Path) -> dict[str, Any]:
    for candidate in (
        artifacts_dir / "serving_summary.json",
        artifacts_dir / "run_manifest.json",
    ):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
    return {}


def _search_roots(artifacts_dir: Path) -> list[Path]:
    roots = [artifacts_dir]
    if artifacts_dir.parent != artifacts_dir:
        roots.append(artifacts_dir.parent)
    return roots


def _find_source_path(artifacts_dir: Path, source: str) -> Path | None:
    if source == "merged_trace":
        return _find_trace_jsonl(artifacts_dir)
    globs = _SOURCE_GLOBS.get(source, (f"**/{source}*", f"**/*{source}*"))
    for root in _search_roots(artifacts_dir):
        for pattern in globs:
            hits = sorted(root.glob(pattern))
            # Prefer non-empty files.
            for hit in hits:
                if hit.is_file() and hit.stat().st_size > 0:
                    return hit
    return None


def _parse_ts_ns_stream(path: Path) -> list[int]:
    """Extract timestamps from amd-smi JSONL or metrics scrape text."""
    stamps: list[int] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return stamps
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("### ts_ns="):
            try:
                stamps.append(int(line.split("=", 1)[1]))
            except ValueError:
                continue
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # amd-smi loop writes ``{"ts_ns":N,"m":...}`` sometimes split; try prefix
            m = re.search(r'"ts_ns"\s*:\s*(\d+)', line)
            if m:
                stamps.append(int(m.group(1)))
            continue
        if isinstance(rec, dict) and isinstance(rec.get("ts_ns"), int):
            stamps.append(rec["ts_ns"])
    return stamps


def _wall_window_ns(
    header: dict[str, Any],
    sidecar: dict[str, Any],
    manifest: dict[str, Any],
    trace: Trace | None,
) -> tuple[int | None, int | None]:
    """Return (wall_start_ns, wall_end_ns) when explicitly recorded — never invent."""
    start = (
        sidecar.get("wall_start_ns")
        or sidecar.get("capture_start_ns")
        or manifest.get("wall_start_ns")
        or header.get("wall_start_ns")
    )
    end = (
        sidecar.get("wall_end_ns")
        or sidecar.get("capture_end_ns")
        or manifest.get("wall_end_ns")
        or header.get("wall_end_ns")
    )
    window_s = sidecar.get("window_s") or (sidecar.get("serving") or {}).get("window_s")
    if start is None and isinstance(header.get("captured_at_ns"), int):
        # captured_at_ns is device/tool domain on inject path — only treat as wall
        # when the sidecar explicitly says so.
        if sidecar.get("captured_at_is_wall"):
            start = header["captured_at_ns"]
    if (
        isinstance(start, int)
        and end is None
        and isinstance(window_s, (int, float))
        and window_s > 0
    ):
        end = start + int(float(window_s) * 1e9)
    if isinstance(start, int) and end is None and isinstance(header.get("duration_ns"), int):
        if sidecar.get("captured_at_is_wall") or header.get("wall_start_ns") is not None:
            end = start + header["duration_ns"]
    if isinstance(start, int) and isinstance(end, int):
        return start, end
    _ = trace
    return (
        start if isinstance(start, int) else None,
        end if isinstance(end, int) else None,
    )


def qualify_run(
    artifacts_dir: Path,
    deployment: DeploymentSpec,
    contract: SignalContract | None = None,
) -> QualifyRunResult:
    """Emit a machine-readable integrity verdict for one run's artifacts."""
    contract = contract or default_signal_contract()
    checks: list[CheckResult] = []
    artifacts_dir = artifacts_dir.resolve()
    identity_required = contract.require_deployment_identity

    # --- contract version ---
    if contract.version != deployment.signal_contract_version:
        checks.append(
            CheckResult(
                "signal_contract_version",
                "fail",
                {
                    "deployment_declares": deployment.signal_contract_version,
                    "contract_file": contract.version,
                },
            )
        )
    else:
        checks.append(
            CheckResult(
                "signal_contract_version",
                "pass",
                {"version": contract.version},
            )
        )

    if contract.capture_backend != deployment.capture_backend:
        checks.append(
            CheckResult(
                "capture_backend",
                "fail",
                {
                    "declared": deployment.capture_backend,
                    "contract": contract.capture_backend,
                },
            )
        )
    else:
        checks.append(
            CheckResult(
                "capture_backend",
                "pass",
                {"backend": contract.capture_backend},
            )
        )

    # Required sources (empty source → unknown, never a silent zero).
    source_paths: dict[str, Path | None] = {}
    for source in contract.required_sources:
        path = _find_source_path(artifacts_dir, source)
        source_paths[source] = path
        if path is None:
            checks.append(
                CheckResult(
                    f"source:{source}",
                    "unknown",
                    {"source": source, "note": "empty_or_absent"},
                )
            )
        else:
            checks.append(
                CheckResult(
                    f"source:{source}",
                    "pass",
                    {"source": source, "path": str(path)},
                )
            )

    trace_path = source_paths.get("merged_trace") or _find_trace_jsonl(artifacts_dir)
    if trace_path is None:
        checks.append(
            CheckResult(
                "merged_trace",
                "unknown",
                {"artifacts_dir": str(artifacts_dir), "note": "no jsonl"},
            )
        )
        result = QualifyRunResult(
            verdict=_aggregate(checks),
            checks=checks,
            deployment_fingerprint=deployment_fingerprint(deployment, contract),
            capture_fingerprint="capture:absent",
            signal_contract_version=contract.version,
        )
        return result

    scan = _scan_jsonl(trace_path)
    checks.append(
        CheckResult(
            "merged_trace",
            "pass",
            {"path": str(trace_path)},
        )
    )

    trace: Trace | None = None
    try:
        trace = _load_trace_jsonl(trace_path)
    except (ValueError, OSError, TypeError, ValidationError) as exc:
        checks.append(
            CheckResult(
                "trace_load",
                "unknown",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        )

    header = scan.get("header") or {}
    if contract.require_nonempty_header and not header and trace is None:
        checks.append(
            CheckResult("capture_header", "unknown", {"note": "missing _header"})
        )
    else:
        checks.append(
            CheckResult(
                "capture_header",
                "pass",
                {
                    "run_id": header.get("run_id") or (trace.run_id if trace else None),
                    "source": header.get("source") or (trace.source if trace else None),
                    "device_count": header.get("device_count")
                    or (trace.device_count if trace else None),
                },
            )
        )

    if contract.require_kernel_events:
        if scan["kernels"] == 0:
            checks.append(
                CheckResult(
                    "kernel_events",
                    "unknown",
                    {"kernels": 0, "note": "empty_capture"},
                )
            )
        else:
            checks.append(
                CheckResult("kernel_events", "pass", {"kernels": scan["kernels"]})
            )

    if contract.require_named_kernels:
        if scan["kernels"] == 0:
            checks.append(
                CheckResult(
                    "named_kernels",
                    "unknown",
                    {"named": 0, "anonymous": 0},
                )
            )
        elif scan["named_kernels"] == 0:
            checks.append(
                CheckResult(
                    "named_kernels",
                    "fail",
                    {"named": 0, "anonymous": scan["anonymous_kernels"]},
                )
            )
        else:
            checks.append(
                CheckResult(
                    "named_kernels",
                    "pass",
                    {
                        "named": scan["named_kernels"],
                        "anonymous": scan["anonymous_kernels"],
                    },
                )
            )

    # Drops / malformed / truncated / duplicates / truncated shards
    if scan["dropped_records"] > contract.max_dropped_records:
        checks.append(
            CheckResult(
                "dropped_records",
                "fail",
                {
                    "dropped": scan["dropped_records"],
                    "max": contract.max_dropped_records,
                },
            )
        )
    else:
        checks.append(
            CheckResult(
                "dropped_records",
                "pass",
                {"dropped": scan["dropped_records"]},
            )
        )

    if scan["malformed_lines"]:
        checks.append(
            CheckResult(
                "malformed_events",
                "fail",
                {"malformed_lines": scan["malformed_lines"]},
            )
        )
    else:
        checks.append(
            CheckResult("malformed_events", "pass", {"malformed_lines": 0})
        )

    if scan["invalid_kernels"]:
        checks.append(
            CheckResult(
                "invalid_kernel_timestamps",
                "fail",
                {"invalid_kernels": scan["invalid_kernels"]},
            )
        )
    else:
        checks.append(
            CheckResult("invalid_kernel_timestamps", "pass", {"invalid_kernels": 0})
        )

    if scan["truncated_names"] and not contract.allow_truncated_names:
        checks.append(
            CheckResult(
                "truncated_names",
                "fail",
                {"truncated_names": scan["truncated_names"], "name_max": NAME_MAX},
            )
        )
    else:
        checks.append(
            CheckResult(
                "truncated_names",
                "pass",
                {"truncated_names": scan["truncated_names"]},
            )
        )

    if scan["truncated_shard"]:
        checks.append(
            CheckResult(
                "truncated_shard",
                "fail",
                {"note": "eof_mid_line_or_meta_truncated", "path": str(trace_path)},
            )
        )
    else:
        checks.append(
            CheckResult("truncated_shard", "pass", {"truncated_shard": False})
        )

    if scan["duplicate_event_keys"]:
        checks.append(
            CheckResult(
                "duplicate_events",
                "fail",
                {"duplicate_event_keys": scan["duplicate_event_keys"]},
            )
        )
    else:
        checks.append(
            CheckResult("duplicate_events", "pass", {"duplicate_event_keys": 0})
        )

    # Provenance — never invent rank/node/shard from event order.
    if contract.require_pid_on_kernels and scan["kernels"]:
        if scan["missing_pid_kernels"]:
            checks.append(
                CheckResult(
                    "pid_provenance",
                    "fail",
                    {"scopes_missing_pid": scan["missing_pid_kernels"]},
                )
            )
        else:
            checks.append(
                CheckResult("pid_provenance", "pass", {"pids": scan["pids"]})
            )
    elif contract.require_pid_on_kernels:
        checks.append(
            CheckResult("pid_provenance", "unknown", {"note": "no_kernels"})
        )

    if contract.require_device_on_kernels and scan["kernels"]:
        if scan["missing_device_kernels"]:
            checks.append(
                CheckResult(
                    "device_provenance",
                    "fail",
                    {"scopes_missing_device": scan["missing_device_kernels"]},
                )
            )
        else:
            checks.append(
                CheckResult("device_provenance", "pass", {"devices": scan["devices"]})
            )
    elif contract.require_device_on_kernels:
        checks.append(
            CheckResult("device_provenance", "unknown", {"note": "no_kernels"})
        )

    if contract.require_rank_on_kernels:
        if not scan["kernels"]:
            checks.append(CheckResult("rank_provenance", "unknown", {"note": "no_kernels"}))
        elif not scan["ranks"]:
            checks.append(
                CheckResult(
                    "rank_provenance",
                    "unknown",
                    {"note": "rank_not_on_wire"},
                )
            )
        else:
            checks.append(
                CheckResult("rank_provenance", "pass", {"ranks": scan["ranks"]})
            )
    else:
        checks.append(
            CheckResult(
                "rank_provenance",
                "pass",
                {
                    "ranks": scan["ranks"],
                    "note": "on_wire" if scan["ranks"] else "optional_in_v0",
                },
            )
        )

    if contract.require_node_on_kernels:
        if not scan["kernels"]:
            checks.append(CheckResult("node_provenance", "unknown", {"note": "no_kernels"}))
        elif not scan["nodes"]:
            checks.append(
                CheckResult("node_provenance", "unknown", {"note": "node_not_on_wire"})
            )
        else:
            checks.append(
                CheckResult("node_provenance", "pass", {"nodes": scan["nodes"]})
            )
    else:
        checks.append(
            CheckResult(
                "node_provenance",
                "pass",
                {
                    "nodes": scan["nodes"],
                    "note": "on_wire" if scan["nodes"] else "optional_in_v0",
                },
            )
        )

    # Shard path is discarded at merge; inventory is checked under topology_shards.
    if scan["shards"]:
        checks.append(
            CheckResult("shard_provenance", "pass", {"shards": scan["shards"]})
        )
    else:
        checks.append(
            CheckResult(
                "shard_provenance",
                "pass",
                {
                    "note": "shard_path_discarded_at_merge_inventory_via_pid",
                    "pids": scan["pids"],
                },
            )
        )

    if scan["kernels"]:
        conflict = any(p is None for p in scan["pids"]) and any(
            p is not None for p in scan["pids"]
        )
        if conflict:
            checks.append(
                CheckResult(
                    "provenance_consistency",
                    "fail",
                    {"pids": scan["pids"], "note": "mixed_null_pid"},
                )
            )
        else:
            checks.append(
                CheckResult(
                    "provenance_consistency",
                    "pass",
                    {"pids": scan["pids"], "devices": scan["devices"]},
                )
            )

    # Deployment identity from MANIFEST / header / sidecars
    manifest = _load_observed_manifest(artifacts_dir)
    sidecar = _serving_sidecar(artifacts_dir)
    observed_model = (
        header.get("fingerprint")
        or (trace.fingerprint if trace else None)
        or sidecar.get("model")
        or sidecar.get("model_id")
    )
    checks.append(
        _cmp("model_repository", deployment.model_repository, observed_model)
    )

    observed_rev = (
        sidecar.get("checkpoint_revision")
        or sidecar.get("revision")
        or manifest.get("checkpoint_revision")
    )
    checks.append(
        _cmp(
            "checkpoint_revision",
            deployment.checkpoint_revision,
            observed_rev,
            required=identity_required,
        )
    )

    observed_engine = (
        sidecar.get("engine")
        or manifest.get("engine")
        or ("vllm" if manifest.get("vllm") or sidecar.get("vllm_version") else None)
    )
    checks.append(
        _cmp(
            "engine_identity",
            deployment.engine,
            observed_engine,
            required=identity_required,
        )
    )

    observed_engine_ver = manifest.get("vllm") or sidecar.get("vllm_version")
    checks.append(
        _cmp(
            "engine_version",
            deployment.engine_version,
            observed_engine_ver,
            required=identity_required,
        )
    )

    observed_rocm = manifest.get("rocm") or sidecar.get("rocm_version")
    checks.append(
        _cmp(
            "rocm_version",
            deployment.rocm_version,
            observed_rocm,
            required=identity_required,
        )
    )

    observed_digest = sidecar.get("image_digest") or manifest.get("image_digest")
    checks.append(
        _cmp(
            "image_digest",
            deployment.image_digest,
            observed_digest,
            required=identity_required,
        )
    )

    # Topology inventory
    topo = deployment.topology
    expected_gpus = topo.gpus
    observed_gpu_count = len([d for d in scan["devices"] if d is not None])
    header_devices = header.get("device_count") or (trace.device_count if trace else None)

    if scan["kernels"] == 0:
        checks.append(
            CheckResult(
                "topology_gpus",
                "unknown",
                {"declared_gpus": expected_gpus, "observed_devices": []},
            )
        )
    elif observed_gpu_count < expected_gpus:
        if isinstance(header_devices, int) and header_devices >= expected_gpus:
            checks.append(
                CheckResult(
                    "topology_gpus",
                    "pass",
                    {
                        "declared_gpus": expected_gpus,
                        "header_device_count": header_devices,
                        "observed_device_ids": scan["devices"],
                    },
                )
            )
        else:
            checks.append(
                CheckResult(
                    "topology_gpus",
                    "fail",
                    {
                        "declared_gpus": expected_gpus,
                        "observed_device_ids": scan["devices"],
                        "header_device_count": header_devices,
                        "note": "missing_device_or_rank",
                    },
                )
            )
    else:
        checks.append(
            CheckResult(
                "topology_gpus",
                "pass",
                {
                    "declared_gpus": expected_gpus,
                    "observed_device_ids": scan["devices"],
                },
            )
        )

    expected_procs = topo.processes if topo.processes is not None else expected_gpus
    expected_ranks = topo.ranks if topo.ranks is not None else expected_procs
    observed_pids = [p for p in scan["pids"] if p is not None]
    if scan["kernels"] == 0:
        checks.append(
            CheckResult(
                "topology_processes",
                "unknown",
                {"declared": expected_procs, "observed_pids": []},
            )
        )
        checks.append(
            CheckResult(
                "topology_ranks",
                "unknown",
                {"declared": expected_ranks, "observed_ranks": scan["ranks"]},
            )
        )
        checks.append(
            CheckResult(
                "topology_shards",
                "unknown",
                {"declared": expected_procs, "observed_shards": scan["shards"]},
            )
        )
    else:
        if len(observed_pids) < expected_procs:
            checks.append(
                CheckResult(
                    "topology_processes",
                    "fail",
                    {
                        "declared_processes": expected_procs,
                        "observed_pids": observed_pids,
                        "note": "missing_rank_or_process",
                    },
                )
            )
        else:
            checks.append(
                CheckResult(
                    "topology_processes",
                    "pass",
                    {
                        "declared_processes": expected_procs,
                        "observed_pids": observed_pids,
                    },
                )
            )

        # Ranks on-wire when present; otherwise inventory-only via pid count
        # (never assign rank ordinals from event order).
        if scan["ranks"]:
            if len(scan["ranks"]) < expected_ranks:
                checks.append(
                    CheckResult(
                        "topology_ranks",
                        "fail",
                        {
                            "declared_ranks": expected_ranks,
                            "observed_ranks": scan["ranks"],
                        },
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        "topology_ranks",
                        "pass",
                        {
                            "declared_ranks": expected_ranks,
                            "observed_ranks": scan["ranks"],
                        },
                    )
                )
        elif len(observed_pids) < expected_ranks:
            checks.append(
                CheckResult(
                    "topology_ranks",
                    "fail",
                    {
                        "declared_ranks": expected_ranks,
                        "observed_pids": observed_pids,
                        "note": "missing_rank_inventory",
                    },
                )
            )
        else:
            checks.append(
                CheckResult(
                    "topology_ranks",
                    "pass",
                    {
                        "declared_ranks": expected_ranks,
                        "observed_pids_as_inventory": observed_pids,
                        "note": "rank_ids_not_on_wire_inventory_only",
                    },
                )
            )

        # Shards: on-wire shard ids, else pid count as inventory proxy only when
        # equal to declared — still unknown for path provenance (above).
        if scan["shards"]:
            if len(scan["shards"]) < expected_procs:
                checks.append(
                    CheckResult(
                        "topology_shards",
                        "fail",
                        {
                            "declared_shards": expected_procs,
                            "observed_shards": scan["shards"],
                        },
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        "topology_shards",
                        "pass",
                        {
                            "declared_shards": expected_procs,
                            "observed_shards": scan["shards"],
                        },
                    )
                )
        elif len(observed_pids) >= expected_procs:
            checks.append(
                CheckResult(
                    "topology_shards",
                    "pass",
                    {
                        "declared_shards": expected_procs,
                        "observed_pids_as_shard_inventory": observed_pids,
                        "note": "shard_files_merged",
                    },
                )
            )
        else:
            checks.append(
                CheckResult(
                    "topology_shards",
                    "fail",
                    {
                        "declared_shards": expected_procs,
                        "observed_pids": observed_pids,
                        "note": "missing_shard",
                    },
                )
            )

    # Nodes: single-node default can pass without on-wire node ids.
    if topo.nodes <= 1:
        checks.append(
            CheckResult(
                "topology_nodes",
                "pass",
                {"declared_nodes": topo.nodes, "observed_nodes": scan["nodes"]},
            )
        )
    elif scan["nodes"]:
        if len(scan["nodes"]) < topo.nodes:
            checks.append(
                CheckResult(
                    "topology_nodes",
                    "fail",
                    {
                        "declared_nodes": topo.nodes,
                        "observed_nodes": scan["nodes"],
                    },
                )
            )
        else:
            checks.append(
                CheckResult(
                    "topology_nodes",
                    "pass",
                    {
                        "declared_nodes": topo.nodes,
                        "observed_nodes": scan["nodes"],
                    },
                )
            )
    else:
        checks.append(
            CheckResult(
                "topology_nodes",
                "unknown",
                {
                    "declared_nodes": topo.nodes,
                    "observed_nodes": [],
                    "note": "node_not_on_wire_not_inferred",
                },
            )
        )

    exports = manifest.get("exports") if isinstance(manifest.get("exports"), dict) else {}
    observed_tp = _extract_flag_int(
        exports, deployment.launch_flags, "tensor-parallel-size", "tp"
    )
    if observed_tp is None:
        observed_tp = sidecar.get("tp") or manifest.get("tp")
    checks.append(
        _cmp(
            "topology_tp",
            topo.tp,
            observed_tp,
            required=identity_required,
        )
    )
    observed_ep = _extract_flag_int(
        exports, deployment.launch_flags, "enable-expert-parallel", "ep"
    )
    if observed_ep is None:
        observed_ep = sidecar.get("ep") or manifest.get("ep")
    checks.append(
        _cmp(
            "topology_ep",
            topo.ep,
            observed_ep if isinstance(observed_ep, int) else None,
            required=False,
        )
    )
    observed_dp = sidecar.get("dp") or manifest.get("dp")
    checks.append(
        _cmp(
            "topology_dp",
            topo.dp,
            observed_dp if isinstance(observed_dp, int) else None,
            required=False,
        )
    )

    observed_workload = header.get("workload_id") or (trace.workload_id if trace else None)
    checks.append(
        _cmp(
            "workload_id",
            deployment.workload_id,
            observed_workload,
            required=identity_required,
        )
    )
    observed_traffic = (
        manifest.get("traffic_manifest_id")
        or sidecar.get("traffic_manifest_id")
        or manifest.get("label")
    )
    checks.append(
        _cmp(
            "traffic_manifest_id",
            deployment.traffic_manifest_id,
            observed_traffic,
            required=identity_required,
        )
    )

    observed_source = header.get("source") or (trace.source if trace else None)
    expected_source = {
        "rocprof-inject": "rocprof",
        "cupti-inject": "cupti",
    }.get(deployment.capture_backend)
    if expected_source and observed_source:
        if observed_source != expected_source:
            checks.append(
                CheckResult(
                    "trace_source",
                    "fail",
                    {
                        "declared_backend": deployment.capture_backend,
                        "source": observed_source,
                    },
                )
            )
        else:
            checks.append(
                CheckResult("trace_source", "pass", {"source": observed_source})
            )
    elif expected_source:
        checks.append(
            CheckResult(
                "trace_source",
                "unknown",
                {
                    "declared_backend": deployment.capture_backend,
                    "source": observed_source,
                },
            )
        )

    extra = str(exports.get("GITM_EXTRA_VLLM_ARGS", "") or "")
    if not deployment.launch_flags:
        checks.append(
            CheckResult(
                "launch_flags",
                "pass",
                {"declared": [], "note": "none_required"},
            )
        )
    elif not exports and not sidecar.get("launch_flags"):
        checks.append(
            CheckResult(
                "launch_flags",
                "unknown",
                {"declared": deployment.launch_flags, "observed_extra": None},
            )
        )
    else:
        observed_blob = extra or " ".join(
            str(x) for x in (sidecar.get("launch_flags") or [])
        )
        missing_obs = [f for f in deployment.launch_flags if f not in observed_blob]
        if missing_obs:
            checks.append(
                CheckResult(
                    "launch_flags",
                    "fail",
                    {
                        "declared": deployment.launch_flags,
                        "observed_extra": observed_blob,
                        "missing": missing_obs,
                    },
                )
            )
        else:
            checks.append(
                CheckResult(
                    "launch_flags",
                    "pass",
                    {
                        "declared": deployment.launch_flags,
                        "observed_extra": observed_blob,
                    },
                )
            )

    # Capture-window coverage (tool-clock domain).
    duration = header.get("duration_ns") or (trace.duration_ns if trace else None)
    if contract.require_capture_window:
        if not isinstance(duration, int) or duration <= 0:
            checks.append(
                CheckResult(
                    "capture_window_coverage",
                    "unknown",
                    {"duration_ns": duration, "note": "window_unobservable"},
                )
            )
        elif scan["kernels"] == 0:
            checks.append(
                CheckResult(
                    "capture_window_coverage",
                    "unknown",
                    {"duration_ns": duration, "note": "empty_capture"},
                )
            )
        else:
            max_end = scan["max_end_ns"]
            # Events filtered to the window at merge; still reject clearly outside.
            if isinstance(max_end, int) and max_end > duration * 2:
                checks.append(
                    CheckResult(
                        "capture_window_coverage",
                        "fail",
                        {
                            "duration_ns": duration,
                            "max_end_ns": max_end,
                            "note": "events_outside_declared_window",
                        },
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        "capture_window_coverage",
                        "pass",
                        {
                            "duration_ns": duration,
                            "min_start_ns": scan["min_start_ns"],
                            "max_end_ns": max_end,
                            "kernels": scan["kernels"],
                        },
                    )
                )

    # Cross-source clock alignment / cadence. Absent state plane is unknown only
    # when that source is required; otherwise not applicable for this contract.
    wall_start, wall_end = _wall_window_ns(header, sidecar, manifest, trace)
    amdsmi_path = source_paths.get("amdsmi") or _find_source_path(artifacts_dir, "amdsmi")
    metrics_path = source_paths.get("metrics") or _find_source_path(artifacts_dir, "metrics")
    state_path = amdsmi_path or metrics_path
    state_required = any(s in contract.required_sources for s in ("amdsmi", "metrics"))
    if state_path is None:
        status: CheckStatus = "unknown" if state_required else "pass"
        note = "no_state_plane_source" if state_required else "state_plane_not_required"
        checks.append(CheckResult("clock_alignment", status, {"note": note}))
        checks.append(CheckResult("sampling_cadence", status, {"note": note}))
        checks.append(CheckResult("sampling_gaps", status, {"note": note}))
    else:
        stamps = _parse_ts_ns_stream(state_path)
        if len(stamps) < 2:
            checks.append(
                CheckResult(
                    "clock_alignment",
                    "unknown",
                    {
                        "source": str(state_path),
                        "samples": len(stamps),
                        "note": "insufficient_samples",
                    },
                )
            )
            checks.append(
                CheckResult(
                    "sampling_cadence",
                    "unknown",
                    {"samples": len(stamps), "note": "insufficient_samples"},
                )
            )
            checks.append(
                CheckResult(
                    "sampling_gaps",
                    "unknown",
                    {"samples": len(stamps), "note": "insufficient_samples"},
                )
            )
        else:
            if wall_start is None or wall_end is None:
                checks.append(
                    CheckResult(
                        "clock_alignment",
                        "unknown",
                        {
                            "note": "wall_window_unobservable_not_inferred",
                            "state_source": str(state_path),
                            "sample_span_ns": [stamps[0], stamps[-1]],
                        },
                    )
                )
            else:
                # Require state samples to overlap the declared wall window.
                overlap = stamps[-1] >= wall_start and stamps[0] <= wall_end
                skew = max(
                    abs(stamps[0] - wall_start),
                    abs(stamps[-1] - wall_end),
                )
                if not overlap:
                    checks.append(
                        CheckResult(
                            "clock_alignment",
                            "fail",
                            {
                                "wall_window_ns": [wall_start, wall_end],
                                "state_span_ns": [stamps[0], stamps[-1]],
                                "note": "no_overlap",
                            },
                        )
                    )
                elif skew > contract.max_clock_skew_ns:
                    checks.append(
                        CheckResult(
                            "clock_alignment",
                            "fail",
                            {
                                "wall_window_ns": [wall_start, wall_end],
                                "state_span_ns": [stamps[0], stamps[-1]],
                                "skew_ns": skew,
                                "max_clock_skew_ns": contract.max_clock_skew_ns,
                            },
                        )
                    )
                else:
                    checks.append(
                        CheckResult(
                            "clock_alignment",
                            "pass",
                            {
                                "wall_window_ns": [wall_start, wall_end],
                                "state_span_ns": [stamps[0], stamps[-1]],
                                "skew_ns": skew,
                            },
                        )
                    )

            deltas = [b - a for a, b in zip(stamps, stamps[1:]) if b >= a]
            if not deltas:
                checks.append(
                    CheckResult(
                        "sampling_cadence",
                        "fail",
                        {"note": "non_monotonic_timestamps"},
                    )
                )
                checks.append(
                    CheckResult(
                        "sampling_gaps",
                        "fail",
                        {"note": "non_monotonic_timestamps"},
                    )
                )
            else:
                expected = contract.expected_sample_period_ns
                median = sorted(deltas)[len(deltas) // 2]
                # Cadence: median near expected (within 50%).
                if abs(median - expected) > expected * 0.5:
                    checks.append(
                        CheckResult(
                            "sampling_cadence",
                            "fail",
                            {
                                "median_delta_ns": median,
                                "expected_period_ns": expected,
                            },
                        )
                    )
                else:
                    checks.append(
                        CheckResult(
                            "sampling_cadence",
                            "pass",
                            {
                                "median_delta_ns": median,
                                "expected_period_ns": expected,
                                "samples": len(stamps),
                            },
                        )
                    )
                gap_limit = int(expected * contract.max_sample_gap_factor)
                big_gaps = [d for d in deltas if d > gap_limit]
                if big_gaps:
                    checks.append(
                        CheckResult(
                            "sampling_gaps",
                            "fail",
                            {
                                "gap_count": len(big_gaps),
                                "max_gap_ns": max(big_gaps),
                                "limit_ns": gap_limit,
                            },
                        )
                    )
                else:
                    checks.append(
                        CheckResult(
                            "sampling_gaps",
                            "pass",
                            {"max_gap_ns": max(deltas), "limit_ns": gap_limit},
                        )
                    )

    return QualifyRunResult(
        verdict=_aggregate(checks),
        checks=checks,
        deployment_fingerprint=deployment_fingerprint(deployment, contract),
        capture_fingerprint=_capture_fingerprint(scan, trace),
        signal_contract_version=contract.version,
    )


def _extract_flag_int(
    exports: dict[str, Any],
    launch_flags: list[str],
    long_name: str,
    short_key: str,
) -> int | None:
    extra = str(exports.get("GITM_EXTRA_VLLM_ARGS", "") or "")
    blob = extra + " " + " ".join(launch_flags)
    m = re.search(rf"--{re.escape(long_name)}[= ]+(\d+)", blob)
    if m:
        return int(m.group(1))
    _ = short_key
    return None


def _aggregate(checks: list[CheckResult]) -> Verdict:
    if any(c.status == "fail" for c in checks):
        return "invalid_run"
    if any(c.status == "unknown" for c in checks):
        return "not_established"
    return "qualified"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifacts",
        type=Path,
        required=True,
        help="Capture directory, run directory, or a trace.jsonl path.",
    )
    ap.add_argument(
        "--deployment-spec",
        type=Path,
        required=True,
        help="Declared deployment JSON (Zhu / operator).",
    )
    ap.add_argument(
        "--signal-contract",
        type=Path,
        default=None,
        help="Signal contract JSON (default: provisional v0 built-in).",
    )
    ap.add_argument(
        "--signal-contract-version",
        default=None,
        help="If set to 'v0', use the built-in provisional contract.",
    )
    args = ap.parse_args(argv)

    try:
        deployment = load_deployment_spec(args.deployment_spec)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({
            "verdict": "not_established",
            "checks": [{
                "name": "deployment_spec",
                "status": "unknown",
                "evidence": {"error": str(exc)},
            }],
            "deployment_fingerprint": "deploy:absent",
            "capture_fingerprint": "capture:absent",
            "signal_contract_version": PROVISIONAL_SIGNAL_CONTRACT_VERSION,
            "qualify_run_revision": QUALIFY_RUN_REVISION,
        }, indent=2))
        return 2

    if args.signal_contract_version == "v0" and args.signal_contract is None:
        contract = default_signal_contract()
    else:
        try:
            contract = load_signal_contract(args.signal_contract)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(json.dumps({
                "verdict": "not_established",
                "checks": [{
                    "name": "signal_contract",
                    "status": "unknown",
                    "evidence": {"error": str(exc)},
                }],
                "deployment_fingerprint": deployment_fingerprint(
                    deployment, default_signal_contract()
                ),
                "capture_fingerprint": "capture:absent",
                "signal_contract_version": PROVISIONAL_SIGNAL_CONTRACT_VERSION,
                "qualify_run_revision": QUALIFY_RUN_REVISION,
            }, indent=2))
            return 2

    result = qualify_run(args.artifacts, deployment, contract)
    print(json.dumps(result.to_dict(), indent=2))
    if result.verdict == "qualified":
        return 0
    if result.verdict == "invalid_run":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
