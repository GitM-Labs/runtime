"""Declared deployment identity and provisional signal contract for qualify_run.

Aravinth owns the real signal-contract versioning. Until that lands, Nathan ships
``signal_contract.v0`` so runs can be verified against an explicit checklist.
Zhu owns the authoritative deployment specification (checkpoint, engine, topology);
operators supply it as JSON — nothing here invents missing identity from event order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROVISIONAL_SIGNAL_CONTRACT_VERSION = "v0"

CaptureBackend = Literal["rocprof-inject", "cupti-inject", "unknown"]


class TopologySpec(BaseModel):
    """Expected process / device / rank layout."""

    model_config = ConfigDict(extra="forbid")

    nodes: int = 1
    gpus: int
    processes: int | None = None  # defaults to gpus when unset
    ranks: int | None = None
    tp: int | None = None
    ep: int | None = None
    dp: int | None = None


class DeploymentSpec(BaseModel):
    """What the operator / Zhu declares the run *should* be.

    Fields that cannot be observed in artifacts remain comparable only when
    present here; absence of required evidence yields ``not_established``, never
    a silent inference.
    """

    model_config = ConfigDict(extra="forbid")

    model_repository: str
    checkpoint_revision: str | None = None
    engine: str = "vllm"
    engine_version: str | None = None
    rocm_version: str | None = None
    image_digest: str | None = None
    launch_flags: list[str] = Field(default_factory=list)
    topology: TopologySpec
    workload_id: str | None = None
    traffic_manifest_id: str | None = None
    capture_backend: CaptureBackend = "rocprof-inject"
    signal_contract_version: str = PROVISIONAL_SIGNAL_CONTRACT_VERSION


class SignalContract(BaseModel):
    """Provisional v0 checklist of required observations for a qualified run."""

    model_config = ConfigDict(extra="forbid")

    version: str = PROVISIONAL_SIGNAL_CONTRACT_VERSION
    capture_backend: CaptureBackend = "rocprof-inject"
    # MI355X bring-up expects amd-smi on the run; missing state plane must not
    # silently pass clock/cadence checks.
    required_sources: list[str] = Field(
        default_factory=lambda: ["merged_trace", "amdsmi"],
    )
    require_kernel_events: bool = True
    require_named_kernels: bool = True
    max_dropped_records: int = 0
    require_pid_on_kernels: bool = True
    require_device_on_kernels: bool = True
    require_rank_on_kernels: bool = False  # not on wire yet; unknown until observable
    require_node_on_kernels: bool = False
    allow_truncated_names: bool = False
    require_nonempty_header: bool = True
    require_capture_window: bool = True
    # State-plane cadence (amd-smi / metrics). Applied only when that source exists.
    expected_sample_period_ns: int = 1_000_000_000
    max_sample_gap_factor: float = 2.5
    # Wall-clock vs device-clock alignment needs an explicit wall window on disk.
    max_clock_skew_ns: int = 5_000_000_000
    # Identity that must be declared + observed for ``qualified`` (never inferred).
    require_deployment_identity: bool = True


def default_signal_contract() -> SignalContract:
    return SignalContract()


def load_deployment_spec(path: Path) -> DeploymentSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    return DeploymentSpec.model_validate(data)


def load_signal_contract(path: Path | None) -> SignalContract:
    if path is None:
        return default_signal_contract()
    data = json.loads(path.read_text(encoding="utf-8"))
    return SignalContract.model_validate(data)


def deployment_fingerprint(spec: DeploymentSpec, contract: SignalContract) -> str:
    """Immutable digest of declared deployment + contract version."""
    import hashlib

    payload = {
        "deployment": spec.model_dump(mode="json"),
        "signal_contract_version": contract.version,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"deploy:{digest}"


def parse_run_manifest(text: str) -> dict[str, Any]:
    """Parse ``$RUN/MANIFEST`` append-only lines into the latest observed keys."""
    observed: dict[str, Any] = {"exports": {}, "phases": []}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("export "):
            body = line[len("export ") :]
            if "=" in body:
                k, _, v = body.partition("=")
                observed["exports"][k.strip()] = v.strip().strip('"')
            continue
        parts: dict[str, str] = {}
        for tok in line.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                parts[k] = v
        if "phase" in parts:
            observed["phases"].append(parts)
        for k, v in parts.items():
            if k in (
                "rocm",
                "vllm",
                "phase",
                "arm",
                "label",
                "ts",
                "image_digest",
                "checkpoint_revision",
                "traffic_manifest_id",
                "engine",
                "tp",
                "ep",
                "dp",
            ):
                observed[k] = v
                if k in ("tp", "ep", "dp"):
                    try:
                        observed[k] = int(v)
                    except ValueError:
                        pass
        if "shard_growth_outside_window" in parts:
            observed["shard_growth_outside_window"] = parts["shard_growth_outside_window"]
    return observed
