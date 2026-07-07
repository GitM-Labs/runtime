"""Deploy invariants are enforced as code: no root, no host escape, no phone-home."""

from __future__ import annotations

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent


def _load_docs(path: Path):
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def test_k8s_operator_is_least_privilege():
    docs = _load_docs(_ROOT / "deploy" / "k8s" / "operator.yaml")
    ds = next(d for d in docs if d.get("kind") == "DaemonSet")
    spec = ds["spec"]["template"]["spec"]

    # no host escape, no SA token (no API / out-of-cluster reach)
    assert spec.get("hostNetwork", False) is False
    assert spec.get("hostPID", False) is False
    assert spec["automountServiceAccountToken"] is False
    assert spec["securityContext"]["runAsNonRoot"] is True

    container = spec["containers"][0]
    sc = container["securityContext"]
    assert sc["allowPrivilegeEscalation"] is False
    assert sc["readOnlyRootFilesystem"] is True
    assert sc["capabilities"]["drop"] == ["ALL"]
    # GPU via the device-plugin resource, never a privileged host mount
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1


def test_no_clusterrole_binding_grants_api_access():
    docs = _load_docs(_ROOT / "deploy" / "k8s" / "operator.yaml")
    kinds = {d.get("kind") for d in docs}
    assert "ClusterRoleBinding" not in kinds and "RoleBinding" not in kinds


def test_container_runs_as_non_root_user():
    text = (_ROOT / "Dockerfile").read_text()
    # a USER directive that isn't root, appearing after the build steps
    user_lines = [ln for ln in text.splitlines() if ln.strip().startswith("USER ")]
    assert user_lines, "Dockerfile must drop to a non-root USER"
    assert all("root" not in ln for ln in user_lines)
