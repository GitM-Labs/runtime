"""Environment probe — ``gitm doctor``."""

from __future__ import annotations

import platform
import sys
from typing import Any

from gitm import __version__
from gitm._paths import s3_root, scratch_root


def doctor() -> dict[str, Any]:
    """Probe the runtime environment and return a JSON-able report."""
    info: dict[str, Any] = {
        "gitm_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "s3_root": s3_root(),  # canonical store; None if $GITM_S3_ROOT unset
        "scratch": str(scratch_root()),  # local ephemeral run dir
    }

    from gitm.telemetry.backends import discover_backends

    diagnostics: list[str] = []
    backends = discover_backends(diagnostics=diagnostics)
    telemetry_backends: list[dict[str, Any]] = []
    for backend in backends:
        try:
            telemetry_backends.append(
                {"vendor": backend.vendor, "device_count": backend.device_count()}
            )
        except Exception as exc:
            diagnostics.append(
                f"{backend.vendor} telemetry probe failed: {type(exc).__name__}: {exc}"
            )
        finally:
            try:
                backend.close()
            except Exception as exc:
                diagnostics.append(
                    f"{backend.vendor} telemetry close failed: {type(exc).__name__}: {exc}"
                )
    info["telemetry_backends"] = telemetry_backends
    info["telemetry_diagnostics"] = diagnostics
    return info
