"""Vendor backend autodiscovery.

Initialize every known vendor backend; return only the live ones. A single
binary works unchanged on NVIDIA-only, AMD-only, and mixed-vendor nodes.
"""

from __future__ import annotations

import warnings

from gitm.telemetry.backends.base import Backend


def discover_backends(*, diagnostics: list[str] | None = None) -> list[Backend]:
    """Return all live vendor backends in discovery order.

    A missing optional vendor library or a valid zero-device backend is expected
    and stays quiet. Unexpected import, initialization, count, or cleanup failures
    are appended to ``diagnostics``; direct callers that do not supply a list get
    a runtime warning instead.
    """
    found: list[Backend] = []

    def record(vendor: str, detail: str) -> None:
        message = f"{vendor} telemetry discovery failed: {detail}"
        if diagnostics is not None:
            diagnostics.append(message)
        else:
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def attempt(vendor: str, factory) -> None:
        backend: Backend | None = None
        try:
            backend = factory()
            count = backend.device_count()
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"device_count returned invalid value {count!r}")
            if count > 0:
                found.append(backend)
                backend = None
        except ImportError:
            return
        except Exception as exc:
            record(vendor, f"{type(exc).__name__}: {exc}")
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    record(vendor, f"backend close failed ({type(exc).__name__}: {exc})")

    def nvidia():
        from gitm.telemetry.backends.nvidia import NvidiaBackend

        return NvidiaBackend()

    def amd():
        from gitm.telemetry.backends.amd import AmdBackend

        return AmdBackend()

    attempt("nvidia", nvidia)
    attempt("amd", amd)
    return found
