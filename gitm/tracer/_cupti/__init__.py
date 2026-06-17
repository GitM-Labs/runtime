"""CUPTI native shim package.

Exposes :func:`load_shim`, which imports the compiled ``_cupti_shim`` extension
if it was built (see :mod:`gitm.tracer._cupti.build`). When the extension is
absent on a GPU box, ``load_shim`` attempts a one-time auto-build so that
``pip install`` + ``gitm run`` works with no manual build step; on a CPU-only
host (or if the build can't find the CUDA toolchain) it returns ``None`` and
the tracer degrades to a well-formed no-op.

Auto-build is best-effort and can be disabled with ``GITM_AUTOBUILD_CUPTI=0``.
"""

from __future__ import annotations

import importlib
import os
import shutil
import threading
from types import ModuleType

_BUILD_ATTEMPTED = False
_BUILD_LOCK = threading.Lock()


def _import_shim() -> ModuleType | None:
    try:
        from gitm.tracer._cupti import _cupti_shim  # type: ignore[attr-defined]
    except Exception:
        return None
    return _cupti_shim


def _maybe_autobuild() -> bool:
    """One-time best-effort shim build. Returns True if a build was attempted.

    Skipped when already tried this process, disabled via env, or not on a GPU
    box (no ``nvidia-smi`` — nothing to compile against)."""
    global _BUILD_ATTEMPTED
    if os.environ.get("GITM_AUTOBUILD_CUPTI", "1") == "0":
        return False
    if shutil.which("nvidia-smi") is None:
        return False
    # Serialize so concurrent callers (e.g. a parallel test runner) don't race
    # to compile the same .so. Double-checked: re-test the flag under the lock.
    with _BUILD_LOCK:
        if _BUILD_ATTEMPTED:
            return False
        _BUILD_ATTEMPTED = True
        try:
            from gitm.tracer._cupti.build import build

            build()
        except (SystemExit, Exception):  # missing toolchain / compile failure → degrade
            pass
        importlib.invalidate_caches()  # so the freshly built .so is discoverable
    return True


def load_shim() -> ModuleType | None:
    """Return the compiled CUPTI shim module, or ``None`` if unavailable.

    Imports the prebuilt extension if present; otherwise tries a one-time
    auto-build (GPU boxes only) and imports again."""
    shim = _import_shim()
    if shim is not None:
        return shim
    if _maybe_autobuild():
        return _import_shim()
    return None


def available() -> bool:
    return load_shim() is not None
