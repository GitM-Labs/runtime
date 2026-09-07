"""ROCm collection support — the AMD half of the injected tracer.

The collector itself is ``rocm_inject.c`` -> ``libgitm_rocm_inject.so`` (build
with ``python -m gitm.tracer._rocm.build``), loaded into every HIP process via
``ROCP_TOOL_LIBRARIES``. This module is the small host-side surface the
capture pipeline needs without initializing HIP:

* :func:`timestamp` — a reading of the SAME clock rocprofiler-sdk stamps
  activity records with, which is what makes the capture window filter correct
  across processes (the exact role ``gitm_cupti_timestamp`` plays on NVIDIA).
  Read through ``librocprofiler-sdk``'s own ``rocprofiler_get_timestamp`` via
  ctypes rather than reimplemented from its definition, so a change in the
  sdk's clock source can never split the domains silently.
* :func:`device_count` — GPU agents from the kfd sysfs topology. No HIP init,
  so it is safe to call while the injected library owns collection.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

LIB_NAME = "libgitm_rocm_inject.so"

_ROCM_HOMES = ("/opt/rocm",)  # versioned installs symlink /opt/rocm -> rocm-X.Y


def lib_path() -> Path:
    """Where the injection tool is built, whether or not it exists yet."""
    return Path(__file__).resolve().parent / LIB_NAME


def _sdk_lib_candidates() -> list[Path]:
    homes = [Path(os.environ["ROCM_PATH"])] if os.environ.get("ROCM_PATH") else []
    homes += [Path(h) for h in _ROCM_HOMES]
    out = []
    for h in homes:
        out += [h / "lib" / "librocprofiler-sdk.so", h / "lib64" / "librocprofiler-sdk.so"]
    return out


_sdk_handle: ctypes.CDLL | None = None
_sdk_tried = False


def _sdk() -> ctypes.CDLL | None:
    """librocprofiler-sdk, loaded once. Loading it does NOT register a tool —
    registration only happens for libraries listed in ROCP_TOOL_LIBRARIES — so
    this cannot fight the injected collector."""
    global _sdk_handle, _sdk_tried
    if _sdk_tried:
        return _sdk_handle
    _sdk_tried = True
    names = [str(p) for p in _sdk_lib_candidates() if p.exists()]
    found = ctypes.util.find_library("rocprofiler-sdk")
    if found:
        names.append(found)
    for name in names:
        try:
            _sdk_handle = ctypes.CDLL(name)
            break
        except OSError:
            continue
    return _sdk_handle


def timestamp() -> int | None:
    """Nanoseconds in the rocprofiler record clock domain, or ``None``."""
    sdk = _sdk()
    if sdk is None:
        return None
    try:
        fn = sdk.rocprofiler_get_timestamp
    except AttributeError:
        return None
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    ts = ctypes.c_uint64(0)
    if fn(ctypes.byref(ts)) != 0:  # ROCPROFILER_STATUS_SUCCESS == 0
        return None
    return ts.value or None


def device_count() -> int:
    """GPU count from the kfd topology; 0 on a non-ROCm host.

    A kfd node is a GPU iff its ``gpu_id`` is non-zero — CPU sockets appear as
    nodes too, with ``gpu_id`` 0.
    """
    nodes = Path("/sys/class/kfd/kfd/topology/nodes")
    if not nodes.is_dir():
        return 0
    count = 0
    for node in nodes.iterdir():
        gpu_id = node / "gpu_id"
        try:
            if gpu_id.read_text().strip() not in ("", "0"):
                count += 1
        except OSError:
            continue
    return count
