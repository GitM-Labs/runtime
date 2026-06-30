"""Binary applicability gate: does this lever apply to this deployment?"""

from __future__ import annotations

from dataclasses import dataclass

from gitm.kernels.spec import InterventionSpec


@dataclass(frozen=True)
class GateContext:
    workload: str
    dtype: str | None = None
    hardware: str | None = None
    kv_cache_len: int | None = None
    num_gpus: int = 1
    has_collective: bool = False
    has_interconnect: bool = False


def applicable(spec: InterventionSpec, ctx: GateContext) -> tuple[bool, str]:
    """Return (True, "") if spec applies to ctx, else (False, reason)

    All conditions are and-ed, the first failure short circuits with its reason
    Unset spec conditions are treated as "no constraint"
    """
    app = spec.applicability
    if ctx.workload not in app.workloads:
        return False, f"workload {ctx.workload!r} not in {app.workloads}"

    if app.requires_dtype:
        if ctx.dtype is None:
            return False, f"Requires dtype in {app.requires_dtype} but run dtype is unknown"
        if ctx.dtype not in app.requires_dtype:
            return False, f"dtype {ctx.dtype!r} not in {app.requires_dtype}"

    if app.requires_hardware:
        hw = (ctx.hardware or "").lower()
        if not hw:
            return False, f"Requires hardware in {app.requires_hardware} but SKU is unknown"
        if not any (req.lower() in hw for req in app.requires_hardware):
            return False, f"Hardware {ctx.hardware!r} matches none of {app.requires_hardware}"

    if app.min_kv_cache_len is not None:
        if ctx.kv_cache_len is None:
            return False, f"Requires kv_cache_len >= {app.min_kv_cache_len} but is unknown"
        if ctx.kv_cache_len < app.min_kv_cache_len:
            return False, f"kv_cache_len {ctx.kv_cache_len} < min {app.min_kv_cache_len}"

    if app.max_kv_cache_len is not None and ctx.kv_cache_len is not None:
        if ctx.kv_cache_len > app.max_kv_cache_len:
            return False, f"kv_cache_len {ctx.kv_cache_len} > max {app.max_kv_cache_len}"

    # Forward-compatible: honor extended fields (min_gpus, requires_collective,
    # requires_interconnect) if a later Applicability revision adds them, without
    # hard-depending on them here.

    min_gpus = getattr(app, "min_gpus", None)
    if min_gpus is not None and ctx.num_gpus < min_gpus:
        return False, f"num_gpus {ctx.num_gpus} < min {min_gpus}"
    if getattr(app, "requires_collective", False) and not ctx.has_collective:
        return False, "requires a collective (multi-GPU) but run is single-GPU/no-collective"
    if getattr(app, "requires_interconnect", False) and not ctx.has_interconnect:
        return False, "requires interconnect (NVLink/IB) but none reported"

    return True, ""
