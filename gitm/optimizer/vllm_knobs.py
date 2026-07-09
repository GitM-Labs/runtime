"""vLLM knob taxonomy — where each library knob lives, and how it can be applied.

The intervention library (:mod:`gitm.kernels.library`) names knobs flat
(``max_num_seqs``, ``block_size``, …). A *live* vLLM engine keeps them on nested
config objects (``scheduler_config.max_num_seqs``, ``cache_config.block_size``,
``parallel_config.tensor_parallel_size``), and only a few are safe to mutate on a
running engine. This module is the single source of truth for both:

* **path** — the dotted location on the engine (tried under several prefixes so
  it survives vLLM laying configs out as ``engine.vllm_config.<cfg>`` vs
  ``engine.<cfg>`` across versions), or ``env:VLLM_*`` for env-var knobs.
* **kind** — ``"scheduling"`` (hot-swappable: takes effect next scheduler step)
  vs ``"structural"`` (requires an engine restart to take effect).

:class:`gitm.optimizer.apply.LiveEngineApplicator` uses this to hot-swap a
scheduling knob in place, and to route a structural knob through a restart hook
(or roll it back cleanly when no restart hook is available) — never to silently
set a structural field that the running engine won't actually honor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from gitm.kernels.spec import InterventionSpec

KnobKind = Literal["scheduling", "structural"]

# Prefixes a config object may hide behind, newest-first. ``""`` = the config is
# a direct attribute of the engine (older vLLM / our test doubles).
_PREFIXES = (
    "",
    "vllm_config.",
    "engine.vllm_config.",
    "llm_engine.vllm_config.",
    "engine.",
    "llm_engine.",
)


@dataclass(frozen=True)
class KnobSpec:
    """Where a flat library knob lives on the engine, and how it can be applied."""

    knob: str
    path: str  # dotted engine path, or "env:NAME" for an env-var knob
    kind: KnobKind

    @property
    def is_env(self) -> bool:
        return self.path.startswith("env:")


# The curated map. Only the three scheduler knobs vLLM honors mid-run are
# "scheduling"; everything else is fixed at construction → "structural".
_KNOBS: dict[str, KnobSpec] = {
    # --- scheduling: live-settable on the scheduler, effective next step -------
    "max_num_seqs": KnobSpec("max_num_seqs", "scheduler_config.max_num_seqs", "scheduling"),
    "max_num_batched_tokens": KnobSpec(
        "max_num_batched_tokens", "scheduler_config.max_num_batched_tokens", "scheduling"
    ),
    "scheduling_policy": KnobSpec("scheduling_policy", "scheduler_config.policy", "scheduling"),
    # --- structural: fixed at engine construction, need a restart to change ----
    "block_size": KnobSpec("block_size", "cache_config.block_size", "structural"),
    "kv_cache_dtype": KnobSpec("kv_cache_dtype", "cache_config.cache_dtype", "structural"),
    "gpu_memory_utilization": KnobSpec(
        "gpu_memory_utilization", "cache_config.gpu_memory_utilization", "structural"
    ),
    "swap_space": KnobSpec("swap_space", "cache_config.swap_space_bytes", "structural"),
    "enable_prefix_caching": KnobSpec(
        "enable_prefix_caching", "cache_config.enable_prefix_caching", "structural"
    ),
    "enable_chunked_prefill": KnobSpec(
        "enable_chunked_prefill", "scheduler_config.chunked_prefill_enabled", "structural"
    ),
    "num_speculative_tokens": KnobSpec(
        "num_speculative_tokens", "speculative_config.num_speculative_tokens", "structural"
    ),
    "enforce_eager": KnobSpec("enforce_eager", "model_config.enforce_eager", "structural"),
    "max_seq_len_to_capture": KnobSpec(
        "max_seq_len_to_capture", "model_config.max_seq_len_to_capture", "structural"
    ),
    "quantization": KnobSpec("quantization", "model_config.quantization", "structural"),
    "tensor_parallel_size": KnobSpec(
        "tensor_parallel_size", "parallel_config.tensor_parallel_size", "structural"
    ),
    "pipeline_parallel_size": KnobSpec(
        "pipeline_parallel_size", "parallel_config.pipeline_parallel_size", "structural"
    ),
    "disable_custom_all_reduce": KnobSpec(
        "disable_custom_all_reduce", "parallel_config.disable_custom_all_reduce", "structural"
    ),
    "distributed_executor_backend": KnobSpec(
        "distributed_executor_backend", "parallel_config.distributed_executor_backend", "structural"
    ),
    # Env-var knob: read by vLLM at construction → structural.
    "VLLM_ATTENTION_BACKEND": KnobSpec(
        "VLLM_ATTENTION_BACKEND", "env:VLLM_ATTENTION_BACKEND", "structural"
    ),
    # Prerequisite flags for the table below — not applied standalone, only read.
    "enable_dbo": KnobSpec("enable_dbo", "scheduler_config.enable_dbo", "structural"),
}


def resolve_knob(knob: str) -> KnobSpec | None:
    """The :class:`KnobSpec` for ``knob``, or ``None`` if it isn't in the taxonomy."""
    return _KNOBS.get(knob)


def _holder_and_leaf(engine: Any, path: str) -> tuple[Any, str] | None:
    """Resolve ``path`` on ``engine`` to ``(holder_object, leaf_attr)``.

    Tries each prefix in turn; returns the first where the parent chain resolves
    to a real object that has the leaf attribute. ``None`` if nothing matches.
    """
    leaf = path.rsplit(".", 1)[-1]
    rel_parents = path.rsplit(".", 1)[0] if "." in path else ""
    for prefix in _PREFIXES:
        full_parent = f"{prefix}{rel_parents}".strip(".")
        holder: Any = engine
        ok = True
        for attr in (a for a in full_parent.split(".") if a):
            holder = getattr(holder, attr, None)
            if holder is None:
                ok = False
                break
        if ok and holder is not None and hasattr(holder, leaf):
            return holder, leaf
    return None


def get_knob(engine: Any, knob: str) -> Any:
    """Read ``knob`` from the live engine via its taxonomy path.

    Falls back to a flat attribute on the engine when the knob isn't in the
    taxonomy or its structured path isn't present (duck-typing across versions
    and test doubles). Raises ``AttributeError`` if nothing resolves.
    """
    spec = resolve_knob(knob)
    if spec is not None and spec.is_env:
        return os.environ.get(spec.path.split(":", 1)[1])
    if spec is not None:
        hl = _holder_and_leaf(engine, spec.path)
        if hl is not None:
            holder, leaf = hl
            return getattr(holder, leaf)
    # Flat fallback.
    if hasattr(engine, knob):
        return getattr(engine, knob)
    raise AttributeError(f"engine has no knob {knob!r} (taxonomy path or flat attr)")


def set_knob(engine: Any, knob: str, value: Any) -> None:
    """Set ``knob`` on the live engine via its taxonomy path (scheduling knobs).

    Raises ``AttributeError`` when the knob can't be located — the caller
    (:class:`~gitm.optimizer.apply.LiveEngineApplicator`) turns that into a
    rollback rather than silently no-op'ing.
    """
    spec = resolve_knob(knob)
    if spec is not None and spec.is_env:
        name = spec.path.split(":", 1)[1]
        # Restoring an originally-unset env var means *unsetting* it — never
        # writing the literal string "None", which vLLM would read as a backend.
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
        return
    if spec is not None:
        hl = _holder_and_leaf(engine, spec.path)
        if hl is not None:
            holder, leaf = hl
            setattr(holder, leaf, value)
            return
    if hasattr(engine, knob):
        setattr(engine, knob, value)
        return
    raise AttributeError(f"engine has no hot-swappable knob {knob!r}")


def knob_kind(knob: str) -> KnobKind:
    """``"scheduling"`` or ``"structural"`` for ``knob``.

    Unknown knobs default to ``"structural"`` — the safe assumption is that a
    knob we don't recognize needs a restart, never that it's safe to hot-swap.
    """
    spec = resolve_knob(knob)
    return spec.kind if spec is not None else "structural"


#: (name substring, prerequisite knob) — a candidate matching the substring
#: only applies when the prerequisite is on (e.g. dbo_prefill_token_threshold
#: needs --enable-dbo). Check the live engine via unmet_prerequisite rather
#: than denylisting forever, including on deployments where it genuinely holds.
KNOB_PREREQUISITES: tuple[tuple[str, str], ...] = (
    ("partial_prefill", "enable_chunked_prefill"),
    ("long_prefill_token_threshold", "enable_chunked_prefill"),
    ("dbo", "enable_dbo"),
)


def unmet_prerequisite(engine: Any | None, knob: str) -> str | None:
    """None if ``knob`` has no prerequisite, or it holds on ``engine`` — else
    the rejection reason. No engine -> can't verify -> reject (conservative,
    like :func:`knob_kind`'s unknown-defaults-unsafe default)."""
    lname = knob.lower()
    prereq = next((p for needle, p in KNOB_PREREQUISITES if needle in lname), None)
    if prereq is None:
        return None
    if engine is None:
        return f"prerequisite {prereq!r} unverifiable: no live engine"
    try:
        if get_knob(engine, prereq):
            return None
        return f"prerequisite {prereq!r} not enabled on this engine"
    except AttributeError:
        return f"prerequisite {prereq!r} unknown on this engine"


def resolve_relative_value(spec: InterventionSpec, engine: Any | None) -> InterventionSpec:
    """Scale a relative catalog lever's value off the engine's CURRENT setting.

    A knob like ``max_num_batched_tokens`` has no single right absolute value
    across deployments (model size/GPU/workload shape vary) — vLLM's own
    auto_tune.sh sweeps it rather than hardcoding a number. With a live engine,
    read its current value and scale by ``value_multiplier``. Falls back to
    the static ``value`` with no multiplier/engine/readable current value.
    """
    if spec.value_multiplier is None or engine is None:
        return spec
    try:
        current = get_knob(engine, spec.knob)
    except AttributeError:
        return spec
    if not isinstance(current, int | float) or isinstance(current, bool) or current <= 0:
        return spec
    scaled = current * spec.value_multiplier
    if spec.value_max is not None:
        scaled = min(scaled, spec.value_max)
    if spec.value_min is not None:
        scaled = max(scaled, spec.value_min)
    new_value = int(round(scaled)) if isinstance(current, int) else scaled
    return spec.model_copy(update={
        "value": new_value,
        "summary": f"{spec.summary} (scaled {spec.value_multiplier:g}x current {current} -> {new_value})",
    })


def expand_relative_candidates(spec: InterventionSpec, engine: Any | None) -> list[InterventionSpec]:
    """Sweep ``value_multiplier_grid`` into one resolved candidate per point —
    same idea as vLLM's auto_tune.sh and autoresearch's value grid, applied to
    a reviewed catalog lever. Each point resolves off the SAME current engine
    value via :func:`resolve_relative_value`, with its own name.

    No grid -> a single :func:`resolve_relative_value` call. No live engine,
    or every point collapsing to the same value (current == 0), -> one
    candidate, not N duplicates.
    """
    if not spec.value_multiplier_grid:
        return [resolve_relative_value(spec, engine)]
    if engine is None:
        return [spec.model_copy(update={"value_multiplier_grid": []})]
    out: list[InterventionSpec] = []
    seen_values: set[Any] = set()
    for m in spec.value_multiplier_grid:
        variant = spec.model_copy(update={"value_multiplier": m, "value_multiplier_grid": []})
        resolved = resolve_relative_value(variant, engine)
        if resolved.value in seen_values:
            continue  # collapsed to an already-queued value (e.g. current == 0)
        seen_values.add(resolved.value)
        suffix = f"x{m:g}".replace(".", "_").replace("-", "neg")
        out.append(resolved.model_copy(update={"name": f"{spec.name}_{suffix}"}))
    return out
