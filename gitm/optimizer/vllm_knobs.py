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

import json
import os
from collections.abc import Callable, Mapping
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


# The curated map. vLLM EngineArgs are treated as construction-time knobs in the
# real in-process engine path, even when a Python config object exposes the same
# field. Route them through restart so the A/B measures a rebuilt engine, not a
# config mutation the scheduler/cache/model may ignore.
_KNOBS: dict[str, KnobSpec] = {
    # --- structural: fixed at engine construction, need a restart to change ----
    "max_num_seqs": KnobSpec("max_num_seqs", "scheduler_config.max_num_seqs", "structural"),
    "max_num_batched_tokens": KnobSpec(
        "max_num_batched_tokens", "scheduler_config.max_num_batched_tokens", "structural"
    ),
    "scheduling_policy": KnobSpec("scheduling_policy", "scheduler_config.policy", "structural"),
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
    "async_scheduling": KnobSpec(
        "async_scheduling", "scheduler_config.async_scheduling", "structural"
    ),
    "scheduler_delay_factor": KnobSpec(
        "scheduler_delay_factor", "scheduler_config.scheduler_delay_factor", "structural"
    ),
    "max_num_partial_prefills": KnobSpec(
        "max_num_partial_prefills", "scheduler_config.max_num_partial_prefills", "structural"
    ),
    "max_long_partial_prefills": KnobSpec(
        "max_long_partial_prefills", "scheduler_config.max_long_partial_prefills", "structural"
    ),
    "long_prefill_token_threshold": KnobSpec(
        "long_prefill_token_threshold", "scheduler_config.long_prefill_token_threshold", "structural"
    ),
    "dbo_decode_token_threshold": KnobSpec(
        "dbo_decode_token_threshold", "scheduler_config.dbo_decode_token_threshold", "structural"
    ),
    "dbo_prefill_token_threshold": KnobSpec(
        "dbo_prefill_token_threshold", "scheduler_config.dbo_prefill_token_threshold", "structural"
    ),
    "num_speculative_tokens": KnobSpec(
        "num_speculative_tokens", "speculative_config.num_speculative_tokens", "structural"
    ),
    "enforce_eager": KnobSpec("enforce_eager", "model_config.enforce_eager", "structural"),
    "max_seq_len_to_capture": KnobSpec(
        "max_seq_len_to_capture", "model_config.max_seq_len_to_capture", "structural"
    ),
    "max_model_len": KnobSpec("max_model_len", "model_config.max_model_len", "structural"),
    "disable_sliding_window": KnobSpec(
        "disable_sliding_window", "model_config.disable_sliding_window", "structural"
    ),
    "quantization": KnobSpec("quantization", "model_config.quantization", "structural"),
    "calculate_kv_scales": KnobSpec(
        "calculate_kv_scales", "cache_config.calculate_kv_scales", "structural"
    ),
    "kv_sharing_fast_prefill": KnobSpec(
        "kv_sharing_fast_prefill", "cache_config.kv_sharing_fast_prefill", "structural"
    ),
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
    # Mixture-of-Experts. All read at construction (they change how expert
    # weights are sharded and which fused-MoE kernel is built) → structural.
    "enable_expert_parallel": KnobSpec(
        "enable_expert_parallel", "parallel_config.enable_expert_parallel", "structural"
    ),
    "enable_eplb": KnobSpec("enable_eplb", "parallel_config.enable_eplb", "structural"),
    "data_parallel_size": KnobSpec(
        "data_parallel_size", "parallel_config.data_parallel_size", "structural"
    ),
    # Env-var knob: read by vLLM at construction → structural.
    "VLLM_ATTENTION_BACKEND": KnobSpec(
        "VLLM_ATTENTION_BACKEND", "env:VLLM_ATTENTION_BACKEND", "structural"
    ),
    # Prerequisite flags for the table below — not applied standalone, only read.
    "enable_dbo": KnobSpec("enable_dbo", "scheduler_config.enable_dbo", "structural"),
    "moe_backend": KnobSpec("moe_backend", "model_config.moe_backend", "structural"),
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
    """Set ``knob`` on the live engine via its taxonomy path (live knobs).

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
    # Expert-parallel load balancing only means anything under EP: with plain TP
    # every rank slices every expert, so routing skew is symmetric across ranks
    # and there is no straggler for EPLB to rebalance.
    ("eplb", "enable_expert_parallel"),
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


# --- knobs vLLM takes nested inside another argument ---------------------------
#
# The catalogue names a lever by the setting it changes. vLLM does not always
# take that setting as an argument of its own: speculative decoding is configured
# through one JSON argument, ``speculative_config``. Passing
# ``num_speculative_tokens`` at the top level is rejected by the engine
# (``LLM(num_speculative_tokens=5)``) and by the server
# (``vllm: error: unrecognized arguments: --num-speculative-tokens 5``), so the
# top-ranked lever of every run and every sweep could never start (P2-1, L-3).
#
# One table, read by the three places a knob leaves gitm: the restart path's
# engine kwargs, the server flags ``gitm propose`` emits, and ``gitm ingest``
# reading those flags back to the lever.

# The lever changes its one field and nothing else. A baseline that already
# speculates keeps its method, draft model and lookup window, so the A/B differs
# from it only in the token count; the n-gram method is filled in only when the
# baseline does not speculate at all.

#: knob -> (nested argument, field inside it, its type, fields filled in only if absent).
_NESTED_KNOBS: dict[str, tuple[str, str, Callable[[Any], Any], dict[str, Any]]] = {
    # n-gram drafting: vLLM fills in the lookup window itself when it is unset.
    "num_speculative_tokens": ("speculative_config", "num_speculative_tokens", int,
                               {"method": "ngram"}),
}


def _as_dict(value: Any) -> dict[str, Any]:
    """A nested argument as a dict, whether it came as a dict or as the JSON a
    server flag carries. Anything unreadable counts as unset."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


def _nested_value(knob: str, value: Any, current: Any) -> dict[str, Any]:
    _arg, field, cast, defaults = _NESTED_KNOBS[knob]
    return {**defaults, **_as_dict(current), field: cast(value)}


def engine_kwargs(values: dict[str, Any],
                  base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``{knob: value}`` as the keyword arguments ``LLM()`` accepts, to update
    ``base`` (the engine's current kwargs) with. A nested knob is merged into
    the argument ``base`` already has, not put in place of it."""
    out: dict[str, Any] = {}
    for knob, value in values.items():
        nested = _NESTED_KNOBS.get(knob)
        if nested is None:
            out[knob] = value
            continue
        arg = nested[0]
        current = out[arg] if arg in out else (base or {}).get(arg)
        out[arg] = _nested_value(knob, value, current)
    return out


def server_arg(knob: str, value: Any, current: Any = None) -> tuple[str, Any]:
    """``(flag, value)`` that sets ``knob`` on ``vllm serve``. ``current`` is the
    value the baseline already passes for that flag, which a nested knob is
    merged into."""
    nested = _NESTED_KNOBS.get(knob)
    if nested is None:
        return "--" + knob.replace("_", "-"), value
    merged = _nested_value(knob, value, current)
    return "--" + nested[0].replace("_", "-"), json.dumps(merged, separators=(",", ":"))


def same_server_value(a: Any, b: Any) -> bool:
    """Whether two values of one server flag set the same thing. A JSON-valued
    flag (``--speculative-config``) is compared as what it parses to, so the
    same config spaced or ordered differently is not a change."""
    if a == b:
        return True
    try:
        return isinstance(a, str) and isinstance(b, str) and json.loads(a) == json.loads(b)
    except ValueError:
        return False


def knob_from_server_arg(name: str, value: Any, baseline: Any = None) -> tuple[str, Any]:
    """The catalogue ``(knob, value)`` a server argument sets; the inverse of
    :func:`server_arg`. ``name`` is the flag without dashes, in snake case, and
    ``baseline`` is what the baseline passed for the same flag.

    A nested argument is read as the knob only when it is exactly what
    :func:`server_arg` builds from ``baseline``: the baseline's config with the
    knob's field changed. A config that also changed the method or anything else
    is not this lever, and crediting the lever with it would file a method change
    as evidence about a token count. Anything else comes back unchanged."""
    for knob, (arg, field, _cast, _defaults) in _NESTED_KNOBS.items():
        if name != arg:
            continue
        got = _as_dict(value)
        if got.get(field) is None or (baseline is not None and got == _as_dict(baseline)):
            continue    # no setting, or the baseline's own: nothing was changed
        try:
            built = _nested_value(knob, got[field], baseline)
        except (TypeError, ValueError):
            continue
        if got == built:
            return knob, got[field]
    return name, value
