"""Read a live vLLM server's model shape and turn it into a predictable spec.

The attach path (:mod:`gitm.serve.attach`) finds the running server and knows the
name it serves under, but not its *shape* — and :func:`predict_moe_graph` without a
model silently falls back to DeepSeek-V4-Flash defaults, so a prediction against
"whatever is running" is really a prediction against a hardcoded checkpoint. The
one place the ground-truth config actually lives is the process itself: its argv
names the model, its environment names the HuggingFace cache, and the `config.json`
under that cache is the shape the graph should predict against.

This module is the bridge. It is deliberately *only* filesystem and string work —
no planner imports beyond the pure spec builder, no `/proc` beyond what
:mod:`gitm.serve.discover` already read — so it stays testable off a real pod (every
entry point takes an injectable ``environ`` and ``cache_root``) and so the planner
never grows HF-cache concerns.

The load-bearing decision is the **gate**. :func:`spec_from_hf_config` is written
for a DeepSeek-V4-class MoE, and every one of its ``cfg.get(key, default)`` calls is
a DeepSeek default; :func:`gitm.planner.roofline.weight_bytes` falls back to bf16 on
an unrecognised dtype. Feed either an off-distribution config and you get a
confident, plausible, wrong spec — exactly the "an estimate read as a measurement"
failure the graph's own docstrings are written against. So :func:`validate_moe_config`
refuses a config that is missing a dominant-term field (expert count,
experts-per-token, expert intermediate size) or carries a quantisation method the
roofline cannot price, and names the keys it looked for rather than defaulting them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gitm.planner.moe_graph import spec_from_hf_config
from gitm.planner.roofline import (
    _WEIGHT_BYTES,
    BatchConfig,
    ShardingConfig,
    SparseMoEModelSpec,
)
from gitm.serve import discover

# Dtype strings the roofline can actually price. ``bf16`` is legitimate for an
# unquantised checkpoint (no ``quantization_config``); anything outside this set that
# is *declared* by the config is a silent-mispricing risk, not a default to accept.
KNOWN_DTYPES = frozenset(_WEIGHT_BYTES)

# Alias -> canonical key that :func:`spec_from_hf_config` reads. Non-DeepSeek MoEs
# name the same tensors differently; without this a valid Mixtral/Qwen3-MoE config
# would fail the gate for "missing" experts it declares under another name.
_EXPERT_COUNT_ALIASES = ("n_routed_experts", "num_local_experts", "num_experts")
_EXPERT_TOPK_ALIASES = ("num_experts_per_tok", "moe_top_k", "top_k")
_EXPERT_INTER_ALIASES = ("moe_intermediate_size", "intermediate_size")


def model_ref_from_cmdline(cmdline: list[str]) -> str | None:
    """The model argument a ``vllm serve`` command was launched with.

    ``vllm serve <model> ...`` puts the model first; the module form
    (``-m vllm.entrypoints...``) passes it as ``--model``. Returns ``None`` when
    neither is present rather than guessing — a missing model ref becomes a refusal
    upstream, not a wrong lookup.
    """
    if not cmdline:
        return None
    # Explicit --model wins wherever it appears. Not ``-m``: that is the Python
    # interpreter's module flag (``python -m vllm.entrypoints...``), not vLLM's.
    for i, a in enumerate(cmdline):
        if a == "--model" and i + 1 < len(cmdline):
            return cmdline[i + 1]
        if a.startswith("--model="):
            return a.split("=", 1)[1]
    # Positional form: the token right after `serve`.
    for i, a in enumerate(cmdline[:-1]):
        if a == "serve":
            nxt = cmdline[i + 1]
            if not nxt.startswith("-"):
                return nxt
    return None


def hub_cache_root(environ: dict[str, str] | None) -> Path:
    """The HuggingFace hub cache directory the *target* process would use.

    Order matches ``huggingface_hub``: ``HF_HUB_CACHE`` points straight at the hub
    dir; ``HF_HOME`` puts it at ``$HF_HOME/hub``; the legacy ``TRANSFORMERS_CACHE``
    is hub-shaped; otherwise ``~/.cache/huggingface/hub``. Read from the target's
    environment, not this process's — a server started under a relocated cache keeps
    its models there, and reading our own env would look in the wrong place.
    """
    env = environ or {}
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"])
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    if env.get("TRANSFORMERS_CACHE"):
        return Path(env["TRANSFORMERS_CACHE"])
    return Path(os.path.expanduser("~/.cache/huggingface/hub"))


def resolve_config_path(
    model_ref: str | None,
    environ: dict[str, str] | None = None,
    *,
    cache_root: Path | None = None,
) -> Path | None:
    """Locate the ``config.json`` for ``model_ref`` — a local path or a hub id.

    Returns ``None`` (never a guess) when nothing resolves; the caller turns that
    into a refusal that names the ref it could not find.
    """
    if not model_ref:
        return None

    # A local checkpoint directory, or a direct path to the config itself.
    p = Path(model_ref)
    if p.is_file() and p.name == "config.json":
        return p
    if p.is_dir() and (p / "config.json").is_file():
        return p / "config.json"

    # A hub id: deepseek-ai/DeepSeek-V4 -> models--deepseek-ai--DeepSeek-V4.
    root = cache_root or hub_cache_root(environ)
    repo_dir = root / ("models--" + model_ref.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    configs = [s / "config.json" for s in snapshots.iterdir() if (s / "config.json").is_file()]
    if not configs:
        return None
    # Most recently materialised snapshot — the one a running server most likely
    # loaded after a `hf download`.
    return max(configs, key=lambda c: c.stat().st_mtime)


def _kv_dtype_from_flag(value: str, act_dtype: str) -> str:
    """Map vLLM's ``--kv-cache-dtype`` onto a roofline dtype.

    ``auto`` means the cache follows the model dtype (usually bf16), so it is *not*
    fp8 — leaving the spec's hardcoded ``fp8`` in place there would overstate the
    saving. The ``fp8_*`` variants all price identically here.
    """
    v = value.lower()
    if v == "auto":
        return act_dtype
    if v.startswith("fp8"):
        return "fp8"
    return v


def _act_dtype_from_flag(value: str) -> str | None:
    """Map vLLM's ``--dtype`` onto a roofline dtype, or ``None`` for ``auto``.

    ``auto`` defers to the checkpoint's ``torch_dtype``, which
    :func:`spec_from_hf_config` already reads — so only an *explicit* dtype overrides.
    """
    v = value.lower()
    if v in ("auto", ""):
        return None
    return {"half": "fp16", "float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}.get(v, v)


def serving_overrides_from_cmdline(cmdline: list[str]) -> dict[str, Any]:
    """Serving decisions that live on the launch command, not in the checkpoint.

    Parsed with the same ``_arg_of``/``resolve_dp`` helpers the launch preflight uses
    (imported lazily so this module does not pull vLLM's serve machinery on import).
    ``kv_dtype`` here is the strongest reason this whole path exists: the spec builder
    hardcodes ``fp8`` and comments that it is "a serving decision, not a model fact" —
    and this is where that fact is recorded.
    """
    from gitm.serve.vllm import _arg_of, resolve_dp

    out: dict[str, Any] = {}

    dtype_flag = _arg_of(cmdline, "--dtype")
    act = _act_dtype_from_flag(dtype_flag) if dtype_flag else None
    if act:
        out["act_dtype"] = act

    kv = _arg_of(cmdline, "--kv-cache-dtype")
    if kv:
        # Priced against the resolved activation dtype for the ``auto`` case.
        out["kv_dtype"] = _kv_dtype_from_flag(kv, act or "bf16")

    mml = _arg_of(cmdline, "--max-model-len")
    if mml and mml.isdigit():
        out["max_model_len"] = int(mml)

    tp_raw = _arg_of(cmdline, "--tensor-parallel-size") or _arg_of(cmdline, "-tp")
    tp = int(tp_raw) if tp_raw and tp_raw.isdigit() else 1
    dp = resolve_dp(cmdline)
    ep_enabled = "--enable-expert-parallel" in cmdline
    out["tp"], out["dp"] = tp, dp
    # vLLM shards whole experts across the parallel ranks when expert parallelism is
    # on; modelled here as EP over the tensor-parallel group (single-node case). A
    # multi-node dp>1 EP deployment is approximated by this and flagged as such.
    out["ep"] = tp if ep_enabled else 1
    out["ep_enabled"] = ep_enabled
    return out


def _first_present(cfg: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return None


def is_sparse_moe_config(cfg: dict[str, Any]) -> bool:
    """True when any routed-expert shape field makes this a sparse candidate.

    This deliberately recognizes partial configs. A config that declares an
    expert count but omits top-k must reach :func:`validate_moe_config` and be
    refused, never fall through to the dense graph because it was incomplete.
    """
    # ``intermediate_size`` is also the standard dense-MLP field, so only the
    # explicitly MoE spelling is a sparse signal on that axis.
    return (
        _first_present(cfg, _EXPERT_COUNT_ALIASES) is not None
        or _first_present(cfg, _EXPERT_TOPK_ALIASES) is not None
        or cfg.get("moe_intermediate_size") is not None
    )


def validate_moe_config(cfg: dict[str, Any]) -> list[str]:
    """Dominant-term fields this config fails to declare usably. Empty == usable.

    Refusing here rather than defaulting is the whole point: a missing expert count
    silently becomes 256, a missing top-k becomes 6, and an unrecognised quant method
    becomes bf16 — each a large error on a term that dominates the decode step. The
    strings returned name the aliases looked for so the operator can see what a
    supported config would carry.
    """
    missing: list[str] = []

    def positive_alias(keys: tuple[str, ...], label: str) -> int | None:
        value = _first_present(cfg, keys)
        joined = " | ".join(keys)
        if value is None:
            missing.append(f"{joined} ({label})")
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            missing.append(f"{joined} ({label}) must be a positive integer, got {value!r}")
            return None
        return value

    n_experts = positive_alias(_EXPERT_COUNT_ALIASES, "routed expert count")
    top_k = positive_alias(_EXPERT_TOPK_ALIASES, "experts per token")
    positive_alias(_EXPERT_INTER_ALIASES, "expert intermediate size")
    if n_experts is not None and top_k is not None and top_k > n_experts:
        missing.append(
            f"experts per token {top_k} exceeds routed expert count {n_experts}"
        )

    # The sparse graph is a V4-shaped attention model, not a generic MoE graph.
    # Every field below changes a node's compute/bytes; defaulting any of them to
    # V4 values would fabricate a plausible graph for a partial or foreign model.
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "o_groups",
        "vocab_size",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "sliding_window",
    ):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            missing.append(f"{key} must be a declared positive integer, got {value!r}")
    qk_rope = cfg.get("qk_rope_head_dim")
    if isinstance(qk_rope, bool) or not isinstance(qk_rope, int) or qk_rope < 0:
        missing.append(
            f"qk_rope_head_dim must be a declared non-negative integer, got {qk_rope!r}"
        )
    shared = cfg.get("n_shared_experts")
    if isinstance(shared, bool) or not isinstance(shared, int) or shared < 0:
        missing.append(
            f"n_shared_experts must be a declared non-negative integer, got {shared!r}"
        )
    if cfg.get("torch_dtype") is None:
        missing.append("torch_dtype must be declared; activation width cannot be guessed")

    n_layers = cfg.get("num_hidden_layers")
    ratios = cfg.get("compress_ratios")
    if not isinstance(ratios, (list, tuple)):
        missing.append("compress_ratios must be declared for the sparse-attention graph")
    elif isinstance(n_layers, int) and n_layers > 0 and len(ratios) < n_layers:
        missing.append(
            f"compress_ratios has {len(ratios)} entries; need at least {n_layers} for {n_layers} layers"
        )
    elif any(isinstance(r, bool) or not isinstance(r, int) or r < 0 for r in ratios):
        missing.append("compress_ratios must contain non-negative integers")

    # Quantisation: a *declared* method must be one the roofline can price. No
    # ``quantization_config`` is fine — that is an unquantised (bf16) checkpoint.
    q = cfg.get("quantization_config") or {}
    if not isinstance(q, dict):
        missing.append("quantization_config must be an object when declared")
        q = {}
    method = q.get("quant_method")
    if method is not None and str(method).lower() not in KNOWN_DTYPES:
        missing.append(
            f"quantization_config.quant_method={method!r} "
            f"(not priceable; known: {', '.join(sorted(KNOWN_DTYPES))})"
        )
    expert_dtype = cfg.get("expert_dtype")
    if expert_dtype is None:
        missing.append(
            "expert_dtype must be declared; routed and shared expert byte width cannot be guessed"
        )
    elif str(expert_dtype).lower() not in KNOWN_DTYPES:
        missing.append(f"expert_dtype={expert_dtype!r} (not priceable)")
    return missing


def validate_priceable_dtypes(spec: SparseMoEModelSpec) -> list[str]:
    """Final resolved dtypes that would make byte-width pricing fall back."""
    missing: list[str] = []
    for field_name in ("weight_dtype", "expert_dtype", "kv_dtype", "act_dtype"):
        value = getattr(spec, field_name)
        if str(value).lower() not in KNOWN_DTYPES:
            missing.append(
                f"{field_name}={value!r} (not priceable; known: "
                f"{', '.join(sorted(KNOWN_DTYPES))})"
            )
    return missing


def normalize_moe_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``cfg`` with alias keys mapped onto the canonical DeepSeek names.

    Only fills a canonical key that is absent, so a config that already uses the
    DeepSeek names is untouched. This keeps :func:`spec_from_hf_config` a pure,
    single-vocabulary builder — the aliasing lives here, at the edge that reads
    foreign configs, and its dict-based tests keep passing unchanged.
    """
    out = dict(cfg)
    for canonical, aliases in (
        ("n_routed_experts", _EXPERT_COUNT_ALIASES),
        ("num_experts_per_tok", _EXPERT_TOPK_ALIASES),
        ("moe_intermediate_size", _EXPERT_INTER_ALIASES),
    ):
        if out.get(canonical) is None:
            val = _first_present(cfg, aliases)
            if val is not None:
                out[canonical] = val
    return out


@dataclass
class LiveSpec:
    """A live server resolved into everything :func:`predict_moe_graph` needs."""

    spec: SparseMoEModelSpec
    sharding: ShardingConfig
    batch: BatchConfig
    source_path: Path
    model_ref: str
    applied_overrides: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    ok: bool = True


@dataclass
class LiveSpecError:
    """Why a live server could not be turned into a trustworthy spec.

    ``missing_keys`` is populated only for the gate failure; a missing config or
    unreadable environment set ``reason`` alone.
    """

    reason: str
    model_ref: str | None = None
    missing_keys: list[str] = field(default_factory=list)
    ok: bool = False

    def render(self) -> str:
        lines = [self.reason]
        if self.missing_keys:
            lines.append("  the config must declare each of:")
            lines += [f"    - {k}" for k in self.missing_keys]
        return "\n".join(lines)


def live_moe_spec(
    target: discover.Target,
    *,
    environ: dict[str, str] | None = None,
    proc: Path = discover.PROC,
    cache_root: Path | None = None,
    default_kv_cache_len: int = 4096,
) -> LiveSpec | LiveSpecError:
    """Resolve a running server (a discover :class:`Target`) into a MoE spec.

    Reads the target's own environment for the cache location unless ``environ`` is
    supplied (tests). Returns :class:`LiveSpec` on success or :class:`LiveSpecError`
    with the reason — never a defaulted spec, so a report can only ever show a
    prediction against the model that is genuinely loaded.

    """
    model_ref = model_ref_from_cmdline(target.cmdline)
    if not model_ref:
        return LiveSpecError(
            reason=f"could not find a model argument on PID {target.pid}'s command line."
        )

    if environ is None:
        environ = discover.read_environ(target.pid, proc) or {}

    cfg_path = resolve_config_path(model_ref, environ, cache_root=cache_root)
    if cfg_path is None:
        return LiveSpecError(
            reason=(
                f"no config.json found for {model_ref!r} — not a local checkpoint dir, "
                f"and not in the hub cache at {cache_root or hub_cache_root(environ)}."
            ),
            model_ref=model_ref,
        )

    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError) as e:
        return LiveSpecError(reason=f"could not read {cfg_path}: {e}", model_ref=model_ref)

    # predict_moe_graph models DeepSeek-V4-class compressed/selected attention;
    # feeding it a Mixtral-style MoE (standard attention) would mis-price the KV
    # path. The dense planner is the right tool for those, so decline rather than
    # emit a confident wrong floor.
    if not is_sparse_moe_config(cfg):
        missing = validate_moe_config(cfg)
        if not missing:
            missing = ["index_topk or compress_ratios"]
        return LiveSpecError(
            reason=(
                f"{model_ref!r} at {cfg_path} is not a DeepSeek-V4-class sparse-MoE model "
                "(no index_topk / compress_ratios); its attention is not what "
                "predict_moe_graph models."
            ),
            model_ref=model_ref,
            missing_keys=missing,
        )

    missing = validate_moe_config(cfg)
    if missing:
        return LiveSpecError(
            reason=(
                f"{model_ref!r} at {cfg_path} is not a sparse-MoE config this planner can "
                "predict without guessing its dominant terms."
            ),
            model_ref=model_ref,
            missing_keys=missing,
        )

    spec = spec_from_hf_config(normalize_moe_config(cfg), name=model_ref)

    overrides = serving_overrides_from_cmdline(target.cmdline)
    spec_changes: dict[str, Any] = {}
    if "kv_dtype" in overrides:
        spec_changes["kv_dtype"] = overrides["kv_dtype"]
    if "act_dtype" in overrides:
        spec_changes["act_dtype"] = overrides["act_dtype"]
        if "kv_dtype" not in overrides:
            # vLLM's absent --kv-cache-dtype means ``auto``: follow the resolved
            # compute dtype, not a model-family-specific fp8 guess.
            spec_changes["kv_dtype"] = overrides["act_dtype"]
    if spec_changes:
        from dataclasses import replace

        spec = replace(spec, **spec_changes)

    unpriceable = validate_priceable_dtypes(spec)
    if unpriceable:
        return LiveSpecError(
            reason=(
                f"{model_ref!r} at {cfg_path} resolves to dtypes this planner cannot "
                "price without substituting bf16 byte widths."
            ),
            model_ref=model_ref,
            missing_keys=unpriceable,
        )

    sharding = ShardingConfig(
        tp=overrides.get("tp", 1), ep=overrides.get("ep", 1), dp=overrides.get("dp", 1)
    )
    batch = BatchConfig(kv_cache_len=overrides.get("max_model_len", default_kv_cache_len))
    warnings: list[str] = []
    if cfg.get("expert_dtype") is None:
        warnings.append(
            "expert_dtype absent; inherited "
            f"weight_dtype={spec.weight_dtype!r} for routed and shared expert weights"
        )
    warnings.append("decode batch was not observed; using batch=1 single-sequence floor")
    if "max_model_len" not in overrides:
        warnings.append(
            f"KV cache length was not declared on the launch command; using "
            f"kv_cache_len={default_kv_cache_len}"
        )
    if overrides.get("ep_enabled") and overrides.get("dp", 1) > 1:
        warnings.append(
            "expert parallelism with data_parallel_size>1 is approximated over the "
            "tensor-parallel group"
        )

    return LiveSpec(
        spec=spec,
        sharding=sharding,
        batch=batch,
        source_path=cfg_path,
        model_ref=model_ref,
        applied_overrides=overrides,
        warnings=warnings,
    )
