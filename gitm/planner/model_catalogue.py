from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

CATALOGUE_DIR = Path(__file__).resolve().parent / "models"

#: Families a catalogue entry may declare, and the spec each one builds.
_FAMILIES = ("hybrid", "sparse_moe", "glm_moe_dsa")


def available() -> list[str]:
    """Catalogue entry names, sorted. The stem of each YAML file."""
    if not CATALOGUE_DIR.is_dir():
        return []
    return sorted(p.stem for p in CATALOGUE_DIR.glob("*.yaml"))


def _resolve(name_or_path: str | Path) -> Path:
    """A catalogue name or a path to a YAML file, resolved to a path."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return p
    candidate = CATALOGUE_DIR / f"{p.name}.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"no catalogue entry {str(name_or_path)!r}. Available: {available() or 'none'}. "
        "Pass a catalogue name or a path to a YAML file."
    )


def _expand_layer_types(value: Any, n_layers: int) -> tuple[str, ...]:
    """Accept a plain list or a ``{pattern, repeat}`` form.

    Forty entries written out is unambiguous but unreadable, and an unreadable
    schedule is one nobody checks. The compact form is expanded and then
    validated against ``n_layers``, so a pattern that does not tile the model is
    a load error rather than a silently truncated schedule — which would leave
    the trailing layers taking the last entry's kind.
    """
    if value is None:
        return ()
    if isinstance(value, dict):
        pattern = value.get("pattern")
        repeat = value.get("repeat")
        if not isinstance(pattern, list) or not pattern:
            raise ValueError("layer_types.pattern must be a non-empty list")
        try:
            repeat = int(repeat)
        except (TypeError, ValueError):
            raise ValueError("layer_types.repeat must be an integer") from None
        expanded = tuple(str(t) for t in pattern) * repeat
    elif isinstance(value, list):
        expanded = tuple(str(t) for t in value)
    else:
        raise ValueError(
            "layer_types must be a list or a {pattern, repeat} mapping, "
            f"got {type(value).__name__}"
        )
    if expanded and len(expanded) != n_layers:
        raise ValueError(
            f"layer_types expands to {len(expanded)} entries but n_layers is "
            f"{n_layers} — the schedule must cover the model exactly"
        )
    return expanded


def load_entry(name_or_path: str | Path, _seen: frozenset[str] = frozenset()) -> dict[str, Any]:
    """The raw catalogue entry, validated for structure but not yet a spec.

    ``extends: <entry>`` merges this entry's ``spec`` over that of another. It
    exists for the case where two entries describe *the same architecture at a
    different precision* — a bf16 release and its FP8 sibling, which share a
    78-entry indexer schedule and a 78-entry MLP schedule verbatim. Copying those
    into both files is 158 lines of duplicated evidence that can drift apart
    silently, and it hides the thing worth seeing: the two entries differ only in
    their dtypes. ``provenance`` is deliberately *not* merged — each checkpoint
    was validated against its own published size and has its own open questions.
    """
    path = _resolve(name_or_path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")

    base_name = data.pop("extends", None)
    if base_name is not None:
        key = str(base_name)
        if key in _seen:
            raise ValueError(f"{path}: 'extends' cycle through {key!r}")
        base = load_entry(key, _seen | {key})
        merged = dict(base.get("spec") or {})
        merged.update(data.get("spec") or {})
        data = {**{k: v for k, v in base.items() if k != "provenance"}, **data}
        data["spec"] = merged

    family = data.get("family")
    if family not in _FAMILIES:
        raise ValueError(
            f"{path}: family must be one of {list(_FAMILIES)}, got {family!r}"
        )
    if not isinstance(data.get("spec"), dict):
        raise ValueError(f"{path}: missing a 'spec' mapping")
    return data


def load_spec(name_or_path: str | Path):
    """Build the model spec a catalogue entry describes.

    Raises
    ------
    ValueError
        On an unknown key. A mistyped field would otherwise be dropped silently,
        leaving the spec holding a reference default while the file appears to
        set it — the exact class of error this catalogue exists to prevent.
    """
    entry = load_entry(name_or_path)
    family = entry["family"]
    raw = dict(entry["spec"])

    if family == "hybrid":
        from gitm.planner.hybrid_graph import HybridMoEModelSpec as cls
    elif family == "glm_moe_dsa":
        from gitm.planner.glm_graph import GlmMoeDsaModelSpec as cls  # type: ignore[assignment]
    else:
        from gitm.planner.roofline import SparseMoEModelSpec as cls  # type: ignore[assignment]

    known = {f.name for f in fields(cls)}
    if "layer_types" in raw:
        raw["layer_types"] = _expand_layer_types(
            raw["layer_types"], int(raw.get("n_layers", 0))
        )
    # Per-layer schedule lists that must reach the frozen dataclass as tuples. A
    # list would make the spec unhashable; a dropped tuple-coercion here is how a
    # schedule silently arrives as the wrong type.
    for key in ("compress_ratios", "dspark_layer_ids", "indexer_types", "mlp_layer_types"):
        if key in raw and isinstance(raw[key], list):
            raw[key] = tuple(raw[key])
    # Per-layer schedules must cover the model exactly. ``spec_from_hf_config``
    # already refuses a short one; the catalogue path did not, and a schedule one
    # entry short does not fail — the missing layers fall through to the modulo
    # fallback and can land on the right answer by luck, which is a plausible
    # total resting on evidence that is not there. That is the exact failure the
    # explicit schedules exist to prevent, so it is an error here too.
    n_layers = int(raw.get("n_layers", 0) or 0)
    for key in ("indexer_types", "mlp_layer_types"):
        sched = raw.get(key)
        if sched and n_layers and len(sched) != n_layers:
            raise ValueError(
                f"{name_or_path}: {key} has {len(sched)} entries for {n_layers} "
                "layers — the schedule must cover the model exactly"
            )
    # Nested one level: ``op_dtype_overrides`` is a list of ``[op, dtype]`` pairs
    # in YAML and must reach the frozen spec as a tuple of tuples. Coercing only
    # the outer list would leave inner lists inside a frozen dataclass — hashable
    # in appearance, not in fact.
    if isinstance(raw.get("op_dtype_overrides"), list):
        raw["op_dtype_overrides"] = tuple(
            tuple(pair) if isinstance(pair, list) else pair
            for pair in raw["op_dtype_overrides"]
        )

    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{name_or_path}: unknown spec field(s) {sorted(unknown)} for family "
            f"{family!r}. Known fields: {sorted(known)}"
        )

    raw.setdefault("name", entry.get("name", str(name_or_path)))
    return cls(**raw)


def predict(
    name_or_path: str | Path,
    hw=None,
    batch=None,
    sharding=None,
    **kwargs: Any,
):
    """``(graph, family)`` for a catalogue entry — the one-call path."""
    entry = load_entry(name_or_path)
    spec = load_spec(name_or_path)
    family = entry["family"]

    if family == "hybrid":
        from gitm.planner.hybrid_graph import predict_hybrid_graph

        return predict_hybrid_graph(spec, hw, batch, sharding, **kwargs), family

    if family == "glm_moe_dsa":
        from gitm.planner.glm_graph import predict_glm_graph

        return predict_glm_graph(spec, hw, batch, sharding, **kwargs), family

    from gitm.planner.moe_graph import predict_moe_graph

    return predict_moe_graph(spec, hw, batch, sharding, **kwargs), family
