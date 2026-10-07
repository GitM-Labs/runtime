"""Attributes beyond ``L{layer}/{op}``, joined on what correlation recovers.

    range_layer        layer_class, layer_kind, mlp_type   predicted graph (static)
    op                 moe_phase                           op vocabulary (static)
    range instance     annotations (``L3/op#wave=2``)      the range name (dynamic)
    kernel             stream, replay, identity            the kernel record

``layer_class`` groups layers the planner predicts identically, so a model's
archetypes come from its own graph with no per-model code. :func:`stratum` is
what attribution groups by when asked to stratify.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

MOE_PHASE = {"moe_router": "route", "moe_permute": "dispatch", "moe_all_to_all": "exchange",
             "moe_routed": "expert", "moe_shared": "shared_expert", "moe_combine": "combine"}


@dataclass
class AttributeIndex:
    by_layer: dict[int, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_graph(cls, graph) -> AttributeIndex:
        per_layer: dict[int, list] = {}
        for n in getattr(graph, "nodes", ()):
            if n.layer is not None:
                per_layer.setdefault(n.layer, []).append(n)
        model = getattr(graph, "model", None)
        classes: dict[tuple, str] = {}
        by_layer = {}
        for layer in sorted(per_layer):
            sig = tuple(sorted((n.op, round(n.prediction.t_pred_s, 15),
                                round(n.prediction.bytes, 6)) for n in per_layer[layer]))
            attrs = {"layer_class": classes.setdefault(sig, f"c{len(classes)}")}
            kind = _layer_kind(model, layer)
            if kind:
                attrs["layer_kind"] = kind
            types = getattr(model, "mlp_layer_types", None)
            if types and layer < len(types):
                attrs["mlp_type"] = str(types[layer])
            by_layer[layer] = attrs
        return cls(by_layer)

    def attributes(self, kernel, op: str | None = None) -> dict[str, str]:
        """Static, then op vocabulary, then the record, then annotations — the most
        specific source wins."""
        get = kernel.get if isinstance(kernel, Mapping) else (lambda k: getattr(kernel, k, None))
        out = dict(self.by_layer.get(get("range_layer"), {}))
        op = op if op is not None else get("range_op")
        if op in MOE_PHASE:
            out["moe_phase"] = MOE_PHASE[op]
        if get("stream_id") is not None:
            out["stream"] = str(get("stream_id"))
        out["replay"] = "graph" if get("graph_id") else "eager"
        if get("identity"):
            out["identity"] = get("identity")
        if isinstance(get("range_attrs"), Mapping):
            out.update({str(k): str(v) for k, v in get("range_attrs").items()})
        return out


def stratum(op: str, attrs: Mapping[str, str], keys: Iterable[str]) -> str:
    """``"mlp_down[wave=0]"``; just ``op`` with no keys. A missing key is ``?``."""
    keys = tuple(keys)
    return op + "[" + ",".join(f"{k}={attrs.get(k, '?')}" for k in keys) + "]" if keys else op


def _layer_kind(model, layer: int) -> str | None:
    fn = getattr(model, "layer_kind", None)
    if not callable(fn):
        return None
    try:
        v = fn(layer)
    except Exception:
        return None
    return None if v is None else str(v)
