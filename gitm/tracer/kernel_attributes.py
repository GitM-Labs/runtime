"""Attributes beyond ``L{layer}/{op}``: a side table joined on what correlation recovers.

Correlation names a kernel's op and layer. Heterogeneous models need more than
that to classify it — a DeepSeek-V4 layer that compresses its KV cache costs
nothing like one that reads another layer's; an MoE layer's dispatch, expert
GEMM and combine run concurrently across expert-parallel waves; a hybrid model
interleaves linear and full attention. Encoding any of that into the range name
would break every consumer that parses it, so it is kept beside the identity,
keyed by what correlation already produced (docs/kernel_identity.md, "Attributes
beyond ``L{layer}/{op}``")::

    key                  attributes                         source
    range_layer          layer_class, layer_kind, mlp_type  the predicted graph (static)
    (layer, op)          moe_phase                          the op vocabulary (static)
    range instance       anything annotated on the range    ``L3/op#wave=2`` (dynamic)
    kernel               stream, replay, identity           the kernel record

``layer_class`` needs no per-model code: two layers are the same class when the
planner predicts the same structure for them (the same ops at the same cost), so
a model's archetypes fall out of its own graph — V4's sliding-window layers 0-1
against its compressed layers, Qwen3-Next's DeltaNet against full-attention
layers — and a dense model has exactly one class. Declared per-layer kinds
(``layer_kind``/``mlp_layer_types`` on the model spec) are added when the model
has them.

Dynamic attributes ride on the range name because rocTX has no payload: the
annotation is the one carrier both vendors share. They are per *instance* — the
same layer's ranges can carry different waves on different steps — which is
exactly what a static table cannot express.

:meth:`AttributeIndex.key` is what attribution groups by. The default, op only,
is the historical behaviour; adding attributes splits an op's residual series
into strata so a cause confined to one archetype or one wave is not averaged
away by the rest of the op.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

#: The MoE phase each MoE op belongs to. Phases overlap in time across waves
#: under expert parallelism, which is why attributing within one needs the
#: phase as well as the op.
MOE_PHASE: dict[str, str] = {
    "moe_router": "route",
    "moe_permute": "dispatch",
    "moe_all_to_all": "exchange",
    "moe_routed": "expert",
    "moe_shared": "shared_expert",
    "moe_combine": "combine",
}


def _class_signature(nodes) -> tuple:
    return tuple(sorted((n.op, round(n.prediction.t_pred_s, 15), round(n.prediction.bytes, 6))
                        for n in nodes))


@dataclass
class AttributeIndex:
    """Static per-layer attributes from a predicted graph, applied to kernels."""

    by_layer: dict[int, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_graph(cls, graph) -> AttributeIndex:
        per_layer: dict[int, list] = {}
        for n in getattr(graph, "nodes", ()):
            if n.layer is not None:
                per_layer.setdefault(n.layer, []).append(n)
        classes: dict[tuple, str] = {}
        by_layer: dict[int, dict[str, str]] = {}
        model = getattr(graph, "model", None)
        for layer in sorted(per_layer):
            sig = _class_signature(per_layer[layer])
            label = classes.setdefault(sig, f"c{len(classes)}")
            attrs = {"layer_class": label}
            kind = _declared(model, "layer_kind", layer)
            if kind:
                attrs["layer_kind"] = kind
            mlp = _declared(model, "mlp_layer_type", layer)
            if mlp:
                attrs["mlp_type"] = mlp
            by_layer[layer] = attrs
        return cls(by_layer)

    def attributes(self, kernel, op: str | None = None) -> dict[str, str]:
        """Every attribute known for ``kernel``; ``op`` overrides its identity.

        Precedence, lowest to highest: static (layer), op vocabulary, kernel
        record, dynamic annotations — the most specific source wins a clash,
        and an annotation is the most specific thing there is.
        """
        out: dict[str, str] = {}
        layer = _get(kernel, "range_layer")
        if layer is not None:
            out.update(self.by_layer.get(layer, {}))
        op = op if op is not None else _get(kernel, "range_op")
        if op in MOE_PHASE:
            out["moe_phase"] = MOE_PHASE[op]
        stream = _get(kernel, "stream_id")
        if stream is not None:
            out["stream"] = str(stream)
        out["replay"] = "graph" if _get(kernel, "graph_id") else "eager"
        ident = _get(kernel, "identity")
        if ident:
            out["identity"] = ident
        dyn = _get(kernel, "range_attrs")
        if isinstance(dyn, Mapping):
            out.update({str(k): str(v) for k, v in dyn.items()})
        return out


def stratum(op: str, attrs: Mapping[str, str], keys: Iterable[str]) -> str:
    """``"mlp_down[layer_class=c1,wave=0]"``; just ``op`` when ``keys`` is empty.

    A key the kernel lacks is written as ``?`` rather than dropped, so kernels
    with and without it never share a stratum by accident.
    """
    keys = tuple(keys)
    if not keys:
        return op
    return op + "[" + ",".join(f"{k}={attrs.get(k, '?')}" for k in keys) + "]"


def _get(obj, name):
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _declared(model, what: str, layer: int) -> str | None:
    """A per-layer kind the model spec declares, or None.

    ``layer_kind(layer)`` is the hybrid models' method; GLM-style specs carry
    ``mlp_layer_types`` as a tuple indexed by layer.
    """
    if model is None:
        return None
    if what == "layer_kind":
        fn = getattr(model, "layer_kind", None)
        if callable(fn):
            try:
                v = fn(layer)
            except Exception:
                return None
            return str(v) if v is not None else None
        return None
    types = getattr(model, "mlp_layer_types", None)
    if types and 0 <= layer < len(types):
        return str(types[layer])
    return None
