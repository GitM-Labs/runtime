"""Read scheduler stats from a vLLM V1-shaped engine.

V1 keeps the scheduler inside EngineCore and exposes a stats object via
make_stats() (num_running_reqs / num_waiting_reqs / kv_cache_usage) instead of
the V0 running/waiting deques. Fake that shape to pin the read path off-GPU; the
exact attr names still need GPU validation on the target vLLM build.
"""

from __future__ import annotations

from types import SimpleNamespace

from gitm.tracer.vllm_stats import read_scheduler_stats


def _v1_engine(running: int, waiting: int, cache: float, max_seqs: int = 256):
    stats = SimpleNamespace(
        num_running_reqs=running, num_waiting_reqs=waiting, kv_cache_usage=cache
    )
    scheduler = SimpleNamespace(make_stats=lambda: stats)
    # llm.llm_engine.engine_core.engine_core.scheduler (in-process V1)
    return SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=scheduler))
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_seqs),
    )


def test_reads_v1_stats_object():
    s = read_scheduler_stats(_v1_engine(running=3, waiting=12, cache=0.87), t_ns=0)
    assert s is not None
    assert s.num_running == 3
    assert s.num_waiting == 12
    assert s.gpu_cache_usage == 0.87
    assert s.batch_occupancy == 3 / 256  # occupancy derived from V1 num_running


def test_none_engine_still_none():
    assert read_scheduler_stats(None) is None


def test_v1_engine_without_stats_is_none():
    # a V1-shaped engine whose scheduler exposes nothing readable -> None, no crash.
    empty = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=object()))
        )
    )
    assert read_scheduler_stats(empty) is None


# ── NVTX model instrumentation ──────────────────────────────────────────────
# push/pop are injectable, so these run without torch or a GPU.


class _FakeModule:
    """Enough of nn.Module to register hooks against."""

    def __init__(self, name=""):
        self._name = name
        self._children: dict[str, _FakeModule] = {}
        self.pre_hooks: list = []
        self.post_hooks: list = []

    def add(self, path: str) -> _FakeModule:
        head, _, rest = path.partition(".")
        child = self._children.setdefault(head, _FakeModule(head))
        return child.add(rest) if rest else child

    def named_modules(self, prefix=""):
        if prefix:
            yield prefix, self
        for name, child in self._children.items():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)

    def register_forward_pre_hook(self, fn):
        self.pre_hooks.append(fn)
        return _FakeHandle(self.pre_hooks, fn)

    def register_forward_hook(self, fn, always_call=False):
        self.post_hooks.append(fn)
        return _FakeHandle(self.post_hooks, fn)

    def forward(self):
        """Run pre-hooks, recurse into children, run post-hooks."""
        for h in self.pre_hooks:
            h(self, ())
        for child in self._children.values():
            child.forward()
        for h in self.post_hooks:
            h(self, (), None)


class _FakeHandle:
    def __init__(self, lst, fn):
        self._lst, self._fn = lst, fn

    def remove(self):
        if self._fn in self._lst:
            self._lst.remove(self._fn)


def _qwen_tree():
    """Two layers of the Qwen3.6 shape: one GDN, one full-attention, both MoE."""
    root = _FakeModule()
    for p in [
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.0.linear_attn.conv1d",
        "model.layers.0.mlp.gate",
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_expert",
        "model.layers.3.self_attn.qkv_proj",
        "model.layers.3.self_attn.o_proj",
        "model.layers.3.mlp.experts",
        "model.embed_tokens",
        "lm_head",
    ]:
        root.add(p)
    return root


def _record():
    """A push/pop pair that records the range stack as it evolves."""
    stack, seen = [], []

    def push(name):
        stack.append(name)
        seen.append(tuple(stack))

    def pop():
        stack.pop()

    return push, pop, stack, seen


def test_layer_index_is_read_from_the_path_not_a_counter():
    """A model whose blocks are built out of order, or shared with a draft head,
    is still labelled by where it actually sits."""
    from gitm.tracer.vllm_stats import op_for_module

    assert op_for_module("model.layers.3.self_attn.qkv_proj") == ("qkv_proj", 3)
    assert op_for_module("model.layers.11.mlp.experts") == ("moe_routed", 11)
    assert op_for_module("lm_head") == ("lm_head", None)
    assert op_for_module("model.embed_tokens") is None
    assert op_for_module("") is None


def test_range_names_round_trip_through_the_correlator():
    """A range whose name does not parse resolves to an op nobody predicted."""
    from gitm.distributed.correlate import parse_range_name
    from gitm.tracer.vllm_stats import op_for_module, range_name

    for path in ("model.layers.7.mlp.experts", "lm_head"):
        op, layer = op_for_module(path)
        assert parse_range_name(range_name(op, layer)) == (op, layer)


def test_every_emitted_op_is_one_the_graph_actually_predicts():
    """The load-bearing consistency check.

    Correlated kernels and name-classified kernels must land on the *same* node.
    If instrumentation invented its own spelling, one op's residual would be
    split across two names and each half would look healthier than the whole.
    """
    from gitm.optimizer.deviation import _OP_RULES
    from gitm.tracer.vllm_stats import _CONTAINER_OPS, _MODULE_OPS

    emitted = set(_MODULE_OPS.values()) | set(_CONTAINER_OPS.values())
    assert emitted <= set(_OP_RULES), f"ops with no predicted node: {emitted - set(_OP_RULES)}"


def test_instrumentation_covers_both_layer_types():
    from gitm.tracer.vllm_stats import instrument_model

    push, pop, _, _ = _record()
    inst = instrument_model(_qwen_tree(), push=push, pop=pop)

    assert "L0/linattn_recurrent" in inst.ranges     # GDN block
    assert "L0/linattn_in_proj" in inst.ranges
    assert "L3/attn_score_value" in inst.ranges      # attention block
    assert "L3/qkv_proj" in inst.ranges
    assert "L0/moe_routed" in inst.ranges
    assert "lm_head" in inst.ranges
    assert inst.n_modules == len(inst.ranges)


def test_container_ranges_enclose_their_children():
    """The fallback that carries most of the value: the attention and GDN kernels
    are launched from inside the block's own forward and are not named
    submodules, so without the enclosing range they would be unattributed. The
    correlation sweep picks the innermost, so a child still wins where present.
    """
    from gitm.tracer.vllm_stats import instrument_model

    push, pop, stack, seen = _record()
    tree = _qwen_tree()
    instrument_model(tree, push=push, pop=pop)
    tree.forward()

    assert stack == [], "every push must be matched by a pop"
    nested = [s for s in seen if s[-1] == "L3/qkv_proj"]
    assert nested, "qkv_proj range never opened"
    assert "L3/attn_score_value" in nested[0], "child range is not inside its block"


def test_removal_detaches_every_hook():
    from gitm.tracer.vllm_stats import instrument_model

    push, pop, stack, seen = _record()
    tree = _qwen_tree()
    inst = instrument_model(tree, push=push, pop=pop)
    inst.remove()
    tree.forward()
    assert seen == [] and stack == []


def test_an_unrecognised_tree_yields_no_ranges_rather_than_raising():
    """vLLM moves module layouts between releases. Fewer ranges is a degraded
    capture; an exception is a dead engine."""
    from gitm.tracer.vllm_stats import instrument_model

    root = _FakeModule()
    for p in ("backbone.block_0.attention_thing", "head.projector"):
        root.add(p)
    assert instrument_model(root, push=lambda n: None, pop=lambda: None).ranges == []


def test_a_model_without_named_modules_is_not_an_error():
    from gitm.tracer.vllm_stats import instrument_model

    assert instrument_model(object(), push=lambda n: None, pop=lambda: None).ranges == []
