"""CUDA-graph identity must survive the native collector, not just the decoder.

The correlation tests build graph_id/graph_node_id into Python dicts directly,
so on their own they would pass against a collector that never emits the
fields. A kernel record without them decodes as an eager launch and takes the
launch range as its op, which is the mis-attribution the fields exist to stop.

Two checks close that gap: a source contract that runs everywhere, and an
end-to-end capture of a real graph replay that runs on a CUDA box with the shim
built.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_CUPTI = pathlib.Path(__file__).resolve().parents[1] / "gitm" / "tracer" / "_cupti"


def _kernel_branch(src: str) -> str:
    """The body of an emitter's GITM_REC_KERNEL branch, up to the next kind."""
    m = re.search(r"if \(r->kind == GITM_REC_KERNEL\)(.*?)else if \(r->kind ==",
                  src, re.DOTALL)
    assert m, "no GITM_REC_KERNEL branch found"
    return m.group(1)


def test_core_reads_graph_identity_off_the_kernel_record():
    src = (_CUPTI / "cupti_core.c").read_text()
    assert "r.graph_id = k->graphId;" in src
    assert "r.graph_node_id = k->graphNodeId;" in src


@pytest.mark.parametrize("emitter", ["cupti_shim.c", "cupti_inject.c"])
def test_both_emitters_write_graph_identity_on_kernels(emitter):
    branch = _kernel_branch((_CUPTI / emitter).read_text())
    for key in ("graph_id", "graph_node_id"):
        assert f'"{key}"' in branch or f'\\"{key}\\"' in branch, (emitter, key)
        assert f"r->{key}" in branch, (emitter, key)


def test_inject_kernel_format_matches_its_arguments():
    """A printf whose specifiers and arguments drift apart writes garbage JSON
    for every kernel, with no error. Count them."""
    branch = _kernel_branch((_CUPTI / "cupti_inject.c").read_text())
    call = branch[branch.index("fprintf("):]
    fmt = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', call.split("\n                (", 1)[0]))
    n_spec = len(re.findall(r"%(?:llu|u|d)", fmt))
    args = call[call.index('\\n",') + len('\\n",'):call.rindex(");")]
    n_args = len([a for a in args.split(",") if a.strip()])
    assert n_spec == n_args


def _gpu_backend():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from gitm.tracer._cupti import available

    if not available():
        pytest.skip("CUPTI shim not built (python -m gitm.tracer._cupti.build)")
    from gitm.tracer.cupti import CuptiBackend

    return torch, CuptiBackend()


def test_graph_replay_kernels_carry_graph_identity_end_to_end():
    torch, backend = _gpu_backend()

    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):  # warm up off the default stream before capture
        (a @ b).relu_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = (a @ b).relu_()
    torch.cuda.synchronize()

    backend.start()
    graph.replay()
    eager = a + b
    torch.cuda.synchronize()
    events = backend.stop()
    del out, eager

    kernels = [e for e in events if e.kind == "kernel"]
    replayed = [k for k in kernels if k.graph_id is not None]
    launched = [k for k in kernels if k.graph_id is None]
    assert replayed, "graph replay produced no kernel with a graph_id"
    assert all(k.graph_node_id is not None for k in replayed)
    assert len({k.graph_node_id for k in replayed}) == len(replayed)
    assert launched, "the eager add must decode as a non-graph launch"


# ── capture-time node map (cupti_core.c, NVTX + RESOURCE callbacks) ──────────

_CORE = (_CUPTI / "cupti_core.c").read_text()


def test_graph_nodes_are_written_before_the_arm_gate():
    """Graphs are captured at engine start, long before a window arms."""
    src = (_CUPTI / "cupti_inject.c").read_text()
    sink = src[src.index("static void file_sink"):]
    assert sink.index("GITM_REC_GRAPH_NODE") < sink.index("if (!g_armed) return;")
    for key in ("graph_node_id", "cloned_from", "node_kind", "name"):
        assert f'\\"{key}\\"' in sink


def test_the_shim_decodes_graph_nodes():
    src = (_CUPTI / "cupti_shim.c").read_text()
    branch = src[src.index("r->kind == GITM_REC_GRAPH_NODE"):]
    branch = branch[:branch.index("} else if")]
    assert all(f'"{k}"' in branch for k in ("graph_node_id", "cloned_from", "node_kind"))
    fmt = re.search(r'"(\{[^"]*\})"', branch).group(1)
    assert fmt.count("s:") == 5


def test_the_node_map_enables_every_callback_its_handler_reads():
    start = _CORE[_CORE.index("static void node_map_start"):]
    start = start[:start.index("\n}\n")]
    assert "CUPTI_CB_DOMAIN_NVTX" in start
    for cbid in ("GRAPHNODE_CREATED", "GRAPHNODE_CLONED"):
        assert f"CUPTI_CBID_RESOURCE_{cbid}" in start
    assert "RUNTIME_INSTANTIATE" in start and "DRIVER_INSTANTIATE" in start
    for push in ("nvtxRangePushA", "nvtxRangePushEx", "nvtxDomainRangePushEx",
                 "nvtxRangePop", "nvtxDomainRangePop"):
        assert f"CUPTI_CBID_NVTX_{push}" in _CORE


def test_nodes_created_by_instantiate_are_not_named():
    """cudaGraphInstantiate fires GRAPHNODE_CREATED for its copies; naming them
    after the range open at instantiate time would rename the captured nodes."""
    assert ("cbid == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED && !tls_nvtx.instantiating"
            in _CORE)
    # ...but a copy that names its original is kept, as a link with no name.
    assert "emit_node(id, orig, \"\", node_kind(g->nodeType));" in _CORE
    assert "node_map_start();" in _CORE and "node_map_stop();" in _CORE


def test_cupti_collector_compiles_against_cuda_headers(tmp_path):
    import importlib.util
    import os

    inc = os.environ.get("GITM_CUDA_INCLUDE")
    if not inc:
        pytest.skip("set GITM_CUDA_INCLUDE (scripts/check_collectors.py --keep DIR)")
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("check", root / "scripts/check_collectors.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    if check.compiler() is None:
        pytest.skip("no C compiler")
    for r in check.compile_cupti(pathlib.Path(inc), tmp_path):
        assert r.returncode == 0, r.stderr[-4000:]


def test_replayed_kernels_take_their_capture_range_end_to_end(monkeypatch):
    """Needs NVTX routed to CUPTI before the first range (NVTX_INJECTION64_PATH)."""
    import os

    if not os.environ.get("NVTX_INJECTION64_PATH"):
        pytest.skip("needs NVTX_INJECTION64_PATH -> libcupti (run_env(..., nvtx=True))")
    monkeypatch.setenv("GITM_TRACE_NVTX", "1")
    torch, backend = _gpu_backend()

    a = torch.randn(256, 256, device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        (a @ a).relu_()
    torch.cuda.synchronize()
    backend.start()  # the node map must be live during capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.cuda.nvtx.range_push("L4/mlp_down")
        out = a @ a
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("decode_step")
    graph.replay()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    events = backend.stop()
    del out
    replayed = [e for e in events if e.kind == "kernel" and e.graph_id is not None]
    assert replayed and all((k.range_op, k.range_layer) == ("mlp_down", 4) for k in replayed)
    assert {k.launch_range for k in replayed} == {"decode_step"}


def test_a_new_session_starts_with_an_empty_range_stack():
    """Pops while callbacks are off are never seen; without a reset a range from
    an earlier session would name the next session's nodes."""
    cb = _CORE[_CORE.index("static void CUPTIAPI on_callback"):]
    assert cb.index("tls_sync();") < cb.index("CUPTI_CB_DOMAIN_NVTX")
    start = _CORE[_CORE.index("static void node_map_start"):]
    assert "atomic_fetch_add(&g_session, 1);" in start[:start.index("\n}\n")]
