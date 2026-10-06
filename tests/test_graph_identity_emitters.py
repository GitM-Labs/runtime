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
