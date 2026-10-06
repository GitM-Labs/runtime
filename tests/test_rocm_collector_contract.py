"""The ROCm collector (rocm_inject.c) and the decoder must agree on the contract.

Nothing on a test box loads rocprofiler-sdk, so the correlation tests feed the
decoder dicts — which would pass against a collector that never wrote the
fields, or wrote them in a different layout. These checks read the C source
itself, so the two sides cannot drift without a failure here:

* the graph-node id layout (flag bit, sequence and ordinal widths, the
  untracked-exec sentinel) is the same number on both sides;
* every ``fprintf`` has as many arguments as conversion specifiers — a drift
  writes garbage JSON for every record, silently;
* each record kind carries the keys the decoder reads;
* every HIP API operation the callback handles is in the configured filter,
  or the handler never runs.

A real compile against rocprofiler-sdk headers runs when ``GITM_ROCM_INCLUDE``
points at a header tree (``scripts/check_rocm_collector.py`` fetches one for a
given ROCm release and runs the same compile).
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

from gitm.distributed import correlate as C
from gitm.tracer import injection, vendor

SRC = (pathlib.Path(__file__).resolve().parents[1] / "gitm" / "tracer" / "_rocm"
       / "rocm_inject.c").read_text()


def _define(name: str) -> str:
    m = re.search(rf"^#define {name}\s+(.+?)\s*(?:/\*.*)?$", SRC, re.MULTILINE)
    assert m, name
    return m.group(1)


def test_node_id_layout_matches_the_decoder():
    assert int(_define("GITM_NODE_SEQ_BITS")) == C.NODE_SEQ_BITS
    assert int(_define("GITM_NODE_ORD_BITS")) == C.NODE_ORDINAL_BITS
    assert _define("GITM_NODE_FLAG") == "(1ULL << 63)"
    assert C.NODE_CAPTURE_FLAG == 1 << 63
    assert _define("GITM_EXEC_UNTRACKED") == "GITM_NODE_SEQ_MASK"
    assert C.GRAPH_UNTRACKED == (1 << C.NODE_SEQ_BITS) - 1


def test_node_id_formula_matches_the_decoder():
    """C: ((seq & SEQ_MASK) << ORD_BITS) | ((ordinal + 1) & ORD_MASK)."""
    body = re.search(r"static inline uint64_t node_id\(.*?\{(.*?)\}", SRC, re.DOTALL).group(1)
    assert "((seq & GITM_NODE_SEQ_MASK) << GITM_NODE_ORD_BITS)" in body
    assert "((ordinal + 1) & GITM_NODE_ORD_MASK)" in body
    seq_mask, ord_mask = (1 << 31) - 1, (1 << 32) - 1
    for seq, ordinal in [(1, 0), (7, 41), (seq_mask, ord_mask - 1)]:
        c_value = ((seq & seq_mask) << 32) | ((ordinal + 1) & ord_mask)
        assert c_value == C.exec_node_id(seq, ordinal)
        assert (1 << 63) | c_value == C.capture_node_id(seq, ordinal)
        assert C.node_ordinal(c_value) == ordinal


def test_a_graph_stamp_decodes_to_the_replay_node_id():
    """decode_stamp: graph_node_id = ext & ~FLAG must equal exec_node_id."""
    body = re.search(r"static stamp_t decode_stamp\(.*?\n\}", SRC, re.DOTALL).group(0)
    assert "s.graph_node_id = ext & ~GITM_NODE_FLAG" in body
    assert "s.graph_id = (ext >> GITM_NODE_ORD_BITS) & GITM_NODE_SEQ_MASK" in body
    ext = (1 << 63) | C.exec_node_id(3, 9)
    assert ext & ~(1 << 63) == C.exec_node_id(3, 9)
    assert (ext >> 32) & ((1 << 31) - 1) == 3


def _fprintf_calls(src: str) -> list[str]:
    calls, i = [], 0
    while (i := src.find("fprintf(", i)) != -1:
        depth, j = 0, i + len("fprintf")
        in_str = False
        while j < len(src):
            ch = src[j]
            if in_str:
                if ch == "\\":
                    j += 2
                    continue
                if ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        calls.append(src[i:j + 1])
        i = j
    return calls


def _split_args(call: str) -> list[str]:
    inner = call[call.index("(") + 1:-1]
    args, depth, cur, in_str, k = [], 0, "", False, 0
    while k < len(inner):
        ch = inner[k]
        if in_str:
            cur += ch
            if ch == "\\":
                cur += inner[k + 1]
                k += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            cur += ch
        elif ch in "([":
            depth += 1
            cur += ch
        elif ch in ")]":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
        k += 1
    args.append(cur.strip())
    return args


@pytest.mark.parametrize("call", _fprintf_calls(SRC), ids=lambda c: c[:60].replace("\n", " "))
def test_every_fprintf_has_one_argument_per_conversion(call):
    args = _split_args(call)
    fmt_parts = []
    rest = args[1:]
    while rest and rest[0].startswith('"'):
        fmt_parts.append(rest.pop(0))
    # Adjacent literals with no comma between them are one argument.
    fmt = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', " ".join(fmt_parts)))
    specs = re.findall(r"%(?:llu|u|d|s|04x)", fmt.replace("%%", ""))
    assert len(specs) == len(rest), (fmt, rest)


def _emitted_keys(marker: str) -> set[str]:
    """Keys written in the branch that starts by emitting ``marker``."""
    start = SRC.index(marker)
    end = min(i for i in (SRC.find("} else if (h->kind", start + 1),
                          SRC.find("\n}\n", start)) if i != -1)
    return set(re.findall(r'\\"([a-z_]+)\\":', SRC[start:end]))


def test_kernel_records_carry_every_identity_field_the_decoder_reads():
    keys = _emitted_keys('fputs("{\\"kind\\":\\"kernel\\",\\"name\\":"')
    assert {"correlation_id", "range_id", "kernel_id", "thread_id", "graph_id",
            "graph_node_id", "grid", "block", "stream_id", "device_id"} <= keys


def test_memcpy_and_runtime_records_carry_stamps():
    assert {"range_id", "graph_id", "graph_node_id", "correlation_id"} <= _emitted_keys(
        '"{\\"kind\\":\\"memcpy\\"')
    assert {"range_id", "graph_launch", "correlation_id", "thread_id"} <= _emitted_keys(
        '"{\\"kind\\":\\"runtime\\"')


def test_structural_records_carry_what_correlation_joins_on():
    node = _emitted_keys('"{\\"kind\\":\\"graph_node\\"')
    assert {"graph_node_id", "capture_id", "node_kind", "kernel_id", "name"} <= node
    assert "grid" in SRC[SRC.index("static void emit_graph_node"):SRC.index(
        "static void emit_graph_exec")]
    assert {"graph_id", "capture_id", "n_nodes"} <= _emitted_keys(
        '"{\\"kind\\":\\"graph_exec\\",\\"graph_id\\":%llu,"')


def test_structural_records_are_written_whether_or_not_armed():
    """Graphs are captured at engine start, long before a window arms."""
    for fn in ("emit_graph_node", "emit_graph_exec", "emit_collector_meta"):
        body = SRC[SRC.index(f"static void {fn}"):]
        body = body[:body.index("\n}\n")]
        assert "g_armed" not in body and "refresh_armed" not in body, fn
    assert set(injection.CORRELATION_KINDS) >= {"graph_node", "graph_exec", "marker", "runtime"}


def test_the_collector_names_itself_as_vendor_classification_expects():
    assert '\\"collector\\":\\"rocprofiler-sdk\\"' in SRC
    recs = [{"kind": "meta", "collector": "rocprofiler-sdk"}]
    assert vendor.classify_trace(recs).vendor == "amd"


def test_every_handled_hip_op_is_in_the_callback_filter():
    filt = SRC[SRC.index("k_hip_ops[] = {"):]
    filt = set(re.findall(r"ROCPROFILER_HIP_RUNTIME_API_ID_(\w+)", filt[:filt.index("};")]))
    handled = SRC[SRC.index("static void capture_node"):SRC.index("static const rocprofiler_"
                                                                  "tracing_operation_t k_hip_ops")]
    used = set(re.findall(r"ROCPROFILER_HIP_RUNTIME_API_ID_(\w+)", handled))
    assert used <= filt, used - filt
    for op in ("hipGraphInstantiate", "hipGraphInstantiateWithFlags",
               "hipGraphInstantiateWithParams", "hipGraphLaunch", "hipGraphLaunch_spt",
               "hipStreamBeginCapture", "hipStreamBeginCaptureToGraph", "hipStreamEndCapture",
               "hipModuleLaunchKernel", "hipExtModuleLaunchKernel", "hipLaunchKernel"):
        assert op in filt, op


def test_stamps_are_requested_for_every_record_kind_that_carries_one():
    block = SRC[SRC.index("rocprofiler_external_correlation_id_request_kind_t kinds[]"):]
    block = block[:block.index("};")]
    for kind in ("KERNEL_DISPATCH", "MEMORY_COPY", "HIP_RUNTIME_API"):
        assert f"ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_{kind}" in block


def _compiler() -> list[str] | None:
    if os.environ.get("CC") and shutil.which(os.environ["CC"]):
        return [os.environ["CC"]]
    for cc in ("cc", "gcc", "clang"):
        if shutil.which(cc):
            return [cc]
    try:
        import ziglang  # noqa: F401
    except ImportError:
        return None
    return [sys.executable, "-m", "ziglang", "cc", "-target", "x86_64-linux-gnu"]


def test_compiles_against_rocprofiler_sdk_headers(tmp_path):
    inc = os.environ.get("GITM_ROCM_INCLUDE")
    if not inc:
        pytest.skip("set GITM_ROCM_INCLUDE (scripts/check_rocm_collector.py fetches one)")
    cc = _compiler()
    if cc is None:
        pytest.skip("no C compiler")
    src = pathlib.Path(__file__).resolve().parents[1] / "gitm/tracer/_rocm/rocm_inject.c"
    out = tmp_path / "rocm_inject.o"
    r = subprocess.run([*cc, "-c", "-fPIC", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
                        "-D__HIP_PLATFORM_AMD__", "-isystem", inc, str(src), "-o", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and out.exists(), r.stderr[-4000:]
