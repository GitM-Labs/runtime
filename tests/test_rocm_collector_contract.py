"""rocm_inject.c and the decoder must agree on the record contract.

Correlation tests feed the decoder dicts, which would pass against a collector
that never wrote the fields; these read the C source. The real compile runs when
GITM_ROCM_INCLUDE points at a header tree (scripts/check_collectors.py).
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re

import pytest

from gitm.distributed import correlate as C
from gitm.tracer import injection, vendor

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = (ROOT / "gitm/tracer/_rocm/rocm_inject.c").read_text()


def _define(name: str) -> str:
    return re.search(rf"^#define {name}\s+(.+?)\s*$", SRC, re.MULTILINE).group(1)


def _between(start: str, end: str) -> str:
    i = SRC.index(start)
    return SRC[i:SRC.index(end, i + len(start))]


def test_node_id_layout_matches_the_decoder():
    assert int(_define("GITM_NODE_SEQ_BITS")) == C.NODE_SEQ_BITS
    assert int(_define("GITM_NODE_ORD_BITS")) == C.NODE_ORDINAL_BITS
    assert _define("GITM_NODE_FLAG") == "(1ULL << 63)" and C.NODE_CAPTURE_FLAG == 1 << 63
    assert _define("GITM_EXEC_UNTRACKED") == "GITM_NODE_SEQ_MASK"
    assert C.GRAPH_UNTRACKED == (1 << C.NODE_SEQ_BITS) - 1
    body = _between("static inline uint64_t node_id(", "}")
    assert "((seq & GITM_NODE_SEQ_MASK) << GITM_NODE_ORD_BITS)" in body
    assert "((ordinal + 1) & GITM_NODE_ORD_MASK)" in body
    for seq, ordinal in [(1, 0), (7, 41), ((1 << 31) - 1, (1 << 32) - 2)]:
        c_value = (seq << 32) | (ordinal + 1)
        assert c_value == C.exec_node_id(seq, ordinal)
        assert (1 << 63) | c_value == C.capture_node_id(seq, ordinal)
        assert C.node_ordinal(c_value) == ordinal


def test_a_graph_stamp_decodes_to_the_replay_node_id():
    body = _between("static stamp_t decode_stamp(", "\n}")
    assert "s.graph_node_id = ext & ~GITM_NODE_FLAG" in body
    assert "s.graph_id = (ext >> GITM_NODE_ORD_BITS) & GITM_NODE_SEQ_MASK" in body


def _fprintf_calls() -> list[tuple[str, list[str]]]:
    """(format, args) per fprintf: string literals are the format, the rest args."""
    out = []
    for m in re.finditer(r"fprintf\(g_fp,", SRC):
        depth, j = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(SRC[j], 0) if SRC[j] != '"' else 0
            if SRC[j] == '"':  # skip the literal
                j += 1
                while SRC[j] != '"':
                    j += 2 if SRC[j] == "\\" else 1
            j += 1
        body = SRC[m.end():j - 1]
        fmt = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', body))
        rest = re.sub(r'"(?:[^"\\]|\\.)*"', "", body)
        depth, args, cur = 0, [], ""
        for ch in rest:
            depth += (ch in "([") - (ch in ")]")
            if ch == "," and depth == 0:
                args.append(cur)
                cur = ""
            else:
                cur += ch
        args.append(cur)
        out.append((fmt, [a for a in args if a.strip()]))
    return out


@pytest.mark.parametrize("fmt,args", _fprintf_calls(), ids=lambda v: str(v)[:40])
def test_every_fprintf_has_one_argument_per_conversion(fmt, args):
    assert len(re.findall(r"%(?:llu|u|d|s|04x)", fmt.replace("%%", ""))) == len(args)


def _keys(marker: str) -> set[str]:
    i = SRC.index(marker)
    return set(re.findall(r'\\"([a-z_]+)\\":', SRC[i:SRC.index("}\\n", i)]))


def test_records_carry_what_the_decoder_joins_on():
    assert {"correlation_id", "range_id", "kernel_id", "thread_id", "graph_id",
            "graph_node_id", "grid", "block"} <= _keys('"{\\"kind\\":\\"kernel\\"')
    assert {"range_id", "graph_id", "graph_node_id"} <= _keys('"{\\"kind\\":\\"memcpy\\"')
    assert {"range_id", "graph_launch"} <= _keys('"{\\"kind\\":\\"runtime\\"')
    assert {"graph_node_id", "capture_id", "node_kind", "kernel_id", "name"} <= _keys(
        '"{\\"kind\\":\\"graph_node\\"')
    assert {"graph_id", "capture_id", "n_nodes"} <= _keys('"{\\"kind\\":\\"graph_exec\\"')


def test_structural_records_are_written_whether_or_not_armed():
    for fn in ("emit_graph_node", "emit_graph_exec", "emit_collector_meta"):
        assert "armed" not in _between(f"static void {fn}(", "\n}\n"), fn
    assert {"graph_node", "graph_exec"} <= set(injection.CORRELATION_KINDS)
    assert '\\"collector\\":\\"rocprofiler-sdk\\"' in SRC
    assert vendor.classify_trace([{"kind": "meta", "collector": "rocprofiler-sdk"}]).vendor == "amd"


def test_every_handled_hip_op_is_in_the_callback_filter():
    ops = lambda text: set(re.findall(r"ROCPROFILER_HIP_RUNTIME_API_ID_(\w+)", text))  # noqa: E731
    filt = ops(_between("k_hip_ops[] = {", "};"))
    handled = ops(_between("static void capture_node(", "static const rocprofiler_tracing"))
    assert handled <= filt, handled - filt
    kinds = _between("rocprofiler_external_correlation_id_request_kind_t kinds[]", "};")
    for k in ("KERNEL_DISPATCH", "MEMORY_COPY", "HIP_RUNTIME_API"):
        assert f"REQUEST_{k}" in kinds


def test_compiles_against_rocprofiler_sdk_headers(tmp_path):
    inc = os.environ.get("GITM_ROCM_INCLUDE")
    if not inc:
        pytest.skip("set GITM_ROCM_INCLUDE (scripts/check_collectors.py --keep DIR)")
    spec = importlib.util.spec_from_file_location("check", ROOT / "scripts/check_collectors.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    if check.compiler() is None:
        pytest.skip("no C compiler")
    r = check.compile_rocm(pathlib.Path(inc), tmp_path)
    assert r.returncode == 0, r.stderr[-4000:]
