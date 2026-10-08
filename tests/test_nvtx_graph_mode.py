"""--nvtx preflight: per-layer identity needs vLLM's hooks to run, i.e. no torch.compile."""

from __future__ import annotations

import pytest

from gitm.serve.vllm import _compilation, check_nvtx_graph_mode

BASE = ["vllm", "serve", "m"]


@pytest.mark.parametrize("extra,compiled,graphs,status", [
    ([], True, True, "warn"),
    (["-O3"], True, True, "warn"),
    (["--enforce-eager"], False, False, "pass"),
    (["-O0"], False, False, "pass"),
    (["-O", "0"], False, False, "pass"),
    (["--compilation-config", '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'],
     False, True, "pass"),
    (['-cc={"mode": 0, "cudagraph_mode": "NONE"}'], False, False, "pass"),
    (["--compilation-config", '{"mode": 3, "cudagraph_mode": "FULL"}'], True, True, "warn"),
    (["--compilation-config", '{"cudagraph_mode": "NONE"}'], True, False, "warn"),
])
def test_launch_shapes(extra, compiled, graphs, status):
    assert _compilation(BASE + extra) == (compiled, graphs)
    assert check_nvtx_graph_mode(BASE + extra)[0].status == status


def test_the_warning_names_a_launch_shape_that_keeps_graphs():
    detail = check_nvtx_graph_mode(BASE)[0].detail
    assert "--enforce-eager" in detail and '"mode": 0' in detail
