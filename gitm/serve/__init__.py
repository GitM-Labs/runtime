"""vLLM server kernel capture — the two ways gitm can get a trace out of a server.

    gitm capture serve ...    launch the server under the collector, then capture
    gitm capture attach ...   adopt a server that is already running, then capture

They exist as a pair because the two situations are genuinely different, not because
one is a convenience wrapper on the other. Launching is what a benchmark needs:
gitm controls the flags, waits out weight load and CUDA-graph capture, drives
reproducible load, and tears the server down. Attaching is what a production
investigation needs: the server is already up, the traffic is real, and the answer
must come without restarting anything or rerouting a single request.

What makes attach possible at all is that the collector's window is opened by a
file — the injected library stats ``$GITM_TRACE_OUT.arm`` on every buffer flush — so
the process that opens the window need not be the process that started the server.
What makes it *impossible* in one specific case is the mirror image of the same
mechanism: ``CUDA_INJECTION64_PATH`` is read by the driver once, at CUDA init, so a
server started without it can never be traced while it runs. :mod:`gitm.serve.discover`
detects that up front and prints the restart that fixes it, rather than letting the
run end in an empty trace.
"""

from gitm.serve.artifacts import CaptureResult
from gitm.serve.attach import AttachOptions, attach_and_capture, describe_targets
from gitm.serve.discover import Target, classify, find_vllm_pids, resolve_target
from gitm.serve.vllm import launch_and_capture

__all__ = [
    "AttachOptions",
    "CaptureResult",
    "Target",
    "attach_and_capture",
    "classify",
    "describe_targets",
    "find_vllm_pids",
    "launch_and_capture",
    "resolve_target",
]
