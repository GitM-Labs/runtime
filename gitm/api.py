"""Public embedded API.

    from gitm import optimize
    optimize(engine, budget="24h", target=0.15)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gitm.scheduler import LoopConfig, run_loop


def optimize(
    engine: Any | None = None,
    *,
    workload: str | None = None,
    budget: str = "24h",
    target: float = 0.15,
    scratch: str | None = None,
    workload_runner: Callable[[], dict[str, Any]] | None = None,
    use_history: bool | None = None,
    rerank: str = "off",
) -> dict[str, Any]:
    """Run the autonomous 24-hour optimization loop and return a report.

    Either pass an ``engine`` (e.g. a running vLLM engine handle) for the
    embedded path, or pass ``workload`` (e.g. ``"vllm-decode"``) for the CLI
    path. ``budget`` and ``target`` follow the SKU contract: a verified floor
    of ``target`` fraction improvement within ``budget`` wall time, or a
    qualification-gate diagnostic explaining why the floor was not committed.

    ``use_history`` ranks candidates from what previous runs measured on this
    GPU rather than from the library's hand-authored estimates. It is never
    asked for here — this entry point does not touch stdin, so an embedded
    caller cannot be blocked by a prompt it did not expect. ``gitm run`` puts
    the question to the operator and passes the answer down.

    ``rerank`` decides what the run does with what it learns between one
    candidate and the next. ``"off"`` keeps the opening order to the end;
    ``"recapture"`` traces the workload again after each applied candidate and
    re-ranks what is left against it.

    ``workload_runner`` optionally supplies an explicit zero-arg callable that
    launches the workload's GPU work; it runs inside the capture window. When
    omitted, the loop resolves ``workload`` against the registry in
    :mod:`gitm.workloads`.
    """
    cfg = LoopConfig(
        engine=engine,
        workload=workload,
        budget=budget,
        target=target,
        scratch=scratch,
        workload_runner=workload_runner,
        use_history=use_history,
        rerank=rerank,
    )
    return run_loop(cfg)
