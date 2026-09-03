"""Seam 2 — run the argv the replay module builds, and say what happened.

:meth:`~gitm.traffic.replay.ReplayPlan.bench_serve_argv` builds the command;
nothing ran it. This module does, and it is deliberately thin — the interesting
work was already done, and everything hard about talking to a vLLM server is
already solved in :mod:`gitm.serve.vllm`, which this reuses rather than reimplements.

Three things it does that a bare ``subprocess.run`` would not:

**The version guard fires first.** Below :data:`~gitm.traffic.replay.VLLM_MIN_VERSION`
there is no ``timed_trace`` dataset, and vLLM's own failure is an argparse
complaint about an unknown dataset name — which reads like a typo in *our*
command rather than a missing feature, and costs an afternoon. Checking before
launching converts that into a sentence naming the version and the flag.

**The model id comes from the server, not from the caller.** ``--served-model-name``
can rename a model, and a completion request with the wrong id is a **404, not a
slow path** — a whole run that looks like it went badly rather than one that never
started. :func:`gitm.serve.vllm.served_model_name` already knows this; asking it
also proves the endpoint is answering before we fire a few thousand requests at it.

**The result is joined, not just carried.** ``bench serve``'s JSON has the
metrics and **no trace identity, no regime, no config capture** — and under
``--self-timed`` it still records the CLI's ``request_rate`` and ``burstiness``
defaults, which are wrong because the real values came from the trace. Pass
``regime=`` and :func:`run_replay` hands the result to
:func:`gitm.traffic.results.join_result` (seam 3), which attaches the workload
identity, drops those two fields with a reason, and reconciles the reported
totals against the trace. Without ``regime=`` the raw result is still carried and
a note says what is missing — joining is not silently skipped.

Verified against real ``vllm bench serve`` 0.28.0: 40/40 requests, paced to
12.008 s against a 12.000 s trace span. The parts that still need a server —
output-length fidelity, and whether the schedule holds when a real server
saturates — are named in the standup's ``verification.md``. :func:`run_replay`
takes ``dry_run`` so the argv and the guard stay exercisable without one.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from gitm.traffic.regime import Regime
from gitm.traffic.replay import VLLM_MIN_VERSION, ReplayPlan
from gitm.traffic.results import BenchRun, join_result
from gitm.traffic.schema import TraceMeta

#: How much of a failed run's output to keep. Enough to see the argparse line or
#: the traceback; not so much that a result row carries a log file.
TAIL_CHARS = 4000


class VllmUnavailable(RuntimeError):
    """vLLM is missing or too old. The message names the version and the flag."""


def installed_vllm_version() -> str | None:
    """The installed vLLM version, or ``None`` if it is not importable.

    Reads package metadata rather than importing ``vllm``: the import pulls in
    torch and CUDA and takes tens of seconds, and this runs before every launch.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("vllm")
        except PackageNotFoundError:
            return None
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib
        return None


def _release(v: str) -> tuple[int, ...]:
    """The numeric release part of a version, for comparison.

    ``0.23.0rc1`` -> ``(0, 23, 0)``; ``0.23.0+cu128`` -> ``(0, 23, 0)``. Not a
    full PEP 440 implementation on purpose — this compares a floor, and the only
    thing that matters is that a release candidate does not read as newer than
    its release, nor a local/build tag as older.
    """
    head = re.split(r"[^0-9.]", v.strip(), maxsplit=1)[0].rstrip(".")
    return tuple(int(p) for p in head.split(".") if p.isdigit()) or (0,)


def vllm_executable() -> str:
    """The ``vllm`` console script **belonging to the running interpreter**.

    Not a bare ``"vllm"`` on ``PATH``, for two reasons and the second is the
    serious one:

    * ``python -m gitm.traffic`` run with an absolute interpreter — a conda env
      invoked without activation, which is the normal case in CI and under WSL —
      has that env's ``bin/`` nowhere on ``PATH``, and the launch dies with
      ``FileNotFoundError: 'vllm'``.
    * **The guard and the run could disagree.** :func:`check_vllm` reads
      ``importlib.metadata.version("vllm")``, which is the version installed
      *for this interpreter*. Resolving the binary from ``PATH`` could then run a
      different environment's vLLM — so the version that was validated and the
      version that runs would not be the same install. Deriving both from
      ``sys.executable`` makes that impossible.

    Falls back to ``"vllm"`` when no sibling script exists, so an unusual layout
    still gets a ``PATH`` lookup rather than a hard failure.
    """
    bindir = Path(sys.executable).parent
    for name in ("vllm", "vllm.exe"):
        cand = bindir / name
        if cand.exists():
            return str(cand)
    found = shutil.which("vllm")
    return found or "vllm"


def check_vllm(min_version: str = VLLM_MIN_VERSION) -> str:
    """Return the installed version, or raise with what to do about it.

    Called before launching, never after: the whole point is that the failure
    arrives as a sentence about ``timed_trace`` rather than as vLLM's argparse
    error about an unknown dataset name.
    """
    found = installed_vllm_version()
    if found is None:
        raise VllmUnavailable(
            f"vllm is not installed. Firing a replay needs vllm>={min_version} "
            f"(the version that added `bench serve --dataset-name timed_trace`). "
            f"Everything else in gitm.traffic is CPU-only and needs no vLLM. "
            f"Install: pip install 'gitm-labs[vllm]'"
        )
    if _release(found) < _release(min_version):
        raise VllmUnavailable(
            f"vllm {found} is too old: `--dataset-name timed_trace` needs "
            f">={min_version}. Below it the run dies on an argparse complaint "
            f"about an unknown dataset name, which reads like a typo in our "
            f"command rather than a missing feature. Upgrade: "
            f"pip install -U 'vllm>={min_version}'"
        )
    return found


class RunResult(BaseModel):
    """What the run did, with the plan's provenance and — given a regime — the join.

    Two levels on purpose: :attr:`result` is vLLM's JSON untouched, and
    :attr:`joined` is that result tied to the workload, with the misleading fields
    dropped and the reconciliation checked. Keeping the raw one means the join is
    auditable rather than something you have to trust.
    """

    model_config = ConfigDict(extra="forbid")

    argv: list[str]
    returncode: int
    duration_s: float
    vllm_version: str | None = None
    resolved_model: str | None = None  # what the server calls it, not what we asked for
    stdout_tail: str = ""
    stderr_tail: str = ""
    result_path: str | None = None
    #: ``bench serve``'s raw result JSON, unmodified. **Do not read
    #: ``request_rate`` or ``burstiness`` out of this** — under ``--self-timed``
    #: they are the CLI's untouched defaults, and the true values are on
    #: :attr:`source`'s regime. :attr:`joined` is that join already done
    #: (:mod:`gitm.traffic.results`); this field is the unmodified original.
    result: dict | None = None
    #: The trace this came from, verbatim from the plan. A measured number whose
    #: workload cannot be identified is not evidence, and this is the field that
    #: stops the two being separated between here and a playbook row.
    source: TraceMeta
    #: The joined record — result metrics tied to the regime that produced them,
    #: with vLLM's misleading ``request_rate`` / ``burstiness`` dropped and the
    #: reconciliation checked. Present when a ``regime`` was supplied and a result
    #: file came back; ``None`` otherwise. See :mod:`gitm.traffic.results`.
    joined: BenchRun | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def summary(self) -> str:
        head = "ok" if self.ok else f"FAILED rc={self.returncode}"
        got = f", {len(self.result)} result keys" if self.result else ""
        return (
            f"{head} in {self.duration_s:.1f}s — {self.source.source} "
            f"({self.source.rows_emitted} requests){got}"
        )


def run_replay(
    plan: ReplayPlan,
    *,
    model: str,
    base_url: str = "http://127.0.0.1:8000",
    backend: str = "openai",
    tokenizer: str | None = None,
    max_concurrency: int | None = None,
    result_dir: str | Path | None = None,
    seed: int = 0,
    regime: Regime | None = None,
    timeout_s: float = 3600.0,
    resolve_model: bool = True,
    dry_run: bool = False,
) -> RunResult:
    """Fire ``plan`` at ``base_url`` with ``vllm bench serve``.

    Order of operations, and the first two are the reason this is not a bare
    ``subprocess.run``:

    1. **Version guard**, before anything is launched.
    2. **Resolve the served model id** off ``/v1/models``, which also proves the
       endpoint is answering. ``resolve_model=False`` skips it and uses ``model``
       verbatim — for a server that does not expose the route.
    3. Run, capturing stdout, stderr and the exit code.
    4. Read back the result JSON if one was written, and — when a ``regime`` is
       supplied — **join** it to the workload that produced it (seam 3). The
       join is what makes the result evidence rather than a number: it attaches
       the regime and trace identity the result JSON has no room for, drops the
       two fields that are wrong under ``--self-timed``, and reconciles the
       reported totals against the trace.

    ``dry_run`` builds and returns everything except the subprocess, so the argv
    and the guard are exercisable with no vLLM and no server.
    """
    notes: list[str] = []
    joined: BenchRun | None = None
    found = None if dry_run else check_vllm()

    resolved = model
    if resolve_model and not dry_run:
        from gitm.serve.vllm import served_model_name

        resolved = served_model_name(base_url, model)
        if resolved != model:
            notes.append(
                f"server calls the model {resolved!r}, not {model!r} "
                "(--served-model-name); using the server's id — the wrong one is a 404"
            )

    result_path: Path | None = None
    if result_dir is not None:
        result_path = Path(result_dir) / f"benchserve_{plan.source.source}_{int(time.time())}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)

    argv = plan.bench_serve_argv(
        model=resolved,
        base_url=base_url,
        backend=backend,
        tokenizer=tokenizer,
        max_concurrency=max_concurrency,
        result_filename=str(result_path) if result_path else None,
        seed=seed,
    )

    if dry_run:
        notes.append("dry run: nothing was launched")
        return RunResult(argv=argv, returncode=0, duration_s=0.0, resolved_model=resolved,
                         result_path=str(result_path) if result_path else None,
                         source=plan.source, notes=notes)

    # argv[0] is the documented command name; run the console script that belongs
    # to THIS interpreter, so the install check_vllm() validated is the one that
    # runs. See vllm_executable().
    exe = vllm_executable()
    if exe != argv[0]:
        notes.append(f"running {exe} (the console script beside {sys.executable})")
        argv = [exe, *argv[1:]]

    started = time.monotonic()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        # A timeout is a result, not an exception to propagate: the partial output
        # is the only evidence of how far it got.
        rc, out, err = 124, (exc.stdout or b"").decode(errors="replace") if isinstance(
            exc.stdout, bytes) else (exc.stdout or ""), f"timed out after {timeout_s:.0f}s"
        notes.append(f"killed at the {timeout_s:.0f}s timeout")
    duration = time.monotonic() - started

    result = None
    if result_path is not None and result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"result file unreadable: {exc}")
        else:
            if regime is None:
                notes.append(
                    "result JSON carries no trace identity, no regime and no config "
                    "capture, and under --self-timed its request_rate and burstiness are "
                    "the CLI defaults rather than the trace's. Pass `regime=` to join it."
                )
            else:
                joined = join_result(result, plan, regime)
                notes.append(
                    f"joined to {joined.regime_label}; dropped {sorted(joined.dropped)} "
                    "as CLI defaults that --self-timed never consulted"
                )
                if not joined.reconciled:
                    notes.append(
                        "RESULT DOES NOT RECONCILE WITH ITS TRACE: "
                        + "; ".join(
                            f"{c.name} expected {c.expected} got {c.actual}"
                            for c in joined.failures()
                        )
                    )
    elif result_path is not None:
        notes.append(f"no result file at {result_path} despite --save-result")

    return RunResult(
        joined=joined,
        argv=argv, returncode=rc, duration_s=duration, vllm_version=found,
        resolved_model=resolved, stdout_tail=out[-TAIL_CHARS:], stderr_tail=err[-TAIL_CHARS:],
        result_path=str(result_path) if result_path else None, result=result,
        source=plan.source, notes=notes,
    )
