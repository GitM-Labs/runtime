"""Turn ranked candidates into the arms a harness can run.

    arms, unreachable = plan_arms(base_argv, ranked)
    write_experiments(path, baseline_argv=base_argv, arms=arms, ...)

The epic says the loop *"emits experiment specs for the runtime experiment
harness"*. Nothing wrote one: every manifest was hand-written, and the nine
intervention slots in ``scripts/kimi_loop/gen_pods.py`` are a person doing this
module's job from the ranked list by eye.

**The output shape is not invented.** An arm is what
:func:`gitm.optimizer.harness_results.read_capture` consumes on the way back —
a served model, a load shape, and a server argv — because the only emitted
format worth having is one whose results can be read back. So this module is
written as the inverse of :func:`~gitm.optimizer.harness_results.knob_difference`
and :func:`~gitm.optimizer.harness_results.resolve_lever`, and a test asserts
that over the whole catalogue: emit an arm for a lever, diff it against the
baseline, resolve it, and get the same lever. That is #125's claim — resolution
matches on knob *and* value, so a catalogue-generated manifest round-trips by
construction — turned into something that fails when it stops being true.

Being the exact inverse is what makes the three awkward cases visible rather
than silently broken:

* **A lever realised by removing a flag.** ``cuda_graphs_enable`` is
  ``enforce_eager: false``, which an arm realises by *not* passing
  ``--enforce-eager``. If the baseline does not pass it either, the arm is the
  baseline, and ``compare`` rightly refuses two arms that ran the same flags. So
  it is reachable only against a baseline that sets it, and that depends on the
  baseline rather than on the lever.
* **A lever the baseline already runs.** Emitting it produces an arm identical
  to the baseline, which measures nothing.
* **An environment-variable lever.** ``VLLM_ATTENTION_BACKEND`` is not a server
  flag, and ``knob_difference`` reads only ``serve_argv``, so nothing in the
  return path can see it. The arm is still worth running; its result cannot be
  attributed automatically, and saying so is better than emitting it as though
  it could.

None of those are refusals of the lever. They are reported alongside the arms,
because a sweep that quietly dropped three of its candidates looks like a sweep
of the ones that remained.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gitm.optimizer.harness_results import LAUNCH_ONLY_FLAGS, parse_flags, realises
from gitm.optimizer.vllm_knobs import knob_from_server_arg, same_server_value, server_arg

__all__ = [
    "Arm",
    "Unreachable",
    "SCHEMA",
    "flag_for",
    "is_env_knob",
    "plan_arms",
    "write_experiments",
]

SCHEMA = "gitm/experiments/v1"

#: A serving-shaped load for a caller that has no baseline to take one from.
#: Never substituted for a missing one: ``read_capture`` refuses two arms whose
#: load shapes differ, so a sweep written under an invented load is a sweep whose
#: results the return path rejects.
DEFAULT_LOAD: dict[str, Any] = {
    "requests": 512, "concurrency": 256, "input_tokens": 1024,
    "output_tokens": 256, "seed": 42,
}


def is_env_knob(knob: str) -> bool:
    """Whether this knob is an environment variable rather than a server flag.

    By spelling, which is the catalogue's own convention and the only signal
    available: ``VLLM_ATTENTION_BACKEND`` against ``enable_expert_parallel``. A
    knob is one or the other, never both, and the server reads them by different
    routes.
    """
    return knob.isupper()


def flag_for(knob: str) -> str:
    """The server flag that sets ``knob``.

    The exact inverse of the normalisation ``resolve_lever`` applies
    (``knob.lstrip("-").replace("-", "_")``). Written here rather than inlined so
    the two sides of the round trip have one place to disagree, and so a test can
    assert they do not.
    """
    return "--" + knob.replace("_", "-")


@dataclass(frozen=True)
class Arm:
    """One experiment the harness can run, in the shape its results come back in."""

    lever: str
    knob: str
    value: Any
    serve_argv: tuple[str, ...]
    #: Extra environment for the server process. Empty for a flag lever.
    env: dict[str, str] = field(default_factory=dict)
    predicted_delta: float | None = None
    #: Whether ``gitm ingest`` can attribute this arm's result to this lever on
    #: the way back. False for an environment-variable lever, whose change is
    #: invisible to a server-argv diff.
    ingestable: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["serve_argv"] = list(self.serve_argv)
        return d


@dataclass(frozen=True)
class Unreachable:
    """A ranked lever that cannot become an arm against this baseline, and why."""

    lever: str
    knob: str
    reason: str


def _find(argv: Sequence[str], flag: str, booleans: frozenset[str] = frozenset()
          ) -> tuple[int, Any] | None:
    # The reader's parser, not a copy of it: see parse_flags.
    for i, tok, value in parse_flags(argv, booleans=booleans):
        if tok == flag:
            return i, value
    return None


def _without(argv: Sequence[str], flag: str, booleans: frozenset[str] = frozenset()
             ) -> list[str]:
    """``argv`` with ``flag`` and the value that belongs to it removed."""
    found = _find(argv, flag, booleans)
    if found is None:
        return list(argv)
    i, value = found
    span = 1 if value is True else 2
    return [*argv[:i], *argv[i + span:]]


def _with(argv: Sequence[str], flag: str, value: Any,
          booleans: frozenset[str] = frozenset()) -> list[str]:
    """``argv`` with ``flag`` set to ``value``, replacing any current setting.

    Replaced in place rather than appended, so an arm never carries the same
    flag twice. Two settings of one flag is a server-dependent precedence
    question, and the reader's parser would report only one of them.
    """
    out = _without(argv, flag, booleans)
    if value is True:
        return [*out, flag]
    return [*out, flag, str(value)]


def plan_arms(
    base_argv: Sequence[str], ranked: Iterable[Any], *, max_arms: int | None = None,
    booleans: frozenset[str] = frozenset(),
) -> tuple[list[Arm], list[Unreachable]]:
    """One arm per ranked candidate, plus the candidates that cannot become one.

    ``ranked`` takes :class:`gitm.agents.policy.RankedCandidate` objects, or bare
    specs. A candidate the ranking already rejected is not emitted: the gate's
    answer is categorical and re-asking it here would spend cluster time on a
    lever the loop declined locally.

    ``booleans`` are the flags that never take a value (see
    :func:`~gitm.optimizer.harness_results.boolean_flags`), so a model placed
    after one is not read as its value and removed with it.

    ``max_arms`` is checked before an arm is built rather than after it is
    appended, so it holds on every path. Checking it after meant an
    environment-variable arm — which takes an early exit — never saw the cap, and
    ``max_arms=0`` still emitted one. One arm is one cluster job, so the cap is a
    budget and has to be exact.
    """
    arms: list[Arm] = []
    out: list[Unreachable] = []
    base = list(base_argv)

    for item in ranked:
        if max_arms is not None and len(arms) >= max_arms:
            break
        spec = getattr(item, "spec", item)
        rejected = getattr(item, "rejected_reason", None)
        predicted = getattr(item, "predicted_delta", None)
        name, knob = spec.name, spec.knob

        if rejected is not None:
            out.append(Unreachable(name, knob, f"not ranked: {rejected}"))
            continue

        extra = dict(getattr(spec, "knobs", {}) or {})
        if len(extra) > 1:
            out.append(Unreachable(
                name, knob,
                f"sets {len(extra)} knobs at once ({', '.join(sorted(extra))}); a "
                "harness arm's delta cannot be credited to one lever"))
            continue
        if not knob and extra:
            knob = next(iter(extra))

        if is_env_knob(knob):
            # Worth running, not attributable on the way back. Emitted with the
            # baseline's own argv, so the server differs only in environment.
            arms.append(Arm(
                lever=name, knob=knob, value=spec.value,
                serve_argv=tuple(base), env={knob: str(spec.value)},
                predicted_delta=predicted, ingestable=False))
            continue

        # The flag that actually sets this knob on the server, which for a knob
        # vLLM takes nested (speculative decoding) is not flag_for(knob).
        flag, arg_value = server_arg(knob, spec.value)
        if flag in LAUNCH_ONLY_FLAGS:
            out.append(Unreachable(
                name, knob,
                f"{flag} is launch-only, so the return path excludes it from the "
                "flags it diffs and this arm would read as identical"))
            continue

        current = _find(base, flag, booleans)
        if current is not None:
            # Merged into what the baseline passes, so the arm changes this one
            # setting rather than replacing the baseline's whole nested config.
            _, arg_value = server_arg(knob, spec.value, current[1])
        if spec.value is False:
            if current is None:
                out.append(Unreachable(
                    name, knob,
                    f"realised only by removing {flag}, which this baseline does "
                    "not set, so the arm would be the baseline"))
                continue
            argv = _without(base, flag, booleans)
        elif current is not None and (
                same_server_value(arg_value, current[1])
                or realises(knob_from_server_arg(flag.lstrip("-").replace("-", "_"),
                                                 current[1])[1], spec.value)):
            out.append(Unreachable(
                name, knob,
                f"the baseline already runs {flag}={current[1]}, so the arm would "
                "measure nothing"))
            continue
        else:
            argv = _with(base, flag, arg_value, booleans)

        arms.append(Arm(
            lever=name, knob=knob, value=spec.value,
            serve_argv=tuple(argv), predicted_delta=predicted))

    return arms, out


def write_experiments(
    path: str | Path, *, baseline_argv: Sequence[str], arms: Sequence[Arm],
    served_model: str, load: dict[str, Any], unreachable: Sequence[Unreachable] = (),
    run_id: str | None = None, notes: str | None = None,
    fingerprint: str | None = None, gpu_sku: str | None = None,
) -> str:
    """Write the sweep as JSON. Returns the path written.

    The baseline is written as its own entry rather than left implicit. Which arm
    is the baseline is not recoverable from a set of directories afterwards — the
    one with fewer flags is a guess, and a wrong guess inverts the sign of every
    delta — so the file that commissioned the sweep is where it has to be said.

    ``fingerprint`` and ``gpu_sku`` are written into the ingest command this file
    carries, because the two halves of the loop have to agree on them or the
    results never reach the next proposal. The command used to be emitted without
    either: an operator following it verbatim got a sweep filed under a key the
    next ``gitm propose`` did not look under, and the loop silently did not close.

    ``load`` is required and written exactly as given, one shape for every arm.
    ``read_capture`` refuses two arms whose load shapes differ, since a lever
    measured under a different load is not an A/B of the baseline — so
    substituting a default for a baseline that declared none would commission a
    sweep the return path then refuses wholesale. The caller establishes the load
    or there is no comparable sweep to write.
    """
    path = Path(path)
    doc = {
        "schema": SCHEMA,
        "run_id": run_id,
        "served_model": served_model,
        "load": dict(load),
        "notes": notes,
        "baseline": {"name": "baseline", "serve_argv": list(baseline_argv)},
        "arms": [a.to_dict() for a in arms],
        # Beside the arms, never omitted. A sweep that quietly dropped three of
        # its candidates looks like a sweep of the ones that remained, and the
        # reasons are the most useful thing here when a lever never gets tested.
        "unreachable": [asdict(u) for u in unreachable],
        "fingerprint": fingerprint,
        "gpu_sku": gpu_sku,
        "ingest": {
            # Complete, including the keys. The ranking filters history on the
            # GPU and the workload digest, so a command missing either files
            # results the next proposal cannot find.
            #
            # Every substituted value goes through shlex.quote, because this is a
            # line someone pastes into a shell. A SKU has spaces in it, a
            # fingerprint is whatever the operator passed, and the directory
            # placeholders are angle brackets — which a shell reads as
            # redirection, not as a blank to fill in.
            "command": " ".join([
                "gitm ingest --baseline", shlex.quote("<baseline dir>"),
                *(f"--candidate {shlex.quote(f'<{a.lever} dir>')}"
                  for a in arms if a.ingestable),
                "--gpu-sku", shlex.quote(gpu_sku or "<sku>"),
                *(["--fingerprint", shlex.quote(fingerprint)] if fingerprint else []),
            ]),
            "not_attributable": [a.lever for a in arms if not a.ingestable],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    return str(path)
