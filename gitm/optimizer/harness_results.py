"""Turn a pair of harness captures into a verification record the loop can read.

    a = read_capture("results/tp2-baseline")
    b = read_capture("results/tp2-ep")
    write_comparison(a, b, out_dir=runs_dir() / run_id)

`runtime-experiment-harness` runs experiments across a cluster and hands the
server to ``gitm capture serve``, which leaves ``serving_summary.json`` and
``run_manifest.json`` in each arm's directory. Nothing read them back.

:mod:`gitm.optimizer.history` aggregates ``runs/<run_id>/verification.json``, and
**only ``run_loop`` writes one** — so every result measured on the cluster was
invisible to the ranking that reads history. A loop that proposes experiments and
then cannot see their results is the failure the history reader exists to
prevent, one layer out.

This converts rather than teaching the reader a second format. ``history.py`` is
the most-tested module here and has been through several rounds of review
findings; giving it another input shape would put that at risk to save a file
write. A converted capture lands beside the loop's own exports and is read by the
same code, so a cluster result and a local one are the same kind of evidence.

**Pairing is explicit, never inferred.** Which arm is the baseline is not
recoverable from two directories — the one with fewer flags is a guess, and a
wrong guess silently inverts the sign of every delta it produces. The caller
says, or nothing is written.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gitm.optimizer.history import EXPORT_NAME
from gitm.optimizer.report import Provenance
from gitm.optimizer.verification_export import VerificationRecord, write_verification
from gitm.serve.discover import vllm_argv_start, vllm_launch_argv

__all__ = [
    "Capture",
    "fingerprint_of",
    "realises",
    "resolve_lever",
    "LAUNCH_ONLY_FLAGS",
    "CaptureError",
    "read_capture",
    "knob_difference",
    "parse_flags",
    "boolean_flags",
    "compare",
    "sweep_id",
    "write_comparison",
    "write_comparisons",
]

#: What an attached capture can honestly say about its tracing: the collector was
#: running, and whether NVTX markers were on is not recorded. Distinct from both
#: ``"cupti"`` and ``"cupti+nvtx"`` on purpose — see :func:`_tracing_from_attach`.
TRACING_NVTX_UNKNOWN = "cupti(nvtx:unknown)"

SUMMARY_NAME = "serving_summary.json"
MANIFEST_NAME = "run_manifest.json"


class CaptureError(ValueError):
    """A capture directory that cannot be read as one, with the reason."""


@dataclass(frozen=True)
class Capture:
    """One arm of a harness experiment, as it lands on disk."""

    path: Path
    served_model: str | None
    #: The server argv the harness launched. The knob under test is the
    #: difference between two of these, which is why the whole list is kept
    #: rather than a parsed subset.
    serve_argv: tuple[str, ...]
    #: Load shape: requests, concurrency, input/output tokens, seed. Two arms
    #: measured under different load are not an A/B, and this is what says so.
    load: dict[str, Any]
    #: ``off`` | ``cupti`` | ``cupti+nvtx``. Tracing costs throughput, so an arm
    #: traced against one that was not measures the tracer, not the knob.
    tracing: str | None
    throughput: float | None
    window_s: float | None
    #: ``True`` when the throughput above is SLO-qualified goodput. Two arms
    #: measured on different metrics are not comparable, and the raw request rate
    #: counts requests that missed their SLO.
    goodput: bool = False
    #: The capture's own trace, when it has one. An untraced arm cannot be
    #: fingerprinted, so it cannot be filed against a workload the loop knows.
    trace_path: Path | None = None
    #: The command that starts this server: entry point, positional model and
    #: flags. ``serve_argv`` is what two arms are *compared* on; this is what a
    #: proposed arm is *run* as. They are the same list for a launched capture,
    #: and differ for an attached one, whose ``serve_argv`` is its flags alone.
    launch_argv: tuple[str, ...] = ()

    @property
    def comparable_key(self) -> tuple:
        """What must match for two arms to be measuring the same thing."""
        return (self.served_model, self.tracing,
                tuple(sorted((k, str(v)) for k, v in self.load.items())))


#: Server flags that say where to listen, not what to compute. Two arms bound to
#: different ports are the same experiment, and folding a port into the knob
#: under test invents a lever no library entry can match.
LAUNCH_ONLY_FLAGS = frozenset({
    "--host", "--port", "--api-key", "--served-model-name", "--download-dir",
    "--uvicorn-log-level", "--root-path", "--allowed-origins", "--ssl-keyfile",
    "--ssl-certfile", "--disable-log-requests", "--disable-log-stats",
})


def fingerprint_of(capture: Capture) -> str:
    """The loop's own workload fingerprint, computed from the capture's trace.

    :func:`gitm.optimizer.qualification.fingerprint` hashes the set of
    ``(kernel name, grid, block)`` and prefixes the vendor, and the loop filters
    history on exactly that string. A record filed under anything else — a model
    name, say — is written and then filtered straight back out, which is
    invisible rather than merely coarse.

    Streamed line by line instead of loading the trace: a real capture is
    millions of kernels, and only the distinct shapes are needed. Pinned against
    the real function by a test, since a digest that drifts silently stops
    matching and nothing says why.
    """
    path = capture.trace_path
    if path is None or not path.exists():
        raise CaptureError(
            f"{capture.path.name}: no trace.jsonl, so no fingerprint. An untraced "
            "arm cannot be filed against a workload the loop would recognise.")
    vendor, shapes = None, set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError as exc:
                raise CaptureError(f"{path.name}: not valid JSONL: {exc}") from exc
            if "_header" in rec or vendor is None and rec.get("vendor"):
                header = rec.get("_header", rec)
                vendor = header.get("vendor", vendor)
                continue
            if rec.get("kind") != "kernel":
                continue
            shapes.add((
                rec.get("name"),
                (rec.get("grid_x") or 1) * (rec.get("grid_y") or 1) * (rec.get("grid_z") or 1),
                (rec.get("block_x") or 1) * (rec.get("block_y") or 1) * (rec.get("block_z") or 1),
            ))
    if not shapes:
        raise CaptureError(f"{path.name}: no kernel records, so no fingerprint")
    digest = hashlib.sha256(repr(sorted(shapes)).encode("utf-8")).hexdigest()[:16]
    return f"{vendor}:{digest}"


def _as_bool(value: Any) -> bool:
    """A flag value as the boolean a server would act on."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def realises(value: Any, spec_value: Any) -> bool:
    """Whether an arm setting a knob to ``value`` is the setting ``spec_value``.

    A lever is a knob *and* the value it puts there — :func:`apply_intervention`
    applies ``{spec.knob: spec.value}``. Matching on the knob alone credits an
    arm to a lever it did not run, and for a boolean it credits it to the
    opposite one: ``--enforce-eager`` sets ``enforce_eager`` true, while the
    only catalog entry on that knob is ``cuda_graphs_enable``, which sets it
    false. That record would be a win for eager mode read as evidence for
    disabling it.
    """
    if isinstance(spec_value, bool) or isinstance(value, bool):
        return _as_bool(value) is _as_bool(spec_value)
    if isinstance(spec_value, int | float):
        try:
            return float(value) == float(spec_value)
        except (TypeError, ValueError):
            return False
    return str(value).strip().lower() == str(spec_value).strip().lower()


def resolve_lever(knob: str, value: Any, library: Iterable[Any]) -> Any | None:
    """The catalog entry this flag change corresponds to, or ``None``.

    Ranking looks a record up by ``spec.name``, so a name invented from the flag
    is a record nothing can find. The names do not follow from the flags:
    ``--enforce-eager`` is the knob ``enforce_eager``, and ``--max-num-seqs`` is
    the lever ``max_num_seqs_dynamic``. Matching on ``knob`` is what bridges
    them, and matching on the value is what keeps the bridge honest: no catalog
    entry shares a knob with another, so a knob-only match would resolve *every*
    setting of that knob to the one entry regardless of what the arm actually
    ran.

    ``value`` is ``None`` when the candidate *removed* the flag. Removing a
    boolean flag realises ``false``, which is how ``cuda_graphs_enable`` — a
    lever that exists only as the absence of ``--enforce-eager`` — is reachable
    at all. Removing a valued flag restores a server default this module does
    not know, so there is no lever to name and it resolves to ``None``.
    """
    knob_name = knob.lstrip("-").replace("-", "_")
    matches = [s for s in library if s.knob == knob_name]
    if value is None:
        return next((s for s in matches if isinstance(s.value, bool) and not s.value), None)
    return next((s for s in matches if realises(value, s.value)), None)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CaptureError(f"{path.name}: unreadable: {exc}") from exc
    except ValueError as exc:
        raise CaptureError(f"{path.name}: not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise CaptureError(f"{path.name}: not a JSON object")
    return doc


def _throughput(summary: dict[str, Any]) -> tuple[float | None, float | None, bool]:
    """Requests per second over the window, the window, and whether it is goodput.

    ``goodput_rps`` where the capture reported one — it counts only requests that
    met their SLO, which is the number a serving change should be judged on. A
    run that met none has a real goodput of ``0.0`` and is not missing data.

    ``null`` is neither: the field is present and carries no number, which is
    what a capture writes when its window had no usable duration. Falling through
    to the raw rate there is how one arm ends up measured on SLO-qualified
    goodput and the other on a rate that counts requests which missed it — the
    third return value is what lets the caller refuse that.
    """
    client = summary.get("client")
    client = client if isinstance(client, dict) else {}
    window = client.get("window_s") or summary.get("wall_s")
    window = float(window) if isinstance(window, int | float) else None

    goodput = client.get("goodput_rps")
    if isinstance(goodput, int | float):
        return float(goodput), window, True

    n = client.get("n_requests")
    if isinstance(n, int | float) and window:
        return float(n) / window, window, False
    return None, window, False


def _find_trace(path: Path, manifest: dict[str, Any]) -> Path | None:
    """The trace belonging to this capture directory.

    The manifest records the path the trace had *on the machine that produced
    it*, and cluster captures are read after ``sync_results.sh`` has mirrored
    them somewhere else. That recorded path is then either missing — and the
    fingerprint fails, so a good measurement never reaches the loop — or, worse,
    still present and holding a different run's trace, which files the
    measurement under a workload it did not run.

    So the directory wins over the manifest: the trace sitting beside the
    artifacts is this capture's trace. The recorded path is consulted last, for
    a capture read in place on the machine that wrote it.
    """
    trace = manifest.get("trace")
    declared = trace.get("path") if isinstance(trace, dict) else None
    here = [path / "trace.jsonl"]
    if declared:
        here.insert(0, path / Path(declared).name)
    for candidate in here:
        if candidate.exists():
            return candidate
    return Path(declared) if declared else None


def _argv_from_attach(manifest: dict[str, Any]) -> list[str]:
    """The server's flags from an ``attach`` manifest, which has no ``serve_argv``.

    ``gitm capture serve`` launches the server and records the argv it used.
    ``gitm capture attach`` did not launch it, so it records what it found in
    ``/proc`` instead, under ``target.cmdline`` — the whole command, interpreter
    and all.

    Reading only ``serve_argv`` meant every attached capture came back with no
    flags at all. That is not a missing nicety: the baseline's flags are what
    every proposed arm is built from, so a sweep proposed against an attached
    baseline would launch servers carrying one flag and nothing else. It also
    silently inverted a lever — ``cuda_graphs_enable`` is realised by *removing*
    ``--enforce-eager``, so an empty baseline reports it unreachable on a server
    that was in fact started with it.

    Everything before the first flag is dropped: the interpreter, the console
    script, the subcommand and the positional model are how the server was
    invoked, not what it was configured with, and ``knob_difference`` compares
    flags.

    Where the flags start is :func:`gitm.serve.discover.vllm_argv_start`, which
    anchors on the vLLM entry point. Two cheaper anchors are both wrong. A
    ``serve`` token misses ``python -m vllm.entrypoints.openai.api_server …``,
    a form discovery already supports, which then comes back with no flags at
    all. The first ``--`` picks up a launcher's own options — under
    ``torchrun --nproc-per-node 2 -m vllm… --model m`` it starts at
    ``--nproc-per-node``, and a proposed arm would hand the server a flag it has
    never heard of.
    """
    cmdline = (manifest.get("target") or {}).get("cmdline")
    if not isinstance(cmdline, list):
        return []
    start = vllm_argv_start([str(a) for a in cmdline])
    return [] if start is None else [str(a) for a in cmdline[start:]]


def _tracing_from_attach(manifest: dict[str, Any]) -> str | None:
    """Whether an attached capture was traced, which its summary does not say.

    ``comparable_key`` includes this because tracing costs throughput, so an arm
    traced against one that was not measures the tracer rather than the knob.
    Left at ``None`` on every attached capture, two of them compared fine with
    each other but never against a launched one, and the reason would have read
    as a mismatch in the data rather than a gap in what was recorded.

    What it cannot say is whether NVTX was on. An attach target records
    ``traceable``, ``inject_lib`` and ``trace_out``, and nothing about markers —
    so ``"cupti"`` would be a claim that NVTX was *off*, which is not something
    the manifest establishes. A traced arm with markers and one without would
    then compare as though they matched, and the marker overhead would land on
    whatever knob was under test.

    So the label says what is known: traced, NVTX unestablished. It equals
    itself, so two attached arms still compare; it equals neither ``"cupti"``
    nor ``"cupti+nvtx"``, so an attached arm and a launched one are refused
    rather than quietly compared. That refusal is the honest outcome until the
    attach path records the marker setting it can already read off the server's
    environment — which is the real fix, and belongs where the capture is
    written rather than where it is read.
    """
    target = manifest.get("target")
    if not isinstance(target, dict) or "traceable" not in target:
        return None
    return TRACING_NVTX_UNKNOWN if target.get("traceable") else "off"


def read_capture(path: str | Path) -> Capture:
    """One arm's directory, read into a :class:`Capture`.

    Raises rather than returning a half-built record: a comparison assembled from
    a capture whose throughput is missing would report a delta against nothing.
    """
    path = Path(path)
    if not path.is_dir():
        raise CaptureError(f"{path}: not a directory")
    summary = _read_json(path / SUMMARY_NAME)
    manifest = _read_json(path / MANIFEST_NAME)

    throughput, window, is_goodput = _throughput(summary)
    if throughput is None:
        raise CaptureError(f"{path.name}: no throughput in {SUMMARY_NAME}")
    trace_path = _find_trace(path, manifest)

    argv = manifest.get("serve_argv")
    if isinstance(argv, list):
        # A launched capture recorded the command it ran, which is both.
        launch = argv
    else:
        argv = _argv_from_attach(manifest)
        cmdline = (manifest.get("target") or {}).get("cmdline")
        launch = (vllm_launch_argv([str(a) for a in cmdline])
                  if isinstance(cmdline, list) else None) or []
    load = manifest.get("load")
    return Capture(
        path=path,
        served_model=manifest.get("served_model"),
        serve_argv=tuple(str(a) for a in argv) if isinstance(argv, list) else (),
        load=load if isinstance(load, dict) else {},
        tracing=summary.get("tracing") or _tracing_from_attach(manifest),
        throughput=throughput,
        window_s=window,
        goodput=is_goodput,
        trace_path=trace_path,
        launch_argv=tuple(str(a) for a in launch),
    )


def boolean_flags(library: Iterable[Any]) -> frozenset[str]:
    """The server flags the catalogue sets to ``True`` or ``False``.

    Those flags never take a value, and that is the only reliable way to read a
    command line where the model follows one: ``vllm serve --enforce-eager
    CHECKPOINT``. Guessing which token is the model does not work. The served
    name can be an alias (``--served-model-name``), and treating the alias as
    the model then stops it being read as that flag's own value.
    """
    return frozenset("--" + s.knob.replace("_", "-") for s in library
                     if getattr(s, "knob", None) and isinstance(s.value, bool))


def parse_flags(
    argv: Sequence[str], *, booleans: frozenset[str] = frozenset()
) -> list[tuple[int, str, Any]]:
    """``[(index, flag, value)]`` for each ``--flag`` in ``argv``.

    The one parser both directions use: :func:`knob_difference` reading an arm
    back and :func:`gitm.optimizer.experiment_specs.plan_arms` writing one. Two
    copies of this rule is how an emitter and a reader come to disagree about
    what an arm says.

    A token after a flag is that flag's value unless it is itself a flag or the
    flag is one of ``booleans`` (see :func:`boolean_flags`). Without the second
    rule, ``vllm serve --enforce-eager CHECKPOINT`` read the checkpoint as
    ``--enforce-eager``'s value, and an arm removing that flag for
    ``cuda_graphs_enable`` removed the checkpoint with it.
    """
    out: list[tuple[int, str, Any]] = []
    i = 0
    while i < len(argv):
        token = str(argv[i])
        if not token.startswith("--"):
            i += 1
            continue
        nxt = str(argv[i + 1]) if i + 1 < len(argv) else None
        if token not in booleans and nxt is not None and not nxt.startswith("--"):
            out.append((i, token, nxt))
            i += 2
        else:
            out.append((i, token, True))
            i += 1
    return out


def knob_difference(
    baseline: Capture, candidate: Capture, *, booleans: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """The server flags the candidate changed, as ``{flag: value}``.

    Launch-only flags are excluded: where a server binds says nothing about what
    it computes, and two arms on different ports would otherwise fold ``--port``
    into the lever under test and produce a name no catalog entry can match.

    A flag the *baseline* carries and the candidate drops is reported too, under
    ``None``. Leaving it out was the earlier behaviour and it was worse than
    incomplete: the measured delta would have been credited entirely to whatever
    the candidate *added*, while a removal had moved it as well. The caller
    refuses those rather than attributing them.
    """
    def flags(c: Capture) -> dict[str, Any]:
        return {flag: value
                for _, flag, value in parse_flags(c.serve_argv, booleans=booleans)
                if flag not in LAUNCH_ONLY_FLAGS}

    base, cand = flags(baseline), flags(candidate)
    moved = {k: v for k, v in cand.items() if base.get(k) != v}
    moved.update({k: None for k in base if k not in cand})
    return moved


def compare(
    baseline: Capture, candidate: Capture, *, library: Iterable[Any],
    agreement_band: float = 0.02,
) -> VerificationRecord:
    """One baseline↔candidate comparison, in the loop's own record shape.

    ``library`` is required, not optional. Ranking looks a record up by
    ``spec.name``, and a name invented from the flag is a record nothing can
    find: ``--enforce-eager`` is the lever ``cuda_graphs_enable``, and
    ``--max-num-seqs 512`` is ``max_num_seqs_dynamic``. A flag with no catalog
    entry is refused rather than filed under a name that will never be looked up.

    Refuses arms that are not measuring the same thing — a different model, load
    shape or tracing arm, or one measured on goodput against one measured on raw
    request rate. Tracing especially: it costs throughput, so a traced candidate
    against an untraced baseline reports the tracer's overhead as the lever's
    effect.

    ``significant`` is the gain clearing ``agreement_band``, not a statistical
    test: a harness arm is one measurement, so there is no scatter to compute and
    a std of ``0.0`` would read as perfect precision rather than as one sample.
    ``reps=1`` says which it is.

    ``kept`` is the gate's own rule applied to this measurement: cleared the band
    and faster. Leaving it False because no rollback gate ran is the tidier
    -sounding choice and the wrong one — the reader maps ``not kept`` to *loss*,
    so every cluster result, including a +49% win, would demote the lever it
    proves. A harness arm runs standalone, so there is nothing to roll back and
    the number is the whole question. ``via="harness"`` records which path
    decided.
    """
    if baseline.comparable_key != candidate.comparable_key:
        raise CaptureError(
            "these arms are not an A/B: "
            f"baseline {baseline.comparable_key} vs candidate {candidate.comparable_key}")
    if baseline.goodput != candidate.goodput:
        raise CaptureError(
            "these arms were measured on different metrics: "
            f"{'goodput' if baseline.goodput else 'request rate'} vs "
            f"{'goodput' if candidate.goodput else 'request rate'}")
    if not baseline.throughput:
        raise CaptureError(f"{baseline.path.name}: baseline throughput is zero")

    library = list(library)
    knobs = knob_difference(baseline, candidate, booleans=boolean_flags(library))
    if not knobs:
        raise CaptureError(
            f"{baseline.path.name} and {candidate.path.name} ran the same server "
            "flags: there is no intervention between them")
    if len(knobs) > 1:
        dropped = sorted(k for k, v in knobs.items() if v is None)
        detail = f"dropping {', '.join(dropped)}, " if dropped else ""
        raise CaptureError(
            f"these arms differ in {len(knobs)} flags ({detail}"
            f"{', '.join(sorted(knobs))}): the measured delta cannot be "
            "credited to one lever")

    knob, value = next(iter(knobs.items()))
    spec = resolve_lever(knob, value, library)
    if spec is None:
        ran = f"removing {knob}" if value is None else f"setting {knob}={value}"
        raise CaptureError(
            f"{ran} matches no catalog entry. The catalog names a knob *and* the "
            "value it puts there, so a record filed under a lever the arm did "
            "not run is evidence for the wrong intervention.")

    speedup = candidate.throughput / baseline.throughput
    delta = speedup - 1.0
    return VerificationRecord(
        intervention_name=spec.name,
        summary=spec.summary,
        knob=spec.knob,
        value=spec.value,
        source=str(candidate.path),
        baseline_tps=baseline.throughput,
        candidate_tps=candidate.throughput,
        speedup=speedup,
        delta=delta,
        baseline_std=0.0,
        candidate_std=0.0,
        reps=1,
        agreement_band=agreement_band,
        significant=abs(delta) > agreement_band,
        kept=delta > 0 and abs(delta) > agreement_band,
        via="harness",
        # Requests, not tokens, and goodput only when the capture reported it —
        # the two arms are refused above unless they agree on which.
        unit="goodput_requests/sec" if baseline.goodput else "requests/sec",
        baseline_config={"serve_argv": list(baseline.serve_argv)},
        candidate_config={"serve_argv": list(candidate.serve_argv)},
    )


def sweep_id(baseline: Capture, candidates: Iterable[Capture]) -> str:
    """A run id for this sweep, derived from what it measured.

    Not from the directory paths. Paths identify a copy, not a sweep: an
    operator who re-downloads the same results into a second directory gets a
    different id, the existing-export refusal does not fire, and history counts
    one sweep twice — reading it as corroboration of itself, which is the single
    thing the record is least able to be.

    Derived instead from the model, the load shape, and each arm's server argv
    with the throughput it produced. Two ingests of the same measurements agree
    wherever the files live; a re-run of the same arms that measured something
    different is a different sweep, which it is.
    """
    def arm(c: Capture) -> str:
        return f"{'|'.join(c.serve_argv)}={c.throughput!r}"

    parts = [
        str(baseline.served_model),
        "|".join(f"{k}={v}" for k, v in sorted(baseline.load.items())),
        arm(baseline),
        # Sorted, so naming the same arms in another order is the same sweep.
        *sorted(arm(c) for c in candidates),
    ]
    return "harness-" + hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def write_comparison(
    baseline: Capture, candidate: Capture, *, out_dir: str | Path,
    library: Iterable[Any], gpu_sku: str | None = None,
    fingerprint: str | None = None, run_id: str | None = None,
) -> str:
    """Write one comparison as a ``verification.json`` under ``out_dir``.

    The single-arm case of :func:`write_comparisons`, which holds the
    documentation for both.
    """
    return write_comparisons(
        baseline, [candidate], out_dir=out_dir, library=library,
        gpu_sku=gpu_sku, fingerprint=fingerprint, run_id=run_id)


def write_comparisons(
    baseline: Capture, candidates: Iterable[Capture], *, out_dir: str | Path,
    library: Iterable[Any], gpu_sku: str | None = None,
    fingerprint: str | None = None, run_id: str | None = None,
    dry_run: bool = False,
) -> str:
    """Write every candidate's comparison against one baseline, as one export.

    A harness sweep is one baseline and several arms, so that is the shape this
    takes. They land in a single export rather than one directory per arm
    because they *are* one run: same baseline, same workload, same box, and
    :func:`gitm.optimizer.history.load_history` aggregates records across an
    export exactly as it does across directories. Splitting them would also mean
    inventing a run id per arm, and a run id is how a result is traced back to
    the thing that produced it.

    ``fingerprint`` is computed from a trace when not given, by the same rule
    :func:`gitm.optimizer.qualification.fingerprint` uses. It is never defaulted
    to the model name: the loop filters history on the trace digest, so a record
    filed under ``Kimi-K2.5`` is written and then filtered straight back out —
    invisible, not merely coarse.

    The trace comes from the **baseline**, and only from an accepted candidate if
    the baseline has none. The digest is over kernel shapes, and a lever changes
    shapes — so an arm's own digest differs from the baseline's, while the
    workload is the same one by construction (``compare`` refuses two arms that
    disagree on model, load or tracing). Keying a sweep on the baseline is what
    makes every arm of it findable under one workload, and what makes it the same
    key ``gitm propose`` ranks history under. Keying on a candidate instead left
    the two commands filing and looking under different digests, so a measured
    result could not reach the next proposal.

    Taking ``candidates[0]`` as given was wrong for two further reasons, and both
    still hold: a refused arm is not part of this sweep, and an untraced first
    arm raised and discarded a sweep whose other arms were fine.

    Every arm is compared, and one unusable arm does not discard the others:
    :func:`compare` refuses an arm that is not an A/B of the baseline, and that
    refusal belongs to that arm. The caller gets the reasons back through
    :class:`CaptureError` only when *nothing* could be compared, because an
    export with no records is a file that says a sweep produced no evidence.

    Refuses to overwrite an existing export. A reused run id would otherwise
    replace a directory's records, and the loop's own exports live under the
    same tree.

    ``dry_run`` performs every check and writes nothing, returning the path it
    would have written. It exists so ``--dry-run`` can refuse what the real
    command would refuse, rather than predicting a write that cannot happen.
    """
    out_dir = Path(out_dir)
    export = out_dir / EXPORT_NAME
    if export.exists():
        raise CaptureError(
            f"{export} already exists. Writing here would replace whatever it "
            "records; pick a run id that is not in use.")

    candidates = list(candidates)
    if not candidates:
        raise CaptureError("no candidate arms to compare against the baseline")

    records: list[VerificationRecord] = []
    accepted: list[Capture] = []
    refused: list[str] = []
    for cand in candidates:
        try:
            records.append(compare(baseline, cand, library=library))
            accepted.append(cand)
        except CaptureError as exc:
            refused.append(f"{cand.path.name}: {exc}")
    if not records:
        raise CaptureError(
            "no arm could be compared against this baseline:\n  "
            + "\n  ".join(refused))

    unfingerprintable: list[str] = []
    if fingerprint is None:
        for source in (baseline, *accepted):
            try:
                fingerprint = fingerprint_of(source)
                break
            except CaptureError as exc:
                # Its own list, not ``refused``. These arms were compared and
                # their measurements are in the export; saying they were refused
                # would tell the operator a result was dropped when it was kept.
                unfingerprintable.append(f"{source.path.name}: {exc}")
        if fingerprint is None:
            raise CaptureError(
                "neither the baseline nor any accepted arm could be "
                "fingerprinted, so these results cannot be filed against a "
                "workload the loop would recognise. Pass --fingerprint, or "
                "capture with tracing on:\n  " + "\n  ".join(unfingerprintable))

    if dry_run:
        return str(export)

    out_dir.mkdir(parents=True, exist_ok=True)
    # The sidecar first. The export is what the "already exists" check guards, so
    # publishing it before the account of what was refused means a failure in
    # between leaves records on disk, no reasons beside them, and a retry
    # refused. Written in the order a reader needs them to be complete.
    if refused or unfingerprintable:
        # Beside the export, not inside it: the export is the evidence the
        # ranking reads, and an arm that could not be compared is not evidence
        # about a lever. It still has to be visible, or a sweep of nine that
        # ingested four looks like a sweep of four.
        #
        # The two lists are separate because they mean opposite things to a
        # reader checking a partial sweep. ``refused`` is a measurement that did
        # not make it in. ``not_fingerprintable`` is one that did, from an arm
        # that merely could not supply the workload digest.
        (out_dir / "ingest_refused.json").write_text(
            json.dumps({"refused": refused,
                        "not_fingerprintable": unfingerprintable}, indent=2) + "\n")
    prov = Provenance(
        workload_id="vllm-serve",
        fingerprint=fingerprint,
        run_id=run_id or out_dir.name,
        git_sha="", gitm_version="", started_at_ns=0, ended_at_ns=0,
    )
    return write_verification(records, prov, export, gpu_sku=gpu_sku)
