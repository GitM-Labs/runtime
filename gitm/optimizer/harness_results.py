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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gitm.optimizer.history import EXPORT_NAME
from gitm.optimizer.report import Provenance
from gitm.optimizer.verification_export import VerificationRecord, write_verification

__all__ = [
    "Capture",
    "fingerprint_of",
    "resolve_lever",
    "LAUNCH_ONLY_FLAGS",
    "CaptureError",
    "read_capture",
    "knob_difference",
    "compare",
    "write_comparison",
]

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


def _realises(value: Any, spec_value: Any) -> bool:
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
    return next((s for s in matches if _realises(value, s.value)), None)


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
    load = manifest.get("load")
    return Capture(
        path=path,
        served_model=manifest.get("served_model"),
        serve_argv=tuple(str(a) for a in argv) if isinstance(argv, list) else (),
        load=load if isinstance(load, dict) else {},
        tracing=summary.get("tracing"),
        throughput=throughput,
        window_s=window,
        goodput=is_goodput,
        trace_path=trace_path,
    )


def knob_difference(baseline: Capture, candidate: Capture) -> dict[str, Any]:
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
    def flags(argv: tuple[str, ...]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        i = 0
        while i < len(argv):
            token = argv[i]
            if not token.startswith("--"):
                i += 1
                continue
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is not None and not nxt.startswith("--"):
                value, i = nxt, i + 2
            else:
                value, i = True, i + 1
            if token not in LAUNCH_ONLY_FLAGS:
                out[token] = value
        return out

    base, cand = flags(baseline.serve_argv), flags(candidate.serve_argv)
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

    knobs = knob_difference(baseline, candidate)
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
        baseline_config={"serve_argv": list(baseline.serve_argv)},
        candidate_config={"serve_argv": list(candidate.serve_argv)},
    )


def write_comparison(
    baseline: Capture, candidate: Capture, *, out_dir: str | Path,
    library: Iterable[Any], gpu_sku: str | None = None,
    fingerprint: str | None = None, run_id: str | None = None,
) -> str:
    """Write the comparison as a ``verification.json`` under ``out_dir``.

    ``fingerprint`` is computed from the candidate's own trace when not given,
    by the same rule :func:`gitm.optimizer.qualification.fingerprint` uses. It is
    never defaulted to the model name: the loop filters history on the trace
    digest, so a record filed under ``Kimi-K2.5`` is written and then filtered
    straight back out — invisible, not merely coarse.

    Refuses to overwrite an existing export. A reused run id would otherwise
    replace a directory's records with this single comparison, and the loop's own
    exports live under the same tree.
    """
    out_dir = Path(out_dir)
    export = out_dir / EXPORT_NAME
    if export.exists():
        raise CaptureError(
            f"{export} already exists. Writing here would replace whatever it "
            "records; pick a run id that is not in use.")

    record = compare(baseline, candidate, library=library)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = Provenance(
        workload_id="vllm-serve",
        fingerprint=fingerprint or fingerprint_of(candidate),
        run_id=run_id or out_dir.name,
        git_sha="", gitm_version="", started_at_ns=0, ended_at_ns=0,
    )
    return write_verification([record], prov, export, gpu_sku=gpu_sku)
