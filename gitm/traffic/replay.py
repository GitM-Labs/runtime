"""Replay mode — fire a canonical trace as-is, through a tool that already exists.

vLLM's bench-serve has a native ``timed_trace`` dataset that consumes
``{"timestamp", "input_length", "output_length", "hash_ids"}`` JSONL and, under
``--self-timed``, schedules each request at **its own timestamp** rather than at
a synthesized rate. That is faithful replay, already written, already maintained.
So this module writes that file and builds that command line. There is no load
generator here and there should never be one.

Two things this module exists to get right, both silent failures otherwise:

**Block coverage.** ``timed_trace`` builds each prompt by expanding ``hash_ids``
at ``--timed-trace-chunk-hash-size`` tokens per id and stops when the ids run
out. Pass 16 (the vLLM default) for a Mooncake trace whose blocks are 512 tokens
and every prompt comes out 32x short while every count in the output still looks
correct. :func:`write_timed_trace` checks ``len(blocks) * block_tokens >=
input_tokens`` for every request and refuses the whole file if it does not hold.

**Sources with no prefix identity.** BurstGPT has no ``hash_ids``. Emitting an
empty list produces a zero-length prompt, not a 472-token one. So blocks are
*synthesized*: each request gets its own fresh run of ids, globally unique, so
lengths are honoured and **no prefix sharing is invented that the source never
had**. The plan records that this happened; a prefix-cache experiment reading a
plan with ``prefix_synthesized=True`` must reject it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from gitm.traffic.schema import Trace, TraceMeta

#: Block size used when synthesizing prefix ids for a source that has none.
#: Arbitrary — the ids are unique per request either way — but it sets how many
#: ids each row carries, so keep it large enough that the file stays small.
SYNTHETIC_BLOCK_TOKENS = 512

#: First vLLM release containing the ``timed_trace`` dataset. The feature landed
#: 2026-05-28 (``bfb9ebc21``), which missed v0.22.0 by one day: v0.22.1 does
#: **not** have it, v0.23.0 does — checked at both tags, not inferred from dates.
#:
#: This matters because the repo's own ``[vllm]`` extra says ``vllm>=0.6`` and
#: ChunkPrefill's Phase B says ``>=0.19.0``. Either would install a vLLM with no
#: ``timed_trace`` at all, and the failure is an argparse complaint about an
#: unknown dataset name — which reads like a typo in our command, not a missing
#: feature. The number lives here rather than in prose so the argv builder can
#: say it out loud.
VLLM_MIN_VERSION = "0.23.0"


class ReplayPlan(BaseModel):
    """What was written, and the command that fires it.

    Carries the source :class:`~gitm.traffic.schema.TraceMeta` verbatim: the
    promotion rule (deliverable 2) requires trace identity on every promoted row,
    and a plan that has forgotten which bytes it came from cannot supply it.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    requests: int
    span_s: float
    #: Token totals of what was written. Here so a *result* can be reconciled
    #: against the trace after the trace object is gone: ``bench serve`` reports
    #: ``total_input_tokens``, and comparing it to this is what catches a run
    #: fired at the wrong ``--timed-trace-chunk-hash-size``. At vLLM's default of
    #: 16 against Mooncake's 512-token blocks the run reports ~1/32 of this while
    #: every other count still looks right. See :mod:`gitm.traffic.results`.
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    chunk_hash_size: int
    sec_multiplier: float = 1.0  # we always emit seconds; vLLM's default is 1
    self_timed: bool = True
    prefix_synthesized: bool = False
    source: TraceMeta
    notes: list[str] = Field(default_factory=list)

    def bench_serve_argv(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:8000",
        backend: str = "openai",
        tokenizer: str | None = None,
        max_concurrency: int | None = None,
        result_filename: str | None = None,
        seed: int = 0,
    ) -> list[str]:
        """The exact ``vllm bench serve`` invocation for this plan.

        **Requires vLLM >= :data:`VLLM_MIN_VERSION`.** Below that there is no
        ``timed_trace`` dataset and the run dies on an argparse error that looks
        like a typo in this command rather than a missing feature.

        ``backend`` must be a completions backend (``openai`` or ``vllm``):
        ``timed_trace`` passes pre-tokenized prompts, which the chat endpoints
        will not take.

        ``tokenizer`` is worth passing whenever the **served** model id is not an
        id HuggingFace can resolve. ``bench serve`` builds a tokenizer from
        ``--model`` when none is given, so a server started with
        ``--served-model-name`` — or any stub — fails inside
        ``AutoTokenizer.from_pretrained`` with *"X is not a local folder and is
        not a valid model identifier"*, long after the endpoint answered fine.
        Pre-tokenized prompts do not save you from this: the tokenizer is still
        constructed for output accounting.

        ``result_filename`` is worth passing. What comes back does **not**
        identify the workload: ``bench serve``'s result JSON carries
        ``backend``, ``model_id``, ``num_prompts``, ``max_concurrency`` and the
        metrics, but no trace identity and no regime. Worse, under
        ``--self-timed`` it still records the CLI's ``request_rate`` and
        ``burstiness`` defaults, which are meaningless — the real values came
        from the trace and are on this plan's ``Regime``. Joining the two is the
        open work; see the spec's "what the result JSON does not carry".
        """
        argv = [
            "vllm", "bench", "serve",
            "--backend", backend,
            "--model", model,
            "--base-url", base_url,
            "--dataset-name", "timed_trace",
            "--dataset-path", self.path,
            "--num-prompts", str(self.requests),
            "--timed-trace-sec-multiplier", f"{self.sec_multiplier:g}",
            "--timed-trace-chunk-hash-size", str(self.chunk_hash_size),
            "--seed", str(seed),
        ]
        if tokenizer is not None:
            argv += ["--tokenizer", tokenizer]
        argv.append("--self-timed" if self.self_timed else "--no-self-timed")
        if max_concurrency is not None:
            argv += ["--max-concurrency", str(max_concurrency)]
        if result_filename is not None:
            argv += ["--save-result", "--result-filename", result_filename]
        return argv


def write_timed_trace(
    trace: Trace,
    out: str | Path,
    *,
    block_tokens: int | None = None,
) -> ReplayPlan:
    """Write ``trace`` as a vLLM ``timed_trace`` JSONL and return the plan.

    ``block_tokens`` defaults to the trace's own
    :attr:`TraceMeta.prefix_block_tokens` when it has prefix identity, and to
    :data:`SYNTHETIC_BLOCK_TOKENS` when ids are being synthesized.

    Raises if any request lacks an output length (replay-as-is is not defined for
    it — use the parameterized mode) or if block coverage would truncate a prompt.
    """
    out = Path(out)
    missing_out = sum(1 for r in trace.requests if r.output_tokens is None)
    if missing_out:
        raise ValueError(
            f"{missing_out} of {len(trace.requests)} requests have no output length; "
            "replay-as-is is undefined for them — use gitm.traffic.parameterize"
        )

    synthesize = not trace.meta.has_prefix_identity
    if block_tokens is None:
        block_tokens = (
            SYNTHETIC_BLOCK_TOKENS if synthesize else (trace.meta.prefix_block_tokens or 0)
        )
    if block_tokens <= 0:
        raise ValueError(
            "block_tokens must be positive — a trace with prefix identity must "
            "record how many tokens one block id stands for"
        )

    lines: list[str] = []
    next_block_id = 0
    for i, req in enumerate(trace.requests):
        if synthesize or not req.prefix_blocks:
            n = math.ceil(req.input_tokens / block_tokens)
            blocks = list(range(next_block_id, next_block_id + n))
            next_block_id += n
        else:
            blocks = list(req.prefix_blocks)
            if len(blocks) * block_tokens < req.input_tokens:
                raise ValueError(
                    f"request {i}: {len(blocks)} blocks x {block_tokens} tokens cannot "
                    f"cover input_tokens={req.input_tokens}. The prompt would be "
                    f"silently truncated — check block_tokens against the source "
                    f"(Mooncake is 512, not vLLM's default 16)."
                )
        lines.append(
            json.dumps(
                {
                    "timestamp": round(req.arrival_s, 6),
                    "input_length": req.input_tokens,
                    "output_length": req.output_tokens,
                    "hash_ids": blocks,
                }
            )
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    notes = ["timestamps emitted in seconds, so sec_multiplier is 1"]
    if synthesize:
        notes.append(
            "prefix blocks SYNTHESIZED (source has no prefix identity): ids are unique "
            "per request, so no prefix sharing is implied. Not valid for prefix-cache "
            "experiments."
        )
        if trace.meta.has_session_identity:
            # BurstGPT_3 is exactly this case: it knows which requests are turns of
            # one conversation, but not what they share. Later turns of a session do
            # re-send the conversation so far, so unique-per-request blocks
            # UNDERSTATE the real reuse. Understating is the safe direction — it
            # never invents a cache hit — but a measured prefix-cache benefit on
            # this trace is a floor, not an estimate, and that has to be said.
            notes.append(
                f"source has session identity ({trace.meta.session_rows} rows in "
                f"{trace.meta.sessions} sessions) but no prefix identity: real "
                "conversations share a prefix that these unique ids do not reproduce. "
                "Prefix-cache reuse is UNDERSTATED, never overstated."
            )
    if trace.meta.has_session_identity:
        # Said on every session-carrying plan, not just the synthesized ones: the
        # timed_trace format has no session field at all, so conversation identity
        # stops at this boundary. Deriving prefix blocks from sessions would make
        # it flow, but only by asserting how much each turn re-sends — an
        # invented cache hit, which is the one thing this module will not do.
        notes.append(
            "session identity is NOT carried into the replay file: vLLM's timed_trace "
            "format has no session field. Sessions are available for analysis and "
            "regime characterization, not for session-aware firing."
        )
    return ReplayPlan(
        path=str(out),
        requests=len(trace.requests),
        span_s=trace.meta.span_s,
        input_tokens_total=sum(r.input_tokens for r in trace.requests),
        output_tokens_total=sum(r.output_tokens or 0 for r in trace.requests),
        chunk_hash_size=block_tokens,
        prefix_synthesized=synthesize,
        source=trace.meta,
        notes=notes,
    )


def read_timed_trace(path: str | Path, *, source: str = "timed_trace") -> Trace:
    """Read a ``timed_trace`` JSONL back into canonical form.

    The inverse of :func:`write_timed_trace`, and the reason the replay claim is
    evidence: :mod:`gitm.traffic.validate` compares this against the trace the
    adapter produced, so "the pipeline preserves the trace" is a measured
    statement about the file the benchmark will actually consume.
    """
    from gitm.traffic.adapters import read_mooncake

    return read_mooncake(path, time_scale=1.0, source=source)
