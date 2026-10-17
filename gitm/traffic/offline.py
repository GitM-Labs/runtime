"""Offline batch inputs from a trace — real lengths and real prefix sharing.

The optimization loop measures each candidate by handing a fixed batch of
requests to an offline engine (``LLM.generate``) and timing it. Until this, that
batch was 256 copies of a five-token prompt asking for 512 tokens: pure decode,
no prefill to speak of, nothing shared between requests. On that traffic stock
vLLM sits at its ceiling and every lever that acts on prefill, prefix caching,
chunked scheduling or KV pressure measures as noise. Kimi K2.6 on H200 rejected
all 17 candidates against it.

This turns a canonical :class:`~gitm.traffic.schema.Trace` into that batch
instead: each request's real input and output length, and prompts built from
its prefix-block chain so that requests which shared a prefix in production
share identical leading tokens here, and the engine's prefix cache sees the same
reuse. It is a *batch*, not a replay: arrival timing is not reproduced (that is
:mod:`gitm.traffic.replay` against a server). What it keeps is the shape of the
work: the prefill/decode mix and the sharing.

CPU-only and vLLM-free: it returns token ids and lengths, and the workload wraps
them in vLLM's types.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from gitm.traffic.adapters import read_burstgpt, read_mooncake
from gitm.traffic.schema import Trace

#: Token ids are drawn from ``[_ID_FLOOR, vocab_size)``, clear of the low ids
#: tokenizers reserve for special and control tokens.
_ID_FLOOR = 1000

READERS = {"mooncake": read_mooncake, "burstgpt": read_burstgpt}


@dataclass
class OfflineBatch:
    """What an offline engine is handed: one entry per request, in trace order."""

    prompt_token_ids: list[list[int]]
    output_tokens: list[int]
    #: Requests read but not used, by reason. Never silently trimmed.
    skipped: dict[str, int] = field(default_factory=dict)
    source: str = ""
    sha256: str = ""

    def summary(self) -> dict:
        n = len(self.output_tokens)
        inputs = [len(p) for p in self.prompt_token_ids]
        return {
            "source": self.source,
            "sha256": self.sha256,
            "requests": n,
            "skipped": dict(self.skipped),
            "input_tokens_mean": round(sum(inputs) / n, 1) if n else 0,
            "output_tokens_mean": round(sum(self.output_tokens) / n, 1) if n else 0,
            "input_tokens_total": sum(inputs),
            "output_tokens_total": sum(self.output_tokens),
        }


def parse_spec(spec: str) -> tuple[str, Path]:
    """``"mooncake:/path/trace.jsonl"`` -> ``("mooncake", Path(...))``."""
    kind, sep, path = spec.partition(":")
    if not sep or kind not in READERS or not path:
        raise ValueError(
            f"traffic spec {spec!r}: expected <source>:<path>, source one of "
            f"{sorted(READERS)}")
    return kind, Path(path)


def _block_tokens(key: str, n: int, vocab_size: int) -> list[int]:
    """The same ``n`` token ids for the same key, every time and on every
    machine: hashed, not drawn from a seeded RNG whose sequence could shift.

    Keys are namespaced (``block:<id>`` for a trace's prefix blocks,
    ``private:<n>`` for a request's own tail), so no trace block id, negative
    ones included, can collide with a tail and share tokens it never shared."""
    span = vocab_size - _ID_FLOOR
    out: list[int] = []
    counter = 0
    while len(out) < n:
        digest = hashlib.blake2b(f"{key}:{counter}".encode(), digest_size=64).digest()
        for i in range(0, len(digest), 4):
            out.append(_ID_FLOOR + int.from_bytes(digest[i:i + 4], "little") % span)
            if len(out) == n:
                break
        counter += 1
    return out


def batch_from_trace(trace: Trace, *, n: int, vocab_size: int,
                     max_model_len: int | None = None) -> OfflineBatch:
    """The first ``n`` usable requests of ``trace`` as an offline batch.

    In arrival order rather than sampled, so a prefix shared by neighbouring
    requests in production stays shared within the batch. A request is skipped,
    and counted, when it has no output length (nothing to generate to) or when
    input plus output would not fit ``max_model_len``. A long document prompt is
    exactly the request worth keeping, so it is never cut down to fit.
    """
    if vocab_size <= _ID_FLOOR:
        raise ValueError(f"vocab_size {vocab_size} leaves no room for prompt token ids")
    block = trace.meta.prefix_block_tokens
    batch = OfflineBatch([], [], source=trace.meta.source, sha256=trace.meta.sha256)
    unique = 0
    for req in trace.requests:
        if len(batch.output_tokens) >= n:
            break
        if not req.output_tokens:
            batch.skipped["no_output_length"] = batch.skipped.get("no_output_length", 0) + 1
            continue
        if max_model_len is not None and req.input_tokens + req.output_tokens > max_model_len:
            batch.skipped["over_max_model_len"] = batch.skipped.get("over_max_model_len", 0) + 1
            continue
        ids: list[int] = []
        if req.prefix_blocks and block:
            for b in req.prefix_blocks:
                ids.extend(_block_tokens(f"block:{b}", block, vocab_size))
                if len(ids) >= req.input_tokens:
                    break
        if len(ids) < req.input_tokens:
            # No prefix identity (BurstGPT), or a chain shorter than the input:
            # the rest is this request's own, shared with nobody.
            unique += 1
            ids.extend(_block_tokens(f"private:{unique}", req.input_tokens - len(ids), vocab_size))
        batch.prompt_token_ids.append(ids[:req.input_tokens])
        batch.output_tokens.append(req.output_tokens)
    return batch


def load_batch(spec: str, *, n: int, vocab_size: int,
               max_model_len: int | None = None) -> OfflineBatch:
    """Read the trace a ``GITM_VLLM_TRAFFIC`` spec names and build the batch."""
    kind, path = parse_spec(spec)
    return batch_from_trace(READERS[kind](path), n=n, vocab_size=vocab_size,
                            max_model_len=max_model_len)
