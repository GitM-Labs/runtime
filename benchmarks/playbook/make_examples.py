"""Generate benchmarks/playbook/examples.json from the REAL D1 regimes.

The regimes are measured off the committed traffic fixtures; the deltas are not
measured and every row says so (evidence=illustrative).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from gitm.bench.manifest import sha256_file
from gitm.playbook.schema import (
    EnvCapture,
    Evidence,
    Invalidation,
    MeasuredDelta,
    Playbook,
    PlaybookRow,
    Provenance,
    RowIdentity,
)
from gitm.traffic import Regime, SourceKind, read_burstgpt, read_mooncake

F = Path("benchmarks/traffic_replay/fixtures")
bg_path, mc_path = F / "burstgpt_slice.csv", F / "mooncake_slice.jsonl"
bg = read_burstgpt(bg_path)
mc = read_mooncake(mc_path)
bg_r = Regime.from_trace(bg)
mc_r = Regime.from_trace(mc)

H100 = "NVIDIA H100 80GB"
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
REV = "95a723d0"
ENV = EnvCapture(engine="vllm", engine_version="0.11.0")
ENV_NEXT = EnvCapture(engine="vllm", engine_version="0.12.0")
T = datetime(2026, 9, 2, tzinfo=timezone.utc)


def prov(trace, path, regime, **kw):
    """Provenance as D1 would emit it: real checksum, real drops, real replay
    conditions. BurstGPT has no prefix identity, so a replay of it always
    synthesizes blocks -- which is what makes ex6 a floor and not a measurement."""
    kw.setdefault("replay_chunk_hash_size", 512)
    kw.setdefault("replay_self_timed", True)
    kw.setdefault("prefix_synthesized", not trace.meta.has_prefix_identity)
    return Provenance(
        trace_source=trace.meta.source,
        trace_sha256=sha256_file(path)[0],
        trace_drops=dict(trace.meta.drops),
        regime_label=regime.label(),
        repeat_raw_data=[],
        verified_at=T,
        **kw,
    )


# A scoreboard regime: Artificial Analysis' fixed-length condition, as its own
# named source_kind. Numbers are the published fixed shape, not a measurement of
# ours -- which is exactly why it must never match a production query.
board_r = mc_r.model_copy(update={
    "source_kind": SourceKind.SCOREBOARD, "trace": "artificialanalysis-fixed",
    "requests": 1000, "rate_rps": 1.0, "io_ratio": 4.0,
    "input_p50": 1024, "input_p95": 1024, "output_p50": 256, "output_p95": 256,
    "burstiness": 0.0, "notes": ["fixed-length scoreboard condition; not production traffic"],
})

rows = [
    PlaybookRow(
        row_id="ex1-prefix-cache-mooncake",
        identity=RowIdentity(model=MODEL, model_revision=REV, gpu_sku=H100, env=ENV,
                             regime=mc_r, knobs={"enable_prefix_caching": True}),
        delta=MeasuredDelta(throughput_pct=14.5, ttft_p99_ms=-18.0, itl_p99_ms=0.3,
                            repeats=5, throughput_ci95_pct=(9.1, 19.4)),
        provenance=prov(mc, mc_path, mc_r),
        evidence=Evidence.ILLUSTRATIVE,
        notes=["Regime is real (measured off the pinned Mooncake fixture). The delta is not: "
               "no run against a live endpoint has happened. Shape to build against, not a claim."],
    ),
    PlaybookRow(
        row_id="ex2-max-num-seqs-burstgpt",
        identity=RowIdentity(model=MODEL, model_revision=REV, gpu_sku=H100, env=ENV,
                             regime=bg_r, knobs={"max_num_seqs": 64}),
        delta=MeasuredDelta(throughput_pct=8.1, ttft_p99_ms=-4.2, itl_p99_ms=1.0,
                            repeats=5, throughput_ci95_pct=(3.3, 12.6)),
        provenance=prov(bg, bg_path, bg_r),
        evidence=Evidence.ILLUSTRATIVE,
        notes=["The counter-example to ex1: same model, same GPU, no prefix identity in the "
               "source at all. A prefix-caching row must never be selected for this regime."],
    ),
    PlaybookRow(
        row_id="ex3-chunked-prefill-qwen",
        identity=RowIdentity(model=MODEL, model_revision=REV, gpu_sku=H100, env=ENV,
                             regime=mc_r, knobs={"enable_chunked_prefill": True,
                                                 "max_num_batched_tokens": 8192}),
        delta=MeasuredDelta(throughput_pct=6.4, ttft_p99_ms=-31.0, itl_p99_ms=2.1, repeats=5),
        provenance=prov(mc, mc_path, mc_r, promotion_rule="pending-adit/promotion-rule"),
        evidence=Evidence.ILLUSTRATIVE,
        notes=["This is the row deliverable 3 would produce. It stays illustrative until "
               "Phase B runs -- that needs one 80 GB card, which we do not have."],
    ),
    PlaybookRow(
        row_id="ex4-retired-engine-bump",
        identity=RowIdentity(model=MODEL, model_revision=REV, gpu_sku=H100, env=ENV_NEXT,
                             regime=mc_r, knobs={"enable_prefix_caching": True}),
        delta=MeasuredDelta(throughput_pct=11.0, ttft_p99_ms=-12.0, itl_p99_ms=0.4, repeats=4),
        provenance=prov(mc, mc_path, mc_r),
        evidence=Evidence.ILLUSTRATIVE,
        invalidated=Invalidation(reason="vLLM 0.11 -> 0.12 scheduler rewrite; the measured "
                                        "delta is against a scheduler that no longer exists",
                                 at=datetime(2026, 9, 2, tzinfo=timezone.utc), by="validation"),
        notes=["Kept, not deleted. A deleted row leaves no record that the claim was made, "
               "which is the first thing a reviewer asks for."],
    ),
    PlaybookRow(
        row_id="ex6-prefix-cache-on-a-synthesized-trace",
        identity=RowIdentity(model=MODEL, model_revision=REV, gpu_sku=H100, env=ENV,
                             regime=bg_r, knobs={"enable_prefix_caching": True}),
        delta=MeasuredDelta(throughput_pct=1.2, ttft_p99_ms=-0.4, itl_p99_ms=0.0, repeats=5),
        provenance=prov(bg, bg_path, bg_r),
        evidence=Evidence.ILLUSTRATIVE,
        notes=["delta_is_floor: BurstGPT has no prefix identity, so D1 synthesized unique "
               "blocks per request and the replay saw the LEAST reuse the real traffic could "
               "have had. +1.2% is a lower bound, never quotable as the gain."],
    ),
    PlaybookRow(
        row_id="ex5-scoreboard-not-production",
        identity=RowIdentity(model=MODEL, model_revision=REV, gpu_sku=H100, env=ENV,
                             regime=board_r, knobs={"max_num_seqs": 64}),
        delta=MeasuredDelta(throughput_pct=22.0, ttft_p99_ms=-9.0, itl_p99_ms=0.1, repeats=3),
        provenance=prov(mc, mc_path, board_r, config_capture="pending-adit"),
        evidence=Evidence.ILLUSTRATIVE,
        notes=["source_kind=scoreboard. The biggest claimed delta in the file, and it is "
               "gated out of every production lookup by equality, not by distance."],
    ),
]

out = Path("benchmarks/playbook/examples.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(Playbook(rows=rows).model_dump(mode="json"), indent=2) + "\n",
               encoding="utf-8")
print(f"wrote {out} — {len(rows)} rows, {sum(r.selectable for r in rows)} selectable")
