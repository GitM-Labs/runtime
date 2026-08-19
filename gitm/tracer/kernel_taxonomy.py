"""What a captured trace is actually made of — coarse kernel buckets and coverage.

The tracer does not filter: ``CONCURRENT_KERNEL`` records every kernel the device
ran, so a vLLM trace already contains the cuBLAS/cutlass GEMMs, the MoE expert
kernels, FlashAttention, NCCL's all-reduces and vLLM's own CUDA kernels. The open
question is never "were they captured" but "can we *see* them in what came back",
and that question has three failure modes this module is built to answer:

* **unnamed work** — a bucket breakdown where ``other`` dominates means the trace is
  full of kernels no rule recognises, and any conclusion drawn per-op is guesswork.
* **truncated names** — the collector's name field is bounded (``NAME_MAX``).
  Mangled cutlass/MoE template instantiations run long, and two different kernels
  truncated to the same 255 bytes become one identity. That is silent: the trace
  looks complete and the distinct expert GEMMs have merged.
* **missing device time** — kernels that ran but were never attributed (CUDA-graph
  replay is the usual cause) leave the GPU looking idle while throughput says
  otherwise. ``gpu_active_share`` is the number that exposes it.

This is deliberately NOT :func:`gitm.optimizer.deviation.classify_op`. That maps a
kernel onto a node of the *predicted graph* and returns ``None`` for anything the
graph doesn't model — the right semantics for alignment, the wrong ones for "what
ran". Here every kernel lands in exactly one bucket, and ``other`` is a finding
rather than a shrug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Must track GITM_NAME_MAX in gitm/tracer/_cupti/cupti_core.h. A name of exactly
# this length was almost certainly cut off — the collector copies at most NAME_MAX
# bytes, so a full-length name is the fingerprint of truncation, not a coincidence.
NAME_MAX = 255

# Ordered: first match wins, case-insensitive substring against the (mangled) name.
# Order is load-bearing where vocabularies overlap:
#   * MoE first — `marlin_moe`/`moe_align` also contain "gemm"/"align", and an MoE
#     GEMM is more usefully an MoE kernel than a GEMM.
#   * collectives before GEMM — NCCL's kernels mention neither, but custom all-reduce
#     paths do carry "reduce", which the elementwise rules would otherwise claim.
#   * attention before GEMM — FlashAttention's inner loop is a GEMM by any honest
#     reading, but attributing it to "gemm" hides the attention cost entirely.
#   * quantised-GEMM names (marlin, machete, scaled_mm) are GEMMs, so they precede
#     the generic quant rules, which are then left with the standalone
#     quantise/dequantise passes.
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("moe", ("moe", "expert", "topk_softmax", "grouped_gemm", "group_gemm",
             "groupedgemm", "gather_scatter", "sort_tokens", "routing", "router")),
    # "cross_device_reduce" is vLLM's own custom all-reduce (the fast path TP=2 takes
    # on NVLink). Without it the generic "reduce" needle files it as elementwise and
    # the collective cost disappears into the noise — which is the one cost a TP run
    # exists to measure.
    ("collective", ("nccl", "all_reduce", "allreduce", "reduce_scatter", "reducescatter",
                    "all_gather", "allgather", "custom_ar", "cross_device", "one_shot",
                    "two_shot")),
    # Linear / recurrent attention, kept separate from softmax attention. Hybrid
    # models (Qwen3-Next-style Gated DeltaNet, Mamba, RWKV) run mostly these and only
    # a few full-attention layers, so folding them together hides the split that
    # matters — and none of the softmax needles below match them, which would leave
    # the dominant layer type sitting in "other". Names come from the flash-linear-
    # attention kernels these models ship with.
    # "local_cumsum", not bare "cumsum": top-p sampling runs a cumulative sum over
    # sorted probabilities, and the loose needle would file it as linear attention.
    # "chunk_o"/"chunk_h" are kept for older flash-linear-attention builds, but
    # they do not match the current kernel names: those put the suffix last
    # (`chunk_fwd_kernel_o`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`),
    # so "chunk_fwd" is the needle that actually fires. Confirmed against a
    # Qwen3.6-35B-A3B capture, where six GDN kernels — three quarters of the
    # model's layers — landed in `other`.
    #
    # "deltarule" carries no underscore on purpose: the CuTeDSL fast path is
    # `_FullyFusedDeltaRuleSm90`, and CamelCase lowercases to one token, so every
    # underscored delta needle misses the SM90 path that Hopper actually runs.
    #
    # The short causal convolution belongs here too: it is part of the GDN layer,
    # not a separate mechanism, and it has no other home in this vocabulary.
    ("linear_attn", ("delta_rule", "gated_delta", "deltanet", "deltarule",
                     "fused_recurrent", "linear_attn", "solve_tril", "wy_fast",
                     "local_cumsum", "chunk_o", "chunk_h", "chunk_fwd",
                     "chunk_scaled_dot", "recompute_w_u", "causal_conv1d",
                     "post_conv", "selective_scan", "mamba")),
    # Sparse/compressed attention needles ride in this bucket rather than a
    # separate one: they are attention by cost and by what a lever would target.
    # The indexer needles are load-bearing — a "lightning_indexer" kernel matches
    # the "index" needle in the elementwise rule below, so without an earlier
    # claim it lands as elementwise. That is worse than `other`: `other` is a
    # visible finding, whereas a misfiled kernel makes the attention bucket look
    # cheap and the elementwise bucket look inexplicably expensive.
    # "flashinfer" is deliberately absent. It is a *vendor*, not an operation:
    # FlashInfer ships attention, sampling, norm and GEMM kernels, and vLLM logs
    # "Using FlashInfer for top-p & top-k sampling" on this stack. With the
    # vendor needle here — ahead of the sampling rule — every FlashInfer sampler
    # was reported as attention time. A misfile is worse than `other`: `other` is
    # a visible finding, whereas this silently inflated a bucket that gets
    # trusted. Its actual attention kernels are named for what they do, so they
    # are matched by name below.
    ("attention", ("flash_fwd", "flash_attn", "flashattn", "fmha", "paged_attention",
                   "paged_attn", "attention", "attn_score", "splitkv", "merge_attn",
                   "mha_fwd", "cutlass_mla", "flash_mla", "mla_sparse", "sparse_mla",
                   "indexer", "lightning_index", "batchprefill", "batchdecode",
                   "pagedkv", "prepare_varlen", "compute_attn")),
    ("kv_cache", ("reshape_and_cache", "slot_mapping", "copy_blocks", "swap_blocks",
                  "concat_and_cache", "block_table")),
    # "nvjet" is cuBLAS's JIT-generated Hopper/Blackwell GEMM family
    # (`nvjet_sm90_tst_128x8_64x12_4x1_v_bz_TNT` — tile shape and BLAS transpose
    # notation). It carries none of the classic needles, so on a Qwen3.6 H200
    # capture 18.7% of all device time sat in `other` while the `gemm` bucket
    # reported 3.6%. Worse, the family was *split*: the `..._splitK_...`
    # variants matched "splitk" and classified, so one kernel family landed in
    # two buckets by accident of naming.
    ("gemm", ("gemm", "cutlass", "sgemm", "hgemm", "s16816", "s161616", "matmul",
              "cublas", "marlin", "machete", "scaled_mm", "wgrad", "tensorop",
              "gemv", "splitk", "nvjet", "xmma")),
    ("quant", ("quant", "dequant", "scaled_fp8", "per_token_group", "awq", "gptq",
               "fp8_", "int8_", "nvfp4", "mxfp4")),
    ("norm", ("rms_norm", "rmsnorm", "layer_norm", "layernorm", "fused_add_rms",
              "l2norm", "l2_norm")),
    ("rope", ("rope", "rotary")),
    ("activation", ("silu", "gelu", "swiglu", "act_and_mul", "relu")),
    # "sampling" and the unpunctuated "topk"/"topp" are load-bearing: FlashInfer
    # names its sampler `TopKTopPSamplingFromProbKernel`, and neither "sample"
    # (not a substring of "sampling") nor "top_k" (not a substring of "TopKTopP")
    # matches it. Removing the vendor needle from `attention` without adding
    # these would move the sampler from a wrong bucket to no bucket.
    # MoE's `topk_softmax` router is claimed by the `moe` rule far above, so the
    # loose needles here cannot reach it.
    ("sampling", ("sample", "sampling", "argmax", "top_k", "top_p", "topk", "topp",
                  "softmax", "penalt", "logits", "logprob", "multinomial", "gumbel")),
    ("elementwise", ("elementwise", "vectorized", "fill", "copy", "cat_", "concat",
                     "index", "transpose", "gather", "scatter", "reduce", "cast",
                     "arange", "zero", "memset", "unrolled")),
)


def classify_kernel(name: str) -> str:
    """Bucket one kernel name. Never ``None`` — an unrecognised kernel is ``other``."""
    n = (name or "").lower()
    if not n or n == "<anonymous>":
        return "unnamed"
    for bucket, needles in _RULES:
        if any(k in n for k in needles):
            return bucket
    return "other"


@dataclass
class BucketStat:
    bucket: str
    n_kernels: int
    time_ns: int
    share: float          # of total kernel time, not of wall time
    top_names: list[str] = field(default_factory=list)


@dataclass
class KernelBreakdown:
    """Bucket shares plus the three coverage signals worth failing a run over."""

    n_kernels: int
    kernel_time_ns: int
    buckets: list[BucketStat]
    per_device_active_ns: dict[int, int]
    window_ns: int | None
    gpu_active_share: float | None   # busiest device's active time / window
    n_truncated_names: int           # records whose name hit NAME_MAX
    n_distinct_truncated: int        # distinct such names — collisions hide here
    n_devices: int

    @property
    def other_share(self) -> float:
        """Kernel-time share that matched no rule. High = the trace is unreadable."""
        for b in self.buckets:
            if b.bucket == "other":
                return b.share
        return 0.0

    def warnings(self) -> list[str]:
        """Everything about this trace that should stop a conclusion being drawn."""
        out: list[str] = []
        if self.n_kernels == 0:
            out.append(
                "no kernels captured at all. The usual cause is decode running as "
                "CUDA-graph replay that this CUPTI does not attribute — re-run with "
                "--enforce-eager to confirm."
            )
            return out
        if self.gpu_active_share is not None and self.gpu_active_share < 0.2:
            out.append(
                f"GPU active only {self.gpu_active_share:.1%} of the window. Either the "
                f"load left the engine idle, or kernels ran without being attributed "
                f"(CUDA-graph replay). Compare against the served throughput before "
                f"trusting any per-kernel share below."
            )
        if self.n_truncated_names:
            out.append(
                f"{self.n_truncated_names} records ({self.n_distinct_truncated} distinct) "
                f"have names at the {NAME_MAX}-byte cap and are truncated. Long mangled "
                f"cutlass/MoE template names collide once cut, so distinct kernels may "
                f"be merged here. Raise GITM_NAME_MAX in cupti_core.h to separate them."
            )
        if self.other_share > 0.25:
            out.append(
                f"{self.other_share:.1%} of kernel time matched no bucket rule — the "
                f"breakdown is not describing most of the work. See the 'other' "
                f"top_names and extend _RULES."
            )
        for want in ("gemm", "moe"):
            hit = next((b for b in self.buckets if b.bucket == want), None)
            if hit is None or hit.n_kernels == 0:
                out.append(
                    f"no '{want}' kernels in the trace. For a dense-MoE decode that is "
                    f"not plausible — expect either graph-replay attribution loss or a "
                    f"vocabulary gap, not a genuine absence."
                )
        return out


def _active_ns(intervals: list[tuple[int, int]]) -> int:
    """Union of [start, end) intervals — wall time with *something* resident.

    Summing durations would double-count: concurrent kernels on one device overlap
    constantly, and the naive sum routinely exceeds the window it is measured
    against, producing an "active share" above 100%.
    """
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    return total + (cur_end - cur_start)


def summarize_kernels(kernels, *, window_ns: int | None = None,
                      top_n: int = 5) -> KernelBreakdown:
    """Bucket a trace's kernel events and measure how readable the result is.

    ``window_ns`` is the capture window; without it ``gpu_active_share`` is ``None``
    rather than divided by a span inferred from the kernels themselves, which would
    define away the very idleness the number exists to detect.
    """
    by_bucket: dict[str, list] = {}
    per_device: dict[int, list[tuple[int, int]]] = {}
    truncated: list[str] = []
    total_time = 0

    for k in kernels:
        name = getattr(k, "name", "") or ""
        start, end = int(k.start_ns), int(k.end_ns)
        dur = max(end - start, 0)
        total_time += dur
        by_bucket.setdefault(classify_kernel(name), []).append((name, dur))
        per_device.setdefault(int(getattr(k, "device_id", 0) or 0), []).append((start, end))
        if len(name) >= NAME_MAX:
            truncated.append(name)

    buckets: list[BucketStat] = []
    for bucket, entries in by_bucket.items():
        t = sum(d for _, d in entries)
        names: dict[str, int] = {}
        for name, d in entries:
            names[name] = names.get(name, 0) + d
        buckets.append(BucketStat(
            bucket=bucket,
            n_kernels=len(entries),
            time_ns=t,
            share=(t / total_time) if total_time else 0.0,
            top_names=[n for n, _ in sorted(names.items(), key=lambda kv: -kv[1])[:top_n]],
        ))
    buckets.sort(key=lambda b: -b.time_ns)

    active = {dev: _active_ns(iv) for dev, iv in per_device.items()}
    share = (max(active.values()) / window_ns) if (active and window_ns) else None

    return KernelBreakdown(
        n_kernels=sum(b.n_kernels for b in buckets),
        kernel_time_ns=total_time,
        buckets=buckets,
        per_device_active_ns=active,
        window_ns=window_ns,
        gpu_active_share=share,
        n_truncated_names=len(truncated),
        n_distinct_truncated=len(set(truncated)),
        n_devices=len(active),
    )


def format_breakdown(bd: KernelBreakdown) -> str:
    """Human-readable table — what the pod run prints when it finishes."""
    lines = [
        f"{bd.n_kernels} kernels, {bd.kernel_time_ns / 1e9:.2f}s kernel time "
        f"across {bd.n_devices} device(s)"
    ]
    if bd.gpu_active_share is not None:
        lines.append(f"GPU active (busiest device): {bd.gpu_active_share:.1%} of the window")
    lines.append(f"{'bucket':<12} {'kernels':>9} {'time_s':>9} {'share':>7}")
    for b in bd.buckets:
        lines.append(f"{b.bucket:<12} {b.n_kernels:>9} {b.time_ns / 1e9:>9.3f} {b.share:>6.1%}")
    return "\n".join(lines)
