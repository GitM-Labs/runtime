# Partial gap fixes

Status update October 14, 2026: [PR #178](https://github.com/GitM-Labs/runtime/pull/178) merged these fields. References below to local or uncommitted work describe the October 11 snapshot.

October 11, 2026. User-authorized local fixes, uncommitted. Existing catalogue defaults and measured evidence remain unchanged.

## Hybrid convolution storage

`HybridMoEModelSpec` now accepts the original submission’s `linear_conv_channels` and `linear_conv_state_dtype` fields. Explicit channels take precedence over the existing fitted `conv_dim`; omission retains that field. Explicit convolution state precision is independent of `ssm_state_dtype`; omission retains the old shared-state precision.

Both the state-byte calculation and emitted convolution read/write traffic use these fields. The planner does not provide a fit ledger for hybrid models, so the state-byte calculation is not a capacity verdict. The HF reader preserves explicitly supplied fields rather than overwriting them with the value-head-width fallback. Explicit channels must divide TP, so a remainder cannot disappear through integer division. Channel and dtype validation rejects malformed opt-in inputs. Fields are appended after existing fields to avoid shifting existing positional arguments.

For the submitted K3 geometry, convolution history has 36864 channels with three previous inputs under the current graph’s history convention. BF16 stores 221184 bytes per sequence/layer instead of 442368 bytes when charged as FP32. Recurrent-state storage remains FP32. This fixes the independent-storage expression; it does not establish the actual engine allocation policy, fusion, or physical KDA lowering.

The original submission is unchanged and still rejected for 14 other scalar keys. Full nested K3 reading and MLA/latent-MoE/KDA integration remain separate work. The two stale state operation overrides are not silently renamed into arithmetic overrides; doing so could introduce an unsupported compute-precision change.

## Independent index cache storage

`GlmMoeDsaModelSpec` now accepts `index_kv_dtype` and `index_scale_bytes_per_token`. Defaults inherit the existing `kv_dtype` and charge no additional index scale bytes, preserving legacy behavior and multiplication order. Explicit fields are preserved by the config reader and catalogue loader. Cache layout is not inferred from model weight precision or an SN40L bundle label.

The new entry-byte helper is used in the resident KV ledger, index-key writes and index-key reads, including prefill and TP-replicated scans. A 128-element FP8 index key with one four-byte scale is 132 bytes per token. BF16 MLA latent plus RoPE remains 1152 bytes per layer/token for rank 512 and RoPE width 64. Scale bytes are per cached key, not per index head, weight block or tensor-parallel shard. Index FLOPs and attention-core traffic remain unchanged.

This closes CF-08’s local representation gap. It does not implement V3.2 family detection or establish SambaNova’s deployed precision. Those remain distinct gaps.

## Verification

The selected planner suite passes 512 cases, including 29 focused cases for these changes. The focused cases check independent storage, capacity and graph consumers, explicit reader/loader support, decode/prefill behavior, TP replication, aliases and rejected malformed inputs. Fourteen legacy catalogue/Qwen output comparisons match the original hashes. Ruff and the whitespace check pass. This is neither the complete repository suite nor a new GPU validation run.

## Remaining partial work

K3 MLA in the hybrid schedule, latent expert execution, router behavior and mixed expert storage need complete consumer paths and evidence. V3.2 detection needs its own supported model reader. SN40L execution rules, placement and cache precision require deployment evidence. Named tiers and result semantics already have optional local implementations; publication and review are separate from representation completeness. The register retains these distinctions rather than changing every status to resolved.

No commit, push, external message or additional Runpod spend occurred.
