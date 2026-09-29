# DeepSeek V3.2 planner design note — static evidence, open execution contract

**Review state:** static checkpoint audit reproduced; engine, target deployment, hardware, traffic, and qualified observation `UNVERIFIED`. Rahul has not signed a technical gate and Zhu's verifier has not accepted this bundle. The catalogue graph is an exploratory structural prediction until those gates are satisfied.

## 1. Pin and reproduction

Public checkpoint: [`deepseek-ai/DeepSeek-V3.2`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) revision `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`.

| Artifact | SHA256 / fact |
|---|---|
| `config.json` | `c7fa8b191e9936d8e6a57d864baab82b792fae16a116416cdd3a75ba76bc5af1` |
| `model.safetensors.index.json` | `2a150b2af4aba7b037edc9fdba70f3b41abf758ee926279e7701d954298f9884` |
| Indexed tensors / shard headers | 92,425 / 163, all headers range-read and checked against the index |
| Exact stored payload | **689,471,107,200 bytes** |

The evidence is `gitm/planner/models/deepseek-v3.2.evidence.json`, including component/dtype ledgers and each header digest. Re-audit without downloading tensor payloads using `scripts/audit_safetensors_headers.py --index <pinned-index> --remote-base https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/ --expected gitm/planner/models/deepseek-v3.2.evidence.json`; the host must honor HTTP byte ranges. A local full checkpoint can use `--shards-dir <checkpoint-dir>` instead.

`verify_deepseek_checkpoint(config,index,revision=...)` checks hashes, architecture dimensions, quantization recipe, indexer/MLP schedules, MTP tensors, and count/ledger closure. Its result is `STATIC_VERIFIED_ENGINE_UNVERIFIED`. It does not compare engine code, weights' contents, or deployment.

## 2. Exact parameter and stored-byte closure

The total indexed element count is **685,396,921,376**, including scale and other tensors. Partition by stored dtype/tensor kind:

| Tensor kind | Count | Elements | Stored bytes |
|---|---:|---:|---:|
| FP8 E4M3 weights | 45,932 | 681,408,069,632 | 681,408,069,632 |
| BF16 weights | 126 | 3,946,184,704 | 7,892,369,408 |
| FP32 block scales | 45,932 | 41,591,584 | 166,366,336 |
| Other FP32 weights | 314 | 1,052,416 | 4,209,664 |
| Other FP32 tensors | 121 | 23,040 | 92,160 |
| **Total** | **92,425** | **685,396,921,376** | **689,471,107,200** |

Component ledger from the same headers (includes scales, norms and MTP within each category):

| Component | Stored bytes |
|---|---:|
| Routed experts, backbone + MTP | 665,345,458,176 |
| Shared experts, backbone + MTP | 2,599,005,696 |
| Attention projections, backbone + MTP | 11,603,874,368 |
| Indexer, backbone + MTP | 894,178,880 |
| Leading dense FFN | 1,189,375,488 |
| Router weights, backbone + MTP | 216,530,944 |
| Backbone embedding + output head | 3,706,716,160 |
| **Separate MTP embedding, output head, fusion and auxiliary norms** | **3,912,381,440** |
| Other norms/biases | 3,586,048 |
| **Total** | **689,471,107,200** |

The worklog's apparent `metadata.total_size` conflict is resolved arithmetically: `1,370,793,842,752 = 2 × 685,396,921,376` indexed elements. That field reflects a two-byte-per-element accounting for this index, **including scales**, and is not the on-disk FP8 payload. This equation is verified for the pinned index; no general rule about other checkpoints is assumed.

The existing analytic `model_weight_bytes` originally omitted a separate MTP BF16 embedding/head/fusion (~3.91 GB). The shared `glm_graph.py` path now represents those resident tensors and separates the router's and indexer's stored dtype from execution dtype. At TP1/EP1 its estimate is **689,467,088,807 B**, **4,018,393 B low** versus the exact header ledger. Norm/bias and block-boundary details remain to reconcile. The exact whole-checkpoint payload is known; **exact per-rank placement and memory are not** until engine sharding/padding are pinned. Do not turn the 4 MB difference into a free efficiency adjustment.

## 3. Graph mapping and what the checkpoint can establish

`glm_moe_dsa` is the current graph shape: 61 MLA/DSA backbone layers, 3 dense then 58 sparse MLP layers, 256 routed experts and top-8 per token plus one shared expert, 128 attention heads, `kv_lora_rank=512`, `qk_rope_head_dim=64`, `index_topk=2048`, and one MTP module. The index has each layer's indexer weights, including the MTP layer. The model structure supports 61 full-indexer graph layers; engine source/trace must still establish whether every selection is recomputed and how MTP executes.

The planner's cached MLA entry has **512 latent + 64 RoPE elements** per token per layer; it is shared across query heads. This is a structural shape, not a cache-byte assertion. The catalogue's BF16 KV scenario prices `512×2 + 64×2 = 1,152` bytes per entry before indexer key, while an FP8 latent/BF16 RoPE scenario changes that quantity. Actual layout, scales and cache kernel depend on the pinned engine launch. The FP8 *weight* checkpoint does not imply an FP8 cache.

The catalogue loads and `model_catalogue.predict('deepseek-v3.2')` builds prefill/decode graph regions and TP/EP collectives. Raw `deepseek_v32` config dispatch now recognizes this family but refuses a config-only prediction, since that file alone lacks the indexer schedule. The live serve path checks a pinned snapshot's config/index first and uses the catalogue spec only when they match. This is the production route, with `checkpoint_evidence` and its `UNVERIFIED` engine/deployment fields carried into attach output. The attach result labels the resulting floor **exploratory**.

The graph still models unabsorbed MLA, an FP32 router execution hypothesis, no expert-capacity padding, and a particular MTP selection path. Those are not checkpoint facts. A BF16 `indexer.weights_proj` lives beside FP8 `wq_b` and `wk`: storage is accounted separately, while mixed runtime kernel traffic and peak selection need pinned engine evidence and, if necessary, Ishaan's schema change. Exact temporary traffic and accumulation precision are likewise `UNVERIFIED`.

## 4. Topology and execution tests

`ShardingConfig(tp, ep, dp)` models per-rank work, with EP within the TP group. Tests check that TP8/EP8 changes memory and emits all-to-all, while an invalid TP7 fails the head-divisibility check; these are **scenarios**, not the engagement topology. The actual rank map, GPU SKU, padding, collective backend and engine revision must come from Adit/Collin. Changing the pinned revision, config dimensions, quantization recipe or tensor schedule fails static verification; the workload contract rejects a changed checkpoint pin and an incompatible topology.

Run the CPU-only checks:

```bash
python -m pytest tests/test_deepseek_v32.py tests/test_glm_graph.py tests/test_serve_model_config.py -q
python -m gitm.serve.deployment_contract docs/deepseek-v3.2/deployment-contract.yaml
```

The second command currently exits with unresolved field blockers. That is expected and prevents a reproducible-launch claim. Existing GLM/Kimi planner tests should also continue passing.

## 5. Evidence needed to advance the claim

| Decision | Required input / reviewer |
|---|---|
| Does this checkpoint match the engagement? | Collin's served checkpoint identity; Zhu's verifier agreement |
| What operations actually execute? | Engine source commit/image, all flags/defaults, backend/kernel selection and padding; Rahul reviews the graph consequence |
| What fits on each rank? | Adit's GPU/rank/TP/EP/DP deployment manifest and engine placement behavior |
| What workload was measured? | Private traffic manifest, replay command and reconciled request/token identity |
| Is the observation admissible? | Nathan's multi-rank qualification result, including deliberate mismatch/missing-rank rejection |
| Do predicted and observed regions agree? | Kernel/transfer/collective mapping, interval predictions, observed variance, unmatched-work residuals and contradiction fix |

See `GAP-MAP.md` for each boundary and `deployment-contract.yaml` for machine-readable unknowns. No customer-facing timing claim follows from this static design note alone.
