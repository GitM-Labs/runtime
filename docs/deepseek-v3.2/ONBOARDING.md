# Model onboarding procedure exercised by DeepSeek V3.2

**Scope:** a procedure derived from the current static integration. Deployment and trace steps are specified but have **not** been exercised for the target engagement. The September 25 deliverables are therefore not represented as a passed end-to-end gate.

## Sequence and responsibility

| Step | Kind | Automated evidence/check | Human decision and failure caught |
|---|---|---|---|
| 1. Pin the target | Generic | Record immutable checkpoint repo/revision, config/index hashes and actual served identity in a versioned contract | Collin confirms engagement identity. A public model with the same name is not proof of what Parasail serves. |
| 2. Verify checkpoint | Generic mechanism, DeepSeek-specific tensor rules | Call Zhu's checkpoint/engine verifier when its interface is available. Current evidence interface is `verify_deepseek_checkpoint`; `scripts/audit_safetensors_headers.py` reproduces dtype/component bytes without reading payloads. | Zhu agrees on artifact set and verifier result. Changed pin, missing shard/tensor, altered dimension or quantization fails before planning. Do not duplicate his verifier logic manually in a playbook. |
| 3. Decide architecture/schema | DeepSeek-specific | Catalogue loader checks fields; tensor index proves dense/MoE and indexer-weight schedule; negative tests exercise drift | Rahul judges MLA/DSA/MTP equivalence to `glm_moe_dsa`. Ishaan owns schema gaps when mixed precision, sharding or execution cannot be represented without distortion. Weight presence alone does not prove runtime selection. |
| 4. Build the planner graph | Generic path, DeepSeek-specific values | `load_spec`, `predict`, raw-config/live-route tests, tensor-class ledger, TP/EP collective tests | Rahul re-derives stored/compute/cache separation and signs the technical gate. The 4,018,393-byte analytic residual and engine assumptions remain visible. |
| 5. Pin engine and deployment | Generic | Fill `deployment-contract.yaml` with engine source/image, complete launch/env, SKU/ranks and TP/EP/DP; `python -m gitm.serve.deployment_contract ...` lists missing fields/owners | Collin/Adit confirm real defaults, kernels, MLA path, expert padding and per-rank placement. A complete field is not automatically verified. |
| 6. Package replay | Generic | Version trace source/hash, canonical request/token totals, prefix and timing rules, replay argv. Existing `gitm/traffic/replay.py` checks block coverage for vLLM `timed_trace`. | Collin supplies representative private traffic; Medha decides valid source kind. Public/synthetic traces cannot be relabeled production. |
| 7. Reproduce and capture | Generic | A second engineer follows bring-up, replay, capture, teardown commands and checks checkpoint/engine/topology/workload hashes | Adit reproduces; Isaiah checks experiment contract. Missing hardware/inputs remain dated blockers, not a pass. |
| 8. Qualify trace | Generic | Nathan's gate verifies rank completeness and all deployment identities, then rejects mismatched/missing-rank controls. `validate_trace.py` only checks offline attribution coverage. | Nathan decides admissibility. An attribution pass is insufficient. |
| 9. Reconcile and fix | Generic method, DeepSeek-specific regions | Align predicted and observed kernels/transfers/collectives; report time interval, variance, traffic, unmatched work and residual classification for each material region | Rahul reviews cause. Correct the highest-impact contradiction in production path and add a regression test; unidentifiable gaps remain explicit. |

## Generated coverage, not a second maintained status table

With the pinned config/index available and a fresh JUnit file, generate state from the evidence and contract:

```bash
python -m pytest tests/test_deepseek_v32.py --junitxml=deepseek-tests.xml -q
python -m gitm.serve.deepseek_coverage \
  docs/deepseek-v3.2/deployment-contract.yaml \
  --config /path/to/pinned/config.json \
  --index /path/to/pinned/model.safetensors.index.json \
  --junit deepseek-tests.xml > deepseek-coverage.json
```

The generator checks static identity, reads actual DeepSeek test case results, and derives blockers from the contract. It keeps the qualified-trace row `UNVERIFIED` until Nathan's accepted gate result can be integrated. If config/index or JUnit is absent, it reports `UNVERIFIED` rather than copying an old green matrix. Treat the JSON as a generated review artifact; the inputs remain the sources of truth.

## Failure modes caught in this integration

- An FP8 weight checkpoint was mistaken for an FP8 KV cache. Cache representation now remains a deployment unknown and the live path reads the serving flag.
- Raw `deepseek_v32` was not routed to the MLA/DSA graph. It now detects the family but refuses a config-only prediction without the tensor index; pinned live snapshots use the catalogue spec.
- The analytic footprint omitted separate MTP BF16 embedding/head/fusion weights (~3.91 GB). The shared footprint path now counts them; remaining ~4 MB is not erased.
- The index's `total_size` looked almost 2× the stored payload. Header inventory shows it is exactly two bytes times all indexed elements, including scales; exact payload uses `data_offsets`.
- A single execution dtype was used to price router and indexer storage. Storage overrides now distinguish the BF16 router and `indexer.weights_proj`; mixed execution traffic remains an open schema/engine question.
- An offline attribution checker could be mistaken for a full trace qualification gate. The handoff calls Nathan's gate separately.

## Review handoff

Give Rahul the pin and evidence manifest, the gap map, this design note, the current contract validator output and the targeted test report. Give Zhu the same hashes and header manifest for his verifier. Give Adit and Nathan a filled contract only when their actual inputs exist. PRs should state what was agent-written and what Medha or a reviewer checked by hand, along with exact commands/results and unresolved assumptions. Keep customer-derived values in the private location designated by the engagement owner.
