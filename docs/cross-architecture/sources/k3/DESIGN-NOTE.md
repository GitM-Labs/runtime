# Kimi K3 decode prediction

I worked from `moonshotai/Kimi-K3` at revision
`f831ab66814297da540d832a5235f8e904f29d06`. For the serving side, I used
SGLang v0.5.19 at commit
`0bcd822377da7b5718e674eaf9c870d349424dd1`. I did not use traces.

The workload is 32 active sequences, 8,192 cached tokens per sequence, and one
new token per sequence. I assumed text-only decode with speculative decoding
turned off.

The first thing I checked was the layer schedule. Kimi K3 has 93 text layers:
69 Kimi Delta Attention (KDA) layers and 24 MLA layers. The config lists the
layers starting at one, while the implementation checks `layer_idx + 1`. After
converting to zero-based indices, the MLA layers are 3, 7, 11, and so on through
91, followed by another MLA layer at 92. That last pair means the schedule
cannot be represented as a simple repeating group all the way to the end.

Layer 0 has a normal dense MLP with intermediate size 33,792. Every later layer
uses a mixture of experts. There are 896 routed experts, 16 selected per token,
and two shared experts. The routed path first reduces the model width from 7,168
to 3,584, runs the selected experts at that smaller width, and projects back to
7,168. This matters a lot for both weight and traffic calculations.

## Engine assumptions

The official SGLang Kimi K3 recipe for H200 uses two eight-GPU nodes with TP16
and EP16. It also pins Marlin for the routed experts, FlashMLA for decode
attention, symmetric memory, and a static memory fraction of 0.85. The recipe
currently labels the H200 setup as verification in progress, so I treated it as
the best published deployment shape rather than a measured result.

SGLang determines several things that are not in the checkpoint:

- **Read from engine source:** decode uses the full CUDA graph backend by
  default, with one captured graph for each supported batch size. I assume batch
  32 is among the captured sizes because the recipe does not disable graphs.
- **Read from engine source:** KDA merges the wide q, k, v, and output-gate
  projections. Its optimized decode path can also combine the short convolution,
  recurrent update, and gated RMSNorm. The smaller beta and forget-gate
  projections remain separate.
- **Read from docs and source:** MLA decode uses FlashMLA. The output gate can run
  on a side stream, and the output projection can write into symmetric memory so
  that its following reduction is fused.
- **Read from engine source:** on H200, Marlin reads the packed MXFP4 expert
  weights but operates on BF16 activations. The checkpoint stores four-bit expert
  weights; this does not make the entire decode step FP4.
- **Assumed:** expert dispatch and combine use an NCCL-compatible all-to-all over
  the two nodes. The recipe does not pin the exact all-to-all backend or the NIC
  bandwidth, so I used 400 Gb/s per GPU as a rough fabric ceiling.
- **Read from defaults:** the published H200 command does not request FP8 KV or a
  lower-precision KDA state. I therefore use BF16 MLA KV, FP32 recurrent state,
  and BF16 convolution state.

A decode layer should roughly run in this order: residual normalization,
attention projection, either a KDA state update or an MLA cache read, output
projection and TP reduction, expert routing, latent projection, EP dispatch,
Marlin expert computation, EP combine, projection back to model width, shared
expert merge, and the final residual update. Some neighboring operations fuse
or overlap in SGLang, so this is the dependency order rather than a claim that
every item launches as a separate kernel.

## Memory fit

The safetensors index gives a total checkpoint size of **1,560.86 GB**. The
routed experts account for about 2.723 trillion logical weights. Their packed
four-bit data takes 1,361.37 GB, and the one-byte scale for every group of 32
weights adds 85.09 GB. The remaining tensors occupy 114.40 GB.

An eight-GPU node would need about 195.11 GB of weights on every GPU if the
weights were divided evenly. An H200 has 141 GB, so the assigned node cannot
hold this checkpoint even before adding KV cache or workspace.

Twelve GPUs are the bare storage lower bound. That is not a usable parallel
layout here because 12 does not evenly divide the 896 experts. Sixteen GPUs are
the first clean layout that divides both 96 attention heads and 896 experts.
This agrees with the published TP16/EP16 recipe.

My per-GPU estimate for that layout is:

| Item | GB per H200 |
| --- | ---: |
| Packed expert weights | 85.09 |
| Expert scale metadata | 5.32 |
| Other weights | 7.15 |
| MLA KV cache | 7.25 |
| KDA recurrent and convolution state | 0.90 |
| Activation and Marlin workspace, estimated | 2.00 |
| Communication buffers, estimated | 4.00 |
| **Working total** | **111.70** |

I reserve 21.15 GB, or 15 percent of the device, for allocator slack, graph
capture, and runtime variation. That leaves about 8.15 GB unassigned. I would
not treat this as proof that the model will start, since expert packing may not
divide perfectly across ranks. I would reject this configuration if the actual
startup log shows less than 4 GB free after graph capture.

For the state calculations, each MLA token stores 512 latent values and 64 RoPE
values in each of 24 MLA layers. At BF16, 32 sequences and 8,192 tokens use
7.25 GB per GPU because this recipe does not use DCP. KDA history does not grow
with context. Its fixed state is about 0.87 GB of FP32 recurrent matrices plus
0.03 GB of BF16 convolution history per GPU.

## Top three bound hypotheses

### 1. Routed-expert computation is limited by HBM traffic

Each rank owns 56 experts under EP16. At batch 32, sequence-parallel routing
gives each rank about 32 assignments per layer. If routing is close to uniform,
those assignments touch about 24.5 distinct local experts. One expert occupies
about 17.55 MB including its scales. Across 92 MoE layers, this gives roughly
39.6 GB of expert-weight reads per step. At the H200's 4.8 TB/s peak bandwidth,
the ideal lower bound is about 8.3 ms. Actual Marlin time will be higher because
each selected expert receives very few rows.

**Experiment:** keep topology, context length, expert placement, and routing IDs
fixed. Sweep batch size through 8, 16, 32, and 64. Record Marlin time, HBM bytes,
tensor-core use, and the number of distinct experts touched. I would reject the
memory-bound hypothesis if tensor utilization stays above 60 percent while HBM
stays below 50 percent, or if time follows the number of assignments more closely
than the number of distinct experts. This test does not show how much of Marlin
time overlaps communication.

### 2. The many small collectives are limited by latency

SGLang dispatches sequence-parallel token shards, so every rank starts with two
of the 32 tokens. That gives 32 token-expert assignments per rank. Sending the
3,584-wide BF16 latent takes about 0.229 MB in each direction per MoE layer.
Across 92 layers, dispatch and combine total about 42.2 MB per rank. The payload
itself is fairly small. The larger concern is the repeated launch and
synchronization cost of as many as 184 EP collectives, plus the TP reductions
after attention and any unfused branches.

**Experiment:** hold the model, batch, routing IDs, context, and kernels fixed.
Compare the two-node layout with an equivalent 16-GPU NVSwitch domain, or vary
only the inter-node latency. Record collective time, exposed synchronization
gaps, overlap, and total step time. I would reject this hypothesis if all
communication stays below 10 percent of the critical path and doubling fabric
latency changes step time by less than 5 percent. This experiment alone cannot
separate a poor collective algorithm from uneven routing.

### 3. FlashMLA is limited by KV-cache bandwidth

The 24 MLA layers read about 7.25 GB of BF16 cache per decode step at the baseline
workload. This is the largest local-memory term that grows with context length.
Using peak HBM bandwidth gives a lower bound near 1.5 ms.

**Experiment:** keep batch 32, routing, topology, and precision fixed. Sweep the
cached context through 2K, 4K, 8K, and 16K. Record FlashMLA time and HBM read
bytes. A memory-bound result should grow close to linearly once fixed overhead is
small. I would reject the hypothesis if doubling context above 4K increases MLA
time by less than 50 percent without a matching rise in achieved bandwidth. This
does not answer whether switching the cache to FP8 preserves quality.

## What I would do next

The current GitM `hybrid` model schema cannot describe four Kimi K3 details at
the same time: MLA inside a hybrid schedule, the 3,584-wide latent experts,
MXFP4 expert storage, and BF16 convolution state beside FP32 recurrent state. I
marked those fields in the YAML instead of replacing them with convenient but
incorrect dimensions.

Given another two hours, I would add those four fields to
`HybridMoEModelSpec`, test the predicted weight total against the exact
1,560,860,324,864-byte index value, and record the all-to-all backend selected by
an actual SGLang launch. Until then, the memory and bound calculations above are
manual. I think that is safer than producing a planner run whose shapes look
valid but do not match the checkpoint.

## Primary sources

- Kimi K3 [`config.json`](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)
- Kimi K3 [`model.safetensors.index.json`](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/model.safetensors.index.json)
- Kimi K3 [`configuration_kimi_k3.py`](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/configuration_kimi_k3.py)
- Kimi K3 [`modeling_kimi_linear.py`](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)
- Official [SGLang Kimi K3 recipe](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
- SGLang [`models/kimi_k3.py`](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/models/kimi_k3.py)
- SGLang [`moe_runner/marlin.py`](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/layers/moe/moe_runner/marlin.py)
- SGLang [`decode_cuda_graph_runner.py`](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)
