# We Gave Our Runtime Four GPU Workloads and No Instructions. It Diagnosed Each and Proved Every Fix.

*The trace picks the lever. A gated A/B proves it.*

**+32.4% on HFT, byte-identical output on 400 million events. On the edge tracks, up to +427% on KITTI and +37% on nuScenes, where the gate is throughput plus detection equivalence. Zero human tuning, every failed change rolled back.**

*July 2026*

[Reproduce our benchmarks](#try-it-yourself)

-----

We think most GPU fleets run far below the hardware they are paying for, and that the waste is a systems problem, not a kernel problem. The GPU is rarely slow because the math is slow. It is slow because it waits: launching thousands of tiny kernels one at a time, stalling at synchronization points, moving memory it did not need to move, or running work in series that could have overlapped.

Our runtime does one thing: it closes the loop between diagnosis and proof.

- **Observe.** Profile the workload from its own telemetry (CUPTI, NVML). No model-code changes or source access.
- **Diagnose.** Read the dominant bottleneck off the trace: launch-bound, compute-bound, or serialized.
- **Apply.** The candidate intervention selected for that bottleneck.
- **Verify.** A gated A/B on real hardware, best-of-N to suppress jitter.
- **Keep or roll back.** Keep only if the candidate is both faster and output-equivalent. Otherwise revert and report nothing.

Every number in this post is a kept decision, not a guess.

A good engineer can tune one workload once. A fleet is thousands of workloads changing weekly: new models, new request mixes, new kernels, new traffic shapes, new failure modes. That is why this has to be a runtime loop, not a consulting pass. The loop earns its place because it runs continuously, per job, where hand-tuning gives up. Any one result below is useful; the point is that the same loop found all of them automatically, verified each one, and rolled back the ones that failed.

Two honest notes before the numbers. These are benchmark workloads, not a customer’s production fleet, and what they prove is the loop, that the runtime can read what actually limits a workload and apply a safe, verified fix without touching model code. Separately, our commercial wedge is inference serving, and the same loop is now pointed there in a vLLM decode benchmark we describe at the end. We are not pretending these four workloads stand in for that one. They show the loop; the next one runs it on what we sell into.

## Where the Waste Actually Lives

Two readings from the trace tell us most of what we need. The first is whether the GPU is **launch-bound** or **compute-bound**: burning time issuing many small kernels, or genuinely busy doing large-matrix arithmetic? The second is **serialized concurrency**, our reading of how much work runs with no overlap.

A serialized concurrency of 1.000 means your GPU is effectively single-threaded at the workload level: thousands of kernels standing in a single-file line, each waiting on the one before it, a piece of hardware built for massive parallelism running its work one at a time.

These two readings decide the lever. A launch-bound workload wants its launches amortized. A compute-bound workload does not care about launch overhead at all, so amortizing launches buys nothing; it wants the arithmetic itself made cheaper. Pick the wrong lever and you do real work for a 0% result.

## Track 1: HFT Limit Order Books (the cleanest proof)

The strongest result first: **+32.4% throughput with byte-identical output, on 400 million events.**

```
optimize hft: 400,000,000 events on NVIDIA A100 80GB PCIe

  baseline : 69,323,880 events/s
  candidate: 91,786,039 events/s  (1.32x)
  identical output: True

  VERDICT: kept candidate — verified +32.4% faster, identical output
```

Here is why it is the trust anchor. The order book is the live list of every outstanding buy and sell for an instrument; HFT churns it constantly into a stream of nanosecond-timestamped events. Real HFT flow is proprietary, so we generate a **deterministic synthetic stream per seed** with the structural properties that exercise scan-bound execution, and we say so plainly: generated data, reproducible from a seed, not a customer’s tape. We are testing an execution path, not claiming market realism. On it we compute three per-symbol microstructure features (top of book, microprice, one-second VWAP).

The trace reads as launch-and-scan-bound, so the lever is not precision and not batch size. It is **doing fewer scans**, collapsing redundant passes over the book while preserving the exact output. Because the computation is deterministic, we hold it to byte-identical output, not a tolerance: two reductions form an output signature, and the candidate is kept only when that signature matches to the byte. +32.4% faster, and every byte the same. No accuracy hand-waving is possible, because the correctness bar is exact.

## Tracks 2 and 3: The Same Lever, Very Different Headroom

The edge tracks are where the diagnosis argument lands. They run the *identical* lever and get wildly different results, and the direction of that difference is exactly what the bottleneck reading predicts.

**KITTI** is the standard autonomous-driving 3D detection benchmark: one LiDAR frame in, scored 3D boxes out. Model is PointPillars (OpenPCDet, pinned `pointpillar.yaml`, `pointpillar_7728.pth`). Its trace is unambiguous: serialized concurrency 1.000, 14,590 kernels in single file, the GPU spending its time launching rather than computing.

**nuScenes** is the heavier cousin: ten accumulated LiDAR sweeps per frame (about 10x the points), ten classes. Model is CenterPoint on a PointPillars backbone. Its baseline runs at about 2.6 frames per second versus KITTI’s 9 to 11, because each frame already spends real time doing math.

Same lever on both: **frame batching**, collating several frames into one forward pass so per-frame launch cost is divided by the batch size. In eval mode the normalization is fixed and convolutions act per sample, so batching does not mix frames; it only shares the launches.

|Batch|KITTI          |nuScenes                  |
|-----|---------------|--------------------------|
|2    |+37.3%         |kept baseline (not faster)|
|4    |+63.2%         |+7.1%                     |
|8    |+174.9%        |+23.2%                    |
|16   |+275.4%        |+34.0%                    |
|32   |+366.0%        |+37.3%                    |
|64   |**+427.0%**    |+25.8%                    |
|128  |**rolled back**|+34.0%                    |
|256+ |OOM            |OOM                       |

A note before reading this table as serving numbers: in deployment the same gate that verifies correctness also enforces the workload’s latency budget, so the sweep shows throughput headroom under this workload’s offline tolerance, not a real-time serving prescription.

KITTI tops out at +427%. nuScenes tops out around +37%. Nobody tuned the lever differently. The runtime read a heavier, more compute-bound workload off the trace and the physics did the rest.

**We did not choose batching because it is clever. We chose it because the trace ruled out everything else.** Batching is a known, mundane optimization; we did not invent it and we are not claiming to. What the runtime did was read the launch-bound signature off the trace and select batching *without being told*, sweep the batch dimension, measure where the headroom ran out per workload, and then **refuse it at KITTI B=128** when the accumulation order and cuDNN algorithm choice drifted a borderline detection past tolerance. The lever is boring on purpose. A person can pick it once. The runtime picks it automatically, per workload, across a fleet that no one has time to profile by hand, and knows when to stop. That is the whole difference between a tune and a substrate.

That B=128 rollback sits in the table as `rolled back`, not quietly dropped. A results table with no failures in it is a marketing table, not a benchmark.

## How We Keep Ourselves Honest

**We gate on correctness, not just speed.** For the deterministic HFT track that means byte-identical output. For the neural detectors, where inference across a different batch shape is never bit-exact, the gate today checks that per-frame detection counts match and sorted confidence scores agree within a small window. We will say what that is not: it is not yet a preserved-mAP (KITTI) or preserved-NDS (nuScenes) guarantee. Hardening the detector gate to a task-accuracy invariant is on the list, and where we have those deltas we will publish them.

**We show the failures.** Rollbacks appear as `rolled back`. Out-of-memory batches appear as `OOM`. Nothing is omitted to make the table cleaner.

**We report best-of-N, and we publish the spread.** Best-of-N isolates a systems change from scheduler jitter, which is standard, but it is also the number most worth auditing, so we publish per-run variance alongside the reproducibility artifacts rather than asking you to trust the best sample.

**We name the model, the hardware, and the config.** Pinned checkpoints, pinned configs, named GPUs, seeded data. The commands below regenerate every number.

## What This Does Not Yet Prove, and What’s Next

Two honest gaps.

First, the compute-bound case. Every track above is launch- or scan-bound, which is where amortizing work pays off. What does the runtime do when the GPU is genuinely busy with arithmetic? The answer should be a completely different lever, and we are running that now on OpenFold (the open-source AlphaFold2 reimplementation), whose Evoformer is dense attention and large matmuls. Frame batching would buy nothing there; the right lever is **bf16 precision**, gated on median plDDT holding within tolerance. We are deliberately not publishing a number until it clears the same gated A/B the others did. An early prediction put it near +17% and an earlier ad-hoc run came back much higher, and rather than pick the flattering one we are re-measuring honestly first.

Second, the commercial edge. None of the four tracks above is an inference-serving workload, and our wedge is inference serving: helping providers get more sellable throughput from the GPUs they already own. The same gated loop is now pointed there. The inference-serving benchmark applies the identical gate to vLLM and SGLang decode, measured in tokens per second under p95 and p99 latency constraints, with named model, GPU count, and request mix, before and after. Inference providers run thousands of decode jobs that drift constantly as models, quantization, and traffic change, which is exactly the condition where a continuous runtime loop compounds and hand-tuning cannot keep up. This post shows the loop working across bottleneck types. The next applies it to the workload we sell into.

## Try It Yourself

Every result regenerates from a single command per track. The HFT A/B, on a dataset you regenerate deterministically from a seed:

```sh
python -m gitm.benchmarks.hft.generate --events 1_000_000_000 --seed 42 --out ./hft_1b_seed42
gitm run --workload hft --optimize --seed 42 --stage ./hft_1b_seed42 --max-events 400_000_000
```

The KITTI batch sweep:

```sh
for B in 1 2 4 8 16 32 64 128 256 512; do
  AB=$(( B < 64 ? 64 : B ))
  GITM_EDGE_CKPT=.../pointpillar_7728.pth \
  GITM_EDGE_CFG=.../pointpillar.yaml \
  GITM_EDGE_FRAMES=$AB GITM_EDGE_AB_FRAMES=$AB GITM_EDGE_BATCH_SIZE=$B \
  python -m gitm.cli run --workload kitti --report kitti_b$B.md
done
```

**Want this run against your own workload?** Point us at one GPU job and we will hand back a headroom assessment: its bottleneck classification (launch-bound, compute-bound, or serialized), the serialized-concurrency reading, the levers we tried, which ones passed the gate and which rolled back, and the verified before/after. You give us the workload and read access to run it; we give you a number that is a proven A/B or does not exist. Turnaround is days, not weeks, and it costs you nothing.

-----

*Thanks to the OpenPCDet and OpenFold projects, and to the KITTI and nuScenes teams, whose open models and datasets made these benchmarks reproducible.*
