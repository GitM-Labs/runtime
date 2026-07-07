# Trace-capture overhead

**Target:** < 10 %, tightening to < 5 % after the buffered/async-I/O pass.

## Methodology

[`measure_overhead.py`](measure_overhead.py) times a workload N times **without**
instrumentation and N times **with** `gitm.tracer.capture(...)` wrapping it, then
reports the mean wall-clock overhead the capture path adds:

```
overhead = (mean_instrumented - mean_baseline) / mean_baseline
```

On a GPU box, point `--trace-dir` at scratch and swap the synthetic workload for
the real decode loop (100 steps, per the ticket):

```bash
python -m benchmarks.overhead.measure_overhead --runs 3 --steps 100 \
    --trace-dir $GITM_SCRATCH/traces
```

The instrumented runs then exercise the live CUPTI tracer (built via
`python -m gitm.tracer._cupti.build`) and write a real trace per run.

## Results

| Host | Tracer | Baseline | Instrumented | Overhead |
| --- | --- | --- | --- | --- |
| CPU laptop (no GPU) | no-op | ~8.5 ms | ~7.3 ms | within noise of 0 % |
| A100 dev box | CUPTI | _TBD_ | _TBD_ | _TBD_ (target <10%) |

**CPU note.** Without the CUPTI shim built, `capture()` is a well-formed no-op,
so the only cost is the context-manager + empty-trace write — below timing noise
(runs here swing ±15 % from CPU jitter alone). This row establishes the *floor of
the method*, not the GPU overhead. The headline A100 numbers must be filled
in on the dev box before sign-off.

## Overhead reduction (roadmap)

If the measurement exceeds 5 %, the hot path is the synchronous per-event
JSONL write in [`capture.py`](../../gitm/tracer/capture.py). The planned fix is
buffered/async I/O: accumulate events in the C shim (already done — records are
batched and only materialized on `stop()`), and write the JSONL on a background
thread. Re-run this script to confirm < 5 %.
