# Real profiler fixtures

All files below are **unchanged** copies of upstream artifacts from
[pytorch/kineto](https://github.com/pytorch/kineto).

| File | Origin | Ref / commit | Notes |
|------|--------|--------------|-------|
| `kineto_gpu_metrics_input.json` | `tb_plugin/test/gpu_metrics_input.json` | tag `v0.4.0` (`486c083`) | Small chrome-trace with `Kernel`/`Memcpy` cats; single device |
| `kineto_resnet50_workers0.pt.trace.json.gz` | `tb_plugin/samples/resnet50_num_workers_0/worker0.1623143089861.pt.trace.json.gz` | tag `v0.4.0` (`486c083`) | Real torch.profiler export (~1.1 MiB gz, ~75k events) |
| `kineto_resnet50_workers4.pt.trace.json.gz` | `tb_plugin/samples/resnet50_num_workers_4/worker0.1623212756351.pt.trace.json.gz` | tag `v0.4.0` (`486c083`) | Real torch.profiler export with dataloader workers |
| `synthetic_4xA100_nccl.json` | **synthetic** (see `../generate_fixtures.py`) | n/a | 4-device chrome-trace with interleaved NCCL kernels, modeled on the event shape of the resnet samples (`cat: Kernel/Memcpy`, args keys `device`/`stream`/`correlation`/`grid`/`block`/`shared memory`/`registers per thread`/`bytes`). Marked synthetic because no multi-GPU sample shipped in kineto v0.4.0. |

Fetch recipe (reproducible):

```bash
REF=v0.4.0
BASE=https://raw.githubusercontent.com/pytorch/kineto/${REF}
curl -L -o kineto_gpu_metrics_input.json \
  ${BASE}/tb_plugin/test/gpu_metrics_input.json
curl -L -o kineto_resnet50_workers0.pt.trace.json.gz \
  ${BASE}/tb_plugin/samples/resnet50_num_workers_0/worker0.1623143089861.pt.trace.json.gz
curl -L -o kineto_resnet50_workers4.pt.trace.json.gz \
  ${BASE}/tb_plugin/samples/resnet50_num_workers_4/worker0.1623212756351.pt.trace.json.gz
```

None of the real traces contain >1 CUDA device; multi-GPU coverage uses
`synthetic_4xA100_nccl.json`.
