# Kimi MI355X — collector bring-up (checked)

Command-checked procedure to stand up capture on the Kimi loop, verify required
signals, identify collector/backend and provisional signal-contract version, and
tear down. Every step is **command → expected result**. No eyeball-only checks.

Companion to [`kimi_mi355x_loop_runbook.md`](kimi_mi355x_loop_runbook.md). Trust
boundaries: [`run_qualification_path.md`](run_qualification_path.md).

Provisional contract: `signal_contract.v0` (Nathan stub until Aravinth versions
the real contract). On a laptop with a wheel installed:

```bash
python -c "from gitm.optimizer.deployment_spec import PROVISIONAL_SIGNAL_CONTRACT_VERSION as v; print(v)"
```

Expected: `v0`.

---

## 0. Prerequisites

| Command | Expected |
|---|---|
| `kubectl config current-context` | `us-mi355x-gitmachine` (or the documented MI355X context) |
| `kubectl get secret hf-token-secret -o name` | `secret/hf-token-secret` |
| `ls /mnt/shared/hf-cache/hub 2>/dev/null \|\| kubectl exec deploy/rex-stage -- ls /mnt/shared/hf-cache/hub \| head` | Directory listing including Kimi / moonshotai cache entries |

---

## 1. Deploy

```bash
kubectl config use-context us-mi355x-gitmachine
python3 -m build --wheel
kubectl cp dist/gitm_labs-*.whl default/rex-stage:/mnt/shared/gitm/wheel/
tar -C scripts/kimi_loop -cf - . | kubectl exec -i rex-stage -- tar -C /mnt/shared/gitm/scripts -xf -
kubectl apply -f deploy/k8s/mi355x-kimi-loop.yaml
kubectl get pods -l app=kimi-loop -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.phase}{" "}{range .status.containerStatuses[*]}{.name}={.ready}{" "}{end}{"\n"}{end}'
```

| Check | Expected |
|---|---|
| Pod phase | `Running` |
| Container ready | `vllm=true gitm=true` (or names as in YAML) → effectively `2/2` |
| Wait loop | `kubectl wait --for=condition=Ready pod -l app=kimi-loop --timeout=7200s` exits `0` |

If stuck `Pending` / SchedulingGated: `kubectl get provisioningrequests` — capacity, not collector failure.

---

## 2. Engine identity

```bash
POD=$(kubectl get pod -l app=kimi-loop -o jsonpath='{.items[0].metadata.name}')
kubectl get pod "$POD" -o jsonpath='{.status.containerStatuses[?(@.name=="vllm")].imageID}{"\n"}'
kubectl exec "$POD" -c gitm -- bash -c 'cat /opt/rocm/.info/version; python -c "import vllm; print(vllm.__version__)"'
kubectl exec "$POD" -c gitm -- curl -sf http://localhost:8000/health
```

| Check | Expected |
|---|---|
| `imageID` | Digests matching the pin in `deploy/k8s/mi355x-kimi-loop.yaml` (compare strings) |
| ROCm version | Non-empty version file contents |
| vLLM version | Non-empty version string |
| `/health` | HTTP success (exit 0), empty or JSON body |

Record these three strings; they are inputs to `deployment_spec` / MANIFEST comparison.

---

## 3. Arm C injection (capture path live)

```bash
kubectl exec -it deploy/kimi-k25-loop -c gitm -- bash /mnt/shared/gitm/scripts/arm.sh C
kubectl exec deploy/kimi-k25-loop -c gitm -- cat /scratch/arm.env
kubectl exec deploy/kimi-k25-loop -c gitm -- bash -c 'test -f /scratch/lib/libgitm_rocm_inject.so && echo INJECT_OK'
```

| Check | Expected |
|---|---|
| `arm.sh` | Ends with `arm C serving` (or `already active and serving`) |
| `GITM_ARM` | `C` |
| `ROCP_TOOL_LIBRARIES` | Path containing `libgitm_rocm_inject.so` (non-empty) |
| `GITM_TRACE_NVTX` | `1` |
| `GITM_EXTRA_VLLM_ARGS` | Contains `--enable-layerwise-nvtx-tracing` |
| Inject library | `INJECT_OK` |

Backend identity for this loop: **rocprofiler inject** (`source=rocprof` on merged traces), not `gitm.telemetry.Collector`.

---

## 4. Capture live (E0-scale)

Inside the gitm sidecar (or via `kubectl exec`):

```bash
export GITM_RUN=$(date -u +%Y%m%d-%H%M%S)
export RUN=/mnt/shared/gitm/results/$GITM_RUN
mkdir -p "$RUN/e0"
python -m gitm.cli capture attach --port 8000 --requests 8 --concurrency 2 \
  --input-tokens 256 --output-tokens 64 --out "$RUN/e0/capture"
python - "$RUN/e0/capture" <<'PY'
import json, glob, sys
d = sys.argv[1]
traces = sorted(glob.glob(d + "/**/*.jsonl", recursive=True))
assert traces, "FAIL: no merged trace"
ev = [json.loads(l) for l in open(traces[-1]) if l.strip()]
kernels = [e for e in ev if e.get("kind") == "kernel"]
meta = [e for e in ev if e.get("kind") == "meta"]
header = next((e.get("_header") for e in ev if isinstance(e, dict) and "_header" in e), None)
# header may be first line only
if header is None and ev:
    pass
dropped = sum(m.get("dropped_records", 0) for m in meta if isinstance(m, dict))
named = [k for k in kernels if k.get("name") and "unknown" not in str(k.get("name")).lower()]
print(f"trace={traces[-1]}")
print(f"kernels={len(kernels)} named={len(named)} dropped={dropped}")
assert kernels, "FAIL: zero kernels"
assert named, "FAIL: anonymous kernels"
assert dropped == 0, f"FAIL: dropped_records={dropped}"
print("CAPTURE_LIVE_OK")
PY
```

| Check | Expected |
|---|---|
| Merged JSONL | At least one `**/*.jsonl` under `$RUN/e0/capture` |
| Kernels | `kernels>0` |
| Names | `named>0` |
| Drops | `dropped=0` |
| Final line | `CAPTURE_LIVE_OK` |

Optional header identity:

```bash
python - <<PY
import json, glob, os
d = os.environ["RUN"] + "/e0/capture"
p = sorted(glob.glob(d + "/**/*.jsonl", recursive=True))[-1]
h = json.loads(open(p).readline()).get("_header", {})
print("source=", h.get("source"))
print("workload_id=", h.get("workload_id"))
print("run_id=", h.get("run_id"))
print("device_count=", h.get("device_count"))
PY
```

| Check | Expected |
|---|---|
| `source` | `rocprof` (injected ROCm path) |
| `workload_id` | `vllm-attach` (what `capture attach` writes; example deployment_spec matches) |

---

## 5. Expected sources present / absent

Still in sidecar after a capture (or after `run_loop.sh e0`):

```bash
# Required for capture integrity
test -s /scratch/trace/kimi.jsonl.* 2>/dev/null || ls /scratch/trace/ | head
ls "$RUN/e0/capture"/**/*.jsonl 2>/dev/null | head -1
test -f /scratch/telemetry/amdsmi.jsonl && echo AMDSMI_PRESENT || echo AMDSMI_ABSENT

# Contract checklist
python - <<'PY'
from pathlib import Path
import os, glob
run = Path(os.environ["RUN"])
required = {
    "merged_trace": bool(glob.glob(str(run / "e0/capture/**/*.jsonl"), recursive=True)),
    "amdsmi": Path("/scratch/telemetry/amdsmi.jsonl").is_file(),
}
for k, ok in required.items():
    print(f"{k}={'PRESENT' if ok else 'ABSENT'}")
missing = [k for k, ok in required.items() if not ok]
print("SOURCES_OK" if not missing else "SOURCES_MISSING " + ",".join(missing))
PY
```

| Source | Expected for arm-C E0 |
|---|---|
| Merged capture JSONL | `PRESENT` |
| Inject shards under `/scratch/trace/` | May be cleared after merge; absence after successful merge is OK if merged JSONL exists |
| `amdsmi.jsonl` | `PRESENT` (always-on in pod). If `ABSENT`, record as missing state-plane source |
| `gitm.telemetry.Collector` Prometheus/OTLP | **Not required** for this loop (absent is OK) |

---

## 6. Contract + collector identity stamp

```bash
python - <<'PY'
from gitm.optimizer.deployment_spec import (
    PROVISIONAL_SIGNAL_CONTRACT_VERSION,
    default_signal_contract,
)
c = default_signal_contract()
print("signal_contract_version=", PROVISIONAL_SIGNAL_CONTRACT_VERSION)
print("backend=", c.capture_backend)
print("required_sources=", ",".join(c.required_sources))
PY
```

| Check | Expected |
|---|---|
| `signal_contract_version` | `v0` |
| `backend` | `rocprof-inject` |
| Printed sources | Includes `merged_trace` |

If these prints fail with `ImportError`, the sidecar wheel is stale — re-copy wheel from step 1.

Qualify the capture once `deployment_spec.json` is prepared (see `gitm.optimizer.qualify_run`):

```bash
python -m gitm.optimizer.qualify_run \
  --artifacts "$RUN/e0/capture" \
  --deployment-spec /path/to/deployment_spec.json \
  --signal-contract-version v0
```

| Check | Expected |
|---|---|
| JSON `verdict` | `qualified` for a good E0; otherwise named failed/unknown checks |
| Exit code | `0` only when `qualified` |

---

## 7. Tear down / leave idle

| Goal | Command | Expected |
|---|---|---|
| Leave pod warm for more runs | `exit` sidecar only | Deployment still `Running 2/2` |
| Stop capture arm (clean numbers) | `bash /mnt/shared/gitm/scripts/arm.sh A` | `arm A serving`; `ROCP_TOOL_LIBRARIES` empty |
| Delete deployment | `kubectl delete -f deploy/k8s/mi355x-kimi-loop.yaml` | Pods terminate; **do not** delete `/mnt/shared/gitm/results/` or HF cache |
| Confirm results preserved | `ls /mnt/shared/gitm/results/$GITM_RUN` | MANIFEST / e0 capture still present |

Do **not** `rm -rf /mnt/shared/hf-cache` or wipe `/mnt/shared/gitm/results` as part of collector tear-down.
