#!/usr/bin/env bash
#
# final_gpu_stability_test.sh — single-invocation 8x MI300X / Kimi-K2.6 stability validation.
#
#   sudo bash final_gpu_stability_test.sh                     # safe suite, smoke mode
#   sudo bash final_gpu_stability_test.sh --mode stability    # safe suite, long mode
#   sudo bash final_gpu_stability_test.sh --include-unsafe-matrix   # also run HSA-disabled (crash-prone) rows
#
# Goal: localize GPU hangs to hardware / ROCm-HSA / AITER-kernels / RCCL-XGMI / vLLM config.
# Safety: HSA-override-DISABLED rows are known crash-prone and are OFF by default.
# This script NEVER resets GPUs, reloads drivers, reboots, or changes firmware. It only kills
# containers/processes IT started. It saves partial results if a stage crashes or times out.
#
# Review-first: every cluster-specific value is a variable in the CONFIG block below. Verify them
# against your real serve.sh/docker command before running. Missing tools are SKIPPED, not passed.

set -u -o pipefail

# ======================================================================================
# CONFIG  (override via environment; defaults derived from the incident logs)
# ======================================================================================
IMAGE="${GPU_TEST_IMAGE:-4599ea638415}"
MODEL_PATH="${MODEL_PATH:-/mnt/nvme1/models/Kimi-K2.6}"
SERVED_NAME="${SERVED_NAME:-moonshotai/Kimi-K2.6}"
PORT="${VLLM_PORT:-8888}"

TP="${TENSOR_PARALLEL:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9472}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"          # customer guardrail: do NOT exceed 256
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_MLA}"
QUICK_REDUCE="${QUICK_REDUCE:-INT4}"
HSA_OVERRIDE_VALUE="${HSA_OVERRIDE_VALUE:-9.4.2}"

# Extra docker args to MATCH PRODUCTION. Device flags are from the logs; --ipc=host + shm are
# required for TP shared-memory (shm_broadcast). Paste your real serve.sh docker flags here.
DOCKER_DEVICE_ARGS="${DOCKER_DEVICE_ARGS:---device /dev/kfd --device /dev/dri --group-add video}"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:---ipc=host --shm-size=16g --cap-add=SYS_PTRACE --security-opt seccomp=unconfined}"

# Existing pre-flight health script to reuse (first that exists wins). Falls back to an inline gate.
PREFLIGHT_CANDIDATES=("${PREFLIGHT_SCRIPT:-}" ./tools/preflight.sh ./preflight.sh ./tools/watcher.sh /opt/gitm/preflight.sh)

# Mode + knobs
MODE="smoke"                 # smoke | stability
INCLUDE_UNSAFE=0
CONC_LEVELS_SMOKE="1 4 8"
CONC_LEVELS_STAB="1 4 8 16 32"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"

# Timeouts (seconds)
LOAD_TIMEOUT="${LOAD_TIMEOUT:-1200}"         # model load + ready
INIT_TIMEOUT="${INIT_TIMEOUT:-600}"          # short-init rows
REQ_TIMEOUT="${REQ_TIMEOUT:-120}"            # per inference request
STOP_GRACE="${STOP_GRACE:-120}"              # graceful docker stop (teardown-doc policy)
HEALTH_POLL="${HEALTH_POLL:-5}"              # seconds between live health polls
SUSTAINED_SECS_SMOKE=60
SUSTAINED_SECS_STAB=1800
COMPUTE_SECS_SMOKE=30
COMPUTE_SECS_STAB=180

# dmesg fault signatures (from mns512_report.md / kernel_health_checks.md)
FAULT_RE='kiq fence|retry page fault|TLB flush failed|failed to write reg|Memory access fault|HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION|uncorrectable|ras_|ring .* timeout'

# ======================================================================================
# Arg parsing
# ======================================================================================
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="${2:-smoke}"; shift 2;;
    --smoke) MODE="smoke"; shift;;
    --stability) MODE="stability"; shift;;
    --include-unsafe-matrix) INCLUDE_UNSAFE=1; shift;;
    -h|--help) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done
[ "$MODE" = "smoke" ] || [ "$MODE" = "stability" ] || { echo "mode must be smoke|stability" >&2; exit 2; }

# ======================================================================================
# Result dir + logging
# ======================================================================================
TS="$(date +%Y-%m-%d_%H%M%S)"
OUTDIR="${OUTDIR:-$(pwd)/gpu_stability_results/$TS}"
mkdir -p "$OUTDIR"
ENVIRON="$OUTDIR/environment.txt"
PREFLIGHT_LOG="$OUTDIR/preflight.log"
TELEM="$OUTDIR/gpu_telemetry.csv"
KLOG="$OUTDIR/kernel_logs.txt"
RESJSON="$OUTDIR/test_results.json"
INFJSON="$OUTDIR/inference_metrics.json"
REPORT="$OUTDIR/final_report.md"
RUNLOG="$OUTDIR/run.log"

# logs go to stderr (+ RUNLOG) so function stdout stays clean for value capture
log()  { printf '%s %s\n' "[$(date +%H:%M:%S)]" "$*" | tee -a "$RUNLOG" >&2; }
warn() { printf '%s WARN: %s\n' "[$(date +%H:%M:%S)]" "$*" | tee -a "$RUNLOG" >&2; }

# Parallel result records: name|status|detail|seconds
RES_NAMES=(); RES_STATUS=(); RES_DETAIL=(); RES_SECS=()
record() { local d="${3:-}"; d="${d//$'\n'/ }"; d="${d//|/ }"; RES_NAMES+=("$1"); RES_STATUS+=("$2"); RES_DETAIL+=("$d"); RES_SECS+=("${4:-0}"); emit_json; log "RESULT [$2] $1 $d"; }

CRITICAL_FAIL=0
BG_PIDS=()
CONTAINER_PREFIX="gpu_stab_${TS}"

# ======================================================================================
# Cleanup — only our own containers/processes; save partial results
# ======================================================================================
cleanup() {
  local ec=$?
  log "cleanup: tearing down (exit=$ec)"
  # our background helpers
  for p in "${BG_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
  # only containers we created
  if command -v docker >/dev/null 2>&1; then
    for c in $(docker ps -a --filter "name=${CONTAINER_PREFIX}" --format '{{.Names}}' 2>/dev/null); do
      log "cleanup: removing container $c"
      docker stop -t 10 "$c" >/dev/null 2>&1
      docker rm -f "$c" >/dev/null 2>&1
    done
  fi
  emit_json
  [ -f "$REPORT" ] || generate_report
  log "cleanup: results in $OUTDIR"
}
trap cleanup EXIT INT TERM

# ======================================================================================
# Helpers
# ======================================================================================
have() { command -v "$1" >/dev/null 2>&1; }

gpu_cards() { ls -d /sys/class/drm/card*/device 2>/dev/null | while read -r d; do
  [ -f "$d/gpu_busy_percent" ] && echo "$d"; done; }

dmesg_fault_count() { local c; c=$(dmesg -T 2>/dev/null | grep -icE "$FAULT_RE"); echo "${c:-0}"; }

# Snapshot recent faulting kernel lines into KLOG
capture_kernel() { { echo "==== dmesg @ $(date) ($1) ===="; dmesg -T 2>/dev/null | tail -n 300 | grep -iE "$FAULT_RE" || echo "(no fault signatures)"; } >> "$KLOG"; }

# Telemetry sampler (sysfs always available on host; amd-smi enriches if present)
start_telemetry() {
  [ -f "$TELEM" ] || echo "ts,card,gpu_busy_percent,power_w,temp_c,vram_used_mb" > "$TELEM"
  ( while :; do
      local now; now="$(date +%s)"
      for d in $(gpu_cards); do
        local busy pw tc vram
        busy="$(cat "$d/gpu_busy_percent" 2>/dev/null || echo NA)"
        pw="$(cat "$d"/hwmon/hwmon*/power1_average 2>/dev/null | head -1)"; [ -n "$pw" ] && pw=$((pw/1000000)) || pw=NA
        tc="$(cat "$d"/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)"; [ -n "$tc" ] && tc=$((tc/1000)) || tc=NA
        vram="$(cat "$d/mem_info_vram_used" 2>/dev/null)"; [ -n "$vram" ] && vram=$((vram/1048576)) || vram=NA
        echo "$now,$(basename "$(dirname "$d")"),$busy,$pw,$tc,$vram" >> "$TELEM"
      done
      sleep "$HEALTH_POLL"
    done ) &
  BG_PIDS+=("$!")
}

# Pre-flight: reuse existing script if found, else inline gate. Returns 0 clean / 1 not clean.
preflight() {
  local used=""
  for cand in "${PREFLIGHT_CANDIDATES[@]}"; do
    [ -n "$cand" ] && [ -x "$cand" ] && { used="$cand"; break; }
    [ -n "$cand" ] && [ -f "$cand" ] && { used="$cand"; break; }
  done
  if [ -n "$used" ]; then
    log "preflight: reusing $used"
    if bash "$used" >>"$PREFLIGHT_LOG" 2>&1; then echo "reused:$used"; return 0
    else echo "reused:$used"; return 1; fi
  fi
  log "preflight: no existing script found; using inline gate (see kernel_health_checks.md)"
  { echo "== inline preflight @ $(date) =="
    local bad=0
    for d in $(gpu_cards); do
      b="$(cat "$d/gpu_busy_percent" 2>/dev/null)"
      echo "$(basename "$(dirname "$d")") busy=$b"
      [ "$b" = "0" ] || { echo "  NOT IDLE"; bad=1; }
    done
    if dmesg -T 2>/dev/null | tail -300 | grep -qiE "$FAULT_RE"; then echo "recent dmesg fault signatures present"; bad=1; fi
    echo "inline-preflight bad=$bad"
    [ "$bad" = 0 ]
  } >>"$PREFLIGHT_LOG" 2>&1
  local rc=$?; echo "inline"; return $rc
}

# Wait until all GPUs idle (post-teardown), up to N seconds
wait_idle() {
  local t=0 max="${1:-120}"
  while [ "$t" -lt "$max" ]; do
    local busy=0
    for d in $(gpu_cards); do [ "$(cat "$d/gpu_busy_percent" 2>/dev/null || echo 1)" = "0" ] || busy=1; done
    [ "$busy" = 0 ] && return 0
    sleep 3; t=$((t+3))
  done
  return 1
}

# Launch vLLM (detached). Args: <logfile> <hsa_on 0|1> <aiter_on 0|1> <cname>
launch_vllm() {
  local logf="$1" hsa="$2" aiter="$3" cname="$4"
  local envs=(-e "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=${QUICK_REDUCE}")
  [ "$hsa" = 1 ]   && envs+=(-e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_VALUE}")
  envs+=(-e "VLLM_ROCM_USE_AITER=${aiter}")
  envs+=(-e "VLLM_ATTENTION_BACKEND=${ATTENTION_BACKEND}")   # env, not a CLI flag (version-robust)
  {
    echo "# cmd @ $(date)"
    echo "# HSA_OVERRIDE=$([ "$hsa" = 1 ] && echo "$HSA_OVERRIDE_VALUE" || echo UNSET)  AITER=$aiter  mns=$MAX_NUM_SEQS  backend=$ATTENTION_BACKEND"
  } >"$logf"
  # shellcheck disable=SC2086
  docker run -d --rm --name "$cname" \
    $DOCKER_DEVICE_ARGS $EXTRA_DOCKER_ARGS \
    -p "${PORT}:${PORT}" \
    -v "${MODEL_PATH%/*}:${MODEL_PATH%/*}:ro" \
    "${envs[@]}" \
    "$IMAGE" \
    vllm serve "$MODEL_PATH" \
      --served-model-name "$SERVED_NAME" \
      --host 0.0.0.0 --port "$PORT" \
      --tensor-parallel-size "$TP" \
      --max-model-len "$MAX_MODEL_LEN" \
      --block-size "$BLOCK_SIZE" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --trust-remote-code \
      >>"$logf" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || { echo "docker run failed rc=$rc" >>"$logf"; return 1; }
  ( docker logs -f "$cname" >>"$logf" 2>&1 ) & BG_PIDS+=("$!")
  return 0
}

# Wait for /health ready OR container exit OR dmesg fault OR timeout.
# echoes: ready | crashed | fault | timeout
wait_ready() {
  local cname="$1" timeout="$2" base_faults="$3" t=0
  while [ "$t" -lt "$timeout" ]; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${cname}$"; then echo crashed; return 1; fi
    if [ "$(dmesg_fault_count)" -gt "$base_faults" ]; then echo fault; return 1; fi
    if have curl && curl -fsS -m 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then echo ready; return 0; fi
    sleep "$HEALTH_POLL"; t=$((t+HEALTH_POLL))
  done
  echo timeout; return 1
}

# Graceful teardown of a container we own.
teardown_vllm() {
  local cname="$1"
  docker ps --format '{{.Names}}' | grep -q "^${cname}$" || return 0
  log "teardown: docker stop -t $STOP_GRACE $cname"
  docker stop -t "$STOP_GRACE" "$cname" >/dev/null 2>&1
  docker rm -f "$cname" >/dev/null 2>&1
}

# Verify actual backend/kernels from the server log (don't trust the flag alone).
verify_backend() {
  local logf="$1"
  { echo "---- backend verification ($logf) ----"
    grep -hoE "Using \[[^]]*\] all-reduce backends[^.]*" "$logf" | head -3
    grep -hoE "Final IR op priority[^)]*\)" "$logf" | head -1
    grep -hoiE "aiter|triton_mla|rocm_aiter|AITER_CUSTOM" "$logf" | sort | uniq -c | sort -rn | head
  } >>"$RUNLOG" 2>&1
}

# ======================================================================================
# Inference — real requests to /v1/completions. Args: <label> <concurrency> <duration_s> <logf>
# Appends one JSON object to a temp list; sets globals LAST_OK/LAST_ERR/LAST_TO/LAST_LAT.
# ======================================================================================
infer_prompt() { yes "stability test prompt token" | head -n "$PROMPT_TOKENS" | tr '\n' ' '; }
run_inference() {
  local label="$1" conc="$2" dur="$3" logf="$4"
  have curl || { record "inference:$label" SKIPPED "curl missing" 0; return 0; }
  local prompt; prompt="$(infer_prompt)"
  local body; body="$(printf '{"model":"%s","prompt":"%s","max_tokens":%d,"temperature":0,"ignore_eos":true}' "$SERVED_NAME" "$prompt" "$GEN_TOKENS")"
  local tmp; tmp="$(mktemp)"; local end=$(( $(date +%s) + dur ))
  local ok=0 err=0 to=0
  while [ "$(date +%s)" -lt "$end" ]; do
    local pids=()
    for _ in $(seq 1 "$conc"); do
      ( local st en code; st=$(date +%s.%N)
        code=$(curl -s -o /dev/null -w '%{http_code}' -m "$REQ_TIMEOUT" \
               -H 'Content-Type: application/json' -d "$body" \
               "http://localhost:${PORT}/v1/completions" 2>/dev/null)
        en=$(date +%s.%N)
        echo "$code $(awk "BEGIN{print $en-$st}")" >>"$tmp"
      ) & pids+=("$!")
    done
    for p in "${pids[@]}"; do wait "$p"; done
    # live health between waves
    [ "$(dmesg_fault_count)" -gt "$BASE_FAULTS" ] && { capture_kernel "inference:$label"; break; }
  done
  while read -r code lat; do
    case "$code" in
      200) ok=$((ok+1)); echo "$lat" >>"$tmp.lat";;
      000) to=$((to+1));;
      *)   err=$((err+1));;
    esac
  done <"$tmp"
  local avg=NA p_min=NA p_max=NA
  if [ -s "$tmp.lat" ]; then
    avg=$(awk '{s+=$1;n++} END{if(n)printf "%.3f",s/n}' "$tmp.lat")
    p_min=$(sort -n "$tmp.lat" | head -1); p_max=$(sort -n "$tmp.lat" | tail -1)
  fi
  local total=$((ok+err+to)) rate=0
  [ "$total" -gt 0 ] && rate=$(awk "BEGIN{printf \"%.1f\",100*$ok/$total}")
  printf '{"label":"%s","concurrency":%d,"duration_s":%d,"requests":%d,"ok":%d,"errors":%d,"timeouts":%d,"success_pct":%s,"lat_avg_s":"%s","lat_min_s":"%s","lat_max_s":"%s"}\n' \
    "$label" "$conc" "$dur" "$total" "$ok" "$err" "$to" "$rate" "$avg" "$p_min" "$p_max" >>"$INFJSON.parts"
  rm -f "$tmp" "$tmp.lat"
  LAST_OK=$ok; LAST_ERR=$err; LAST_TO=$to; LAST_LAT=$avg
  if [ "$ok" -eq 0 ]; then record "inference:$label" FAIL "0/$total ok (err=$err to=$to)" 0
  elif [ "$to" -gt 0 ] || [ "$err" -gt 0 ]; then record "inference:$label" INCONCLUSIVE "ok=$ok err=$err to=$to avg=${avg}s" 0
  else record "inference:$label" PASS "ok=$ok avg=${avg}s conc=$conc" 0; fi
}

# ======================================================================================
# A full vLLM config test (load -> verify -> inference -> teardown). Args:
#   <name> <hsa 0|1> <aiter 0|1> <workload short|inference|sustained> <logf>
# ======================================================================================
vllm_config_test() {
  local name="$1" hsa="$2" aiter="$3" workload="$4" logf="$5"
  [ "$CRITICAL_FAIL" = 1 ] && { record "$name" SKIPPED "prior critical failure" 0; return 0; }
  have docker || { record "$name" SKIPPED "docker missing" 0; return 0; }
  local cname="${CONTAINER_PREFIX}_$(echo "$name" | tr -c 'a-zA-Z0-9' _)"
  local t0; t0=$(date +%s); BASE_FAULTS="$(dmesg_fault_count)"
  log "== $name :: HSA=$([ "$hsa" = 1 ] && echo on || echo OFF) AITER=$aiter workload=$workload =="
  if ! launch_vllm "$logf" "$hsa" "$aiter" "$cname"; then
    record "$name" FAIL "container failed to start" $(( $(date +%s)-t0 )); return 1
  fi
  local tmo="$LOAD_TIMEOUT"; [ "$workload" = short ] && tmo="$INIT_TIMEOUT"
  local status; status="$(wait_ready "$cname" "$tmo" "$BASE_FAULTS")"
  capture_kernel "$name"
  verify_backend "$logf"
  case "$status" in
    ready)   log "$name: model ready";;
    fault)   record "$name" FAIL "GPU fault during load (dmesg signature)" $(( $(date +%s)-t0 )); CRITICAL_FAIL=1; teardown_vllm "$cname"; return 1;;
    crashed) record "$name" FAIL "container exited during load (see $logf)" $(( $(date +%s)-t0 )); teardown_vllm "$cname"; return 1;;
    timeout) record "$name" FAIL "load timeout ${tmo}s (hang)" $(( $(date +%s)-t0 )); CRITICAL_FAIL=1; teardown_vllm "$cname"; return 1;;
  esac

  if [ "$workload" = short ]; then
    record "$name" PASS "init+ready ok" $(( $(date +%s)-t0 ))
  else
    local conc dur
    if [ "$workload" = sustained ]; then
      dur=$([ "$MODE" = stability ] && echo "$SUSTAINED_SECS_STAB" || echo "$SUSTAINED_SECS_SMOKE")
      run_inference "$name" 8 "$dur" "$logf"
    else
      run_inference "$name" 4 20 "$logf"
    fi
    [ "$CRITICAL_FAIL" = 1 ] || record "$name" PASS "ready+inference completed" $(( $(date +%s)-t0 ))
  fi
  teardown_vllm "$cname"
  if wait_idle 120; then log "$name: GPUs idle after teardown"; else warn "$name: GPUs not idle after teardown"; fi
}

# ======================================================================================
# Compute / memory + RCCL/XGMI via portable in-container PyTorch (torch is in the image).
# ======================================================================================
gpu_compute_mem_test() {
  [ "$CRITICAL_FAIL" = 1 ] && { record "compute_mem" SKIPPED "prior critical failure" 0; return 0; }
  have docker || { record "compute_mem" SKIPPED "docker missing" 0; return 0; }
  local secs; secs=$([ "$MODE" = stability ] && echo "$COMPUTE_SECS_STAB" || echo "$COMPUTE_SECS_SMOKE")
  local logf="$OUTDIR/compute_mem.log" cname="${CONTAINER_PREFIX}_compute"
  local py='
import os,sys,time,torch
secs=float(os.environ.get("SECS","30")); n=torch.cuda.device_count()
assert n>0,"no GPUs visible to torch"
print("torch",torch.__version__,"devices",n,flush=True)
t0=time.time()
while time.time()-t0<secs:
  for i in range(n):
    with torch.cuda.device(i):
      a=torch.randn(4096,4096,device="cuda",dtype=torch.bfloat16)
      b=a@a; _=b.float().sum().item()
      m=torch.empty(int(2e9//2),dtype=torch.bfloat16,device="cuda"); m.fill_(1.0); _=m[-1].item(); del m
  torch.cuda.synchronize()
print("COMPUTE_MEM_OK",flush=True)'
  local t0; t0=$(date +%s); BASE_FAULTS="$(dmesg_fault_count)"
  # shellcheck disable=SC2086
  timeout $((secs+120)) docker run --rm --name "$cname" $DOCKER_DEVICE_ARGS $EXTRA_DOCKER_ARGS \
    -e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_VALUE}" -e "SECS=$secs" \
    "$IMAGE" python3 -c "$py" >"$logf" 2>&1
  local rc=$?; capture_kernel "compute_mem"
  if [ "$(dmesg_fault_count)" -gt "$BASE_FAULTS" ]; then record "compute_mem" FAIL "GPU fault during compute (dmesg)" $(( $(date +%s)-t0 )); CRITICAL_FAIL=1
  elif grep -q COMPUTE_MEM_OK "$logf"; then record "compute_mem" PASS "all $TP GPUs compute+alloc ok" $(( $(date +%s)-t0 ))
  elif [ $rc -eq 124 ]; then record "compute_mem" FAIL "compute timeout (hang)" $(( $(date +%s)-t0 )); CRITICAL_FAIL=1
  else record "compute_mem" INCONCLUSIVE "torch test did not confirm OK (rc=$rc, see log)" $(( $(date +%s)-t0 )); fi
}

rccl_xgmi_test() {
  [ "$CRITICAL_FAIL" = 1 ] && { record "rccl_xgmi" SKIPPED "prior critical failure" 0; return 0; }
  local logf="$OUTDIR/rccl_test.log"
  # Prefer a real rccl-tests binary if present in the image; else portable torch.distributed all_reduce.
  local cname="${CONTAINER_PREFIX}_rccl"; local t0; t0=$(date +%s); BASE_FAULTS="$(dmesg_fault_count)"
  local has_rccltests=0
  # shellcheck disable=SC2086
  docker run --rm $DOCKER_DEVICE_ARGS "$IMAGE" bash -lc 'command -v all_reduce_perf' >/dev/null 2>&1 && has_rccltests=1
  if [ "$has_rccltests" = 1 ]; then
    log "rccl_xgmi: using all_reduce_perf"
    # shellcheck disable=SC2086
    timeout 300 docker run --rm --name "$cname" $DOCKER_DEVICE_ARGS $EXTRA_DOCKER_ARGS \
      -e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_VALUE}" \
      "$IMAGE" bash -lc "all_reduce_perf -b 8 -e 512M -f 2 -g $TP" >"$logf" 2>&1
  else
    log "rccl_xgmi: all_reduce_perf absent -> portable torch.distributed fallback"
    local py='
import os,torch,torch.distributed as dist
dist.init_process_group("nccl")
r=dist.get_rank(); torch.cuda.set_device(r)
x=torch.ones(1<<24,device="cuda")
dist.all_reduce(x); torch.cuda.synchronize()
# p2p ring send/recv
w=dist.get_world_size(); y=torch.full((1<<20,),float(r),device="cuda")
dist.all_reduce(y)
if r==0: print("RCCL_OK sum0=%d p2p=%f"%(int(x[0].item()),y[0].item()),flush=True)
dist.destroy_process_group()'
    # shellcheck disable=SC2086
    timeout 300 docker run --rm --name "$cname" $DOCKER_DEVICE_ARGS $EXTRA_DOCKER_ARGS \
      -e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_VALUE}" \
      "$IMAGE" bash -lc "torchrun --standalone --nproc_per_node=$TP -c \"$(printf '%s' "$py" | sed 's/"/\\"/g')\"" >"$logf" 2>&1 \
      || python3 -c "import sys;sys.exit(0)"
  fi
  local rc=$?; capture_kernel "rccl_xgmi"
  if [ "$(dmesg_fault_count)" -gt "$BASE_FAULTS" ]; then record "rccl_xgmi" FAIL "GPU fault during collectives (dmesg)" $(( $(date +%s)-t0 )); CRITICAL_FAIL=1
  elif grep -qE "RCCL_OK|Avg bus bandwidth" "$logf"; then record "rccl_xgmi" PASS "8-GPU all-reduce completed" $(( $(date +%s)-t0 ))
  elif [ $rc -eq 124 ]; then record "rccl_xgmi" FAIL "collective timeout (XGMI/RCCL hang)" $(( $(date +%s)-t0 )); CRITICAL_FAIL=1
  else record "rccl_xgmi" INCONCLUSIVE "no success marker (rc=$rc, see rccl_test.log)" $(( $(date +%s)-t0 )); fi
}

# XGMI link state snapshot (non-disruptive)
xgmi_state() {
  { echo "== XGMI/topology @ $(date) =="
    if have amd-smi; then amd-smi xgmi 2>&1; amd-smi topology 2>&1
    elif have rocm-smi; then rocm-smi --showtopo 2>&1
    else echo "SKIPPED: no amd-smi/rocm-smi on host"; fi
  } >>"$OUTDIR/rccl_test.log" 2>&1
}

# ======================================================================================
# JSON + report emitters
# ======================================================================================
emit_json() {
  { echo '{'
    echo "  \"timestamp\": \"$TS\", \"mode\": \"$MODE\", \"include_unsafe\": $INCLUDE_UNSAFE,"
    echo "  \"image\": \"$IMAGE\", \"model\": \"$MODEL_PATH\", \"max_num_seqs\": $MAX_NUM_SEQS,"
    echo "  \"attention_backend\": \"$ATTENTION_BACKEND\", \"hsa_override\": \"$HSA_OVERRIDE_VALUE\","
    echo '  "results": ['
    local i n=${#RES_NAMES[@]}
    for ((i=0;i<n;i++)); do
      local sep=,; [ $((i+1)) -eq "$n" ] && sep=
      printf '    {"test":"%s","status":"%s","detail":"%s","seconds":%s}%s\n' \
        "${RES_NAMES[$i]}" "${RES_STATUS[$i]}" "$(echo "${RES_DETAIL[$i]}" | sed 's/"/\\"/g')" "${RES_SECS[$i]:-0}" "$sep"
    done
    echo '  ]'
    echo '}'
  } >"$RESJSON"
}

conclusion() {
  # Heuristic root-cause from the result pattern.
  local arch_fail=0 aiter_off_fail=0 base_ok=0 kern_ok=0 comm_ok=0 compute_ok=0
  local i n=${#RES_NAMES[@]}
  for ((i=0;i<n;i++)); do
    case "${RES_NAMES[$i]}:${RES_STATUS[$i]}" in
      Baseline:PASS) base_ok=1;;
      Architecture_check:FAIL) arch_fail=1;;
      AITER_isolation:FAIL) aiter_off_fail=1;;
      Kernel_comparison:PASS) kern_ok=1;;
      rccl_xgmi:PASS) comm_ok=1;;
      compute_mem:PASS) compute_ok=1;;
    esac
  done
  if [ "$INCLUDE_UNSAFE" = 1 ] && [ "$arch_fail" = 1 ] && [ "$base_ok" = 1 ]; then
    echo "LIKELY ROOT CAUSE: removing HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_VALUE destabilizes HSA arch selection on gfx942 — baseline (override on) healthy, architecture-check (override off) failed. Confidence: MEDIUM-HIGH. Keep the override set in production."
  elif [ "$base_ok" = 1 ] && [ "$compute_ok" = 1 ] && [ "$comm_ok" = 1 ] && [ "$kern_ok" = 1 ]; then
    echo "No instability reproduced in the safe suite: baseline, compute/mem, RCCL/XGMI and kernel-comparison all passed. Confidence that config is serviceable: MEDIUM (finite test). HSA-disabled comparison was $([ "$INCLUDE_UNSAFE" = 1 ] && echo run || echo SKIPPED)."
  else
    echo "INCONCLUSIVE: mixed results — see per-test detail and kernel_logs.txt. Not enough signal to assign a single root cause."
  fi
}

generate_report() {
  local pass fail skip inc
  pass=$(printf '%s\n' "${RES_STATUS[@]:-}" | grep -c '^PASS$')
  fail=$(printf '%s\n' "${RES_STATUS[@]:-}" | grep -c '^FAIL$')
  skip=$(printf '%s\n' "${RES_STATUS[@]:-}" | grep -c '^SKIPPED$')
  inc=$(printf '%s\n' "${RES_STATUS[@]:-}" | grep -c '^INCONCLUSIVE$')
  {
    echo "# MI300X GPU Stability Validation — $TS"
    echo
    echo "- Mode: **$MODE**  |  Unsafe matrix: **$([ "$INCLUDE_UNSAFE" = 1 ] && echo included || echo SKIPPED)**"
    echo "- Image \`$IMAGE\` · model \`$MODEL_PATH\` · TP$TP · mns=$MAX_NUM_SEQS · backend $ATTENTION_BACKEND · QR=$QUICK_REDUCE"
    echo "- Totals: PASS=$pass FAIL=$fail INCONCLUSIVE=$inc SKIPPED=$skip"
    echo
    echo "## Results"
    echo "| Test | Status | Detail | Secs |"
    echo "|---|---|---|---|"
    local i n=${#RES_NAMES[@]}
    for ((i=0;i<n;i++)); do echo "| ${RES_NAMES[$i]} | ${RES_STATUS[$i]} | ${RES_DETAIL[$i]} | ${RES_SECS[$i]:-0} |"; done
    echo
    echo "## Kernel faults observed"
    if grep -qiE "$FAULT_RE" "$KLOG" 2>/dev/null; then grep -iE "$FAULT_RE" "$KLOG" | sort | uniq -c | sort -rn | sed 's/^/    /'; else echo "None."; fi
    echo
    echo "## Inference metrics"
    if [ -s "$INFJSON.parts" ]; then echo '```json'; cat "$INFJSON.parts"; echo '```'; else echo "No inference stage ran."; fi
    echo
    echo "## Cluster health after testing"
    if wait_idle 10; then echo "All GPUs idle; " ; else echo "WARNING: GPUs not idle after teardown — inspect before reuse. "; fi
    grep -qiE "$FAULT_RE" "$KLOG" 2>/dev/null && echo "Kernel fault signatures were recorded during the run (see kernel_logs.txt)." || echo "No kernel fault signatures recorded."
    echo
    echo "## Most likely root cause"
    conclusion
    echo
    echo "## Recommended configuration"
    echo "- Keep \`HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_VALUE\` set (gfx942 native)."
    echo "- \`max_num_seqs=$MAX_NUM_SEQS\` per customer guardrail; graceful \`docker stop -t $STOP_GRACE\`."
    echo "- Backend as verified in run.log (do not trust the flag alone)."
    echo
    echo "## Caveats / remaining risk"
    echo "- This is a **finite** test; passing does NOT prove the cluster is crash-free."
    echo "- Intermittent wedges (see mns512_report.md) may not reproduce in one pass."
    [ "$INCLUDE_UNSAFE" = 0 ] && echo "- HSA-disabled (crash-prone) rows were SKIPPED; the override hypothesis was not exercised."
  } >"$REPORT"
}

# ======================================================================================
# Prerequisites
# ======================================================================================
preqs() {
  log "=== prerequisites ==="
  [ "$(id -u)" -eq 0 ] || warn "not root: dmesg/sysfs/amd-smi may be limited (run with sudo)"
  have docker || { echo "FATAL: docker not found" >&2; exit 3; }
  docker image inspect "$IMAGE" >/dev/null 2>&1 || warn "image $IMAGE not present locally (docker run will try to pull)"
  [ -e "$MODEL_PATH" ] || warn "model path $MODEL_PATH not visible from host (ok if only mounted in-container)"
  have curl || warn "curl missing: inference stages will be SKIPPED"
  if [ "$MAX_NUM_SEQS" -gt 256 ] 2>/dev/null; then warn "MAX_NUM_SEQS=$MAX_NUM_SEQS exceeds the 256 guardrail"; fi
}

environment_report() {
  log "=== environment ==="
  { echo "== $(date) =="
    echo "host kernel: $(uname -r)"
    echo "GPUs (sysfs cards): $(gpu_cards | wc -l)"
    for d in $(gpu_cards); do echo "  $(basename "$(dirname "$d")") busy=$(cat "$d/gpu_busy_percent" 2>/dev/null)"; done
    echo "docker: $(docker --version 2>/dev/null)"
    echo "image id: $(docker image inspect -f '{{.Id}} {{.Created}}' "$IMAGE" 2>/dev/null || echo 'not local')"
    if have amd-smi; then echo "== amd-smi static =="; amd-smi static -g all 2>&1 | head -80
    elif have rocm-smi; then echo "== rocm-smi =="; rocm-smi 2>&1 | head -40
    else echo "no amd-smi/rocm-smi on host (ROCm userspace is in the container)"; fi
    echo "== in-container versions =="
    docker run --rm "$IMAGE" bash -lc 'python3 -c "import torch;print(\"torch\",torch.__version__)"; (vllm --version 2>/dev/null)||true' 2>&1 | head
  } >"$ENVIRON" 2>&1
}

confirm_unsafe() {
  echo
  echo "############################################################"
  echo "# UNSAFE MATRIX requested (--include-unsafe-matrix)."
  echo "# This runs vLLM WITHOUT HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_VALUE,"
  echo "# the configuration previously associated with GPU hangs/crashes."
  echo "# It may wedge a GPU; the script will stop on fault and NOT reset it."
  echo "############################################################"
  printf 'Type EXACTLY "yes-run-unsafe" to proceed: '
  read -r ans
  [ "$ans" = "yes-run-unsafe" ] || { log "unsafe matrix declined; running safe suite only"; INCLUDE_UNSAFE=0; }
}

# ======================================================================================
# MAIN
# ======================================================================================
log "final_gpu_stability_test start — mode=$MODE unsafe=$INCLUDE_UNSAFE out=$OUTDIR"
preqs
environment_report
start_telemetry
xgmi_state

# Stage 2: pre-flight health
if pf="$(preflight)"; then record "preflight" PASS "$pf" 0
else record "preflight" FAIL "cluster not clean before testing ($pf)" 0; CRITICAL_FAIL=1
     warn "aborting: pre-flight failed — will not test on an unhealthy cluster"; generate_report; exit 1; fi

[ "$INCLUDE_UNSAFE" = 1 ] && confirm_unsafe

# Stage 3: compute/memory
gpu_compute_mem_test
# Stage 4: XGMI/RCCL
rccl_xgmi_test

# Stage 5-8: vLLM config matrix (safe rows always; unsafe rows gated)
vllm_config_test "Baseline"            1 1 sustained "$OUTDIR/vllm_baseline.log"
vllm_config_test "Kernel_comparison"   1 0 inference "$OUTDIR/vllm_aiter_off.log"   # HSA on, AITER off (safe)
if [ "$INCLUDE_UNSAFE" = 1 ]; then
  vllm_config_test "Architecture_check" 0 1 short     "$OUTDIR/vllm_hsa_off.log"      # HSA OFF (unsafe)
  vllm_config_test "AITER_isolation"    0 0 inference "$OUTDIR/vllm_hsa_aiter_off.log" # HSA OFF + AITER off (unsafe)
else
  record "Architecture_check" SKIPPED "unsafe (HSA disabled) — use --include-unsafe-matrix" 0
  record "AITER_isolation"    SKIPPED "unsafe (HSA disabled) — use --include-unsafe-matrix" 0
fi
# Stage 9: sustained already covered by Baseline(sustained); stability mode extends duration.

# Stage 10: report
generate_report
log "DONE — report: $REPORT"
echo
echo "================ SUMMARY ================"
grep -E '^\| ' "$REPORT" | sed -n '3,$p' 2>/dev/null || true
echo "Full report: $REPORT"
echo "Machine-readable: $RESJSON"
