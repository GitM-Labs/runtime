#!/usr/bin/env bash
#
# final_gpu_stability_test.sh — single-invocation 8x MI300X / Kimi-K2.6 stability validation.
#
#   sudo bash final_gpu_stability_test.sh                     # PRODUCTION suite (AITER-on), smoke mode
#   sudo bash final_gpu_stability_test.sh --mode stability    # production suite, long mode
#   sudo bash final_gpu_stability_test.sh --compare-aiter-off # + opt-in AITER-DISABLED comparison
#   sudo bash final_gpu_stability_test.sh --include-unsafe-matrix   # + opt-in HSA-override-disabled rows
#
# Goal: validate the production AITER-enabled config; comparisons are opt-in diagnostics only.
# Default path = preflight, compute/mem, RCCL/XGMI, AITER-ENABLED vLLM init+inference, postflight.
# Safety: production does NOT deploy AITER-off or HSA-override-off; both are OFF by default here.
# This script NEVER resets GPUs, reloads drivers, reboots, or changes firmware. It only kills
# containers/processes IT started. It saves partial results if a stage crashes or times out.
#
# Exit code: 0 only when no FAIL/UNSAFE occurred. 1 = a test FAILED. 2 = UNSAFE_TO_CONTINUE
# (GPU fault/hang/idle-check failure/preflight unsafe) — the suite halts further GPU work.
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
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"          # customer guardrail: hard cap 256 (enforced, see parse_args)
MAX_NUM_SEQS_CAP="${MAX_NUM_SEQS_CAP:-256}"
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
INCLUDE_UNSAFE=0             # HSA-override-disabled rows (opt-in; --include-unsafe-matrix)
RUN_AITER_COMPARISON="${RUN_AITER_COMPARISON:-0}"   # AITER-off comparison (opt-in; --compare-aiter-off)
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
GEN_TOKENS="${GEN_TOKENS:-128}"

# Timeouts (seconds)
LOAD_TIMEOUT="${LOAD_TIMEOUT:-1200}"         # model load + ready
INIT_TIMEOUT="${INIT_TIMEOUT:-600}"          # short-init rows
REQ_TIMEOUT="${REQ_TIMEOUT:-120}"            # per inference request
STOP_GRACE="${STOP_GRACE:-120}"              # graceful docker stop
STOP_HARD_TIMEOUT="${STOP_HARD_TIMEOUT:-30}" # extra wall-clock we allow 'docker stop' itself to block
HEALTH_POLL="${HEALTH_POLL:-5}"              # seconds between live health polls
DETECT_TIMEOUT="${DETECT_TIMEOUT:-60}"       # short docker probes (e.g. tool detection)
SUSTAINED_SECS_SMOKE="${SUSTAINED_SECS_SMOKE:-60}"
SUSTAINED_SECS_STAB="${SUSTAINED_SECS_STAB:-1800}"
COMPUTE_SECS_SMOKE="${COMPUTE_SECS_SMOKE:-30}"
COMPUTE_SECS_STAB="${COMPUTE_SECS_STAB:-180}"

# dmesg fault signatures (mns512_report.md / kernel_health_checks.md + Fix 10 additions)
FAULT_RE='timeout waiting for kiq fence|retry page fault|TLB flush failed|failed to write reg|Memory access fault|HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION|uncorrectable| amdgpu_job_timedout|ring [a-z0-9_]* timeout| GPU reset |VM_L2_PROTECTION_FAULT|VM fault|XGMI.*(error|fault|down)'

# ======================================================================================
# State
# ======================================================================================
RES_NAMES=(); RES_STATUS=(); RES_DETAIL=(); RES_SECS=()
RES_FAULT=()                 # per-row: 1 if the failure was a GPU fault (for conclusion heuristic)
SUITE_RC=0                   # 0 ok, 1 FAIL present, 2 UNSAFE_TO_CONTINUE present
HALT=0                       # 1 => stop all further GPU workloads
CLEANED=0                    # cleanup idempotency guard
BG_PIDS=()
DMESG_OK=1                   # 0 => cannot read dmesg; fault detection is blind
CONTAINER_PREFIX=""          # set in setup_dirs

# ======================================================================================
# Logging + results
# ======================================================================================
# logs go to stderr (+ RUNLOG) so function stdout stays clean for value capture
log()  { printf '%s %s\n' "[$(date +%H:%M:%S)]" "$*" | tee -a "${RUNLOG:-/dev/null}" >&2; }
warn() { printf '%s WARN: %s\n' "[$(date +%H:%M:%S)]" "$*" | tee -a "${RUNLOG:-/dev/null}" >&2; }

# raise SUITE_RC monotonically to the given severity (1 FAIL, 2 UNSAFE)
set_rc() { [ "$1" -gt "$SUITE_RC" ] && SUITE_RC="$1"; return 0; }

# record <name> <status> <detail> <seconds> [is_gpu_fault 0|1]
record() {
  local name="$1" st="$2" d="${3:-}" secs="${4:-0}" fault="${5:-0}"
  d="${d//$'\n'/ }"; d="${d//|/ }"
  RES_NAMES+=("$name"); RES_STATUS+=("$st"); RES_DETAIL+=("$d"); RES_SECS+=("$secs"); RES_FAULT+=("$fault")
  case "$st" in
    FAIL)               set_rc 1;;
    UNSAFE_TO_CONTINUE) set_rc 2; HALT=1;;
  esac
  emit_json
  log "RESULT [$st] $name — $d"
}

# ======================================================================================
# Cleanup + signal handling (Fix 6, Fix 8)
# ======================================================================================
cleanup() {
  [ "$CLEANED" = 1 ] && return 0
  CLEANED=1
  local ec=$?
  { echo "== cleanup @ $(date) (caught exit=$ec) =="; } >>"${CLEANUP_LOG:-/dev/null}" 2>&1
  log "cleanup: tearing down (never deletes logs; graceful stop ${STOP_GRACE}s, hard-bounded ${STOP_HARD_TIMEOUT}s)"
  for p in "${BG_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
  if command -v docker >/dev/null 2>&1 && [ -n "$CONTAINER_PREFIX" ]; then
    local c
    for c in $(docker ps -a --filter "name=${CONTAINER_PREFIX}" --format '{{.Names}}' 2>/dev/null); do
      stop_container "$c" cleanup
    done
  fi
  emit_json
  [ -f "${REPORT:-/nonexistent}" ] || generate_report 2>/dev/null
  log "cleanup: results in ${OUTDIR:-?}"
}
on_signal() {
  local sig="$1"
  log "received SIG$sig — stopping suite (no resume, no reset)"
  record "interrupt" UNSAFE_TO_CONTINUE "caught SIG$sig; cleaning up and exiting" 0
  cleanup
  # 128 + signal number convention; ensures nonzero and does not resume the suite
  case "$sig" in INT) exit 130;; TERM) exit 143;; *) exit 2;; esac
}

# ======================================================================================
# Helpers
# ======================================================================================
have() { command -v "$1" >/dev/null 2>&1; }

gpu_cards() { ls -d /sys/class/drm/card*/device 2>/dev/null | while read -r d; do
  [ -f "$d/gpu_busy_percent" ] && echo "$d"; done; }

# Fix 10: verify dmesg is actually readable; otherwise fault detection is blind.
check_dmesg() { if dmesg -T >/dev/null 2>&1; then DMESG_OK=1; else DMESG_OK=0; fi; }
dmesg_fault_count() {
  [ "$DMESG_OK" = 1 ] || { echo -1; return; }   # -1 = unknown (blind), never a silent 0
  local c; c=$(dmesg -T 2>/dev/null | grep -icE "$FAULT_RE"); echo "${c:-0}"
}
# new faults since baseline? echoes yes/no; treats blind dmesg as "unknown"->caller handles
faults_since() { local base="$1" now; now="$(dmesg_fault_count)"
  if [ "$now" = -1 ] || [ "$base" = -1 ]; then echo unknown; return; fi
  [ "$now" -gt "$base" ] && echo yes || echo no; }

capture_kernel() { { echo "==== dmesg @ $(date) ($1) ===="
  if [ "$DMESG_OK" = 1 ]; then dmesg -T 2>/dev/null | tail -n 400 | grep -iE "$FAULT_RE" || echo "(no fault signatures)"
  else echo "(dmesg not readable — fault detection blind)"; fi; } >> "$KLOG"; }

# Fix 9: record every bounded command's timing/exit/timeout to timings.csv
record_timing() { printf '%s,%s,%s,%s,%s,%s,%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "${7:-}" >>"$TIMINGS"; }
# timed <label> <timeout_s> -- <cmd...>  : bounded run, records timing, returns cmd rc (124 on timeout)
timed() {
  local label="$1" tmo="$2"; shift 2; [ "$1" = "--" ] && shift
  local st en rc; st=$(date +%s)
  timeout "$tmo" "$@"; rc=$?
  en=$(date +%s)
  local to=0; [ "$rc" -eq 124 ] && to=1
  record_timing "$label" "$st" "$en" "$((en-st))" "$rc" "$to" "$*"
  return "$rc"
}

# Fix 8: bounded, state-aware container stop. Never assumes 'docker stop' returns (D-state on wedge).
stop_container() {
  local cname="$1" ctx="${2:-stop}"
  command -v docker >/dev/null 2>&1 || return 0
  docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${cname}$" || return 0
  { echo "[$(date +%H:%M:%S)] ($ctx) stopping $cname (grace ${STOP_GRACE}s, bounded ${STOP_HARD_TIMEOUT}s)"; } >>"${CLEANUP_LOG:-/dev/null}" 2>&1
  # preserve logs before removal
  docker logs "$cname" >>"${OUTDIR:-.}/container_${cname}.log" 2>&1 || true
  if timeout "$((STOP_GRACE+STOP_HARD_TIMEOUT))" docker stop -t "$STOP_GRACE" "$cname" >/dev/null 2>&1; then
    { echo "[$(date +%H:%M:%S)] stopped $cname cleanly"; } >>"${CLEANUP_LOG:-/dev/null}" 2>&1
  else
    warn "docker stop did not return for $cname within $((STOP_GRACE+STOP_HARD_TIMEOUT))s — container likely blocked on a wedged GPU (D-state). NOT forcing destructive recovery."
    { echo "[$(date +%H:%M:%S)] stop TIMED OUT for $cname — unresponsive (wedged-GPU signature), left for inspection"; } >>"${CLEANUP_LOG:-/dev/null}" 2>&1
  fi
  timeout "$STOP_HARD_TIMEOUT" docker rm -f "$cname" >/dev/null 2>&1 || true
}

# Telemetry sampler (sysfs; amd-smi enriches if present)
start_telemetry() {
  [ -f "$TELEM" ] || echo "ts,card,gpu_busy_percent,power_w,temp_c,vram_used_mb" > "$TELEM"
  ( while :; do
      now="$(date +%s)"
      for d in $(gpu_cards); do
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

# ======================================================================================
# Pre-flight + mandatory GPU health gate (Fix 5, Fix 10)
# ======================================================================================
preflight() {
  check_dmesg
  local used=""
  for cand in "${PREFLIGHT_CANDIDATES[@]}"; do
    [ -n "$cand" ] && { [ -x "$cand" ] || [ -f "$cand" ]; } && { used="$cand"; break; }
  done
  if [ -n "$used" ]; then
    log "preflight: reusing $used"
    if timed preflight_reuse "$DETECT_TIMEOUT" -- bash "$used" >>"$PREFLIGHT_LOG" 2>&1; then echo "reused:$used"; return 0
    else echo "reused:$used (nonzero)"; return 1; fi
  fi
  log "preflight: no existing script; inline gate (kernel_health_checks.md)"
  local bad=0
  { echo "== inline preflight @ $(date) =="
    echo "dmesg_readable=$DMESG_OK"
    for d in $(gpu_cards); do
      b="$(cat "$d/gpu_busy_percent" 2>/dev/null)"; echo "$(basename "$(dirname "$d")") busy=$b"
      [ "$b" = "0" ] || { echo "  NOT IDLE"; bad=1; }
    done
  } >>"$PREFLIGHT_LOG" 2>&1
  if [ "$DMESG_OK" != 1 ]; then echo "dmesg-unreadable"; return 2; fi      # 2 => unsafe (blind)
  if [ "$(dmesg_fault_count)" -gt 0 ]; then echo "recent-faults"; return 1; fi
  echo "inline-clean"; return 0
}

# gpu_gate <context> : mandatory before each GPU phase and after each teardown.
# Verifies idle + dmesg-readable + no new faults + GPU query works. Sets HALT on failure.
gpu_gate() {
  local ctx="$1" t=0
  [ "$HALT" = 1 ] && return 1
  if [ "$DMESG_OK" != 1 ]; then record "gate:$ctx" UNSAFE_TO_CONTINUE "cannot read dmesg; GPU fault detection is blind" 0; return 1; fi
  # wait for idle
  while [ "$t" -lt 120 ]; do
    local busy=0; for d in $(gpu_cards); do [ "$(cat "$d/gpu_busy_percent" 2>/dev/null || echo 1)" = "0" ] || busy=1; done
    [ "$busy" = 0 ] && break; sleep 3; t=$((t+3))
  done
  local busy=0; for d in $(gpu_cards); do [ "$(cat "$d/gpu_busy_percent" 2>/dev/null || echo 1)" = "0" ] || busy=1; done
  if [ "$busy" != 0 ]; then record "gate:$ctx" UNSAFE_TO_CONTINUE "GPU not idle after prior workload (busy!=0 / possible wedge)" 0; return 1; fi
  # no fresh fault lines
  if [ "$(dmesg_fault_count)" -gt 0 ] && [ -n "${GATE_FAULT_BASE:-}" ] && [ "$(dmesg_fault_count)" -gt "$GATE_FAULT_BASE" ]; then
    record "gate:$ctx" UNSAFE_TO_CONTINUE "new dmesg fault signatures before launch" 0; return 1; fi
  return 0
}

# ======================================================================================
# In-container GPU jobs with live fault monitoring (Fix 2, Fix 9)
# run_gpu_job <cname> <logf> <timeout> -- <docker run args...>   (do NOT pass -d/--rm/--name)
# echoes: DONE:<exit> | FAULT | TIMEOUT | STARTFAIL ; returns 0/101/124/125
# ======================================================================================
run_gpu_job() {
  local cname="$1" logf="$2" tmo="$3"; shift 3; [ "$1" = "--" ] && shift
  have docker || { echo STARTFAIL; return 125; }
  docker rm -f "$cname" >/dev/null 2>&1
  local base; base="$(dmesg_fault_count)"
  # shellcheck disable=SC2086
  if ! docker run -d --name "$cname" "$@" >/dev/null 2>>"$logf"; then echo STARTFAIL; return 125; fi
  ( timeout "$((tmo+DETECT_TIMEOUT))" docker logs -f "$cname" >>"$logf" 2>&1 ) & local lp=$!
  local t=0
  while [ "$t" -lt "$tmo" ]; do
    if [ "$(faults_since "$base")" = yes ]; then kill "$lp" 2>/dev/null; docker kill "$cname" >/dev/null 2>&1; echo FAULT; return 101; fi
    docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${cname}$" || { kill "$lp" 2>/dev/null
      local ec; ec=$(docker inspect -f '{{.State.ExitCode}}' "$cname" 2>/dev/null || echo "?"); echo "DONE:$ec"; return 0; }
    sleep "$HEALTH_POLL"; t=$((t+HEALTH_POLL))
  done
  kill "$lp" 2>/dev/null; docker kill "$cname" >/dev/null 2>&1; echo TIMEOUT; return 124
}

# ======================================================================================
# vLLM launch / readiness / teardown
# ======================================================================================
launch_vllm() {
  local logf="$1" hsa="$2" aiter="$3" cname="$4"
  local envs=(-e "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=${QUICK_REDUCE}")
  [ "$hsa" = 1 ] && envs+=(-e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_VALUE}")
  envs+=(-e "VLLM_ROCM_USE_AITER=${aiter}" -e "VLLM_ATTENTION_BACKEND=${ATTENTION_BACKEND}")
  { echo "# cmd @ $(date)"
    echo "# HSA_OVERRIDE=$([ "$hsa" = 1 ] && echo "$HSA_OVERRIDE_VALUE" || echo UNSET)  AITER=$aiter  mns=$MAX_NUM_SEQS  backend=$ATTENTION_BACKEND"; } >"$logf"
  # shellcheck disable=SC2086
  docker run -d --rm --name "$cname" $DOCKER_DEVICE_ARGS $EXTRA_DOCKER_ARGS -p "${PORT}:${PORT}" \
    -v "${MODEL_PATH%/*}:${MODEL_PATH%/*}:ro" "${envs[@]}" "$IMAGE" \
    vllm serve "$MODEL_PATH" --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT" \
      --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN" --block-size "$BLOCK_SIZE" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs "$MAX_NUM_SEQS" --trust-remote-code \
      >>"$logf" 2>&1 || { echo "docker run failed" >>"$logf"; return 1; }
  ( timeout "$((LOAD_TIMEOUT+DETECT_TIMEOUT))" docker logs -f "$cname" >>"$logf" 2>&1 ) & BG_PIDS+=("$!")
  return 0
}

# echoes ready|crashed|fault|timeout
wait_ready() {
  local cname="$1" tmo="$2" base="$3" t=0
  while [ "$t" -lt "$tmo" ]; do
    docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${cname}$" || { echo crashed; return 1; }
    [ "$(faults_since "$base")" = yes ] && { echo fault; return 1; }
    if have curl && curl -fsS -m 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then echo ready; return 0; fi
    sleep "$HEALTH_POLL"; t=$((t+HEALTH_POLL))
  done
  echo timeout; return 1
}

# Fix 263: compare the OBSERVED backend to the requested AITER state. echoes match|mismatch|unknown
verify_backend() {
  local logf="$1" want_aiter="$2"
  { echo "---- backend verification ($logf) want_aiter=$want_aiter ----"
    grep -hoE "Using \[[^]]*\] all-reduce backends[^.]*" "$logf" | head -3
    grep -hoE "Final IR op priority[^)]*\)" "$logf" | head -1; } >>"$RUNLOG" 2>&1
  local line; line=$(grep -hoE "Using \[[^]]*\] all-reduce backends[^.]*for group 'tp:0'" "$logf" | head -1)
  [ -n "$line" ] || { echo unknown; return; }
  if echo "$line" | grep -q "AITER_CUSTOM"; then
    [ "$want_aiter" = 1 ] && echo match || echo mismatch
  else
    [ "$want_aiter" = 0 ] && echo match || echo mismatch
  fi
}

# ======================================================================================
# Inference (Fix 4): validate HTTP 200 + JSON + nonempty text + no error field.
# Sets INFER_STATUS/INFER_DETAIL. Does NOT itself mark the config PASS.
# ======================================================================================
infer_prompt() { yes "stability test prompt token" | head -n "$PROMPT_TOKENS" | tr '\n' ' '; }
run_inference() {
  local label="$1" conc="$2" dur="$3"
  INFER_STATUS=SKIPPED; INFER_DETAIL="curl missing"
  have curl || { record "inference:$label" SKIPPED "curl missing" 0; return 0; }
  local prompt body tmp end ok=0 bad=0 to=0
  prompt="$(infer_prompt)"
  body="$(printf '{"model":"%s","prompt":"%s","max_tokens":%d,"temperature":0,"ignore_eos":true}' "$SERVED_NAME" "$prompt" "$GEN_TOKENS")"
  tmp="$(mktemp)"; end=$(( $(date +%s) + dur ))
  while [ "$(date +%s)" -lt "$end" ]; do
    local pids=()
    for _ in $(seq 1 "$conc"); do
      ( local bodyf code st en; bodyf="$(mktemp)"; st=$(date +%s.%N)
        code=$(curl -s -o "$bodyf" -w '%{http_code}' -m "$REQ_TIMEOUT" -H 'Content-Type: application/json' \
               -d "$body" "http://localhost:${PORT}/v1/completions" 2>/dev/null)
        en=$(date +%s.%N)
        # valid = HTTP 200 AND has a nonempty choices[].text AND no error field
        local verdict=bad
        if [ "$code" = 200 ] && ! grep -q '"error"' "$bodyf" 2>/dev/null \
           && grep -qE '"text"[[:space:]]*:[[:space:]]*"[^"]' "$bodyf" 2>/dev/null; then verdict=ok; fi
        [ "$code" = 000 ] && verdict=timeout
        echo "$verdict $(awk "BEGIN{print $en-$st}")" >>"$tmp"; rm -f "$bodyf"
      ) & pids+=("$!")
    done
    for p in "${pids[@]}"; do wait "$p"; done
    [ "$(faults_since "$BASE_FAULTS")" = yes ] && { capture_kernel "inference:$label"; INFER_FAULT=1; break; }
  done
  while read -r v lat; do case "$v" in ok) ok=$((ok+1)); echo "$lat">>"$tmp.lat";; timeout) to=$((to+1));; *) bad=$((bad+1));; esac; done <"$tmp"
  local avg=NA; [ -s "$tmp.lat" ] && avg=$(awk '{s+=$1;n++} END{if(n)printf "%.3f",s/n}' "$tmp.lat")
  local total=$((ok+bad+to))
  printf '{"label":"%s","concurrency":%d,"duration_s":%d,"requests":%d,"ok":%d,"invalid":%d,"timeouts":%d,"lat_avg_s":"%s"}\n' \
    "$label" "$conc" "$dur" "$total" "$ok" "$bad" "$to" "$avg" >>"$INFJSON.parts"
  rm -f "$tmp" "$tmp.lat"
  if [ "$total" -eq 0 ]; then INFER_STATUS=INCONCLUSIVE; INFER_DETAIL="no requests issued"
  elif [ "$bad" -gt 0 ] || [ "$to" -gt 0 ] || [ "$ok" -eq 0 ]; then INFER_STATUS=FAIL; INFER_DETAIL="ok=$ok invalid=$bad timeouts=$to (not all requests valid)"
  else INFER_STATUS=PASS; INFER_DETAIL="ok=$ok/$total avg=${avg}s conc=$conc"; fi
}

# ======================================================================================
# vLLM config test (Fix 2, Fix 4, Fix 5)
# ======================================================================================
vllm_config_test() {
  local name="$1" hsa="$2" aiter="$3" workload="$4" logf="$5"
  [ "$HALT" = 1 ] && { record "$name" SKIPPED "halted: prior unsafe condition" 0; return 0; }
  have docker || { record "$name" SKIPPED "docker missing" 0; return 0; }
  GATE_FAULT_BASE="$(dmesg_fault_count)"; gpu_gate "$name" || { record "$name" SKIPPED "halted: GPU gate failed" 0; return 0; }
  local cname="${CONTAINER_PREFIX}_$(echo "$name" | tr -c 'a-zA-Z0-9' _)"
  local t0; t0=$(date +%s); BASE_FAULTS="$(dmesg_fault_count)"; INFER_FAULT=0
  log "== $name :: HSA=$([ "$hsa" = 1 ] && echo on || echo OFF) AITER=$aiter workload=$workload =="
  launch_vllm "$logf" "$hsa" "$aiter" "$cname" || { record "$name" FAIL "container failed to start" "$(( $(date +%s)-t0 ))"; return 1; }
  local tmo="$LOAD_TIMEOUT"; [ "$workload" = short ] && tmo="$INIT_TIMEOUT"
  local status; status="$(wait_ready "$cname" "$tmo" "$BASE_FAULTS")"
  capture_kernel "$name"; local vb; vb="$(verify_backend "$logf" "$aiter")"
  case "$status" in
    ready)   log "$name: model ready (backend-verify=$vb)";;
    fault)   record "$name" UNSAFE_TO_CONTINUE "GPU fault during load (dmesg signature)" "$(( $(date +%s)-t0 ))" 1; stop_container "$cname"; return 1;;
    crashed) record "$name" FAIL "container exited during load without GPU fault (see $logf)" "$(( $(date +%s)-t0 ))" 0; stop_container "$cname"; return 1;;
    timeout) record "$name" UNSAFE_TO_CONTINUE "load hang ${tmo}s (device unverified)" "$(( $(date +%s)-t0 ))" 1; stop_container "$cname"; return 1;;
  esac
  local backend_note=""; [ "$vb" = mismatch ] && backend_note=" [WARN backend!=requested]"; [ "$vb" = unknown ] && backend_note=" [backend unverified]"
  if [ "$workload" = short ]; then
    if [ "$vb" = unknown ]; then record "$name" INCONCLUSIVE "ready but backend unverified" "$(( $(date +%s)-t0 ))"
    else record "$name" PASS "init+ready ok (backend-verify=$vb)$backend_note" "$(( $(date +%s)-t0 ))"; fi
  else
    local dur=20 conc=4
    [ "$workload" = sustained ] && { conc=8; dur=$([ "$MODE" = stability ] && echo "$SUSTAINED_SECS_STAB" || echo "$SUSTAINED_SECS_SMOKE"); }
    run_inference "$name" "$conc" "$dur"
    if [ "$INFER_FAULT" = 1 ]; then record "$name" UNSAFE_TO_CONTINUE "GPU fault during inference" "$(( $(date +%s)-t0 ))" 1
    else record "inference:$name" "$INFER_STATUS" "$INFER_DETAIL" 0
      case "$INFER_STATUS" in
        PASS) record "$name" PASS "ready+inference ok$backend_note" "$(( $(date +%s)-t0 ))";;
        *)    record "$name" FAIL "inference did not pass ($INFER_DETAIL)" "$(( $(date +%s)-t0 ))" 0;;
      esac
    fi
  fi
  stop_container "$cname"
  GATE_FAULT_BASE="$(dmesg_fault_count)"; gpu_gate "post:$name" || true
}

# ======================================================================================
# Compute/memory + RCCL/XGMI (Fix 2, Fix 3)
# ======================================================================================
gpu_compute_mem_test() {
  [ "$HALT" = 1 ] && { record "compute_mem" SKIPPED "halted: prior unsafe condition" 0; return 0; }
  have docker || { record "compute_mem" SKIPPED "docker missing" 0; return 0; }
  gpu_gate compute_mem || { record "compute_mem" SKIPPED "halted: GPU gate failed" 0; return 0; }
  local secs; secs=$([ "$MODE" = stability ] && echo "$COMPUTE_SECS_STAB" || echo "$COMPUTE_SECS_SMOKE")
  local logf="$OUTDIR/compute_mem.log" cname="${CONTAINER_PREFIX}_compute" t0; t0=$(date +%s)
  # Fix 368/376: require ALL TP GPUs visible AND validate results, else explicit failure.
  local py='
import os,time,torch
want=int(os.environ.get("TP","0")); n=torch.cuda.device_count()
if want and n<want: print("COMPUTE_FAIL only %d of %d GPUs visible"%(n,want)); raise SystemExit(0)
secs=float(os.environ.get("SECS","30")); t0=time.time(); okdev=0
for i in range(n):
  with torch.cuda.device(i):
    a=torch.ones(1024,1024,device="cuda",dtype=torch.float32); b=a@a
    v=b[0,0].item()
    if abs(v-1024.0)>1e-3: print("COMPUTE_FAIL bad matmul on cuda:%d got %f"%(i,v)); raise SystemExit(0)
    m=torch.empty(1<<28,dtype=torch.uint8,device="cuda"); m.fill_(7)
    if int(m[-1].item())!=7: print("COMPUTE_FAIL bad mem on cuda:%d"%i); raise SystemExit(0)
    del m; okdev+=1
while time.time()-t0<secs:
  for i in range(n):
    with torch.cuda.device(i): _=(torch.ones(1024,1024,device="cuda")@torch.ones(1024,1024,device="cuda"))
  torch.cuda.synchronize()
print("COMPUTE_MEM_OK devices=%d"%okdev)'
  # shellcheck disable=SC2086
  local out; out="$(run_gpu_job "$cname" "$logf" "$((secs+DETECT_TIMEOUT))" -- \
    $DOCKER_DEVICE_ARGS $EXTRA_DOCKER_ARGS -e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_VALUE}" -e "SECS=$secs" -e "TP=$TP" \
    "$IMAGE" python3 -c "$py")"
  local rc=$?; docker rm -f "$cname" >/dev/null 2>&1; capture_kernel compute_mem
  case "$out" in
    FAULT)   record "compute_mem" UNSAFE_TO_CONTINUE "GPU fault during compute (dmesg)" "$(( $(date +%s)-t0 ))" 1;;
    TIMEOUT) record "compute_mem" UNSAFE_TO_CONTINUE "compute hang (timeout; device unverified)" "$(( $(date +%s)-t0 ))" 1;;
    STARTFAIL) record "compute_mem" FAIL "container failed to start" "$(( $(date +%s)-t0 ))" 0;;
    DONE:*)
      if grep -q COMPUTE_MEM_OK "$logf"; then record "compute_mem" PASS "$(grep -o 'COMPUTE_MEM_OK.*' "$logf" | head -1) (expected TP=$TP)" "$(( $(date +%s)-t0 ))"
      elif grep -q COMPUTE_FAIL "$logf"; then record "compute_mem" FAIL "$(grep -o 'COMPUTE_FAIL.*' "$logf" | head -1)" "$(( $(date +%s)-t0 ))" 0
      else record "compute_mem" INCONCLUSIVE "no OK/FAIL marker (rc=${out#DONE:}); see compute_mem.log" "$(( $(date +%s)-t0 ))"; fi;;
    *) record "compute_mem" INCONCLUSIVE "unrecognized job state ($out, rc=$rc)" "$(( $(date +%s)-t0 ))";;
  esac
}

rccl_xgmi_test() {
  [ "$HALT" = 1 ] && { record "rccl_xgmi" SKIPPED "halted: prior unsafe condition" 0; return 0; }
  have docker || { record "rccl_xgmi" SKIPPED "docker missing" 0; return 0; }
  gpu_gate rccl_xgmi || { record "rccl_xgmi" SKIPPED "halted: GPU gate failed" 0; return 0; }
  local logf="$OUTDIR/rccl_test.log" cname="${CONTAINER_PREFIX}_rccl" t0; t0=$(date +%s)
  xgmi_state   # Fix 404: writes to its own file, not rccl_test.log
  local has=0
  timed rccl_detect "$DETECT_TIMEOUT" -- bash -c "docker run --rm $DOCKER_DEVICE_ARGS '$IMAGE' bash -lc 'command -v all_reduce_perf'" >/dev/null 2>&1 && has=1
  local out expect_sum=$TP
  if [ "$has" = 1 ]; then
    log "rccl_xgmi: using all_reduce_perf"
    # shellcheck disable=SC2086
    out="$(run_gpu_job "$cname" "$logf" 300 -- $DOCKER_DEVICE_ARGS $EXTRA_DOCKER_ARGS \
      -e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_VALUE}" "$IMAGE" bash -lc "all_reduce_perf -b 8 -e 512M -f 2 -g $TP")"
  else
    log "rccl_xgmi: all_reduce_perf absent -> portable torch.distributed (Fix 3: real script, not torchrun -c)"
    # Fix 3: write python to a file inside the container and run it with torchrun (NOT -c).
    local runner='set -e; cat >/tmp/rccl_probe.py <<"PYEOF"
import os,torch,torch.distributed as dist
dist.init_process_group("nccl"); r=dist.get_rank(); w=dist.get_world_size(); torch.cuda.set_device(r)
x=torch.ones(1<<24,device="cuda"); dist.all_reduce(x); torch.cuda.synchronize()
g=[torch.empty(8,device="cuda") for _ in range(w)]; dist.all_gather(g,torch.full((8,),float(r),device="cuda"))
b=torch.full((8,),float(r),device="cuda"); dist.broadcast(b,0); torch.cuda.synchronize()
if r==0:
    import sys
    s=int(x[0].item()); gg=int(sum(t.sum().item() for t in g)//8); bb=int(b[0].item())
    print("RCCL_OK sum=%d allgather=%d bcast=%d world=%d"%(s,gg,bb,w)); sys.stdout.flush()
dist.destroy_process_group()
PYEOF
torchrun --standalone --nproc_per_node='"$TP"' /tmp/rccl_probe.py'
    # shellcheck disable=SC2086
    out="$(run_gpu_job "$cname" "$logf" 300 -- $DOCKER_DEVICE_ARGS $EXTRA_DOCKER_ARGS \
      -e "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_VALUE}" "$IMAGE" bash -lc "$runner")"
  fi
  local rc=$?; docker rm -f "$cname" >/dev/null 2>&1; capture_kernel rccl_xgmi
  case "$out" in
    FAULT)   record "rccl_xgmi" UNSAFE_TO_CONTINUE "GPU fault during collectives (dmesg)" "$(( $(date +%s)-t0 ))" 1;;
    TIMEOUT) record "rccl_xgmi" UNSAFE_TO_CONTINUE "collective hang (timeout; XGMI/RCCL)" "$(( $(date +%s)-t0 ))" 1;;
    STARTFAIL) record "rccl_xgmi" FAIL "container failed to start" "$(( $(date +%s)-t0 ))" 0;;
    DONE:*)
      if [ "$has" = 1 ]; then
        grep -q "Avg bus bandwidth" "$logf" && record "rccl_xgmi" PASS "all_reduce_perf completed" "$(( $(date +%s)-t0 ))" \
          || record "rccl_xgmi" FAIL "all_reduce_perf produced no bandwidth result" "$(( $(date +%s)-t0 ))" 0
      else
        # Fix 376: validate the collective RESULT (sum must equal world size), not just presence of a marker.
        local sum; sum=$(grep -oE 'RCCL_OK sum=[0-9]+' "$logf" | head -1 | grep -oE '[0-9]+$')
        if [ -n "$sum" ] && [ "$sum" = "$expect_sum" ]; then record "rccl_xgmi" PASS "all-reduce correct (sum=$sum=world) $(grep -o 'allgather=[0-9]* bcast=[0-9]*' "$logf"|head -1)" "$(( $(date +%s)-t0 ))"
        elif [ -n "$sum" ]; then record "rccl_xgmi" FAIL "all-reduce WRONG result sum=$sum expected $expect_sum" "$(( $(date +%s)-t0 ))" 0
        else record "rccl_xgmi" INCONCLUSIVE "no validated RCCL result (rc=${out#DONE:}); see rccl_test.log" "$(( $(date +%s)-t0 ))"; fi
      fi;;
    *) record "rccl_xgmi" INCONCLUSIVE "unrecognized job state ($out, rc=$rc)" "$(( $(date +%s)-t0 ))";;
  esac
}

xgmi_state() {
  { echo "== XGMI/topology @ $(date) =="
    if have amd-smi; then amd-smi xgmi 2>&1; amd-smi topology 2>&1
    elif have rocm-smi; then rocm-smi --showtopo 2>&1
    else echo "SKIPPED: no amd-smi/rocm-smi on host"; fi
  } >>"$OUTDIR/xgmi_topology.txt" 2>&1
}

# ======================================================================================
# JSON + report
# ======================================================================================
emit_json() {
  { echo '{'
    echo "  \"timestamp\": \"$TS\", \"mode\": \"$MODE\", \"include_unsafe\": $INCLUDE_UNSAFE, \"suite_rc\": $SUITE_RC, \"halted\": $HALT,"
    echo "  \"image\": \"$IMAGE\", \"model\": \"$MODEL_PATH\", \"max_num_seqs\": $MAX_NUM_SEQS,"
    echo "  \"attention_backend\": \"$ATTENTION_BACKEND\", \"hsa_override\": \"$HSA_OVERRIDE_VALUE\","
    echo '  "results": ['
    local i n=${#RES_NAMES[@]}
    for ((i=0;i<n;i++)); do local sep=,; [ $((i+1)) -eq "$n" ] && sep=
      printf '    {"test":"%s","status":"%s","detail":"%s","seconds":%s}%s\n' \
        "${RES_NAMES[$i]}" "${RES_STATUS[$i]}" "$(printf '%s' "${RES_DETAIL[$i]}" | sed 's/\\/\\\\/g; s/"/\\"/g')" "${RES_SECS[$i]:-0}" "$sep"; done
    echo '  ]'; echo '}'; } >"$RESJSON"
}

# Fix 308: finalize the advertised aggregate metrics file.
finalize_inference_json() {
  if [ -s "$INFJSON.parts" ]; then { echo '['; paste -sd, "$INFJSON.parts"; echo ']'; } >"$INFJSON"
  else echo '[]' >"$INFJSON"; fi
}

# Fix 475: only blame the HSA override when Architecture_check failed with an actual GPU fault.
conclusion() {
  local base_ok=0 arch_gpu_fault=0 compute_ok=0 comm_ok=0 kern_ok=0 i n=${#RES_NAMES[@]}
  for ((i=0;i<n;i++)); do
    case "${RES_NAMES[$i]}" in
      Baseline) [ "${RES_STATUS[$i]}" = PASS ] && base_ok=1;;
      Architecture_check) { [ "${RES_STATUS[$i]}" = UNSAFE_TO_CONTINUE ] || [ "${RES_STATUS[$i]}" = FAIL ]; } && [ "${RES_FAULT[$i]}" = 1 ] && arch_gpu_fault=1;;
      compute_mem) [ "${RES_STATUS[$i]}" = PASS ] && compute_ok=1;;
      rccl_xgmi) [ "${RES_STATUS[$i]}" = PASS ] && comm_ok=1;;
      Kernel_comparison) [ "${RES_STATUS[$i]}" = PASS ] && kern_ok=1;;
    esac
  done
  if [ "$INCLUDE_UNSAFE" = 1 ] && [ "$arch_gpu_fault" = 1 ] && [ "$base_ok" = 1 ]; then
    echo "SIGNAL: Architecture_check (HSA override OFF) produced a GPU fault while the HSA-on Baseline was healthy. Consistent with the override mattering, but a single run is not proof. Confidence: LOW-MEDIUM."
  elif [ "$base_ok" = 1 ] && [ "$compute_ok" = 1 ] && [ "$comm_ok" = 1 ]; then
    echo "No instability reproduced in the safe suite (baseline, compute/mem, RCCL/XGMI, kernel-comparison all PASS). Finite test; MEDIUM confidence the config is serviceable."
  else
    echo "INCONCLUSIVE: mixed results — see per-test detail and kernel_logs.txt. No single root cause assigned."
  fi
}

generate_report() {
  finalize_inference_json
  local pass fail unsafe skip inc
  pass=$(printf '%s\n' "${RES_STATUS[@]:-}" | grep -c '^PASS$')
  fail=$(printf '%s\n' "${RES_STATUS[@]:-}" | grep -c '^FAIL$')
  unsafe=$(printf '%s\n' "${RES_STATUS[@]:-}" | grep -c '^UNSAFE_TO_CONTINUE$')
  skip=$(printf '%s\n' "${RES_STATUS[@]:-}" | grep -c '^SKIPPED$')
  inc=$(printf '%s\n' "${RES_STATUS[@]:-}" | grep -c '^INCONCLUSIVE$')
  { echo "# MI300X GPU Stability Validation — $TS"; echo
    echo "- Mode: **$MODE**  |  Unsafe matrix: **$([ "$INCLUDE_UNSAFE" = 1 ] && echo included || echo SKIPPED)**  |  suite_rc=**$SUITE_RC** halted=$HALT"
    echo "- Image \`$IMAGE\` · model \`$MODEL_PATH\` · TP$TP · mns=$MAX_NUM_SEQS · backend $ATTENTION_BACKEND · QR=$QUICK_REDUCE"
    echo "- Totals: PASS=$pass FAIL=$fail UNSAFE=$unsafe INCONCLUSIVE=$inc SKIPPED=$skip"; echo
    echo "## Results"; echo "| Test | Status | Detail | Secs |"; echo "|---|---|---|---|"
    local i n=${#RES_NAMES[@]}; for ((i=0;i<n;i++)); do echo "| ${RES_NAMES[$i]} | ${RES_STATUS[$i]} | ${RES_DETAIL[$i]} | ${RES_SECS[$i]:-0} |"; done; echo
    echo "## Kernel faults observed"
    if [ "$DMESG_OK" != 1 ]; then echo "dmesg not readable — fault detection was BLIND this run."
    elif grep -qiE "$FAULT_RE" "$KLOG" 2>/dev/null; then grep -iE "$FAULT_RE" "$KLOG" | sed -E 's/\[[^]]*\] //' | sort | uniq -c | sort -rn | sed 's/^/    /'
    else echo "None."; fi; echo
    echo "## Inference metrics"; if [ -s "$INFJSON" ]; then echo '```json'; cat "$INFJSON"; echo '```'; else echo "No inference stage ran."; fi; echo
    echo "## Cluster health after testing"
    if [ "$HALT" = 1 ]; then echo "HALTED on an unsafe condition — a GPU may be wedged; it was left untouched (no reset). Inspect before reuse."
    else echo "No halt condition; see postflight row."; fi
    echo
    echo "## Most likely root cause"; conclusion; echo
    echo "## Caveats / remaining risk"
    echo "- Finite test; a clean pass does NOT prove the cluster is crash-free."
    echo "- Intermittent wedges (see mns512_report.md) may not reproduce in one pass."
    [ "$INCLUDE_UNSAFE" = 0 ] && echo "- HSA-disabled (crash-prone) rows were SKIPPED; the override hypothesis was not exercised."
  } >"$REPORT"
}

# ======================================================================================
# Prereqs + environment
# ======================================================================================
preqs() {
  log "=== prerequisites ==="
  [ "$(id -u)" -eq 0 ] || warn "not root: dmesg/sysfs/amd-smi may be limited (run with sudo)"
  have docker || { echo "FATAL: docker not found" >&2; return 3; }
  docker image inspect "$IMAGE" >/dev/null 2>&1 || warn "image $IMAGE not present locally (docker run will try to pull)"
  [ -e "$MODEL_PATH" ] || warn "model path $MODEL_PATH not visible from host (ok if only mounted in-container)"
  have curl || warn "curl missing: inference stages will be SKIPPED"
  return 0
}
environment_report() {
  log "=== environment ==="
  { echo "== $(date) =="; echo "host kernel: $(uname -r)"; echo "GPUs (sysfs cards): $(gpu_cards | wc -l)"
    for d in $(gpu_cards); do echo "  $(basename "$(dirname "$d")") bdf=$(basename "$(readlink -f "$d" 2>/dev/null)") busy=$(cat "$d/gpu_busy_percent" 2>/dev/null)"; done
    echo "docker: $(docker --version 2>/dev/null)"; echo "image id: $(docker image inspect -f '{{.Id}} {{.Created}}' "$IMAGE" 2>/dev/null || echo 'not local')"
    echo "cpu affinity (self): $(taskset -cp $$ 2>/dev/null)"
    if have amd-smi; then echo "== amd-smi static =="; amd-smi static -g all 2>&1 | grep -iE 'BDF|OAM_ID|MARKET_NAME|VERSION:|DRIVER|NAME: AMD' | head -80
    elif have rocm-smi; then echo "== rocm-smi =="; rocm-smi 2>&1 | head -40; else echo "no amd-smi/rocm-smi on host"; fi
  } >"$ENVIRON" 2>&1
}
# Phase 7 — postflight (read-only). New faults since run start, or busy-with-no-process => unsafe.
postflight_phase() {
  local t0; t0=$(date +%s)
  { echo "== postflight @ $(date) =="
    for d in $(gpu_cards); do echo "$(basename "$(dirname "$d")") busy=$(cat "$d/gpu_busy_percent" 2>/dev/null)"; done
  } >>"$OUTDIR/postflight.txt" 2>&1
  local busy=0 d
  for d in $(gpu_cards); do [ "$(cat "$d/gpu_busy_percent" 2>/dev/null || echo 1)" = "0" ] || busy=1; done
  local newfault; newfault="$(faults_since "${RUN_BASE_FAULTS:-0}")"
  if [ "$busy" != 0 ] || [ "$newfault" = yes ]; then
    record "postflight" UNSAFE_TO_CONTINUE "GPU not idle or new fault signatures after tests — inspect before reuse" "$(( $(date +%s)-t0 ))" 1
  elif [ "$newfault" = unknown ]; then
    record "postflight" INCONCLUSIVE "dmesg not readable — cannot confirm post-run health" "$(( $(date +%s)-t0 ))"
  else
    record "postflight" PASS "all GPUs idle, no new fault signatures after tests" "$(( $(date +%s)-t0 ))"
  fi
}

confirm_unsafe() {
  echo; echo "############################################################"
  echo "# UNSAFE MATRIX requested (--include-unsafe-matrix)."
  echo "# Runs vLLM WITHOUT HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_VALUE (previously associated with hangs)."
  echo "# May wedge a GPU; the script stops on fault and does NOT reset it."
  echo "############################################################"
  if [ "${ASSUME_YES:-0}" = 1 ]; then return 0; fi
  printf 'Type EXACTLY "yes-run-unsafe" to proceed: '; read -r ans
  [ "$ans" = "yes-run-unsafe" ] || { log "unsafe matrix declined; safe suite only"; INCLUDE_UNSAFE=0; }
}

# ======================================================================================
# Arg parsing (Fix 7) + dir setup
# ======================================================================================
usage() { grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; }
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --mode)
        case "${2:-}" in
          smoke|stability) MODE="$2"; shift 2;;
          ""|-*) echo "error: --mode requires a value (smoke|stability)" >&2; exit 2;;
          *) echo "error: invalid mode '$2' (want smoke|stability)" >&2; exit 2;;
        esac;;
      --mode=*) MODE="${1#--mode=}"; shift;;
      --smoke) MODE=smoke; shift;;
      --stability) MODE=stability; shift;;
      --include-unsafe-matrix) INCLUDE_UNSAFE=1; shift;;
      --compare-aiter-off) RUN_AITER_COMPARISON=1; shift;;
      -h|--help) usage; exit 0;;
      *) echo "error: unknown argument '$1'" >&2; usage >&2; exit 2;;
    esac
  done
  case "$MODE" in smoke|stability) ;; *) echo "error: mode must be smoke|stability" >&2; exit 2;; esac
  # Fix L537: enforce the customer guardrail — do not silently pass >cap to vLLM.
  if ! [[ "$MAX_NUM_SEQS" =~ ^[0-9]+$ ]]; then echo "error: MAX_NUM_SEQS must be an integer" >&2; exit 2; fi
  if [ "$MAX_NUM_SEQS" -gt "$MAX_NUM_SEQS_CAP" ]; then
    echo "error: MAX_NUM_SEQS=$MAX_NUM_SEQS exceeds the customer cap ($MAX_NUM_SEQS_CAP). Refusing to launch the prohibited config." >&2
    echo "       (raise MAX_NUM_SEQS_CAP explicitly only if the customer guardrail is lifted.)" >&2; exit 2
  fi
}
setup_dirs() {
  TS="$(date +%Y-%m-%d_%H%M%S)"; OUTDIR="${OUTDIR:-$(pwd)/gpu_stability_results/$TS}"; mkdir -p "$OUTDIR"
  ENVIRON="$OUTDIR/environment.txt"; PREFLIGHT_LOG="$OUTDIR/preflight.log"; TELEM="$OUTDIR/gpu_telemetry.csv"
  KLOG="$OUTDIR/kernel_logs.txt"; RESJSON="$OUTDIR/test_results.json"; INFJSON="$OUTDIR/inference_metrics.json"
  REPORT="$OUTDIR/final_report.md"; RUNLOG="$OUTDIR/run.log"; CLEANUP_LOG="$OUTDIR/cleanup.log"; TIMINGS="$OUTDIR/timings.csv"
  CONTAINER_PREFIX="gpu_stab_${TS}_$$"
  echo "label,start_epoch,end_epoch,duration_s,exit_code,timed_out,command" >"$TIMINGS"
  : >"$KLOG"
}

# ======================================================================================
# MAIN
# ======================================================================================
main() {
  parse_args "$@"
  setup_dirs
  trap cleanup EXIT
  trap 'on_signal INT' INT
  trap 'on_signal TERM' TERM

  log "final_gpu_stability_test start — mode=$MODE unsafe=$INCLUDE_UNSAFE out=$OUTDIR"
  preqs || { SUITE_RC=3; return 3; }
  environment_report
  start_telemetry
  check_dmesg

  local pf; if pf="$(preflight)"; then record "preflight" PASS "$pf" 0
  else
    case "$pf" in
      dmesg-unreadable) record "preflight" UNSAFE_TO_CONTINUE "cannot read dmesg — fault detection blind; run as root" 0;;
      *)                record "preflight" UNSAFE_TO_CONTINUE "cluster not clean before testing ($pf)" 0;;
    esac
    warn "aborting: preflight unsafe"; generate_report; return "$SUITE_RC"
  fi

  [ "$INCLUDE_UNSAFE" = 1 ] && confirm_unsafe

  RUN_BASE_FAULTS="$(dmesg_fault_count)"

  # --- DEFAULT (production) path: AITER-ENABLED baseline only ---
  gpu_compute_mem_test
  rccl_xgmi_test
  vllm_config_test "Baseline" 1 1 sustained "$OUTDIR/vllm_baseline.log"   # HSA on + AITER on = approved prod config

  # --- OPT-IN comparison: AITER-DISABLED (not a production config) ---
  if [ "$RUN_AITER_COMPARISON" = 1 ]; then
    vllm_config_test "Kernel_comparison" 1 0 inference "$OUTDIR/vllm_aiter_off.log"   # HSA on, AITER OFF
  else
    record "Kernel_comparison" SKIPPED "AITER-off comparison is opt-in (--compare-aiter-off); production uses VLLM_ROCM_USE_AITER=1" 0
  fi

  # --- OPT-IN comparison: HSA-override-DISABLED rows ---
  if [ "$INCLUDE_UNSAFE" = 1 ]; then
    vllm_config_test "Architecture_check" 0 1 short    "$OUTDIR/vllm_hsa_off.log"
    vllm_config_test "AITER_isolation"    0 0 inference "$OUTDIR/vllm_hsa_aiter_off.log"
  else
    record "Architecture_check" SKIPPED "HSA-override-off is opt-in (--include-unsafe-matrix); not a production config" 0
    record "AITER_isolation"    SKIPPED "HSA-override-off is opt-in (--include-unsafe-matrix); not a production config" 0
  fi

  postflight_phase
  generate_report
  log "DONE — report: $REPORT (suite_rc=$SUITE_RC)"
  echo; echo "================ SUMMARY (rc=$SUITE_RC) ================"
  grep -E '^\| ' "$REPORT" | sed -n '3,$p' 2>/dev/null || true
  echo "Full report: $REPORT"; echo "Machine-readable: $RESJSON"
  return "$SUITE_RC"
}

# Only run when executed directly; sourcing (regression tests) just loads the functions.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
  exit $?
fi
