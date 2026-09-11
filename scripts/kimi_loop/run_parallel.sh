#!/bin/bash
# One-hour parallel loop: per-pod workload, run INSIDE that pod's gitm sidecar.
#
#   bash /mnt/shared/gitm/scripts/run_parallel.sh <role>
#
# roles (one per pod, arms baked at pod birth — this script never switches arms):
#   sweep-chat   arm A: E2 chat 1024/256, c in {1,4,16,64,128,256} + E1 A reps
#   sweep-rag    arm A: E2 rag 4096/512, same grid
#   sweep-long   arm A: E2 long 8192/1024, same grid
#   traced       arm B: E1 B reps + the E7 capture window at rag c=64
#   correlate    arm C (the ORIGINAL kimi-k25-loop pod): E0 gate then E3/E4
#   intervene    arm I (fp8 KV): E8 after-arm reps at rag c=64
#
# Compressed timings for the one-hour budget: 120 s per sweep point, 3x120 s
# reps for overhead arms. All results share one run id on /mnt/shared.
set -euo pipefail

ROLE="${1:?usage: run_parallel.sh <role>}"
export ROCP_TOOL_LIBRARIES=/scratch/lib/libgitm_rocm_inject.so
export GITM_TRACE_OUT=${GITM_TRACE_OUT:-/scratch/trace/kimi.jsonl}
GITM_RUN="${GITM_RUN:-20260911-loop1}"
RUN="/mnt/shared/gitm/results/$GITM_RUN"
MODEL=moonshotai/Kimi-K2.5
EP=http://localhost:8000
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RUN"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

manifest() {
  {
    echo "role=$ROLE label=$1 ts=$(date -u +%FT%TZ)"
    echo "rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo '?')"
    grep -h '^export' /scratch/arm.env
  } >> "$RUN/MANIFEST.$ROLE"
}

guide() {  # guide <outfile> <streams> <prompt> <output> <seconds>
  guidellm run \
    --backend "kind=openai_http,target=$EP/v1,model=$MODEL,max_tokens=$4" \
    --profile "kind=concurrent,streams=$2,warmup=0.1,cooldown=0.1" \
    --constraint "kind=max_duration,seconds=$5" \
    --data "kind=synthetic_text,prompt_tokens=$3,output_tokens=$4" \
    --output "kind=json,path=$1"
}

scrape() {
  while :; do
    printf '### ts_ns=%s\n' "$(date +%s%N)"
    curl -sf "$EP/metrics" | grep -E 'vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|num_preemptions)' || true
    sleep 1
  done >> "$1"
}

capture() { python -m gitm.cli capture attach --port 8000 --duration "$2" --out "$1"; }

sweep() {  # sweep <name> <prompt> <output>
  local D="$RUN/e2"; mkdir -p "$D"
  manifest "sweep $1"
  for c in 1 4 16 64 128 256; do
    log "E2 $1 c=$c"
    scrape "$D/metrics_$1_c${c}.prom" & S=$!
    guide "$D/guidellm_$1_c${c}.json" "$c" "$2" "$3" 120
    kill "$S" 2>/dev/null || true
  done
}

case "$ROLE" in
  sweep-chat)
    sweep chat 1024 256
    # E1 clean reps ride on this pod after its sweep.
    D="$RUN/e1"; mkdir -p "$D"; manifest "e1 arm A"
    for r in 1 2 3; do
      log "E1 A rep $r"
      guide "$D/guidellm_A_r${r}.json" 64 1024 256 120
    done
    ;;
  sweep-rag) sweep rag 4096 512 ;;
  sweep-long) sweep long 8192 1024 ;;
  traced)
    D="$RUN/e1"; mkdir -p "$D"; manifest "e1 arm B"
    for r in 1 2 3; do
      log "E1 B rep $r (collecting)"
      capture "$D/cap_B_r${r}" 130 & C=$!
      sleep 3
      guide "$D/guidellm_B_r${r}.json" 64 1024 256 120
      wait "$C" || true
    done
    # The E7 window: traced capture under the headline sweep point.
    D="$RUN/e2"; mkdir -p "$D"; manifest "e7 window rag c=64"
    capture "$D/cap_B_rag_c64" 130 & C=$!
    sleep 3
    guide "$D/guidellm_B_rag_c64.json" 64 4096 512 120
    wait "$C" || true
    ;;
  correlate)
    # The original pod, already arm C eager. E0 gates, then the layer capture.
    env GITM_RUN="$GITM_RUN" bash "$SCRIPTS/run_loop.sh" e0
    env GITM_RUN="$GITM_RUN" bash "$SCRIPTS/run_loop.sh" e3e4
    ;;
  intervene)
    D="$RUN/e8"; mkdir -p "$D"; manifest "e8 fp8 kv (after-arm)"
    echo "--kv-cache-dtype fp8" > "$D/LEVER"
    for r in 1 2 3; do
      log "E8 I rep $r"
      guide "$D/guidellm_after_r${r}.json" 64 4096 512 120
    done
    # Also one traced window so E8's delta carries a kernel-level explanation.
    capture "$D/cap_I" 130 & C=$!
    sleep 3
    guide "$D/guidellm_after_traced.json" 64 4096 512 120
    wait "$C" || true
    ;;
  bench)
    # Generic intervention arm: the pod was born with a lever baked into its
    # arm.env; GITM_ARM is its label. Run the headline point (rag c=64) three
    # times plus one traced window, named by the label, so every intervention
    # is directly comparable to the traced-pod baseline at the same point.
    LABEL="${GITM_ARM:-unknown}"
    D="$RUN/interventions/$LABEL"; mkdir -p "$D"
    manifest "intervention $LABEL"
    grep -h GITM_EXTRA_VLLM_ARGS /scratch/arm.env > "$D/LEVER" 2>/dev/null || true
    for r in 1 2 3; do
      log "intervention $LABEL rep $r"
      guide "$D/guidellm_r${r}.json" 64 4096 512 120
    done
    capture "$D/cap" 130 & C=$!
    sleep 3
    guide "$D/guidellm_traced.json" 64 4096 512 120
    wait "$C" || true
    ;;
  *) echo "unknown role: $ROLE"; exit 2 ;;
esac
cp /scratch/telemetry/amdsmi.jsonl "$RUN/amdsmi.$ROLE.jsonl" 2>/dev/null || true
log "role $ROLE complete -> $RUN"
