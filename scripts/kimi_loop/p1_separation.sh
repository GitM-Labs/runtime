#!/bin/bash
# P1 separating experiment: is fp8 KV's effect on MLA decode attention a
# slowdown/speedup (multiplicative) or a fixed cost (additive)?
# Spec: docs/experiments/p1_fp8_kv_separation.md. Run INSIDE the gitm sidecar.
#
#   bash /mnt/shared/gitm/scripts/p1_separation.sh traced   # node 1: arms B, I, B
#   bash /mnt/shared/gitm/scripts/p1_separation.sh off      # node 2: arms A, L
#
# Both nodes share one run id (GITM_RUN) so the analysis finds both halves:
#   python -m gitm.optimizer.separation analyze <pulled results/$GITM_RUN>
#
# Order inside a phase is the same shuffled order on both nodes; the traced node
# re-runs the baseline after the candidate (base2) to measure drift.
set -euo pipefail

NODE="${1:?usage: p1_separation.sh traced|off}"
GITM_RUN="${GITM_RUN:?set GITM_RUN to a shared run id, e.g. 20260929-p1}"
RUN="/mnt/shared/gitm/results/$GITM_RUN"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
MODEL=moonshotai/Kimi-K2.5
EP=http://localhost:8000
LEVER="--kv-cache-dtype fp8"
C=16                 # offered concurrency (decode batch)
OUT_TOK=256          # decode length: kv varies by at most this within a point
LOAD_S=150           # GuideLLM duration per point
RAMP_S=40            # let every stream finish its first prefill before capturing
WINDOW_S=30          # capture window: ~60k attention launches at c=16
# Pre-registered, fixed orders (one per rep). Never edit after the first run.
ORDERS=("8192 2048 32768 4096 16384"
        "16384 32768 4096 2048 8192"
        "4096 16384 2048 8192 32768")

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
# On any exit, stop background scrapers and load generators; the scraper loops forever.
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

manifest() {  # manifest <dir> <label>
  {
    echo "node=$NODE label=$2 ts=$(date -u +%FT%TZ) c=$C out=$OUT_TOK"
    echo "rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo '?')"
    grep -h '^export' /scratch/arm.env
  } >> "$1/MANIFEST"
}

guide() {  # guide <outfile> <prompt_tokens>
  guidellm run \
    --backend "kind=openai_http,target=$EP/v1,model=$MODEL,max_tokens=$OUT_TOK" \
    --profile "kind=concurrent,streams=$C,warmup=0.1,cooldown=0.1" \
    --constraint "kind=max_duration,seconds=$LOAD_S" \
    --data "kind=synthetic_text,prompt_tokens=$2,output_tokens=$OUT_TOK" \
    --output "kind=json,path=$1"
}

scrape() {
  while :; do
    printf '### ts_ns=%s\n' "$(date +%s%N)"
    curl -sf "$EP/metrics" | grep -E 'vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|num_preemptions)' || true
    sleep 1
  done >> "$1"
}

phase() {  # phase <node-dir> <phase> <arm> [lever...]
  local node_dir="$1" name="$2" arm="$3"; shift 3
  local P="$RUN/p1/$node_dir/$name"; mkdir -p "$P"
  log "== $node_dir/$name: arm $arm $*"
  bash "$SCRIPTS/arm.sh" "$arm" "$@"
  manifest "$P" "$name"
  # Correctness gate: stop here rather than spend the phase on a broken arm.
  python -m gitm.optimizer.separation sanity --model "$MODEL" --out "$P/sanity.json" \
    || { log "!!! correctness gate failed for $name — stopping"; exit 1; }
  log "warm-up"
  guide "$P/warmup.json" 2048 >/dev/null || log "warm-up load failed (not fatal)"
  for rep in 1 2 3; do
    for L in ${ORDERS[$((rep - 1))]}; do
      local D="$P/L${L}_r${rep}"; mkdir -p "$D"
      log "$name L=$L rep=$rep"
      scrape "$D/metrics.prom" & local S=$!
      guide "$D/guidellm.json" "$L" & local G=$!
      if [ "$node_dir" = traced ]; then
        sleep "$RAMP_S"
        # A failed window is recorded, not fatal: the analysis flags the missing
        # trace and returns inconclusive rather than losing the rest of the run.
        python -m gitm.cli capture attach --port 8000 --duration "$WINDOW_S" --out "$D/cap" \
          || { log "!!! capture failed at $name L=$L rep=$rep"; echo "$name L=$L rep=$rep" >> "$P/FAILED"; }
      fi
      wait "$G" || { log "!!! guidellm failed at $name L=$L rep=$rep"; echo "$name L=$L rep=$rep guidellm" >> "$P/FAILED"; }
      kill "$S" 2>/dev/null || true
    done
  done
}

case "$NODE" in
  traced)
    phase traced base1 B
    phase traced cand  I $LEVER
    phase traced base2 B
    ;;
  off)
    phase off base1 A
    phase off cand  L $LEVER
    ;;
  *) echo "unknown node: $NODE" >&2; exit 2 ;;
esac
cp /scratch/telemetry/amdsmi.jsonl "$RUN/p1/amdsmi.$NODE.jsonl" 2>/dev/null || true
log "p1 $NODE complete -> $RUN/p1"
