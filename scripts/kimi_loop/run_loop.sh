#!/bin/bash
# The Kimi K2.5 / MI355X loop — execution side. Run INSIDE the gitm sidecar:
#
#   kubectl exec -it deploy/kimi-k25-loop -c gitm -- bash /mnt/shared/gitm/scripts/run_loop.sh <phase>
#
# Phases (docs/mi355x_experiment_plan.md numbering):
#   e0      smoke + clock domain (arm C, one windowed capture, hard gates)
#   e1      tracer overhead: arms A/B/C x fixed GuideLLM load x 3 reps
#   e2      concurrency x length sweep (arm A; one B window at the midpoint)
#   e3e4    layer-by-layer kernels: arm C, saturated decode, windowed capture
#   burst   BurstGPT replay (arm A + one B window) via gitm.traffic
#   e8      intervention arm: re-run the headline point under $INTERVENTION
#   all     e0 e1 e2 e3e4 burst
#
# Results land under /mnt/shared/gitm/results/$GITM_RUN (survives the pod).
# Every phase writes MANIFEST lines: arm, image digest, rocm version, ts —
# the standing rule is no anonymous numbers.
#
# The predicted side of the loop (scripts/kimi_loop/predict_sweep.py) and the
# deviation/report side (scripts/kimi_loop/analyze.sh) run on the laptop; this
# script only touches the server, the load generators, and the tracer.
set -euo pipefail

PHASE="${1:-all}"
# The sidecar runs no GPU code, but capture windowing (injection.clock_now)
# and shard handling dispatch on active_vendor(), which reads THIS process's
# environment. Without the hook variable the vendor resolves to None, the
# window clock reads None, and every capture comes back empty with no error —
# the exact silent failure docs/rocm.md warns about.
export ROCP_TOOL_LIBRARIES=/scratch/lib/libgitm_rocm_inject.so
export GITM_TRACE_OUT=${GITM_TRACE_OUT:-/scratch/trace/kimi.jsonl}
BASE=/mnt/shared/gitm/results
GITM_RUN="${GITM_RUN:-$(date -u +%Y%m%d-%H%M%S)}"
RUN="$BASE/$GITM_RUN"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
MODEL=moonshotai/Kimi-K2.5
EP=http://localhost:8000
mkdir -p "$RUN"

# Mirror of predict_sweep.py LENGTH_CONFIGS / CONCURRENCY — the join key.
LENGTH_CONFIGS="chat:1024:256 rag:4096:512 long:8192:1024"
CONCURRENCY="1 4 16 64 128 256"
HEADLINE="rag:4096:512"; HEADLINE_C=64

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

manifest() {  # manifest <phase> <arm> <label>
  {
    echo "phase=$1 arm=$2 label=$3 ts=$(date -u +%FT%TZ)"
    echo "rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo '?')"
    echo "vllm=$(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
    grep -h '^export' /scratch/arm.env
  } >> "$RUN/MANIFEST"
}

guide() {  # guide <outfile> <streams> <prompt> <output> <seconds>
  guidellm run \
    --backend "kind=openai_http,target=$EP/v1,model=$MODEL,max_tokens=$4" \
    --profile "kind=concurrent,streams=$2,warmup=0.1,cooldown=0.1" \
    --constraint "kind=max_duration,seconds=$5" \
    --data "kind=synthetic_text,prompt_tokens=$3,output_tokens=$4" \
    --output "kind=json,path=$1"
}

scrape_metrics() {  # scrape_metrics <outfile> — 1 Hz until killed
  while :; do
    printf '### ts_ns=%s\n' "$(date +%s%N)"
    curl -sf "$EP/metrics" | grep -E 'vllm:(num_requests_running|num_requests_waiting|gpu_cache_usage_perc|num_preemptions)' || true
    sleep 1
  done >> "$1"
}

no_shard_growth_check() {  # standing rule: shards must not grow outside windows
  local before after
  before=$(stat -c %s /scratch/trace/kimi.jsonl.* 2>/dev/null | awk '{s+=$1} END {print s+0}')
  sleep 10
  after=$(stat -c %s /scratch/trace/kimi.jsonl.* 2>/dev/null | awk '{s+=$1} END {print s+0}')
  if [ "$before" != "$after" ]; then
    log "!!! shards grew outside a capture window ($before -> $after) — A-arm numbers are suspect"
    echo "shard_growth_outside_window=true" >> "$RUN/MANIFEST"
  fi
}

capture_window() {  # capture_window <outdir> <duration_s>  (observe mode)
  python -m gitm.cli capture attach --port 8000 --duration "$2" --out "$1"
}

e0() {
  log "E0: smoke + clock domain (arm C)"
  bash "$SCRIPTS/arm.sh" C
  manifest e0 C smoke
  local D="$RUN/e0"; mkdir -p "$D"
  python -m gitm.cli capture attach --port 8000 --requests 8 --concurrency 2 \
    --input-tokens 256 --output-tokens 64 --out "$D/capture"
  python - "$D/capture" <<'PY'
import json, glob, sys
d = sys.argv[1]
traces = sorted(glob.glob(d + "/**/*.jsonl", recursive=True))
assert traces, f"E0 FAIL: no merged trace under {d}"
ev = [json.loads(l) for l in open(traces[-1])]
kernels = [e for e in ev if e.get("kind") == "kernel"]
meta = [e for e in ev if e.get("kind") == "meta"]
dropped = sum(m.get("dropped_records", 0) for m in meta)
named = [k for k in kernels if k.get("name") and "unknown" not in str(k.get("name")).lower()]
ranged = [k for k in kernels if k.get("range_op")]
print(f"events={len(ev)} kernels={len(kernels)} named={len(named)} "
      f"range_op={len(ranged)} dropped={dropped}")
assert kernels, "E0 FAIL: zero kernel events — tool not loading or window empty (clock domain?)"
assert named, "E0 FAIL: kernels are anonymous — code-object callback not resolving names"
assert dropped == 0, f"E0 FAIL: {dropped} dropped records"
if not ranged:
    print("E0 WARN: no range_op — layerwise rocTX not reaching the tool; "
          "grep the server log for layerwise flag acceptance before running E3/E4")
print("E0 PASS")
PY
  log "E0 done -> $D"
}

e1() {
  log "E1: tracer overhead, arms A/B/C x 3 reps (fixed load: chat c=$HEADLINE_C)"
  local D="$RUN/e1"; mkdir -p "$D"
  for arm in A B C; do
    bash "$SCRIPTS/arm.sh" "$arm"
    manifest e1 "$arm" overhead
    [ "$arm" = A ] && no_shard_growth_check
    for rep in 1 2 3; do
      log "E1 arm=$arm rep=$rep"
      if [ "$arm" != A ]; then
        # The tracer must be COLLECTING during B/C — dormant-but-injected
        # would measure arm A twice and read as "tracing is free".
        capture_window "$D/cap_${arm}_r${rep}" 190 &
        CAPPID=$!
        sleep 3
      fi
      guide "$D/guidellm_${arm}_r${rep}.json" "$HEADLINE_C" 1024 256 180
      [ "$arm" != A ] && wait "$CAPPID" || true
    done
  done
  log "E1 done -> $D  (analyze.sh renders the overhead table)"
}

e2() {
  log "E2: concurrency x length sweep (arm A)"
  local D="$RUN/e2"; mkdir -p "$D"
  bash "$SCRIPTS/arm.sh" A
  manifest e2 A sweep
  no_shard_growth_check
  for cfg in $LENGTH_CONFIGS; do
    IFS=: read -r name prompt output <<< "$cfg"
    for c in $CONCURRENCY; do
      log "E2 $name c=$c"
      scrape_metrics "$D/metrics_${name}_c${c}.prom" & SCRAPE=$!
      guide "$D/guidellm_${name}_c${c}.json" "$c" "$prompt" "$output" 240
      kill "$SCRAPE" 2>/dev/null || true
    done
  done
  # One traced point mid-grid so E7 has a sweep-conditions capture.
  bash "$SCRIPTS/arm.sh" B
  manifest e2 B "traced midpoint $HEADLINE c=$HEADLINE_C"
  IFS=: read -r name prompt output <<< "$HEADLINE"
  capture_window "$D/cap_B_${name}_c${HEADLINE_C}" 190 & CAPPID=$!
  sleep 3
  guide "$D/guidellm_B_${name}_c${HEADLINE_C}.json" "$HEADLINE_C" "$prompt" "$output" 180
  wait "$CAPPID" || true
  log "E2 done -> $D"
}

e3e4() {
  log "E3/E4: layer-by-layer kernel attribution (arm C, saturated decode)"
  local D="$RUN/e3e4"; mkdir -p "$D"
  bash "$SCRIPTS/arm.sh" C
  manifest e3e4 C "saturated decode chat c=256"
  # Background load; let the prefill wave pass, then window pure decode.
  guide "$D/guidellm_load.json" 256 1024 1024 420 & LOAD=$!
  sleep 90
  capture_window "$D/capture" 120
  wait "$LOAD" || true
  # Sanity inline; full taxonomy/attribution tables come from analyze.sh.
  python - "$D/capture" <<'PY'
import json, glob, sys
from collections import Counter
d = sys.argv[1]
traces = sorted(glob.glob(d + "/**/*.jsonl", recursive=True))
ev = [json.loads(l) for l in open(traces[-1])]
k = [e for e in ev if e.get("kind") == "kernel"]
attr = [e for e in k if e.get("range_op")]
lay = Counter(e.get("range_layer") for e in attr)
t_total = sum(e.get("end_ns", 0) - e.get("start_ns", 0) for e in k)
t_attr = sum(e.get("end_ns", 0) - e.get("start_ns", 0) for e in attr)
print(f"kernels={len(k)} attributed={len(attr)} layers_seen={len(lay)} "
      f"kernel_time_attributed={t_attr / max(t_total, 1):.1%}")
PY
  log "E3/E4 done -> $D"
}

burst() {
  log "BurstGPT replay (arm A + one B window)"
  local D="$RUN/burstgpt"; mkdir -p "$D"
  local CSV=/mnt/shared/gitm/burstgpt/BurstGPT_1.csv
  if [ ! -f "$CSV" ]; then
    mkdir -p "$(dirname "$CSV")"
    curl -fL -o "$CSV" \
      "${BURSTGPT_URL:-https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_1.csv}" \
      || { log "!!! BurstGPT download failed — set BURSTGPT_URL and re-run"; return 1; }
  fi
  python -m gitm.traffic --describe burstgpt "$CSV" | tee "$D/describe.txt"
  bash "$SCRIPTS/arm.sh" A
  manifest burst A replay
  python -m gitm.traffic --replay burstgpt "$CSV" --out "$D/replay.jsonl" \
    --model "$MODEL" --fire --result-dir "$D/arm_A" | tee "$D/replay_A.log"
  bash "$SCRIPTS/arm.sh" B
  manifest burst B replay+window
  capture_window "$D/cap_B" 190 & CAPPID=$!
  sleep 3
  python -m gitm.traffic --replay burstgpt "$CSV" --out "$D/replay.jsonl" \
    --model "$MODEL" --fire --result-dir "$D/arm_B" | tee "$D/replay_B.log"
  wait "$CAPPID" || true
  log "BurstGPT done -> $D"
}

e8() {
  local LEVER="${INTERVENTION:?set INTERVENTION='--kv-cache-dtype fp8' (or another lever) for e8}"
  log "E8: intervention arm: $LEVER (headline point, before/after)"
  local D="$RUN/e8"; mkdir -p "$D"
  IFS=: read -r name prompt output <<< "$HEADLINE"
  bash "$SCRIPTS/arm.sh" B
  manifest e8 B "baseline for intervention"
  capture_window "$D/cap_before" 190 & CAPPID=$!; sleep 3
  guide "$D/guidellm_before.json" "$HEADLINE_C" "$prompt" "$output" 180
  wait "$CAPPID" || true
  bash "$SCRIPTS/arm.sh" I $LEVER
  manifest e8 I "$LEVER"
  capture_window "$D/cap_after" 190 & CAPPID=$!; sleep 3
  guide "$D/guidellm_after.json" "$HEADLINE_C" "$prompt" "$output" 180
  wait "$CAPPID" || true
  echo "$LEVER" > "$D/LEVER"
  log "E8 done -> $D"
}

log "run id: $GITM_RUN -> $RUN"
case "$PHASE" in
  e0) e0 ;;
  e1) e1 ;;
  e2) e2 ;;
  e3e4) e3e4 ;;
  burst) burst ;;
  e8) e8 ;;
  all) e0; e1; e2; e3e4; burst ;;
  *) echo "unknown phase: $PHASE (e0|e1|e2|e3e4|burst|e8|all)"; exit 2 ;;
esac
# Telemetry slice for the whole run, whatever the phase.
cp /scratch/telemetry/amdsmi.jsonl "$RUN/amdsmi.jsonl" 2>/dev/null || true
log "phase '$PHASE' complete. Results: $RUN"
