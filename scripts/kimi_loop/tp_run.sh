#!/bin/bash
# Throughput-only intervention run — no capture window, so the tracer stays
# dormant and cannot livelock the server (the failure the bench role hit).
# Run INSIDE an intervention pod's gitm sidecar; it self-waits for the server.
#
#   setsid nohup bash /mnt/shared/gitm/scripts/tp_run.sh < /dev/null \
#     > /scratch/logs/tp.log 2>&1 &
#
# Reads the pod's own arm label + lever from /scratch/arm.env, runs the
# headline point (rag c=64) three times, writes to interventions/<label>/.
set -uo pipefail

RUN="${GITM_RUN:-20260911-loop1}"
EP=http://localhost:8000
MODEL=moonshotai/Kimi-K2.5

echo "[$(date -u +%H:%M:%S)] waiting for server /health ..."
until curl -sf -o /dev/null "$EP/health"; do sleep 15; done
echo "[$(date -u +%H:%M:%S)] server up"

L=$(grep -oE 'GITM_ARM=[^ ]*' /scratch/arm.env | head -1 | cut -d= -f2)
L="${L:-unknown}"
D="/mnt/shared/gitm/results/$RUN/interventions/$L"
mkdir -p "$D"
grep GITM_EXTRA_VLLM_ARGS /scratch/arm.env > "$D/LEVER" 2>/dev/null || true
echo "[$(date -u +%H:%M:%S)] intervention=$L -> $D"

for r in 1 2 3; do
  echo "[$(date -u +%H:%M:%S)] $L rep $r"
  guidellm run \
    --backend "kind=openai_http,target=$EP/v1,model=$MODEL,max_tokens=512" \
    --profile "kind=concurrent,streams=64,warmup=0.1" \
    --constraint "kind=max_duration,seconds=120" \
    --data "kind=synthetic_text,prompt_tokens=4096,output_tokens=512" \
    --output "kind=json,path=$D/guidellm_r${r}.json" || echo "rep $r failed"
done
echo "done $(date -u +%FT%TZ)" > "$D/DONE"
echo "[$(date -u +%H:%M:%S)] $L complete"
