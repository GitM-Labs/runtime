#!/bin/bash
# Extend each intervention to the chat and long length regimes at c=64
# (rag c=64 already collected under the label root). Run INSIDE an intervention
# pod's gitm sidecar; self-waits for the server. Servers already serving
# (tracer cleared), so this starts immediately.
#
#   setsid nohup bash /mnt/shared/gitm/scripts/tp_run_lengths.sh < /dev/null \
#     > /scratch/logs/tp_len.log 2>&1 &
set -uo pipefail

RUN="${GITM_RUN:-20260911-loop1}"
EP=http://localhost:8000
MODEL=moonshotai/Kimi-K2.5

echo "[$(date -u +%H:%M:%S)] waiting for server /health ..."
until curl -sf -o /dev/null "$EP/health"; do sleep 15; done
L=$(grep -oE 'GITM_ARM=[^ ]*' /scratch/arm.env | head -1 | cut -d= -f2)
L="${L:-unknown}"
echo "[$(date -u +%H:%M:%S)] intervention=$L"

for spec in "chat:1024:256" "long:8192:1024"; do
  IFS=: read -r cfg pin pout <<< "$spec"
  D="/mnt/shared/gitm/results/$RUN/interventions/$L/$cfg"
  mkdir -p "$D"
  for r in 1 2 3; do
    echo "[$(date -u +%H:%M:%S)] $L $cfg rep $r"
    guidellm run \
      --backend "kind=openai_http,target=$EP/v1,model=$MODEL,max_tokens=$pout" \
      --profile "kind=concurrent,streams=64,warmup=0.1" \
      --constraint "kind=max_duration,seconds=120" \
      --data "kind=synthetic_text,prompt_tokens=$pin,output_tokens=$pout" \
      --output "kind=json,path=$D/guidellm_r${r}.json" || echo "rep $r failed"
  done
done
echo "done $(date -u +%FT%TZ)" > "/mnt/shared/gitm/results/$RUN/interventions/$L/DONE_LENGTHS"
echo "[$(date -u +%H:%M:%S)] $L chat+long complete"
