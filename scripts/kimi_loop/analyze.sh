#!/bin/bash
# Pull one run's results off the cluster and run the analysis — laptop side.
#
#   scripts/kimi_loop/analyze.sh <run-id>        # e.g. 20260911-081500
#
# Then E7 (deviation) per captured point, which needs the point's batch and
# kv-len — the examples below match run_loop.sh's fixed points.
set -euo pipefail

RUN_ID="${1:?usage: analyze.sh <run-id>  (ls /mnt/shared/gitm/results in the sidecar)}"
LOCAL="evidence/kimi-mi355x/runs/$RUN_ID"
mkdir -p "$LOCAL"

POD=$(kubectl get pods -l app=kimi-loop -o name | head -1 | cut -d/ -f2)
echo "==> pulling results/$RUN_ID from $POD"
# Exclude the 1 Hz telemetry stream and the multi-hundred-MB trace shards:
# they choke the exec tar stream and analyze_join reads neither for the
# headline tables. Pull traces separately and targeted when needed.
kubectl exec "$POD" -c gitm -- tar -C /mnt/shared/gitm/results \
  --exclude='*.jsonl' --exclude='*.jsonl.*' -cf - "$RUN_ID" \
  | tar -C "$(dirname "$LOCAL")" -xf -

python3 scripts/kimi_loop/analyze_join.py "$LOCAL"

echo
echo "==> E7 deviation commands (run the ones whose captures exist):"
cat <<EOF
# E1/E2 traced midpoint: rag c=64, decode-heavy window
python -m gitm.cli deviate \$(ls $LOCAL/e2/cap_B_rag_c64/**/*.jsonl | tail -1) \\
    --model kimi-k2.5 --gpu MI355X --tp 8 --batch 64 --kv-len 4352 --json \\
    > $LOCAL/analysis/deviation_rag_c64.json

# E3/E4 saturated decode window: chat-long c=256
python -m gitm.cli deviate \$(ls $LOCAL/e3e4/capture/**/*.jsonl | tail -1) \\
    --model kimi-k2.5 --gpu MI355X --tp 8 --batch 256 --kv-len 1536 --json \\
    > $LOCAL/analysis/deviation_saturated.json
EOF
