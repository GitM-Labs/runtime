#!/usr/bin/env bash
# Live on-node smoke for gitm-node-health (no Slurm submit, no fault injection).
#
#   ./health-check.sh              # hardware-only; no serve expected
#   ./health-check.sh --require-healthy  # also require a live vLLM process
#
# Run on an H200 node after setup-node.sh has installed gitm-node-health.
set -euo pipefail

JOB="health-check-$$"
OUT_DIR="/tmp/gitm-health-check"
WAIT_S=15
REQUIRE_HEALTHY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --require-healthy) REQUIRE_HEALTHY=1; shift ;;
    --wait) WAIT_S="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --job) JOB="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

HEALTH_BIN=$(command -v gitm-node-health 2>/dev/null || true)
HEALTH_BIN=${HEALTH_BIN:-/usr/local/bin/gitm-node-health}
[[ -x "$HEALTH_BIN" ]] || {
  echo "gitm-node-health not installed; run: sudo ./setup-node.sh --skip-image" >&2
  exit 1
}

OUT="$OUT_DIR/node_health.jsonl"
WIN="$OUT_DIR/node_health_window.json"
mkdir -p "$OUT_DIR"
rm -f "$OUT" "$OUT.sampler.pid" "$WIN"

"$HEALTH_BIN" --print-thresholds >/dev/null
echo "pre --job $JOB (sampler ~${WAIT_S}s)…"
HEALTH_ARGS=()
[[ "$REQUIRE_HEALTHY" -eq 1 ]] && HEALTH_ARGS+=(--require-vllm)
"$HEALTH_BIN" pre --job "$JOB" --out "$OUT" "${HEALTH_ARGS[@]}"
sleep "$WAIT_S"
"$HEALTH_BIN" post --job "$JOB" --out "$OUT" "${HEALTH_ARGS[@]}"
set +e
"$HEALTH_BIN" summarize --job "$JOB" --out "$OUT" --window "$WIN"
set -e

echo "--- $WIN ---"
cat "$WIN"

# Fail on every required signal; vLLM is required only with --require-healthy.
python3 - "$WIN" <<'PY'
import json, sys

win = json.load(open(sys.argv[1]))
failed = set(win.get("failed_signals") or [])

if int(win.get("n_samples") or 0) < 1:
    sys.exit("health-check: no samples (monitoring_coverage / wait longer)")

unexpected = sorted(failed)
if unexpected:
    sys.exit(f"health-check: unexpected failures {unexpected}")

print(
    "health-check: ok",
    f"healthy={win.get('healthy')}",
    f"failed_signals={sorted(failed)}",
    f"n_samples={win.get('n_samples')}",
)
PY
