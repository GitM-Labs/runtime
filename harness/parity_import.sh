#!/usr/bin/env bash
# Manual hardware parity: same workload under gitm CUPTI capture vs external
# nsys / torch.profiler, then diff headroom numbers.
#
# Requires a GPU box with CUDA, nsys, and a torch+CUDA env. The synthetic
# unit-test parity (tests/test_importers.py::test_parity_metrics_and_headroom)
# stands in until this is run.
set -euo pipefail

OUT="${1:-/tmp/gitm_parity_import}"
mkdir -p "$OUT"

echo "== 1. Live gitm capture (CUPTI) =="
echo "    Run a short CUDA workload under: python scripts/gpu_live_capture.py"
echo "    Or: gitm run --workload <id>  (produces JSONL under \$GITM_SCRATCH)"
echo "    Copy the JSONL to: $OUT/live.jsonl"

echo "== 2. Nsight Systems on the same command =="
echo "    nsys profile -o $OUT/nsys --trace=cuda --force-overwrite=true -- <same-command>"
echo "    nsys export --type sqlite --output $OUT/nsys.sqlite $OUT/nsys.nsys-rep"

echo "== 3. Torch profiler on the same step (if applicable) =="
echo "    export_chrome_trace → $OUT/torch.json"

echo "== 4. Analyze external dumps =="
echo "    gitm analyze $OUT/nsys.sqlite $OUT/torch.json --out $OUT/import_report.md --json $OUT/import.json --sku \"\$GITM_GPU_SKU\""

echo "== 5. Diff =="
echo "    Compare busy_fraction, stall_fraction, ceiling_distance from:"
echo "      - live path (compute_metrics + build_headroom on live.jsonl)"
echo "      - $OUT/import.json"
echo "    Expect close agreement on timing-derived metrics; MBU/HFU and"
echo "    confidence will differ (imports are trace-only)."

echo "Documented procedure only — not executed by CI."
