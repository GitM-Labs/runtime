#!/usr/bin/env bash
# Verify the GITM infra layer. Runs every check that does not need a GPU, and
# prints the GPU-box checklist it cannot run here. Reports pass/fail per check
# and exits non-zero if any runnable check failed.
#
#   ./scripts/verify_infra.sh
#
# Run on a GPU dev box too: the Tier-3 section lists the checks to do there.

set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
fails=0
pass() { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fails=$((fails+1)); }
skip() { printf '  \033[33mSKIP\033[0m %s\n' "$1"; }
# Run each check in a subshell so a check's `cd`/`exit` can't leak into the script.
# Exit 0 = pass, 2 = skip (e.g. GPU check on a CPU box), anything else = fail.
check() {
  ( eval "$2" ) >/tmp/verify_infra.log 2>&1; local rc=$?
  if [ "$rc" -eq 0 ]; then pass "$1"
  elif [ "$rc" -eq 2 ]; then skip "$1"; sed 's/^/      /' </tmp/verify_infra.log | tail -1
  else fail "$1"; tail -5 /tmp/verify_infra.log | sed 's/^/      /'; fi
}

echo "==================== TIER 1: static / CI ===================="
check "pytest (full suite)"            "$PY -m pytest -q"
check "ruff (lint, new surface)"       "ruff check gitm/bench gitm/tracer/_cupti gitm/tracer/_cupti_decode.py gitm/tracer/cupti.py gitm/optimizer/apply.py benchmarks tests/test_bench.py tests/test_bench_datasets.py tests/test_cupti.py tests/test_hft_harness.py tests/test_framework_harnesses.py tests/test_apply_rollback.py"
check "import sanity (all new modules)" "$PY - <<'EOF'
import importlib
for m in ['gitm.bench.schema','gitm.bench.manifest','gitm.bench.baseline','gitm.bench.profile','gitm.bench.edge_manifest','gitm.bench.results','gitm.bench.runner','gitm.bench.reproduce','gitm.bench.cli','gitm.tracer.cupti','gitm.tracer._cupti_decode','gitm.optimizer.apply','gitm.benchmarks.hft.harness','gitm.runtime_driver','gitm.workloads','benchmarks.hft.generate','benchmarks.hft.harness','benchmarks.biotech.fetch','benchmarks.biotech.harness','benchmarks.edge.fetch','benchmarks.edge.harness','benchmarks.skeleton.measure_overhead']:
    importlib.import_module(m)
EOF"
check "intervention library validates" "$PY -c 'from gitm.kernels import load_library; assert len(load_library())==18'"
check "wheel build + data files"        "$PY -m build --wheel >/dev/null 2>&1 && $PY - <<'EOF'
import zipfile, glob
n = zipfile.ZipFile(sorted(glob.glob('dist/*.whl'))[-1]).namelist()
for f in ['gitm/bench/templates/results.md.j2','gitm/tracer/_cupti/cupti_shim.c','gitm/tracer/_cupti/cupti_core.c','gitm/tracer/_cupti/cupti_core.h','gitm/tracer/_cupti/cupti_inject.c','gitm/kernels/library.yaml']:
    assert f in n, f
EOF"
rm -rf dist build gitm.egg-info 2>/dev/null

# C shim syntax check, only if a compiler is available.
if command -v clang >/dev/null 2>&1; then
  STUB=$(mktemp -d); PYINC=$($PY -c 'import sysconfig; print(sysconfig.get_path("include"))')
  printf 'typedef int cudaError_t;\n#define cudaSuccess 0\ncudaError_t cudaGetDeviceCount(int*);\n' > "$STUB/cuda_runtime.h"
  cat > "$STUB/cupti.h" <<'EOF'
#include <stdint.h>
#define CUPTIAPI
typedef enum { CUPTI_SUCCESS=0, CUPTI_ERROR_MAX_LIMIT_REACHED=1 } CUptiResult;
typedef enum { CUPTI_ACTIVITY_KIND_MEMCPY=1, CUPTI_ACTIVITY_KIND_KERNEL=3, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL=10, CUPTI_ACTIVITY_KIND_SYNCHRONIZATION=30 } CUpti_ActivityKind;
typedef struct { CUpti_ActivityKind kind; } CUpti_Activity;
typedef struct { CUpti_ActivityKind kind; const char *name; uint64_t start,end; uint32_t deviceId,contextId,streamId,correlationId; int32_t gridX,gridY,gridZ,blockX,blockY,blockZ,staticSharedMemory,dynamicSharedMemory; uint16_t registersPerThread; } CUpti_ActivityKernel9;
typedef struct { CUpti_ActivityKind kind; uint8_t copyKind; uint64_t bytes,start,end; uint32_t deviceId,contextId,streamId,correlationId; } CUpti_ActivityMemcpy5;
typedef struct { CUpti_ActivityKind kind; uint32_t type; uint64_t start,end; uint32_t contextId,streamId,correlationId; } CUpti_ActivitySynchronization;
typedef void* CUcontext;
typedef void (*CUpti_BuffersCallbackRequestFunc)(uint8_t**,size_t*,size_t*);
typedef void (*CUpti_BuffersCallbackCompleteFunc)(CUcontext,uint32_t,uint8_t*,size_t,size_t);
CUptiResult cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc,CUpti_BuffersCallbackCompleteFunc);
CUptiResult cuptiActivityEnable(CUpti_ActivityKind); CUptiResult cuptiActivityDisable(CUpti_ActivityKind);
CUptiResult cuptiActivityFlushAll(uint32_t); CUptiResult cuptiActivityGetNextRecord(uint8_t*,size_t,CUpti_Activity**);
CUptiResult cuptiActivityFlushPeriod(uint32_t);
CUptiResult cuptiGetTimestamp(uint64_t*);
CUptiResult cuptiGetResultString(CUptiResult,const char**);
EOF
  CI="-I'$STUB' -Igitm/tracer/_cupti"
  check "CUPTI core compiles (stub headers)" "clang -fsyntax-only -Wall -Wextra $CI gitm/tracer/_cupti/cupti_core.c"
  check "CUPTI shim compiles (stub headers)" "clang -fsyntax-only -Wall -Wextra $CI -I'$PYINC' gitm/tracer/_cupti/cupti_shim.c"
  # The injection lib must never pull in libpython: the CUDA driver dlopens it into
  # arbitrary processes at CUDA init. Compiling it with NO Python include path is
  # the check — if someone adds #include <Python.h>, this fails.
  check "CUPTI injection lib compiles without libpython" \
    "clang -fsyntax-only -Wall -Wextra $CI gitm/tracer/_cupti/cupti_inject.c"
  rm -rf "$STUB"
fi

echo "==================== TIER 2: functional end-to-end (CPU) ===================="
for B in hft biotech edge; do
  check "smoke loop: $B (generate->freeze->verify->baseline->reproduce)" \
    "cd benchmarks/$B && export GITM_SCRATCH=\$(mktemp -d) && make smoke 2>&1 | grep -q 'REPRODUCIBLE'; rc=\$?; rm -f manifest.yaml manifest.jsonl results.md; rm -rf \$GITM_SCRATCH; exit \$rc"
done
check "embedded optimize() loop" "GITM_SCRATCH=\$(mktemp -d) $PY -c \"
from gitm import optimize; import pathlib
r=optimize(workload='vllm-decode', budget='1s', target=0.15)
assert pathlib.Path(r['summary']['report_path']).exists()\""
check "gitm doctor"  "GITM_SCRATCH=\$(mktemp -d) $PY -m gitm.cli doctor"
check "gitm.bench CLI help" "$PY -m gitm.bench --help"
check "overhead harness runs" "$PY -m benchmarks.skeleton.measure_overhead --runs 2 --steps 30"

echo "==================== TIER 3: GPU box (auto-detected) ===================="
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^/  GPU: /'
  check "CUPTI shim builds (real CUDA toolkit)" "$PY -m gitm.tracer._cupti.build"
  check "live CUPTI capture yields sane kernel events" "$PY scripts/gpu_live_capture.py"
  # HFT on cuDF/CuPy — only meaningful when those are installed.
  check "HFT harness runs on GPU (gpu_name != cpu)" "$PY - <<'EOF'
import sys
try:
    import cudf, cupy  # noqa: F401
except Exception:
    print('cuDF/CuPy not installed; harness would fall back to pandas'); sys.exit(2)
import tempfile, os, json, io, contextlib
from pathlib import Path
from benchmarks.hft.generate import GenConfig, generate
from benchmarks.hft import harness
stage = Path(tempfile.mkdtemp())
generate(GenConfig(events=200000, seed=42, events_per_file=200000), stage/'hft_x_seed42')
os.environ['GITM_BENCH_STAGE'] = str(stage)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    harness.main(['--seed','42','--warm-seconds','60'])
payload = json.loads(buf.getvalue().strip().splitlines()[-1])
assert payload['gpu_name'] != 'cpu', 'harness fell back to CPU: '+payload['gpu_name']
print('HFT on', payload['gpu_name'], '->', f\"{payload['metric_value']:.3g} events/s\")
EOF"
  echo "  Manual follow-ups (drive by hand — need weights/DBs/scale):"
  echo "    [ ] benchmarks/biotech,edge: fill load_*_runner bodies; run real inference"
  echo "    [ ] real 1B-event baselines: spread <2% and GPU-active <85% on hardware"
else
  echo "  No GPU detected — skipping. On a RunPod CUDA box, first run:"
  echo "    bash scripts/gpu_setup.sh   # install deps + build the CUPTI shim"
  echo "    ./scripts/verify_infra.sh   # this section then auto-runs"
fi

echo "==================== reproducibility fingerprint ===================="
$PY scripts/emit_report.py --out verify_report.json >/dev/null 2>&1 \
  && echo "  wrote verify_report.json (compare with: python scripts/compare_results.py reference.json verify_report.json)" \
  || echo "  WARN: could not write verify_report.json"

echo "============================================================"
if [ "$fails" -eq 0 ]; then
  printf '\033[32mAll runnable infra checks passed.\033[0m Tier 3 must be run on a GPU box.\n'
else
  printf '\033[31m%d check(s) failed.\033[0m See output above.\n' "$fails"
fi
exit "$fails"
