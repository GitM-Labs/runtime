#!/bin/bash
# Switch the serving arm — run INSIDE the gitm sidecar of kimi-k25-loop.
#
#   arm.sh A|B|C|I [extra vllm args for I]
#
#   A  clean:      no tool loaded — the headline-numbers arm
#   B  traced:     GITM tracer injected, dormant until a window is armed
#   C  correlate:  B + rocTX correlation + vLLM layerwise ranges
#   I  intervene:  B + caller-supplied vllm args (e.g. --kv-cache-dtype fp8) —
#                  the E8 intervention arm; pass the lever as $2...
#
# Writes /scratch/arm.env (ALL five variables, every time — the supervisor
# re-sources with `set -a`, so an omitted variable would leak from the previous
# arm), kills the server (shared pid namespace), and waits for /health to go
# down and come back. ROCP_TOOL_LIBRARIES is read at HIP init, so a process
# restart is exactly sufficient — and nothing less is.
set -euo pipefail

ARM="${1:?usage: arm.sh A|B|C|I [extra vllm args]}"
shift || true
TOOL=/scratch/lib/libgitm_rocm_inject.so
OUT=/scratch/trace/kimi.jsonl

SHIM=/scratch/lib/libgitm_roctx_shim.so
case "$ARM" in
  A) ROCP="";      NVTX=0; EXTRA="";                                PRELOAD="" ;;
  B) ROCP="$TOOL"; NVTX=0; EXTRA="";                                PRELOAD="" ;;
  # C also preloads the roctx shim: torch emits markers through the LEGACY
  # libroctx64, which rocprofiler-sdk's marker service cannot see (found live
  # on ROCm 7.2.3 — ranges vanished silently). The shim forwards them.
  # And C runs EAGER: layerwise ranges are nn.Module forward hooks, which
  # never fire inside a CUDA/HIP-graph replay (vLLM warns exactly this).
  # C is the attribution arm — its throughput is never the headline, and the
  # eager-vs-graphs delta is itself a measurement (the H200 residual story).
  C) ROCP="$TOOL"; NVTX=1; EXTRA="--enable-layerwise-nvtx-tracing --enforce-eager"; PRELOAD="$SHIM" ;;
  I) ROCP="$TOOL"; NVTX=0; EXTRA="$*";                              PRELOAD="" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

# Idempotent: if this exact arm is already written AND the server is healthy,
# don't pay a model reload for nothing.
WANT=$(printf 'export GITM_ARM=%s\nexport ROCP_TOOL_LIBRARIES=%s\nexport GITM_TRACE_OUT=%s\nexport GITM_TRACE_NVTX=%s\nexport GITM_EXTRA_VLLM_ARGS="%s"\nexport LD_PRELOAD=%s\n' \
  "$ARM" "$ROCP" "$OUT" "$NVTX" "$EXTRA" "$PRELOAD")
if [ -f /scratch/arm.env ] && [ "$(cat /scratch/arm.env)" = "$WANT" ] \
   && curl -sf -o /dev/null http://localhost:8000/health; then
  echo "==> arm $ARM already active and serving — no restart"
  exit 0
fi

cat > /scratch/arm.env <<EOF
export GITM_ARM=$ARM
export ROCP_TOOL_LIBRARIES=$ROCP
export GITM_TRACE_OUT=$OUT
export GITM_TRACE_NVTX=$NVTX
export GITM_EXTRA_VLLM_ARGS="$EXTRA"
export LD_PRELOAD=$PRELOAD
EOF
echo "==> arm $ARM written: ROCP_TOOL_LIBRARIES='$ROCP' NVTX=$NVTX extra='$EXTRA'"

# Stale shards from the previous arm would merge into the next window, and a
# stale .arm marker (a capture whose owner died mid-window) blocks the next
# preflight. The restart makes both unambiguously stale.
rm -f /scratch/trace/kimi.jsonl.* /scratch/trace/kimi.jsonl.arm 2>/dev/null || true

pkill -f 'vllm serve' || echo "(no server was running)"

echo "==> waiting for /health to drop..."
for _ in $(seq 60); do
  curl -sf -o /dev/null http://localhost:8000/health || break
  sleep 2
done

echo "==> waiting for /health to return (model reload; up to 60 min)..."
for i in $(seq 720); do
  if curl -sf -o /dev/null http://localhost:8000/health; then
    echo "==> arm $ARM serving (waited $((i * 5))s)"
    # Belt and braces: the server log must agree about which arm it launched.
    tail -1 /scratch/logs/server_arm${ARM}.log >/dev/null 2>&1 || true
    exit 0
  fi
  sleep 5
done
echo "!!! server did not come back within 60 min — check /scratch/logs/" >&2
exit 1
