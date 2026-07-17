#!/usr/bin/env bash
# Ship this local repo to a pod via tar-over-SSH — no git push, and works with
# RunPod's SSH *proxy* (ssh.runpod.io), which does not support scp/sftp. The
# tarball streams through the SSH shell channel and unpacks on the pod.
#
#   scripts/ship_to_pod.sh <ssh-destination> [port] [remote_parent]
#
# RunPod proxy (no port):  scripts/ship_to_pod.sh n1laqoy57ugl52-64412064@ssh.runpod.io
# Exposed TCP (with port): scripts/ship_to_pod.sh root@1.2.3.4 40000
#
# Re-run any time after local edits. Override the key with SSH_KEY=...
set -euo pipefail
DEST=${1:?usage: ship_to_pod.sh <ssh-destination> [port] [remote_parent=/workspace]}
PORT=${2:-}
PARENT=${3:-/workspace}
KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$(basename "$ROOT")"          # "runtime"
SSH_ARGS=(-i "$KEY" -o StrictHostKeyChecking=accept-new)
[ -n "$PORT" ] && SSH_ARGS+=(-p "$PORT")

echo "==> shipping $NAME -> $DEST:$PARENT/$NAME  (tar over ssh)"
# COPYFILE_DISABLE stops macOS tar emitting ._AppleDouble sidecars for every file.
# --no-same-owner stops the remote tar trying to chown to the Mac's uid (it runs as
# root, the uid doesn't exist there, and it errors on every single entry — which
# looks exactly like a failed transfer while actually being harmless).
COPYFILE_DISABLE=1 tar czf - \
  --exclude .venv --exclude .git --exclude __pycache__ --exclude .pytest_cache \
  --exclude '*.egg-info' --exclude dist --exclude build --exclude verify_report.json \
  --exclude .ruff_cache --exclude .DS_Store --exclude '*.so' \
  -C "$(dirname "$ROOT")" "$NAME" \
  | ssh "${SSH_ARGS[@]}" "$DEST" \
      "mkdir -p '$PARENT' && rm -rf '$PARENT/$NAME' && tar xzf - --no-same-owner -C '$PARENT' && echo '   unpacked at $PARENT/$NAME'"

echo "==> done. On the pod:"
echo "    cd $PARENT/$NAME && bash scripts/gpu_setup.sh && ./scripts/verify_infra.sh"
