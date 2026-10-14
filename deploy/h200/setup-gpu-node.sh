#!/usr/bin/env bash
# Join one 8x H200 VM to the control VM's cluster as a GPU worker.
#
# Run setup-control.sh first; it prints this exact command with the values
# filled in:
#
#   sudo ./setup-gpu-node.sh --server https://10.128.0.5:6443 --token K10... \
#        --k3s-version v1.34.1+k3s1 --name h200-a --hf-cache /home/adit/hf_hub
#
# Idempotent. The NVIDIA driver must already be installed (`nvidia-smi` works).
# --hf-cache and --results-dir must be the SAME paths on every GPU node: jobs
# mount them by path, whichever node they land on.
set -euo pipefail

SERVER="" TOKEN="" K3S_VERSION="" NAME=""
HF_CACHE="/var/lib/gitm/hf-cache"
RESULTS_DIR="/var/lib/gitm/results"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --server)      SERVER="$2"; shift 2 ;;
    --token)       TOKEN="$2"; shift 2 ;;
    --k3s-version) K3S_VERSION="$2"; shift 2 ;;
    --name)        NAME="$2"; shift 2 ;;
    --hf-cache)    HF_CACHE="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    -h|--help)     sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }
for v in SERVER TOKEN K3S_VERSION NAME; do
  [[ -n "${!v}" ]] || { echo "--$(tr '[:upper:]_' '[:lower:]-' <<<"$v") is required" >&2; exit 1; }
done
[[ "$NAME" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] || { echo "--name: lowercase letters, digits, dashes" >&2; exit 1; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "Checking GPUs"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first" >&2; exit 1; }
NGPU=$(nvidia-smi -L | grep -c '^GPU ' || true)
echo "found $NGPU GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | tr '\n' ' ')"
[[ "$NGPU" -eq 8 ]] || echo "WARNING: expected 8 GPUs; jobs request exactly 8 and will never fit here"

step "Reaching the control VM"
HOSTPORT="${SERVER#https://}"
if ! timeout 5 bash -c "exec 3<>/dev/tcp/${HOSTPORT%:*}/${HOSTPORT##*:}" 2>/dev/null; then
  echo "cannot reach $SERVER. Open the GCP firewall between the VMs (see README: tcp 6443, 10250; udp 8472)." >&2
  exit 1
fi
echo "ok"

step "NVIDIA container toolkit"
if ! command -v nvidia-ctk >/dev/null; then
  command -v apt-get >/dev/null || { echo "no apt-get: install nvidia-container-toolkit manually, then re-run" >&2; exit 1; }
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit
fi
# GPUs reach an unprivileged container only through the device plugin's volume
# mounts, never via NVIDIA_VISIBLE_DEVICES=all — otherwise any pod could grab
# GPUs the queue thinks are free.
nvidia-ctk config --in-place \
  --set accept-nvidia-visible-devices-envvar-when-unprivileged=false \
  --set accept-nvidia-visible-devices-as-volume-mounts=true

step "Shared paths"
mkdir -p "$HF_CACHE" "$RESULTS_DIR"
chmod 1777 "$RESULTS_DIR"   # jobs write here; sticky so runs can't delete each other
df -h "$HF_CACHE" | tail -1 | awk '{print "hf cache disk: " $4 " free of " $2}'

step "k3s agent ${K3S_VERSION}"
mkdir -p /etc/rancher/k3s
cat > /etc/rancher/k3s/config.yaml <<EOF
default-runtime: nvidia
node-name: ${NAME}
node-label:
  - gitm.dev/gpu-node=true
EOF
if ! systemctl is-active --quiet k3s-agent; then
  curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$K3S_VERSION" K3S_URL="$SERVER" K3S_TOKEN="$TOKEN" sh -s - agent
else
  systemctl restart k3s-agent
fi
for _ in $(seq 60); do
  ls /var/lib/rancher/k3s/agent/etc/containerd/config.toml* >/dev/null 2>&1 && break; sleep 2
done
grep -q nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml* \
  || { echo "k3s did not detect the nvidia runtime (is nvidia-container-runtime on PATH?)" >&2; exit 1; }

step "Done: ${NAME} joined"
echo "On the control VM, check it:  sudo k3s kubectl get nodes   (then: sudo ./setup-control.sh --smoke)"
