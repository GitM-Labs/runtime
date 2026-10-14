#!/usr/bin/env bash
# Set up the small CPU "control" VM: Kubernetes control plane, the shared
# 8-GPU-per-job queue, the gitm-job CLI for the team, and the status pusher
# that feeds the web page. GPU VMs then join with setup-gpu-node.sh.
#
#   sudo ./setup-control.sh --status-repo aditchawdhary/gpu-status \
#                           --hf-cache /var/lib/gitm/hf-cache
#   sudo ./setup-control.sh --smoke     # after both GPU nodes have joined
#
# Idempotent: safe to re-run.
set -euo pipefail

DEVICE_PLUGIN_VERSION="v0.17.1"
KUEUE_VERSION="v0.20.1"

STATUS_REPO=""
HF_CACHE="/var/lib/gitm/hf-cache"
RESULTS_DIR="/var/lib/gitm/results"
SMOKE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --status-repo) STATUS_REPO="$2"; shift 2 ;;
    --hf-cache)    HF_CACHE="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --smoke)       SMOKE_ONLY=1; shift ;;
    -h|--help)     sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
CONF_DIR=/etc/gitm-gpu
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# Smoke test: one queued job per GPU node, both must run and see 8 GPUs.
# ---------------------------------------------------------------------------
smoke() {
  step "Smoke test"
  mapfile -t NODES < <(kubectl get nodes -l gitm.dev/gpu-node=true -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
  [[ ${#NODES[@]} -gt 0 ]] || { echo "no GPU nodes have joined yet (run setup-gpu-node.sh on each H200 VM)" >&2; exit 1; }
  for n in "${NODES[@]}"; do
    cap=$(kubectl get node "$n" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')
    echo "$n: nvidia.com/gpu=${cap:-0}"
  done
  for n in "${NODES[@]}"; do
    kubectl -n gitm delete job "smoke-$n" --ignore-not-found >/dev/null
    cat <<EOF | kubectl apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata:
  name: smoke-$n
  namespace: gitm
  labels: {kueue.x-k8s.io/queue-name: gpu, gitm.dev/owner: setup}
  annotations: {gitm.dev/display-name: "smoke test ($n)", gitm.dev/model: "-"}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      nodeSelector: {kubernetes.io/hostname: "$n"}
      containers:
      - name: main
        image: ubuntu:24.04
        command: ["nvidia-smi", "-L"]
        resources: {limits: {nvidia.com/gpu: 8}}
EOF
  done
  ok=1
  for n in "${NODES[@]}"; do
    if kubectl -n gitm wait "job/smoke-$n" --for=condition=Complete --timeout=10m >/dev/null; then
      seen=$(kubectl -n gitm logs "job/smoke-$n" | grep -c '^GPU ' || true)
      echo "$n: job saw $seen GPU(s)"; [[ "$seen" -eq 8 ]] || ok=0
    else
      echo "$n: smoke job did not complete (kubectl -n gitm describe job smoke-$n)"; ok=0
    fi
  done
  if kubectl -n gitm create job gitm-bypass --image=ubuntu:24.04 -- true >/dev/null 2>&1; then
    kubectl -n gitm delete job gitm-bypass >/dev/null; echo "admission policy NOT enforcing"; ok=0
  else
    echo "admission policy rejects unqueued jobs: ok"
  fi
  [[ $ok -eq 1 ]] && echo "SMOKE TEST PASSED" || { echo "SMOKE TEST FAILED" >&2; exit 1; }
}
if [[ $SMOKE_ONLY -eq 1 ]]; then smoke; exit 0; fi

[[ -n "$STATUS_REPO" ]] || { echo "--status-repo is required (e.g. aditchawdhary/gpu-status)" >&2; exit 1; }

# ---------------------------------------------------------------------------
step "k3s server (control plane only; jobs never land here)"
mkdir -p /etc/rancher/k3s
cat > /etc/rancher/k3s/config.yaml <<EOF
write-kubeconfig-mode: "0600"
disable:
  - traefik
  - servicelb
EOF
if ! command -v k3s >/dev/null; then
  curl -sfL https://get.k3s.io | sh -s - server
else
  systemctl restart k3s
fi
until kubectl get nodes >/dev/null 2>&1; do sleep 2; done
kubectl wait node --all --for=condition=Ready --timeout=5m

# ---------------------------------------------------------------------------
step "NVIDIA device plugin ${DEVICE_PLUGIN_VERSION} (GPU nodes only)"
kubectl apply -f "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/${DEVICE_PLUGIN_VERSION}/deployments/static/nvidia-device-plugin.yml"
kubectl -n kube-system patch daemonset nvidia-device-plugin-daemonset --type merge \
  -p '{"spec":{"template":{"spec":{"nodeSelector":{"gitm.dev/gpu-node":"true"}}}}}' >/dev/null
kubectl -n kube-system set env daemonset/nvidia-device-plugin-daemonset DEVICE_LIST_STRATEGY=volume-mounts >/dev/null

# ---------------------------------------------------------------------------
step "Kueue ${KUEUE_VERSION} and the shared queue"
kubectl apply --server-side -f "https://github.com/kubernetes-sigs/kueue/releases/download/${KUEUE_VERSION}/manifests.yaml"
kubectl -n kueue-system wait deploy/kueue-controller-manager --for=condition=available --timeout=5m
for i in $(seq 30); do   # Kueue's webhook lags its deployment by a few seconds
  kubectl apply -f "$HERE/cluster.yaml" && break
  [[ $i -eq 30 ]] && exit 1; sleep 5
done

# ---------------------------------------------------------------------------
step "Team access (Unix group gitm-users)"
getent group gitm-users >/dev/null || groupadd gitm-users
mkdir -p "$CONF_DIR"
TOKEN=""
for _ in $(seq 30); do
  TOKEN=$(kubectl -n gitm get secret team-token -o jsonpath='{.data.token}' 2>/dev/null | base64 -d || true)
  [[ -n "$TOKEN" ]] && break; sleep 1
done
[[ -n "$TOKEN" ]] || { echo "team-token secret never got a token" >&2; exit 1; }
CA=$(kubectl config view --raw -o jsonpath='{.clusters[0].cluster.certificate-authority-data}')
( umask 027
  cat > "$CONF_DIR/kubeconfig" <<EOF
apiVersion: v1
kind: Config
clusters:
- name: gitm
  cluster: {server: "https://127.0.0.1:6443", certificate-authority-data: "${CA}"}
users:
- name: team
  user: {token: "${TOKEN}"}
contexts:
- name: gitm
  context: {cluster: gitm, user: team, namespace: gitm}
current-context: gitm
EOF
)
chown root:gitm-users "$CONF_DIR/kubeconfig"; chmod 640 "$CONF_DIR/kubeconfig"
cat > "$CONF_DIR/config" <<EOF
# Read by gitm-job and gitm-status. Re-run setup-control.sh to change.
# HF_CACHE and RESULTS_DIR are paths ON THE GPU NODES (same on each).
STATUS_REPO=${STATUS_REPO}
HF_CACHE=${HF_CACHE}
RESULTS_DIR=${RESULTS_DIR}
EOF

# ---------------------------------------------------------------------------
step "gitm-job CLI and status pusher"
install -m 755 "$HERE/gitm-job" /usr/local/bin/gitm-job
install -m 644 "$HERE/gitm-status.service" /etc/systemd/system/gitm-status.service
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
[[ -f "$CONF_DIR/deploy_key" ]] || ssh-keygen -q -t ed25519 -N '' -C "gitm-status" -f "$CONF_DIR/deploy_key"
chmod 600 "$CONF_DIR/deploy_key"
systemctl daemon-reload
systemctl enable gitm-status.service >/dev/null
systemctl restart gitm-status.service

# ---------------------------------------------------------------------------
IP=$(hostname -I | awk '{print $1}')
K3S_VERSION=$(k3s --version | awk 'NR==1{print $3}')
NODE_TOKEN=$(cat /var/lib/rancher/k3s/server/node-token)
step "Control VM ready"
cat <<EOF

1. On EACH H200 VM (use h200-a, h200-b), copy deploy/h200/ over and run:

   sudo ./setup-gpu-node.sh --server https://${IP}:6443 \\
        --token ${NODE_TOKEN} \\
        --k3s-version ${K3S_VERSION} --name h200-a \\
        --hf-cache ${HF_CACHE} --results-dir ${RESULTS_DIR}

   (The token lets a machine join the cluster: keep it private.)

2. Back here, once both have joined:   sudo ./setup-control.sh --smoke

3. Let the status pusher write to GitHub. Add this as a deploy key WITH WRITE
   ACCESS at https://github.com/${STATUS_REPO}/settings/keys/new :

   $(cat "$CONF_DIR/deploy_key.pub")

   then:  sudo systemctl restart gitm-status && journalctl -u gitm-status -f

4. Give teammates access (they log out and back in afterwards):
   sudo usermod -aG gitm-users <username>

5. Optional, for gated models, a Hugging Face token all jobs can use:
   sudo k3s kubectl -n gitm create secret generic hf-token --from-literal=token=hf_xxx
EOF
