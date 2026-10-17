#!/usr/bin/env bash
# Prepare one 8x H200 Slurm node to run gitm jobs in throwaway containers.
#
#   sudo ./setup-node.sh --report-key "ssh-ed25519 AAAA... gitm-report"
#
# Run on EVERY GPU node (gitmachine-2 included). The report key is printed by
# setup-controller.sh; it lets the status pusher read GPU readings and job log
# tails from this node, and nothing else.
#
# Installs (pinned): nvidia-container-toolkit (GPU hook), Enroot ${ENROOT_VERSION}
# (unprivileged containers), Pyxis ${PYXIS_VERSION} (srun --container-image=...).
# Idempotent. Does not restart running jobs; restarts slurmd only if Pyxis
# was newly added to the plugin stack.
set -euo pipefail

ENROOT_VERSION="4.2.1"
PYXIS_VERSION="v0.24.0"

REPORT_KEY=""
RESULTS_DIR="/var/lib/gitm/results"
HF_CACHE="/var/lib/gitm/hf-cache"
IMAGE_DIR="/var/lib/gitm/images"
DEFAULT_IMAGE_REF="vllm/vllm-openai:latest"
SKIP_IMAGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --report-key)  REPORT_KEY="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --hf-cache)    HF_CACHE="$2"; shift 2 ;;
    --image)       DEFAULT_IMAGE_REF="$2"; shift 2 ;;
    --skip-image)  SKIP_IMAGE=1; shift ;;
    -h|--help)     sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
step "Checks"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first" >&2; exit 1; }
command -v slurmd >/dev/null || { echo "slurmd not found: this node must be a Slurm node first" >&2; exit 1; }
NGPU=$(nvidia-smi -L | grep -c '^GPU ' || true)
echo "GPUs: $NGPU x $(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | tr '\n' ' ')"
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[[ "$BUSY" -eq 0 ]] || warn "$BUSY process(es) are using GPUs outside Slurm right now (nvidia-smi lists them). Jobs from the queue will collide with them."
if grep -q 'RmProfilingAdminOnly: 1' /proc/driver/nvidia/params 2>/dev/null; then
  warn "CUPTI hardware counters are restricted to root on this node. Kernel tracing works; for counters run:
    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf
    update-initramfs -u && reboot"
fi

# ---------------------------------------------------------------------------
step "Packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl gpg jq git build-essential squashfs-tools parallel fuse-overlayfs squashfuse pigz zstd >/dev/null
if ! command -v nvidia-container-cli >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y -qq libnvidia-container-tools >/dev/null
fi
echo "nvidia-container-cli $(nvidia-container-cli --version | head -1 | awk '{print $NF}')"

# ---------------------------------------------------------------------------
step "Enroot ${ENROOT_VERSION}"
if [[ "$(enroot version 2>/dev/null || true)" != "$ENROOT_VERSION" ]]; then
  ARCH=$(dpkg --print-architecture)
  TMP=$(mktemp -d)
  for pkg in "enroot_${ENROOT_VERSION}-1_${ARCH}.deb" "enroot+caps_${ENROOT_VERSION}-1_${ARCH}.deb"; do
    curl -fsSL -o "$TMP/$pkg" "https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_VERSION}/$pkg"
  done
  apt-get install -y -qq "$TMP"/*.deb >/dev/null
  rm -rf "$TMP"
fi
echo "enroot $(enroot version)"
# Container data lives on local disk, one tree per user, removed by Pyxis when
# a job ends. Nothing goes into home directories.
mkdir -p /var/lib/enroot/{cache,data} /run/enroot
chmod 1777 /var/lib/enroot/cache /var/lib/enroot/data /run/enroot
# Enroot bind-mounts the host's /etc/hostname into every container; some
# images (GCP Ubuntu) never write one, and then every container start fails.
[[ -f /etc/hostname ]] || hostname > /etc/hostname
mkdir -p /etc/enroot/enroot.conf.d
cat > /etc/enroot/enroot.conf.d/gitm.conf <<'EOF'
# gitm: per-user container storage on local disk, never in $HOME
ENROOT_RUNTIME_PATH    /run/enroot/user-$(id -u)
ENROOT_CACHE_PATH      /var/lib/enroot/cache/user-$(id -u)
ENROOT_DATA_PATH       /var/lib/enroot/data/user-$(id -u)
ENROOT_SQUASH_OPTIONS  -comp lz4 -noD
ENROOT_MOUNT_HOME      no
ENROOT_RESTRICT_DEV    yes
ENROOT_ROOTFS_WRITABLE yes
EOF

# ---------------------------------------------------------------------------
step "Pyxis ${PYXIS_VERSION}"
SPANK_H=""
for d in /usr/include /usr/local/include /opt/slurm/include; do
  if [[ -f "$d/slurm/spank.h" ]]; then SPANK_H="$d"; break; fi
done
if [[ -z "$SPANK_H" ]]; then
  apt-get install -y -qq libslurm-dev >/dev/null 2>&1 || true
  [[ -f /usr/include/slurm/spank.h ]] && SPANK_H=/usr/include
fi
[[ -n "$SPANK_H" ]] || { echo "slurm/spank.h not found: install the Slurm development headers that match $(sinfo -V)" >&2; exit 1; }
PYXIS_SO=/usr/local/lib/slurm/spank_pyxis.so
PYXIS_STAMP=/usr/local/share/pyxis/.built-${PYXIS_VERSION}-$(sinfo -V | awk '{print $2}')
if [[ ! -f "$PYXIS_SO" || ! -f "$PYXIS_STAMP" ]]; then
  TMP=$(mktemp -d)
  git -c advice.detachedHead=false clone -q --depth 1 --branch "$PYXIS_VERSION" https://github.com/NVIDIA/pyxis "$TMP/pyxis"
  # CPPFLAGS via the environment: the Makefile appends it to its own flags,
  # whereas a make-argument CPPFLAGS would replace them.
  CPPFLAGS="-I$SPANK_H" make -C "$TMP/pyxis" -s >/dev/null
  make -C "$TMP/pyxis" -s install prefix=/usr/local >/dev/null
  rm -rf "$TMP"
  mkdir -p "$(dirname "$PYXIS_STAMP")"; touch "$PYXIS_STAMP"
fi
SLURM_CONF=$(scontrol show config 2>/dev/null | awk -F'= ' '/^SLURM_CONF/{print $2}')
SLURM_DIR=$(dirname "${SLURM_CONF:-/etc/slurm/slurm.conf}")
PLUGSTACK=$(scontrol show config 2>/dev/null | awk -F'= ' '/^PlugStackConfig/{print $2}')
[[ -z "$PLUGSTACK" || "$PLUGSTACK" == "(null)" ]] && PLUGSTACK="$SLURM_DIR/plugstack.conf"
mkdir -p "$SLURM_DIR/plugstack.conf.d"
CHANGED=0
if ! grep -qsF "$PYXIS_SO" "$SLURM_DIR/plugstack.conf.d/pyxis.conf"; then
  echo "required $PYXIS_SO" > "$SLURM_DIR/plugstack.conf.d/pyxis.conf"; CHANGED=1
fi
# Only add an include line if Pyxis isn't already picked up (some Slurm
# installs read plugstack.conf.d by themselves; loading it twice fails).
if ! srun --help 2>/dev/null | grep -q -- '--container-image'; then
  if ! grep -qs "plugstack.conf.d" "$PLUGSTACK"; then
    [[ -f "$PLUGSTACK" ]] && cp -a "$PLUGSTACK" "$PLUGSTACK.bak.$(date +%s)"
    echo "include $SLURM_DIR/plugstack.conf.d/*" >> "$PLUGSTACK"; CHANGED=1
  fi
fi
if [[ $CHANGED -eq 1 ]]; then
  systemctl restart slurmd
  echo "pyxis registered ($SLURM_DIR/plugstack.conf.d/pyxis.conf); slurmd restarted"
fi
srun --help 2>/dev/null | grep -q -- '--container-image' \
  && echo "srun knows --container-image: ok" \
  || warn "srun does not list --container-image yet (restart your shell; on the controller run setup-controller.sh)"

# ---------------------------------------------------------------------------
step "gitm folders, CLI and node reporter"
mkdir -p "$RESULTS_DIR" "$HF_CACHE" "$IMAGE_DIR" /etc/gitm-gpu
chmod 1777 "$RESULTS_DIR"          # jobs write here; sticky so runs can't delete each other
# Shared Hugging Face cache: every user's job reads and adds models here.
# Default ACLs keep new files writable by all, whatever each job's umask.
chmod 0777 "$HF_CACHE"
command -v setfacl >/dev/null || apt-get install -y -qq acl >/dev/null
setfacl -d -m u::rwx,g::rwx,o::rwx "$HF_CACHE"
install -m 755 "$HERE/gitm-job" /usr/local/bin/gitm-job
install -m 755 "$HERE/gitm-node-report" /usr/local/bin/gitm-node-report
touch /etc/gitm-gpu/config
setkey() { grep -q "^$1=" /etc/gitm-gpu/config && sed -i "s#^$1=.*#$1=$2#" /etc/gitm-gpu/config || echo "$1=$2" >> /etc/gitm-gpu/config; }
setkey RESULTS_DIR "$RESULTS_DIR"
setkey HF_CACHE "$HF_CACHE"
if [[ -n "$REPORT_KEY" ]]; then
  [[ "$REPORT_KEY" =~ ^ssh-(ed25519|rsa)\  ]] || { echo "--report-key must be an ssh public key line" >&2; exit 1; }
  # A dedicated unprivileged account (works where root SSH login is disabled):
  # its key can run gitm-node-report and nothing else.
  id gitm-report >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/gitm-report --shell /bin/sh gitm-report
  install -d -m 700 -o gitm-report -g gitm-report /var/lib/gitm-report/.ssh
  echo "command=\"/usr/local/bin/gitm-node-report\",restrict $(awk '{print $1, $2}' <<<"$REPORT_KEY") gitm-report" \
    > /var/lib/gitm-report/.ssh/authorized_keys
  chown gitm-report:gitm-report /var/lib/gitm-report/.ssh/authorized_keys
  chmod 600 /var/lib/gitm-report/.ssh/authorized_keys
  echo "report key installed for user gitm-report (can only run gitm-node-report)"
fi

# ---------------------------------------------------------------------------
if [[ $SKIP_IMAGE -eq 0 ]]; then
  SQSH="$IMAGE_DIR/$(echo "$DEFAULT_IMAGE_REF" | tr '/:#@' '____').sqsh"
  step "Pre-importing $DEFAULT_IMAGE_REF (saves every job a multi-GB download)"
  if [[ ! -f "$SQSH" ]]; then
    ENROOT_CACHE_PATH=/var/lib/enroot/cache/root ENROOT_DATA_PATH=/var/lib/enroot/data/root \
      enroot import -o "$SQSH.tmp" "docker://$DEFAULT_IMAGE_REF"
    mv "$SQSH.tmp" "$SQSH"
  fi
  chmod 644 "$SQSH"
  setkey DEFAULT_IMAGE "$SQSH"
  echo "default image: $SQSH ($(du -h "$SQSH" | cut -f1))"
fi

step "Done: $(hostname -s)"
echo "Containers: srun --container-image=... works on this node. Default job image: $(grep ^DEFAULT_IMAGE= /etc/gitm-gpu/config | cut -d= -f2)"
