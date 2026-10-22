#!/usr/bin/env bash
# Run on the second H200 machine to join the Slurm cluster run by gitmachine-2.
#
#   sudo ./setup-join.sh --bundle ~/gitm-join.tgz
#
# The bundle comes from setup-controller.sh: munge key, slurm.conf (+gres,
# cgroup), team accounts with their UIDs, and the status-report key. After
# this, the machine runs slurmd, takes jobs from the same queue, and accepts
# the same Slurm tokens (tokens are checked by the controller only).
set -euo pipefail

BUNDLE=""
SKIP_IMAGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)     BUNDLE="$2"; shift 2 ;;
    --skip-image) SKIP_IMAGE=1; shift ;;
    -h|--help)    sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }
[[ -f "$BUNDLE" ]] || { echo "--bundle FILE is required (made by setup-controller.sh)" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

B=$(mktemp -d); trap 'rm -rf "$B"' EXIT
tar -C "$B" -xzf "$BUNDLE"
# shellcheck disable=SC1091
source "$B/info"
THIS=$(hostname -s)

step "Joining $THIS to the cluster on $CONTROLLER (Slurm $SLURM_VER)"
[[ " $NODES " == *" $THIS "* ]] || die "this machine is '$THIS' but the controller added: $NODES
  Re-run setup-controller.sh with --add-node $THIS=$(hostname -I | awk '{print $1}')"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver first"

# ---------------------------------------------------------------------------
step "Reaching the controller"
if ! getent hosts "$CONTROLLER" >/dev/null; then
  echo "$CONTROLLER_IP $CONTROLLER_FQDN $CONTROLLER" >> /etc/hosts
  echo "added $CONTROLLER to /etc/hosts"
fi
timeout 3 bash -c "</dev/tcp/$CONTROLLER_IP/6817" 2>/dev/null \
  || die "cannot reach slurmctld at $CONTROLLER_IP:6817. Ask the provider to allow TCP 6817-6818 (and srun's ports) between the two machines."
echo "slurmctld port reachable"

# ---------------------------------------------------------------------------
step "Slurm $SLURM_VER and munge"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq munge >/dev/null
if [[ -f "$B/slurm_user" && -s "$B/slurm_user" ]]; then
  IFS=: read -r gname _ ggid _ < "$B/slurm_group" || true
  [[ -n "${ggid:-}" ]] && { getent group "$ggid" >/dev/null || groupadd -r -g "$ggid" "$gname"; }
  IFS=: read -r suser _ suid sgid sgecos shome sshell < "$B/slurm_user"
  id "$suser" >/dev/null 2>&1 || useradd -r -u "$suid" -g "$sgid" -c "$sgecos" -d "$shome" -M -s "$sshell" "$suser"
fi
if [[ -f "$B/slurm-install.tgz" ]]; then
  # The controller's own source build, file for file.
  if [[ -s "$B/slurm-deps.txt" ]]; then
    # shellcheck disable=SC2046
    apt-get install -y -qq $(cat "$B/slurm-deps.txt") >/dev/null || warn "some runtime packages failed to install"
  fi
  tar -C / -xzf "$B/slurm-install.tgz"
  ldconfig
  systemctl daemon-reload
  # The build may live outside sudo's PATH (e.g. /opt/slurm): find it from the
  # bundle, use it here, and add it to every login shell's PATH.
  SLURMD_P=$(tar -tzf "$B/slurm-install.tgz" | grep -E '(^|/)s?bin/slurmd$' | head -1 || true)
  SRUN_P=$(tar -tzf "$B/slurm-install.tgz" | grep -E '(^|/)bin/srun$' | head -1 || true)
  if [[ -z "$SLURMD_P" ]]; then
    for d in /usr/local/sbin /usr/sbin /opt/slurm/sbin /usr/local/slurm/sbin /opt/slurm/bin /usr/local/bin; do
      if [[ -x "$d/slurmd" ]]; then SLURMD_P="${d#/}/slurmd"; break; fi
    done
  fi
  [[ -n "$SLURMD_P" ]] || die "slurmd binary not found in the bundle. On gitmachine-2 run: command -v slurmd; sudo sh -c 'command -v slurmd'"
  echo "slurmd: /$SLURMD_P   srun: /${SRUN_P:-?}"
  ADD=""
  for p in "/$SLURMD_P" "/${SRUN_P:-$SLURMD_P}"; do
    d=$(dirname "$p")
    case ":$PATH:" in *":$d:"*) ;; *) PATH="$d:$PATH"; ADD="$ADD:$d" ;; esac
  done
  export PATH
  if [[ -n "$ADD" ]]; then
    echo "export PATH=\"\$PATH$ADD\"" > /etc/profile.d/gitm-slurm-path.sh
    echo "Slurm binaries are in ${ADD#:} (added to PATH for login shells)"
  fi
  MISSING=$(ldd "/$SLURMD_P" 2>/dev/null | grep 'not found' || true)
  [[ -z "$MISSING" ]] || die "slurmd is missing libraries:
$MISSING"
elif ! command -v slurmd >/dev/null; then
  apt-get install -y -qq slurmd slurm-client >/dev/null
fi
HAVE=$(slurmd -V | awk '{print $2}')
echo "controller: $SLURM_VER   this machine: $HAVE"
[[ "${HAVE%.*}" == "${SLURM_VER%.*}" ]] || die "Slurm versions differ ($HAVE vs $SLURM_VER)."

# ---------------------------------------------------------------------------
step "Team accounts (same UIDs as $CONTROLLER)"
while IFS=: read -r gname _ gid _; do
  getent group "$gid" >/dev/null || groupadd -g "$gid" "$gname"
done < "$B/group"
CONFLICT=0
while IFS=: read -r name _ uid gid gecos home shell; do
  if cur=$(id -u "$name" 2>/dev/null); then
    [[ "$cur" == "$uid" ]] || { warn "$name has UID $cur here but $uid on $CONTROLLER"; CONFLICT=1; }
    continue
  fi
  if getent passwd "$uid" >/dev/null; then
    warn "UID $uid is taken here by $(getent passwd "$uid" | cut -d: -f1); cannot create $name"; CONFLICT=1; continue
  fi
  useradd -u "$uid" -g "$gid" -c "$gecos" -d "$home" -M -s "$shell" "$name"
  mkdir -p "$home"; chown "$uid:$gid" "$home"; chmod 750 "$home"
  echo "created $name ($uid)"
done < "$B/passwd"
[[ $CONFLICT -eq 0 ]] || warn "jobs from the users above will fail on this machine until their UIDs match"

# ---------------------------------------------------------------------------
step "munge key and Slurm config"
install -m 400 -o munge -g munge "$B/munge.key" /etc/munge/munge.key
systemctl enable munge >/dev/null; systemctl restart munge
munge -n | unmunge >/dev/null && echo "munge: ok"
mkdir -p "$SLURM_DIR"
for f in "$B"/slurm/*; do install -m 644 "$f" "$SLURM_DIR/"; done
SPOOL=$(awk -F= 'tolower($1)=="slurmdspooldir"{print $2}' "$SLURM_DIR/slurm.conf"); SPOOL=${SPOOL:-/var/spool/slurmd}
mkdir -p "$SPOOL" /var/log/slurm
scontrol ping | grep -q UP || die "scontrol can't talk to the controller (munge key or network)"
echo "controller answers: ok"

# ---------------------------------------------------------------------------
step "Containers (Enroot + Pyxis), reporter, gitm CLI"
mkdir -p /etc/gitm-gpu
install -m 644 "$B/gitm/config" /etc/gitm-gpu/config
RES=$(grep '^RESULTS_DIR=' /etc/gitm-gpu/config | cut -d= -f2-)
HF=$(grep '^HF_CACHE=' /etc/gitm-gpu/config | cut -d= -f2-)
IFS=, read -r -a HEALTH_USERS <<<"$(grep '^HEALTH_USERS=' /etc/gitm-gpu/config | cut -d= -f2-)"
NODE_ARGS=(--report-key "$(cat "$B/gitm/report_key.pub")" --results-dir "$RES" --hf-cache "$HF")
for user in "${HEALTH_USERS[@]}"; do NODE_ARGS+=(--health-user "$user"); done
[[ $SKIP_IMAGE -eq 1 ]] && NODE_ARGS+=(--skip-image)
"$HERE/setup-node.sh" "${NODE_ARGS[@]}"   # also restarts slurmd if Pyxis was added

# ---------------------------------------------------------------------------
step "Starting slurmd"
systemctl enable slurmd >/dev/null
systemctl restart slurmd
for _ in $(seq 30); do
  st=$(sinfo -h -n "$THIS" -o '%T' 2>/dev/null | head -1)
  [[ "$st" == idle* || "$st" == alloc* || "$st" == mix* ]] && break; sleep 2
done
sinfo -N
case "$st" in
  idle*|alloc*|mix*) echo -e "\n$THIS joined the cluster ($st). Next, on $CONTROLLER:  sudo ./setup-controller.sh --smoke" ;;
  *) warn "$THIS is '$st'. Check: scontrol show node $THIS | grep -i reason ; journalctl -u slurmd -n 30
  If the reason is 'Low RealMemory' or similar, on $CONTROLLER run: scontrol update nodename=$THIS state=resume" ;;
esac
