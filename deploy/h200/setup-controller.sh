#!/usr/bin/env bash
# Run on the Slurm controller (gitmachine-2). Turns the existing Slurm into the
# team's whole-node GPU queue, adds the second H200 machine, and starts the
# status feed for https://aditchawdhary.github.io/gpu/.
#
#   sudo ./setup-controller.sh --add-node NODE2NAME=NODE2_IP \
#                              --status-repo aditchawdhary/gpu-status
#   sudo ./setup-controller.sh --smoke        # after setup-join.sh on the 2nd machine
#
# NODE2NAME is what `hostname -s` prints on the second machine; NODE2_IP is its
# internal IP (`hostname -I`). Changes to slurm.conf are shown as a diff and
# backed up first; if slurmctld does not come back, the backup is restored.
set -euo pipefail

PARTITION="gpu"
STATUS_REPO=""
RESULTS_DIR="/var/lib/gitm/results"
HF_CACHE="/var/lib/gitm/hf-cache"
ADD_NODES=()
SMOKE=0
SKIP_IMAGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --add-node)    ADD_NODES+=("$2"); shift 2 ;;
    --status-repo) STATUS_REPO="$2"; shift 2 ;;
    --partition)   PARTITION="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --hf-cache)    HF_CACHE="$2"; shift 2 ;;
    --skip-image)  SKIP_IMAGE=1; shift ;;
    --smoke)       SMOKE=1; shift ;;
    -h|--help)     sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
CONF_DIR=/etc/gitm-gpu
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
ADMIN_USER="${SUDO_USER:-root}"

systemctl is-active --quiet slurmctld || die "slurmctld is not running here; run this on the Slurm controller"
SLURM_CONF=$(scontrol show config | awk -F'= ' '/^SLURM_CONF/{print $2}')
SLURM_DIR=$(dirname "$SLURM_CONF")
SLURM_VER=$(sinfo -V | awk '{print $2}')
THIS=$(hostname -s)

# ---------------------------------------------------------------------------
# Smoke test: one containerized job per node, each must see 8 GPUs.
# ---------------------------------------------------------------------------
if [[ $SMOKE -eq 1 ]]; then
  step "Smoke test (partition $PARTITION)"
  sinfo -N -p "$PARTITION"
  IMAGE=$(grep '^DEFAULT_IMAGE=' "$CONF_DIR/config" 2>/dev/null | cut -d= -f2-); IMAGE=${IMAGE:-vllm/vllm-openai:latest}
  GRES=$(grep '^GRES=' "$CONF_DIR/config" 2>/dev/null | cut -d= -f2-)
  ok=1
  for n in $(sinfo -h -N -p "$PARTITION" -o '%N' | sort -u); do
    printf '%s: ' "$n"
    out=$(runuser -u "$ADMIN_USER" -- srun -p "$PARTITION" -w "$n" -N1 --exclusive ${GRES:+--gres=$GRES} \
            -t 10 --job-name=smoke-test --container-image="$IMAGE" --no-container-mount-home \
            nvidia-smi -L 2>&1) || true
    seen=$(grep -c '^GPU ' <<<"$out" || true)
    if [[ "$seen" -eq 8 ]]; then echo "container saw 8 GPUs: ok"; else echo "FAILED"; echo "$out" | tail -8; ok=0; fi
  done
  if systemctl is-active --quiet gitm-status; then
    echo "status pusher: $(journalctl -u gitm-status -n 1 --no-pager -o cat)"
  else
    warn "gitm-status service is not running"; ok=0
  fi
  [[ $ok -eq 1 ]] && echo -e "\nSMOKE TEST PASSED" || { echo -e "\nSMOKE TEST FAILED" >&2; exit 1; }
  exit 0
fi

[[ -n "$STATUS_REPO" ]] || die "--status-repo is required (e.g. aditchawdhary/gpu-status)"

# ---------------------------------------------------------------------------
step "Slurm $SLURM_VER on $THIS ($SLURM_CONF)"
if ! dpkg -S "$(command -v slurmd)" >/dev/null 2>&1; then
  warn "slurmd here was not installed from a .deb package. setup-join.sh installs Slurm from apt on the
  second machine, which must match $SLURM_VER. If you built Slurm from source, install the same build there first."
fi
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [[ "$BUSY" -gt 0 ]]; then
  warn "$BUSY process(es) hold GPUs on $THIS outside Slurm. Queued jobs will run out of GPU memory until they stop:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read -r p m; do
    echo "   $(ps -o user=,pid=,etime= -p "$p")  $m  $(ps -o cmd= -p "$p" | cut -c1-90)"; done
fi

# ---------------------------------------------------------------------------
step "Updating slurm.conf"
BACKUP="$SLURM_CONF.bak.$(date +%Y%m%d-%H%M%S)"
cp -a "$SLURM_CONF" "$BACKUP"
NEW=$(mktemp)
python3 - "$SLURM_CONF" "$NEW" "$PARTITION" "$THIS" "${ADD_NODES[@]}" <<'PY'
import re, sys
src, dst, part, this, *adds = sys.argv[1:]
lines = open(src).read().splitlines()

def kv(line):
    return re.findall(r"(\w+)=(\S+)", line)

def names_of(spec):
    # expand simple lists: a,b,c (bracket ranges are kept as-is)
    return [n for n in re.split(r",(?![^\[]*\])", spec) if n]

# --- the node definition for this host, to clone for new nodes
tmpl, tmpl_i = None, None
for i, l in enumerate(lines):
    s = l.strip()
    if s.startswith("NodeName=") and not s.startswith("NodeName=DEFAULT"):
        nm = dict(kv(s)).get("NodeName", "")
        if this in names_of(nm) or nm == this:
            tmpl, tmpl_i = s, i
if adds and tmpl is None:
    sys.exit(f"no NodeName line for {this} in slurm.conf; add the second node by hand")

existing = set()
for l in lines:
    s = l.strip()
    if s.startswith("NodeName="):
        existing.update(names_of(dict(kv(s)).get("NodeName", "")))

new_nodes = []
for a in adds:
    if "=" not in a:
        sys.exit(f"--add-node expects NAME=IP, got {a!r}")
    name, ip = a.split("=", 1)
    if name in existing:
        print(f"node {name} already defined; leaving its line alone")
        new_nodes.append(name)
        continue
    rest = [f"{k}={v}" for k, v in kv(tmpl) if k not in ("NodeName", "NodeAddr", "NodeHostname")]
    tmpl_i += 1   # keep node definitions together, ahead of the partitions
    lines.insert(tmpl_i, f"NodeName={name} NodeAddr={ip} " + " ".join(rest))
    print(f"added: NodeName={name} NodeAddr={ip} (cloned from {this})")
    new_nodes.append(name)

# --- the partition: whole-node jobs, and include the new nodes
found = False
for i, l in enumerate(lines):
    s = l.strip()
    if s.startswith("PartitionName=") and dict(kv(s)).get("PartitionName") == part:
        found = True
        d = dict(kv(s))
        nodes = d.get("Nodes", "")
        if nodes.upper() != "ALL":
            cur = names_of(nodes)
            for n in new_nodes:
                if n not in cur:
                    cur.append(n)
            s = re.sub(r"\bNodes=\S+", "Nodes=" + ",".join(cur), s) if "Nodes=" in s else s + " Nodes=" + ",".join(cur)
        if re.search(r"\bOverSubscribe=\S+", s):
            s = re.sub(r"\bOverSubscribe=\S+", "OverSubscribe=EXCLUSIVE", s)
        else:
            s += " OverSubscribe=EXCLUSIVE"
        lines[i] = s
if not found:
    sys.exit(f"no PartitionName={part} in slurm.conf")

# --- finished-job history for the status page when accounting is off
glob = {}
for l in lines:
    s = l.strip()
    if s and not s.startswith("#") and not s.startswith(("NodeName", "PartitionName")):
        for k, v in kv(s):
            glob[k.lower()] = v
if glob.get("jobcomptype", "jobcomp/none").lower() == "jobcomp/none":
    lines = [l for l in lines if not l.strip().lower().startswith(("jobcomptype=", "jobcomploc="))]
    lines += ["JobCompType=jobcomp/filetxt", "JobCompLoc=/var/log/slurm/jobcomp.log"]
    print("added: JobCompType=jobcomp/filetxt (job history for the status page)")
if "gres=gpu" in (tmpl or "").lower() and "grestypes" not in glob:
    lines.append("GresTypes=gpu")
    print("added: GresTypes=gpu")

open(dst, "w").write("\n".join(lines) + "\n")
PY
diff -u "$SLURM_CONF" "$NEW" | sed 's/^/   /' || true
# gres.conf lines pinned to this host (NodeName=...) need a twin for each new node.
GRES_CONF="$SLURM_DIR/gres.conf"
if [[ -f "$GRES_CONF" ]] && grep -q "NodeName=$THIS\b" "$GRES_CONF"; then
  cp -a "$GRES_CONF" "$GRES_CONF.bak.$(date +%Y%m%d-%H%M%S)"
  for a in "${ADD_NODES[@]}"; do
    n="${a%%=*}"
    grep -q "NodeName=$n\b" "$GRES_CONF" || grep "NodeName=$THIS\b" "$GRES_CONF" | sed "s/NodeName=$THIS\b/NodeName=$n/" >> "$GRES_CONF"
  done
  echo "gres.conf: GPU lines added for ${ADD_NODES[*]%%=*}"
fi
install -m 644 "$NEW" "$SLURM_CONF"; rm -f "$NEW"
mkdir -p /var/log/slurm
SLURM_USER=$(scontrol show config | awk -F'= ' '/^SlurmUser/{print $2}' | cut -d'(' -f1)
touch /var/log/slurm/jobcomp.log; chown "${SLURM_USER:-slurm}" /var/log/slurm/jobcomp.log; chmod 644 /var/log/slurm/jobcomp.log

systemctl restart slurmctld
for _ in $(seq 20); do scontrol ping >/dev/null 2>&1 && break; sleep 1; done
if ! scontrol ping 2>/dev/null | grep -q UP; then
  warn "slurmctld did not come back; restoring $BACKUP"
  cp -a "$BACKUP" "$SLURM_CONF"; systemctl restart slurmctld
  journalctl -u slurmctld -n 20 --no-pager
  die "slurm.conf change rejected (see log above). Your previous config is restored."
fi
systemctl restart slurmd
echo "slurmctld up with the new config (backup: $BACKUP)"
JOBCOMP_LOG=$(scontrol show config | awk -F'= ' '/^JobCompLoc/{print $2}')

# ---------------------------------------------------------------------------
step "Keys"
mkdir -p "$CONF_DIR"; chmod 755 "$CONF_DIR"
[[ -f "$CONF_DIR/report_key" ]] || ssh-keygen -q -t ed25519 -N '' -C gitm-report -f "$CONF_DIR/report_key"
[[ -f "$CONF_DIR/deploy_key" ]] || ssh-keygen -q -t ed25519 -N '' -C gitm-status -f "$CONF_DIR/deploy_key"
chmod 600 "$CONF_DIR/report_key" "$CONF_DIR/deploy_key"

NODE_GRES=$(scontrol show node "$THIS" | grep -o 'Gres=[^ ]*' | head -1 | cut -d= -f2)
GRES=""; [[ "$NODE_GRES" == gpu* ]] && GRES="gpu:8"
touch "$CONF_DIR/config"
setkey() { grep -q "^$1=" "$CONF_DIR/config" && sed -i "s#^$1=.*#$1=$2#" "$CONF_DIR/config" || echo "$1=$2" >> "$CONF_DIR/config"; }
setkey PARTITION "$PARTITION"
setkey STATUS_REPO "$STATUS_REPO"
setkey GRES "$GRES"
setkey JOBCOMP_LOG "${JOBCOMP_LOG:-}"
setkey RESULTS_DIR "$RESULTS_DIR"
setkey HF_CACHE "$HF_CACHE"

# ---------------------------------------------------------------------------
step "Containers on $THIS (Enroot + Pyxis)"
NODE_ARGS=(--report-key "$(cat "$CONF_DIR/report_key.pub")" --results-dir "$RESULTS_DIR" --hf-cache "$HF_CACHE")
[[ $SKIP_IMAGE -eq 1 ]] && NODE_ARGS+=(--skip-image)
"$HERE/setup-node.sh" "${NODE_ARGS[@]}"

# ---------------------------------------------------------------------------
step "Join bundle for the other machine(s)"
B=$(mktemp -d); mkdir -p "$B/slurm" "$B/gitm"
for f in slurm.conf gres.conf cgroup.conf; do [[ -f "$SLURM_DIR/$f" ]] && cp -a "$SLURM_DIR/$f" "$B/slurm/"; done
cp -a /etc/munge/munge.key "$B/munge.key"
cp "$CONF_DIR/report_key.pub" "$B/gitm/"
grep -v '^STATUS_REPO=' "$CONF_DIR/config" > "$B/gitm/config"
cat > "$B/info" <<EOF
SLURM_VER=$SLURM_VER
SLURM_DIR=$SLURM_DIR
CONTROLLER=$THIS
CONTROLLER_FQDN=$(hostname -f)
CONTROLLER_IP=$(hostname -I | awk '{print $1}')
NODES="$(for a in "${ADD_NODES[@]}"; do echo -n "${a%%=*} "; done)"
EOF
# Accounts that run jobs must exist with the same UID/GID on every node.
getent passwd | awk -F: '$3>=1000 && $3<60000' > "$B/passwd"
cut -d: -f4 "$B/passwd" | sort -u | while read -r g; do getent group "$g"; done > "$B/group"
BUNDLE="/root/gitm-join.tgz"
tar -C "$B" -czf "$BUNDLE" .; rm -rf "$B"; chmod 600 "$BUNDLE"
if [[ "$ADMIN_USER" != root ]]; then
  HOME_ADMIN=$(getent passwd "$ADMIN_USER" | cut -d: -f6)
  install -m 600 -o "$ADMIN_USER" "$BUNDLE" "$HOME_ADMIN/gitm-join.tgz"
  BUNDLE="$HOME_ADMIN/gitm-join.tgz"
fi
echo "bundle: $BUNDLE  (contains the munge key: delete it after joining)"

# ---------------------------------------------------------------------------
step "Status pusher"
install -m 644 "$HERE/gitm-status.service" /etc/systemd/system/gitm-status.service
systemctl daemon-reload
systemctl enable gitm-status >/dev/null
systemctl restart gitm-status

# ---------------------------------------------------------------------------
step "Controller ready"
cat <<EOF

1. Copy this folder and the bundle to the second machine, then run there:
     scp -r $HERE $BUNDLE <you>@<second machine>:~/
     ssh <second machine>
     sudo ~/$(basename "$HERE")/setup-join.sh --bundle ~/gitm-join.tgz && rm ~/gitm-join.tgz

2. Back here:   sudo $HERE/setup-controller.sh --smoke

3. Status page feed: add this as a deploy key WITH WRITE ACCESS at
   https://github.com/${STATUS_REPO}/settings/keys/new
     $(cat "$CONF_DIR/deploy_key.pub")
   then:  sudo systemctl restart gitm-status && journalctl -u gitm-status -f

4. Optional Hugging Face token for gated models (same file on every node):
     echo hf_xxx | sudo tee $CONF_DIR/hf_token >/dev/null && sudo chmod 644 $CONF_DIR/hf_token
EOF
