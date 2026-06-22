#!/usr/bin/env bash
set -euo pipefail

# Boot sequence for the single-node SLURM CI container (PID 1 via tini):
#   patch hostname into slurm.conf -> munged -> slurmctld (wait for ping) ->
#   slurmd -> wait for node IDLE (resume if DOWN) -> exec test cmd as slurmuser.
# Hostname is patched because slurmd requires NodeName to equal `hostname -s`.

log() { echo "[start-slurm] $*"; }

# Put this container's hostname into the SLURM config (must match or slurmd won't start).
HOST="$(hostname -s)"
log "Container hostname: ${HOST}"
sed -i "s/PLACEHOLDER_HOST/${HOST}/g" /etc/slurm/slurm.conf

# Start munge auth (needed before slurmctld/slurmd).
log "Starting munged..."
chown -R munge:munge /run/munge
runuser -u munge -- /usr/sbin/munged --force
if ! munge -n | unmunge >/dev/null 2>&1; then
    log "ERROR: munge round-trip failed."
    cat /var/log/munge/munged.log 2>/dev/null || true
    exit 1
fi
log "munge OK."

# Start the controller (slurmctld) as the slurm user.
log "Starting slurmctld..."
runuser -u slurm -- /usr/sbin/slurmctld
for _ in $(seq 1 30); do
    if scontrol ping >/dev/null 2>&1; then break; fi
    sleep 1
done

# Start the compute daemon (slurmd) as root (needs root to run jobs as other users).
log "Starting slurmd..."
if ! /usr/sbin/slurmd; then
    log "ERROR: slurmd failed to start. Diagnostics:"
    echo "--- slurmd.log ---";    tail -n 200 /var/log/slurm/slurmd.log    2>/dev/null || echo "(no slurmd.log)"
    echo "--- slurmctld.log ---"; tail -n 200 /var/log/slurm/slurmctld.log 2>/dev/null || echo "(no slurmctld.log)"
    echo "--- foreground 'slurmd -Dvvv' (10s, shows the real error) ---"
    timeout 10 /usr/sbin/slurmd -Dvvv 2>&1 | head -n 120 || true
    exit 1
fi

# Wait for the node to become IDLE; resume it if it came up DOWN/DRAINED.
log "Waiting for node to become IDLE..."
ready=0
for i in $(seq 1 60); do
    state="$(sinfo -h -o '%T' 2>/dev/null | head -n1 || true)"
    log "  attempt ${i}: partition state='${state}'"
    case "${state}" in
        idle*|mixed*|allocated*)
            ready=1
            break
            ;;
        *)
            scontrol update nodename="${HOST}" state=resume 2>/dev/null || true
            ;;
    esac
    sleep 2
done

if [ "${ready}" -ne 1 ]; then
    log "ERROR: node did not reach IDLE. Diagnostics:"
    sinfo -N -l || true
    scontrol show nodes || true
    echo "--- slurmctld.log ---"; tail -n 100 /var/log/slurm/slurmctld.log 2>/dev/null || true
    echo "--- slurmd.log ---";    tail -n 100 /var/log/slurm/slurmd.log    2>/dev/null || true
    exit 1
fi

log "Cluster is up:"
sinfo

# Run the passed command as slurmuser, from the repo dir.
if [ "$#" -eq 0 ]; then
    set -- bash -l
fi
log "Executing as slurmuser: $*"
cd /opt/queens
exec runuser -u slurmuser -- env HOME=/home/slurmuser "$@"
