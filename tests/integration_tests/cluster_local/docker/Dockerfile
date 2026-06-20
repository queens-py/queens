# syntax=docker/dockerfile:1
#
# Single-node, container-safe SLURM image for the `cluster_local` integration
# test. It boots slurmctld + slurmd + munge inside one container and runs the
# test as a non-root user against that SLURM.
#
# condaforge/miniforge3 is an Ubuntu 24.04 image, so both `apt-get install
# slurm-wlm munge` and `conda env create -f environment.yml` work here.
#
# Grounded in: dask-jobqueue ci/slurm, giovtorres/slurm-docker-cluster.
FROM condaforge/miniforge3:latest

ENV DEBIAN_FRONTEND=noninteractive

# --- SLURM (controller + compute + client) and munge auth ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        slurm-wlm \
        slurm-client \
        munge \
        libmunge2 \
        tini \
        procps \
        iproute2 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- non-root user that submits/owns the SLURM jobs ---
RUN useradd --create-home --shell /bin/bash slurmuser

# --- runtime dirs + ownership for the slurm/munge daemons ---
RUN mkdir -p \
        /var/spool/slurm /var/spool/slurmd /var/lib/slurm /var/log/slurm /run/slurm \
        /run/munge /var/log/munge /var/lib/munge /etc/munge /etc/slurm \
    && chown -R slurm:slurm /var/spool/slurm /var/spool/slurmd /var/lib/slurm /var/log/slurm /run/slurm \
    && chown -R munge:munge /run/munge /etc/munge /var/lib/munge /var/log/munge

# --- munge key (create-munge-key was removed in munge 0.5.14; dd is portable) ---
RUN dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null \
    && chown munge:munge /etc/munge/munge.key \
    && chmod 0400 /etc/munge/munge.key

# --- SLURM config (hostname placeholder patched at boot) ---
COPY tests/integration_tests/cluster_local/docker/slurm.conf /etc/slurm/slurm.conf
RUN chown slurm:slurm /etc/slurm/slurm.conf && chmod 0644 /etc/slurm/slurm.conf

# --- QUEENS environment baked in (the dask worker's python = this env) ---
WORKDIR /opt/queens
COPY . /opt/queens
# Pre-install CPU-only torch (as the GitHub CI action does) so the much larger
# default GPU build is not pulled in, then install QUEENS with its pinned
# runtime + dev deps (the latter provide pytest).
RUN conda env create -f environment.yml \
    && conda run -n queens pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && conda run -n queens pip install -e .[safe_develop] \
    && conda clean -afy \
    && chown -R slurmuser:slurmuser /opt/queens

COPY tests/integration_tests/cluster_local/docker/start-slurm.sh /usr/local/bin/start-slurm.sh
RUN chmod +x /usr/local/bin/start-slurm.sh

# tini (PID 1) reaps the backgrounded munged/slurmctld/slurmd cleanly.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/start-slurm.sh"]
CMD ["bash", "-lc", "echo 'Pass a command to run as slurmuser'"]
