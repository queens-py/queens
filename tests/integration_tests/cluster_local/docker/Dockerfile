# syntax=docker/dockerfile:1
# Single-node SLURM image for the cluster_local integration test.
# ubuntu 22.04 ships slurm 21.08, which runs without cgroups.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# SLURM (controller + compute + client), munge auth, and fetch tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        slurm-wlm \
        slurm-client \
        munge \
        libmunge2 \
        tini \
        procps \
        iproute2 \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user that submits/owns the SLURM jobs.
RUN useradd --create-home --shell /bin/bash slurmuser

# Runtime dirs + ownership for the slurm/munge daemons.
RUN mkdir -p \
        /var/spool/slurm /var/spool/slurmd /var/lib/slurm /var/log/slurm /run/slurm \
        /run/munge /var/log/munge /var/lib/munge /etc/munge /etc/slurm \
    && chown -R slurm:slurm /var/spool/slurm /var/spool/slurmd /var/lib/slurm /var/log/slurm /run/slurm \
    && chown -R munge:munge /run/munge /etc/munge /var/lib/munge /var/log/munge

# Generate the munge key (dd works on all munge versions).
RUN dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null \
    && chown munge:munge /etc/munge/munge.key \
    && chmod 0400 /etc/munge/munge.key

# SLURM config (hostname placeholder patched at boot). No cgroup.conf needed on slurm 21.08.
COPY tests/integration_tests/cluster_local/docker/slurm.conf /etc/slurm/slurm.conf
RUN chown slurm:slurm /etc/slurm/slurm.conf && chmod 0644 /etc/slurm/slurm.conf

# Install pixi.
ARG PIXI_VERSION=v0.68.1
ENV PIXI_HOME=/opt/pixi
ENV PATH=/opt/pixi/bin:$PATH
RUN curl -fsSL https://pixi.sh/install.sh | \
        env PIXI_VERSION="${PIXI_VERSION}" PIXI_NO_PATH_UPDATE=1 bash \
    && pixi --version

# Install QUEENS into the dev pixi environment.
WORKDIR /opt/queens
COPY . /opt/queens
RUN pixi install --locked -e dev \
    && pixi run --locked -e dev install-editable \
    && chown -R slurmuser:slurmuser /opt/queens

COPY tests/integration_tests/cluster_local/docker/start-slurm.sh /usr/local/bin/start-slurm.sh
RUN chmod +x /usr/local/bin/start-slurm.sh

# tini (PID 1) reaps the backgrounded munged/slurmctld/slurmd cleanly.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/start-slurm.sh"]
CMD ["pixi", "run", "--frozen", "-e", "dev", "pytest", "-m", "cluster_local", "tests/integration_tests/cluster_local/", "-v"]
