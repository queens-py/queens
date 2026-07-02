# Single-node SLURM image for the cluster_local integration test.
# ubuntu 22.04 ships SLURM 21.08 -- the last series that runs cgroup-free, so it
# works in an unprivileged container. Builds SLURM + munge + pixi, installs
# QUEENS into the dev env, then runs the cluster_local suite (boot: start-slurm.sh).
FROM ubuntu:22.04

# Accept package defaults (no interactive prompts during build)
ENV DEBIAN_FRONTEND=noninteractive

# Install SLURM, munge auth, init + fetch tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        slurm-wlm \
        slurm-client \
        munge \
        tini \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user to run the jobs (SLURM rejects root)
RUN useradd --create-home --shell /bin/bash slurmuser

# Create daemon dirs, each owned by its service user
RUN mkdir -p \
        /var/spool/slurm /var/spool/slurmd /var/lib/slurm /var/log/slurm /run/slurm \
        /run/munge /var/log/munge /var/lib/munge /etc/munge /etc/slurm \
    && chown -R slurm:slurm /var/spool/slurm /var/spool/slurmd /var/lib/slurm /var/log/slurm /run/slurm \
    && chown -R munge:munge /run/munge /etc/munge /var/lib/munge /var/log/munge

# Generate the shared munge auth key ('dd' works on all munge versions)
RUN dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null \
    && chown munge:munge /etc/munge/munge.key \
    && chmod 0400 /etc/munge/munge.key

# Add SLURM config (hostname patched at boot)
COPY tests/integration_tests/cluster_local/docker/slurm.conf /etc/slurm/slurm.conf
RUN chown slurm:slurm /etc/slurm/slurm.conf && chmod 0644 /etc/slurm/slurm.conf

# Install pixi (version from .pixi-version)
COPY .pixi-version /tmp/.pixi-version
ENV PIXI_HOME=/opt/pixi
ENV PATH=/opt/pixi/bin:$PATH
RUN curl -fsSL https://pixi.sh/install.sh | \
        env PIXI_VERSION="$(tr -d '[:space:]' < /tmp/.pixi-version)" PIXI_NO_PATH_UPDATE=1 bash \
    && pixi --version

# Install QUEENS into the dev pixi environment
WORKDIR /opt/queens
COPY . /opt/queens
RUN pixi install --locked -e dev \
    && pixi run --locked -e dev install-editable \
    && chown -R slurmuser:slurmuser /opt/queens

# Add the boot script that starts SLURM
COPY tests/integration_tests/cluster_local/docker/start-slurm.sh /usr/local/bin/start-slurm.sh
RUN chmod +x /usr/local/bin/start-slurm.sh

# Start tini as the main process so it cleans up
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/start-slurm.sh"]
CMD ["pixi", "run", "--frozen", "-e", "dev", "pytest", "-m", "cluster_local", "tests/integration_tests/cluster_local/", "-v"]
