#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Cluster scheduler for QUEENS runs."""

import logging
import time
from pathlib import Path
from typing import Sequence

from dask.distributed import Client

from queens.schedulers._cluster_base import _BaseCluster
from queens.utils.config_directories import experiment_directory  # Do not change this import!
from queens.utils.config_directories import create_directory
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class Cluster(_BaseCluster):
    """Cluster (remote) scheduler for QUEENS."""

    @log_init_args
    def __init__(
        self,
        experiment_name,
        workload_manager,
        walltime,
        remote_connection,
        num_jobs=1,
        min_jobs=0,
        num_procs=1,
        num_nodes=1,
        queue=None,
        cluster_internal_address=None,
        restart_workers=False,
        allowed_failures=5,
        verbose=True,
        experiment_base_dir=None,
        overwrite_existing_experiment=False,
        job_script_prologue=None,
    ):
        """Init method for the remote cluster scheduler.

        The total number of cores per job is given by num_procs*num_nodes.

        Args:
            experiment_name (str): Name of the current experiment
            workload_manager (str): Workload manager ("pbs" or "slurm")
            walltime (str): Walltime for each worker job. Format (hh:mm:ss)
            remote_connection (RemoteConnection): SSH connection to the remote host
            num_jobs (int, opt): Maximum number of parallel jobs
            min_jobs (int, opt): Minimum number of active workers for the cluster
            num_procs (int, opt): Number of processors per job per node
            num_nodes (int, opt): Number of cluster nodes per job
            queue (str, opt): Destination queue for each worker job
            cluster_internal_address (str, opt): Internal address of cluster
            restart_workers (bool): If True, restart workers after each finished job. For larger
                jobs (>1min) this should be set to True in most cases.
            allowed_failures (int): Number of allowed failures for a task before an error is raised
            verbose (bool, opt): Verbosity of evaluations. Defaults to True.
            experiment_base_dir (str, Path): Base directory for the simulation outputs
            overwrite_existing_experiment (bool): If True, overwrite experiment directory if it
                exists already. If False, prompt user for confirmation before overwriting.
            job_script_prologue (list, opt): List of commands to be executed before starting a
                worker.
        """
        self.remote_connection = remote_connection
        self.remote_connection.open()

        # sync remote source code with local state
        self.remote_connection.sync_remote_repository()

        super().__init__(
            experiment_name=experiment_name,
            workload_manager=workload_manager,
            walltime=walltime,
            num_jobs=num_jobs,
            min_jobs=min_jobs,
            num_procs=num_procs,
            num_nodes=num_nodes,
            queue=queue,
            cluster_internal_address=cluster_internal_address,
            restart_workers=restart_workers,
            allowed_failures=allowed_failures,
            verbose=verbose,
            experiment_base_dir=experiment_base_dir,
            overwrite_existing_experiment=overwrite_existing_experiment,
            job_script_prologue=job_script_prologue,
        )

    def _get_experiment_dir(
        self, experiment_name, experiment_base_dir, overwrite_existing_experiment
    ):
        """Get experiment directory on remote host.

        Args:
            experiment_name (str): name of the current experiment
            experiment_base_dir (str, Path): Remote base directory for the simulation outputs
            overwrite_existing_experiment (bool): If true, continue and overwrite experiment
                directory. If false, prompt user for confirmation before continuing and overwriting.

        Returns:
            experiment_dir (Path): Path to experiment directory on remote host.
        """
        experiment_dir, experiment_dir_exists = self.remote_connection.run_function(
            experiment_directory, experiment_name, experiment_base_dir
        )
        if not overwrite_existing_experiment and experiment_dir_exists:
            self.get_user_confirmation_to_overwrite(experiment_dir)
        self.remote_connection.run_function(create_directory, experiment_dir)

        return experiment_dir

    def _start_cluster(self, dask_cluster_kwargs, dask_cluster_adapt_kwargs):
        """Start a Dask cluster and connect a client on remote host.

        Returns:
            client (Client): Dask client that is connected to and submits computations to a Dask
                cluster.
        """
        local_port, remote_port = self.remote_connection.open_port_forwarding()
        local_port_dashboard, remote_port_dashboard = self.remote_connection.open_port_forwarding()

        scheduler_options = {
            "port": remote_port,
            "dashboard_address": remote_port_dashboard,
            "allowed_failures": self.allowed_failures,
        }
        if self.cluster_internal_address is not None:
            scheduler_options["contact_address"] = f"{self.cluster_internal_address}:{remote_port}"

        dask_cluster_kwargs["scheduler_options"] = scheduler_options

        stdout, stderr = self.remote_connection.start_cluster(
            self.workload_manager,
            dask_cluster_kwargs,
            dask_cluster_adapt_kwargs,
            self.experiment_dir,
        )
        _logger.debug(stdout)
        _logger.debug(stderr)

        for i in range(20, 0, -1):  # 20 tries to connect
            _logger.debug("Trying to connect to Dask Cluster: try #%d", i)
            try:
                client = Client(address=f"localhost:{local_port}", timeout=10)
                break
            except OSError as exc:
                if i == 1:
                    raise OSError(
                        stdout.read().decode("ascii") + stderr.read().decode("ascii")
                    ) from exc
                time.sleep(1)

        return client, local_port_dashboard

    def copy_files_to_experiment_dir(self, paths):
        """Copy file to experiment directory.

        Args:
            paths (Path, list): Paths to files or directories that should be copied to experiment
                directory
        """
        self.remote_connection.copy_to_remote(paths, self.experiment_dir)

    def copy_files_from_experiment_dir(
        self,
        destination: Path | None = None,
        verbose: bool = True,
        exclude: str | Sequence | None = None,
        filters: str | None = None,
    ):
        """Copy files from remote experiment directory to the local machine.

        Args:
            destination (Path): Path to the local directory where the files from the remote
                experiment directory should be copied to. If None, the default base directory
                `~/queens-experiments/` is used.
            verbose: True for verbose
            exclude: Options to exclude
            filters: Filters for rsync
        """
        if destination is None:
            # We use None as experiment_base_dir to get the default base directory, since we do not
            # save it explicitly in the constructor but only use it to construct the location of
            # the remote experiment_dir. That said, we cannot easily retrieve the used
            # experiment_base_dir from self.experiment_dir here because the remote os might be
            # different from the local os, resulting in a different path structure (i.e., different
            # home directory names in case of experiment_base_dir=None). If the user wants to
            # specify a custom destination, they can do so via the destination argument.
            destination = self.local_experiment_dir(
                self.experiment_name, None, overwrite_existing_experiment=True
            ).parent
        self.remote_connection.copy_from_remote(
            self.experiment_dir, destination, verbose, exclude, filters
        )
