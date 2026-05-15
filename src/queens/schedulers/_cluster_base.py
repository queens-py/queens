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
"""Base class for QUEENS cluster schedulers."""

import logging
from abc import abstractmethod
from datetime import timedelta

from dask_jobqueue import PBSCluster, SLURMCluster

from queens.schedulers._dask import Dask
from queens.utils.logger_settings import log_init_args
from queens.utils.valid_options import get_option

_logger = logging.getLogger(__name__)

VALID_WORKLOAD_MANAGERS = {
    "slurm": {
        "dask_cluster_cls": SLURMCluster,
        "job_extra_directives": lambda nodes, cores: f"--ntasks={nodes * cores}",
        "job_directives_skip": [
            "#SBATCH -n 1",
            "#SBATCH --mem=",
            "#SBATCH --cpus-per-task=",
        ],
    },
    "pbs": {
        "dask_cluster_cls": PBSCluster,
        "job_extra_directives": lambda nodes, cores: f"-l nodes={nodes}:ppn={cores}",
        "job_directives_skip": ["#PBS -l select"],
    },
}


def timedelta_to_str(timedelta_obj):
    """Format a timedelta object to str.

    This function seems unnecessarily complicated, but unfortunately the datetime library does not
     support this formatting for timedeltas. Returns the format HH:MM:SS.

    Args:
        timedelta_obj (datetime.timedelta): Timedelta object to format

    Returns:
        str: String of the timedelta object
    """
    # Time in seconds
    time_in_seconds = int(timedelta_obj.total_seconds())
    (minutes, seconds) = divmod(time_in_seconds, 60)
    (hours, minutes) = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def _initialize_dask_cluster(
    logger, dask_cluster_cls, dask_cluster_kwargs, dask_cluster_adapt_kwargs, experiment_dir
):
    """Initialize a Dask cluster.

    Start dask cluster, adapt it to the requested worker settings, and
    write jobscript.
    """
    logger.info("Starting dask cluster of type: %s", dask_cluster_cls)
    logger.debug("Dask cluster kwargs:")
    logger.debug(dask_cluster_kwargs)
    cluster = dask_cluster_cls(**dask_cluster_kwargs)

    logger.info("Adapting dask cluster settings")
    logger.debug("Dask cluster adapt kwargs:")
    logger.debug(dask_cluster_adapt_kwargs)
    cluster.adapt(**dask_cluster_adapt_kwargs)

    logger.info("Dask cluster info:")
    logger.info(cluster)

    dask_jobscript = experiment_dir / "dask_jobscript.sh"
    logger.info("Writing dask jobscript to:")
    logger.info(dask_jobscript)
    dask_jobscript.write_text(str(cluster.job_script()))

    return cluster


class _BaseCluster(Dask):
    """Abstract base class for QUEENS cluster schedulers."""

    @log_init_args
    def __init__(
        self,
        experiment_name,
        workload_manager,
        walltime,
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
        """Init method for the abstract cluster scheduler.

        The total number of cores per job is given by num_procs*num_nodes.

        Args:
            experiment_name (str): Name of the current experiment
            workload_manager (str): Workload manager ("pbs" or "slurm")
            walltime (str): Walltime for each worker job. Format (hh:mm:ss)
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
        self.workload_manager = workload_manager
        self.walltime = walltime
        self.min_jobs = min_jobs
        self.num_nodes = num_nodes
        self.queue = queue
        self.cluster_internal_address = cluster_internal_address
        self.allowed_failures = allowed_failures
        self.job_script_prologue = job_script_prologue

        experiment_dir = self._get_experiment_dir(
            experiment_name, experiment_base_dir, overwrite_existing_experiment
        )

        _logger.debug("experiment directory: %s", experiment_dir)

        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            num_jobs=num_jobs,
            num_procs=num_procs,
            restart_workers=restart_workers,
            verbose=verbose,
        )

    @abstractmethod
    def _get_experiment_dir(
        self, experiment_name, experiment_base_dir, overwrite_existing_experiment
    ):
        """Get experiment directory."""

    @abstractmethod
    def _start_cluster(self, dask_cluster_kwargs, dask_cluster_adapt_kwargs):
        """Start cluster and return connected client and dashboard port."""

    @abstractmethod
    def copy_files_to_experiment_dir(self, paths):
        """Copy file to experiment directory.

        Args:
            paths (Path, list): Paths to files or directories that should be copied to experiment
                directory
        """

    def _start_cluster_and_connect_client(self):
        """Start a Dask cluster and a client that connects to it.

        Returns:
            client (Client): Dask client that is connected to and submits computations to a Dask
                cluster.
        """
        # collect all settings for the dask cluster
        dask_cluster_options = get_option(VALID_WORKLOAD_MANAGERS, self.workload_manager)
        job_extra_directives = dask_cluster_options["job_extra_directives"](
            self.num_nodes, self.num_procs
        )
        job_directives_skip = dask_cluster_options["job_directives_skip"]
        if self.queue is None:
            job_directives_skip.append("#SBATCH -p")

        hours, minutes, seconds = map(int, self.walltime.split(":"))
        walltime_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)

        # Increase jobqueue walltime by 5 minutes to kill dask workers in time
        increased_walltime = timedelta_to_str(walltime_delta + timedelta(minutes=5))

        # dask worker lifetime = walltime - 3m +/- 2m
        worker_lifetime = str(int((walltime_delta + timedelta(minutes=2)).total_seconds())) + "s"

        dask_cluster_kwargs = {
            "job_name": self.experiment_name,
            "queue": self.queue,
            "memory": "10TB",
            "walltime": increased_walltime,
            "log_directory": str(self.experiment_dir),
            "job_directives_skip": job_directives_skip,
            "job_extra_directives": [job_extra_directives],
            "worker_extra_args": ["--lifetime", worker_lifetime, "--lifetime-stagger", "2m"],
            "job_script_prologue": self.job_script_prologue,
            # keep this hardcoded to 1, the number of threads for the mpi run is handled by
            # job_extra_directives. Note that the number of workers is not the number of
            # parallel simulations!
            "cores": 1,
            "processes": 1,
            "n_workers": 1,
        }
        dask_cluster_adapt_kwargs = {
            "minimum_jobs": self.min_jobs,
            "maximum_jobs": self.num_jobs,
        }

        # start dask cluster
        client, dashboard_port = self._start_cluster(dask_cluster_kwargs, dask_cluster_adapt_kwargs)

        _logger.debug("Submitting dummy job to check basic functionality of client.")
        client.submit(lambda: "Dummy job").result(timeout=180)
        _logger.debug("Dummy job was successful.")
        _logger.info(
            "To view the Dask dashboard open this link in your browser: "
            "http://localhost:%i/status",
            dashboard_port,
        )
        return client

    def restart_worker(self, worker):
        """Restart a worker.

        This method retires a dask worker. The Client.adapt method of dask takes cares of submitting
        new workers subsequently.

        Args:
            worker (str, tuple): Worker to restart. This can be a worker address, name, or a both.
        """
        self.client.retire_workers(workers=list(worker))

    @staticmethod
    def delete_experiment_dir_if_empty(_):
        """The remote experiment directory will never be empty, so pass."""
