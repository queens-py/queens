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
from datetime import timedelta

from dask.distributed import Client
from dask_jobqueue import PBSCluster, SLURMCluster

from queens.schedulers._dask import Dask
from queens.utils.config_directories import experiment_directory  # Do not change this import!
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


class Cluster(Dask):
    """Cluster scheduler for QUEENS."""

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
    ):
        """Init method for the cluster scheduler.

        The total number of cores per job is given by num_procs*num_nodes.

        Args:
            experiment_name (str): name of the current experiment
            workload_manager (str): Workload manager ("pbs" or "slurm")
            walltime (str): Walltime for each worker job. Format (hh:mm:ss)
            remote_connection (RemoteConnection): ssh connection to the remote host
            num_jobs (int, opt): Maximum number of parallel jobs
            min_jobs (int, opt): Minimum number of active workers for the cluster
            num_procs (int, opt): Number of processors per job per node
            num_nodes (int, opt): Number of cluster nodes per job
            queue (str, opt): Destination queue for each worker job
            cluster_internal_address (str, opt): Internal address of cluster
            restart_workers (bool): If true, restart workers after each finished job. For larger
                                    jobs (>1min) this should be set to true in most cases.
            allowed_failures (int): Number of allowed failures for a task before an error is raised
            verbose (bool, opt): Verbosity of evaluations. Defaults to True.
        """
        self.remote_connection = remote_connection
        self.remote_connection.open()

        # sync remote source code with local state
        self.remote_connection.sync_remote_repository()

        # get the path of the experiment directory on remote host
        experiment_dir = self.remote_connection.run_function(experiment_directory, experiment_name)
        _logger.debug(
            "experiment directory on %s@%s: %s",
            self.remote_connection.user,
            self.remote_connection.host,
            experiment_dir,
        )

        # collect all settings for the dask cluster
        dask_cluster_options = get_option(VALID_WORKLOAD_MANAGERS, workload_manager)
        job_extra_directives = dask_cluster_options["job_extra_directives"](num_nodes, num_procs)
        job_directives_skip = dask_cluster_options["job_directives_skip"]
        if queue is None:
            job_directives_skip.append("#SBATCH -p")

        hours, minutes, seconds = map(int, walltime.split(":"))
        walltime_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)

        # Increase jobqueue walltime by 5 minutes to kill dask workers in time
        walltime = timedelta_to_str(walltime_delta + timedelta(minutes=5))

        # dask worker lifetime = walltime - 3m +/- 2m
        worker_lifetime = str(int((walltime_delta + timedelta(minutes=2)).total_seconds())) + "s"

        local_port, remote_port = self.remote_connection.open_port_forwarding()
        local_port_dashboard, remote_port_dashboard = self.remote_connection.open_port_forwarding()

        scheduler_options = {
            "port": remote_port,
            "dashboard_address": remote_port_dashboard,
            "allowed_failures": allowed_failures,
        }
        if cluster_internal_address:
            scheduler_options["contact_address"] = f"{cluster_internal_address}:{remote_port}"
        dask_cluster_kwargs = {
            "job_name": experiment_name,
            "queue": queue,
            "memory": "10TB",
            "scheduler_options": scheduler_options,
            "walltime": walltime,
            "log_directory": str(experiment_dir),
            "job_directives_skip": job_directives_skip,
            "job_extra_directives": [job_extra_directives],
            "worker_extra_args": ["--lifetime", worker_lifetime, "--lifetime-stagger", "2m"],
            # keep this hardcoded to 1, the number of threads for the mpi run is handled by
            # job_extra_directives. Note that the number of workers is not the number of parallel
            # simulations!
            "cores": 1,
            "processes": 1,
            "n_workers": 1,
        }
        dask_cluster_adapt_kwargs = {
            "minimum_jobs": min_jobs,
            "maximum_jobs": num_jobs,
        }

        # actually start the dask cluster on remote host
        stdout, stderr = self.remote_connection.start_cluster(
            workload_manager,
            dask_cluster_kwargs,
            dask_cluster_adapt_kwargs,
            experiment_dir,
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

        _logger.debug("Submitting dummy job to check basic functionality of client.")
        client.submit(lambda: "Dummy job").result(timeout=180)
        _logger.debug("Dummy job was successful.")
        _logger.info(
            "To view the Dask dashboard open this link in your browser: "
            "http://localhost:%i/status",
            local_port_dashboard,
        )

        # pylint: disable=duplicate-code
        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            num_jobs=num_jobs,
            num_procs=num_procs,
            client=client,
            restart_workers=restart_workers,
            verbose=verbose,
        )
        # pylint: enable=duplicate-code

    def copy_files_to_experiment_dir(self, paths):
        """Copy file to experiment directory.

        Args:
            paths (Path, list): paths to files or directories that should be copied to experiment
                                directory
        """
        destination = f"{self.experiment_dir}/"
        self.remote_connection.copy_to_remote(paths, destination)
