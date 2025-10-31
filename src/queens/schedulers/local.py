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
"""Local scheduler for QUEENS runs."""

import logging

from dask.distributed import Client, LocalCluster

from queens.schedulers._dask import Dask
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class Local(Dask):
    """Local scheduler class for QUEENS."""

    @log_init_args
    def __init__(
        self,
        experiment_name,
        num_jobs=1,
        num_procs=1,
        restart_workers=False,
        verbose=True,
        experiment_base_dir=None,
        overwrite_existing_experiment=False,
    ):
        """Initialize local scheduler.

        Args:
            experiment_name (str): name of the current experiment
            num_jobs (int, opt): Maximum number of parallel jobs
            num_procs (int, opt): number of processors per job
            restart_workers (bool): If true, restart workers after each finished job. Try setting it
                                    to true in case you are experiencing memory-leakage warnings.
            verbose (bool, opt): Verbosity of evaluations. Defaults to True.
            experiment_base_dir (str, Path): Base directory for the simulation outputs
            overwrite_existing_experiment (bool): If true, overwrite experiment directory if it
                exists already. If false, prompt user for confirmation before overwriting.
        """
        # pylint: disable=duplicate-code
        experiment_dir = self.local_experiment_dir(
            experiment_name, experiment_base_dir, overwrite_existing_experiment
        )
        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            num_jobs=num_jobs,
            num_procs=num_procs,
            restart_workers=restart_workers,
            verbose=verbose,
        )

    def _start_cluster_and_connect_client(self):
        """Start a Dask cluster and a client that connects to it.

        Returns:
               client (Client): Dask client that is connected to and submits computations to a
                                Dask cluster.
        """
        cluster = LocalCluster(
            n_workers=self.num_jobs,
            processes=True,
            threads_per_worker=1,  # keep this hardcoded to 1,
            # the number of threads for the mpi run is handled by the driver.
            silence_logs=False,
        )
        client = Client(cluster)
        _logger.info(
            "To view the Dask dashboard open this link in your browser: %s",
            client.dashboard_link,
        )
        return client

    def restart_worker(self, worker):
        """Restart a worker.

        Args:
            worker (str, tuple): Worker to restart. This can be a worker address, name, or a both.
        """
        self.client.restart_workers(workers=list(worker))
