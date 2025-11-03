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
"""QUEENS dask scheduler parent class."""

import abc
import logging
import time

import numpy as np
import tqdm
from dask.distributed import as_completed

from queens.schedulers._scheduler import Scheduler
from queens.utils.printing import get_str_table

_logger = logging.getLogger(__name__)

SHUTDOWN_CLIENTS = []


class Dask(Scheduler):
    """Abstract base class for schedulers in QUEENS.

    Attributes:
        num_procs (int): Number of processors per job
        client (Client): Dask client that connects to and submits computation to a Dask cluster
        restart_workers (bool): If True, restart workers after each finished job
    """

    def __init__(
        self,
        experiment_name,
        experiment_dir,
        num_jobs,
        num_procs,
        restart_workers,
        verbose=True,
    ):
        """Initialize scheduler.

        Args:
            experiment_name (str): Name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.
            num_jobs (int): Maximum number of parallel jobs
            num_procs (int): Number of processors per job
            restart_workers (bool): If True, restart workers after each finished job
            verbose (bool, opt): Verbosity of evaluations. Defaults to True.
        """
        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            num_jobs=num_jobs,
            verbose=verbose,
        )
        self.num_procs = num_procs
        self.restart_workers = restart_workers

        self.client = None
        self.start_cluster_and_connect_client()

    @abc.abstractmethod
    def _start_cluster_and_connect_client(self):
        """Start a Dask cluster and a client that connects to it.

        Returns:
            client (Client): Dask client that is connected to and submits computations to a Dask
                cluster.
        """

    def start_cluster_and_connect_client(self):
        """Start a Dask cluster and a client that connects to it."""
        if self.client is None or self.client.status == "closed":
            client = self._start_cluster_and_connect_client()
            self.register_shutdown(client)
            self.client = client

    def register_shutdown(self, client):
        """Register shutdown callback.

        The Dask client and cluster will be shut down when leaving the GlobalSettings context.

        Args:
            client (Client): Dask client that is connected to and submits computations to a Dask
                cluster.
        """
        global SHUTDOWN_CLIENTS  # pylint: disable=global-variable-not-assigned
        SHUTDOWN_CLIENTS.append(client.shutdown)

    def evaluate(self, samples, driver, job_ids=None):
        """Submit jobs to driver.

        Args:
            samples (np.array): Array of samples
            driver (Driver): Driver object that runs simulation
            job_ids (lst, opt): List of job IDs corresponding to samples

        Returns:
            result_dict (dict): Dictionary containing results
        """
        self.start_cluster_and_connect_client()

        if self.restart_workers:
            # This is necessary, because the subprocess in the driver does not get killed
            # sometimes when the worker is restarted.
            def run_driver(*args, **kwargs):
                time.sleep(5)
                return driver.run(*args, **kwargs)

        else:
            run_driver = driver.run

        if job_ids is None:
            job_ids = self.get_job_ids(len(samples))
        futures = self.client.map(
            run_driver,
            samples,
            job_ids,
            pure=False,
            num_procs=self.num_procs,
            experiment_dir=self.experiment_dir,
            experiment_name=self.experiment_name,
        )

        # The theoretical number of sequential jobs
        num_sequential_jobs = int(np.ceil(len(samples) / self.num_jobs))

        results = {future.key: None for future in futures}
        with tqdm.tqdm(total=len(futures)) as progressbar:
            for future in as_completed(futures):
                results[future.key] = future.result()
                progressbar.update(1)
                if self.restart_workers:
                    worker = list(self.client.who_has(future).values())[0]
                    self.restart_worker(worker)

            if self.verbose:
                elapsed_time = progressbar.format_dict["elapsed"]
                averaged_time_per_job = elapsed_time / num_sequential_jobs

                run_time_dict = {
                    "number of jobs": len(samples),
                    "number of parallel jobs": self.num_jobs,
                    "number of procs": self.num_procs,
                    "total elapsed time": f"{elapsed_time:.3e}s",
                    "average time per parallel job": f"{averaged_time_per_job:.3e}s",
                }
                _logger.info(
                    get_str_table(
                        f"Batch summary for jobs {min(job_ids)} - {max(job_ids)}", run_time_dict
                    )
                )

        result_dict = {"result": [], "gradient": []}
        for result in results.values():
            # We should remove this squeeze! It is only introduced for consistency with old test.
            result_dict["result"].append(np.atleast_1d(np.array(result[0]).squeeze()))
            result_dict["gradient"].append(result[1])
        result_dict["result"] = np.array(result_dict["result"])
        result_dict["gradient"] = np.array(result_dict["gradient"])
        return result_dict

    @abc.abstractmethod
    def restart_worker(self, worker):
        """Restart a worker."""

    async def shutdown_client(self):
        """Shutdown the DASK client."""
        await self.client.shutdown()
