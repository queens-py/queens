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

import logging
import random
import time
from typing import TYPE_CHECKING

import numpy as np
from dask.distributed import progress
from distributed import WorkerPlugin

from queens.schedulers._scheduler import Scheduler
from queens.utils.printing import get_str_table

if TYPE_CHECKING:
    from typing import Any

    from dask.typing import Key
    from distributed.worker import Worker
    from distributed.worker_state_machine import TaskStateState as WorkerTaskStateState
_logger = logging.getLogger(__name__)

SHUTDOWN_CLIENTS = []


class ShutdownAfterFirstTask(WorkerPlugin):
    """Shutdown a worker after the first task is finished."""

    def setup(self, worker):
        """Setup the worker plugin.

        Run when the plugin is attached to a worker.

        Args:
            worker (Worker): The worker that has this plugin.
        """
        self.worker = worker  # pylint: disable=attribute-defined-outside-init
        self.has_shutdown = False  # pylint: disable=attribute-defined-outside-init

    def transition(self, key, start, finish, **kwargs):
        """Called when task changes its state.

        Args:
            key (Key): Key of the task.
            start (WorkerTaskStateState): Start state of the transition. One of waiting, ready,
                                          executing, long-running, memory, error.
            finish (WorkerTaskStateState): Final state of the transition.
            kwargs (Any): More options passed when transitioning
        """
        if start == "executing" and finish == "memory" and not self.has_shutdown:
            self.has_shutdown = True  # pylint: disable=attribute-defined-outside-init

            async def shutdown():
                await self.worker.close_gracefully(reason=f"Shutdown after task {key}")

            self.worker.loop.call_later(2, shutdown)


class Dask(Scheduler):
    """Abstract base class for schedulers in QUEENS.

    Attributes:
        num_procs (int): number of processors per job
        client (Client): Dask client that connects to and submits computation to a Dask cluster
        restart_workers (bool): If true, restart workers after each finished job
    """

    def __init__(
        self,
        experiment_name,
        experiment_dir,
        num_jobs,
        num_procs,
        client,
        restart_workers,
        max_wait_time=0.0,
        verbose=True,
    ):
        """Initialize scheduler.

        Args:
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.
            num_jobs (int): Maximum number of parallel jobs
            num_procs (int): number of processors per job
            client (Client): Dask client that connects to and submits computation to a Dask cluster
            restart_workers (bool): If true, restart workers after each finished job
            max_wait_time (float, opt): Maximum wait time before a job is started in seconds.
                                        Defaults to 0.0 (so all jobs are started immediately).
                                        The wait time for each job is uniformly distributed between
                                        0 and max_wait_time.
                                        Starting many jobs (several hundreds) at the same time can
                                        overwhelm the hardware of a resource, the randomized wait
                                        time can help to spread the load.
            verbose (bool, opt): Verbosity of evaluations. Defaults to True.
        """
        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            num_jobs=num_jobs,
            verbose=verbose,
        )
        self.num_procs = num_procs
        self.client = client
        self.restart_workers = restart_workers
        self.max_wait_time = max_wait_time

        # Register this plugin on all workers
        if self.restart_workers:
            self.client.register_plugin(ShutdownAfterFirstTask(), name="shutdown_after_first")

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
        # the initial batch can overwhelm the hardware of a cluster
        # by starting many jobs at the same time
        # -> introduce a random wait time for jobs to spread the load
        max_wait_time = float(self.max_wait_time)  # this float copy allows pickle serialization

        def run_driver(*args, **kwargs):
            random_wait_time = random.uniform(0.0, max_wait_time)
            time.sleep(random_wait_time)
            return driver.run(*args, **kwargs)

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

        start_time = time.time()
        progress(futures)
        results_values = self.client.gather(futures)

        if self.verbose:
            elapsed_time = time.time() - start_time
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
        # for result in results:
        for result in results_values:
            # We should remove this squeeze! It is only introduced for consistency with old test.
            result_dict["result"].append(np.atleast_1d(np.array(result[0]).squeeze()))
            result_dict["gradient"].append(result[1])
        result_dict["result"] = np.array(result_dict["result"])
        result_dict["gradient"] = np.array(result_dict["gradient"])
        return result_dict

    async def shutdown_client(self):
        """Shutdown the DASK client."""
        await self.client.shutdown()
