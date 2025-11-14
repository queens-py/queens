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
"""Pool scheduler for QUEENS runs."""

import logging
from collections.abc import Iterable
from functools import partial

from tqdm import tqdm

from queens.schedulers._scheduler import Scheduler, SchedulerCallableSignature
from queens.utils.logger_settings import log_init_args
from queens.utils.pool import create_pool

_logger = logging.getLogger(__name__)


class Pool(Scheduler):
    """Pool scheduler class for QUEENS.

    Attributes:
        pool (pathos pool): Multiprocessing pool.
    """

    @log_init_args
    def __init__(
        self,
        experiment_name,
        num_jobs=1,
        verbose=True,
        experiment_base_dir=None,
        overwrite_existing_experiment=False,
    ):
        """Initialize Pool.

        Args:
            experiment_name (str): Name of the current experiment
            num_jobs (int, opt): Maximum number of parallel jobs
            verbose (bool, opt): Verbosity of evaluations. Defaults to True.
            experiment_base_dir (str, Path): Base directory for the simulation outputs
            overwrite_existing_experiment (bool): If True, overwrite experiment directory if it
                exists already. If False, prompt user for confirmation before overwriting.
        """
        # pylint: disable=duplicate-code
        experiment_dir = self.local_experiment_dir(
            experiment_name, experiment_base_dir, overwrite_existing_experiment
        )
        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            num_jobs=num_jobs,
            verbose=verbose,
        )
        self.pool = create_pool(num_jobs)

    def evaluate(
        self, samples: Iterable, function: SchedulerCallableSignature, job_ids: Iterable = None
    ) -> dict:
        """Submit jobs to driver.

        Args:
            samples (np.array): Array of samples
            function (Callable): Callable to evaluate in the scheduler
            job_ids (lst, opt): List of job IDs corresponding to samples

        Returns:
            result_dict (dict): Dictionary containing results
        """
        partial_function = partial(
            function,
            num_procs=1,
            experiment_dir=self.experiment_dir,
            experiment_name=self.experiment_name,
        )
        if job_ids is None:
            job_ids = self.get_job_ids(len(samples))
        # Pool or no pool
        if self.pool:
            results = self.pool.map(partial_function, samples, job_ids)
        elif self.verbose:
            results = list(map(partial_function, tqdm(samples), job_ids))
        else:
            results = list(map(partial_function, samples, job_ids))

        return results
