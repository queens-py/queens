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
"""Integration test for the ClusterLocal scheduler.

This test runs a small QUEENS Monte Carlo study through ``ClusterLocal`` against
a real single-node SLURM installation. It is marked ``cluster_local`` and
therefore only runs in the dedicated SLURM environment (e.g. the SLURM CI
container), not in the regular test suite.
"""

import logging

import numpy as np
import pytest

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.monte_carlo import MonteCarlo
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.cluster_local import ClusterLocal
from queens.utils.io import load_result

_logger = logging.getLogger(__name__)

pytestmark = pytest.mark.cluster_local


def test_cluster_local_monte_carlo(global_settings, tmp_path):
    """Run a Monte Carlo study on a local SLURM cluster, end to end."""
    num_samples = 4
    bound = float(np.pi)

    x1 = Uniform(lower_bound=-bound, upper_bound=bound)
    x2 = Uniform(lower_bound=-bound, upper_bound=bound)
    x3 = Uniform(lower_bound=-bound, upper_bound=bound)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    driver = Function(parameters=parameters, function="ishigami90")
    scheduler = ClusterLocal(
        experiment_name=global_settings.experiment_name,
        workload_manager="slurm",
        walltime="00:05:00",
        num_jobs=1,
        min_jobs=1,
        num_procs=1,
        num_nodes=1,
        experiment_base_dir=tmp_path,
        overwrite_existing_experiment=True,
    )
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = MonteCarlo(
        seed=42,
        num_samples=num_samples,
        result_description={"write_results": True, "plot_results": False},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    run_iterator(iterator, global_settings=global_settings)

    results = load_result(global_settings.result_file(".pickle"))
    output = np.asarray(results["raw_output_data"]["result"])

    assert output.shape[0] == num_samples
    assert np.all(np.isfinite(output))
