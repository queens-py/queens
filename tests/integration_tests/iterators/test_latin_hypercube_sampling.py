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
"""Integration test for the Latin Hyper Cube iterator.

The test is based on the low-fidelity Borehole function.
"""

import numpy as np
import pytest

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.latin_hypercube_sampling import LatinHypercubeSampling
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


def test_latin_hypercube_sampling_borehole83(global_settings):
    """Test case for latin hyper cube iterator."""
    # Parameters
    rw = Uniform(lower_bound=0.05, upper_bound=0.15)
    r = Uniform(lower_bound=100, upper_bound=50000)
    tu = Uniform(lower_bound=63070, upper_bound=115600)
    hu = Uniform(lower_bound=990, upper_bound=1110)
    tl = Uniform(lower_bound=63.1, upper_bound=116)
    hl = Uniform(lower_bound=700, upper_bound=820)
    l = Uniform(lower_bound=1120, upper_bound=1680)
    kw = Uniform(lower_bound=9855, upper_bound=12045)
    parameters = Parameters(rw=rw, r=r, tu=tu, hu=hu, tl=tl, hl=hl, l=l, kw=kw)

    # Setup iterator
    driver = Function(parameters=parameters, function="borehole83_lofi")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = LatinHypercubeSampling(
        seed=42,
        num_samples=1000,
        num_iterations=5,
        result_description={"write_results": True, "plot_results": False},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_allclose(results["mean"], 61.910468085219456, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(results["var"], 1336.5420586597304, rtol=1e-6, atol=1e-12)


@pytest.mark.max_time_for_test(20)
def test_latin_hypercube_sampling_branin78(global_settings):
    """Test Latin hypercube sampling for the high-fidelity Branin function.

    The test samples the high-fidelity Branin benchmark function

        f(x_1, x_2) =
            (-1.275 x_1^2 / pi^2 + 5 x_1 / pi + x_2 - 6)^2
            + (10 - 5 / (4 pi)) cos(x_1) + 10

    with independent uniform input distributions

        x_1 ~ U[-5, 10],
        x_2 ~ U[0, 15].

    For these distributions, the exact moments are analytically defined by

        E[f] = 1 / 15^2 int_{-5}^{10} int_0^{15} f(x_1, x_2) dx_2 dx_1
             = 54.3071982719085,

        Var[f] = 1 / 15^2 int_{-5}^{10} int_0^{15} (f(x_1, x_2) - E[f])^2 dx_2 dx_1
               = 2626.687312415944.

    The assertions below check the deterministic sample statistics of the seeded
    Latin hypercube run, not the exact distribution moments.
    """
    # Parameters
    x1 = Uniform(lower_bound=-5, upper_bound=10)
    x2 = Uniform(lower_bound=0, upper_bound=15)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = Function(parameters=parameters, function="branin78_hifi")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = LatinHypercubeSampling(
        seed=42,
        num_samples=1000,
        num_iterations=10,
        result_description={"write_results": True, "plot_results": False},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_allclose(results["mean"], 54.25531895299926, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(results["var"], 2483.786406285974, rtol=1e-6, atol=1e-12)
