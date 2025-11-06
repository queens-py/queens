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
"""Integration test for the Polynomial Chaos iterator."""

import pytest

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.polynomial_chaos import PolynomialChaos
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


def test_polynomial_chaos_pseudo_spectral_borehole(global_settings):
    """Test case for the PC iterator using a pseudo spectral approach."""
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
    iterator = PolynomialChaos(
        approach="pseudo_spectral",
        seed=42,
        num_collocation_points=50,
        sampling_rule="gaussian",
        sparse=True,
        polynomial_order=2,
        result_description={"write_results": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))
    assert results["mean"] == pytest.approx(61.78966587)
    assert results["covariance"] == pytest.approx([1312.23414971])


def test_polynomial_chaos_collocation_borehole(global_settings):
    """Test for the PC iterator using a collocation approach."""
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
    iterator = PolynomialChaos(
        approach="collocation",
        seed=42,
        num_collocation_points=50,
        sampling_rule="sobol",
        polynomial_order=2,
        result_description={"write_results": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))
    assert results["mean"] == pytest.approx(62.05018243)
    assert results["covariance"] == pytest.approx([1273.81372103])
