#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2025, QUEENS contributors.
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
"""Integration tests for the control varaites iterator."""

import pytest

from queens.distributions.uniform import UniformDistribution
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.control_variates_iterator import ControlVariatesIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result


def test_control_variates_with_given_num_samples(global_settings):
    """Test function for control variates with a given number of samples."""
    n0 = 100

    # Parameters
    rw = UniformDistribution(lower_bound=0.05, upper_bound=0.15)
    r = UniformDistribution(lower_bound=100, upper_bound=50000)
    tu = UniformDistribution(lower_bound=63070, upper_bound=115600)
    hu = UniformDistribution(lower_bound=990, upper_bound=1110)
    tl = UniformDistribution(lower_bound=63.1, upper_bound=116)
    hl = UniformDistribution(lower_bound=700, upper_bound=820)
    l = UniformDistribution(lower_bound=1120, upper_bound=1680)
    kw = UniformDistribution(lower_bound=9855, upper_bound=12045)
    parameters = Parameters(rw=rw, r=r, tu=tu, hu=hu, tl=tl, hl=hl, l=l, kw=kw)

    # Set up scheduler
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)

    # Set up drivers
    driver0 = FunctionDriver(parameters=parameters, function="borehole83_lofi")
    driver1 = FunctionDriver(parameters=parameters, function="borehole83_hifi")

    # Set up models
    model0 = SimulationModel(scheduler=scheduler, driver=driver0)
    model1 = SimulationModel(scheduler=scheduler, driver=driver1)

    iterator = ControlVariatesIterator(
        model=model1,
        control_variate=model0,
        parameters=parameters,
        global_settings=global_settings,
        seed=42,
        num_samples=n0,
        num_samples_cv=10 * n0,
        use_optimal_num_samples=False,
    )

    run_iterator(iterator=iterator, global_settings=global_settings)
    res = load_result(global_settings.result_file(".pickle"))

    assert res["mean"] == pytest.approx(77.03460846952085)
    assert res["std"] == pytest.approx(1.3774480043137558)
    assert res["num_samples_cv"] == pytest.approx(1000)
    assert res["mean_cv"] == pytest.approx(61.63815600352344)
    assert res["std_cv_mean_estimator"] == pytest.approx(1.1561278589420407)
    assert res["alpha"] == pytest.approx(1.1296035845358712)


def test_control_variates_with_optimal_num_samples(global_settings):
    """Test function for control variates with optimal number of samples."""
    n0 = 4

    # Parameters
    rw = UniformDistribution(lower_bound=0.05, upper_bound=0.15)
    r = UniformDistribution(lower_bound=100, upper_bound=50000)
    tu = UniformDistribution(lower_bound=63070, upper_bound=115600)
    hu = UniformDistribution(lower_bound=990, upper_bound=1110)
    tl = UniformDistribution(lower_bound=63.1, upper_bound=116)
    hl = UniformDistribution(lower_bound=700, upper_bound=820)
    l = UniformDistribution(lower_bound=1120, upper_bound=1680)
    kw = UniformDistribution(lower_bound=9855, upper_bound=12045)
    parameters = Parameters(rw=rw, r=r, tu=tu, hu=hu, tl=tl, hl=hl, l=l, kw=kw)

    # Set up scheduler
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)

    # Set up drivers
    driver0 = FunctionDriver(parameters=parameters, function="borehole83_lofi")
    driver1 = FunctionDriver(parameters=parameters, function="borehole83_hifi")

    # Set up models
    model0 = SimulationModel(scheduler=scheduler, driver=driver0)
    model1 = SimulationModel(scheduler=scheduler, driver=driver1)

    c1 = 1
    c0 = 0.9999999

    iterator = ControlVariatesIterator(
        model=model1,
        control_variate=model0,
        parameters=parameters,
        global_settings=global_settings,
        seed=42,
        num_samples=n0,
        num_samples_cv=10 * n0,
        use_optimal_num_samples=True,
        cost_model=c1,
        cost_cv=c0,
    )

    run_iterator(iterator=iterator, global_settings=global_settings)
    res = load_result(global_settings.result_file(".pickle"))

    assert res["mean"] == pytest.approx(77.6457414342444)
    assert res["std"] == pytest.approx(0.039169722018672436)
    assert res["num_samples_cv"] == pytest.approx(1353264)
    assert res["mean_cv"] == pytest.approx(61.78825592166509)
    assert res["std_cv_mean_estimator"] == pytest.approx(0.03117012579709094)
    assert res["alpha"] == pytest.approx(1.2566383731008297)
    assert res["beta"] == pytest.approx(338316.21441286104)
