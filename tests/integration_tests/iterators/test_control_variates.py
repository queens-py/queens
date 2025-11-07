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
"""Integration tests for the control variates iterator."""

import pytest

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.control_variates import ControlVariates
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


@pytest.fixture(name="parameters")
def fixture_parameters():
    """Parameters for the integration tests."""
    rw = Uniform(lower_bound=0.05, upper_bound=0.15)
    r = Uniform(lower_bound=100, upper_bound=50000)
    tu = Uniform(lower_bound=63070, upper_bound=115600)
    hu = Uniform(lower_bound=990, upper_bound=1110)
    tl = Uniform(lower_bound=63.1, upper_bound=116)
    hl = Uniform(lower_bound=700, upper_bound=820)
    l = Uniform(lower_bound=1120, upper_bound=1680)
    kw = Uniform(lower_bound=9855, upper_bound=12045)
    parameters = Parameters(rw=rw, r=r, tu=tu, hu=hu, tl=tl, hl=hl, l=l, kw=kw)

    return parameters


@pytest.fixture(name="scheduler")
def fixture_scheduler(global_settings):
    """Scheduler for the integration tests."""
    # Set up scheduler
    scheduler = Pool(experiment_name=global_settings.experiment_name)

    return scheduler


@pytest.fixture(name="control_variate")
def fixture_control_variate(parameters, scheduler):
    """Control variate model for the integration tests."""
    # Set up driver.
    driver = Function(parameters=parameters, function="borehole83_lofi")
    # Set up model.
    model = Simulation(scheduler=scheduler, driver=driver)

    return model


@pytest.fixture(name="model_main")
def fixture_model_main(parameters, scheduler):
    """Main model for the integration tests."""
    # Set up driver.
    driver = Function(parameters=parameters, function="borehole83_hifi")
    # Set up model.
    model = Simulation(scheduler=scheduler, driver=driver)

    return model


def test_control_variates_with_given_num_samples(
    global_settings, parameters, model_main, control_variate
):
    """Test function for control variates with a given number of samples."""
    # Number of samples on the cross-model estimator.
    n0 = 100

    # Set up iterator.
    iterator = ControlVariates(
        model=model_main,
        control_variate=control_variate,
        parameters=parameters,
        global_settings=global_settings,
        seed=42,
        num_samples=n0,
        num_samples_cv=10 * n0,
        use_optimal_num_samples=False,
    )

    # Run iterator and load results.
    run_iterator(iterator=iterator, global_settings=global_settings)
    res = load_result(global_settings.result_file(".pickle"))

    # Test outputs.
    assert res["mean"] == pytest.approx(77.03460846952085)
    assert res["std"] == pytest.approx(1.3774480043137558)
    assert res["num_samples_cv"] == pytest.approx(1000)
    assert res["mean_cv"] == pytest.approx(61.63815600352344)
    assert res["std_cv_mean_estimator"] == pytest.approx(1.1561278589420407)
    assert res["cv_influence_coeff"] == pytest.approx(1.1296035845358712)


def test_control_variates_with_optimal_num_samples(
    global_settings, parameters, model_main, control_variate
):
    """Test function for control variates with optimal number of samples."""
    # Number of samples on the cross-model estimator.
    n0 = 4
    # Cost of evaluating the main model.
    cost_model_main = 1
    # Cost of evaluating the control variate.
    cost_control_variate = 0.9999999

    # Set up iterator.
    iterator = ControlVariates(
        model=model_main,
        control_variate=control_variate,
        parameters=parameters,
        global_settings=global_settings,
        seed=42,
        num_samples=n0,
        num_samples_cv=10 * n0,
        use_optimal_num_samples=True,
        cost_model=cost_model_main,
        cost_cv=cost_control_variate,
    )

    # Run iterator and load results.
    run_iterator(iterator=iterator, global_settings=global_settings)
    res = load_result(global_settings.result_file(".pickle"))

    # Test outputs.
    assert res["mean"] == pytest.approx(77.6457414342444)
    assert res["std"] == pytest.approx(0.039169722018672436)
    assert res["num_samples_cv"] == pytest.approx(1353264)
    assert res["mean_cv"] == pytest.approx(61.78825592166509)
    assert res["std_cv_mean_estimator"] == pytest.approx(0.03117012579709094)
    assert res["cv_influence_coeff"] == pytest.approx(1.2566383731008297)
    assert res["sample_ratio"] == pytest.approx(338316.21441286104)
