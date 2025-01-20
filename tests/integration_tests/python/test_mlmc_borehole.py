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
"""Integration tests for the multilevel Monte Carlo iterator.

The tests are based on the low-fidelity and the high-fidelity Borehole
function.
"""

import pytest

from queens.distributions.uniform import UniformDistribution
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.mlmc_iterator import MLMCIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result


@pytest.fixture(name="parameters")
def fixture_parameters():
    """Parameters for the integration tests of the MLMC iterator."""
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

    return parameters


@pytest.fixture(name="scheduler")
def fixture_scheduler(global_settings):
    """Scheduler for the integration tests of the MLMC iterator."""
    # Set up scheduler.
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)

    return scheduler


@pytest.fixture(name="models")
def fixture_models(parameters, scheduler):
    """Models for the integration tests of the MLMC iterator."""
    # Set up drivers.
    driver0 = FunctionDriver(parameters=parameters, function="borehole83_lofi")
    driver1 = FunctionDriver(parameters=parameters, function="borehole83_hifi")
    # Set up models.
    model0 = SimulationModel(scheduler=scheduler, driver=driver0)
    model1 = SimulationModel(scheduler=scheduler, driver=driver1)

    return [model0, model1]


def test_mlmc_borehole_given_num_samples(global_settings, parameters, models):
    """Test case for the iterator with a given number of samples."""
    # Set up iterator.
    iterator = MLMCIterator(
        seed=42,
        num_samples=[1000, 100],
        models=models,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Run iterator and load results.
    run_iterator(iterator=iterator, global_settings=global_settings)
    result = load_result(path_to_result_file=global_settings.result_file(".pickle"))

    # Test outputs.
    assert result["mean"] == pytest.approx(76.52224796054254)
    assert result["var"] == pytest.approx(2.0511312684237075)
    assert result["std"] == pytest.approx(1.4321771079107875)
    assert result["mean_estimators"] == pytest.approx([60.4546131, 16.06763486])
    assert result["var_estimators"] == pytest.approx([1266.89995688, 78.42313115])
    assert result["num_samples"] == pytest.approx([1000, 100])


def test_mlmc_borehole_bootstrap(global_settings, parameters, models):
    """Test case for the bootstrap estimate of the MLMC standard deviation."""
    # Set up iterator.
    iterator = MLMCIterator(
        seed=42,
        num_samples=[1000, 100],
        models=models,
        parameters=parameters,
        global_settings=global_settings,
        num_bootstrap_samples=200,
    )

    # Run iterator and load results.
    run_iterator(iterator=iterator, global_settings=global_settings)
    result = load_result(path_to_result_file=global_settings.result_file(".pickle"))

    assert result["mean"] == pytest.approx(76.52224796054254)
    assert result["var"] == pytest.approx(2.0511312684237075)
    assert result["std"] == pytest.approx(1.4321771079107875)
    assert result["mean_estimators"] == pytest.approx([60.4546131, 16.06763486])
    assert result["var_estimators"] == pytest.approx([1266.89995688, 78.42313115])
    assert result["num_samples"] == pytest.approx([1000, 100])
    assert result["std_bootstrap"] == pytest.approx(1.4177144502392238)


def test_mlmc_borehole_optimal_num_samples(global_settings, parameters, models):
    """Test case for the iterator with an optimal number of samples."""
    # Set up iterator.
    iterator_optimal = MLMCIterator(
        seed=42,
        num_samples=[1000, 100],
        models=models,
        parameters=parameters,
        global_settings=global_settings,
        use_optimal_num_samples=True,
        cost_models=[1, 1000],
    )

    # Run iterator and load results.
    run_iterator(iterator=iterator_optimal, global_settings=global_settings)
    result = load_result(path_to_result_file=global_settings.result_file(".pickle"))

    # Test outputs.
    assert result["mean"] == pytest.approx(77.9589082063506)
    assert result["var"] == pytest.approx(0.8857499559585561)
    assert result["std"] == pytest.approx(0.9411428987983472)
    assert result["mean_estimators"] == pytest.approx([61.89127335, 16.06763486])
    assert result["var_estimators"] == pytest.approx([1290.91108238, 78.42313115])
    assert result["num_samples"] == pytest.approx([12716, 100])
