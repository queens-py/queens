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
"""Fixtures needed across unit_tests for iterators."""

from copy import deepcopy

import pytest

from queens.distributions.lognormal import LogNormal
from queens.distributions.normal import Normal
from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.local import Local


@pytest.fixture(name="ishigami_90_uniform")
def fixture_ishigami_90_uniform(default_parameters_uniform_3d):
    """Default simulation model."""
    driver = Function(parameters=default_parameters_uniform_3d, function="ishigami90")
    scheduler = Local(experiment_name="dummy_experiment_name")
    model = Simulation(scheduler=scheduler, driver=driver)
    return model


@pytest.fixture(name="ishigami_90_mixed")
def fixture_ishigami_90_mixed(default_parameters_mixed):
    """Default simulation model."""
    driver = Function(parameters=default_parameters_mixed, function="ishigami90")
    scheduler = Local(experiment_name="dummy_experiment_name")
    model = Simulation(scheduler=scheduler, driver=driver)
    return model


@pytest.fixture(name="default_parameters_uniform_2d")
def fixture_default_parameters_uniform_2d():
    """Parameters with 2 uniform distributions."""
    x1 = Uniform(lower_bound=-2, upper_bound=2)
    x2 = Uniform(lower_bound=-2, upper_bound=2)
    return Parameters(x1=x1, x2=x2)


@pytest.fixture(name="default_parameters_uniform_3d")
def fixture_default_parameters_uniform_3d():
    """Parameters with 3 uniform distributions."""
    random_variable = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    return Parameters(
        x1=random_variable, x2=deepcopy(random_variable), x3=deepcopy(random_variable)
    )


@pytest.fixture(name="default_parameters_mixed")
def fixture_default_parameters_mixed():
    """Parameters with different distributions."""
    x1 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = Normal(mean=0, covariance=4)
    x3 = LogNormal(normal_mean=0.3, normal_covariance=1)
    return Parameters(x1=x1, x2=x2, x3=x3)
