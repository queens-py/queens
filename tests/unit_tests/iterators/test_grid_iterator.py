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
"""Unit tests for the GridIterator."""

from copy import deepcopy

import numpy as np
import pytest
from mock import Mock

from queens.distributions.uniform import Uniform
from queens.iterators.grid_iterator import GridIterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters


@pytest.fixture(name="grid_dict_one")
def fixture_grid_dict_one():
    """A grid dictionary with one axis."""
    axis_description = {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"}
    grid_dict_dummy = {"x1": axis_description}
    return grid_dict_dummy


@pytest.fixture(name="grid_dict_two")
def fixture_grid_dict_two():
    """A grid dictionary with two axes."""
    axis_description = {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"}
    grid_dict_dummy = {"x1": axis_description, "x2": axis_description}
    return grid_dict_dummy


@pytest.fixture(name="grid_dict_three")
def fixture_grid_dict_three():
    """A grid dictionary with three axes."""
    axis_description = {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"}
    grid_dict_dummy = {"x1": axis_description, "x2": axis_description, "x3": axis_description}
    return grid_dict_dummy


@pytest.fixture(name="parameters_one")
def fixture_parameters_one():
    """Parameters with one uniform distribution."""
    random_variable = Uniform(lower_bound=-2, upper_bound=2)
    return Parameters(x1=random_variable)


@pytest.fixture(name="parameters_two")
def fixture_parameters_two():
    """Parameters with two uniform distributions."""
    random_variable = Uniform(lower_bound=-2, upper_bound=2)
    return Parameters(x1=random_variable, x2=deepcopy(random_variable))


@pytest.fixture(name="parameters_three")
def fixture_parameters_three():
    """Parameters with three uniform distributions."""
    random_variable = Uniform(lower_bound=-2, upper_bound=2)
    return Parameters(
        x1=random_variable, x2=deepcopy(random_variable), x3=deepcopy(random_variable)
    )


@pytest.fixture(name="expected_samples_one")
def fixture_expected_samples_one():
    """Expected samples for one parameter."""
    x1 = np.linspace(-2, 2, 5)
    return np.atleast_2d(x1).T


@pytest.fixture(name="expected_samples_two")
def fixture_expected_samples_two():
    """Expected samples for two parameters."""
    x1 = np.linspace(-2, 2, 5)
    x2 = np.linspace(-2, 2, 5)
    x1, x2 = np.meshgrid(x1, x2)
    samples = np.array([x1.flatten(), x2.flatten()]).T
    return samples


@pytest.fixture(name="expected_samples_three")
def fixture_expected_samples_three():
    """Expected samples for three parameters."""
    x1 = np.linspace(-2, 2, 5)
    x2 = np.linspace(-2, 2, 5)
    x3 = np.linspace(-2, 2, 5)
    x1, x2, x3 = np.meshgrid(x1, x2, x3)
    samples = np.array([x1.flatten(), x2.flatten(), x3.flatten()]).T
    return samples


# fixtures for some objects
@pytest.fixture(name="default_model")
def fixture_default_model():
    """A default simulation model."""
    model = SimulationModel(scheduler=Mock(), driver=Mock())
    return model


@pytest.fixture(name="default_grid_iterator")
def fixture_default_grid_iterator(
    global_settings, grid_dict_two, parameters_two, default_model, result_description
):
    """A default grid iterator."""
    # create iterator object
    my_grid_iterator = GridIterator(
        model=default_model,
        parameters=parameters_two,
        global_settings=global_settings,
        result_description=result_description,
        grid_design=grid_dict_two,
    )
    return my_grid_iterator


# -------------- actual unit_tests --------------------------------------------------
def test_init(
    mocker,
    global_settings,
    grid_dict_two,
    parameters_two,
    default_model,
    result_description,
):
    """Test the initialization of the GridIterator class."""
    # some default input for testing
    num_parameters = 2
    mp = mocker.patch("queens.iterators.iterator.Iterator.__init__")
    GridIterator.parameters = Mock()
    GridIterator.parameters.num_parameters = num_parameters

    my_grid_iterator = GridIterator(
        model=default_model,
        parameters=parameters_two,
        global_settings=global_settings,
        result_description=result_description,
        grid_design=grid_dict_two,
    )

    # tests / asserts
    mp.assert_called_once_with(default_model, parameters_two, global_settings)
    assert my_grid_iterator.grid_dict == grid_dict_two
    assert my_grid_iterator.result_description == result_description
    assert my_grid_iterator.samples is None
    assert my_grid_iterator.output is None
    assert not my_grid_iterator.num_grid_points_per_axis
    assert my_grid_iterator.num_parameters == num_parameters
    assert not my_grid_iterator.scale_type


def test_model_evaluate(default_grid_iterator, mocker):
    """Test the evaluate method of the SimulationModel class."""
    mp = mocker.patch("queens.models.simulation_model.SimulationModel.evaluate", return_value=None)
    default_grid_iterator.model.evaluate(None)
    mp.assert_called_once()


def test_pre_run_one(
    grid_dict_one,
    parameters_one,
    expected_samples_one,
    result_description,
    default_model,
    global_settings,
):
    """Test the pre_run method for a single parameter."""
    grid_iterator = GridIterator(
        model=default_model,
        parameters=parameters_one,
        global_settings=global_settings,
        result_description=result_description,
        grid_design=grid_dict_one,
    )
    grid_iterator.pre_run()
    np.testing.assert_array_equal(grid_iterator.samples, expected_samples_one)


def test_pre_run_two(
    grid_dict_two,
    parameters_two,
    expected_samples_two,
    default_model,
    global_settings,
):
    """Test the pre_run method for two parameters."""
    grid_iterator = GridIterator(
        model=default_model,
        parameters=parameters_two,
        global_settings=global_settings,
        result_description={},
        grid_design=grid_dict_two,
    )
    grid_iterator.pre_run()
    np.testing.assert_array_equal(grid_iterator.samples, expected_samples_two)


def test_pre_run_three(
    grid_dict_three,
    parameters_three,
    expected_samples_three,
    result_description,
    default_model,
    global_settings,
):
    """Test the pre_run method for three parameters."""
    grid_iterator = GridIterator(
        model=default_model,
        parameters=parameters_three,
        global_settings=global_settings,
        result_description=result_description,
        grid_design=grid_dict_three,
    )
    grid_iterator.pre_run()
    np.testing.assert_array_equal(grid_iterator.samples, expected_samples_three)


def test_core_run(mocker, default_grid_iterator, expected_samples_two):
    """Test the core_run method of the GridIterator class."""
    mocker.patch("queens.models.simulation_model.SimulationModel.evaluate", return_value=2)
    default_grid_iterator.samples = expected_samples_two
    default_grid_iterator.core_run()
    np.testing.assert_array_equal(default_grid_iterator.samples, expected_samples_two)
    assert default_grid_iterator.output == 2


def test_post_run(mocker, default_grid_iterator):
    """Test the post_run method of the GridIterator class."""
    # test if save results is called

    visualization = Mock()
    mp2 = mocker.patch.object(visualization, "plot_qoi_grid", return_value=1)
    mp1 = mocker.patch("queens.iterators.grid_iterator.write_results", return_value=None)
    default_grid_iterator.visualization = visualization

    default_grid_iterator.post_run()
    mp1.assert_called_once()
    mp2.assert_called_once()
