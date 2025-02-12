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
"""Unit tests for the BMFMC iterator."""

# pylint: disable=invalid-name
import numpy as np
import pytest
from mock import Mock

from queens.iterators.bmfmc import BMFMC


@pytest.fixture(name="approx_name")
def fixture_approx_name():
    """Approximation name."""
    name = "gp_approximation_gpflow"
    return name


@pytest.fixture(name="config")
def fixture_config():
    """Configuration dictionary."""
    config = {
        "type": "gp_approximation_gpflow",
        "features_config": "opt_features",
        "num_features": 1,
        "X_cols": 1,
    }
    return config


@pytest.fixture(name="default_bmfmc_model")
def fixture_default_bmfmc_model():
    """A default BMFMC model."""
    np.random.seed(1)
    model = Mock()
    model.X_mc = np.random.random((20, 2))
    model.Y_LFs_mc = np.random.random((20, 2))
    return model


@pytest.fixture(name="experiment_dir")
def fixture_experiment_dir():
    """Experiment directory."""
    return "my_dummy_dir"


@pytest.fixture(name="initial_design")
def fixture_initial_design():
    """Initial design dictionary."""
    return {"num_HF_eval": 5, "num_bins": 5, "method": "diverse_subset"}


@pytest.fixture(name="default_bmfmc_iterator")
def fixture_default_bmfmc_iterator(
    global_settings,
    default_parameters_uniform_2d,
    default_bmfmc_model,
    result_description,
    initial_design,
):
    """A default BMFMC instance."""
    my_bmfmc_iterator = BMFMC(
        model=default_bmfmc_model,
        parameters=default_parameters_uniform_2d,
        global_settings=global_settings,
        result_description=result_description,
        initial_design=initial_design,
    )
    return my_bmfmc_iterator


# ------ actual unit_tests --------------------------------------------
def test_init(
    default_parameters_uniform_2d,
    default_bmfmc_model,
    result_description,
    initial_design,
    global_settings,
):
    """Test initialization of BMFMC."""
    my_bmfmc_iterator = BMFMC(
        model=default_bmfmc_model,
        parameters=default_parameters_uniform_2d,
        global_settings=global_settings,
        result_description=result_description,
        initial_design=initial_design,
    )
    # tests / asserts
    assert my_bmfmc_iterator.result_description == result_description
    assert my_bmfmc_iterator.X_train is None
    assert my_bmfmc_iterator.Y_LFs_train is None
    assert my_bmfmc_iterator.output is None
    assert my_bmfmc_iterator.initial_design == initial_design


def test_core_run(mocker, default_bmfmc_iterator, default_bmfmc_model):
    """Test core_run method of BMFMC."""
    mp1 = mocker.patch("queens.iterators.bmfmc.BMFMC.calculate_optimal_X_train")

    default_bmfmc_iterator.core_run()

    # -------- Load MC data from model -----------------------
    default_bmfmc_model.load_sampling_data.assert_called_once()

    # ---- determine optimal input points for which HF should be simulated -------
    mp1.assert_called_once()

    # ----- build model on training points and evaluate it -----------------------
    default_bmfmc_model.evaluate.assert_called_once()


def test_calculate_optimal_X_train(mocker, default_bmfmc_iterator):
    """Test calculate_optimal_X_train method of BMFMC."""
    mp1 = mocker.patch("queens.iterators.bmfmc.BMFMC.diverse_subset_design")
    default_bmfmc_iterator.X_train = np.array([1.0, 1.0, 1.0])
    default_bmfmc_iterator.Y_LFs_train = np.array([2.0, 2.0, 2.0])
    default_bmfmc_iterator.calculate_optimal_X_train()

    mp1.assert_called_once()
    np.testing.assert_array_equal(default_bmfmc_iterator.model.X_train, np.array([1.0, 1.0, 1.0]))
    np.testing.assert_array_equal(
        default_bmfmc_iterator.model.Y_LFs_train, np.array([2.0, 2.0, 2.0])
    )


def test_get_design_method(mocker, default_bmfmc_iterator):
    """Test get_design_method method of BMFMC."""
    mocker.patch("queens.iterators.bmfmc.BMFMC.random_design", return_value="random")
    mocker.patch(
        "queens.iterators.bmfmc.BMFMC.diverse_subset_design",
        return_value="diverse",
    )
    mocker.patch("queens.models.bmfmc_model.BMFMCModel.calculate_extended_gammas")

    method = default_bmfmc_iterator.get_design_method("random")
    assert method() == "random"

    method = default_bmfmc_iterator.get_design_method("diverse_subset")
    assert method() == "diverse"

    with pytest.raises(NotImplementedError):
        default_bmfmc_iterator.get_design_method("blabla")


def test_diverse_subset_design(mocker, default_bmfmc_iterator):
    """Test diverse_subset_design method of BMFMC."""
    mocker.patch("queens.models.bmfmc_model.BMFMCModel.calculate_extended_gammas")
    n_points = 3
    np.random.seed(1)
    default_bmfmc_iterator.model.gammas_ext_mc = np.random.random((10, 2))
    default_bmfmc_iterator.diverse_subset_design(n_points)
    X_train = default_bmfmc_iterator.X_train
    Y_LFs_train = default_bmfmc_iterator.Y_LFs_train

    expected_X_train = np.array(
        np.array([[0.20445225, 0.87811744], [0.4173048, 0.55868983], [0.14038694, 0.19810149]])
    )
    expected_Y_LFs_train = np.array(
        np.array([[0.21162812, 0.26554666], [0.57411761, 0.14672857], [0.58930554, 0.69975836]])
    )
    np.testing.assert_array_almost_equal(X_train, expected_X_train, decimal=6)
    np.testing.assert_array_almost_equal(Y_LFs_train, expected_Y_LFs_train, decimal=6)


def test_random_design(mocker, default_bmfmc_iterator):
    """Test random_design method of BMFMC."""
    mocker.patch("queens.models.bmfmc_model.BMFMCModel.calculate_extended_gammas")
    n_points = 3
    np.random.seed(1)
    default_bmfmc_iterator.model.gammas_ext_mc = np.random.random((10, 2))
    default_bmfmc_iterator.initial_design = {
        "num_HF_eval": 3,
        "num_bins": 3,
        "method": "random",
        "seed": 1,
        "master_LF": 1,
    }
    default_bmfmc_iterator.random_design(n_points)
    X_train = default_bmfmc_iterator.X_train
    Y_LFs_train = default_bmfmc_iterator.Y_LFs_train

    expected_X_train = np.array(
        np.array([[0.6918771, 0.3155156], [0.3134242, 0.6923226], [0.1403869, 0.1981015]])
    )
    expected_Y_LFs_train = np.array(
        np.array([[0.3976768, 0.1653542], [0.6944002, 0.4141793], [0.5893055, 0.6997584]])
    )
    np.testing.assert_array_almost_equal(X_train, expected_X_train, decimal=6)
    np.testing.assert_array_almost_equal(Y_LFs_train, expected_Y_LFs_train, decimal=6)


def test_model_evaluate(default_bmfmc_iterator):
    """Test model evaluate method of BMFMC."""
    default_bmfmc_iterator.model.evaluate(None)
    default_bmfmc_iterator.model.evaluate.assert_called_once()


def test_post_run(mocker, default_bmfmc_iterator):
    """Test post_run method of BMFMC."""
    visualization = Mock()
    mp1 = mocker.patch.object(visualization, "plot_pdfs")
    mp2 = mocker.patch.object(visualization, "plot_manifold")
    mp3 = mocker.patch("queens.iterators.bmfmc.write_results")
    default_bmfmc_iterator.visualization = visualization

    default_bmfmc_iterator.post_run()

    mp1.assert_called_once()
    mp2.assert_called_once()
    mp3.assert_called_once()
