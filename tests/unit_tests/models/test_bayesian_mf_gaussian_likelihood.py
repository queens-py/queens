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
"""Unit tests for Bayesian multi-fidelity Gaussian likelihood function."""

# pylint: disable=invalid-name
from unittest import mock

import numpy as np
import pytest
from mock import Mock, patch

from queens.models.likelihood_models.bayesian_mf_gaussian_likelihood import (
    BMFGaussianModel,
    BmfiaInterface,
)
from queens.models.simulation_model import SimulationModel


# ------------ fixtures and params ---------------
@pytest.fixture(name="default_interface")
def fixture_default_interface():
    """Dummy BMFIA interface for testing."""
    num_processors_multi_processing = 2
    coord_labels = ["x1", "x2"]
    time_vec = None
    coords_mat = np.array([[1, 0], [1, 0]])

    interface = BmfiaInterface(
        num_processors_multi_processing=num_processors_multi_processing,
        probabilistic_mapping_type="per_coordinate",
    )
    interface.time_vec = time_vec
    interface.coords_mat = coords_mat
    interface.coord_labels = coord_labels
    return interface


@pytest.fixture(name="default_bmfia_iterator")
def fixture_default_bmfia_iterator(get_patched_bmfia_iterator):
    """Dummy iterator for testing."""
    parameters = "dummy_parameters"
    hf_model = "dummy_hf_model"
    lf_model = Mock()

    iterator = get_patched_bmfia_iterator(parameters, hf_model, lf_model)

    return iterator


@pytest.fixture(name="default_mf_likelihood")
def fixture_default_mf_likelihood(
    mocker,
    dummy_simulation_model,
    default_interface,
    default_bmfia_iterator,
):
    """Default multi-fidelity Gaussian likelihood object."""
    forward_model = dummy_simulation_model
    coords_mat = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 2, 3, 4])
    y_obs = np.array([[1, 2], [3, 4]])
    output_label = ["a", "b"]
    coord_labels = ["c", "d"]
    mf_interface = default_interface
    bmfia_subiterator = default_bmfia_iterator
    noise_var = np.array([0.1])
    num_refinement_samples = 0
    likelihood_evals_for_refinement_lst = []
    dummy_normal_distr = "dummy"

    experimental_data_reader = Mock()
    experimental_data_reader.get_experimental_data = lambda: (
        y_obs,
        coords_mat,
        time_vec,
        None,
        None,
        coord_labels,
        output_label,
    )
    mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel"
        ".build_approximation"
    )
    mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood.MeanFieldNormal",
        return_value=dummy_normal_distr,
    )
    mf_likelihood = BMFGaussianModel(
        forward_model=forward_model,
        mf_interface=mf_interface,
        mf_subiterator=bmfia_subiterator,
        noise_value=noise_var,
        num_refinement_samples=num_refinement_samples,
        likelihood_evals_for_refinement=likelihood_evals_for_refinement_lst,
        experimental_data_reader=experimental_data_reader,
        mf_approx=Mock(),
    )

    return mf_likelihood


class InstanceMock:
    """Mock class."""

    def __init__(self, *_args):
        """Initialize dummy class."""
        self.counter = 0

    def plot(self, *_args, **_kwargs):
        """Mock plot function."""
        self.counter += 1
        return 1


@pytest.fixture(name="mock_visualization")
def fixture_mock_visualization():
    """Mock visualization."""
    my_mock = InstanceMock()
    return my_mock


class InstanceMockModel:
    """Mock model."""

    def __init__(self):
        """Init method for mock model."""
        self.variables = 1

    def evaluate(self, *_args, **_kwargs):
        """Mock evaluate method."""
        return {"result": 1}


@pytest.fixture(name="mock_model")
def fixture_mock_model():
    """Mock model fixture."""
    my_mock = InstanceMockModel()
    return my_mock


# ------------ unit_tests -------------------------
def test_init(mocker, dummy_simulation_model, default_interface, default_bmfia_iterator):
    """Test the init of the multi-fidelity Gaussian likelihood function."""
    forward_model = dummy_simulation_model
    coords_mat = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 2, 3, 4])
    y_obs = np.array([[1], [3]])
    output_label = ["a", "b"]
    coord_labels = ["c", "d"]
    mf_interface = default_interface
    bmfia_subiterator = default_bmfia_iterator
    noise_var = 1.0
    mean_field_normal = "dummy_mean_field"
    num_refinement_samples = 0
    likelihood_evals_for_refinement_lst = []

    experimental_data_reader = Mock()
    experimental_data_reader.get_experimental_data = lambda: (
        y_obs,
        coords_mat,
        time_vec,
        None,
        None,
        coord_labels,
        output_label,
    )
    mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BMFGaussianModel"
        ".build_approximation"
    )
    mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood.MeanFieldNormal",
        return_value=mean_field_normal,
    )
    model = BMFGaussianModel(
        forward_model=forward_model,
        mf_interface=mf_interface,
        mf_subiterator=bmfia_subiterator,
        noise_value=noise_var,
        num_refinement_samples=num_refinement_samples,
        likelihood_evals_for_refinement=likelihood_evals_for_refinement_lst,
        experimental_data_reader=experimental_data_reader,
        mf_approx=Mock(),
    )

    # tests / asserts ----------------------------------
    assert model.forward_model == forward_model
    np.testing.assert_array_equal(model.coords_mat, coords_mat)
    np.testing.assert_array_equal(model.time_vec, time_vec)
    np.testing.assert_array_equal(model.y_obs, y_obs)
    assert model.output_label == output_label
    assert model.coord_labels == coord_labels

    assert model.mf_interface == mf_interface
    assert model.mf_subiterator == bmfia_subiterator
    assert model.min_log_lik_mf is None
    assert model.normal_distribution == mean_field_normal
    assert model.noise_var == noise_var
    assert model.likelihood_counter == 1
    assert model.num_refinement_samples == num_refinement_samples


def test_evaluate(default_mf_likelihood, mocker):
    """Compare return value with the expected value using a single point."""
    likelihood_output = np.array(np.array([7, 7]))
    y_lf_mat = np.array([[1, 2]])
    samples = np.array([[2, 3]])
    # on purpose transpose y_lf_mat here to check if this is wrong orientation is corrected
    mp1 = mocker.patch(
        "queens.models.simulation_model.SimulationModel.evaluate",
        return_value={"result": y_lf_mat.T},
    )

    mp2 = mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood."
        "BMFGaussianModel.evaluate_from_output",
        return_value=likelihood_output,
    )

    mf_log_likelihood = default_mf_likelihood.evaluate(samples)["result"]

    # assert statements
    mp1.assert_called_once()
    mp2.assert_called_once()
    np.testing.assert_array_equal(samples, mp1.call_args[0][0])
    np.testing.assert_array_equal(samples, mp2.call_args[0][0])
    np.testing.assert_array_equal(y_lf_mat, mp2.call_args[0][1])
    np.testing.assert_array_equal(mf_log_likelihood, likelihood_output)


def test_evaluate_from_output(default_mf_likelihood, mocker):
    """Compare return value with the expected value using a single point."""
    samples = np.array([[1, 2], [3, 4]])
    forward_model_output = np.array([[5], [6]])
    mf_log_likelihood_exp = np.array([[7], [9]])
    mp1 = mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood."
        "BMFGaussianModel.evaluate_mf_likelihood",
        return_value=mf_log_likelihood_exp,
    )

    y_lf_mat = np.array([[1, 2]])

    # test without adaptivity
    mf_log_likelihood = default_mf_likelihood.evaluate_from_output(samples, forward_model_output)
    mp1.assert_called_once()
    mp1.assert_called_with(samples, forward_model_output)
    np.testing.assert_array_equal(mf_log_likelihood, mf_log_likelihood_exp)
    assert default_mf_likelihood.likelihood_counter == 2

    # test with adaptivity
    mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood."
        "BMFGaussianModel._adaptivity_trigger",
        return_value=True,
    )
    mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood."
        "BMFGaussianModel._refine_mf_likelihood"
    )
    with pytest.raises(NotImplementedError):
        default_mf_likelihood.evaluate_from_output(mf_log_likelihood_exp, y_lf_mat)


def test_evaluate_mf_likelihood(default_mf_likelihood, mocker):
    """Test the evaluation of the log multi-fidelity Gaussian likelihood."""
    # --- define some vectors and matrices -----
    y_lf_mat = np.array(
        [[1, 1, 1], [2, 2, 2]]
    )  # three dim output per point x in x_batch (row-wise)
    x_batch = np.array([[0, 0], [0, 1]])  # make points have distance 1
    diff_mat = np.array([[1, 1, 1], [2, 2, 2]])  # for simplicity we assume diff_mat equals
    var_y_mat = np.array([[1, 1, 1], [1, 1, 1]])
    z_mat = y_lf_mat
    m_f_mat = np.array([[1, 1], [1, 1]])

    # mock the normal distribution
    distribution_mock = mock.MagicMock()
    distribution_mock.update_variance.return_value = None
    distribution_mock.logpdf.return_value = np.array([[1.0]])
    default_mf_likelihood.normal_distribution = distribution_mock

    mp1 = mocker.patch(
        "queens.iterators.bmfia.BMFIA.set_feature_strategy",
        return_value=(z_mat),
    )
    mp2 = mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BmfiaInterface.evaluate",
        return_value=(m_f_mat, var_y_mat),
    )

    log_lik_mf = default_mf_likelihood.evaluate_mf_likelihood(x_batch, y_lf_mat)

    # ------ assert and test statements ------------------------------------
    mp1.assert_called_once()
    np.testing.assert_array_equal(y_lf_mat, mp1.call_args[0][0])
    np.testing.assert_array_equal(x_batch, mp1.call_args[0][1])
    np.testing.assert_array_equal(
        default_mf_likelihood.coords_mat[: y_lf_mat.shape[0]], mp1.call_args[0][2]
    )

    # test evaluate method
    mp2.assert_called_once()
    np.testing.assert_array_equal(z_mat, mp2.call_args[0][0])

    # test covariance update and logpdf count
    assert distribution_mock.update_variance.call_count == diff_mat.shape[0]
    assert distribution_mock.logpdf.call_count == diff_mat.shape[0]

    # test logpdf output and input
    np.testing.assert_array_equal(log_lik_mf, np.array([1, 1]))


def test_grad(default_mf_likelihood):
    """Test grad method."""
    # define inputs
    np.random.seed(42)
    samples = np.random.rand(3, 2)
    forward_model_output = np.random.rand(3, 4)
    upstream_gradient = np.random.rand(3, 1)
    partial_grad = np.random.rand(3, 4)
    like_grad = np.random.rand(3, 2)
    default_mf_likelihood.response = {"forward_model_output": forward_model_output}

    with patch.object(BMFGaussianModel, "partial_grad_evaluate", return_value=partial_grad) as mp1:
        with patch.object(SimulationModel, "grad", return_value=like_grad) as mp2:
            grad_out = default_mf_likelihood.grad(samples, upstream_gradient)
            mp1.assert_called_once_with(samples, forward_model_output)
            mp2.assert_called_once()
            np.testing.assert_array_equal(mp2.call_args.args[0], samples)
            np.testing.assert_array_equal(mp2.call_args.args[1], upstream_gradient * partial_grad)
            np.testing.assert_array_equal(grad_out, like_grad)


def test_partial_grad_evaluate(mocker, default_mf_likelihood):
    """Test the partial grad evaluate method."""
    # --- define some vectors and matrices -----
    forward_model_output = np.array(
        [[1, 1, 1], [2, 2, 2]]
    )  # three dim output per point x in x_batch (row-wise)
    forward_model_input = np.array([[0, 0], [0, 1]])  # make points have distance 1
    diff_mat = np.array([[1, 1, 1], [2, 2, 2]])  # for simplicity we assume diff_mat equals
    var_y_mat = np.array([[1, 1, 1], [1, 1, 1]])
    z_mat = forward_model_output
    m_f_mat = np.array([[1, 1], [1, 1]])
    grad_m_f_mat = np.array([[6, 7, 8], [9, 10, 11]])
    grad_var_y_mat = np.array([[12, 13, 14], [15, 16, 17]])

    # create mock attribute for mf_likelihood
    distribution_mock = mock.MagicMock()
    distribution_mock.update_variance.return_value = None
    distribution_mock.logpdf.return_value = np.array([[1]])
    default_mf_likelihood.normal_distribution = distribution_mock

    mp1 = mocker.patch(
        "queens.iterators.bmfia.BMFIA.set_feature_strategy",
        return_value=z_mat,
    )
    mp2 = mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BmfiaInterface"
        ".evaluate_and_gradient",
        return_value=(m_f_mat, var_y_mat, grad_m_f_mat, grad_var_y_mat),
    )

    mocker.patch(
        "queens.distributions.mean_field_normal.MeanFieldNormal.logpdf",
        return_value=0.1,
    )
    mp3 = mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood."
        "BMFGaussianModel.grad_log_pdf_d_ylf",
        return_value=np.array([[0.2]]),
    )
    grad_out = default_mf_likelihood.partial_grad_evaluate(
        forward_model_input, forward_model_output
    )

    # ------ assert and test statements ------------------------------------
    mp1.assert_called_once()
    np.testing.assert_array_equal(forward_model_output, mp1.call_args[0][0])
    np.testing.assert_array_equal(forward_model_input, mp1.call_args[0][1])
    np.testing.assert_array_equal(
        default_mf_likelihood.coords_mat[: forward_model_output.shape[0]], mp1.call_args[0][2]
    )

    # test evaluate method
    mp2.assert_called_once()
    np.testing.assert_array_equal(z_mat, mp2.call_args[0][0])

    # test covariance update and logpdf count
    assert distribution_mock.update_variance.call_count == diff_mat.shape[0]
    assert distribution_mock.logpdf.call_count == diff_mat.shape[0]

    # test grad logpdf features
    assert mp3.call_count == diff_mat.shape[0]

    # test logpdf output and input
    np.testing.assert_array_equal(grad_out, np.array([[0.2], [0.2]]))


def test_grad_log_pdf_d_ylf(default_mf_likelihood):
    """Test grad log pdf d ylf method."""
    m_f_vec = np.array([[1, 4]])
    grad_m_f = np.array([[3, 5]])
    grad_var_y = np.array([[7, 10]])

    d_log_lik_d_mf = np.array([[1], [2]])  # two samples, output is scalar
    d_log_lik_d_var = np.array([[3], [4]])  # two samples, output is scalar

    # create mock attribute for mf_likelihood
    distribution_mock = mock.MagicMock()
    distribution_mock.grad_logpdf.return_value = d_log_lik_d_mf
    distribution_mock.grad_logpdf_var.return_value = d_log_lik_d_var
    default_mf_likelihood.normal_distribution = distribution_mock

    # run the method
    d_log_lik_d_z = default_mf_likelihood.grad_log_pdf_d_ylf(m_f_vec, grad_m_f, grad_var_y)

    # --- assert and test statements ----------------------------------------
    distribution_mock.grad_logpdf.assert_called_once()
    distribution_mock.grad_logpdf.assert_called_with(m_f_vec)

    distribution_mock.grad_logpdf_var.assert_called_once()
    distribution_mock.grad_logpdf_var.assert_called_with(m_f_vec)

    expected_d_log_lik_d_z = np.array([[24, 50]])
    np.testing.assert_array_equal(d_log_lik_d_z, expected_d_log_lik_d_z)


def test_initialize_bmfia_iterator(default_bmfia_iterator, mocker):
    """Test the initialization of the mf likelihood model."""
    coords_mat = np.array([[1, 2, 3], [2, 2, 2]])
    time_vec = np.linspace(1, 10, 3)
    y_obs = np.array([[5, 5, 5], [6, 6, 6]])

    mo_1 = mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood."
        "print_bmfia_acceleration",
        return_value=None,
    )

    BMFGaussianModel.initialize_bmfia_iterator(coords_mat, time_vec, y_obs, default_bmfia_iterator)

    # actual tests / asserts
    mo_1.assert_called_once()
    np.testing.assert_array_almost_equal(
        default_bmfia_iterator.coords_experimental_data, coords_mat, decimal=4
    )
    np.testing.assert_array_almost_equal(default_bmfia_iterator.time_vec, time_vec, decimal=4)
    np.testing.assert_array_almost_equal(default_bmfia_iterator.y_obs, y_obs, decimal=4)


def test_build_approximation(default_bmfia_iterator, default_interface, mocker):
    """Test for the build stage of the probabilistic regression model."""
    z_train = np.array([[1, 1, 1], [2, 2, 2]])
    y_hf_train = np.array([[1, 1], [2, 2]])
    coord_labels = ["x", "y", "z"]
    time_vec = default_bmfia_iterator.time_vec
    coords_mat = default_bmfia_iterator.coords_experimental_data
    approx = Mock()

    mo_1 = mocker.patch(
        "queens.iterators.bmfia.BMFIA.core_run",
        return_value=(z_train, y_hf_train),
    )
    mo_2 = mocker.patch(
        "queens.models.likelihood_models.bayesian_mf_gaussian_likelihood.BmfiaInterface"
        ".build_approximation",
        return_value=None,
    )

    BMFGaussianModel.build_approximation(
        default_bmfia_iterator,
        default_interface,
        approx,
        coord_labels,
        time_vec,
        coords_mat,
    )

    # actual asserts/tests
    mo_1.assert_called_once()
    mo_2.assert_called_once_with(z_train, y_hf_train, approx, coord_labels, time_vec, coords_mat)


def test_evaluate_forward_model(default_mf_likelihood, mock_model):
    """Test if forward model (lf model) is updated and evaluated correctly."""
    y_mat_expected = 1
    default_mf_likelihood.forward_model = mock_model
    y_mat = default_mf_likelihood.forward_model.evaluate(None)["result"]

    # actual tests / asserts
    np.testing.assert_array_almost_equal(y_mat, y_mat_expected, decimal=4)
