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
"""Collection of jitted kernel objects for a GP."""

import warnings

import numpy as np
from numba import jit, njit, prange
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from numpy.linalg.linalg import cholesky

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


# --- squared exponential covariance function -------------------
@njit(parallel=True)
def squared_exponential(x_train_mat, hyper_param_lst):
    """Jit the kernel for squared exponential covariance function.

    Also compute/pre-compile necessary
    derivatives for finding the MAP estimate of the GP. The covariance
    function here is the squared exponential covariance function.

    Args:
        x_train_mat (np.array): Training input points for the GP. Row-wise samples are stored,
                                different columns correspond to different input dimensions.
        hyper_param_lst (lst): List with the hyper-parameters for the kernel

    Returns:
        k_mat (np.array): Assembled covariance matrix of the GP
        k_mat_inv (np.array): Inverse of the assembled covariance matrix
        cholesky_k_mat (np.array): Lower cholesky decomposition of the covariance matrix
        partial_derivatives_hyper_params_lst (lst): List with partial derivatives of the
                                                    evidence w.r.t. the hyper-parameters
    """
    sigma_0_sq, l_scale_sq, sigma_n_sq = hyper_param_lst
    k_mat = np.zeros((x_train_mat.shape[0], x_train_mat.shape[0]), dtype=np.float64)
    partial_l_scale_sq = np.zeros((x_train_mat.shape[0], x_train_mat.shape[0]), dtype=np.float64)
    partial_sigma_0_sq = np.zeros((x_train_mat.shape[0], x_train_mat.shape[0]), dtype=np.float64)

    # pylint: disable=not-an-iterable
    for i in prange(x_train_mat.shape[0]):
        for j in prange(x_train_mat.shape[0]):
            # pylint: enable=not-an-iterable
            if i == j:
                noise_var = sigma_n_sq
            else:
                noise_var = 0.0

            x = x_train_mat[i, :]
            y = x_train_mat[j, :]

            delta = np.linalg.norm(x - y)
            k_mat[i, j] = sigma_0_sq * np.exp(-(delta**2) / (2.0 * l_scale_sq)) + noise_var
            partial_l_scale_sq[i, j] = (
                sigma_0_sq
                * np.exp(-(delta**2) / (2.0 * l_scale_sq))
                * delta**2
                / (2.0 * l_scale_sq**2)
            )
            partial_sigma_0_sq[i, j] = np.exp(-(delta**2) / (2.0 * l_scale_sq))

    # calculate first the cholesky decomposition
    cholesky_k_mat = cholesky(k_mat)

    partial_sigma_n = np.eye(k_mat.shape[0])

    # partial_sigma_0_sq (np.array): Derivative of the covariance function w.r.t. sigma_0**2
    # partial_l_scale_sq (np.array): Derivative of the covariance function w.r.t. l_scale**2
    # partial_sigma_n_sq (np.array): Derivative of the covariance function w.r.t. sigma_n**2
    partial_derivatives_hyper_params_lst = [
        partial_sigma_0_sq,
        partial_l_scale_sq,
        partial_sigma_n,
    ]

    return (k_mat, cholesky_k_mat, partial_derivatives_hyper_params_lst)


@jit(nopython=True)
def posterior_mean_squared_exponential(
    k_mat_inv,
    x_test_mat,
    x_train_mat,
    y_train_vec,
    hyper_param_lst,
):
    """Jit the posterior mean function of the Gaussian Process.

    The mean function is based on the squared exponential covariance function.

    Args:
        k_mat_inv (np.array): Inverse of the assembled covariance matrix
        x_test_mat (np.array): Testing input points for the GP. Individual samples row-wise,
                    columns correspond to different dimensions.
        x_train_mat (np.array): Training input points for the GP. Individual samples row-wise,
                                columns correspond to different dimensions.
        y_train_vec (np.array): Training outputs for the GP. Column vector where rows correspond
                                to different samples.
        hyper_param_lst (lst): List with the hyper-parameters of the kernel

    Returns:
        mu_vec (np.array): Posterior mean vector of the Gaussian Process evaluated at x_test_vec
    """
    sigma_0_sq, l_scale_sq, _ = hyper_param_lst

    k_vec = np.zeros((x_train_mat.shape[0], x_test_mat.shape[0]), dtype=np.float64)
    for j, x_test in enumerate(x_test_mat):
        for i, x_train in enumerate(x_train_mat):
            delta = np.linalg.norm(x_test - x_train)
            k_vec[i, j] = sigma_0_sq * np.exp(-(delta**2) / (2 * l_scale_sq))

    mu_vec = np.dot(np.dot(k_vec.T, k_mat_inv), (y_train_vec))

    return mu_vec


@jit(nopython=True)
def grad_posterior_mean_squared_exponential(
    k_mat_inv,
    x_test_mat,
    x_train_mat,
    y_train_vec,
    hyper_param_lst,
):
    """Jit the gradient of the posterior mean function of the GP.

    The mean function is based on the squared exponential covariance function.

    Args:
        k_mat_inv (np.array): Inverse of the assembled covariance matrix
        x_test_mat (np.array): Testing input points for the GP. Individual samples row-wise,
                    columns correspond to different dimensions.
        x_train_mat (np.array): Training input points for the GP. Individual samples row-wise,
                                columns correspond to different dimensions.
        y_train_vec (np.array): Training outputs for the GP. Column vector where rows correspond
                                to different samples.
        hyper_param_lst (lst): List with the hyper-parameters of the kernel

    Returns:
        grad_mu_mat (np.array): Gradient of the posterior mean vector (along columns) of the
                                Gaussian Process evaluated at each x_test_vec
                                (row-wise)
    """
    sigma_0_sq, l_scale_sq, _ = hyper_param_lst

    grad_k_vec = np.zeros((x_train_mat.shape[0], x_train_mat.shape[1]), dtype=np.float64)
    grad_mu_mat = np.zeros(x_test_mat.shape, dtype=np.float64)

    for j, x_test in enumerate(x_test_mat):
        for i, x_train in enumerate(x_train_mat):
            delta = np.linalg.norm(x_test - x_train)
            grad_k_vec[i] = (
                sigma_0_sq
                * np.exp(-(delta**2) / (2 * l_scale_sq))
                * (-(x_test - x_train))
                / (l_scale_sq)
            )
        grad_mu_mat[j] = np.dot(np.dot(grad_k_vec.T, k_mat_inv), (y_train_vec)).flatten()
    return grad_mu_mat


@jit(nopython=True)
def posterior_var_squared_exponential(
    k_mat_inv,
    x_test_mat,
    x_train_mat,
    hyper_param_lst,
    support,
):
    """Jit the posterior variance function of the Gaussian Process.

    The posterior is based on the squared exponential covariance function.

    Args:
        k_mat_inv (np.array): Inverse of the assembled covariance matrix
        x_test_mat (np.array): Testing input points for the GP. Individual samples row-wise,
                    columns correspond to different dimensions.
        x_train_mat (np.array): Training input points for the GP. Individual samples row-wise,
                                columns correspond to different dimensions.
        hyper_param_lst (lst): List with the hyper-parameters of the kernel
        support (str): Support type for the posterior distribution. For 'y' the posterior
                    is computed w.r.t. the output data; For 'f' the GP is computed w.r.t. the
                    latent function f.

    Returns:
        posterior_variance_vec (np.array): Posterior variance vector of the GP evaluated
                                           at the testing points x_test_vec
    """
    sigma_0_sq, l_scale_sq, sigma_n_sq = hyper_param_lst
    posterior_variance_vec = np.zeros((x_test_mat.shape[0], 1), dtype=np.float64)
    k_vec_test_train = np.zeros((x_train_mat.shape[0], 1), dtype=np.float64)
    for j, x_test in enumerate(x_test_mat):
        for i, x_train in enumerate(x_train_mat):
            delta = np.linalg.norm(x_test - x_train)
            k_vec_test_train[i, 0] = sigma_0_sq * np.exp(-(delta**2) / (2 * l_scale_sq))
            k_test = sigma_0_sq

        posterior_variance_vec[j] = k_test - np.dot(
            np.dot(k_vec_test_train.T, k_mat_inv), k_vec_test_train
        )
    if support == "y":
        posterior_variance_vec = posterior_variance_vec + sigma_n_sq

    return posterior_variance_vec


@jit(nopython=True)
def grad_posterior_var_squared_exponential(
    k_mat_inv,
    x_test_mat,
    x_train_mat,
    hyper_param_lst,
):
    """Jit the gradient of the posterior variance function.

    The posterior is based on the squared exponential covariance function.

    Args:
        k_mat_inv (np.array): Inverse of the assembled covariance matrix
        x_test_mat (np.array): Testing input points for the GP. Individual samples row-wise,
                    columns correspond to different dimensions.
        x_train_mat (np.array): Training input points for the GP. Individual samples row-wise,
                                columns correspond to different dimensions.
        hyper_param_lst (lst): List with the hyper-parameters of the kernel

    Returns:
        grad_posterior_variance (np.array): Gradient of the posterior variance
                                            of the GP evaluated at the testing points
                                            x_test_vec
    """
    sigma_0_sq, l_scale_sq, _ = hyper_param_lst
    grad_posterior_variance = np.zeros((x_test_mat.shape[0], x_test_mat.shape[1]), dtype=np.float64)
    grad_k_vec_test_train = np.zeros((x_train_mat.shape[0], x_test_mat.shape[1]), dtype=np.float64)
    k_vec_test_train = np.zeros((x_train_mat.shape[0], 1), dtype=np.float64)
    for j, x_test in enumerate(x_test_mat):
        for i, x_train in enumerate(x_train_mat):
            delta = np.linalg.norm(x_test - x_train)
            grad_k_vec_test_train[i, :] = (
                sigma_0_sq
                * np.exp(-(delta**2) / (2 * l_scale_sq))
                * (-(x_test - x_train))
                / (l_scale_sq)
            )
            k_vec_test_train[i, 0] = sigma_0_sq * np.exp(-(delta**2) / (2 * l_scale_sq))

        grad_posterior_variance[j] = np.dot(
            (-2 * np.dot(k_vec_test_train.T, k_mat_inv)), grad_k_vec_test_train
        )
    return grad_posterior_variance


@jit(nopython=True)
def grad_log_evidence_squared_exponential(
    param_vec, y_train_vec, x_train_vec, k_mat_inv, partial_derivatives_hyper_params_lst
):
    """Calculate gradient of log-evidence.

    Gradient of the log evidence function of the GP w.r.t. the
    variational hyper-parameters. The latter might be a transformed
    representation of the actual hyper-parameters.
    The evidence is based on the squared exponential covariance function.

    Args:
        param_vec (np.array): Vector containing values of hyper-parameters.
                                Note this is already used here in some of the other input
                                values are computed beforehand and stored as attributes.
        y_train_vec (np.array): Output training vector of the GP
        x_train_vec (np.array): Input training vector for the GP
        k_mat_inv (np.array): Current inverse of the GP covariance matrix
        partial_derivatives_hyper_params_lst (lst): Partial derivatives of the log evidence
                                                    w.r.t the hyper-parameters

    Returns:
        grad (np.array): Gradient vector of the evidence w.r.t. the parameterization
                            of the hyper-parameters
    """
    sigma_0_sq_param, l_scale_sq_param, sigma_n_sq_param = param_vec

    # partial_sigma_0_sq (np.array): Partial derivative of covariance matrix w.r.t. signal
    #                                variance variational parameter
    # partial_l_scale_sq (np.array): Partial derivative of covariance matrix w.r.t. length
    #                                squared scale variational parameter
    # partial_sigma_n_sq (np.array): Partial derivative of covariance matrix w.r.t. noise
    #                                variance variational parameter
    (
        partial_sigma_0_sq,
        partial_l_scale_sq,
        partial_sigma_n_sq,
    ) = partial_derivatives_hyper_params_lst

    data_minus_prior_mean = y_train_vec - x_train_vec
    alpha = np.dot(k_mat_inv, data_minus_prior_mean)

    grad_ev_sigma_0_sq_param = 0.5 * np.trace(
        (np.dot(alpha, alpha.T) - k_mat_inv) @ (partial_sigma_0_sq) * np.exp(sigma_0_sq_param)
    )
    grad_ev_sigma_n_sq_param = 0.5 * np.trace(
        (np.dot(alpha, alpha.T) - k_mat_inv) @ (partial_sigma_n_sq) * np.exp(sigma_n_sq_param)
    )
    grad_ev_l_scale_sq_param = 0.5 * np.trace(
        (np.dot(alpha, alpha.T) - k_mat_inv) @ (partial_l_scale_sq) * np.exp(l_scale_sq_param)
    )
    grad = np.array(
        [grad_ev_sigma_0_sq_param, grad_ev_l_scale_sq_param, grad_ev_sigma_n_sq_param]
    ).flatten()

    return grad


# -- Matern 3-2 covariance function ------------------------------------
@jit(nopython=True)
def matern_3_2(x_train_mat, hyper_param_lst):
    """Jit the kernel for the Matern 3/2 function.

    Also compute/pre-compile necessary
    derivatives for finding the MAP estimate of the GP. The covariance
    function here is the squared exponential covariance function.

    Args:
        x_train_mat (np.array): Training input points for the GP. Row-wise samples are stored,
                                different columns correspond to different input dimensions.
        hyper_param_lst (lst): List with the hyper-parameters for the kernel

    Returns:
        k_mat (np.array): Assembled covariance matrix of the GP
        k_mat_inv (np.array): Inverse of the assembled covariance matrix
        cholesky_k_mat (np.array): Lower cholesky decomposition of the covariance matrix
        partial_derivatives_hyper_params_lst (lst): List with partial derivatives of the
                                                    evidence w.r.t. the hyper-parameters
    """
    sigma_0_sq, l_scale, sigma_n_sq = hyper_param_lst
    k_mat = np.zeros((x_train_mat.shape[0], x_train_mat.shape[0]), dtype=np.float64)
    partial_l_scale = np.zeros((x_train_mat.shape[0], x_train_mat.shape[0]), dtype=np.float64)
    partial_sigma_0_sq = np.zeros((x_train_mat.shape[0], x_train_mat.shape[0]), dtype=np.float64)

    # pylint: disable=not-an-iterable
    for i in prange(x_train_mat.shape[0]):
        for j in prange(x_train_mat.shape[0]):
            # pylint: enable=not-an-iterable
            if i == j:
                noise_var = sigma_n_sq
            else:
                noise_var = 0.0

            x = x_train_mat[i, :]
            y = x_train_mat[j, :]

            delta = np.linalg.norm(x - y)
            k_mat[i, j] = (
                sigma_0_sq
                * (1 + np.sqrt(3) * delta / l_scale)
                * np.exp(-np.sqrt(3) * delta / l_scale)
                + noise_var
            )
            partial_l_scale[i, j] = (
                sigma_0_sq * np.exp(-np.sqrt(3) * delta / l_scale) * 3 * delta**2 / (l_scale**3)
            )
            partial_sigma_0_sq[i, j] = (1 + np.sqrt(3) * delta / l_scale) * np.exp(
                -np.sqrt(3) * delta / l_scale
            )

    # calculate first the cholesky decomposition
    cholesky_k_mat = cholesky(k_mat)

    partial_sigma_n = np.eye(k_mat.shape[0])

    # partial_sigma_0_sq (np.array): Derivative of the covariance function w.r.t. sigma_0**2
    # partial_l_scale_(np.array): Derivative of the covariance function w.r.t. l_scale
    # partial_sigma_n_sq (np.array): Derivative of the covariance function w.r.t. sigma_n**2
    partial_derivatives_hyper_params_lst = [
        partial_sigma_0_sq,
        partial_l_scale,
        partial_sigma_n,
    ]

    return (k_mat, cholesky_k_mat, partial_derivatives_hyper_params_lst)


@jit(nopython=True)
def posterior_mean_matern_3_2(
    k_mat_inv,
    x_test_mat,
    x_train_mat,
    y_train_vec,
    hyper_param_lst,
):
    """Jit the posterior mean function of the Gaussian Process.

    The mean function is based on the Matern 3/2 covariance function.

    Args:
        k_mat_inv (np.array): Inverse of the assembled covariance matrix
        x_test_mat (np.array): Testing input points for the GP. Individual samples row-wise,
                    columns correspond to different dimensions.
        x_train_mat (np.array): Training input points for the GP. Individual samples row-wise,
                                columns correspond to different dimensions.
        y_train_vec (np.array): Training outputs for the GP. Column vector where rows correspond
                                to different samples.
        hyper_param_lst (lst): List with the hyper-parameters of the kernel

    Returns:
        mu_vec (np.array): Posterior mean vector of the Gaussian Process evaluated at x_test_vec
    """
    sigma_0_sq, l_scale, _ = hyper_param_lst

    k_vec = np.zeros((x_train_mat.shape[0], x_test_mat.shape[0]), dtype=np.float64)
    for j, x_test in enumerate(x_test_mat):
        for i, x_train in enumerate(x_train_mat):
            delta = np.linalg.norm(x_test - x_train)
            k_vec[i, j] = (
                sigma_0_sq
                * (1 + np.sqrt(3) * delta / l_scale)
                * np.exp(-np.sqrt(3) * delta / l_scale)
            )

    mu_vec = np.dot(np.dot(k_vec.T, k_mat_inv), (y_train_vec))

    return mu_vec


@jit(nopython=True)
def posterior_var_matern_3_2(
    k_mat_inv,
    x_test_mat,
    x_train_mat,
    hyper_param_lst,
    support,
):
    """Jit the posterior variance function of the Gaussian Process.

    The posterior is based on the Matern 3/2 kernel.

    Args:
        k_mat_inv (np.array): Inverse of the assembled covariance matrix
        x_test_mat (np.array): Testing input points for the GP. Individual samples row-wise,
                    columns correspond to different dimensions.
        x_train_mat (np.array): Training input points for the GP. Individual samples row-wise,
                                columns correspond to different dimensions.
        hyper_param_lst (lst): List with the hyper-parameters of the kernel
        support (str): Support type for the posterior distribution. For 'y' the posterior
                    is computed w.r.t. the output data; For 'f' the GP is computed w.r.t. the
                    latent function f.

    Returns:
        posterior_variance_vec (np.array): Posterior variance vector of the GP evaluated
                                            at the testing points x_test_vec
    """
    sigma_0_sq, l_scale, sigma_n_sq = hyper_param_lst
    k_mat_test_train = np.zeros((x_train_mat.shape[0], x_test_mat.shape[0]), dtype=np.float64)
    for j, x_test in enumerate(x_test_mat):
        for i, x_train in enumerate(x_train_mat):
            delta = np.linalg.norm(x_test - x_train)
            k_mat_test_train[i, j] = (
                sigma_0_sq
                * (1 + np.sqrt(3) * delta / l_scale)
                * np.exp(-np.sqrt(3) * delta / l_scale)
            )

    k_mat_test = np.zeros((x_test_mat.shape[0], x_test_mat.shape[0]), dtype=np.float64)
    for j, x_test1 in enumerate(x_test_mat):
        for i, x_test2 in enumerate(x_test_mat):
            delta = np.linalg.norm(x_test1 - x_test2)
            k_mat_test[i, j] = (
                sigma_0_sq
                * (1 + np.sqrt(3) * delta / l_scale)
                * np.exp(-np.sqrt(3) * delta / l_scale)
            )

    posterior_variance_vec = np.diag(
        k_mat_test - np.dot(np.dot(k_mat_test_train.T, k_mat_inv), k_mat_test_train)
    )

    if support == "y":
        posterior_variance_vec = posterior_variance_vec + sigma_n_sq

    return posterior_variance_vec


def grad_posterior_mean_matern_3_2(*_args):
    """Calculate gradient of posterior mean for Matern 3/2 kernel."""
    raise NotImplementedError(
        "The gradient of the posterior mean is not implemented yet for the Matern 3/2 kernel."
    )


def grad_posterior_var_matern_3_2(*_args):
    """Calculate gradient of posterior variance for Matern 3/2 kernel."""
    raise NotImplementedError(
        "The gradient of the posterior variance is not implemented yet for the Matern 3/2 kernel."
    )


@jit(nopython=True)
def grad_log_evidence_matern_3_2(
    param_vec, y_train_vec, x_train_vec, k_mat_inv, partial_derivatives_hyper_params_lst
):
    """Calculate gradient of log-evidence for Matern 3/2 kernel.

    Gradient of the log evidence function of the GP w.r.t. the
    variational hyper-parameters. The latter might be a transformed
    representation of the actual hyper-parameters.

    Args:
        param_vec (np.array): Vector containing values of hyper-parameters.
                                Note this is already used here in some of the other input
                                values are computed beforehand and stored as attributes.
        y_train_vec (np.array): Output training vector of the GP
        x_train_vec (np.array): Input training vector for the GP
        k_mat_inv (np.array): Current inverse of the GP covariance matrix
        partial_derivatives_hyper_params_lst (lst): Partial derivatives of the log evidence
                                                    w.r.t the hyper-parameters

    Returns:
        grad (np.array): Gradient vector of the evidence w.r.t. the parameterization
                            of the hyper-parameters
    """
    sigma_0_sq_param, l_scale_param, sigma_n_sq_param = param_vec

    # partial_sigma_0_sq (np.array): Partial derivative of covariance matrix w.r.t. signal
    #                                variance variational parameter
    # partial_l_scale (np.array): Partial derivative of covariance matrix w.r.t. length
    #                             scale variational parameter
    # partial_sigma_n_sq (np.array): Partial derivative of covariance matrix w.r.t. noise
    #                                variance variational parameter
    (
        partial_sigma_0_sq,
        partial_l_scale,
        partial_sigma_n_sq,
    ) = partial_derivatives_hyper_params_lst

    data_minus_prior_mean = y_train_vec - x_train_vec
    alpha = np.dot(k_mat_inv, data_minus_prior_mean)

    grad_ev_sigma_0_sq_param = 0.5 * np.trace(
        (np.dot(alpha, alpha.T) - k_mat_inv) @ (partial_sigma_0_sq) * np.exp(sigma_0_sq_param)
    )
    grad_ev_sigma_n_sq_param = 0.5 * np.trace(
        (np.dot(alpha, alpha.T) - k_mat_inv) @ (partial_sigma_n_sq) * np.exp(sigma_n_sq_param)
    )
    grad_ev_l_scale_param = 0.5 * np.trace(
        (np.dot(alpha, alpha.T) - k_mat_inv) @ (partial_l_scale) * np.exp(l_scale_param)
    )
    grad = np.array(
        [grad_ev_sigma_0_sq_param, grad_ev_l_scale_param, grad_ev_sigma_n_sq_param]
    ).flatten()

    return grad
