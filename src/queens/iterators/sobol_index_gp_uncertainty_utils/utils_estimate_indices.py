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
"""Utils for Sobol index estimation.

All functions below are independent functions so that they can be used for parallel computations
with multiprocessing.

Important: Do not use XArrays in parallel processes as they are very slow!
"""

import numpy as np


def bootstrap(prediction, bootstrap_indices):
    """Bootstrap samples.

    Args:
        prediction (ndarray): realizations of Gaussian process
        bootstrap_indices (ndarray): bootstrap indices

    Returns:
        current_bootstrap_sample (ndarray): current bootstrap sample
    """
    current_bootstrap_sample = prediction[bootstrap_indices]
    return current_bootstrap_sample


def calculate_indices_first_total_order(
    prediction, bootstrap_idx, input_dim, number_bootstrap_samples, first_order_estimator
):
    """Estimate first and total-order Sobol indices.

    Args:
        prediction (ndarray): realizations of Gaussian process
        bootstrap_idx (ndarray): bootstrap indices
        input_dim (int): input-parameter index
        number_bootstrap_samples (int): number of bootstrap samples
        first_order_estimator (str): estimator for first-order indices

    Returns:
        estimates_first_order (ndarray): estimates of first-order Sobol index
        estimates_total_order (ndarray): estimates of total-order Sobol index
    """
    estimates_first_order = np.empty(number_bootstrap_samples)
    estimates_total_order = np.empty(number_bootstrap_samples)

    bootstrap_samples = bootstrap(prediction, bootstrap_idx)
    for idx_bootstrap in np.arange(number_bootstrap_samples):
        current_bootstrap_sample = bootstrap_samples[idx_bootstrap, :]
        sample_matrix_a, sample_matrix_b, sample_matrix_ab = _extract_sample_matrices(
            current_bootstrap_sample, input_dim
        )
        estimates_first_order[idx_bootstrap] = _estimate_first_order_index(
            sample_matrix_a, sample_matrix_b, sample_matrix_ab, first_order_estimator
        )
        estimates_total_order[idx_bootstrap] = _estimate_total_order_index(
            sample_matrix_a, sample_matrix_b, sample_matrix_ab
        )

    return estimates_first_order, estimates_total_order


def calculate_indices_second_order_gp_realizations(
    prediction,
    bootstrap_indices,
    input_dim_i,
    number_bootstrap_samples,
    number_parameters,
    first_order_estimator,
):
    """Estimate first, second and total-order Sobol indices.

    Based on Gaussian process realizations and parallelized over those realizations.

    Args:
        prediction (ndarray): realizations of Gaussian process
        bootstrap_indices (ndarray): bootstrap indices
        input_dim_i (int): input-parameter index
        number_bootstrap_samples (int): number of bootstrap samples
        number_parameters (int): number of input-space dimensions
        first_order_estimator (str): estimator for first-order indices

    Returns:
        estimates_first_order (ndarray): estimates of first-order Sobol index
        estimates_second_order (ndarray): estimates of second-order Sobol index
        estimates_total_order (ndarray): estimates of total-order Sobol index
    """
    estimates_first_order, estimates_total_order = calculate_indices_first_total_order(
        prediction, bootstrap_indices, input_dim_i, number_bootstrap_samples, first_order_estimator
    )

    estimates_second_order = np.empty(
        (number_bootstrap_samples, number_parameters - (input_dim_i + 1))
    )
    bootstrap_samples = bootstrap(prediction, bootstrap_indices)
    for b in np.arange(number_bootstrap_samples):
        current_bootstrap_sample = bootstrap_samples[b, :]
        idx_loop = 0
        for input_dim_j in range(input_dim_i + 1, number_parameters):
            (
                sample_matrix_a,
                sample_matrix_b,
                sample_matrix_ab_i,
                sample_matrix_ab_j,
                sample_matrix_ba_i,
            ) = _extract_sample_matrices_second_order(
                current_bootstrap_sample, input_dim_i, input_dim_j, number_parameters
            )
            estimates_second_order[b, idx_loop] = _estimate_second_order_index(
                sample_matrix_a,
                sample_matrix_ab_i,
                sample_matrix_ab_j,
                sample_matrix_ba_i,
                sample_matrix_b,
                first_order_estimator,
            )
            idx_loop += 1

    return estimates_first_order, estimates_total_order, estimates_second_order


def calculate_indices_second_order_gp_mean(
    prediction, bootstrap_indices, input_dim_i, number_parameters, first_order_estimator
):
    """Estimate first, second and total-order Sobol indices.

    Based on Gaussian process mean and parallelized over bootstrapping samples.

    Args:
        prediction (ndarray): prediction
        bootstrap_indices (ndarray): bootstrap indices
        input_dim_i (int): input-parameter index
        number_parameters (int): number of input-space dimensions
        first_order_estimator (str): estimator for first-order indices

    Returns:
        estimates_first_order (ndarray): estimates of first-order Sobol index
        estimates_second_order (ndarray): estimates of second-order Sobol index
        estimates_total_order (ndarray): estimates of total-order Sobol index
    """
    bootstrap_sample = bootstrap(prediction, bootstrap_indices)
    sample_matrix_a, sample_matrix_b, sample_matrix_ab = _extract_sample_matrices(
        bootstrap_sample, input_dim_i
    )
    estimates_first_order = _estimate_first_order_index(
        sample_matrix_a, sample_matrix_b, sample_matrix_ab, first_order_estimator
    )
    estimates_total_order = _estimate_total_order_index(
        sample_matrix_a, sample_matrix_b, sample_matrix_ab
    )

    estimates_second_order = np.empty(number_parameters - (input_dim_i + 1))
    idx_loop = 0
    for input_dim_j in range(input_dim_i + 1, number_parameters):
        (
            sample_matrix_a,
            sample_matrix_b,
            sample_matrix_ab_i,
            sample_matrix_ab_j,
            sample_matrix_ba_i,
        ) = _extract_sample_matrices_second_order(
            bootstrap_sample, input_dim_i, input_dim_j, number_parameters
        )
        estimates_second_order[idx_loop] = _estimate_second_order_index(
            sample_matrix_a,
            sample_matrix_ab_i,
            sample_matrix_ab_j,
            sample_matrix_ba_i,
            sample_matrix_b,
            first_order_estimator,
        )
        idx_loop += 1

    return estimates_first_order, estimates_total_order, estimates_second_order


def calculate_indices_third_order(
    prediction,
    bootstrap_indices,
    number_boostrap_samples,
    number_parameters,
    first_order_estimator,
):
    """Estimate third-order Sobol indices.

    Based on Gaussian process realizations and parallelized over those realizations.

    Args:
        prediction (ndarray): realizations of Gaussian process
        bootstrap_indices (ndarray): bootstrap indices
        number_boostrap_samples (int): number of bootstrap samples
        number_parameters (int): number of input-space dimensions
        first_order_estimator (str): estimator for first-order indices

    Returns:
        estimates_third_order (ndarray): estimates for third-order Sobol index
    """
    # 1. Estimate first-order Sobol indices
    estimates_first_order = np.empty((number_boostrap_samples, number_parameters))
    estimates_total_order = np.empty((number_boostrap_samples, number_parameters))
    for i in range(number_parameters):
        (
            estimates_first_order[:, i],
            estimates_total_order[:, i],
        ) = calculate_indices_first_total_order(
            prediction, bootstrap_indices, i, number_boostrap_samples, first_order_estimator
        )

    # 2. Estimate second-order Sobol indices
    estimates_second_order = np.empty((number_boostrap_samples, number_parameters))
    bootstrap_samples = bootstrap(prediction, bootstrap_indices)
    for b in np.arange(number_boostrap_samples):
        current_bootstrap_sample = bootstrap_samples[b, :]
        idx_loop = 0
        for idx_i, i in enumerate(range(number_parameters)):
            for j in range(idx_i + 1, number_parameters):
                (
                    sample_matrix_a,
                    sample_matrix_b,
                    sample_matrix_ab_i,
                    sample_matrix_ab_j,
                    sample_matrix_ba_i,
                ) = _extract_sample_matrices_second_order(
                    current_bootstrap_sample, i, j, number_parameters
                )
                estimates_second_order[b, idx_loop] = _estimate_second_order_index(
                    sample_matrix_a,
                    sample_matrix_ab_i,
                    sample_matrix_ab_j,
                    sample_matrix_ba_i,
                    sample_matrix_b,
                    first_order_estimator,
                )
                idx_loop += 1

    # 3. Estimate closed third-order Sobol indices (includes lower order indices)
    closed_estimates_third_order = np.empty((number_boostrap_samples, 1))
    for b in np.arange(number_boostrap_samples):
        current_bootstrap_sample = bootstrap_samples[b, :]
        (
            sample_matrix_a,
            sample_matrix_b,
            sample_matrix_ab_ijk,
        ) = _extract_sample_matrices_third_order(current_bootstrap_sample)
        closed_estimates_third_order[b, 0] = _estimate_closed_third_order_index(
            sample_matrix_b, sample_matrix_ab_ijk
        )

    # 4. Subtract lower order indices
    estimates_third_order = (
        closed_estimates_third_order.flatten()
        - estimates_second_order.sum(axis=1)
        - estimates_first_order.sum(axis=1)
    )

    return estimates_third_order


def _extract_sample_matrices(prediction, i):
    """Extract sampling matrices A, B, AB from Gaussian process prediction.

    Format:
        [AB_1, AB_2, ..., AB_D, BA_1, ..., BA_D, A, B]

    Args:
        prediction (ndarray): realizations of Gaussian process
        i (int): index to calculate

    Returns:
        sample_matrix_a (ndarray): Saltelli A sample matrix
        sample_matrix_b (ndarray): Saltelli B sample matrix
        sample_matrix_ab (ndarray): Saltelli AB sample matrix
    """
    sample_matrix_ab = prediction[:, i]
    sample_matrix_a = prediction[:, -2]
    sample_matrix_b = prediction[:, -1]

    return sample_matrix_a, sample_matrix_b, sample_matrix_ab


def _extract_sample_matrices_second_order(prediction, i, j, number_parameters):
    """Extract sampling matrices A, B, AB from Gaussian process prediction.

    Format:
        [AB_1, AB_2, ..., AB_D, BA_1, ..., BA_D, A, B]

    Args:
        prediction (ndarray): realizations of Gaussian process
        i (int): index to calculate
        j (int): index to calculate
        number_parameters (int): number of input space dimensions

    Returns:
        sample_matrix_a (ndarray): Saltelli A sample matrix
        sample_matrix_b (ndarray): Saltelli B sample matrix
        sample_matrix_ab_i (ndarray): Saltelli ABi sample matrix
        sample_matrix_ab_j (ndarray): Saltelli ABj sample matrix
        sample_matrix_ba_i (ndarray): Saltelli BAi sample matrix
    """
    sample_matrix_ab_i = prediction[:, i]
    sample_matrix_ab_j = prediction[:, j]
    sample_matrix_ba_i = prediction[:, number_parameters + i]
    sample_matrix_a = prediction[:, -2]
    sample_matrix_b = prediction[:, -1]

    return (
        sample_matrix_a,
        sample_matrix_b,
        sample_matrix_ab_i,
        sample_matrix_ab_j,
        sample_matrix_ba_i,
    )


def _extract_sample_matrices_third_order(prediction):
    """Extract sampling matrices A, B, AB from Gaussian process prediction.

    Format:
        [AB_1, AB_2, ..., AB_D, BA_1, ..., BA_D, AB_123, A, B]

    Args:
        prediction (ndarray): realizations of Gaussian process

    Returns:
        sample_matrix_a (ndarray): Saltelli A sample matrix
        sample_matrix_b (ndarray): Saltelli B sample matrix
        sample_matrix_ab_ijk (ndarray): Saltelli AB_ijk sample matrix
    """
    sample_matrix_ab_ijk = prediction[:, -3]
    sample_matrix_a = prediction[:, -2]
    sample_matrix_b = prediction[:, -1]

    return sample_matrix_a, sample_matrix_b, sample_matrix_ab_ijk


def _estimate_first_order_index(sample_matrix_a, sample_matrix_b, sample_matrix_ab, estimator):
    """Compute first-order Sobol indices.

    References for estimators:

    [Janon2014] Janon, Alexandre, Thierry Klein, Agnès Lagnoux, Maëlle Nodet, and Clémentine
    Prieur. ‘Asymptotic Normality and Efficiency of Two Sobol Index Estimators’. ESAIM: Probability
    and Statistics 18 (2014): 342–64. https://doi.org/10.1051/ps/2013040.

    [Gratiet2014] Le Gratiet, Loic, Claire Cannamela, and Bertrand Iooss. ‘A Bayesian Approach
    for Global Sensitivity Analysis of (Multifidelity) Computer Codes’. SIAM/ASA Journal on
    Uncertainty Quantification 2, no. 1 (1 January 2014): 336–63. https://doi.org/10.1137/130926869.

    [Saltelli2010] Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design
    and Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
    (1 February 2010): 259–270. https://doi.org/10.1016/j.cpc.2009.09.018.

    Args:
        sample_matrix_a (ndarray): results corresponding to A sample matrix
        sample_matrix_b (ndarray): results corresponding B sample matrix
        sample_matrix_ab (ndarray): results corresponding to AB sample matrix
        estimator (str): estimator for first-order indices

    Returns:
        first_order (ndarray): first-order Sobol index estimates
    """
    if estimator == "Janon2014":
        # [Janon2014] Equation (2.5)
        first_order = (
            np.mean(sample_matrix_b * sample_matrix_ab)
            - (0.5 * np.mean(sample_matrix_b + sample_matrix_ab)) ** 2
        ) / (
            np.mean(sample_matrix_b * sample_matrix_b)
            - (0.5 * np.mean(sample_matrix_b + sample_matrix_ab)) ** 2
        )

    elif estimator == "Janon2014alt":
        # [Janon2014] Equation (2.8)
        first_order = np.sum(
            (sample_matrix_b - 0.5 * (sample_matrix_b.mean() + sample_matrix_ab.mean()))
            * (sample_matrix_ab - 0.5 * (sample_matrix_b.mean() + sample_matrix_ab.mean()))
        ) / np.sum(
            0.5 * (sample_matrix_b**2 + sample_matrix_ab**2)
            - (0.5 * (sample_matrix_b.mean() + sample_matrix_ab.mean())) ** 2
        )

    elif estimator == "Gratiet2014":
        # [Gratiet2014] Equation (4.1)
        first_order = (
            np.mean(sample_matrix_b * sample_matrix_ab)
            - sample_matrix_b.mean() * sample_matrix_ab.mean()
        ) / (np.mean(sample_matrix_b * sample_matrix_b) - sample_matrix_b.mean() ** 2)

    elif estimator == "Saltelli2010":
        # [Saltelli2010] also used in SALib library
        first_order = np.mean(
            sample_matrix_b * (sample_matrix_ab - sample_matrix_a), axis=0
        ) / np.var(np.r_[sample_matrix_a, sample_matrix_b], axis=0)

    else:
        raise ValueError(
            "Unknown first-order estimator. Valid estimators are Janon2014, "
            "Janon2014alt, Gratiet2014 or Saltelli2010."
        )

    return first_order


def _estimate_total_order_index(sample_matrix_a, sample_matrix_b, sample_matrix_ab):
    """Estimate total-order Sobol indices.

    Reference:
    [Saltelli2010] Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design
    and Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
    (1 February 2010): 259–270. https://doi.org/10.1016/j.cpc.2009.09.018.

    Args:
        sample_matrix_a (ndarray): results corresponding to A sample matrix
        sample_matrix_b (ndarray): results corresponding to B sample matrix
        sample_matrix_ab (ndarray): results corresponding to AB sample matrix

    Returns:
        total_order (ndarray): total-order Sobol index estimates
    """
    # from SALib library [Saltelli2010]
    total_order = (
        0.5
        * np.mean((sample_matrix_a - sample_matrix_ab) ** 2, axis=0)
        / np.var(np.r_[sample_matrix_a, sample_matrix_b], axis=0)
    )

    return total_order


def _estimate_second_order_index(
    sample_matrix_a,
    sample_matrix_ab_j,
    sample_matrix_ab_k,
    sample_matrix_ba_j,
    sample_matrix_b,
    first_order_estimator,
):
    """Estimate second-order Sobol indices.

    [Saltelli2010] Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design
    and Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
    (1 February 2010): 259–270. https://doi.org/10.1016/j.cpc.2009.09.018.

    Args:
        sample_matrix_a (ndarray): results corresponding to A sample matrix
        sample_matrix_ab_j (ndarray): results corresponding to AB sample matrix
        sample_matrix_ab_k (ndarray): results corresponding to AB sample matrix
        sample_matrix_ba_j (ndarray): results corresponding to BA sample matrix
        sample_matrix_b (ndarray): results corresponding to B sample matrix
        first_order_estimator (str): estimator for first-order indices

    Returns:
        second_order (ndarray): second-order Sobol index estimates
    """
    # from SALib library [Saltelli2010]
    total_second_order_effect_jk = np.mean(
        sample_matrix_ba_j * sample_matrix_ab_k - sample_matrix_a * sample_matrix_b, axis=0
    ) / np.var(np.r_[sample_matrix_a, sample_matrix_b], axis=0)
    first_order_index_j = _estimate_first_order_index(
        sample_matrix_a, sample_matrix_b, sample_matrix_ab_j, first_order_estimator
    )
    first_order_index_k = _estimate_first_order_index(
        sample_matrix_a, sample_matrix_b, sample_matrix_ab_k, first_order_estimator
    )
    second_order = total_second_order_effect_jk - first_order_index_j - first_order_index_k

    return second_order


def _estimate_closed_third_order_index(sample_matrix_b, sample_matrix_ab_ijk):
    """Estimate closed third-order Sobol indices.

    Estimator based on Equation (4.1) in

    Le Gratiet, Loic, Claire Cannamela, and Bertrand Iooss. ‘A Bayesian Approach for Global
    Sensitivity Analysis of (Multifidelity) Computer Codes’. SIAM/ASA Journal on Uncertainty
    Quantification 2, no. 1 (1 January 2014): 336–63.
    https://doi.org/10.1137/130926869.

    Args:
        sample_matrix_b (ndarray): results corresponding to B
        sample_matrix_ab_ijk (ndarray): results corresponding to AB_ijk

    Returns:
        second_order (ndarray): second-order Sobol index estimates
    """
    closed_third_order = (
        np.mean(sample_matrix_b * sample_matrix_ab_ijk)
        - sample_matrix_b.mean() * sample_matrix_ab_ijk.mean()
    ) / (np.mean(sample_matrix_b * sample_matrix_b) - sample_matrix_b.mean() ** 2)

    return closed_third_order
