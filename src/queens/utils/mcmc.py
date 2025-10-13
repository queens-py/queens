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
"""Collection of utils for Markov Chain Monte Carlo algorithms."""

from typing import Tuple, Union

import numpy as np
from numpy import ndarray


def mh_select(
    log_acceptance_probability: ndarray, current_sample: ndarray, proposed_sample: ndarray
) -> Tuple[ndarray, ndarray]:
    """Perform Metropolis-Hastings selection.

    The Metropolis-Hastings algorithm is used in Markov Chain Monte Carlo (MCMC) methods to
    accept or reject a proposed sample based on the log of the acceptance probability. This function
    compares the acceptance probability with a random number between 0 and 1 to decide if each
    proposed sample should replace the current sample.
    If the random number is smaller than the acceptance probability, the proposed sample is
    accepted. The function further checks whether the `log_acceptance_probability` is finite. If
    it is infinite or NaN, the function will not accept the respective proposed sample.

    Args:
        log_acceptance_probability: Logarithm of the acceptance probability for each sample. This
            represents the log of the ratio of the probability densities of the proposed sample to
            the probability densities of the proposed sample to the current sample.
        current_sample: The current sample values from the MCMC chain.
        proposed_sample: The proposed sample values to be considered for acceptance.

    Returns:
        The sample values selected after the Metropolis-Hastings step.
            If the proposed sample is accepted, it will be returned; otherwise, the current sample
            is returned.
        A bool array indicating whether each proposed sample was accepted.
    """
    isfinite = np.isfinite(log_acceptance_probability)
    accept = (
        np.log(np.random.uniform(size=log_acceptance_probability.shape))
        < log_acceptance_probability
    )

    bool_idx = isfinite * accept

    selected_samples = np.where(bool_idx[:, np.newaxis], proposed_sample, current_sample)

    return selected_samples, bool_idx


def tune_scale_covariance(
    scale_covariance: Union[ndarray, float], accept_rate: Union[ndarray, float]
) -> ndarray:
    r"""Adjust the covariance scaling factor based on the acceptance rate.

    This function tunes the covariance scaling factor used in Metropolis-Hastings or similar MCMC
    algorithms based on the observed acceptance rate of proposed samples. The goal is to maintain an
    acceptance rate within the range of 20% to 50%, which is considered optimal for many MCMC
    algorithms. The covariance scaling factor is adjusted according to the following rules:

        +------------------+-----------------------------+
        | Acceptance Rate  |  Variance adaptation factor |
        +==================+=============================+
        |     <0.001       |              x 0.1          |
        +------------------+-----------------------------+
        |      <0.05       |              x 0.5          |
        +------------------+-----------------------------+
        |      <0.2        |              x 0.9          |
        +------------------+-----------------------------+
        |      >0.5        |              x 1.1          |
        +------------------+-----------------------------+
        |      >0.75       |              x 2            |
        +------------------+-----------------------------+
        |      >0.95       |              x 10           |
        +------------------+-----------------------------+

    Reference:
    [1]: https://github.com/pymc-devs/pymc/blob/main/pymc/step_methods/metropolis.py

    Args:
        scale_covariance: The current covariance scaling factor for the proposal distribution.
        accept_rate: The observed acceptance rate of the proposed samples. This value should be
            between 0 and 1.

    Returns:
        The updated covariance scaling factor adjusted according to the acceptance rate.
    """
    scale_covariance = np.where(accept_rate < 0.001, scale_covariance * 0.1, scale_covariance)
    scale_covariance = np.where(
        (accept_rate >= 0.001) * (accept_rate < 0.05), scale_covariance * 0.5, scale_covariance
    )
    scale_covariance = np.where(
        (accept_rate >= 0.05) * (accept_rate < 0.2), scale_covariance * 0.9, scale_covariance
    )
    scale_covariance = np.where(
        (accept_rate > 0.5) * (accept_rate <= 0.75), scale_covariance * 1.1, scale_covariance
    )
    scale_covariance = np.where(
        (accept_rate > 0.75) * (accept_rate <= 0.95), scale_covariance * 2.0, scale_covariance
    )
    scale_covariance = np.where((accept_rate > 0.95), scale_covariance * 10.0, scale_covariance)

    return scale_covariance
