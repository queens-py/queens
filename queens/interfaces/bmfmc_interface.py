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
"""Interface for grouping outputs with inputs."""

from queens.utils.logger_settings import log_init_args

# pylint: disable=invalid-name


class BmfmcInterface:
    """Interface for grouping outputs with inputs.

    Interface for grouping the outputs of several simulations with identical
    input to one data point. The *BmfmcInterface* is basically a version of the
    *approximation_interface* class, that allows vectorized mapping and
    implicit function relationships.

    Attributes:
        probabilistic_mapping (obj): Instance of the probabilistic mapping, which models the
                                     probabilistic dependency between high-fidelity model,
                                     low-fidelity models and informative input features.

    Returns:
        BMFMCInterface (obj): Instance of the BMFMCInterface
    """

    @log_init_args
    def __init__(self, probabilistic_mapping):
        """Initialize the interface.

        Args:
            probabilistic_mapping (obj): Instance of the probabilistic mapping, which models the
                                         probabilistic dependency between high-fidelity model,
                                         low-fidelity models and informative input features.
        """
        self.probabilistic_mapping = probabilistic_mapping

    def evaluate(self, samples, support="y", full_cov=False, gradient_bool=False):
        r"""Predict on probabilistic mapping.

        Call the probabilistic mapping and predict the mean and variance
        for the high-fidelity model, given the inputs *z_lf* (called samples here).

        Args:
            samples (np.array): Low-fidelity feature vector *z_lf* that contains the corresponding
                                Monte-Carlo points on which the probabilistic mapping should
                                be evaluated
            support (str): Support/variable for which we predict the mean and (co)variance. For
                           *support=f*  the Gaussian process predicts w.r.t. the latent function
                           *f*. For the choice of *support=y* we predict w.r.t. the
                           simulation/experimental output *y*
            full_cov (bool): Boolean that specifies whether the entire posterior covariance matrix
                             should be returned or only the posterior variance
            gradient_bool (bool): Flag to determine whether the gradient of the function at
                                  the evaluation point is expected (*True*) or not (*False*)

        Returns:
            mean_Y_HF_given_Z_LF (np.array): Vector of mean predictions
              :math:`\mathbb{E}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*, \mathcal{D}_{f})]`
              for the HF model given the low-fidelity feature input
            var_Y_HF_given_Z_LF (np.array): Vector of variance predictions
              :math:`\mathbb{V}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,\mathcal{D}_{f})]`
              for the HF model given the low-fidelity feature input
        """
        if self.probabilistic_mapping is None:
            raise RuntimeError(
                "The probabilistic mapping has not been properly initialized, cannot continue!"
            )
        if gradient_bool:
            raise NotImplementedError(
                "The gradient response is not implemented for this interface. Please set "
                "`gradient_bool=False`. Abort..."
            )

        output = self.probabilistic_mapping.predict(samples, support=support, full_cov=full_cov)
        mean_Y_HF_given_Z_LF = output["result"]
        var_Y_HF_given_Z_LF = output["variance"]
        return mean_Y_HF_given_Z_LF, var_Y_HF_given_Z_LF

    def build_approximation(self, Z_LF_train, Y_HF_train):
        r"""Build and train the probabilistic mapping.

        Based on the training inputs
        :math:`\mathcal{D}_f={Y_{HF},Z_{LF}}`.

        Args:
            Z_LF_train (np.array): Training inputs for probabilistic mapping
            Y_HF_train (np.array): Training outputs for probabilistic mapping
        """
        self.probabilistic_mapping.setup(Z_LF_train, Y_HF_train)
        self.probabilistic_mapping.train()
