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
"""Models.

Modules for multi-query mapping of inputs to outputs, such as parameter
samples to model evaluations.
"""

from queens.models.bmfm import BMFM
from queens.models.differentiable_adjoint import DifferentiableAdjoint
from queens.models.differentiable_fd import DifferentiableFD
from queens.models.likelihoods.bayesian_mf_gaussian import BMFGaussian, BmfiaInterface
from queens.models.likelihoods.gaussian import Gaussian
from queens.models.simulation_model import Simulation
from queens.models.surrogates.bayesian_neural_network import GaussianBayesianNeuralNetwork
from queens.models.surrogates.gaussian_neural_network import GaussianNeuralNetwork
from queens.models.surrogates.gp_approximation_gpflow import GPFlowRegression
from queens.models.surrogates.gp_approximation_gpflow_svgp import GPflowSVGP
from queens.models.surrogates.gp_approximation_jitted import GPJitted
from queens.models.surrogates.gp_heteroskedastic_gpflow import HeteroskedasticGP

VALID_TYPES = {
    "simulation_model": Simulation,
    "bmfmc_model": BMFM,
    "gaussian": Gaussian,
    "bmf_gaussian": BMFGaussian,
    "bmfia_interface": BmfiaInterface,
    "differentiable_simulation_model_fd": DifferentiableFD,
    "differentiable_simulation_model_adjoint": DifferentiableAdjoint,
    "heteroskedastic_gp": HeteroskedasticGP,
    "gp_approximation_gpflow": GPFlowRegression,
    "gaussian_bayesian_neural_network": GaussianBayesianNeuralNetwork,
    "gp_jitted": GPJitted,
    "gp_approximation_gpflow_svgp": GPflowSVGP,
    "gaussian_nn": GaussianNeuralNetwork,
}
