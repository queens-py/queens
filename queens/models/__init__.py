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

from queens.models.adjoint import Adjoint
from queens.models.bmfmc import BMFMC
from queens.models.finite_difference import FiniteDifference
from queens.models.likelihoods.bmf_gaussian import BMFGaussian, BmfiaInterface
from queens.models.likelihoods.gaussian import Gaussian
from queens.models.reinforcement_learning.reinforcement_learning import ReinforcementLearning
from queens.models.simulation import Simulation
from queens.models.surrogates.bayesian_neural_network import GaussianBayesianNeuralNetwork
from queens.models.surrogates.gaussian_neural_network import GaussianNeuralNetwork
from queens.models.surrogates.gaussian_process import GaussianProcess
from queens.models.surrogates.heteroskedastic_gaussian_process import HeteroskedasticGaussianProcess
from queens.models.surrogates.jitted_gaussian_process import JittedGaussianProcess
from queens.models.surrogates.variational_gaussian_process import VariationalGaussianProcess

VALID_TYPES = {
    "simulation_model": Simulation,
    "bmfmc_model": BMFMC,
    "gaussian": Gaussian,
    "bmf_gaussian": BMFGaussian,
    "bmfia_interface": BmfiaInterface,
    "differentiable_simulation_model_fd": FiniteDifference,
    "differentiable_simulation_model_adjoint": Adjoint,
    "heteroskedastic_gp": HeteroskedasticGaussianProcess,
    "gp_approximation_gpflow": GaussianProcess,
    "gaussian_bayesian_neural_network": GaussianBayesianNeuralNetwork,
    "gp_jitted": JittedGaussianProcess,
    "gp_approximation_gpflow_svgp": VariationalGaussianProcess,
    "gaussian_nn": GaussianNeuralNetwork,
    "reinforcement_learning": ReinforcementLearning,
}
