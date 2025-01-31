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

Map inputs to outputs, such as parameter samples to model evaluations
and possibly gradients.
"""

from queens.models.bmfmc_model import BMFMCModel
from queens.models.differentiable_simulation_model_adjoint import (
    DifferentiableSimulationModelAdjoint,
)
from queens.models.differentiable_simulation_model_fd import DifferentiableSimulationModelFD
from queens.models.likelihood_models.bayesian_mf_gaussian_likelihood import BMFGaussianModel
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.models.simulation_model import SimulationModel
from queens.models.surrogate_models.bayesian_neural_network import (
    GaussianBayesianNeuralNetworkModel,
)
from queens.models.surrogate_models.gaussian_neural_network import GaussianNeuralNetworkModel
from queens.models.surrogate_models.gp_approximation_gpflow import GPFlowRegressionModel
from queens.models.surrogate_models.gp_approximation_gpflow_svgp import GPflowSVGPModel
from queens.models.surrogate_models.gp_approximation_jitted import GPJittedModel
from queens.models.surrogate_models.gp_heteroskedastic_gpflow import HeteroskedasticGPModel

VALID_TYPES = {
    "simulation_model": SimulationModel,
    "bmfmc_model": BMFMCModel,
    "gaussian": GaussianLikelihood,
    "bmf_gaussian": BMFGaussianModel,
    "differentiable_simulation_model_fd": DifferentiableSimulationModelFD,
    "differentiable_simulation_model_adjoint": DifferentiableSimulationModelAdjoint,
    "heteroskedastic_gp": HeteroskedasticGPModel,
    "gp_approximation_gpflow": GPFlowRegressionModel,
    "gaussian_bayesian_neural_network": GaussianBayesianNeuralNetworkModel,
    "gp_jitted": GPJittedModel,
    "gp_approximation_gpflow_svgp": GPflowSVGPModel,
    "gaussian_nn": GaussianNeuralNetworkModel,
}
