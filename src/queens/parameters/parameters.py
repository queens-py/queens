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
"""Parameters."""

import logging

import numpy as np

from queens.distributions._distribution import Continuous, Discrete
from queens.parameters.random_fields._random_field import RandomField
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


def _add_parameters_keys(parameters_keys, parameter_name, dimension):
    """Add parameter keys to existing parameter keys.

    If the dimension of a parameter is larger than one, a separate unique key is added for each
    parameter member.
    Example: If parameter x1 is 3-dimensional the keys x1_0, x1_1, x1_2 is added.
             If parameter x1 is 1-dimensional the key x1 is added.

    Args:
        parameters_keys (list): List of existing parameter keys
        parameter_name (str): Parameter name to be added
        dimension (int): dimension of Parameter

    Returns:
        parameters_keys (list): List of keys for all parameter members
    """
    if dimension == 1:
        parameters_keys.append(parameter_name)
    else:
        parameters_keys.extend([f"{parameter_name}_{i}" for i in range(dimension)])
    return parameters_keys


class Parameters:
    """Parameters class.

    Attributes:
        dict (dict): Random variables and random fields stored in a dict.
        parameters_keys (list): List of keys for all parameter members.
        num_parameters (int): Number of (truncated) parameters.
        random_field_flag (bool): Specifies if random fields are used.
        names (list): Parameter names.
    """

    @log_init_args
    def __init__(self, **parameters):
        """Initialize Parameters object.

        Args:
            **parameters (Distribution, RandomField): parameters as keyword arguments
        """
        self.dict = {}
        self.parameters_keys = []
        self.num_parameters = 0
        self.random_field_flag = False
        self.names = []

        # Initialize with any parameters passed in
        if parameters:
            self.add_parameters(**parameters)

    def add_parameters(self, **parameters):
        """Add parameters to the Parameters object.

        Args:
            **parameters (Distribution, RandomField): parameters as keyword arguments
        """
        for name, obj in parameters.items():
            if name in self.dict:
                raise ValueError(f"Parameter '{name}' already exists.")

            if isinstance(obj, (Continuous, Discrete)):
                self.parameters_keys = _add_parameters_keys(
                    self.parameters_keys, name, obj.dimension
                )
                self.num_parameters += obj.dimension
            elif isinstance(obj, RandomField):
                self.parameters_keys += obj.coords["keys"]
                self.num_parameters += obj.dimension
                self.random_field_flag = True
            else:
                raise NotImplementedError(
                    f"Parameter class '{obj.__class__.__name__}' not supported."
                )

            self.dict[name] = obj
            self.names.append(name)

    def draw_samples(self, num_samples):
        """Draw samples from all parameters.

        Args:
            num_samples (int): The number of samples to draw for each parameter.

        Returns:
            samples (np.ndarray): Drawn samples
        """
        samples = np.zeros((num_samples, self.num_parameters))
        current_index = 0
        for parameter in self.to_list():
            samples[:, current_index : current_index + parameter.dimension] = parameter.draw(
                num_samples
            )
            current_index += parameter.dimension
        return samples

    def joint_logpdf(self, samples):
        """Evaluate the logpdf summed over all parameters.

        Args:
            samples (np.ndarray): Samples for which to evaluate the joint logpdf. Each row
                                  represents a sample and each column corresponds to a parameter
                                  dimension.

        Returns:
            logpdf (np.ndarray): logpdf summed over all parameters
        """
        samples = samples.reshape(-1, self.num_parameters)
        logpdf = 0
        i = 0
        for parameter in self.to_list():
            logpdf += parameter.logpdf(samples[:, i : i + parameter.dimension])
            i += parameter.dimension
        return logpdf

    def grad_joint_logpdf(self, samples):
        """Evaluate the gradient of the joint logpdf w.r.t. the samples.

        Args:
            samples (np.ndarray): Samples for which to evaluate the gradient of the joint logpdf.
                                  Each row represents a sample and each column corresponds to a
                                  parameter dimension.

        Returns:
            grad_logpdf (np.ndarray): Gradient of the joint logpdf w.r.t. the samples
        """
        samples = samples.reshape(-1, self.num_parameters)
        grad_logpdf = np.zeros(samples.shape)
        j = 0
        for parameter in self.to_list():
            grad_logpdf[:, j : j + parameter.dimension] = parameter.grad_logpdf(
                samples[:, j : j + parameter.dimension]
            )
            j += parameter.dimension
        return grad_logpdf

    def latent_grad(self, upstream_gradient):
        """Gradient of the rvs and rfs w.r.t. latent variables.

        Args:
            upstream_gradient (np.array): Upstream gradient
        Returns:
            gradient (np.ndarray): Gradient of the joint rvs/rfs w.r.t. the samples
        """
        if self.random_field_flag:
            upstream_gradient = np.atleast_2d(upstream_gradient)
            gradient = np.zeros(shape=(upstream_gradient.shape[0], self.num_parameters))
            index_latent = 0
            index_field = 0
            for parameter in self.to_list():
                if isinstance(parameter, RandomField):
                    gradient[:, index_latent : index_latent + parameter.dimension] = (
                        parameter.latent_gradient(
                            upstream_gradient[:, index_field : index_field + parameter.dim_coords]
                        )
                    )
                    index_field += parameter.dim_coords
                else:
                    gradient[:, index_latent : index_latent + parameter.dimension] = (
                        upstream_gradient[:, index_field : index_field + parameter.dimension]
                    )
                    index_field += parameter.dimension
                index_latent += parameter.dimension
            return gradient
        return upstream_gradient

    def inverse_cdf_transform(self, samples):
        """Transform samples to unit interval.

        Args:
            samples (np.ndarray): Samples that should be transformed.

        Returns:
            transformed_samples (np.ndarray): Transformed samples
        """
        samples = samples.reshape(-1, self.num_parameters)
        transformed_samples = np.zeros(samples.shape)
        for i, parameter in enumerate(self.to_list()):
            if parameter.dimension != 1:
                raise ValueError("Only 1D Random variables can be transformed!")
            transformed_samples[:, i] = parameter.ppf(samples[:, i])
        return transformed_samples

    def sample_as_dict(self, sample):
        """Return sample as a dict.

        Args:
            sample (np.ndarray): A single sample

        Returns:
            sample_dict (dict): Dictionary containing sample members and the corresponding parameter
            keys
        """
        sample_dict = {}
        sample = sample.reshape(-1)
        if self.random_field_flag:
            sample = self.expand_random_field_realization(sample)
        for j, key in enumerate(self.parameters_keys):
            sample_dict[key] = sample[j]
        return sample_dict

    def expand_random_field_realization(self, truncated_sample):
        """Expand truncated representation of random fields.

        Args:
            truncated_sample (np.ndarray): Truncated representation of sample

        Returns:
            sample_expanded (np.ndarray): Expanded representation of sample
        """
        sample_expanded = np.zeros(len(self.parameters_keys))
        index_truncated = 0
        index_expanded = 0
        for parameter in self.to_list():
            if isinstance(parameter, RandomField):
                sample_expanded[index_expanded : index_expanded + parameter.dim_coords] = (
                    parameter.expanded_representation(
                        truncated_sample[index_truncated : index_truncated + parameter.dimension]
                    )
                )
                index_expanded += parameter.dim_coords
                index_truncated += parameter.dimension
            else:
                sample_expanded[index_expanded : index_expanded + parameter.dimension] = (
                    truncated_sample[index_truncated : index_truncated + parameter.dimension]
                )
                index_expanded += parameter.dimension
                index_truncated += parameter.dimension
        return sample_expanded

    def to_list(self):
        """Return parameters as list.

        Returns:
            parameter_list (list): List of parameters
        """
        parameter_list = list(self.dict.values())
        return parameter_list

    def to_distribution_list(self):
        """Return the distributions of the parameters as list.

        Returns:
            distribution_list (list): List of distributions of parameters
        """
        distribution_list = []
        for parameter in self.to_list():
            if isinstance(parameter, RandomField):
                distribution_list.append(parameter.distribution)
            else:
                distribution_list.append(parameter)

        return distribution_list
