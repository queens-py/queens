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
"""Base class for variational inference iterator."""

import abc
import logging
import time

import numpy as np

from queens.iterators._iterator import Iterator
from queens.utils.process_outputs import write_results
from queens.variational_distributions import FullRankNormal, MeanFieldNormal
from queens.visualization.variational_inference_visualization import VIVisualization

_logger = logging.getLogger(__name__)

VALID_EXPORT_FIELDS = [
    "elbo",
    "n_sims",
    "samples",
    "likelihood_variance",
    "variational_parameters",
    "learning_rate",
]


class VariationalInference(Iterator):
    """Stochastic variational inference iterator.

    References:
        [1]: Mohamed et al. "Monte Carlo Gradient Estimation in Machine Learning". Journal of
             Machine Learning Research. 21(132):1−62, 2020.
        [2]: Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A
             Review for Statisticians. Journal of the American Statistical Association, 112(518),
             859–877. https://doi.org/10.1080/01621459.2017.1285773
        [3]: Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). Stochastic variational
             inference. Journal of Machine Learning Research, 14(1), 1303–1347.



    Attributes:
        result_description (dict): Settings for storing and visualizing the results.
        variational_params_initialization_approach (str): Flag to decide how to initialize the
                                                          variational parameters.
        n_samples_per_iter (int): Batch size per iteration (number of simulations per iteration to
                                  estimate the involved expectations).
        variational_transformation (str): String encoding the transformation that will be applied to
                                          the variational density.
        natural_gradient_bool (boolean): *True* if natural gradient should be used.
        fim_decay_start_iter (float): Iteration at which the FIM dampening is started.
        fim_dampening_coefficient (float): Initial nugget term value for the FIM dampening.
        fim_dampening_lower_bound (float): Lower bound on the FIM dampening coefficient.
        fim_dampening_bool (boolean): *True* if FIM dampening should be used.
        random_seed (int): Seed for the random number generators.
        max_feval (int): Maximum number of simulation runs for this analysis.
        num_parameters (int): Actual number of model input parameters that should be calibrated.
        stochastic_optimizer (obj): QUEENS stochastic optimizer object.
        variational_distribution (Variational): Variational distribution object.
        variational_params (np.array): Row vector containing the variational parameters.
        elbo: Evidence lower bound.
        nan_in_gradient_counter (int): Count how many times *NaNs* appeared in the gradient estimate
                                       in a row.
        iteration_data (CollectionObject): Object to store iteration data if desired.
        verbose_every_n_iter (int): Number of iterations between printing, plotting, and saving
    """

    def __init__(
        self,
        model,
        parameters,
        global_settings,
        result_description,
        variational_distribution,
        variational_params_initialization,
        n_samples_per_iter,
        variational_transformation,
        random_seed,
        max_feval,
        natural_gradient,
        FIM_dampening,
        decay_start_iter,
        dampening_coefficient,
        FIM_dampening_lower_bound,
        stochastic_optimizer,
        iteration_data,
        verbose_every_n_iter=10,
    ):
        """Initialize VI iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            result_description (dict): Settings for storing and visualizing the results
            variational_distribution (Variational): Variational distribution object
            variational_params_initialization (str): Flag to decide how to initialize the
                                                     variational parameters
            n_samples_per_iter (int): Batch size per iteration (number of simulations per iteration
                                                to estimate the involved expectations)
            variational_transformation (str): String encoding the transformation that will be
                                              applied to the variational density
            random_seed (int): Seed for the random number generators
            max_feval (int): Maximum number of simulation runs for this analysis
            natural_gradient (boolean): True if natural gradient should be used
            FIM_dampening (boolean): True if FIM dampening should be used
            decay_start_iter (int): Iteration at which the FIM dampening is started
            dampening_coefficient (float): Initial nugget term value for the FIM dampening
            FIM_dampening_lower_bound (float): Lower bound on the FIM dampening coefficient
            stochastic_optimizer (obj): QUEENS stochastic optimizer object
            iteration_data (CollectionObject): Object to store iteration data if desired
            verbose_every_n_iter (int): Number of iterations between printing, plotting, and saving
        Returns:
            Initialise variational inference iterator
        """
        super().__init__(model, parameters, global_settings)

        self.result_description = result_description
        self.variational_params_initialization_approach = variational_params_initialization
        self.n_samples_per_iter = n_samples_per_iter
        self.variational_transformation = variational_transformation
        self.natural_gradient_bool = natural_gradient
        self.fim_decay_start_iter = decay_start_iter
        self.fim_dampening_coefficient = dampening_coefficient
        self.fim_dampening_lower_bound = FIM_dampening_lower_bound
        self.fim_dampening_bool = FIM_dampening
        self.random_seed = random_seed
        self.max_feval = max_feval
        self.num_parameters = self.parameters.num_parameters
        self.stochastic_optimizer = stochastic_optimizer
        self.variational_distribution = variational_distribution
        self.variational_params = None
        self.elbo = -np.inf
        self.nan_in_gradient_counter = 0
        self.iteration_data = iteration_data
        self.verbose_every_n_iter = verbose_every_n_iter

        self.visualization = None
        if result_description.get("plotting_options"):
            self.visualization = VIVisualization.from_config_create(
                result_description["plotting_options"]
            )

    def core_run(self):
        """Core run for stochastic variational inference."""
        start = time.time()

        self.iteration_data.add(variational_parameters=self.variational_params)
        old_parameters = self.variational_params.copy()

        # Stochastic optimization
        for self.variational_params in self.stochastic_optimizer:
            self._catch_non_converging_simulations(old_parameters)

            # Just to avoid constant spamming
            if self.stochastic_optimizer.iteration % self.verbose_every_n_iter == 0:
                self._verbose_output()
                self._write_results()
                self._clearing_and_plots()

            # Stop the optimizer in case of too many simulations
            if self.model.num_evaluations >= self.max_feval:
                break

            self.iteration_data.add(learning_rate=self.stochastic_optimizer.learning_rate)
            self.iteration_data.add(variational_parameters=self.variational_params)
            old_parameters = self.variational_params.copy()

        end = time.time()

        if self.model.num_evaluations > self.max_feval:
            _logger.warning("Maximum probabilistic model calls reached")
        elif np.any(np.isnan(self.stochastic_optimizer.rel_l2_change)):
            _logger.warning("NaN(s) in the relative change of variational parameters")
        else:
            _logger.info("Finished successfully! :-)")
        _logger.info("Variational inference took %s seconds.", end - start)

    def _catch_non_converging_simulations(self, old_parameters):
        """Reset variational parameters in case of failed simulations."""
        if np.isnan(self.stochastic_optimizer.rel_l2_change):
            self.variational_params = old_parameters

    def pre_run(self):
        """Initialize the prior model and variational parameters."""
        _logger.info("Initialize Optimization run.")
        if self.parameters.random_field_flag:
            raise NotImplementedError(
                "Variational inference for random fields is not yet implemented! Abort..."
            )
        self._initialize_variational_params()

        # set the gradient according to input
        self.stochastic_optimizer.set_gradient_function(self.get_gradient_function())
        self.stochastic_optimizer.current_variational_parameters = self.variational_params

    def post_run(self):
        """Write results and potentially visualize them."""
        if self.result_description["write_results"]:
            result_dict = self._prepare_result_description()
            write_results(result_dict, self.global_settings.result_file(".pickle"))

        if self.visualization:
            self.visualization.save_plots()

    def _verbose_output(self):
        """Give some informative outputs during the BBVI iterations."""
        mean_change = self.stochastic_optimizer.rel_l2_change * 100
        _logger.info("So far %s simulation runs", self.model.num_evaluations)
        _logger.info("L2 change of all variational parameters: %.4f %%", mean_change)
        _logger.info("The elbo is: %.2f", self.elbo)
        # Avoids a busy screen
        if self.variational_params.shape[0] > 24:
            _logger.info(
                "First 24 of %s variational parameters : \n", self.variational_params.shape[0]
            )
            _logger.info(self.variational_params[:24])
        else:
            _logger.info("Values of variational parameters: \n")
            _logger.info(self.variational_params)

    def _write_results(self):
        """Write results to output file."""
        if self.result_description["write_results"]:
            result_dict = self._prepare_result_description()
            write_results(result_dict, self.global_settings.result_file(".pickle"))

    def _initialize_variational_params(self):
        """Initialize the variational parameters.

        There are two possibilities:
            1. Random initialization:
                Is handled by the variational distribution object
            2. Initialization based on the prior modeling (only for normal distributions!)
                Extract the prior moments and initialize the parameters based on them
        """
        if self.variational_params_initialization_approach == "random":
            self.variational_params = (
                self.variational_distribution.initialize_variational_parameters(random=True)
            )
        elif self.variational_params_initialization_approach == "prior":
            if isinstance(
                self.variational_distribution,
                (MeanFieldNormal, FullRankNormal),
            ):
                mu, cov = self._initialize_variational_params_from_prior()
                var_params = self.variational_distribution.construct_variational_parameters(mu, cov)
                self.variational_params = var_params
            else:
                raise ValueError(
                    "Initializing the variational parameters based on the prior is only possible"
                    "for distribution family 'normal'"
                )
        else:
            valid_initialization_types = {"random", "prior"}
            raise NotImplementedError(
                f"{self.variational_params_initialization_approach} is not known.\n"
                f"Valid options are {valid_initialization_types}"
            )

    def _initialize_variational_params_from_prior(self):
        """Initialize the variational parameters based on the prior.

        The variational distribution might be transformed in a second
        step such that the actual variational distribution is of a
        different family. Only is used for normal distributions.
        """
        # Get the first and second moments of the prior distributions
        mean_list_prior = []
        cov_list_prior = []
        if self.parameters.num_parameters > 0:
            for params in self.parameters.to_distribution_list():
                mean_list_prior.append(params.mean)
                cov_list_prior.append(params.covariance.squeeze())
        # Set the mean and variance params of the variational distr such that the
        # transformed distribution would match the moments of the prior
        if self.variational_transformation == "exp":
            mean_list_variational = [
                np.log(E**2 / np.sqrt(E**2 + C)) for E, C in zip(mean_list_prior, cov_list_prior)
            ]
            cov_list_variational = [
                np.sqrt(np.log(1 + C / E**2)) for E, C in zip(mean_list_prior, cov_list_prior)
            ]
        elif self.variational_transformation is None:
            mean_list_variational = mean_list_prior
            cov_list_variational = cov_list_prior
        else:
            raise ValueError(
                f"The transformation type {self.variational_transformation} for the "
                f"variational density is unknown! Abort..."
            )

        return np.array(mean_list_variational), np.diag(cov_list_variational)

    def _transform_samples(self, x_mat):
        """Transform samples.

        Transform samples of the variational distribution according to the specified
        transformation mapping.

        Args:
            x_mat (np.array): Samples of the variational distribution

        Returns:
            x_mat_trans (np.array): Transformed samples of variational distribution
        """
        if self.variational_transformation == "exp":
            x_mat_trans = np.exp(x_mat)
        elif self.variational_transformation is None:
            x_mat_trans = x_mat
        else:
            raise ValueError(
                f"The transformation type {self.variational_transformation} for the "
                f"variational density is unknown! Abort..."
            )
        return x_mat_trans

    def _prepare_result_description(self):
        """Create the dictionary for the result pickle file.

        Returns:
            result_description (dict): Dictionary with result summary of the analysis
        """
        result_description = {
            "final_elbo": self.elbo,
            "final_variational_parameters": self.variational_params,
            "batch_size": self.n_samples_per_iter,
            "number_of_sim": self.model.num_evaluations,
            "natural_gradient": self.natural_gradient_bool,
            "fim_dampening": self.fim_dampening_bool,
            "variational_parameter_initialization": self.variational_params_initialization_approach,
        }

        if self.iteration_data:
            result_description.update({"iteration_data": self.iteration_data.to_dict()})

        distribution_dict = self.variational_distribution.export_dict(self.variational_params)
        if self.variational_transformation == "exp":
            distribution_dict.update(
                {"variational_transformation": self.variational_transformation}
            )

        result_description.update({"variational_distribution": distribution_dict})
        return result_description

    @abc.abstractmethod
    def _calculate_elbo_gradient(self, variational_parameters):
        """Estimate the ELBO gradient expression.

        Args:
            variational_parameters (np.array): Variational parameters

        Returns:
            elbo gradient as column vector (np.array)
        """

    def _clearing_and_plots(self):
        """Visualization and clear some internal variables."""
        # some plotting and output
        if self.visualization and self.visualization.plot_boolean:
            self.visualization.plot_convergence(
                self.stochastic_optimizer.iteration,
                self.iteration_data.variational_parameters,
                self.iteration_data.elbo,
            )

    def _get_fim(self):
        """Get the FIM for the current variational distribution.

        Add dampening if desired.

        Returns:
            fisher (np.array): fisher information matrix of the variational distribution
        """
        fim = self.variational_distribution.fisher_information_matrix(self.variational_params)
        if self.fim_dampening_bool:
            if self.stochastic_optimizer.iteration > self.fim_decay_start_iter:
                dampening_coefficient = self.fim_dampening_coefficient * np.exp(
                    -(self.stochastic_optimizer.iteration - self.fim_decay_start_iter)
                    / self.fim_decay_start_iter
                )
                dampening_coefficient = max(self.fim_dampening_lower_bound, dampening_coefficient)
            else:
                dampening_coefficient = self.fim_dampening_coefficient
            fim = fim + np.eye(len(fim)) * dampening_coefficient
        return fim

    def get_gradient_function(self):
        """Select the gradient function for the stochastic optimizer.

        Two options exist, with or without natural gradient.

        Returns:
            obj: function to evaluate the gradient
        """
        safe_gradient = self.handle_gradient_nan(self._calculate_elbo_gradient)
        if self.natural_gradient_bool:

            def my_gradient(variational_parameters):
                return np.linalg.solve(self._get_fim(), safe_gradient(variational_parameters))

            gradient = my_gradient

        else:
            gradient = safe_gradient

        return gradient

    def handle_gradient_nan(self, gradient_function):
        """Handle *NaN* in gradient estimations.

        Args:
            gradient_function (function): Function that estimates the gradient

        Returns:
             function: Gradient function wrapped with the counter
        """

        def nan_counter_and_warner(*args, **kwargs):
            """Count iterations with NaNs and write warning."""
            gradient = gradient_function(*args, **kwargs)
            if np.isnan(gradient).any():
                _logger.warning(
                    "Gradient estimate contains NaNs (number of iterations in a row with NaNs:"
                    " %s)",
                    self.nan_in_gradient_counter,
                )
                gradient = np.nan_to_num(gradient)
                self.nan_in_gradient_counter += 1
            else:
                self.nan_in_gradient_counter = 0

            # Arbitrary number, this will be changed in the future
            if self.nan_in_gradient_counter == 10:
                raise ValueError(
                    "Variational inference stopped: 10 iterations in a row failed to compute a"
                    "bounded gradient!"
                )
            return gradient

        return nan_counter_and_warner
