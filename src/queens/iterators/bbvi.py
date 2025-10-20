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
"""Black box variational inference iterator."""

import logging

import numpy as np

from queens.iterators._variational_inference import VALID_EXPORT_FIELDS, VariationalInference
from queens.utils.collection import CollectionObject
from queens.utils.logger_settings import log_init_args
from queens.utils.valid_options import check_if_valid_options

_logger = logging.getLogger(__name__)


class BBVI(VariationalInference):
    """Black box variational inference (BBVI) iterator.

    For Bayesian inverse problems. BBVI does not require model gradients and can hence be used with
    any simulation model and without the need for adjoint implementations. The algorithm is based
    on [1]. The expectations for the gradient computations are computed using an importance
    sampling approach where the IS-distribution is constructed as a mixture of the variational
    distribution from previous iterations (similar as in [2]).

    **Keep in mind:** This algorithm requires the logpdf of the variational distribution to be
    differentiable w.r.t. the variational parameters. This is not the case for certain
    distributions, e.g. uniform distribution, and can therefore not be used in combination with
    this algorithm (see [3] page 13)!

    References:
        [1]: Ranganath, Rajesh, Sean Gerrish, and David M. Blei. "Black Box Variational Inference."
             Proceedings of the Seventeenth International Conference on Artificial Intelligence
             and Statistics. 2014.
        [2]: Arenz, Neumann & Zhong. "Efficient Gradient-Free Variational Inference using Policy
             Search." Proceedings of the 35th International Conference on Machine Learning (2018) in
             PMLR 80:234-243
        [3]: Mohamed et al. "Monte Carlo Gradient Estimation in Machine Learning". Journal of
             Machine Learning Research. 21(132):1−62, 2020.

    Attributes:
        control_variates_scaling_type (str): Flag to decide how to compute control variate scaling.
        loo_cv_bool (boolean): *True* if leave-one-out procedure is used for the control variate
                               scaling estimations. Is quite slow!
        random_seed (int): Seed for the random number generators.
        max_feval (int): Maximum number of simulation runs for this analysis.
        memory (int): Number of previous iterations that should be included in the MC ELBO
                      gradient estimations. For *memory=0* the algorithm reduces to standard the
                      standard BBVI algorithm. (Better variable name is welcome.)
        model_eval_iteration_period (int): If the iteration number is a multiple of this number,
                                           the probabilistic model is sampled independent of the
                                           other conditions.
        resample (bool): *True* is resampling should be used.
        log_variational_mat (np.array): Logpdf evaluations of the variational distribution.
        grad_params_log_variational_mat (np.array): Column-wise grad params logpdf (score function)
                                                    of the variational distribution.
        log_posterior_unnormalized (np.array): Row-vector logarithmic probabilistic model evaluation
                                               (generally unnormalized).
        samples_list (list): List of samples from previous iterations for the ISMC gradient.
        parameter_list (list): List of parameters from previous iterations for the ISMC gradient.
        log_posterior_unnormalized_list (list): List of probabilistic model evaluations from
                                                previous iterations for the ISMC gradient.
        ess (float): Effective sample size of the current iteration (in case IS is used).
        sampling_bool (bool): *True* if probabilistic model has to be sampled. If importance
                              sampling is used the forward model might not evaluated in
                              every iteration.
        sample_set (np.ndarray): Set of samples used to evaluate the probabilistic model is
                                 not needed in other VI methods.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        result_description,
        variational_distribution,
        n_samples_per_iter,
        random_seed,
        max_feval,
        control_variates_scaling_type,
        loo_control_variates_scaling,
        stochastic_optimizer,
        variational_transformation=None,
        variational_parameter_initialization=None,
        memory=0,
        natural_gradient=True,
        FIM_dampening=True,
        decay_start_iteration=50,
        dampening_coefficient=1e-2,
        FIM_dampening_lower_bound=1e-8,
        model_eval_iteration_period=1000,
        resample=False,
        verbose_every_n_iter=10,
    ):
        """Initialize BBVI iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            result_description (dict): Settings for storing and visualizing the results
            variational_distribution (Variational): variational distribution object
            n_samples_per_iter (int): Batch size per iteration (number of simulations per iteration
                                                to estimate the involved expectations)
            random_seed (int): Seed for the random number generators
            max_feval (int): Maximum number of simulation runs for this analysis
            control_variates_scaling_type (str): Flag to decide how to compute control variate
                                                scaling
            loo_control_variates_scaling: True if leave-one-out procedure is used for the control
                                          variate scaling estimations. Is quite slow!
            stochastic_optimizer (obj): QUEENS stochastic optimizer object
            variational_transformation (str): String encoding the transformation that will be
                                              applied to the variational density
            variational_parameter_initialization (str): Flag to decide how to initialize the
                                                        variational parameters
            memory (int): Number of previous iterations that should be included in the MC ELBO
                          gradient estimations. For memory=0 the algorithm reduces to standard the
                          standard BBVI algorithm. (Better variable name is welcome)
            natural_gradient (boolean): True if natural gradient should be used
            FIM_dampening (boolean): True if FIM dampening should be used
            decay_start_iteration (int): Iteration at which the FIM dampening is started
            dampening_coefficient (float): Initial nugget term value for the FIM dampening
            FIM_dampening_lower_bound (float): Lower bound on the FIM dampening coefficient

            model_eval_iteration_period (int): If the iteration number is a multiple of this number
                                               the probabilistic model is sampled independent of the
                                               other conditions
            resample (bool): True is resampling should be used
            verbose_every_n_iter (int): Number of iterations between printing, plotting, and saving

        Returns:
            bbvi_obj (obj): Instance of the BBVI
        """
        # pylint: disable=duplicate-code
        valid_export_fields = ["ess", "weights"] + VALID_EXPORT_FIELDS
        iterative_data_names = result_description.get("iterative_field_names", [])
        check_if_valid_options(valid_export_fields, iterative_data_names)
        iteration_data = CollectionObject(*iterative_data_names)

        super().__init__(
            model=model,
            parameters=parameters,
            global_settings=global_settings,
            result_description=result_description,
            variational_distribution=variational_distribution,
            variational_params_initialization=variational_parameter_initialization,
            n_samples_per_iter=n_samples_per_iter,
            variational_transformation=variational_transformation,
            random_seed=random_seed,
            max_feval=max_feval,
            natural_gradient=natural_gradient,
            FIM_dampening=FIM_dampening,
            decay_start_iter=decay_start_iteration,
            dampening_coefficient=dampening_coefficient,
            FIM_dampening_lower_bound=FIM_dampening_lower_bound,
            stochastic_optimizer=stochastic_optimizer,
            iteration_data=iteration_data,
            verbose_every_n_iter=verbose_every_n_iter,
        )

        if not memory:
            model_eval_iteration_period = 1

        self.control_variates_scaling_type = control_variates_scaling_type
        self.loo_cv_bool = loo_control_variates_scaling
        self.random_seed = random_seed
        self.max_feval = max_feval
        self.memory = memory
        self.model_eval_iteration_period = model_eval_iteration_period
        self.resample = resample
        self.log_variational_mat = None
        self.grad_params_log_variational_mat = None
        self.log_posterior_unnormalized = None
        self.samples_list = []
        self.parameter_list = []
        self.log_posterior_unnormalized_list = []
        self.ess = 0
        self.sampling_bool = True
        self.sample_set = None

    def core_run(self):
        """Core run for black-box variational inference."""
        _logger.info("Starting black box Bayesian variational inference...")
        super().core_run()

    def eval_log_likelihood(self, samples):
        """Calculate the log-likelihood of the observation data.

        Evaluation of the likelihood model for all inputs of the sample batch will trigger
        the actual forward simulation.

        Args:
            samples (np.array): Samples (n_samples x n_dimension)

        Returns:
            log_likelihood (np.array): Vector of the log-likelihood function for all input
            samples of the current batch
        """
        # The first samples belong to simulation input
        # get simulation output (run actual forward problem)
        log_likelihood = self.model.evaluate(samples)["result"]

        return log_likelihood.flatten()

    def get_log_prior(self, samples):
        """Evaluate the log prior of the model for a sample batch.

        The samples are transformed according to the selected
        transformation.

        Args:
            samples (np.array): Samples (n_samples x n_dimension)

        Returns:
            log_prior (np.array): log-prior vector evaluated for current sample batch
        """
        return self.parameters.joint_logpdf(samples)

    def get_log_posterior_unnormalized(self, samples):
        """Calculate the unnormalized log posterior for a sample batch.

        Args:
            samples (np.array): Samples (n_samples x n_dimension)

        Returns:
            unnormalized_log_posterior (np.array): Values of unnormalized log posterior
            distribution at positions of sample batch
        """
        # Transform the samples
        samples = self._transform_samples(samples)
        log_prior = self.get_log_prior(samples)
        log_likelihood = self.eval_log_likelihood(samples)
        log_posterior_unnormalized = log_likelihood + log_prior
        return log_posterior_unnormalized.flatten()

    def _verbose_output(self):
        """Give some informative outputs during the BBVI iterations."""
        _logger.info("-" * 80)
        _logger.info("Iteration %s of BBVI algorithm", self.stochastic_optimizer.iteration + 1)

        super()._verbose_output()

        if self.memory > 0 and self.stochastic_optimizer.iteration > 0:
            _logger.info("ESS: %.2f of %s", self.ess, (self.memory + 1) * self.n_samples_per_iter)
        if self.stochastic_optimizer.iteration > 1:
            _logger.info("Likelihood noise variance: %s", self.model.normal_distribution.covariance)
        _logger.info("-" * 80)

    def _prepare_result_description(self):
        """Creates the dictionary for the result pickle file.

        Returns:
            result_description (dict): Dictionary with result summary of the analysis
        """
        result_description = super()._prepare_result_description()
        result_description.update(
            {
                "control_variates_scaling_type": self.control_variates_scaling_type,
                "loo_control_variates": self.loo_cv_bool,
            }
        )

        if self.memory > 0:
            result_description.update({"memory": self.memory})

        if self.iteration_data:
            result_description["iteration_data"].update(self.iteration_data.to_dict())
        return result_description

    @staticmethod
    def _averaged_control_variates_scalings(f_mat, h_mat, weights_is):
        """Averaged control variate scalings.

        This function computes the control variate scaling averaged over the
        components of the control variate.

        Args:
            f_mat (np.array): MC gradient samples (n_variational_parameters x n_samples)
            h_mat (np.array): Control variate samples (n_variational_parameters x n_samples)
            weights_is (np.array): importance sampling weights (1 x n_samples)

        Returns:
            cv_scaling (np.array): Control variate scalings (n_variational_parameters x 1)
        """
        dim = len(h_mat)
        cov_sum = 0
        var_sum = 0
        for ielbo, covariate in zip(f_mat, h_mat):
            cov_sum += np.cov(ielbo, covariate, aweights=weights_is)[0, 1]
            # Use cov instead of np.var to use weights
            var_sum += float(np.cov(covariate, aweights=weights_is))
        cv_scaling = np.ones((dim, 1)) * cov_sum / var_sum
        return cv_scaling

    @staticmethod
    def _componentwise_control_variates_scalings(f_mat, h_mat, weights_is):
        """Computes the componentwise control variates scaling.

        I.e., every component of the control variate separately is computed separately.

        Args:
            f_mat (np.array): MC gradient samples (n_variational_parameters x n_samples)
            h_mat (np.array): Control variate samples (n_variational_parameters x n_samples)
            weights_is (np.array): importance sampling weights (1 x n_samples)

        Returns:
            cv_scaling (np.array): Control variate scalings (n_variational_parameters x 1)
        """
        n_parameters = len(h_mat)
        cv_scaling = np.ones((n_parameters, 1))
        for i in range(n_parameters):
            cv_scaling[i] = np.cov(f_mat[i], h_mat[i])[0, 1] / float(
                np.cov(h_mat[i], aweights=weights_is)
            )
        return cv_scaling

    @staticmethod
    def _loo_control_variates_scalings(cv_obj, f_mat, h_mat, weights_is):
        """Leave one out control variates.

        To reduce bias in the MC and control variate scaling estimation
        Ranganath proposed a leave-one-out procedure to estimate the control
        variate scalings. Each sample has its own scaling that is computed
        using f_mat and h_mat without the values related to itself. (see
        http://arks.princeton.edu/ark:/88435/dsp01pr76f608w) Is slow!

        Args:
            cv_obj (control variate function): A control variate scaling function
            f_mat (np.array): MC gradient samples (n_variational_parameters x n_samples)
            h_mat (np.array): Control variate samples (n_variational_parameters x n_samples)
            weights_is (np.array): importance sampling weights (1 x n_samples)

        Returns:
            cv_scaling (np.array): Control variate scalings (n_variational_parameters x n_samples)
        """
        cv_scaling = []
        for i in range(f_mat.shape[1]):
            scv = cv_obj(np.delete(f_mat, i, 1), np.delete(h_mat, i, 1), np.delete(weights_is, i))
            cv_scaling.append(scv)
        cv_scaling = np.concatenate(cv_scaling, axis=1)
        return cv_scaling

    def _get_control_variates_scalings(self, f_mat, h_mat, weights_is):
        """Calculate the control variate scalings.

        Args:
            f_mat (np.array): MC gradient samples (n_variational_parameters x n_samples)
            h_mat (np.array): Control variate samples (n_variational_parameters x n_samples)
            weights_is (np.array): importance sampling weights (1 x n_samples)

        Returns:
            cv_scaling (np.array): for loo CV: (n_variational_parameters x n_samples)
                                         else: (n_variational_parameters x 1)
        """
        if self.control_variates_scaling_type == "componentwise":
            cv_scaling_obj = self._componentwise_control_variates_scalings
        elif self.control_variates_scaling_type == "averaged":
            cv_scaling_obj = self._averaged_control_variates_scalings
        else:
            valid_options = {"componentwise", "averaged"}
            raise NotImplementedError(
                f"{self.control_variates_scaling_type} unknown, valid types are {valid_options}"
            )
        if self.loo_cv_bool:
            cv_scaling = self._loo_control_variates_scalings(
                cv_scaling_obj, f_mat, h_mat, weights_is
            )
        else:
            cv_scaling = cv_scaling_obj(f_mat, h_mat, weights_is)
        return cv_scaling

    def _calculate_elbo_gradient(self, variational_parameters):
        """Estimate the ELBO gradient expression.

        Based on MC with importance sampling with the samples of previous iterations if desired.
        The score function is used as a control variate. No Rao-Blackwellization scheme
        is used.

        Args:
            variational_parameters (np.array): Variational parameters

        Returns:
            grad_elbo (np.array): ELBO gradient
        """
        self.variational_params = variational_parameters

        # Check if evaluating the probabilistic model is necessary
        self._check_if_sampling_necessary()
        if self.sampling_bool:
            self._sample_probabilistic_model()

        # Use IS sampling (if enabled)
        selfnormalized_weights_is, normalizing_constant_is = self._prepare_importance_sampling()
        if self.stochastic_optimizer.iteration > self.memory and self.memory > 0 and self.resample:
            # Number of samples to resample
            n_samples = int(self.n_samples_per_iter * self.memory)
            # Resample
            self._resample(selfnormalized_weights_is, n_samples)
            # After resampling the weights
            selfnormalized_weights_is, normalizing_constant_is = 1 / n_samples, n_samples

        #  Evaluate the logpdf and grad params logpdf function of the variational distribution
        self._evaluate_variational_distribution_for_batch()
        self._filter_failed_simulations()

        # Compute the MC samples, without control variates
        f_mat = self.grad_params_log_variational_mat * (
            self.log_posterior_unnormalized - self.log_variational_mat
        )

        # Compute the control variate at the given samples
        # We assume the expectation of the control variate is zero
        h_mat = self.grad_params_log_variational_mat

        if isinstance(selfnormalized_weights_is, (float, int)):
            weights = (
                np.ones(self.log_posterior_unnormalized.shape)
                * selfnormalized_weights_is
                * normalizing_constant_is
            )
        else:
            weights = normalizing_constant_is * selfnormalized_weights_is

        # Get control variate scalings
        control_variate_scalings = self._get_control_variates_scalings(
            f_mat, h_mat, weights.flatten()
        )

        # MC gradient estimation with control variates
        grad_elbo = normalizing_constant_is * np.mean(
            selfnormalized_weights_is * (f_mat - control_variate_scalings * h_mat), axis=1
        )

        # Compute the logpdf for the elbo estimate (here no IS is used)
        self._calculate_elbo(selfnormalized_weights_is, normalizing_constant_is)

        self.iteration_data.add(
            samples=self.sample_set,
            weights=weights,
            n_sims=self.model.num_evaluations,
            likelihood_variance=self.model.normal_distribution.covariance,
        )

        return grad_elbo

    def _sample_probabilistic_model(self):
        """Evaluate probabilistic model."""
        # Draw samples for the current iteration
        self.sample_set = self.variational_distribution.draw(
            self.variational_params, self.n_samples_per_iter
        )

        # Calls the (unnormalized) probabilistic model
        self.log_posterior_unnormalized = self.get_log_posterior_unnormalized(self.sample_set)

    def _evaluate_variational_distribution_for_batch(self):
        """Evaluate logpdf and score function."""
        self.log_variational_mat = self.variational_distribution.logpdf(
            self.variational_params, self.sample_set
        )

        self.grad_params_log_variational_mat = self.variational_distribution.grad_params_logpdf(
            self.variational_params, self.sample_set
        )

        # Convert if NaNs to floats. For high dimensional RV floating point issues
        # might be avoided this way
        self.log_variational_mat = np.nan_to_num(self.log_variational_mat)
        self.grad_params_log_variational_mat = np.nan_to_num(self.grad_params_log_variational_mat)

    def _resample(self, selfnormalized_weights, n_samples):
        """Stratified resampling.

        Args:
            selfnormalized_weights (np.array): weights of the samples
            n_samples (int): number of samples
        """
        random_sample_within_bins = (np.random.rand(n_samples) + np.arange(n_samples)) / n_samples

        idx = []
        cumulative_probability = np.cumsum(selfnormalized_weights)
        i, j = 0, 0
        while i < n_samples:
            if random_sample_within_bins[i] < cumulative_probability[j]:
                idx.append(j)
                i += 1
            else:
                j += 1

        self.log_posterior_unnormalized = self.log_posterior_unnormalized[idx]
        self.sample_set = self.sample_set[idx]

    def _check_if_sampling_necessary(self):
        """Check if resampling is necessary.

        Sampling is necessary if one of the following condition is met:
            1. No memory is used
            2. Not enough samples in memory
            3. ESS number is too small
            4. Every model_eval_iteration_period
        """
        max_ess = self.n_samples_per_iter * (self.memory)
        self.sampling_bool = (
            self.memory == 0
            or self.stochastic_optimizer.iteration <= self.memory
            or self.ess < 0.5 * max_ess
            or self.stochastic_optimizer.iteration % self.model_eval_iteration_period == 0
        )

    def _calculate_elbo(self, selfnormalized_weights, normalizing_constant):
        """Calculate the ELBO.

        Args:
            selfnormalized_weights (np.array): selfnormalized importance sampling weights
            normalizing_constant (int): Importance sampling normalizing constant
        """
        instant_elbo = selfnormalized_weights * (
            self.log_posterior_unnormalized - self.log_variational_mat
        )
        elbo = normalizing_constant * np.mean(instant_elbo)
        self.iteration_data.add(elbo=elbo)
        self.elbo = elbo

    def _prepare_importance_sampling(self):
        r"""Helper functions for the importance sampling.

        Importance sampling based gradient computation (if enabled). This includes:
            1. Store samples, variational parameters and probabilistic model evaluations
            2. Update variables samples and log_posterior_unnormalized
            3. Compute autonormalized weights and the normalizing constant

        The normalizing constant is a constant in order to recover the proper weights values. The
        gradient estimation is multiplied with this constant in order to avoid a bias. This can
        be done since for any constant :math:`a`:
        :math:`\int_x h(x) p(x) dx = a \int_x h(x) \frac{1}{a}p(x) dx`

        Returns:
            selfnormalized_weights (np.array): (1 x n_samples)
            normalizing_constant (int): Normalizing constant
            samples (np.array): Samples (n_samples x n_dimension)
        """
        # Values if no IS is used or for the first iteration
        selfnormalized_weights = 1
        normalizing_constant = 1

        # If importance sampling is used
        if self.memory > 0:
            ess = len(self.sample_set)
            self._update_sample_and_posterior_lists()

            # The number of iterations that we want to keep the samples and model evals
            if self.stochastic_optimizer.iteration > 0:
                weights_is = self.get_importance_sampling_weights(
                    self.parameter_list, self.sample_set
                )

                # Self normalize weighs
                normalizing_constant = np.sum(weights_is)
                selfnormalized_weights = weights_is / normalizing_constant
                ess = 1 / np.sum(selfnormalized_weights**2)
            self.ess = ess
            self.iteration_data.add(ess=ess)

        return selfnormalized_weights, normalizing_constant

    def _update_sample_and_posterior_lists(self):
        """Assemble the samples for IS MC gradient estimation."""
        # Check if probabilistic model was sampled
        if self.sampling_bool:
            # Store the current samples, parameters and probabilistic model evals
            self.parameter_list.append(self.variational_params)
            self.samples_list.append(self.sample_set)
            self.log_posterior_unnormalized_list.append(self.log_posterior_unnormalized)

        # The number of iterations that we want to keep the samples and model evals
        if self.stochastic_optimizer.iteration >= self.memory:
            self.parameter_list = self.parameter_list[-(self.memory + 1) :]
            self.samples_list = self.samples_list[-(self.memory + 1) :]
            self.log_posterior_unnormalized_list = self.log_posterior_unnormalized_list[
                -(self.memory + 1) :
            ]

            self.sample_set = np.concatenate(self.samples_list, axis=0)
            self.log_posterior_unnormalized = np.concatenate(
                self.log_posterior_unnormalized_list, axis=0
            )

    def get_importance_sampling_weights(self, variational_params_list, samples):
        r"""Get the importance sampling weights for the MC gradient estimation.

        Uses a special computation of the weights using the logpdfs to reduce
        numerical issues:

        :math:`w=\frac{q_i}{\sum_{j=0}^{memory+1} \frac{1}{memory+1}q_j}=\frac{memory+1}
        {\sum_{j=0}^{memory+1}exp(ln(q_j)-ln(q_i))}`

        and is therefore slightly slower. Assumes the mixture coefficients are all equal.

        Args:
            variational_params_list (list): Variational parameters list of the current and the
                                            desired previous iterations
            samples (np.array): Samples (n_samples x n_dimension)

        Returns:
            weights (np.array): (Unnormalized) weights for the ISMC evaluated for the
            given samples (1 x n_samples)
        """
        inv_weights = 0
        n_mixture = len(variational_params_list)
        log_pdf_current_iteration = self.variational_distribution.logpdf(
            self.variational_params, samples
        )
        for params in variational_params_list:
            inv_weights += np.exp(
                self.variational_distribution.logpdf(params, samples) - log_pdf_current_iteration
            )
        weights = n_mixture / inv_weights
        return weights

    def _filter_failed_simulations(self):
        """Filter samples failed simulations."""
        # Indices where the log joint is a nan
        idx = np.where(~np.isnan(self.log_posterior_unnormalized))[0]
        if len(idx) != len(self.log_posterior_unnormalized):
            _logger.warning("At least one probabilistic model call resulted in a NaN")
        if self.log_variational_mat.ndim > 1:
            self.log_variational_mat = self.log_variational_mat[:, idx]
        else:
            self.log_variational_mat = self.log_variational_mat[idx]
        self.grad_params_log_variational_mat = self.grad_params_log_variational_mat[:, idx]
        self.log_posterior_unnormalized = self.log_posterior_unnormalized[idx]
