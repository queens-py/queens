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
"""Multi-fidelity Gaussian likelihood model."""

import copy
import logging
import multiprocessing as mp
import time
from multiprocessing import get_context

import numpy as np
import tqdm

from queens.distributions.mean_field_normal import MeanFieldNormal
from queens.models.likelihoods._likelihood import Likelihood
from queens.utils.ascii_art import print_bmfia_acceleration
from queens.utils.logger_settings import log_init_args
from queens.utils.valid_options import get_option

_logger = logging.getLogger(__name__)


class BMFGaussian(Likelihood):
    """Multi fidelity likelihood function.

    Multi-fidelity likelihood of the Bayesian multi-fidelity inverse
    analysis scheme [1, 2].

    Attributes:
        coords_mat (np.array): Row-wise coordinates at which the observations were recorded
        time_vec (np.array): Vector of observation times
        output_label (str): Name of the experimental outputs (column label in csv-file)
        coord_labels (lst): List with coordinate labels for (column labels in csv-file)
        mf_interface (obj): QUEENS multi-fidelity interface
        mf_subiterator (obj): Subiterator to select the training data of the
                                probabilistic regression model
        normal_distribution (obj): Mean field normal distribution object
        noise_var (np.array): Noise variance of the observations
        likelihood_counter (int): Internal counter for the likelihood evaluation
        num_refinement_samples (int): Number of additional samples to train the multi-fidelity
                                      dependency in refinement step
        likelihood_evals_for_refinement (lst):  List with necessary number of likelihood
                                                evaluations before the refinement step is
                                                conducted

    Returns:
        Instance of BMFGaussian. This is a multi-fidelity version of the
        Gaussian noise likelihood model.


    References:
        [1] Nitzler, J.; Biehler, J.; Fehn, N.; Koutsourelakis, P.-S. and Wall, W.A. (2020),
            "A Generalized Probabilistic Learning Approach for Multi-Fidelity Uncertainty
            Propagation in Complex Physical Simulations", arXiv:2001.02892

        [2] Nitzler J.; Wall, W. A. and Koutsourelakis P.-S. (2021),
            "An efficient Bayesian multi-fidelity framework for the solution of high-dimensional
            inverse problems in computationally demanding simulations", unpublished internal report
    """

    @log_init_args
    def __init__(
        self,
        forward_model,
        mf_interface,
        mf_subiterator,
        experimental_data_reader,
        mf_approx,
        noise_value=None,
        num_refinement_samples=None,
        likelihood_evals_for_refinement=None,
    ):
        """Instantiate the multi-fidelity likelihood class.

        Args:
            forward_model (obj): Forward model to iterate; here: the low fidelity model
            mf_interface (obj): QUEENS multi-fidelity interface
            mf_subiterator (obj): Subiterator to select the training data of the probabilistic
                                  regression model
            experimental_data_reader (obj): Experimental data reader object
            mf_approx (Model): Probabilistic mapping
            noise_value (array_like): Noise variance of the observations
            num_refinement_samples (int): Number of additional samples to train the multi-fidelity
                                          dependency in refinement step
            likelihood_evals_for_refinement (lst): List with necessary number of likelihood
                                                   evaluations before the refinement step is
                                                   conducted
            plotting_options (dict): Options for plotting
        """
        (
            y_obs,
            self.coords_mat,
            self.time_vec,
            _,
            _,
            self.coord_labels,
            self.output_label,
        ) = experimental_data_reader.get_experimental_data()
        super().__init__(forward_model, y_obs)

        if not isinstance(mf_interface, BmfiaInterface):
            raise ValueError("The interface type must be 'bmfia_interface' for BMFGaussian!")

        # ----------------------- initialize the mean field normal distribution ------------------
        noise_variance = np.array(noise_value)
        dimension = y_obs.size

        # build distribution with dummy values; parameters might change during runtime
        mean_field_normal = MeanFieldNormal(
            mean=y_obs, variance=noise_variance, dimension=dimension
        )

        # ---------------------- initialize some model settings/train surrogates -----------------
        self.initialize_bmfia_iterator(self.coords_mat, self.time_vec, y_obs, mf_subiterator)
        self.build_approximation(
            mf_subiterator,
            mf_interface,
            mf_approx,
            self.coord_labels,
            self.time_vec,
            self.coords_mat,
        )

        self.mf_interface = mf_interface
        self.mf_subiterator = mf_subiterator
        self.min_log_lik_mf = None
        self.normal_distribution = mean_field_normal
        self.noise_var = noise_variance
        self.likelihood_counter = 1
        self.num_refinement_samples = num_refinement_samples
        self.likelihood_evals_for_refinement = likelihood_evals_for_refinement

    def _evaluate(self, samples):
        """Evaluate multi-fidelity likelihood.

        Evaluation with current set of variables
        which are an attribute of the underlying low-fidelity simulation model.

        Args:
            samples (np.ndarray): Evaluated samples

        Returns:
            dict: Vector of log-likelihood values per model input.
        """
        # reshape the model output according to the number of coordinates
        num_coordinates = self.coords_mat.shape[0]
        num_samples = samples.shape[0]

        # we explicitly cut the array at the variable size as within one batch several chains
        # e.g., in MCMC might be calculated; we only want the last chain here
        forward_model_output = self.forward_model.evaluate(samples)["result"].reshape(
            -1, num_coordinates
        )[:num_samples, :]

        mf_log_likelihood = self.evaluate_from_output(samples, forward_model_output)
        self.response = {
            "forward_model_output": forward_model_output,
            "mf_log_likelihood": mf_log_likelihood,
        }

        return {"result": mf_log_likelihood}

    def grad(self, samples, upstream_gradient):
        r"""Evaluate gradient of model w.r.t. current set of input samples.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
                                          :math:`\frac{\partial g}{\partial f}`

        Returns:
            gradient (np.array): Gradient w.r.t. current set of input samples
                                 :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """
        partial_grad = self.partial_grad_evaluate(samples, self.response["forward_model_output"])
        upstream_gradient = upstream_gradient * partial_grad
        gradient = self.forward_model.grad(samples, upstream_gradient)
        return gradient

    def evaluate_from_output(self, samples, forward_model_output):
        """Evaluate multi-fidelity likelihood from forward model output.

        Args:
            samples (np.ndarray): Samples to evaluate
            forward_model_output (np.ndarray): Forward model output

        Returns:
            mf_log_likelihood (np.array): Vector of log-likelihood values per model input.
        """
        if self._adaptivity_trigger():
            raise NotImplementedError("Adaptivity is not yet implemented for BMFGaussian!")

        # evaluate the modified multi-fidelity likelihood expression with LF model response
        mf_log_likelihood = self.evaluate_mf_likelihood(samples, forward_model_output)
        self.likelihood_counter += 1
        return mf_log_likelihood

    def partial_grad_evaluate(self, forward_model_input, forward_model_output):
        """Implement the partial derivative of the evaluate method.

        The partial derivative w.r.t. the output of the sub-model is for example
        required to calculate gradients of the current model w.r.t. to the sample
        input.

        Args:
            forward_model_input (np.array): Sample inputs of the model run (here not required).
            forward_model_output (np.array): Output of the underlying sub- or forward model
                                             for the current batch of sample inputs.

        Returns:
            grad_out (np.array): Evaluated partial derivative of the evaluation function
                                 w.r.t. the output of the underlying sub-model.
        """
        # construct LF feature matrix
        z_mat = self.mf_subiterator.set_feature_strategy(
            forward_model_output,
            forward_model_input,
            self.coords_mat[: forward_model_output.shape[0]],
        )

        # Get the response matrices of the multi-fidelity mapping
        m_f_mat, var_y_mat, grad_m_f_mat, grad_var_y_mat = self.mf_interface.evaluate_and_gradient(
            z_mat
        )

        if grad_m_f_mat.ndim == 3:
            grad_m_f_mat = grad_m_f_mat[:, :, 0]  # extract only derivative w.r.t. to LF output
            grad_var_y_mat = grad_var_y_mat[:, :, 0]  # extract only derivative w.r.t. to LF output

        assert np.array_equal(
            m_f_mat.shape[1], np.atleast_2d(self.y_obs).shape[1]
        ), "Column dimension of the probab. regression output and y_obs do not agree!"

        # here we iterate over samples meaning we
        # iterate here over all surrogates simultaneously such that
        # the new m is a vector of all e.g. first entries in all surrogates
        log_lik_mf_lst = []
        grad_log_lik_lst = []
        for m_f_vec, variance_vec, grad_m_f, grad_var_y in zip(
            m_f_mat, var_y_mat, grad_m_f_mat, grad_var_y_mat, strict=True
        ):
            self.normal_distribution.update_variance(
                variance_vec.flatten() + self.noise_var.flatten()
            )
            log_lik_mf_lst.append(self.normal_distribution.logpdf(m_f_vec.reshape(1, -1)))
            grad_log_lik_lst.append(
                self.grad_log_pdf_d_ylf(m_f_vec, grad_m_f, grad_var_y).flatten()
            )

        log_lik_mf_output = np.array(log_lik_mf_lst).reshape(-1, 1)
        grad_out = np.array(grad_log_lik_lst)

        if self.min_log_lik_mf is None:
            self.min_log_lik_mf = np.min(log_lik_mf_output)

        return grad_out

    def _adaptivity_trigger(self):
        """Triggers adaptive refinement for the m_f_likelihood."""
        if (
            self.likelihood_evals_for_refinement
            and self.likelihood_counter in self.likelihood_evals_for_refinement
        ):
            return True
        return False

    def _refine_mf_likelihood(self, additional_x_train, additional_y_lf_train=None):
        """Refine multi-fidelity likelihood.

        Args:
            additional_x_train (np.array): New input training points.
            additional_y_lf_train (np.array, optional): New output training points.
                                                        Defaults to None.
        """
        z_train, y_hf_train = self.mf_subiterator.expand_training_data(
            additional_x_train, additional_y_lf_train=additional_y_lf_train
        )
        _logger.info("Start updating the probabilistic model...")
        self.mf_interface.build_approximation(z_train, y_hf_train)
        _logger.info("---------------------------------------------------------------------")
        _logger.info("Probabilistic model was updated successfully!")
        _logger.info("---------------------------------------------------------------------")

    def evaluate_mf_likelihood(self, x_batch, y_lf_mat):
        """Evaluate the Bayesian multi-fidelity likelihood as described in [1].

        Args:
            x_batch (np.array): Input batch matrix; rows correspond to one input vector;
                                different dimensions along columns

            y_lf_mat (np.array): Response matrix of the low-fidelity model; Row-wise corresponding
                                 to rows in x_batch input batch matrix. Different coordinate
                                 locations along the columns

        Returns:
            log_lik_mf_output (tuple): Tuple with vector of log-likelihood values
                                       per model input and potentially the gradient
                                       of the model w.r.t. its inputs


        References:
            [1] Nitzler, J., Biehler, J., Fehn, N., Koutsourelakis, P.-S. and Wall, W.A. (2020),
                "A Generalized Probabilistic Learning Approach for Multi-Fidelity Uncertainty
                Propagation in Complex Physical Simulations", arXiv:2001.02892
        """
        # construct LF feature matrix
        z_mat = self.mf_subiterator.set_feature_strategy(
            y_lf_mat, x_batch, self.coords_mat[: y_lf_mat.shape[0]]
        )
        # Get the response matrices of the multi-fidelity mapping
        m_f_mat, var_y_mat = self.mf_interface.evaluate(z_mat)
        assert np.array_equal(
            m_f_mat.shape[1], np.atleast_2d(self.y_obs).shape[1]
        ), "Column dimension of the probab. regression output and y_obs do not agree! Abort..."

        # iterate here over all surrogates simultaneously such that the
        # new m is a vector of all, e.g., first entries in all surrogates
        log_lik_mf_lst = []
        for m_f_vec, variance_vec in zip(m_f_mat, var_y_mat, strict=True):
            self.normal_distribution.update_variance(
                variance_vec.flatten() + self.noise_var.flatten()
            )
            log_lik_mf_lst.append(self.normal_distribution.logpdf(m_f_vec.reshape(1, -1)))
        log_lik_mf_output = np.array(log_lik_mf_lst).reshape(-1)

        if self.min_log_lik_mf is None:
            self.min_log_lik_mf = np.min(log_lik_mf_output)

        return log_lik_mf_output

    def grad_log_pdf_d_ylf(self, m_f_vec, grad_m_f_dy, grad_var_y_dy):
        """Calculate the gradient of the logpdf w.r.t. to the LF model output.

        The gradient is calculated from the individual partial derivatives
        and then composed in this method.

        Args:
            m_f_vec (np.array): mean vector of the probabilistic surrogate evaluated at sample
                                points
            grad_m_f_dy (np.array): gradient of the mean function/vector of the probabilistic
                                 regression model w.r.t. the regression model's input
            grad_var_y_dy (np.array): gradient of the variance function/vector of the probabilistic
                                   regression model w.r.t. the regression model's input

        Returns:
            d_log_lik_d_z (np.array): gradient of the logpdf w.r.t. y_lf
        """
        d_log_lik_d_m_f = self.normal_distribution.grad_logpdf(m_f_vec).reshape(1, -1)
        d_log_lik_d_var = self.normal_distribution.grad_logpdf_var(m_f_vec).reshape(1, -1)

        d_log_lik_d_y = d_log_lik_d_m_f * grad_m_f_dy.reshape(
            1, -1
        ) + d_log_lik_d_var * grad_var_y_dy.reshape(1, -1)

        return d_log_lik_d_y

    @staticmethod
    def initialize_bmfia_iterator(coords_mat, time_vec, y_obs, bmfia_subiterator):
        """Initialize the bmfia iterator.

        Args:
            coords_mat (np.array): Coordinates of the experimental data.
            time_vec (np.array): Time vector of the experimental data.
            y_obs (np.array): Experimental data observations at coordinates
            bmfia_subiterator (bmfia_subiterator): BMFIA subiterator object.
        """
        _logger.info("---------------------------------------------------------------------")
        _logger.info("Speed-up through Bayesian multi-fidelity inverse analysis (BMFIA)!")
        _logger.info("---------------------------------------------------------------------")
        print_bmfia_acceleration()

        bmfia_subiterator.coords_experimental_data = coords_mat
        bmfia_subiterator.time_vec = time_vec
        bmfia_subiterator.y_obs = y_obs

    @staticmethod
    def build_approximation(
        bmfia_subiterator,
        bmfia_interface,
        approx,
        coord_labels,
        time_vec,
        coords_mat,
    ):
        """Construct the probabilistic surrogate / mapping.

        Surrogate is calculated based on the provided training-data and
        optimize the hyper-parameters by maximizing the data's evidence
        or its lower bound (ELBO).

        Args:
            bmfia_subiterator (bmfia_subiterator): BMFIA subiterator object.
            bmfia_interface (bmfia_interface): BMFIA interface object.
            approx (Model): Approximation for probabilistic mapping.
            coord_labels (list): List of coordinate labels.
            time_vec (np.array): Time vector of the experimental data.
            coords_mat (np.array): (Spatial) Coordinates of the experimental data.
        """
        # Start the bmfia (sub)iterator to create the training data for the probabilistic mapping
        z_train, y_hf_train = bmfia_subiterator.core_run()
        # ----- train regression model on the data ----------------------------------------
        bmfia_interface.build_approximation(
            z_train, y_hf_train, approx, coord_labels, time_vec, coords_mat
        )
        _logger.info("---------------------------------------------------------------------")
        _logger.info("Probabilistic model was built successfully!")
        _logger.info("---------------------------------------------------------------------")


# pylint: disable=invalid-name
# pylint: disable=too-many-lines


class BmfiaInterface:
    r"""Interface class for Bayesian multi-fidelity inverse analysis.

    Interface for grouping the outputs of several simulation models with
    identical model inputs to one multi-fidelity data point in the
    multi-fidelity space.

    Attributes:
        num_processors_multi_processing (int): Number of processors that should be used in the
                                               multi-processing pool.
        evaluate_method (method): Configured method to evaluate the probabilistic mapping
        evaluate_and_gradient_method (method): Configured method to evaluate the probabilistic
                                                mapping and its gradient
        instantiate_probabilistic_mappings (method): Configured method to instantiate the
                                                      probabilistic mapping objects
        probabilistic_mapping_obj_lst (lst): List of probabilistic mapping objects, which
                                             models the probabilistic dependency between
                                             high-fidelity model, low-fidelity models and
                                             informative input features for
                                             each coordinate tuple of
                                             :math:`y_{lf} \times y_{hf} \times \gamma_i`
                                             individually.
        update_mappings_method (method): Configured method to update the probabilistic mapping
        coord_labels (str): Labels / names of the coordinates in the experimental data file
        time_vec (np.array): Vector with time-stamps of observations
        coords_mat (np.array): Matrix with coordinate values for observations

    Returns:
        BmfiaInterface (obj): Instance of the BmfiaInterface
    """

    @staticmethod
    def instantiate_per_coordinate(
        z_lf_train,
        y_hf_train,
        _time_vec,
        _coords_mat,
        approx,
    ):
        """Instantiate probabilistic mappings per coordinate.

        Args:
            z_lf_train (np.array): Training inputs for probabilistic mapping.
                                   Rows: Samples, Columns: Coordinates
            y_hf_train (np.array): Training outputs for probabilistic mapping.
                                   Rows: Samples, Columns: Coordinates
            _time_vec (np.array): Time vector of the experimental data.
            _coords_mat (np.array): (Spatial) Coordinates of the experimental data.
            approx (Model): Probabilistic mapping

        Returns:
            z_lf_train (np.array): Input matrix in correct ordering
            probabilistic_mapping_obj_lst (list): List of probabilistic mapping objects
        """
        if z_lf_train.ndim != 3:
            raise IndexError("z_lf_train must be a 3d tensor!")

        probabilistic_mapping_obj_lst = []
        for z_lf, y_hf in zip(z_lf_train.T, y_hf_train.T, strict=True):
            probabilistic_mapping_obj_lst.append(copy.deepcopy(approx))
            probabilistic_mapping_obj_lst[-1].setup(np.atleast_2d(z_lf), np.atleast_2d(y_hf).T)

        return z_lf_train, y_hf_train, probabilistic_mapping_obj_lst

    @staticmethod
    def instantiate_per_time_step(z_lf_train, y_hf_train, time_vec, coords_mat, approx):
        """Instantiate probabilistic mappings per time step.

        This means that one probabilistic mapping is build for all space locations
        combined but different mappings per time instance.

        Args:
            z_lf_train (np.array): Training inputs for probabilistic mapping.
                                   Rows: Samples, Columns: Coordinates
            y_hf_train (np.array): Training outputs for probabilistic mapping.
                                   Rows: Samples, Columns: Coordinates
            time_vec (np.array): Time vector of the experimental data.
            coords_mat (np.array): (Spatial) Coordinates of the experimental data.
            approx (Model): Probabilistic mapping

        Returns:
            z_lf_array (np.array): Input matrix in correct ordering
        """
        # determine the number of time steps and check coordinate compliance
        _, t_size = BmfiaInterface.check_coordinates_return_dimensions(
            z_lf_train, time_vec, coords_mat
        )

        # prepare LF and HF training array
        z_lf_array = BmfiaInterface.prepare_z_lf_for_time_steps(z_lf_train, t_size, coords_mat)

        y_hf_lst = np.array_split(y_hf_train.T, t_size)
        y_hf_array = np.array([y_hf.T.reshape(-1, 1) for y_hf in y_hf_lst])

        if z_lf_array.ndim != 3 or y_hf_array.ndim != 3:
            raise IndexError(
                "The input arrays for the probabilistic mapping must be 3-dimensional."
            )

        # loop over all time steps and instantiate the probabilistic mapping
        probabilistic_mapping_obj_lst = []
        for z_lf, y_hf in zip(z_lf_array, y_hf_array, strict=True):
            probabilistic_mapping_obj_lst.append(copy.deepcopy(approx))
            probabilistic_mapping_obj_lst[-1].setup(z_lf, y_hf)

        return z_lf_array, y_hf_array, probabilistic_mapping_obj_lst

    @staticmethod
    def evaluate_per_coordinate(
        z_lf, support, probabilistic_mapping_obj_lst, _time_vec, _coords_mat
    ):
        r"""Map the lf features to a probabilistic response for the hf model.

        Here a probabilistic mapping per coordinate combination (time and space)
        is evaluated.

        Calls the probabilistic mapping and predicts the mean and variance,
        respectively covariance, for the high-fidelity model, given the inputs
        *z_lf*.

        Args:
            z_lf (np.array): Low-fidelity feature vector that contains the corresponding Monte-Carlo
                             points on which the probabilistic mapping should be evaluated.
                             Dimensions: Rows: different multi-fidelity vector/points
                             (each row is one multi-fidelity point).
                             Columns: different model outputs/informative features.
            support (str): Support/variable for which we predict the mean and (co)variance. For
                           `support=f` the Gaussian process predicts w.r.t. the latent function
                           `f`. For the choice of `support=y` we predict w.r.t. to the
                           simulation/experimental output `y`,
            probabilistic_mapping_obj_lst (list): List of probabilistic mapping objects.
            _time_vec (np.array): Time vector of the experimental data.
            _coords_mat (np.array): (Spatial) Coordinates of the experimental data.

        Returns:
            mean (np.array): Vector of mean predictions
                             :math:`\mathbb{E}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*, \mathcal{D}_{f})]`
                             for the HF model given the low-fidelity feature input. Different HF
                             predictions per row. Each row corresponds to one multi-fidelity
                             input vector in :math:`\Omega_{y_{lf}\times\gamma_i}`.

            variance (np.array): Vector of variance predictions
                                 :math:`\mathbb{V}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,\mathcal{D}_{f})]`
                                 for the HF model given the low-fidelity feature input.
                                 Different HF predictions per row. Each row corresponds to one
                                 multi-fidelity input vector in
                                 :math:`\Omega_{y_{lf}\times\gamma_i}`.
        """
        mean_y_hf_given_z_lf = []
        var_y_hf_given_z_lf = []

        for z_test_per_coordinate, probabilistic_mapping_obj in zip(
            z_lf.T, probabilistic_mapping_obj_lst, strict=True
        ):
            if z_test_per_coordinate.ndim == 1:
                z_test_per_coordinate = np.atleast_2d(z_test_per_coordinate).T

            output = probabilistic_mapping_obj.predict(
                z_test_per_coordinate,
                support=support,
                gradient_bool=False,
            )

            mean_y_hf_given_z_lf.append(output["result"].squeeze())
            var_y_hf_given_z_lf.append(output["variance"].squeeze())

        mean = np.atleast_2d(np.array(mean_y_hf_given_z_lf)).T
        variance = np.atleast_2d(np.array(var_y_hf_given_z_lf)).T
        return mean, variance

    @staticmethod
    def evaluate_per_time_step(z_lf, support, probabilistic_mapping_obj_lst, time_vec, coords_mat):
        r"""Map the LF features to a probabilistic response for the hf model.

        Here a probabilistic mapping per time-step but for all locations combined
        is evaluated.

        Calls the probabilistic mapping and predicts the mean and variance,
        respectively covariance, for the high-fidelity model, given the inputs
        *z_lf*.

        Args:
            z_lf (np.array): Low-fidelity feature vector that contains the corresponding Monte-Carlo
                             points on which the probabilistic mapping should be evaluated.
                             Dimensions: Rows: different multi-fidelity vector/points
                             (each row is one multi-fidelity point).
                             Columns: different model outputs/informative features.
                             Note: z_lf might be a 3d tensor here
                             Dims z_lf: gamma_dim x num_samples x coord_dim
                             Dims z_lf.T: coord_dim x num_samples x gamma_dim --> iterate over
                             coord_dim (= combines space and time, in the order of batches
                             per time step)

            support (str): Support/variable for which we predict the mean and (co)variance. For
                            `support=f` the Gaussian process predicts w.r.t. the latent function
                            `f`. For the choice of `support=y` we predict w.r.t. to the
                            simulation/experimental output `y`,
            probabilistic_mapping_obj_lst (list): List of probabilistic mapping objects.
            time_vec (np.array): Vector of time coordinate points.
            coords_mat (np.array): Matrix of spatial coordinate points.

        Returns:
            mean (np.array): Vector of mean predictions
                             :math:`\mathbb{E}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*, \mathcal{D}_{f})]`
                             for the HF model given the low-fidelity feature input. Different HF
                             predictions per row. Each row corresponds to one multi-fidelity
                             input vector in :math:`\Omega_{y_{lf}\times\gamma_i}`.

            variance (np.array): Vector of variance predictions
                                 :math:`\mathbb{V}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,\mathcal{D}_{f})]`
                                 for the HF model given the low-fidelity feature input.
                                 Different HF predictions per row. Each row corresponds to one
                                 multi-fidelity input vector in
                                 :math:`\Omega_{y_{lf}\times\gamma_i}`.
        """
        # determine the number of time steps and check coordinate compliance
        num_coords, t_size = BmfiaInterface.check_coordinates_return_dimensions(
            z_lf, time_vec, coords_mat
        )
        z_lf_array = BmfiaInterface.prepare_z_lf_for_time_steps(z_lf, t_size, coords_mat)

        (
            mean,
            variance,
            _,
            _,
        ) = BmfiaInterface.iterate_over_time_steps(
            z_lf_array, support, num_coords, probabilistic_mapping_obj_lst, gradient_bool=False
        )

        return mean, variance

    @staticmethod
    def evaluate_and_gradient_per_coordinate(
        z_lf, support, probabilistic_mapping_obj_lst, _time_vec, _coords_mat
    ):
        r"""Evaluate probabilistic mapping and gradient for space point.

        Here a probabilistic mapping per coordinate combination (time and space)
        is evaluated along with the gradient of the underlying probabilistic mapping.

        Calls the probabilistic mapping and predicts the mean and variance,
        respectively covariance, for the high-fidelity model, given the inputs
        *z_lf* along with the gradients of the mean and the variance function of the
        underlying probabilistic mapping.

        Args:
            z_lf (np.array): Low-fidelity feature vector that contains the corresponding Monte-Carlo
                points on which the probabilistic mapping should be evaluated.
                Dimensions: Rows: different multi-fidelity vector/points
                (each row is one multi-fidelity point).
                Columns: different model outputs/informative features.
                Note: z_lf might be a 3d tensor here:
                Dims z_lf: gamma_dim x num_samples x coord_dim or
                Dims z_lf.T: coord_dim x num_samples x gamma_dim --> iterate over coord_dim.
            support (str): Support/variable for which we predict the mean and (co)variance. For
                `support=f`, the Gaussian process predicts w.r.t. the latent function
                `f`. For the choice of `support=y` we predict w.r.t. to the
                simulation/experimental output `y`.
            probabilistic_mapping_obj_lst (list): List of probabilistic mapping objects
            _time_vec (np.array): Vector of time points
            _coords_mat (np.array): Matrix of spatial coordinates

        Returns:
            mean (np.array): Vector of mean predictions
                :math:`\mathbb{E}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*, \mathcal{D}_{f})]`
                for the HF model given the low-fidelity feature input. Different HF
                predictions per row. Each row corresponds to one multi-fidelity
                input vector in :math:`\Omega_{y_{lf}\times\gamma_i}`.
            variance (np.array): Vector of variance predictions
                :math:`\mathbb{V}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,\mathcal{D}_{f})]`
                for the HF model given the low-fidelity feature input.
                Different HF predictions per row. Each row corresponds to one
                multi-fidelity input vector in :math:`\Omega_{y_{lf}\times\gamma_i}`.
            grad_mean (np.array): Gradient matrix for the mean prediction.
                Different HF predictions per row, and gradient vector entries per column.
            grad_variance (np.array): Gradient matrix for the mean prediction.
                Different HF predictions per row, and gradient vector entries per column.
        """
        mean_Y_HF_given_Z_LF = []
        var_Y_HF_given_Z_LF = []
        grad_mean = []
        grad_variance = []

        if len(probabilistic_mapping_obj_lst) != z_lf.T.shape[0]:
            raise IndexError(
                "The length of the list with probabilistic mapping objects "
                "must agree with the row numbers in Z_LF.T (coordinate dimension)! Abort..."
            )

        for z_test_per_coordinate, probabilistic_mapping_obj in zip(
            z_lf.T, probabilistic_mapping_obj_lst, strict=True
        ):
            if z_test_per_coordinate.ndim == 1:
                z_test_per_coordinate = np.atleast_2d(z_test_per_coordinate).T
            output = probabilistic_mapping_obj.predict(
                z_test_per_coordinate, support=support, gradient_bool=True
            )

            mean_Y_HF_given_Z_LF.append(output["result"].squeeze())
            var_Y_HF_given_Z_LF.append(output["variance"].squeeze())

            grad_mean.append(output["grad_mean"].squeeze().T)
            grad_variance.append(output["grad_var"].squeeze().T)

        mean = np.atleast_2d(np.array(mean_Y_HF_given_Z_LF)).T
        variance = np.atleast_2d(np.array(var_Y_HF_given_Z_LF)).T

        # only select the gradient w.r.t. the LF model
        grad_mean = np.array(grad_mean).T
        grad_variance = np.array(grad_variance).T

        if grad_mean.ndim == 3:
            grad_mean = grad_mean[:, 0]
            grad_variance = grad_variance[:, 0]

        return mean, variance, grad_mean, grad_variance

    @staticmethod
    def evaluate_and_gradient_per_time_step(
        z_lf, support, probabilistic_mapping_obj_lst, time_vec, coords_mat
    ):
        r"""Evaluate probabilistic mapping and gradient for time step.

        Here a probabilistic mapping per time-step but for all locations combined
        is evaluated along with the gradient of the mean and variance function of the
        underlying probabilistic mapping.

        Args:
            z_lf (np.array): Low-fidelity feature vector that contains the corresponding Monte Carlo
                             points on which the probabilistic mapping should be evaluated.
                             Dimensions: Rows: different multi-fidelity vector/points
                             (each row is one multi-fidelity point).
                             Columns: different model outputs/informative features.
                             Note: z_lf might be a 3d tensor here
                             Dims z_lf: gamma_dim x num_samples x coord_dim
                             Dims z_lf.T: coord_dim x num_samples x gamma_dim --> iterate over
                             coord_dim
            support (str): Support/variable for which we predict the mean and (co)variance. For
                            `support=f` the Gaussian process predicts w.r.t. the latent function
                            `f`. For the choice of `support=y` we predict w.r.t. to the
                            simulation/experimental output `y`,
            probabilistic_mapping_obj_lst (list): List of probabilistic mapping objects.
            time_vec (np.array): Time vector for which the probabilistic mapping is evaluated.
            coords_mat (np.array): Coordinates for which the probabilistic mapping is evaluated.

        Returns:
            mean (np.array): Vector of mean predictions
                :math:`\mathbb{E}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*, \mathcal{D}_{f})]`
                for the HF model given the low-fidelity feature input. Different HF
                predictions per row. Each row corresponds to one multi-fidelity
                input vector in :math:`\Omega_{y_{lf}\times\gamma_i}`.
            variance (np.array): Vector of variance predictions
                :math:`\mathbb{V}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,\mathcal{D}_{f})]`
                for the HF model given the low-fidelity feature input.
                Different HF predictions per row. Each row corresponds to one
                multi-fidelity input vector in
                :math:`\Omega_{y_{lf}\times\gamma_i}`.
            grad_mean (np.array): Gradient matrix for the mean prediction.
                Different HF predictions per row, and gradient vector entries per column
            grad_variance (np.array): Gradient matrix for the mean prediction.
                Different HF predictions per row, and gradient vector entries per column
        """
        # determine the number of time steps and check coordinate compliance
        num_coords, t_size = BmfiaInterface.check_coordinates_return_dimensions(
            z_lf, time_vec, coords_mat
        )

        z_lf_array = BmfiaInterface.prepare_z_lf_for_time_steps(z_lf, t_size, coords_mat)

        (
            mean,
            variance,
            grad_mean,
            grad_variance,
        ) = BmfiaInterface.iterate_over_time_steps(
            z_lf_array, support, num_coords, probabilistic_mapping_obj_lst, gradient_bool=True
        )

        # reshape arrays back
        grad_mean = grad_mean.swapaxes(1, 2)
        grad_variance = grad_variance.swapaxes(1, 2)

        return mean, variance, grad_mean, grad_variance

    @staticmethod
    def update_mappings_per_coordinate(
        probabilistic_mapping_obj_lst, z_lf_train, y_hf_train, _time_vec, _coords_mat
    ):
        """Update the probabilistic mappings per coordinate.

        Args:
            probabilistic_mapping_obj_lst (list): List of probabilistic mapping objects
            z_lf_train (np.array): Training inputs for probabilistic mapping.
            y_hf_train (np.array): Training outputs for probabilistic mapping.
            _time_vec (np.array): Time vector for which the probabilistic mapping is evaluated.
            _coords_mat (np.array): Coordinates for which the probabilistic mapping is evaluated.

        Returns:
            z_lf_train (np.array): Training inputs for probabilistic mapping.
            y_hf_train (np.array): Training outputs for probabilistic mapping.
            probabilistic_mapping_obj_lst (list): List of probabilistic mapping objects
        """
        if z_lf_train.ndim != 3:
            raise IndexError("z_lf_train must be a 3d tensor!")

        for probabilistic_model, z_lf, y_hf in zip(
            probabilistic_mapping_obj_lst, z_lf_train.T, y_hf_train.T, strict=True
        ):
            probabilistic_model.update_training_data(np.atleast_2d(z_lf), np.atleast_2d(y_hf))

        return z_lf_train, y_hf_train, probabilistic_mapping_obj_lst

    @staticmethod
    def update_mappings_per_time_step(
        probabilistic_mapping_obj_lst, z_lf_array, y_hf_array, time_vec, coords_mat
    ):
        """Update the probabilistic mappings per time step.

        Args:
            probabilistic_mapping_obj_lst (list): List of probabilistic mapping objects
            z_lf_array (np.array): Input matrix in correct ordering
            y_hf_array (np.array): Output matrix in correct ordering
            time_vec (np.array): Time vector for which the probabilistic mapping is evaluated.
            coords_mat (np.array): Coordinates for which the probabilistic mapping is evaluated.

        Returns:
            z_lf_train (np.array): Training inputs for probabilistic mapping.
            y_hf_train (np.array): Training outputs for probabilistic mapping.
            probabilistic_mapping_obj_lst (list): List of probabilistic mapping objects
        """
        # determine the number of time steps and check coordinate compliance
        _, t_size = BmfiaInterface.check_coordinates_return_dimensions(
            z_lf_array, time_vec, coords_mat
        )

        # prepare LF and HF training array
        z_lf_array = BmfiaInterface.prepare_z_lf_for_time_steps(z_lf_array, t_size, coords_mat)

        if y_hf_array.T.shape[0] != t_size:
            raise IndexError(
                "The number of time steps must be equal to the number of time steps in the "
                "high-fidelity model."
            )
        y_hf_lst = np.array_split(y_hf_array.T, t_size)
        y_hf_array = np.array([y_hf.T.reshape(-1, 1) for y_hf in y_hf_lst])

        if z_lf_array.ndim != 3 or y_hf_array.ndim != 3:
            raise IndexError(
                "The input arrays for the probabilistic mapping must be 3-dimensional."
            )

        for probabilistic_model, z_lf, y_hf in zip(
            probabilistic_mapping_obj_lst, z_lf_array, y_hf_array, strict=True
        ):
            probabilistic_model.update_training_data(np.atleast_2d(z_lf), np.atleast_2d(y_hf).T)

        return z_lf_array, y_hf_array, probabilistic_mapping_obj_lst

    valid_probabilistic_mappings_configurations = {
        "per_coordinate": (
            instantiate_per_coordinate,
            evaluate_per_coordinate,
            evaluate_and_gradient_per_coordinate,
            update_mappings_per_coordinate,
        ),
        "per_time_step": (
            instantiate_per_time_step,
            evaluate_per_time_step,
            evaluate_and_gradient_per_time_step,
            update_mappings_per_time_step,
        ),
    }

    @log_init_args
    def __init__(
        self,
        num_processors_multi_processing=1,
        probabilistic_mapping_type="per_coordinate",
    ):
        """Instantiate a BMFIA interface.

        Args:
            num_processors_multi_processing (int): Number of processors that should be used in the
                                                   multi-processing pool.
            probabilistic_mapping_type (str): Configured method to instantiate the  probabilistic
                                              mapping objects
        """
        # instantiate probabilistic mapping objects
        (
            instantiate_probabilistic_mappings,
            evaluate_method,
            evaluate_and_gradient_method,
            update_mappings_method,
        ) = get_option(
            BmfiaInterface.valid_probabilistic_mappings_configurations, probabilistic_mapping_type
        )

        self.instantiate_probabilistic_mappings = instantiate_probabilistic_mappings
        self.num_processors_multi_processing = num_processors_multi_processing
        self.probabilistic_mapping_obj_lst = []
        self.evaluate_method = evaluate_method
        self.evaluate_and_gradient_method = evaluate_and_gradient_method
        self.update_mappings_method = update_mappings_method
        self.coord_labels = None
        self.time_vec = None
        self.coords_mat = None

    def evaluate(self, samples, support="y"):
        r"""Map the lf-features to a probabilistic response for the hf model.

        Calls the probabilistic mapping and predicts the mean and variance,
        respectively covariance, for the high-fidelity model, given the inputs
        *z_lf* (called samples here).

        Args:
            samples (np.array): Low-fidelity feature vector *z_lf* that contains the corresponding
                                Monte-Carlo points, on which the probabilistic mapping should
                                be evaluated.
                                Dimensions:

                                * Rows: different multi-fidelity vector/points (each row is one
                                  multi-fidelity point)
                                * Columns: different model outputs/informative features
            support (str): Support/variable for which we predict the mean and (co)variance. For
                            *support=f*  the Gaussian process predicts w.r.t. the latent function
                            *f*. For the choice of *support=y* we predict w.r.t. the
                            simulation/experimental output *y*

        Returns:
            mean (np.array): Vector of mean predictions
                                             :math:`\mathbb{E}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,
                                             \mathcal{D}_{f})]` for the HF model given the
                                             low-fidelity feature input. Different HF predictions
                                             per row. Each row corresponds to one multi-fidelity
                                             input vector in
                                             :math:`\Omega_{y_{lf}\times\gamma_i}`.

            variance (np.array): Vector of variance predictions :math:`\mathbb{V}_{
                                            f^*}[p(y_{HF}^*|f^*,z_{LF}^*,\mathcal{D}_{f})]` for the
                                            HF model given the low-fidelity feature input.
                                            Different HF predictions
                                            per row. Each row corresponds to one multi-fidelity
                                            input vector in
                                            :math:`\Omega_{y_{lf}\times\gamma_i}`.
        """
        mean, variance = self.evaluate_method(
            samples, support, self.probabilistic_mapping_obj_lst, self.time_vec, self.coords_mat
        )
        return mean, variance

    def evaluate_and_gradient(self, z_lf, support="y"):
        r"""Evaluate probabilistic mapping and its gradient.

        Calls the probabilistic mapping and predicts the mean and variance,
        respectively covariance, for the high-fidelity model, given the inputs
        *z_lf* as well as its gradient w.r.t. the low-fidelity model.

        Args:
            z_lf (np.array): Low-fidelity feature vector that contains the corresponding Monte-Carlo
                             points, on which the probabilistic mapping should be evaluated.
                             Dimensions:

                             * Rows: different multi-fidelity vector/points (each row is one
                               multi-fidelity point)
                             * Columns: different model outputs/informative features
            support (str): Support/variable for which we predict the mean and (co)variance. For
                            *support=f*  the Gaussian process predicts w.r.t. the latent function
                            *f*. For the choice of *support=y* we predict w.r.t. the
                            simulation/experimental output *y*

        Returns:
            mean (np.array): Vector of mean predictions
                             :math:`\mathbb{E}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,
                             \mathcal{D}_{f})]` for the HF model given the
                             low-fidelity feature input. Different HF predictions
                             per row. Each row corresponds to one multi-fidelity
                             input vector in :math:`\Omega_{y_{lf}\times\gamma_i}`.

            variance (np.array): Vector of variance predictions
                                 :math:`\mathbb{V}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,\mathcal{D}_{f})]`
                                 for the HF model given the low-fidelity feature input.
                                 Different HF predictions per row. Each row corresponds to one
                                 multi-fidelity input vector in
                                 :math:`\Omega_{y_{lf}\times\gamma_i}`.

            grad_mean (np.array, optional): Gradient matrix for the mean prediction.
                                            Different HF predictions per row, and gradient
                                            vector entries per column
            grad_variance (np.array, optional): Gradient matrix for the mean prediction.
                                                Different HF predictions per row, and gradient
                                                vector entries per column
        """
        mean, variance, grad_mean, grad_variance = self.evaluate_and_gradient_method(
            z_lf, support, self.probabilistic_mapping_obj_lst, self.time_vec, self.coords_mat
        )
        return mean, variance, grad_mean, grad_variance

    def build_approximation(
        self, z_lf_train, y_hf_train, approx, coord_labels, time_vec, coords_mat
    ):
        r"""Build the probabilistic regression models.

        Build and train the probabilistic mapping objects based on the
        training inputs :math:`\mathcal{D}_f={Y_{HF},Z_{LF}}` per coordinate
        point/measurement point in the inverse problem.

        Args:
            z_lf_train (np.array): Training inputs for probabilistic mapping.

                *  Rows: Samples
                *  Columns: Coordinates

            y_hf_train (np.array): Training outputs for probabilistic mapping.

                *  Rows: Samples
                *  Columns: Coordinates

            approx (Model): Probabilistic mapping in configuration dictionary.
            coord_labels (list): List of coordinate labels.
            time_vec (np.array): Time vector of the experimental data.
            coords_mat (np.array): (Spatial) Coordinates of the experimental data.
        """
        # initialize further attributes
        self.coord_labels = coord_labels
        self.time_vec = time_vec
        self.coords_mat = coords_mat

        # check input dimensions
        if z_lf_train.ndim != 3:
            raise IndexError("z_lf_train must be a 3d tensor!")

        if z_lf_train.T.shape[0] != y_hf_train.T.shape[0]:
            raise IndexError("Dimension of z_lf_train and y_hf_train do not agree!")

        # Instantiate list of probabilistic mapping
        (
            z_lf_train,
            y_hf_train,
            self.probabilistic_mapping_obj_lst,
        ) = self.instantiate_probabilistic_mappings(
            z_lf_train, y_hf_train, self.time_vec, self.coords_mat, approx
        )

        if self.num_processors_multi_processing > 1:
            # Conduct training of probabilistic mappings in parallel
            num_coords = z_lf_train.T.shape[2]
            optimized_mapping_states_lst = BmfiaInterface.train_probabilistic_mappings_in_parallel(
                num_coords, self.num_processors_multi_processing, self.probabilistic_mapping_obj_lst
            )
            # Set the optimized hyper-parameters for probabilistic regression model
            self.set_optimized_state_of_probabilistic_mappings(optimized_mapping_states_lst)
        else:
            # conduct serial training of probabilistic mapping
            self._train_probabilistic_mappings_serial()

    def _train_probabilistic_mappings_serial(self):
        """Train the probabilistic models in series."""
        t_s = time.time()

        num_map = len(self.probabilistic_mapping_obj_lst)
        for num, probabilistic_model in enumerate(self.probabilistic_mapping_obj_lst):
            _logger.info("Starting training of probabilistic mapping %d of %d...", num + 1, num_map)
            probabilistic_model.train()
            _logger.info(
                "Finished training of probabilistic mapping '%d' of '%d'!\n",
                num + 1,
                num_map,
            )

        t_e = time.time()
        t_total = t_e - t_s
        _logger.info("Total time for training of all probabilistic mappings: %d s", t_total)

    @staticmethod
    def train_probabilistic_mappings_in_parallel(
        num_coords, num_processors_multi_processing, probabilistic_mapping_obj_lst
    ):
        """Train the probabilistic regression models in parallel.

        We use a multi-processing tool to conduct the training of the
        probabilistic regression models in parallel.

        Args:
            num_coords (int): number of coordinates in the current inverse problem
            num_processors_multi_processing (int): number of processors to use for
                                                    the multi-processing pool
            probabilistic_mapping_obj_lst (list): List of probabilistic mapping objects

        Returns:
            optimized_mapping_states_lst (lst): List of updated / trained states for
                                                the probabilistic regression models.
        """
        # prepare parallel pool for training
        num_processors_available = mp.cpu_count()

        if (
            num_processors_multi_processing is None
            or num_processors_multi_processing == 0
            or not isinstance(num_processors_multi_processing, int)
        ):
            raise RuntimeError(
                "You have to specify a valid number of processors for the "
                "multi-processing pool in the BMFIA-interface! You specified "
                f"{num_processors_multi_processing}, number of processors, which is not a "
                "valid choice. A valid choice is an integer between 0 and the "
                "number of available processors on your resource. Abort..."
            )
        if num_processors_multi_processing > num_processors_available:
            raise RuntimeError(
                f"You specified {num_processors_multi_processing} for the multi-processing "
                f"pool but the system only has {num_processors_available}! "
                f"Please specify a number of processors that is smaller or equal "
                f"to {num_processors_available}! Abort..."
            )

        num_processors_for_job = min(num_processors_multi_processing, num_coords)

        _logger.info(
            "Run generation and training of probabilistic surrogates in parallel on %d processors.",
            num_processors_for_job,
        )

        # Init multi-processing pool
        with get_context("spawn").Pool(processes=num_processors_for_job) as pool:
            # Actual parallel training of the models
            optimized_mapping_states_lst = list(
                tqdm.tqdm(
                    pool.imap(BmfiaInterface.optimize_hyper_params, probabilistic_mapping_obj_lst),
                    total=len(probabilistic_mapping_obj_lst),
                )
            )

        return optimized_mapping_states_lst

    def set_optimized_state_of_probabilistic_mappings(self, optimized_mapping_states_lst):
        """Set the new states of the trained probabilistic mappings.

        Args:
            optimized_mapping_states_lst (lst): List of updated / trained states for
                                                the probabilistic regression models.
        """
        for optimized_state_dict, mapping in zip(
            optimized_mapping_states_lst, self.probabilistic_mapping_obj_lst, strict=True
        ):
            mapping.set_state(optimized_state_dict)

    @staticmethod
    def optimize_hyper_params(probabilistic_mapping):
        """Train one probabilistic surrogate model.

        Args:
            probabilistic_mapping (obj): Instantiated but untrained probabilistic mapping

        Returns:
            optimized_mapping_state_dict (dict): Dictionary with optimized state of the trained
                                                 probabilistic regression model
        """
        probabilistic_mapping.train()
        optimized_mapping_state_dict = probabilistic_mapping.get_state()
        return optimized_mapping_state_dict

    @staticmethod
    def check_coordinates_return_dimensions(z_lf, time_vec, coords_mat):
        """Check the compliance of Z_LF with the coordinates and time vector.

        Args:
            z_lf (np.array): Low fidelity coordinates
            time_vec (np.array): Coordinates time vector
            coords_mat (np.array): (spatial) coordinates matrix

        Returns:
            num_coords (int): number of coordinates in the current inverse problem
            t_size (int): number of time steps in the current inverse problem
        """
        # get number of time steps
        t_size = time_vec.size if np.any(time_vec) else 1
        # get number of coordinates (spatial)
        if coords_mat.ndim == 1:
            coords_mat = np.atleast_2d(coords_mat)
        num_coords = coords_mat.shape[0]

        # get number of coordinates in Z_LF after splitting along time dimension
        num_coords_in_z = z_lf.T.shape[0] / t_size
        if num_coords != num_coords_in_z:
            raise ValueError(
                "Number of coordinates in Z_LF must agree with number of coordinates in "
                "self.coords_mat. "
                f"num_coords={num_coords}, num_coords_in_z={num_coords_in_z}"
            )
        return num_coords, t_size

    @staticmethod
    def prepare_z_lf_for_time_steps(z_lf, t_size, coords_mat):
        """Prepare the low fidelity coordinates for the time steps.

        Args:
            z_lf (np.array): Low fidelity variables before reshaping
            t_size (int): number of time steps in the current inverse problem
            coords_mat (np.array): (spatial) coordinates matrix

        Returns:
            z_lf_out (np.array): Low fidelity variables after reshaping
        """
        # vertically expand coordinates to match number of samples
        num_samples = z_lf.T.shape[1]  # number of samples
        coords_vec = np.tile(coords_mat, (num_samples, 1))

        # split Z_LF along time dimension
        z_lf_time_splitted = np.array_split(z_lf.T.squeeze(), t_size)

        # one list element (per time step) that is build below has the following
        # dimensions: num_samples x (ylf, gamma, coords)
        z_lf_lst = []
        for z_lf_time in z_lf_time_splitted:
            # z_lf has dim coords x num_samples and z_lf.T has dim num_samples x coords
            # reshaping in C order will not break the order of the coordinates
            z_lf_reshaped = z_lf_time.T.reshape(-1, 1)
            z_lf_stacked = np.hstack((z_lf_reshaped, coords_vec))
            z_lf_lst.append(z_lf_stacked)
        z_lf_out = np.array(z_lf_lst)

        return z_lf_out

    @staticmethod
    def iterate_over_time_steps(
        z_lf_array, support, num_coords, probabilistic_mapping_obj_lst, gradient_bool=None
    ):
        """Iterate and arrange data over different time steps.

        Evaluate and iterate over the regression models for the time
        steps.

        Args:
            z_lf_array (np.array): Low fidelity variables after reshaping
            support (np.array): Support of the random variable
            num_coords (int): Number of coordinates
            probabilistic_mapping_obj_lst (lst): List of probabilistic mapping objects
            gradient_bool (bool, optional): If True, the gradient of the mean and variance is
                                            returned

        Returns:
            mean (np.array): Mean of the high fidelity variables
            variance (np.array): Variance of the high fidelity variables
            grad_mean (np.array): Gradient of the mean
            grad_variance (np.array): Gradient of the variance
        """
        mean_y_hf_given_z_lf = []
        var_y_hf_given_z_lf = []
        grad_mean = []
        grad_variance = []

        if z_lf_array.ndim != 3:
            raise ValueError("Dimension of z_lf_array must be 3.")

        for z_test_per_time_step, probabilistic_mapping_obj in zip(
            z_lf_array, probabilistic_mapping_obj_lst, strict=True
        ):
            output = probabilistic_mapping_obj.predict(
                z_test_per_time_step, support=support, gradient_bool=gradient_bool
            )
            mean_y_hf_given_z_lf.append(output["result"].flatten())
            var_y_hf_given_z_lf.append(output["variance"].flatten())

            if gradient_bool:
                grad_mean.append(output["grad_mean"])
                grad_variance.append(output["grad_var"])

        # convert list to arrays and reshape arrays back
        mean = np.array(mean_y_hf_given_z_lf).T
        variance = np.array(var_y_hf_given_z_lf).T

        # reshape back to original dimensions in C order keeps the order of the coordinates
        # intact row wise, order now is: (n_time_steps x num_samples) x num_coords
        mean = mean.reshape(-1, num_coords)
        variance = variance.reshape(-1, num_coords)

        if gradient_bool:
            # list over different time steps; each element has shape
            # (num_coords * samples) x dim_z_lf
            grad_mean = np.array(grad_mean).reshape(-1, z_lf_array.shape[2])
            grad_variance = np.array(grad_variance).reshape(-1, z_lf_array.shape[2])

            # split first dimension of arrays along num_samples
            num_samples = int(grad_mean.shape[0] / num_coords)
            grad_mean_lst = np.split(grad_mean, num_samples, axis=0)
            grad_var_lst = np.split(grad_variance, num_samples, axis=0)

            # stack arrays per sample in new dimension
            grad_mean = np.swapaxes(np.stack(grad_mean_lst, axis=0), 1, 2)
            grad_variance = np.swapaxes(np.stack(grad_var_lst, axis=0), 1, 2)

        return mean, variance, grad_mean, grad_variance
