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
"""Iterator for Bayesian multi-fidelity UQ."""

# pylint: disable=invalid-name
import logging

import numpy as np
import pandas as pd
from diversipy import psa_select

from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import process_outputs, write_results
from queens.visualization.bmfmc_visualization import BMFMCVisualization

_logger = logging.getLogger(__name__)


class BMFMCIterator(Iterator):
    r"""Iterator for the Bayesian multi-fidelity Monte-Carlo method.

    The iterator fulfills the following tasks:

    1.  Load the low-fidelity Monte Carlo data.
    2.  Based on low-fidelity data, calculate optimal *X_train* to evaluate the high-fidelity model.
    3.  Based on *X_train* return the corresponding *Y_LFs_train*.
    4.  Initialize the *BMFMC_model* (this is not the high-fidelity model but the probabilistic
        mapping) with *X_train* and *Y_LFs_train*. Note that the *BMFMC_model* itself triggers the
        computation of the high-fidelity training data *Y_HF_train*.
    5.  Trigger the evaluation of the *BMFMC_model*. Here evaluation refers to computing the
        posterior statistics of the high-fidelity model. This is implemented in the *BMFMC_model*
        itself.

    Attributes:
        model (obj): Instance of the BMFMCModel.
        result_description (dict): Dictionary containing settings for plotting and saving
                                   data/results.
        X_train (np.array): Corresponding input for the simulations that are used to train the
                            probabilistic mapping.
        Y_LFs_train (np.array): Outputs of the low-fidelity models that correspond to the training
                                inputs *X_train*.
        output (dict): Dictionary containing the output quantities:

            *  ``Z_mc``: Corresponding Monte-Carlo point in LF informative feature space
            *  ``m_f_mc``: Corresponding Monte-Carlo points of posterior mean of
                           the probabilistic mapping
            *  ``var_y_mc``: Corresponding Monte-Carlo posterior variance samples of the
                             probabilistic mapping
            *  ``y_pdf_support``: Support vector for QoI output distribution
            *  ``p_yhf_mean``: Vector containing mean function of HF output
                               posterior distribution
            *  ``p_yhf_var``: Vector containing posterior variance function of HF output
                              distribution
            *  ``p_yhf_mean_BMFMC``: Vector containing mean function of HF output
                                     posterior distribution calculated without informative
                                     features :math:`\boldsymbol{\gamma}`
            *  ``p_yhf_var_BMFMC``: Vector containing posterior variance function of HF
                                    output distribution calculated without informative
                                    features :math:`\boldsymbol{\gamma}`
            *  ``p_ylf_mc``: Vector with low-fidelity output distribution (kde from MC
                             data)
            *  ``p_yhf_mc``: Vector with reference HF output distribution (kde from MC
                             reference data)
            *  ``Z_train``: Corresponding training data in LF feature space
            *  ``Y_HF_train``: Outputs of the high-fidelity model that correspond to the
                               training inputs *X_train* such that :math:`Y_{HF}=y_{HF}(X)`
            *  ``X_train``: Corresponding input for the simulations that are used to
                            train the probabilistic mapping
        initial_design (dict): Dictionary containing settings for the selection strategy/initial
                               design of training points for the probabilistic mapping.
        visualization (BMFMCVisualization): Visualization object for BMFMC.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        result_description,
        initial_design,
        plotting_options=None,
    ):
        r"""Initialize BMFMC iterator object.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            result_description (dict): Dictionary containing settings for plotting and saving data/
                                       results
            initial_design (dict): Dictionary containing settings for the selection strategy/initial
                                   design of training points for the probabilistic mapping
            plotting_options (dict): Plotting options
        """
        super().__init__(model, parameters, global_settings)  # Input prescribed by iterator.py
        self.result_description = result_description
        self.X_train = None
        self.Y_LFs_train = None
        self.output = None
        self.initial_design = initial_design

        self.visualization = None
        if plotting_options:
            self.visualization = BMFMCVisualization.from_config_create(
                plotting_options, model.predictive_var, model.BMFMC_reference
            )

            self.model.visulization = self.visualization

    def core_run(self):
        r"""Main run of the BMFMCIterator.

        The BMFMCIterator covers the following points:

        1.  Reading the sampling data from the low-fidelity model in QUEENS.
        2.  Based on LF data, determine optimal *X_train* for which the high-fidelity model should
            be evaluated :math:`Y_{HF}=y_{HF}(X)`.
        3.  Update the BMFMCModel with the partial training data set of *X_train*, *Y_LF_train*
            (*Y_HF_train* is determined in the BMFMCModel).
        4.  Evaluate the BMFMCModel, which means that the posterior statistics
            :math:`\mathbb{E}_{f}\left[p(y_{HF}^*|f,\mathcal{D})\right]` and
            :math:`\mathbb{V}_{f}\left[p(y_{HF}^*|f,\mathcal{D})\right]` are computed based
            on the BMFMC algorithm, which is implemented in the BMFMCModel.
        """
        # -------- Load MC data from model -----------------------
        self.model.load_sampling_data()

        # ---- determine optimal input points for which HF should be simulated -------
        self.calculate_optimal_X_train()

        # ----- build model on training points and evaluate it -----------------------
        self.output = self.model.evaluate(self.X_train)

    def calculate_optimal_X_train(self):
        r"""Calculate the optimal model inputs *X_train*.

        Based on the low-fidelity sampling data, calculate the optimal model
        inputs *X_train*, on which the high-fidelity model should be evaluated to
        construct the training data set for BMFMC. This selection is performed
        based on the following method options:

        *   **random**: Divides the :math:`y_{LF}` data set in bins and selects training
            candidates randomly from each bin until :math:`n_{train}` is reached.
        *   **diverse subset**: Determine the most important input features :math:`\gamma_i`
            (this information is provided by the BMFMCModel), and find a space
            filling subset (diverse subset), given the LF sampling data with
            respect to the most important features :math:`\gamma_i`. The
            number of features to be considered can be set in the input file.

            **Remark**: An optimization routine for the optimal number of
            features to be considered will be added in the future.
        """
        design_method = self.initial_design.get("method")
        n_points = self.initial_design.get("num_HF_eval")
        run_design_method = self.get_design_method(design_method)
        run_design_method(n_points)

        # update the Bmfmc model variables
        # TODO: normally done by model.update_model_from_sample_batch() ! # pylint: disable=fixme
        self.model.X_train = self.X_train
        self.model.Y_LFs_train = self.Y_LFs_train

    def get_design_method(self, design_method):
        """Get the design method for selecting the HF data.

        Get the design method for selecting the HF data from the LF MC dataset.

        Args:
            design_method (str): Design method specified in input file

        Returns:
            run_design_method (obj): Design method for selecting the HF training set
        """
        self.model.calculate_extended_gammas()
        if design_method == "random":
            run_design_method = self.random_design

        elif design_method == "diverse_subset":
            run_design_method = self.diverse_subset_design

        else:
            raise NotImplementedError(
                f"You specified the non-valid method"
                f" '{self.initial_design['method']}'. This is not "
                f"implemented! The only valid methods are 'random' or "
                f"'diverse_subset'. Abort..."
            )

        return run_design_method

    def diverse_subset_design(self, n_points):
        """Calculate the HF training points based on psa_select.

        Calculate the HF training points from large LF-MC data-set based on
        the diverse subset strategy based on the psa_select method from **diversipy**.

        Args:
             n_points (int): Number of HF training points to be selected
        """
        design = self.model.gammas_ext_mc
        prelim_subset = psa_select(design, n_points, selection_target="max_dist_from_boundary")

        # return training data for outputs and corresponding inputs
        index = np.vstack(
            np.array(
                np.all((design[:, None, :] == prelim_subset[None, :, :]), axis=-1).nonzero()
            ).T.tolist()
        )[:, 0]

        # set the training data and indices in the BMFMC model and iterator
        self.model.training_indices = index
        self.X_train = self.model.X_mc[index, :]
        self.Y_LFs_train = self.model.Y_LFs_mc[index, :]

    def random_design(self, n_points):
        """Calculate the HF training points based on random selection.

        Calculate the HF training points from large LF-MC data-set based on random selection from
        bins over y_LF.

        Args:
            n_points (int): Number of HF training points to be selected
        """
        n_bins = self.initial_design.get("num_bins")
        np.random.seed(self.initial_design.get("seed"))
        master_LF = self.initial_design.get("master_LF", 0)
        ylf_min = np.amin(self.model.Y_LFs_mc)
        ylf_max = np.amax(self.model.Y_LFs_mc)
        break_points = np.linspace(ylf_min, ylf_max, n_bins + 1)

        bin_vec = pd.cut(
            self.model.Y_LFs_mc[:, master_LF],
            bins=break_points,
            labels=False,
            include_lowest=True,
            retbins=True,
        )[0]

        # Go through all bins and  randomly select points
        all_indices = np.arange(0, bin_vec.shape[0], dtype=int)
        training_indices = []
        for current_bin in range(n_bins):
            bin_indices = all_indices[bin_vec == current_bin]  # array of booleans
            training_indices.append(
                list(
                    np.random.choice(
                        bin_indices, min(n_points // n_bins, bin_indices.shape[0]), replace=False
                    )
                )
            )

        self.model.training_indices = np.array(
            [item for sublist in training_indices for item in sublist]
        )
        self.X_train = self.model.X_mc[self.model.training_indices]
        self.Y_LFs_train = self.model.Y_LFs_mc[self.model.training_indices]
        if self.model.training_indices.shape[0] < n_points:
            _logger.warning(
                "The chosen number of training points (%s) "
                "for the HF-LF mapping is smaller than specified (%s). "
                "Reduce the number of bins to increase the number of training points!",
                self.model.training_indices.shape[0],
                n_points,
            )

    # ------------------- BELOW JUST PLOTTING AND SAVING RESULTS ------------------
    def post_run(self):
        """Saving and plotting the results."""
        if self.visualization:
            self.visualization.plot_pdfs(self.output)
            self.visualization.plot_manifold(
                self.output, self.model.Y_LFs_mc, self.model.Y_HF_mc, self.model.Y_HF_train
            )

        if self.result_description["write_results"]:
            results = process_outputs(self.output, self.result_description)
            write_results(results, self.global_settings.result_file(".pickle"))
