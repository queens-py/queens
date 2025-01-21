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
"""Iterator for Bayesian multi-fidelity inverse analysis."""

# pylint: disable=invalid-name
import logging

import numpy as np

from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.sobol_sequence import sample_sobol_sequence

_logger = logging.getLogger(__name__)


class BMFIAIterator(Iterator):
    """Bayesian multi-fidelity inverse analysis iterator.

    Iterator for Bayesian multi-fidelity inverse analysis. Here, we build
    the multi-fidelity probabilistic surrogate, determine optimal training
    points *X_train* and evaluate the low- and high-fidelity model for these
    training inputs, to yield *Y_LF_train* and *Y_HF_train* training data. The
    actual inverse problem is not solved or iterated in this module but instead
    we iterate over the training data to approximate the probabilistic mapping
    *p(yhf|ylf)*.

    Attributes:
        X_train (np.array): Input training matrix for HF and LF model.
        Y_LF_train (np.array): Corresponding LF model response to *X_train* input.
        Y_HF_train (np.array): Corresponding HF model response to *X_train* input.
        Z_train (np.array): Corresponding LF informative features to *X_train* input.
        features_config (str): Type of feature selection method.
        hf_model (obj): High-fidelity model object.
        lf_model (obj): Low-fidelity model object.
        coords_experimental_data (np.array): Coordinates of the experimental data.
        time_vec (np.array): Time vector of experimental observations.
        y_obs_vec (np.array): Output data of experimental observations.
        x_cols (list): List of columns for features taken from input variables.
        num_features (int): Number of features to be selected.
        coord_cols (list): List of columns for coordinates taken from input variables.

    Returns:
       BMFIAIterator (obj): Instance of the BMFIAIterator
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        global_settings,
        features_config,
        hf_model,
        lf_model,
        initial_design,
        X_cols=None,
        num_features=None,
        coord_cols=None,
    ):
        """Instantiate the BMFIAIterator object.

        Args:
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            features_config (str): Type of feature selection method.
            hf_model (obj): High-fidelity model object.
            lf_model (obj): Low-fidelity model object.
            initial_design (dict): Dictionary describing initial design.
            X_cols (list, opt): List of columns for features taken from input variables.
            num_features (int, opt): Number of features to be selected.
            coord_cols (list, opt): List of columns for coordinates taken from input variables.
        """
        super().__init__(None, parameters, global_settings)  # Input prescribed by iterator.py

        # ---------- calculate the initial training samples via classmethods ----------
        x_train = self.calculate_initial_x_train(initial_design, parameters)

        self.X_train = x_train
        self.Y_LF_train = None
        self.Y_HF_train = None
        self.Z_train = None
        self.features_config = features_config
        self.hf_model = hf_model
        self.lf_model = lf_model
        self.coords_experimental_data = None
        self.time_vec = None
        self.y_obs_vec = None
        self.x_cols = X_cols
        self.num_features = num_features
        self.coord_cols = coord_cols

    @classmethod
    def calculate_initial_x_train(cls, initial_design_dict, parameters):
        """Optimal training data set for probabilistic model.

        Based on the selected design method, determine the optimal set of
        input points X_train to run the HF and the LF model on for the
        construction of the probabilistic surrogate.

        Args:
            initial_design_dict (dict): Dictionary with description of initial design.
            model (obj): A model object on which the calculation is performed (only needed for
                         interfaces here. The model is not evaluated here)
            parameters (obj): Parameters object

        Returns:
            x_train (np.array): Optimal training input samples
        """
        run_design_method = cls.get_design_method(initial_design_dict)
        x_train = run_design_method(initial_design_dict, parameters)
        return x_train

    @classmethod
    def get_design_method(cls, initial_design_dict):
        """Get the design method for initial training data.

        Select the method for the generation of the initial training data
        for the probabilistic regression model.

        Args:
            initial_design_dict (dict): Dictionary with description of initial design.

        Returns:
            run_design_method (obj): Design method for selecting the HF training set
        """
        # check correct inputs
        assert isinstance(
            initial_design_dict, dict
        ), "Input argument 'initial_design_dict' must be of type 'dict'! Abort..."

        assert (
            "type" in initial_design_dict.keys()
        ), "No key 'type' found in 'initial_design_dict'. Abort..."

        # choose design method
        if initial_design_dict["type"] == "random":
            run_design_method = cls.random_design
        elif initial_design_dict["type"] == "sobol":
            run_design_method = cls._sobol_design
        else:
            raise NotImplementedError(
                "The design type you chose for selecting training data is not valid! "
                f"You chose {initial_design_dict['type']} but the only valid options "
                "is 'random'!"
            )

        return run_design_method

    @staticmethod
    def random_design(initial_design_dict, parameters):
        """Generate a uniformly random design strategy.

        Get a random initial design using the Monte-Carlo sampler with a uniform distribution.

        Args:
            initial_design_dict (dict): Dictionary with description of initial design.
            model (obj): A model object on which the calculation is performed (only needed for
                         interfaces here. The model is not evaluated here)
            parameters (obj): Parameters object

        Returns:
            x_train (np.array): Optimal training input samples
        """
        seed = initial_design_dict["seed"]
        num_samples = initial_design_dict["num_HF_eval"]
        np.random.seed(seed)
        x_train = parameters.draw_samples(num_samples)
        return x_train

    @staticmethod
    def _sobol_design(initial_design_dict, parameters):
        """Generate  quasi random design using the Sobol sequence.

        Args:
            initial_design_dict (dict): Dictionary with description of initial design.
            model (obj): A model object on which the calculation is performed (only needed for
                         interfaces here. The model is not evaluated here)
            parameters (obj): Parameters object

        Returns:
            x_train (np.array): Training input samples from Sobol sequence
        """
        x_train = sample_sobol_sequence(
            dimension=parameters.num_parameters,
            number_of_samples=initial_design_dict["num_HF_eval"],
            parameters=parameters,
            randomize=False,
            seed=initial_design_dict["seed"],
        )
        return x_train

    # ----------- main methods of the object form here ----------------------------------------
    def core_run(self):
        """Trigger main or core run of the BMFIA iterator.

        It summarizes the actual evaluation of the HF and LF models for these data and the
        determination of LF informative features.

        Returns:
            Z_train (np.array): Matrix with low-fidelity feature training data
            Y_HF_train (np.array): Matrix with HF training data
        """
        # ----- build model on training points and evaluate it -----------------------
        self.eval_model()

        # ----- Set the feature strategy of the probabilistic mapping (select gammas)
        self.Z_train = self.set_feature_strategy(
            self.Y_LF_train, self.X_train, self.coords_experimental_data[: self.Y_LF_train.shape[0]]
        )

        return self.Z_train, self.Y_HF_train

    def expand_training_data(self, additional_x_train, additional_y_lf_train=None):
        """Update or expand the training data.

        Data is appended by an additional input/output vector of data.

        Args:
            additional_x_train (np.array): Additional input vector
            additional_y_lf_train (np.array, optional): Additional LF model response corresponding
                                                        to additional input vector. Default to None

        Returns:
            Z_train (np.array): Matrix with low-fidelity feature training data
            Y_HF_train (np.array): Matrix with HF training data
        """
        if additional_y_lf_train is None:
            _logger.info("Starting to compute additional Y_LF_train...")
            num_coords = self.coords_experimental_data.shape[0]
            additional_y_lf_train = self.lf_model.evaluate(additional_x_train)["result"].reshape(
                -1, num_coords
            )
            _logger.info("Additional Y_LF_train were successfully computed!")

        _logger.info("Starting to compute additional Y_LF_train...")
        additional_y_hf_train = self.hf_model.evaluate(additional_x_train)["result"].reshape(
            -1, num_coords
        )
        _logger.info("Additional Y_HF_train were successfully computed!")

        self.X_train = np.vstack((self.X_train, additional_x_train))
        self.Y_LF_train = np.vstack((self.Y_LF_train, additional_y_lf_train))
        self.Y_HF_train = np.vstack((self.Y_HF_train, additional_y_hf_train))
        _logger.info("Training data was successfully expanded!")

        self.Z_train = self.set_feature_strategy(
            self.Y_LF_train, self.X_train, self.coords_experimental_data[: self.Y_LF_train.shape[0]]
        )

        return self.Z_train, self.Y_HF_train

    def evaluate_LF_model_for_X_train(self):
        """Evaluate the low-fidelity model for the X_train input data-set."""
        # reshape the scalar output by the coordinate dimension
        num_coords = self.coords_experimental_data.shape[0]
        self.Y_LF_train = self.lf_model.evaluate(self.X_train)["result"].reshape(-1, num_coords)

    def evaluate_HF_model_for_X_train(self):
        """Evaluate the high-fidelity model for the X_train input data-set."""
        # reshape the scalar output by the coordinate dimension
        num_coords = self.coords_experimental_data.shape[0]
        self.Y_HF_train = self.hf_model.evaluate(self.X_train)["result"].reshape(-1, num_coords)

    def set_feature_strategy(self, y_lf_mat, x_mat, coords_mat):
        """Get the low-fidelity feature matrix.

        Compose the low-fidelity feature matrix that consists of the low-
        fidelity model outputs and the low-fidelity informative features.

            y_lf_mat (np.array): Low-fidelity output matrix with row-wise model realizations.
                                 Columns are different dimensions of the output.
            x_mat (np.array): Input matrix for the simulation model with row-wise input points,
                              and colum-wise variable dimensions.
            coords_mat (np.array): Coordinate matrix for the observations with row-wise coordinate
                                   points and different dimensions per column.

        Returns:
            z_mat (np.array): Extended low-fidelity matrix containing
                              informative feature dimensions. Every row is one data point with
                              dimensions per column.
        """
        # check dimensions of the input
        assert (
            y_lf_mat.ndim == 2
        ), f"Dimension of y_lf_mat must be 2 but you provided dim={y_lf_mat.ndim}. Abort..."
        assert (
            x_mat.ndim == 2
        ), f"Dimension of x_mat must be 2 but you provided dim={x_mat.ndim}. Abort..."
        assert (
            coords_mat.ndim == 2
        ), f"Dimension of coords_mat must be 2 but you provided dim={coords_mat.ndim}. Abort..."

        feature_dict = {
            "man_features": self._get_man_features,
            "opt_features": self._get_opt_features,
            "coord_features": self._get_coord_features,
            "no_features": self._get_no_features,
            "time_features": self._get_time_features,
        }
        try:
            feature_fun = feature_dict.get(self.features_config, None)
        except KeyError as my_error:
            raise KeyError(
                "The key 'features_config' was not available in the dictionary "
                "'settings_probab_mapping'!"
            ) from my_error

        if feature_fun:
            z_mat = feature_fun(x_mat, y_lf_mat, coords_mat)
        else:
            raise ValueError(
                "Feature space method specified in 'features_config' is not valid! "
                f"You provided: {self.features_config} "
                f"but valid options are: {feature_dict.keys()}."
            )

        return z_mat

    def _get_man_features(self, x_mat, y_lf_mat, _):
        """Get the low-fidelity feature matrix with manual features.

        Args:
            x_mat (np.array): Input matrix for the simulation model with row-wise input points,
                              and colum-wise variable dimensions.
            y_lf_mat (np.array): Low-fidelity output matrix with row-wise model realizations.
                                 Columns are different dimensions of the output.

        Returns:
            z_mat (np.array): Extended low-fidelity matrix containing
                              informative feature dimensions. Every row is one data point with
                              dimensions per column.
        """
        try:
            idx_lst = self.x_cols
            # Catch wrong data type
            assert isinstance(idx_lst, list), "Entries of X_cols must be in list format! Abort..."
            # Catch empty list
            assert (
                idx_lst != []
            ), "The index list for selection of manual features must not be empty!, Abort..."
            gamma_mat = x_mat[:, idx_lst]
            assert (
                gamma_mat.shape[0] == y_lf_mat.shape[0]
            ), "Dimensions of gamma_mat and y_lf_mat do not agree! Abort..."
            z_lst = []
            for y_per_coordinate in y_lf_mat.T:
                z_lst.append(np.hstack([y_per_coordinate.reshape(-1, 1), gamma_mat]))

            z_mat = np.array(z_lst).squeeze().T

            assert z_mat.ndim == 3, "z_mat should be a 3d tensor if man features are used! Abort..."

        except KeyError as my_error:
            raise KeyError(
                "The settings for the probabilistic mapping need a key 'X_cols' if "
                "you want to use the feature configuration 'man_features'!"
            ) from my_error

        return z_mat

    def _get_opt_features(self, *_):
        """Get the low-fidelity feature matrix with optimal features.

        Returns:
            z_mat (np.array): Extended low-fidelity matrix containing
                              informative feature dimensions. Every row is one data point with
                              dimensions per column.
        """
        assert isinstance(
            self.num_features, int
        ), "Number of informative features must be an integer! Abort..."
        assert (
            self.num_features >= 1
        ), "Number of informative features must be an integer greater than one! Abort..."

        z_mat = self.update_probabilistic_mapping_with_features()
        return z_mat

    def _get_coord_features(self, _, y_lf_mat, coords_mat):
        """Get the low-fidelity feature matrix with coordinate features.

        Args:
            x_mat (np.array): Input matrix for the simulation model with row-wise input points,
                              and colum-wise variable dimensions.
            y_lf_mat (np.array): Low-fidelity output matrix with row-wise model realizations.
                                 Columns are different dimensions of the output.
            coords_mat (np.array): Coordinates matrix.

        Returns:
            z_mat (np.array): Extended low-fidelity matrix containing
                              informative feature dimensions. Every row is one data point with
                              dimensions per column.
        """
        try:
            idx_lst = self.coord_cols
            # Catch wrong data type
            assert isinstance(
                idx_lst, list
            ), "Entries of coord_cols must be in list format! Abort..."
            # Catch empty list
            assert (
                idx_lst != []
            ), "The index list for selection of manual features must not be empty!, Abort..."

            coord_feature = coords_mat[:, idx_lst]
            assert (
                coord_feature.shape[0] == y_lf_mat.shape[0]
            ), "Dimensions of coords_feature and y_lf_mat do not agree! Abort..."

            z_lst = []
            for y_per_coordinate in y_lf_mat.T:
                z_lst.append(np.hstack([y_per_coordinate.reshape(-1, 1), coord_feature]))

            z_mat = np.array(z_lst).squeeze().T
            assert (
                z_mat.ndim == 3
            ), "z_mat should be a 3d tensor if coord_features are used! Abort..."

        except KeyError as my_error:
            raise KeyError(
                "The settings for the probabilistic mapping need a key 'coord_cols' "
                "if you want to use the feature configuration 'coord_features'! Abort..."
            ) from my_error

        return z_mat

    def _get_no_features(self, _x_mat, y_lf_mat, __):
        """Get the low-fidelity feature matrix without additional features.

        Args:
            y_lf_mat (np.array): Low-fidelity output matrix with row-wise model realizations.
                                 Columns are different dimensions of the output.

        Returns:
            z_mat (np.array): Extended low-fidelity matrix containing
                              informative feature dimensions. Every row is one data point with
                              dimensions per column.
        """
        z_mat = y_lf_mat[None, :, :]
        return z_mat

    def _get_time_features(self, _, y_lf_mat, __):
        """Get the low-fidelity feature matrix with time features.

        Args:
            y_lf_mat (np.array): Low-fidelity output matrix with row-wise model realizations.
                                 Columns are different dimensions of the output.

        Returns:
            z_mat (np.array): Extended low-fidelity matrix containing
                              informative feature dimensions. Every row is one data point with
                              dimensions per column.
        """
        time_repeat = int(y_lf_mat.shape[0] / self.time_vec.size)
        time_vec = np.repeat(self.time_vec.reshape(-1, 1), repeats=time_repeat, axis=0)

        z_mat = np.hstack([y_lf_mat, time_vec])
        return z_mat

    def update_probabilistic_mapping_with_features(self):
        """Update multi-fidelity mapping with optimal lf-features."""
        raise NotImplementedError(
            "Optimal features for inverse problems are not yet implemented! Abort..."
        )

    def eval_model(self):
        """Evaluate the LF and HF model to for the training inputs.

        *X_train*.
        """
        # ---- run LF model on X_train (potentially we need to iterate over this and the previous
        # step to determine optimal X_train; for now just one sequence)
        _logger.info("-------------------------------------------------------------------")
        _logger.info("Starting to evaluate the low-fidelity model for training points....")
        _logger.info("-------------------------------------------------------------------")

        self.evaluate_LF_model_for_X_train()

        _logger.info("-------------------------------------------------------------------")
        _logger.info("Successfully calculated the low-fidelity training points!")
        _logger.info("-------------------------------------------------------------------")

        # ---- run HF model on X_train
        _logger.info("-------------------------------------------------------------------")
        _logger.info("Starting to evaluate the high-fidelity model for training points...")
        _logger.info("-------------------------------------------------------------------")

        self.evaluate_HF_model_for_X_train()

        _logger.info("-------------------------------------------------------------------")
        _logger.info("Successfully calculated the high-fidelity training points!")
        _logger.info("-------------------------------------------------------------------")
