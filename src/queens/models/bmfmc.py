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
"""Bayesian multi-fidelity Monte-Carlo model."""

# pylint: disable=invalid-name
import logging

import numpy as np
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import queens.utils.pdf_estimation as est
from queens.iterators.data import Data
from queens.models._model import Model
from queens.parameters.random_fields.karhunen_loeve import KarhunenLoeve as RandomField
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class BMFMC(Model):
    r"""Bayesian multi-fidelity Monte-Carlo model.

    Bayesian multi-fidelity Monte-Carlo model for uncertainty quantification, which is a
    probabilistic mapping between a high-fidelity simulation model (HF) and one or
    more low fidelity simulation models (LFs), respectively informative
    features :math:`(\gamma)` from the input space. Based on this mapping and the LF samples
    :math:`\mathcal{D}_{LF}^*=\\{Z^*,Y_{LF}^*\\}`, the BMFMC model computes the
    posterior statistics:

    :math:`\mathbb{E}_{f^*}\left[p(y_{HF}^*|f^*,D_f)\right]`, equation (14) in [1]
    and

    :math:`\mathbb{V}_{f^*}\left[p(y_{HF}^*|f^*,D_f)\right]`, equation (15) in [1]

    of the HF model's output uncertainty.

    The BMFMC model is designed to be constructed upon the sampling data of a LF model
    :math:`\mathcal{D}_{LF}^*=\\{Z^*,Y_{LF}^*\\}`, that is provided by pickle or csv-files,
    and offers then different options to obtain the HF data:

    1.  Provide HF training data in a file (**Attention:** The user needs to make sure that
        this training set is representative and its input :math:`Z` is a subset of the LF model
        sampling set:
        :math:`Z\subset Z^*`).
    2.  Run optimal HF simulations based on LF data. This requires a suitable simulation sub-model
        and sub-iterator for the HF model.
    3.  Provide HF sampling data (as a file), calculated with same :math:`Z^*` as the
        LF simulation runs and select optimal HF training set from this data. This is just helpful
        for scientific benchmarking when a ground-truth solution for the HF output uncertainty has
        been sampled before and this data exists anyway.

    Attributes:
        parameters (obj): Parameters object
        interface (obj): Interface object.
        features_config (str): strategy that will be used to calculate the low-fidelity features
                                   :math:`Z_{\text{LF}}`: `opt_features`, `no_features` or
                                   `man_features`
        X_cols (list): for `man_features`, columns of X-matrix that should be used as an informative
                       feature
        num_features (int): for `opt_features`, number of features to be used
        high_fidelity_model (obj): HF (simulation) model to run simulations that yield the HF
                                 training set :math:`\mathcal{D}_{HF}=\\{Z, Y_{HF}\\}`.
        X_train (np.array): Matrix of simulation inputs corresponding to the training
                                      data-set of the multi-fidelity mapping.
        Y_HF_train (np.array): Vector or matrix of HF output that correspond to training input
                               according to :math:`Y_{HF} = y_{HF}(X)`.
        Y_LFs_train (np.array): Output vector/matrix of one or multiple LF models that correspond to
                                the training input according to :math:`Y_{LF,i}=y_{LF,i}(X)`.
        X_mc (np.array): Matrix of simulation inputs that were used in the Monte-Carlo sampling
                         of the LF models. Each row is one input set for a simulation. Columns
                         refer to different *field_realizations* of the same variable.
        Y_LFs_mc (np.array): Output vector/matrix for the LF models that correspond to the *X_mc*
                            according to :math:`Y_{LF,i}^*=y_{LF,i}(X^*)`. At the moment *Y_LF_mc*
                            contains in one row scalar results for different LF models. (In the
                            future we will change the format to pandas dataframes to handle
                            vectorized/functional outputs for different models more elegantly).
        Y_HF_mc (np.array): (optional for benchmarking) Output vector/matrix for the HF model
                            that corresponds to the *X_mc* according to
                            :math:`Y_{HF}^*=y_{HF}(X^*)`.
        gammas_ext_mc (np.array): Matrix of extended low-fidelity informative features
                                :math:`\boldsymbol{\Gamma}^*` corresponding to Monte-Carlo
                                input :math:`X^*`.
        gammas_ext_train (np.array): Matrix of extended low-fidelity informative features
                                   :math:`\boldsymbol{\Gamma}` corresponding to the training
                                   input :math:`X`.
        Z_train (np.array): Training matrix of low-fidelity features according to
                            :math:`Z=\left[y_{LF,i}(X),\Gamma\right]`.
        Z_mc (np.array): Monte-Carlo matrix of low-fidelity features according to
                         :math:`Z^*=\left[y_{LF,i}(X^*),\Gamma^*\right]`.
        m_f_mc (np.array): Vector of posterior mean values of multi-fidelity mapping
                           corresponding to the Monte-Carlo input *Z_mc* according to
                           :math:`\mathrm{m}_{f^*}(Z^*)`.
        var_y_mc (np.array): Vector of posterior variance of multi-fidelity mapping
                             corresponding to the Monte-Carlo input *Z_mc* according to
                             :math:`\mathrm{m}_{f^*}(Z^*)`.
        p_yhf_mean (np.array): Vector that contains the mean approximation of the HF
                               output density defined on *y_hf_support*. The vector *p_yhf_mean* is
                               defined as:
                               :math:`\mathbb{E}_{f^*}\left[p(y_{HF}^*|f^*,D_f)\right]`
                               according to eq. (14) in [1].
        p_yhf_var (np.array): Vector that contains the variance approximation of the HF output
                              density defined on *y_hf_support*. The vector *p_yhf_var* is
                              defined as:
                              :math:`\mathbb{V}_{f^*}\left[p(y_{HF}^*|f^*,D_f)\right]`
                              according to eq. (15) in [1].
        predictive_var_bool (bool): Flag that determines whether *p_yhf_var* should be computed.
        p_yhf_mc (np.array): (optional) Monte-Carlo based kernel-density estimate of the HF output.
        p_ylf_mc (np.array): (optional) Kernel density estimate for LF model output.
                            **Note:** For BMFMC the explicit density is never required, only the
                            :math:`\mathcal{D}_{LF}` is used in the algorithm.
        no_features_comparison_bool (bool): If flag is *True*, the result will be compared to a
                                            prediction that used no LF input features.
        eigenfunc_random_fields (np.array): Matrix containing the discretized eigenfunctions of a
                                            underlying random field. **Note:** This is an
                                            intermediate solution and should be moved to the
                                            variables module! The current solution works so far
                                            only for one random field!
        eigenvals: Eigenvalues corresponding to the eigenfunctions.
        f_mean_train (np.array): Vector of predicted mean values of multi-fidelity mapping
                                 corresponding to the training input *Z_train* according to
                                 :math:`\mathrm{m}_{f^*}(Z)`.
        y_pdf_support (np.array): Support grid for HF output density :math:`p(y_{HF})`.
        lf_data_iterators (obj): Data iterators to load sampling data of low-fidelity models from a
                                 file.
        hf_data_iterator (obj):  Data iterator to load the benchmark sampling data from a HF model
                                 from a file (optional and only for scientific benchmark).
        training_indices (np.array): Vector with indices to select the training data subset from
                                     the larger data set of Monte-Carlo data.
        uncertain_parameters (obj): UncertainParameters object.
        visualization (BMFMCVisualization): BMFMC visualization object.

    References:
        [1] Nitzler, J., Biehler, J., Fehn, N., Koutsourelakis, P.-S. and Wall, W.A. (2020),
            "A Generalized Probabilistic Learning Approach for Multi-Fidelity Uncertainty
            Propagation in Complex Physical Simulations", arXiv:2001.02892
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        global_settings,
        probabilistic_mapping,
        features_config,
        predictive_var,
        BMFMC_reference,
        y_pdf_support_max,
        y_pdf_support_min,
        X_cols=None,
        num_features=None,
        hf_model=None,
        path_to_lf_mc_data=None,
        path_to_hf_mc_reference_data=None,
    ):
        r"""Initialize the Bayesian Multi-Fidelity Monte-Carlo (BMFMC) Model.

        Initializes the BMFMC model by setting up data iterators, configuring features, and
        preparing various components required for the BMFMC analysis.

        Args:
            parameters (Parameters): Parameters object.
            global_settings (GlobalSettings): Settings for the QUEENS experiment, including its name
                                              and output directory.
            probabilistic_mapping (obj): Instance of the probabilistic mapping object that models
                                         the probabilistic dependency between high-fidelity and
                                         low-fidelity models along with informative input features.
            features_config (str): Strategy used to calculate low-fidelity features
                                   :math:`Z_{\text{LF}}`. Possible values are `opt_features`
                                   (optimal features), `no_features`, or `man_features` (manual
                                   features).
            predictive_var (bool): Boolean flag that determines whether to compute the posterior
                                   variance :math:`\mathbb{V}_{f}\left[p(y_{HF}^*|f,\mathcal{
                                   D})\right]`.
            BMFMC_reference (bool): Boolean flag that triggers the BMFMC solution without
                                    informative features :math:`\boldsymbol{\gamma}` for
                                    comparison.
            y_pdf_support_max (float): Maximum value for the probability density function (PDF)
                                       support used in this analysis.
            y_pdf_support_min (float): Minimum value for the PDF support used in this analysis.
            X_cols (list, optional): For `man_features`, list of columns from the X-matrix to be
                                     used as informative features.
            num_features (int, optional): For `opt_features`, number of features to be used.
            hf_model (model, optional): High-fidelity model used to run simulations and generate
                                        training data.
            path_to_lf_mc_data (str or list of str, optional): Path(s) to low-fidelity MC data
                                                               files.
            path_to_hf_mc_reference_data (str, optional): Path to high-fidelity MC reference data
                                                          file.
        """
        # TODO the unlabeled treatment of raw data for eigenfunc_random_fields and input vars and # pylint: disable=fixme
        #  random fields is prone to errors and should be changed! The implementation should
        #  rather use the variable module and reconstruct the eigenfunctions of the random fields
        #  if not provided in the data field
        self.parameters = parameters
        interface = BmfmcInterface(probabilistic_mapping=probabilistic_mapping)

        if path_to_hf_mc_reference_data is not None:
            hf_data_iterator = Data(path_to_hf_mc_reference_data, None, global_settings, None)
        else:
            hf_data_iterator = None

        # ----------------------- create subordinate data iterators ------------------------------
        self.lf_data_iterators = [
            Data(path, None, global_settings, None) for path in path_to_lf_mc_data
        ]

        self.interface = interface
        self.features_config = features_config
        self.X_cols = X_cols
        self.num_features = num_features
        self.high_fidelity_model = hf_model
        self.X_train = None
        self.Y_HF_train = None
        self.Y_LFs_train = None
        self.X_mc = None
        self.Y_LFs_mc = None
        self.Y_HF_mc = None
        self.gammas_ext_mc = None
        self.gammas_ext_train = None
        self.Z_train = None
        self.Z_mc = None
        self.m_f_mc = None
        self.var_y_mc = None
        self.p_yhf_mean = None
        self.p_yhf_var = None
        self.predictive_var_bool = predictive_var
        self.p_yhf_mc = None
        self.p_ylf_mc = None
        self.no_features_comparison_bool = BMFMC_reference
        self.eigenfunc_random_fields = (
            None  # TODO this should be moved to the variable class! # pylint: disable=fixme
        )
        self.eigenvals = None
        self.f_mean_train = None
        self.y_pdf_support = np.linspace(y_pdf_support_min, y_pdf_support_max, 200)
        self.hf_data_iterator = hf_data_iterator
        self.training_indices = None
        self.uncertain_parameters = None
        self.visualization = None

        super().__init__()

    def _evaluate(self, samples):
        """Evaluate the Bayesian Multi-Fidelity Monte-Carlo (BMFMC) model.

        This method evaluates the BMFMC routine by performing two main tasks:

        1. **Probabilistic Mapping Evaluation**: Evaluates the probabilistic mapping for both the LF
                                                 Monte Carlo points and the LF training points.
        2. **BMFMC Posterior Statistics Evaluation**: Uses the results from the probabilistic
                                                      mapping to compute the BMFMC posterior
                                                      statistics.

        Args:
            samples (any type): Currently not used.

        Returns:
            output (dict): A dictionary containing the results of the evaluation. The dictionary
                           includes:
                - **Z_mc**: (np.array) Low-fidelity features Monte-Carlo data used in the
                                       evaluation.
                - **m_f_mc**: (np.array) Posterior mean values of the probabilistic mapping
                                         (denoted *f*) for LF Monte-Carlo inputs. These values
                                         represent the mean predictions based on the Monte-Carlo
                                         simulations.
                - **var_y_mc**: (np.array) Posterior variance of the probabilistic mapping
                                           (denoted *f*) for LF Monte-Carlo inputs. These values
                                           indicate the uncertainty associated with the mean
                                           predictions.
                - **y_pdf_support**: (np.array) Support grid for the probability density function
                                                (PDF) of the HF output used in this analysis.
                                                This provides the range over which the PDF is
                                                evaluated.
                - **p_yhf_mean**: (np.array) Posterior mean prediction of the HF output PDF,
                                             representing the expected values of the HF output
                                             given the model and data.
                - **p_yhf_var**: (np.array) Posterior variance prediction of the HF output PDF,
                                            indicating the uncertainty in the HF output predictions.
                - **p_yhf_mean_BMFMC**: (np.array or None) Reference posterior mean prediction of
                                                           the HF output PDF without using
                                                           informative features, for comparison
                                                           purposes. This value is calculated if
                                                           `self.no_features_comparison_bool` is
                                                           True.
                - **p_yhf_var_BMFMC**: (np.array or None) Reference posterior variance prediction of
                                                          the HF output PDF without using
                                                          informative features, for comparison
                                                          purposes. This value is calculated if
                                                          `self.no_features_comparison_bool` is
                                                          True.
                - **p_ylf_mc**: (np.array) Kernel density estimate of the LF model output PDF, used
                                           for illustration purposes.
                - **p_yhf_mc**: (np.array) Kernel density estimate of the HF model output PDF based
                                           on a full Monte-Carlo simulation. Used for benchmarking
                                           and comparison.
                - **Z_train**: (np.array) LF feature vector used for training the probabilistic
                                          mapping. This contains the features that were used to
                                          fit the model.
                - **X_train**: (np.array) Input matrix for simulations used to train the
                                          probabilistic mapping, corresponding to the training data.
                - **Y_HF_train**: (np.array) Outputs of the high-fidelity model corresponding to the
                                             training inputs *X_train*, representing the HF
                                             output values used for model fitting.
        """
        p_yhf_mean_BMFMC = None
        p_yhf_var_BMFMC = None

        self.compute_pymc_reference()

        # ------------------ STANDARD BMFMC (no additional features) for comparison ----------------
        if self.no_features_comparison_bool:
            p_yhf_mean_BMFMC, p_yhf_var_BMFMC = self.run_BMFMC_without_features()

        # ------------------- Generalized BMFMC with features --------------------------------------
        self.run_BMFMC()

        # gather and return the output
        output = {
            "Z_mc": self.Z_mc,
            "m_f_mc": self.m_f_mc,
            "var_y_mc": self.var_y_mc,
            "y_pdf_support": self.y_pdf_support,
            "p_yhf_mean": self.p_yhf_mean,
            "p_yhf_var": self.p_yhf_var,
            "p_yhf_mean_BMFMC": p_yhf_mean_BMFMC,
            "p_yhf_var_BMFMC": p_yhf_var_BMFMC,
            "p_ylf_mc": self.p_ylf_mc,
            "p_yhf_mc": self.p_yhf_mc,
            "Z_train": self.Z_train,
            "X_train": self.X_train,
            "Y_HF_train": self.Y_HF_train,
        }
        return output

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
        raise NotImplementedError("Gradient method is not implemented in `bmfmc_model`")

    def run_BMFMC(self):
        """Run BMFMC.

        Run with additional informative features.
        """
        # construct the probabilistic mapping between y_HF and LF features z_LF
        self.build_approximation(approx_case=True)

        # Evaluate probabilistic mapping for certain Z-points
        self.m_f_mc, self.var_y_mc = self.interface.evaluate(self.Z_mc.T)
        self.f_mean_train, _ = self.interface.evaluate(self.Z_train.T)
        # TODO the variables (here) manifold must probably an object from the variable class! # pylint: disable=fixme

        # actual 'evaluation' of generalized BMFMC routine
        self.compute_pyhf_statistics()

    def run_BMFMC_without_features(self):
        """Run BMFMC without further informative LF features.

        Returns:
            p_yhf_mean_BMFMC (np.array): Posterior mean function of the HF output density
            p_yhf_var_BMFMC (np.array): Posterior variance function of the HF output density
        """
        # construct the probabilistic mapping between y_HF and y_LF
        self.build_approximation(approx_case=False)

        # Evaluate probabilistic mapping for LF points
        self.m_f_mc, self.var_y_mc = self.interface.evaluate(self.Y_LFs_mc.T)
        self.f_mean_train, _ = self.interface.evaluate(self.Y_LFs_train.T)

        # actual 'evaluation' of BMFMC routine
        self.compute_pyhf_statistics()
        p_yhf_mean_BMFMC = self.p_yhf_mean  # this is just for comparison so no class attribute
        p_yhf_var_BMFMC = self.p_yhf_var  # this is just for comparison so no class attribute

        return p_yhf_mean_BMFMC, p_yhf_var_BMFMC

    def load_sampling_data(self):
        """Load sampling data.

        Load the low-fidelity sampling data from a pickle file into
        QUEENS. Check if high-fidelity benchmark data is available and
        load this as well.
        """
        # --------------------- load description for random fields/ parameters ---------
        # here we load the random parameter description from the pickle file
        # we load the description of the uncertain parameters from the first lf iterator
        # (note: all lf iterators have the same description)
        self.uncertain_parameters = (
            self.lf_data_iterators[0].read_pickle_file().get("input_description")
        )

        # --------------------- load LF sampling raw data with data iterators --------------
        self.X_mc = self.lf_data_iterators[0].read_pickle_file().get("input_data")
        # here we assume that all lfs have the same input vector
        try:
            self.eigenfunc_random_fields = (
                self.lf_data_iterators[0].read_pickle_file().get("eigenfunc")
            )
            self.eigenvals = self.lf_data_iterators[0].read_pickle_file().get("eigenvalue")
        except IOError:
            self.eigenfunc_random_fields = None
            self.eigenvals = None

        Y_LFs_mc = [
            lf_data_iterator.read_pickle_file().get("output")[:, 0]
            for lf_data_iterator in self.lf_data_iterators
        ]

        self.Y_LFs_mc = np.atleast_2d(np.vstack(Y_LFs_mc)).T

        # ------------------- Deal with potential HF-MC data --------------------------
        if self.hf_data_iterator is not None:
            try:
                self.Y_HF_mc = self.hf_data_iterator.read_pickle_file().get("output")[:, 0]
            except FileNotFoundError as ex:
                raise FileNotFoundError(
                    "The file containing the high-fidelity Monte-Carlo data "
                    "was not found! Abort..."
                ) from ex

    def get_hf_training_data(self):
        """Get high-fidelity training data.

        Given the low-fidelity sampling data and the optimal training input
        :math:`X`, either simulate the high-fidelity response for :math:`X` or
        load the corresponding high-fidelity response from the high-fidelity
        benchmark data provided by a pickle file.
        """
        # check if training simulation input was correctly calculated in iterator
        if self.X_train is None:
            raise ValueError(
                "The training input X_train cannot be 'None'! The training inputs "
                "should have been calculated in the iterator! Abort..."
            )

        # check how we should get the corresponding HF simulation output
        # Directly start simulations of HF model for optimal input batch X_train
        if (self.high_fidelity_model is not None) and (self.Y_HF_mc is None):
            _logger.info(
                "High-fidelity model found! Starting now simulation on HF model for BMFMC "
                "training data..."
            )
            # Evaluate High Fidelity Model
            self.high_fidelity_model.evaluate(self.X_train)

            # Get the HF-model training data for BMFMC
            self.Y_HF_train = self.high_fidelity_model.response["result"]
            _logger.info(
                "High-fidelity simulations finished successfully!\n Starting now BMFMC "
                "routine..."
            )
        elif (self.Y_HF_mc is not None) and (self.high_fidelity_model is None):
            # match Y_HF_mc data with X_train do determine Y_HF_train
            index_rows = [
                np.where(np.all(self.X_mc == self.X_train[i, :], axis=1))[0][0]
                for i, _ in enumerate(self.X_train[:, 0])
            ]

            self.Y_HF_train = np.atleast_2d(
                np.asarray([self.Y_HF_mc[index] for index in index_rows])
            ).T
        else:
            raise RuntimeError(
                "Please make sure to provide either a pickle file with "
                "high-fidelity Monte-Carlo data or an appropriate high-fidelity "
                "model to compute the high-fidelity training data! Abort..."
            )

    def build_approximation(self, approx_case=True):
        r"""Train surrogate model.

        Construct the probabilistic surrogate/mapping based on the
        provided training-data and optimize the hyper-parameters by maximizing
        the data's evidence or its lower bound (ELBO).

        Args:
            approx_case (bool):  Boolean that switches input features :math:`\boldsymbol{\gamma}`
                                 off if set to *False*. If not specified or set to *True*
                                 informative input features will be used in the BMFMC framework
        """
        # get the HF output data (from file or by starting a simulation, dependent on config)
        self.get_hf_training_data()

        # ----- train regression model on the data ----------------------------------------
        if approx_case:
            self.set_feature_strategy()
            self.interface.build_approximation(self.Z_train, self.Y_HF_train)
        else:
            self.interface.build_approximation(self.Y_LFs_train, self.Y_HF_train)

    def compute_pyhf_statistics(self):
        """Compute high-fidelity output density prediction.

        Calculate the high-fidelity output density prediction
        *p_yhf_mean* and its credible bounds *p_yhf_var* on the support
        *y_pdf_support* according to equation (14) and (15) in [1].
        """
        self.calculate_p_yhf_mean()

        if self.predictive_var_bool:
            self.calculate_p_yhf_var()
        else:
            self.p_yhf_var = None

    def calculate_p_yhf_mean(self):
        """Calculate the posterior mean estimate for the HF density."""
        standard_deviation = np.sqrt(self.var_y_mc)
        pdf_mat = st.norm.pdf(self.y_pdf_support, loc=self.m_f_mc, scale=standard_deviation)
        pyhf_mean_vec = np.sum(pdf_mat, axis=0)
        self.p_yhf_mean = 1 / self.m_f_mc.size * pyhf_mean_vec

    def calculate_p_yhf_var(self):
        """Calculate the posterior variance of the HF density prediction."""
        # calculate full posterior covariance matrix for testing points
        _, k_post = self.interface.evaluate(self.Z_mc.T)

        spacing = 1
        f_mean_pred = self.m_f_mc[0::spacing, :]
        yhf_var_pred = self.var_y_mc[0::spacing, :]
        k_post = k_post[0::spacing, 0::spacing]

        # Define support structure for computation
        points = np.vstack((self.y_pdf_support, self.y_pdf_support)).T

        # Define the outer loop (addition of all multivariate normal distributions
        yhf_pdf_grid = np.zeros((points.shape[0],))
        i = 1
        _logger.info("\n")

        # TODO we should speed this up with multiprocessing # pylint: disable=fixme
        for num1, (mean1, var1) in enumerate(
            zip(tqdm(f_mean_pred, desc=r"Calculating Var_f[p(y_HF|f,z,D)]"), yhf_var_pred)
        ):
            for num2, (mean2, var2) in enumerate(
                zip(f_mean_pred[num1 + 1 :], yhf_var_pred[num1 + 1 :])
            ):
                num2 = num1 + num2
                covariance = k_post[num1, num2]
                mean_vec = np.array([mean1, mean2])
                diff = points - mean_vec.T
                det_sigma = var1 * var2 - covariance**2

                if det_sigma < 0:
                    det_sigma = 1e-6
                    covariance = 0.95 * covariance

                inv_sigma = (
                    1
                    / det_sigma
                    * np.array(
                        [np.append(var2, -covariance), np.insert(var1, 0, -covariance)],
                        dtype=np.float64,
                    )
                )

                a = np.dot(diff, inv_sigma)
                b = np.einsum("ij,ij->i", a, diff)
                c = np.sqrt(4 * np.pi**2 * det_sigma)
                args = -0.5 * b + np.log(1 / c)
                args[args > 40] = 40  # limit arguments for for better conditioning
                yhf_pdf_grid += np.exp(args)
                i = i + 1

                # Define inner loop (add rows of 2D domain to yield variance function)
                self.p_yhf_var = 1 / (i - 1) * yhf_pdf_grid - 0.9995 * self.p_yhf_mean**2

    def compute_pymc_reference(self):
        """Compute reference kernel density estimate.

        Given a high-fidelity Monte-Carlo benchmark dataset, compute the
        reference kernel density estimate for the quantity of interest
        and optimize the bandwidth of the kde.
        """
        # optimize the bandwidth for the kde
        bandwidth_lfmc = est.estimate_bandwidth_for_kde(
            self.Y_LFs_mc[:, 0], np.amin(self.Y_LFs_mc[:, 0]), np.amax(self.Y_LFs_mc[:, 0])
        )

        if self.Y_HF_mc is not None:
            # perform kde with the optimized bandwidth in case HF MC is given
            self.p_yhf_mc, _ = est.estimate_pdf(
                np.atleast_2d(self.Y_HF_mc),
                bandwidth_lfmc,
                support_points=np.atleast_2d(self.y_pdf_support),
            )

        if self.Y_LFs_train.shape[1] < 2:
            self.p_ylf_mc, _ = est.estimate_pdf(
                np.atleast_2d(self.Y_LFs_mc).T,
                bandwidth_lfmc,
                support_points=np.atleast_2d(self.y_pdf_support),
            )  # TODO: make this also work for several lfs # pylint: disable=fixme

    def set_feature_strategy(self):
        r"""Set feature strategy.

        Depending on the method specified in the input file, set the
        strategy that will be used to calculate the low-fidelity
        features :math:`Z_{\text{LF}}`.
        """
        if self.features_config == "man_features":
            idx_vec = self.X_cols
            self.gammas_ext_train = np.atleast_2d(self.X_train[:, idx_vec]).T
            self.gammas_ext_mc = np.atleast_2d(self.X_mc[:, idx_vec]).T
            self.Z_train = np.hstack([self.Y_LFs_train, self.gammas_ext_train])
            self.Z_mc = np.hstack([self.Y_LFs_mc, self.gammas_ext_mc])
        elif self.features_config == "opt_features":
            if self.num_features < 1:
                raise ValueError(
                    f"You specified {self.num_features} features, "
                    "which is an "
                    f"invalid value! Please only specify integer values greater than zero! Abort..."
                )
            self.update_probabilistic_mapping_with_features()
        elif self.features_config == "no_features":
            self.Z_train = self.Y_LFs_train
            self.Z_mc = self.Y_LFs_mc
        else:
            raise IOError("Feature space method specified in input file is unknown!")

    def calculate_extended_gammas(self):
        r"""Calculate extended input features.

        Given the low-fidelity sampling data, calculate the extended
        input features :math:`\boldsymbol{\gamma_{\text{LF,ext}}}`. The
        informative input features :math:`\boldsymbol{\gamma}` are
        calculated so that they would maximize the Pearson correlation
        coefficient between :math:`\boldsymbol{\gamma_i^*}` and
        :math:`Y_{\text{LF}}^*`. Afterwards :math:`z_{\text{LF}}` is
        composed by :math:`y_{\text{LF}}` and
        :math:`\boldsymbol{\gamma_{LF}}`.
        """
        x_red = self.input_dim_red()  # this is also standardized
        x_iter_test = x_red
        self.gammas_ext_mc = np.empty((x_iter_test.shape[0], 0))

        # standardize the LF output vector for better performance
        Y_LFS_mc_stdized = StandardScaler().fit_transform(self.Y_LFs_mc)

        # Iteratively sort reduced input space by importance of its dimensions
        for counter in range(x_red.shape[1]):
            # calculate the scores/ranking of candidates for informative input features gamma_i
            # by projecting the (dim. reduced) input on the LFs output
            corr_coef_unnorm = np.abs(np.dot(x_iter_test.T, Y_LFS_mc_stdized))

            # --------- plot the rankings/scores for first iteration -------------------------------
            if self.visualization:
                if counter == 0:
                    ele = np.arange(1, x_iter_test.shape[1] + 1)
                    self.visualization.plot_feature_ranking(ele, corr_coef_unnorm, counter)
            # --------------------------------------------------------------------------------------

            # select input feature with the highest score
            select_bool = corr_coef_unnorm == np.max(corr_coef_unnorm)
            test_iter = np.dot(x_iter_test, select_bool)

            # substract last choice from candidate pool
            x_iter_test = np.atleast_2d(x_iter_test[:, ~select_bool[:, 0]])

            # Scale features linearly to LF output data so that probabilistic model
            # can be fit easier
            features_test = linear_scale_a_to_b(test_iter, self.Y_LFs_mc)

            # Assemble feature vectors and informative features
            self.gammas_ext_mc = np.hstack((self.gammas_ext_mc, features_test))

    def update_probabilistic_mapping_with_features(self):
        r"""Update probabilistic mapping.

        Given the number of additional informative features of the input
        and the extended feature matrix :math:`\Gamma_{LF,ext}`,
        assemble first the LF feature matrix :math:`Z_{LF}`. In a next
        step, update the probabilistic mapping with the LF-features. The
        former steps includes a training and prediction step. The
        determination of optimal training points is outsourced to the
        BMFMC iterator and the results get only called at this place.
        """
        # Select demanded number of features
        gamma_mc = self.gammas_ext_mc[:, 0 : self.num_features]
        self.Z_mc = np.hstack([self.Y_LFs_mc, gamma_mc])

        # Get training data from training_indices previously calculated in the iterator
        if self.training_indices is not None:
            self.Z_train = self.Z_mc[self.training_indices, :]
        else:
            raise ValueError("The training indices are still set to None! Abort...")

        # update dataset for probabilistic mapping with new feature dimensions
        self.interface.build_approximation(self.Z_train, self.Y_HF_train)
        self.m_f_mc, self.var_y_mc = self.interface.evaluate(self.Z_mc.T)
        self.f_mean_train, _ = self.interface.evaluate(self.Z_train.T)

    def input_dim_red(self):
        """Reduce dimensionality of the input space.

        Unsupervised dimensionality reduction of the input space. The random
        are first expressed via a truncated Karhunen-Loeve expansion that still
        contains, e.g. 95 % of the field's variance. Afterwards, input samples
        of the random fields get projected on the reduced basis and the
        coefficients of the projection sever as the new reduced encoding for
        the latter. Eventually the uncorrelated input samples and the reduced
        representation of random field samples get assembled to a new reduced
        input vector which is also standardized along each of the remaining
        dimensions.

        Returns:
            X_red_test_stdizd (np.array): Dimensionality reduced input matrix corresponding to
            testing/sampling data for the probabilistic mapping
        """
        x_uncorr, truncated_basis_dict = self.get_random_fields_and_truncated_basis(
            explained_var=95
        )
        if truncated_basis_dict is not None:
            coefs_mat = project_samples_on_truncated_basis(truncated_basis_dict, self.X_mc.shape[0])
        else:
            coefs_mat = None

        X_red_test_stdizd = assemble_x_red_stdizd(x_uncorr, coefs_mat)
        return X_red_test_stdizd

    def get_random_fields_and_truncated_basis(self, explained_var=95.0):
        """Get random fields and their truncated basis.

        Get the random fields and their description from the data files
        (pickle-files) and return their truncated basis. The truncation is
        determined based on the explained variance threshold (*explained_var*).

        Args:
            explained_var (float): Threshold for truncation in percent.

        Returns:
            random_fields_trunc_dict (dict): Dictionary containing samples of the random fields
              as well as their truncated basis
            x_uncorr (np.array): Samples of remaining uncorrelated random variables.
        """
        # determine uncorrelated random variables
        num_random_var = 0
        random_fields_list = []
        for parameter_name, parameter in self.parameters.dict.items():
            if isinstance(parameter, RandomField):
                random_fields_list.append((parameter_name, parameter))
            else:
                num_random_var += parameter.dimension

        x_uncorr = self.X_mc[:, 0:num_random_var]

        # iterate over all random fields
        dim_random_fields = 0
        if self.parameters.random_field_flag:
            for random_field, basis, eigenvals in zip(
                random_fields_list,
                self.eigenfunc_random_fields.items(),
                self.eigenvals.items(),
            ):
                # write the simulated samples of the random fields also in the new dictionary
                # Attention: Here we assume that X_mc contains in the first columns uncorrelated
                #            random variables until the column id 'num_random_var' and then only
                #            random fields
                random_fields_trunc_dict = {
                    random_field[0]: {
                        "samples": self.X_mc[
                            :,
                            num_random_var
                            + dim_random_fields : num_random_var
                            + dim_random_fields
                            + random_field[1].dimension,
                        ]
                    }
                }

                # determine the truncation basis
                idx_truncation = [
                    idx for idx, eigenval in enumerate(eigenvals[1]) if eigenval >= explained_var
                ][0]

                # write the truncated basis also in the dictionary
                random_fields_trunc_dict[random_field[0]].update(
                    {"trunc_basis": basis[1][0:idx_truncation]}
                )

                # adjust the counter for next iteration
                dim_random_fields += random_field[1].dimension
        else:
            random_fields_trunc_dict = None

        return x_uncorr, random_fields_trunc_dict


def project_samples_on_truncated_basis(truncated_basis_dict, num_samples):
    """Project samples on truncated basis.

    Project the high-dimensional samples of the random field on the truncated bases to yield the
    projection coefficients of the series expansion that serve as a new reduced representation of
    the random fields.

    Args:
        truncated_basis_dict (dic): Dictionary containing random field samples and truncated bases.
        num_samples (int): Number of Monte-Carlo samples.

    Returns:
        coefs_mat (np.array): Matrix containing the reduced representation of all random fields
                              stacked together along the columns.
    """
    coefs_mat = np.empty((num_samples, 0))

    # iterate over random fields
    for basis in truncated_basis_dict.items():
        coefs_mat = np.hstack((coefs_mat, np.dot(basis[1]["samples"], basis[1]["trunc_basis"].T)))

    return coefs_mat


def linear_scale_a_to_b(data_a, data_b):
    """Scale linearly.

    Scale a data vector 'data_a' linearly to the range of data vector
    'data_b'.

    Args:
        data_a (np.array): Data vector that should be scaled.
        data_b (np.array): Reference data vector that provides the range for scaling.

    Returns:
       scaled_a (np.array): Scaled data_a vector.
    """
    min_b = np.min(data_b)
    max_b = np.max(data_b)
    scaled_a = min_b + (data_a - np.min(data_a)) * (
        (max_b - min_b) / (np.max(data_a) - np.min(data_a))
    )
    return scaled_a


def assemble_x_red_stdizd(x_uncorr, coef_mat):
    """Assemble and standardize the dimension-reduced input x_red.

    Args:
        x_uncorr (np.array): Samples of remaining uncorrelated random variables.
        coef_mat (np.array): Optional coefficient matrix to concatenate.

    Returns:
        X_red_test_stdizd (np.array): Standardized dimension-reduced input.
    """
    if coef_mat is not None:
        x_red = np.hstack((x_uncorr, coef_mat))
    else:
        x_red = x_uncorr
    X_red_test_stdizd = StandardScaler().fit_transform(x_red)
    return X_red_test_stdizd


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
