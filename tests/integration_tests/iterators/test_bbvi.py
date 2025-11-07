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
"""Integration tests for the BBVI iterator."""

from collections import namedtuple

import numpy as np
import pytest
from mock import Mock, patch
from scipy.stats import multivariate_normal as mvn

from queens.distributions.normal import Normal
from queens.drivers.function import Function
from queens.global_settings import GlobalSettings
from queens.iterators.bbvi import BBVI
from queens.main import run_iterator
from queens.models.likelihoods.gaussian import Gaussian
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.stochastic_optimizers import Adam
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io import load_result
from queens.utils.iterative_averaging import MovingAveraging
from queens.variational_distributions import FullRankNormal, MeanFieldNormal


def test_bbvi_density_match(
    mocker,
    dummy_bbvi_instance,
):
    """Matching a Gaussian distribution."""
    # fix the random seed
    np.random.seed(1)

    # mock all parts of the algorithm that has to do with initialization or an underlying model
    mocker.patch(
        "queens.iterators.bbvi.BBVI.pre_run",
        return_value=None,
    )

    # actual main call of bbvi with patched density for posterior
    with patch.object(BBVI, "get_log_posterior_unnormalized", target_density):
        variational_distr_obj = dummy_bbvi_instance.variational_distribution
        mean = np.array([0.1, 0.7, 0.2, 0.3, 0.25])
        cov = np.exp(np.diag([0.5, 0.5, 0.5, 0.5, 0.5]) * 2)
        var_params = variational_distr_obj.construct_variational_parameters(mean, cov)
        dummy_bbvi_instance.variational_params = var_params
        dummy_bbvi_instance.stochastic_optimizer.set_gradient_function(
            dummy_bbvi_instance.get_gradient_function()
        )
        dummy_bbvi_instance.stochastic_optimizer.current_variational_parameters = var_params
        dummy_bbvi_instance.run()  # actual run of the algorithm

        opt_variational_params = np.array(dummy_bbvi_instance.variational_params)

        elbo = dummy_bbvi_instance.iteration_data.elbo

    # Actual tests
    opt_variational_samples = variational_distr_obj.draw(opt_variational_params, 10000)
    variational_logpdf = variational_distr_obj.logpdf(
        opt_variational_params, opt_variational_samples
    )
    target_logpdf = target_density(
        dummy_bbvi_instance, x=opt_variational_samples, pdf=False
    ).flatten()
    kl_divergence = np.abs(np.mean(variational_logpdf - target_logpdf))
    # Test if KL divergence is not too large
    assert kl_divergence < 10**5
    # Test if the elbo declined on average
    assert np.mean(elbo[-3:]) > np.mean(elbo[:3])


def test_bbvi_park91a_hifi(
    tmp_path, _create_experimental_data_park91a_hifi_on_grid, global_settings
):
    """Test for the bbvi iterator based on the *park91a_hifi* function."""
    experimental_data_path = tmp_path
    plot_dir = tmp_path
    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)

    # Parameters
    x1 = Normal(mean=0.6, covariance=0.2)
    x2 = Normal(mean=0.3, covariance=0.1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    variational_distribution = FullRankNormal(dimension=2)
    stochastic_optimizer = Adam(
        learning_rate=0.01,
        optimization_type="max",
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
        max_iteration=10000000,
    )
    noise_var_iterative_averaging = MovingAveraging(num_iter_for_avg=10)
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=experimental_data_path,
        output_label="y_obs",
        coordinate_labels=["x3", "x4"],
    )
    driver = Function(parameters=parameters, function="park91a_hifi_on_grid")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    forward_model = Simulation(scheduler=scheduler, driver=driver)
    model = Gaussian(
        noise_type="MAP_jeffrey_variance",
        nugget_noise_variance=1e-08,
        noise_var_iterative_averaging=noise_var_iterative_averaging,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = BBVI(
        max_feval=100,
        n_samples_per_iter=2,
        memory=20,
        model_eval_iteration_period=1000,
        natural_gradient=True,
        FIM_dampening=True,
        decay_start_iteration=50,
        dampening_coefficient=0.01,
        FIM_dampening_lower_bound=1e-08,
        variational_transformation=None,
        variational_parameter_initialization="prior",
        random_seed=1,
        control_variates_scaling_type="averaged",
        loo_control_variates_scaling=False,
        result_description={
            "write_results": True,
            "iterative_field_names": ["elbo"],
            "plotting_options": {
                "plot_boolean": False,
                "plotting_dir": plot_dir,
                "plot_name": "variational_params_convergence.eps",
                "save_bool": False,
            },
        },
        variational_distribution=variational_distribution,
        stochastic_optimizer=stochastic_optimizer,
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    elbo_list = results["iteration_data"]["elbo"]

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.2
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5
    assert np.mean(elbo_list[-5:]) > np.mean(elbo_list[:5])


@pytest.fixture(name="dummy_bbvi_instance")
def fixture_dummy_bbvi_instance(tmp_path, my_variational_distribution):
    """A BBVI instance."""
    #  ----- interesting params one might want to change ---------------------------
    n_samples_per_iter = 5
    max_feval = 10 * n_samples_per_iter
    num_variables = 5
    memory = 10
    natural_gradient_bool = False
    fim_dampening_bool = True
    variational_params_initialization_approach = "random"
    fim_decay_start_iter = 50
    fim_dampening_coefficient = 1e-2
    fim_dampening_lower_bound = 1e-8
    control_variates_scaling_type = "averaged"
    loo_cv_bool = False
    # ------ params we want to keep fixed -----------------------------------------
    variational_transformation = None
    experiment_name = "density_match"
    result_description = {
        "iterative_field_names": ["elbo"],
        "write_results": False,
        "plotting_options": {
            "plot_boolean": False,
            "plotting_dir": tmp_path,
            "plot_name": "variational_params_convergence.eps",
            "save_bool": False,
        },
    }
    stochastic_optimizer = Adam(
        learning_rate=0.01,
        optimization_type="max",
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
        max_iteration=10000000,
    )

    # ------ other params ----------------------------------------------------------
    model = namedtuple("model", ("normal_distribution", "num_evaluations"))(
        normal_distribution=namedtuple("normal_distribution", "covariance")(covariance=0),
        num_evaluations=0,
    )
    global_settings = GlobalSettings(experiment_name, output_dir=tmp_path)
    random_seed = 1

    with global_settings:
        parameters = Mock()
        parameters.num_parameters = num_variables

        bbvi_instance = BBVI(
            model=model,
            parameters=parameters,
            global_settings=global_settings,
            result_description=result_description,
            variational_parameter_initialization=variational_params_initialization_approach,
            n_samples_per_iter=n_samples_per_iter,
            variational_transformation=variational_transformation,
            random_seed=random_seed,
            max_feval=max_feval,
            memory=memory,
            natural_gradient=natural_gradient_bool,
            FIM_dampening=fim_dampening_bool,
            decay_start_iteration=fim_decay_start_iter,
            dampening_coefficient=fim_dampening_coefficient,
            FIM_dampening_lower_bound=fim_dampening_lower_bound,
            control_variates_scaling_type=control_variates_scaling_type,
            loo_control_variates_scaling=loo_cv_bool,
            variational_distribution=my_variational_distribution,
            stochastic_optimizer=stochastic_optimizer,
            model_eval_iteration_period=1,
            resample=True,
        )
    return bbvi_instance


def target_density(self, x=None, pdf=False):  # pylint: disable=unused-argument
    """Create visualization module."""
    output_array = []
    mean = (np.array([0.5, 0.2, 0.6, 0.1, 0.2])).reshape(
        -1,
    )
    std_vec = np.array([0.1, 0.2, 0.01, 0.3, 0.1])
    cov = np.diag(std_vec**2)
    if not pdf:
        for value in x:
            output_array.append(mvn.logpdf(value, mean=mean, cov=cov))
    else:
        for value in x:
            output_array.append(mvn.pdf(value, mean=mean, cov=cov))

    output_array = np.array(output_array)
    self.model = self.model._replace(num_evaluations=self.model.num_evaluations + len(x))
    return output_array


@pytest.fixture(name="my_variational_distribution")
def fixture_my_variational_distribution():
    """A variational distribution."""
    return MeanFieldNormal(dimension=5)
