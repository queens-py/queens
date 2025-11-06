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
"""Integration tests for the RPVI iterator."""

import numpy as np
import pandas as pd
import pytest
from mock import patch

from queens.data_processors import CsvFile
from queens.distributions import Normal
from queens.drivers import Function, Jobscript
from queens.iterators import RPVI
from queens.main import run_iterator
from queens.models import Adjoint, FiniteDifference, Gaussian, Simulation
from queens.parameters import Parameters
from queens.schedulers import Local, Pool
from queens.stochastic_optimizers import Adam
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io import load_result
from queens.utils.run_subprocess import run_subprocess
from queens.variational_distributions import FullRankNormal, MeanFieldNormal


def test_rpvi_park91a_hifi(
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    global_settings,
):
    """Integration test for the rpvi iterator.

    Based on the *park91a_hifi* function.
    """
    # Parameters
    x1 = Normal(mean=0.6, covariance=0.2)
    x2 = Normal(mean=0.3, covariance=0.1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    variational_distribution = MeanFieldNormal(dimension=2)
    stochastic_optimizer = Adam(
        optimization_type="max",
        learning_rate=0.025,
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
        max_iteration=500,
    )
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
        coordinate_labels=["x3", "x4"],
    )
    driver = Function(parameters=parameters, function="park91a_hifi_on_grid")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    forward_model = FiniteDifference(
        scheduler=scheduler, driver=driver, finite_difference_method="2-point", step_size=1e-07
    )
    model = Gaussian(
        noise_type="MAP_jeffrey_variance",
        nugget_noise_variance=1e-08,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = RPVI(
        max_feval=1000,
        n_samples_per_iter=2,
        score_function_bool=True,
        natural_gradient=True,
        FIM_dampening=True,
        decay_start_iteration=50,
        dampening_coefficient=0.01,
        FIM_dampening_lower_bound=1e-08,
        variational_transformation=None,
        variational_parameter_initialization="prior",
        random_seed=1,
        result_description={
            "write_results": True,
            "plotting_options": {
                "plot_boolean": False,
                "plotting_dir": tmp_path,
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

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.1
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


def test_rpvi_park91a_hifi_provided_gradient(
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    global_settings,
):
    """Test rpvi on *park91a_hifi* function with analytical gradients."""
    # Parameters
    x1 = Normal(mean=0.6, covariance=0.2)
    x2 = Normal(mean=0.3, covariance=0.1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    variational_distribution = MeanFieldNormal(dimension=2)
    stochastic_optimizer = Adam(
        optimization_type="max",
        learning_rate=0.025,
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
        max_iteration=500,
    )
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
        coordinate_labels=["x3", "x4"],
    )
    driver = Function(parameters=parameters, function="park91a_hifi_on_grid_with_gradients")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    forward_model = Simulation(scheduler=scheduler, driver=driver)
    model = Gaussian(
        noise_type="MAP_jeffrey_variance",
        nugget_noise_variance=1e-08,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = RPVI(
        max_feval=1000,
        n_samples_per_iter=2,
        score_function_bool=True,
        natural_gradient=True,
        FIM_dampening=True,
        decay_start_iteration=50,
        dampening_coefficient=0.01,
        FIM_dampening_lower_bound=1e-08,
        variational_transformation=None,
        variational_parameter_initialization="prior",
        random_seed=1,
        result_description={
            "write_results": True,
            "plotting_options": {
                "plot_boolean": False,
                "plotting_dir": tmp_path,
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

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.1
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


likelihood_mean = np.array([-2.0, 1.0])
likelihood_covariance = np.diag(np.array([0.1, 10.0]))
likelihood = Normal(likelihood_mean, likelihood_covariance)


def target_density(self, samples):
    """Target posterior density."""
    log_likelihood_output = likelihood.logpdf(samples)
    grad_log_likelihood = likelihood.grad_logpdf(samples)
    self.num_evaluations += len(samples)

    return log_likelihood_output, grad_log_likelihood


@pytest.fixture(name="forward_model", scope="module", params=["simulation_model", "fd_model"])
def fixture_forward_model(request):
    """Different forward models."""
    return request.param


@pytest.mark.max_time_for_test(20)
def test_rpvi_gaussian(tmp_path, _create_experimental_data, forward_model, global_settings):
    """Test RPVI with univariate Gaussian."""
    # Parameters
    x1 = Normal(mean=0.0, covariance=1.0)
    x2 = Normal(mean=10.0, covariance=100.0)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    variational_distribution = MeanFieldNormal(dimension=2)
    stochastic_optimizer = Adam(
        learning_rate=0.05,
        optimization_type="max",
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
        max_iteration=10000000,
    )
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
    )
    driver = Function(parameters=parameters, function="patch_for_likelihood")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    forward_model = Simulation(scheduler=scheduler, driver=driver)
    model = Gaussian(
        noise_type="fixed_variance",
        noise_value=1,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = RPVI(
        max_feval=100000,
        n_samples_per_iter=10,
        score_function_bool=False,
        natural_gradient=True,
        FIM_dampening=True,
        decay_start_iteration=50,
        dampening_coefficient=0.01,
        FIM_dampening_lower_bound=1e-08,
        variational_transformation=None,
        variational_parameter_initialization="prior",
        random_seed=1,
        verbose_every_n_iter=1000,
        result_description={
            "write_results": True,
            "plotting_options": {
                "plot_boolean": False,
                "plotting_dir": tmp_path,
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
    with patch.object(Gaussian, "evaluate_and_gradient", target_density):
        run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    posterior_covariance = np.diag(np.array([1 / 11, 100 / 11]))
    posterior_mean = np.array([-20 / 11, 20 / 11]).reshape(-1, 1)

    # Actual tests
    np.testing.assert_almost_equal(
        results["variational_distribution"]["mean"], posterior_mean, decimal=3
    )
    np.testing.assert_almost_equal(
        results["variational_distribution"]["covariance"], posterior_covariance, decimal=4
    )


@pytest.fixture(name="_create_experimental_data")
def fixture_create_experimental_data(tmp_path):
    """Create a csv file with experimental data."""
    data_dict = {"y_obs": np.zeros(1)}
    experimental_data_path = tmp_path / "experimental_data.csv"
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)


@pytest.fixture(name="module_path")
def fixture_module_path(tmp_path):
    """Path to likelihood module."""
    my_module_path = tmp_path / "my_likelihood_module.py"
    return str(my_module_path)


@pytest.fixture(name="_write_custom_likelihood_model")
def fixture_write_custom_likelihood_model(module_path):
    """Create a python file with a custom likelihood class."""
    custom_class_lst = [
        "from queens.models.likelihoods.gaussian_likelihood import Gaussian\n",
        "class MyLikelihood(Gaussian):\n",
        "   pass",
    ]
    with open(module_path, "w", encoding="utf-8") as f:
        for my_string in custom_class_lst:
            f.writelines(my_string)


@pytest.fixture(name="python_path")
def fixture_python_path():
    """Current python path."""
    _, _, stdout, _ = run_subprocess("which python")
    return stdout.strip()


@pytest.fixture(name="rpvi_jobscript_template", scope="session")
def fixture_rpvi_jobscript_template():
    """Jobscript template for the Jobscript driver in rpvi test cases."""
    return "{{ executable }} {{ input_file }} {{ output_file }}"


def test_rpvi_exe_park91a_hifi_provided_gradient(
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    example_simulator_fun_dir,
    _create_input_file_executable_park91a_hifi_on_grid,
    python_path,
    rpvi_jobscript_template,
    global_settings,
):
    """Test for the *rpvi* iterator based on the *park91a_hifi* function."""
    # generate json input file from template
    third_party_input_file = tmp_path / "input_file_executable_park91a_hifi_on_grid.csv"
    experimental_data_path = tmp_path
    executable = example_simulator_fun_dir / "executable_park91a_hifi_on_grid_with_gradients.py"
    executable = f"{python_path} {executable} p"
    plot_dir = tmp_path
    # Parameters
    x1 = Normal(mean=0.6, covariance=0.2)
    x2 = Normal(mean=0.3, covariance=0.1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    variational_distribution = FullRankNormal(dimension=2)
    stochastic_optimizer = Adam(
        optimization_type="max",
        learning_rate=0.02,
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
        max_iteration=10000000,
    )
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="experimental_data.csv",
        csv_data_base_dir=experimental_data_path,
        output_label="y_obs",
        coordinate_labels=["x3", "x4"],
    )
    scheduler = Local(
        num_procs=1,
        num_jobs=1,
        experiment_name=global_settings.experiment_name,
    )
    data_processor = CsvFile(
        file_name_identifier="*_output.csv",
        file_options_dict={
            "delete_field_data": False,
            "filter": {"type": "entire_file"},
        },
    )
    gradient_data_processor = CsvFile(
        file_name_identifier="*_gradient.csv",
        file_options_dict={
            "delete_field_data": False,
            "filter": {"type": "entire_file"},
        },
    )
    driver = Jobscript(
        parameters=parameters,
        input_templates=third_party_input_file,
        executable=executable,
        data_processor=data_processor,
        gradient_data_processor=gradient_data_processor,
        jobscript_template=rpvi_jobscript_template,
    )
    forward_model = Simulation(scheduler=scheduler, driver=driver)
    model = Gaussian(
        noise_type="MAP_jeffrey_variance",
        nugget_noise_variance=1e-08,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = RPVI(
        max_feval=10,
        n_samples_per_iter=3,
        score_function_bool=True,
        natural_gradient=True,
        FIM_dampening=True,
        decay_start_iteration=50,
        dampening_coefficient=0.01,
        FIM_dampening_lower_bound=1e-08,
        variational_transformation=None,
        variational_parameter_initialization="prior",
        random_seed=1,
        result_description={
            "write_results": True,
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

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.15
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


@pytest.mark.max_time_for_test(20)
def test_rpvi_exe_park91a_hifi_finite_differences_gradient(
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    example_simulator_fun_dir,
    _create_input_file_executable_park91a_hifi_on_grid,
    python_path,
    rpvi_jobscript_template,
    global_settings,
):
    """Test for the *rpvi* iterator based on the *park91a_hifi* function."""
    # generate json input file from template
    third_party_input_file = tmp_path / "input_file_executable_park91a_hifi_on_grid.csv"
    experimental_data_path = tmp_path
    executable = example_simulator_fun_dir / "executable_park91a_hifi_on_grid_with_gradients.py"
    executable = f"{python_path} {executable} s"
    plot_dir = tmp_path
    # Parameters
    x1 = Normal(mean=0.6, covariance=0.2)
    x2 = Normal(mean=0.3, covariance=0.1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    variational_distribution = FullRankNormal(dimension=2)
    stochastic_optimizer = Adam(
        optimization_type="max",
        learning_rate=0.02,
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
        max_iteration=10000000,
    )
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="experimental_data.csv",
        csv_data_base_dir=experimental_data_path,
        output_label="y_obs",
        coordinate_labels=["x3", "x4"],
    )
    scheduler = Local(
        num_procs=1,
        num_jobs=1,
        experiment_name=global_settings.experiment_name,
    )
    data_processor = CsvFile(
        file_name_identifier="*_output.csv",
        file_options_dict={
            "delete_field_data": False,
            "filter": {"type": "entire_file"},
        },
    )
    driver = Jobscript(
        parameters=parameters,
        input_templates=third_party_input_file,
        executable=executable,
        data_processor=data_processor,
        jobscript_template=rpvi_jobscript_template,
    )
    forward_model = FiniteDifference(
        scheduler=scheduler, driver=driver, finite_difference_method="2-point"
    )
    model = Gaussian(
        noise_type="MAP_jeffrey_variance",
        nugget_noise_variance=1e-08,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = RPVI(
        max_feval=10,
        n_samples_per_iter=3,
        score_function_bool=True,
        natural_gradient=True,
        FIM_dampening=True,
        decay_start_iteration=50,
        dampening_coefficient=0.01,
        FIM_dampening_lower_bound=1e-08,
        variational_transformation=None,
        variational_parameter_initialization="prior",
        random_seed=1,
        result_description={
            "write_results": True,
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

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.15
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


def test_rpvi_exe_park91a_hifi_adjoint_gradient(
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    example_simulator_fun_dir,
    _create_input_file_executable_park91a_hifi_on_grid,
    python_path,
    rpvi_jobscript_template,
    global_settings,
):
    """Test the *rpvi* iterator based on the *park91a_hifi* function."""
    # generate json input file from template
    third_party_input_file = tmp_path / "input_file_executable_park91a_hifi_on_grid.csv"
    experimental_data_path = tmp_path
    executable = example_simulator_fun_dir / "executable_park91a_hifi_on_grid_with_gradients.py"
    executable = f"{python_path} {executable} s"
    # adjoint executable (here we actually use the same executable but call it with
    # a different flag "a" for adjoint)
    adjoint_executable = (
        example_simulator_fun_dir / "executable_park91a_hifi_on_grid_with_gradients.py"
    )
    adjoint_executable = f"{python_path} {adjoint_executable} a"
    plot_dir = tmp_path
    # Parameters
    x1 = Normal(mean=0.6, covariance=0.2)
    x2 = Normal(mean=0.3, covariance=0.1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    variational_distribution = FullRankNormal(dimension=2)
    stochastic_optimizer = Adam(
        optimization_type="max",
        learning_rate=0.02,
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
        max_iteration=10000000,
    )
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="experimental_data.csv",
        csv_data_base_dir=experimental_data_path,
        output_label="y_obs",
        coordinate_labels=["x3", "x4"],
    )
    scheduler = Local(
        num_procs=1,
        num_jobs=1,
        experiment_name=global_settings.experiment_name,
    )
    data_processor = CsvFile(
        file_name_identifier="*_output.csv",
        file_options_dict={
            "delete_field_data": False,
            "filter": {"type": "entire_file"},
        },
    )
    driver = Jobscript(
        parameters=parameters,
        input_templates=third_party_input_file,
        executable=executable,
        data_processor=data_processor,
        jobscript_template=rpvi_jobscript_template,
    )
    gradient_data_processor = CsvFile(
        file_name_identifier="*_gradient.csv",
        file_options_dict={
            "delete_field_data": False,
            "filter": {"type": "entire_file"},
        },
    )
    adjoint_driver = Jobscript(
        parameters=parameters,
        input_templates=third_party_input_file,
        executable=adjoint_executable,
        data_processor=gradient_data_processor,
        jobscript_template=rpvi_jobscript_template,
    )
    forward_model = Adjoint(
        adjoint_file="grad_objective.csv",
        scheduler=scheduler,
        driver=driver,
        gradient_driver=adjoint_driver,
    )
    model = Gaussian(
        noise_type="MAP_jeffrey_variance",
        nugget_noise_variance=1e-08,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = RPVI(
        max_feval=10,
        n_samples_per_iter=3,
        score_function_bool=True,
        natural_gradient=True,
        FIM_dampening=True,
        decay_start_iteration=50,
        dampening_coefficient=0.01,
        FIM_dampening_lower_bound=1e-08,
        variational_transformation=None,
        variational_parameter_initialization="prior",
        random_seed=1,
        result_description={
            "write_results": True,
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

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.15
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


@pytest.fixture(name="_create_input_file_executable_park91a_hifi_on_grid")
def fixture_create_input_file_executable_park91a_hifi_on_grid(tmp_path):
    """Create a csv file as temporary input file for executable."""
    input_path = tmp_path / "input_file_executable_park91a_hifi_on_grid.csv"
    input_path.write_text("{{ x1 }}\n{{ x2 }}", encoding="utf-8")
