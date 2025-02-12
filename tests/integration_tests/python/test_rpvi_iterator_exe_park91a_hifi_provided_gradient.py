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
"""Integration test for reparametrization trick VI as executable."""

import numpy as np
import pytest

from queens.data_processors import Csv
from queens.distributions import NormalDistribution
from queens.drivers import MpiDriver
from queens.iterators import RPVIIterator
from queens.main import run_iterator
from queens.models import (
    DifferentiableSimulationModelAdjoint,
    DifferentiableSimulationModelFD,
    GaussianLikelihood,
    SimulationModel,
)
from queens.parameters import Parameters
from queens.schedulers import LocalScheduler
from queens.stochastic_optimizers import Adam
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io_utils import load_result
from queens.utils.run_subprocess import run_subprocess
from queens.variational_distributions import FullRankNormalVariational


@pytest.fixture(name="python_path")
def fixture_python_path():
    """Current python path."""
    _, _, stdout, _ = run_subprocess("which python")
    return stdout.strip()


@pytest.fixture(name="mpirun_path", scope="session")
def fixture_mpi_run_path():
    """Path to the mpirun executable."""
    _, _, stdout, _ = run_subprocess("which mpirun")
    return stdout.strip()


@pytest.fixture(name="mpi_command", scope="session")
def fixture_mpi_command(mpirun_path):
    """Base command to call mpirun with MpiDriver."""
    return mpirun_path + " --bind-to none"


def test_rpvi_iterator_exe_park91a_hifi_provided_gradient(
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    example_simulator_fun_dir,
    _create_input_file_executable_park91a_hifi_on_grid,
    python_path,
    mpi_command,
    global_settings,
):
    """Test for the *rpvi* iterator based on the *park91a_hifi* function."""
    # pylint: disable=duplicate-code
    # generate json input file from template
    third_party_input_file = tmp_path / "input_file_executable_park91a_hifi_on_grid.csv"
    experimental_data_path = tmp_path
    executable = example_simulator_fun_dir / "executable_park91a_hifi_on_grid_with_gradients.py"
    executable = f"{python_path} {executable} p"
    plot_dir = tmp_path
    # Parameters
    x1 = NormalDistribution(mean=0.6, covariance=0.2)
    x2 = NormalDistribution(mean=0.3, covariance=0.1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    variational_distribution = FullRankNormalVariational(dimension=2)
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
    scheduler = LocalScheduler(
        num_procs=1,
        num_jobs=1,
        experiment_name=global_settings.experiment_name,
    )
    data_processor = Csv(
        file_name_identifier="*_output.csv",
        file_options_dict={
            "delete_field_data": False,
            "filter": {"type": "entire_file"},
        },
    )
    gradient_data_processor = Csv(
        file_name_identifier="*_gradient.csv",
        file_options_dict={
            "delete_field_data": False,
            "filter": {"type": "entire_file"},
        },
    )
    driver = MpiDriver(
        parameters=parameters,
        input_templates=third_party_input_file,
        executable=executable,
        data_processor=data_processor,
        gradient_data_processor=gradient_data_processor,
        mpi_cmd=mpi_command,
    )
    forward_model = SimulationModel(scheduler=scheduler, driver=driver)
    model = GaussianLikelihood(
        noise_type="MAP_jeffrey_variance",
        nugget_noise_variance=1e-08,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = RPVIIterator(
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
def test_rpvi_iterator_exe_park91a_hifi_finite_differences_gradient(
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    example_simulator_fun_dir,
    _create_input_file_executable_park91a_hifi_on_grid,
    python_path,
    mpi_command,
    global_settings,
):
    """Test for the *rpvi* iterator based on the *park91a_hifi* function."""
    # pylint: disable=duplicate-code
    # generate json input file from template
    third_party_input_file = tmp_path / "input_file_executable_park91a_hifi_on_grid.csv"
    experimental_data_path = tmp_path
    executable = example_simulator_fun_dir / "executable_park91a_hifi_on_grid_with_gradients.py"
    executable = f"{python_path} {executable} s"
    plot_dir = tmp_path
    # Parameters
    x1 = NormalDistribution(mean=0.6, covariance=0.2)
    x2 = NormalDistribution(mean=0.3, covariance=0.1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    variational_distribution = FullRankNormalVariational(dimension=2)
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
    scheduler = LocalScheduler(
        num_procs=1,
        num_jobs=1,
        experiment_name=global_settings.experiment_name,
    )
    data_processor = Csv(
        file_name_identifier="*_output.csv",
        file_options_dict={
            "delete_field_data": False,
            "filter": {"type": "entire_file"},
        },
    )
    driver = MpiDriver(
        parameters=parameters,
        input_templates=third_party_input_file,
        executable=executable,
        data_processor=data_processor,
        mpi_cmd=mpi_command,
    )
    forward_model = DifferentiableSimulationModelFD(
        scheduler=scheduler, driver=driver, finite_difference_method="2-point"
    )
    model = GaussianLikelihood(
        noise_type="MAP_jeffrey_variance",
        nugget_noise_variance=1e-08,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = RPVIIterator(
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


def test_rpvi_iterator_exe_park91a_hifi_adjoint_gradient(
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    example_simulator_fun_dir,
    _create_input_file_executable_park91a_hifi_on_grid,
    python_path,
    mpi_command,
    global_settings,
):
    """Test the *rpvi* iterator based on the *park91a_hifi* function."""
    # pylint: disable=duplicate-code
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
    x1 = NormalDistribution(mean=0.6, covariance=0.2)
    x2 = NormalDistribution(mean=0.3, covariance=0.1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    variational_distribution = FullRankNormalVariational(dimension=2)
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
    scheduler = LocalScheduler(
        num_procs=1,
        num_jobs=1,
        experiment_name=global_settings.experiment_name,
    )
    data_processor = Csv(
        file_name_identifier="*_output.csv",
        file_options_dict={
            "delete_field_data": False,
            "filter": {"type": "entire_file"},
        },
    )
    driver = MpiDriver(
        parameters=parameters,
        input_templates=third_party_input_file,
        executable=executable,
        data_processor=data_processor,
        mpi_cmd=mpi_command,
    )
    gradient_data_processor = Csv(
        file_name_identifier="*_gradient.csv",
        file_options_dict={
            "delete_field_data": False,
            "filter": {"type": "entire_file"},
        },
    )
    adjoint_driver = MpiDriver(
        parameters=parameters,
        input_templates=third_party_input_file,
        executable=adjoint_executable,
        data_processor=gradient_data_processor,
        mpi_cmd=mpi_command,
    )
    forward_model = DifferentiableSimulationModelAdjoint(
        adjoint_file="grad_objective.csv",
        scheduler=scheduler,
        driver=driver,
        gradient_driver=adjoint_driver,
    )
    model = GaussianLikelihood(
        noise_type="MAP_jeffrey_variance",
        nugget_noise_variance=1e-08,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = RPVIIterator(
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
