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
"""Test remote 4C simulations with ensight data-processor."""

import getpass
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest
from testbook import testbook

from queens.data_processors.pvd_file import PvdFile
from queens.distributions.uniform import Uniform
from queens.drivers import Jobscript
from queens.iterators.monte_carlo import MonteCarlo
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.cluster import Cluster
from queens.utils.config_directories import experiment_directory
from queens.utils.io import load_result
from queens.utils.path import relative_path_from_root
from queens.utils.remote_operations import RemoteConnection
from test_utils.integration_tests import fourc_build_path_from_home

_logger = logging.getLogger(__name__)


THOUGHT_CLUSTER_TYPE = "thought"
BRUTEFORCE_CLUSTER_TYPE = "bruteforce"
CHARON_CLUSTER_TYPE = "charon"


@pytest.mark.parametrize(
    "cluster",
    [
        pytest.param(THOUGHT_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        # pytest.param(BRUTEFORCE_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        pytest.param(CHARON_CLUSTER_TYPE, marks=pytest.mark.imcs_cluster),
    ],
    indirect=True,
)
class TestCluster:
    """Test class collecting all test with Dask jobqueue clusters and 4C.

    NOTE: we use a class here since our fixture are set to autouse, but we only want to call them
    for these tests.
    """

    def pytest_base_directory_on_cluster(self):
        """Remote directory containing several pytest runs."""
        return "~/queens-tests"

    @pytest.fixture(name="test_base_directory_on_cluster")
    def fixture_test_base_directory_on_cluster(self, pytest_id):
        """Remote directory containing all experiments of a single pytest run.

        This directory is conceptually equivalent to the usual base
        directory for non-pytest runs, i.e., production experiments. The
        goal is to separate the testing data from production data of the
        user.
        """
        return self.pytest_base_directory_on_cluster() + f"/{pytest_id}"

    @pytest.fixture(name="experiment_dir")
    def fixture_experiment_dir(
        self, global_settings, remote_connection, test_base_directory_on_cluster
    ):
        """Fixture providing the remote experiment directory."""
        experiment_dir, _ = remote_connection.run_function(
            experiment_directory,
            global_settings.experiment_name,
            test_base_directory_on_cluster,
        )
        return experiment_dir

    @pytest.fixture(name="_create_experiment_dir")
    def fixture_create_experiment_dir(self, remote_connection, experiment_dir):
        """Fixture providing the remote experiment directory."""

        def create_experiment_dir_and_assert_it_exists():
            """Create experiment directory on remote and assert it exists."""
            os.mkdir(experiment_dir)
            return experiment_dir.exists()

        assert remote_connection.run_function(create_experiment_dir_and_assert_it_exists)

    @pytest.fixture(name="basic_cluster_kwargs")
    def fixture_basic_cluster_kwargs(self, cluster_settings, test_base_directory_on_cluster):
        """Basic keyword arguments to initialize the cluster scheduler."""
        return {
            "workload_manager": cluster_settings["workload_manager"],
            "queue": cluster_settings.get("queue"),
            "cluster_internal_address": cluster_settings["cluster_internal_address"],
            "experiment_base_dir": test_base_directory_on_cluster,
        }

    @pytest.fixture(name="cluster_kwargs")
    def fixture_cluster_kwargs(self, basic_cluster_kwargs, remote_connection, test_name):
        """Keyword arguments to initialize the cluster scheduler."""
        return basic_cluster_kwargs | {
            "walltime": "00:10:00",
            "num_jobs": 1,
            "min_jobs": 1,
            "num_procs": 1,
            "num_nodes": 1,
            "remote_connection": remote_connection,
            "experiment_name": test_name,
        }

    def test_new_experiment_dir(self, cluster_kwargs, remote_connection, experiment_dir):
        """Test cluster init when experiment dir does not exist."""
        experiment_dir_exists = remote_connection.run_function(experiment_dir.exists)
        assert not experiment_dir_exists

        Cluster(**cluster_kwargs)

        experiment_dir_exists = remote_connection.run_function(experiment_dir.exists)
        assert experiment_dir_exists

    def test_overwriting_existing_experiment_dir(self, cluster_kwargs, _create_experiment_dir):
        """Test cluster init when overwriting experiment dir via flag."""
        Cluster(**cluster_kwargs, overwrite_existing_experiment=True)

    def test_no_prompt_input_for_existing_experiment_dir(
        self, cluster_kwargs, mocker, _create_experiment_dir
    ):
        """Test cluster init when not overwriting experiment dir via flag.

        Since the experiment directory already exists, the scheduler
        prompts the user for input. In this test case, the user does not
        provide any prompt input, leading to an abort.
        """
        mocker.patch("select.select", return_value=(None, None, None))
        with pytest.raises(SystemExit) as exit_info:
            Cluster(**cluster_kwargs, overwrite_existing_experiment=False)
        assert exit_info.value.code == 1

    def test_empty_prompt_input_for_existing_experiment_dir(
        self, cluster_kwargs, mocker, _create_experiment_dir
    ):
        """Test cluster init when not overwriting experiment dir via flag.

        Since the experiment directory already exists, the scheduler
        prompts the user for input. In this test case, the user provides
        empty input, leading to an abort.
        """
        mocker.patch("select.select", return_value=(True, None, None))
        mocker.patch("sys.stdin.readline", return_value="")
        with pytest.raises(SystemExit) as exit_info:
            Cluster(**cluster_kwargs, overwrite_existing_experiment=False)
        assert exit_info.value.code == 1

    @pytest.mark.parametrize("user_input", ["y", "yes"])
    def test_y_prompt_input_for_existing_experiment_dir(
        self, cluster_kwargs, mocker, user_input, _create_experiment_dir
    ):
        """Test cluster init when not overwriting experiment dir via flag.

        Since the experiment directory already exists, the scheduler
        prompts the user for input. In this test case, the user provides
        the input 'y' or 'yes', allowing the initialization to proceed.
        """
        mocker.patch("select.select", return_value=(True, None, None))
        mocker.patch("sys.stdin.readline", return_value=user_input)
        Cluster(**cluster_kwargs, overwrite_existing_experiment=False)

    def test_fourc_mc_cluster(
        self,
        third_party_inputs,
        cluster_settings,
        cluster_kwargs,
        remote_connection,
        fourc_cluster_path,
        fourc_example_expected_output,
        global_settings,
    ):
        """Test remote 4C simulations with DASK jobqueue and MC iterator.

        Test for remote 4C simulations on a remote cluster in combination
        with
        - DASK jobqueue cluster
        - Monte-Carlo (MC) iterator
        - 4C ensight data-processor.


        Args:
            third_party_inputs (Path): Path to the 4C input files
            cluster_settings (dict): Cluster settings
            cluster_kwargs (dict): Keyword arguments to initialize the cluster scheduler
            remote_connection (RemoteConnection): Remote connection object
            fourc_cluster_path (Path): paths to 4C executable on the cluster
            fourc_example_expected_output (np.ndarray): Expected output for the MC samples
            global_settings (GlobalSettings): object containing experiment name and tmp_path
        """
        fourc_input_file_template = third_party_inputs / "fourc" / "solid_runtime_hex8.4C.yaml"

        # Parameters
        parameter_1 = Uniform(lower_bound=0.0, upper_bound=1.0)
        parameter_2 = Uniform(lower_bound=0.0, upper_bound=1.0)
        parameters = Parameters(parameter_1=parameter_1, parameter_2=parameter_2)

        data_processor = PvdFile(
            field_name="displacement",
            file_name_identifier="output-structure.pvd",
            file_options_dict={},
        )

        scheduler = Cluster(**cluster_kwargs)

        driver = Jobscript(
            parameters=parameters,
            input_templates=fourc_input_file_template,
            jobscript_template=cluster_settings["jobscript_template"],
            executable=fourc_cluster_path,
            data_processor=data_processor,
            extra_options={"cluster_script": cluster_settings["cluster_script_path"]},
        )
        model = Simulation(scheduler=scheduler, driver=driver)
        iterator = MonteCarlo(
            seed=42,
            num_samples=2,
            result_description={"write_results": True, "plot_results": False},
            model=model,
            parameters=parameters,
            global_settings=global_settings,
        )

        # Actual analysis
        run_iterator(iterator, global_settings=global_settings)

        # Load results
        results = load_result(global_settings.result_file(".pickle"))

        # The data has to be deleted before the assertion
        self.delete_old_simulation_data(remote_connection)

        # assert statements
        np.testing.assert_array_almost_equal(
            results["raw_output_data"]["result"], fourc_example_expected_output, decimal=6
        )

    @staticmethod
    @testbook(
        "tutorials/3_grid_iterator_fourc_remote.ipynb",
    )
    def test_fourc_remote_tutorial(
        tb, tmp_path, test_name, cluster_settings, fourc_cluster_path, basic_cluster_kwargs
    ):
        """Test for tutorial 3: Remote 4C simulation with grid iterator.

        The notebook is run with injected lines of code to replace placeholders.
        It is checked that the replaced dict entries already exist in the notebook.
        """
        jobscript_driver_kwargs = {
            "jobscript_template": cluster_settings["jobscript_template"],
            "executable": fourc_cluster_path,
            "extra_options": {"cluster_script": cluster_settings["cluster_script_path"]},
        }
        remote_connection_kwargs = {
            "host": cluster_settings["host"],
            "user": cluster_settings["user"],
            "remote_python": cluster_settings["remote_python"],
            "remote_queens_repository": cluster_settings["remote_queens_repository"],
            "gateway": cluster_settings["gateway"],
        }

        kwargs_dicts = [jobscript_driver_kwargs, remote_connection_kwargs, basic_cluster_kwargs]
        dict_names = [
            "jobscript_driver_kwargs",
            "remote_connection_kwargs",
            "cluster_scheduler_kwargs",
        ]

        injected_cell = """from pathlib import PosixPath"""

        for kwargs_dict, dict_name in zip(kwargs_dicts, dict_names):
            dict_name_injected = f"{dict_name}_injected"
            injected_cell += f"""
{dict_name_injected} = {kwargs_dict}
if not {dict_name}.keys() == {dict_name_injected}.keys():
    raise KeyError(
        f"The keys of the injected dictionary are not the same as the keys of the "
        f"placeholder dictionary in the notebook.\\n"
        f"Injected keys: {{{dict_name_injected}.keys()}}\\n"
        f"Placeholder keys: {{{dict_name}.keys()}}"
    )
{dict_name} = {dict_name_injected}
            """

        # replace placeholder dicts
        tb.inject(injected_cell, after=6, run=False)
        # replace experiment name and output dir
        tb.inject(
            f"experiment_name = {test_name!r}\noutput_dir = {tmp_path!r}",
            after=8,
            run=False,
        )
        # assert expected output
        tb.inject(
            "np.testing.assert_allclose(max_displacement_magnitude_per_run, "
            "[0.17606783, 0.22969808, 0.27944426, 0.22969808, 0.2782447,  0.32395894, 0.27944426, "
            "0.32395894, 0.36635981])",
            after=14,
            run=False,
        )

        # run the notebook
        tb.execute()

    def delete_old_simulation_data(self, remote_connection):
        """Delete old simulation data on the cluster.

        This approach deletes test simulation data older than seven days.

        Args:
            remote_connection (RemoteConnection): connection to remote cluster.
        """
        # Delete data from tests older than 1 week
        command = (
            "find "
            + str(self.pytest_base_directory_on_cluster())
            + " -mindepth 1 -maxdepth 1 -mtime +7 -type d -exec rm -rv {} \\;"
        )
        result = remote_connection.run(command, in_stream=False)
        _logger.debug("Deleting old simulation data:\n%s", result.stdout)


@dataclass(frozen=True)
class ClusterConfig:
    """Configuration data of cluster.

    Attributes:
        name (str):                         name of cluster
        host (str):                         hostname or ip address to reach cluster from network
        workload_manager (str):             type of work load scheduling software (PBS or SLURM)
        jobscript_template (Path):          absolute path to jobscript template file
        cluster_internal_address (str)      ip address of login node in cluster internal network
        default_python_path (str):          path indicating the default remote python location
        cluster_script_path (Path):          path to the cluster_script which defines functions
                                            needed for the jobscript
        queue (str, opt):                   Destination queue for each worker job
    """

    name: str
    host: str
    workload_manager: str
    jobscript_template: Path
    cluster_internal_address: str | None
    default_python_path: str
    cluster_script_path: Path
    queue: str | None = None

    dict = asdict


THOUGHT_CONFIG = ClusterConfig(
    name="thought",
    host="129.187.58.22",
    workload_manager="slurm",
    queue="normal",
    jobscript_template=relative_path_from_root("templates/jobscripts/fourc_thought.sh"),
    cluster_internal_address=None,
    default_python_path="$HOME/anaconda/miniconda/envs/queens/bin/python",
    cluster_script_path=Path("/lnm/share/donottouch.sh"),
)


BRUTEFORCE_CONFIG = ClusterConfig(
    name="bruteforce",
    host="bruteforce.lnm.ed.tum.de",
    workload_manager="slurm",
    jobscript_template=relative_path_from_root("templates/jobscripts/fourc_bruteforce.sh"),
    cluster_internal_address="10.10.0.1",
    default_python_path="$HOME/anaconda/miniconda/envs/queens/bin/python",
    cluster_script_path=Path("/lnm/share/donottouch.sh"),
)
CHARON_CONFIG = ClusterConfig(
    name="charon",
    host="charon.bauv.unibw-muenchen.de",
    workload_manager="slurm",
    jobscript_template=relative_path_from_root("templates/jobscripts/fourc_charon.sh"),
    cluster_internal_address="192.168.2.253",
    default_python_path="$HOME/miniconda3/envs/queens/bin/python",
    cluster_script_path=Path(),
)

CLUSTER_CONFIGS = {
    THOUGHT_CLUSTER_TYPE: THOUGHT_CONFIG,
    BRUTEFORCE_CLUSTER_TYPE: BRUTEFORCE_CONFIG,
    CHARON_CLUSTER_TYPE: CHARON_CONFIG,
}


# CLUSTER TESTS ------------------------------------------------------------------------------------
@pytest.fixture(name="user", scope="session")
def fixture_user():
    """Name of user calling the test suite."""
    return getpass.getuser()


@pytest.fixture(name="remote_user", scope="session")
def fixture_remote_user(pytestconfig):
    """Name of cluster account user used in tests."""
    return pytestconfig.getoption("remote_user")


@pytest.fixture(name="gateway", scope="session")
def fixture_gateway(pytestconfig):
    """Gateway connection (proxyjump)."""
    gateway = pytestconfig.getoption("gateway")
    if isinstance(gateway, str):
        gateway = json.loads(gateway)
    return gateway


@pytest.fixture(name="cluster", scope="session")
def fixture_cluster(request):
    """Name of the cluster to run a test on.

    The actual parameterization is done on a per test basis which also
    defines the parameterized markers of the tests.
    """
    return request.param


@pytest.fixture(name="cluster_settings", scope="session")
def fixture_cluster_settings(
    cluster, remote_user, gateway, remote_python, remote_queens_repository
):
    """All cluster settings."""
    settings = CLUSTER_CONFIGS.get(cluster).dict()
    _logger.debug("raw cluster config: %s", settings)
    settings["cluster"] = cluster
    settings["user"] = remote_user
    settings["remote_python"] = remote_python
    settings["remote_queens_repository"] = remote_queens_repository
    settings["gateway"] = gateway
    return settings


@pytest.fixture(name="remote_python", scope="session")
def fixture_remote_python(pytestconfig):
    """Path to the Python environment on remote host."""
    return pytestconfig.getoption("remote_python")


@pytest.fixture(name="remote_connection", scope="session")
def fixture_remote_connection(cluster_settings):
    """A fabric connection to a remote host."""
    return RemoteConnection(
        host=cluster_settings["host"],
        user=cluster_settings["user"],
        remote_python=cluster_settings["remote_python"],
        remote_queens_repository=cluster_settings["remote_queens_repository"],
        gateway=cluster_settings["gateway"],
    )


@pytest.fixture(name="remote_queens_repository", scope="session")
def fixture_remote_queens_repository(pytestconfig):
    """Path to the queens repository on remote host."""
    remote_queens = pytestconfig.getoption("remote_queens_repository", skip=True)
    return remote_queens


@pytest.fixture(name="fourc_cluster_path", scope="session")
def fixture_fourc_cluster_path(remote_connection):
    """Paths to 4C executable on the clusters.

    Checks also for existence of the executable.
    """
    result = remote_connection.run("echo ~", in_stream=False)
    remote_home = Path(result.stdout.rstrip())

    fourc = fourc_build_path_from_home(remote_home)

    # Check for existence of 4C on remote machine.
    find_result = remote_connection.run(f"find {fourc}", in_stream=False)
    Path(find_result.stdout.rstrip())

    return fourc
