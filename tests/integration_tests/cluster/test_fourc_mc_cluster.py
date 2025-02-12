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

import logging
from pathlib import Path

import numpy as np
import pytest

import queens.schedulers.cluster_scheduler as cluster_scheduler  # pylint: disable=consider-using-from-import
from queens.data_processors.pvd import Pvd
from queens.distributions.uniform import Uniform
from queens.drivers import Jobscript
from queens.iterators.monte_carlo import MonteCarlo
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.utils import config_directories
from queens.utils.io_utils import load_result
from tests.integration_tests.conftest import (  # BRUTEFORCE_CLUSTER_TYPE,
    CHARON_CLUSTER_TYPE,
    THOUGHT_CLUSTER_TYPE,
)

_logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "cluster",
    [
        pytest.param(THOUGHT_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        # pytest.param(BRUTEFORCE_CLUSTER_TYPE, marks=pytest.mark.lnm_cluster),
        pytest.param(CHARON_CLUSTER_TYPE, marks=pytest.mark.imcs_cluster),
    ],
    indirect=True,
)
class TestDaskCluster:
    """Test class collecting all test with Dask jobqueue clusters and 4C.

    NOTE: we use a class here since our fixture are set to autouse, but we only want to call them
    for these tests.
    """

    @pytest.fixture(autouse=True)
    def mock_experiment_dir(self, monkeypatch, cluster_settings, patched_base_directory):
        """Mock the experiment directory on the cluster.

        The goal is to separate the testing data from production data of the user.
        NOTE: It is necessary to mock the whole experiment_directory method.
        Otherwise, the mock is not loaded properly remote.
        This is in contrast to the local mocking where it suffices to mock
        config_directories.EXPERIMENTS_BASE_FOLDER_NAME.
        Note that we also rely on this local mock here!
        """

        def patch_experiments_directory(experiment_name):
            """Base directory for all experiments on the computing machine."""
            experiments_dir = (
                Path(patched_base_directory.replace("$HOME", str(Path.home()))) / experiment_name
            )
            Path.mkdir(experiments_dir, parents=True, exist_ok=True)
            return experiments_dir

        monkeypatch.setattr(cluster_scheduler, "experiment_directory", patch_experiments_directory)
        _logger.debug("Mocking of dask experiment_directory  was successful.")
        _logger.debug(
            "dask experiment_directory is mocked to '%s/<experiment_name>' on %s@%s",
            patched_base_directory,
            cluster_settings["user"],
            cluster_settings["host"],
        )

    def experiment_dir_on_cluster(self):
        """Remote experiment path."""
        return (
            f"$HOME/{config_directories.BASE_DATA_DIR}/{config_directories.TESTS_BASE_FOLDER_NAME}"
        )

    @pytest.fixture(name="patched_base_directory")
    def fixture_patched_base_directory(self, pytest_id):
        """Path of the tests on the remote cluster."""
        return self.experiment_dir_on_cluster() + f"/{pytest_id}"

    def test_fourc_mc_cluster(
        self,
        third_party_inputs,
        cluster_settings,
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
            remote_connection (RemoteConnection): Remote connection object
            fourc_cluster_path (Path): paths to 4C executable on the cluster
            fourc_example_expected_output (np.ndarray): Expected output for the MC samples
            global_settings (GlobalSettings): object containing experiment name and tmp_path
        """
        fourc_input_file_template = third_party_inputs / "fourc" / "solid_runtime_hex8.dat"

        # Parameters
        parameter_1 = Uniform(lower_bound=0.0, upper_bound=1.0)
        parameter_2 = Uniform(lower_bound=0.0, upper_bound=1.0)
        parameters = Parameters(parameter_1=parameter_1, parameter_2=parameter_2)

        data_processor = Pvd(
            field_name="displacement",
            file_name_identifier="*.pvd",
            file_options_dict={},
        )

        scheduler = cluster_scheduler.ClusterScheduler(
            workload_manager=cluster_settings["workload_manager"],
            walltime="00:10:00",
            num_jobs=1,
            min_jobs=1,
            num_procs=1,
            num_nodes=1,
            remote_connection=remote_connection,
            cluster_internal_address=cluster_settings["cluster_internal_address"],
            experiment_name=global_settings.experiment_name,
            queue=cluster_settings.get("queue"),
        )

        driver = Jobscript(
            parameters=parameters,
            input_templates=fourc_input_file_template,
            jobscript_template=cluster_settings["jobscript_template"],
            executable=fourc_cluster_path,
            data_processor=data_processor,
            extra_options={"cluster_script": cluster_settings["cluster_script_path"]},
        )
        model = SimulationModel(scheduler=scheduler, driver=driver)
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
        self.delete_simulation_data(remote_connection)

        # assert statements
        np.testing.assert_array_almost_equal(
            results["raw_output_data"]["result"], fourc_example_expected_output, decimal=6
        )

    def delete_simulation_data(self, remote_connection):
        """Delete simulation data on the cluster.

        This approach deletes test simulation data older than seven days
        Args:
            remote_connection (RemoteConnection): connection to remote cluster.
        """
        # Delete data from tests older then 1 week
        command = (
            "find "
            + str(self.experiment_dir_on_cluster())
            + " -mtime +7 -mindepth 1 -maxdepth 1 -type d -exec rm -rv {} \\;"
        )
        result = remote_connection.run(command, in_stream=False)
        _logger.debug("Deleting old simulation data:\n%s", result.stdout)
