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
"""Collect fixtures used by the integration tests."""

import getpass
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from example_simulator_functions.currin88 import currin88_hifi, currin88_lofi
from example_simulator_functions.park91a import X3, X4, park91a_hifi_on_grid
from queens.utils.path import relative_path_from_root
from queens.utils.pdf_estimation import estimate_bandwidth_for_kde
from queens.utils.process_outputs import write_results
from queens.utils.remote_operations import RemoteConnection
from test_utils.integration_tests import fourc_build_paths_from_home

_logger = logging.getLogger(__name__)

THOUGHT_CLUSTER_TYPE = "thought"
BRUTEFORCE_CLUSTER_TYPE = "bruteforce"
CHARON_CLUSTER_TYPE = "charon"


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

    fourc, _, _ = fourc_build_paths_from_home(remote_home)

    # Check for existence of 4C on remote machine.
    find_result = remote_connection.run(f"find {fourc}", in_stream=False)
    Path(find_result.stdout.rstrip())

    return fourc


@pytest.fixture(name="fourc_example_expected_output")
def fixture_fourc_example_expected_output():
    """Expected outputs for the 4C example."""
    result = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [0.1195746416995907, -0.002800078802811129, -0.005486393250866545],
                [0.1260656705382511, -0.002839272898349505, -0.005591796485367413],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.1322571001660478, -0.00290530963354552, -0.005635750492708091],
                [0.1387400363966301, -0.002944141371541845, -0.005740445608910146],
                [0.0, 0.0, 0.0],
                [0.1195746416995907, -0.002800078802811129, -0.005486393250866545],
                [0.2289195764727486, -0.01428888900910762, -0.02789834740243489],
                [0.24879304060717, -0.01437712967153365, -0.02801932699697155],
                [0.1260656705382511, -0.002839272898349505, -0.005591796485367413],
                [0.1322571001660478, -0.00290530963354552, -0.005635750492708091],
                [0.2674182375147958, -0.01440529789560568, -0.02822643380369276],
                [0.2865938203259575, -0.01448421374089244, -0.02832919236100399],
                [0.1387400363966301, -0.002944141371541845, -0.005740445608910146],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.1324695606381951, -0.00626315166779166, -0.003933720977121313],
                [0.1472676510502675, -0.006473749929398404, -0.004119847415735578],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.1417895586931143, -0.006449559916104635, -0.004017410516711057],
                [0.1565760384270568, -0.006658631448567143, -0.004202575461905436],
                [0.0, 0.0, 0.0],
                [0.1324695606381951, -0.00626315166779166, -0.003933720977121313],
                [0.250425194294125, -0.0327911259093084, -0.02066877798205112],
                [0.2973138520084483, -0.03326225476238921, -0.02086982506734491],
                [0.1472676510502675, -0.006473749929398404, -0.004119847415735578],
                [0.1417895586931143, -0.006449559916104635, -0.004017410516711057],
                [0.2801976826134401, -0.03299992343589411, -0.0208604824244337],
                [0.3257952884928654, -0.03343065913125062, -0.02103608880631498],
                [0.1565760384270568, -0.006658631448567143, -0.004202575461905436],
            ],
        ]
    )
    return result


@pytest.fixture(name="_create_experimental_data_park91a_hifi_on_grid")
def fixture_create_experimental_data_park91a_hifi_on_grid(tmp_path):
    """Create a csv file with experimental data."""
    # Fix random seed
    np.random.seed(seed=1)

    # True input values
    x1 = 0.5
    x2 = 0.2

    y_vec = park91a_hifi_on_grid(x1, x2)

    # Artificial noise
    sigma_n = 0.001
    noise_vec = np.random.normal(loc=0, scale=sigma_n, size=(y_vec.size,))

    # Inverse crime: Add artificial noise to model output for the true value
    y_fake = y_vec + noise_vec

    # write fake data to csv
    data_dict = {
        "x3": X3,
        "x4": X4,
        "y_obs": y_fake,
    }
    experimental_data_path = tmp_path / "experimental_data.csv"
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)


# fixtures for Elementary Effects Sobol tests
@pytest.fixture(name="expected_result_mu")
def fixture_expected_result_mu():
    """Expected Mu result."""
    expected_result_mu = np.array(
        [
            25.8299150077341,
            19.28297176050532,
            -14.092164789704626,
            5.333475971922498,
            -11.385141403296364,
            13.970208961715421,
            -3.0950202483238303,
            0.6672725255532903,
            7.2385092339309445,
            -7.7664016980947075,
        ]
    )
    return expected_result_mu


@pytest.fixture(name="expected_result_mu_star")
def fixture_expected_result_mu_star():
    """Expected Mu star result."""
    expected_result_mu_star = np.array(
        [
            29.84594504725642,
            21.098173537614855,
            16.4727722348437,
            26.266876218598668,
            16.216603266281044,
            18.051629859410895,
            3.488313966697564,
            2.7128638920479147,
            7.671230484535577,
            10.299932289624746,
        ]
    )
    return expected_result_mu_star


@pytest.fixture(name="expected_result_sigma")
def fixture_expected_result_sigma():
    """Expected sigma result."""
    expected_result_sigma = np.array(
        [
            53.88783786787971,
            41.02192670857979,
            29.841807478998156,
            43.33349033575829,
            29.407676882180404,
            31.679653142831512,
            5.241491105224932,
            4.252334015139214,
            10.38274186974731,
            18.83046700807382,
        ]
    )
    return expected_result_sigma


# pylint: disable=invalid-name
# ---- fixtures for bmfmc tests----------------------------------------------------------------
@pytest.fixture(name="monte_carlo_samples_x")
def fixture_monte_carlo_samples_x():
    """1000 uniform Monte Carlo samples for x1 and x2 between 0 and 1."""
    np.random.seed(1)
    n_samples = 1000
    monte_carlo_samples_x = np.random.uniform(low=0.0, high=1.0, size=(n_samples, 2))
    return monte_carlo_samples_x


@pytest.fixture(name="lf_mc_data")
def fixture_lf_mc_data(monte_carlo_samples_x):
    """Samples of low-fidelity model output using currin88_lofi."""
    y = []
    for x_vec in monte_carlo_samples_x:
        params = {"x1": x_vec[0], "x2": x_vec[1]}
        y.append(currin88_lofi(**params))

    y_lf_mc = np.array(y).reshape((monte_carlo_samples_x.shape[0], -1))

    return y_lf_mc


@pytest.fixture(name="hf_mc_data")
def fixture_hf_mc_data(monte_carlo_samples_x):
    """Samples of high-fidelity model output using currin88_hifi."""
    y = []
    for x_vec in monte_carlo_samples_x:
        params = {"x1": x_vec[0], "x2": x_vec[1]}
        y.append(currin88_hifi(**params))

    y_lf_mc = np.array(y).reshape((monte_carlo_samples_x.shape[0], -1))

    return y_lf_mc


@pytest.fixture(name="bandwidth_lf_mc")
def fixture_bandwidth_lf_mc(lf_mc_data):
    """Estimated bandwidth for KDE for low-fidelity data."""
    bandwidth_lf_mc = estimate_bandwidth_for_kde(
        lf_mc_data[:, 0], np.amin(lf_mc_data[:, 0]), np.amax(lf_mc_data[:, 0])
    )
    return bandwidth_lf_mc


@pytest.fixture(name="_write_lf_mc_data_to_pickle")
def fixture_write_lf_mc_data_to_pickle(tmp_path, monte_carlo_samples_x, lf_mc_data):
    """Write low-fidelity model data to a pickle file."""
    file_name = "LF_MC_data.pickle"
    input_description = {
        "x1": {
            "type": "uniform",
            "lower_bound": 0.0,
            "upper_bound": 1.0,
        },
        "x2": {
            "type": "uniform",
            "lower_bound": 0.0,
            "upper_bound": 1.0,
        },
    }
    data = {
        "input_data": monte_carlo_samples_x,
        "input_description": input_description,
        "output": lf_mc_data,
        "eigenfunc": None,
        "eigenvalue": None,
    }
    write_results(data, tmp_path / file_name)


@pytest.fixture(name="design_method", params=["random", "diverse_subset"])
def fixture_design_method(request):
    """Different design methods for parameterized tests."""
    design = request.param
    return design


# ---- fixtures for bmfia tests-------------------------------------------
@pytest.fixture(name="expected_samples")
def fixture_expected_samples():
    """Expected SMC samples."""
    samples = np.array(
        [
            [0.51711296, 0.55200585],
            [0.4996905, 0.6673229],
            [0.48662203, 0.68802404],
            [0.49806929, 0.66276797],
            [0.49706481, 0.68586978],
            [0.50424704, 0.65139028],
            [0.51437955, 0.57678317],
            [0.51275639, 0.58981357],
            [0.50163956, 0.65389397],
            [0.52127371, 0.61237995],
        ]
    )

    return samples


@pytest.fixture(name="expected_weights")
def fixture_expected_weights():
    """Expected SMC weights."""
    weights = np.array(
        [
            0.00183521,
            0.11284748,
            0.16210619,
            0.07066473,
            0.10163831,
            0.09845534,
            0.10742886,
            0.15461861,
            0.09222745,
            0.0981778,
        ]
    )
    return weights


@pytest.fixture(name="expected_variational_cov")
def fixture_expected_variational_cov():
    """Expected variational covariance."""
    exp_var_cov = np.array([[0.00142648, 0.0], [0.0, 0.00347234]])
    return exp_var_cov


@pytest.fixture(name="expected_variational_mean_nn")
def fixture_expected_variational_mean_nn():
    """Expected variational mean."""
    exp_var_mean = np.array([0.19221321, 0.33134219]).reshape(-1, 1)

    return exp_var_mean


@pytest.fixture(name="expected_variational_cov_nn")
def fixture_expected_variational_cov_nn():
    """Expected variational covariance."""
    exp_var_cov = np.array([[0.01245263, 0.0], [0.0, 0.01393423]])
    return exp_var_cov


# pylint: enable=invalid-name
# add fixtures not related to bmfia or bmfmc below
