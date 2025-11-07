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
"""Configuration module for the entire test suite (highest level)."""

import getpass
import logging
import socket
from time import perf_counter

import numpy as np
import pytest

from queens.global_settings import GlobalSettings
from queens.utils import config_directories
from queens.utils.logger_settings import reset_logging
from queens.utils.path import relative_path_from_root

_logger = logging.getLogger(__name__)


NAME_OF_HOST = socket.gethostname()


def pytest_addoption(parser):
    """Add pytest options."""
    # default remote_user is same as local_user
    local_user = getpass.getuser()
    parser.addoption("--remote-user", action="store", default=local_user)
    parser.addoption("--remote-python", action="store", default=None)
    parser.addoption("--remote-queens-repository", action="store", default="null")
    parser.addoption("--pipeline-id", action="store", default=None)
    parser.addoption(
        "--test-timing",
        action="store_true",
        default=False,
        help="Turn on test timing, so exceptions are raised if tests are too slow. To change "
        "the maximum test time use @pytest.marker.max_time_for_test(time_in_seconds)",
    )
    parser.addoption(
        "--gateway",
        action="store",
        default=None,
        help=(
            "gateway connection (proxyjump) for remote connection in json format,"
            ' e.g. \'{"host": "user@host"}\''
        ),
    )


def check_item_for_marker(item, marker_name):
    """Check if item is marked with marker_name.

    Args:
        item (pytest): pytest.item object
        marker_name (str): Name of the marker to check

    Returns:
        bool: True if the test is marked
    """
    return marker_name in [mark.name for mark in item.own_markers]


def pytest_collection_finish(session):
    """Print information after collection of tests."""
    # After all the tests are collected we give some information for our test setup
    if len(session.items) > 0:
        # pylint: disable-next=protected-access
        tmp_path = session.items[0].config._tmp_path_factory.getbasetemp()
        # Use print as otherwise logger will not necessarily be shown in terminal
        print(f"\nThe pytest directory for this test run is {tmp_path}")

    if pipeline_id := session.config.getoption("--pipeline-id"):
        print(f"The pipeline id for this test run is {pipeline_id}")

    # Options to be displayed in the terminal
    options_dict = {
        # Options for remote tests
        "For remote cluster tests": [
            "--remote-user",
            "--remote-python",
            "--remote-queens-repository",
            "--gateway",
        ],
        # Other options
        "Other options": ["--test-timing", "--pipeline-id"],
    }

    print("\nThe following QUEENS pytest options are used (if needed):")
    for options_section, options_list in options_dict.items():
        print("\n  " + options_section)
        for option in options_list:
            option_value = session.config.getoption(option)
            print(f"     {option}: {option_value}")


def pytest_collection_modifyitems(items):
    """Automatically add pytest markers based on testpath."""
    for item in items:
        if "benchmarks/" in item.nodeid:
            item.add_marker(pytest.mark.benchmark)
        elif "integration_tests/fourc/" in item.nodeid:
            item.add_marker(pytest.mark.integration_tests_fourc)

            # Add default max_time_for_test if none was set
            if not check_item_for_marker(item, "max_time_for_test"):
                item.add_marker(pytest.mark.max_time_for_test(10))
        elif "integration_tests/cluster/" in item.nodeid:
            # Pytest markers are set individually in each cluster integration test
            continue
        elif "integration_tests/" in item.nodeid:
            item.add_marker(pytest.mark.integration_tests)

            # Add default max_time_for_test if none was set
            if not check_item_for_marker(item, "max_time_for_test"):
                item.add_marker(pytest.mark.max_time_for_test(10))
        elif "unit_tests/" in item.nodeid:
            item.add_marker(pytest.mark.unit_tests)

            # Add default max_time_for_test if none was set
            if not check_item_for_marker(item, "max_time_for_test"):
                item.add_marker(pytest.mark.max_time_for_test(2))


@pytest.fixture(name="time_tests", autouse=True, scope="function")
def fixture_time_tests(request):
    """Time tests if desired."""
    # Check if test timing is on
    if request.config.getoption("--test-timing"):
        # Measure time
        start_time = perf_counter()
        yield
        total_time = perf_counter() - start_time

        # Check if max_time is provided
        if "max_time_for_test" in request.keywords:
            max_time = request.keywords["max_time_for_test"].args[0]

            if total_time > max_time:
                raise TimeoutError(
                    f"Test exceeded time constraint {total_time:.03f}s > {max_time}s"
                )
    else:
        # Do nothing
        yield


@pytest.fixture(name="hostname", scope="session")
def fixture_hostname(name_of_host=NAME_OF_HOST):
    """Hostname calling the test suite."""
    _logger.debug("Tests are run on: %s", name_of_host)
    return name_of_host


@pytest.fixture(name="pytest_id", autouse=True, scope="session")
def fixture_pytest_id(tmp_path_factory):
    """Pipeline id."""
    pytest_tests_path = tmp_path_factory.getbasetemp()
    pytest_run_id = pytest_tests_path.name
    return pytest_run_id


@pytest.fixture(name="global_mock_local_base_dir", autouse=True)
def fixture_global_mock_local_base_dir(monkeypatch, tmp_path):
    """Mock the local base directory for all tests.

    This is necessary to keep the base directory of a user clean from
    testing data. pytest temp_path supplies a perfect location for this
    (see pytest docs).
    """

    def mock_local_base_dir():
        return tmp_path

    monkeypatch.setattr(config_directories, "base_directory", mock_local_base_dir)
    _logger.debug("Mocking of local base dir was successful.")
    _logger.debug("local base dir is mocked to: %s", config_directories.base_directory())


@pytest.fixture(name="inputdir", scope="session")
def fixture_inputdir():
    """Return the path to the json input-files of the function test."""
    input_files_path = relative_path_from_root("tests/input_files/queens")
    return input_files_path


@pytest.fixture(name="third_party_inputs", scope="session")
def fixture_third_party_inputs():
    """Return the path to the json input-files of the function test."""
    input_files_path = relative_path_from_root("tests/input_files/third_party")
    return input_files_path


@pytest.fixture(name="config_dir", scope="session")
def fixture_config_dir():
    """Return the path to the json input-files of the function test."""
    config_dir_path = relative_path_from_root("config")
    return config_dir_path


@pytest.fixture(name="fourc_link_paths", scope="session")
def fixture_fourc_link_paths(config_dir):
    """Set symbolic links for 4C on testing machine."""
    fourc_build_dir = config_dir / "4C_build"
    fourc = fourc_build_dir / "4C"
    post_ensight = fourc_build_dir / "post_ensight"
    post_processor = fourc_build_dir / "post_processor"
    return fourc, post_ensight, post_processor


@pytest.fixture(name="example_simulator_fun_dir", scope="session")
def fixture_example_simulator_fun_dir():
    """Return the path to the example simulator functions."""
    input_files_path = relative_path_from_root("src/example_simulator_functions")
    return input_files_path


@pytest.fixture(name="ref_result_iterator")
def fixture_ref_result_iterator():
    """Reference results of an iterator."""
    ref_result_iterator = np.array(
        [
            [1.7868040337],
            [-13.8624183835],
            [6.3423271929],
            [6.1674472752],
            [5.3528917433],
            [-0.7472766806],
            [5.0007066283],
            [6.4763926539],
            [-6.4173504897],
            [3.1739282221],
        ]
    )
    return ref_result_iterator


def pytest_sessionfinish():
    """Register a hook to suppress logging errors after the session."""
    logging.raiseExceptions = False


@pytest.fixture(name="reset_loggers", autouse=True)
def fixture_reset_loggers():
    """Reset loggers.

    This fixture is called at every test due to `autouse=True`. It acts
    as a generator and allows us to close all loggers after each test.
    This should avoid duplicate logger output.
    """
    # Do the test.
    yield

    # Test is done, now reset the loggers.
    reset_logging()


@pytest.fixture(name="test_name")
def fixture_test_name(request):
    """Return the test's name."""
    test_name = request.node.name
    _logger.info("Name of current test: %s", test_name)
    _logger.info("the above name will be used as the experiment_name")
    return test_name


@pytest.fixture(name="global_settings")
def fixture_global_settings(test_name, tmp_path):
    """Initialize GlobalSettings object."""
    # Setup and initialize global settings
    global_settings = GlobalSettings(experiment_name=test_name, output_dir=tmp_path)

    # wrap the tests in a global settings context
    with global_settings:
        yield global_settings


@pytest.fixture(name="disable_matplot_show", autouse=True)
def fixture_disable_matplotlib_show():
    """Do not show plots in matplotlib gui during testing.

    Note that the plots are still saved during testing.
    """
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
