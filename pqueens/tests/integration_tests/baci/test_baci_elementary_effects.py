"""Test suite for integration tests for the Morris-Salib Iterator.

Estimate Elementary Effects for local simulations with BACI using the
INVAAA minimal model.
"""

import json
import os
import pickle

import numpy as np
import pytest

from pqueens.main import main
from pqueens.utils import injector
from pqueens.utils.run_subprocess import run_subprocess


@pytest.fixture(scope="session")
def output_directory_forward(tmpdir_factory):
    """Create two temporary output directories for test runs with singularity.

    with singularity (<...>_true) and without singularity (<...>_false)

    Args:
        tmpdir_factory: fixture used to create arbitrary temporary directories

    Returns:
        output_directory_forward (dict): temporary output directories for simulation without and
                                         with singularity
    """
    path_singularity_true = tmpdir_factory.mktemp("test_baci_elementary_effects_true")
    path_singularity_false = tmpdir_factory.mktemp("test_baci_elementary_effects_false")

    return {True: path_singularity_true, False: path_singularity_false}


@pytest.fixture()
def experiment_directory(output_directory_forward, singularity_bool):
    """Return experiment directory depending on singularity_bool.

    Returns:
        experiment_directory (LocalPath): experiment directory depending on singularity_bool
    """
    return output_directory_forward[singularity_bool]


@pytest.fixture()
def check_experiment_directory(experiment_directory):
    """Check if experiment directory contains subdirectories.

    Raises:
        AssertionError: If experiment directory does not contain subdirectories.
    """
    number_subdirectories = count_subdirectories(experiment_directory)

    assert (
        number_subdirectories != 0
    ), "Empty output directory. Run test_baci_elementary_effects first."


def count_subdirectories(current_directory):
    """Count subdirectories in current_directory.

    Returns:
        number_subdirectories (int): number of subdirectories
    """
    number_subdirectories = 0
    for current_subdirectory in os.listdir(current_directory):
        path_current_subdirectory = os.path.join(current_directory, current_subdirectory)
        if os.path.isdir(path_current_subdirectory):
            number_subdirectories += 1
    return number_subdirectories


def remove_job_output_directory(experiment_directory, jobid):
    """Remove output directory of job #jobid from experiment_directory."""
    rm_cmd = "rm -r " + str(experiment_directory) + "/" + str(jobid)
    run_subprocess(rm_cmd)


@pytest.mark.integration_tests_baci
def test_baci_elementary_effects(
    inputdir, third_party_inputs, baci_link_paths, singularity_bool, experiment_directory
):
    """Integration test for the Elementary Effects Iterator together with BACI.

    The test runs a local native BACI simulation as well as a local Singularity
    based BACI simulation for elementary effects.

    Args:
        inputdir (str): Path to the JSON input file
        third_party_inputs (str): Path to the BACI input files
        baci_link_paths(str): Path to the links pointing to baci-release and post_drt_monitor
        singularity_bool (str): String that encodes a boolean that is parsed to the JSON input file
        experiment_directory (LocalPath): experiment directory depending on singularity_bool

    Returns:
        None
    """
    template = os.path.join(inputdir, "baci_local_elementary_effects_template.json")
    input_file = os.path.join(experiment_directory, "elementary_effects_baci_local_invaaa.json")
    third_party_input_file = os.path.join(third_party_inputs, "baci_input_files", "invaaa_ee.dat")
    experiment_name = "ee_invaaa_local_singularity_" + json.dumps(singularity_bool)

    baci_release, post_drt_monitor, _, _ = baci_link_paths

    dir_dict = {
        'experiment_name': experiment_name,
        'experiment_dir': str(experiment_directory),
        'baci_input': third_party_input_file,
        'baci-release': baci_release,
        'post_drt_monitor': post_drt_monitor,
        'singularity_boolean': json.dumps(singularity_bool),
    }

    injector.inject(dir_dict, template, input_file)
    arguments = ['--input=' + input_file, '--output=' + str(experiment_directory)]

    main(arguments)

    result_file_name = experiment_name + ".pickle"
    result_file = os.path.join(experiment_directory, result_file_name)
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # test results of SA analysis
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu"], np.array([-1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star"], np.array([1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["sigma"], np.array([0.198629, 0.198629]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star_conf"], np.array([0.136631, 0.140794]), rtol=1.0e-3
    )


@pytest.mark.integration_tests_baci
def test_restart_from_output_folders_baci(
    inputdir,
    tmpdir,
    third_party_inputs,
    baci_link_paths,
    singularity_bool,
    experiment_directory,
    check_experiment_directory,
):
    """Integration test for the restart functionality.

     Test restart from results in output folders.

    - test with and without singularity
    - drop_database_boolean = true

    Restart based on results of previous tests "test_baci_elementary_effects".

    Args:
        inputdir (str): Path to the JSON input file
        tmpdir (str): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        baci_link_paths(str): Path to the links pointing to baci-release and post_drt_monitor
        singularity_bool (str): String that encodes a boolean that is parsed to the JSON input file
        experiment_directory (LocalPath): experiment directory depending on singularity_bool
        check_experiment_directory: Check if experiment directory contains subdirectories

    Returns:
        None
    """
    template = os.path.join(inputdir, "baci_local_restart_elementary_effects_template.json")
    input_file = os.path.join(tmpdir, "elementary_effects_baci_local_invaaa_restart.json")
    third_party_input_file = os.path.join(third_party_inputs, "baci_input_files", "invaaa_ee.dat")
    experiment_name = "ee_invaaa_local_singularity_" + json.dumps(singularity_bool)

    baci_release, post_drt_monitor, _, _ = baci_link_paths

    dir_dict = {
        'experiment_name': experiment_name,
        'experiment_dir': str(experiment_directory),
        'baci_input': third_party_input_file,
        'baci-release': baci_release,
        'post_drt_monitor': post_drt_monitor,
        'singularity_boolean': json.dumps(singularity_bool),
        'drop_database_boolean': "true",
    }

    injector.inject(dir_dict, template, input_file)
    arguments = ['--input=' + input_file, '--output=' + str(tmpdir)]

    main(arguments)

    result_file_name = experiment_name + ".pickle"
    result_file = os.path.join(tmpdir, result_file_name)
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # test results of SA analysis
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu"], np.array([-1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star"], np.array([1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["sigma"], np.array([0.198629, 0.198629]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star_conf"], np.array([0.136631, 0.140794]), rtol=1.0e-3
    )


@pytest.mark.integration_tests_baci
def test_block_restart_baci(
    inputdir, tmpdir, third_party_inputs, baci_link_paths, output_directory_forward
):
    """Integration test for the block-restart functionality.

     Delete last results and block-restart those results (only without singularity).

    Args:
        inputdir (str): Path to the JSON input file
        tmpdir (str): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        baci_link_paths(str): Path to the links pointing to baci-release and post_drt_monitor
        output_directory_forward (dict): paths to output directory for test with and without
                                         singularity

    Returns:
        None
    """
    # Test without singularity:
    singularity_bool = False
    output_directory = output_directory_forward[singularity_bool]
    number_of_output_directories = count_subdirectories(output_directory)
    experiment_name = "ee_invaaa_local_singularity_" + json.dumps(singularity_bool)

    # Delete last results
    for jobid in range(number_of_output_directories - 4, number_of_output_directories + 1):
        remove_job_output_directory(output_directory, jobid)

    template = os.path.join(inputdir, "baci_local_restart_elementary_effects_template.json")
    input_file = os.path.join(tmpdir, "elementary_effects_baci_local_invaaa_restart.json")
    third_party_input_file = os.path.join(third_party_inputs, "baci_input_files", "invaaa_ee.dat")

    baci_release, post_drt_monitor, _, _ = baci_link_paths

    dir_dict = {
        'experiment_name': experiment_name,
        'experiment_dir': str(output_directory),
        'baci_input': third_party_input_file,
        'baci-release': baci_release,
        'post_drt_monitor': post_drt_monitor,
        'singularity_boolean': json.dumps(singularity_bool),
        'drop_database_boolean': "true",
    }

    injector.inject(dir_dict, template, input_file)
    arguments = ['--input=' + input_file, '--output=' + str(tmpdir)]

    main(arguments)

    result_file_name = experiment_name + ".pickle"
    result_file = os.path.join(tmpdir, result_file_name)
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # test results of SA analysis
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu"], np.array([-1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star"], np.array([1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["sigma"], np.array([0.198629, 0.198629]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star_conf"], np.array([0.136631, 0.140794]), rtol=1.0e-3
    )


@pytest.mark.integration_tests_baci
def test_restart_from_db_baci(
    inputdir, tmpdir, third_party_inputs, baci_link_paths, output_directory_forward
):
    """Integration test for the restart functionality.

    Test restart from results in database.

    - test only without singularity
    - drop_database_boolean = false

    Args:
        inputdir (str): Path to the JSON input file
        tmpdir (str): Temporary directory in which the pytests are run
        third_party_inputs (str): Path to the BACI input files
        baci_link_paths(str): Path to the links pointing to baci-release and post_drt_monitor
        output_directory_forward (dict): paths to output directory for test with and without
                                         singularity

    Returns:
        None
    """
    # This test itself does not submit jobs and thus does not depend on singularity.
    # Set singularity_boolean for database reference only.
    singularity_bool = False
    output_directory = output_directory_forward[singularity_bool]
    experiment_name = "ee_invaaa_local_singularity_" + json.dumps(singularity_bool)

    # Find number of output directories from previous run and delete all output folders:
    # Make sure this test fails when results are not found in database.
    number_of_output_directories = count_subdirectories(output_directory)
    for jobid in range(1, number_of_output_directories + 1):
        remove_job_output_directory(output_directory, jobid)

    template = os.path.join(inputdir, "baci_local_restart_elementary_effects_template.json")
    input_file = os.path.join(tmpdir, "elementary_effects_baci_local_invaaa_restart.json")
    third_party_input_file = os.path.join(third_party_inputs, "baci_input_files", "invaaa_ee.dat")

    baci_release, post_drt_monitor, _, _ = baci_link_paths

    dir_dict = {
        'experiment_name': experiment_name,
        'experiment_dir': str(output_directory),
        'baci_input': third_party_input_file,
        'baci-release': baci_release,
        'post_drt_monitor': post_drt_monitor,
        'singularity_boolean': json.dumps(singularity_bool),
        'drop_database_boolean': "false",
    }

    injector.inject(dir_dict, template, input_file)
    arguments = ['--input=' + input_file, '--output=' + str(tmpdir)]

    main(arguments)

    result_file_name = experiment_name + ".pickle"
    result_file = os.path.join(tmpdir, result_file_name)
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # test results of SA analysis
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu"], np.array([-1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star"], np.array([1.361395, 0.836351]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["sigma"], np.array([0.198629, 0.198629]), rtol=1.0e-3
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star_conf"], np.array([0.136631, 0.140794]), rtol=1.0e-3
    )