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
"""Unit tests for local schedulers with a (non-)existing experiment dir."""

import os

import pytest

from queens.schedulers import Local, Pool


@pytest.mark.parametrize("scheduler_class", [Local, Pool])
def test_new_experiment_dir(tmp_path, test_name, experiment_dir, scheduler_class):
    """Test scheduler initialization when experiment dir does not exist."""
    assert not experiment_dir.exists()
    scheduler_class(
        experiment_name=test_name,
        experiment_base_dir=tmp_path,
    )
    assert experiment_dir.exists()


@pytest.mark.parametrize("scheduler_class", [Local, Pool])
def test_overwriting_existing_experiment_dir(
    tmp_path, test_name, _create_experiment_dir, scheduler_class
):
    """Test scheduler init when overwriting experiment directory via flag."""
    scheduler_class(
        experiment_name=test_name,
        experiment_base_dir=tmp_path,
        overwrite_existing_experiment=True,
    )


@pytest.mark.parametrize("scheduler_class", [Local, Pool])
def test_no_prompt_input_for_existing_experiment_dir(
    tmp_path, test_name, _create_experiment_dir, scheduler_class, mocker
):
    """Test scheduler init when not overwriting experiment directory via flag.

    Since the experiment directory already exists, the scheduler prompts
    the user for input. In this test case, the user does not provide any
    prompt input, leading to an abort.
    """
    mocker.patch("select.select", return_value=(None, None, None))
    with pytest.raises(SystemExit) as exit_info:
        scheduler_class(
            experiment_name=test_name,
            experiment_base_dir=tmp_path,
            overwrite_existing_experiment=False,
        )
    assert exit_info.value.code == 1


@pytest.mark.parametrize("scheduler_class", [Local, Pool])
def test_empty_prompt_input_for_existing_experiment_dir(
    tmp_path, test_name, _create_experiment_dir, scheduler_class, mocker
):
    """Test scheduler init when not overwriting experiment directory via flag.

    Since the experiment directory already exists, the scheduler prompts
    the user for input. In this test case, the user provides empty
    input, leading to an abort.
    """
    mocker.patch("select.select", return_value=(True, None, None))
    mocker.patch("sys.stdin.readline", return_value="")
    with pytest.raises(SystemExit) as exit_info:
        scheduler_class(
            experiment_name=test_name,
            experiment_base_dir=tmp_path,
            overwrite_existing_experiment=False,
        )
    assert exit_info.value.code == 1


@pytest.mark.parametrize("scheduler_class,user_input", [(Local, "y"), (Pool, "yes")])
def test_y_prompt_input_for_existing_experiment_dir(
    tmp_path, test_name, _create_experiment_dir, scheduler_class, mocker, user_input
):
    """Test scheduler init when not overwriting experiment directory via flag.

    Since the experiment directory already exists, the scheduler prompts
    the user for input. In this test case, the user provides the input
    'y' or 'yes', allowing the initialization to proceed.
    """
    mocker.patch("select.select", return_value=(True, None, None))
    mocker.patch("sys.stdin.readline", return_value=user_input)
    scheduler_class(
        experiment_name=test_name,
        experiment_base_dir=tmp_path,
        overwrite_existing_experiment=False,
    )


@pytest.fixture(name="experiment_dir")
def fixture_experiment_dir(tmp_path, test_name):
    """Fixture for the experiment directory."""
    return tmp_path / test_name


@pytest.fixture(name="_create_experiment_dir")
def fixture_create_experiment_dir(experiment_dir):
    """Create the experiment directory."""
    os.mkdir(experiment_dir)
    assert experiment_dir.exists()


@pytest.mark.parametrize("scheduler_class", [Local, Pool])
def test_deletion_of_empty_experiment_dir(
    global_settings, tmp_path, test_name, experiment_dir, scheduler_class
):
    """Test the deletion of an empty experiment directory.

    The experiment directory should be deleted when exiting the global
    settings context.
    """
    with global_settings:
        scheduler_class(
            experiment_name=test_name,
            experiment_base_dir=tmp_path,
        )
        assert experiment_dir.exists()
        assert not any(experiment_dir.iterdir())

    assert not experiment_dir.exists()


@pytest.mark.parametrize("scheduler_class", [Local, Pool])
def test_deletion_of_experiment_dir_with_files(
    global_settings, tmp_path, test_name, experiment_dir, scheduler_class
):
    """Test the deletion of an experiment directory containing files.

    The experiment directory should NOT be deleted when exiting the
    global settings context.
    """
    with global_settings:
        scheduler_class(
            experiment_name=test_name,
            experiment_base_dir=tmp_path,
        )
        assert experiment_dir.exists()
        test_file = experiment_dir / "test_file.txt"
        test_file.write_text("test content")
        assert test_file.exists()

    assert experiment_dir.exists()
    assert test_file.exists()
