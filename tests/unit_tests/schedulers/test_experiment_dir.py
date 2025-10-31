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
def test_new_experiment_dir(tmp_path, experiment_name, experiment_dir, scheduler_class):
    """Test scheduler initialization when experiment dir does not exist."""
    assert not experiment_dir.exists()
    scheduler_class(
        experiment_name=experiment_name,
        experiment_base_dir=tmp_path,
    )
    assert experiment_dir.exists()


@pytest.mark.parametrize("scheduler_class", [Local, Pool])
def test_overwriting_existing_experiment_dir(
    tmp_path, experiment_name, _create_experiment_dir, scheduler_class
):
    """Test scheduler init when overwriting experiment directory via flag."""
    scheduler_class(
        experiment_name=experiment_name,
        experiment_base_dir=tmp_path,
        overwrite_existing_experiment=True,
    )


@pytest.mark.parametrize("scheduler_class", [Local, Pool])
def test_no_prompt_input_for_existing_experiment_dir(
    tmp_path, experiment_name, _create_experiment_dir, scheduler_class, mocker
):
    """Test scheduler init when not overwriting experiment directory via flag.

    Since the experiment directory already exists, the scheduler prompts
    the user for input. In this test case, the user does not provide any
    prompt input, leading to an abort.
    """
    mocker.patch("select.select", return_value=(None, None, None))
    with pytest.raises(SystemExit) as exit_info:
        scheduler_class(
            experiment_name=experiment_name,
            experiment_base_dir=tmp_path,
            overwrite_existing_experiment=False,
        )
    assert exit_info.value.code == 2


@pytest.mark.parametrize("scheduler_class", [Local, Pool])
def test_empty_prompt_input_for_existing_experiment_dir(
    tmp_path, experiment_name, _create_experiment_dir, scheduler_class, mocker
):
    """Test scheduler init when not overwriting experiment directory via flag.

    Since the experiment directory already exists, the scheduler prompts
    the user for input. In this test case, the user provides 'n' as
    prompt input, leading to an abort.
    """
    mocker.patch("select.select", return_value=(True, None, None))
    mocker.patch("sys.stdin.readline", return_value="")
    with pytest.raises(SystemExit) as exit_info:
        scheduler_class(
            experiment_name=experiment_name,
            experiment_base_dir=tmp_path,
            overwrite_existing_experiment=False,
        )
    assert exit_info.value.code == 2


@pytest.mark.parametrize("scheduler_class", [Local, Pool])
def test_y_prompt_input_for_existing_experiment_dir(
    tmp_path, experiment_name, _create_experiment_dir, scheduler_class, mocker
):
    """Test scheduler init when not overwriting experiment directory via flag.

    Since the experiment directory already exists, the scheduler prompts
    the user for input. In this test case, the user provides 'y' as
    prompt input, allowing the initialization to proceed.
    """
    mocker.patch("select.select", return_value=(True, None, None))
    mocker.patch("sys.stdin.readline", return_value="y")
    scheduler_class(
        experiment_name=experiment_name,
        experiment_base_dir=tmp_path,
        overwrite_existing_experiment=False,
    )


@pytest.fixture(name="experiment_dir")
def fixture_experiment_dir(tmp_path, experiment_name):
    """Fixture for the experiment directory."""
    return tmp_path / experiment_name


@pytest.fixture(name="_create_experiment_dir")
def fixture_create_experiment_dir(experiment_dir):
    """Create the experiment directory."""
    os.makedirs(experiment_dir)
    assert experiment_dir.exists()
