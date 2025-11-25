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
"""Unit tests for the adjoint model."""

from pathlib import Path

import numpy as np
import pytest
from mock import Mock

from queens.models import adjoint
from queens.models.adjoint import Adjoint


# ------------------ some fixtures ------------------------------- #
@pytest.fixture(name="default_adjoint_model")
def fixture_default_adjoint_model():
    """A default adjoint model."""
    model_obj = Adjoint(
        scheduler=Mock(),
        driver=Mock(),
        gradient_driver=Mock(),
        adjoint_file="my_adjoint_file",
    )
    return model_obj


# ------------------ actual unit tests --------------------------- #
def test_init():
    """Test the init method of the adjoint model."""
    scheduler = Mock()
    driver = Mock()
    gradient_driver = Mock()
    adjoint_file = "my_adjoint_file"

    # Test without grad handler
    model_obj = Adjoint(
        scheduler=scheduler,
        driver=driver,
        gradient_driver=gradient_driver,
        adjoint_file=adjoint_file,
    )
    assert model_obj.scheduler == scheduler
    assert model_obj.function == driver
    assert model_obj.gradient_driver == gradient_driver
    assert model_obj.adjoint_file == adjoint_file


@pytest.fixture(name="scheduler_response")
def fixture_scheduler_response():
    """Scheduler fixture."""

    def scheduler_response(samples, driver, job_ids=None):  # pylint: disable=unused-argument
        """Scheduler response."""
        return [{"result": x**2} for x in samples]

    return scheduler_response


def test_evaluate(default_adjoint_model, scheduler_response):
    """Test the evaluation method."""
    default_adjoint_model.scheduler.evaluate = scheduler_response
    samples = np.array([[2.0]])
    response = default_adjoint_model.evaluate(samples)
    expected_response = {"result": samples**2}
    assert response == expected_response
    assert default_adjoint_model.response == expected_response


def test_grad(default_adjoint_model, scheduler_response):
    """Test grad method."""
    experiment_dir = Path("path_to_experiment_dir")
    adjoint.write_to_csv = Mock()
    default_adjoint_model.scheduler.next_job_id = 7
    default_adjoint_model.scheduler.experiment_dir = experiment_dir
    default_adjoint_model.scheduler.evaluate = scheduler_response

    np.random.seed(42)
    samples = np.random.random((2, 3))
    upstream_gradient = np.random.random((2, 4))
    grad_out = default_adjoint_model.grad(samples, upstream_gradient=upstream_gradient)

    expected_grad = samples**2
    np.testing.assert_almost_equal(expected_grad, grad_out)

    assert adjoint.write_to_csv.call_count == 2
    assert (
        adjoint.write_to_csv.call_args_list[0].args[0]
        == experiment_dir / "5" / default_adjoint_model.adjoint_file
    )
    assert (
        adjoint.write_to_csv.call_args_list[1].args[0]
        == experiment_dir / "6" / default_adjoint_model.adjoint_file
    )
    np.testing.assert_equal(
        adjoint.write_to_csv.call_args_list[0].args[1],
        upstream_gradient[0].reshape(1, -1),
    )
    np.testing.assert_equal(
        adjoint.write_to_csv.call_args_list[1].args[1],
        upstream_gradient[1].reshape(1, -1),
    )
