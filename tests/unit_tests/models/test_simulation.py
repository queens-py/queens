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
"""Unit tests for the simulation model."""

import numpy as np
import pytest
from mock import Mock

from queens.models.simulation import Simulation


# ------------------ actual unit tests --------------------------- #
def test_init():
    """Test the init method of the simulation model."""
    scheduler = Mock()
    driver = Mock()
    model_obj = Simulation(scheduler=scheduler, driver=driver)
    assert model_obj.scheduler == scheduler
    assert model_obj.function == driver


@pytest.fixture(name="scheduler_response")
def fixture_scheduler_response():
    """Scheduler response fixture."""

    def scheduler_response(samples, driver, job_ids=None):  # pylint: disable=unused-argument
        """Scheduler response."""
        return [{"result": x**2, "gradient": 2 * x} for x in samples]

    return scheduler_response


def test_evaluate(scheduler_response):
    """Test the evaluation method."""
    model_obj = Simulation(scheduler=Mock(), driver=Mock())
    model_obj.scheduler.evaluate = scheduler_response

    samples = np.array([[2.0]])
    response = model_obj.evaluate(samples)
    expected_response = {"result": samples**2, "gradient": 2 * samples}
    assert response == expected_response
    assert model_obj.response == expected_response


def test_grad():
    """Test grad method."""
    model = Simulation(scheduler=Mock(), driver=Mock())
    np.random.seed(42)
    upstream_gradient = np.random.random((2, 4))
    gradient = np.random.random((2, 3, 4))
    model.response = {"mean": None, "gradient": gradient}
    grad_out = model.grad(None, upstream_gradient=upstream_gradient)
    expected_grad = np.sum(
        upstream_gradient[:, :, np.newaxis] * np.swapaxes(gradient, 1, 2), axis=1
    )
    np.testing.assert_almost_equal(expected_grad, grad_out)

    model.response = {"mean": None}
    with pytest.raises(ValueError):
        model.grad(None, upstream_gradient=upstream_gradient)
