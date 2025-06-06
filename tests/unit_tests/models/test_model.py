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
"""Test-module for abstract Model class."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from queens.models._model import Model


class DummyModel(Model):
    """Dummy model class."""

    def _evaluate(self, samples):
        """Evaluate model with current set of samples."""

    def grad(self, samples, upstream_gradient):
        """Evaluate gradient of model with current set of samples."""


@pytest.fixture(name="model")
def fixture_model():
    """An instance of an empty Model class."""
    return DummyModel()


def test_init(model):
    """Test init."""
    assert model.response is None
    assert model.num_evaluations == 0
    assert model.num_gradient_evaluations == 0
    assert not model.evaluate_and_gradient_bool


def test_evaluate_and_gradient(model):
    """Test evaluate_and_gradient method."""
    assert not model.evaluate_and_gradient_bool

    def model_eval(self, x):
        assert self.evaluate_and_gradient_bool
        return {"result": np.sum(x**2, axis=1, keepdims=True)}

    model.grad = Mock(
        side_effect=lambda x, upstream_gradient: np.sum(
            upstream_gradient[:, :, np.newaxis] * 2 * x[:, np.newaxis, :], axis=1
        )
    )

    num_samples = 3
    samples = np.random.random((num_samples, 4))
    with patch.object(DummyModel, "_evaluate", new=model_eval):
        model_out, model_grad = model.evaluate_and_gradient(samples, upstream_gradient=None)
        assert model.grad.call_count == 1
        assert model.num_evaluations == num_samples
        assert model.num_gradient_evaluations == num_samples
        np.testing.assert_array_equal(model.grad.call_args.args[0], samples)
        np.testing.assert_array_equal(
            model.grad.call_args.kwargs["upstream_gradient"], np.ones((samples.shape[0], 1))
        )

        expected_model_out = np.sum(samples**2, axis=1, keepdims=True)
        expected_model_grad = 2 * samples
        np.testing.assert_array_equal(expected_model_out, model_out)
        np.testing.assert_array_equal(expected_model_grad, model_grad)

        # test with upstream_gradient
        upstream_ = np.random.random(samples.shape[0])
        model.evaluate_and_gradient(samples, upstream_gradient=upstream_)
        assert model.grad.call_count == 2
        assert model.num_evaluations == 2 * num_samples
        assert model.num_gradient_evaluations == 2 * num_samples
        np.testing.assert_array_equal(model.grad.call_args.args[0], samples)
        np.testing.assert_array_equal(
            model.grad.call_args.kwargs["upstream_gradient"], upstream_[:, np.newaxis]
        )

        assert not model.evaluate_and_gradient_bool
