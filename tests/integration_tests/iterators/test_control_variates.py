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
"""Integration tests for the control variates iterator."""

import pytest

from queens.iterators.control_variates import ControlVariates
from queens.main import run_iterator
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


@pytest.fixture(name="scheduler")
def fixture_scheduler(global_settings):
    """Scheduler for the integration tests."""
    return Pool(experiment_name=global_settings.experiment_name)


def test_control_variates_with_given_num_samples(
    global_settings, borehole_parameters, borehole83_hifi_model, borehole83_lofi_model
):
    """Test function for control variates with a given number of samples."""
    # Number of samples on the cross-model estimator.
    n0 = 100

    # Set up iterator.
    iterator = ControlVariates(
        model=borehole83_hifi_model,
        control_variate=borehole83_lofi_model,
        parameters=borehole_parameters,
        global_settings=global_settings,
        seed=42,
        num_samples=n0,
        num_samples_cv=10 * n0,
        use_optimal_num_samples=False,
    )

    # Run iterator and load results.
    run_iterator(iterator=iterator, global_settings=global_settings)
    res = load_result(global_settings.result_file(".pickle"))

    # Test outputs.
    assert res["mean"] == pytest.approx(77.03460846952085)
    assert res["std"] == pytest.approx(1.3774480043137558)
    assert res["num_samples_cv"] == pytest.approx(1000)
    assert res["mean_cv"] == pytest.approx(61.63815600352344)
    assert res["std_cv_mean_estimator"] == pytest.approx(1.1561278589420407)
    assert res["cv_influence_coeff"] == pytest.approx(1.1296035845358712)


def test_control_variates_with_optimal_num_samples(
    global_settings, park91a_parameters, park91a_hifi_model, park91a_lofi_model
):
    """Test function for control variates with optimal number of samples."""
    # Number of samples on the cross-model estimator.
    n0 = 4
    # Cost of evaluating the main model.
    cost_model_main = 1
    # Cost of evaluating the control variate.
    cost_control_variate = 0.9999999

    # Set up iterator.
    iterator = ControlVariates(
        model=park91a_hifi_model,
        control_variate=park91a_lofi_model,
        parameters=park91a_parameters,
        global_settings=global_settings,
        seed=42,
        num_samples=n0,
        num_samples_cv=10 * n0,
        use_optimal_num_samples=True,
        cost_model=cost_model_main,
        cost_cv=cost_control_variate,
    )

    # Run iterator and load results.
    run_iterator(iterator=iterator, global_settings=global_settings)
    res = load_result(global_settings.result_file(".pickle"))

    # Test outputs.
    assert res["mean"] == pytest.approx(8.486285171375979)
    assert res["std"] == pytest.approx(1.0843641376888087)
    assert res["num_samples_cv"] == pytest.approx(18)
    assert res["mean_cv"] == pytest.approx(8.877129681945862)
    assert res["std_cv_mean_estimator"] == pytest.approx(1.3203990388887525)
    assert res["cv_influence_coeff"] == pytest.approx(0.676596107631806)
    assert res["sample_ratio"] == pytest.approx(4.72194840557595)
