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
"""Integration test for Sobol indices estimation with Sobol's G function."""

import numpy as np

from queens.distributions.uniform import Uniform
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.sobol_index_iterator import SobolIndexIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result
from test_utils.integration_tests import assert_sobol_index_iterator_results


def test_sobol_indices_sobol(global_settings):
    """Test Sobol Index iterator with Sobol G-function.

    Including first, second and total order indices. The test should
    converge to the analytical solution defined in Sobol's G-function
    implementation (see *sobol.py*).
    """
    # Parameters
    x1 = Uniform(lower_bound=0, upper_bound=1)
    x2 = Uniform(lower_bound=0, upper_bound=1)
    x3 = Uniform(lower_bound=0, upper_bound=1)
    x4 = Uniform(lower_bound=0, upper_bound=1)
    x5 = Uniform(lower_bound=0, upper_bound=1)
    x6 = Uniform(lower_bound=0, upper_bound=1)
    x7 = Uniform(lower_bound=0, upper_bound=1)
    x8 = Uniform(lower_bound=0, upper_bound=1)
    x9 = Uniform(lower_bound=0, upper_bound=1)
    x10 = Uniform(lower_bound=0, upper_bound=1)
    parameters = Parameters(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6, x7=x7, x8=x8, x9=x9, x10=x10)

    # Setup iterator
    driver = FunctionDriver(parameters=parameters, function="sobol_g_function")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    model = SimulationModel(scheduler=scheduler, driver=driver)
    iterator = SobolIndexIterator(
        seed=42,
        calc_second_order=True,
        num_samples=128,
        confidence_level=0.95,
        num_bootstrap_samples=10,
        result_description={"write_results": True, "plot_results": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
        skip_values=1024,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    expected_result = {}

    expected_result["S1"] = np.array(
        [
            0.0223308716,
            0.1217603520,
            0.0742536887,
            0.0105281513,
            0.0451664441,
            0.0103643039,
            -0.0243893613,
            -0.0065963022,
            0.0077115277,
            0.0087332959,
        ]
    )
    expected_result["S1_conf"] = np.array(
        [
            0.0805685374,
            0.3834399385,
            0.0852274149,
            0.0455336021,
            0.0308612621,
            0.0320150143,
            0.0463744331,
            0.0714009860,
            0.0074505447,
            0.0112548095,
        ]
    )

    expected_result["ST"] = np.array(
        [
            0.7680857789,
            0.4868735760,
            0.3398667460,
            0.2119195462,
            0.2614132922,
            0.3189091311,
            0.6505384437,
            0.2122730632,
            0.0091166496,
            0.0188473672,
        ]
    )

    expected_result["ST_conf"] = np.array(
        [
            0.3332995622,
            0.6702803374,
            0.3789328006,
            0.1061256016,
            0.1499369465,
            0.2887465421,
            0.4978127348,
            0.7285189769,
            0.0088588230,
            0.0254845356,
        ]
    )

    expected_result["S2"] = np.array(
        [
            [
                np.nan,
                0.1412835702,
                -0.0139270230,
                -0.0060290464,
                0.0649029079,
                0.0029081424,
                0.0711209478,
                0.0029761017,
                -0.0040965718,
                0.0020644536,
            ],
            [
                np.nan,
                np.nan,
                -0.0995909726,
                -0.0605137390,
                -0.1084396644,
                -0.0723118849,
                -0.0745624634,
                -0.0774015700,
                -0.0849434447,
                -0.0839125029,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                -0.0246418033,
                -0.0257497932,
                -0.0193201341,
                -0.0077236185,
                -0.0330585164,
                -0.0345501232,
                -0.0302764363,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0311150448,
                0.0055202682,
                0.0033339784,
                -0.0030970794,
                -0.0072451869,
                -0.0063212065,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0028320819,
                -0.0104508084,
                -0.0052688338,
                -0.0078624231,
                -0.0076410622,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0030222662,
                0.0027860256,
                0.0028227848,
                0.0035368873,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0201030574,
                0.0210914390,
                0.0202893663,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0078664740,
                0.0106712221,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, -0.0102325515],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )

    expected_result["S2_conf"] = np.array(
        [
            [
                np.nan,
                0.9762064146,
                0.1487396176,
                0.1283905049,
                0.2181870269,
                0.1619544753,
                0.1565960033,
                0.1229244812,
                0.1309522579,
                0.1455652199,
            ],
            [
                np.nan,
                np.nan,
                0.3883751512,
                0.3554957308,
                0.3992635683,
                0.4020261874,
                0.3767426554,
                0.3786542992,
                0.3790355847,
                0.3889345096,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                0.0758005266,
                0.0737757790,
                0.0738589320,
                0.1032391772,
                0.0713230587,
                0.0806156892,
                0.0847106864,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.1018303925,
                0.1047654360,
                0.0683036422,
                0.0874356406,
                0.1080467182,
                0.1046926153,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0415102405,
                0.0337889266,
                0.0301212961,
                0.0355450299,
                0.0353899382,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0392075204,
                0.0454072312,
                0.0464493854,
                0.0440356854,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0825175719,
                0.0821124198,
                0.0790512360,
            ],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0685979162,
                0.0668528158,
            ],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0295934940],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )

    assert_sobol_index_iterator_results(results, expected_result)
