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
"""Integration tests for Sobol Index iterator."""

import numpy as np

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators import LatinHypercubeSampling, SobolIndex
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.models.surrogates.gaussian_process import GaussianProcess
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


def test_sobol_index_borehole(global_settings):
    """Test case for Sobol Index iterator."""
    # Parameters
    rw = Uniform(lower_bound=0.05, upper_bound=0.15)
    r = Uniform(lower_bound=100, upper_bound=50000)
    tu = Uniform(lower_bound=63070, upper_bound=115600)
    hu = Uniform(lower_bound=990, upper_bound=1110)
    tl = Uniform(lower_bound=63.1, upper_bound=116)
    hl = Uniform(lower_bound=700, upper_bound=820)
    l = Uniform(lower_bound=1120, upper_bound=1680)
    kw = Uniform(lower_bound=9855, upper_bound=12045)
    parameters = Parameters(rw=rw, r=r, tu=tu, hu=hu, tl=tl, hl=hl, l=l, kw=kw)

    # Setup iterator
    driver = Function(parameters=parameters, function="borehole83_lofi")
    scheduler = Pool(experiment_name=global_settings.experiment_name, num_jobs=2)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = SobolIndex(
        seed=42,
        calc_second_order=True,
        num_samples=1024,
        confidence_level=0.95,
        num_bootstrap_samples=1000,
        result_description={"write_results": True, "plot_results": False},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    expected_first_order_indices = np.array(
        [
            0.8275788005095177,
            3.626326582692376e-05,
            1.7993448562887368e-09,
            0.04082350205109995,
            -1.0853339811788176e-05,
            0.0427473897346278,
            0.038941629762778956,
            0.009001905983634081,
        ]
    )

    np.testing.assert_allclose(results["sensitivity_indices"]["S1"], expected_first_order_indices)


def test_sobol_index_sobol(global_settings):
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
    driver = Function(parameters=parameters, function="sobol_g_function")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = SobolIndex(
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


def test_sobol_index_ishigami(global_settings):
    """Test case for Sobol index iterator with Ishigami function."""
    # Parameters
    x1 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x3 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Setup iterator
    driver = Function(parameters=parameters, function="ishigami90")
    scheduler = Pool(experiment_name=global_settings.experiment_name, verbose=True)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = SobolIndex(
        seed=42,
        calc_second_order=True,
        num_samples=16,
        confidence_level=0.95,
        num_bootstrap_samples=1000,
        result_description={"write_results": True, "plot_results": False},
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

    expected_result["S1"] = np.array([0.12572757495660558, 0.3888444532476749, -0.1701023677236496])

    expected_result["S1_conf"] = np.array(
        [0.3935803586836114, 0.6623091120357786, 0.2372589075839736]
    )

    expected_result["ST"] = np.array([0.32520201992825987, 0.5263552164769918, 0.1289289258091274])

    expected_result["ST_conf"] = np.array(
        [0.24575185898081872, 0.5535870474744364, 0.15792828597131078]
    )

    expected_result["S2"] = np.array(
        [
            [np.nan, 0.6350854922111611, 1.0749774123116016],
            [np.nan, np.nan, 0.32907368546743065],
            [np.nan, np.nan, np.nan],
        ]
    )

    expected_result["S2_conf"] = np.array(
        [
            [np.nan, 0.840605849268133, 1.2064077218919202],
            [np.nan, np.nan, 0.5803799668636836],
            [np.nan, np.nan, np.nan],
        ]
    )

    assert_sobol_index_iterator_results(results, expected_result)


def test_sobol_index_gaussian_process_ishigami(global_settings):
    """Test Sobol indices estimation with Gaussian process surrogate."""
    # Parameters
    x1 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x3 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Setup iterator
    driver = Function(parameters=parameters, function="ishigami90")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    simulation_model = Simulation(scheduler=scheduler, driver=driver)
    training_iterator = LatinHypercubeSampling(
        seed=42,
        num_samples=50,
        num_iterations=10,
        result_description={"write_results": True, "plot_results": False},
        model=simulation_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    gpflow_regression_model = GaussianProcess(
        number_restarts=10,
        number_training_iterations=1000,
        dimension_lengthscales=3,
        training_iterator=training_iterator,
    )
    iterator = SobolIndex(
        seed=42,
        calc_second_order=False,
        num_samples=128,
        confidence_level=0.95,
        num_bootstrap_samples=1000,
        result_description={"write_results": True, "plot_results": True},
        model=gpflow_regression_model,
        parameters=parameters,
        global_settings=global_settings,
        skip_values=1024,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    expected_result_s1 = np.array([0.37365542, 0.49936914, -0.00039217])
    expected_result_s1_conf = np.array([0.14969221, 0.18936135, 0.0280309])

    np.testing.assert_allclose(results["sensitivity_indices"]["S1"], expected_result_s1, atol=1e-05)
    np.testing.assert_allclose(
        results["sensitivity_indices"]["S1_conf"], expected_result_s1_conf, atol=1e-05
    )


def assert_sobol_index_iterator_results(results, expected_results):
    """Assert the equality of the results with the expected values.

    Args:
        results (dict): Results dictionary from pickle file
        expected_results (dict): Dictionary with expected results
    """
    np.testing.assert_allclose(results["sensitivity_indices"]["S1"], expected_results["S1"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["S1_conf"], expected_results["S1_conf"]
    )

    np.testing.assert_allclose(results["sensitivity_indices"]["ST"], expected_results["ST"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["ST_conf"], expected_results["ST_conf"]
    )

    np.testing.assert_allclose(results["sensitivity_indices"]["S2"], expected_results["S2"])
    np.testing.assert_allclose(
        results["sensitivity_indices"]["S2_conf"], expected_results["S2_conf"]
    )
