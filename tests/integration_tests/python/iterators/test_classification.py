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
"""Integration test for the classification iterator."""

import numpy as np
from sklearn.neural_network._multilayer_perceptron import MLPClassifier

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.classification import Classification
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.classifier import ActiveLearningClassifier
from queens.utils.io import load_result


def test_classification_rosenbrock60(tmp_path, global_settings):
    """Integration test for the classification iterator."""

    def classification_function(x):
        """Classification function.

        Classes are defined in the following way:
        1 or True if the value of x is larger than a threshold.
        0 or False if the value of x is smaller than equal to a threshold.

        Args:
            x (np.array): unclassified data
        Returns:
             classified data (np.array)
        """
        return x > 80

    # Parameters
    x1 = Uniform(lower_bound=-2, upper_bound=2)
    x2 = Uniform(lower_bound=-2, upper_bound=2)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    classifier_obj = MLPClassifier()
    classifier = ActiveLearningClassifier(n_params=2, batch_size=4, classifier_obj=classifier_obj)
    driver = Function(parameters=parameters, function="rosenbrock60")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = Classification(
        num_sample_points=10000,
        num_model_calls=12,
        random_sampling_frequency=2,
        seed=42,
        result_description={
            "write_results": True,
            "plotting_options": {
                "plot_bool": False,
                "plotting_dir": tmp_path,
                "plot_name": "neural_classifier_convergence_plot",
                "save_bool": True,
            },
        },
        classification_function=classification_function,
        classifier=classifier,
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    expected_results_classified = np.ones((12, 1))
    expected_results_classified[-2:] = 0
    expected_samples = np.array(
        [
            [-1.08480734, -0.7215275],
            [0.47755988, -1.70552695],
            [1.2057831, -0.18745467],
            [-0.82740908, -1.73532511],
            [-0.60210352, 1.98595546],
            [-0.66345435, 1.99530424],
            [-0.46880494, 1.98952451],
            [-0.31552438, 1.98624568],
            [-0.86489638, -0.2715313],
            [1.87330784, 1.77265221],
            [0.88234568, 0.14938235],
            [0.3827193, 0.81934479],
        ]
    )
    obtained_results = results["classified_outputs"]
    obtained_samples = results["input_samples"]
    np.testing.assert_array_almost_equal(expected_samples, obtained_samples)
    np.testing.assert_array_equal(expected_results_classified, obtained_results)
