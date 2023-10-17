"""Binary classification iterator.

This iterator trains a classification algorithm based on a forward and
classification model.
"""

import logging

import numpy as np
from skactiveml.utils import MISSING_LABEL

from pqueens.iterators.iterator import Iterator
from pqueens.utils.ascii_art import print_classification
from pqueens.utils.process_outputs import write_results
from pqueens.visualization.classification import ClassificationVisualization

_logger = logging.getLogger(__name__)


def default_classification_function(features):
    """Default classification function checking for NaNs.

    Args:
        features (np.array): input array containing the values that should be classified
    Returns:
        np.array: boolean array predictions where True represents non-NaN values,
                  and False represents NaN values in the original array x
    """
    return ~np.isnan(features.astype(float))


class ClassificationIterator(Iterator):
    """Iterator for machine leaning based classification.

    Attributes:
        model (model): Model to be evaluated by iterator
        result_description (dict):  Description of desired results
        num_sample_points (int): number of total points
        num_model_calls (int): total number of model calls
        random_sampling_frequency (int): in case of active sampling every iteration index that
                                        is a multiple of this number the samples are selected
                                        randomly
        classifier (obj): queens classifier object
        visualization_obj (obj): object for visualization
        classification_function (fun): function that classifies the model output
        samples (np.array): samples on which the model was evaluated at
        classified_outputs (np.array): classified output of evaluated at samples
    """

    def __init__(
        self,
        model,
        parameters,
        result_description,
        num_sample_points,
        num_model_calls,
        random_sampling_frequency,
        classifier,
        seed,
        classification_function=default_classification_function,
    ):
        """Initialize classification iterator.

        Args:
            model (model): Model to be evaluated by iterator
            parameters (obj): Parameters object
            result_description (dict):  Description of desired results
            num_sample_points (int): number of total points
            num_model_calls (int): total number of model calls
            random_sampling_frequency (int): in case of active sampling every iteration index that
                                             is a multiple of this number the samples are selected
                                             randomly
            classifier (obj): queens classifier object
            seed (int): random initial seed
            classification_function (fun): function that classifies the model output
        """
        if classifier.is_active and num_model_calls >= num_sample_points:
            raise ValueError("Number of sample points needs to be greater than num_model_calls.")
        super().__init__(model, parameters)
        self.result_description = result_description
        self.samples = np.empty((0, self.parameters.num_parameters))
        self.classified_outputs = np.empty((0, 1))
        self.num_sample_points = num_sample_points
        self.num_model_calls = num_model_calls
        self.classifier = classifier
        self.seed = seed
        self.random_sampling_frequency = random_sampling_frequency

        if visualization_obj := result_description.get("plotting_options") is not None:
            visualization_obj = ClassificationVisualization(
                **result_description.get("plotting_options")
            )
        self.visualization_obj = visualization_obj

        self.classification_function = classification_function
        np.random.seed(self.seed)

    def core_run(self):
        """Evaluate the samples on model and classify them."""
        # Essential
        print_classification()

        # use active learning
        if self.classifier.is_active:
            _logger.info("Active classification.")
            samples = self.parameters.draw_samples(self.num_sample_points)
            classification_data = np.full(shape=len(samples), fill_value=MISSING_LABEL)
            for i in range(self.num_model_calls // self.classifier.batch_size):
                query_idx = self.classifier.train(samples, classification_data)
                if self.random_sampling_frequency and i % self.random_sampling_frequency == 0:
                    query_idx = self._select_random_samples(classification_data, query_idx)
                classification_data[query_idx.reshape(-1, 1)] = self.binarize(samples[query_idx])

                if self.visualization_obj:
                    self.visualization_obj.plot_decision_boundary(
                        self.classified_outputs,
                        self.samples,
                        self.classifier,
                        self.parameters.names,
                        i,
                    )
                if i % 10 == 0:
                    _logger.info(
                        "Active iteration %i, model calls %i", i, len(self.classified_outputs)
                    )
        else:
            _logger.info("Classification.")
            samples = self.parameters.draw_samples(self.num_model_calls)
            classification_data = self.binarize(samples)
            self.classifier.train(samples, classification_data)

    def _select_random_samples(self, classification_data, query_idx):
        """Select random samples.

        Args:
            classification_data (np.array): Classification of all sample points
            query_idx (np.array): array with the next sample indices

        Returns:
            np.array: new query_idx
        """
        not_predicted_idx = np.argwhere(np.isnan(classification_data))
        idx = np.array(
            list(
                filter(
                    lambda x: x not in query_idx,
                    not_predicted_idx,
                )
            )
        )
        np.random.shuffle(idx)
        return idx[: self.classifier.batch_size].flatten()

    def post_run(self):
        """Analyze the results."""
        if self.result_description:
            results = {"classified_outputs": self.classified_outputs, "input_samples": self.samples}
            if self.result_description["write_results"]:
                write_results(
                    results,
                    self.output_dir,
                    self.experiment_name,
                )

        # plot decision boundary for the trained classifier
        if self.visualization_obj:
            self.visualization_obj.plot_decision_boundary(
                self.classified_outputs,
                self.samples,
                self.classifier,
                self.parameters.names,
                "final",
            )

        # save checkpoint
        self.classifier.save(
            self.output_dir,
            self.experiment_name + "_classifier",
        )

    def _evaluate_model(self, samples):
        """Evaluate model.

        Args:
            samples (np.array): Samples where to evaluate the model

        Returns:
            np.array: model output
        """
        output = self.model.evaluate(samples)["mean"]
        self.samples = np.row_stack((self.samples, samples))
        return output

    def binarize(self, samples):
        """Classify the output.

        Args:
            samples (np.array): Samples where to evaluate the model

        Returns:
            np.array: classified output
        """
        output = self.classification_function(self._evaluate_model(samples))
        self.classified_outputs = np.row_stack((self.classified_outputs, output)).astype(int)
        return output