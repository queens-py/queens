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
"""Functionality for training and evaluating an RL model with QUEENS.

For an introduction to RL in the context of QUEENS, we refer to the documentation of the
:py:mod:`queens.models.reinforcement_learning` module.
"""

import logging
import time
from collections import defaultdict

import numpy as np

from queens.iterators._iterator import Iterator
from queens.models.reinforcement_learning.reinforcement_learning import (
    ReinforcementLearning as RLModel,
)
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


class ReinforcementLearning(Iterator):
    """Iterator for enabling the training or evaluation of an RL model.

    Attributes:
        _interaction_steps (int): Number of interaction steps to be performed.
            This variable is only relevant in ``"evaluation"`` mode and determines
            the number of interaction steps that should be performed with the model.
        _mode (str): Mode of the ``ReinforcementLearning`` iterator. This variable can be either
            ``"training"`` or ``"evaluation"``, depending on whether the user
            wants to train an RL model or use a trained model for evaluation
            purposes, e.g., as surrogate.
        initial_observation (np.ndarray): Initial observation of the environment.
            This variable is only relevant in ``"evaluation"`` mode and determines
            the initial observation of the environment, i.e., the starting
            point of the interaction loop.
        output (dict): Dictionary for storing the output of the iterator in
            ``"evaluation"`` mode. Since an ``ReinforcementLearning`` iterator instance behaves
            differently than other iterators in the sense that the outputs are only collected
            iteratively via an interaction loop, the dictionary contains lists
            during an evaluation run, which are filled dynamically. Once the evaluation
            run is complete, these lists are converted to ``numpy`` arrarys, so that
            the resulting dict associates string keys, with ``numpy`` arrays.
            The ``output`` dict of an ``ReinforcementLearning`` iterator instance contains 6 keys:

            * ``"result"``: This key is kept for compatibility with other QUEENS
              iterators and contains the recorded actions.
            * ``"action"``: Contains the actions that were predicted by the agent
              as result of the provided observations.
            * ``"new_obs"``: The new observation after applying the predicted action
              to the environment.
            * ``"reward"``: The reward corresponding to the predicted action.
            * ``"info"``: Additional information about the interaction step.
              Please note that each entry in ``output["info"]`` will again be a
              dictionary.
            * ``"done"``: Flag indicating whether the undertaken action completed
              an episode.
        result_description (dict):  Description of desired results.
        samples (np.ndarray): Observations that were used as model inputs during the
            evaluation interaction loop. Similarly to the :py:attr:`output` member,
            the samples are only generated iteratively during an evaluation run
            of an ``ReinforcementLearning`` iterator instance. Thus, this variable is initially
            initialized as list and only converted to a numpy array at the end of each evaluation
            run.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        result_description=None,
        mode="training",
        interaction_steps=1000,
        initial_observation=None,
    ):
        """Initialize an ``ReinforcementLearning`` iterator.

        Args:
            model (RLModel): Model to be evaluated by the iterator.
            parameters (Parameters): Parameters object.

                .. note::
                    This parameter is required by the base class, but is
                    currently not used in the ``ReinforcementLearning`` iterator.

            global_settings (GlobalSettings): Settings of the QUEENS experiment including its name
                and the output directory.
            result_description (dict): Description of desired results.
            mode (str): Mode of the ``ReinforcementLearning`` iterator. This variable can be either
                ``"training"`` or ``"evaluation"``.
            interaction_steps (int): Number of interaction steps to be performed.
            initial_observation (np.ndarray): Initial observation of the environment.
        """
        # Make sure that a valid model has been provided
        if not isinstance(model, RLModel):
            raise ValueError(
                "Unsupported model:\n"
                "`ReinforcementLearning` only supports models that inherit from `RLModel`."
            )

        super().__init__(model, parameters, global_settings)

        self.result_description = result_description
        self.mode = mode
        self.interaction_steps = interaction_steps
        self.initial_observation = initial_observation

        # Initialize samples and output members
        self.samples = None
        self.output = None

    @property
    def mode(self):
        """Access the mode of the ``ReinforcementLearning`` iterator.

        Returns:
            str: Mode of the ``ReinforcementLearning`` iterator.
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        """Set the mode of the ``ReinforcementLearning`` iterator.

        Perform sanity checks to ensure that mode has a valid value.

        Args:
            value (str): Mode of the ``ReinforcementLearning`` iterator.
        """
        if value not in ["training", "evaluation"]:
            raise ValueError(
                f"Unsupported mode: {value}\nThe mode must be either `training` or `evaluation`."
            )
        self._mode = value

    @property
    def interaction_steps(self):
        """Access the number of interaction steps.

        Returns:
            int: Number of interaction steps to be performed.
        """
        return self._interaction_steps

    @interaction_steps.setter
    def interaction_steps(self, value):
        """Set the number of interaction steps.

        Perform sanity checks to ensure that the number of interaction steps
        has a valid value.

        Args:
            value (int): Number of interaction steps to be performed.
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Unsupported number of interaction steps: {value}\n"
                "The number of interaction steps must be a positive integer."
            )
        self._interaction_steps = value

    def pre_run(self):
        """Prepare the core run of the RL iterator (not needed here)."""
        _logger.info("Initialize ReinforcementLearning.")

    def core_run(self):
        """Core run of ``ReinforcementLearning`` iterator.

        Depending on the :py:attr:`_mode` of the ``ReinforcementLearning`` iterator, the agent is
        either trained or used for evaluation. In case of evaluation, the results of the
        interactions are stored in the :py:attr:`output` dictionary.
        """
        _logger.info("Welcome to Reinforcement Learning core run.")
        if self.mode == "training":
            _logger.info("Starting agent training.")
            start = time.time()
            # Start the model training
            self.model.train()
            end = time.time()
            _logger.info("Agent training took %E seconds.", end - start)
        else:  # self._mode == "evaluation"
            _logger.info("Starting interaction loop.")
            # Reset samples member (initialize as list since data needs to be added dynamically)
            self.samples = []
            # Reset output member (initilize as a dictionary with lists since
            # data needs to be added dynamically)
            self.output = defaultdict(list)
            if self.initial_observation is None:
                _logger.debug(
                    "No initial observation provided.\n"
                    "Resetting environment to generate an initial observation."
                )
                obs = self.model.reset()
            else:  # initial observation has been provided by the user
                _logger.debug("Using provided initial observation.")
                obs = self.initial_observation
            start = time.time()
            # Perform as many interaction steps as set by the user
            for _ in range(self._interaction_steps):
                result = self.model.interact(obs)
                # Extract the observation for the next iteration
                obs = result["new_obs"]
                # Update the samples and outputs
                self.update_samples_and_outputs(obs, result)
            end = time.time()
            _logger.info("Interaction loop took %E seconds.", end - start)
            # convert the generated samples and outputs to numpy arrays
            self.convert_to_numpy()

    def post_run(self):
        """Optionally export the results of the core run depending on the mode.

        If the mode is set to ``"training"``, the trained agent is
        stored for further processing. If the mode is set to
        ``"evaluation"``, the interaction outputs are stored for further
        processing.
        """
        if self.result_description:
            if self.result_description["write_results"]:
                if self._mode == "training":
                    _logger.info("Storing the trained agent for further processing.")
                    self.model.save(self.global_settings)
                else:  # self._mode == "evaluation"
                    _logger.info("Processing interaction output...")
                    results = {"samples": self.samples, "outputs": self.output}
                    _logger.info("Storing processed output for further processing.")
                    write_results(results, self.global_settings.result_file(".pickle"))

    def update_samples_and_outputs(self, obs, result):
        """Stores step information of the current interaction step.

        Dynamically updates the :py:attr:`samples` and :py:attr:`output` members
        with the provided data from the last interaction step.

        Args:
            obs (np.ndarray) : The observation used as an input to start the last
                interaction step.
            result (dict): The results generated by the last interaction step.
        """
        self.samples.append(obs)
        for key in result.keys():
            self.output[key].append(result[key])

    def convert_to_numpy(self):
        """Converts members :py:attr:`samples` and :py:attr:`output` to numpy.

        This function is called at the end of :py:meth:`core_run` when executed
        with ``mode==evaluation``.
        """
        self.samples = np.array(self.samples)
        for key, value in self.output.items():
            self.output[key] = np.array(value)
