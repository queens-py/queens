#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2025, QUEENS contributors.
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
"""Base class for RL capabilities."""

import logging
import time

from queens.iterators.iterator import Iterator
from queens.models.rl_models.rl_model import RLModel
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


class RLIterator(Iterator):
    """Iterator for enabling the solution of RL problems.

    For an introduction to RL with QUEENS, we refer to the documentation of
    the :py:mod:`queens.models.rl_models.rl_model` module.

    Attributes:
        _interaction_steps (int): Number of interaction steps to be performed.
            This variable is only relevant in ``'evaluation'`` mode and determines
            the number of interaction steps that should be performed with the model.
        _mode (str): Mode of the RLIterator. This variable can be either
            ``'training'`` or ``'evaluation'``, depending on whether the user
            wants to train an RL model or use a trained model for evaluation
            purposes, e.g., as surrogate.
        initial_observation (np.ndarray): Initial observation of the environment.
            This variable is only relevant in ``'evaluation'`` mode and determines
            the initial observation of the environment, i.e., the starting
            point of the interaction loop.
        output (dict): Dictionary for storing the output of the iterator in
            ``'evaluation'`` mode. The dictionary contains two keys: ``'step'``
            and ``'result'``. The key ``'step'`` contains the list of steps
            performed and the key ``'result'`` contains the list of results
            corresponding to each step.
        result_description (dict):  Description of desired results.
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
        """Initialize an RLIterator.

        Args:
            model (RLModel): Model to be evaluated by the iterator.
            parameters (Parameters): Parameters object.
                .. note::
                        This parameter is required by the base class, but is
                        currently not used in the RLIterator.
            global_settings (GlobalSettings): Settings of the QUEENS experiment including its name
                and the output directory.
            result_description (dict): Description of desired results.
            mode (str): Mode of the RLIterator. This variable can be either ``'training'``
                or ``'evaluation'``.
            interaction_steps (int): Number of interaction steps to be performed.
            initial_observation (np.ndarray): Initial observation of the environment.
        """
        # Make sure that a valid model has been provided
        if not isinstance(model, RLModel):
            raise ValueError(
                "Unsupported model:\n"
                "`RLIterator` only supports models that inherit from `RLModel`."
            )

        super().__init__(model, parameters, global_settings)

        self.result_description = result_description
        self.mode = mode
        self.interaction_steps = interaction_steps
        self.initial_observation = initial_observation

        # Create a dictionary with empty lists to store the interaction data
        self.output = {"step": [], "result": []}

    @property
    def mode(self):
        """Access the mode of the RLIterator.

        Returns:
            str: Mode of the RLIterator.
        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        """Set the mode of the RLIterator.

        Perform sanity checks to ensure that mode has a valid value.

        Args:
            mode (str): Mode of the RLIterator.
        """
        if mode not in ["training", "evaluation"]:
            raise ValueError(
                f"Unsupported mode: {mode}\n" "The mode must be either `training` or `evaluation`."
            )
        self._mode = mode

    @property
    def interaction_steps(self):
        """Access the number of interaction steps.

        Returns:
            int: Number of interaction steps to be performed.
        """
        return self._interaction_steps

    @interaction_steps.setter
    def interaction_steps(self, eval_steps):
        """Set the number of interaction steps.

        Perform sanity checks to ensure that the number of interaction steps
        has a valid value.

        Args:
            eval_steps (int): Number of interaction steps to be performed.
        """
        if eval_steps < 0:
            raise ValueError(
                f"Unsupported number of interaction steps: {eval_steps}\n"
                "The number of interaction steps must be a positive integer."
            )
        self._interaction_steps = eval_steps

    def pre_run(self):
        """Prepare the core run of the RLIterator (not needed here)."""
        _logger.info("Initialize RLIterator.")

    def core_run(self):
        """Core run of RLIterator.

        Depending on the :py:attr:`_mode` of the RLIterator, the agent is either
        trained or used for evaluation. In case of evaluation, the results of
        the interactions are stored in the :py:attr:`output` dictionary.
        """
        _logger.info("Welcome to Reinforcement Learning core run.")
        if self._mode == "training":
            _logger.info("Starting agent training.")
            start = time.time()
            # Start the model training
            self.model.train()
            end = time.time()
            _logger.info("Agent training took %E seconds.", end - start)
        else:  # self._mode == 'evaluation'
            _logger.info("Starting interaction loop.")
            if self.initial_observation is None:
                _logger.debug(
                    "No initial observation provided.\n"
                    "Resetting environment to generate an initial observation."
                )
                obs = self.model.get_initial_observation()
            else:  # initial observation has been provided by the user
                _logger.debug("Using provided initial observation.")
                obs = self.initial_observation
            start = time.time()
            # Perform as many interaction steps as set by the user
            for step in range(self._interaction_steps):
                result = self.model.interact(obs)
                # Extract the observation for the next iteration
                obs = result["observation"]
                # Update the output dictionary
                self.output["step"].append(step)
                self.output["result"].append(result)
            end = time.time()
            _logger.info("Interaction loop took %E seconds.", end - start)

    def post_run(self):
        """Optionally export the results of the core run depending on the mode.

        If the mode is set to ``'training'``, the trained agent is
        stored for further processing. If the mode is set to
        ``'evaluation'``, the interaction outputs are stored for further
        processing.
        """
        if self.result_description:
            if self.result_description["write_results"]:
                if self._mode == "training":
                    _logger.info("Storing the trained agent for further processing.")
                    self.model.save(self.global_settings)
                else:  # self._mode == 'evaluation'
                    _logger.info("Storing interaction output for further processing.")
                    write_results(self.output, self.global_settings.result_file(".pickle"))
