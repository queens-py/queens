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
"""Model for running RL tasks.

TODO: Add a disclaimer about the nomenclature of reinforcement learning (RL) models
with respect to standard QUEENS nomenclature.
"""

import logging

from queens.models.model import Model
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class RLModel(Model):
    """TODO."""

    @log_init_args
    def __init__(self, agent, total_timesteps=10000, render_on_evaluation=False, render_args=()):
        """TODO."""
        super().__init__()

        self.is_trained = False

        # Store the provided agent instance
        self._agent = agent
        # Retrieve a (vectorized) stable-baseline3 environment for evaluation
        self._vec_env = self._agent.get_env()

        self._total_timesteps = total_timesteps
        self._render_on_evaluation = render_on_evaluation
        self._render_args = render_args

    def interact(self, observation):
        """Interaction step of a (trained) RL agent with an environment.

        Args:
            observation (np.ndarray): The observation of the current state of
                                      the environment.

        Returns:
            next_observation (dict): Dictionary with the model output
                                     (corresponding to the observation of the
                                     new state of the environment)
        """
        _logger.info("Computing one agent-enviroment interaction (i.e., one timestep).")
        result = self.predict(observation)
        # return values are: observation, reward, done, info
        obs, _, _, _ = self.step(result["action"])
        if self._render_on_evaluation:
            self.render()
        _logger.info("Interaction completed.")
        return obs

    def get_initial_observation(self):
        """TODO."""
        return self._vec_env.reset()

    def grad(self, samples, upstream_gradient):
        """TODO Check if this is always the case."""
        raise NotImplementedError(
            "Gradient information not available. \n"
            "If you need gradients, please use a different model or implement "
            "the `grad` method in the child class."
        )

    def evaluate(self, samples):
        """Evaluate the model on a given set of input samples.

        Delegates the call to py:meth:`predict` internally and stores the of
        the model evaluation in the internal storage variable :py:attr:`response`.

        Args:
            samples (np.ndarray): Input samples, i.e., multiple observations

        Returns:
            dict: Results and actions corresponding to current set of input samples
        """
        # Predict the next actions (and states) based on the current
        # observations of the environment
        self.response = self.predict(samples)
        return self.response

    def predict(self, observations):
        """Make predictions with a trained RL agent for given observations.

        Args:
            observations (np.ndarray): Either a single observation or a batch of observations.

        Returns:
            result (dict): Results and actions corresponding to given set of input observations.
        """
        _logger.debug("Predicting the next action based on the current state of the environment.")
        # Predict the agent's action and the new state of the environment
        # based on the current observation
        actions, states = self._agent.predict(observations)
        # combine information in a dict
        result = {
            "result": actions,  # this is redundant, but kept for compatibility
            "action": actions,
            "state": states,
        }
        return result

    def render(self):
        """Render the current state of the environment."""
        self._vec_env.render(*self._render_args)

    def step(self, action):
        """Execute a single step in the environment.

        Args:
            action (np.ndarray): Action to be executed

        Returns:
            observation (np.ndarray): Observation of the new state of the environment
            reward (float): Reward obtained from the environment
            done (bool): Flag indicating whether the episode has finished
            info (dict): Additional information
        """
        _logger.debug("Applying an action to the environment.")
        return self._vec_env.step(action)

    def train(self):
        """Train the RL agent."""
        _logger.info("Training the RL agent for a total of %d timesteps.", self._total_timesteps)
        # Train the agent for the desired number of timesteps
        self._agent.learn(total_timesteps=self._total_timesteps)
        _logger.info("Training completed.")
        self.is_trained = True
