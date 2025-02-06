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
"""Functionality for performing Reinforcement Learning (RL) with *QUEENS*.

.. note::
        If you have no prior experience with RL, a good starting point might be
        the introduction of Spinning Up in Deep RL by OpenAI:
        https://spinningup.openai.com/en/latest/spinningup/rl_intro.html.

In the follwing, we provide a brief overview of RL concepts and terminology and
their relation to *QUEENS*.

In its essence, Reinformcement Learning (RL) is a type of machine learning which
tries to mimick they way how humans learn to accomplish a new task, namely by performing
trial-and-error interactions with their environment and learning from the gathered experience.

In RL, this interaction happens between a so-called **agent** (i.e., a learning algorithm)
and an **environment** (i.e., the task or problem to be solved). The agent can perform
**actions** in the environment in order to modify its state and receives **observations**
(i.e., of the new state of the environment after applying the action) and **rewards**
(i.e., a numerical reward signal quantifying how well the undertaken action was
with respect to solving the problem encoded in the environment) in return.
One interaction step between the agent and the environment is called a **timestep**.
The goal of the agent is to learn a **policy** (i.e., a mapping from observations to actions)
allowing it to solve the task encoded in the environment by maximizing the cumulative
reward signal obtained after performing an action.

In *QUEENS* terminology, the environment in it's most general form can be thought
of as a **model** which encodes the problem at hand, e.g., in the form of a physical simulation,
and can be evaluated in forward fashion.
The RL agent is trained by letting the algorithm repeatedly interact with the environment
and learning a suitable policy from the collected experience. Once the agent is trained,
it can be used to make predictions about the next action to perform based on a given
observation. As such, the agent can be considered as a **surrogate model** as
it first needs to be trained before being able to make predictions. Following the
*QUEENS* terminology for models, a **sample** corresponds to an **observation** and
the **response** of the RL model corresponds to the **action** to be taken.

This interpretation of RL in the context of *QUEENS* has been reflected in the
design of the :py:class:`RLModel` class.
"""

import logging

from queens.models.model import Model
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class RLModel(Model):
    """Main class for administering RL tasks within *QUEENS*.

    The training or evaluation of an RLModel instance can be performed by using
    an instance of type :py:class:`queens.iterators.RLIterator`.

    Attributes:
        _agent (object): An instance of a *stable-baselines3* agent.
        _render_on_evaluation (bool): Flag indicating whether to render the
            environment during evaluation.
        _total_timesteps (int): Total number of timesteps to train the agent.
        _vec_env (object): A vectorized environment for evaluation.
        is_trained (bool): Flag indicating whether the agent has been trained.
        response (dict): The response of the model evaluation.
    """

    @log_init_args
    def __init__(self, agent, total_timesteps=10000, render_on_evaluation=False):
        """Initialize an RLModel instance.

        Args:
            agent (object): An instance of a *stable-baselines3* agent.
            total_timesteps (int): Total number of timesteps to train the agent.
            render_on_evaluation (bool): Flag indicating whether to render the
                environment during evaluation.
        """
        super().__init__()

        self.is_trained = False

        # Store the provided agent instance
        self._agent = agent
        # Retrieve a (vectorized) stable-baseline3 environment for evaluation
        self._vec_env = self._agent.get_env()

        self._total_timesteps = total_timesteps
        self._render_on_evaluation = render_on_evaluation

    def interact(self, observation):
        """Perform one interaction step of an RL agent with an environment.

        One interaction consists of the following steps:
            1. Predict the next action based on the current observation, see :py:meth:`predict()`.
            2. Apply the predicted action to the environment, see :py:meth:`step()`.
            3. Optionally render the environment depending on the value of
               :py:attr:`_render_on_evaluation`, see :py:meth:`render()`.
            4. Return the new observation of the environment.

        Args:
            observation (np.ndarray): The observation of the current state of
                                      the environment.

        Returns:
            result (dict): A dictionary containing all the results generated during
                this interaction step, such as the undertaken action, the new observation,
                and the reward obtained from the environment.
        """
        _logger.info("Computing one agent-enviroment interaction (i.e., one timestep).")
        result = self.predict(observation)
        # return values are: observation, reward, done, info
        obs, reward, done, info = self.step(result["action"])
        if self._render_on_evaluation:
            self.render()
        _logger.info("Interaction completed.")
        # add the additional information to the result dict
        result.update(
            {
                "observation": obs,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        return result

    def get_initial_observation(self):
        """Returns a (random) initial state of the environment.

        This method can be used to generate an inital observation of the
        environment to be used as the starting point for an evalution of a
        trained agent.

        Returns:
            np.ndarray: (Random) Initial observation of the environment.
        """
        return self._vec_env.reset()

    def grad(self, samples, upstream_gradient):
        """Evaluate the gradient of the model wrt. the provided input samples.

        .. warning::
                This method is currently not implemented for RL models.

        Raises:
            NotImplementedError: If the method is called.
        """
        raise NotImplementedError(
            "Gradient information not available. \n"
            "If you need gradients, please use a different model or implement "
            "the `grad` method in the child class."
        )

    def evaluate(self, samples):
        """Evaluate the model (agent) on the provided samples (observations).

        Delegates the call to :py:meth:`predict()` internally and stores the of
        the model evaluation in the internal storage variable :py:attr:`response`.

        Args:
            samples (np.ndarray): Input samples, i.e., multiple observations.

        Returns:
            dict: Results (actions) corresponding to current set of input samples.
        """
        # Predict the next actions (and states) based on the current
        # observations of the environment
        self.response = self.predict(samples)
        return self.response

    def predict(self, observations):
        """Predict the actions to be undertaken for given observations.

        Args:
            observations (np.ndarray): Either a single observation or a batch of observations.

        Returns:
            result (dict): Actions and new (hidden) states corresponding to the provided
                observations. The prediction actions are stored as the main result of this
                model.
        """
        _logger.debug("Predicting the next action based on the current state of the environment.")
        # Predict the agent's action and the new (hidden) state of the
        # environment based on the current observation
        actions, hidden_states = self._agent.predict(observations)
        # combine information into a dict
        result = {
            "result": actions,  # this is redundant, but kept for compatibility
            "action": actions,
            "hidden_state": hidden_states,
        }
        return result

    def render(self):
        """Render the current state of the environment.

        .. note::
                Internally delegates the call to the ``render()`` method of the
                vectorized environment. Render settings can be controlled via
                the constructor of the environment.
        """
        self._vec_env.render()

    def step(self, action):
        """Perform a single step in the environment.

        Applys the provided action to the environment.

        Args:
            action (np.ndarray): Action to be executed.

        Returns:
            observation (np.ndarray): Observation of the new state of the environment.
            reward (float): Reward obtained from the environment.
            done (bool): Flag indicating whether the episode has finished.
            info (dict): Additional information.
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
