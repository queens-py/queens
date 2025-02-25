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
"""Utiltiy functions for working with stable-baselines3 agents.

This module provides utility functions for working with stable-
baselines3 agents in QUEENS. The idea is to enable a QUEENS user who is
not familiar with the stable-baselines3 reinforcement learning library
but wants to try out RL for their problem to easily create and use a
reinforcement learning agent in QUEENS without first studying the
package. If you are familiar with stable-baselines3, you can as well
create the agents yourself.
"""

import inspect
import logging
import os
import random

import numpy as np
import stable_baselines3 as sb3
import torch

from queens.utils.valid_options_utils import check_if_valid_options

_logger = logging.getLogger(__name__)

_supported_sb3_agents = {
    name: obj
    for name, obj in inspect.getmembers(sb3)
    if inspect.isclass(obj) and issubclass(obj, sb3.common.base_class.BaseAlgorithm)
}


def get_supported_sb3_policies(agent_class):
    """Looks up the supported policies for a stable-baselines3 agent class.

    Args:
        agent_class (class): A stable-baselines3 agent class.

    Returns:
        list: A list of strings representing the supported policies for the given agent class.

    Raises:
        ValueError: If the provided class is not a stable-baselines3 agent class or does not provide
            a ``policy_aliases`` attribute.
    """
    if issubclass(agent_class, sb3.common.base_class.BaseAlgorithm):
        if hasattr(agent_class, "policy_aliases"):
            return list(agent_class.policy_aliases.keys())
        error_message = f"{agent_class.__name__} does not provide a 'policy_aliases' attribute."
    else:
        error_message = f"{agent_class.__name__} is not a stable-baselines3 agent."

    raise ValueError(error_message)


def create_sb3_agent(agent_name, policy_name, env, agent_options=None):
    """Creates a stable-baselines3 agent based on its name as string.

    Looks up whether the provided agent name corresponds to an agent supported by
    stable-baselines3 and creates an instance of the agent with the provided policy
    and environment. Options for modifying the agent's optional parameters can be
    provided as a dictionary.

    Args:
        agent_name (str): The name of the agent to create.
        policy_name (str): The name of the policy to use with the agent.
        env (gymnasium.Env): The environment to train the agent on. For a convenience function to
            create a predefined gymnasium environment, see
            :py:meth:`queens.models.reinforcement_learning.utils.gymnasium.create_gym_environment`.
        agent_options (dict, optional): A dictionary of optional parameters to pass to the agent.

    Returns:
        agent (stable_baselines3.BaseAlgorithm): An instance of the created agent.

    Raises:
        ValueError: If the provided agent name is not supported by stable-baselines3
        InvalidOptionError: If the provided agent name is not known to stable-baselines3 or the
            provided policy name is not supported by the chosen agent class.
    """
    # Check that a valid agent has been provided
    check_if_valid_options(_supported_sb3_agents, agent_name, "Agent unknown to stable-baselines3!")

    # Retrieve the corresponding agent class
    agent_class = _supported_sb3_agents[agent_name]

    # Check that the provided policy is compatible with the chosen agent
    supported_sb3_policies = get_supported_sb3_policies(agent_class)
    check_if_valid_options(
        supported_sb3_policies, policy_name, f"Unsupported policy for agent {agent_name}!"
    )

    # if no options are provided, create an empty dictionary to be able to
    # unpack it without errors
    agent_options = agent_options or {}

    # create the agent instance with the provided parameters
    agent = agent_class(policy_name, env, **agent_options)

    return agent


def make_deterministic(seed, disable_gpu=True):
    """Make the random number generation deterministic for an agent training.

    This is achieved by setting the random seed for all python libraries
    that are involved in training and stable-baselines3 agent.

    .. note::
            This function should be called before creating the agent.
            Make sure to also pass a seed to the agent and the environment.

            * For the environment, you wanna invoke the ``reset()`` method
              after creation and pass the seed as parameter, e.g., ``env.reset(seed=seed)``.
              This needs to be done before passing the environment to the agent.
            * For the agent, this can be achieved by adding the entry
              ``"seed": seed,`` to the ``agent_options`` dict.

    .. warning::
            Since GPU computations can result in non-deterministic computations,
            this functions modifies the ``CUDA_VISIBLE_DEVICES`` environment
            variable to disable GPU computations. This behavior can be changed
            by adapting the ``disable_gpu`` parameter.

    Args:
        seed (int): The random seed to set for all libraries.
        disable_gpu (bool, optional): Flag indicating whether to disable GPU
                computations. Defaults to True.
    """
    _logger.info("Setting random seed of libraries to %s", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sb3.common.utils.set_random_seed(seed)

    if disable_gpu:
        _logger.info("Disabling GPU computations.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_model(agent_name, path, experiment_name, env=None):
    """Convenience function for loading a stable-baselines3 agent from file.

    Checks whether the agent is of an off-policy type and if so loads its
    replay buffer as well. The latter is required if a continuation of the
    training is desired.

    Args:
        agent_name (str): The name of the agent to load.
        path (str, Path): The path to the directory containing the agent to load.
        experiment_name (str): The name of the QUEENS experiment that was used
            to train the agent (contained in the filename).
        env (gymnasium.Env, optional): The environment on which the agent was trained.

    Returns:
        agent (stable_baselines3.BaseAlgorithm): The loaded agent.
    """
    # Determine the file stem (works for both Path objects and strings)
    file_stem = f"{path}/{experiment_name}"

    # Check that a valid agent has been provided
    check_if_valid_options(_supported_sb3_agents, agent_name, "Agent unknown to stable-baselines3!")
    # Retrieve the corresponding agent class
    agent_class = _supported_sb3_agents[agent_name]

    # Load the agent from file
    agent_file = f"{file_stem}_agent.zip"
    _logger.info("Loading agent from file %s", agent_file)
    agent = agent_class.load(agent_file, env=env)

    if env is None:
        _logger.warning(
            "No environment provided!\n"
            "The agent is loaded without an environment and cannot be used for "
            "predictions. Make sure to attach an environment by calling `set_env()` "
            "on the agent instance to attach an environment before using it for "
            "evaluation."
        )

    # Check and load replay buffer if it exists
    if hasattr(agent, "replay_buffer"):
        _logger.debug("Agent is of off-policy type. Attempting to load replay buffer.")
        replay_buffer_file = f"{file_stem}_replay_buffer.pickle"
        if os.path.exists(replay_buffer_file):
            _logger.debug("Loading replay buffer from file %s", replay_buffer_file)
            agent.load_replay_buffer(replay_buffer_file)
        else:
            _logger.warning(
                "No replay buffer found for the agent.\n"
                "You cannot continue the training with this agent."
            )

    return agent


def save_model(agent, gs):
    """Save a (trained) stable-baselines3 agent to a file.

    Checks whether the agent is of an off-policy type and if so stores its
    replay buffer as well. The latter is required if a continuation of the
    training is desired.

    Args:
        agent (stable_baselines3.BaseAlgorithm): The trained agent to save.
        gs (GlobalSettings): The global settings of the QUEENS experiment
            (needed to retrieve the experiment name and the output directory of
            the current run).
    """
    # Save the agent to file
    agent_file = gs.result_file(".zip", suffix="_agent")
    _logger.info("Saving agent to %s", agent_file)
    agent.save(agent_file)

    # Check and save replay buffer if it exists
    if hasattr(agent, "replay_buffer") and agent.replay_buffer is not None:
        _logger.debug(
            "Agent is of off-policy type and has a replay buffer. "
            "Saving the replay buffer as well."
        )
        replay_buffer_file = gs.result_file(".pickle", suffix="_replay_buffer")
        _logger.debug("Writing replay buffer to %s", replay_buffer_file)
        agent.save_replay_buffer(replay_buffer_file)
