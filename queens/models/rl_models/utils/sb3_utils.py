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
"""Utiltiy functions for working with *stable-baselines3* agents.

This module provides utility functions for working with *stable-
baselines3* agents in *QUEENS*. The idea is to enable a *QUEENS* user
who is not familiar with the *stable-baselines* reinforcement learning
library but wants to try out RL for their problem to easily create and
use a reinforcement learning agent in *QUEENS* without first studying
the package. If you are familiar with *stable-baselines3*, you can as
well create the agents yourself.
"""

import inspect

import stable_baselines3 as sb3


def _create_supported_agents_dict():
    """Create a dictionary of supported *stable-baselines3* agents.

    In order to work even with updates of the *stable-baselines3* library, this
    function dynamically looks up the supported agents based on the currently
    installed library version.

    Returns:
        supported_agents (dict): A dictionary of the currently supported *stable-baselines3* agents.
    """
    # Get all agent classes in the stable_baselines3 module
    supported_agents = {
        name: obj
        for name, obj in inspect.getmembers(sb3)
        if inspect.isclass(obj) and issubclass(obj, sb3.common.base_class.BaseAlgorithm)
    }
    return supported_agents


_supported_sb3_agents = _create_supported_agents_dict()


def get_supported_sb3_policies(agent_class):
    """Looks up the supported policies for a *stable-baselines3* agent class.

    Args:
        agent_class (class): A *stable-baselines3* agent class.

    Returns:
        list: A list of strings representing the supported policies for the given agent class.

    Raises:
        ValueError: If the provided class is not a *stable-baselines3* agent class.
    """
    if issubclass(agent_class, sb3.common.base_class.BaseAlgorithm):
        return list(agent_class.policy_aliases.keys())

    raise ValueError(
        f"{agent_class.__name__} is not a stable-baselines3 agent.\n"
        "Cannot find supported policies for this class!"
    )


def create_sb3_agent(agent_name, policy_name, env, agent_options):
    """Creates a *stable-baselines3* agent based on its name as string.

    Looks up whether the provided agent name corresponds to an agent supported by
    *stable-baselines3* and creates an instance of the agent with the provided policy
    and environment. Options for modifying the agent's optional parameters can be
    provided as a dictionary.

    Args:
        agent_name (str): The name of the agent to create.
        policy_name (str): The name of the policy to use with the agent.
        env (gymnasium.Env): The environment to train the agent on. For a convenience
            function to create a predefined *gymnasium* environment, see
            :py:meth:`queens.models.rl_models.utils.gym_utils.create_gym_environment`.
        agent_options (dict): A dictionary of optional parameters to pass to the agent.

    Returns:
        agent (stable_baselines3.BaseAlgorithm): An instance of the created agent.

    Raises:
        ValueError: If the provided agent name is not supported by *stable-baselines3*
            or does not support the provided policy
    """
    # Check that a valid agent has been provided
    agent_name = agent_name.upper()
    if agent_name not in _supported_sb3_agents.keys():
        raise ValueError(f"Unsupported agent: {agent_name}")

    # Retrieve the corresponding agent class
    agent_class = _supported_sb3_agents[agent_name]

    # Check that the provided policy is compatible with the chosen agent
    supported_sb3_policies = get_supported_sb3_policies(agent_class)
    if policy_name not in supported_sb3_policies:
        raise ValueError(
            f"Unsupported policy: `{policy_name}` for agent `{agent_name}`:\n"
            f"Agent {agent_name} only supports the following policies: "
            f"{supported_sb3_policies}!"
        )

    # create the agent instance with the provided parameters
    agent = agent_class(policy_name, env, **agent_options)

    return agent
