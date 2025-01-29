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
"""TODO: Document this."""

import inspect

import gymnasium as gym
import stable_baselines3 as sb3


def _create_supported_agents_dict():
    """TODO: Document this."""
    # Get all agent classes in the stable_baselines3 module
    supported_agents = {
        name: obj
        for name, obj in inspect.getmembers(sb3)
        if inspect.isclass(obj) and issubclass(obj, sb3.common.base_class.BaseAlgorithm)
    }
    return supported_agents


_supported_sb3_agents = _create_supported_agents_dict()


def get_supported_sb3_policies(agent_class):
    """TODO: Document this."""
    if issubclass(agent_class, sb3.common.base_class.BaseAlgorithm):
        return list(agent_class.policy_aliases.keys())

    raise ValueError(
        f"{agent_class.__name__} is not a stable-baselines3 agent.\n"
        "Cannot find supported policies for this class!"
    )


_supported_gym_environments = list(gym.envs.registry.keys())


def create_gym_environment(env_name, env_options):
    """TODO: Document this."""
    if env_name not in _supported_gym_environments:
        raise ValueError(f"Environment `{env_name}` is not known to gymnasium")

    # If the environment name is known, create an environment instance
    env = gym.make(env_name, **env_options)

    return env


def create_sb3_agent(agent_name, policy_name, env, agent_options):
    """TODO: Document this."""
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
