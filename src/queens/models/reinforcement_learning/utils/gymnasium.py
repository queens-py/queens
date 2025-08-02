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
"""Utility functions for working with gymnasium environments.

This module essentially provides a function to create a gymnasium
environment from its name to facilitate the creation of gymnasium
environments in QUEENS for users who are not familiar with the package.
"""

import logging

import gymnasium as gym

from queens.utils.valid_options import check_if_valid_options

_logger = logging.getLogger(__name__)
_supported_gym_environments = list(gym.envs.registry.keys())


def create_gym_environment(env_name, env_options=None, seed=None):
    """Convenience function to create a gymnasium environment.

    Args:
        env_name (str): Name of the gymnasium environment to create.
        env_options (dict, optional): Dictionary of options to pass to the environment.
        seed (int, optional): Seed to use for the environment.

    Returns:
        env (gymnasium.Env): An instance of the created gymnasium environment.

    Raises:
        InvalidOptionError: If the provided environment name is not known to gymnasium.
    """
    check_if_valid_options(_supported_gym_environments, env_name, "Unknown gymnasium environment!")

    # if no options are provided, create an empty dictionary to be able to
    # unpack it without errors
    env_options = env_options or {}

    # If the environment name is known, create an environment instance
    env = gym.make(env_name, **env_options)

    if seed is not None:
        _logger.debug("Setting seed for the environment to %d.", seed)
        env.reset(seed=seed)

    return env
