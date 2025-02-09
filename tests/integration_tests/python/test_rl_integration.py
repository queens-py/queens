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
"""Integration test for the reinforcement learning iterator."""

import numpy as np
import pytest

from queens.iterators.rl_iterator import RLIterator
from queens.main import run_iterator
from queens.models.rl_models.rl_model import RLModel
from queens.models.rl_models.utils.gym_utils import create_gym_environment
from queens.models.rl_models.utils.sb3_utils import create_sb3_agent, make_deterministic
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result

SEED = 429


def create_agent(agent_name, environment_name):
    """Provide a valid agent instance for testing."""
    # create a gym environment
    env = create_gym_environment(environment_name, seed=SEED)
    # create an agent
    agent = create_sb3_agent(agent_name, "MlpPolicy", env, agent_options={"seed": SEED})
    return agent


def create_model(agent):
    """Provide a valid (i.e., deterministic) RL model for testing."""
    # Create the model
    model = RLModel(agent, total_timesteps=500, deterministic_actions=True)
    return model


def create_iterator(agent_name, environment_name, gs):
    """Provide a vald RL iterator for training or evaluation."""
    # Create the agent
    agent = create_agent(agent_name, environment_name)
    # Create the model
    model = create_model(agent)
    # Create the iterator
    iterator = RLIterator(
        model,
        parameters=Parameters(),
        global_settings=gs,
        result_description={"write_results": True},
    )
    return iterator


# ------------------ actual unit tests --------------------------- #
@pytest.mark.parametrize(
    "agent_name, environment_name",
    [
        ("A2C", "CartPole-v1"),
        ("PPO", "CartPole-v1"),
        ("DQN", "CartPole-v1"),
        ("A2C", "Pendulum-v1"),
        ("PPO", "Pendulum-v1"),
        ("SAC", "Pendulum-v1"),
        ("TD3", "Pendulum-v1"),
        ("DDPG", "Pendulum-v1"),
    ],
)
def test_rl_integration(agent_name, environment_name, global_settings):
    """Integration test for the reinforcement learning iterator."""
    # Fix all randomness with a seed
    make_deterministic(SEED)

    ### STEP 1 - Train model A ###
    iterator_a = create_iterator(agent_name, environment_name, global_settings)
    run_iterator(iterator_a, global_settings=global_settings)

    ### STEP 2 - Evaluate model A ###
    # Evaluate the trained agent
    iterator_a.mode = "evaluation"
    iterator_a.interaction_steps = 1000
    # RL-training
    run_iterator(iterator_a, global_settings=global_settings)

    ### STEP 3 - Train model B ###
    iterator_b = create_iterator(agent_name, environment_name, global_settings)
    run_iterator(iterator_b, global_settings=global_settings)
    model_b = iterator_b.model

    ### STEP 4 - Load stored result of model A evaluation ###
    results = load_result(global_settings.result_file(".pickle"))

    # TODO  # pylint: disable=W0511
    # Check entries of results dictionary

    # retrieve the data as numpy arrays
    expected_results = results["raw_output_data"]["actions"]
    # extract the number of observations...
    n_obs = results["input_data"].shape[-1]
    # ... and reshape the array to allow for batch evaluation
    inputs = results["input_data"].reshape(-1, n_obs)

    ### STEP 5 - Evaluate model B on the inputs of model A ###
    predictions = model_b.predict(inputs)

    ### STEP 6 - Compare results ###
    np.testing.assert_array_equal(expected_results, predictions)
