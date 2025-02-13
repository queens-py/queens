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
"""Unit test for RLIterator."""

import numpy as np
import pytest
from mock import Mock

from queens.iterators.rl_iterator import RLIterator
from queens.main import run_iterator
from queens.models.rl_models.rl_model import RLModel
from queens.models.rl_models.utils.gym_utils import create_gym_environment
from queens.models.rl_models.utils.sb3_utils import create_sb3_agent, load_model

SEED = 429


def create_iterator(agent_name, environment_name, global_settings):
    """Create a fully-functional RLIterator instance."""
    env = create_gym_environment(environment_name, seed=SEED)
    agent = create_sb3_agent(agent_name, "MlpPolicy", env, agent_options={"seed": SEED})
    iterator = RLIterator(
        model=RLModel(agent, total_timesteps=500),
        parameters=Mock(),
        global_settings=global_settings,
        result_description={"write_results": True},
    )
    return iterator


# ------------------ actual unit tests --------------------------- #
@pytest.mark.parametrize(
    "mode,steps",
    [
        ("evaluation", 1_000),
        ("training", 500),
        pytest.param("interaction", 100, marks=pytest.mark.xfail(strict=True, raises=ValueError)),
        pytest.param("evaluation", -5, marks=pytest.mark.xfail(strict=True, raises=ValueError)),
        pytest.param("training", -10_000, marks=pytest.mark.xfail(strict=True, raises=ValueError)),
    ],
)
def test_rl_iterator_initialization_and_properties(mode, steps):
    """Test the constructor."""
    # prepare the mock parameters
    model = Mock(spec=RLModel)
    parameters = Mock()
    global_settings = Mock()

    # prepare the meaningful parameters
    result_description = {
        "write_results": True,
    }

    # generate a random observation
    obs = np.random.random(size=(5, 1))

    # create the iterator instance
    iterator = RLIterator(
        model,
        parameters,
        global_settings,
        result_description=result_description,
        mode=mode,
        interaction_steps=steps,
        initial_observation=obs,
    )

    # check whether initialization worked correctly
    assert iterator.result_description == result_description
    assert iterator.mode == mode
    assert iterator.interaction_steps == steps
    np.testing.assert_array_equal(iterator.initial_observation, obs)
    assert iterator.samples is None
    assert iterator.output is None


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
def test_save_and_load(agent_name, environment_name, global_settings):
    """Test saving and loading of an RLIterator instance."""
    ### STEP 1 - Create, train, and save iterator A ###
    iterator = create_iterator(agent_name, environment_name, global_settings)
    run_iterator(iterator, global_settings)
    rl_model = iterator.model

    ### STEP 2 - Create a new agent instance and load state of trained agent ###
    sb3_agent = load_model(
        agent_name,
        global_settings.result_file(".pickle").parent,
        global_settings.result_file(".pickle").stem,
        create_gym_environment(environment_name, seed=SEED),
    )

    ### STEP 3 - Generate random observations ###
    obs = rl_model.reset()
    n_obs = obs.shape[-1]
    samples = np.random.normal(0.0, 2.0, size=(1000, n_obs))

    ### STEP 4 - Make deterministic predictions with both models A and B
    rl_model_result = rl_model.predict(samples, deterministic=True)
    sb3_agent_result = sb3_agent.predict(samples, deterministic=True)

    # Extract the respective predictions
    rl_model_actions = rl_model_result["action"]
    sb3_agent_actions = sb3_agent_result[0]  # tuple, first entry are actions

    # The results need to be the same, if the stored model is identical to the
    # one still in memory
    np.testing.assert_array_equal(rl_model_actions, sb3_agent_actions)
