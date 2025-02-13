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
"""Unit tests for the RLModel class."""

import numpy as np
import pytest
from mock import Mock

from queens.models.rl_models.rl_model import RLModel
from queens.models.rl_models.utils.gym_utils import create_gym_environment
from queens.models.rl_models.utils.sb3_utils import create_sb3_agent, make_deterministic

SEED = 429


@pytest.fixture(name="custom_agent")
def fixture_custom_agent(agent_name, environment_name):
    """Provide a valid agent instance for testing."""
    make_deterministic(SEED)
    # create a gym environment
    env = create_gym_environment(environment_name, seed=SEED)
    # create an agent
    agent = create_sb3_agent(agent_name, "MlpPolicy", env, agent_options={"seed": SEED})
    return agent


# ------------------ actual unit tests --------------------------- #
@pytest.mark.parametrize(
    "render_mode",
    [
        None,
        "human",
        "rgb_array",
        "ansi",
        pytest.param("rgb-array", marks=pytest.mark.xfail(strict=True, raises=ValueError)),
    ],
)
def test_rl_model_init(render_mode):
    """Unit tests for the RLModel class."""
    # Create the model
    agent = Mock()
    model = RLModel(agent, render_mode=render_mode, total_timesteps=1_000)

    # Check whether setting up the model worked correctly
    assert model._agent == agent  # pylint: disable=W0212
    assert model._deterministic_actions is False  # pylint: disable=W0212
    assert model._render_mode == render_mode  # pylint: disable=W0212
    assert model._total_timesteps == 1_000  # pylint: disable=W0212
    assert model._vec_env is not None  # pylint: disable=W0212
    assert isinstance(model.frames, list) and len(model.frames) == 0
    assert model.is_trained is False


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
def test_rl_model_training_and_evaluation(custom_agent):
    """Test the training of the RLModel class."""
    # Create the model and make it fully deterministic
    model = RLModel(custom_agent, total_timesteps=100, deterministic_actions=True)

    # Check whether the model is not trained yet
    assert model.is_trained is False

    # Train the model
    model.train()

    # Check whether the model is trained
    assert model.is_trained is True

    # Generate an initial observation
    obs = model.reset(seed=SEED)

    # Perform an interaction with the trained agent
    interaction_result = model.interact(obs)

    # Make sure the resulting dict contains all expected values
    assert "result" in interaction_result
    assert interaction_result["result"] is not None
    assert "action" in interaction_result
    assert interaction_result["action"] is not None
    assert "new_obs" in interaction_result
    assert interaction_result["new_obs"] is not None
    assert "reward" in interaction_result
    assert interaction_result["reward"] is not None
    assert "done" in interaction_result
    assert interaction_result["done"] is not None
    assert "info" in interaction_result
    assert interaction_result["info"] is not None

    # result and action are supposed to be the same fields
    np.testing.assert_array_equal(interaction_result["action"], interaction_result["result"])

    # reset the state of the environment
    model.reset(seed=SEED)

    # Perform a prediction with the agent, using the same observation as above
    prediction_result = model.predict(obs, deterministic=True)

    # Make sure the resulting dict contains all expected values
    assert "result" in prediction_result
    assert prediction_result["result"] is not None
    assert "action" in prediction_result
    assert prediction_result["action"] is not None

    # result and action are supposed to be the same fields
    np.testing.assert_array_equal(prediction_result["action"], prediction_result["result"])

    # Make sure that the shared entries of prediction_result and interaction_result
    # are identical, because they were invoked with the same observation and same
    # trained state of the agent and everything should be deterministic
    np.testing.assert_array_equal(interaction_result["action"], prediction_result["action"])
    np.testing.assert_array_equal(interaction_result["result"], prediction_result["result"])
