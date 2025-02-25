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
"""Module for reinforcement learning capabilities.

.. note::
        If you have no prior experience with RL, a good starting point might be
        the introduction of Spinning Up in Deep RL by OpenAI:
        https://spinningup.openai.com/en/latest/spinningup/rl_intro.html.

In the follwing, we provide a brief overview of RL concepts and terminology and their relation to
QUEENS.

In its essence, Reinformcement Learning (RL) is a type of machine learning which tries to mimick
they way how humans learn to accomplish a new task, namely by performing trial-and-error
interactions with their environment and learning from the gathered experience.

In RL, this interaction happens between a so-called **agent** (i.e., a learning algorithm) and an
**environment** (i.e., the task or problem to be solved). The agent can perform **actions** in the
environment in order to modify its state and receives **observations** (i.e., of the new state of
the environment after applying the action) and **rewards** (i.e., a numerical reward signal
quantifying how well the undertaken action was with respect to solving the problem encoded in the
environment) in return. One interaction step between the agent and the environment is called a
**timestep**. The goal of the agent is to learn a **policy** (i.e., a mapping from observations to
actions) allowing it to solve the task encoded in the environment by maximizing the cumulative
reward signal obtained after performing an action.

In QUEENS terminology, the environment in it's most general form can be thought of as a **model**
which encodes the problem at hand, e.g., in the form of a physical simulation, and can be evaluated
in forward fashion.
The RL agent is trained by letting the algorithm repeatedly interact with the environment and
learning a suitable policy from the collected experience. Once the agent is trained, it can be used
to make predictions about the next action to perform based on a given observation. As such, the
agent can be considered as a **surrogate model** as it first needs to be trained before being able
to make predictions. Following the QUEENS terminology for models, a **sample** corresponds to an
**observation** and the **response** of the RL model corresponds to the **action** to be taken.

This interpretation of RL in the context of QUEENS has been reflected in the design of the
:py:class:`queens.models.reinforcement_learning.reinforcement_learning.ReinforcementLearning` class.
"""
