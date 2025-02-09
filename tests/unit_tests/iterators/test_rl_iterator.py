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
from queens.models.rl_models.rl_model import RLModel


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
