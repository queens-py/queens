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
"""Fixtures for the integration tests."""

import numpy as np
import pytest

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.models.simulation import Simulation
from queens.parameters import Parameters
from queens.schedulers.pool import Pool


@pytest.fixture(name="pool_scheduler")
def fixture_pool_scheduler(global_settings):
    """Pool scheduler for the integration tests."""
    return Pool(experiment_name=global_settings.experiment_name)


@pytest.fixture(name="borehole_parameters")
def fixture_borehole_parameters():
    """Parameters for the Borehole benchmark function."""
    rw = Uniform(lower_bound=0.05, upper_bound=0.15)
    r = Uniform(lower_bound=100, upper_bound=50000)
    tu = Uniform(lower_bound=63070, upper_bound=115600)
    hu = Uniform(lower_bound=990, upper_bound=1110)
    tl = Uniform(lower_bound=63.1, upper_bound=116)
    hl = Uniform(lower_bound=700, upper_bound=820)
    l = Uniform(lower_bound=1120, upper_bound=1680)
    kw = Uniform(lower_bound=9855, upper_bound=12045)
    return Parameters(rw=rw, r=r, tu=tu, hu=hu, tl=tl, hl=hl, l=l, kw=kw)


@pytest.fixture(name="park91a_parameters")
def fixture_park91a_parameters():
    """Parameters for the Park91a benchmark function."""
    x1 = Uniform(lower_bound=0, upper_bound=1)
    x2 = Uniform(lower_bound=0, upper_bound=1)
    x3 = Uniform(lower_bound=0, upper_bound=1)
    x4 = Uniform(lower_bound=0, upper_bound=1)
    return Parameters(x1=x1, x2=x2, x3=x3, x4=x4)


@pytest.fixture(name="borehole83_lofi_model")
def fixture_borehole83_lofi_model(borehole_parameters, pool_scheduler):
    """Simulation model wrapping the low-fidelity Borehole function."""
    driver = Function(parameters=borehole_parameters, function="borehole83_lofi")
    return Simulation(scheduler=pool_scheduler, driver=driver)


@pytest.fixture(name="borehole83_hifi_model")
def fixture_borehole83_hifi_model(borehole_parameters, pool_scheduler):
    """Simulation model wrapping the high-fidelity Borehole function."""
    driver = Function(parameters=borehole_parameters, function="borehole83_hifi")
    return Simulation(scheduler=pool_scheduler, driver=driver)


@pytest.fixture(name="park91a_lofi_model")
def fixture_park91a_lofi_model(park91a_parameters, pool_scheduler):
    """Simulation model wrapping the low-fidelity Park91a function."""
    driver = Function(parameters=park91a_parameters, function="park91a_lofi")
    return Simulation(scheduler=pool_scheduler, driver=driver)


@pytest.fixture(name="park91a_hifi_model")
def fixture_park91a_hifi_model(park91a_parameters, pool_scheduler):
    """Simulation model wrapping the high-fidelity Park91a function."""
    driver = Function(parameters=park91a_parameters, function="park91a_hifi")
    return Simulation(scheduler=pool_scheduler, driver=driver)


@pytest.fixture(name="fourc_example_expected_output")
def fixture_fourc_example_expected_output():
    """Expected outputs for the 4C example."""
    result = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [0.1195746416995907, -0.002800078802811129, -0.005486393250866545],
                [0.1260656705382511, -0.002839272898349505, -0.005591796485367413],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.1322571001660478, -0.00290530963354552, -0.005635750492708091],
                [0.1387400363966301, -0.002944141371541845, -0.005740445608910146],
                [0.0, 0.0, 0.0],
                [0.1195746416995907, -0.002800078802811129, -0.005486393250866545],
                [0.2289195764727486, -0.01428888900910762, -0.02789834740243489],
                [0.24879304060717, -0.01437712967153365, -0.02801932699697155],
                [0.1260656705382511, -0.002839272898349505, -0.005591796485367413],
                [0.1322571001660478, -0.00290530963354552, -0.005635750492708091],
                [0.2674182375147958, -0.01440529789560568, -0.02822643380369276],
                [0.2865938203259575, -0.01448421374089244, -0.02832919236100399],
                [0.1387400363966301, -0.002944141371541845, -0.005740445608910146],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.1324695606381951, -0.00626315166779166, -0.003933720977121313],
                [0.1472676510502675, -0.006473749929398404, -0.004119847415735578],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.1417895586931143, -0.006449559916104635, -0.004017410516711057],
                [0.1565760384270568, -0.006658631448567143, -0.004202575461905436],
                [0.0, 0.0, 0.0],
                [0.1324695606381951, -0.00626315166779166, -0.003933720977121313],
                [0.250425194294125, -0.0327911259093084, -0.02066877798205112],
                [0.2973138520084483, -0.03326225476238921, -0.02086982506734491],
                [0.1472676510502675, -0.006473749929398404, -0.004119847415735578],
                [0.1417895586931143, -0.006449559916104635, -0.004017410516711057],
                [0.2801976826134401, -0.03299992343589411, -0.0208604824244337],
                [0.3257952884928654, -0.03343065913125062, -0.02103608880631498],
                [0.1565760384270568, -0.006658631448567143, -0.004202575461905436],
            ],
        ]
    )
    return result
