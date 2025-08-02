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
"""Base module for iterators or methods."""

import abc


class Iterator(metaclass=abc.ABCMeta):
    """Base class for Iterator hierarchy.

    This Iterator class is the base class for one of the primary class
    hierarchies in QUEENS. The job of the iterator hierarchy is to
    coordinate and execute simulations/function evaluations.

    Attributes:
        model (obj): Model to be evaluated by iterator.
        parameters: Parameters object
        global_settings (GlobalSettings): settings of the QUEENS experiment including its name and
                                          the output directory
    """

    def __init__(self, model, parameters, global_settings):
        """Initialize iterator object.

        Args:
            model (Model): Model to be evaluated by iterator.
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
        """
        self.model = model
        self.global_settings = global_settings
        self.parameters = parameters

    def pre_run(self):
        """Optional pre-run portion of run."""

    @abc.abstractmethod
    def core_run(self):
        """Core part of the run, implemented by all derived classes."""

    def post_run(self):
        """Optional post-run portion of run.

        E.g. for doing some post processing.
        """

    def run(self):
        """Orchestrate pre/core/post phases."""
        self.pre_run()
        self.core_run()
        self.post_run()
