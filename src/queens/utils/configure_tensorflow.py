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
"""Utils related to tensorflow and friends."""

import logging
import os
from types import ModuleType

_logger = logging.getLogger(__name__)


def configure_tensorflow(tensorflow: ModuleType) -> None:
    """Configure tensorflow.

    Args:
        tensorflow: The module to configure
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tensorflow.get_logger().setLevel(logging.ERROR)

    # Use GPU acceleration if possible
    if tensorflow.test.gpu_device_name() != "/device:GPU:0":
        _logger.info("WARNING: GPU device not found.")
    else:
        _logger.info("SUCCESS: Found GPU: %s", tensorflow.test.gpu_device_name())


def configure_keras(tf_keras: ModuleType) -> None:
    """Configure tf keras.

    Args:
        tf_keras: The module configuration
    """
    tf_keras.backend.set_floatx("float64")
