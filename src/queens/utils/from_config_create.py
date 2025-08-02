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
"""From config create utils."""

import logging
import types

from queens.data_processors import VALID_TYPES as VALID_DATA_PROCESSOR_TYPES
from queens.distributions import VALID_TYPES as VALID_DISTRIBUTION_TYPES
from queens.drivers import VALID_TYPES as VALID_DRIVER_TYPES
from queens.drivers._driver import Driver
from queens.external_geometries import VALID_TYPES as VALID_EXTERNAL_GEOMETRY_TYPES
from queens.iterators import VALID_TYPES as VALID_ITERATOR_TYPES
from queens.iterators._iterator import Iterator
from queens.models import VALID_TYPES as VALID_MODEL_TYPES
from queens.models.bmfmc import BMFMC
from queens.parameters.parameters import from_config_create_parameters
from queens.parameters.random_fields import VALID_TYPES as VALID_RANDOM_FIELD_TYPES
from queens.schedulers import VALID_TYPES as VALID_SCHEDULER_TYPES
from queens.schedulers import Scheduler
from queens.stochastic_optimizers import VALID_TYPES as VALID_STOCHASTIC_OPTIMIZER_TYPES
from queens.utils.classifier import VALID_CLASSIFIER_LEARNING_TYPES, VALID_CLASSIFIER_TYPES
from queens.utils.exceptions import InvalidOptionError
from queens.utils.experimental_data_reader import (
    VALID_TYPES as VALID_EXPERIMENTAL_DATA_READER_TYPES,
)
from queens.utils.imports import get_module_class
from queens.utils.iterative_averaging import VALID_TYPES as VALID_ITERATIVE_AVERAGING_TYPES
from queens.utils.remote_operations import VALID_CONNECTION_TYPES
from queens.variational_distributions import VALID_TYPES as VALID_VARIATIONAL_DISTRIBUTION_TYPES

_logger = logging.getLogger(__name__)


VALID_TYPES = {
    **VALID_CLASSIFIER_LEARNING_TYPES,
    **VALID_CLASSIFIER_TYPES,
    **VALID_CONNECTION_TYPES,
    **VALID_DATA_PROCESSOR_TYPES,
    **VALID_DISTRIBUTION_TYPES,
    **VALID_DRIVER_TYPES,
    **VALID_EXPERIMENTAL_DATA_READER_TYPES,
    **VALID_EXTERNAL_GEOMETRY_TYPES,
    **VALID_ITERATIVE_AVERAGING_TYPES,
    **VALID_ITERATOR_TYPES,
    **VALID_MODEL_TYPES,
    **VALID_RANDOM_FIELD_TYPES,
    **VALID_SCHEDULER_TYPES,
    **VALID_STOCHASTIC_OPTIMIZER_TYPES,
    **VALID_VARIATIONAL_DISTRIBUTION_TYPES,
}


def from_config_create_iterator(config, global_settings):
    """Create main iterator for queens run from config.

    A bottom up approach is used here to create all objects from the description. First, the objects
    that do not need any other so far uninitialized objects for initialization are initialized.
    These objects are put into the description of the other objects, where they are referenced.
    Then again the objects that do not need any other so far uninitialized objects for
    initialization are initialized. This process repeats until the main iterator (indicated by the
    'method' keyword) is initialized.

    Args:
        config (dict): Description of the queens run
        global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                          and the output directory

    Returns:
        new_obj (iterator): Main queens iterator with all initialized objects.
    """
    # do pre-processing
    random_field_preprocessor = None
    random_field_preprocessor_options = config.pop("random_field_preprocessor", None)
    if random_field_preprocessor_options:
        random_field_preprocessor = from_config_create_object(random_field_preprocessor_options)
        random_field_preprocessor.main_run()
        random_field_preprocessor.write_random_fields_to_dat()

    parameters = from_config_create_parameters(
        config.pop("parameters", {}), random_field_preprocessor
    )
    obj_key = None
    for _ in range(1000):  # Instead of 'while True' we only allow 1000 iterations for safety
        deadlock = True
        for obj_key, obj_dict in config.items():
            if isinstance(obj_dict, dict):
                reference_to_uninitialized_object = check_for_reference(obj_dict)
                if not reference_to_uninitialized_object:
                    deadlock = False
                    break
        if deadlock or obj_key is None:
            raise RuntimeError(
                "Queens run can not be configured due to missing 'method' "
                "description, circular dependencies or missing object descriptions! "
                f"Remaining uninitialized objects are: {list(config.keys())}"
            )

        try:
            obj_description = config.pop(obj_key)
            new_obj = from_config_create_object(obj_description, global_settings, parameters)
        except (TypeError, InvalidOptionError) as err:
            raise InvalidOptionError(f"Object '{obj_key}' can not be initialized.") from err

        if obj_key == "method":
            if config:
                _logger.warning("Unused settings:")
                _logger.warning(config)
            return new_obj  # returns initialized iterator

        config = insert_new_obj(config, obj_key, new_obj)

    raise RuntimeError(
        "Queens run can not be configured. If you provided less than 1000 object descriptions in "
        "the input file, this behaviour is unexpected and you should raise an issue."
    )


def from_config_create_object(obj_description, global_settings=None, parameters=None):
    """Create object from description.

    Args:
        obj_description (dict): Description of the object
        global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                          and the output directory
        parameters (obj, optional): Parameters object

    Returns:
        obj: Initialized object
    """
    object_class = get_module_class(obj_description, VALID_TYPES)
    if isinstance(object_class, types.FunctionType):
        return object_class
    if issubclass(object_class, (Iterator, Driver, BMFMC)):
        obj_description["parameters"] = parameters
    if issubclass(object_class, (Iterator, BMFMC)):
        obj_description["global_settings"] = global_settings
    if issubclass(object_class, Scheduler):
        obj_description["experiment_name"] = global_settings.experiment_name
    return object_class(**obj_description)


def check_for_reference(obj_description):
    """Check if another uninitialized object is referenced.

    Indicated by a keyword that ends with '_name'. Sub-dictionaries are also checked.

    Args:
        obj_description (dict): Description of the object

    Returns:
        bool: True, if another uninitialized object is referenced.
    """
    for key, value in obj_description.items():
        if (
            key.endswith("_name") and key != "plot_name"
        ):  # TODO: rename plot_name keyword # pylint: disable=fixme
            return True
        if isinstance(value, dict):
            reference_check = check_for_reference(value)
            if reference_check:
                return True
    return False


def insert_new_obj(config, new_obj_key, new_obj):
    """Insert initialized object in other object descriptions.

    Args:
        config (dict): Description of queens run, or sub dictionary
        new_obj_key (str): Key of initialized object
        new_obj (obj): Initialized object

    Returns:
        bool: True, if another uninitialized object is referenced.
    """
    referenced_keys = []
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = insert_new_obj(value, new_obj_key, new_obj)
        elif key.endswith("_name") and value == new_obj_key:
            referenced_keys.append(key)

    for key in referenced_keys:
        config.pop(key)  # remove key "<example>_name"
        config[key.removesuffix("_name")] = new_obj  # add key "<example>" with initialized object
    return config
