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
"""Utils to collect data during iterative processes."""

from __future__ import annotations

from typing import Iterable

from queens.utils.printing import get_str_table


class CollectionObject:
    """Collection object which stores data.

    This object can be indexed by iteration i: *collection_object[i]* but also using the collected
    fields *collection_object.field1*.
    """

    def __init__(self, *field_names: str) -> None:
        """Initialize the collection item.

        Args:
            field_names: Names of fields to be stored
        """
        for key in field_names:
            self.__dict__.update({key: []})

    @classmethod
    def create_collection_object_from_dict(cls, data_dict: dict) -> CollectionObject:
        """Create collection item from dict.

        Args:
            data_dict: Dictionary with values to be stored in this object

        Returns:
            Collection object created from dict
        """
        collection_object = cls()
        collection_object.__dict__.update(data_dict)
        return collection_object

    def add(self, **field_names_and_values: dict) -> None:
        """Add data to the object.

        This function can be called with one or multiple fields, i.e.:
        *collection_object.add(field1=value1)* or
        *collection_object.add(field1=value1, field2=value2)*. An error
        is raised if one tries to add data to a field for a new
        iteration before all fields are filled for the current
        iteration.
        """
        # Select only the fields to be stored the others are ignored
        field_names_to_be_stored = {
            key: field_names_and_values[key] for key in self.keys() & field_names_and_values.keys()
        }
        for key, value in field_names_to_be_stored.items():
            # Check if current iteration is completed
            if len(self.__dict__[key]) > min(self._get_lens()):
                fields_with_lens = ", ".join(
                    [f"{key}: {length}" for key, length in zip(self.keys(), self._get_lens())]
                )
                raise ValueError(
                    f"Can not add value to {key} list as it has length {len(self.__dict__[key])} "
                    f"but the other entries are at {len(self)}: {fields_with_lens}"
                )

            self.__dict__[key].append(value)

    def _get_lens(self) -> list[int]:
        """Get number of elements per field.

        Returns:
            List of number of elements per field
        """
        return [len(value) for value in self.values()]

    def __str__(self) -> str:
        """Print table of current collection.

        Returns:
            Print table
        """
        return get_str_table(f"Collection object with {self.__len__()} iterations", self.__dict__)

    def values(self) -> Iterable:
        """Values of the current object.

        This allows to use the object like a dict.

        Returns:
            Values of the collection object
        """
        return self.__dict__.values()

    def items(self) -> Iterable:
        """Items of the current object.

        This allows to use the object like a dict.

        Returns:
            Items of the collection object
        """
        return self.__dict__.items()

    def keys(self) -> Iterable:
        """Keys, i.e. field names of the current object.

        This allows to use the object like a dict.

        Returns:
            Keys of the collection object
        """
        return self.__dict__.keys()

    def __len__(self) -> int:
        """Len function for the object.

        Returns:
            number of complete iterations
        """
        lens = self._get_lens()

        if lens:
            return min(self._get_lens())

        # No data
        return 0

    def __getitem__(self, i: int | slice) -> CollectionObject:
        """Python intern method to index the collection object.

        Args:
            i: int or slice

        Returns:
            collection object with values and field names for provided indexes
        """
        if isinstance(i, int):
            if i > len(self):
                raise IndexError(f"index {i} out of range for size {self.__len__()}")

        new_dict = {}
        for key, value in self.items():
            new_dict.update({key: value[i]})
        return self.create_collection_object_from_dict(new_dict)

    def to_dict(self) -> dict:
        """Create a dictionary from the collection object.

        Returns:
            Dictionary with all data
        """
        return self.__dict__

    def __bool__(self) -> bool:
        """Bool value of the object.

        Returns:
            Returns True if data is stored in the object
        """
        lens = self._get_lens()
        return any(lens)
