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
"""Test module for the metadata utils."""

import numpy as np
import pytest
import yaml

from queens.utils.metadata import SimulationMetadata, get_metadata_path, hash_input

INPUT = np.array([1.5, -2.0])


@pytest.fixture(name="job_input")
def fixture_input(parameters):
    """Input of a job as created by *Parameters.sample_as_dict*."""
    return parameters.sample_as_dict(INPUT)


def test_hash_input_is_deterministic(job_input):
    """Test that hashing the same input twice yields the same hash."""
    assert hash_input(job_input) == hash_input(job_input)


def test_hash_input_is_independent_of_key_order(job_input):
    """Test that the hash does not depend on the order of the parameters."""
    reordered_input = dict(reversed(list(job_input.items())))

    assert list(reordered_input) != list(job_input)
    assert hash_input(reordered_input) == hash_input(job_input)


def test_hash_input_is_independent_of_numeric_type(job_input):
    """Test that numpy and python numbers of equal value hash equally."""
    standard_type_input = {key: float(value) for key, value in job_input.items()}

    assert all(type(standard_type_input[key]) is not type(v) for key, v in job_input.items())
    assert hash_input(standard_type_input) == hash_input(job_input)


def test_hash_input_differs_for_different_values(parameters, job_input):
    """Test that a changed parameter value changes the hash."""
    changed_input = parameters.sample_as_dict(INPUT + np.array([0.0, 1.0e-12]))

    assert hash_input(changed_input) != hash_input(job_input)


def test_hash_input_differs_for_different_parameter_names(job_input):
    """Test that renaming a parameter changes the hash."""
    renamed_input = {
        "parameter_1": job_input["parameter_1"],
        "parameter_3": job_input["parameter_2"],
    }

    assert hash_input(renamed_input) != hash_input(job_input)


def test_hash_input_for_array_valued_parameters():
    """Test that array valued parameters, e.g. random fields, are hashed."""
    job_input = {"random_field": np.array([1.0, 2.0, 3.0])}
    changed_input = {"random_field": np.array([1.0, 2.0, 4.0])}

    assert hash_input(job_input) == hash_input({"random_field": np.array([1.0, 2.0, 3.0])})
    assert hash_input(job_input) != hash_input(changed_input)


def test_hash_input_does_not_modify_input():
    """Test that hashing leaves the input untouched.

    The conversion to standard types is done in place, so the input has
    to be copied before hashing it.
    """
    array = np.array([1.0, 2.0, 3.0])
    job_input = {"random_field": array}

    hash_input(job_input)

    assert isinstance(job_input["random_field"], np.ndarray)
    np.testing.assert_array_equal(job_input["random_field"], array)


def test_metadata_holds_input_hash(tmp_path, job_input):
    """Test that the exported metadata holds the hash of the input."""
    metadata = SimulationMetadata(job_id=1, job_input=job_input, job_dir=tmp_path)

    metadata.export()

    exported_metadata = yaml.safe_load(get_metadata_path(tmp_path).read_text(encoding="utf-8"))
    assert exported_metadata["input_hash"] == hash_input(job_input)


def test_metadata_of_successful_section(tmp_path):
    """Test the timing of a code section that does not raise."""
    metadata = SimulationMetadata(job_id=1, job_input={"parameter_1": 1.0}, job_dir=tmp_path)

    with metadata.time_code("dummy_section"):
        pass

    exported_metadata = yaml.safe_load(get_metadata_path(tmp_path).read_text(encoding="utf-8"))
    assert exported_metadata["job_successful"] is True
    dummy_section = exported_metadata["times"]["dummy_section"]
    assert dummy_section["status"] == "successful"
    assert dummy_section["time"] >= 0
    assert dummy_section["timestamp_start"]


def test_metadata_of_failed_section(tmp_path):
    """Test that a failing code section marks the job as unsuccessful."""
    metadata = SimulationMetadata(job_id=1, job_input={"parameter_1": 1.0}, job_dir=tmp_path)

    with pytest.raises(ValueError, match="dummy error"):
        with metadata.time_code("dummy_section"):
            raise ValueError("dummy error")

    assert metadata.job_successful is False

    exported_metadata = yaml.safe_load(get_metadata_path(tmp_path).read_text(encoding="utf-8"))
    assert exported_metadata["job_successful"] is False
    assert exported_metadata["times"]["dummy_section"]["status"] == "failed"


def test_metadata_init_from_file(tmp_path, job_input):
    """Test that an exported metadata file is read in correctly."""
    metadata = SimulationMetadata(job_id=1, job_input=job_input, job_dir=tmp_path)
    with metadata.time_code("dummy_section"):
        pass

    read_in_metadata = SimulationMetadata.init_from_file(tmp_path)

    assert read_in_metadata.to_dict() == metadata.to_dict()
