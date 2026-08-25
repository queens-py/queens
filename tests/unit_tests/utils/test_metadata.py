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

from queens.utils.metadata import SimulationMetadata, get_metadata_path, hash_inputs

INPUTS = np.array([1.5, -2.0])


@pytest.fixture(name="inputs")
def fixture_inputs(parameters):
    """Inputs of a job as created by *Parameters.sample_as_dict*."""
    return parameters.sample_as_dict(INPUTS)


def test_hash_inputs_is_deterministic(inputs):
    """Test that hashing the same inputs twice yields the same hash."""
    assert hash_inputs(inputs) == hash_inputs(inputs)


def test_hash_inputs_is_independent_of_key_order(inputs):
    """Test that the hash does not depend on the order of the parameters."""
    reordered_inputs = dict(reversed(list(inputs.items())))

    assert list(reordered_inputs) != list(inputs)
    assert hash_inputs(reordered_inputs) == hash_inputs(inputs)


def test_hash_inputs_is_independent_of_numeric_type(inputs):
    """Test that numpy and python numbers of equal value hash equally."""
    standard_type_inputs = {key: float(value) for key, value in inputs.items()}

    assert all(type(standard_type_inputs[key]) is not type(v) for key, v in inputs.items())
    assert hash_inputs(standard_type_inputs) == hash_inputs(inputs)


def test_hash_inputs_differs_for_different_values(parameters, inputs):
    """Test that a changed parameter value changes the hash."""
    changed_inputs = parameters.sample_as_dict(INPUTS + np.array([0.0, 1.0e-12]))

    assert hash_inputs(changed_inputs) != hash_inputs(inputs)


def test_hash_inputs_differs_for_different_parameter_names(inputs):
    """Test that renaming a parameter changes the hash."""
    renamed_inputs = {"parameter_1": inputs["parameter_1"], "parameter_3": inputs["parameter_2"]}

    assert hash_inputs(renamed_inputs) != hash_inputs(inputs)


def test_hash_inputs_for_array_valued_parameters():
    """Test that array valued parameters, e.g. random fields, are hashed."""
    inputs = {"random_field": np.array([1.0, 2.0, 3.0])}
    changed_inputs = {"random_field": np.array([1.0, 2.0, 4.0])}

    assert hash_inputs(inputs) == hash_inputs({"random_field": np.array([1.0, 2.0, 3.0])})
    assert hash_inputs(inputs) != hash_inputs(changed_inputs)


def test_hash_inputs_does_not_modify_inputs():
    """Test that hashing leaves the inputs untouched.

    The conversion to standard types is done in place, so the inputs
    have to be copied before hashing them.
    """
    array = np.array([1.0, 2.0, 3.0])
    inputs = {"random_field": array}

    hash_inputs(inputs)

    assert isinstance(inputs["random_field"], np.ndarray)
    np.testing.assert_array_equal(inputs["random_field"], array)


def test_metadata_holds_hash_instead_of_inputs(tmp_path, inputs):
    """Test that the exported metadata holds the hash of the inputs."""
    metadata = SimulationMetadata(job_id=1, inputs=inputs, job_dir=tmp_path)

    metadata.export()

    exported_metadata = yaml.safe_load(get_metadata_path(tmp_path).read_text(encoding="utf-8"))
    assert exported_metadata["inputs_hash"] == hash_inputs(inputs)


def test_metadata_of_successful_section(tmp_path):
    """Test the timing of a code section that does not raise."""
    metadata = SimulationMetadata(job_id=1, inputs={"parameter_1": 1.0}, job_dir=tmp_path)

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
    metadata = SimulationMetadata(job_id=1, inputs={"parameter_1": 1.0}, job_dir=tmp_path)

    with pytest.raises(ValueError, match="dummy error"):
        with metadata.time_code("dummy_section"):
            raise ValueError("dummy error")

    assert metadata.job_successful is False

    exported_metadata = yaml.safe_load(get_metadata_path(tmp_path).read_text(encoding="utf-8"))
    assert exported_metadata["job_successful"] is False
    assert exported_metadata["times"]["dummy_section"]["status"] == "failed"


def test_metadata_init_from_file(tmp_path, inputs):
    """Test that an exported metadata file is read in correctly."""
    metadata = SimulationMetadata(job_id=1, inputs=inputs, job_dir=tmp_path)
    with metadata.time_code("dummy_section"):
        pass

    read_in_metadata = SimulationMetadata.init_from_file(tmp_path)

    assert read_in_metadata.to_dict() == metadata.to_dict()
