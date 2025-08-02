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
"""Extract QUEENS example from the readme."""


from pathlib import Path

from queens.utils.path import relative_path_from_root


def extract_from_markdown_file_by_marker(marker_name, md_file):
    """Extract section from an markdown file.

    Args:
        marker_name (str): Name of the section to extract.
        md_file (str, pathlib.Path): Path to the markdown file

    Returns:
        str: section as string
    """
    marker = f"<!---{marker_name} marker, do not remove this comment-->"

    # Split the example in the readme using the marker
    text = Path(md_file).read_text(encoding="utf-8").split(marker)

    # Only one example should appear
    if len(text) != 3:
        raise ValueError(f"Could not extract the section marked with '{marker}' from {md_file}!")

    return text[1]


def get_queens_example_from_readme(output_dir):
    """Extract the example from the readme.

    Args:
        output_dir (str): Output directory for the QUEENS run.
    """
    readme_path = relative_path_from_root("README.md")

    example_source = (
        extract_from_markdown_file_by_marker("example", readme_path)
        .replace("```python", "")
        .replace("```", "")
        .replace('output_dir="."', f'output_dir="{output_dir}"')
    )

    return example_source
