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
"""Module to create license header check rc file.

Module to create render jinja2 template for the licenserc.yaml file
based on the license_header.tmpl file.
"""
import argparse
from pathlib import Path

from jinja2 import Template


def render_template(template_file, text_file, output_file, placeholder):
    """Render a Jinja2 template inserting text from a file.

    Args:
        template_file (Path or str): Path to the Jinja2 template file.
        text_file (Path or str): Path to the text file containing the content to insert.
        output_file (Path or str): Path where the rendered output file will be saved.
        placeholder (str): The name of the placeholder variable in the template
            where the text content will be inserted.

    Returns:
        None
    """
    # Convert to Path objects if they are not already
    template_file = Path(template_file)
    text_file = Path(text_file)
    output_file = Path(output_file)

    # Check if input files exist
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")
    if not text_file.exists():
        raise FileNotFoundError(f"Text file not found: {text_file}")

    # Read the template file
    template_content = template_file.read_text(encoding="utf-8")

    # Read the text file
    text_content = text_file.read_text(encoding="utf-8")  # Create a Jinja2 template
    template = Template(template_content)

    # Render the template with the text content inserted at the placeholder
    rendered_content = template.render({placeholder: text_content})

    # Write the rendered content to the output file
    output_file.write_text(rendered_content, encoding="utf-8")

    print(f"Rendered template written to {output_file}")


def main():
    """Command-line interface for rendering a Jinja2 template.

    Usage:
        python create_licenserc.py --template_file TEMPLATE_FILE --text_file TEXT_FILE
                              --output_file OUTPUT_FILE --placeholder PLACEHOLDER
    """
    parser = argparse.ArgumentParser(
        description="Render a Jinja2 template with content from a text file."
    )
    parser.add_argument(
        "--template_file", type=Path, required=True, help="Path to the Jinja2 template file."
    )
    parser.add_argument(
        "--text_file", type=Path, required=True, help="Path to the text file to insert."
    )
    parser.add_argument(
        "--output_file", type=Path, required=True, help="Path to save the rendered output file."
    )
    parser.add_argument(
        "--placeholder", type=str, required=True, help="Placeholder in the template to replace."
    )

    args = parser.parse_args()

    render_template(args.template_file, args.text_file, args.output_file, args.placeholder)


# Example usage
if __name__ == "__main__":
    main()
