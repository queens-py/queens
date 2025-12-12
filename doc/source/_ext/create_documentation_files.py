#!/usr/bin/env python3
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
"""Documentation creation utils."""

import pydoc
import re
import shutil
import sys
from pathlib import Path

import requests

from queens.utils.injector import inject
from queens.utils.path import relative_path_from_root

sys.path.insert(1, str(relative_path_from_root("test_utils").resolve()))
from get_queens_example_from_readme import (  # pylint: disable=import-error, wrong-import-position,wrong-import-order
    extract_from_markdown_file_by_marker,
)


def get_template_path_by_name(template_name):
    """Get template path by name from the template folder.

    Args:
        template_name (str): Temple name in the template folder.

    Returns:
        pathlib.Path: Path to template
    """
    template_path = (Path(__file__).parent / "templates") / template_name
    return template_path


def relative_to_doc_source(relative_path):
    """Relative path from documentation source.

    Args:
        relative_path (str): path relative to `doc/source/`

    Returns:
        pathlib.Path: Path relative from documentation
    """
    return relative_path_from_root("doc/source/" + relative_path)


def remove_markdown_emojis(md_text):
    """Remove emojis from markdown text.

    Args:
        md_text (str): Markdown text

    Returns:
        str: Cleaned text
    """
    # Define a regex pattern for matching markdown-style emojis
    emoji_pattern = re.compile(r":\w+:")

    for emoji in emoji_pattern.findall(md_text):

        # Remove markdown emojis from the text
        md_text = md_text.replace(emoji, "")

        # Replace emojis in reference to other sections
        md_text = md_text.replace(emoji[1:-1], "")

    return md_text


def prepend_relative_links(md_text, base_url):
    """Prepend url for relative links.

    Args:
        md_text (str): Text to check
        base_url (str): Base URL to add

    Returns:
        str: Prepended markdown text
    """
    md_link_regex = "\\[([^]]+)]\\(\\s*(.*)\\s*\\)"
    for match in re.findall(md_link_regex, md_text):
        _, link = match

        # For local reference that started with an emoji
        if link.strip().startswith("#-"):
            new_link = "#" + link[2:].strip().lower()
            md_text = md_text.replace(f"({link})", f"({new_link})")

        # No http links or proper references within the file references
        if not link.strip().startswith("http") and not link.strip().startswith("#"):
            md_text = md_text.replace(f"({link})", f"({base_url}/{link.strip()})")

    return md_text


def clean_markdown(md_text):
    """Clean markdown.

    Removes emojis and prepends links.

    Args:
        md_text (str): Original markdown text.

    Returns:
        str: Markdown text
    """
    md_text = remove_markdown_emojis(md_text)
    md_text = prepend_relative_links(md_text, "https://www.github.com/queens-py/queens/blob/main")
    return md_text


def clean_markdown_file(md_path, new_path):
    """Load markdown and escape relative links and emojis.

    Args:
        md_path (pathlib.Path, str): Path to an existing markdown file
        new_path (pathlib.Path, str): Path for the cleaned file

    Returns:
        str: file name of the new markdown file
    """
    md_text = clean_markdown(relative_path_from_root(md_path).read_text())
    new_path = Path(new_path)
    new_path.write_text(md_text, encoding="utf-8")
    return new_path.name


def create_development():
    """Create development page."""
    development_template = get_template_path_by_name("development.rst.j2")
    development_path = relative_to_doc_source("development.rst")

    md_paths = []
    md_paths.append(
        clean_markdown_file(
            relative_path_from_root("CONTRIBUTING.md"), relative_to_doc_source("contributing.md")
        )
    )
    md_paths.append(
        clean_markdown_file(
            relative_path_from_root("tests/README.md"), relative_to_doc_source("testing.md")
        )
    )
    inject({"md_paths": md_paths}, development_template, development_path)


def create_introduction():
    """Generate landing page."""
    intro_template = get_template_path_by_name("introduction.md.j2")
    into_path = relative_to_doc_source("introduction.md")

    def extract_from_markdown_by_marker(marker_name, md_path):
        return clean_markdown(extract_from_markdown_file_by_marker(marker_name, md_path))

    inject(
        {
            "readme_path": relative_path_from_root("README.md"),
            "contributing_path": relative_path_from_root("CONTRIBUTING.md"),
            "extract_from_markdown_by_marker": extract_from_markdown_by_marker,
        },
        intro_template,
        into_path,
    )


def create_packages():
    """Create overview of the QUEENS package."""
    overview_template = get_template_path_by_name("packages.rst.j2")
    overview_path = relative_to_doc_source("packages.rst")

    def get_module_description(python_file):
        """Get module description.

        Args:
            python_file (pathlib.Path): Path to python file.

        Returns:
            str: Module description.
        """
        module_documentation = pydoc.importfile(str(python_file)).__doc__.split("\n\n")
        return "\n\n".join([m.replace("\n", " ") for m in module_documentation[1:]])

    queens_base_path = relative_path_from_root("src/queens")
    queens_modules = []
    for path in sorted(queens_base_path.iterdir()):
        if path.name.startswith("__") or not path.is_dir():
            continue

        description = get_module_description(path / "__init__.py")
        name = path.stem

        queens_modules.append(
            {
                "name": name.replace("_", " ").title(),
                "module": "queens." + name,
                "description": description,
            }
        )

    queens_interfaces_path = relative_path_from_root("src/queens_interfaces")
    queens_interfaces_modules = []
    for path in sorted(queens_interfaces_path.iterdir()):
        if path.name.startswith("__") or not path.is_dir():
            continue

        description = get_module_description(path / "__init__.py")
        name = path.stem

        queens_interfaces_modules.append(
            {
                "name": name.replace("_", " ").title(),
                "module": "queens_interfaces." + name,
                "description": description,
            }
        )

    name = queens_interfaces_path.stem
    description = get_module_description(queens_interfaces_path / "__init__.py")
    queens_interfaces = {
        "name": name.replace("_", " ").title(),
        "module": "queens_interfaces",
        "description": description,
    }

    example_simulator_functions = relative_path_from_root(
        "src/example_simulator_functions/__init__.py"
    )
    example_simulator_functions_module = []

    description = get_module_description(example_simulator_functions)
    name = example_simulator_functions.parent.stem

    example_simulator_functions_module = {
        "name": name.replace("_", " ").title(),
        "module": "example_simulator_functions",
        "description": description,
    }

    inject(
        {
            "queens_modules": queens_modules,
            "queens_interfaces_modules": queens_interfaces_modules,
            "queens_interfaces": queens_interfaces,
            "example_simulator_functions": example_simulator_functions_module,
            "len": len,
        },
        overview_template,
        overview_path,
    )


def download_images():
    """Download images."""

    def download_file_from_url(url, file_name):
        """Download file from an url."""
        url_request = requests.get(url, timeout=10)
        with open(file_name, "wb") as f:
            f.write(url_request.content)

    download_file_from_url(
        "https://raw.githubusercontent.com/queens-py/queens-design/main/logo/queens_text_night.svg",
        relative_to_doc_source("images/queens_logo_night.svg"),
    )
    download_file_from_url(
        "https://raw.githubusercontent.com/queens-py/queens-design/main/logo/queens_text.svg",
        relative_to_doc_source("images/queens_logo_day.svg"),
    )
    download_file_from_url(
        "https://raw.githubusercontent.com/queens-py/queens-design/main/crown/queens_crown.svg",
        relative_to_doc_source("images/queens_logo_no_text.svg"),
    )


def copy_tutorials():
    """Copy tutorials from source to doc."""
    for tutorial in relative_path_from_root("tutorials").glob("*.ipynb"):
        destination = relative_to_doc_source("tutorials/" + tutorial.name)

        if destination.exists():
            destination.unlink()

        shutil.copyfile(tutorial, destination)


def main():
    """Create all the rst files."""
    create_introduction()
    copy_tutorials()
    create_development()
    create_packages()
