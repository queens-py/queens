#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2025, QUEENS contributors.
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
"""Create changelog from tags."""

import sys
from pathlib import Path

import gitlab  # pylint: disable=import-error

CHANGE_KEY = "change:"


def create_section(name, link, date, message, header_symbol, newline=True):
    """Create section of changelog.

    Args:
        name (str): Name of the tag
        link (str): Link of the tag
        date (obj): Date of the tag
        message (str): Tag message
        header_symbol (str): Symbol of the section
        newline (bool): Add newline after the section name

    Returns:
        str: section for the markdown file
    """
    section = f"{header_symbol} [{name}]({link}) ({date})"
    if newline:
        section += "\n"
    section += message + "\n\n"
    return section


def generate_changelog(token, project_id, changelog_path):  # pylint: disable=redefined-outer-name
    """Generate changelog file.

    Args:
        token (str): Token to access the project
        project_id (int): Id of the QUEENS project
        changelog_path (str): Path to the changelog file
    """
    gitlab_api = gitlab.Gitlab(url="https://gitlab.lrz.de", private_token=token)
    project = gitlab_api.projects.get(id=project_id)

    tags = project.tags.list(iterator=True)

    tags_list = []
    for tag in tags:
        tags_list.append(tag.asdict())

    def data_sort(tag_as_dict):
        date = tag_as_dict["commit"]["created_at"].split("T")[0]
        y, m, d = date.split("-")
        return int(y) * 10000 + 100 * int(m) + int(d)

    # Sort by date of commit
    tags_list = sorted(tags_list, key=data_sort, reverse=True)

    # Create markdown file
    change_log = """<!---
To add changes to the changelog create a tag where the message starts with 'change: ' the rest is
done automatically by a pipeline. Releases are added automatically. Changes in this file will
be overwritten!
-->\n\n"""

    # Header
    change_log += "# Changelog\n\n"

    # Loop through tags
    for tag in tags_list:
        date = tag["commit"]["created_at"].split("T")[0]
        name = tag["name"]
        link = tag["commit"]["web_url"]
        message = tag["message"]

        if tag["release"] is not None:
            change_log += create_section(name, link, date, message, "##")
        # if the tag start with the desired key
        elif message.lower().find(CHANGE_KEY) == 0:
            message = message[len(CHANGE_KEY) :].strip()
            message = ": " + message.replace("\r\n", " ")
            change_log += create_section(name, link, date, message, "*", newline=False)
        else:
            print(
                f'\n Tag "{name}" is not added to the changelog as it does not start with the key '
                f'{CHANGE_KEY}".'
            )
    change_log_path = Path(changelog_path)
    change_log_path.write_text(change_log, encoding="utf-8")
    print(f"\nUpdated changelog: {change_log_path.resolve()}")


if __name__ == "__main__":
    args = sys.argv
    token = args[1]
    project_id = args[2]
    changelog_path = args[3]
    generate_changelog(
        token=token,
        project_id=project_id,
        changelog_path=changelog_path,
    )
