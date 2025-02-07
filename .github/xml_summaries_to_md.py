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
"""Generate a md test summary from the xml test reports."""

import re
import sys
import xml.etree.ElementTree as ET
from copy import copy
from pathlib import Path


def xml_to_dict(xml_element, root=True):
    """Xml to dict converter.

    Args:
        xml_element (ET): ElementTree object.
        root (bool, optional): Is root object. Defaults to True.

    Returns:
        dict: data as dictionary
    """
    if root:
        return {xml_element.tag: xml_to_dict(xml_element, False)}
    dictionary = copy(xml_element.attrib)
    if xml_element.text:
        dictionary["_text"] = xml_element.text
    for x in xml_element.findall("./*"):
        if x.tag not in dictionary:
            dictionary[x.tag] = []
        dictionary[x.tag].append(xml_to_dict(x, False))
    return dictionary


def create_md_table(data_list, header):
    """Generate simple markdown table.

    Args:
        data_list (list): List of rows.
        header (list): Headers of the table.

    Returns:
        str: table string
    """

    def add_seperators(row):
        """Add markdown table seperators.

        Args:
            row (list): List of row data.

        Returns:
            str: single row for markdown table
        """
        return "|" + "|".join([str(s) for s in row]) + "|"

    table = [add_seperators(header)]
    table.append(add_seperators(["--"] * len(header)))
    table.extend([add_seperators(k) for k in data_list])
    return "\n".join(table)


def collapsable(full_text, summary):
    """Create collapsable section.

    Args:
        full_text (str): Full text.
        summary (str): Summary text.

    Returns:
        str: collapse section
    """
    return f"<details>\n<summary>{summary}</summary>\n\n{full_text}\n\n</details>" ""


def generate_md_summary(path_to_junit_xml, path_to_coverage_xml):
    """Generate markdown summary.

    Args:
        path_to_junit_xml (str): Path to junit xml file.
        path_to_coverage_xml (str): Path to coverage xml file.

    Returns:
        str: pytest summary.
    """
    root = ET.fromstring(Path(path_to_junit_xml).read_text(encoding="utf-8"))
    junit_report = xml_to_dict(root)

    root = ET.fromstring(Path(path_to_coverage_xml).read_text(encoding="utf-8"))
    coverage_summary = xml_to_dict(root)

    testing_time = junit_report["testsuites"]["testsuite"][0]["time"]
    total_number_of_tests = junit_report["testsuites"]["testsuite"][0]["tests"]
    unit_tests = []
    integration_tests = []
    failed = []
    for k in junit_report["testsuites"]["testsuite"][0]["testcase"]:
        k.pop("system-out")
        k.pop("system-err")
        k["time"] = float(k["time"])
        k["name"] = re.sub(r"\[(.*?)\]", r"[**\1**]", k["name"])

        if k.pop("failure", False):
            failed.append(list(k.values()))
        if k["classname"].startswith("tests.unit_tests"):
            k["classname"] = k["classname"][len("tests.unit_tests.") :]
            unit_tests.append(list(k.values()))
        elif k["classname"].startswith("tests.integration_tests"):
            k["classname"] = k["classname"][len("tests.integration_tests.") :]
            integration_tests.append(list(k.values()))
        else:
            raise ValueError(f"Unknown test type {k['classname']}.")

    integration_tests = sorted(integration_tests, key=lambda x: -x[-1])
    unit_tests = sorted(unit_tests, key=lambda x: -x[-1])

    text = "# :blue_heart: Pytest summary\n"

    text += f"\n### {total_number_of_tests} tests took {int(float(testing_time))}s.\n"

    text += "\n## :umbrella: Coverage \n\n"
    text += f'\n - by line rate **{int(float(coverage_summary["coverage"]["line-rate"]) * 100)}%**'
    text += (
        f'\n - by branch rate **{int(float(coverage_summary["coverage"]["branch-rate"]) * 100)}%**'
    )

    text += "\n\n## :bullettrain_side: Integration tests\n\n"
    text += collapsable(
        "\n > only showing the top 50 slowest tests\n\n"
        + create_md_table(integration_tests[:50], header=["Test Path", "Name", "Time (s)"]),
        f"{len(integration_tests)} integration tests took "
        f"{int(sum((k[-1] for k in integration_tests)))}s.",
    )

    text += "\n\n## :speedboat: Unit tests\n\n"
    text += collapsable(
        "\n > only showing the top 50 slowest tests\n\n"
        + create_md_table(unit_tests[:50], header=["Test Path", "Name", "Time (s)"]),
        f"{len(unit_tests)} unit tests took {int(sum((k[-1] for k in unit_tests)))}s.",
    )
    text += "\n\n"

    if failed:
        text += "\n\n## :worried: Failed tests\n\n"
        text += f"\n{len(failed)} test(s) failed.\n\n"
        text += create_md_table(failed, header=["Test Path", "Name", "Time (s)"])

    return text


if __name__ == "__main__":
    try:
        print(generate_md_summary(sys.argv[1], sys.argv[2]))
    except:  # pylint: disable=bare-except
        print("Could not generate the summary :sob:")
