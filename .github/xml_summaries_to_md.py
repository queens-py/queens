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
"""Generate a markdown test summary from XML test reports."""

import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

MAX_SLOW_TESTS = 50

JUNIT_REPORT_PATTERN = re.compile(
    r"test_junit_(?P<group>tutorials_fourc|tutorials|fourc|core)_(?P<runner>[^.]+)\.xml$"
)
GROUP_LABELS = {
    "core": "Core tests",
    "tutorials": "Tutorial tests",
    "fourc": "4C integration tests",
    "tutorials_fourc": "4C tutorial tests",
}
RUNNER_LABELS = {
    "Linux": "Ubuntu",
    "macOS": "macOS",
}


@dataclass
class TestCase:
    """Single JUnit testcase entry."""

    group: str
    classname: str
    name: str
    status: str
    time: float


@dataclass
class TestReport:
    """Aggregated JUnit report for one workflow test group."""

    group: str
    tests: int
    passed: int
    skipped: int
    failures: int
    errors: int
    time: float
    cases: list[TestCase]


def _xml_root(path):
    """Read an XML file and return its root element."""
    return ET.fromstring(Path(path).read_text(encoding="utf-8"))


def _split_xml_paths(paths):
    """Split input XML paths into JUnit reports and one coverage report.

    Args:
        paths (list): Input XML paths.

    Returns:
        tuple: JUnit paths and coverage path.
    """
    junit_paths = []
    coverage_paths = []
    for path in paths:
        root = _xml_root(path)
        if root.tag == "coverage":
            coverage_paths.append(path)
        elif root.tag in {"testsuite", "testsuites"}:
            junit_paths.append(path)
        else:
            raise ValueError(f"Unknown XML report type {root.tag!r} in {path}.")

    if not coverage_paths:
        raise ValueError("No coverage XML report was provided.")
    if len(coverage_paths) > 1:
        raise ValueError("Expected exactly one coverage XML report.")
    if not junit_paths:
        raise ValueError("No JUnit XML reports were provided.")

    return junit_paths, coverage_paths[0]


def _iter_testsuites(root):
    """Yield all testsuite elements from a JUnit XML root."""
    if root.tag == "testsuite":
        yield root
        return

    yield from root.findall(".//testsuite")


def _workflow_group_from_path(path):
    """Infer workflow test group label from a JUnit XML file name."""
    match = JUNIT_REPORT_PATTERN.fullmatch(Path(path).name)
    if match is None:
        return Path(path).stem.replace("_", " ")

    group = GROUP_LABELS[match.group("group")]
    runner = RUNNER_LABELS.get(match.group("runner"), match.group("runner"))
    return f"{group} ({runner})"


def _testcase_status(testcase):
    """Return the status of a JUnit testcase element."""
    if testcase.find("error") is not None:
        return "error"
    if testcase.find("failure") is not None:
        return "failed"
    if testcase.find("skipped") is not None:
        return "skipped"
    return "passed"


def _shorten_classname(classname):
    """Shorten pytest class names for display."""
    prefixes = (
        "tests.unit_tests.",
        "tests.integration_tests.",
        "tests.tutorial_tests.",
        "tests.benchmarks.",
    )
    for prefix in prefixes:
        if classname.startswith(prefix):
            return classname[len(prefix) :]
    return classname


def _format_test_name(name):
    """Highlight pytest parametrization values in test names."""
    return re.sub(r"\[(.*?)\]", r"[**\1**]", name)


def _parse_junit_report(path):
    """Parse one JUnit XML report into a TestReport."""
    root = _xml_root(path)
    group = _workflow_group_from_path(path)
    suites = list(_iter_testsuites(root))
    cases = []

    for suite in suites:
        for testcase in suite.findall("testcase"):
            cases.append(
                TestCase(
                    group=group,
                    classname=_shorten_classname(testcase.get("classname", "")),
                    name=_format_test_name(testcase.get("name", "")),
                    status=_testcase_status(testcase),
                    time=float(testcase.get("time", 0.0)),
                )
            )

    passed = sum(testcase.status == "passed" for testcase in cases)
    skipped = sum(testcase.status == "skipped" for testcase in cases)
    failures = sum(testcase.status == "failed" for testcase in cases)
    errors = sum(testcase.status == "error" for testcase in cases)
    runtime = sum(float(suite.get("time", 0.0)) for suite in suites)

    return TestReport(
        group=group,
        tests=len(cases),
        passed=passed,
        skipped=skipped,
        failures=failures,
        errors=errors,
        time=runtime,
        cases=cases,
    )


def create_md_table(data_list, header):
    """Generate simple markdown table.

    Args:
        data_list (list): List of rows.
        header (list): Headers of the table.

    Returns:
        str: table string
    """

    def markdown_cell(value):
        """Format a value as markdown table cell."""
        return str(value).replace("|", r"\|").replace("\n", "<br>")

    def add_separators(row):
        """Add markdown table separators.

        Args:
            row (list): List of row data.

        Returns:
            str: single row for markdown table
        """
        return "|" + "|".join([markdown_cell(s) for s in row]) + "|"

    if not data_list:
        return "_No entries._"

    table = [add_separators(header)]
    table.append(add_separators(["--"] * len(header)))
    table.extend([add_separators(k) for k in data_list])
    return "\n".join(table)


def collapsible(full_text, summary):
    """Create collapsible section.

    Args:
        full_text (str): Full text.
        summary (str): Summary text.

    Returns:
        str: collapse section
    """
    return f"<details>\n<summary>{summary}</summary>\n\n{full_text}\n\n</details>"


def collapsable(full_text, summary):
    """Create collapsible section.

    Keep the old misspelled function name for compatibility.
    """
    return collapsible(full_text, summary)


def _format_time(seconds):
    """Format a time value for markdown output."""
    return f"{seconds:.3f}"


def _format_test_count(count):
    """Format a test count with the correct singular or plural label."""
    return f"{count} test" if count == 1 else f"{count} tests"


def _report_summary(report):
    """Create a compact text summary for one grouped report."""
    failed = report.failures + report.errors
    status_summary = f"{report.passed} passed"
    if report.skipped:
        status_summary += f", {report.skipped} skipped"
    if failed:
        status_summary += f", {failed} failed or errored"
    return f"{_format_test_count(report.tests)} took {int(report.time)}s ({status_summary})."


def _overview_rows(reports):
    """Create the workflow group overview table rows."""
    return [
        [
            report.group,
            report.tests,
            report.passed,
            report.skipped,
            report.failures,
            report.errors,
            _format_time(report.time),
        ]
        for report in reports
    ]


def _case_rows(cases):
    """Create markdown table rows for testcase entries."""
    return [
        [
            testcase.group,
            testcase.classname,
            testcase.name,
            testcase.status,
            _format_time(testcase.time),
        ]
        for testcase in cases
    ]


def generate_md_summary(*paths):
    """Generate markdown summary.

    Args:
        paths (str): Paths to JUnit XML files and one coverage XML file. The
            order is flexible; report types are detected from the XML root tag.

    Returns:
        str: pytest summary.
    """
    flattened_paths = []
    for path in paths:
        if isinstance(path, (list, tuple)):
            flattened_paths.extend(path)
        else:
            flattened_paths.append(path)

    junit_paths, coverage_path = _split_xml_paths(flattened_paths)
    reports = [_parse_junit_report(path) for path in junit_paths]
    coverage_root = _xml_root(coverage_path)

    total_number_of_tests = sum(report.tests for report in reports)
    testing_time = sum(report.time for report in reports)
    failed_cases = [
        testcase
        for report in reports
        for testcase in report.cases
        if testcase.status in {"failed", "error"}
    ]

    text = "# Pytest summary\n"

    text += (
        f"\n### {_format_test_count(total_number_of_tests)} took {int(testing_time)}s "
        f"across {len(reports)} workflow report(s).\n"
    )

    text += "\n## Coverage\n\n"
    text += f'\n - by line rate **{int(float(coverage_root.get("line-rate", 0.0)) * 100)}%**'
    text += f'\n - by branch rate **{int(float(coverage_root.get("branch-rate", 0.0)) * 100)}%**'

    text += "\n\n## Workflow Groups\n\n"
    text += create_md_table(
        _overview_rows(reports),
        header=["Group", "Tests", "Passed", "Skipped", "Failures", "Errors", "Time (s)"],
    )

    for report in reports:
        slowest_cases = sorted(report.cases, key=lambda testcase: -testcase.time)[:MAX_SLOW_TESTS]
        text += f"\n\n## {report.group}\n\n"
        text += collapsible(
            f"\n > only showing the top {MAX_SLOW_TESTS} slowest tests\n\n"
            + create_md_table(
                _case_rows(slowest_cases),
                header=["Group", "Test Path", "Name", "Status", "Time (s)"],
            ),
            _report_summary(report),
        )

    if failed_cases:
        text += "\n\n## Failed Tests\n\n"
        text += f"\n{_format_test_count(len(failed_cases))} failed or errored.\n\n"
        text += create_md_table(
            _case_rows(failed_cases),
            header=["Group", "Test Path", "Name", "Status", "Time (s)"],
        )

    return text


if __name__ == "__main__":
    try:
        print(generate_md_summary(*sys.argv[1:]))
    except Exception as error:  # pylint: disable=broad-except
        print(f"Could not generate the summary: {error}")
