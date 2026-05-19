#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check durations for tests declared in changed test files."""

from __future__ import annotations

import ast
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

JUNIT_ROOT = Path("test-results")
WARN_SECONDS = 10.0
FAIL_SECONDS = 30.0

TestRecord = tuple[str, str, float, str]


def collect_test_classes(source: str) -> set[str]:
    tree = ast.parse(source)
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name.startswith("Test")}


def parse_junit_records(root: Path) -> list[TestRecord]:
    records: list[TestRecord] = []

    for xml_path in sorted(root.rglob("*.xml")):
        environment = xml_path.parent.name.removeprefix("test-results-")
        for testcase in ET.parse(xml_path).getroot().iter("testcase"):
            classname = testcase.get("classname")
            name = testcase.get("name")
            if not classname or not name:
                continue
            records.append((classname, name, float(testcase.get("time") or "0"), environment))

    return records


def evaluate_records(
    records: list[TestRecord],
    changed_classes: set[str],
) -> tuple[list[TestRecord], list[TestRecord]]:
    changed_records = [record for record in records if record[0] in changed_classes]
    failures = [record for record in changed_records if record[2] >= FAIL_SECONDS]
    warnings = [record for record in changed_records if WARN_SECONDS <= record[2] < FAIL_SECONDS]
    failures.sort(key=lambda record: record[2], reverse=True)
    warnings.sort(key=lambda record: record[2], reverse=True)
    return failures, warnings


def main() -> int:
    changed_classes = _collect_changed_test_classes(
        os.environ["BASE_REF"],
        os.environ.get("HEAD_REF", "HEAD"),
    )
    records = parse_junit_records(JUNIT_ROOT)

    if not records:
        raise SystemExit(f"No JUnit XML test records found under {JUNIT_ROOT}")

    failures, warnings = evaluate_records(records, changed_classes)
    markdown = _format_summary(failures, warnings)
    print(markdown)

    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(summary_file).write_text(markdown + "\n", encoding="utf-8")

    return 1 if failures else 0


def _collect_changed_test_classes(base_ref: str, head_ref: str = "HEAD") -> set[str]:
    classes: set[str] = set()
    paths = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=AMR",
            f"{base_ref}...{head_ref}",
            "--",
            "newton/tests/test*.py",
            "newton/tests/**/test*.py",
        ],
        text=True,
    ).splitlines()

    for path in (Path(path) for path in paths):
        if path.exists():
            classes.update(collect_test_classes(path.read_text(encoding="utf-8")))
    return classes


def _format_summary(
    failures: list[TestRecord],
    warnings: list[TestRecord],
) -> str:
    lines = [
        "## Changed Test Duration Check",
        "",
        f"Warn threshold: `{WARN_SECONDS:.1f}s`; fail threshold: `{FAIL_SECONDS:.1f}s`.",
    ]
    if failures:
        lines.extend(["", "### Blocking Slow Tests", "", _records_table(failures)])
    if warnings:
        lines.extend(["", "### Slow Test Warnings", "", _records_table(warnings)])
    if not failures and not warnings:
        lines.extend(["", "No changed tests exceeded the warning threshold."])

    return "\n".join(lines)


def _records_table(records: list[TestRecord]) -> str:
    lines = ["| Test | Environment | Time |", "| --- | --- | ---: |"]
    for classname, name, duration, environment in records:
        lines.append(f"| `{classname}::{name}` | `{environment}` | `{duration:.3f}s` |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
