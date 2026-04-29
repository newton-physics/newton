#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check PR test durations from existing JUnit XML artifacts."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple


class LineRange(NamedTuple):
    start: int
    end: int

    def intersects(self, other: LineRange) -> bool:
        return self.start <= other.end and other.start <= self.end


class TestRecord(NamedTuple):
    classname: str
    name: str
    duration: float
    environment: str

    @property
    def computed_name(self) -> str:
        return f"{self.classname}::{self.name}"


class TestPattern(NamedTuple):
    classname: str
    name: str | None = None
    name_prefix: str | None = None

    def matches(self, classname: str, name: str) -> bool:
        if self.classname != classname:
            return False
        if self.name is not None:
            return self.name == name
        if self.name_prefix is not None:
            return name == self.name_prefix or name.startswith(f"{self.name_prefix}_")
        return True


class AffectedTests:
    def __init__(self):
        self._patterns: set[TestPattern] = set()

    def add_class(self, classname: str) -> None:
        self._patterns.add(TestPattern(classname=classname))

    def add_method(self, classname: str, name: str) -> None:
        self._patterns.add(TestPattern(classname=classname, name=name))

    def add_prefix(self, classname: str, name_prefix: str) -> None:
        self._patterns.add(TestPattern(classname=classname, name_prefix=name_prefix))

    def matches(self, classname: str, name: str) -> bool:
        return any(pattern.matches(classname, name) for pattern in self._patterns)

    def __bool__(self) -> bool:
        return bool(self._patterns)

    def __len__(self) -> int:
        return len(self._patterns)


class TestReport(NamedTuple):
    failures: list[TestRecord]
    warnings: list[TestRecord]
    checked_count: int
    ignored_count: int


class MethodInfo(NamedTuple):
    name: str
    line_range: LineRange


class ClassInfo(NamedTuple):
    name: str
    line_range: LineRange
    methods: list[MethodInfo]


class DynamicRegistration(NamedTuple):
    classname: str
    test_name: str
    call_range: LineRange
    function_range: LineRange | None


_DIFF_HEADER_RE = re.compile(r"^\+\+\+ b/(.+)$")
_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def parse_changed_ranges(diff_text: str) -> dict[str, list[LineRange]]:
    changed_ranges: dict[str, list[LineRange]] = {}
    current_path: str | None = None

    for line in diff_text.splitlines():
        header_match = _DIFF_HEADER_RE.match(line)
        if header_match:
            current_path = header_match.group(1)
            continue

        hunk_match = _DIFF_HUNK_RE.match(line)
        if not hunk_match or current_path is None:
            continue

        start = int(hunk_match.group(1))
        count = int(hunk_match.group(2) or "1")
        end = start + max(count, 1) - 1
        changed_ranges.setdefault(current_path, []).append(LineRange(start, end))

    return changed_ranges


def collect_affected_tests(
    sources: dict[str, str],
    changed_ranges: dict[str, list[LineRange]],
) -> AffectedTests:
    affected = AffectedTests()

    for path, ranges in changed_ranges.items():
        source = sources.get(path)
        if source is None:
            continue

        classes, registrations = _analyze_source(source)
        matched_any = False

        for changed_range in ranges:
            for registration in registrations:
                if registration.call_range.intersects(changed_range) or (
                    registration.function_range is not None and registration.function_range.intersects(changed_range)
                ):
                    affected.add_prefix(registration.classname, registration.test_name)
                    matched_any = True

            for class_info in classes:
                if not class_info.line_range.intersects(changed_range):
                    continue

                changed_methods = [
                    method for method in class_info.methods if method.line_range.intersects(changed_range)
                ]
                if changed_methods:
                    for method in changed_methods:
                        affected.add_method(class_info.name, method.name)
                else:
                    affected.add_class(class_info.name)
                matched_any = True

        if not matched_any:
            for class_info in classes:
                affected.add_class(class_info.name)

    return affected


def parse_junit_records(root: Path) -> list[TestRecord]:
    records: list[TestRecord] = []

    for xml_path in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            raise SystemExit(f"Failed to parse JUnit XML {xml_path}: {exc}") from exc

        environment = _environment_from_path(xml_path)
        for testcase in tree.getroot().iter("testcase"):
            classname = testcase.get("classname")
            name = testcase.get("name")
            if not classname or not name:
                continue
            try:
                duration = float(testcase.get("time") or "0")
            except ValueError:
                duration = 0.0
            records.append(TestRecord(classname, name, duration, environment))

    return records


def evaluate_records(
    records: list[TestRecord],
    affected_tests: AffectedTests,
    warn_seconds: float,
    fail_seconds: float,
) -> TestReport:
    failures: list[TestRecord] = []
    warnings: list[TestRecord] = []
    ignored_count = 0

    for record in records:
        if not affected_tests.matches(record.classname, record.name):
            ignored_count += 1
            continue
        if record.duration >= fail_seconds:
            failures.append(record)
        elif record.duration >= warn_seconds:
            warnings.append(record)

    failures.sort(key=lambda record: record.duration, reverse=True)
    warnings.sort(key=lambda record: record.duration, reverse=True)

    return TestReport(
        failures=failures,
        warnings=warnings,
        checked_count=len(records) - ignored_count,
        ignored_count=ignored_count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--junit-root", type=Path, required=True, help="Directory containing downloaded JUnit XML artifacts."
    )
    parser.add_argument("--base-ref", required=True, help="Base commit/ref for the PR diff.")
    parser.add_argument("--head-ref", default="HEAD", help="Head commit/ref for the PR diff.")
    parser.add_argument(
        "--warn-seconds", type=float, default=10.0, help="Warn for affected tests at or above this time."
    )
    parser.add_argument(
        "--fail-seconds", type=float, default=30.0, help="Fail for affected tests at or above this time."
    )
    parser.add_argument("--summary-file", type=Path, help="Markdown summary output path.")
    args = parser.parse_args(argv)

    diff_text = _run_git_diff(args.base_ref, args.head_ref)
    changed_ranges = {path: ranges for path, ranges in parse_changed_ranges(diff_text).items() if _is_test_path(path)}
    sources = {path: Path(path).read_text(encoding="utf-8") for path in changed_ranges if Path(path).exists()}
    affected_tests = collect_affected_tests(sources, changed_ranges)
    records = parse_junit_records(args.junit_root)

    if not records:
        print(f"No JUnit XML test records found under {args.junit_root}", file=sys.stderr)
        return 1

    report = evaluate_records(records, affected_tests, args.warn_seconds, args.fail_seconds)
    markdown = _format_summary(report, args.warn_seconds, args.fail_seconds)
    print(markdown)

    if args.summary_file is not None:
        with args.summary_file.open("a", encoding="utf-8") as summary:
            summary.write(markdown)
            summary.write("\n")

    if report.failures:
        return 1
    return 0


def _analyze_source(source: str) -> tuple[list[ClassInfo], list[DynamicRegistration]]:
    tree = ast.parse(source)
    functions = {
        node.name: _node_range(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    classes: list[ClassInfo] = []
    registrations: list[DynamicRegistration] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                MethodInfo(name=child.name, line_range=_node_range(child))
                for child in node.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef) and child.name.startswith("test")
            ]
            classes.append(ClassInfo(name=node.name, line_range=_node_range(node), methods=methods))
        elif isinstance(node, ast.Call):
            registration = _dynamic_registration(node, functions)
            if registration is not None:
                registrations.append(registration)

    return classes, registrations


def _dynamic_registration(
    node: ast.Call,
    functions: dict[str, LineRange],
) -> DynamicRegistration | None:
    func_name = _name(node.func)
    if func_name not in {"add_function_test", "add_function_test_register_kernel", "add_kernel_test"}:
        return None
    if len(node.args) < 2:
        return None

    classname = _name(node.args[0])
    test_name = _constant_string(node.args[1])
    if classname is None or test_name is None:
        return None

    function_name = _name(node.args[2]) if len(node.args) >= 3 else None
    function_range = functions.get(function_name) if function_name is not None else None
    return DynamicRegistration(classname, test_name, _node_range(node), function_range)


def _node_range(node: ast.AST) -> LineRange:
    return LineRange(node.lineno, getattr(node, "end_lineno", node.lineno))


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _environment_from_path(xml_path: Path) -> str:
    for part in reversed(xml_path.parts):
        if part.startswith("test-results-"):
            return part.removeprefix("test-results-")
    return xml_path.parent.name or "unknown"


def _run_git_diff(base_ref: str, head_ref: str) -> str:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--unified=0",
            "--no-ext-diff",
            "--diff-filter=AM",
            f"{base_ref}...{head_ref}",
            "--",
            "newton/**/test*.py",
            "newton/tests/test*.py",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def _is_test_path(path: str) -> bool:
    path_obj = Path(path)
    return path_obj.name.startswith("test") and path_obj.suffix == ".py" and "newton" in path_obj.parts


def _format_summary(
    report: TestReport,
    warn_seconds: float,
    fail_seconds: float,
) -> str:
    lines = [
        "## Changed Test Duration Check",
        "",
        f"Checked `{report.checked_count}` affected test timing record(s); ignored `{report.ignored_count}` unchanged record(s).",
        f"Warn threshold: `{warn_seconds:.1f}s`; fail threshold: `{fail_seconds:.1f}s`.",
    ]
    if report.failures:
        lines.extend(["", "### Blocking Slow Tests", "", _records_table(report.failures)])
    if report.warnings:
        lines.extend(["", "### Slow Test Warnings", "", _records_table(report.warnings)])
    if not report.failures and not report.warnings:
        lines.extend(["", "No affected tests exceeded the warning threshold."])

    return "\n".join(lines)


def _records_table(records: list[TestRecord], limit: int = 20) -> str:
    lines = ["| Test | Environment | Time |", "| --- | --- | ---: |"]
    for record in records[:limit]:
        lines.append(f"| `{record.computed_name}` | `{record.environment}` | `{record.duration:.3f}s` |")
    if len(records) > limit:
        lines.append(f"| ...and {len(records) - limit} more | | |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
