# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "detect_api_changes.py"

spec = importlib.util.spec_from_file_location("detect_api_changes", SCRIPT)
assert spec is not None and spec.loader is not None
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)


def _write_package(root: Path, files: dict[str, str]) -> None:
    for path, source in files.items():
        file_path = root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(source, encoding="utf-8")


class TestDetectApiChanges(unittest.TestCase):
    def test_compare_reports_nested_public_types_and_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            head = root / "head"

            shared_init = "from ._src.model import Actuator, Model\n"
            _write_package(
                base,
                {
                    "newton/__init__.py": shared_init,
                    "newton/_src/model.py": """
from dataclasses import dataclass
from enum import Enum


class Model:
    class AttributeFrequency(Enum):
        ONCE = "once"
        BODY = "body"


class Actuator:
    @dataclass
    class State:
        position: float
""",
                },
            )
            _write_package(
                head,
                {
                    "newton/__init__.py": shared_init,
                    "newton/_src/model.py": """
from dataclasses import dataclass
from enum import Enum


class Model:
    class AttributeFrequency(Enum):
        ONCE = "once"
        SHAPE = "shape"


class Actuator:
    @dataclass
    class State:
        position: float
        velocity: float
""",
                },
            )

            diff = detector.compare_symbols(
                detector.extract_api_symbols(base),
                detector.extract_api_symbols(head),
            )

            added_paths = {item["path"] for item in diff["added"]}
            removed_paths = {item["path"] for item in diff["removed"]}
            changed_paths = {item["path"] for item in diff["changed"]}
            self.assertIn("newton.Model.AttributeFrequency.SHAPE", added_paths)
            self.assertIn("newton.Model.AttributeFrequency.BODY", removed_paths)
            self.assertIn("newton.Actuator.State.__init__", changed_paths)
            self.assertIn("newton.Actuator.State.velocity", added_paths)

    def test_compare_reports_dataclass_constructor_order_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            head = root / "head"

            shared_init = "from ._src.actuators import ActuatorParsed\n"
            _write_package(
                base,
                {
                    "newton/__init__.py": shared_init,
                    "newton/_src/actuators.py": """
from dataclasses import dataclass


@dataclass
class ActuatorParsed:
    name: str
    path: str
""",
                },
            )
            _write_package(
                head,
                {
                    "newton/__init__.py": shared_init,
                    "newton/_src/actuators.py": """
from dataclasses import dataclass


@dataclass
class ActuatorParsed:
    path: str
    name: str
""",
                },
            )

            diff = detector.compare_symbols(
                detector.extract_api_symbols(base),
                detector.extract_api_symbols(head),
            )

            constructor_changes = [item for item in diff["changed"] if item["path"] == "newton.ActuatorParsed.__init__"]
            self.assertEqual(len(constructor_changes), 1)
            signature_change = constructor_changes[0]["changes"][0]
            self.assertEqual(signature_change["field"], "signature")
            self.assertEqual(
                signature_change["before"],
                "__init__(self: ActuatorParsed, name: str, path: str)",
            )
            self.assertEqual(
                signature_change["after"],
                "__init__(self: ActuatorParsed, path: str, name: str)",
            )

    def test_main_reports_skipped_modules_without_false_removals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            head = root / "head"
            (base / "newton").mkdir(parents=True)
            (head / "newton").mkdir(parents=True)

            (base / "newton" / "__init__.py").write_text("from .bad import public\n", encoding="utf-8")
            (head / "newton" / "__init__.py").write_text("from .bad import public\n", encoding="utf-8")
            (base / "newton" / "bad.py").write_text(
                "def public() -> int:\n    return 1\n",
                encoding="utf-8",
            )
            (head / "newton" / "bad.py").write_text("def public(:\n    return 1\n", encoding="utf-8")

            output = io.StringIO()
            warnings = io.StringIO()
            with (
                mock.patch.object(sys, "argv", ["detect_api_changes.py", str(base), str(head)]),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(warnings),
            ):
                self.assertEqual(detector.main(), 0)

            report = json.loads(output.getvalue())
            self.assertIn("has_analysis_warnings", report)
            self.assertIn("skipped_modules", report)
            self.assertTrue(report["has_analysis_warnings"])
            self.assertEqual(report["skipped_modules"]["base"], [])
            self.assertEqual(report["skipped_modules"]["head"][0]["module"], "newton.bad")
            self.assertIn("SyntaxError", report["skipped_modules"]["head"][0]["reason"])
            self.assertEqual(report["diff"]["summary"]["removed_count"], 0)
            self.assertFalse(report["diff"]["has_changes"])
            self.assertIn("Analysis warnings", report["comment"])
            self.assertIn("WARNING: skipping newton/bad.py", warnings.getvalue())

    def test_main_reports_decode_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            head = root / "head"
            (base / "newton").mkdir(parents=True)
            (head / "newton").mkdir(parents=True)

            (base / "newton" / "__init__.py").write_text("", encoding="utf-8")
            (head / "newton" / "__init__.py").write_text("", encoding="utf-8")
            (base / "newton" / "bad.py").write_text("", encoding="utf-8")
            (head / "newton" / "bad.py").write_bytes(b"\xff")

            output = io.StringIO()
            warnings = io.StringIO()
            with (
                mock.patch.object(sys, "argv", ["detect_api_changes.py", str(base), str(head)]),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(warnings),
            ):
                self.assertEqual(detector.main(), 0)

            report = json.loads(output.getvalue())
            self.assertTrue(report["has_analysis_warnings"])
            self.assertEqual(report["skipped_modules"]["head"][0]["module"], "newton.bad")
            self.assertIn("UnicodeDecodeError", report["skipped_modules"]["head"][0]["reason"])
            self.assertIn("WARNING: skipping newton/bad.py", warnings.getvalue())

    def test_format_comment_escapes_markdown_control_characters(self):
        comment = detector.format_comment(
            {
                "has_changes": True,
                "summary": {
                    "added_count": 1,
                    "removed_count": 0,
                    "changed_count": 0,
                    "total": 1,
                },
                "added": [
                    {
                        "path": "newton.public.MALICIOUS",
                        "kind": "constant",
                        "signature": "MALICIOUS: </details><img src=x>",
                        "value": "'closing </details>\\n### injected\\n`tick`'",
                    }
                ],
                "removed": [],
                "changed": [],
            }
        )

        self.assertNotIn("</details>", comment)
        self.assertNotIn("<img", comment)
        self.assertNotIn("\n### injected", comment)
        self.assertIn("&lt;/details&gt;", comment)
        self.assertIn("\\n### injected", comment)
        self.assertIn("`tick`", comment)


if __name__ == "__main__":
    unittest.main(verbosity=2)
