# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
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

    def test_compare_ignores_instance_attribute_initializer_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            head = root / "head"

            shared_init = "from ._src.builder import ModelBuilder\nfrom .constants import FOO\n"
            _write_package(
                base,
                {
                    "newton/__init__.py": shared_init,
                    "newton/constants.py": "FOO = 1\n",
                    "newton/_src/builder.py": """
class ModelBuilder:
    def __init__(self):
        self.shape_collision_filter_pairs: list[tuple[int, int]] = []
""",
                },
            )
            _write_package(
                head,
                {
                    "newton/__init__.py": shared_init,
                    "newton/constants.py": "FOO = 2\n",
                    "newton/_src/builder.py": """
class _BuilderShapeCollisionFilterPairs:
    pass


class ModelBuilder:
    def __init__(self):
        self.shape_collision_filter_pairs = _BuilderShapeCollisionFilterPairs()
""",
                },
            )

            diff = detector.compare_symbols(
                detector.extract_api_symbols(base),
                detector.extract_api_symbols(head),
            )

            changed_paths = {item["path"] for item in diff["changed"]}
            self.assertIn("newton.FOO", changed_paths)
            self.assertIn("newton.constants.FOO", changed_paths)
            self.assertNotIn("newton.ModelBuilder.shape_collision_filter_pairs", changed_paths)

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

    def test_copy_symbol_preserves_existing_source_module(self):
        symbol = detector._copy_symbol(
            {
                "kind": "class",
                "signature": "class Public",
                "source_module": "newton.pkg.leaf",
            },
            source_module="newton.pkg",
        )

        self.assertEqual(symbol["source_module"], "newton.pkg.leaf")

    def test_init_reexport_propagation_records_source_module(self):
        tree = ast.parse("from .leaf import Public\n")
        all_definitions = {
            "newton.pkg": {},
            "newton.pkg.leaf": {
                "Public": detector._make_symbol("class", "class Public"),
                "Public.method": detector._make_symbol("method", "method(self: Public) -> int"),
            },
        }

        self.assertTrue(detector._propagate_init_reexports("newton.pkg", tree, all_definitions))

        self.assertEqual(all_definitions["newton.pkg"]["Public"]["source_module"], "newton.pkg.leaf")
        self.assertEqual(all_definitions["newton.pkg"]["Public.method"]["source_module"], "newton.pkg.leaf")

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

    def test_format_comment_compacts_long_signature_changes(self):
        before = (
            "add_shape(self: ModelBuilder, *, body: int, type: int, xform: Transform | None = None, "
            "cfg: ShapeConfig | None = None, scale: Vec3 | None = None, "
            "src: Mesh | Gaussian | Heightfield | Any | None = None, is_static: bool = False, "
            "color: Vec3 | None = None, label: str | None = None, "
            "custom_attributes: dict[str, Any] | None = None) -> int"
        )
        after = (
            "add_shape(self: ModelBuilder, *, body: int, type: int, xform: Transform | None = None, "
            "cfg: ShapeConfig | None = None, scale: Vec3 | None = None, "
            "src: Mesh | Gaussian | Heightfield | Any | None = None, is_static: bool = False, "
            "color: Vec3 | None = None, opacity: float | None = None, label: str | None = None, "
            "custom_attributes: dict[str, Any] | None = None) -> int"
        )

        comment = detector.format_comment(
            {
                "has_changes": True,
                "summary": {
                    "added_count": 0,
                    "removed_count": 0,
                    "changed_count": 1,
                    "total": 1,
                },
                "added": [],
                "removed": [],
                "changed": [
                    {
                        "path": "newton.ModelBuilder.add_shape",
                        "kind": "method",
                        "changes": [
                            {
                                "field": "signature",
                                "before": before,
                                "after": after,
                            }
                        ],
                    }
                ],
            }
        )

        self.assertNotIn("src: Mesh | Gaussian", comment)
        self.assertNotIn("-&gt;", comment)
        self.assertIn(
            "  - before:\n"
            "    ```python\n"
            "    add_shape(..., color: Vec3 | None = None, label: str | None = None, ...) -> int\n"
            "    ```",
            comment,
        )
        self.assertIn(
            "  - after:\n"
            "    ```python\n"
            "    add_shape(..., color: Vec3 | None = None, opacity: float | None = None, "
            "label: str | None = None, ...) -> int\n"
            "    ```",
            comment,
        )
        self.assertNotIn("`add_shape", comment)

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
        self.assertIn("```python", comment)
        self.assertIn("&lt;/details>", comment)
        self.assertIn("&lt;img src=x>", comment)
        self.assertIn("\\n### injected", comment)
        self.assertIn("`tick`", comment)


if __name__ == "__main__":
    unittest.main(verbosity=2)
