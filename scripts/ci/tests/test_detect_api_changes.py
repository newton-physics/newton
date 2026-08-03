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

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ci" / "detect_api_changes.py"

spec = importlib.util.spec_from_file_location("detect_api_changes", SCRIPT)
assert spec is not None and spec.loader is not None
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)


def _write(root: Path, files: dict[str, str]) -> None:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


def _snapshot(root: Path) -> tuple[dict, dict]:
    notes = {}
    return detector.extract_api_symbols(root, notes), notes


def _run(base: Path, head: Path) -> dict:
    output = io.StringIO()
    with mock.patch.object(sys, "argv", [str(SCRIPT), str(base), str(head)]), contextlib.redirect_stdout(output):
        assert detector.main() == 0
    return json.loads(output.getvalue())


class TestDetectApiChanges(unittest.TestCase):
    def test_extracts_common_declarations_and_bindings(self):
        """Extract functions, overloads, properties, bindings, and data fields."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                {
                    "newton/__init__.py": "from ._src.api import Example, Record, Pair, convert\n__all__ = ['Example', 'Record', 'Pair', 'convert']\n",
                    "newton/_src/api.py": """
from dataclasses import dataclass
from typing import NamedTuple, overload

@overload
def convert(value: int) -> int: ...
@overload
def convert(value: str) -> str: ...
def convert(value: object) -> object: return value

class Example:
    @classmethod
    def build(cls, value: int = 1) -> "Example": ...
    @staticmethod
    def check(value: int) -> bool: ...
    @property
    def value(self) -> int: ...
    @value.setter
    def value(self, new_value: int) -> None: ...

@dataclass(kw_only=True)
class Record:
    count: int
    label: str = "item"

Pair = NamedTuple("Pair", [("left", int), ("right", str)])
""",
                },
            )

            symbols, notes = _snapshot(root)

            self.assertFalse(notes)
            self.assertEqual(symbols["newton.Example.build"]["binding"], "classmethod")
            self.assertEqual(symbols["newton.Example.check"]["binding"], "staticmethod")
            self.assertEqual(set(symbols["newton.Example.value"]["accessors"]), {"getter", "setter"})
            self.assertEqual(len(symbols["newton.convert"]["overloads"]), 2)
            self.assertTrue(symbols["newton.Record"]["dataclass"])
            self.assertEqual([field["name"] for field in symbols["newton.Record"]["fields"]], ["count", "label"])
            self.assertEqual([field["name"] for field in symbols["newton.Pair"]["fields"]], ["left", "right"])

    def test_detects_public_constructor_signature_changes(self):
        """Detect changes to a public class constructor signature."""
        template = """
__all__ = ["Example"]

class Example:
    def __init__(self, {parameters}): ...
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, head = root / "base", root / "head"
            for tree, parameters in (
                (base, "value: int"),
                (head, "value: str, mode: bool = False"),
            ):
                _write(tree, {"newton/__init__.py": template.format(parameters=parameters)})

            report = _run(base, head)

            self.assertEqual(report["decision"], "changed")
            self.assertTrue(report["needs_review"])
            changed = next(item for item in report["diff"]["changed"] if item["path"] == "newton.Example")
            self.assertIn("constructors", {item["field"] for item in changed["changes"]})

    def test_marks_changed_dynamic_exports_as_unknown(self):
        """Request review when a dependency of an unsupported export changes."""
        template = """
_EXPORTS = ["{export}"]
__all__ = _EXPORTS

class Foo: ...
class Bar: ...
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, head = root / "base", root / "head"
            for tree, export in ((base, "Foo"), (head, "Bar")):
                _write(tree, {"newton/__init__.py": template.format(export=export)})

            report = _run(base, head)

            self.assertEqual(report["decision"], "unknown")
            self.assertTrue(report["needs_review"])
            self.assertTrue(report["has_analysis_warnings"])

    def test_resolves_known_solver_lazy_export_map(self):
        """Keep the known Newton lazy solver facade equivalent to direct imports."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(
                root,
                {
                    "newton/__init__.py": "from . import solvers\n__all__ = ['solvers']\n",
                    "newton/solvers.py": "from ._src import solvers as _solvers\n__all__ = [*_solvers.__all__]\n",
                    "newton/_src/__init__.py": "",
                    "newton/_src/solvers/__init__.py": """
__all__ = ["SolverExample"]
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "SolverExample": (".example", "SolverExample"),
}
""",
                    "newton/_src/solvers/example.py": "class SolverExample:\n    def step(self, dt: float) -> None: ...\n",
                },
            )

            symbols, notes = _snapshot(root)

            self.assertFalse(notes)
            self.assertIn("newton.solvers.SolverExample.step", symbols)

    def test_detects_newton_solver_custom_attribute_addition(self):
        """Detect the custom geom_group registration added by Newton PR 3283."""
        source = """
class SolverMuJoCo:
    @classmethod
    def register_custom_attributes(cls, builder):
        {body}
"""
        addition = """builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="geom_group",
                frequency=AttributeFrequency.SHAPE,
                dtype=wp.int32,
                namespace="mujoco",
            )
        )"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, head = root / "base", root / "head"
            common_init = "from ._src.solvers.mujoco import SolverMuJoCo\n__all__ = ['SolverMuJoCo']\n"
            for tree, body in ((base, "pass"), (head, addition)):
                _write(
                    tree,
                    {
                        "newton/__init__.py": common_init,
                        "newton/_src/solvers/mujoco.py": source.format(body=body),
                    },
                )

            report = _run(base, head)

            self.assertEqual(report["decision"], "changed")
            added = {item["path"]: item for item in report["diff"]["added"]}
            path = next(path for path in added if path.endswith(".custom_attribute.mujoco:geom_group"))
            self.assertEqual(added[path]["frequency"], "AttributeFrequency.SHAPE")
            self.assertEqual(added[path]["dtype"], "wp.int32")

    def test_detects_custom_frequency_and_attribute_metadata_changes(self):
        """Detect frequency additions and attribute type or frequency modifications."""
        template = """
class SolverExample:
    def register_custom_attributes(self, builder):
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="row", namespace="demo"))
        builder.add_custom_attribute(ModelBuilder.CustomAttribute(
            name="value", namespace="demo", frequency={frequency}, dtype={dtype}))
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, head = root / "base", root / "head"
            for tree, frequency, dtype in (
                (base, "Frequency.BODY", "wp.float32"),
                (head, "'demo:row'", "wp.vec3"),
            ):
                _write(
                    tree,
                    {
                        "newton/__init__.py": "",
                        "newton/_src/solvers/example.py": template.format(frequency=frequency, dtype=dtype),
                    },
                )

            report = _run(base, head)

            changed = [item for item in report["diff"]["changed"] if item["kind"] == "custom_attribute"]
            self.assertEqual(len(changed), 1)
            fields = {item["field"] for item in changed[0]["changes"]}
            self.assertTrue({"frequency", "dtype"} <= fields)
            base_symbols, _ = _snapshot(base)
            self.assertTrue(any(path.endswith(".custom_frequency.demo:row") for path in base_symbols))

    def test_detects_registrations_in_delegated_solver_helpers(self):
        """Detect registrations delegated to a helper in another solver module."""
        helper = """
def register_demo_attributes(builder):
    ca = type(builder).CustomAttribute
    builder.add_custom_attribute(ca(name="existing", namespace="demo", dtype=wp.float32))
    {addition}
"""
        solver = """
from .registration import register_demo_attributes

class SolverExample:
    @classmethod
    def register_custom_attributes(cls, builder):
        register_demo_attributes(builder)
"""
        addition = 'builder.add_custom_frequency(type(builder).CustomFrequency(name="row", namespace="demo"))'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, head = root / "base", root / "head"
            for tree, body in ((base, ""), (head, addition)):
                _write(
                    tree,
                    {
                        "newton/__init__.py": "",
                        "newton/_src/solvers/demo/__init__.py": "",
                        "newton/_src/solvers/demo/registration.py": helper.format(addition=body),
                        "newton/_src/solvers/demo/solver.py": solver,
                    },
                )

            report = _run(base, head)

            self.assertEqual(report["decision"], "changed")
            self.assertFalse(report["has_analysis_warnings"])
            added = {item["path"] for item in report["diff"]["added"]}
            self.assertTrue(any(path.endswith(".custom_frequency.demo:row") for path in added))
            base_symbols, _ = _snapshot(base)
            self.assertTrue(any(path.endswith(".custom_attribute.demo:existing") for path in base_symbols))

    def test_marks_changed_delegation_as_unknown(self):
        """Request review when unsupported registration delegation changes."""
        template = """
class SolverExample:
    def register_custom_attributes(self, builder):
        {body}
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, head = root / "base", root / "head"
            for tree, body in ((base, "pass"), (head, "Config.register_custom_attributes(builder)")):
                _write(
                    tree,
                    {
                        "newton/__init__.py": "",
                        "newton/_src/solvers/example.py": template.format(body=body),
                    },
                )

            report = _run(base, head)

            self.assertEqual(report["decision"], "unknown")
            self.assertTrue(report["needs_review"])
            self.assertTrue(report["has_analysis_warnings"])

    def test_stable_unknown_construct_does_not_label_every_pr(self):
        """Ignore unchanged uncertainty fingerprints between revisions."""
        source = """
class SolverExample:
    def register_custom_attributes(self, builder):
        Config.register_custom_attributes(builder)
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, head = root / "base", root / "head"
            for tree in (base, head):
                _write(tree, {"newton/__init__.py": "", "newton/_src/solvers/example.py": source})

            report = _run(base, head)

            self.assertEqual(report["decision"], "unchanged")
            self.assertFalse(report["needs_review"])

    def test_marks_changed_registration_helper_as_unknown(self):
        """Request review when an unsupported registration helper changes."""
        template = """
def register_specs(builder):
    add_specs(builder, "{value}")

class SolverExample:
    def register_custom_attributes(self, builder):
        register_specs(builder)
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, head = root / "base", root / "head"
            for tree, value in ((base, "before"), (head, "after")):
                _write(tree, {"newton/__init__.py": "", "newton/_src/solvers/example.py": template.format(value=value)})

            report = _run(base, head)

            self.assertEqual(report["decision"], "unknown")
            self.assertTrue(report["needs_review"])
            self.assertTrue(report["has_analysis_warnings"])

    def test_bounds_untrusted_source_reads(self):
        """Reject oversized source files before parsing their full contents."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, {"newton/__init__.py": "VALUE = '123456789'\n__all__ = ['VALUE']\n"})
            notes = {}
            with mock.patch.object(detector, "MAX_FILE_BYTES", 8):
                symbols = detector.extract_api_symbols(root, notes)

            self.assertEqual(symbols, {})
            self.assertIn("limit:file:newton", notes)

    def test_rejects_symlinked_source_files(self):
        """Reject source symlinks before reading outside the checkout."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, {"newton/__init__.py": ""})
            outside = root / "outside.py"
            outside.write_text("ESCAPED = True\n", encoding="utf-8")
            link = root / "newton" / "escaped.py"
            try:
                link.symlink_to(outside)
            except OSError as error:
                self.skipTest(f"cannot create symlinks: {error}")

            symbols, notes = _snapshot(root)

            self.assertNotIn("newton.escaped", symbols)
            self.assertIn("unsafe-file:newton.escaped", notes)

    def test_rejects_symlinked_package_before_traversal(self):
        """Reject a package symlink before starting source traversal."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkout"
            outside = Path(tmp) / "outside"
            root.mkdir()
            _write(outside, {"__init__.py": "ESCAPED = True\n"})
            try:
                (root / "newton").symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"cannot create symlinks: {error}")

            notes = {}
            with mock.patch.object(Path, "rglob", side_effect=AssertionError("package traversal started")):
                symbols = detector.extract_api_symbols(root, notes)

            self.assertEqual(symbols, {})
            self.assertIn("unsafe-package:newton", notes)

    def test_bounds_and_escapes_comment_output(self):
        """Cap report items and escape attacker-controlled HTML."""
        head = {f"newton.<img src=x>.value{i}": {"kind": "constant"} for i in range(25)}
        diff = detector.compare_symbols({}, head)

        comment = detector.format_comment(diff, "changed", [])

        self.assertEqual(diff["summary"]["added_count"], 25)
        self.assertEqual(len(diff["added"]), detector.MAX_REPORT_ITEMS)
        self.assertNotIn("<img", comment)
        self.assertIn("&lt;img", comment)
        self.assertIn("and 5 more", comment)
        self.assertLessEqual(len(comment), detector.MAX_COMMENT_CHARS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
