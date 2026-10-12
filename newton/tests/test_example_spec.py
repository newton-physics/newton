# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import importlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import newton.examples
from newton.examples._spec import load_example_spec, render_example_spec


class TestExampleSpec(unittest.TestCase):
    def test_loads_literal_without_executing_module(self):
        """Load static metadata without executing surrounding module code."""
        source = '''"""Example description."""
raise RuntimeError("must not execute")
_EXAMPLE_SPEC = {
    "schema_version": 1,
    "success_criteria": ("It works.",),
    "runs": ({"args": (), "success_criteria": ()},),
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example_test.py"
            path.write_text(source, encoding="utf-8")
            spec = load_example_spec(path)

        self.assertIsNotNone(spec)
        self.assertEqual(spec.description, "Example description.")
        self.assertEqual(spec.success_criteria, ("It works.",))

    def test_rejects_non_literal_metadata(self):
        """Reject metadata that would require executing the example module."""
        source = '''"""Example description."""
_EXAMPLE_SPEC = make_spec()
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example_test.py"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a literal value"):
                load_example_spec(path)

    def test_rejects_invalid_schema_versions(self):
        """Require the schema version to be the exact supported integer value."""
        for schema_version in ("True", "1.0", "2"):
            with self.subTest(schema_version=schema_version):
                source = f'''"""Example description."""
_EXAMPLE_SPEC = {{
    "schema_version": {schema_version},
    "success_criteria": ("It works.",),
    "runs": ({{"args": (), "success_criteria": ()}},),
}}
'''
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "example_test.py"
                    path.write_text(source, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "unsupported example spec schema version"):
                        load_example_spec(path)

    def test_renders_markdown_commands_and_criteria(self):
        """Render common and run-specific criteria as readable Markdown."""
        path = Path(__file__).parents[1] / "examples" / "cable" / "example_cable_bundle_hysteresis.py"
        spec = load_example_spec(path)
        output = render_example_spec("cable_bundle_hysteresis", spec, output_format="markdown")

        self.assertIn("# `cable_bundle_hysteresis`", output)
        self.assertIn("Demonstrates Dahl friction model for cable bending hysteresis.", output)
        self.assertIn("showing realistic memory effects in cable dynamics.", output)
        self.assertIn("--test --viewer null", output)
        self.assertIn("python -m newton.examples cable_bundle_hysteresis --no-dahl", output)
        self.assertIn("The cables are mostly straight.", output)

    def test_renders_json_for_agents(self):
        """Render a structured JSON form with complete commands."""
        path = Path(__file__).parents[1] / "examples" / "robot" / "example_robot_asroballet.py"
        spec = load_example_spec(path)
        payload = json.loads(render_example_spec("robot_asroballet", spec, output_format="json"))

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["name"], "robot_asroballet")
        self.assertIn("Demonstrates LQR and learned-policy control for the asRoBallet robot", payload["description"])
        self.assertIn('default; pass "--controller lqr" to use LQR.', payload["description"])
        self.assertEqual(len(payload["runs"]), 2)
        self.assertEqual(
            payload["runs"][1]["command"],
            "python -m newton.examples robot_asroballet --controller lqr",
        )

    def test_prototype_examples_have_valid_specs(self):
        """Validate the metadata embedded in each prototype example."""
        examples = Path(__file__).parents[1] / "examples"
        paths = {
            "newton.examples.robot.example_robot_omniwheel": examples / "robot" / "example_robot_omniwheel.py",
            "newton.examples.robot.example_robot_asroballet": examples / "robot" / "example_robot_asroballet.py",
            "newton.examples.cable.example_cable_bundle_hysteresis": (
                examples / "cable" / "example_cable_bundle_hysteresis.py"
            ),
        }

        for module_path, path in paths.items():
            with self.subTest(path=path):
                spec = load_example_spec(path)
                self.assertIsNotNone(spec)

                module = importlib.import_module(module_path)
                create_parser = getattr(module.Example, "create_parser", newton.examples.create_parser)
                parser = create_parser()
                for run in spec.runs:
                    parser.parse_args(run.args)

    def test_all_examples_have_specs_and_descriptions(self):
        """Require every registered example to define a spec and general description."""
        examples_directory = Path(newton.examples.get_source_directory())
        errors = []
        for example_name, module_path in newton.examples.get_examples().items():
            relative_module = module_path.removeprefix("newton.examples.")
            source_path = examples_directory.joinpath(*relative_module.split(".")).with_suffix(".py")
            try:
                spec = load_example_spec(source_path)
            except ValueError as error:
                errors.append(f"{example_name}: {error}")
                continue
            if spec is None:
                errors.append(f"{example_name}: does not define _EXAMPLE_SPEC")
            elif not spec.description.strip():
                errors.append(f"{example_name}: does not have a module description")

        self.assertEqual(errors, [], "Invalid example specifications:\n- " + "\n- ".join(errors))

    def test_describe_cli_renders_json_without_running_example(self):
        """Dispatch the JSON CLI without executing the selected example."""
        output = io.StringIO()
        argv = ["newton.examples", "--describe", "robot_asroballet", "--format", "json"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(runpy, "run_module") as run_module,
            contextlib.redirect_stdout(output),
            self.assertRaises(SystemExit) as exit_context,
        ):
            newton.examples.main()

        self.assertEqual(exit_context.exception.code, 0)
        run_module.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["name"], "robot_asroballet")

    def test_describe_cli_reports_missing_metadata(self):
        """Report a clear CLI error when an example has no embedded specification."""
        error = io.StringIO()
        argv = ["newton.examples", "--describe", "basic_pendulum"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch("newton.examples._spec.load_example_spec", return_value=None),
            contextlib.redirect_stderr(error),
            self.assertRaises(SystemExit) as exit_context,
        ):
            newton.examples.main()

        self.assertEqual(exit_context.exception.code, 2)
        self.assertIn("does not define an example specification", error.getvalue())


if __name__ == "__main__":
    unittest.main()
