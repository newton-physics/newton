# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import contextlib
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


def _declared_options(source_path):
    """Read literal argparse option declarations without importing an example."""
    options = set()
    for node in ast.walk(ast.parse(source_path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        flags = {arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
        options.update(flag for flag in flags if flag.startswith("-"))
        for keyword in node.keywords:
            action = keyword.value
            if keyword.arg == "action" and (
                (isinstance(action, ast.Name) and action.id == "BooleanOptionalAction")
                or (isinstance(action, ast.Attribute) and action.attr == "BooleanOptionalAction")
            ):
                options.update("--no-" + flag[2:] for flag in flags if flag.startswith("--"))
    return options


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
        self.assertEqual(output.count("python -m newton.examples cable_bundle_hysteresis --no-dahl"), 1)
        self.assertNotIn("Run interactively:", output)
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
        self.assertEqual(len(payload["runs"]), 3)
        self.assertEqual(
            payload["runs"][1]["command"],
            "python -m newton.examples robot_asroballet --controller lqr",
        )

    def test_all_examples_have_specs_and_descriptions(self):
        """Require descriptions, valid specs, and declared run options across the registry.

        This static check detects stale option names, including generated negative
        boolean options; it does not validate argument values or execute examples.
        """
        examples_directory = Path(newton.examples.get_source_directory())
        shared_options = _declared_options(examples_directory / "__init__.py")
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
            if spec is not None:
                options = shared_options | _declared_options(source_path)
                for run in spec.runs:
                    for argument in run.args:
                        option = argument.partition("=")[0]
                        if option.startswith("--") and option not in options:
                            errors.append(f"{example_name}: run uses undeclared option {option}")

        self.assertEqual(errors, [], "Invalid example specifications:\n- " + "\n- ".join(errors))

    def test_preserves_documented_run_variants(self):
        """Retain commands removed from descriptions in the rendered run catalog."""
        documented_runs = {
            "basic_conveyor": ((), ("--solver", "kamino")),
            "basic_conveyor_forces": (("--solver", "kamino"),),
            "basic_dzhanibekov": (("--solver", "xpbd"), ("--solver", "mujoco"), ("--solver", "kamino")),
            "basic_heightfield": (("--solver", "mujoco"), ("--solver", "kamino")),
            "basic_joints": (("--solver", "xpbd"), ("--solver", "vbd"), ("--solver", "kamino")),
            "basic_pendulum": (("--solver", "xpbd"), ("--solver", "kamino")),
            "basic_plotting": (("--world-count", "4"),),
            "basic_shapes": ((), ("--solver", "kamino")),
            "basic_urdf": ((), ("--solver", "kamino")),
            "nut_bolt_hydro": (("--solver", "kamino"),),
            "nut_bolt_sdf": ((), ("--solver", "kamino")),
            "pyramid": ((), ("--solver", "kamino")),
            "controller_differential_ik": (("--ik-method", "adaptive_damping"),),
            "controller_joint_impedance_heterogeneous": (("--solver", "kamino"),),
            "diffsim_spring_cage": ((),),
            "ik_cube_stacking": (("--world-count", "16"),),
            "kamino_basic_dr_testmech": (("--world-count", "16"),),
            "kamino_basic_fourbar": (("--world-count", "16"),),
            "kamino_robot_anymal_d": (("--world-count", "16"),),
            "kamino_robot_dr_legs": (("--world-count", "16"),),
            "mpm_anymal": (("--viewer", "gl"),),
            "admm_contact_solver": (("--solver", "free"),),
            "robot_allegro_hand": (("--world-count", "16"), ("--solver", "kamino")),
            "robot_anymal_d": (("--world-count", "16"), ("--solver", "kamino")),
            "robot_asroballet": ((),),
            "robot_cartpole": (("--world-count", "100"),),
            "robot_g1": (("--world-count", "16"),),
            "robot_h1": (("--world-count", "16"), ("--solver", "kamino")),
            "robot_panda_hydro": (("--scene", "pen", "--world-count", "1"),),
            "robot_policy": (("--robot", "anymal", "--physx"),),
            "robot_ur10": (("--world-count", "16"), ("--solver", "kamino")),
            "selection_articulations": ((), ("--solver", "kamino")),
            "selection_cartpole": (("--world-count", "16", "--max-worlds", "8"),),
            "selection_materials": ((), ("--solver", "kamino")),
            "selection_multiple": ((), ("--solver", "kamino")),
        }
        examples_directory = Path(newton.examples.get_source_directory())
        examples = newton.examples.get_examples()
        for example_name, expected_args in documented_runs.items():
            with self.subTest(example=example_name):
                relative_module = examples[example_name].removeprefix("newton.examples.")
                source_path = examples_directory.joinpath(*relative_module.split(".")).with_suffix(".py")
                spec = load_example_spec(source_path)
                payload = json.loads(render_example_spec(example_name, spec, output_format="json"))
                actual_args = {tuple(run["args"]) for run in payload["runs"]}
                self.assertTrue(
                    set(expected_args) <= actual_args, f"Missing commands: {set(expected_args) - actual_args}"
                )
                if example_name == "admm_contact_solver":
                    criteria_by_args = {tuple(run["args"]): run["success_criteria"] for run in payload["runs"]}
                    coupled_criterion = "The tray responds to particle contact through ADMM contact coupling."
                    self.assertNotIn(coupled_criterion, payload["success_criteria"])
                    self.assertIn(coupled_criterion, criteria_by_args[()])
                    self.assertNotIn(coupled_criterion, criteria_by_args[("--solver", "free")])
                    self.assertTrue(any("pass through the tray" in c for c in criteria_by_args[("--solver", "free")]))

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
