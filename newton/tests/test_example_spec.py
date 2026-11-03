# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import contextlib
import io
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import newton.examples


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


def _documented_runs(description, example_name):
    """Read command arguments from the QA prose."""
    runs = set()
    for line in description.partition("\nRuns:\n")[2].splitlines():
        if line.startswith("- `"):
            command = shlex.split(line[3:-1])
            if command[:4] != ["python", "-m", "newton.examples", example_name]:
                raise ValueError(f"unexpected example command: {line}")
            args = tuple(command[4:])
            if args in runs:
                raise ValueError(f"duplicate example command: {line}")
            runs.add(args)
    return runs


class TestExampleSpec(unittest.TestCase):
    def test_all_examples_have_specs_and_descriptions(self):
        """Require descriptions, QA criteria, and declared command options for every example."""
        examples_directory = Path(newton.examples.get_source_directory())
        shared_options = _declared_options(examples_directory / "__init__.py")
        errors = []
        for example_name, module_path in newton.examples.get_examples().items():
            relative_module = module_path.removeprefix("newton.examples.")
            path = examples_directory.joinpath(*relative_module.split(".")).with_suffix(".py")
            description = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
            introduction, separator, criteria = description.partition("\nSuccess criteria:\n")
            if not introduction.strip() or not separator or not criteria.partition("\nRuns:\n")[0].strip():
                errors.append(f"{example_name}: missing description or success criteria")
            try:
                runs = _documented_runs(description, example_name)
            except ValueError as error:
                errors.append(f"{example_name}: {error}")
                continue
            if not runs:
                errors.append(f"{example_name}: no documented commands")
            options = shared_options | _declared_options(path)
            for args in runs:
                for argument in args:
                    option = argument.partition("=")[0]
                    if option.startswith("--") and option not in options:
                        errors.append(f"{example_name}: run uses undeclared option {option}")
        self.assertEqual(errors, [], "Invalid example documentation:\n- " + "\n- ".join(errors))

    def test_describe_cli_without_executing_example(self):
        """Print the complete docstring without executing example code, or report its absence."""
        description = (
            "Example probe.\n\nSuccess criteria:\n- It stays upright.\n\nRuns:\n- `python -m newton.examples probe`"
        )
        with tempfile.TemporaryDirectory() as directory:
            category = Path(directory) / "basic"
            category.mkdir()
            path = category / "example_probe.py"
            for text in (description, None):
                with self.subTest(description=text):
                    source = (repr(text) + "\n" if text else "") + 'raise RuntimeError("must not execute")\n'
                    path.write_text(source, encoding="utf-8")
                    output, error = io.StringIO(), io.StringIO()
                    with (
                        mock.patch.object(newton.examples, "get_source_directory", return_value=directory),
                        mock.patch.object(sys, "argv", ["newton.examples", "--describe", "probe"]),
                        contextlib.redirect_stdout(output),
                        contextlib.redirect_stderr(error),
                        self.assertRaises(SystemExit) as exit_context,
                    ):
                        newton.examples.main()
                    self.assertEqual(exit_context.exception.code, 0 if text else 2)
                    if text:
                        self.assertEqual(output.getvalue(), description + "\n")
                    else:
                        self.assertIn("does not have a module docstring", error.getvalue())


if __name__ == "__main__":
    unittest.main()
