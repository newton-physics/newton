# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load and render static metadata embedded in Newton examples."""

from __future__ import annotations

import ast
import json
import shlex
from dataclasses import dataclass
from pathlib import Path

_SPEC_NAME = "_EXAMPLE_SPEC"
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExampleRunSpec:
    """Describe one supported invocation of an example."""

    args: tuple[str, ...]
    success_criteria: tuple[str, ...]


@dataclass(frozen=True)
class ExampleSpec:
    """Describe the stable behavior demonstrated by an example.

    The description is sourced from the example module's docstring.
    """

    description: str
    success_criteria: tuple[str, ...]
    runs: tuple[ExampleRunSpec, ...]


def _read_string_sequence(value, *, field: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list or tuple of strings")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain only non-empty strings")
    return tuple(value)


def _validate_spec(raw_spec, *, description: str) -> ExampleSpec:
    if not isinstance(raw_spec, dict):
        raise ValueError(f"{_SPEC_NAME} must be a dictionary literal")

    expected_keys = {"schema_version", "success_criteria", "runs"}
    if set(raw_spec) != expected_keys:
        raise ValueError(f"{_SPEC_NAME} must contain exactly {sorted(expected_keys)}")
    schema_version = raw_spec["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != _SCHEMA_VERSION:
        raise ValueError(f"unsupported example spec schema version: {schema_version!r}")

    success_criteria = _read_string_sequence(raw_spec["success_criteria"], field="success_criteria", allow_empty=False)
    raw_runs = raw_spec["runs"]
    if not isinstance(raw_runs, (list, tuple)) or not raw_runs:
        raise ValueError("runs must be a non-empty list or tuple")

    runs = []
    seen_args = set()
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict) or set(raw_run) != {"args", "success_criteria"}:
            raise ValueError(f"runs[{index}] must contain exactly 'args' and 'success_criteria'")
        args = _read_string_sequence(raw_run["args"], field=f"runs[{index}].args", allow_empty=True)
        if args in seen_args:
            raise ValueError(f"runs[{index}].args duplicates an earlier run")
        seen_args.add(args)
        runs.append(
            ExampleRunSpec(
                args=args,
                success_criteria=_read_string_sequence(
                    raw_run["success_criteria"],
                    field=f"runs[{index}].success_criteria",
                    allow_empty=True,
                ),
            )
        )

    return ExampleSpec(description=description, success_criteria=success_criteria, runs=tuple(runs))


def load_example_spec(source_path: str | Path) -> ExampleSpec | None:
    """Load an example specification without importing or executing its module.

    Args:
        source_path: Path to an example Python module.

    Returns:
        The validated specification, or ``None`` when the module does not define one.

    Raises:
        ValueError: If the module defines malformed or non-literal metadata.
    """
    source_path = Path(source_path)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    description = ast.get_docstring(tree, clean=True)

    values = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == _SPEC_NAME for target in node.targets
        ):
            values.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == _SPEC_NAME:
            values.append(node.value)

    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"{source_path}: {_SPEC_NAME} must be assigned exactly once")
    if not description:
        raise ValueError(f"{source_path}: an example with {_SPEC_NAME} must have a module docstring")

    try:
        raw_spec = ast.literal_eval(values[0])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source_path}: {_SPEC_NAME} must be a literal value") from error

    try:
        return _validate_spec(raw_spec, description=description)
    except ValueError as error:
        raise ValueError(f"{source_path}: {error}") from error


def _command(example_name: str, args: tuple[str, ...]) -> str:
    return shlex.join(("python", "-m", "newton.examples", example_name, *args))


def render_example_spec(example_name: str, spec: ExampleSpec, *, output_format: str) -> str:
    """Render an example specification for human or machine consumption.

    Args:
        example_name: Short example name accepted by ``python -m newton.examples``.
        spec: Validated example specification.
        output_format: Either ``"markdown"`` or ``"json"``.

    Returns:
        The rendered specification.

    Raises:
        ValueError: If ``output_format`` is unsupported.
    """
    if output_format == "json":
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "name": example_name,
            "description": spec.description,
            "success_criteria": list(spec.success_criteria),
            "runs": [
                {
                    "args": list(run.args),
                    "command": _command(example_name, run.args),
                    "success_criteria": list(run.success_criteria),
                }
                for run in spec.runs
            ],
        }
        return json.dumps(payload, indent=2)

    if output_format != "markdown":
        raise ValueError(f"unsupported output format: {output_format!r}")

    lines = [f"# `{example_name}`", "", spec.description, "", "## Success criteria", ""]
    lines.extend(f"{index}. {criterion}" for index, criterion in enumerate(spec.success_criteria, 1))
    lines.extend(("", "## Runs"))
    for index, run in enumerate(spec.runs, 1):
        lines.extend(("", f"### Run {index}", "", "```bash", _command(example_name, run.args), "```"))
        if run.success_criteria:
            lines.extend(("", "Additional success criteria:", ""))
            lines.extend(f"{item}. {criterion}" for item, criterion in enumerate(run.success_criteria, 1))
    return "\n".join(lines)
