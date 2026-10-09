# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""DC-motor import and construction-time validation, without eager backend imports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mujoco import MjsActuator

    from ...sim.model import Model


_INPUT_CODES = {"pos": 1, "vel": 2, "voltage": 8}
_INPUT_NAMES = {code: name for name, code in _INPUT_CODES.items()}
_INPUT_ALIASES = {"position": "pos", "velocity": "vel"}
_CONSTRUCTION_PARAMETERS = (
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_dynprm",
    "actuator_dyntype",
    "actuator_gaintype",
    "actuator_biastype",
    "actuator_actdim",
    "actuator_actearly",
    "actuator_forcerange",
    "actuator_forcelimited",
    "actuator_ctrlspec",
)


def parse_input(value: str | int) -> int:
    """Parse a supported single-input control signature or reject unsupported inputs."""
    text = str(value).strip().lower()
    try:
        signature = int(value) if isinstance(value, int | np.integer) else int(text)
    except ValueError:
        # Accept enum representations as well as the MJCF names and Newton aliases.
        name = text.rsplit(".", 1)[-1].rsplit("_", 1)[-1]
        signature = _INPUT_CODES.get(_INPUT_ALIASES.get(name, name))
    if signature not in _INPUT_CODES.values():
        raise NotImplementedError(
            "SolverMuJoCo supports one DC-motor control input per actuator row: "
            "'voltage', 'pos', or 'vel'. MuJoCo 'ff', 'none', and combined input "
            "signatures are not supported by Newton's one-control-per-actuator mapping."
        )
    return signature


def resolve_mjcf_parameters(root: ET.Element) -> dict[ET.Element, dict[str, str]]:
    """Resolve shortcuts and inherited defaults into attributes for the MJCF importer."""
    motors = root.findall("actuator/dcmotor")
    if not motors:
        return {}
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "Importing MJCF <dcmotor> actuators requires MuJoCo. Install it with `uv pip install 'newton[sim]'`."
        ) from exc

    # Resolve native shortcuts without compiling a model or loading geometry/assets.
    spec = mujoco.MjSpec.from_string(_actuator_only_xml(root, motors))
    return {motor: _parameter_attributes(actuator) for motor, actuator in zip(motors, spec.actuators, strict=True)}


def _actuator_only_xml(root: ET.Element, motors: list[ET.Element]) -> str:
    """Preserve default hierarchy and shortcut order while excluding unrelated content."""
    native_root = ET.Element("mujoco")
    actuator_tags = {
        "general",
        "motor",
        "position",
        "velocity",
        "intvelocity",
        "damper",
        "cylinder",
        "muscle",
        "adhesion",
        "dcmotor",
        "orientation",
        "pid",
    }
    # Defaults inherit compiled values, not the original XML vectors.
    for default in root.findall("default"):
        copied = deepcopy(default)
        for node in copied.iter("default"):
            for child in list(node):
                if child.tag != "default" and child.tag not in actuator_tags:
                    node.remove(child)
        native_root.append(copied)
    ET.SubElement(native_root, "actuator").extend(deepcopy(motor) for motor in motors)
    for element in native_root.iter():
        if "input" in element.attrib:
            value = element.attrib["input"].strip().lower()
            element.set("input", _INPUT_ALIASES.get(value, value))
    return ET.tostring(native_root, encoding="unicode")


def _parameter_attributes(actuator: MjsActuator) -> dict[str, str]:
    """Reuse the importer's normal attribute converters for resolved native parameters."""
    values = {}
    for name in (
        "gainprm",
        "biasprm",
        "dynprm",
        "gear",
        "ctrlrange",
        "forcerange",
        "actrange",
        "lengthrange",
        "damping",
    ):
        values[name] = " ".join(str(value) for value in getattr(actuator, name))
    for name in ("dyntype", "gaintype", "biastype", "actdim"):
        values[name] = str(int(getattr(actuator, name)))
    for name in ("ctrllimited", "forcelimited", "actlimited"):
        values[name] = {0: "false", 1: "true", 2: "auto"}[int(getattr(actuator, name))]
    for name in ("cranklength", "armature"):
        values[name] = str(getattr(actuator, name))
    values["actearly"] = str(bool(actuator.actearly)).lower()
    # Zero is MuJoCo's automatic signature, which defaults to voltage for DC motors.
    values["input"] = _INPUT_NAMES[parse_input(int(actuator.ctrlspec) or _INPUT_CODES["voltage"])]
    return values


def validate_worlds(model: Model, *, dcmotor_type: int) -> None:
    """Require matching shortcut layouts and construction parameters across worlds."""
    world_count = model.world_count
    attributes = getattr(model, "mujoco", None)
    world_attribute = getattr(attributes, "actuator_world", None)
    type_attribute = getattr(attributes, "ctrl_type", None)
    if world_count <= 1 or world_attribute is None or type_attribute is None:
        return
    ctrl_type = type_attribute.numpy()
    if not np.any(ctrl_type == dcmotor_type):
        return

    # Group once in actuator order, rather than scanning every row for each world.
    actuator_world = world_attribute.numpy()
    rows = np.flatnonzero((actuator_world >= 0) & (actuator_world < world_count))
    row_counts = np.bincount(actuator_world[rows], minlength=world_count)
    layout = "matching high-level DC-motor actuator layouts"
    _require_matching_worlds(row_counts, layout)
    rows = rows[np.argsort(actuator_world[rows], kind="stable")].reshape(world_count, row_counts[0])
    dcmotor_rows = ctrl_type[rows] == dcmotor_type
    _require_matching_worlds(dcmotor_rows, layout)
    rows = rows[:, dcmotor_rows[0]]

    # The template model is replicated across worlds. These resolved shortcut
    # parameters, unlike general-actuator arrays, are not updated independently.
    for name in _CONSTRUCTION_PARAMETERS:
        attribute = getattr(attributes, name, None)
        if attribute is None:
            raise ValueError(f"High-level DC-motor actuator rows are missing mujoco:{name}.")
        _require_matching_worlds(attribute.numpy()[rows], f"identical high-level DC-motor parameters (mujoco:{name})")


def _require_matching_worlds(values: np.ndarray, description: str) -> None:
    """Report the first world whose scalar or array values differ from world zero."""
    differences = (values[1:] != values[0]).reshape(len(values) - 1, -1)
    mismatches = np.flatnonzero(np.any(differences, axis=1))
    if mismatches.size:
        raise ValueError(
            f"SolverMuJoCo with separate_worlds=True requires {description}; "
            f"world {mismatches[0] + 1} differs from world 0."
        )
