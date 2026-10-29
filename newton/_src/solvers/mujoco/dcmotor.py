# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""DC-motor authoring, import, and validation, without eager backend imports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np

from ...sim import ModelBuilder
from .actuators import ActuatorTarget, _add_actuator

if TYPE_CHECKING:
    from mujoco import MjsActuator

    from ...sim.model import Model


_INPUT_CODES = {"pos": 1, "vel": 2, "voltage": 8}
_INPUT_NAMES = {code: name for name, code in _INPUT_CODES.items()}
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
)


def parse_input(value: str | int) -> int:
    """Parse a supported single-input control signature or reject unsupported inputs."""
    signature = int(value) if isinstance(value, int | np.integer) else _INPUT_CODES.get(value)
    if isinstance(value, bool) or signature not in _INPUT_CODES.values():
        raise NotImplementedError(
            "SolverMuJoCo supports one DC-motor control input per actuator row: "
            "'voltage', 'pos', or 'vel'. MuJoCo 'ff', 'none', and combined input "
            "signatures are not supported by Newton's one-control-per-actuator mapping."
        )
    return signature


def _import_mujoco():
    """Load the native shortcut resolver only when a DC motor is authored/imported."""
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "Authoring or importing <dcmotor> actuators requires MuJoCo. Install it with `uv pip install 'newton[sim]'`."
        ) from exc
    return mujoco


def add_actuator_dcmotor(
    builder: ModelBuilder,
    target: ActuatorTarget,
    *,
    motorconst: Sequence[float] = (0.0,),
    resistance: float = 0.0,
    nominal: Sequence[float] = (0.0,),
    saturation: Sequence[float] = (0.0,),
    inductance: Sequence[float] = (0.0,),
    cogging: Sequence[float] = (0.0,),
    controller: Sequence[float] = (0.0,),
    thermal: Sequence[float] = (0.0,),
    lugre: Sequence[float] = (0.0,),
    input: str | int = "voltage",
    gear: Sequence[float] = (1.0,),
    damping: Sequence[float] = (0.0,),
    armature: float = 0.0,
    ctrlrange: Sequence[float] | None = None,
    ctrllimited: bool | int | str = "auto",
    cranklength: float | None = None,
    ctrl: float = 0.0,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a native DC motor controlled through ``control.mujoco.ctrl``.

    Requires the optional ``mujoco`` package (included in ``newton[sim]``),
    but not MuJoCo-Warp. Native MuJoCo resolves the electrical and controller
    parameters; these are construction-time properties, not runtime-editable
    general-actuator gains. No model is compiled by this helper. Parameter
    vectors are zero-padded before native default inference, as in MJCF.
    Force limits are derived from ``saturation``, not separate
    ``forcerange``/``forcelimited`` options, matching the native shortcut.

    Parameter ordering and inference follow the
    `MuJoCo DC-motor reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-dcmotor>`_.

    Args:
        builder: Model builder receiving the actuator.
        target: Typed actuator transmission target.
        motorconst: Torque constant [N·m/A] and back-EMF constant [V·s/rad].
        resistance: Winding resistance [ohm]. Supply sufficient nonzero
            ``motorconst``, ``resistance``, or ``nominal`` values for native inference.
        nominal: Voltage [V], stall torque [N·m], and no-load speed [rad/s].
        saturation: Torque [N·m], current [A], and current-rate [A/s] limits.
        inductance: Winding inductance [H] and electrical time constant [s].
        cogging: Torque amplitude [N·m], pole count, and phase [rad].
        controller: ``kp``, ``ki``, ``kd``, slew limit, integral limit, and voltage
            limit; see the linked native reference for input-dependent units.
        thermal: Thermal resistance [K/W], capacitance [J/K], time constant [s],
            resistance temperature coefficient [1/K], reference and ambient
            temperatures [degrees Celsius].
        lugre: Bristle stiffness [N·m/rad], damping [N·m·s/rad], Coulomb and static
            friction torques [N·m], and Stribeck speed [rad/s].
        input: Single control input: ``voltage`` (8), ``pos`` (1), or ``vel`` (2).
            Accepts a native keyword or its integer control signature.
            Other signatures raise :class:`NotImplementedError`.
        gear: Native transmission gear, padded to six values. Ball/free joints
            require explicit native gear as described in :class:`ActuatorTarget`.
            With nonzero damping on the Warp backend, gear must match across
            replicated worlds and cannot change after solver construction.
        damping: Coefficients of ``-d0*v - d1*abs(v)*v - d2*v**3`` in actuator
            velocity ``v``. Coefficient ``dn`` has units [N·m·(s/rad)^(n+1)]
            for angular transmissions or [N·(s/m)^(n+1)] for linear ones.
        armature: Passive rotor inertia [kg·m² or kg for linear transmissions].
            Currently supported only by native MuJoCo.
        ctrlrange: Optional two-value control range: voltage [V], position
            [m or rad], or velocity [m/s or rad/s], in actuator coordinates.
        ctrllimited: Control-limit tri-state (false, true, or auto).
        cranklength: Slider-crank length [m]. Required for slider-crank targets.
        ctrl: Initial control value, with the same units as ``ctrlrange``.
        custom_attributes: Additional registered ``mujoco:actuator`` attributes.

    Returns:
        The ``mujoco:actuator`` row index.
    """
    from .solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    signature = parse_input(input)
    parameters = {}
    for name, values, size in (
        ("motorconst", motorconst, 2),
        ("nominal", nominal, 3),
        ("saturation", saturation, 3),
        ("inductance", inductance, 2),
        ("cogging", cogging, 3),
        ("controller", controller, 6),
        ("thermal", thermal, 6),
        ("lugre", lugre, 5),
    ):
        if not 1 <= len(values) <= size:
            raise ValueError(f"{name} requires one to {size} values, got {len(values)}.")
        parameters[name] = [float(value) for value in values] + [0.0] * (size - len(values))
    if not 1 <= len(damping) <= 3:
        raise ValueError(f"damping requires one to 3 values, got {len(damping)}.")
    padded_damping = [float(value) for value in damping] + [0.0] * (3 - len(damping))
    spec = _import_mujoco().MjSpec()
    actuator = spec.add_actuator()
    actuator.damping = padded_damping
    actuator.set_to_dcmotor(resistance=resistance, ctrlspec=signature, **parameters)
    return _add_actuator(
        builder,
        target,
        ctrl_type=int(SolverMuJoCo.CtrlType.DCMOTOR),
        dyntype=int(actuator.dyntype),
        gaintype=int(actuator.gaintype),
        biastype=int(actuator.biastype),
        dynprm=actuator.dynprm,
        gainprm=actuator.gainprm,
        biasprm=actuator.biasprm,
        actdim=actuator.actdim,
        actearly=actuator.actearly,
        forcerange=actuator.forcerange,
        forcelimited=int(actuator.forcelimited),
        ctrlspec=signature,
        gear=gear,
        damping=damping,
        armature=armature,
        ctrlrange=ctrlrange,
        ctrllimited=ctrllimited,
        cranklength=cranklength,
        ctrl=ctrl,
        custom_attributes=custom_attributes,
    )


def resolve_mjcf_parameters(root: ET.Element) -> dict[ET.Element, dict[str, str]]:
    """Resolve shortcuts and inherited defaults into attributes for the MJCF importer."""
    motors = root.findall("actuator/dcmotor")
    if not motors:
        return {}
    mujoco = _import_mujoco()

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


def validate_worlds(model: Model, rows: np.ndarray, *, dcmotor_type: int) -> None:
    """Require matching shortcut layouts and construction parameters across worlds."""
    attributes = model.mujoco
    type_attribute = getattr(attributes, "ctrl_type", None)
    if type_attribute is None:
        return
    dcmotor_rows = type_attribute.numpy()[rows] == dcmotor_type
    if not np.any(dcmotor_rows):
        return
    _require_matching_worlds(dcmotor_rows, "matching high-level DC-motor actuator layouts")
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


__all__ = ["add_actuator_dcmotor"]
