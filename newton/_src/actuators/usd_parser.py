# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .clamping import Clamping, ClampingDCMotor, ClampingMaxForce, ClampingPositionBased
from .controllers import Controller, ControllerNetLSTM, ControllerNetMLP, ControllerPD, ControllerPID
from .delay import Delay


@dataclass
class SchemaEntry:
    """Maps an API schema to a component class and its USD→kwarg param names."""

    component_class: type
    param_map: dict[str, str]
    is_controller: bool = False
    validate: Callable[[dict[str, Any]], None] | None = None


def _validate_clamp_position_based(kwargs: dict[str, Any]) -> None:
    path = kwargs.get("lookup_table_path")
    if not path:
        raise ValueError("NewtonClampingPositionBasedAPI requires a lookupTablePath attribute")
    table_path = Path(path)
    if not table_path.is_file():
        raise ValueError(f"Lookup table file not found: {path}")
    angles, torques = [], []
    for line in table_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        angles.append(float(parts[0]))
        torques.append(float(parts[1]))
    kwargs.pop("lookup_table_path")
    kwargs["lookup_angles"] = angles
    kwargs["lookup_torques"] = torques


def _validate_clamp_velocity_based(kwargs: dict[str, Any]) -> None:
    vel_lim = kwargs.get("velocity_limit")
    if vel_lim is not None and vel_lim <= 0.0:
        raise ValueError(
            f"NewtonClampingDCMotorAPI requires velocity_limit > 0 (division by velocity_limit "
            f"in torque-speed computation); got {vel_lim}"
        )


# Temporary registry until the actual USD schema is merged.
SCHEMA_REGISTRY: dict[str, SchemaEntry] = {
    "NewtonControllerPDAPI": SchemaEntry(
        component_class=ControllerPD,
        param_map={"kp": "kp", "kd": "kd", "constForce": "constant_force"},
        is_controller=True,
    ),
    "NewtonControllerPIDAPI": SchemaEntry(
        component_class=ControllerPID,
        param_map={"kp": "kp", "ki": "ki", "kd": "kd", "integralMax": "integral_max", "constForce": "constant_force"},
        is_controller=True,
    ),
    "NewtonClampingMaxForceAPI": SchemaEntry(
        component_class=ClampingMaxForce,
        param_map={"maxForce": "max_force"},
    ),
    "NewtonDelayAPI": SchemaEntry(
        component_class=Delay,
        param_map={"delay": "delay"},
    ),
    "NewtonClampingDCMotorAPI": SchemaEntry(
        component_class=ClampingDCMotor,
        param_map={"saturationEffort": "saturation_effort", "velocityLimit": "velocity_limit", "maxForce": "max_force"},
        validate=_validate_clamp_velocity_based,
    ),
    # Position-based clamping uses a file path rather than inline USD
    # attributes because lookup tables can have 100+ entries, which would
    # be unwieldy as individual prim attributes.  This mirrors the
    # networkPath convention used by the neural-network controllers.
    "NewtonClampingPositionBasedAPI": SchemaEntry(
        component_class=ClampingPositionBased,
        param_map={"lookupTablePath": "lookup_table_path"},
        validate=_validate_clamp_position_based,
    ),
    # Neural-network controllers
    # input_order / input_idx (MLP) are intentionally left out of the schema;
    # they are framework-specific (e.g. IsaacLab) and I would siggest to set programmatically.
    "NewtonControllerNetMLPAPI": SchemaEntry(
        component_class=ControllerNetMLP,
        param_map={"networkPath": "network_path"},
        is_controller=True,
    ),
    "NewtonControllerNetLSTMAPI": SchemaEntry(
        component_class=ControllerNetLSTM,
        param_map={"networkPath": "network_path"},
        is_controller=True,
    ),
}


@dataclass
class ActuatorParsed:
    """Result of parsing a USD actuator prim.

    Each detected API schema produces a (class, kwargs) entry.
    The controller is separated out; everything else goes into
    component_specs (delay, clamping, etc.).
    """

    controller_class: type[Controller]
    controller_kwargs: dict[str, Any] = field(default_factory=dict)
    component_specs: list[tuple[type[Clamping | Delay], dict[str, Any]]] = field(default_factory=list)
    target_paths: list[str] = field(default_factory=list)
    """Joint targets. Currently only a single target is supported."""


def get_attribute(prim, name: str, default: Any = None) -> Any:
    """Get attribute value from a USD prim, returning default if not found."""
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return default
    return attr.Get()


def get_relationship_targets(prim, name: str) -> list[str]:
    """Get relationship target paths from a USD prim."""
    rel = prim.GetRelationship(name)
    if not rel:
        return []
    return [str(t) for t in rel.GetTargets()]


def get_schemas_from_prim(prim) -> list[str]:
    """Get applied schemas that match the registry.

    Uses prim metadata to find applied schema tokens, since our custom
    schemas (e.g. ``NewtonControllerPDAPI``) are not registered USD schema types
    and therefore are not returned by ``GetAppliedSchemas()``.
    """
    # GetAppliedSchemas() only returns registered USD schema types.
    # Our custom schemas live in the apiSchemas metadata token list.
    # TODO: replace with proper USD schema type checks once the Newton schema is merged.
    meta = prim.GetMetadata("apiSchemas")
    if meta is None:
        return []
    # SdfTokenListOp: use .GetAddedOrExplicitItems() or iterate directly
    try:
        tokens = list(meta.GetAddedOrExplicitItems())
    except AttributeError:
        tokens = list(meta)
    return [s for s in tokens if s in SCHEMA_REGISTRY]


def parse_actuator_prim(prim) -> ActuatorParsed | None:
    """Parse a USD Actuator prim into a composed actuator specification.

    Each detected schema directly maps to a component class with its
    extracted params. Returns ``None`` if the prim is not a
    ``NewtonActuator`` or has no target relationship.

    Raises:
        ValueError: If the prim is a valid actuator but has malformed
            schemas (e.g. multiple controllers or no controller schema).
    """
    if prim.GetTypeName() != "NewtonActuator":
        return None

    target_paths = get_relationship_targets(prim, "newton:actuator:targets")
    if not target_paths:
        return None
    if len(target_paths) != 1:
        raise ValueError(
            f"Actuator prim currently supports exactly one target; got {len(target_paths)}"
        )

    schemas = get_schemas_from_prim(prim)
    controller_class = None
    controller_kwargs: dict[str, Any] = {}
    component_specs: list[tuple[type, dict[str, Any]]] = []

    for schema_name in schemas:
        entry = SCHEMA_REGISTRY.get(schema_name)
        if entry is None:
            continue

        kwargs: dict[str, Any] = {}
        for usd_name, kwarg_name in entry.param_map.items():
            value = get_attribute(prim, f"newton:actuator:{usd_name}")
            if value is not None:
                kwargs[kwarg_name] = value

        if entry.validate is not None:
            entry.validate(kwargs)

        if entry.is_controller:
            if controller_class is not None:
                raise ValueError(
                    f"Actuator prim has multiple controllers: "
                    f"{controller_class.__name__} and {entry.component_class.__name__}"
                )
            controller_class = entry.component_class
            controller_kwargs = kwargs
        else:
            component_specs.append((entry.component_class, kwargs))

    if controller_class is None:
        raise ValueError(
            f"Actuator prim has no controller schema applied (applied schemas: {prim.GetAppliedSchemas()})"
        )

    return ActuatorParsed(
        controller_class=controller_class,
        controller_kwargs=controller_kwargs,
        component_specs=component_specs,
        target_paths=target_paths,
    )
