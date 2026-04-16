# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .clamping import Clamping, ClampingDCMotor, ClampingMaxForce
from .controllers import Controller, ControllerNetLSTM, ControllerNetMLP, ControllerPD, ControllerPID
from .delay import Delay


@dataclass
class SchemaEntry:
    """Maps an API schema to a component class and its USD→kwarg param names."""

    component_class: type
    param_map: dict[str, str]
    is_controller: bool = False
    validate: Callable[[dict[str, Any]], None] | None = None


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
    # Neural-network controllers
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
    target_path: str = ""


def get_attribute(prim, name: str, default: Any = None) -> Any:
    """Get attribute value from a USD prim, returning default if not found."""
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return default
    return attr.Get()


def get_relationship_target(prim, name: str) -> str | None:
    """Get the single relationship target path from a USD prim."""
    rel = prim.GetRelationship(name)
    if not rel:
        return None
    targets = rel.GetTargets()
    if len(targets) != 1:
        return None
    return str(targets[0])


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

    target_path = get_relationship_target(prim, "newton:actuator:target")
    if target_path is None:
        return None

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
        target_path=target_path,
    )
