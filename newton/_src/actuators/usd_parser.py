# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actuators import ActuatorDCMotor, ActuatorDelayedPD, ActuatorPD, ActuatorPID


@dataclass
class ParsedActuator:
    """Result of parsing a USD actuator prim."""

    actuator_class: type
    target_paths: list[str]
    kwargs: dict[str, Any] = field(default_factory=dict)
    transmission: list[float] | None = None


API_SCHEMA_HANDLERS: dict[str, dict[str, str]] = {
    "PDControllerAPI": {
        "kp": "kp",
        "kd": "kd",
        "maxForce": "max_force",
        "constForce": "constant_force",
    },
    "PIDControllerAPI": {
        "kp": "kp",
        "ki": "ki",
        "kd": "kd",
        "maxForce": "max_force",
        "integralMax": "integral_max",
        "constForce": "constant_force",
    },
    "DelayAPI": {
        "delay": "delay",
    },
    "DCMotorAPI": {
        "saturationEffort": "saturation_effort",
        "velocityLimit": "velocity_limit",
    },
}


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


def get_actuator_attribute_names(prim) -> set[str]:
    """Get all newton:actuator attribute names from a prim."""
    names = set()
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        if attr_name.startswith("newton:actuator:"):
            names.add(attr_name.split(":")[-1])
    return names


def infer_schemas_from_prim(prim) -> list[str]:
    """Infer actuator schemas from attribute names."""
    attr_names = get_actuator_attribute_names(prim)
    schemas = []

    if "ki" in attr_names:
        schemas.append("PIDControllerAPI")
    elif "kp" in attr_names or "kd" in attr_names:
        schemas.append("PDControllerAPI")
    if "delay" in attr_names:
        schemas.append("DelayAPI")
    if "saturationEffort" in attr_names and "velocityLimit" in attr_names:
        schemas.append("DCMotorAPI")

    return schemas


def determine_actuator_class(schemas: list[str]) -> type:
    """Determine actuator class from inferred schemas."""
    has_delay = "DelayAPI" in schemas
    has_pid = "PIDControllerAPI" in schemas
    has_pd = "PDControllerAPI" in schemas
    has_dc_motor = "DCMotorAPI" in schemas

    if has_dc_motor and has_pd:
        return ActuatorDCMotor
    elif has_delay and has_pd:
        return ActuatorDelayedPD
    elif has_pid:
        return ActuatorPID
    elif has_pd:
        return ActuatorPD
    else:
        return ActuatorPD


def validate_kwargs(schemas: list[str], kwargs: dict[str, Any]) -> None:
    """Validate extracted kwargs. Raises ValueError on invalid values."""
    if "DCMotorAPI" in schemas:
        vel_lim = kwargs.get("velocity_limit")
        if vel_lim is not None and vel_lim <= 0.0:
            raise ValueError(
                f"DCMotorAPI requires velocity_limit > 0 (division by velocity_limit "
                f"in saturation computation); got {vel_lim}"
            )


def extract_kwargs_from_prim(prim, schemas: list[str]) -> dict[str, Any]:
    """Extract actuator parameters from prim attributes using newton:actuator:{attr} format."""
    kwargs = {}
    for schema_name in schemas:
        if schema_name not in API_SCHEMA_HANDLERS:
            continue
        param_map = API_SCHEMA_HANDLERS[schema_name]
        for usd_name, kwarg_name in param_map.items():
            value = get_attribute(prim, f"newton:actuator:{usd_name}")
            if value is not None:
                kwargs[kwarg_name] = value
    validate_kwargs(schemas, kwargs)
    return kwargs


def parse_actuator_prim(prim) -> ParsedActuator | None:
    """Parse a USD Actuator prim. Returns ``None`` if not a valid actuator."""
    if prim.GetTypeName() != "Actuator":
        return None

    target_paths = get_relationship_targets(prim, "newton:actuator:target")
    if not target_paths:
        return None

    schemas = infer_schemas_from_prim(prim)

    transmission = get_attribute(prim, "newton:actuator:transmission")

    return ParsedActuator(
        actuator_class=determine_actuator_class(schemas),
        target_paths=target_paths,
        kwargs=extract_kwargs_from_prim(prim, schemas),
        transmission=list(transmission) if transmission else None,
    )
