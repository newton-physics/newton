# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .clamping import Clamping, ClampingDCMotor, ClampingMaxEffort, ClampingPositionBased
from .controllers import Controller, ControllerNetLSTM, ControllerNetMLP, ControllerPD, ControllerPID
from .delay import Delay


@dataclass
class SchemaEntry:
    """Maps an API schema to a component class and its USD-to-kwarg param names."""

    component_class: type
    param_map: dict[str, str]
    is_controller: bool = False
    validate: Callable[[dict[str, Any]], None] | None = None


def _validate_neural_control(kwargs: dict[str, Any]) -> None:
    model_path = kwargs.get("model_path")
    if not model_path:
        raise ValueError("NewtonNeuralControlAPI requires a non-empty newton:modelPath attribute")


_NEURAL_CONTROLLER_TYPES: dict[str, type[Controller]] = {
    "mlp": ControllerNetMLP,
    "lstm": ControllerNetLSTM,
}


def _resolve_neural_controller(kwargs: dict[str, Any]) -> tuple[type[Controller], dict[str, Any]]:
    """Read checkpoint metadata and dispatch to MLP or LSTM runtime.

    The checkpoint must contain ``"model_type"`` in its metadata
    (``"mlp"`` or ``"lstm"``).  Only the metadata is loaded here;
    the full model is loaded later by the controller constructor.

    Raises:
        ValueError: If ``model_type`` is missing or not recognised.
    """
    from .utils import load_metadata

    model_path = kwargs["model_path"]
    metadata = load_metadata(model_path)

    model_type = metadata.get("model_type")
    if model_type is None:
        raise ValueError(
            f"Checkpoint at '{model_path}' is missing 'model_type' in metadata; "
            f"expected one of {sorted(_NEURAL_CONTROLLER_TYPES)}"
        )
    cls = _NEURAL_CONTROLLER_TYPES.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unsupported model_type '{model_type}' in checkpoint metadata "
            f"at '{model_path}'; expected one of {sorted(_NEURAL_CONTROLLER_TYPES)}"
        )
    return cls, {"model_path": model_path}


# ---------------------------------------------------------------------------
# Schema registry
#
# Maps USD API schema tokens to runtime component classes.
# Kept as a "fake" registry until the newton-usd-schemas PR merges and
# proper USD schema type checks become available.
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[str, SchemaEntry] = {
    # ── Controllers ────────────────────────────────────────────────────
    "NewtonPDControlAPI": SchemaEntry(
        component_class=ControllerPD,
        param_map={"constEffort": "const_effort", "kp": "kp", "kd": "kd"},
        is_controller=True,
    ),
    "NewtonPIDControlAPI": SchemaEntry(
        component_class=ControllerPID,
        param_map={
            "constEffort": "const_effort",
            "kp": "kp",
            "ki": "ki",
            "kd": "kd",
            "integralMax": "integral_max",
        },
        is_controller=True,
    ),
    "NewtonNeuralControlAPI": SchemaEntry(
        component_class=ControllerNetMLP,
        param_map={"modelPath": "model_path"},
        is_controller=True,
        validate=_validate_neural_control,
    ),
    # ── Clamping ───────────────────────────────────────────────────────
    "NewtonMaxEffortClampingAPI": SchemaEntry(
        component_class=ClampingMaxEffort,
        param_map={"maxEffort": "max_effort"},
    ),
    "NewtonDCMotorClampingAPI": SchemaEntry(
        component_class=ClampingDCMotor,
        param_map={
            "saturationEffort": "saturation_effort",
            "velocityLimit": "velocity_limit",
            "maxMotorEffort": "max_motor_effort",
        },
    ),
    "NewtonPositionBasedClampingAPI": SchemaEntry(
        component_class=ClampingPositionBased,
        param_map={
            "lookupPositions": "lookup_positions",
            "lookupEfforts": "lookup_efforts",
        },
    ),
    # ── Delay ──────────────────────────────────────────────────────────
    "NewtonActuatorDelayAPI": SchemaEntry(
        component_class=Delay,
        param_map={"delaySteps": "delay"},
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
    """Joint target path (USD prim path of the driven joint)."""


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
    schemas (e.g. ``NewtonPDControlAPI``) are not registered USD schema types
    and therefore are not returned by ``GetAppliedSchemas()``.
    """
    meta = prim.GetMetadata("apiSchemas")
    if meta is None:
        return []
    try:
        tokens = list(meta.GetAddedOrExplicitItems())
    except AttributeError:
        tokens = list(meta)
    return [s for s in tokens if s in SCHEMA_REGISTRY]


def parse_actuator_prim(prim) -> ActuatorParsed | None:
    """Parse a USD Actuator prim into a composed actuator specification.

    Each detected schema directly maps to a component class with its
    extracted params. Returns ``None`` if the prim is not a
    ``NewtonActuator``.

    Raises:
        ValueError: If the prim is a ``NewtonActuator`` but:
            - has no authored ``newton:targets`` relationship,
            - has multiple controller schemas applied,
            - has no controller schema, or
            - has a ``NewtonNeuralControlAPI`` with an unsupported model.
    """
    if prim.GetTypeName() != "NewtonActuator":
        return None

    target_paths = get_relationship_targets(prim, "newton:targets")
    if not target_paths:
        raise ValueError(
            f"Actuator prim '{prim.GetPath()}' has no authored 'newton:targets' relationship; "
            f"deactivate the prim instead of leaving the target empty"
        )
    if len(target_paths) > 1:
        warnings.warn(
            f"Actuator prim {prim.GetPath()} has {len(target_paths)} targets; "
            f"only the first is used, additional targets are ignored",
            stacklevel=2,
        )
        target_paths = target_paths[:1]

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
            value = get_attribute(prim, f"newton:{usd_name}")
            if value is not None:
                kwargs[kwarg_name] = value

        if entry.validate is not None:
            try:
                entry.validate(kwargs)
            except ValueError as exc:
                raise ValueError(f"Actuator prim '{prim.GetPath()}': {exc}") from None

        if entry.is_controller:
            if controller_class is not None:
                raise ValueError(
                    f"Actuator prim has multiple controllers: "
                    f"{controller_class.__name__} and {entry.component_class.__name__}"
                )
            if schema_name == "NewtonNeuralControlAPI":
                controller_class, controller_kwargs = _resolve_neural_controller(kwargs)
            else:
                controller_class = entry.component_class
                controller_kwargs = kwargs
        else:
            component_specs.append((entry.component_class, kwargs))

    if controller_class is None:
        raise ValueError(
            f"Actuator prim '{prim.GetPath()}' has no controller schema (matched schemas from metadata: {schemas})"
        )

    return ActuatorParsed(
        controller_class=controller_class,
        controller_kwargs=controller_kwargs,
        component_specs=component_specs,
        target_path=target_paths[0],
    )
