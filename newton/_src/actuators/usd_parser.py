# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from newton._src.usd.utils import _resolve_asset_path

from .clamping import Clamping, ClampingDCMotor, ClampingMaxEffort, ClampingPositionBased
from .controllers import Controller, ControllerNeuralLSTM, ControllerNeuralMLP, ControllerPD, ControllerPID
from .delay import Delay
from .utils import load_metadata


class ComponentKind(enum.Enum):
    """Classification of actuator component schemas."""

    CONTROLLER = "controller"
    CLAMPING = "clamping"
    DELAY = "delay"


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


_CAMEL_RE = re.compile(r"(?<=[a-z0-9])([A-Z])")


def _camel_to_snake(name: str) -> str:
    """Convert a camelCase name to snake_case."""
    return _CAMEL_RE.sub(r"_\1", name).lower()


def _read_schema_attrs(prim, schema_name: str) -> dict[str, Any]:
    """Read all authored ``newton:`` attributes for *schema_name* from *prim*.

    Uses :meth:`pxr.Usd.Prim.GetPropertiesInNamespace` scoped to the
    schema's property list so that only attributes belonging to
    *schema_name* are returned (not attributes from other applied schemas
    that share the ``newton:`` namespace).

    Returns:
        Mapping of snake_case kwarg names to their authored values.
        Attributes without an authored value are omitted.
    """
    from pxr import Usd

    defn = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema_name)
    if defn is None:
        raise RuntimeError(
            f"Schema {schema_name!r} not found in Usd.SchemaRegistry; is newton-usd-schemas >= 0.2.0rc1 installed?"
        )
    schema_props = set(defn.GetPropertyNames())

    from pxr import Sdf

    kwargs: dict[str, Any] = {}
    for prop in prim.GetAuthoredPropertiesInNamespace("newton"):
        if prop.GetName() not in schema_props:
            continue
        if not prop.IsValid() or not prop.HasAuthoredValue():
            continue
        camel = prop.GetName().removeprefix("newton:")
        val = prop.Get()
        if isinstance(val, Sdf.AssetPath):
            val = _resolve_asset_path(val, prim, prop)
        kwargs[_camel_to_snake(camel)] = val
    return kwargs


@dataclass
class _SchemaEntry:
    """Maps a USD API schema to a runtime component class."""

    component_class: type | Callable[[dict[str, Any]], type]
    """Concrete class, or a callable that receives the parsed kwargs and
    returns the concrete class (e.g. for neural controllers that pick
    MLP vs LSTM at parse time).  The callable may also validate kwargs
    and raise :class:`ValueError`.
    """
    kind: ComponentKind


_NEURAL_CONTROLLER_TYPES: dict[str, type[Controller]] = {
    "mlp": ControllerNeuralMLP,
    "lstm": ControllerNeuralLSTM,
}


def _resolve_neural_control(kwargs: dict[str, Any]) -> type[Controller]:
    """Validate neural-control kwargs and return the concrete controller class.

    Inspects the checkpoint's ``model_type`` metadata to choose between
    :class:`ControllerNeuralMLP` and :class:`ControllerNeuralLSTM`.

    Raises:
        ValueError: If ``model_path`` is empty or the checkpoint's
            ``model_type`` metadata is missing / not recognised.
    """
    model_path = kwargs.get("model_path")
    if not model_path:
        raise ValueError("NewtonNeuralControlAPI requires a non-empty newton:modelPath attribute")

    metadata = load_metadata(model_path)

    model_type = metadata.get("model_type")
    if model_type is None:
        raise ValueError(
            f"Checkpoint at '{model_path}' is missing 'model_type' in metadata; "
            f"expected one of {sorted(_NEURAL_CONTROLLER_TYPES)}"
        )
    resolved_cls = _NEURAL_CONTROLLER_TYPES.get(model_type)
    if resolved_cls is None:
        raise ValueError(
            f"Unsupported model_type '{model_type}' in checkpoint metadata "
            f"at '{model_path}'; expected one of {sorted(_NEURAL_CONTROLLER_TYPES)}"
        )
    return resolved_cls


def _get_relationship_targets(prim, name: str) -> list[str]:
    """Get relationship target paths from a USD prim."""
    rel = prim.GetRelationship(name)
    if not rel:
        return []
    return [str(t) for t in rel.GetTargets()]


_SCHEMA_REGISTRY: dict[str, _SchemaEntry] = {
    # ── Controllers ────────────────────────────────────────────────────
    "NewtonPDControlAPI": _SchemaEntry(
        component_class=ControllerPD,
        kind=ComponentKind.CONTROLLER,
    ),
    "NewtonPIDControlAPI": _SchemaEntry(
        component_class=ControllerPID,
        kind=ComponentKind.CONTROLLER,
    ),
    "NewtonNeuralControlAPI": _SchemaEntry(
        component_class=_resolve_neural_control,
        kind=ComponentKind.CONTROLLER,
    ),
    # ── Clamping ───────────────────────────────────────────────────────
    "NewtonMaxEffortClampingAPI": _SchemaEntry(
        component_class=ClampingMaxEffort,
        kind=ComponentKind.CLAMPING,
    ),
    "NewtonDCMotorClampingAPI": _SchemaEntry(
        component_class=ClampingDCMotor,
        kind=ComponentKind.CLAMPING,
    ),
    "NewtonPositionBasedClampingAPI": _SchemaEntry(
        component_class=ClampingPositionBased,
        kind=ComponentKind.CLAMPING,
    ),
    # ── Delay ──────────────────────────────────────────────────────────
    "NewtonActuatorDelayAPI": _SchemaEntry(
        component_class=Delay,
        kind=ComponentKind.DELAY,
    ),
}


def register_actuator_component(
    schema_name: str,
    component_class: type | Callable[[dict[str, Any]], type],
    kind: ComponentKind,
) -> None:
    """Register a USD API schema for actuator parsing.

    Args:
        schema_name: USD API schema token (e.g. ``"MyCustomControlAPI"``).
            Must be registered with :class:`pxr.Usd.SchemaRegistry`.
        component_class: Concrete class, or a callable that receives
            the parsed kwargs dict and returns the concrete class.
            A callable may also validate kwargs and raise
            :class:`ValueError`.
        kind: Whether this schema is a controller, clamping, delay, etc.

    If *schema_name* is already registered, a warning is emitted and the
    existing entry is overwritten.
    """
    if schema_name in _SCHEMA_REGISTRY:
        warnings.warn(
            f"Actuator schema {schema_name!r} is already registered; overwriting",
            stacklevel=2,
        )
    _SCHEMA_REGISTRY[schema_name] = _SchemaEntry(
        component_class=component_class,
        kind=kind,
    )


def parse_actuator_prim(prim) -> ActuatorParsed | None:
    """Parse a USD Actuator prim into a composed actuator specification.

    Each detected schema directly maps to a component class with its
    extracted params. Returns ``None`` if the prim is not a
    ``NewtonActuator``.

    Raises:
        ValueError: If the prim is a ``NewtonActuator`` but:
            - has no authored ``newton:targets`` relationship,
            - the target prim does not exist or is not a
              ``PhysicsRevoluteJoint`` / ``PhysicsPrismaticJoint``,
            - has multiple controller schemas applied,
            - has no controller schema, or
            - has a ``NewtonNeuralControlAPI`` with an unsupported model.
    """
    if prim.GetTypeName() != "NewtonActuator":
        return None

    target_paths = _get_relationship_targets(prim, "newton:targets")
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

    _SUPPORTED_JOINT_TYPES = {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}
    stage = prim.GetStage()
    target_prim = stage.GetPrimAtPath(target_paths[0]) if stage else None
    if target_prim is None or not target_prim.IsValid():
        raise ValueError(
            f"Actuator prim '{prim.GetPath()}' targets '{target_paths[0]}' which does not exist on the stage"
        )
    target_type = target_prim.GetTypeName()
    if target_type not in _SUPPORTED_JOINT_TYPES:
        raise ValueError(
            f"Actuator prim '{prim.GetPath()}' targets '{target_paths[0]}' "
            f"of type '{target_type}'; only {sorted(_SUPPORTED_JOINT_TYPES)} "
            f"are supported"
        )

    controller_class = None
    controller_kwargs: dict[str, Any] = {}
    component_specs: list[tuple[type, dict[str, Any]]] = []
    detected: list[str] = []

    for schema_name in prim.GetAppliedSchemas():
        entry = _SCHEMA_REGISTRY.get(schema_name)
        if entry is None:
            continue
        detected.append(schema_name)

        kwargs = _read_schema_attrs(prim, schema_name)

        if isinstance(entry.component_class, type):
            cls = entry.component_class
        else:
            try:
                cls = entry.component_class(kwargs)
            except ValueError as exc:
                raise ValueError(f"Actuator prim '{prim.GetPath()}': {exc}") from None

        if entry.kind is ComponentKind.CONTROLLER:
            if controller_class is not None:
                raise ValueError(
                    f"Actuator prim '{prim.GetPath()}' has multiple controllers: "
                    f"{controller_class.__name__} and {cls.__name__}"
                )
            controller_class = cls
            controller_kwargs = kwargs
        else:
            component_specs.append((cls, kwargs))

    if controller_class is None:
        raise ValueError(f"Actuator prim '{prim.GetPath()}' has no controller schema (detected schemas: {detected})")

    return ActuatorParsed(
        controller_class=controller_class,
        controller_kwargs=controller_kwargs,
        component_specs=component_specs,
        target_path=target_paths[0],
    )
