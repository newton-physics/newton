# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from newton._src.usd.utils import _resolve_asset_path, get_applied_api_schemas

from .clamping import ClampingBase, ClampingDCMotor, ClampingMaxEffort, ClampingPositionBased
from .delay import Delay
from .drives import DriveBase, DriveNeuralLSTM, DriveNeuralMLP, DrivePD, DrivePID
from .utils import load_metadata

_DEPRECATED_UNSET = object()
_COMPONENT_KIND_CONTROLLER_DEPRECATION_MSG = (
    "ComponentKind.CONTROLLER is deprecated in Newton 1.6; use ComponentKind.DRIVE instead."
)
_PARSED_CONTROLLER_CLASS_DEPRECATION_MSG = (
    "ActuatorParsed.controller_class is deprecated in Newton 1.6; use drive_class instead."
)
_PARSED_CONTROLLER_KWARGS_DEPRECATION_MSG = (
    "ActuatorParsed.controller_kwargs is deprecated in Newton 1.6; use drive_kwargs instead."
)


class _ComponentKindMeta(enum.EnumMeta):
    """Provide a warning-producing compatibility member for ``CONTROLLER``."""

    def __getattr__(cls, name: str):
        if name == "CONTROLLER":
            warnings.warn(_COMPONENT_KIND_CONTROLLER_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            return cls.DRIVE
        return super().__getattr__(name)

    def __getitem__(cls, name: str):
        if name == "CONTROLLER":
            warnings.warn(_COMPONENT_KIND_CONTROLLER_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            return cls.DRIVE
        return super().__getitem__(name)


class ComponentKind(enum.Enum, metaclass=_ComponentKindMeta):
    """Classification of actuator component schemas."""

    DRIVE = "drive"
    CLAMPING = "clamping"
    DELAY = "delay"

    @classmethod
    def _missing_(cls, value: object):
        if value == "controller":
            warnings.warn(_COMPONENT_KIND_CONTROLLER_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            return cls.DRIVE
        return None


@dataclass(init=False)
class ActuatorParsed:
    """Result of parsing a USD actuator prim.

    Each detected API schema produces a (class, kwargs) entry.
    The drive is separated out; everything else goes into
    component_specs (delay, clamping, etc.).
    """

    drive_class: type[DriveBase]
    drive_kwargs: dict[str, Any] = field(default_factory=dict)
    component_specs: list[tuple[type[ClampingBase | Delay], dict[str, Any]]] = field(default_factory=list)
    target_path: str = ""
    """Joint target path (USD prim path of the driven joint)."""

    def __init__(
        self,
        drive_class: type[DriveBase] | object = _DEPRECATED_UNSET,
        drive_kwargs: dict[str, Any] | object = _DEPRECATED_UNSET,
        component_specs: list[tuple[type[ClampingBase | Delay], dict[str, Any]]] | None = None,
        target_path: str = "",
        *,
        controller_class: type[DriveBase] | object = _DEPRECATED_UNSET,
        controller_kwargs: dict[str, Any] | object = _DEPRECATED_UNSET,
    ) -> None:
        """Initialize a parsed actuator specification.

        Args:
            drive_class: Parsed drive class.
            drive_kwargs: Parsed drive constructor arguments.
            component_specs: Parsed delay and clamping specifications.
            target_path: USD prim path of the driven joint.
            controller_class: Deprecated in Newton 1.6; use ``drive_class``.
            controller_kwargs: Deprecated in Newton 1.6; use ``drive_kwargs``.
        """
        if controller_class is not _DEPRECATED_UNSET and drive_class is not _DEPRECATED_UNSET:
            raise TypeError("Specify only one of 'drive_class' and deprecated 'controller_class'.")
        if controller_kwargs is not _DEPRECATED_UNSET and drive_kwargs is not _DEPRECATED_UNSET:
            raise TypeError("Specify only one of 'drive_kwargs' and deprecated 'controller_kwargs'.")

        if controller_class is not _DEPRECATED_UNSET:
            warnings.warn(_PARSED_CONTROLLER_CLASS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            drive_class = controller_class
        if controller_kwargs is not _DEPRECATED_UNSET:
            warnings.warn(_PARSED_CONTROLLER_KWARGS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
            drive_kwargs = controller_kwargs
        if drive_class is _DEPRECATED_UNSET:
            raise TypeError("ActuatorParsed() missing required argument: 'drive_class'")

        self.drive_class = drive_class
        self.drive_kwargs = {} if drive_kwargs is _DEPRECATED_UNSET else drive_kwargs
        self.component_specs = [] if component_specs is None else component_specs
        self.target_path = target_path

    @property
    def controller_class(self) -> type[DriveBase]:
        """Deprecated alias for :attr:`drive_class`.

        .. deprecated:: 1.6
            Use :attr:`drive_class` instead.
        """
        warnings.warn(_PARSED_CONTROLLER_CLASS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.drive_class

    @controller_class.setter
    def controller_class(self, value: type[DriveBase]) -> None:
        warnings.warn(_PARSED_CONTROLLER_CLASS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.drive_class = value

    @property
    def controller_kwargs(self) -> dict[str, Any]:
        """Deprecated alias for :attr:`drive_kwargs`.

        .. deprecated:: 1.6
            Use :attr:`drive_kwargs` instead.
        """
        warnings.warn(_PARSED_CONTROLLER_KWARGS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.drive_kwargs

    @controller_kwargs.setter
    def controller_kwargs(self, value: dict[str, Any]) -> None:
        warnings.warn(_PARSED_CONTROLLER_KWARGS_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self.drive_kwargs = value


_CAMEL_RE = re.compile(r"(?<=[a-z0-9])([A-Z])")


def _camel_to_snake(name: str) -> str:
    """Convert a camelCase name to snake_case."""
    return _CAMEL_RE.sub(r"_\1", name).lower()


def _read_schema_attrs(prim, schema_name: str) -> dict[str, Any]:
    """Return authored ``newton:`` attributes for *schema_name* as snake_case kwargs.

    Filters to properties defined by *schema_name* when the schema is registered.
    Falls back to all authored ``newton:`` attributes when the plugin is not loaded.

    Returns:
        Authored attribute values keyed by snake_case name; unset attributes omitted.
    """
    from pxr import Sdf, Usd

    defn = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema_name)
    schema_props = set(defn.GetPropertyNames()) if defn is not None else None

    kwargs: dict[str, Any] = {}
    for prop in prim.GetAuthoredPropertiesInNamespace("newton"):
        if not isinstance(prop, Usd.Attribute):
            continue
        if schema_props is not None and prop.GetName() not in schema_props:
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
    returns the concrete class (e.g. for neural drives that pick
    MLP vs LSTM at parse time).  The callable may also validate kwargs
    and raise :class:`ValueError`.
    """
    kind: ComponentKind


_NEURAL_DRIVE_TYPES: dict[str, type[DriveBase]] = {
    "mlp": DriveNeuralMLP,
    "lstm": DriveNeuralLSTM,
}


def _resolve_neural_drive(kwargs: dict[str, Any]) -> type[DriveBase]:
    """Validate neural-control kwargs and return the concrete drive class.

    Inspects the checkpoint's ``model_type`` metadata to choose between
    :class:`DriveNeuralMLP` and :class:`DriveNeuralLSTM`.

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
            f"expected one of {sorted(_NEURAL_DRIVE_TYPES)}"
        )
    resolved_cls = _NEURAL_DRIVE_TYPES.get(model_type)
    if resolved_cls is None:
        raise ValueError(
            f"Unsupported model_type '{model_type}' in checkpoint metadata "
            f"at '{model_path}'; expected one of {sorted(_NEURAL_DRIVE_TYPES)}"
        )
    return resolved_cls


def _get_relationship_targets(prim, name: str) -> list[str]:
    """Get relationship target paths from a USD prim."""
    rel = prim.GetRelationship(name)
    if not rel:
        return []
    return [str(t) for t in rel.GetTargets()]


class SchemaNames:
    """Canonical USD tokens from ``newton-usd-schemas``"""

    ACTUATOR = "NewtonActuator"

    PD_CONTROL = "NewtonPDControlAPI"
    PID_CONTROL = "NewtonPIDControlAPI"
    NEURAL_CONTROL = "NewtonNeuralControlAPI"

    MAX_EFFORT_CLAMPING = "NewtonMaxEffortClampingAPI"
    DC_MOTOR_CLAMPING = "NewtonDCMotorClampingAPI"
    POSITION_BASED_CLAMPING = "NewtonPositionBasedClampingAPI"

    DELAY = "NewtonActuatorDelayAPI"


_SCHEMA_REGISTRY: dict[str, _SchemaEntry] = {
    SchemaNames.PD_CONTROL: _SchemaEntry(DrivePD, ComponentKind.DRIVE),
    SchemaNames.PID_CONTROL: _SchemaEntry(DrivePID, ComponentKind.DRIVE),
    SchemaNames.NEURAL_CONTROL: _SchemaEntry(_resolve_neural_drive, ComponentKind.DRIVE),
    SchemaNames.MAX_EFFORT_CLAMPING: _SchemaEntry(ClampingMaxEffort, ComponentKind.CLAMPING),
    SchemaNames.DC_MOTOR_CLAMPING: _SchemaEntry(ClampingDCMotor, ComponentKind.CLAMPING),
    SchemaNames.POSITION_BASED_CLAMPING: _SchemaEntry(ClampingPositionBased, ComponentKind.CLAMPING),
    SchemaNames.DELAY: _SchemaEntry(Delay, ComponentKind.DELAY),
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
        kind: Whether this schema is a drive, clamping, delay, etc.

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
            - has multiple drive schemas applied,
            - has no drive schema, or
            - has a ``NewtonNeuralControlAPI`` with an unsupported model.
    """
    if prim.GetTypeName() != SchemaNames.ACTUATOR:
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

    drive_class = None
    drive_kwargs: dict[str, Any] = {}
    component_specs: list[tuple[type, dict[str, Any]]] = []
    detected: list[str] = []

    for schema_name in get_applied_api_schemas(prim):
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

        if entry.kind is ComponentKind.DRIVE:
            if drive_class is not None:
                raise ValueError(
                    f"Actuator prim '{prim.GetPath()}' has multiple drives: {drive_class.__name__} and {cls.__name__}"
                )
            drive_class = cls
            drive_kwargs = kwargs
        else:
            component_specs.append((cls, kwargs))

    if drive_class is None:
        raise ValueError(f"Actuator prim '{prim.GetPath()}' has no drive schema (detected schemas: {detected})")

    return ActuatorParsed(
        drive_class=drive_class,
        drive_kwargs=drive_kwargs,
        component_specs=component_specs,
        target_path=target_paths[0],
    )
