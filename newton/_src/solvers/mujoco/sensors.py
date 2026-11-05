# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo sensor storage and programmatic authoring helpers.

MuJoCo sensors live on the model as rows of the ``mujoco:sensor`` custom
frequency. Each row stores the MuJoCo sensor type together with the object and
reference entities it observes. :class:`~newton.solvers.SolverMuJoCo` exports
the rows as native MuJoCo sensors, whose outputs are available in
``sensordata`` of the MuJoCo data.
"""

from __future__ import annotations

from array import array
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any

import warp as wp

from ...geometry import ShapeFlags
from ...sim.enums import JointType
from ...sim.model import Model
from ._authoring import _add_custom_frequency_row, _ensure_mujoco_attributes, _vector

if TYPE_CHECKING:
    from ...sim.builder import ModelBuilder


class SensorType(IntEnum):
    """Supported MuJoCo sensor types.

    Values mirror MuJoCo's ``mjtSensor`` ordinals. Camera-projection, tactile,
    plugin, and user sensors are not supported.
    """

    TOUCH = 0
    """Normal contact force magnitude summed over contacts inside a site volume [N]."""
    ACCELEROMETER = 1
    """Linear acceleration of a site, including gravity, in the site frame [m/s²]."""
    VELOCIMETER = 2
    """Linear velocity of a site in the site frame [m/s]."""
    GYRO = 3
    """Angular velocity of a site in the site frame [rad/s]."""
    FORCE = 4
    """Interaction force between a site's body and its parent, in the site frame [N]."""
    TORQUE = 5
    """Interaction torque between a site's body and its parent, in the site frame [N·m]."""
    MAGNETOMETER = 6
    """Magnetic flux in the site frame."""
    RANGEFINDER = 7
    """Distance along a site's positive z-axis to the nearest geom [m], or -1 if nothing is hit."""
    JOINTPOS = 9
    """Scalar joint position [m or rad]."""
    JOINTVEL = 10
    """Scalar joint velocity [m/s or rad/s]."""
    TENDONPOS = 11
    """Tendon length [m]."""
    TENDONVEL = 12
    """Tendon velocity [m/s]."""
    ACTUATORPOS = 13
    """Actuator transmission length."""
    ACTUATORVEL = 14
    """Actuator transmission velocity."""
    ACTUATORFRC = 15
    """Scalar actuator force."""
    JOINTACTFRC = 16
    """Total actuator force acting on a scalar joint."""
    TENDONACTFRC = 17
    """Total actuator force acting on a tendon."""
    BALLQUAT = 18
    """Ball-joint orientation quaternion in MuJoCo ``wxyz`` order."""
    BALLANGVEL = 19
    """Ball-joint angular velocity [rad/s]."""
    JOINTLIMITPOS = 20
    """Joint limit distance minus margin."""
    JOINTLIMITVEL = 21
    """Joint limit velocity."""
    JOINTLIMITFRC = 22
    """Joint limit force."""
    TENDONLIMITPOS = 23
    """Tendon limit distance minus margin [m]."""
    TENDONLIMITVEL = 24
    """Tendon limit velocity [m/s]."""
    TENDONLIMITFRC = 25
    """Tendon limit force [N]."""
    FRAMEPOS = 26
    """Object position, optionally relative to a reference frame [m]."""
    FRAMEQUAT = 27
    """Object orientation quaternion in MuJoCo ``wxyz`` order, optionally relative to a reference frame."""
    FRAMEXAXIS = 28
    """Object frame x-axis, optionally relative to a reference frame."""
    FRAMEYAXIS = 29
    """Object frame y-axis, optionally relative to a reference frame."""
    FRAMEZAXIS = 30
    """Object frame z-axis, optionally relative to a reference frame."""
    FRAMELINVEL = 31
    """Object linear velocity, optionally relative to a reference frame [m/s]."""
    FRAMEANGVEL = 32
    """Object angular velocity, optionally relative to a reference frame [rad/s]."""
    FRAMELINACC = 33
    """Object linear acceleration [m/s²]."""
    FRAMEANGACC = 34
    """Object angular acceleration [rad/s²]."""
    SUBTREECOM = 35
    """Center of mass of a body subtree [m]."""
    SUBTREELINVEL = 36
    """Linear velocity of a body subtree's center of mass [m/s]."""
    SUBTREEANGMOM = 37
    """Angular momentum of a body subtree about its center of mass [kg·m²/s]."""
    INSIDESITE = 38
    """``1`` if an object frame lies inside a reference site's volume, else ``0``."""
    GEOMDIST = 39
    """Signed distance between two geoms or bodies [m]."""
    GEOMNORMAL = 40
    """Normal of the closest points between two geoms or bodies."""
    GEOMFROMTO = 41
    """Closest points between two geoms or bodies [m]."""
    CONTACT = 42
    """Contact data matching the object and reference filters."""
    E_POTENTIAL = 43
    """Potential energy [J]."""
    E_KINETIC = 44
    """Kinetic energy [J]."""
    CLOCK = 45
    """Simulation time [s]."""


# MuJoCo mjtObj ordinals used by sensor object and reference columns.
class _SensorObjType(IntEnum):
    UNKNOWN = 0
    BODY = 1
    XBODY = 2
    JOINT = 3  # Stored as a Newton joint-DOF index.
    GEOM = 5
    SITE = 6
    TENDON = 18
    ACTUATOR = 19


@dataclass(frozen=True)
class SensorObject:
    """A typed object observed by a MuJoCo sensor.

    Construct objects with :meth:`body`, :meth:`xbody`, :meth:`subtree`,
    :meth:`joint`, :meth:`joint_dof`, :meth:`geom`, :meth:`site`,
    :meth:`tendon`, or :meth:`actuator`. Objects must belong to the builder's
    current world or the global world (``-1``).
    """

    class Kind(Enum):
        """Sensor object kinds supported by :class:`SensorObject`."""

        BODY = "body"
        """A Newton body, observed at its center of mass (inertial frame)."""

        XBODY = "xbody"
        """A Newton body, observed at its body frame."""

        SUBTREE = "subtree"
        """The kinematic subtree rooted at a Newton body; used by contact sensors."""

        JOINT = "joint"
        """A Newton joint, optionally restricted to one local DOF."""

        JOINT_DOF = "joint_dof"
        """The MuJoCo joint containing an absolute Newton joint-DOF index."""

        GEOM = "geom"
        """A Newton collision shape."""

        SITE = "site"
        """A Newton site shape."""

        TENDON = "tendon"
        """A ``mujoco:tendon`` row."""

        ACTUATOR = "actuator"
        """A ``mujoco:actuator`` row."""

    kind: SensorObject.Kind
    """Object kind."""

    index: int
    """Newton entity index. ``-1`` selects the world body for :attr:`Kind.BODY` and :attr:`Kind.XBODY`."""

    dof: int | None = None
    """Local joint DOF, when :attr:`kind` is :attr:`Kind.JOINT`."""

    @classmethod
    def body(cls, body: int) -> SensorObject:
        """Observe a Newton body at its center of mass.

        Args:
            body: Newton body index, or ``-1`` for the world body.

        Returns:
            A typed body object.
        """
        return cls(cls.Kind.BODY, int(body))

    @classmethod
    def xbody(cls, body: int) -> SensorObject:
        """Observe a Newton body at its body frame.

        Args:
            body: Newton body index, or ``-1`` for the world body.

        Returns:
            A typed body-frame object.
        """
        return cls(cls.Kind.XBODY, int(body))

    @classmethod
    def subtree(cls, body: int) -> SensorObject:
        """Select the kinematic subtree rooted at a Newton body.

        Args:
            body: Newton body index of the subtree root.

        Returns:
            A typed subtree object.
        """
        return cls(cls.Kind.SUBTREE, int(body))

    @classmethod
    def joint(cls, joint: int, *, dof: int | None = None) -> SensorObject:
        """Observe a Newton joint, selecting a scalar axis for multi-DOF joints.

        Args:
            joint: Newton joint index.
            dof: Local DOF within a D6 joint. Required for D6 joints with more
                than one DOF. Omit for scalar, ball, and free joints.

        Returns:
            A typed joint object.
        """
        return cls(cls.Kind.JOINT, int(joint), None if dof is None else int(dof))

    @classmethod
    def joint_dof(cls, dof: int) -> SensorObject:
        """Observe the MuJoCo joint containing an absolute Newton joint-DOF index.

        Args:
            dof: Absolute Newton joint-DOF index.

        Returns:
            A typed joint-DOF object.
        """
        return cls(cls.Kind.JOINT_DOF, int(dof))

    @classmethod
    def geom(cls, shape: int) -> SensorObject:
        """Observe a Newton collision shape.

        Args:
            shape: Newton shape index.

        Returns:
            A typed geom object.
        """
        return cls(cls.Kind.GEOM, int(shape))

    @classmethod
    def site(cls, site: int) -> SensorObject:
        """Observe a Newton site shape.

        Args:
            site: Newton site-shape index.

        Returns:
            A typed site object.
        """
        return cls(cls.Kind.SITE, int(site))

    @classmethod
    def tendon(cls, tendon: int) -> SensorObject:
        """Observe a ``mujoco:tendon`` row.

        Args:
            tendon: Row index returned by :func:`add_tendon_fixed` or :func:`add_tendon_spatial`.

        Returns:
            A typed tendon object.
        """
        return cls(cls.Kind.TENDON, int(tendon))

    @classmethod
    def actuator(cls, actuator: int) -> SensorObject:
        """Observe a ``mujoco:actuator`` row.

        Args:
            actuator: Row index returned by one of the actuator helpers.

        Returns:
            A typed actuator object.
        """
        return cls(cls.Kind.ACTUATOR, int(actuator))


_OBJTYPE_BY_KIND = {
    SensorObject.Kind.BODY: _SensorObjType.BODY,
    SensorObject.Kind.XBODY: _SensorObjType.XBODY,
    SensorObject.Kind.SUBTREE: _SensorObjType.XBODY,
    SensorObject.Kind.JOINT: _SensorObjType.JOINT,
    SensorObject.Kind.JOINT_DOF: _SensorObjType.JOINT,
    SensorObject.Kind.GEOM: _SensorObjType.GEOM,
    SensorObject.Kind.SITE: _SensorObjType.SITE,
    SensorObject.Kind.TENDON: _SensorObjType.TENDON,
    SensorObject.Kind.ACTUATOR: _SensorObjType.ACTUATOR,
}

_K = SensorObject.Kind
_FRAME_KINDS = frozenset((_K.BODY, _K.XBODY, _K.GEOM, _K.SITE))
_DISTANCE_KINDS = frozenset((_K.BODY, _K.GEOM))
_CONTACT_OBJ_KINDS = frozenset((_K.BODY, _K.SUBTREE, _K.GEOM, _K.SITE))
_CONTACT_REF_KINDS = frozenset((_K.BODY, _K.SUBTREE, _K.GEOM))
_NONE: frozenset[SensorObject.Kind] = frozenset()

# (allowed object kinds, object required, allowed reference kinds, reference required)
_SENSOR_SIGNATURES: dict[SensorType, tuple[frozenset, bool, frozenset, bool]] = {}
for _t in (
    SensorType.TOUCH,
    SensorType.ACCELEROMETER,
    SensorType.VELOCIMETER,
    SensorType.GYRO,
    SensorType.FORCE,
    SensorType.TORQUE,
    SensorType.MAGNETOMETER,
    SensorType.RANGEFINDER,
):
    _SENSOR_SIGNATURES[_t] = (frozenset((_K.SITE,)), True, _NONE, False)
for _t in (
    SensorType.JOINTPOS,
    SensorType.JOINTVEL,
    SensorType.JOINTACTFRC,
    SensorType.BALLQUAT,
    SensorType.BALLANGVEL,
    SensorType.JOINTLIMITPOS,
    SensorType.JOINTLIMITVEL,
    SensorType.JOINTLIMITFRC,
):
    _SENSOR_SIGNATURES[_t] = (frozenset((_K.JOINT, _K.JOINT_DOF)), True, _NONE, False)
for _t in (
    SensorType.TENDONPOS,
    SensorType.TENDONVEL,
    SensorType.TENDONACTFRC,
    SensorType.TENDONLIMITPOS,
    SensorType.TENDONLIMITVEL,
    SensorType.TENDONLIMITFRC,
):
    _SENSOR_SIGNATURES[_t] = (frozenset((_K.TENDON,)), True, _NONE, False)
for _t in (SensorType.ACTUATORPOS, SensorType.ACTUATORVEL, SensorType.ACTUATORFRC):
    _SENSOR_SIGNATURES[_t] = (frozenset((_K.ACTUATOR,)), True, _NONE, False)
for _t in (
    SensorType.FRAMEPOS,
    SensorType.FRAMEQUAT,
    SensorType.FRAMEXAXIS,
    SensorType.FRAMEYAXIS,
    SensorType.FRAMEZAXIS,
    SensorType.FRAMELINVEL,
    SensorType.FRAMEANGVEL,
):
    _SENSOR_SIGNATURES[_t] = (_FRAME_KINDS, True, _FRAME_KINDS, False)
for _t in (SensorType.FRAMELINACC, SensorType.FRAMEANGACC):
    _SENSOR_SIGNATURES[_t] = (_FRAME_KINDS, True, _NONE, False)
for _t in (SensorType.SUBTREECOM, SensorType.SUBTREELINVEL, SensorType.SUBTREEANGMOM):
    _SENSOR_SIGNATURES[_t] = (frozenset((_K.BODY,)), True, _NONE, False)
_SENSOR_SIGNATURES[SensorType.INSIDESITE] = (_FRAME_KINDS, True, frozenset((_K.SITE,)), True)
for _t in (SensorType.GEOMDIST, SensorType.GEOMNORMAL, SensorType.GEOMFROMTO):
    _SENSOR_SIGNATURES[_t] = (_DISTANCE_KINDS, True, _DISTANCE_KINDS, True)
_SENSOR_SIGNATURES[SensorType.CONTACT] = (_CONTACT_OBJ_KINDS, False, _CONTACT_REF_KINDS, False)
for _t in (SensorType.E_POTENTIAL, SensorType.E_KINETIC, SensorType.CLOCK):
    _SENSOR_SIGNATURES[_t] = (_NONE, False, _NONE, False)
del _t

# MuJoCo's default sensor integer parameters: rangefinder data="dist" and
# contact data="found", num=1.
_DEFAULT_INTPRM = {
    SensorType.RANGEFINDER: (1, 0, 0),
    SensorType.CONTACT: (1, 0, 1),
}

_SENSOR_INTERP = {"zoh": 0, "linear": 1, "cubic": 2}


def _remap_sensor_ids(values: list[Any], context: dict[str, Any], type_key: str) -> list[Any]:
    """Offset heterogeneous sensor object indices during builder composition."""
    objtype_attr = context["builder"].custom_attributes.get(type_key)
    if not values or objtype_attr is None:
        return values

    entity_offsets = context["entity_offsets"]
    frequency_offsets = context["custom_frequency_offsets"]
    offsets = {
        int(_SensorObjType.BODY): entity_offsets["body"],
        int(_SensorObjType.XBODY): entity_offsets["body"],
        int(_SensorObjType.JOINT): entity_offsets["joint_dof"],
        int(_SensorObjType.GEOM): entity_offsets["shape"],
        int(_SensorObjType.SITE): entity_offsets["shape"],
        int(_SensorObjType.TENDON): frequency_offsets.get("mujoco:tendon", 0),
        int(_SensorObjType.ACTUATOR): frequency_offsets.get("mujoco:actuator", 0),
    }

    authored_types = objtype_attr.values or []
    is_sparse = isinstance(authored_types, dict)
    authored_count = len(authored_types)
    default_type = objtype_attr.default
    ids = array("i", (-1 if value is None else int(value) for value in values))
    for i, row in enumerate(context["row_indices"]):
        if ids[i] < 0:
            continue
        if is_sparse:
            objtype = authored_types.get(row)
        else:
            objtype = authored_types[row] if row < authored_count else None
        ids[i] += offsets.get(int(default_type if objtype is None else objtype), 0)
    return [None if value is None else ids[i] for i, value in enumerate(values)]


def _remap_sensor_objid(values: list[Any], context: dict[str, Any]) -> list[Any]:
    """Remap ``mujoco:sensor_objid`` according to ``mujoco:sensor_objtype``."""
    return _remap_sensor_ids(values, context, "mujoco:sensor_objtype")


def _remap_sensor_refid(values: list[Any], context: dict[str, Any]) -> list[Any]:
    """Remap ``mujoco:sensor_refid`` according to ``mujoco:sensor_reftype``."""
    return _remap_sensor_ids(values, context, "mujoco:sensor_reftype")


def _parse_sensor_interp(value: Any, _context: dict[str, Any] | None = None) -> int:
    """Parse the MJCF ``interp`` sensor attribute."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _SENSOR_INTERP:
            return _SENSOR_INTERP[key]
        value = int(key)
    result = int(value)
    if result not in _SENSOR_INTERP.values():
        raise ValueError(f"Expected sensor interp zoh, linear, or cubic, got {value!r}.")
    return result


def _parse_sensor_user(value: Any, _context: dict[str, Any] | None = None) -> str:
    """Normalize sensor user data to space-separated numbers."""
    if isinstance(value, str):
        value = value.split()
    return " ".join(repr(float(component)) for component in value)


def _register_sensor_attributes(builder: ModelBuilder) -> None:
    """Declare the ``mujoco:sensor`` frequency and its custom attributes on ``builder``.

    Idempotent: re-registration with the same spec is a no-op.
    """
    ca = type(builder).CustomAttribute
    freq = "mujoco:sensor"
    model_assignment = Model.AttributeAssignment.MODEL

    builder.add_custom_frequency(type(builder).CustomFrequency(name="sensor", namespace="mujoco"))

    def add(name: str, dtype: Any, default: Any, **kwargs: Any) -> None:
        builder.add_custom_attribute(
            ca(
                name=name,
                frequency=freq,
                assignment=model_assignment,
                dtype=dtype,
                default=default,
                namespace="mujoco",
                **kwargs,
            )
        )

    add("sensor_world", wp.int32, 0, references="world")
    add("sensor_label", str, "", mjcf_attribute_name="name")
    add("sensor_type", wp.int32, int(SensorType.TOUCH))
    add("sensor_objtype", wp.int32, int(_SensorObjType.UNKNOWN))
    add("sensor_objid", wp.int32, -1, reference_value_transformer=_remap_sensor_objid)
    add("sensor_reftype", wp.int32, int(_SensorObjType.UNKNOWN))
    add("sensor_refid", wp.int32, -1, reference_value_transformer=_remap_sensor_refid)
    add("sensor_intprm", wp.vec3i, wp.vec3i(0, 0, 0))
    add("sensor_noise", wp.float32, 0.0, mjcf_attribute_name="noise")
    add("sensor_cutoff", wp.float32, 0.0, mjcf_attribute_name="cutoff")
    add("sensor_nsample", wp.int32, 0, mjcf_attribute_name="nsample")
    add(
        "sensor_interp",
        wp.int32,
        0,
        mjcf_attribute_name="interp",
        mjcf_value_transformer=_parse_sensor_interp,
    )
    add("sensor_delay", wp.float32, 0.0, mjcf_attribute_name="delay")
    add("sensor_interval", wp.vec2, wp.vec2(0.0, 0.0), mjcf_attribute_name="interval")
    # Variable-length MuJoCo user data, stored as space-separated numbers.
    add("sensor_user", str, "", mjcf_attribute_name="user", mjcf_value_transformer=_parse_sensor_user)


# Sensor attributes that MJCF import and the authoring helper set explicitly.
_MANAGED_SENSOR_ATTRIBUTES = frozenset(
    (
        "sensor_world",
        "sensor_type",
        "sensor_objtype",
        "sensor_objid",
        "sensor_reftype",
        "sensor_refid",
        "sensor_intprm",
    )
)


def _validate_world(builder: ModelBuilder, name: str, index: int, world: int) -> None:
    if world not in (-1, builder.current_world):
        raise ValueError(
            f"{name} index {index} belongs to world {world}, but the sensor is added to world {builder.current_world}."
        )


def _validate_index(name: str, index: int, count: int) -> None:
    if index < 0 or index >= count:
        raise IndexError(f"{name} index {index} is outside [0, {count}).")


def _custom_row_world(builder: ModelBuilder, key: str, row: int) -> int:
    attr = builder.custom_attributes[key]
    values = attr.values or []
    value = values.get(row) if isinstance(values, dict) else (values[row] if row < len(values) else None)
    return int(attr.default if value is None else value)


def _resolve_object(builder: ModelBuilder, obj: SensorObject, name: str) -> tuple[int, int]:
    """Validate ``obj`` and return its ``(objtype, objid)`` storage pair."""
    if not isinstance(obj, SensorObject):
        raise TypeError(f"{name} must be a SensorObject.")
    kind = obj.kind
    index = obj.index

    if kind in (_K.BODY, _K.XBODY, _K.SUBTREE):
        if index == -1 and kind != _K.SUBTREE:
            return int(_OBJTYPE_BY_KIND[kind]), -1
        _validate_index(f"{name} body", index, len(builder.body_mass))
        _validate_world(builder, f"{name} body", index, int(builder.body_world[index]))
    elif kind == _K.JOINT:
        _validate_index(f"{name} joint", index, len(builder.joint_type))
        linear_dofs, angular_dofs = builder.joint_dof_dim[index]
        dof_count = linear_dofs + angular_dofs
        if obj.dof is None:
            if dof_count != 1 and builder.joint_type[index] not in (JointType.BALL, JointType.FREE):
                raise ValueError(f"Joint {index} has {dof_count} DOFs; specify the local dof in SensorObject.joint().")
            local_dof = 0
        else:
            local_dof = obj.dof
            _validate_index(f"{name} joint dof", local_dof, dof_count)
        _validate_world(builder, f"{name} joint", index, int(builder.joint_world[index]))
        index = int(builder.joint_qd_start[index]) + local_dof
    elif kind == _K.JOINT_DOF:
        _validate_index(f"{name} joint dof", index, len(builder.joint_qd))
        joint = bisect_right(builder.joint_qd_start, index) - 1
        _validate_world(builder, f"{name} joint", joint, int(builder.joint_world[joint]))
    elif kind in (_K.GEOM, _K.SITE):
        _validate_index(f"{name} shape", index, len(builder.shape_flags))
        is_site = bool(int(builder.shape_flags[index]) & int(ShapeFlags.SITE))
        if kind == _K.SITE and not is_site:
            raise ValueError(f"{name} shape index {index} does not identify a Newton site.")
        if kind == _K.GEOM and is_site:
            raise ValueError(f"{name} shape index {index} identifies a site, not a geom.")
        _validate_world(builder, f"{name} shape", index, int(builder.shape_world[index]))
    elif kind == _K.TENDON:
        _validate_index(f"{name} tendon", index, builder._custom_frequency_counts.get("mujoco:tendon", 0))
        _validate_world(builder, f"{name} tendon", index, _custom_row_world(builder, "mujoco:tendon_world", index))
    elif kind == _K.ACTUATOR:
        _validate_index(f"{name} actuator", index, builder._custom_frequency_counts.get("mujoco:actuator", 0))
        _validate_world(builder, f"{name} actuator", index, _custom_row_world(builder, "mujoco:actuator_world", index))
    else:
        raise ValueError(f"Unsupported sensor object kind {kind!r}.")
    return int(_OBJTYPE_BY_KIND[kind]), int(index)


def _sensor_type(value: SensorType | int | str) -> SensorType:
    if isinstance(value, str):
        key = value.strip().upper()
        try:
            return SensorType[key]
        except KeyError as error:
            raise ValueError(f"Unknown sensor type {value!r}.") from error
    try:
        return SensorType(int(value))
    except ValueError as error:
        raise ValueError(f"Unsupported MuJoCo sensor type {value!r}.") from error


def add_sensor(
    builder: ModelBuilder,
    sensor_type: SensorType | int | str,
    obj: SensorObject | None = None,
    *,
    ref: SensorObject | None = None,
    label: str = "",
    noise: float = 0.0,
    cutoff: float = 0.0,
    intprm: Sequence[int] | None = None,
    nsample: int = 0,
    interp: int | str = "zoh",
    delay: float = 0.0,
    interval: Sequence[float] = (0.0, 0.0),
    user: Sequence[float] = (),
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a native MuJoCo sensor.

    The sensor belongs to :attr:`~newton.ModelBuilder.current_world`.
    :class:`~newton.solvers.SolverMuJoCo` exports it to the MuJoCo model, where
    its output is written to ``sensordata``. Use
    :attr:`~newton.solvers.SolverMuJoCo.mjc_sensor_to_newton_sensor` to map
    MuJoCo sensors back to rows returned by this function.

    Args:
        builder: Model builder receiving the sensor.
        sensor_type: Sensor type, as a :class:`SensorType`, its ordinal, or its
            case-insensitive name such as ``"touch"``.
        obj: Observed object. Required for all types except contact, energy,
            and clock sensors.
        ref: Reference object for frame, in-site, distance, and contact sensors.
        label: MuJoCo sensor name. Empty labels receive a generated name.
        noise: Standard deviation of the sensor noise. MuJoCo stores this value
            but does not apply it.
        cutoff: Absolute output limit, or ``0`` for no limit.
        intprm: MuJoCo integer parameters. Contact sensors use
            ``(dataspec, reduce, num)`` and rangefinders use ``(dataspec, 0, 0)``;
            ``None`` selects MuJoCo's defaults.
        nsample: Number of samples kept in the sensor history buffer.
        interp: History interpolation, one of ``"zoh"``, ``"linear"``, or
            ``"cubic"``.
        delay: Sensor delay [s].
        interval: Sensor sampling period and phase [s].
        user: MuJoCo sensor user data. MuJoCo pads all sensors to the longest
            user-data length.
        custom_attributes: Additional registered ``mujoco:sensor`` attributes.

    Returns:
        The ``mujoco:sensor`` row index.
    """
    _ensure_mujoco_attributes(builder, "mujoco:sensor_type")
    stype = _sensor_type(sensor_type)
    obj_kinds, obj_required, ref_kinds, ref_required = _SENSOR_SIGNATURES[stype]

    def check(name: str, value: SensorObject | None, kinds: frozenset, required: bool) -> tuple[int, int]:
        if value is None:
            if required:
                raise ValueError(f"{stype.name} sensors require {name}.")
            return int(_SensorObjType.UNKNOWN), -1
        if value.kind not in kinds:
            allowed = ", ".join(sorted(kind.value for kind in kinds)) or "none"
            raise ValueError(f"{stype.name} sensors accept {name} kinds: {allowed}; got {value.kind.value}.")
        return _resolve_object(builder, value, name)

    objtype, objid = check("obj", obj, obj_kinds, obj_required)
    reftype, refid = check("ref", ref, ref_kinds, ref_required)

    if intprm is None:
        intprm = _DEFAULT_INTPRM.get(stype, (0, 0, 0))
    if len(intprm) > 3:
        raise ValueError(f"intprm requires at most 3 values, got {len(intprm)}.")
    intprm_values = [int(value) for value in intprm] + [0] * (3 - len(intprm))
    if cutoff < 0.0:
        raise ValueError("cutoff must be non-negative.")
    if nsample < 0:
        raise ValueError("nsample must be non-negative.")

    values = {
        "mujoco:sensor_world": builder.current_world,
        "mujoco:sensor_label": str(label),
        "mujoco:sensor_type": int(stype),
        "mujoco:sensor_objtype": objtype,
        "mujoco:sensor_objid": objid,
        "mujoco:sensor_reftype": reftype,
        "mujoco:sensor_refid": refid,
        "mujoco:sensor_intprm": wp.vec3i(*intprm_values),
        "mujoco:sensor_noise": float(noise),
        "mujoco:sensor_cutoff": float(cutoff),
        "mujoco:sensor_nsample": int(nsample),
        "mujoco:sensor_interp": _parse_sensor_interp(interp),
        "mujoco:sensor_delay": float(delay),
        "mujoco:sensor_interval": _vector(builder, "mujoco:sensor_interval", interval, 2),
        "mujoco:sensor_user": _parse_sensor_user(user),
    }
    return _add_custom_frequency_row(
        builder,
        "mujoco:sensor",
        values,
        index_key="mujoco:sensor_type",
        custom_attributes=custom_attributes,
    )


__all__ = ["SensorObject", "SensorType", "add_sensor"]
