# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Programmatic authoring helpers for MuJoCo actuators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import warp as wp

from ...geometry import ShapeFlags
from ...sim import ModelBuilder
from ._authoring import _add_custom_frequency_row, _ensure_mujoco_attributes
from .enums import _ActuatorBiasType, _ActuatorDynamicsType, _ActuatorGainType


@dataclass(frozen=True)
class ActuatorTarget:
    """A typed MuJoCo actuator transmission target.

    Construct targets with :meth:`joint`, :meth:`joint_dof`, :meth:`tendon`,
    :meth:`site`, :meth:`body`, or :meth:`slider_crank`. The authoring helper
    resolves the target to MuJoCo's heterogeneous ``trnid`` representation.
    """

    kind: str
    """Transmission target kind."""

    index: int
    """Primary Newton entity index."""

    secondary_index: int = -1
    """Optional secondary Newton entity index."""

    dof: int | None = None
    """Local joint DOF, when :attr:`kind` is ``"joint"``."""

    @classmethod
    def joint(cls, joint: int, dof: int | None = None) -> ActuatorTarget:
        """Target a Newton joint, optionally selecting one local DOF.

        Args:
            joint: Newton joint index.
            dof: Local DOF within the joint. Required for multi-DOF joints.

        Returns:
            A typed joint target.
        """
        return cls("joint", int(joint), dof=None if dof is None else int(dof))

    @classmethod
    def joint_dof(cls, dof: int) -> ActuatorTarget:
        """Target an absolute Newton joint-DOF index.

        Args:
            dof: Absolute Newton joint-DOF index.

        Returns:
            A typed joint-DOF target.
        """
        return cls("joint_dof", int(dof))

    @classmethod
    def tendon(cls, tendon: int) -> ActuatorTarget:
        """Target a ``mujoco:tendon`` row.

        Args:
            tendon: Row index returned by :func:`add_tendon_fixed` or
                :func:`add_tendon_spatial`.

        Returns:
            A typed tendon target.
        """
        return cls("tendon", int(tendon))

    @classmethod
    def site(cls, site: int, refsite: int | None = None) -> ActuatorTarget:
        """Target a Newton site shape, optionally relative to another site.

        Args:
            site: Newton site-shape index.
            refsite: Optional reference site-shape index.

        Returns:
            A typed site target.
        """
        return cls("site", int(site), -1 if refsite is None else int(refsite))

    @classmethod
    def body(cls, body: int) -> ActuatorTarget:
        """Target a Newton body.

        Args:
            body: Newton body index.

        Returns:
            A typed body target.
        """
        return cls("body", int(body))

    @classmethod
    def slider_crank(cls, cranksite: int, slidersite: int) -> ActuatorTarget:
        """Target the crank and slider sites of a slider-crank transmission.

        Args:
            cranksite: Newton crank site-shape index.
            slidersite: Newton slider site-shape index.

        Returns:
            A typed slider-crank target.
        """
        return cls("slider_crank", int(cranksite), int(slidersite))


def _validate_index(name: str, index: int, count: int) -> None:
    if index < 0 or index >= count:
        raise IndexError(f"{name} index {index} is outside [0, {count}).")


def _validate_site(builder: ModelBuilder, site: int, name: str) -> None:
    _validate_index(name, site, len(builder.shape_flags))
    if not (int(builder.shape_flags[site]) & int(ShapeFlags.SITE)):
        raise ValueError(f"{name} index {site} does not identify a Newton site.")


def _resolve_target(builder: ModelBuilder, target: ActuatorTarget) -> tuple[int, wp.vec2i]:
    from .solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    if not isinstance(target, ActuatorTarget):
        raise TypeError("target must be an ActuatorTarget.")

    if target.kind == "joint":
        _validate_index("joint", target.index, len(builder.joint_type))
        linear_dofs, angular_dofs = builder.joint_dof_dim[target.index]
        dof_count = linear_dofs + angular_dofs
        if target.dof is None:
            if dof_count != 1:
                raise ValueError(
                    f"Joint {target.index} has {dof_count} DOFs; specify the local dof in ActuatorTarget.joint()."
                )
            local_dof = 0
        else:
            local_dof = target.dof
        _validate_index("joint dof", local_dof, dof_count)
        dof = int(builder.joint_qd_start[target.index]) + local_dof
        return int(SolverMuJoCo.TrnType.JOINT), wp.vec2i(dof, -1)

    if target.kind == "joint_dof":
        _validate_index("joint dof", target.index, len(builder.joint_qd))
        return int(SolverMuJoCo.TrnType.JOINT), wp.vec2i(target.index, -1)

    if target.kind == "tendon":
        tendon_count = builder._custom_frequency_counts.get("mujoco:tendon", 0)
        _validate_index("tendon", target.index, tendon_count)
        return int(SolverMuJoCo.TrnType.TENDON), wp.vec2i(target.index, -1)

    if target.kind == "site":
        _validate_site(builder, target.index, "site")
        if target.secondary_index >= 0:
            _validate_site(builder, target.secondary_index, "refsite")
        return int(SolverMuJoCo.TrnType.SITE), wp.vec2i(target.index, target.secondary_index)

    if target.kind == "body":
        _validate_index("body", target.index, len(builder.body_mass))
        return int(SolverMuJoCo.TrnType.BODY), wp.vec2i(target.index, -1)

    if target.kind == "slider_crank":
        _validate_site(builder, target.index, "cranksite")
        _validate_site(builder, target.secondary_index, "slidersite")
        return int(SolverMuJoCo.TrnType.SLIDERCRANK), wp.vec2i(target.index, target.secondary_index)

    raise ValueError(f"Unsupported actuator target kind {target.kind!r}.")


def _vector(
    builder: ModelBuilder,
    key: str,
    values: Sequence[float],
    length: int,
    *,
    exact: bool = False,
) -> Any:
    components = [float(value) for value in values]
    if (exact and len(components) != length) or len(components) > length:
        requirement = "exactly" if exact else "at most"
        raise ValueError(f"{key} requires {requirement} {length} values, got {len(components)}.")
    components.extend([0.0] * (length - len(components)))
    return builder.custom_attributes[key].dtype(*components)


def _tristate(value: bool | int | str) -> int:
    if isinstance(value, str):
        try:
            return {"false": 0, "true": 1, "auto": 2}[value.lower().strip()]
        except KeyError as error:
            raise ValueError(f"Expected false, true, or auto, got {value!r}.") from error
    result = int(value)
    if result not in (0, 1, 2):
        raise ValueError(f"Expected a MuJoCo tri-state value in {{0, 1, 2}}, got {value!r}.")
    return result


def _enum_value(value: int | str, mapping: dict[str, int], name: str) -> int:
    if isinstance(value, str):
        try:
            return mapping[value.lower().replace("_", "").strip()]
        except KeyError as error:
            choices = ", ".join(sorted(mapping))
            raise ValueError(f"Unknown {name} {value!r}; expected one of {choices}.") from error
    return int(value)


def _dynamics_type(value: int | str) -> int:
    return _enum_value(
        value,
        {
            "none": int(_ActuatorDynamicsType.NONE),
            "integrator": int(_ActuatorDynamicsType.INTEGRATOR),
            "filter": int(_ActuatorDynamicsType.FILTER),
            "filterexact": int(_ActuatorDynamicsType.FILTER_EXACT),
            "muscle": int(_ActuatorDynamicsType.MUSCLE),
            "dcmotor": int(_ActuatorDynamicsType.DCMOTOR),
            "user": int(_ActuatorDynamicsType.USER),
        },
        "dyntype",
    )


def _gain_type(value: int | str) -> int:
    return _enum_value(
        value,
        {
            "fixed": int(_ActuatorGainType.FIXED),
            "affine": int(_ActuatorGainType.AFFINE),
            "muscle": int(_ActuatorGainType.MUSCLE),
            "dcmotor": int(_ActuatorGainType.DCMOTOR),
            "so3": int(_ActuatorGainType.SO3),
            "user": int(_ActuatorGainType.USER),
        },
        "gaintype",
    )


def _bias_type(value: int | str) -> int:
    return _enum_value(
        value,
        {
            "none": int(_ActuatorBiasType.NONE),
            "affine": int(_ActuatorBiasType.AFFINE),
            "muscle": int(_ActuatorBiasType.MUSCLE),
            "dcmotor": int(_ActuatorBiasType.DCMOTOR),
            "so3": int(_ActuatorBiasType.SO3),
            "user": int(_ActuatorBiasType.USER),
        },
        "biastype",
    )


def _input_mode(value: int | str) -> int:
    return _enum_value(value, {"voltage": 0, "position": 1, "velocity": 2}, "DC-motor input mode")


def _add_actuator(
    builder: ModelBuilder,
    target: ActuatorTarget,
    *,
    ctrl_type: int,
    dyntype: int | str = "none",
    gaintype: int | str = "fixed",
    biastype: int | str = "none",
    dynprm: Sequence[float] = (1.0,),
    gainprm: Sequence[float] = (1.0,),
    biasprm: Sequence[float] = (0.0,),
    gear: Sequence[float] = (1.0,),
    ctrlrange: Sequence[float] | None = None,
    ctrllimited: bool | int | str = "auto",
    forcerange: Sequence[float] | None = None,
    forcelimited: bool | int | str = "auto",
    actrange: Sequence[float] | None = None,
    actlimited: bool | int | str = "auto",
    actdim: int | None = None,
    actearly: bool = False,
    cranklength: float | None = None,
    damping: float = 0.0,
    armature: float = 0.0,
    ctrl: float = 0.0,
    specific_values: dict[str, Any] | None = None,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    from .solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    _ensure_mujoco_attributes(builder, "mujoco:actuator_trnid")
    trntype, trnid = _resolve_target(builder, target)
    if trntype == int(SolverMuJoCo.TrnType.SLIDERCRANK):
        if cranklength is None or cranklength <= 0.0:
            raise ValueError("A slider-crank actuator requires a positive cranklength [m].")

    values: dict[str, Any] = {
        "mujoco:actuator_trnid": trnid,
        "mujoco:actuator_target_label": "",
        "mujoco:actuator_trntype": trntype,
        "mujoco:actuator_dyntype": _dynamics_type(dyntype),
        "mujoco:actuator_gaintype": _gain_type(gaintype),
        "mujoco:actuator_biastype": _bias_type(biastype),
        "mujoco:actuator_world": builder.current_world,
        "mujoco:actuator_ctrllimited": _tristate(ctrllimited),
        "mujoco:actuator_forcelimited": _tristate(forcelimited),
        "mujoco:actuator_ctrlrange": _vector(
            builder,
            "mujoco:actuator_ctrlrange",
            ctrlrange if ctrlrange is not None else (),
            2,
        ),
        "mujoco:actuator_has_ctrlrange": int(ctrlrange is not None),
        "mujoco:actuator_forcerange": _vector(
            builder,
            "mujoco:actuator_forcerange",
            forcerange if forcerange is not None else (),
            2,
        ),
        "mujoco:actuator_has_forcerange": int(forcerange is not None),
        "mujoco:actuator_gear": _vector(builder, "mujoco:actuator_gear", gear, 6),
        "mujoco:actuator_damping": float(damping),
        "mujoco:actuator_armature": float(armature),
        "mujoco:actuator_cranklength": 0.0 if cranklength is None else float(cranklength),
        "mujoco:actuator_dynprm": _vector(builder, "mujoco:actuator_dynprm", dynprm, 10),
        "mujoco:actuator_gainprm": _vector(builder, "mujoco:actuator_gainprm", gainprm, 10),
        "mujoco:actuator_biasprm": _vector(builder, "mujoco:actuator_biasprm", biasprm, 10),
        "mujoco:actuator_actlimited": _tristate(actlimited),
        "mujoco:actuator_actrange": _vector(
            builder,
            "mujoco:actuator_actrange",
            actrange if actrange is not None else (),
            2,
        ),
        "mujoco:actuator_has_actrange": int(actrange is not None),
        "mujoco:actuator_actdim": -1 if actdim is None else int(actdim),
        "mujoco:actuator_actearly": bool(actearly),
        "mujoco:ctrl": float(ctrl),
        "mujoco:ctrl_source": int(SolverMuJoCo.CtrlSource.CTRL_DIRECT),
        "mujoco:ctrl_type": int(ctrl_type),
    }
    if specific_values:
        values.update(specific_values)

    return _add_custom_frequency_row(
        builder,
        "mujoco:actuator",
        values,
        index_key="mujoco:actuator_trnid",
        custom_attributes=custom_attributes,
    )


def add_actuator_general(
    builder: ModelBuilder,
    target: ActuatorTarget,
    *,
    dyntype: int | str = "none",
    gaintype: int | str = "fixed",
    biastype: int | str = "none",
    dynprm: Sequence[float] = (1.0,),
    gainprm: Sequence[float] = (1.0,),
    biasprm: Sequence[float] = (0.0,),
    gear: Sequence[float] = (1.0,),
    ctrlrange: Sequence[float] | None = None,
    ctrllimited: bool | int | str = "auto",
    forcerange: Sequence[float] | None = None,
    forcelimited: bool | int | str = "auto",
    actrange: Sequence[float] | None = None,
    actlimited: bool | int | str = "auto",
    actdim: int | None = None,
    actearly: bool = False,
    cranklength: float | None = None,
    damping: float = 0.0,
    armature: float = 0.0,
    ctrl: float = 0.0,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a MuJoCo-native general actuator controlled through ``control.mujoco.ctrl``.

    Args:
        builder: Model builder receiving the actuator.
        target: Typed actuator transmission target.
        dyntype: MuJoCo activation dynamics type.
        gaintype: MuJoCo gain type.
        biastype: MuJoCo bias type.
        dynprm: Activation dynamics parameters, padded to ten values.
        gainprm: Gain parameters, padded to ten values.
        biasprm: Bias parameters, padded to ten values.
        gear: Transmission gear, padded to six values.
        ctrlrange: Optional control range.
        ctrllimited: Control-limit tri-state (false, true, or auto).
        forcerange: Optional actuator force range [N or N·m].
        forcelimited: Force-limit tri-state (false, true, or auto).
        actrange: Optional activation-state range.
        actlimited: Activation-limit tri-state (false, true, or auto).
        actdim: Activation-state dimension, or ``None`` for automatic selection.
        actearly: Whether force uses the next activation state.
        cranklength: Slider-crank length [m]. Required for slider-crank targets.
        damping: Actuator damping [N·s/m or N·m·s/rad].
        armature: Actuator armature [kg or kg·m²].
        ctrl: Initial MuJoCo control value.
        custom_attributes: Additional registered ``mujoco:actuator`` attributes.

    Returns:
        The ``mujoco:actuator`` row index.
    """
    from .solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    return _add_actuator(
        builder,
        target,
        ctrl_type=int(SolverMuJoCo.CtrlType.GENERAL),
        dyntype=dyntype,
        gaintype=gaintype,
        biastype=biastype,
        dynprm=dynprm,
        gainprm=gainprm,
        biasprm=biasprm,
        gear=gear,
        ctrlrange=ctrlrange,
        ctrllimited=ctrllimited,
        forcerange=forcerange,
        forcelimited=forcelimited,
        actrange=actrange,
        actlimited=actlimited,
        actdim=actdim,
        actearly=actearly,
        cranklength=cranklength,
        damping=damping,
        armature=armature,
        ctrl=ctrl,
        custom_attributes=custom_attributes,
    )


def add_actuator_motor(
    builder: ModelBuilder,
    target: ActuatorTarget,
    *,
    gear: Sequence[float] = (1.0,),
    ctrlrange: Sequence[float] | None = None,
    ctrllimited: bool | int | str = "auto",
    forcerange: Sequence[float] | None = None,
    forcelimited: bool | int | str = "auto",
    cranklength: float | None = None,
    damping: float = 0.0,
    armature: float = 0.0,
    ctrl: float = 0.0,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a MuJoCo motor shortcut controlled through ``control.mujoco.ctrl``.

    Args:
        builder: Model builder receiving the actuator.
        target: Typed actuator transmission target.
        gear: Transmission gear, padded to six values.
        ctrlrange: Optional control range.
        ctrllimited: Control-limit tri-state (false, true, or auto).
        forcerange: Optional actuator force range [N or N·m].
        forcelimited: Force-limit tri-state (false, true, or auto).
        cranklength: Slider-crank length [m]. Required for slider-crank targets.
        damping: Actuator damping [N·s/m or N·m·s/rad].
        armature: Actuator armature [kg or kg·m²].
        ctrl: Initial MuJoCo control value.
        custom_attributes: Additional registered ``mujoco:actuator`` attributes.

    Returns:
        The ``mujoco:actuator`` row index.
    """
    return add_actuator_general(
        builder,
        target,
        gainprm=(1.0,),
        biasprm=(0.0,),
        gear=gear,
        ctrlrange=ctrlrange,
        ctrllimited=ctrllimited,
        forcerange=forcerange,
        forcelimited=forcelimited,
        cranklength=cranklength,
        damping=damping,
        armature=armature,
        ctrl=ctrl,
        custom_attributes=custom_attributes,
    )


def add_actuator_position(
    builder: ModelBuilder,
    target: ActuatorTarget,
    *,
    kp: float = 1.0,
    kv: float = 0.0,
    dampratio: float = 0.0,
    gear: Sequence[float] = (1.0,),
    ctrlrange: Sequence[float] | None = None,
    ctrllimited: bool | int | str = "auto",
    forcerange: Sequence[float] | None = None,
    forcelimited: bool | int | str = "auto",
    cranklength: float | None = None,
    damping: float = 0.0,
    armature: float = 0.0,
    ctrl: float = 0.0,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a MuJoCo position shortcut controlled through ``control.mujoco.ctrl``.

    Args:
        builder: Model builder receiving the actuator.
        target: Typed actuator transmission target.
        kp: Position feedback gain.
        kv: Velocity feedback gain.
        dampratio: Damping ratio used when ``kv`` is zero.
        gear: Transmission gear, padded to six values.
        ctrlrange: Optional control range.
        ctrllimited: Control-limit tri-state (false, true, or auto).
        forcerange: Optional actuator force range [N or N·m].
        forcelimited: Force-limit tri-state (false, true, or auto).
        cranklength: Slider-crank length [m]. Required for slider-crank targets.
        damping: Actuator damping [N·s/m or N·m·s/rad].
        armature: Actuator armature [kg or kg·m²].
        ctrl: Initial MuJoCo control value.
        custom_attributes: Additional registered ``mujoco:actuator`` attributes.

    Returns:
        The ``mujoco:actuator`` row index.
    """
    from .solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    if kp <= 0.0:
        raise ValueError("kp must be positive.")
    if kv != 0.0 and dampratio != 0.0:
        raise ValueError("Specify either kv or dampratio, not both.")
    bias_damping = -float(kv) if kv != 0.0 else float(dampratio)
    return _add_actuator(
        builder,
        target,
        ctrl_type=int(SolverMuJoCo.CtrlType.POSITION),
        gaintype="fixed",
        biastype="affine",
        gainprm=(kp,),
        biasprm=(0.0, -kp, bias_damping),
        gear=gear,
        ctrlrange=ctrlrange,
        ctrllimited=ctrllimited,
        forcerange=forcerange,
        forcelimited=forcelimited,
        cranklength=cranklength,
        damping=damping,
        armature=armature,
        ctrl=ctrl,
        custom_attributes=custom_attributes,
    )


def add_actuator_velocity(
    builder: ModelBuilder,
    target: ActuatorTarget,
    *,
    kv: float = 1.0,
    gear: Sequence[float] = (1.0,),
    ctrlrange: Sequence[float] | None = None,
    ctrllimited: bool | int | str = "auto",
    forcerange: Sequence[float] | None = None,
    forcelimited: bool | int | str = "auto",
    cranklength: float | None = None,
    damping: float = 0.0,
    armature: float = 0.0,
    ctrl: float = 0.0,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a MuJoCo velocity shortcut controlled through ``control.mujoco.ctrl``.

    Args:
        builder: Model builder receiving the actuator.
        target: Typed actuator transmission target.
        kv: Velocity feedback gain.
        gear: Transmission gear, padded to six values.
        ctrlrange: Optional control range.
        ctrllimited: Control-limit tri-state (false, true, or auto).
        forcerange: Optional actuator force range [N or N·m].
        forcelimited: Force-limit tri-state (false, true, or auto).
        cranklength: Slider-crank length [m]. Required for slider-crank targets.
        damping: Actuator damping [N·s/m or N·m·s/rad].
        armature: Actuator armature [kg or kg·m²].
        ctrl: Initial MuJoCo control value.
        custom_attributes: Additional registered ``mujoco:actuator`` attributes.

    Returns:
        The ``mujoco:actuator`` row index.
    """
    from .solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    if kv <= 0.0:
        raise ValueError("kv must be positive.")
    return _add_actuator(
        builder,
        target,
        ctrl_type=int(SolverMuJoCo.CtrlType.VELOCITY),
        gaintype="fixed",
        biastype="affine",
        gainprm=(kv,),
        biasprm=(0.0, 0.0, -kv),
        gear=gear,
        ctrlrange=ctrlrange,
        ctrllimited=ctrllimited,
        forcerange=forcerange,
        forcelimited=forcelimited,
        cranklength=cranklength,
        damping=damping,
        armature=armature,
        ctrl=ctrl,
        custom_attributes=custom_attributes,
    )


def add_actuator_dcmotor(
    builder: ModelBuilder,
    target: ActuatorTarget,
    *,
    motorconst: Sequence[float] = (0.0, 0.0),
    resistance: float = 0.0,
    nominal: Sequence[float] = (0.0, 0.0, 0.0),
    saturation: Sequence[float] = (0.0, 0.0, 0.0),
    inductance: Sequence[float] = (0.0, 0.0),
    cogging: Sequence[float] = (0.0, 0.0, 0.0),
    controller: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    thermal: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    lugre: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0),
    input_mode: int | str = "voltage",
    gear: Sequence[float] = (1.0,),
    ctrlrange: Sequence[float] | None = None,
    ctrllimited: bool | int | str = "auto",
    forcerange: Sequence[float] | None = None,
    forcelimited: bool | int | str = "auto",
    actrange: Sequence[float] | None = None,
    actlimited: bool | int | str = "auto",
    cranklength: float | None = None,
    damping: float = 0.0,
    armature: float = 0.0,
    ctrl: float = 0.0,
    custom_attributes: dict[str, Any] | None = None,
) -> int:
    """Add a MuJoCo DC-motor shortcut controlled through ``control.mujoco.ctrl``.

    The high-level values are preserved until :class:`SolverMuJoCo` builds its
    native MuJoCo model, where ``MjsActuator.set_to_dcmotor()`` compiles them.

    Args:
        builder: Model builder receiving the actuator.
        target: Typed actuator transmission target.
        motorconst: Motor torque and back-EMF constants.
        resistance: Terminal resistance [ohm].
        nominal: Nominal voltage, current, and speed.
        saturation: Current, voltage, and controller saturation values.
        inductance: Inductance and current-loop bandwidth.
        cogging: Cogging-friction parameters.
        controller: Controller parameters.
        thermal: Thermal-model parameters.
        lugre: LuGre-friction parameters.
        input_mode: ``"voltage"``, ``"position"``, ``"velocity"``, or its integer code.
        gear: Transmission gear, padded to six values.
        ctrlrange: Optional control range.
        ctrllimited: Control-limit tri-state (false, true, or auto).
        forcerange: Optional actuator force range [N or N·m].
        forcelimited: Force-limit tri-state (false, true, or auto).
        actrange: Optional activation-state range.
        actlimited: Activation-limit tri-state (false, true, or auto).
        cranklength: Slider-crank length [m]. Required for slider-crank targets.
        damping: Actuator damping [N·s/m or N·m·s/rad].
        armature: Actuator armature [kg or kg·m²].
        ctrl: Initial MuJoCo control value.
        custom_attributes: Additional registered ``mujoco:actuator`` attributes.

    Returns:
        The ``mujoco:actuator`` row index.
    """
    from .solver_mujoco import SolverMuJoCo  # noqa: PLC0415

    _ensure_mujoco_attributes(builder, "mujoco:actuator_trnid")
    SolverMuJoCo._register_dcmotor_custom_attributes(builder)
    specific_values = {
        "mujoco:actuator_dcmotor_motorconst": _vector(
            builder, "mujoco:actuator_dcmotor_motorconst", motorconst, 2, exact=True
        ),
        "mujoco:actuator_dcmotor_resistance": float(resistance),
        "mujoco:actuator_dcmotor_nominal": _vector(builder, "mujoco:actuator_dcmotor_nominal", nominal, 3, exact=True),
        "mujoco:actuator_dcmotor_saturation": _vector(
            builder, "mujoco:actuator_dcmotor_saturation", saturation, 3, exact=True
        ),
        "mujoco:actuator_dcmotor_inductance": _vector(
            builder, "mujoco:actuator_dcmotor_inductance", inductance, 2, exact=True
        ),
        "mujoco:actuator_dcmotor_cogging": _vector(builder, "mujoco:actuator_dcmotor_cogging", cogging, 3, exact=True),
        "mujoco:actuator_dcmotor_controller": _vector(
            builder, "mujoco:actuator_dcmotor_controller", controller, 6, exact=True
        ),
        "mujoco:actuator_dcmotor_thermal": _vector(builder, "mujoco:actuator_dcmotor_thermal", thermal, 6, exact=True),
        "mujoco:actuator_dcmotor_lugre": _vector(builder, "mujoco:actuator_dcmotor_lugre", lugre, 5, exact=True),
        "mujoco:actuator_dcmotor_input": _input_mode(input_mode),
    }
    return _add_actuator(
        builder,
        target,
        ctrl_type=int(SolverMuJoCo.CtrlType.DCMOTOR),
        gear=gear,
        ctrlrange=ctrlrange,
        ctrllimited=ctrllimited,
        forcerange=forcerange,
        forcelimited=forcelimited,
        actrange=actrange,
        actlimited=actlimited,
        cranklength=cranklength,
        damping=damping,
        armature=armature,
        ctrl=ctrl,
        specific_values=specific_values,
        custom_attributes=custom_attributes,
    )


__all__ = [
    "ActuatorTarget",
    "add_actuator_dcmotor",
    "add_actuator_general",
    "add_actuator_motor",
    "add_actuator_position",
    "add_actuator_velocity",
]
