# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Provides builders for testing supported joint and geometry types.

This module defines a set of functions for creating
model builders to test and demonstrate all the types
of joints and geometries supported by Kamino.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ......core import Axis
from ......geometry import GeoType
from ......sim import JointTargetMode, JointType, ModelBuilder

###
# Module interface
###

__all__ = [
    "build_all_joints_test",
    "build_binary_cartesian_joint_test",
    "build_binary_cylindrical_joint_test",
    "build_binary_prismatic_joint_test",
    "build_binary_revolute_joint_test",
    "build_binary_spherical_joint_test",
    "build_binary_universal_joint_test",
    "build_free_joint_test",
    "build_shape_pairs_test",
    "build_unary_cartesian_joint_test",
    "build_unary_cylindrical_joint_test",
    "build_unary_prismatic_joint_test",
    "build_unary_revolute_joint_test",
    "build_unary_spherical_joint_test",
    "build_unary_universal_joint_test",
    "get_shape_bottom_position",
    "make_shape_initial_position",
    "make_single_shape_pair_builder",
    "shape_default_dims",
    "shape_name_to_type",
]


###
# Helpers
###


def _shape_cfg_basic(collision_group: int = 1) -> ModelBuilder.ShapeConfig:
    """Shape config matching Kamino testing defaults (zero margin and gap)."""
    return ModelBuilder.ShapeConfig(margin=0.0, gap=0.0, collision_group=collision_group)


def _identity_inertia() -> wp.mat33:
    """Identity inertia tensor matching Kamino's ``I_3`` used throughout ``testing.py``."""
    return wp.mat33(np.eye(3, dtype=np.float32))


def _actuator_mode(implicit_pd: bool) -> JointTargetMode:
    """Select the actuator mode used by Kamino's testing builders given ``implicit_pd``."""
    return JointTargetMode.POSITION_VELOCITY if implicit_pd else JointTargetMode.EFFORT


def _dof_config(
    axis,
    *,
    limit_lower: float | None = None,
    limit_upper: float | None = None,
    armature: float = 0.0,
    friction: float = 0.0,
    target_ke: float = 0.0,
    target_kd: float = 0.0,
    effort_limit: float | None = None,
    actuator_mode: JointTargetMode | None = None,
) -> ModelBuilder.JointDofConfig:
    """Build a :class:`ModelBuilder.JointDofConfig` from Kamino-style per-DoF parameters."""
    kwargs: dict = {
        "axis": axis,
        "armature": armature,
        "friction": friction,
        "target_ke": target_ke,
        "target_kd": target_kd,
    }
    if limit_lower is not None:
        kwargs["limit_lower"] = limit_lower
    if limit_upper is not None:
        kwargs["limit_upper"] = limit_upper
    if effort_limit is not None:
        kwargs["effort_limit"] = effort_limit
    if actuator_mode is not None:
        kwargs["actuator_mode"] = actuator_mode
    return ModelBuilder.JointDofConfig(**kwargs)


def _follower_body(
    builder: ModelBuilder,
    label: str,
    xyz: tuple[float, float, float],
) -> int:
    """Add a unit-mass rigid body with identity inertia at ``xyz`` and return its index."""
    return builder.add_link(
        label=label,
        mass=1.0,
        inertia=_identity_inertia(),
        xform=wp.transformf(xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )


def _cylindrical_axes(
    axis: Axis,
    *,
    limits: bool,
    dynamic: bool,
    implicit_pd: bool,
) -> tuple[list[ModelBuilder.JointDofConfig], list[ModelBuilder.JointDofConfig]]:
    """Build linear + angular axes for a cylindrical joint sharing a single spatial axis."""
    mode = _actuator_mode(implicit_pd)
    kw_lin = {
        "armature": 0.1 if dynamic else 0.0,
        "friction": 0.01 if dynamic else 0.0,
        "target_ke": 10.0 if implicit_pd else 0.0,
        "target_kd": 0.01 if implicit_pd else 0.0,
        "actuator_mode": mode,
    }
    kw_ang = {
        "armature": 0.2 if dynamic else 0.0,
        "friction": 0.02 if dynamic else 0.0,
        "target_ke": 20.0 if implicit_pd else 0.0,
        "target_kd": 0.02 if implicit_pd else 0.0,
        "actuator_mode": mode,
    }
    if limits:
        kw_lin["limit_lower"] = -0.5
        kw_lin["limit_upper"] = 0.5
        kw_ang["limit_lower"] = -0.6 * math.pi
        kw_ang["limit_upper"] = 0.6 * math.pi
    lin = ModelBuilder.JointDofConfig(axis=axis, **kw_lin)
    ang = ModelBuilder.JointDofConfig(axis=axis, **kw_ang)
    return [lin], [ang]


def _universal_axes(limits: bool) -> list[ModelBuilder.JointDofConfig]:
    """Two rotational axes (X, Y) with optional limits, matching Kamino's universal joint."""
    kw_x = {"axis": Axis.X, "actuator_mode": JointTargetMode.EFFORT}
    kw_y = {"axis": Axis.Y, "actuator_mode": JointTargetMode.EFFORT}
    if limits:
        kw_x["limit_lower"] = -0.6 * math.pi
        kw_x["limit_upper"] = 0.6 * math.pi
        kw_y["limit_lower"] = -0.6 * math.pi
        kw_y["limit_upper"] = 0.6 * math.pi
    return [
        ModelBuilder.JointDofConfig(**kw_x),
        ModelBuilder.JointDofConfig(**kw_y),
    ]


def _cartesian_axes(
    *,
    limits: bool,
    dynamic: bool,
    implicit_pd: bool,
) -> list[ModelBuilder.JointDofConfig]:
    """Three translational axes (X, Y, Z) with optional limits and PD parameters."""
    mode = _actuator_mode(implicit_pd)
    armature = [0.1, 0.2, 0.3] if dynamic else [0.0, 0.0, 0.0]
    friction = [0.01, 0.02, 0.03] if dynamic else [0.0, 0.0, 0.0]
    ke = [10.0, 20.0, 30.0] if implicit_pd else [0.0, 0.0, 0.0]
    kd = [0.01, 0.02, 0.03] if implicit_pd else [0.0, 0.0, 0.0]
    axes = [Axis.X, Axis.Y, Axis.Z]
    configs: list[ModelBuilder.JointDofConfig] = []
    for i, axis in enumerate(axes):
        kw = {
            "axis": axis,
            "armature": armature[i],
            "friction": friction[i],
            "target_ke": ke[i],
            "target_kd": kd[i],
            "actuator_mode": mode,
        }
        if limits:
            kw["limit_lower"] = -1.0
            kw["limit_upper"] = 1.0
        configs.append(ModelBuilder.JointDofConfig(**kw))
    return configs


def _apply_angular_limits(
    builder: ModelBuilder,
    joint_id: int,
    limits_per_axis: list[tuple[float, float]],
) -> None:
    """Overwrite angular DoF limits of a previously-added joint.

    ``add_joint_ball`` does not accept per-axis limits, so we patch the ModelBuilder's
    flat DoF arrays for the joint's angular DoFs to mirror Kamino's ``q_j_min``/``q_j_max``.
    """
    dof_start = builder.joint_qd_start[joint_id]
    lin, _ = builder.joint_dof_dim[joint_id]
    ang_start = dof_start + lin
    for i, (lo, hi) in enumerate(limits_per_axis):
        idx = ang_start + i
        builder.joint_limit_lower[idx] = lo
        builder.joint_limit_upper[idx] = hi


def _add_ground_box(builder: ModelBuilder, z_offset: float = 0.0) -> None:
    """Static ground box matching Kamino ``testing`` builders (center at z = -1.5 + z_offset)."""
    builder.add_shape_box(
        label="ground",
        body=-1,
        hx=10.0,
        hy=10.0,
        hz=0.5,
        xform=wp.transformf(0.0, 0.0, -1.5 + z_offset, 0.0, 0.0, 0.0, 1.0),
        cfg=_shape_cfg_basic(),
    )


###
# Builders - Joint Tests
###


def build_free_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
) -> ModelBuilder:
    """
    Builds a world to test free joints.

    This world consists of a single rigid body connected to the world via a unary
    free joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degrees of freedom.
        ground: Whether to include a static ground box in the world.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="unary_free_joint_test")

    # Define test system
    bid_F = _follower_body(_builder, "follower", (0.0, 0.0, z_offset))

    lin_lim = (-2.0, 2.0)
    ang_lim = (-0.6 * math.pi, 0.6 * math.pi)
    eff_lim = 100.0
    j_free = _builder.add_joint(
        label="world_to_follower_free",
        joint_type=JointType.FREE,
        parent=-1,
        child=bid_F,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
        linear_axes=[
            _dof_config(Axis.X, limit_lower=lin_lim[0], limit_upper=lin_lim[1], effort_limit=eff_lim),
            _dof_config(Axis.Y, limit_lower=lin_lim[0], limit_upper=lin_lim[1], effort_limit=eff_lim),
            _dof_config(Axis.Z, limit_lower=lin_lim[0], limit_upper=lin_lim[1], effort_limit=eff_lim),
        ],
        angular_axes=[
            _dof_config(Axis.X, limit_lower=ang_lim[0], limit_upper=ang_lim[1], effort_limit=eff_lim),
            _dof_config(Axis.Y, limit_lower=ang_lim[0], limit_upper=ang_lim[1], effort_limit=eff_lim),
            _dof_config(Axis.Z, limit_lower=ang_lim[0], limit_upper=ang_lim[1], effort_limit=eff_lim),
        ],
    )
    _builder.add_articulation([j_free])

    # Add body collision geometries
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.5,
        hy=0.5,
        hz=0.5,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_unary_revolute_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
) -> ModelBuilder:
    """
    Builds a world to test unary revolute joints.

    This world consists of a single rigid body connected to the world via a unary
    revolute joint, with optional limits applied to the joint degree of freedom.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degree of freedom.
        ground: Whether to include a static ground box in the world.
        dynamic: Whether to set the joint to be dynamic, with non-zero armature and friction.
        implicit_pd: Whether to use implicit PD control for the joint.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="unary_revolute_joint_test")

    # Define test system
    bid_F = _follower_body(_builder, "follower", (0.5, -0.25, z_offset))

    j_revolute = _builder.add_joint_revolute(
        label="world_to_follower_revolute",
        parent=-1,
        child=bid_F,
        axis=Axis.Y,
        parent_xform=wp.transformf(0.0, -0.15, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0),
        actuator_mode=_actuator_mode(implicit_pd),
        limit_lower=-0.25 * math.pi if limits else None,
        limit_upper=0.25 * math.pi if limits else None,
        armature=0.1 if dynamic else None,
        friction=0.01 if dynamic else None,
        target_ke=10.0 if implicit_pd else None,
        target_kd=0.01 if implicit_pd else None,
        effort_limit=100.0,
    )
    _builder.add_articulation([j_revolute])

    # Add body collision geometries
    _builder.add_shape_box(
        label="base/box",
        body=-1,
        hx=0.15,
        hy=0.15,
        hz=0.15,
        cfg=_shape_cfg_basic(collision_group=0),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.5,
        hy=0.1,
        hz=0.1,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_binary_revolute_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
) -> ModelBuilder:
    """
    Builds a world to test binary revolute joints.

    This world consists of two rigid bodies connected via a binary revolute
    joint, with optional limits applied to the joint degree of freedom. The base
    is attached to the world by a fixed joint.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degree of freedom.
        ground: Whether to include a static ground box in the world.
        dynamic: Whether to set the joint to be dynamic, with non-zero armature and friction.
        implicit_pd: Whether to use implicit PD control for the joint.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="binary_revolute_joint_test")

    # Define test system
    bid_B = _follower_body(_builder, "base", (0.0, 0.0, z_offset))
    bid_F = _follower_body(_builder, "follower", (0.5, -0.25, z_offset))

    j_fixed = _builder.add_joint_fixed(
        label="world_to_base",
        parent=-1,
        child=bid_B,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )

    j_revolute = _builder.add_joint_revolute(
        label="base_to_follower_revolute",
        parent=bid_B,
        child=bid_F,
        axis=Axis.Y,
        parent_xform=wp.transformf(0.0, -0.15, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0),
        actuator_mode=_actuator_mode(implicit_pd),
        limit_lower=-0.25 * math.pi if limits else None,
        limit_upper=0.25 * math.pi if limits else None,
        armature=0.1 if dynamic else None,
        friction=0.01 if dynamic else None,
        target_ke=10.0 if implicit_pd else None,
        target_kd=0.01 if implicit_pd else None,
        effort_limit=100.0,
    )
    _builder.add_articulation([j_fixed, j_revolute])

    # Add body collision geometries
    _builder.add_shape_box(
        label="base/box",
        body=bid_B,
        hx=0.15,
        hy=0.15,
        hz=0.15,
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.5,
        hy=0.1,
        hz=0.1,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_unary_prismatic_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
) -> ModelBuilder:
    """
    Builds a world to test unary prismatic joints.

    This world consists of a single rigid body connected to the world via a unary
    prismatic joint, with optional limits applied to the joint degree of freedom.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degree of freedom.
        ground: Whether to include a static ground box in the world.
        dynamic: Whether to set the joint to be dynamic, with non-zero armature and friction.
        implicit_pd: Whether to use implicit PD control for the joint.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="unary_prismatic_joint_test")

    # Define test system
    bid_F = _follower_body(_builder, "follower", (0.0, 0.0, z_offset))

    j_prismatic = _builder.add_joint_prismatic(
        label="world_to_follower_prismatic",
        parent=-1,
        child=bid_F,
        axis=Axis.Z,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
        actuator_mode=_actuator_mode(implicit_pd),
        limit_lower=-0.5 if limits else None,
        limit_upper=0.5 if limits else None,
        armature=0.1 if dynamic else None,
        friction=0.01 if dynamic else None,
        target_ke=10.0 if implicit_pd else None,
        target_kd=0.01 if implicit_pd else None,
        effort_limit=100.0,
    )
    _builder.add_articulation([j_prismatic])

    # Add body collision geometries
    _builder.add_shape_box(
        label="base/box",
        body=-1,
        hx=0.025,
        hy=0.025,
        hz=0.5,
        cfg=_shape_cfg_basic(collision_group=0),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.05,
        hy=0.05,
        hz=0.05,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_binary_prismatic_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
) -> ModelBuilder:
    """
    Builds a world to test binary prismatic joints.

    This world consists of two rigid bodies connected via a binary prismatic
    joint, with optional limits applied to the joint degree of freedom. The base
    is attached to the world by a fixed joint.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degree of freedom.
        ground: Whether to include a static ground box in the world.
        dynamic: Whether to set the joint to be dynamic, with non-zero armature and friction.
        implicit_pd: Whether to use implicit PD control for the joint.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="binary_prismatic_joint_test")

    # Define test system
    bid_B = _follower_body(_builder, "base", (0.0, 0.0, z_offset))
    bid_F = _follower_body(_builder, "follower", (0.0, 0.0, z_offset))

    j_fixed = _builder.add_joint_fixed(
        label="world_to_base",
        parent=-1,
        child=bid_B,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )

    j_prismatic = _builder.add_joint_prismatic(
        label="base_to_follower_prismatic",
        parent=bid_B,
        child=bid_F,
        axis=Axis.Z,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
        actuator_mode=_actuator_mode(implicit_pd),
        limit_lower=-0.5 if limits else None,
        limit_upper=0.5 if limits else None,
        armature=0.1 if dynamic else None,
        friction=0.01 if dynamic else None,
        target_ke=10.0 if implicit_pd else None,
        target_kd=0.01 if implicit_pd else None,
        effort_limit=100.0,
    )
    _builder.add_articulation([j_fixed, j_prismatic])

    # Add body collision geometries
    _builder.add_shape_box(
        label="base/box",
        body=bid_B,
        hx=0.025,
        hy=0.025,
        hz=0.5,
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.05,
        hy=0.05,
        hz=0.05,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_unary_cylindrical_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
) -> ModelBuilder:
    """
    Builds a world to test unary cylindrical joints.

    This world consists of a single rigid body connected to the world via a unary
    cylindrical joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degrees of freedom.
        ground: Whether to include a static ground box in the world.
        dynamic: Whether to set the joint to be dynamic, with non-zero armature and friction.
        implicit_pd: Whether to use implicit PD control for the joint.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="unary_cylindrical_joint_test")

    # Define test system
    bid_F = _follower_body(_builder, "follower", (0.0, 0.0, z_offset))

    lin, ang = _cylindrical_axes(Axis.Z, limits=limits, dynamic=dynamic, implicit_pd=implicit_pd)
    j_cyl = _builder.add_joint_d6(
        label="world_to_follower_cylindrical",
        parent=-1,
        child=bid_F,
        linear_axes=lin,
        angular_axes=ang,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )
    _builder.add_articulation([j_cyl])

    # Add body collision geometries
    _builder.add_shape_cylinder(
        label="base/cylinder",
        body=-1,
        radius=0.025,
        half_height=0.5,
        xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        cfg=_shape_cfg_basic(collision_group=2),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.05,
        hy=0.05,
        hz=0.05,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_binary_cylindrical_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
) -> ModelBuilder:
    """
    Builds a world to test binary cylindrical joints.

    This world consists of two rigid bodies connected via a binary cylindrical
    joint, with optional limits applied to the joint degrees of freedom. The base
    is attached to the world by a fixed joint.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degrees of freedom.
        ground: Whether to include a static ground box in the world.
        dynamic: Whether to set the joint to be dynamic, with non-zero armature and friction.
        implicit_pd: Whether to use implicit PD control for the joint.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="binary_cylindrical_joint_test")

    # Define test system
    bid_B = _follower_body(_builder, "base", (0.0, 0.0, z_offset))
    bid_F = _follower_body(_builder, "follower", (0.0, 0.0, z_offset))

    j_fixed = _builder.add_joint_fixed(
        label="world_to_base",
        parent=-1,
        child=bid_B,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )

    lin, ang = _cylindrical_axes(Axis.Z, limits=limits, dynamic=dynamic, implicit_pd=implicit_pd)
    j_cyl = _builder.add_joint_d6(
        label="base_to_follower_cylindrical",
        parent=bid_B,
        child=bid_F,
        linear_axes=lin,
        angular_axes=ang,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )
    _builder.add_articulation([j_fixed, j_cyl])

    # Add body collision geometries
    _builder.add_shape_cylinder(
        label="base/cylinder",
        body=bid_B,
        radius=0.025,
        half_height=0.5,
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.05,
        hy=0.05,
        hz=0.05,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_unary_universal_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
) -> ModelBuilder:
    """
    Builds a world to test unary universal joints.

    This world consists of a single rigid body connected to the world via a unary
    universal joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degrees of freedom.
        ground: Whether to include a static ground box in the world.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="unary_universal_joint_test")

    # Define test system
    bid_F = _follower_body(_builder, "follower", (0.5, 0.0, z_offset))

    j_universal = _builder.add_joint_d6(
        label="world_to_follower_universal",
        parent=-1,
        child=bid_F,
        angular_axes=_universal_axes(limits),
        parent_xform=wp.transformf(0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
    )
    _builder.add_articulation([j_universal])

    # Add body collision geometries
    _builder.add_shape_box(
        label="base/box",
        body=-1,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        cfg=_shape_cfg_basic(collision_group=0),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_binary_universal_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
) -> ModelBuilder:
    """
    Builds a world to test binary universal joints.

    This world consists of two rigid bodies connected via a binary universal
    joint, with optional limits applied to the joint degrees of freedom. The base
    is attached to the world by a fixed joint.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degrees of freedom.
        ground: Whether to include a static ground box in the world.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="binary_universal_joint_test")

    # Define test system
    bid_B = _follower_body(_builder, "base", (0.0, 0.0, z_offset))
    bid_F = _follower_body(_builder, "follower", (0.5, 0.0, z_offset))

    j_fixed = _builder.add_joint_fixed(
        label="world_to_base",
        parent=-1,
        child=bid_B,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )
    j_universal = _builder.add_joint_d6(
        label="base_to_follower_universal",
        parent=bid_B,
        child=bid_F,
        angular_axes=_universal_axes(limits),
        parent_xform=wp.transformf(0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
    )
    _builder.add_articulation([j_fixed, j_universal])

    # Add body collision geometries
    _builder.add_shape_box(
        label="base/box",
        body=bid_B,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_unary_spherical_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
) -> ModelBuilder:
    """
    Builds a world to test unary spherical (ball) joints.

    This world consists of a single rigid body connected to the world via a unary
    spherical joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degrees of freedom.
        ground: Whether to include a static ground box in the world.

    Returns:
        The populated :class:`ModelBuilder`.

    Notes:
        :meth:`ModelBuilder.add_joint_ball` does not expose per-axis limits, so
        when ``limits=True`` the limits are patched directly on the joint's
        angular DoF entries after creation to mirror Kamino behavior.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="unary_spherical_joint_test")

    # Define test system
    bid_F = _follower_body(_builder, "follower", (0.5, 0.0, z_offset))

    j_spherical = _builder.add_joint_ball(
        label="world_to_follower_spherical",
        parent=-1,
        child=bid_F,
        parent_xform=wp.transformf(0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
        actuator_mode=JointTargetMode.EFFORT,
    )
    _builder.add_articulation([j_spherical])

    # Patch per-axis angular limits post-creation (not exposed by add_joint_ball)
    if limits:
        _apply_angular_limits(_builder, j_spherical, [(-0.6 * math.pi, 0.6 * math.pi)] * 3)

    # Add body collision geometries
    _builder.add_shape_box(
        label="base/box",
        body=-1,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        cfg=_shape_cfg_basic(collision_group=0),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_binary_spherical_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
) -> ModelBuilder:
    """
    Builds a world to test binary spherical (ball) joints.

    This world consists of two rigid bodies connected via a binary spherical
    joint, with optional limits applied to the joint degrees of freedom. The base
    is attached to the world by a fixed joint.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degrees of freedom.
        ground: Whether to include a static ground box in the world.

    Returns:
        The populated :class:`ModelBuilder`.

    Notes:
        :meth:`ModelBuilder.add_joint_ball` does not expose per-axis limits, so
        when ``limits=True`` the limits are patched directly on the joint's
        angular DoF entries after creation to mirror Kamino behavior.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="binary_spherical_joint_test")

    # Define test system
    bid_B = _follower_body(_builder, "base", (0.0, 0.0, z_offset))
    bid_F = _follower_body(_builder, "follower", (0.5, 0.0, z_offset))

    j_fixed = _builder.add_joint_fixed(
        label="world_to_base",
        parent=-1,
        child=bid_B,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )
    j_spherical = _builder.add_joint_ball(
        label="base_to_follower_spherical",
        parent=bid_B,
        child=bid_F,
        parent_xform=wp.transformf(0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
        actuator_mode=JointTargetMode.EFFORT,
    )
    _builder.add_articulation([j_fixed, j_spherical])

    # Patch per-axis angular limits post-creation (not exposed by add_joint_ball)
    if limits:
        _apply_angular_limits(_builder, j_spherical, [(-0.6 * math.pi, 0.6 * math.pi)] * 3)

    # Add body collision geometries
    _builder.add_shape_box(
        label="base/box",
        body=bid_B,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_unary_cartesian_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
) -> ModelBuilder:
    """
    Builds a world to test unary cartesian joints.

    This world consists of a single rigid body connected to the world via a unary
    cartesian joint (three translational DoFs), with optional limits applied to the
    joint degrees of freedom.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degrees of freedom.
        ground: Whether to include a static ground box in the world.
        dynamic: Whether to set the joint to be dynamic, with non-zero armature and friction.
        implicit_pd: Whether to use implicit PD control for the joint.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="unary_cartesian_joint_test")

    # Define test system
    bid_F = _follower_body(_builder, "follower", (0.5, 0.0, z_offset))

    j_cartesian = _builder.add_joint_d6(
        label="world_to_follower_cartesian",
        parent=-1,
        child=bid_F,
        linear_axes=_cartesian_axes(limits=limits, dynamic=dynamic, implicit_pd=implicit_pd),
        parent_xform=wp.transformf(0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
    )
    _builder.add_articulation([j_cartesian])

    # Add body collision geometries
    _builder.add_shape_box(
        label="base/box",
        body=-1,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        cfg=_shape_cfg_basic(collision_group=0),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


def build_binary_cartesian_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
) -> ModelBuilder:
    """
    Builds a world to test binary cartesian joints.

    This world consists of two rigid bodies connected via a binary cartesian
    joint (three translational DoFs), with optional limits applied to the joint
    degrees of freedom. The base is attached to the world by a fixed joint.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the entities will be added.
        z_offset: A vertical offset to apply to the rigid body position [m].
        new_world: Whether to open a new world in the builder, to which entities will be added.
            If ``False``, the caller must already be inside an active
            ``begin_world`` / ``end_world`` context on ``builder``.
            If ``True`` (or ``builder`` is ``None``), the builder's
            ``begin_world`` / ``end_world`` pair is issued around the scene.
        limits: Whether to enable limits on the joint degrees of freedom.
        ground: Whether to include a static ground box in the world.
        dynamic: Whether to set the joint to be dynamic, with non-zero armature and friction.
        implicit_pd: Whether to use implicit PD control for the joint.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Open a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="binary_cartesian_joint_test")

    # Define test system
    bid_B = _follower_body(_builder, "base", (0.0, 0.0, z_offset))
    bid_F = _follower_body(_builder, "follower", (0.5, 0.0, z_offset))

    j_fixed = _builder.add_joint_fixed(
        label="world_to_base",
        parent=-1,
        child=bid_B,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )
    j_cartesian = _builder.add_joint_d6(
        label="base_to_follower_cartesian",
        parent=bid_B,
        child=bid_F,
        linear_axes=_cartesian_axes(limits=limits, dynamic=dynamic, implicit_pd=implicit_pd),
        parent_xform=wp.transformf(0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-0.25, -0.25, -0.25, 0.0, 0.0, 0.0, 1.0),
    )
    _builder.add_articulation([j_fixed, j_cartesian])

    # Add body collision geometries
    _builder.add_shape_box(
        label="base/box",
        body=bid_B,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid_F,
        hx=0.25,
        hy=0.25,
        hz=0.25,
        cfg=_shape_cfg_basic(),
    )

    # Add a static ground box beneath the scene
    if ground:
        _add_ground_box(_builder)

    # Close the world if it was opened here
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated builder
    return _builder


###
# Aggregate builder
###


def build_all_joints_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    ground: bool = False,
) -> ModelBuilder:
    """
    Constructs a model builder containing a world for each joint type.

    Args:
        builder: An optional existing :class:`ModelBuilder` to which the joint-type
            worlds will be added. If ``None``, a new builder is created.
        z_offset: A vertical offset to apply to the initial position of the follower
            bodies in each scene [m].
        ground: Whether to include a static ground box in each scene.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    # Create a new builder to populate if none is provided
    if builder is not None:
        _builder = builder
    else:
        _builder = ModelBuilder()

    builder_kwargs = {"z_offset": z_offset, "ground": ground}

    # Add a new world for each joint type
    _builder.add_world(build_free_joint_test(**builder_kwargs))
    _builder.add_world(build_unary_revolute_joint_test(**builder_kwargs))
    _builder.add_world(build_binary_revolute_joint_test(**builder_kwargs))
    _builder.add_world(build_unary_prismatic_joint_test(**builder_kwargs))
    _builder.add_world(build_binary_prismatic_joint_test(**builder_kwargs))
    _builder.add_world(build_unary_cylindrical_joint_test(**builder_kwargs))
    _builder.add_world(build_binary_cylindrical_joint_test(**builder_kwargs))
    _builder.add_world(build_unary_universal_joint_test(**builder_kwargs))
    _builder.add_world(build_binary_universal_joint_test(**builder_kwargs))
    _builder.add_world(build_unary_spherical_joint_test(**builder_kwargs))
    _builder.add_world(build_binary_spherical_joint_test(**builder_kwargs))
    _builder.add_world(build_unary_cartesian_joint_test(**builder_kwargs))
    _builder.add_world(build_binary_cartesian_joint_test(**builder_kwargs))

    # Return the populated builder
    return _builder


###
# Builders - Geometry Tests
###


shape_name_to_type: dict[str, GeoType] = {
    "sphere": GeoType.SPHERE,
    "cylinder": GeoType.CYLINDER,
    "cone": GeoType.CONE,
    "capsule": GeoType.CAPSULE,
    "box": GeoType.BOX,
    "ellipsoid": GeoType.ELLIPSOID,
    "plane": GeoType.PLANE,
}
"""Mapping from shape name to GeoType enum."""


shape_default_dims: dict[GeoType, tuple] = {
    GeoType.SPHERE: (0.5,),
    GeoType.CYLINDER: (0.5, 0.5),
    GeoType.CONE: (0.5, 0.5),
    GeoType.CAPSULE: (0.5, 0.5),
    GeoType.BOX: (0.5, 0.5, 0.5),
    GeoType.ELLIPSOID: (1.0, 1.0, 0.5),
    GeoType.PLANE: (0.0, 0.0, 1.0, 0.0),
}
"""Mapping from GeoType enum to default dimensions (Newton convention: half-extents)."""


_SHAPE_EXPECTED_LEN: dict[GeoType, int] = {
    GeoType.SPHERE: 1,
    GeoType.CYLINDER: 2,
    GeoType.CONE: 2,
    GeoType.CAPSULE: 2,
    GeoType.BOX: 3,
    GeoType.ELLIPSOID: 3,
    GeoType.PLANE: 4,
}


def make_shape_initial_position(type: GeoType, dims: tuple, is_top: bool = True) -> wp.vec3:
    """
    Computes the initial position along the z-axis for a given shape.

    This function calculates the position required to place a shape just above
    (or below) the origin along the z-axis, based on its type and dimensions.

    Args:
        type: The :class:`GeoType` of the shape (e.g. ``GeoType.SPHERE``,
            ``GeoType.BOX``, ``GeoType.CAPSULE``, etc.).
        dims: The dimensions of the shape. The expected format depends on the
            shape type; see :data:`shape_default_dims` for the Newton convention
            (half-extents / half-heights).
        is_top: If ``True``, computes the position for a top shape (above the
            origin). If ``False``, computes the position for a bottom shape
            (below the origin).

    Returns:
        The computed position vector along the z-axis.
    """
    # Check that the shape type is a known GeoType
    if type not in _SHAPE_EXPECTED_LEN:
        raise ValueError(f"Unsupported shape type: {type}")

    # Validate the dimension count against the expected length for this type
    expected_len = _SHAPE_EXPECTED_LEN[type]
    if len(dims) != expected_len:
        raise ValueError(f"Invalid dimensions for shape '{type}': expected {expected_len} values, got {len(dims)}")

    # Compute the initial position along z-axis that places the shape just above.
    # Dimensions use Newton convention (half-extents, half-heights).
    if type == GeoType.SPHERE:
        r = wp.vec3(0.0, 0.0, dims[0])
    elif type == GeoType.BOX:
        r = wp.vec3(0.0, 0.0, dims[2])
    elif type == GeoType.CAPSULE:
        r = wp.vec3(0.0, 0.0, dims[1] + dims[0])
    elif type == GeoType.CYLINDER:
        r = wp.vec3(0.0, 0.0, dims[1])
    elif type == GeoType.CONE:
        r = wp.vec3(0.0, 0.0, dims[1])
    elif type == GeoType.ELLIPSOID:
        r = wp.vec3(0.0, 0.0, dims[2])
    elif type == GeoType.PLANE:
        r = wp.vec3(0.0, 0.0, dims[3])
    else:
        raise ValueError(f"Unsupported shape type: {type}")

    # Invert the position if it's the bottom shape
    if not is_top:
        r = -r

    # Return the computed position
    return r


def get_shape_bottom_position(center: wp.vec3, type: GeoType, dims: tuple) -> wp.vec3:
    """
    Computes the position of the bottom along the z-axis for a given shape.

    Args:
        center: The center position of the shape [m].
        type: The :class:`GeoType` of the shape (e.g. ``GeoType.SPHERE``,
            ``GeoType.BOX``, ``GeoType.CAPSULE``, etc.).
        dims: The dimensions of the shape; same convention as
            :data:`shape_default_dims` (half-extents / half-heights).

    Returns:
        The computed bottom position of the shape along the z-axis.
    """
    # Compute and return the bottom position along z-axis.
    # Dimensions use Newton convention (half-extents, half-heights).
    if type == GeoType.SPHERE:
        return center - wp.vec3(0.0, 0.0, dims[0])
    if type == GeoType.BOX:
        return center - wp.vec3(0.0, 0.0, dims[2])
    if type == GeoType.CAPSULE:
        return center - wp.vec3(0.0, 0.0, dims[1] + dims[0])
    if type == GeoType.CYLINDER:
        return center - wp.vec3(0.0, 0.0, dims[1])
    if type == GeoType.CONE:
        return center - wp.vec3(0.0, 0.0, dims[1])
    if type == GeoType.ELLIPSOID:
        return center - wp.vec3(0.0, 0.0, dims[2])
    if type == GeoType.PLANE:
        return center - wp.vec3(0.0, 0.0, dims[3])
    raise ValueError(f"Unsupported shape type: {type}")


def _add_shape_to_body(
    builder: ModelBuilder,
    name: str,
    body: int,
    label: str,
    dims: tuple,
) -> int:
    """Dispatch to the appropriate ``add_shape_*`` method based on the shape name."""
    cfg = _shape_cfg_basic()
    if name == "sphere":
        return builder.add_shape_sphere(body=body, radius=dims[0], label=label, cfg=cfg)
    if name == "box":
        return builder.add_shape_box(body=body, hx=dims[0], hy=dims[1], hz=dims[2], label=label, cfg=cfg)
    if name == "capsule":
        return builder.add_shape_capsule(body=body, radius=dims[0], half_height=dims[1], label=label, cfg=cfg)
    if name == "cylinder":
        return builder.add_shape_cylinder(body=body, radius=dims[0], half_height=dims[1], label=label, cfg=cfg)
    if name == "cone":
        return builder.add_shape_cone(body=body, radius=dims[0], half_height=dims[1], label=label, cfg=cfg)
    if name == "ellipsoid":
        return builder.add_shape_ellipsoid(body=body, rx=dims[0], ry=dims[1], rz=dims[2], label=label, cfg=cfg)
    if name == "plane":
        return builder.add_shape_plane(body=body, plane=(dims[0], dims[1], dims[2], dims[3]), label=label, cfg=cfg)
    raise ValueError(f"Unsupported shape type: {name}")


def _add_ground_box_offset(builder: ModelBuilder, z_offset: float) -> None:
    """Add a static ground box with its top surface at ``z_offset``."""
    builder.add_shape_box(
        label="ground",
        body=-1,
        hx=2.5,
        hy=2.5,
        hz=0.5,
        xform=wp.transformf(0.0, 0.0, -0.5 + z_offset, 0.0, 0.0, 0.0, 1.0),
        cfg=_shape_cfg_basic(),
    )


def _add_ground_plane_offset(builder: ModelBuilder, z_offset: float) -> None:
    """Add a static ground plane at ``z_offset``."""
    builder.add_shape_plane(
        label="ground",
        body=-1,
        plane=(0.0, 0.0, 1.0, -z_offset),
        cfg=_shape_cfg_basic(),
    )


def make_single_shape_pair_builder(
    shapes: tuple[str, str],
    bottom_dims: tuple | None = None,
    bottom_xyz: tuple | None = None,
    bottom_rpy: tuple | None = None,
    top_dims: tuple | None = None,
    top_xyz: tuple | None = None,
    top_rpy: tuple | None = None,
    distance: float = 0.0,
    ground_box: bool = False,
    ground_plane: bool = False,
    ground_z: float | None = None,
) -> ModelBuilder:
    """
    Generates a :class:`ModelBuilder` for a given shape combination with specified parameters.

    The first shape in the combination is placed above the second shape along the
    z-axis, effectively generating a ``shapes[0]`` atop ``shapes[1]`` configuration.

    Args:
        shapes: A tuple specifying the names of the top and bottom shapes, in
            that order (e.g. ``("box", "sphere")`` places a box on top of a sphere).
        bottom_dims: Dimensions for the bottom shape. If ``None``, defaults are
            taken from :data:`shape_default_dims`.
        bottom_xyz: Position (x, y, z) for the bottom shape [m]. If ``None``,
            defaults to just below the origin along z.
        bottom_rpy: Orientation (roll, pitch, yaw) for the bottom shape [rad].
            If ``None``, defaults to ``(0, 0, 0)``.
        top_dims: Dimensions for the top shape. If ``None``, defaults are taken
            from :data:`shape_default_dims`.
        top_xyz: Position (x, y, z) for the top shape [m]. If ``None``, defaults
            to just above the origin along z.
        top_rpy: Orientation (roll, pitch, yaw) for the top shape [rad]. If
            ``None``, defaults to ``(0, 0, 0)``.
        distance: Mutual distance along the z-axis between the two shapes [m].
            If zero, the shapes are exactly touching.
            If positive, they are separated by that distance.
            If negative, they are penetrating by that distance.
        ground_box: Whether to add a static ground box below the bottom shape.
        ground_plane: Whether to add a static ground plane below the bottom shape.
        ground_z: Explicit z position for the ground; if ``None``, defaults to
            the bottom shape's bottom position along z.

    Returns:
        The constructed :class:`ModelBuilder` with the specified shape combination.
    """
    # Check that the shape combination is a tuple of strings
    if not (isinstance(shapes, tuple) and len(shapes) == 2 and all(isinstance(s, str) for s in shapes)):
        raise ValueError(f"Shape combination must be a tuple of two strings: {shapes}")

    # Check that each shape name is valid
    for shape_name in shapes:
        if shape_name not in shape_name_to_type:
            raise ValueError(f"Unsupported shape name: {shape_name}")

    # Define top and bottom shape names
    top_name = shapes[0]
    bottom_name = shapes[1]

    # Retrieve shape types
    top_shape = shape_name_to_type[top_name]
    bottom_shape = shape_name_to_type[bottom_name]

    # Define default arguments for those not provided
    if bottom_dims is None:
        bottom_dims = shape_default_dims[bottom_shape]
    if bottom_xyz is None:
        bottom_xyz = tuple(make_shape_initial_position(bottom_shape, bottom_dims, is_top=False))
    if bottom_rpy is None:
        bottom_rpy = (0.0, 0.0, 0.0)
    if top_dims is None:
        top_dims = shape_default_dims[top_shape]
    if top_xyz is None:
        top_xyz = tuple(make_shape_initial_position(top_shape, top_dims, is_top=True))
    if top_rpy is None:
        top_rpy = (0.0, 0.0, 0.0)

    # Compute positions and orientations, shifting each shape by half the mutual distance
    dz = 0.5 * distance
    r_b = wp.vec3(bottom_xyz[0], bottom_xyz[1], bottom_xyz[2] - dz)
    q_b = wp.quat_rpy(float(bottom_rpy[0]), float(bottom_rpy[1]), float(bottom_rpy[2]))
    r_t = wp.vec3(top_xyz[0], top_xyz[1], top_xyz[2] + dz)
    q_t = wp.quat_rpy(float(top_rpy[0]), float(top_rpy[1]), float(top_rpy[2]))

    # Create the builder and open a world for the shape pair
    builder = ModelBuilder()
    builder.begin_world(label=f"{top_name}_on_{bottom_name}")

    # Add bodies for the bottom and top shapes
    bid0 = builder.add_body(
        label="bottom_" + bottom_name,
        mass=1.0,
        inertia=_identity_inertia(),
        xform=wp.transform(r_b, q_b),
        lock_inertia=True,
    )
    bid1 = builder.add_body(
        label="top_" + top_name,
        mass=1.0,
        inertia=_identity_inertia(),
        xform=wp.transform(r_t, q_t),
        lock_inertia=True,
    )

    # Attach the corresponding collision geometries to each body
    _add_shape_to_body(builder, bottom_name, bid0, "bottom_" + bottom_name, bottom_dims)
    _add_shape_to_body(builder, top_name, bid1, "top_" + top_name, top_dims)

    # Optionally add a ground geometry below the bottom shape
    if ground_box or ground_plane:
        if ground_z is not None:
            z_g_offset = float(ground_z)
        else:
            bottom_center_z = float(bottom_xyz[2]) - dz
            bottom_bottom = get_shape_bottom_position(wp.vec3(0.0, 0.0, bottom_center_z), bottom_shape, bottom_dims)
            z_g_offset = float(bottom_bottom[2])
        if ground_box:
            _add_ground_box_offset(builder, z_offset=z_g_offset)
        if ground_plane:
            _add_ground_plane_offset(builder, z_offset=z_g_offset)

    builder.end_world()

    # Return the constructed builder
    return builder


def build_shape_pairs_test(
    shape_pairs: list[tuple[str, str]],
    builder: ModelBuilder | None = None,
    per_shape_pair_args: dict | None = None,
    distance: float | None = None,
    ground_box: bool = False,
    ground_plane: bool = False,
    ground_z: float | None = None,
) -> ModelBuilder:
    """
    Generates a builder containing a world for each specified shape combination.

    Args:
        shape_pairs: A list of tuples specifying the names of the top and bottom
            shapes for each combination (e.g. ``[("box", "sphere"), ("cylinder", "cone")]``).
        builder: An optional existing :class:`ModelBuilder` to which the new
            worlds will be added. If ``None``, a new builder is created.
        per_shape_pair_args: Optional per-pair keyword overrides, keyed by the
            ``shapes`` tuple, forwarded to :func:`make_single_shape_pair_builder`.
        distance: Mutual z-axis distance between the two shapes [m], applied to
            every pair. If ``None``, the per-pair or default value is used.
        ground_box: Whether to add a static ground box below the bottom shape in
            each world.
        ground_plane: Whether to add a static ground plane below the bottom
            shape in each world.
        ground_z: Explicit z position for the ground in each world; if ``None``,
            defaults to the bottom shape's bottom position along z.

    Returns:
        A :class:`ModelBuilder` with one world per shape combination.
    """
    # Create a new builder if none is provided
    if builder is not None:
        _builder = builder
    else:
        _builder = ModelBuilder()

    # Iterate over each shape pair and add its world to the main builder
    for shapes in shape_pairs:
        # Pick up per-pair arguments if provided
        if per_shape_pair_args is not None:
            shape_pair_args = dict(per_shape_pair_args.get(shapes, {}))
        else:
            shape_pair_args = {}

        # Override distance if specified
        if distance is not None:
            shape_pair_args["distance"] = distance

        # Create the single shape pair world and add it to the main builder
        _builder.add_world(
            make_single_shape_pair_builder(
                shapes,
                ground_box=ground_box,
                ground_plane=ground_plane,
                ground_z=ground_z,
                **shape_pair_args,
            )
        )

    # Return the populated builder
    return _builder
