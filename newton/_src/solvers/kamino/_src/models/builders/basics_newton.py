# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Factory methods for building 'basic' models.

This module provides a set of functions to create simple mechanical assemblies
using the :class:`newton.ModelBuilder` interface. These include fundamental
configurations such as a box on a plane, a box pendulum, a cartpole, and various
linked box systems.

Each function constructs a specific model by adding rigid bodies, joints, and
collision geometries to a :class:`newton.ModelBuilder` instance. The models are
designed to serve as foundational examples for testing and demonstration
purposes, and each features a certain subset of ill-conditioned dynamics.

World context:
    Unlike :class:`ModelBuilderKamino`, Newton has no ``world_index`` argument.
    When ``new_world`` is ``False``, the caller must already be inside an active
    world (i.e. between :meth:`ModelBuilder.begin_world` and
    :meth:`ModelBuilder.end_world`).
"""

from __future__ import annotations

import math

import warp as wp

from ......core import Axis
from ......sim import JointTargetMode, ModelBuilder
from ...core import inertia
from ...core.joints import JOINT_QMAX, JOINT_QMIN

###
# Module interface
###

__all__ = [
    "build_box_on_plane",
    "build_box_pendulum",
    "build_box_pendulum_vertical",
    "build_boxes_fourbar",
    "build_boxes_hinged",
    "build_boxes_nunchaku",
    "build_boxes_nunchaku_vertical",
    "build_cartpole",
    "make_basics_heterogeneous_builder",
]


###
# Helpers
###


def _shape_cfg_basic() -> ModelBuilder.ShapeConfig:
    """Default shape config matching the Kamino ``basics`` factories.

    Uses zero contact margin and zero contact gap so that collision geometry
    dimensions map one-to-one to the original Kamino ``BoxShape`` / ``SphereShape``
    half-extents.
    """
    return ModelBuilder.ShapeConfig(margin=0.0, gap=0.0)


def _add_ground_box(builder: ModelBuilder) -> None:
    """Add a static collision geometry for the ground plane.

    The ground is modelled as a large, thick, static box attached to the world
    body (``body=-1``), matching the convention used by the Kamino ``basics``
    factories.
    """
    builder.add_shape_box(
        label="ground",
        body=-1,
        hx=10.0,
        hy=10.0,
        hz=0.5,
        xform=wp.transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
        cfg=_shape_cfg_basic(),
    )


###
# Functions
###


def build_box_on_plane(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    ground: bool = True,
    new_world: bool = True,
) -> ModelBuilder:
    """
    Constructs a basic model of a free-floating 'box' body and a ground box geom.

    Args:
        builder (ModelBuilder | None):
            An optional existing model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float):
            A vertical offset to apply to the initial position of the box.
        ground (bool):
            Whether to add a static ground plane to the model.
        new_world (bool):
            Whether to begin a new world in the builder for this model.\n
            If `True` (or `builder` is `None`), the model is wrapped in a new world context
            opened via :meth:`ModelBuilder.begin_world` and closed via
            :meth:`ModelBuilder.end_world`.\n
            If `False`, the caller must already be inside an active world; the model is then
            added to that currently active world.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Begin a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="box_on_plane")

    # Add first body
    i_I = inertia.solid_cuboid_body_moment_of_inertia(1.0, 0.2, 0.2, 0.2)
    xform = wp.transformf(0.0, 0.0, 0.1 + z_offset, 0.0, 0.0, 0.0, 1.0)
    bid0 = _builder.add_body(
        label="box",
        mass=1.0,
        inertia=i_I,
        xform=xform,
        lock_inertia=True,
    )

    # Add collision geometries
    _builder.add_shape_box(
        label="box_geom",
        body=bid0,
        hx=0.1,
        hy=0.1,
        hz=0.1,
        cfg=_shape_cfg_basic(),
    )

    # Add a static collision geometry for the plane
    if ground:
        _add_ground_box(_builder)

    # Close the world context if we opened one
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated model builder
    return _builder


def build_box_pendulum(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.7,
    ground: bool = True,
    new_world: bool = True,
    dynamic_joints: bool = False,
    implicit_pd: bool = False,
) -> ModelBuilder:
    """
    Constructs a basic model of a single box pendulum body with a unary revolute joint.

    This version initializes the pendulum in a horizontal configuration.

    Args:
        builder (ModelBuilder | None):
            An optional existing model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float):
            A vertical offset to apply to the initial position of the box.
        ground (bool):
            Whether to add a static ground plane to the model.
        new_world (bool):
            Whether to begin a new world in the builder for this model.\n
            If `True` (or `builder` is `None`), the model is wrapped in a new world context
            opened via :meth:`ModelBuilder.begin_world` and closed via
            :meth:`ModelBuilder.end_world`.\n
            If `False`, the caller must already be inside an active world; the model is then
            added to that currently active world.
        dynamic_joints (bool):
            Whether to attach non-zero armature and friction terms to the revolute joint
            so that its dynamics are better conditioned for stiff integrators.
        implicit_pd (bool):
            Whether to configure the revolute joint with a position/velocity target mode
            (implicit PD) instead of the default effort-based actuation.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Begin a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="box_pendulum")

    # Model constants
    m = 1.0
    d = 0.5
    w = 0.1
    h = 0.1
    z_0 = z_offset  # Initial z offset for the body

    # Add box pendulum body
    i_I = inertia.solid_cuboid_body_moment_of_inertia(m, d, w, h)
    q_i = wp.transformf(0.5 * d, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0)
    bid0 = _builder.add_link(
        label="pendulum",
        mass=m,
        inertia=i_I,
        xform=q_i,
        lock_inertia=True,
    )

    # Build the joint DoF config (implicit PD vs. effort actuation)
    if implicit_pd:
        axis_cfg = ModelBuilder.JointDofConfig(
            axis=Axis.Y,
            actuator_mode=JointTargetMode.POSITION_VELOCITY,
            target_ke=100.0,
            target_kd=1.0,
            armature=1.0 if dynamic_joints else 0.0,
            friction=0.1 if dynamic_joints else 0.0,
        )
    else:
        axis_cfg = ModelBuilder.JointDofConfig(
            axis=Axis.Y,
            actuator_mode=JointTargetMode.EFFORT,
            armature=1.0 if dynamic_joints else 0.0,
            friction=0.1 if dynamic_joints else 0.0,
        )

    # Add a revolute joint between the world and the pendulum body
    j0 = _builder.add_joint_revolute(
        label="world_to_pendulum",
        parent=-1,
        child=bid0,
        axis=axis_cfg,
        parent_xform=wp.transformf(0.0, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-0.5 * d, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    _builder.add_articulation([j0])

    # Add collision geometries
    _builder.add_shape_box(
        label="box",
        body=bid0,
        hx=0.5 * d,
        hy=0.5 * w,
        hz=0.5 * h,
        cfg=_shape_cfg_basic(),
    )

    # Add a static collision geometry for the plane
    if ground:
        _add_ground_box(_builder)

    # Close the world context if we opened one
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated model builder
    return _builder


def build_box_pendulum_vertical(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.7,
    ground: bool = True,
    new_world: bool = True,
) -> ModelBuilder:
    """
    Constructs a basic model of a single box pendulum body with a unary revolute joint.

    This version initializes the pendulum in a vertical configuration.

    Args:
        builder (ModelBuilder | None):
            An optional existing model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float):
            A vertical offset to apply to the initial position of the box.
        ground (bool):
            Whether to add a static ground plane to the model.
        new_world (bool):
            Whether to begin a new world in the builder for this model.\n
            If `True` (or `builder` is `None`), the model is wrapped in a new world context
            opened via :meth:`ModelBuilder.begin_world` and closed via
            :meth:`ModelBuilder.end_world`.\n
            If `False`, the caller must already be inside an active world; the model is then
            added to that currently active world.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Begin a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="box_pendulum_vertical")

    # Model constants
    m = 1.0
    d = 0.1
    w = 0.1
    h = 0.5
    z_0 = z_offset  # Initial z offset for the body

    # Add box pendulum body
    i_I = inertia.solid_cuboid_body_moment_of_inertia(m, d, w, h)
    q_i = wp.transformf(0.0, 0.0, -0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0)
    bid0 = _builder.add_link(
        label="pendulum",
        mass=m,
        inertia=i_I,
        xform=q_i,
        lock_inertia=True,
    )

    # Add a revolute joint between the world and the pendulum body
    axis_cfg = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=JointTargetMode.EFFORT,
    )
    j0 = _builder.add_joint_revolute(
        label="world_to_pendulum",
        parent=-1,
        child=bid0,
        axis=axis_cfg,
        parent_xform=wp.transformf(0.0, 0.0, z_0, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(0.0, 0.0, 0.5 * h, 0.0, 0.0, 0.0, 1.0),
    )
    _builder.add_articulation([j0])

    # Add collision geometries
    _builder.add_shape_box(
        label="box",
        body=bid0,
        hx=0.5 * d,
        hy=0.5 * w,
        hz=0.5 * h,
        cfg=_shape_cfg_basic(),
    )

    # Add a static collision geometry for the plane
    if ground:
        _add_ground_box(_builder)

    # Close the world context if we opened one
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated model builder
    return _builder


def build_cartpole(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    ground: bool = True,
    new_world: bool = True,
    limits: bool = True,
) -> ModelBuilder:
    """
    Constructs a basic model of a cartpole mounted onto a rail.

    Args:
        builder (ModelBuilder | None):
            An optional existing model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float):
            A vertical offset to apply to the initial position of the box.
        ground (bool):
            Whether to add a static ground plane to the model.
        new_world (bool):
            Whether to begin a new world in the builder for this model.\n
            If `True` (or `builder` is `None`), the model is wrapped in a new world context
            opened via :meth:`ModelBuilder.begin_world` and closed via
            :meth:`ModelBuilder.end_world`.\n
            If `False`, the caller must already be inside an active world; the model is then
            added to that currently active world.
        limits (bool):
            Whether to apply finite position limits on the prismatic rail joint.\n
            If `True`, the cart is restricted to the range `[-4, 4]` along the rail.\n
            If `False`, the joint limits are set to the largest representable float32 range.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Begin a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="cartpole")

    # Model constants
    m_cart = 1.0
    m_pole = 0.2
    dims_rail = (0.03, 8.0, 0.03)  # full dimensions (used for inertia, positions)
    dims_cart = (0.2, 0.5, 0.2)
    dims_pole = (0.05, 0.05, 0.75)
    half_dims_rail = (0.5 * dims_rail[0], 0.5 * dims_rail[1], 0.5 * dims_rail[2])
    half_dims_cart = (0.5 * dims_cart[0], 0.5 * dims_cart[1], 0.5 * dims_cart[2])
    half_dims_pole = (0.5 * dims_pole[0], 0.5 * dims_pole[1], 0.5 * dims_pole[2])

    # Add box cart body
    bid0 = _builder.add_link(
        label="cart",
        mass=m_cart,
        inertia=inertia.solid_cuboid_body_moment_of_inertia(m_cart, *dims_cart),
        xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )

    # Add box pole body
    x_0_pole = 0.5 * dims_pole[0] + 0.5 * dims_cart[0]
    z_0_pole = 0.5 * dims_pole[2] + z_offset
    bid1 = _builder.add_link(
        label="pole",
        mass=m_pole,
        inertia=inertia.solid_cuboid_body_moment_of_inertia(m_pole, *dims_pole),
        xform=wp.transformf(x_0_pole, 0.0, z_0_pole, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )

    # Prismatic rail limits
    if limits:
        p_lo, p_hi = -4.0, 4.0
    else:
        p_lo, p_hi = float(JOINT_QMIN), float(JOINT_QMAX)

    # Add a prismatic joint for the cart
    prism_axis = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=JointTargetMode.EFFORT,
        limit_lower=p_lo,
        limit_upper=p_hi,
        effort_limit=1000.0,
    )
    j0 = _builder.add_joint_prismatic(
        label="rail_to_cart",
        parent=-1,
        child=bid0,
        axis=prism_axis,
        parent_xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )

    # Add a revolute joint for the pendulum
    rev_passive = ModelBuilder.JointDofConfig(
        axis=Axis.X,
        actuator_mode=JointTargetMode.NONE,
    )
    j1 = _builder.add_joint_revolute(
        label="cart_to_pole",
        parent=bid0,
        child=bid1,
        axis=rev_passive,
        parent_xform=wp.transformf(0.5 * dims_cart[0], 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(
            -0.5 * dims_pole[0],
            0.0,
            -0.5 * dims_pole[2],
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )
    _builder.add_articulation([j0, j1])

    # Add collision geometries
    _builder.add_shape_box(
        label="cart",
        body=bid0,
        hx=half_dims_cart[0],
        hy=half_dims_cart[1],
        hz=half_dims_cart[2],
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="pole",
        body=bid1,
        hx=half_dims_pole[0],
        hy=half_dims_pole[1],
        hz=half_dims_pole[2],
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="rail",
        body=-1,
        hx=half_dims_rail[0],
        hy=half_dims_rail[1],
        hz=half_dims_rail[2],
        xform=wp.transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        cfg=ModelBuilder.ShapeConfig(margin=0.0, gap=0.0, collision_group=0),
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_shape_box(
            label="ground",
            body=-1,
            hx=10.0,
            hy=10.0,
            hz=0.5,
            xform=wp.transformf(0.0, 0.0, -1.0 + z_offset, 0.0, 0.0, 0.0, 1.0),
            cfg=_shape_cfg_basic(),
        )

    # Close the world context if we opened one
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated model builder
    return _builder


def build_boxes_hinged(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    ground: bool = True,
    dynamic_joints: bool = False,
    implicit_pd: bool = False,
    new_world: bool = True,
) -> ModelBuilder:
    """
    Constructs a basic model of a two floating boxes connected via revolute joint.

    .. note::
        The Kamino version of this model has no explicit world joint; Newton requires
        each articulation to be rooted at the world, so a free joint is inserted
        between the world and the ``base`` body to form a valid articulation tree.

    Args:
        builder (ModelBuilder | None):
            An optional existing model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float):
            A vertical offset to apply to the initial position of the box.
        ground (bool):
            Whether to add a static ground plane to the model.
        dynamic_joints (bool):
            Whether to attach non-zero armature and friction terms to the hinge joint
            so that its dynamics are better conditioned for stiff integrators.
        implicit_pd (bool):
            Whether to configure the hinge joint with a position/velocity target mode
            (implicit PD) instead of the default effort-based actuation.
        new_world (bool):
            Whether to begin a new world in the builder for this model.\n
            If `True` (or `builder` is `None`), the model is wrapped in a new world context
            opened via :meth:`ModelBuilder.begin_world` and closed via
            :meth:`ModelBuilder.end_world`.\n
            If `False`, the caller must already be inside an active world; the model is then
            added to that currently active world.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Begin a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="boxes_hinged")

    # Model constants
    m_0 = 1.0
    m_1 = 1.0
    d = 0.5
    w = 0.1
    h = 0.1
    z0 = z_offset  # Initial z offset for the bodies

    # Add first body
    bid0 = _builder.add_link(
        label="base",
        mass=m_0,
        inertia=inertia.solid_cuboid_body_moment_of_inertia(m_0, d, w, h),
        xform=wp.transformf(0.25, -0.05, 0.05 + z0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )

    # Add second body
    bid1 = _builder.add_link(
        label="follower",
        mass=m_1,
        inertia=inertia.solid_cuboid_body_moment_of_inertia(m_1, d, w, h),
        xform=wp.transformf(0.75, 0.05, 0.05 + z0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )

    # Attach the base to the world with a free joint so the chain is a valid
    # Newton articulation (Kamino's version implicitly treats unconnected bodies as free)
    jf = _builder.add_joint_free(
        label="world_to_base",
        parent=-1,
        child=bid0,
        parent_xform=wp.transform_identity(dtype=wp.float32),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )

    # Build the hinge joint DoF config (implicit PD vs. effort actuation)
    if implicit_pd:
        hinge_axis = ModelBuilder.JointDofConfig(
            axis=Axis.Y,
            actuator_mode=JointTargetMode.POSITION_VELOCITY,
            target_ke=100.0,
            target_kd=1.0,
            armature=1.0 if dynamic_joints else 0.0,
            friction=0.1 if dynamic_joints else 0.0,
        )
    else:
        hinge_axis = ModelBuilder.JointDofConfig(
            axis=Axis.Y,
            actuator_mode=JointTargetMode.EFFORT,
            armature=1.0 if dynamic_joints else 0.0,
            friction=0.1 if dynamic_joints else 0.0,
        )

    # Add a revolute joint between the two bodies
    jh = _builder.add_joint_revolute(
        label="hinge",
        parent=bid0,
        child=bid1,
        axis=hinge_axis,
        parent_xform=wp.transformf(0.25, 0.05, 0.0, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-0.25, -0.05, 0.0, 0.0, 0.0, 0.0, 1.0),
    )
    _builder.add_articulation([jf, jh])

    # Add collision geometries
    _builder.add_shape_box(
        label="base/box",
        body=bid0,
        hx=0.5 * d,
        hy=0.5 * w,
        hz=0.5 * h,
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="follower/box",
        body=bid1,
        hx=0.5 * d,
        hy=0.5 * w,
        hz=0.5 * h,
        cfg=_shape_cfg_basic(),
    )

    # Add a static collision geometry for the plane
    if ground:
        _add_ground_box(_builder)

    # Close the world context if we opened one
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated model builder
    return _builder


def build_boxes_nunchaku(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    ground: bool = True,
    new_world: bool = True,
) -> ModelBuilder:
    """
    Constructs a basic model of a faux nunchaku consisting of
    two boxes and one sphere connected via spherical joints.

    This version initializes the nunchaku in a horizontal configuration.

    .. note::
        Newton's :meth:`ModelBuilder.add_joint_ball` is used in place of Kamino's
        spherical joint type. A free joint is inserted between the world and the
        ``box_bottom`` body so the chain is a valid Newton articulation tree.

    Args:
        builder (ModelBuilder | None):
            An optional existing model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float):
            A vertical offset to apply to the initial position of the box.
        ground (bool):
            Whether to add a static ground plane to the model.
        new_world (bool):
            Whether to begin a new world in the builder for this model.\n
            If `True` (or `builder` is `None`), the model is wrapped in a new world context
            opened via :meth:`ModelBuilder.begin_world` and closed via
            :meth:`ModelBuilder.end_world`.\n
            If `False`, the caller must already be inside an active world; the model is then
            added to that currently active world.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Begin a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="boxes_nunchaku")

    # Model constants
    m_0 = 1.0
    m_1 = 1.0
    m_2 = 1.0
    d = 0.5
    w = 0.1
    h = 0.1
    r = 0.05

    # Constant to set an initial z offset for the bodies
    # NOTE: for testing purposes, recommend values are {0.0, -0.001}
    z_0 = z_offset

    # Add first body
    bid0 = _builder.add_link(
        label="box_bottom",
        mass=m_0,
        inertia=inertia.solid_cuboid_body_moment_of_inertia(m_0, d, w, h),
        xform=wp.transformf(0.5 * d, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )

    # Add second body
    bid1 = _builder.add_link(
        label="sphere_middle",
        mass=m_1,
        inertia=inertia.solid_sphere_body_moment_of_inertia(m_1, r),
        xform=wp.transformf(r + d, 0.0, r + z_0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )

    # Add third body
    bid2 = _builder.add_link(
        label="box_top",
        mass=m_2,
        inertia=inertia.solid_cuboid_body_moment_of_inertia(m_2, d, w, h),
        xform=wp.transformf(1.5 * d + 2.0 * r, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )

    # Attach the first body to the world with a free joint so the chain is a valid
    # Newton articulation (Kamino's version implicitly treats unconnected bodies as free)
    jf = _builder.add_joint_free(
        label="world_to_box_bottom",
        parent=-1,
        child=bid0,
        parent_xform=wp.transform_identity(dtype=wp.float32),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )

    # Add a joint between the first and second body
    j1 = _builder.add_joint_ball(
        label="box_bottom_to_sphere_middle",
        parent=bid0,
        child=bid1,
        parent_xform=wp.transformf(0.5 * d, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-r, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        actuator_mode=JointTargetMode.NONE,
    )

    # Add a joint between the second and third body
    j2 = _builder.add_joint_ball(
        label="sphere_middle_to_box_top",
        parent=bid1,
        child=bid2,
        parent_xform=wp.transformf(r, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(-0.5 * d, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        actuator_mode=JointTargetMode.NONE,
    )
    _builder.add_articulation([jf, j1, j2])

    # Add collision geometries
    cfg = _shape_cfg_basic()
    _builder.add_shape_box(
        label="box_bottom",
        body=bid0,
        hx=0.5 * d,
        hy=0.5 * w,
        hz=0.5 * h,
        cfg=cfg,
    )
    _builder.add_shape_sphere(
        label="sphere_middle",
        body=bid1,
        radius=r,
        cfg=cfg,
    )
    _builder.add_shape_box(
        label="box_top",
        body=bid2,
        hx=0.5 * d,
        hy=0.5 * w,
        hz=0.5 * h,
        cfg=cfg,
    )

    # Add a static collision geometry for the plane
    if ground:
        _add_ground_box(_builder)

    # Close the world context if we opened one
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated model builder
    return _builder


def build_boxes_nunchaku_vertical(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    ground: bool = True,
    new_world: bool = True,
) -> ModelBuilder:
    """
    Constructs a basic model of a faux nunchaku consisting of
    two boxes and one sphere connected via spherical joints.

    This version initializes the nunchaku in a vertical configuration.

    .. note::
        Newton's :meth:`ModelBuilder.add_joint_ball` is used in place of Kamino's
        spherical joint type. A free joint is inserted between the world and the
        ``box_bottom`` body so the chain is a valid Newton articulation tree.

    Args:
        builder (ModelBuilder | None):
            An optional existing model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float):
            A vertical offset to apply to the initial position of the box.
        ground (bool):
            Whether to add a static ground plane to the model.
        new_world (bool):
            Whether to begin a new world in the builder for this model.\n
            If `True` (or `builder` is `None`), the model is wrapped in a new world context
            opened via :meth:`ModelBuilder.begin_world` and closed via
            :meth:`ModelBuilder.end_world`.\n
            If `False`, the caller must already be inside an active world; the model is then
            added to that currently active world.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Begin a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="boxes_nunchaku_vertical")

    # Model constants
    m_0 = 1.0
    m_1 = 1.0
    m_2 = 1.0
    d = 0.1
    w = 0.1
    h = 0.5
    r = 0.05

    # Constant to set an initial z offset for the bodies
    # NOTE: for testing purposes, recommend values are {0.0, -0.001}
    z_0 = z_offset

    # Add first body
    bid0 = _builder.add_link(
        label="box_bottom",
        mass=m_0,
        inertia=inertia.solid_cuboid_body_moment_of_inertia(m_0, d, w, h),
        xform=wp.transformf(0.0, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )

    # Add second body
    bid1 = _builder.add_link(
        label="sphere_middle",
        mass=m_1,
        inertia=inertia.solid_sphere_body_moment_of_inertia(m_1, r),
        xform=wp.transformf(0.0, 0.0, h + r + z_0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )

    # Add third body
    bid2 = _builder.add_link(
        label="box_top",
        mass=m_2,
        inertia=inertia.solid_cuboid_body_moment_of_inertia(m_2, d, w, h),
        xform=wp.transformf(0.0, 0.0, 1.5 * h + 2.0 * r + z_0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )

    # Attach the first body to the world with a free joint so the chain is a valid
    # Newton articulation (Kamino's version implicitly treats unconnected bodies as free)
    jf = _builder.add_joint_free(
        label="world_to_box_bottom",
        parent=-1,
        child=bid0,
        parent_xform=wp.transform_identity(dtype=wp.float32),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )

    # Add a joint between the first and second body
    j1 = _builder.add_joint_ball(
        label="box_bottom_to_sphere_middle",
        parent=bid0,
        child=bid1,
        parent_xform=wp.transformf(0.0, 0.0, 0.5 * h, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(0.0, 0.0, -r, 0.0, 0.0, 0.0, 1.0),
        actuator_mode=JointTargetMode.NONE,
    )

    # Add a joint between the second and third body
    j2 = _builder.add_joint_ball(
        label="sphere_middle_to_box_top",
        parent=bid1,
        child=bid2,
        parent_xform=wp.transformf(0.0, 0.0, r, 0.0, 0.0, 0.0, 1.0),
        child_xform=wp.transformf(0.0, 0.0, -0.5 * h, 0.0, 0.0, 0.0, 1.0),
        actuator_mode=JointTargetMode.NONE,
    )
    _builder.add_articulation([jf, j1, j2])

    # Add collision geometries
    cfg = _shape_cfg_basic()
    _builder.add_shape_box(
        label="box_bottom",
        body=bid0,
        hx=0.5 * d,
        hy=0.5 * w,
        hz=0.5 * h,
        cfg=cfg,
    )
    _builder.add_shape_sphere(
        label="sphere_middle",
        body=bid1,
        radius=r,
        cfg=cfg,
    )
    _builder.add_shape_box(
        label="box_top",
        body=bid2,
        hx=0.5 * d,
        hy=0.5 * w,
        hz=0.5 * h,
        cfg=cfg,
    )

    # Add a static collision geometry for the plane
    if ground:
        _add_ground_box(_builder)

    # Close the world context if we opened one
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated model builder
    return _builder


def build_boxes_fourbar(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    fixedbase: bool = False,
    floatingbase: bool = False,
    limits: bool = True,
    ground: bool = True,
    dynamic_joints: bool = False,
    implicit_pd: bool = False,
    verbose: bool = False,
    new_world: bool = True,
    actuator_ids: list[int] | None = None,
) -> ModelBuilder:
    """
    Constructs a basic model of a four-bar linkage.

    Args:
        builder (ModelBuilder | None):
            An optional existing model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float):
            A vertical offset to apply to the initial position of the box.
        fixedbase (bool):
            Whether to attach ``link_1`` to the world with a fixed joint.
        floatingbase (bool):
            Whether to attach ``link_1`` to the world with a free (6-DoF) joint.
        limits (bool):
            Whether to apply finite position limits on every revolute joint.\n
            If `True`, each hinge is restricted to `[-pi/4, pi/4]`.\n
            If `False`, the joint limits are set to the largest representable float32 range.
        ground (bool):
            Whether to add a static ground plane to the model.
        dynamic_joints (bool):
            Whether to attach non-zero armature and friction terms to the first
            actuated revolute joint so that its dynamics are better conditioned
            for stiff integrators.
        implicit_pd (bool):
            Whether to configure the first actuated revolute joint with a
            position/velocity target mode (implicit PD) instead of the default
            effort-based actuation.
        verbose (bool):
            If `True`, prints the computed body inertias and the initial body and
            joint positions during construction.
        new_world (bool):
            Whether to begin a new world in the builder for this model.\n
            If `True` (or `builder` is `None`), the model is wrapped in a new world context
            opened via :meth:`ModelBuilder.begin_world` and closed via
            :meth:`ModelBuilder.end_world`.\n
            If `False`, the caller must already be inside an active world; the model is then
            added to that currently active world.
        actuator_ids (list[int] | None):
            1-based indices of the revolute joints (``1`` through ``4``) that should be
            driven by an actuator. Any joint whose index is not listed is treated as a
            passive revolute joint.\n
            In the original Kamino factory the special index ``0`` selected actuation of
            the free-base joint; Newton's free joint does not expose an analogous flag
            and the value is currently ignored for the base joint.\n
            If `None`, defaults to `[1, 3]`.

    Returns:
        ModelBuilder: A model builder containing the four-bar linkage.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Begin a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world(label="boxes_fourbar")

    # Set default actuator IDs if none are provided
    if actuator_ids is None:
        actuator_ids = [1, 3]
    elif not isinstance(actuator_ids, list):
        raise TypeError("actuator_ids, if specified, must be provided as a list of integers.")

    ###
    # Base Parameters
    ###

    # Constant to set an initial z offset for the bodies
    # NOTE: for testing purposes, recommend values are {0.0, -0.001}
    z_0 = z_offset

    # Box dimensions
    d = 0.01
    w = 0.01
    h = 0.1

    # Margins
    mj = 0.001
    dj = 0.5 * d + mj

    ###
    # Body parameters
    ###

    # Box dimensions
    d_1 = h
    w_1 = w
    h_1 = d
    d_2 = d
    w_2 = w
    h_2 = h
    d_3 = h
    w_3 = w
    h_3 = d
    d_4 = d
    w_4 = w
    h_4 = h

    # Inertial properties
    m_i = 1.0
    i_I_i_1 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_1, w_1, h_1)
    i_I_i_2 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_2, w_2, h_2)
    i_I_i_3 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_3, w_3, h_3)
    i_I_i_4 = inertia.solid_cuboid_body_moment_of_inertia(m_i, d_4, w_4, h_4)
    if verbose:
        print(f"i_I_i_1:\n{i_I_i_1}")
        print(f"i_I_i_2:\n{i_I_i_2}")
        print(f"i_I_i_3:\n{i_I_i_3}")
        print(f"i_I_i_4:\n{i_I_i_4}")

    # Initial body positions
    r_0 = wp.vec3f(0.0, 0.0, z_0)
    dr_b1 = wp.vec3f(0.0, 0.0, 0.5 * d)
    dr_b2 = wp.vec3f(0.5 * h + dj, 0.0, 0.5 * h + dj)
    dr_b3 = wp.vec3f(0.0, 0.0, 0.5 * d + h + dj + mj)
    dr_b4 = wp.vec3f(-0.5 * h - dj, 0.0, 0.5 * h + dj)

    # Initial positions of the bodies
    r_b1 = r_0 + dr_b1
    r_b2 = r_b1 + dr_b2
    r_b3 = r_b1 + dr_b3
    r_b4 = r_b1 + dr_b4
    if verbose:
        print(f"r_b1: {r_b1}")
        print(f"r_b2: {r_b2}")
        print(f"r_b3: {r_b3}")
        print(f"r_b4: {r_b4}")

    # Initial body poses
    q_i_1 = wp.transformf(r_b1, wp.quat_identity(dtype=wp.float32))
    q_i_2 = wp.transformf(r_b2, wp.quat_identity(dtype=wp.float32))
    q_i_3 = wp.transformf(r_b3, wp.quat_identity(dtype=wp.float32))
    q_i_4 = wp.transformf(r_b4, wp.quat_identity(dtype=wp.float32))

    # Initial joint positions
    r_j1 = wp.vec3f(r_b2.x, 0.0, r_b1.z)
    r_j2 = wp.vec3f(r_b2.x, 0.0, r_b3.z)
    r_j3 = wp.vec3f(r_b4.x, 0.0, r_b3.z)
    r_j4 = wp.vec3f(r_b4.x, 0.0, r_b1.z)
    if verbose:
        print(f"r_j1: {r_j1}")
        print(f"r_j2: {r_j2}")
        print(f"r_j3: {r_j3}")
        print(f"r_j4: {r_j4}")

    ###
    # Bodies
    ###

    bid1 = _builder.add_link(
        label="link_1",
        mass=m_i,
        inertia=i_I_i_1,
        xform=q_i_1,
        lock_inertia=True,
    )

    bid2 = _builder.add_link(
        label="link_2",
        mass=m_i,
        inertia=i_I_i_2,
        xform=q_i_2,
        lock_inertia=True,
    )

    bid3 = _builder.add_link(
        label="link_3",
        mass=m_i,
        inertia=i_I_i_3,
        xform=q_i_3,
        lock_inertia=True,
    )

    bid4 = _builder.add_link(
        label="link_4",
        mass=m_i,
        inertia=i_I_i_4,
        xform=q_i_4,
        lock_inertia=True,
    )

    ###
    # Geometries
    ###

    # Add collision geometries
    _builder.add_shape_box(
        label="box_1",
        body=bid1,
        hx=0.5 * d_1,
        hy=0.5 * w_1,
        hz=0.5 * h_1,
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="box_2",
        body=bid2,
        hx=0.5 * d_2,
        hy=0.5 * w_2,
        hz=0.5 * h_2,
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="box_3",
        body=bid3,
        hx=0.5 * d_3,
        hy=0.5 * w_3,
        hz=0.5 * h_3,
        cfg=_shape_cfg_basic(),
    )
    _builder.add_shape_box(
        label="box_4",
        body=bid4,
        hx=0.5 * d_4,
        hy=0.5 * w_4,
        hz=0.5 * h_4,
        cfg=_shape_cfg_basic(),
    )

    # Add a static collision geometry for the plane
    if ground:
        _add_ground_box(_builder)

    ###
    # Joints
    ###

    # Revolute joint position limits
    if limits:
        qmin = -0.25 * math.pi
        qmax = 0.25 * math.pi
    else:
        qmin = float(JOINT_QMIN)
        qmax = float(JOINT_QMAX)

    # Optional fixed base: attach link_1 rigidly to the world
    if fixedbase:
        _builder.add_joint_fixed(
            label="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transformf(-r_b1, wp.quat_identity(dtype=wp.float32)),
        )

    # Optional floating base: attach link_1 to the world with a 6-DoF free joint
    if floatingbase:
        _builder.add_joint_free(
            label="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )

    # Per-DoF configurations reused across the revolute joints
    passive_joint_dof_config = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=JointTargetMode.NONE,
        limit_lower=qmin,
        limit_upper=qmax,
    )
    effort_joint_1 = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=JointTargetMode.EFFORT,
        limit_lower=qmin,
        limit_upper=qmax,
        armature=0.1 if dynamic_joints else 0.0,
        friction=0.001 if dynamic_joints else 0.0,
    )
    effort_joint_other = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=JointTargetMode.EFFORT,
        limit_lower=qmin,
        limit_upper=qmax,
    )
    pd_joint_dof_config = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=JointTargetMode.POSITION_VELOCITY,
        armature=0.1 if dynamic_joints else 0.0,
        friction=0.001 if dynamic_joints else 0.0,
        target_ke=1000.0,
        target_kd=20.0,
        limit_lower=qmin,
        limit_upper=qmax,
    )

    # Add a revolute joint between link 1 and link 2
    joint_1_axis = (
        pd_joint_dof_config
        if implicit_pd and 1 in actuator_ids
        else effort_joint_1
        if 1 in actuator_ids
        else passive_joint_dof_config
    )
    _builder.add_joint_revolute(
        label="link1_to_link2",
        parent=bid1,
        child=bid2,
        axis=joint_1_axis,
        parent_xform=wp.transformf(r_j1 - r_b1, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j1 - r_b2, wp.quat_identity(dtype=wp.float32)),
    )

    # Add a revolute joint between link 2 and link 3
    _builder.add_joint_revolute(
        label="link2_to_link3",
        parent=bid2,
        child=bid3,
        axis=effort_joint_other if 2 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j2 - r_b2, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j2 - r_b3, wp.quat_identity(dtype=wp.float32)),
    )

    # Add a revolute joint between link 3 and link 4
    _builder.add_joint_revolute(
        label="link3_to_link4",
        parent=bid3,
        child=bid4,
        axis=effort_joint_other if 3 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j3 - r_b3, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j3 - r_b4, wp.quat_identity(dtype=wp.float32)),
    )

    # Add a revolute joint between link 4 and link 1 (closes the loop)
    _builder.add_joint_revolute(
        label="link4_to_link1",
        parent=bid4,
        child=bid1,
        axis=effort_joint_other if 4 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j4 - r_b4, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j4 - r_b1, wp.quat_identity(dtype=wp.float32)),
    )

    # Close the world context if we opened one
    if new_world or builder is None:
        _builder.end_world()

    # Return the populated model builder
    return _builder


def make_basics_heterogeneous_builder(
    builder: ModelBuilder | None = None,
    ground: bool = True,
    dynamic_joints: bool = False,
    implicit_pd: bool = False,
) -> ModelBuilder:
    """
    Creates a multi-world builder with different worlds in each model.

    This function constructs a model builder containing all basic models. Each scene
    is built into its own sub-builder and then merged in via
    :meth:`ModelBuilder.add_world`, which preserves the Kamino ordering of the
    original ``make_basics_heterogeneous_builder``.

    Args:
        builder (ModelBuilder | None):
            An optional existing model builder to populate.\n
            If `None`, a new builder is created.
        ground (bool):
            Whether to add a static ground plane to each sub-model.
        dynamic_joints (bool):
            Whether to enable dynamic (armature/friction) joint terms in the sub-models
            that expose the option (box pendulum, hinged boxes, four-bar).
        implicit_pd (bool):
            Whether to drive the actuated joints of those sub-models with a
            position/velocity target (implicit PD) instead of effort-based actuation.

    Returns:
        ModelBuilder: The constructed model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Add each basic model as its own world, following the Kamino ordering
    _builder.add_world(
        build_boxes_fourbar(
            ground=ground,
            dynamic_joints=dynamic_joints,
            implicit_pd=implicit_pd,
            new_world=True,
        )
    )
    _builder.add_world(build_boxes_nunchaku(ground=ground, new_world=True))
    _builder.add_world(
        build_boxes_hinged(
            ground=ground,
            dynamic_joints=dynamic_joints,
            implicit_pd=implicit_pd,
            new_world=True,
        )
    )
    _builder.add_world(
        build_box_pendulum(
            ground=ground,
            dynamic_joints=dynamic_joints,
            implicit_pd=implicit_pd,
            new_world=True,
        )
    )
    _builder.add_world(build_box_on_plane(ground=ground, new_world=True))
    _builder.add_world(build_cartpole(z_offset=0.5, ground=ground, new_world=True))

    # Return the populated model builder
    return _builder
