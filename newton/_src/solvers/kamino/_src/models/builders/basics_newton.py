# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Factory methods for building 'basic' models using :class:`newton.ModelBuilder`.

This module provides a set of functions to create simple mechanical assemblies using the
:class:`newton.ModelBuilder` interface. These include fundamental configurations such as
a box on a plane, a box pendulum, a cartpole, and various linked box systems.

Each function constructs a specific model by adding rigid bodies, joints, and collision
geometries to a :class:`newton.ModelBuilder` instance. The  models are designed to serve
as foundational examples for testing and  demonstration purposes, and each features a
certain subset of ill-conditioned dynamics.
"""

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
    "build_boxes_fourbar",
    "build_boxes_nunchaku",
]


###
# Functions
###


def build_boxes_fourbar(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    fixedbase: bool = False,
    floatingbase: bool = True,
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
        ground (bool):
            Whether to add a static ground plane to the model.
        new_world (bool):
            Whether to create a new world in the builder for this model.\n
            If `True`, a new world is created and added to the builder.

    Returns:
        ModelBuilder: A model builder containing the four-bar linkage.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder()
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        _builder.begin_world()

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

    _builder.add_shape_box(
        label="box_1",
        body=bid1,
        hx=0.5 * d_1,
        hy=0.5 * w_1,
        hz=0.5 * h_1,
        cfg=ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )
    _builder.add_shape_box(
        label="box_2",
        body=bid2,
        hx=0.5 * d_2,
        hy=0.5 * w_2,
        hz=0.5 * h_2,
        cfg=ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )

    _builder.add_shape_box(
        label="box_3",
        body=bid3,
        hx=0.5 * d_3,
        hy=0.5 * w_3,
        hz=0.5 * h_3,
        cfg=ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )

    _builder.add_shape_box(
        label="box_4",
        body=bid4,
        hx=0.5 * d_4,
        hy=0.5 * w_4,
        hz=0.5 * h_4,
        cfg=ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_shape_box(
            label="ground",
            body=-1,
            hx=10.0,
            hy=10.0,
            hz=0.5,
            xform=wp.transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            cfg=ModelBuilder.ShapeConfig(margin=0.0, gap=0.0),
        )

    ###
    # Joints
    ###

    if limits:
        qmin = -0.25 * math.pi
        qmax = 0.25 * math.pi
    else:
        qmin = float(JOINT_QMIN)
        qmax = float(JOINT_QMAX)

    if fixedbase:
        _builder.add_joint_fixed(
            label="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transformf(-r_b1, wp.quat_identity(dtype=wp.float32)),
        )

    if floatingbase:
        _builder.add_joint_free(
            label="world_to_link1",
            parent=-1,
            child=bid1,
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
        )

    passive_joint_dof_config = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=JointTargetMode.NONE,
        limit_lower=qmin,
        limit_upper=qmax,
    )
    effort_joint_dof_config = ModelBuilder.JointDofConfig(
        axis=Axis.Y,
        actuator_mode=JointTargetMode.EFFORT,
        limit_lower=qmin,
        limit_upper=qmax,
        armature=0.1 if dynamic_joints else 0.0,
        friction=0.001 if dynamic_joints else 0.0,
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

    joint_1_config_if_implicit_pd = pd_joint_dof_config if implicit_pd else effort_joint_dof_config
    joint_1_config_if_actuated = joint_1_config_if_implicit_pd if 1 in actuator_ids else passive_joint_dof_config
    _builder.add_joint_revolute(
        label="link1_to_link2",
        parent=bid1,
        child=bid2,
        axis=joint_1_config_if_actuated if 1 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j1 - r_b1, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j1 - r_b2, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        label="link2_to_link3",
        parent=bid2,
        child=bid3,
        axis=effort_joint_dof_config if 2 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j2 - r_b2, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j2 - r_b3, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        label="link3_to_link4",
        parent=bid3,
        child=bid4,
        axis=effort_joint_dof_config if 3 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j3 - r_b3, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j3 - r_b4, wp.quat_identity(dtype=wp.float32)),
    )

    _builder.add_joint_revolute(
        label="link4_to_link1",
        parent=bid4,
        child=bid1,
        axis=effort_joint_dof_config if 4 in actuator_ids else passive_joint_dof_config,
        parent_xform=wp.transformf(r_j4 - r_b4, wp.quat_identity(dtype=wp.float32)),
        child_xform=wp.transformf(r_j4 - r_b1, wp.quat_identity(dtype=wp.float32)),
    )

    # Signal the end of setting-up the new world
    if new_world or builder is None:
        _builder.end_world()

    # Return the lists of element indices
    return _builder


def build_boxes_nunchaku(
    builder: ModelBuilder | None = None,
    ground: bool = True,
) -> ModelBuilder:
    """
    Constructs a nunchaku model: two boxes connected by a sphere via ball joints.

    Three bodies (two boxes + one sphere) connected by spherical joints,
    optionally resting on a ground plane.  Produces 9 contacts with the
    ground (4 per box + 1 sphere).

    Args:
        builder: An optional existing model builder to populate.
            If ``None``, a new builder is created.
        ground: Whether to add a static ground plane.

    Returns:
        The populated :class:`ModelBuilder`.
    """
    if builder is None:
        builder = ModelBuilder()

    d, w, h, r = 0.5, 0.1, 0.1, 0.05
    no_gap = ModelBuilder.ShapeConfig(gap=0.0)

    b0 = builder.add_link()
    builder.add_shape_box(b0, hx=d / 2, hy=w / 2, hz=h / 2, cfg=no_gap)

    b1 = builder.add_link()
    builder.add_shape_sphere(b1, radius=r, cfg=no_gap)

    b2 = builder.add_link()
    builder.add_shape_box(b2, hx=d / 2, hy=w / 2, hz=h / 2, cfg=no_gap)

    j0 = builder.add_joint_ball(
        parent=-1,
        child=b0,
        parent_xform=wp.transform(p=wp.vec3(d / 2, 0.0, h / 2), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
    )
    j1 = builder.add_joint_ball(
        parent=b0,
        child=b1,
        parent_xform=wp.transform(p=wp.vec3(d / 2, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-r, 0.0, 0.0), q=wp.quat_identity()),
    )
    j2 = builder.add_joint_ball(
        parent=b1,
        child=b2,
        parent_xform=wp.transform(p=wp.vec3(r, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-d / 2, 0.0, 0.0), q=wp.quat_identity()),
    )
    builder.add_articulation([j0, j1, j2])

    if ground:
        builder.add_ground_plane()

    return builder
