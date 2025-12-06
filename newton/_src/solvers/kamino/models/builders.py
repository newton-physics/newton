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

"""Defines model builders and utilities for test models"""

import math

import numpy as np
import warp as wp

from ..core import ModelBuilder
from ..core.inertia import (
    solid_cuboid_body_moment_of_inertia,
    solid_sphere_body_moment_of_inertia,
)
from ..core.joints import JointActuationType, JointDoFType
from ..core.math import FLOAT32_MAX, FLOAT32_MIN, I_3, quat_from_euler_xyz
from ..core.shapes import (
    BoxShape,
    CapsuleShape,
    ConeShape,
    CylinderShape,
    EllipsoidShape,
    PlaneShape,
    ShapeDescriptorType,
    ShapeType,
    SphereShape,
)
from ..core.types import Axis, mat33f, transformf, vec3f, vec6f
from ..utils import logger as msg
from ..utils.io.usd import USDImporter

###
# Module interface
###

__all__ = [
    "add_body_pose_offset",
    "add_body_twist_offset",
    "add_ground_geom",
    "build_box_on_plane",
    "build_box_pendulum",
    "build_box_pendulum_vertical",
    "build_boxes_fourbar",
    "build_boxes_hinged",
    "build_boxes_nunchaku",
    "build_boxes_nunchaku_vertical",
    "build_usd_model",
    "make_shape_pairs_builder",
    "make_single_shape_pair_builder",
]


###
# Builder modifiers
###


def add_ground_plane(
    builder: ModelBuilder,
    layer: str = "world",
    group: int = 1,
    collides: int = 1,
    world_index: int = 0,
    z_offset: float = 0.0,
) -> int:
    """
    Adds a static collision layer and geometry to a given builder to represent a flat ground plane.

    Args:
        builder (ModelBuilder): The model builder to which the ground geom should be added.
        group (int): The collision group for the ground geometry.
        collides (int): The collision mask for the ground geometry.
        world_index (int): The index of the world in the builder where the ground geom should be added.

    Returns:
        int: The ID of the added ground geometry.
    """
    return builder.add_collision_geometry(
        shape=PlaneShape(vec3f(0.0, 0.0, 1.0), 0.0),
        offset=transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        name="ground",
        layer=layer,
        group=group,
        collides=collides,
        world_index=world_index,
    )


def add_ground_geom(
    builder: ModelBuilder,
    layer: str = "world",
    group: int = 1,
    collides: int = 1,
    world_index: int = 0,
    z_offset: float = 0.0,
) -> int:
    """
    Adds a static collision layer and geometry to a given builder to represent a flat ground plane.

    Args:
        builder (ModelBuilder): The model builder to which the ground geom should be added.
        group (int): The collision group for the ground geometry.
        collides (int): The collision mask for the ground geometry.
        world_index (int): The index of the world in the builder where the ground geom should be added.

    Returns:
        int: The ID of the added ground geometry.
    """
    return builder.add_collision_geometry(
        shape=BoxShape(20.0, 20.0, 1.0),
        offset=transformf(0.0, 0.0, -0.5 + z_offset, 0.0, 0.0, 0.0, 1.0),
        name="ground",
        layer=layer,
        group=group,
        collides=collides,
        world_index=world_index,
    )


def add_body_pose_offset(builder: ModelBuilder, offset: transformf):
    """
    Offsets a model builder by a given transform.

    Args:
        builder (ModelBuilder): The model builder to offset.
        offset (transformf): The transform offset to apply to each body in the builder.
    """
    for i in range(builder.num_bodies):
        builder.bodies[i].q_i_0 = wp.mul(offset, builder.bodies[i].q_i_0)


def add_body_twist_offset(builder: ModelBuilder, offset: vec6f):
    """
    Offsets a model builder by a given transform.

    Args:
        builder (ModelBuilder): The model builder to offset.
        offset (vec6f): The twist offset to apply to each body in the builder.
    """
    for i in range(builder.num_bodies):
        builder.bodies[i].u_i_0 += offset


###
# Builders for joint tests
###


def build_free_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test free joints.

    This world consists of a single rigid body connected to the world via a unary
    free joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="unary_free_joint_test")

    # Define test system
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_follower_free",
        dof_type=JointDoFType.FREE,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=I_3,
        q_j_min=[-2.0, -2.0, -2.0, -0.6 * math.pi, -0.6 * math.pi, -0.6 * math.pi] if limits else None,
        q_j_max=[2.0, 2.0, 2.0, 0.6 * math.pi, 0.6 * math.pi, 0.6 * math.pi] if limits else None,
        tau_j_max=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(1.0, 1.0, 1.0),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_revolute_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test unary revolute joints.

    This world consists of a single rigid body connected to the world via a unary
    revolute joint, with optional limits applied to the joint degree of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degree of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="unary_revolute_joint_test")

    # Define test system
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.5, -0.25, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_follower_revolute",
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, -0.15, z_offset),
        F_r_Fj=vec3f(-0.5, 0.1, 0.0),
        X_j=Axis.Y.to_mat33(),
        q_j_min=[-0.25 * math.pi] if limits else None,
        q_j_max=[0.25 * math.pi] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.3, 0.3, 0.3),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(1.0, 0.2, 0.2),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_revolute_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test binary revolute joints.

    This world consists of two rigid bodies connected via a binary revolute
    joint, with optional limits applied to the joint degree of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degree of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="binary_revolute_joint_test")

    # Define test system
    bid_B = _builder.add_rigid_body(
        name="base",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.5, -0.25, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_base",
        dof_type=JointDoFType.FIXED,
        act_type=JointActuationType.PASSIVE,
        bid_B=-1,
        bid_F=bid_B,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Y.to_mat33(),
        world_index=world_index,
    )
    _builder.add_joint(
        name="base_to_follower_revolute",
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=bid_B,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, -0.15, z_offset),
        F_r_Fj=vec3f(-0.5, 0.1, 0.0),
        X_j=Axis.Y.to_mat33(),
        q_j_min=[-0.25 * math.pi] if limits else None,
        q_j_max=[0.25 * math.pi] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.3, 0.3, 0.3),
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(1.0, 0.2, 0.2),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_prismatic_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test unary prismatic joints.

    This world consists of a single rigid body connected to the world via a unary
    prismatic joint, with optional limits applied to the joint degree of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degree of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="unary_prismatic_joint_test")

    # Define test system
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_follower_prismatic",
        dof_type=JointDoFType.PRISMATIC,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Z.to_mat33(),
        q_j_min=[-0.5] if limits else None,
        q_j_max=[0.5] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.05, 0.05, 1.0),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.1, 0.1, 0.1),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_prismatic_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test binary prismatic joints.

    This world consists of two rigid bodies connected via a binary prismatic
    joint, with optional limits applied to the joint degree of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degree of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="binary_prismatic_joint_test")

    # Define test system
    bid_B = _builder.add_rigid_body(
        name="base",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_base",
        dof_type=JointDoFType.FIXED,
        act_type=JointActuationType.PASSIVE,
        bid_B=-1,
        bid_F=bid_B,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Y.to_mat33(),
        world_index=world_index,
    )
    _builder.add_joint(
        name="base_to_follower_prismatic",
        dof_type=JointDoFType.PRISMATIC,
        act_type=JointActuationType.FORCE,
        bid_B=bid_B,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Z.to_mat33(),
        q_j_min=[-0.5] if limits else None,
        q_j_max=[0.5] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.05, 0.05, 1.0),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.1, 0.1, 0.1),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_cylindrical_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test unary cylindrical joints.

    This world consists of a single rigid body connected to the world via a unary
    cylindrical joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="unary_cylindrical_joint_test")

    # Define test system
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_follower_cylindrical",
        dof_type=JointDoFType.CYLINDRICAL,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Z.to_mat33(),
        q_j_min=[-0.5, -0.6 * math.pi] if limits else None,
        q_j_max=[0.5, 0.6 * math.pi] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/cylinder",
        body=-1,
        shape=CylinderShape(0.025, 1.0),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.1, 0.1, 0.1),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_cylindrical_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test binary cylindrical joints.

    This world consists of two rigid bodies connected via a binary cylindrical
    joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="binary_cylindrical_joint_test")

    # Define test system
    bid_B = _builder.add_rigid_body(
        name="base",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_base",
        dof_type=JointDoFType.FIXED,
        act_type=JointActuationType.PASSIVE,
        bid_B=-1,
        bid_F=bid_B,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Y.to_mat33(),
        world_index=world_index,
    )
    _builder.add_joint(
        name="base_to_follower_cylindrical",
        dof_type=JointDoFType.CYLINDRICAL,
        act_type=JointActuationType.FORCE,
        bid_B=bid_B,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Z.to_mat33(),
        q_j_min=[-0.5, -0.6 * math.pi] if limits else None,
        q_j_max=[0.5, 0.6 * math.pi] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/cylinder",
        body=bid_B,
        shape=CylinderShape(0.025, 1.0),
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.1, 0.1, 0.1),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_universal_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test unary universal joints.

    This world consists of a single rigid body connected to the world via a unary
    universal joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="unary_universal_joint_test")

    # Define test system
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.5, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_follower_universal",
        dof_type=JointDoFType.UNIVERSAL,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.25, -0.25, -0.25),
        F_r_Fj=vec3f(-0.25, -0.25, -0.25),
        X_j=Axis.X.to_mat33(),
        q_j_min=[-0.6 * math.pi, -0.6 * math.pi] if limits else None,
        q_j_max=[0.6 * math.pi, 0.6 * math.pi] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_universal_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test binary universal joints.

    This world consists of two rigid bodies connected via a binary universal
    joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="binary_cylindrical_joint_test")

    # Define test system
    bid_B = _builder.add_rigid_body(
        name="base",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.5, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_base",
        dof_type=JointDoFType.FIXED,
        act_type=JointActuationType.PASSIVE,
        bid_B=-1,
        bid_F=bid_B,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Y.to_mat33(),
        world_index=world_index,
    )
    _builder.add_joint(
        name="base_to_follower_universal",
        dof_type=JointDoFType.UNIVERSAL,
        act_type=JointActuationType.FORCE,
        bid_B=bid_B,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.25, -0.25, -0.25),
        F_r_Fj=vec3f(-0.25, -0.25, -0.25),
        X_j=Axis.X.to_mat33(),
        q_j_min=[-0.6 * math.pi, -0.6 * math.pi] if limits else None,
        q_j_max=[0.6 * math.pi, 0.6 * math.pi] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_spherical_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test unary spherical joints.

    This world consists of a single rigid body connected to the world via a unary
    spherical joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="unary_spherical_joint_test")

    # Define test system
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.5, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_follower_spherical",
        dof_type=JointDoFType.SPHERICAL,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.25, -0.25, -0.25),
        F_r_Fj=vec3f(-0.25, -0.25, -0.25),
        X_j=Axis.X.to_mat33(),
        q_j_min=[-0.6 * math.pi, -0.6 * math.pi, -0.6 * math.pi] if limits else None,
        q_j_max=[0.6 * math.pi, 0.6 * math.pi, 0.6 * math.pi] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_spherical_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test binary spherical joints.

    This world consists of two rigid bodies connected via a binary spherical
    joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="binary_spherical_joint_test")

    # Define test system
    bid_B = _builder.add_rigid_body(
        name="base",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.5, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_base",
        dof_type=JointDoFType.FIXED,
        act_type=JointActuationType.PASSIVE,
        bid_B=-1,
        bid_F=bid_B,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Y.to_mat33(),
        world_index=world_index,
    )
    _builder.add_joint(
        name="base_to_follower_spherical",
        dof_type=JointDoFType.SPHERICAL,
        act_type=JointActuationType.FORCE,
        bid_B=bid_B,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.25, -0.25, -0.25),
        F_r_Fj=vec3f(-0.25, -0.25, -0.25),
        X_j=Axis.X.to_mat33(),
        q_j_min=[-0.6 * math.pi, -0.6 * math.pi, -0.6 * math.pi] if limits else None,
        q_j_max=[0.6 * math.pi, 0.6 * math.pi, 0.6 * math.pi] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_gimbal_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test unary gimbal joints.

    This world consists of a single rigid body connected to the world via a unary
    gimbal joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="unary_gimbal_joint_test")

    # Define test system
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.5, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_follower_gimbal",
        dof_type=JointDoFType.GIMBAL,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.25, -0.25, -0.25),
        F_r_Fj=vec3f(-0.25, -0.25, -0.25),
        X_j=Axis.X.to_mat33(),
        # q_j_min=[-0.4 * math.pi, -0.4 * math.pi, -0.4 * math.pi] if limits else None,
        # q_j_max=[0.4 * math.pi, 0.4 * math.pi, 0.4 * math.pi] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_gimbal_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test binary gimbal joints.

    This world consists of two rigid bodies connected via a binary gimbal
    joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="binary_gimbal_joint_test")

    # Define test system
    bid_B = _builder.add_rigid_body(
        name="base",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.5, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_base",
        dof_type=JointDoFType.FIXED,
        act_type=JointActuationType.PASSIVE,
        bid_B=-1,
        bid_F=bid_B,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Y.to_mat33(),
        world_index=world_index,
    )
    _builder.add_joint(
        name="base_to_follower_gimbal",
        dof_type=JointDoFType.GIMBAL,
        act_type=JointActuationType.FORCE,
        bid_B=bid_B,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.25, -0.25, -0.25),
        F_r_Fj=vec3f(-0.25, -0.25, -0.25),
        X_j=Axis.X.to_mat33(),
        # q_j_min=[-0.4 * math.pi, -0.4 * math.pi, -0.4 * math.pi] if limits else None,
        # q_j_max=[0.4 * math.pi, 0.4 * math.pi, 0.4 * math.pi] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_cartesian_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test unary cartesian joints.

    This world consists of a single rigid body connected to the world via a unary
    cartesian joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="unary_cartesian_joint_test")

    # Define test system
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.5, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_follower_cartesian",
        dof_type=JointDoFType.CARTESIAN,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.25, -0.25, -0.25),
        F_r_Fj=vec3f(-0.25, -0.25, -0.25),
        X_j=Axis.X.to_mat33(),
        q_j_min=[-1.0, -1.0, -1.0] if limits else None,
        q_j_max=[1.0, 1.0, 1.0] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_cartesian_joint_test(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Builds a world to test binary cartesian joints.

    This world consists of two rigid bodies connected via a binary cartesian
    joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilder | None): An optional existing ModelBuilder to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        ground (bool): Whether to include a ground plane in the world.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="binary_gimbal_joint_test")

    # Define test system
    bid_B = _builder.add_rigid_body(
        name="base",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    bid_F = _builder.add_rigid_body(
        name="follower",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.5, 0.0, z_offset), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )
    _builder.add_joint(
        name="world_to_base",
        dof_type=JointDoFType.FIXED,
        act_type=JointActuationType.PASSIVE,
        bid_B=-1,
        bid_F=bid_B,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Y.to_mat33(),
        world_index=world_index,
    )
    _builder.add_joint(
        name="base_to_follower_cartesian",
        dof_type=JointDoFType.CARTESIAN,
        act_type=JointActuationType.FORCE,
        bid_B=bid_B,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.25, -0.25, -0.25),
        F_r_Fj=vec3f(-0.25, -0.25, -0.25),
        X_j=Axis.X.to_mat33(),
        q_j_min=[-1.0, -1.0, -1.0] if limits else None,
        q_j_max=[1.0, 1.0, 1.0] if limits else None,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_all_joints_test_model(
    z_offset: float = 0.0,
    ground: bool = False,
) -> ModelBuilder:
    """
    Constructs a model builder containing a world for each joint type.

    Args:
        z_offset (float): A vertical offset to apply to the initial position of the box.
        ground (bool): Whether to add a static ground plane to the model.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder to populate
    _builder = ModelBuilder(default_world=False)

    # Add a new world for each joint type
    _builder.add_builder(build_free_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_unary_revolute_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_binary_revolute_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_unary_prismatic_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_binary_prismatic_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_unary_cylindrical_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_binary_cylindrical_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_unary_universal_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_binary_universal_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_unary_spherical_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_binary_spherical_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_unary_gimbal_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_binary_gimbal_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_unary_cartesian_joint_test(z_offset=z_offset, ground=ground))
    _builder.add_builder(build_binary_cartesian_joint_test(z_offset=z_offset, ground=ground))

    # Return the lists of element indices
    return _builder


###
# Builders for geometry tests
###


shape_name_to_type: dict[str, ShapeType] = {
    "sphere": ShapeType.SPHERE,
    "cylinder": ShapeType.CYLINDER,
    "cone": ShapeType.CONE,
    "capsule": ShapeType.CAPSULE,
    "box": ShapeType.BOX,
    "ellipsoid": ShapeType.ELLIPSOID,
    "plane": ShapeType.PLANE,
}
"""Mapping from shape name to ShapeType enum."""


shape_type_to_descriptor: dict[ShapeType, ShapeDescriptorType] = {
    ShapeType.SPHERE: SphereShape,
    ShapeType.CYLINDER: CylinderShape,
    ShapeType.CONE: ConeShape,
    ShapeType.CAPSULE: CapsuleShape,
    ShapeType.BOX: BoxShape,
    ShapeType.ELLIPSOID: EllipsoidShape,
    ShapeType.PLANE: PlaneShape,
}
"""Mapping from ShapeType enum to corresponding ShapeDescriptorType."""


shape_default_dims: dict[ShapeType, tuple] = {
    ShapeType.SPHERE: (0.5,),
    ShapeType.CYLINDER: (0.5, 1.0),
    ShapeType.CONE: (0.5, 1.0),
    ShapeType.CAPSULE: (0.5, 1.0),
    ShapeType.BOX: (1.0, 1.0, 1.0),
    ShapeType.ELLIPSOID: (1.0, 1.0, 0.5),
    ShapeType.PLANE: (0.0, 0.0, 1.0, 0.0),
}
"""Mapping from ShapeType enum to default dimensions."""


def make_shape_initial_position(name: str, dims: tuple, is_top: bool = True) -> vec3f:
    """
    Computes the initial position along the z-axis for a given shape.

    This function calculates the position required to place a shape just above
    (or below) the origin along the z-axis, based on its type and dimensions.

    Args:
        name (str):
            The name of the shape (e.g., "sphere", "box", "capsule", etc.).
        dims (tuple):
            The dimensions of the shape. The expected format depends on the shape type.
        is_top (bool):
            If True, computes the position for a top shape (above the origin).
            If False, computes the position for a bottom shape (below the origin).

    Returns:
        vec3f:
            The computed position vector along the z-axis.
    """
    # Retrieve and check the shape type
    shape_type = shape_name_to_type.get(name)
    if shape_type is None:
        raise ValueError(f"Unsupported shape name: {name}")

    # Check dimensions length
    if len(dims) != shape_type.num_params:
        raise ValueError(f"Invalid dimensions for shape '{name}': expected {shape_type.num_params}, got {len(dims)}")

    # Compute the initial position along z-axis that places the shape just above
    if shape_type == ShapeType.SPHERE:
        r = vec3f(0.0, 0.0, dims[0])
    elif shape_type == ShapeType.BOX:
        r = vec3f(0.0, 0.0, 0.5 * dims[2])
    elif shape_type == ShapeType.CAPSULE:
        r = vec3f(0.0, 0.0, 0.5 * dims[1] + dims[0])
    elif shape_type == ShapeType.CYLINDER:
        r = vec3f(0.0, 0.0, 0.5 * dims[1])
    elif shape_type == ShapeType.CONE:
        r = vec3f(0.0, 0.0, 0.5 * dims[1])
    elif shape_type == ShapeType.ELLIPSOID:
        r = vec3f(0.0, 0.0, dims[2])
    elif shape_type == ShapeType.PLANE:
        r = vec3f(0.0, 0.0, dims[3])
    else:
        raise ValueError(f"Unsupported shape type: {shape_type}")

    # Invert the position if it's the bottom shape
    if not is_top:
        r = -r

    # Return the computed position
    return r


def get_shape_bottom_position(center: vec3f, shape: ShapeDescriptorType) -> vec3f:
    """
    Computes the position of the bottom along the z-axis for a given shape.

    Args:
        center (vec3f):
            The center position of the shape.
        shape (ShapeDescriptorType):
            The shape descriptor instance.

    Returns:
        vec3f:
            The computed bottom position of the shape along the z-axis.
    """
    # Compute and return the initial position along z-axis that places the shape just above
    r_bottom = vec3f(0.0)
    if shape.type == ShapeType.SPHERE:
        r_bottom = center - vec3f(0.0, 0.0, shape.params)
    elif shape.type == ShapeType.BOX:
        r_bottom = center - vec3f(0.0, 0.0, 0.5 * shape.params[2])
    elif shape.type == ShapeType.CAPSULE:
        r_bottom = center - vec3f(0.0, 0.0, 0.5 * shape.params[1] + shape.params[0])
    elif shape.type == ShapeType.CYLINDER:
        r_bottom = center - vec3f(0.0, 0.0, 0.5 * shape.params[1])
    elif shape.type == ShapeType.CONE:
        r_bottom = center - vec3f(0.0, 0.0, 0.5 * shape.params[1])
    elif shape.type == ShapeType.ELLIPSOID:
        r_bottom = center - vec3f(0.0, 0.0, shape.params[2])
    elif shape.type == ShapeType.PLANE:
        r_bottom = center - vec3f(0.0, 0.0, shape.params[3])
    else:
        raise ValueError(f"Unsupported shape type: {shape.type}")

    # Return the bottom position of the given shape
    return r_bottom


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
    Generates a ModelBuilder for a given shape combination with specified parameters.

    The first shape in the combination is placed below the second shape along
    the z-axis, effectively generating a "shape[0] atop shape[1]" configuration.

    Args:
        shapes (tuple[str, str]):
            A tuple specifying the names of the bottom and top shapes (e.g., ("box", "sphere")).
        bottom_dims (tuple | None):
            Dimensions for the bottom shape. If None, defaults are used.
        bottom_xyz (tuple | None):
            Position (x, y, z) for the bottom shape. If None, defaults to (0, 0, 0).
        bottom_rpy (tuple | None):
            Orientation (roll, pitch, yaw) for the bottom shape. If None, defaults to (0, 0, 0).
        top_dims (tuple | None):
            Dimensions for the top shape. If None, defaults are used.
        top_xyz (tuple | None):
            Position (x, y, z) for the top shape. If None, defaults to (0, 0, 0).
        top_rpy (tuple | None):
            Orientation (roll, pitch, yaw) for the top shape. If None, defaults to (0, 0, 0).
        distance (float):
            Mutual distance along the z-axis between the two shapes.\n
            If zero, the shapes are exactly touching.\n
            If positive, they are separated by that distance.\n
            If negative, they are penetrating by that distance.

    Returns:
        ModelBuilder:
            The constructed ModelBuilder with the specified shape combination.
    """
    # Check that the shape combination is tuple of strings
    if not (isinstance(shapes, tuple) and len(shapes) == 2 and all(isinstance(s, str) for s in shapes)):
        raise ValueError(f"Shape combination must be a tuple of two strings: {shapes}")

    # Check that each shape name is valid
    for shape_name in shapes:
        if shape_name not in shape_name_to_type:
            raise ValueError(f"Unsupported shape name: {shape_name}")

    # Define bottom and top shapes
    top = shapes[0]
    bottom = shapes[1]

    # Retrieve shape types
    top_type = shape_name_to_type[top]
    bottom_type = shape_name_to_type[bottom]

    # Define default arguments for those not provided
    if bottom_dims is None:
        bottom_dims = shape_default_dims[bottom_type]
    if bottom_xyz is None:
        bottom_xyz = make_shape_initial_position(shapes[1], bottom_dims, is_top=False)
    if bottom_rpy is None:
        bottom_rpy = (0.0, 0.0, 0.0)
    if top_dims is None:
        top_dims = shape_default_dims[top_type]
    if top_xyz is None:
        top_xyz = make_shape_initial_position(shapes[0], top_dims, is_top=True)
    if top_rpy is None:
        top_rpy = (0.0, 0.0, 0.0)

    # Retrieve the shape type
    bottom_descriptor = shape_type_to_descriptor[bottom_type]
    top_descriptor = shape_type_to_descriptor[top_type]

    # Define the mutual separation along z-axis
    r_dz = vec3f(0.0, 0.0, 0.5 * distance)

    # Compute bottom box position and orientation
    r_b = vec3f(bottom_xyz) - r_dz
    q_b = quat_from_euler_xyz(vec3f(*bottom_rpy))

    # Compute top sphere position and orientation
    r_t = vec3f(top_xyz) + r_dz
    q_t = quat_from_euler_xyz(vec3f(*top_rpy))

    # Create the shape descriptors for bottom and top shapes
    # with special handling for PlaneShape
    if bottom_type == ShapeType.PLANE:
        bottom_shape = bottom_descriptor(vec3f(*bottom_dims[0:3]), bottom_dims[3])
    else:
        bottom_shape = bottom_descriptor(*bottom_dims)
    if top_type == ShapeType.PLANE:
        top_shape = top_descriptor(vec3f(*top_dims[0:3]), top_dims[3])
    else:
        top_shape = top_descriptor(*top_dims)

    # Create model builder and add corresponding bodies and their collision geometries
    builder: ModelBuilder = ModelBuilder(default_world=True)
    bid0 = builder.add_rigid_body(
        name="bottom_" + bottom,
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(r_b, q_b),
    )
    bid1 = builder.add_rigid_body(
        name="top_" + top,
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(r_t, q_t),
    )
    builder.add_collision_geometry(body=bid0, name="bottom_" + bottom, shape=bottom_shape)
    builder.add_collision_geometry(body=bid1, name="top_" + top, shape=top_shape)

    # Optionally add a ground geom below the bottom shape
    if ground_box or ground_plane:
        if ground_z is not None:
            z_g_offset = ground_z
        else:
            z_g_offset = float(get_shape_bottom_position(r_b, bottom_shape).z - r_dz.z)
        if ground_box:
            add_ground_geom(builder, z_offset=z_g_offset)
        if ground_plane:
            add_ground_plane(builder, z_offset=z_g_offset)

    # Debug output
    msg.debug(
        "[%s]:\nBODIES:\n%s\nGEOMS:\n%s\n",
        shapes,
        builder.bodies,
        builder.collision_geoms,
    )

    # Return the constructed builder
    return builder


def make_shape_pairs_builder(
    shape_pairs: list[tuple[str, str]],
    per_shape_pair_args: dict | None = None,
    distance: float | None = None,
    ground_box: bool = False,
    ground_plane: bool = False,
    ground_z: float | None = None,
) -> ModelBuilder:
    """
    Generates a builder containing a world for each specified shape combination.

    Args:
        shape_pairs (list[tuple[str, str]]):
            A list of tuples specifying the names of the bottom and top shapes
            for each combination (e.g., [("box", "sphere"), ("cylinder", "cone")]).
        **kwargs:
            Additional keyword arguments to be passed to `make_single_shape_pair_builder`.
    Returns:
        ModelBuilder
            A ModelBuilder containing a world for each specified shape combination.
    """
    # Create an empty ModelBuilder to hold all shape pair worlds
    builder = ModelBuilder(default_world=False)

    # Iterate over each shape pair and add its builder to the main builder
    for shapes in shape_pairs:
        # Set shape-pair-specific arguments if provided
        if per_shape_pair_args is not None:
            # Check if per_shape_pair_args contains arguments for this shape pair
            shape_pair_args = per_shape_pair_args.get(shapes, {})
        else:
            shape_pair_args = {}

        # Override distance if specified
        if distance is not None:
            shape_pair_args["distance"] = distance

        # Create the single shape pair builder and add it to the main builder
        single_pair_builder = make_single_shape_pair_builder(
            shapes, ground_box=ground_box, ground_plane=ground_plane, ground_z=ground_z, **shape_pair_args
        )
        builder.add_builder(single_pair_builder)

    # Return the combined builder
    return builder


###
# Builders for basic models
###


def build_box_on_plane(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    ground: bool = True,
    new_world: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Constructs a basic model of free-floating box body and a ground plane geom.

    Args:
        builder (ModelBuilder | None): An optional model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float): A vertical offset to apply to the initial position of the box.
        ground (bool): Whether to add a static ground plane to the model.
        new_world (bool): Whether to create a new world in the builder for this model.\n
            If `False`, the model is added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        world_index (int): The index of the world to which the model should be added if `new_world` is False.\n
            If `new_world` is True, this argument is ignored.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="box_on_plane")

    # Add first body
    bid0 = _builder.add_rigid_body(
        m_i=1.0,
        i_I_i=solid_cuboid_body_moment_of_inertia(1.0, 0.2, 0.2, 0.2),
        q_i_0=transformf(0.0, 0.0, 0.1 + z_offset, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add a collision layer and geometries
    _builder.add_collision_geometry(body=bid0, shape=BoxShape(0.2, 0.2, 0.2), world_index=world_index)

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the lists of element indices
    return _builder


def build_box_pendulum(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.7,
    ground: bool = True,
    new_world: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Constructs a basic model of a single box pendulum body with a unary revolute joint.

    This version initializes the pendulum in a horizontal configuration.

    Args:
        builder (ModelBuilder | None): An optional model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float): A vertical offset to apply to the initial position of the box.
        ground (bool): Whether to add a static ground plane to the model.
        new_world (bool): Whether to create a new world in the builder for this model.\n
            If `False`, the model is added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.\n
        world_index (int): The index of the world to which the model should be added if `new_world` is False.\n
            If `new_world` is True, this argument is ignored.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="box_pendulum")

    # Model constants
    m = 1.0
    d = 0.5
    w = 0.1
    h = 0.1
    z_0 = z_offset  # Initial z offset for the body

    # Add box pendulum body
    bid0 = _builder.add_rigid_body(
        name="pendulum",
        m_i=m,
        i_I_i=solid_cuboid_body_moment_of_inertia(m, d, w, h),
        q_i_0=transformf(0.5 * d, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add a joint between the two bodies
    _builder.add_joint(
        name="world_to_pendulum",
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid0,
        B_r_Bj=vec3f(0.0, 0.0, 0.5 * h + z_0),
        F_r_Fj=vec3f(-0.5 * d, 0.0, 0.0),
        X_j=Axis.Y.to_mat33(),
        world_index=world_index,
    )

    # Add a collision layer and geometries
    _builder.add_collision_geometry(
        name="box",
        body=bid0,
        shape=BoxShape(d, w, h),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            name="ground",
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the lists of element indices
    return _builder


def build_box_pendulum_vertical(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.7,
    ground: bool = True,
    new_world: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Constructs a basic model of a single box pendulum body with a unary revolute joint.

    This version initializes the pendulum in a vertical configuration.

    Args:
        builder (ModelBuilder | None): An optional model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float): A vertical offset to apply to the initial position of the box.
        ground (bool): Whether to add a static ground plane to the model.
        new_world (bool): Whether to create a new world in the builder for this model.\n
            If `False`, the model is added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.\n
        world_index (int): The index of the world to which the model should be added if `new_world` is False.\n
            If `new_world` is True, this argument is ignored.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="box_pendulum_vertical")

    # Model constants
    m = 1.0
    d = 0.1
    w = 0.1
    h = 0.5
    z_0 = z_offset  # Initial z offset for the body

    # Add box pendulum body
    bid0 = _builder.add_rigid_body(
        name="pendulum",
        m_i=m,
        i_I_i=solid_cuboid_body_moment_of_inertia(m, d, w, h),
        q_i_0=transformf(0.0, 0.0, -0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add a joint between the two bodies
    _builder.add_joint(
        name="world_to_pendulum",
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid0,
        B_r_Bj=vec3f(0.0, 0.0, 0.0 + z_0),
        F_r_Fj=vec3f(0.0, 0.0, 0.5 * h),
        X_j=Axis.Y.to_mat33(),
        world_index=world_index,
    )

    # Add a collision layer and geometries
    _builder.add_collision_geometry(
        name="box",
        body=bid0,
        shape=BoxShape(d, w, h),
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            name="ground",
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the lists of element indices
    return _builder


def build_cartpole(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    ground: bool = True,
    new_world: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Constructs a basic model of a cartpole mounted onto a rail.

    Args:
        builder (ModelBuilder | None): An optional model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float): A vertical offset to apply to the initial position of the box.
        ground (bool): Whether to add a static ground plane to the model.
        new_world (bool): Whether to create a new world in the builder for this model.\n
            If `False`, the model is added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.\n
        world_index (int): The index of the world to which the model should be added if `new_world` is False.\n
            If `new_world` is True, this argument is ignored.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="cartpole")

    # Model constants
    m_cart = 1.0
    m_pole = 0.2
    dims_rail = (0.03, 8.0, 0.03)
    dims_cart = (0.2, 0.5, 0.2)
    dims_pole = (0.05, 0.05, 0.75)

    # Add box cart body
    bid0 = _builder.add_rigid_body(
        name="cart",
        m_i=m_cart,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_cart, *dims_cart),
        q_i_0=transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add box pole body
    x_0_pole = 0.5 * dims_pole[0] + 0.5 * dims_cart[0]
    z_0_pole = 0.5 * dims_pole[2] + z_offset
    bid1 = _builder.add_rigid_body(
        name="pole",
        m_i=m_pole,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_pole, *dims_pole),
        q_i_0=transformf(x_0_pole, 0.0, z_0_pole, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add a prismatic joint for the cart
    _builder.add_joint(
        name="rail_to_cart",
        dof_type=JointDoFType.PRISMATIC,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid0,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Y.to_mat33(),
        q_j_min=[-4.0],
        q_j_max=[4.0],
        tau_j_max=[1000.0],
        world_index=world_index,
    )

    # Add a revolute joint for the pendulum
    _builder.add_joint(
        name="cart_to_pole",
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid0,
        bid_F=bid1,
        B_r_Bj=vec3f(0.5 * dims_cart[0], 0.0, 0.0),
        F_r_Fj=vec3f(-0.5 * dims_pole[0], 0.0, -0.5 * dims_pole[2]),
        X_j=Axis.X.to_mat33(),
        world_index=world_index,
    )

    # Add a collision layer and geometries
    _builder.add_collision_geometry(
        name="cart",
        layer="primary",
        body=bid0,
        shape=BoxShape(*dims_cart),
        group=2,
        collides=2,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="pole",
        layer="primary",
        body=bid1,
        shape=BoxShape(*dims_pole),
        group=3,
        collides=3,
        world_index=world_index,
    )
    _builder.add_collision_geometry(
        name="rail",
        layer="world",
        body=-1,
        shape=BoxShape(*dims_rail),
        offset=transformf(0.0, 0.0, z_offset, 0.0, 0.0, 0.0, 1.0),
        group=1,
        collides=1,
        world_index=world_index,
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            name="ground",
            layer="world",
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.0 + z_offset, 0.0, 0.0, 0.0, 1.0),
            group=1,
            collides=1,
            world_index=world_index,
        )

    # Return the lists of element indices
    return _builder


def build_boxes_hinged(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    ground: bool = True,
    new_world: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Constructs a basic model of a two floating boxes connected via revolute joint.

    Args:
        builder (ModelBuilder | None): An optional model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float): A vertical offset to apply to the initial position of the box.
        ground (bool): Whether to add a static ground plane to the model.
        new_world (bool): Whether to create a new world in the builder for this model.\n
            If `False`, the model is added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.\n
        world_index (int): The index of the world to which the model should be added if `new_world` is False.\n
            If `new_world` is True, this argument is ignored.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="boxes_hinged")

    # Model constants
    m_0 = 1.0
    m_1 = 1.0
    d = 0.5
    w = 0.1
    h = 0.1
    z0 = z_offset  # Initial z offset for the bodies

    # Add first body
    bid0 = _builder.add_rigid_body(
        name="base",
        m_i=m_0,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_0, d, w, h),
        q_i_0=transformf(0.25, -0.05, 0.05 + z0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add second body
    bid1 = _builder.add_rigid_body(
        name="follower",
        m_i=m_1,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_1, d, w, h),
        q_i_0=transformf(0.75, 0.05, 0.05 + z0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add a joint between the two bodies
    _builder.add_joint(
        name="hinge",
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=bid0,
        bid_F=bid1,
        B_r_Bj=vec3f(0.25, 0.05, 0.0),
        F_r_Fj=vec3f(-0.25, -0.05, 0.0),
        X_j=Axis.Y.to_mat33(),
        world_index=world_index,
    )

    # Add a collision layer and geometries
    _builder.add_collision_geometry(
        name="base/box", body=bid0, shape=BoxShape(d, w, h), group=2, collides=3, world_index=world_index
    )
    _builder.add_collision_geometry(
        name="follower/box", body=bid1, shape=BoxShape(d, w, h), group=3, collides=5, world_index=world_index
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            name="ground",
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            group=1,
            collides=7,
            world_index=world_index,
        )

    # Return the lists of element indices
    return _builder


def build_boxes_nunchaku(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    ground: bool = True,
    new_world: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Constructs a basic model of a faux nunchaku consisting of
    two boxes and one sphere connected via spherical joints.

    This version initializes the nunchaku in a horizontal configuration.

    Args:
        builder (ModelBuilder | None): An optional model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float): A vertical offset to apply to the initial position of the box.
        ground (bool): Whether to add a static ground plane to the model.
        new_world (bool): Whether to create a new world in the builder for this model.\n
            If `False`, the model is added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.\n
        world_index (int): The index of the world to which the model should be added if `new_world` is False.\n
            If `new_world` is True, this argument is ignored.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="boxes_nunchaku")

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
    bid0 = _builder.add_rigid_body(
        name="box_bottom",
        m_i=m_0,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_0, d, w, h),
        q_i_0=transformf(0.5 * d, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add second body
    bid1 = _builder.add_rigid_body(
        name="sphere_middle",
        m_i=m_1,
        i_I_i=solid_sphere_body_moment_of_inertia(m_1, r),
        q_i_0=transformf(r + d, 0.0, r + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add third body
    bid2 = _builder.add_rigid_body(
        name="box_top",
        m_i=m_2,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_2, d, w, h),
        q_i_0=transformf(1.5 * d + 2.0 * r, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add a joint between the first and second body
    _builder.add_joint(
        name="box_bottom_to_sphere_middle",
        dof_type=JointDoFType.SPHERICAL,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid0,
        bid_F=bid1,
        B_r_Bj=vec3f(0.5 * d, 0.0, 0.0),
        F_r_Fj=vec3f(-r, 0.0, 0.0),
        X_j=I_3,
        world_index=world_index,
    )

    # Add a joint between the second and third body
    _builder.add_joint(
        name="sphere_middle_to_box_top",
        dof_type=JointDoFType.SPHERICAL,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid1,
        bid_F=bid2,
        B_r_Bj=vec3f(r, 0.0, 0.0),
        F_r_Fj=vec3f(-0.5 * d, 0.0, 0.0),
        X_j=I_3,
        world_index=world_index,
    )

    # Add a collision layer and geometries
    _builder.add_collision_geometry(
        name="box_bottom", body=bid0, shape=BoxShape(d, w, h), group=2, collides=3, world_index=world_index
    )
    _builder.add_collision_geometry(
        name="sphere_middle", body=bid1, shape=SphereShape(r), group=3, collides=5, world_index=world_index
    )
    _builder.add_collision_geometry(
        name="box_top", body=bid2, shape=BoxShape(d, w, h), group=2, collides=3, world_index=world_index
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            name="ground",
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            group=1,
            collides=7,
            world_index=world_index,
        )

    # Return the lists of element indices
    return _builder


def build_boxes_nunchaku_vertical(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    ground: bool = True,
    new_world: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Constructs a basic model of a faux nunchaku consisting of
    two boxes and one sphere connected via spherical joints.

    This version initializes the nunchaku in a vertical configuration.

    Args:
        builder (ModelBuilder | None): An optional model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float): A vertical offset to apply to the initial position of the box.
        ground (bool): Whether to add a static ground plane to the model.
        new_world (bool): Whether to create a new world in the builder for this model.\n
            If `False`, the model is added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.\n
        world_index (int): The index of the world to which the model should be added if `new_world` is False.\n
            If `new_world` is True, this argument is ignored.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="boxes_nunchaku_vertical")

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
    bid0 = _builder.add_rigid_body(
        name="box_bottom",
        m_i=m_0,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_0, d, w, h),
        q_i_0=transformf(0.0, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add second body
    bid1 = _builder.add_rigid_body(
        name="sphere_middle",
        m_i=m_1,
        i_I_i=solid_sphere_body_moment_of_inertia(m_1, r),
        q_i_0=transformf(0.0, 0.0, h + r + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add third body
    bid2 = _builder.add_rigid_body(
        name="box_top",
        m_i=m_2,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_2, d, w, h),
        q_i_0=transformf(0.0, 0.0, 1.5 * h + 2.0 * r + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        world_index=world_index,
    )

    # Add a joint between the first and second body
    _builder.add_joint(
        name="box_bottom_to_sphere_middle",
        dof_type=JointDoFType.SPHERICAL,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid0,
        bid_F=bid1,
        B_r_Bj=vec3f(0.0, 0.0, 0.5 * h),
        F_r_Fj=vec3f(0.0, 0.0, -r),
        X_j=I_3,
        world_index=world_index,
    )

    # Add a joint between the second and third body
    _builder.add_joint(
        name="sphere_middle_to_box_top",
        dof_type=JointDoFType.SPHERICAL,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid1,
        bid_F=bid2,
        B_r_Bj=vec3f(0.0, 0.0, r),
        F_r_Fj=vec3f(0.0, 0.0, -0.5 * h),
        X_j=I_3,
        world_index=world_index,
    )

    # Add a collision layer and geometries
    _builder.add_collision_geometry(
        name="box_bottom", body=bid0, shape=BoxShape(d, w, h), group=2, collides=3, world_index=world_index
    )
    _builder.add_collision_geometry(
        name="sphere_middle", body=bid1, shape=SphereShape(r), group=3, collides=5, world_index=world_index
    )
    _builder.add_collision_geometry(
        name="box_top", body=bid2, shape=BoxShape(d, w, h), group=2, collides=3, world_index=world_index
    )

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            name="ground",
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            group=1,
            collides=7,
            world_index=world_index,
        )

    # Return the lists of element indices
    return _builder


def build_boxes_fourbar(
    builder: ModelBuilder | None = None,
    z_offset: float = 0.0,
    fixedbase: bool = False,
    limits: bool = True,
    ground: bool = True,
    verbose: bool = False,
    new_world: bool = True,
    world_index: int = 0,
) -> ModelBuilder:
    """
    Constructs a basic model of a faux nunchaku consisting of
    two boxes and one sphere connected via spherical joints.

    This version initializes the nunchaku in a vertical configuration.

    Args:
        builder (ModelBuilder | None): An optional model builder to populate.\n
            If `None`, a new builder is created.
        z_offset (float): A vertical offset to apply to the initial position of the box.
        ground (bool): Whether to add a static ground plane to the model.
        new_world (bool): Whether to create a new world in the builder for this model.\n
            If `False`, the model is added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.\n
        world_index (int): The index of the world to which the model should be added if `new_world` is False.\n
            If `new_world` is True, this argument is ignored.

    Returns:
        ModelBuilder: The populated model builder.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilder(default_world=False)
    else:
        _builder = builder

    # Create a new world in the builder if requested or if a new builder was created
    if new_world or builder is None:
        world_index = _builder.add_world(name="boxes_fourbar")

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
    i_I_i_1 = solid_cuboid_body_moment_of_inertia(m_i, d_1, w_1, h_1)
    i_I_i_2 = solid_cuboid_body_moment_of_inertia(m_i, d_2, w_2, h_2)
    i_I_i_3 = solid_cuboid_body_moment_of_inertia(m_i, d_3, w_3, h_3)
    i_I_i_4 = solid_cuboid_body_moment_of_inertia(m_i, d_4, w_4, h_4)
    if verbose:
        print(f"i_I_i_1:\n{i_I_i_1}")
        print(f"i_I_i_2:\n{i_I_i_2}")
        print(f"i_I_i_3:\n{i_I_i_3}")
        print(f"i_I_i_4:\n{i_I_i_4}")

    # Initial body positions
    r_0 = vec3f(0.0, 0.0, z_0)
    dr_b1 = vec3f(0.0, 0.0, 0.5 * d)
    dr_b2 = vec3f(0.5 * h + dj, 0.0, 0.5 * h + dj)
    dr_b3 = vec3f(0.0, 0.0, 0.5 * d + h + dj + mj)
    dr_b4 = vec3f(-0.5 * h - dj, 0.0, 0.5 * h + dj)

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
    q_i_1 = transformf(r_b1, wp.quat_identity())
    q_i_2 = transformf(r_b2, wp.quat_identity())
    q_i_3 = transformf(r_b3, wp.quat_identity())
    q_i_4 = transformf(r_b4, wp.quat_identity())

    # Initial joint positions
    r_j1 = vec3f(r_b2.x, 0.0, r_b1.z)
    r_j2 = vec3f(r_b2.x, 0.0, r_b3.z)
    r_j3 = vec3f(r_b4.x, 0.0, r_b3.z)
    r_j4 = vec3f(r_b4.x, 0.0, r_b1.z)
    if verbose:
        print(f"r_j1: {r_j1}")
        print(f"r_j2: {r_j2}")
        print(f"r_j3: {r_j3}")
        print(f"r_j4: {r_j4}")

    # Joint axes matrix
    X_j = Axis.Y.to_mat33()

    ###
    # Bodies
    ###

    bid1 = _builder.add_rigid_body(
        name="link_1",
        m_i=m_i,
        i_I_i=i_I_i_1,
        q_i_0=q_i_1,
        u_i_0=vec6f(0.0),
        world_index=world_index,
    )

    bid2 = _builder.add_rigid_body(
        name="link_2",
        m_i=m_i,
        i_I_i=i_I_i_2,
        q_i_0=q_i_2,
        u_i_0=vec6f(0.0),
        world_index=world_index,
    )

    bid3 = _builder.add_rigid_body(
        name="link_3",
        m_i=m_i,
        i_I_i=i_I_i_3,
        q_i_0=q_i_3,
        u_i_0=vec6f(0.0),
        world_index=world_index,
    )

    bid4 = _builder.add_rigid_body(
        name="link_4",
        m_i=m_i,
        i_I_i=i_I_i_4,
        q_i_0=q_i_4,
        u_i_0=vec6f(0.0),
        world_index=world_index,
    )

    ###
    # Joints
    ###

    if limits:
        qmin = -0.25 * math.pi
        qmax = 0.25 * math.pi
    else:
        qmin = float(FLOAT32_MIN)
        qmax = float(FLOAT32_MAX)

    if fixedbase:
        _builder.add_joint(
            name="world_to_link1",
            dof_type=JointDoFType.FIXED,
            act_type=JointActuationType.PASSIVE,
            bid_B=-1,
            bid_F=bid1,
            B_r_Bj=vec3f(0.0),
            F_r_Fj=-r_b1,
            X_j=I_3,
            world_index=world_index,
        )

    _builder.add_joint(
        name="link1_to_link2",
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=bid1,
        bid_F=bid2,
        B_r_Bj=r_j1 - r_b1,
        F_r_Fj=r_j1 - r_b2,
        X_j=X_j,
        q_j_min=[qmin],
        q_j_max=[qmax],
        world_index=world_index,
    )

    _builder.add_joint(
        name="link2_to_link3",
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid2,
        bid_F=bid3,
        B_r_Bj=r_j2 - r_b2,
        F_r_Fj=r_j2 - r_b3,
        X_j=X_j,
        q_j_min=[qmin],
        q_j_max=[qmax],
        world_index=world_index,
    )

    _builder.add_joint(
        name="link3_to_link4",
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=bid3,
        bid_F=bid4,
        B_r_Bj=r_j3 - r_b3,
        F_r_Fj=r_j3 - r_b4,
        X_j=X_j,
        q_j_min=[qmin],
        q_j_max=[qmax],
        world_index=world_index,
    )

    _builder.add_joint(
        name="link4_to_link1",
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid4,
        bid_F=bid1,
        B_r_Bj=r_j4 - r_b4,
        F_r_Fj=r_j4 - r_b1,
        X_j=X_j,
        q_j_min=[qmin],
        q_j_max=[qmax],
        world_index=world_index,
    )

    ###
    # Geometries
    ###

    # Add a collision layer and geometries
    _builder.add_collision_geometry(name="box_1", body=bid1, shape=BoxShape(d_1, w_1, h_1), world_index=world_index)
    _builder.add_collision_geometry(name="box_2", body=bid2, shape=BoxShape(d_2, w_2, h_2), world_index=world_index)
    _builder.add_collision_geometry(name="box_3", body=bid3, shape=BoxShape(d_3, w_3, h_3), world_index=world_index)
    _builder.add_collision_geometry(name="box_4", body=bid4, shape=BoxShape(d_4, w_4, h_4), world_index=world_index)

    # Add a static collision layer and geometry for the plane
    if ground:
        _builder.add_collision_geometry(
            name="ground",
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the lists of element indices
    return _builder


def build_usd_model(
    source: str,
    load_static_geometry: bool = True,
    ground: bool = True,
) -> ModelBuilder:
    """
    Imports a USD model and optionally adds a ground plane.

    Each call creates a new world with the USD model and optional ground plane.

    Args:
        source: Path to USD file
        load_static_geometry: Whether to load static geometry from USD
        ground: Whether to add a ground plane

    Returns:
        ModelBuilder with imported USD model and optional ground plane
    """
    # Import the USD model
    importer = USDImporter()
    _builder = importer.import_from(
        source=source,
        load_static_geometry=load_static_geometry,
    )

    ground_world_idx = _builder.num_worlds - 1

    # Add ground plane
    if ground:
        add_ground_geom(builder=_builder, group=1, collides=1, world_index=ground_world_idx)

    return _builder
