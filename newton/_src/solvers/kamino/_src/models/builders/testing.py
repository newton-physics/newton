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
Provides builders for testing supported joint and geometry types.

This module defines a set of functions for creating
model builders to test and demonstrate all the types
of joints and geometries supported by Kamino.
"""

import math

import numpy as np
import warp as wp

from ...core import ModelBuilderKamino
from ...core.joints import JointActuationType, JointDoFType
from ...core.math import I_3, quat_from_euler_xyz
from ...core.shapes import (
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
from ...core.types import Axis, mat33f, transformf, vec3f, vec6f
from ...utils import logger as msg
from . import utils

###
# Module interface
###

__all__ = [
    "build_binary_cartesian_joint_test",
    "build_binary_cylindrical_joint_test",
    "build_binary_gimbal_joint_test",
    "build_binary_prismatic_joint_test",
    "build_binary_revolute_joint_test",
    "build_binary_spherical_joint_test",
    "build_binary_universal_joint_test",
    "build_free_joint_test",
    "build_unary_cartesian_joint_test",
    "build_unary_cylindrical_joint_test",
    "build_unary_gimbal_joint_test",
    "build_unary_prismatic_joint_test",
    "build_unary_revolute_joint_test",
    "build_unary_spherical_joint_test",
    "build_unary_universal_joint_test",
    "make_shape_pairs_builder",
    "make_single_shape_pair_builder",
]


###
# Builders - Joint Tests
###


def build_free_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test free joints.

    This world consists of a single rigid body connected to the world via a unary
    free joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
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
        _builder = ModelBuilderKamino(default_world=False)
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
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(1.0, 1.0, 1.0),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_revolute_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test unary revolute joints.

    This world consists of a single rigid body connected to the world via a unary
    revolute joint, with optional limits applied to the joint degree of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degree of freedom.
        ground (bool): Whether to include a ground plane in the world.
        dynamic (bool): Whether to enable dynamic properties for the joint.
        implicit_pd (bool): Whether to enable implicit PD control for the joint.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilderKamino(default_world=False)
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
        act_type=JointActuationType.POSITION_VELOCITY if implicit_pd else JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, -0.15, z_offset),
        F_r_Fj=vec3f(-0.5, 0.1, 0.0),
        X_j=Axis.Y.to_mat33(),
        q_j_min=[-0.25 * math.pi] if limits else None,
        q_j_max=[0.25 * math.pi] if limits else None,
        a_j=0.1 if dynamic else None,
        b_j=0.01 if dynamic else None,
        k_p_j=10.0 if implicit_pd else None,
        k_d_j=0.01 if implicit_pd else None,
        world_index=world_index,
    )
    _builder.add_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.3, 0.3, 0.3),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(1.0, 0.2, 0.2),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_revolute_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test binary revolute joints.

    This world consists of two rigid bodies connected via a binary revolute
    joint, with optional limits applied to the joint degree of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degree of freedom.
        ground (bool): Whether to include a ground plane in the world.
        dynamic (bool): Whether to set the joint to be dynamic, with non-zero armature and damping.
        implicit_pd (bool): Whether to use implicit PD control for the joint.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilderKamino(default_world=False)
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
        act_type=JointActuationType.POSITION_VELOCITY if implicit_pd else JointActuationType.FORCE,
        bid_B=bid_B,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, -0.15, z_offset),
        F_r_Fj=vec3f(-0.5, 0.1, 0.0),
        X_j=Axis.Y.to_mat33(),
        q_j_min=[-0.25 * math.pi] if limits else None,
        q_j_max=[0.25 * math.pi] if limits else None,
        a_j=0.1 if dynamic else None,
        b_j=0.01 if dynamic else None,
        k_p_j=10.0 if implicit_pd else None,
        k_d_j=0.01 if implicit_pd else None,
        world_index=world_index,
    )
    _builder.add_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.3, 0.3, 0.3),
        world_index=world_index,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(1.0, 0.2, 0.2),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_prismatic_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test unary prismatic joints.

    This world consists of a single rigid body connected to the world via a unary
    prismatic joint, with optional limits applied to the joint degree of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
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
        _builder = ModelBuilderKamino(default_world=False)
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
        act_type=JointActuationType.POSITION_VELOCITY if implicit_pd else JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Z.to_mat33(),
        q_j_min=[-0.5] if limits else None,
        q_j_max=[0.5] if limits else None,
        a_j=0.1 if dynamic else None,
        b_j=0.01 if dynamic else None,
        k_p_j=10.0 if implicit_pd else None,
        k_d_j=0.01 if implicit_pd else None,
        world_index=world_index,
    )
    _builder.add_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.05, 0.05, 1.0),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.1, 0.1, 0.1),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_prismatic_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test binary prismatic joints.

    This world consists of two rigid bodies connected via a binary prismatic
    joint, with optional limits applied to the joint degree of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
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
        _builder = ModelBuilderKamino(default_world=False)
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
        act_type=JointActuationType.POSITION_VELOCITY if implicit_pd else JointActuationType.FORCE,
        bid_B=bid_B,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Z.to_mat33(),
        q_j_min=[-0.5] if limits else None,
        q_j_max=[0.5] if limits else None,
        a_j=0.1 if dynamic else None,
        b_j=0.01 if dynamic else None,
        k_p_j=10.0 if implicit_pd else None,
        k_d_j=0.01 if implicit_pd else None,
        world_index=world_index,
    )
    _builder.add_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.05, 0.05, 1.0),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.1, 0.1, 0.1),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_cylindrical_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test unary cylindrical joints.

    This world consists of a single rigid body connected to the world via a unary
    cylindrical joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        ground (bool): Whether to include a ground plane in the world.
        dynamic (bool): Whether to enable dynamic properties for the joint.
        implicit_pd (bool): Whether to enable implicit PD control for the joint.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilderKamino(default_world=False)
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
        act_type=JointActuationType.POSITION_VELOCITY if implicit_pd else JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Z.to_mat33(),
        q_j_min=[-0.5, -0.6 * math.pi] if limits else None,
        q_j_max=[0.5, 0.6 * math.pi] if limits else None,
        a_j=[0.1, 0.2] if dynamic else None,
        b_j=[0.01, 0.02] if dynamic else None,
        k_p_j=[10.0, 20.0] if implicit_pd else None,
        k_d_j=[0.01, 0.02] if implicit_pd else None,
        world_index=world_index,
    )
    _builder.add_geometry(
        name="base/cylinder",
        body=-1,
        shape=CylinderShape(0.025, 1.0),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.1, 0.1, 0.1),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_cylindrical_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test binary cylindrical joints.

    This world consists of two rigid bodies connected via a binary cylindrical
    joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        ground (bool): Whether to include a ground plane in the world.
        dynamic (bool): Whether to enable dynamic properties for the joint.
        implicit_pd (bool): Whether to enable implicit PD control for the joint.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilderKamino(default_world=False)
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
        act_type=JointActuationType.POSITION_VELOCITY if implicit_pd else JointActuationType.FORCE,
        bid_B=bid_B,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.0, 0.0, z_offset),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=Axis.Z.to_mat33(),
        q_j_min=[-0.5, -0.6 * math.pi] if limits else None,
        q_j_max=[0.5, 0.6 * math.pi] if limits else None,
        a_j=[0.1, 0.2] if dynamic else None,
        b_j=[0.01, 0.02] if dynamic else None,
        k_p_j=[10.0, 20.0] if implicit_pd else None,
        k_d_j=[0.01, 0.02] if implicit_pd else None,
        world_index=world_index,
    )
    _builder.add_geometry(
        name="base/cylinder",
        body=bid_B,
        shape=CylinderShape(0.025, 1.0),
        world_index=world_index,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.1, 0.1, 0.1),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_universal_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test unary universal joints.

    This world consists of a single rigid body connected to the world via a unary
    universal joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
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
        _builder = ModelBuilderKamino(default_world=False)
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
    _builder.add_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_universal_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test binary universal joints.

    This world consists of two rigid bodies connected via a binary universal
    joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
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
        _builder = ModelBuilderKamino(default_world=False)
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
    _builder.add_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_spherical_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test unary spherical joints.

    This world consists of a single rigid body connected to the world via a unary
    spherical joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
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
        _builder = ModelBuilderKamino(default_world=False)
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
    _builder.add_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_spherical_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test binary spherical joints.

    This world consists of two rigid bodies connected via a binary spherical
    joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
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
        _builder = ModelBuilderKamino(default_world=False)
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
    _builder.add_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_gimbal_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test unary gimbal joints.

    This world consists of a single rigid body connected to the world via a unary
    gimbal joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
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
        _builder = ModelBuilderKamino(default_world=False)
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
    _builder.add_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_gimbal_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test binary gimbal joints.

    This world consists of two rigid bodies connected via a binary gimbal
    joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
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
        _builder = ModelBuilderKamino(default_world=False)
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
    _builder.add_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_unary_cartesian_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test unary cartesian joints.

    This world consists of a single rigid body connected to the world via a unary
    cartesian joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        ground (bool): Whether to include a ground plane in the world.
        dynamic (bool): Whether to enable dynamic properties for the joint.
        implicit_pd (bool): Whether to enable implicit PD control for the joint.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilderKamino(default_world=False)
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
        act_type=JointActuationType.POSITION_VELOCITY if implicit_pd else JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.25, -0.25, -0.25),
        F_r_Fj=vec3f(-0.25, -0.25, -0.25),
        X_j=Axis.X.to_mat33(),
        q_j_min=[-1.0, -1.0, -1.0] if limits else None,
        q_j_max=[1.0, 1.0, 1.0] if limits else None,
        a_j=[0.1, 0.2, 0.3] if dynamic else None,
        b_j=[0.01, 0.02, 0.03] if dynamic else None,
        k_p_j=[10.0, 20.0, 30.0] if implicit_pd else None,
        k_d_j=[0.01, 0.02, 0.03] if implicit_pd else None,
        world_index=world_index,
    )
    _builder.add_geometry(
        name="base/box",
        body=-1,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
        group=2,
        collides=2,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
            body=-1,
            shape=BoxShape(20.0, 20.0, 1.0),
            offset=transformf(0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 1.0),
            world_index=world_index,
        )

    # Return the populated builder
    return _builder


def build_binary_cartesian_joint_test(
    builder: ModelBuilderKamino | None = None,
    z_offset: float = 0.0,
    new_world: bool = True,
    limits: bool = True,
    ground: bool = True,
    dynamic: bool = False,
    implicit_pd: bool = False,
    world_index: int = 0,
) -> ModelBuilderKamino:
    """
    Builds a world to test binary cartesian joints.

    This world consists of two rigid bodies connected via a binary cartesian
    joint, with optional limits applied to the joint degrees of freedom.

    Args:
        builder (ModelBuilderKamino | None): An optional existing ModelBuilderKamino to which the entities will be added.
        z_offset (float): A vertical offset to apply to the rigid body position.
        new_world (bool): Whether to create a new world in the builder, to which entities will be added.\n
            If `False`, the contents are added to the existing world specified by `world_index`.\n
            If `True`, a new world is created and added to the builder. In this case the `world_index`
            argument is ignored, and the index of the newly created world will be used instead.
        limits (bool): Whether to enable limits on the joint degrees of freedom.
        ground (bool): Whether to include a ground plane in the world.
        dynamic (bool): Whether to enable dynamic properties for the joint.
        implicit_pd (bool): Whether to enable implicit PD control for the joint.
        world_index (int): The index of the world in the builder where the test model should be added.
    """
    # Create a new builder if none is provided
    if builder is None:
        _builder = ModelBuilderKamino(default_world=False)
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
        act_type=JointActuationType.POSITION_VELOCITY if implicit_pd else JointActuationType.FORCE,
        bid_B=bid_B,
        bid_F=bid_F,
        B_r_Bj=vec3f(0.25, -0.25, -0.25),
        F_r_Fj=vec3f(-0.25, -0.25, -0.25),
        X_j=Axis.X.to_mat33(),
        q_j_min=[-1.0, -1.0, -1.0] if limits else None,
        q_j_max=[1.0, 1.0, 1.0] if limits else None,
        a_j=[0.1, 0.2, 0.3] if dynamic else None,
        b_j=[0.01, 0.02, 0.03] if dynamic else None,
        k_p_j=[10.0, 20.0, 30.0] if implicit_pd else None,
        k_d_j=[0.01, 0.02, 0.03] if implicit_pd else None,
        world_index=world_index,
    )
    _builder.add_geometry(
        name="base/box",
        body=bid_B,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )
    _builder.add_geometry(
        name="follower/box",
        body=bid_F,
        shape=BoxShape(0.5, 0.5, 0.5),
        world_index=world_index,
    )

    # Add a static collision geometry for the plane
    if ground:
        _builder.add_geometry(
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
) -> ModelBuilderKamino:
    """
    Constructs a model builder containing a world for each joint type.

    Args:
        z_offset (float): A vertical offset to apply to the initial position of the box.
        ground (bool): Whether to add a static ground plane to the model.

    Returns:
        ModelBuilderKamino: The populated model builder.
    """
    # Create a new builder to populate
    _builder = ModelBuilderKamino(default_world=False)

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
# Builders - Geometry Tests
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
) -> ModelBuilderKamino:
    """
    Generates a ModelBuilderKamino for a given shape combination with specified parameters.

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
        ModelBuilderKamino:
            The constructed ModelBuilderKamino with the specified shape combination.
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
    builder: ModelBuilderKamino = ModelBuilderKamino(default_world=True)
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
    builder.add_geometry(body=bid0, name="bottom_" + bottom, shape=bottom_shape)
    builder.add_geometry(body=bid1, name="top_" + top, shape=top_shape)

    # Optionally add a ground geom below the bottom shape
    if ground_box or ground_plane:
        if ground_z is not None:
            z_g_offset = ground_z
        else:
            z_g_offset = float(get_shape_bottom_position(r_b, bottom_shape).z - r_dz.z)
        if ground_box:
            utils.add_ground_box(builder, z_offset=z_g_offset)
        if ground_plane:
            utils.add_ground_plane(builder, z_offset=z_g_offset)

    # Debug output
    msg.debug(
        "[%s]:\nBODIES:\n%s\nGEOMS:\n%s\n",
        shapes,
        builder.bodies,
        builder.geoms,
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
) -> ModelBuilderKamino:
    """
    Generates a builder containing a world for each specified shape combination.

    Args:
        shape_pairs (list[tuple[str, str]]):
            A list of tuples specifying the names of the bottom and top shapes
            for each combination (e.g., [("box", "sphere"), ("cylinder", "cone")]).
        **kwargs:
            Additional keyword arguments to be passed to `make_single_shape_pair_builder`.
    Returns:
        ModelBuilderKamino
            A ModelBuilderKamino containing a world for each specified shape combination.
    """
    # Create an empty ModelBuilderKamino to hold all shape pair worlds
    builder = ModelBuilderKamino(default_world=False)

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
