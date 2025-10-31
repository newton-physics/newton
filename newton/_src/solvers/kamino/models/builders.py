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
KAMINO: MODELS: MODEL BUILDER FUNCTIONS
"""

import math

import warp as wp

from ..core import ModelBuilder
from ..core.inertia import (
    solid_cuboid_body_moment_of_inertia,
    solid_cylinder_body_moment_of_inertia,
    solid_sphere_body_moment_of_inertia,
)
from ..core.joints import JointActuationType, JointDoFType
from ..core.math import FLOAT32_MAX, FLOAT32_MIN, I_3
from ..core.shapes import BoxShape, CylinderShape, SphereShape
from ..core.types import Axis, transformf, vec3f, vec6f

###
# Module interface
###

__all__ = [
    "add_ground_geom",
    "build_box_on_plane",
    "build_box_pendulum",
    "build_box_pendulum_vertical",
    "build_boxes_fourbar",
    "build_boxes_hinged",
    "build_boxes_nunchaku",
    "build_boxes_nunchaku_vertical",
    "offset_builder",
]


###
# Builder modifiers
###


def add_ground_geom(builder: ModelBuilder, group: int = 1, collides: int = 1) -> int:
    """Adds a static collision layer and geometry to a given builder to represent a flat ground plane."""
    builder.add_collision_layer("world")
    gid = builder.add_collision_geometry(
        body_id=-1,
        shape=BoxShape(20.0, 20.0, 1.0),
        offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
        group=group,
        collides=collides,
    )
    return gid


def offset_builder(builder: ModelBuilder, offset: transformf):
    """Offsets a model builder by a given transform."""
    for i in range(builder.num_bodies):
        # Eigen::Ref<Math::Vector7> s_i(state.segment<7>(7 * i));
        # Math::Matrix3 R_i = R_offset * R_of(s_i.segment<4>(3));
        # Math::Vector3 r_i = r_offset + R_offset * s_i.segment<3>(0);
        # s_i << r_i, to_quaternion(R_i);
        builder.bodies[i].q_i_0 = wp.mul(offset, builder.bodies[i].q_i_0)


def add_velocity_bias(builder: ModelBuilder, bias: vec6f):
    """Offsets a model builder by a given transform."""
    for i in range(builder.num_bodies):
        # Eigen::Ref<Math::Vector7> s_i(state.segment<7>(7 * i));
        # Math::Matrix3 R_i = R_offset * R_of(s_i.segment<4>(3));
        # Math::Vector3 r_i = r_offset + R_offset * s_i.segment<3>(0);
        # s_i << r_i, to_quaternion(R_i);
        builder.bodies[i].u_i_0 += bias


###
# Builders for unit-tests
###


def build_revolute_joint_test_system(builder: ModelBuilder):
    ###
    # Model parameters
    ###

    # Define the Base body pose and twist
    r_B_0 = vec3f(0.0, 0.0, 0.0)
    q_B_0 = wp.quat_identity()
    u_B_0 = vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Define the joint parameters
    B_r_Bj = vec3f(0.5, 0.0, 0.0)  # Offset of the joint frame in the Base body
    F_r_Fj = vec3f(-0.5, 0.0, 0.0)  # Offset of the joint frame in the Follower body
    X_j = Axis.Y.to_mat33()  # Joint axis in the Base body frame

    # Define the Follower body pose and twist
    r_F_0 = vec3f(1.0, 0.0, 0.0)
    q_F_0 = wp.quat_identity()
    u_F_0 = vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ###
    # Model definition
    ###

    # Define the Base body
    bid0 = builder.add_body(
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(r_B_0, q_B_0),
        u_i_0=u_B_0,
    )

    # Define the Follower body
    bid1 = builder.add_body(
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(r_F_0, q_F_0),
        u_i_0=u_F_0,
    )

    # Define the joint between the two bodies
    builder.add_joint(
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=bid0,
        bid_F=bid1,
        B_r_Bj=B_r_Bj,
        F_r_Fj=F_r_Fj,
        X_j=X_j,
        q_j_min=[-0.25 * math.pi],
        q_j_max=[0.25 * math.pi],
    )


###
# Builders for primitives models
###

BuilderInfo = tuple[list[int], list[int], list[int]]


def build_box_on_plane(builder: ModelBuilder, z_offset: float = 0.0, ground: bool = True) -> BuilderInfo:
    # Create lists of BIDs, JIDs and GIDs
    bids = []
    jids = []
    gids = []

    # Add first body
    bid0 = builder.add_body(
        m_i=1.0,
        i_I_i=solid_cuboid_body_moment_of_inertia(1.0, 0.2, 0.2, 0.2),
        q_i_0=transformf(0.0, 0.0, 0.1 + z_offset, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid0)

    # Add a collision layer and geometries
    builder.add_collision_layer("box")
    gids.append(builder.add_collision_geometry(body_id=bid0, shape=BoxShape(0.2, 0.2, 0.2)))

    # Add a static collision layer and geometry for the plane
    if ground:
        builder.add_collision_layer("world")
        gids.append(
            builder.add_collision_geometry(
                body_id=-1,
                shape=BoxShape(20.0, 20.0, 1.0),
                offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            )
        )

    # Return the lists of element indices
    return bids, jids, gids


def build_box_pendulum(builder: ModelBuilder, z_offset: float = 0.7, ground: bool = True) -> BuilderInfo:
    # Create lists of BIDs, JIDs and GIDs
    bids = []
    jids = []
    gids = []

    # Model constants
    m = 1.0
    d = 0.5
    w = 0.1
    h = 0.1
    z_0 = z_offset  # Initial z offset for the body

    # Add box pendulum body
    bid0 = builder.add_body(
        m_i=m,
        i_I_i=solid_cuboid_body_moment_of_inertia(m, d, w, h),
        q_i_0=transformf(0.5 * d, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid0)

    # Add a joint between the two bodies
    jid0 = builder.add_joint(
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid0,
        B_r_Bj=vec3f(0.0, 0.0, 0.5 * h + z_0),
        F_r_Fj=vec3f(-0.5 * d, 0.0, 0.0),
        X_j=Axis.Y.to_mat33(),
    )
    jids.append(jid0)

    # Add a collision layer and geometries
    builder.add_collision_layer("primary")
    gids.append(builder.add_collision_geometry(body_id=bid0, shape=BoxShape(d, w, h)))

    # Add a static collision layer and geometry for the plane
    if ground:
        builder.add_collision_layer("world")
        gids.append(
            builder.add_collision_geometry(
                body_id=-1,
                shape=BoxShape(20.0, 20.0, 1.0),
                offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            )
        )

    # Return the lists of element indices
    return bids, jids, gids


def build_box_pendulum_vertical(builder: ModelBuilder, z_offset: float = 0.7, ground: bool = True) -> BuilderInfo:
    # Create lists of BIDs, JIDs and GIDs
    bids = []
    jids = []
    gids = []

    # Model constants
    m = 1.0
    d = 0.1
    w = 0.1
    h = 0.5
    z_0 = z_offset  # Initial z offset for the body

    # Add box pendulum body
    bid0 = builder.add_body(
        m_i=m,
        i_I_i=solid_cuboid_body_moment_of_inertia(m, d, w, h),
        q_i_0=transformf(0.0, 0.0, -0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid0)

    # Add a joint between the two bodies
    jid0 = builder.add_joint(
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid0,
        B_r_Bj=vec3f(0.0, 0.0, 0.0 + z_0),
        F_r_Fj=vec3f(0.0, 0.0, 0.5 * h),
        X_j=Axis.Y.to_mat33(),
    )
    jids.append(jid0)

    # Add a collision layer and geometries
    builder.add_collision_layer("primary")
    gids.append(builder.add_collision_geometry(body_id=bid0, shape=BoxShape(d, w, h)))

    # Add a static collision layer and geometry for the plane
    if ground:
        builder.add_collision_layer("world")
        gids.append(
            builder.add_collision_geometry(
                body_id=-1,
                shape=BoxShape(20.0, 20.0, 1.0),
                offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            )
        )

    # Return the lists of element indices
    return bids, jids, gids


def build_cartpole(builder: ModelBuilder, z_offset: float = 0.0, ground: bool = False) -> BuilderInfo:
    # Create lists of BIDs, JIDs and GIDs
    bids = []
    jids = []
    gids = []

    # Model constants
    m = 1.0
    dims_cart = (0.1, 0.1, 0.1)
    dims_pendulum = (0.02, 0.4)
    z_0 = z_offset  # Initial z offset for the body

    # Add box cart body
    z_0_cart = z_0
    bid0 = builder.add_body(
        m_i=m,
        i_I_i=solid_cuboid_body_moment_of_inertia(m, *dims_cart),
        q_i_0=transformf(0.0, 0.0, z_0_cart, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid0)

    # Add box pendulum body
    z_0_pendulum = 0.5 * dims_pendulum[1] + z_0_cart
    bid1 = builder.add_body(
        m_i=m,
        i_I_i=solid_cylinder_body_moment_of_inertia(m, *dims_pendulum),
        q_i_0=transformf(0.0, 0.0, z_0_pendulum, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid1)

    # Add a prismatic joint for the cart
    jid0 = builder.add_joint(
        dof_type=JointDoFType.PRISMATIC,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid0,
        B_r_Bj=vec3f(0.0, 0.0, z_0),
        F_r_Fj=vec3f(0.0, 0.0, z_0_cart),
        X_j=Axis.X.to_mat33(),
    )
    jids.append(jid0)

    # Add a revolute joint for the pendulum
    jid0 = builder.add_joint(
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=bid0,
        bid_F=bid1,
        B_r_Bj=vec3f(0.0, 0.0, 0.0),
        F_r_Fj=vec3f(0.0, 0.0, -0.5 * dims_pendulum[1]),
        X_j=Axis.Y.to_mat33(),
    )
    jids.append(jid0)

    # Add a collision layer and geometries
    builder.add_collision_layer("primary")
    gids.append(builder.add_collision_geometry(body_id=bid0, shape=BoxShape(*dims_cart)))
    gids.append(builder.add_collision_geometry(body_id=bid1, shape=CylinderShape(*dims_pendulum)))

    # Add a static collision layer and geometry for the plane
    if ground:
        builder.add_collision_layer("world")
        gids.append(
            builder.add_collision_geometry(
                body_id=-1,
                shape=BoxShape(20.0, 20.0, 1.0),
                offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            )
        )

    # Return the lists of element indices
    return bids, jids, gids


def build_boxes_hinged(builder: ModelBuilder, z_offset: float = 0.0, ground: bool = True) -> BuilderInfo:
    # Create lists of BIDs, JIDs and GIDs
    bids = []
    jids = []
    gids = []

    # Model constants
    m_0 = 1.0
    m_1 = 1.0
    d = 0.5
    w = 0.1
    h = 0.1
    z0 = z_offset  # Initial z offset for the bodies

    # Add first body
    bid0 = builder.add_body(
        m_i=m_0,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_0, d, w, h),
        q_i_0=transformf(0.25, -0.05, 0.05 + z0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid0)

    # Add second body
    bid1 = builder.add_body(
        m_i=m_1,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_1, d, w, h),
        q_i_0=transformf(0.75, 0.05, 0.05 + z0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid1)

    # Add a joint between the two bodies
    jid0 = builder.add_joint(
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=bid0,
        bid_F=bid1,
        B_r_Bj=vec3f(0.25, 0.05, 0.0),
        F_r_Fj=vec3f(-0.25, -0.05, 0.0),
        X_j=Axis.Y.to_mat33(),
    )
    jids.append(jid0)

    # Add a collision layer and geometries
    builder.add_collision_layer("primary")
    gids.append(builder.add_collision_geometry(body_id=bid0, shape=BoxShape(d, w, h), group=2, collides=3))
    gids.append(builder.add_collision_geometry(body_id=bid1, shape=BoxShape(d, w, h), group=3, collides=5))

    # Add a static collision layer and geometry for the plane
    if ground:
        builder.add_collision_layer("world")
        gids.append(
            builder.add_collision_geometry(
                body_id=-1,
                shape=BoxShape(20.0, 20.0, 1.0),
                offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
                group=1,
                collides=7,
            )
        )

    # Return the lists of element indices
    return bids, jids, gids


def build_boxes_nunchaku(builder: ModelBuilder, z_offset: float = 0.0, ground: bool = True) -> BuilderInfo:
    # Create lists of BIDs, JIDs and GIDs
    bids = []
    jids = []
    gids = []

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
    bid0 = builder.add_body(
        m_i=m_0,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_0, d, w, h),
        q_i_0=transformf(0.5 * d, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid0)

    # Add second body
    bid1 = builder.add_body(
        m_i=m_1,
        i_I_i=solid_sphere_body_moment_of_inertia(m_1, r),
        q_i_0=transformf(r + d, 0.0, r + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid1)

    # Add third body
    bid2 = builder.add_body(
        m_i=m_2,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_2, d, w, h),
        q_i_0=transformf(1.5 * d + 2.0 * r, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid2)

    # Add a joint between the first and second body
    jid0 = builder.add_joint(
        dof_type=JointDoFType.SPHERICAL,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid0,
        bid_F=bid1,
        B_r_Bj=vec3f(0.5 * d, 0.0, 0.0),
        F_r_Fj=vec3f(-r, 0.0, 0.0),
        X_j=I_3,
    )
    jids.append(jid0)

    # Add a joint between the second and third body
    jid1 = builder.add_joint(
        dof_type=JointDoFType.SPHERICAL,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid1,
        bid_F=bid2,
        B_r_Bj=vec3f(r, 0.0, 0.0),
        F_r_Fj=vec3f(-0.5 * d, 0.0, 0.0),
        X_j=I_3,
    )
    jids.append(jid1)

    # Add a collision layer and geometries
    builder.add_collision_layer("primary")
    gids.append(builder.add_collision_geometry(body_id=bid0, shape=BoxShape(d, w, h), group=2, collides=3))
    gids.append(builder.add_collision_geometry(body_id=bid1, shape=SphereShape(r), group=3, collides=5))
    gids.append(builder.add_collision_geometry(body_id=bid2, shape=BoxShape(d, w, h), group=2, collides=3))

    # Add a static collision layer and geometry for the plane
    if ground:
        builder.add_collision_layer("world")
        gids.append(
            builder.add_collision_geometry(
                body_id=-1,
                shape=BoxShape(20.0, 20.0, 1.0),
                offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
                group=1,
                collides=7,
            )
        )

    # Return the lists of element indices
    return bids, jids, gids


def build_boxes_nunchaku_vertical(builder: ModelBuilder, z_offset: float = 0.0, ground: bool = True) -> BuilderInfo:
    # Create lists of BIDs, JIDs and GIDs
    bids = []
    jids = []
    gids = []

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
    bid0 = builder.add_body(
        m_i=m_0,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_0, d, w, h),
        q_i_0=transformf(0.0, 0.0, 0.5 * h + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid0)

    # Add second body
    bid1 = builder.add_body(
        m_i=m_1,
        i_I_i=solid_sphere_body_moment_of_inertia(m_1, r),
        q_i_0=transformf(0.0, 0.0, h + r + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid1)

    # Add third body
    bid2 = builder.add_body(
        m_i=m_2,
        i_I_i=solid_cuboid_body_moment_of_inertia(m_2, d, w, h),
        q_i_0=transformf(0.0, 0.0, 1.5 * h + 2.0 * r + z_0, 0.0, 0.0, 0.0, 1.0),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    bids.append(bid2)

    # Add a joint between the first and second body
    jid0 = builder.add_joint(
        dof_type=JointDoFType.SPHERICAL,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid0,
        bid_F=bid1,
        B_r_Bj=vec3f(0.0, 0.0, 0.5 * h),
        F_r_Fj=vec3f(0.0, 0.0, -r),
        X_j=I_3,
    )
    jids.append(jid0)

    # Add a joint between the second and third body
    jid1 = builder.add_joint(
        dof_type=JointDoFType.SPHERICAL,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid1,
        bid_F=bid2,
        B_r_Bj=vec3f(0.0, 0.0, r),
        F_r_Fj=vec3f(0.0, 0.0, -0.5 * h),
        X_j=I_3,
    )
    jids.append(jid1)

    # Add a collision layer and geometries
    builder.add_collision_layer("primary")
    gids.append(builder.add_collision_geometry(body_id=bid0, shape=BoxShape(d, w, h), group=2, collides=3))
    gids.append(builder.add_collision_geometry(body_id=bid1, shape=SphereShape(r), group=3, collides=5))
    gids.append(builder.add_collision_geometry(body_id=bid2, shape=BoxShape(d, w, h), group=2, collides=3))

    # Add a static collision layer and geometry for the plane
    if ground:
        builder.add_collision_layer("world")
        gids.append(
            builder.add_collision_geometry(
                body_id=-1,
                shape=BoxShape(20.0, 20.0, 1.0),
                offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
                group=1,
                collides=7,
            )
        )

    # Return the lists of element indices
    return bids, jids, gids


def build_boxes_fourbar(
    builder: ModelBuilder,
    z_offset: float = 0.0,
    fixedbase: bool = False,
    limits: bool = True,
    ground: bool = True,
    verbose: bool = False,
) -> BuilderInfo:
    # Create lists of BIDs, JIDs and GIDs
    bids = []
    jids = []
    gids = []

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

    bid1 = builder.add_body(
        m_i=m_i,
        i_I_i=i_I_i_1,
        q_i_0=q_i_1,
        u_i_0=vec6f(0.0),
    )
    bids.append(bid1)

    bid2 = builder.add_body(
        m_i=m_i,
        i_I_i=i_I_i_2,
        q_i_0=q_i_2,
        u_i_0=vec6f(0.0),
    )
    bids.append(bid2)

    bid3 = builder.add_body(
        m_i=m_i,
        i_I_i=i_I_i_3,
        q_i_0=q_i_3,
        u_i_0=vec6f(0.0),
    )
    bids.append(bid3)

    bid4 = builder.add_body(
        m_i=m_i,
        i_I_i=i_I_i_4,
        q_i_0=q_i_4,
        u_i_0=vec6f(0.0),
    )
    bids.append(bid4)

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
        jid1 = builder.add_joint(
            dof_type=JointDoFType.FIXED,
            act_type=JointActuationType.PASSIVE,
            bid_B=-1,
            bid_F=bid1,
            B_r_Bj=vec3f(0.0),
            F_r_Fj=-r_b1,
            X_j=I_3,
        )
        jids.append(jid1)

    jid1 = builder.add_joint(
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=bid1,
        bid_F=bid2,
        B_r_Bj=r_j1 - r_b1,
        F_r_Fj=r_j1 - r_b2,
        X_j=X_j,
        q_j_min=[qmin],
        q_j_max=[qmax],
    )
    jids.append(jid1)

    jid2 = builder.add_joint(
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid2,
        bid_F=bid3,
        B_r_Bj=r_j2 - r_b2,
        F_r_Fj=r_j2 - r_b3,
        X_j=X_j,
        q_j_min=[qmin],
        q_j_max=[qmax],
    )
    jids.append(jid2)

    jid3 = builder.add_joint(
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.FORCE,
        bid_B=bid3,
        bid_F=bid4,
        B_r_Bj=r_j3 - r_b3,
        F_r_Fj=r_j3 - r_b4,
        X_j=X_j,
        q_j_min=[qmin],
        q_j_max=[qmax],
    )
    jids.append(jid3)

    jid4 = builder.add_joint(
        dof_type=JointDoFType.REVOLUTE,
        act_type=JointActuationType.PASSIVE,
        bid_B=bid4,
        bid_F=bid1,
        B_r_Bj=r_j4 - r_b4,
        F_r_Fj=r_j4 - r_b1,
        X_j=X_j,
        q_j_min=[qmin],
        q_j_max=[qmax],
    )
    jids.append(jid4)

    ###
    # Geometries
    ###

    # Add a collision layer and geometries
    builder.add_collision_layer("primary")
    gids.append(builder.add_collision_geometry(body_id=bid1, shape=BoxShape(d_1, w_1, h_1), group=1, collides=1))
    gids.append(builder.add_collision_geometry(body_id=bid2, shape=BoxShape(d_2, w_2, h_2), group=1, collides=1))
    gids.append(builder.add_collision_geometry(body_id=bid3, shape=BoxShape(d_3, w_3, h_3), group=1, collides=1))
    gids.append(builder.add_collision_geometry(body_id=bid4, shape=BoxShape(d_4, w_4, h_4), group=1, collides=1))

    # Add a static collision layer and geometry for the plane
    if ground:
        builder.add_collision_layer("world")
        gids.append(
            builder.add_collision_geometry(
                body_id=-1,
                shape=BoxShape(20.0, 20.0, 1.0),
                offset=transformf(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0),
            )
        )

    # Return the lists of element indices
    return bids, jids, gids
