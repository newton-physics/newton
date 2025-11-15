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
KAMINO: UNIT TESTS: CORE: BUILDER
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.bodies import RigidBodyDescriptor
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.geometry import CollisionGeometryDescriptor, GeometryDescriptor
from newton._src.solvers.kamino.core.gravity import (
    GRAVITY_ACCEL_DEFAULT,
    GRAVITY_DIREC_DEFAULT,
    GRAVITY_NAME_DEFAULT,
)
from newton._src.solvers.kamino.core.joints import JointActuationType, JointDescriptor, JointDoFType
from newton._src.solvers.kamino.core.materials import MaterialDescriptor
from newton._src.solvers.kamino.core.model import Model
from newton._src.solvers.kamino.core.shapes import SphereShape
from newton._src.solvers.kamino.core.types import Axis, mat33f, transformf, vec6f
from newton._src.solvers.kamino.models.builders import (
    build_box_on_plane,
    build_box_pendulum,
    build_boxes_fourbar,
    build_boxes_hinged,
    build_boxes_nunchaku,
    build_cartpole,
)
from newton._src.solvers.kamino.models.utils import (
    make_heterogeneous_builder,
    make_homogeneous_builder,
)
from newton._src.solvers.kamino.utils import logger as msg

###
# Utilities
###


def assert_model_matches_builder(test: unittest.TestCase, builder: ModelBuilder, model: Model):
    """
    Assert that a constructed model matches the specifications of the given builder.

    Args:
        test: The test case instance.
        builder: The model builder instance.
        model: The constructed model instance.
    """
    # Check model sizes and offsets
    for w, world in enumerate(model.worlds):
        test.assertEqual(world.wid, w)
        test.assertEqual(model.info.num_bodies.numpy()[w], world.num_bodies)
        test.assertEqual(model.info.num_joints.numpy()[w], world.num_joints)
        test.assertEqual(model.info.num_body_dofs.numpy()[w], world.num_body_dofs)
        test.assertEqual(model.info.num_joint_coords.numpy()[w], world.num_joint_coords)
        test.assertEqual(model.info.num_joint_dofs.numpy()[w], world.num_joint_dofs)
        test.assertEqual(model.info.num_passive_joint_coords.numpy()[w], world.num_passive_joint_coords)
        test.assertEqual(model.info.num_passive_joint_dofs.numpy()[w], world.num_passive_joint_dofs)
        test.assertEqual(model.info.num_actuated_joint_coords.numpy()[w], world.num_actuated_joint_coords)
        test.assertEqual(model.info.num_actuated_joint_dofs.numpy()[w], world.num_actuated_joint_dofs)
        test.assertEqual(model.info.num_joint_cts.numpy()[w], world.num_joint_cts)
        test.assertEqual(model.info.bodies_offset.numpy()[w], world.bodies_idx_offset)
        test.assertEqual(model.info.joints_offset.numpy()[w], world.joints_idx_offset)
        test.assertEqual(model.info.body_dofs_offset.numpy()[w], world.body_dofs_idx_offset)
        test.assertEqual(model.info.joint_coords_offset.numpy()[w], world.joint_coords_idx_offset)
        test.assertEqual(model.info.joint_dofs_offset.numpy()[w], world.joint_dofs_idx_offset)
        test.assertEqual(model.info.joint_passive_coords_offset.numpy()[w], world.passive_joint_coords_idx_offset)
        test.assertEqual(model.info.joint_passive_dofs_offset.numpy()[w], world.passive_joint_dofs_idx_offset)
        test.assertEqual(model.info.joint_actuated_coords_offset.numpy()[w], world.actuated_joint_coords_idx_offset)
        test.assertEqual(model.info.joint_actuated_dofs_offset.numpy()[w], world.actuated_joint_dofs_idx_offset)
        test.assertEqual(model.info.joint_cts_offset.numpy()[w], world.joint_cts_idx_offset)

    for i in range(model.size.sum_of_num_bodies):
        test.assertEqual(model.bodies.wid.numpy()[i], builder.bodies[i].wid)
        test.assertEqual(model.bodies.bid.numpy()[i], builder.bodies[i].bid)

    for i in range(model.size.sum_of_num_joints):
        wid = builder.joints[i].wid
        bid_offset = builder.worlds[wid].bodies_idx_offset
        test.assertEqual(model.joints.wid.numpy()[i], builder.joints[i].wid)
        test.assertEqual(model.joints.jid.numpy()[i], builder.joints[i].jid)
        test.assertEqual(
            model.joints.bid_B.numpy()[i], builder.joints[i].bid_B + bid_offset if builder.joints[i].bid_B >= 0 else -1
        )
        test.assertEqual(
            model.joints.bid_F.numpy()[i], builder.joints[i].bid_F + bid_offset if builder.joints[i].bid_F >= 0 else -1
        )

    for i in range(model.size.sum_of_num_collision_geoms):
        wid = builder.collision_geoms[i].wid
        bid_offset = builder.worlds[wid].bodies_idx_offset
        test.assertEqual(model.cgeoms.wid.numpy()[i], builder.collision_geoms[i].wid)
        test.assertEqual(model.cgeoms.gid.numpy()[i], builder.collision_geoms[i].gid)
        test.assertEqual(
            model.cgeoms.bid.numpy()[i],
            builder.collision_geoms[i].bid + bid_offset if builder.collision_geoms[i].bid >= 0 else -1,
        )

    for i in range(model.size.sum_of_num_physical_geoms):
        wid = builder.physical_geoms[i].wid
        bid_offset = builder.worlds[wid].bodies_idx_offset
        test.assertEqual(model.pgeoms.wid.numpy()[i], builder.physical_geoms[i].wid)
        test.assertEqual(model.pgeoms.gid.numpy()[i], builder.physical_geoms[i].gid)
        test.assertEqual(
            model.pgeoms.bid.numpy()[i],
            builder.physical_geoms[i].bid + bid_offset if builder.physical_geoms[i].bid >= 0 else -1,
        )

    # Optional printout for debugging
    msg.info("model.bodies.wid: %s", model.bodies.wid)
    msg.info("model.bodies.bid: %s\n", model.bodies.bid)
    msg.info("model.joints.wid: %s", model.joints.wid)
    msg.info("model.joints.jid: %s", model.joints.jid)
    msg.info("model.joints.bid_B: %s", model.joints.bid_B)
    msg.info("model.joints.bid_F: %s\n", model.joints.bid_F)
    msg.info("model.cgeoms.wid: %s", model.cgeoms.wid)
    msg.info("model.cgeoms.gid: %s", model.cgeoms.gid)
    msg.info("model.cgeoms.bid: %s", model.cgeoms.bid)
    msg.info("model.pgeoms.wid: %s", model.pgeoms.wid)
    msg.info("model.pgeoms.gid: %s", model.pgeoms.gid)
    msg.info("model.pgeoms.bid: %s\n", model.pgeoms.bid)
    msg.info("model.info.bodies_offset: %s", model.info.bodies_offset)
    msg.info("model.info.joints_offset: %s", model.info.joints_offset)
    msg.info("model.info.body_dofs_offset: %s", model.info.body_dofs_offset)
    msg.info("model.info.joint_coords_offset: %s", model.info.joint_coords_offset)
    msg.info("model.info.joint_dofs_offset: %s", model.info.joint_dofs_offset)
    msg.info("model.info.joint_cts_offset: %s\n", model.info.joint_cts_offset)
    msg.info("model.info.joint_passive_coords_offset: %s", model.info.joint_passive_coords_offset)
    msg.info("model.info.joint_passive_dofs_offset: %s", model.info.joint_passive_dofs_offset)
    msg.info("model.info.joint_actuated_coords_offset: %s", model.info.joint_actuated_coords_offset)
    msg.info("model.info.joint_actuated_dofs_offset: %s\n", model.info.joint_actuated_dofs_offset)
    msg.info("model.info.base_body_index: %s", model.info.base_body_index)
    msg.info("model.info.base_joint_index: %s", model.info.base_joint_index)


###
# Tests
###


class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True to enable verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default(self):
        builder = ModelBuilder()
        self.assertEqual(builder.num_worlds, 0)
        self.assertEqual(builder.num_bodies, 0)
        self.assertEqual(builder.num_joints, 0)
        self.assertEqual(builder.num_collision_geoms, 0)
        self.assertEqual(builder.num_physical_geoms, 0)
        self.assertEqual(builder.num_materials, 1)  # Default material is always created
        self.assertEqual(builder.num_body_dofs, 0)
        self.assertEqual(builder.num_joint_dofs, 0)
        self.assertEqual(builder.num_passive_joint_dofs, 0)
        self.assertEqual(builder.num_actuated_joint_dofs, 0)
        self.assertEqual(builder.num_joint_cts, 0)
        self.assertEqual(len(builder.bodies), 0)
        self.assertEqual(len(builder.joints), 0)
        self.assertEqual(len(builder.collision_geoms), 0)
        self.assertEqual(len(builder.physical_geoms), 0)
        self.assertEqual(len(builder.materials), 1)  # Default material is always created

    def test_01_make_default_with_world(self):
        builder = ModelBuilder(default_world=True)
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 0)
        self.assertEqual(builder.num_joints, 0)
        self.assertEqual(builder.num_collision_geoms, 0)
        self.assertEqual(builder.num_physical_geoms, 0)
        self.assertEqual(builder.num_materials, 1)  # Default material is always created
        self.assertEqual(builder.num_body_dofs, 0)
        self.assertEqual(builder.num_joint_dofs, 0)
        self.assertEqual(builder.num_passive_joint_dofs, 0)
        self.assertEqual(builder.num_actuated_joint_dofs, 0)
        self.assertEqual(builder.num_joint_cts, 0)
        self.assertEqual(len(builder.bodies), 0)
        self.assertEqual(len(builder.joints), 0)
        self.assertEqual(len(builder.collision_geoms), 0)
        self.assertEqual(len(builder.physical_geoms), 0)
        self.assertEqual(len(builder.materials), 1)  # Default material is always created

    def test_02_add_world(self):
        builder = ModelBuilder()
        wid = builder.add_world(name="test_world", up_axis=Axis.Y)
        self.assertEqual(wid, 0)
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.worlds[wid].wid, wid)
        self.assertEqual(builder.worlds[wid].name, "test_world")
        self.assertEqual(builder.up_axes[wid], Axis.Y)
        self.assertEqual(builder.gravity[wid].name, GRAVITY_NAME_DEFAULT)
        self.assertEqual(builder.gravity[wid].acceleration, GRAVITY_ACCEL_DEFAULT)
        np.testing.assert_array_equal(builder.gravity[wid].direction, np.array(GRAVITY_DIREC_DEFAULT, dtype=np.float32))

    def test_03_add_rigid_body(self):
        builder = ModelBuilder()
        wid = builder.add_world(name="test_world", up_axis=Axis.Z)

        bid = builder.add_rigid_body(
            name="test_rigid_body",
            m_i=1.0,
            i_I_i=mat33f(np.eye(3, dtype=np.float32)),
            q_i_0=transformf(),
            u_i_0=vec6f(),
            world_index=wid,
        )

        self.assertEqual(builder.num_bodies, 1)
        self.assertEqual(bid, 0)
        self.assertEqual(bid, builder.bodies[bid].bid)
        self.assertEqual(builder.bodies[bid].name, "test_rigid_body")
        self.assertEqual(builder.bodies[bid].wid, wid)
        self.assertEqual(builder.bodies[bid].m_i, 1.0)
        np.testing.assert_array_equal(builder.bodies[bid].i_I_i, np.eye(3, dtype=np.float32).flatten())
        np.testing.assert_array_equal(builder.bodies[bid].q_i_0, np.array(transformf(), dtype=np.float32))
        np.testing.assert_array_equal(builder.bodies[bid].u_i_0, np.zeros(6, dtype=np.float32))

    def test_04_add_rigid_body_descriptor(self):
        builder = ModelBuilder()
        wid = builder.add_world(name="test_world", up_axis=Axis.Z)

        body = RigidBodyDescriptor(
            name="test_rigid_body",
            m_i=2.0,
            i_I_i=mat33f(2.0 * np.eye(3, dtype=np.float32)),
            q_i_0=transformf(),
            u_i_0=vec6f(),
        )
        bid = builder.add_rigid_body_descriptor(body, world_index=wid)

        self.assertEqual(builder.num_bodies, 1)
        self.assertEqual(bid, 0)
        self.assertEqual(bid, builder.bodies[bid].bid)
        self.assertEqual(builder.bodies[bid].name, "test_rigid_body")
        self.assertEqual(builder.bodies[bid].wid, wid)
        self.assertEqual(builder.bodies[bid].m_i, 2.0)
        np.testing.assert_array_equal(builder.bodies[bid].i_I_i, 2.0 * np.eye(3, dtype=np.float32).flatten())
        np.testing.assert_array_equal(builder.bodies[bid].q_i_0, np.array(transformf(), dtype=np.float32))
        np.testing.assert_array_equal(builder.bodies[bid].u_i_0, np.zeros(6, dtype=np.float32))

    def test_05_add_duplicate_rigid_body(self):
        builder = ModelBuilder()
        wid = builder.add_world(name="test_world", up_axis=Axis.Z)

        body_0 = RigidBodyDescriptor(
            name="test_rigid_body",
            m_i=2.0,
            i_I_i=mat33f(2.0 * np.eye(3, dtype=np.float32)),
            q_i_0=transformf(),
            u_i_0=vec6f(),
        )
        builder.add_rigid_body_descriptor(body_0, world_index=wid)

        # Attempt to add the same body again and expect an error
        self.assertRaises(ValueError, builder.add_rigid_body_descriptor, body_0, world_index=wid)

    def test_06_add_joint(self):
        builder = ModelBuilder()
        wid = builder.add_world(name="test_world", up_axis=Axis.Z)

        # Define two rigid bodies to connect with a joint
        body_0 = RigidBodyDescriptor(
            name="test_rigid_body_0",
            m_i=2.0,
            i_I_i=mat33f(2.0 * np.eye(3, dtype=np.float32)),
            q_i_0=transformf(),
            u_i_0=vec6f(),
        )
        body_1 = RigidBodyDescriptor(
            name="test_rigid_body_1",
            m_i=1.0,
            i_I_i=mat33f(1.0 * np.eye(3, dtype=np.float32)),
            q_i_0=transformf(),
            u_i_0=vec6f(),
        )
        bid_0 = builder.add_rigid_body_descriptor(body_0, world_index=wid)
        bid_1 = builder.add_rigid_body_descriptor(body_1, world_index=wid)

        # Define a joint descriptor
        joint = JointDescriptor(
            name="test_joint",
            bid_B=bid_0,
            bid_F=bid_1,
            dof_type=JointDoFType.PRISMATIC,
            act_type=JointActuationType.FORCE,
        )
        jid = builder.add_joint_descriptor(joint, world_index=wid)

        self.assertEqual(builder.num_bodies, 2)
        self.assertEqual(builder.num_joints, 1)
        self.assertEqual(jid, 0)
        self.assertEqual(jid, builder.joints[jid].jid)
        self.assertEqual(builder.joints[jid].name, "test_joint")
        self.assertEqual(builder.joints[jid].wid, wid)
        self.assertEqual(builder.joints[jid].bid_B, bid_0)
        self.assertEqual(builder.joints[jid].bid_F, bid_1)
        self.assertEqual(builder.joints[jid].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder.joints[jid].act_type, JointActuationType.FORCE)

    def test_07_add_duplicate_joint(self):
        builder = ModelBuilder()
        wid = builder.add_world(name="test_world", up_axis=Axis.Z)

        # Define two rigid bodies to connect with a joint
        body_0 = RigidBodyDescriptor(
            name="test_rigid_body_0",
            m_i=2.0,
            i_I_i=mat33f(2.0 * np.eye(3, dtype=np.float32)),
            q_i_0=transformf(),
            u_i_0=vec6f(),
        )
        body_1 = RigidBodyDescriptor(
            name="test_rigid_body_1",
            m_i=1.0,
            i_I_i=mat33f(1.0 * np.eye(3, dtype=np.float32)),
            q_i_0=transformf(),
            u_i_0=vec6f(),
        )
        bid_0 = builder.add_rigid_body_descriptor(body_0, world_index=wid)
        bid_1 = builder.add_rigid_body_descriptor(body_1, world_index=wid)

        # Define a joint descriptor
        joint = JointDescriptor(
            name="test_joint",
            bid_B=bid_0,
            bid_F=bid_1,
            dof_type=JointDoFType.PRISMATIC,
            act_type=JointActuationType.FORCE,
        )
        builder.add_joint_descriptor(joint, world_index=wid)

        # Attempt to add the same joint again and expect an error
        self.assertRaises(ValueError, builder.add_joint_descriptor, joint, world_index=wid)

    def test_08_add_invalid_joint(self):
        builder = ModelBuilder()
        wid = builder.add_world(name="test_world", up_axis=Axis.Z)

        # Define a joint descriptor
        joint = JointDescriptor(
            name="test_joint",
            dof_type=JointDoFType.PRISMATIC,
            act_type=JointActuationType.FORCE,
        )
        # Attempt to add a joint without specifying bodies and expect an error
        self.assertRaises(ValueError, builder.add_joint_descriptor, joint, world_index=wid)

    def test_09_add_geometry(self):
        builder = ModelBuilder()
        wid = builder.add_world(name="test_world", up_axis=Axis.Z)
        self.assertTrue(builder.num_physical_geoms == 0)
        self.assertTrue(builder.num_collision_geoms == 0)
        self.assertEqual(len(builder.worlds[wid].physical_geometry_layers), 0)
        self.assertEqual(len(builder.worlds[wid].collision_geometry_layers), 0)

        # Define a geometry descriptor
        gid = builder.add_physical_geometry(
            name="test_geom", layer="foo", shape=SphereShape(radius=1.0), world_index=wid
        )
        self.assertEqual(builder.num_physical_geoms, 1)
        self.assertEqual(gid, 0)
        self.assertEqual(gid, builder.physical_geoms[gid].gid)
        self.assertEqual(builder.physical_geoms[gid].name, "test_geom")
        self.assertEqual(builder.physical_geoms[gid].layer, "foo")
        self.assertEqual(builder.physical_geoms[gid].bid, -1)
        self.assertEqual(builder.physical_geoms[gid].wid, wid)
        self.assertEqual(builder.physical_geoms[gid].lid, 0)
        self.assertEqual(len(builder.worlds[wid].physical_geometry_layers), 1)
        self.assertEqual(builder.worlds[wid].physical_geometry_layers[0], "foo")

        # Create a collision geometry descriptor from the geometry descriptor
        cgid = builder.add_collision_geometry(
            name="test_cgeom",
            layer="foo",
            shape=SphereShape(radius=1.0),
            world_index=wid,
        )
        self.assertEqual(builder.num_collision_geoms, 1)
        self.assertEqual(cgid, 0)
        self.assertEqual(cgid, builder.collision_geoms[cgid].gid)
        self.assertEqual(builder.collision_geoms[cgid].name, "test_cgeom")
        self.assertEqual(builder.collision_geoms[cgid].layer, "foo")
        self.assertEqual(builder.collision_geoms[cgid].bid, -1)
        self.assertEqual(builder.collision_geoms[cgid].wid, wid)
        self.assertEqual(builder.collision_geoms[cgid].lid, 0)
        self.assertEqual(builder.collision_geoms[cgid].mid, 0)
        self.assertEqual(len(builder.worlds[wid].collision_geometry_layers), 1)
        self.assertEqual(builder.worlds[wid].collision_geometry_layers[0], "foo")

    def test_10_add_geometry_descriptors(self):
        builder = ModelBuilder()
        wid = builder.add_world(name="test_world", up_axis=Axis.Z)
        self.assertTrue(builder.num_physical_geoms == 0)
        self.assertTrue(builder.num_collision_geoms == 0)
        self.assertEqual(len(builder.worlds[wid].physical_geometry_layers), 0)
        self.assertEqual(len(builder.worlds[wid].collision_geometry_layers), 0)

        # Define a geometry descriptor
        geom = GeometryDescriptor(name="test_geom", shape=SphereShape(radius=1.0))
        gid = builder.add_physical_geometry_descriptor(geom, world_index=wid)
        self.assertEqual(builder.num_physical_geoms, 1)
        self.assertEqual(gid, 0)
        self.assertEqual(gid, builder.physical_geoms[gid].gid)
        self.assertEqual(builder.physical_geoms[gid].name, "test_geom")
        self.assertEqual(builder.physical_geoms[gid].layer, "default")
        self.assertEqual(builder.physical_geoms[gid].bid, -1)
        self.assertEqual(builder.physical_geoms[gid].wid, wid)
        self.assertEqual(builder.physical_geoms[gid].lid, 0)
        self.assertEqual(len(builder.worlds[wid].physical_geometry_layers), 1)
        self.assertEqual(builder.worlds[wid].physical_geometry_layers[0], "default")

        # Create a collision geometry descriptor from the geometry descriptor
        cgeom = CollisionGeometryDescriptor(name="test_cgeom", base=geom)
        cgid = builder.add_collision_geometry_descriptor(cgeom, world_index=wid)
        self.assertEqual(builder.num_collision_geoms, 1)
        self.assertEqual(cgid, 0)
        self.assertEqual(cgid, builder.collision_geoms[cgid].gid)
        self.assertEqual(builder.collision_geoms[cgid].name, "test_cgeom")
        self.assertEqual(builder.collision_geoms[cgid].layer, "default")
        self.assertEqual(builder.collision_geoms[cgid].bid, -1)
        self.assertEqual(builder.collision_geoms[cgid].wid, wid)
        self.assertEqual(builder.collision_geoms[cgid].lid, 0)
        self.assertEqual(builder.collision_geoms[cgid].mid, 0)
        self.assertEqual(len(builder.worlds[wid].collision_geometry_layers), 1)
        self.assertEqual(builder.worlds[wid].collision_geometry_layers[0], "default")

    def test_11_add_material(self):
        builder = ModelBuilder()
        wid = builder.add_world(name="test_world", up_axis=Axis.Z)
        self.assertEqual(builder.num_materials, 1)  # Default material exists

        material = MaterialDescriptor(
            name="test_material", density=500.0, restitution=0.8, static_friction=0.6, dynamic_friction=0.4
        )

        mid = builder.add_material(material=material)
        self.assertEqual(builder.num_materials, 2)
        self.assertEqual(mid, 1)
        self.assertEqual(mid, builder.materials[mid].mid)
        self.assertEqual(builder.materials[mid].name, "test_material")
        self.assertEqual(builder.materials[mid].wid, wid)
        self.assertEqual(builder.materials[mid].density, 500.0)
        self.assertEqual(builder.materials[mid].restitution, 0.8)
        self.assertEqual(builder.materials[mid].static_friction, 0.6)
        self.assertEqual(builder.materials[mid].dynamic_friction, 0.4)

    def test_12_make_builder_box_on_plane(self):
        # Construct box-on-plane model
        builder = build_box_on_plane()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 1)
        self.assertEqual(builder.num_joints, 0)
        self.assertEqual(builder.num_collision_geoms, 2)
        self.assertEqual(builder.num_physical_geoms, 0)
        self.assertEqual(builder.worlds[0].name, "box_on_plane")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies]
        jids = [joint.jid for joint in builder.joints]
        gids = [geom.gid for geom in builder.collision_geoms]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.collision_geoms[i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.num_worlds, 1)
        self.assertEqual(model.size.sum_of_num_bodies, 1)
        self.assertEqual(model.size.sum_of_num_joints, 0)
        self.assertEqual(model.size.sum_of_num_collision_geoms, 2)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_13_make_builder_box_pendulum(self):
        # Construct box-pendulum model
        builder = build_box_pendulum()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 1)
        self.assertEqual(builder.num_joints, 1)
        self.assertEqual(builder.num_collision_geoms, 2)
        self.assertEqual(builder.num_physical_geoms, 0)
        self.assertEqual(builder.worlds[0].name, "box_pendulum")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies]
        jids = [joint.jid for joint in builder.joints]
        gids = [geom.gid for geom in builder.collision_geoms]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.collision_geoms[i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.num_worlds, 1)
        self.assertEqual(model.size.sum_of_num_bodies, 1)
        self.assertEqual(model.size.sum_of_num_joints, 1)
        self.assertEqual(model.size.sum_of_num_collision_geoms, 2)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_14_make_builder_boxes_hinged(self):
        # Construct boxes-hinged model
        builder = build_boxes_hinged()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 2)
        self.assertEqual(builder.num_joints, 1)
        self.assertEqual(builder.num_collision_geoms, 3)
        self.assertEqual(builder.num_physical_geoms, 0)
        self.assertEqual(builder.worlds[0].name, "boxes_hinged")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies]
        jids = [joint.jid for joint in builder.joints]
        gids = [geom.gid for geom in builder.collision_geoms]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.collision_geoms[i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.num_worlds, 1)
        self.assertEqual(model.size.sum_of_num_bodies, 2)
        self.assertEqual(model.size.sum_of_num_joints, 1)
        self.assertEqual(model.size.sum_of_num_collision_geoms, 3)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_15_make_builder_boxes_nunchaku(self):
        # Construct boxes-nunchaku model
        builder = build_boxes_nunchaku()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 3)
        self.assertEqual(builder.num_joints, 2)
        self.assertEqual(builder.num_collision_geoms, 4)
        self.assertEqual(builder.num_physical_geoms, 0)
        self.assertEqual(builder.worlds[0].name, "boxes_nunchaku")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies]
        jids = [joint.jid for joint in builder.joints]
        gids = [geom.gid for geom in builder.collision_geoms]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.collision_geoms[i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.sum_of_num_bodies, 3)
        self.assertEqual(model.size.sum_of_num_joints, 2)
        self.assertEqual(model.size.sum_of_num_collision_geoms, 4)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_16_make_builder_boxes_fourbar(self):
        # Construct boxes-fourbar model
        builder = build_boxes_fourbar()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 4)
        self.assertEqual(builder.num_joints, 4)
        self.assertEqual(builder.num_collision_geoms, 5)
        self.assertEqual(builder.num_physical_geoms, 0)
        self.assertEqual(builder.worlds[0].name, "boxes_fourbar")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies]
        jids = [joint.jid for joint in builder.joints]
        gids = [geom.gid for geom in builder.collision_geoms]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.collision_geoms[i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.sum_of_num_bodies, 4)
        self.assertEqual(model.size.sum_of_num_joints, 4)
        self.assertEqual(model.size.sum_of_num_collision_geoms, 5)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_17_make_builder_cartpole(self):
        # Construct cartpole model
        builder = build_cartpole()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 2)
        self.assertEqual(builder.num_joints, 2)
        self.assertEqual(builder.num_collision_geoms, 4)
        self.assertEqual(builder.num_physical_geoms, 0)
        self.assertEqual(builder.worlds[0].name, "cartpole")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies]
        jids = [joint.jid for joint in builder.joints]
        gids = [geom.gid for geom in builder.collision_geoms]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.collision_geoms[i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.sum_of_num_bodies, 2)
        self.assertEqual(model.size.sum_of_num_joints, 2)
        self.assertEqual(model.size.sum_of_num_collision_geoms, 4)
        self.assertEqual(model.size.sum_of_num_physical_geoms, 0)
        self.assertEqual(model.device, self.default_device)

    def test_18_add_two_cartpole_worlds_to_builder(self):
        # Construct cartpole model
        builder = ModelBuilder(default_world=False)
        builder = build_cartpole(builder=builder, new_world=True)
        builder = build_cartpole(builder=builder, new_world=True)
        builder = build_cartpole(builder=builder, new_world=True)
        self.assertEqual(builder.num_worlds, 3)
        self.assertEqual(builder.num_bodies, 6)
        self.assertEqual(builder.num_joints, 6)
        self.assertEqual(builder.num_collision_geoms, 12)
        self.assertEqual(builder.num_physical_geoms, 0)

        # Build the model
        model = builder.finalize(self.default_device)

        # Verify that the contents of the model matches those of the combined builder
        self.assertIs(model.worlds, builder.worlds)
        assert_model_matches_builder(self, builder, model)

    def test_19_add_two_cartpole_builders(self):
        # Construct cartpole model
        builder0 = build_cartpole()
        builder1 = build_cartpole()
        builder2 = build_cartpole()

        # Combine two builders into one with two worlds
        builder0.add_builder(builder1)
        builder0.add_builder(builder2)
        self.assertEqual(builder0.num_worlds, 3)
        self.assertEqual(builder0.num_bodies, 6)
        self.assertEqual(builder0.num_joints, 6)
        self.assertEqual(builder0.num_collision_geoms, 12)
        self.assertEqual(builder0.num_physical_geoms, 0)

        # Build the model
        model = builder0.finalize(self.default_device)

        # Verify that the contents of the model matches those of the combined builder
        self.assertIs(model.worlds, builder0.worlds)
        assert_model_matches_builder(self, builder0, model)

    def test_20_make_homogeneous_multi_cartpole_builder(self):
        # Construct cartpole model
        builder = make_homogeneous_builder(num_worlds=3, build_fn=build_cartpole)
        self.assertEqual(builder.num_worlds, 3)
        self.assertEqual(builder.num_bodies, 6)
        self.assertEqual(builder.num_joints, 6)
        self.assertEqual(builder.num_collision_geoms, 12)
        self.assertEqual(builder.num_physical_geoms, 0)

        # Build the model
        model = builder.finalize(self.default_device)

        # Verify that the contents of the model matches those of the combined builder
        self.assertIs(model.worlds, builder.worlds)
        assert_model_matches_builder(self, builder, model)

    def test_21_make_heterogeneous_test_builder(self):
        # Construct cartpole model
        builder = make_heterogeneous_builder()
        self.assertEqual(builder.num_worlds, 6)
        self.assertEqual(builder.num_bodies, 13)
        self.assertEqual(builder.num_joints, 10)
        self.assertEqual(builder.num_collision_geoms, 20)
        self.assertEqual(builder.num_physical_geoms, 0)

        # Build the model
        model = builder.finalize(self.default_device)

        # Verify that the contents of the model matches those of the combined builder
        self.assertIs(model.worlds, builder.worlds)
        assert_model_matches_builder(self, builder, model)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
