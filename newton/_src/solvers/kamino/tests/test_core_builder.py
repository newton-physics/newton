# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: CORE: BUILDER
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.bodies import RigidBodyDescriptor
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.geometry import GeometryDescriptor
from newton._src.solvers.kamino._src.core.gravity import (
    GRAVITY_ACCEL_DEFAULT,
    GRAVITY_DIREC_DEFAULT,
    GRAVITY_NAME_DEFAULT,
)
from newton._src.solvers.kamino._src.core.joints import JointActuationType, JointDescriptor, JointDoFType
from newton._src.solvers.kamino._src.core.materials import MaterialDescriptor
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.shapes import SphereShape
from newton._src.solvers.kamino._src.core.types import Axis, mat33f, transformf, vec6f
from newton._src.solvers.kamino._src.models.builders import basics
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Utilities
###


def assert_model_matches_builder(test: unittest.TestCase, builder: ModelBuilderKamino, model: ModelKamino):
    """
    Assert that a constructed model matches the specifications of the given builder.

    Args:
        test: The test case instance.
        builder: The model builder instance.
        model: The constructed model instance.
    """
    # Check model sizes and offsets
    for w, world in enumerate(builder.worlds):
        test.assertEqual(world.wid, w)
        test.assertEqual(model.info.num_bodies.numpy()[w], world.num_bodies)
        test.assertEqual(model.info.num_joints.numpy()[w], world.num_joints)
        test.assertEqual(model.info.num_passive_joints.numpy()[w], world.num_passive_joints)
        test.assertEqual(model.info.num_actuated_joints.numpy()[w], world.num_actuated_joints)
        test.assertEqual(model.info.num_dynamic_joints.numpy()[w], world.num_dynamic_joints)
        test.assertEqual(model.info.num_geoms.numpy()[w], world.num_geoms)
        test.assertEqual(model.info.num_body_dofs.numpy()[w], world.num_body_dofs)
        test.assertEqual(model.info.num_joint_coords.numpy()[w], world.num_joint_coords)
        test.assertEqual(model.info.num_joint_dofs.numpy()[w], world.num_joint_dofs)
        test.assertEqual(model.info.num_passive_joint_coords.numpy()[w], world.num_passive_joint_coords)
        test.assertEqual(model.info.num_passive_joint_dofs.numpy()[w], world.num_passive_joint_dofs)
        test.assertEqual(model.info.num_actuated_joint_coords.numpy()[w], world.num_actuated_joint_coords)
        test.assertEqual(model.info.num_actuated_joint_dofs.numpy()[w], world.num_actuated_joint_dofs)
        test.assertEqual(model.info.num_joint_cts.numpy()[w], world.num_joint_cts)
        test.assertEqual(model.info.num_joint_dynamic_cts.numpy()[w], world.num_dynamic_joint_cts)
        test.assertEqual(model.info.num_joint_kinematic_cts.numpy()[w], world.num_kinematic_joint_cts)
        test.assertEqual(model.info.bodies_offset.numpy()[w], world.bodies_idx_offset)
        test.assertEqual(model.info.joints_offset.numpy()[w], world.joints_idx_offset)
        test.assertEqual(model.info.geoms_offset.numpy()[w], world.geoms_idx_offset)
        test.assertEqual(model.info.body_dofs_offset.numpy()[w], world.body_dofs_idx_offset)
        test.assertEqual(model.info.joint_coords_offset.numpy()[w], world.joint_coords_idx_offset)
        test.assertEqual(model.info.joint_dofs_offset.numpy()[w], world.joint_dofs_idx_offset)
        test.assertEqual(model.info.joint_passive_coords_offset.numpy()[w], world.joint_passive_coords_idx_offset)
        test.assertEqual(model.info.joint_passive_dofs_offset.numpy()[w], world.joint_passive_dofs_idx_offset)
        test.assertEqual(model.info.joint_actuated_coords_offset.numpy()[w], world.joint_actuated_coords_idx_offset)
        test.assertEqual(model.info.joint_actuated_dofs_offset.numpy()[w], world.joint_actuated_dofs_idx_offset)
        # TODO: test.assertEqual(model.info.joint_cts_offset.numpy()[w], world.joint_cts_idx_offset)
        test.assertEqual(model.info.joint_dynamic_cts_offset.numpy()[w], world.joint_dynamic_cts_idx_offset)
        test.assertEqual(model.info.joint_kinematic_cts_offset.numpy()[w], world.joint_kinematic_cts_idx_offset)

    test.assertEqual(builder.num_bodies, model.size.sum_of_num_bodies)
    for i, body in enumerate(builder.all_bodies):
        test.assertEqual(model.bodies.wid.numpy()[i], body.wid)
        test.assertEqual(model.bodies.bid.numpy()[i], body.bid)

    test.assertEqual(builder.num_joints, model.size.sum_of_num_joints)
    for i, joint in enumerate(builder.all_joints):
        wid = joint.wid
        bid_offset = builder.worlds[wid].bodies_idx_offset
        test.assertEqual(model.joints.wid.numpy()[i], joint.wid)
        test.assertEqual(model.joints.jid.numpy()[i], joint.jid)
        test.assertEqual(model.joints.bid_B.numpy()[i], joint.bid_B + bid_offset if joint.bid_B >= 0 else -1)
        test.assertEqual(model.joints.bid_F.numpy()[i], joint.bid_F + bid_offset if joint.bid_F >= 0 else -1)

    test.assertEqual(builder.num_geoms, model.size.sum_of_num_geoms)
    for i, geom in enumerate(builder.all_geoms):
        wid = geom.wid
        bid_offset = builder.worlds[wid].bodies_idx_offset
        test.assertEqual(model.geoms.wid.numpy()[i], geom.wid)
        test.assertEqual(model.geoms.gid.numpy()[i], geom.gid)
        test.assertEqual(
            model.geoms.bid.numpy()[i],
            geom.body + bid_offset if geom.body >= 0 else -1,
        )

    # Optional printout for debugging
    msg.info("model.bodies.wid: %s", model.bodies.wid)
    msg.info("model.bodies.bid: %s\n", model.bodies.bid)
    msg.info("model.joints.wid: %s", model.joints.wid)
    msg.info("model.joints.jid: %s", model.joints.jid)
    msg.info("model.joints.bid_B: %s", model.joints.bid_B)
    msg.info("model.joints.bid_F: %s\n", model.joints.bid_F)
    msg.info("model.geoms.wid: %s", model.geoms.wid)
    msg.info("model.geoms.gid: %s", model.geoms.gid)
    msg.info("model.geoms.bid: %s", model.geoms.bid)
    msg.info("model.info.bodies_offset: %s", model.info.bodies_offset)
    msg.info("model.info.joints_offset: %s", model.info.joints_offset)
    msg.info("model.info.body_dofs_offset: %s", model.info.body_dofs_offset)
    msg.info("model.info.joint_coords_offset: %s", model.info.joint_coords_offset)
    msg.info("model.info.joint_dofs_offset: %s", model.info.joint_dofs_offset)
    msg.info("model.info.joint_cts_offset: %s\n", model.info.joint_cts_offset)
    msg.info("model.info.joint_dynamic_cts_offset: %s\n", model.info.joint_dynamic_cts_offset)
    msg.info("model.info.joint_kinematic_cts_offset: %s\n", model.info.joint_kinematic_cts_offset)
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
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default(self):
        builder = ModelBuilderKamino()
        self.assertEqual(builder.num_worlds, 0)
        self.assertEqual(builder.num_bodies, 0)
        self.assertEqual(builder.num_joints, 0)
        self.assertEqual(builder.num_geoms, 0)
        self.assertEqual(builder.num_materials, 1)  # Default material is always created
        self.assertEqual(builder.num_body_dofs, 0)
        self.assertEqual(builder.num_joint_dofs, 0)
        self.assertEqual(builder.num_passive_joint_dofs, 0)
        self.assertEqual(builder.num_actuated_joint_dofs, 0)
        self.assertEqual(builder.num_joint_cts, 0)
        self.assertEqual(builder.num_dynamic_joint_cts, 0)
        self.assertEqual(builder.num_kinematic_joint_cts, 0)
        self.assertEqual(len(builder.bodies), 0)
        self.assertEqual(len(builder.joints), 0)
        self.assertEqual(len(builder.geoms), 0)
        self.assertEqual(len(builder.materials), 1)  # Default material is always created

    def test_01_make_default_with_world(self):
        builder = ModelBuilderKamino(default_world=True)
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 0)
        self.assertEqual(builder.num_joints, 0)
        self.assertEqual(builder.num_geoms, 0)
        self.assertEqual(builder.num_materials, 1)  # Default material is always created
        self.assertEqual(builder.num_body_dofs, 0)
        self.assertEqual(builder.num_joint_dofs, 0)
        self.assertEqual(builder.num_passive_joint_dofs, 0)
        self.assertEqual(builder.num_actuated_joint_dofs, 0)
        self.assertEqual(builder.num_joint_cts, 0)
        self.assertEqual(len(builder.bodies), 1)
        self.assertEqual(len(builder.bodies[0]), 0)
        self.assertEqual(len(builder.joints), 1)
        self.assertEqual(len(builder.joints[0]), 0)
        self.assertEqual(len(builder.geoms), 1)
        self.assertEqual(len(builder.geoms[0]), 0)
        self.assertEqual(len(builder.materials), 1)  # Default material is always created

    def test_02_add_world(self):
        builder = ModelBuilderKamino()
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
        builder = ModelBuilderKamino()
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
        self.assertEqual(bid, builder.bodies[wid][bid].bid)
        self.assertEqual(builder.bodies[wid][bid].name, "test_rigid_body")
        self.assertEqual(builder.bodies[wid][bid].wid, wid)
        self.assertEqual(builder.bodies[wid][bid].m_i, 1.0)
        np.testing.assert_array_equal(builder.bodies[wid][bid].i_I_i, np.eye(3, dtype=np.float32).flatten())
        np.testing.assert_array_equal(builder.bodies[wid][bid].q_i_0, np.array(transformf(), dtype=np.float32))
        np.testing.assert_array_equal(builder.bodies[wid][bid].u_i_0, np.zeros(6, dtype=np.float32))

    def test_04_add_rigid_body_descriptor(self):
        builder = ModelBuilderKamino()
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
        self.assertEqual(bid, builder.bodies[wid][bid].bid)
        self.assertEqual(builder.bodies[wid][bid].name, "test_rigid_body")
        self.assertEqual(builder.bodies[wid][bid].wid, wid)
        self.assertEqual(builder.bodies[wid][bid].m_i, 2.0)
        np.testing.assert_array_equal(builder.bodies[wid][bid].i_I_i, 2.0 * np.eye(3, dtype=np.float32).flatten())
        np.testing.assert_array_equal(builder.bodies[wid][bid].q_i_0, np.array(transformf(), dtype=np.float32))
        np.testing.assert_array_equal(builder.bodies[wid][bid].u_i_0, np.zeros(6, dtype=np.float32))

    def test_05_add_duplicate_rigid_body(self):
        builder = ModelBuilderKamino()
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
        builder = ModelBuilderKamino()
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
            a_j=1.0,
            b_j=1.0,
        )
        jid = builder.add_joint_descriptor(joint, world_index=wid)

        self.assertEqual(builder.num_bodies, 2)
        self.assertEqual(builder.num_joints, 1)
        self.assertEqual(jid, 0)
        self.assertEqual(jid, builder.joints[wid][jid].jid)
        self.assertEqual(builder.joints[wid][jid].name, "test_joint")
        self.assertEqual(builder.joints[wid][jid].wid, wid)
        self.assertEqual(builder.joints[wid][jid].bid_B, bid_0)
        self.assertEqual(builder.joints[wid][jid].bid_F, bid_1)
        self.assertEqual(builder.joints[wid][jid].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder.joints[wid][jid].act_type, JointActuationType.FORCE)
        self.assertEqual(builder.joints[wid][jid].a_j, [1.0])
        self.assertEqual(builder.joints[wid][jid].b_j, [1.0])
        self.assertTrue(builder.joints[wid][jid].is_dynamic)
        self.assertTrue(builder.joints[wid][jid].num_kinematic_cts, 5)
        self.assertTrue(builder.joints[wid][jid].num_dynamic_cts, 1)

    def test_07_add_duplicate_joint(self):
        builder = ModelBuilderKamino()
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
        builder = ModelBuilderKamino()
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
        builder = ModelBuilderKamino()
        wid = builder.add_world(name="test_world", up_axis=Axis.Z)
        self.assertTrue(builder.num_geoms == 0)

        # Create a collision geometry descriptor from the geometry descriptor
        gid = builder.add_geometry(
            name="test_geom",
            shape=SphereShape(radius=1.0),
            world_index=wid,
        )
        self.assertEqual(builder.num_geoms, 1)
        self.assertEqual(gid, 0)
        self.assertEqual(gid, builder.geoms[wid][gid].gid)
        self.assertEqual(builder.geoms[wid][gid].name, "test_geom")
        self.assertEqual(builder.geoms[wid][gid].body, -1)
        self.assertEqual(builder.geoms[wid][gid].wid, wid)
        self.assertEqual(builder.geoms[wid][gid].mid, 0)

    def test_10_add_geometry_descriptors(self):
        builder = ModelBuilderKamino()
        wid = builder.add_world(name="test_world", up_axis=Axis.Z)
        self.assertTrue(builder.num_geoms == 0)

        # Define a geometry descriptor
        geom = GeometryDescriptor(name="test_geom", shape=SphereShape(radius=1.0))
        gid = builder.add_geometry_descriptor(geom, world_index=wid)
        self.assertEqual(builder.num_geoms, 1)
        self.assertEqual(gid, 0)
        self.assertEqual(gid, builder.geoms[wid][gid].gid)
        self.assertEqual(builder.geoms[wid][gid].name, "test_geom")
        self.assertEqual(builder.geoms[wid][gid].body, -1)
        self.assertEqual(builder.geoms[wid][gid].wid, wid)
        self.assertEqual(builder.geoms[wid][gid].mid, -1)

    def test_11_add_material(self):
        builder = ModelBuilderKamino()
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
        builder = basics.build_box_on_plane()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 1)
        self.assertEqual(builder.num_joints, 0)
        self.assertEqual(builder.num_geoms, 2)
        self.assertEqual(builder.worlds[0].name, "box_on_plane")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies[0]]
        jids = [joint.jid for joint in builder.joints[0]]
        gids = [geom.gid for geom in builder.geoms[0]]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[0][i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[0][i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.geoms[0][i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.num_worlds, 1)
        self.assertEqual(model.size.sum_of_num_bodies, 1)
        self.assertEqual(model.size.sum_of_num_joints, 0)
        self.assertEqual(model.size.sum_of_num_geoms, 2)
        self.assertEqual(model.device, self.default_device)

    def test_13_make_builder_box_pendulum(self):
        # Construct box-pendulum model
        builder = basics.build_box_pendulum()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 1)
        self.assertEqual(builder.num_joints, 1)
        self.assertEqual(builder.num_geoms, 2)
        self.assertEqual(builder.worlds[0].name, "box_pendulum")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies[0]]
        jids = [joint.jid for joint in builder.joints[0]]
        gids = [geom.gid for geom in builder.geoms[0]]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[0][i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[0][i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.geoms[0][i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.num_worlds, 1)
        self.assertEqual(model.size.sum_of_num_bodies, 1)
        self.assertEqual(model.size.sum_of_num_joints, 1)
        self.assertEqual(model.size.sum_of_num_geoms, 2)
        self.assertEqual(model.device, self.default_device)

    def test_14_make_builder_boxes_hinged(self):
        # Construct boxes-hinged model
        builder = basics.build_boxes_hinged()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 2)
        self.assertEqual(builder.num_joints, 1)
        self.assertEqual(builder.num_geoms, 3)
        self.assertEqual(builder.worlds[0].name, "boxes_hinged")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies[0]]
        jids = [joint.jid for joint in builder.joints[0]]
        gids = [geom.gid for geom in builder.geoms[0]]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[0][i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[0][i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.geoms[0][i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.num_worlds, 1)
        self.assertEqual(model.size.sum_of_num_bodies, 2)
        self.assertEqual(model.size.sum_of_num_joints, 1)
        self.assertEqual(model.size.sum_of_num_geoms, 3)
        self.assertEqual(model.device, self.default_device)

    def test_15_make_builder_boxes_nunchaku(self):
        # Construct boxes-nunchaku model
        builder = basics.build_boxes_nunchaku()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 3)
        self.assertEqual(builder.num_joints, 2)
        self.assertEqual(builder.num_geoms, 4)
        self.assertEqual(builder.worlds[0].name, "boxes_nunchaku")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies[0]]
        jids = [joint.jid for joint in builder.joints[0]]
        gids = [geom.gid for geom in builder.geoms[0]]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[0][i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[0][i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.geoms[0][i].gid)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.sum_of_num_bodies, 3)
        self.assertEqual(model.size.sum_of_num_joints, 2)
        self.assertEqual(model.size.sum_of_num_geoms, 4)
        self.assertEqual(model.device, self.default_device)

    def test_16_make_builder_boxes_fourbar(self):
        # Construct boxes-fourbar model
        builder = basics.build_boxes_fourbar()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 4)
        self.assertEqual(builder.num_joints, 4)
        self.assertEqual(builder.num_geoms, 5)
        self.assertEqual(builder.worlds[0].name, "boxes_fourbar")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies[0]]
        jids = [joint.jid for joint in builder.joints[0]]
        gids = [geom.gid for geom in builder.geoms[0]]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[0][i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[0][i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.geoms[0][i].gid)

        # Generate meta-data for collision detection and contacts allocation
        model_candidate_pairs, pair_offsets = builder.make_collision_candidate_pairs(allow_neighbors=False)
        model_excluded_pairs, _ = builder.make_collision_excluded_pairs(allow_neighbors=False)
        world_num_collidables, model_num_collidables = builder.compute_num_collidable_geoms(
            model_candidate_pairs, pair_offsets
        )
        model_min_contacts, world_min_contacts = builder.compute_required_contact_capacity(
            model_candidate_pairs, pair_offsets
        )

        # Optional printouts for debugging
        msg.info("model_candidate_pairs: %s", model_candidate_pairs)
        msg.info("model_candidate_pairs_count: %s", len(model_candidate_pairs))
        msg.info("pair_offsets: %s", pair_offsets)
        msg.info("model_excluded_pairs: %s", model_excluded_pairs)
        msg.info("model_excluded_pairs_count: %s", len(model_excluded_pairs))
        msg.info("world_num_collidables: %s", world_num_collidables)
        msg.info("model_num_collidables: %s", model_num_collidables)
        msg.info("model_min_contacts: %s", model_min_contacts)
        msg.info("world_min_contacts: %s", world_min_contacts)

        # Check that the generated meta-data matches expected values for this model
        expected_contacts_per_world = 2 * len(model_candidate_pairs) * 12  # 12 is the max contacts per pair
        self.assertEqual(world_num_collidables[0], 5)
        self.assertEqual(model_num_collidables, 5)
        self.assertEqual(len(model_candidate_pairs), 6)
        self.assertEqual(len(model_excluded_pairs), 4)
        self.assertEqual(model_min_contacts, expected_contacts_per_world)
        self.assertEqual(world_min_contacts[0], expected_contacts_per_world)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.sum_of_num_bodies, 4)
        self.assertEqual(model.size.sum_of_num_joints, 4)
        self.assertEqual(model.size.sum_of_num_geoms, 5)
        self.assertEqual(model.device, self.default_device)

    def test_17_make_builder_cartpole(self):
        # Construct cartpole model
        builder = basics.build_cartpole()
        self.assertEqual(builder.num_worlds, 1)
        self.assertEqual(builder.num_bodies, 2)
        self.assertEqual(builder.num_joints, 2)
        self.assertEqual(builder.num_geoms, 4)
        self.assertEqual(builder.worlds[0].name, "cartpole")
        self.assertEqual(builder.worlds[0].wid, 0)

        # Extract the IDs of bodies, joints, and collision geometries
        bids = [body.bid for body in builder.bodies[0]]
        jids = [joint.jid for joint in builder.joints[0]]
        gids = [geom.gid for geom in builder.geoms[0]]

        # Check the number of bodies, joints, and collision geometries
        for i, bid in enumerate(bids):
            self.assertEqual(bid, i)
            self.assertEqual(bid, builder.bodies[0][i].bid)
        for i, jid in enumerate(jids):
            self.assertEqual(jid, i)
            self.assertEqual(jid, builder.joints[0][i].jid)
        for i, gid in enumerate(gids):
            self.assertEqual(gid, i)
            self.assertEqual(gid, builder.geoms[0][i].gid)

        # Generate meta-data for collision detection and contacts allocation
        model_candidate_pairs, pair_offsets = builder.make_collision_candidate_pairs(allow_neighbors=False)
        model_excluded_pairs, _ = builder.make_collision_excluded_pairs(allow_neighbors=False)
        world_num_collidables, model_num_collidables = builder.compute_num_collidable_geoms(
            model_candidate_pairs, pair_offsets
        )
        model_min_contacts, world_min_contacts = builder.compute_required_contact_capacity(
            model_candidate_pairs, pair_offsets
        )

        # Optional printouts for debugging
        msg.info("model_candidate_pairs: %s", model_candidate_pairs)
        msg.info("model_candidate_pairs_count: %s", len(model_candidate_pairs))
        msg.info("pair_offsets: %s", pair_offsets)
        msg.info("model_excluded_pairs: %s", model_excluded_pairs)
        msg.info("model_excluded_pairs_count: %s", len(model_excluded_pairs))
        msg.info("world_num_collidables: %s", world_num_collidables)
        msg.info("model_num_collidables: %s", model_num_collidables)
        msg.info("model_min_contacts: %s", model_min_contacts)
        msg.info("world_min_contacts: %s", world_min_contacts)

        # Check that the generated meta-data matches expected values for this model
        expected_contacts_per_world = 2 * len(model_candidate_pairs) * 12  # 12 is the max contacts per pair
        self.assertEqual(world_num_collidables[0], 3)
        self.assertEqual(model_num_collidables, 3)
        self.assertEqual(len(model_candidate_pairs), 2)
        self.assertEqual(len(model_excluded_pairs), 4)
        self.assertEqual(model_min_contacts, expected_contacts_per_world)
        self.assertEqual(world_min_contacts[0], expected_contacts_per_world)

        # Build the model
        model = builder.finalize(self.default_device)
        self.assertEqual(model.size.sum_of_num_bodies, 2)
        self.assertEqual(model.size.sum_of_num_joints, 2)
        self.assertEqual(model.size.sum_of_num_geoms, 4)
        self.assertEqual(model.device, self.default_device)

    def test_18_add_two_cartpole_worlds_to_builder(self):
        # Construct cartpole model
        builder = ModelBuilderKamino(default_world=False)
        builder = basics.build_cartpole(builder=builder, new_world=True)
        builder = basics.build_cartpole(builder=builder, new_world=True)
        builder = basics.build_cartpole(builder=builder, new_world=True)
        self.assertEqual(builder.num_worlds, 3)
        self.assertEqual(builder.num_bodies, 6)
        self.assertEqual(builder.num_joints, 6)
        self.assertEqual(builder.num_geoms, 12)

        # Build the model
        model = builder.finalize(self.default_device)

        # Verify that the contents of the model matches those of the combined builder
        assert_model_matches_builder(self, builder, model)

    def test_19_add_two_cartpole_builders(self):
        # Construct cartpole model
        builder0 = basics.build_cartpole()
        builder1 = basics.build_cartpole()
        builder2 = basics.build_cartpole()

        # Combine two builders into one with two worlds
        builder0.add_builder(builder1)
        builder0.add_builder(builder2)
        self.assertEqual(builder0.num_worlds, 3)
        self.assertEqual(builder0.num_bodies, 6)
        self.assertEqual(builder0.num_joints, 6)
        self.assertEqual(builder0.num_geoms, 12)

        # Build the model
        model = builder0.finalize(self.default_device)

        # Verify that the contents of the model matches those of the combined builder
        assert_model_matches_builder(self, builder0, model)

    def test_20_make_homogeneous_multi_cartpole_builder(self):
        # Construct cartpole model
        builder = make_homogeneous_builder(num_worlds=3, build_fn=basics.build_cartpole)
        self.assertEqual(builder.num_worlds, 3)
        self.assertEqual(builder.num_bodies, 6)
        self.assertEqual(builder.num_joints, 6)
        self.assertEqual(builder.num_geoms, 12)

        # Build the model
        model = builder.finalize(self.default_device)

        # Verify that the contents of the model matches those of the combined builder
        assert_model_matches_builder(self, builder, model)

    def test_21_make_homogeneous_multi_fourbar_builder(self):
        # Construct fourbar model
        builder = make_homogeneous_builder(num_worlds=3, build_fn=basics.build_boxes_fourbar)
        self.assertEqual(builder.num_worlds, 3)
        self.assertEqual(builder.num_bodies, 12)
        self.assertEqual(builder.num_joints, 12)
        self.assertEqual(builder.num_geoms, 15)

        # Generate meta-data for collision detection and contacts allocation
        model_candidate_pairs, pair_offsets = builder.make_collision_candidate_pairs(allow_neighbors=False)
        model_excluded_pairs, _ = builder.make_collision_excluded_pairs(allow_neighbors=False)
        world_num_collidables, model_num_collidables = builder.compute_num_collidable_geoms(
            model_candidate_pairs, pair_offsets
        )
        model_min_contacts, world_min_contacts = builder.compute_required_contact_capacity(
            model_candidate_pairs, pair_offsets
        )

        # Optional printouts for debugging
        msg.info("model_candidate_pairs: %s", model_candidate_pairs)
        msg.info("model_candidate_pairs_count: %s", len(model_candidate_pairs))
        msg.info("pair_offsets: %s", pair_offsets)
        msg.info("model_excluded_pairs: %s", model_excluded_pairs)
        msg.info("model_excluded_pairs_count: %s", len(model_excluded_pairs))
        msg.info("world_num_collidables: %s", world_num_collidables)
        msg.info("model_num_collidables: %s", model_num_collidables)
        msg.info("model_min_contacts: %s", model_min_contacts)
        msg.info("world_min_contacts: %s", world_min_contacts)

        # Check that the generated meta-data matches expected values for this model
        expected_contacts_per_world = 2 * 6 * 12  # 12 is the max contacts per pair
        self.assertEqual(model_num_collidables, 5 * builder.num_worlds)
        self.assertEqual(world_num_collidables, [5] * builder.num_worlds)
        self.assertEqual(len(model_candidate_pairs), 6 * builder.num_worlds)
        self.assertEqual(len(model_excluded_pairs), 4 * builder.num_worlds)
        self.assertEqual(model_min_contacts, expected_contacts_per_world * builder.num_worlds)
        self.assertEqual(world_min_contacts, [expected_contacts_per_world] * builder.num_worlds)

        # Build the model
        model = builder.finalize(self.default_device)

        # Verify that the contents of the model matches those of the combined builder
        assert_model_matches_builder(self, builder, model)

    def test_22_make_heterogeneous_test_builder(self):
        # Construct cartpole model
        builder = basics.make_basics_heterogeneous_builder(ground=True)
        self.assertEqual(builder.num_worlds, 6)
        self.assertEqual(builder.num_bodies, 13)
        self.assertEqual(builder.num_joints, 10)
        self.assertEqual(builder.num_geoms, 20)

        # Build the model
        model = builder.finalize(self.default_device)

        # Verify that the contents of the model matches those of the combined builder
        assert_model_matches_builder(self, builder, model)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
