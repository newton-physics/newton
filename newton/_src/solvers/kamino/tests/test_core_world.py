# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the WorldDescriptor container in Kamino"""

import math
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.bodies import RigidBodyDescriptor
from newton._src.solvers.kamino._src.core.geometry import GeometryDescriptor
from newton._src.solvers.kamino._src.core.gravity import (
    GRAVITY_ACCEL_DEFAULT,
    GRAVITY_DIREC_DEFAULT,
    GRAVITY_NAME_DEFAULT,
    GravityDescriptor,
)
from newton._src.solvers.kamino._src.core.joints import (
    JOINT_DQMAX,
    JOINT_QMAX,
    JOINT_QMIN,
    JOINT_TAUMAX,
    JointActuationType,
    JointDescriptor,
    JointDoFType,
)
from newton._src.solvers.kamino._src.core.materials import (
    DEFAULT_DENSITY,
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    MaterialDescriptor,
)
from newton._src.solvers.kamino._src.core.shapes import ShapeType, SphereShape
from newton._src.solvers.kamino._src.core.world import WorldDescriptor
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestGravityDescriptor(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose

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

    def test_00_default_construction(self):
        gravity = GravityDescriptor()
        msg.info(f"gravity: {gravity}")
        self.assertIsInstance(gravity, GravityDescriptor)
        self.assertEqual(gravity.name, GRAVITY_NAME_DEFAULT)
        self.assertEqual(gravity.enabled, True)
        self.assertEqual(gravity.acceleration, GRAVITY_ACCEL_DEFAULT)
        expected_direction = np.array(GRAVITY_DIREC_DEFAULT, dtype=np.float32)
        expected_dir_accel = np.array([*GRAVITY_DIREC_DEFAULT, GRAVITY_ACCEL_DEFAULT], dtype=np.float32)
        expected_vector = np.array([0.0, 0.0, -GRAVITY_ACCEL_DEFAULT, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(gravity.direction, expected_direction)
        np.testing.assert_array_equal(gravity.dir_accel(), expected_dir_accel)
        np.testing.assert_array_equal(gravity.vector(), expected_vector)

    def test_01_with_parameters_and_dir_as_list(self):
        gravity = GravityDescriptor(name="test_gravity", enabled=False, acceleration=15.0, direction=[1.0, 0.0, 0.0])
        msg.info(f"gravity: {gravity}")
        self.assertIsInstance(gravity, GravityDescriptor)
        self.assertEqual(gravity.name, "test_gravity")
        self.assertEqual(gravity.enabled, False)
        self.assertEqual(gravity.acceleration, 15.0)
        np.testing.assert_array_equal(gravity.direction, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(gravity.dir_accel(), np.array([1.0, 0.0, 0.0, 15.0], dtype=np.float32))
        np.testing.assert_array_equal(gravity.vector(), np.array([15.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def test_02_with_parameters_and_dir_as_tuple(self):
        gravity = GravityDescriptor(name="test_gravity", enabled=False, acceleration=9.0, direction=(1.0, 0.0, 0.0))
        msg.info(f"gravity: {gravity}")
        self.assertIsInstance(gravity, GravityDescriptor)
        self.assertEqual(gravity.name, "test_gravity")
        self.assertEqual(gravity.enabled, False)
        self.assertEqual(gravity.acceleration, 9.0)
        np.testing.assert_array_equal(gravity.direction, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(gravity.dir_accel(), np.array([1.0, 0.0, 0.0, 9.0], dtype=np.float32))
        np.testing.assert_array_equal(gravity.vector(), np.array([9.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def test_03_with_parameters_and_dir_as_nparray(self):
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        gravity = GravityDescriptor(name="test_gravity", enabled=False, acceleration=12.0, direction=direction)
        msg.info(f"gravity: {gravity}")
        self.assertIsInstance(gravity, GravityDescriptor)
        self.assertEqual(gravity.name, "test_gravity")
        self.assertEqual(gravity.enabled, False)
        self.assertEqual(gravity.acceleration, 12.0)
        np.testing.assert_array_equal(gravity.direction, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(gravity.dir_accel(), np.array([1.0, 0.0, 0.0, 12.0], dtype=np.float32))
        np.testing.assert_array_equal(gravity.vector(), np.array([12.0, 0.0, 0.0, 0.0], dtype=np.float32))


class TestBodyDescriptor(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose

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

    def test_00_default_construction(self):
        body = RigidBodyDescriptor(name="test_body")
        self.assertIsInstance(body, RigidBodyDescriptor)
        msg.info(f"body: {body}")
        self.assertEqual(body.name, "test_body")
        self.assertEqual(body.m_i, 0.0)
        np.testing.assert_array_equal(body.i_I_i, np.zeros(9, dtype=np.float32))
        np.testing.assert_array_equal(body.q_i_0, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(body.u_i_0, np.zeros(6, dtype=np.float32))
        self.assertEqual(body.wid, -1)
        self.assertEqual(body.bid, -1)


class TestJointDescriptor(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose

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

    def test_00_default_construction(self):
        joint = JointDescriptor(name="test_joint")
        self.assertIsInstance(joint, JointDescriptor)
        msg.info(f"joint: {joint}")
        self.assertEqual(joint.name, "test_joint")
        self.assertIsInstance(joint.dof_type, JointDoFType)
        self.assertIsInstance(joint.act_type, JointActuationType)
        self.assertEqual(joint.dof_type, JointDoFType.FREE)
        self.assertEqual(joint.act_type, JointActuationType.PASSIVE)
        self.assertEqual(joint.bid_B, -1)
        self.assertEqual(joint.bid_F, -1)
        self.assertEqual(joint.num_coords, 7)
        self.assertEqual(joint.num_dofs, 6)
        self.assertEqual(joint.num_dynamic_cts, 0)
        self.assertEqual(joint.num_kinematic_cts, 0)
        np.testing.assert_array_equal(joint.B_r_Bj, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(joint.F_r_Fj, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(joint.X_j, np.zeros(9, dtype=np.float32))
        np.testing.assert_array_equal(joint.q_j_min, np.full(6, JOINT_QMIN, dtype=np.float32))
        np.testing.assert_array_equal(joint.q_j_max, np.full(6, JOINT_QMAX, dtype=np.float32))
        np.testing.assert_array_equal(joint.dq_j_max, np.full(6, JOINT_DQMAX, dtype=np.float32))
        np.testing.assert_array_equal(joint.tau_j_max, np.full(6, JOINT_TAUMAX, dtype=np.float32))
        np.testing.assert_array_equal(joint.a_j, np.zeros(6, dtype=np.float32))
        np.testing.assert_array_equal(joint.b_j, np.zeros(6, dtype=np.float32))
        np.testing.assert_array_equal(joint.k_p_j, np.zeros(6, dtype=np.float32))
        np.testing.assert_array_equal(joint.k_d_j, np.zeros(6, dtype=np.float32))
        self.assertEqual(joint.wid, -1)
        self.assertEqual(joint.jid, -1)
        self.assertEqual(joint.coords_offset, -1)
        self.assertEqual(joint.dofs_offset, -1)
        self.assertEqual(joint.passive_coords_offset, -1)
        self.assertEqual(joint.passive_dofs_offset, -1)
        self.assertEqual(joint.actuated_coords_offset, -1)
        self.assertEqual(joint.actuated_dofs_offset, -1)
        self.assertEqual(joint.kinematic_cts_offset, -1)
        self.assertEqual(joint.dynamic_cts_offset, -1)
        # Check property methods
        self.assertEqual(joint.is_actuated, False)
        self.assertEqual(joint.is_passive, True)
        self.assertEqual(joint.is_binary, False)
        self.assertEqual(joint.is_unary, True)
        self.assertEqual(joint.is_dynamic, False)

    def test_01_actuated_revolute_joint_with_effort_dynamics(self):
        joint = JointDescriptor(
            name="test_joint_revolute_dynamic",
            dof_type=JointDoFType.REVOLUTE,
            act_type=JointActuationType.FORCE,
            bid_B=0,
            bid_F=1,
            a_j=1.0,
            b_j=1.0,
        )
        msg.info(f"joint: {joint}")

        # Check values
        self.assertIsInstance(joint, JointDescriptor)
        self.assertEqual(joint.name, "test_joint_revolute_dynamic")
        self.assertIsInstance(joint.dof_type, JointDoFType)
        self.assertIsInstance(joint.act_type, JointActuationType)
        self.assertEqual(joint.dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(joint.act_type, JointActuationType.FORCE)
        self.assertEqual(joint.bid_B, 0)
        self.assertEqual(joint.bid_F, 1)
        self.assertEqual(joint.num_coords, 1)
        self.assertEqual(joint.num_dofs, 1)
        self.assertEqual(joint.num_dynamic_cts, 1)
        self.assertEqual(joint.num_kinematic_cts, 5)
        np.testing.assert_array_equal(joint.B_r_Bj, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(joint.F_r_Fj, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(joint.X_j, np.zeros(9, dtype=np.float32))
        np.testing.assert_array_equal(joint.q_j_min, float(JOINT_QMIN))
        np.testing.assert_array_equal(joint.q_j_max, float(JOINT_QMAX))
        np.testing.assert_array_equal(joint.dq_j_max, float(JOINT_DQMAX))
        np.testing.assert_array_equal(joint.tau_j_max, float(JOINT_TAUMAX))
        np.testing.assert_array_equal(joint.a_j, 1.0)
        np.testing.assert_array_equal(joint.b_j, 1.0)
        np.testing.assert_array_equal(joint.k_p_j, 0.0)
        np.testing.assert_array_equal(joint.k_d_j, 0.0)
        self.assertEqual(joint.wid, -1)
        self.assertEqual(joint.jid, -1)
        self.assertEqual(joint.coords_offset, -1)
        self.assertEqual(joint.dofs_offset, -1)
        self.assertEqual(joint.passive_coords_offset, -1)
        self.assertEqual(joint.passive_dofs_offset, -1)
        self.assertEqual(joint.actuated_coords_offset, -1)
        self.assertEqual(joint.actuated_dofs_offset, -1)
        self.assertEqual(joint.kinematic_cts_offset, -1)
        self.assertEqual(joint.dynamic_cts_offset, -1)
        # Check property methods
        self.assertEqual(joint.is_actuated, True)
        self.assertEqual(joint.is_passive, False)
        self.assertEqual(joint.is_binary, True)
        self.assertEqual(joint.is_unary, False)
        self.assertEqual(joint.is_dynamic, True)


class TestGeometryDescriptor(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose

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

    def test_00_default_construction(self):
        geom = GeometryDescriptor(name="test_geom", body=0)
        msg.info(f"geom: {geom}")

        self.assertIsInstance(geom, GeometryDescriptor)
        self.assertEqual(geom.name, "test_geom")
        self.assertEqual(geom.body, 0)
        self.assertEqual(geom.shape, None)
        self.assertEqual(geom.wid, -1)
        self.assertEqual(geom.gid, -1)
        self.assertEqual(geom.material, None)
        self.assertEqual(geom.group, 1)
        self.assertEqual(geom.collides, 1)
        self.assertEqual(geom.max_contacts, 0)
        self.assertEqual(geom.margin, 0.0)
        self.assertEqual(geom.mid, -1)

    def test_01_with_shape(self):
        cgeom = GeometryDescriptor(name="test_geom", body=0, shape=SphereShape(radius=1.0, name="test_sphere"))
        msg.info(f"cgeom: {cgeom}")

        self.assertIsInstance(cgeom, GeometryDescriptor)
        self.assertEqual(cgeom.name, "test_geom")
        self.assertEqual(cgeom.body, 0)
        self.assertEqual(cgeom.shape.type, ShapeType.SPHERE)
        self.assertEqual(cgeom.shape.radius, 1.0)
        self.assertEqual(cgeom.wid, -1)
        self.assertEqual(cgeom.gid, -1)
        self.assertEqual(cgeom.material, None)
        self.assertEqual(cgeom.group, 1)
        self.assertEqual(cgeom.collides, 1)
        self.assertEqual(cgeom.max_contacts, 0)
        self.assertEqual(cgeom.margin, 0.0)
        self.assertEqual(cgeom.mid, -1)

    def test_02_with_shape_and_material(self):
        test_material = MaterialDescriptor(name="test_material")
        cgeom = GeometryDescriptor(
            name="test_geom",
            body=0,
            shape=SphereShape(radius=1.0, name="test_sphere"),
            material=test_material.name,
        )
        msg.info(f"cgeom: {cgeom}")

        self.assertIsInstance(cgeom, GeometryDescriptor)
        self.assertEqual(cgeom.name, "test_geom")
        self.assertEqual(cgeom.body, 0)
        self.assertEqual(cgeom.shape.type, ShapeType.SPHERE)
        self.assertEqual(cgeom.shape.radius, 1.0)
        self.assertEqual(cgeom.wid, -1)
        self.assertEqual(cgeom.gid, -1)
        self.assertEqual(cgeom.material, test_material.name)
        self.assertEqual(cgeom.group, 1)
        self.assertEqual(cgeom.collides, 1)
        self.assertEqual(cgeom.max_contacts, 0)
        self.assertEqual(cgeom.margin, 0.0)
        self.assertEqual(cgeom.mid, -1)

    def test_03_from_base_geometry(self):
        geom = GeometryDescriptor(name="test_geom", body=0, shape=SphereShape(radius=1.0, name="test_sphere"))
        msg.info(f"geom: {geom}")

        self.assertIsInstance(geom, GeometryDescriptor)
        self.assertEqual(geom.name, "test_geom")
        self.assertEqual(geom.body, 0)
        self.assertEqual(geom.shape.type, ShapeType.SPHERE)
        self.assertEqual(geom.shape.radius, 1.0)
        self.assertEqual(geom.wid, -1)
        self.assertEqual(geom.gid, -1)
        self.assertEqual(geom.material, None)
        self.assertEqual(geom.group, 1)
        self.assertEqual(geom.collides, 1)
        self.assertEqual(geom.max_contacts, 0)
        self.assertEqual(geom.margin, 0.0)
        self.assertEqual(geom.mid, -1)


class TestMaterialDescriptor(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose

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

    def test_00_default_construction(self):
        mat = MaterialDescriptor(name="test_mat")
        msg.info(f"mat: {mat}")

        self.assertIsInstance(mat, MaterialDescriptor)
        self.assertEqual(mat.name, "test_mat")
        self.assertEqual(mat.density, DEFAULT_DENSITY)
        self.assertEqual(mat.restitution, DEFAULT_RESTITUTION)
        self.assertEqual(mat.static_friction, DEFAULT_FRICTION)
        self.assertEqual(mat.dynamic_friction, DEFAULT_FRICTION)
        self.assertEqual(mat.wid, -1)
        self.assertEqual(mat.mid, -1)

    def test_01_with_properties(self):
        mat = MaterialDescriptor(
            name="test_mat",
            density=500.0,
            restitution=0.5,
            static_friction=0.6,
            dynamic_friction=0.4,
        )
        msg.info(f"mat: {mat}")

        self.assertIsInstance(mat, MaterialDescriptor)
        self.assertEqual(mat.name, "test_mat")
        self.assertEqual(mat.density, 500.0)
        self.assertEqual(mat.restitution, 0.5)
        self.assertEqual(mat.static_friction, 0.6)
        self.assertEqual(mat.dynamic_friction, 0.4)
        self.assertEqual(mat.wid, -1)
        self.assertEqual(mat.mid, -1)


class TestWorldDescriptor(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose

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

    def test_00_default_construction(self):
        world = WorldDescriptor(name="test_world")
        self.assertIsInstance(world, WorldDescriptor)
        msg.info(f"world.name: {world.name}")
        msg.info(f"world.uid: {world.uid}")
        msg.info(f"world.num_bodies: {world.num_bodies}")
        msg.info(f"world.num_joints: {world.num_joints}")
        msg.info(f"world.body_names: {world.body_names}")
        msg.info(f"world.joint_names: {world.joint_names}")
        msg.info(f"world.mass_min: {world.mass_min}")
        msg.info(f"world.mass_max: {world.mass_max}")
        msg.info(f"world.mass_total: {world.mass_total}")
        msg.info(f"world.inertia_total: {world.inertia_total}")
        self.assertEqual(world.name, "test_world")
        self.assertEqual(world.num_bodies, 0)
        self.assertEqual(world.num_joints, 0)
        self.assertEqual(world.num_geoms, 0)
        self.assertEqual(len(world.body_names), 0)
        self.assertEqual(len(world.joint_names), 0)
        self.assertEqual(len(world.geom_names), 0)
        self.assertEqual(world.mass_min, math.inf)
        self.assertEqual(world.mass_max, 0.0)
        self.assertEqual(world.mass_total, 0.0)
        self.assertEqual(world.inertia_total, 0.0)

    def test_10_add_body(self):
        world = WorldDescriptor(name="test_world", wid=37)
        msg.info(f"world.name: {world.name}")
        msg.info(f"world.uid: {world.uid}")

        # Add two bodies to the world
        body_0 = RigidBodyDescriptor(name="body_0", m_i=1.0)
        world.add_body(body_0)
        msg.info(f"body_0: {body_0}")
        self.assertEqual(body_0.bid, 0)
        self.assertEqual(body_0.wid, world.wid)
        body_1 = RigidBodyDescriptor(name="body_1", m_i=0.5)
        world.add_body(body_1)
        msg.info(f"body_1: {body_1}")
        self.assertEqual(body_1.bid, 1)
        self.assertEqual(body_1.wid, world.wid)

        # Verify world properties
        self.assertEqual(world.num_bodies, 2)
        self.assertIn(body_0.name, world.body_names)
        self.assertIn(body_1.name, world.body_names)
        self.assertEqual(world.mass_min, 0.5)
        self.assertEqual(world.mass_max, 1.0)
        self.assertEqual(world.mass_total, 1.5)
        self.assertEqual(world.inertia_total, 4.5)

    def test_20_add_joint_revolute_passive(self):
        world = WorldDescriptor(name="test_world", wid=42)
        msg.info(f"world.name: {world.name}")
        msg.info(f"world.uid: {world.uid}")

        # Add two bodies to the world
        body_0 = RigidBodyDescriptor(name="body_0", m_i=1.0)
        world.add_body(body_0)
        msg.info(f"body_0: {body_0}")
        self.assertEqual(body_0.bid, 0)
        self.assertEqual(body_0.wid, world.wid)
        body_1 = RigidBodyDescriptor(name="body_1", m_i=0.5)
        world.add_body(body_1)
        msg.info(f"body_1: {body_1}")
        self.assertEqual(body_1.bid, 1)
        self.assertEqual(body_1.wid, world.wid)

        # Define a joint between two bodies
        joint_0 = JointDescriptor(
            name="body_0_to_1",
            dof_type=JointDoFType.REVOLUTE,
            act_type=JointActuationType.PASSIVE,
            bid_B=body_0.bid,
            bid_F=body_1.bid,
        )
        world.add_joint(joint_0)
        msg.info(f"joint_0: {joint_0}")
        self.assertEqual(joint_0.jid, 0)
        self.assertEqual(joint_0.wid, world.wid)
        self.assertFalse(joint_0.is_actuated)
        self.assertTrue(joint_0.is_binary)
        self.assertTrue(joint_0.is_connected_to_body(body_0.bid))
        self.assertTrue(joint_0.is_connected_to_body(body_1.bid))

        # Verify world properties
        self.assertEqual(world.num_bodies, 2)
        self.assertEqual(world.num_joints, 1)
        self.assertIn(body_0.name, world.body_names)
        self.assertIn(body_1.name, world.body_names)
        self.assertIn(joint_0.name, world.joint_names)
        self.assertIn(joint_0.name, world.passive_joint_names)
        self.assertEqual(world.mass_min, 0.5)
        self.assertEqual(world.mass_max, 1.0)
        self.assertEqual(world.mass_total, 1.5)
        self.assertEqual(world.inertia_total, 4.5)

    def test_21_add_joint_revolute_actuated_dynamic(self):
        world = WorldDescriptor(name="test_world", wid=42)
        msg.info(f"world.name: {world.name}")
        msg.info(f"world.uid: {world.uid}")

        # Add two bodies to the world
        body_0 = RigidBodyDescriptor(name="body_0", m_i=1.0)
        world.add_body(body_0)
        msg.info(f"body_0: {body_0}")
        self.assertEqual(body_0.bid, 0)
        self.assertEqual(body_0.wid, world.wid)
        body_1 = RigidBodyDescriptor(name="body_1", m_i=0.5)
        world.add_body(body_1)
        msg.info(f"body_1: {body_1}")
        self.assertEqual(body_1.bid, 1)
        self.assertEqual(body_1.wid, world.wid)

        # Define a joint between two bodies
        joint_0 = JointDescriptor(
            name="body_0_to_1",
            dof_type=JointDoFType.REVOLUTE,
            act_type=JointActuationType.PASSIVE,
            bid_B=body_0.bid,
            bid_F=body_1.bid,
            a_j=1.0,
            b_j=1.0,
        )
        world.add_joint(joint_0)
        msg.info(f"joint_0: {joint_0}")
        self.assertEqual(joint_0.jid, 0)
        self.assertEqual(joint_0.wid, world.wid)
        self.assertFalse(joint_0.is_actuated)
        self.assertTrue(joint_0.is_binary)
        self.assertTrue(joint_0.is_dynamic)
        self.assertTrue(joint_0.is_connected_to_body(body_0.bid))
        self.assertTrue(joint_0.is_connected_to_body(body_1.bid))

        # Verify world properties
        self.assertEqual(world.num_bodies, 2)
        self.assertEqual(world.num_joints, 1)
        self.assertIn(body_0.name, world.body_names)
        self.assertIn(body_1.name, world.body_names)
        self.assertIn(joint_0.name, world.joint_names)
        self.assertIn(joint_0.name, world.passive_joint_names)
        self.assertTrue(world.has_passive_dofs)
        self.assertFalse(world.has_actuated_dofs)
        self.assertTrue(world.has_implicit_dofs)

    def test_30_add_geometry(self):
        world = WorldDescriptor(name="test_world", wid=42)
        msg.info(f"world.name: {world.name}")
        msg.info(f"world.uid: {world.uid}")

        # Add two bodies to the world
        body_0 = RigidBodyDescriptor(name="body_0", m_i=1.0)
        world.add_body(body_0)
        msg.info(f"body_0: {body_0}")
        self.assertEqual(body_0.bid, 0)
        self.assertEqual(body_0.wid, world.wid)

        # Add physical geometry to body_0
        geom = GeometryDescriptor(name="test_geom", body=body_0.bid, shape=SphereShape(radius=1.0, name="test_sphere"))
        world.add_geometry(geom)
        msg.info(f"geom: {geom}")
        self.assertEqual(geom.name, "test_geom")
        self.assertEqual(geom.body, body_0.bid)
        self.assertEqual(geom.shape.type, ShapeType.SPHERE)
        self.assertEqual(geom.shape.radius, 1.0)
        self.assertEqual(geom.wid, world.wid)
        self.assertEqual(geom.gid, 0)

        # Verify world properties
        self.assertEqual(world.num_geoms, 1)
        self.assertIn(body_0.name, world.body_names)
        self.assertIn(geom.name, world.geom_names)

    def test_40_add_material(self):
        world = WorldDescriptor(name="test_world", wid=42)
        msg.info(f"world.name: {world.name}")
        msg.info(f"world.uid: {world.uid}")

        # Add a material to the world
        mat = MaterialDescriptor(name="test_mat")
        world.add_material(mat)
        msg.info(f"mat: {mat}")
        self.assertEqual(mat.name, "test_mat")
        self.assertEqual(mat.wid, world.wid)
        self.assertEqual(mat.mid, 0)

        # Verify world properties
        self.assertEqual(world.num_materials, 1)
        self.assertIn(mat.name, world.material_names)
        self.assertIn(mat.uid, world.material_uids)

    def test_50_set_base_body(self):
        world = WorldDescriptor(name="test_world", wid=42)
        msg.info(f"world.name: {world.name}")
        msg.info(f"world.uid: {world.uid}")

        # Add some bodies to the world
        body_0 = RigidBodyDescriptor(name="body_0", m_i=1.0)
        world.add_body(body_0)
        msg.info(f"body_0: {body_0}")
        self.assertEqual(body_0.bid, 0)
        self.assertEqual(body_0.wid, world.wid)
        body_1 = RigidBodyDescriptor(name="body_1", m_i=0.5)
        world.add_body(body_1)
        msg.info(f"body_1: {body_1}")
        self.assertEqual(body_1.bid, 1)
        self.assertEqual(body_1.wid, world.wid)
        body_2 = RigidBodyDescriptor(name="body_2", m_i=0.25)
        world.add_body(body_2)
        msg.info(f"body_2: {body_2}")
        self.assertEqual(body_2.bid, 2)
        self.assertEqual(body_2.wid, world.wid)

        # Set body_0 as the base body
        world.set_base_body(2)
        self.assertEqual(world.base_body_idx, body_2.bid)
        self.assertTrue(world.has_base_body)

        # Attempt to set an invalid body as the base body
        self.assertRaises(ValueError, world.set_base_body, 3)

    def test_51_set_base_joint(self):
        world = WorldDescriptor(name="test_world", wid=42)
        msg.info(f"world.name: {world.name}")
        msg.info(f"world.uid: {world.uid}")

        # Add some bodies to the world
        body_0 = RigidBodyDescriptor(name="body_0", m_i=1.0)
        world.add_body(body_0)
        msg.info(f"body_0: {body_0}")
        self.assertEqual(body_0.bid, 0)
        self.assertEqual(body_0.wid, world.wid)
        body_1 = RigidBodyDescriptor(name="body_1", m_i=0.5)
        world.add_body(body_1)
        msg.info(f"body_1: {body_1}")
        self.assertEqual(body_1.bid, 1)
        self.assertEqual(body_1.wid, world.wid)
        body_2 = RigidBodyDescriptor(name="body_2", m_i=0.25)
        world.add_body(body_2)
        msg.info(f"body_2: {body_2}")
        self.assertEqual(body_2.bid, 2)
        self.assertEqual(body_2.wid, world.wid)

        # Add some joints to the world
        joint_0 = JointDescriptor(
            name="body_0_to_1",
            dof_type=JointDoFType.REVOLUTE,
            act_type=JointActuationType.PASSIVE,
            bid_B=body_0.bid,
            bid_F=body_1.bid,
        )
        world.add_joint(joint_0)
        msg.info(f"joint_0: {joint_0}")
        self.assertEqual(joint_0.jid, 0)
        self.assertEqual(joint_0.wid, world.wid)
        joint_1 = JointDescriptor(
            name="body_1_to_2",
            dof_type=JointDoFType.REVOLUTE,
            act_type=JointActuationType.PASSIVE,
            bid_F=body_2.bid,
        )
        world.add_joint(joint_1)
        msg.info(f"joint_1: {joint_1}")
        self.assertEqual(joint_1.jid, 1)
        self.assertEqual(joint_1.wid, world.wid)

        # Set joint_1 as the base joint
        world.set_base_joint(1)
        self.assertEqual(world.base_joint_idx, joint_1.jid)
        self.assertTrue(world.has_base_joint)

        # Attempt to set an invalid joint as the base joint
        self.assertRaises(ValueError, world.set_base_joint, 2)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
