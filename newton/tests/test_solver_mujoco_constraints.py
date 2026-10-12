# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton._src.core.types import vec5
from newton._src.solvers.mujoco.equality import _add_equality_constraint
from newton._src.solvers.mujoco.utils import MJC_OBJ_JOINT, MjcEqualityTargetKind
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import assert_np_equal
from newton.tests.utils.mujoco import _add_deprecated_mimic_constraint


class TestMuJoCoSolverEqualityConstraintProperties(unittest.TestCase):
    def test_connect_reference_anchors_use_free_joint_coordinates(self):
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        body1_xform = wp.transform(wp.vec3(0.8, -0.3, 0.5), wp.quat_rpy(0.2, -0.4, 0.1))
        body2_xform = wp.transform(wp.vec3(-0.6, 1.1, -0.2), wp.quat_rpy(-0.3, 0.2, 0.4))
        parent1_xform = wp.transform(wp.vec3(0.2, -0.1, 0.3), wp.quat_rpy(0.1, 0.3, -0.2))
        child1_xform = wp.transform(wp.vec3(-0.2, 0.4, 0.1), wp.quat_rpy(-0.2, 0.1, 0.3))
        parent2_xform = wp.transform(wp.vec3(-0.3, 0.2, -0.1), wp.quat_rpy(0.3, -0.1, 0.2))
        child2_xform = wp.transform(wp.vec3(0.1, -0.2, 0.4), wp.quat_rpy(0.2, 0.4, -0.3))

        body1 = builder.add_link(xform=body1_xform, mass=1.0, inertia=wp.mat33(np.eye(3)))
        body2 = builder.add_link(xform=body2_xform, mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint1 = builder.add_joint_free(
            child=body1,
            parent_xform=parent1_xform,
            child_xform=child1_xform,
        )
        joint2 = builder.add_joint_free(
            child=body2,
            parent_xform=parent2_xform,
            child_xform=child2_xform,
        )
        builder.add_articulation([joint1])
        builder.add_articulation([joint2])

        anchor1 = wp.vec3(0.25, -0.15, 0.35)
        constraint = _add_equality_constraint(
            builder,
            constraint_type=SolverMuJoCo.EqType.CONNECT,
            body1=body1,
            body2=body2,
            anchor=anchor1,
        )

        model = builder.finalize()
        solver = SolverMuJoCo(model, disable_contacts=True)

        mapping = solver.mjc_eq_to_newton_eq.numpy()[0]
        matches = np.flatnonzero(mapping == constraint)
        self.assertEqual(len(matches), 1)
        eq_data = solver.mjw_model.eq_data.numpy()[0, matches[0]]

        anchor_world = wp.transform_point(body1_xform, anchor1)
        expected_anchor2 = wp.transform_point(wp.transform_inverse(body2_xform), anchor_world)
        np.testing.assert_allclose(eq_data[:3], np.array(anchor1), atol=1.0e-6)
        np.testing.assert_allclose(eq_data[3:6], np.array(expected_anchor2), atol=1.0e-5)

        ref_q = SolverMuJoCo._build_ref_q(model)
        np.testing.assert_allclose(ref_q.numpy(), model.joint_q.numpy(), atol=1.0e-6)
        ref_body_q = SolverMuJoCo._compute_body_poses_at_qref(model, ref_q).numpy()
        assert_np_equal(ref_body_q[body1], np.array(body1_xform), tol=1.0e-5)
        assert_np_equal(ref_body_q[body2], np.array(body2_xform), tol=1.0e-5)

    def test_eq_solver_parameters_conversion_and_update(self):
        """Convert and export equality solver parameters, then update them independently."""
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)
        b1 = template_builder.add_link()
        j1 = template_builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        template_builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j1])
        b2 = template_builder.add_link()
        j2 = template_builder.add_joint_revolute(-1, b2, axis=(0, 0, 1))
        template_builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j2])
        _add_equality_constraint(
            template_builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT,
            body1=b1,
            body2=b2,
            anchor=wp.vec3(0.1, 0.0, 0.0),
        )
        world_count = 2
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "eq_solref"))
        self.assertEqual(model.mujoco.equality_constraint_count, world_count)  # 1 constraint per world
        total_eq = model.mujoco.equality_constraint_count
        initial_values_solref = np.zeros((total_eq, 2), dtype=np.float32)
        for i in range(total_eq):
            # Unique pattern for 2-element solref
            initial_values_solref[i] = [
                0.01 + (i * 0.005) % 0.05,  # timeconst
                0.5 + (i * 0.2) % 1.5,  # dampratio
            ]
        model.mujoco.eq_solref.assign(initial_values_solref)
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "eq_solimp"))
        self.assertEqual(model.mujoco.equality_constraint_count, world_count)  # 1 constraint per world
        total_eq = model.mujoco.equality_constraint_count
        initial_values_solimp = np.zeros((total_eq, 5), dtype=np.float32)
        for i in range(total_eq):
            # Unique pattern for 5-element solimp (dmin, dmax, width, midpoint, power)
            initial_values_solimp[i] = [
                0.85 + (i * 0.02) % 0.1,  # dmin
                0.92 + (i * 0.01) % 0.05,  # dmax
                0.001 + (i * 0.0005) % 0.005,  # width
                0.4 + (i * 0.05) % 0.2,  # midpoint
                1.8 + (i * 0.2) % 1.0,  # power
            ]
        model.mujoco.eq_solimp.assign(wp.array(initial_values_solimp, dtype=vec5, device=model.device))
        with tempfile.TemporaryDirectory() as directory:
            xml_path = os.path.join(directory, "model.xml")
            solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, save_to_mjcf=xml_path)
            np.testing.assert_allclose(solver.mj_model.eq_solimp[0], initial_values_solimp[0], rtol=1e-5)
            connect_elems = list(ET.parse(xml_path).iter("connect"))
            self.assertEqual(len(connect_elems), 1)
            solimp_str = connect_elems[0].get("solimp")
            self.assertIsNotNone(solimp_str)
            np.testing.assert_allclose([float(x) for x in solimp_str.split()], initial_values_solimp[0], rtol=1e-4)

        with self.subTest(property="eq_solref", phase="conversion"):
            mjc_eq_to_newton_eq = solver.mjc_eq_to_newton_eq.numpy()
            mjw_eq_solref = solver.mjw_model.eq_solref.numpy()
            neq = mjc_eq_to_newton_eq.shape[1]  # Number of MuJoCo equality constraints

            def check_values_solref(expected_values, actual_mjw_values, msg_prefix):
                for w in range(world_count):
                    for mjc_eq in range(neq):
                        newton_eq = mjc_eq_to_newton_eq[w, mjc_eq]
                        if newton_eq < 0:
                            continue

                        expected = expected_values[newton_eq]
                        actual = actual_mjw_values[w, mjc_eq]

                        np.testing.assert_allclose(
                            actual,
                            expected,
                            rtol=1e-5,
                            err_msg=f"{msg_prefix} mismatch at World {w}, MuJoCo eq {mjc_eq}, Newton eq {newton_eq}",
                        )

            check_values_solref(initial_values_solref, mjw_eq_solref, "Initial conversion")

        with self.subTest(property="eq_solimp", phase="conversion"):
            mjc_eq_to_newton_eq = solver.mjc_eq_to_newton_eq.numpy()
            mjw_eq_solimp = solver.mjw_model.eq_solimp.numpy()
            neq = mjc_eq_to_newton_eq.shape[1]  # Number of MuJoCo equality constraints

            def check_values_solimp(expected_values, actual_mjw_values, msg_prefix):
                for w in range(world_count):
                    for mjc_eq in range(neq):
                        newton_eq = mjc_eq_to_newton_eq[w, mjc_eq]
                        if newton_eq < 0:
                            continue

                        expected = expected_values[newton_eq]
                        actual = actual_mjw_values[w, mjc_eq]

                        np.testing.assert_allclose(
                            actual,
                            expected,
                            rtol=1e-5,
                            err_msg=f"{msg_prefix} mismatch at World {w}, MuJoCo eq {mjc_eq}, Newton eq {newton_eq}",
                        )

            check_values_solimp(initial_values_solimp, mjw_eq_solimp, "Initial conversion")

        with self.subTest(property="eq_solref", phase="update"):
            updated_values_solref = np.zeros((total_eq, 2), dtype=np.float32)
            for i in range(total_eq):
                updated_values_solref[i] = [
                    0.05 - (i * 0.005) % 0.04,  # timeconst
                    2.0 - (i * 0.2) % 1.0,  # dampratio
                ]
            model.mujoco.eq_solref.assign(updated_values_solref)
            solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)
            mjw_eq_solref_updated = solver.mjw_model.eq_solref.numpy()
            check_values_solref(updated_values_solref, mjw_eq_solref_updated, "Runtime update")
            self.assertFalse(
                np.allclose(mjw_eq_solref_updated[0, 0], initial_values_solref[0]),
                "Value did not change from initial!",
            )

        with self.subTest(property="eq_solimp", phase="update"):
            updated_values_solimp = np.zeros((total_eq, 5), dtype=np.float32)
            for i in range(total_eq):
                updated_values_solimp[i] = [
                    0.80 - (i * 0.02) % 0.08,  # dmin
                    0.88 - (i * 0.01) % 0.04,  # dmax
                    0.005 - (i * 0.0005) % 0.003,  # width
                    0.55 - (i * 0.05) % 0.15,  # midpoint
                    2.2 - (i * 0.2) % 0.8,  # power
                ]
            model.mujoco.eq_solimp.assign(wp.array(updated_values_solimp, dtype=vec5, device=model.device))
            solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)
            mjw_eq_solimp_updated = solver.mjw_model.eq_solimp.numpy()
            check_values_solimp(updated_values_solimp, mjw_eq_solimp_updated, "Runtime update")
            self.assertFalse(
                np.allclose(mjw_eq_solimp_updated[0, 0], initial_values_solimp[0]),
                "Value did not change from initial!",
            )

    def test_eq_data_conversion_and_update(self):
        """
        Test validation of eq_data update from Newton equality constraint properties:
        - equality_constraint_anchor
        - equality_constraint_relpose
        - equality_constraint_polycoef
        - equality_constraint_torquescale

        Tests:
        1. Initial conversion from Model to MuJoCo (multi-world)
        2. Runtime updates (multi-world)
        """
        # Create template with multiple constraint types
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)

        # Create 3 bodies with free joints for CONNECT and WELD constraints
        b1 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        b2 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        b3 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        j1 = template_builder.add_joint_free(child=b1)
        j2 = template_builder.add_joint_free(child=b2)
        j3 = template_builder.add_joint_free(child=b3)
        template_builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_shape_box(body=b3, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j1])
        template_builder.add_articulation([j2])
        template_builder.add_articulation([j3])

        # Create 2 bodies with revolute joints for JOINT constraint
        b4 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        b5 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        j4 = template_builder.add_joint_revolute(parent=-1, child=b4, axis=wp.vec3(0.0, 0.0, 1.0))
        j5 = template_builder.add_joint_revolute(parent=-1, child=b5, axis=wp.vec3(0.0, 0.0, 1.0))
        template_builder.add_shape_box(body=b4, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_shape_box(body=b5, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j4])
        template_builder.add_articulation([j5])

        # Add a CONNECT constraint
        _add_equality_constraint(
            template_builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT,
            body1=b1,
            body2=b2,
            anchor=wp.vec3(0.1, 0.2, 0.3),
        )

        # Add a WELD constraint with specific relpose values
        weld_relpose = wp.transform(wp.vec3(0.01, 0.02, 0.03), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.1))
        _add_equality_constraint(
            template_builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.WELD,
            body1=b2,
            body2=b3,
            anchor=wp.vec3(0.5, 0.6, 0.7),
            relpose=weld_relpose,
            torquescale=0.5,
        )

        # Add a JOINT constraint with specific polycoef values
        joint_polycoef = [0.1, 0.2, 0.3, 0.4, 0.5]
        _add_equality_constraint(
            template_builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.JOINT,
            joint1=j4,
            joint2=j5,
            polycoef=joint_polycoef,
        )

        # Create main builder with multiple worlds
        world_count = 2
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # --- Step 1: Verify initial conversion ---
        mjc_eq_to_newton_eq = solver.mjc_eq_to_newton_eq.numpy()
        mjw_eq_data = solver.mjw_model.eq_data.numpy()
        neq = mjc_eq_to_newton_eq.shape[1]

        eq_constraint_anchor = model.mujoco.equality_constraint_anchor.numpy()
        eq_constraint_relpose = model.mujoco.equality_constraint_relpose.numpy()
        eq_constraint_polycoef = model.mujoco.equality_constraint_polycoef.numpy()
        eq_constraint_torquescale = model.mujoco.equality_constraint_torquescale.numpy()
        eq_constraint_type = model.mujoco.equality_constraint_type.numpy()

        for w in range(world_count):
            for mjc_eq in range(neq):
                newton_eq = mjc_eq_to_newton_eq[w, mjc_eq]
                if newton_eq < 0:
                    continue

                constraint_type = eq_constraint_type[newton_eq]
                actual = mjw_eq_data[w, mjc_eq]

                if constraint_type == 0:  # CONNECT
                    expected_anchor = eq_constraint_anchor[newton_eq]
                    np.testing.assert_allclose(
                        actual[:3],
                        expected_anchor,
                        rtol=1e-5,
                        err_msg=f"Initial CONNECT anchor mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )
                elif constraint_type == 1:  # WELD
                    expected_anchor = eq_constraint_anchor[newton_eq]
                    np.testing.assert_allclose(
                        actual[:3],
                        expected_anchor,
                        rtol=1e-5,
                        err_msg=f"Initial WELD anchor mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )
                    # Verify relpose translation (indices 3:6)
                    expected_relpose = eq_constraint_relpose[newton_eq]
                    expected_trans = expected_relpose[:3]  # translation is first 3 elements
                    np.testing.assert_allclose(
                        actual[3:6],
                        expected_trans,
                        rtol=1e-5,
                        err_msg=f"Initial WELD relpose translation mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )
                    # Verify relpose quaternion (indices 6:10)
                    # Newton stores as xyzw, MuJoCo expects wxyz
                    expected_quat_xyzw = expected_relpose[3:7]  # quaternion is elements 3-6
                    expected_quat_wxyz = [
                        expected_quat_xyzw[3],  # w
                        expected_quat_xyzw[0],  # x
                        expected_quat_xyzw[1],  # y
                        expected_quat_xyzw[2],  # z
                    ]
                    np.testing.assert_allclose(
                        actual[6:10],
                        expected_quat_wxyz,
                        rtol=1e-5,
                        err_msg=f"Initial WELD relpose quaternion mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )
                    # Verify torquescale (index 10)
                    expected_torquescale = eq_constraint_torquescale[newton_eq]
                    self.assertAlmostEqual(
                        actual[10],
                        expected_torquescale,
                        places=5,
                        msg=f"Initial WELD torquescale mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )
                elif constraint_type == 2:  # JOINT
                    # Verify polycoef (indices 0:5)
                    expected_polycoef = eq_constraint_polycoef[newton_eq]
                    np.testing.assert_allclose(
                        actual[:5],
                        expected_polycoef,
                        rtol=1e-5,
                        err_msg=f"Initial JOINT polycoef mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )

        # --- Step 2: Runtime update ---

        # Update anchor for all constraints
        new_anchors = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
            ],
            dtype=np.float32,
        )
        model.mujoco.equality_constraint_anchor.assign(new_anchors[: model.mujoco.equality_constraint_count])

        # Update torquescale for WELD constraints
        new_torquescale = np.array([0.0, 0.9, 0.0, 0.0, 0.8, 0.0], dtype=np.float32)
        model.mujoco.equality_constraint_torquescale.assign(new_torquescale[: model.mujoco.equality_constraint_count])

        # Update relpose for WELD constraints
        new_relpose = np.zeros((model.mujoco.equality_constraint_count, 7), dtype=np.float32)
        # Set new relpose for WELD constraint (index 1 in template, indices 1 and 4 after replication)
        new_trans = [0.11, 0.22, 0.33]
        new_quat_xyzw = [0.0, 0.0, 0.38268343, 0.92387953]  # 45 degrees around Z
        new_relpose[1] = new_trans + new_quat_xyzw
        new_relpose[4] = new_trans + new_quat_xyzw
        model.mujoco.equality_constraint_relpose.assign(new_relpose)

        # Update polycoef for JOINT constraints
        new_polycoef = np.zeros((model.mujoco.equality_constraint_count, 5), dtype=np.float32)
        # Set new polycoef for JOINT constraint (index 2 in template, indices 2 and 5 after replication)
        new_polycoef[2] = [1.1, 1.2, 1.3, 1.4, 1.5]
        new_polycoef[5] = [1.1, 1.2, 1.3, 1.4, 1.5]
        model.mujoco.equality_constraint_polycoef.assign(new_polycoef)

        # Notify solver
        solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)

        # Verify updates
        mjw_eq_data_updated = solver.mjw_model.eq_data.numpy()
        eq_constraint_anchor_updated = model.mujoco.equality_constraint_anchor.numpy()
        eq_constraint_relpose_updated = model.mujoco.equality_constraint_relpose.numpy()
        eq_constraint_polycoef_updated = model.mujoco.equality_constraint_polycoef.numpy()
        eq_constraint_torquescale_updated = model.mujoco.equality_constraint_torquescale.numpy()

        for w in range(world_count):
            for mjc_eq in range(neq):
                newton_eq = mjc_eq_to_newton_eq[w, mjc_eq]
                if newton_eq < 0:
                    continue

                constraint_type = eq_constraint_type[newton_eq]
                actual = mjw_eq_data_updated[w, mjc_eq]

                if constraint_type == 0:  # CONNECT
                    expected_anchor = eq_constraint_anchor_updated[newton_eq]
                    np.testing.assert_allclose(
                        actual[:3],
                        expected_anchor,
                        rtol=1e-5,
                        err_msg=f"Updated CONNECT anchor mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )
                elif constraint_type == 1:  # WELD
                    expected_anchor = eq_constraint_anchor_updated[newton_eq]
                    np.testing.assert_allclose(
                        actual[:3],
                        expected_anchor,
                        rtol=1e-5,
                        err_msg=f"Updated WELD anchor mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )
                    # Verify updated relpose translation (indices 3:6)
                    expected_relpose = eq_constraint_relpose_updated[newton_eq]
                    expected_trans = expected_relpose[:3]
                    np.testing.assert_allclose(
                        actual[3:6],
                        expected_trans,
                        rtol=1e-5,
                        err_msg=f"Updated WELD relpose translation mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )
                    # Verify updated relpose quaternion (indices 6:10)
                    expected_quat_xyzw = expected_relpose[3:7]
                    expected_quat_wxyz = [
                        expected_quat_xyzw[3],  # w
                        expected_quat_xyzw[0],  # x
                        expected_quat_xyzw[1],  # y
                        expected_quat_xyzw[2],  # z
                    ]
                    np.testing.assert_allclose(
                        actual[6:10],
                        expected_quat_wxyz,
                        rtol=1e-5,
                        err_msg=f"Updated WELD relpose quaternion mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )
                    # Verify updated torquescale (index 10)
                    expected_torquescale = eq_constraint_torquescale_updated[newton_eq]
                    self.assertAlmostEqual(
                        actual[10],
                        expected_torquescale,
                        places=5,
                        msg=f"Updated WELD torquescale mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )
                elif constraint_type == 2:  # JOINT
                    # Verify updated polycoef (indices 0:5)
                    expected_polycoef = eq_constraint_polycoef_updated[newton_eq]
                    np.testing.assert_allclose(
                        actual[:5],
                        expected_polycoef,
                        rtol=1e-5,
                        err_msg=f"Updated JOINT polycoef mismatch at World {w}, MuJoCo eq {mjc_eq}",
                    )

    def test_eq_active_update_preserves_connect_anchor(self):
        """Toggle constraints per world without overwriting CONNECT's second anchor."""
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)
        b1 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        j1 = template_builder.add_joint_free(child=b1)
        template_builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j1])
        b2 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        j2 = template_builder.add_joint_free(child=b2)
        template_builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j2])
        _add_equality_constraint(
            template_builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT,
            body1=b1,
            body2=b2,
            anchor=wp.vec3(0.1, 0.2, 0.3),
            enabled=True,
        )
        world_count = 2
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()
        self.assertEqual(model.mujoco.equality_constraint_count, world_count)  # 1 constraint per world
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        initial_eq_data = solver.mjw_model.eq_data.numpy().copy()

        mjc_eq_to_newton_eq = solver.mjc_eq_to_newton_eq.numpy()
        mjw_eq_active = solver.mjw_data.eq_active.numpy()
        neq = mjc_eq_to_newton_eq.shape[1]
        eq_enabled = model.mujoco.equality_constraint_enabled.numpy()
        for w in range(world_count):
            for mjc_eq in range(neq):
                newton_eq = mjc_eq_to_newton_eq[w, mjc_eq]
                if newton_eq < 0:
                    continue

                expected = eq_enabled[newton_eq]
                actual = mjw_eq_active[w, mjc_eq]
                self.assertEqual(
                    bool(actual),
                    bool(expected),
                    f"Initial eq_active mismatch at World {w}, MuJoCo eq {mjc_eq}: expected {expected}, got {actual}",
                )
        solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)
        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy()[:, :, 3:6], initial_eq_data[:, :, 3:6], rtol=1e-5)

        new_enabled = np.array([False, True], dtype=bool)
        model.mujoco.equality_constraint_enabled.assign(new_enabled)
        solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)
        mjw_eq_active_updated = solver.mjw_data.eq_active.numpy()
        for w in range(world_count):
            for mjc_eq in range(neq):
                newton_eq = mjc_eq_to_newton_eq[w, mjc_eq]
                if newton_eq < 0:
                    continue

                expected = new_enabled[newton_eq]
                actual = mjw_eq_active_updated[w, mjc_eq]
                self.assertEqual(
                    bool(actual),
                    bool(expected),
                    f"Updated eq_active mismatch at World {w}, MuJoCo eq {mjc_eq}: expected {expected}, got {actual}",
                )
        new_enabled = np.array([True, True], dtype=bool)
        model.mujoco.equality_constraint_enabled.assign(new_enabled)
        solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)
        mjw_eq_active_reenabled = solver.mjw_data.eq_active.numpy()
        for w in range(world_count):
            for mjc_eq in range(neq):
                newton_eq = mjc_eq_to_newton_eq[w, mjc_eq]
                if newton_eq < 0:
                    continue

                actual = mjw_eq_active_reenabled[w, mjc_eq]
                self.assertEqual(
                    bool(actual),
                    True,
                    f"Re-enabled eq_active mismatch at World {w}, MuJoCo eq {mjc_eq}: expected True, got {actual}",
                )
        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy()[:, :, 3:6], initial_eq_data[:, :, 3:6], rtol=1e-5)


class TestMuJoCoSolverMimicConstraints(unittest.TestCase):
    """Tests for mimic constraint support in SolverMuJoCo."""

    def _make_two_revolute_model(self, coef0=0.0, coef1=1.0, enabled=True):
        """Create a model with two revolute joints and a mimic constraint.

        Args:
            coef0: Offset coefficient for the mimic constraint (joint0 = coef0 + coef1 * joint1).
            coef1: Scale coefficient for the mimic constraint.
            enabled: Whether the mimic constraint is active.

        Returns:
            Finalized Newton Model with two revolute joints linked by a mimic constraint.
        """
        builder = newton.ModelBuilder()
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        j1 = builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        j2 = builder.add_joint_revolute(-1, b2, axis=(0, 0, 1))
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([j1, j2])
        _add_deprecated_mimic_constraint(
            builder,
            joint0=j2,
            joint1=j1,
            coef0=coef0,
            coef1=coef1,
            enabled=enabled,
        )
        return builder.finalize()

    def test_mimic_constraint_conversion(self):
        """Test that mimic constraints are converted to MuJoCo mjEQ_JOINT constraints."""
        model = self._make_two_revolute_model(coef0=0.5, coef1=2.0)
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify MuJoCo has 1 equality constraint of type JOINT
        self.assertEqual(solver.mj_model.neq, 1)
        self.assertEqual(solver.mj_model.eq_type[0], SolverMuJoCo._mujoco.mjtEq.mjEQ_JOINT)

        # Verify polycoef data: [coef0, coef1, 0, 0, 0]
        eq_data = solver.mjw_model.eq_data.numpy()
        np.testing.assert_allclose(eq_data[0, 0, :5], [0.5, 2.0, 0.0, 0.0, 0.0], rtol=1e-5)

        # Verify mapping exists
        self.assertIsNotNone(solver.mjc_eq_to_newton_mimic)
        mimic_map = solver.mjc_eq_to_newton_mimic.numpy()
        self.assertEqual(mimic_map[0, 0], 0)

    def test_joint_mimic_conversion_and_runtime_update(self):
        """Verify MuJoCo lowers and updates joint-owned mimic metadata."""
        builder = newton.ModelBuilder()
        body0 = builder.add_link(mass=1.0, com=wp.vec3(), inertia=wp.mat33(np.eye(3)))
        body1 = builder.add_link(mass=1.0, com=wp.vec3(), inertia=wp.mat33(np.eye(3)))
        reference = builder.add_joint_revolute(-1, body0, axis=newton.Axis.Z)
        follower = builder.add_joint_revolute(body0, body1, axis=newton.Axis.Z)
        builder.add_shape_box(body=body0, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([reference, follower])
        builder.set_joint_mimic(follower, reference, coeffs=(0.5, 2.0))
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        self.assertEqual(model.constraint_mimic_count, 0)
        self.assertEqual(solver.mj_model.neq, 1)
        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy()[0, 0, :5], [0.5, 2.0, 0.0, 0.0, 0.0])
        self.assertEqual(solver.mjc_eq_to_newton_joint_mimic.numpy()[0, 0], follower)
        self.assertEqual(solver.mjc_eq_to_newton_mimic.numpy()[0, 0], -1)

        coeffs = model.joint_mimic_coeffs.numpy()
        coeffs[follower] = (1.0, -3.0)
        model.joint_mimic_coeffs.assign(coeffs)
        solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)

        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy()[0, 0, :5], [1.0, -3.0, 0.0, 0.0, 0.0])

    def test_joint_mimic_d6_conversion(self):
        """Verify MuJoCo lowers multi-axis D6 mimic metadata componentwise."""
        builder = newton.ModelBuilder()
        body0 = builder.add_link(mass=1.0, com=wp.vec3(), inertia=wp.mat33(np.eye(3)))
        body1 = builder.add_link(mass=1.0, com=wp.vec3(), inertia=wp.mat33(np.eye(3)))
        axis = newton.ModelBuilder.JointDofConfig.create_unlimited
        axes = [axis(newton.Axis.X), axis(newton.Axis.Y)]
        reference = builder.add_joint_d6(-1, body0, linear_axes=axes)
        follower = builder.add_joint_d6(body0, body1, linear_axes=axes)
        builder.add_shape_box(body=body0, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([reference, follower])
        builder.set_joint_mimic(follower, reference, coeffs=(0.5, 2.0))
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        self.assertEqual(solver.mj_model.neq, 2)
        np.testing.assert_allclose(
            solver.mjw_model.eq_data.numpy()[0, :, :5],
            [[0.5, 2.0, 0.0, 0.0, 0.0], [0.5, 2.0, 0.0, 0.0, 0.0]],
        )
        np.testing.assert_array_equal(solver.mjc_eq_to_newton_joint_mimic.numpy()[0], [follower, follower])

    def test_joint_mimic_multi_world_mapping(self):
        """Verify dense mimic mappings and coefficients remain per world."""
        template = newton.ModelBuilder()
        body0 = template.add_link(mass=1.0, com=wp.vec3(), inertia=wp.mat33(np.eye(3)))
        body1 = template.add_link(mass=1.0, com=wp.vec3(), inertia=wp.mat33(np.eye(3)))
        reference = template.add_joint_revolute(-1, body0, axis=newton.Axis.Z)
        follower = template.add_joint_revolute(body0, body1, axis=newton.Axis.Z)
        template.add_shape_box(body=body0, hx=0.1, hy=0.1, hz=0.1)
        template.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1)
        template.add_articulation([reference, follower])
        template.set_joint_mimic(follower, reference, coeffs=(0.5, 2.0))

        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        np.testing.assert_array_equal(solver.mjc_eq_to_newton_joint_mimic.numpy()[:, 0], [1, 3])
        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy()[:, 0, :2], [[0.5, 2.0], [0.5, 2.0]])

        coeffs = model.joint_mimic_coeffs.numpy()
        coeffs[3] = (-0.25, -4.0)
        model.joint_mimic_coeffs.assign(coeffs)
        solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)

        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy()[:, 0, :2], [[0.5, 2.0], [-0.25, -4.0]])

    def test_mimic_constraint_runtime_update(self):
        """Test that mimic constraint properties can be updated at runtime."""
        model = self._make_two_revolute_model(coef0=0.5, coef1=2.0)
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Modify coefficients via assign
        model.constraint_mimic_coef0.assign(np.array([1.0], dtype=np.float32))
        model.constraint_mimic_coef1.assign(np.array([3.0], dtype=np.float32))
        model.constraint_mimic_enabled.assign(np.array([False], dtype=bool))

        # Trigger update
        solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)

        # Verify updated values
        eq_data = solver.mjw_model.eq_data.numpy()
        np.testing.assert_allclose(eq_data[0, 0, 0], 1.0, rtol=1e-5)
        np.testing.assert_allclose(eq_data[0, 0, 1], 3.0, rtol=1e-5)
        eq_active = solver.mjw_data.eq_active.numpy()
        self.assertFalse(eq_active[0, 0])

    def test_preserved_mimic_runtime_update_keeps_mujoco_data(self):
        """Preserved MuJoCo mimic equalities sync active state without flattening polycoef."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        polycoef = np.array(
            [
                [0.25, 1.5, -0.2, 0.05, 0.01],
                [-0.5, 0.75, 0.3, -0.1, 0.02],
            ],
            dtype=np.float32,
        )
        solref = np.array([[0.04, 1.2], [0.08, 0.9]], dtype=np.float32)
        solimp = np.array(
            [
                [0.82, 0.91, 0.002, 0.45, 1.8],
                [0.75, 0.88, 0.004, 0.55, 1.6],
            ],
            dtype=np.float32,
        )

        for world in range(2):
            builder.begin_world()
            b1 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
            b2 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
            j1 = builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
            j2 = builder.add_joint_revolute(-1, b2, axis=(0, 0, 1))
            builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
            builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
            builder.add_articulation([j1, j2])
            mimic = _add_deprecated_mimic_constraint(
                builder,
                joint0=j2,
                joint1=j1,
                coef0=10.0 + world,
                coef1=20.0 + world,
                enabled=world == 0,
            )
            _add_equality_constraint(
                builder,
                constraint_type=newton.solvers.SolverMuJoCo.EqType.JOINT,
                joint1=j2,
                joint2=j1,
                polycoef=polycoef[world].tolist(),
                enabled=world == 0,
                custom_attributes={
                    "mujoco:eq_solref": wp.vec2(*solref[world]),
                    "mujoco:eq_solimp": vec5(*solimp[world]),
                    "mujoco:equality_constraint_target_kind": int(MjcEqualityTargetKind.MIMIC),
                    "mujoco:equality_constraint_target": mimic,
                    "mujoco:equality_constraint_objtype": MJC_OBJ_JOINT,
                },
            )
            builder.end_world()

        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        self.assertEqual(solver.mj_model.neq, 1)
        np.testing.assert_array_equal(solver.mjc_eq_to_newton_eq.numpy()[:, 0], [0, 1])
        np.testing.assert_array_equal(solver.mjc_eq_to_newton_mimic.numpy(), np.full((2, 1), -1, dtype=np.int32))
        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy()[:, 0, :5], polycoef, rtol=1e-5)
        np.testing.assert_allclose(solver.mjw_model.eq_solref.numpy()[:, 0], solref, rtol=1e-5)
        np.testing.assert_allclose(solver.mjw_model.eq_solimp.numpy()[:, 0], solimp, rtol=1e-5)
        np.testing.assert_array_equal(solver.mjw_data.eq_active.numpy()[:, 0], [True, False])

        # Normal Newton mimic coefficients must not overwrite preserved MuJoCo polycoef data.
        model.constraint_mimic_coef0.assign(np.array([100.0, 200.0], dtype=np.float32))
        model.constraint_mimic_coef1.assign(np.array([300.0, 400.0], dtype=np.float32))
        model.constraint_mimic_enabled.assign(np.array([False, True], dtype=bool))
        model.mujoco.equality_constraint_enabled.assign(np.array([False, True], dtype=bool))
        solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)

        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy()[:, 0, :5], polycoef, rtol=1e-5)
        np.testing.assert_allclose(solver.mjw_model.eq_solref.numpy()[:, 0], solref, rtol=1e-5)
        np.testing.assert_allclose(solver.mjw_model.eq_solimp.numpy()[:, 0], solimp, rtol=1e-5)
        np.testing.assert_array_equal(solver.mjw_data.eq_active.numpy()[:, 0], [False, True])

    def test_preserved_mimic_runtime_update_syncs_cpu_backend(self):
        """Preserved MuJoCo mimic equality updates are visible to MuJoCo-C."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        b1 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        j1 = builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        j2 = builder.add_joint_revolute(-1, b2, axis=(0, 0, 1))
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([j1, j2])
        mimic = _add_deprecated_mimic_constraint(
            builder,
            joint0=j2,
            joint1=j1,
            coef0=10.0,
            coef1=20.0,
            enabled=True,
        )
        _add_equality_constraint(
            builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.JOINT,
            joint1=j2,
            joint2=j1,
            polycoef=[0.25, 1.5, -0.2, 0.05, 0.01],
            enabled=True,
            custom_attributes={
                "mujoco:eq_solref": wp.vec2(0.04, 1.2),
                "mujoco:eq_solimp": vec5(0.82, 0.91, 0.002, 0.45, 1.8),
                "mujoco:equality_constraint_target_kind": int(MjcEqualityTargetKind.MIMIC),
                "mujoco:equality_constraint_target": mimic,
                "mujoco:equality_constraint_objtype": MJC_OBJ_JOINT,
            },
        )

        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, use_mujoco_cpu=True)

        updated_polycoef = np.array([[0.3, 1.7, -0.4, 0.08, 0.02]], dtype=np.float32)
        updated_solref = np.array([[0.08, 0.9]], dtype=np.float32)
        updated_solimp = np.array([[0.75, 0.88, 0.004, 0.55, 1.6]], dtype=np.float32)

        model.constraint_mimic_coef0.assign(np.array([100.0], dtype=np.float32))
        model.constraint_mimic_coef1.assign(np.array([300.0], dtype=np.float32))
        model.constraint_mimic_enabled.assign(np.array([False], dtype=bool))
        model.mujoco.equality_constraint_polycoef.assign(updated_polycoef)
        model.mujoco.equality_constraint_enabled.assign(np.array([False], dtype=bool))
        model.mujoco.eq_solref.assign(wp.array(updated_solref, dtype=wp.vec2, device=model.device))
        model.mujoco.eq_solimp.assign(wp.array(updated_solimp, dtype=vec5, device=model.device))
        solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)

        np.testing.assert_allclose(solver.mj_model.eq_data[:, :5], updated_polycoef, rtol=1e-5)
        np.testing.assert_allclose(solver.mj_model.eq_solref, updated_solref, rtol=1e-5)
        np.testing.assert_allclose(solver.mj_model.eq_solimp, updated_solimp, rtol=1e-5)
        np.testing.assert_array_equal(solver.mj_data.eq_active, [False])

    def test_mimic_no_constraints(self):
        """Test solver works with zero mimic constraints."""
        builder = newton.ModelBuilder()
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        j1 = builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([j1])
        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        self.assertEqual(model.constraint_mimic_count, 0)
        # No MuJoCo eq constraints created, so mapping should be all -1
        self.assertTrue(np.all(solver.mjc_eq_to_newton_mimic.numpy() == -1))

    def test_mimic_mixed_with_equality_constraints(self):
        """Test mimic constraints coexist with regular equality constraints."""
        builder = newton.ModelBuilder()
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        j1 = builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        j2 = builder.add_joint_revolute(-1, b2, axis=(0, 0, 1))
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([j1, j2])

        # Add a regular JOINT equality constraint
        _add_equality_constraint(
            builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.JOINT,
            joint1=j1,
            joint2=j2,
            polycoef=[0.0, 1.0, 0.0, 0.0, 0.0],
        )
        # Add a mimic constraint
        _add_deprecated_mimic_constraint(builder, joint0=j2, joint1=j1, coef0=0.0, coef1=1.0)

        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # 1 regular eq + 1 mimic = 2 MuJoCo eq constraints
        self.assertEqual(solver.mj_model.neq, 2)
        self.assertEqual(solver.mj_model.eq_type[0], SolverMuJoCo._mujoco.mjtEq.mjEQ_JOINT)
        self.assertEqual(solver.mj_model.eq_type[1], SolverMuJoCo._mujoco.mjtEq.mjEQ_JOINT)

    def test_mimic_constraint_simulation(self):
        """Test that mimic constraint enforces joint tracking during simulation."""
        # Use coef1=2.0 so the relationship is non-trivial: j2 = 2.0 * j1
        model = self._make_two_revolute_model(coef0=0.0, coef1=2.0)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        # Derive DOF indices from mimic constraint metadata
        mimic_joint1 = model.constraint_mimic_joint1.numpy()[0]  # leader
        mimic_joint0 = model.constraint_mimic_joint0.numpy()[0]  # follower
        joint_qd_start = model.joint_qd_start.numpy()
        leader_dof = joint_qd_start[mimic_joint1]
        follower_dof = joint_qd_start[mimic_joint0]

        # Set initial velocity on leader joint to create motion
        qd = state_in.joint_qd.numpy()
        qd[leader_dof] = 1.0
        state_in.joint_qd.assign(qd)

        solver = SolverMuJoCo(model, iterations=50, disable_contacts=True)

        dt = 0.01
        for _ in range(200):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

        # After simulation, follower (j2) should approximately equal 2.0 * leader (j1)
        q = state_in.joint_q.numpy()
        leader_q = float(q[leader_dof])
        follower_q = float(q[follower_dof])
        self.assertNotAlmostEqual(leader_q, 0.0, places=1, msg="Leader joint should have moved from initial position")
        np.testing.assert_allclose(
            follower_q, 2.0 * leader_q, atol=0.1, err_msg="Mimic follower should track 2x leader"
        )

    def test_mimic_constraint_multi_world_randomized(self):
        """Test mimic constraints with per-world randomized coefficients."""
        template_builder = newton.ModelBuilder()
        b1 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        b2 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        j1 = template_builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        j2 = template_builder.add_joint_revolute(-1, b2, axis=(0, 0, 1))
        template_builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j1, j2])
        _add_deprecated_mimic_constraint(template_builder, joint0=j2, joint1=j1, coef0=0.0, coef1=1.0)

        world_count = 3
        builder = newton.ModelBuilder()
        builder.replicate(template_builder, world_count)
        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify initial state
        self.assertEqual(model.constraint_mimic_count, world_count)
        self.assertEqual(solver.mj_model.neq, 1)

        # Randomize coefficients per world
        rng = np.random.default_rng(42)
        new_coef0 = rng.uniform(-1.0, 1.0, size=world_count).astype(np.float32)
        new_coef1 = rng.uniform(0.5, 3.0, size=world_count).astype(np.float32)
        model.constraint_mimic_coef0.assign(new_coef0)
        model.constraint_mimic_coef1.assign(new_coef1)

        solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)

        # Verify each world got its own coefficients
        eq_data = solver.mjw_model.eq_data.numpy()
        for w in range(world_count):
            np.testing.assert_allclose(
                eq_data[w, 0, 0], new_coef0[w], rtol=1e-5, err_msg=f"coef0 mismatch in world {w}"
            )
            np.testing.assert_allclose(
                eq_data[w, 0, 1], new_coef1[w], rtol=1e-5, err_msg=f"coef1 mismatch in world {w}"
            )


class TestEqualityWeldConstraintDefaults(unittest.TestCase):
    def test_equality_weld_constraint_defaults(self):
        """Test the default values of equality weld constraints."""
        mjcf = """<?xml version="1.0" encoding="utf-8"?>
    <mujoco model="equality_weld_constraint">
    <option timestep="0.002" gravity="0 0 0"/>

    <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
     <inertial pos="0 0 0" mass="1" diaginertia="1.0 1.0 1.0"/>

     <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1.0 1.0 1.0"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1.0 1.0 1.0"/>
      </body>
    </body>
  </worldbody>

    <!-- Equality constraint tying the two bodies together -->
  <equality>
    <!-- type="weld" constrains body positions; here link1 follows link2 -->
    <weld name="body_couple" body1="link1"/>
  </equality>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, ignore_inertial_definitions=False, convert_mjc_equality_constraints=False)
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        measured_torquescale = model.mujoco.equality_constraint_torquescale.numpy()[0]
        expected_torquescale = 1.0
        self.assertAlmostEqual(
            expected_torquescale,
            measured_torquescale,
            places=4,
            msg=f"expected_torquescale is {expected_torquescale}, measured_torquescale is {measured_torquescale}",
        )

        measured_body2 = model.mujoco.equality_constraint_body2.numpy()[0]
        expected_body2 = -1
        self.assertEqual(
            expected_body2,
            measured_body2,
            msg=f"expected_body2 is {expected_body2}, measured_body2 is {measured_body2}",
        )

        measured_anchor = model.mujoco.equality_constraint_anchor.numpy()[0]
        expected_anchor = [0, 0, 0]
        for i in range(0, 3):
            self.assertEqual(
                measured_anchor[i],
                expected_anchor[i],
                msg=f"expected_anchor[{i}] is {expected_anchor[i]}, measured_anchor[{i}] is {measured_anchor[i]}",
            )

        measured_relpose = model.mujoco.equality_constraint_relpose.numpy()[0]
        expected_relpose = [0, 1, 0, 0, 0, 0, 0]
        for i in range(0, 7):
            self.assertEqual(
                measured_relpose[i],
                expected_relpose[i],
                msg=f"expected_relpose[{i}] is {expected_relpose[i]}, measured_relpose[{i}] is {measured_relpose[i]}",
            )

        measured_solimp = solver.mjw_model.eq_solimp.numpy()[0]
        expected_solimp = [0.9, 0.95, 0.001, 0.5, 2]
        for i in range(0, 5):
            self.assertAlmostEqual(
                measured_solimp[0][i],
                expected_solimp[i],
                places=4,
                msg=f"expected_solimp[{i}] is {expected_solimp[i]}, measured_solimp[{i}] is {measured_solimp[0][i]}",
            )

        measured_solref = solver.mjw_model.eq_solref.numpy()[0]
        expected_solref = [0.02, 1]
        for i in range(0, 2):
            self.assertAlmostEqual(
                measured_solref[0][i],
                expected_solref[i],
                places=4,
                msg=f"expected_solref[{i}] is {expected_solref[i]}, measured_solref[{i}] is {measured_solref[0][i]}",
            )

    def test_weld_constraint_quat_spec_conversion(self):
        """Test that WELD constraint quaternion is correctly converted to MuJoCo wxyz format in the spec."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        b1 = builder.add_link()
        j1 = builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([j1])

        b2 = builder.add_link()
        j2 = builder.add_joint_revolute(-1, b2, axis=(0, 0, 1))
        builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([j2])

        # 90 degree rotation around Z axis
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0)
        _add_equality_constraint(
            builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.WELD,
            body1=b1,
            body2=b2,
            relpose=wp.transform(wp.vec3(0.1, 0.0, 0.0), rot),
        )

        model = builder.finalize()

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            xml_path = f.name
        try:
            solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, save_to_mjcf=xml_path)

            # Verify the compiled mj_model has the quaternion in wxyz format
            mj_eq_data = solver.mj_model.eq_data[0]
            quat_in_model = mj_eq_data[6:10]

            # Warp quaternion is xyzw: (0, 0, sin(pi/4), cos(pi/4)) = (0, 0, 0.7071, 0.7071)
            # MuJoCo wxyz should be: (cos(pi/4), 0, 0, sin(pi/4)) = (0.7071, 0, 0, 0.7071)
            expected_wxyz = [np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)]
            np.testing.assert_allclose(
                quat_in_model,
                expected_wxyz,
                atol=1e-4,
                err_msg=f"WELD quaternion in compiled model is {quat_in_model}, expected wxyz {expected_wxyz}",
            )

            # Parse the saved XML and verify the weld element has correct quaternion
            tree = ET.parse(xml_path)
            weld_elems = list(tree.iter("weld"))
            self.assertEqual(len(weld_elems), 1, "Expected one weld equality constraint")
            weld = weld_elems[0]
            data_str = weld.get("relpose")
            self.assertIsNotNone(data_str, "relpose attribute missing from weld constraint in saved MJCF")
            relpose_values = [float(x) for x in data_str.split()]
            # relpose is "px py pz qw qx qy qz" in MuJoCo XML
            quat_in_xml = relpose_values[3:7]
            np.testing.assert_allclose(
                quat_in_xml,
                expected_wxyz,
                atol=1e-4,
                err_msg=f"WELD quaternion in saved MJCF is {quat_in_xml}, expected wxyz {expected_wxyz}",
            )
        finally:
            os.unlink(xml_path)


class TestEqualityJointObjType(unittest.TestCase):
    """Verify eq_objtype matches native MuJoCo for joint equality constraints.

    Regression test: Newton's solver previously passed objtype=mjOBJ_JOINT to
    spec.add_equality(), causing eq_objtype=3 in the compiled model. Native
    MuJoCo (from XML) produces eq_objtype=0. The fix: don't pass objtype for
    joint equality constraints, letting MuJoCo set it automatically.
    """

    def test_joint_equality_objtype_matches_native(self):
        """eq_objtype should match between Newton and native for joint equality."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        b1 = builder.add_link()
        j1 = builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        b2 = builder.add_link()
        j2 = builder.add_joint_revolute(-1, b2, axis=(0, 0, 1))
        builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([j1])
        builder.add_articulation([j2])
        _add_equality_constraint(
            builder, constraint_type=newton.solvers.SolverMuJoCo.EqType.JOINT, joint1=j1, joint2=j2
        )
        model = builder.finalize()

        solver = SolverMuJoCo(model)

        # Native: load equivalent MJCF
        import mujoco

        native_model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body><joint name="j1" type="hinge" axis="0 0 1"/><geom type="box" size="0.1 0.1 0.1"/></body>
            <body><joint name="j2" type="hinge" axis="0 0 1"/><geom type="box" size="0.1 0.1 0.1"/></body>
          </worldbody>
          <equality>
            <joint joint1="j1" joint2="j2"/>
          </equality>
        </mujoco>
        """)

        self.assertEqual(solver.mj_model.neq, native_model.neq, "neq count mismatch")
        np.testing.assert_array_equal(
            solver.mj_model.eq_objtype,
            native_model.eq_objtype,
            err_msg="eq_objtype mismatch: Newton should match native MuJoCo for joint equality",
        )
