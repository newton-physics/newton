# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton._src.core.types import vec5
from newton._src.solvers.mujoco.equality import _add_equality_constraint
from newton._src.solvers.mujoco.utils import MJC_OBJ_BODY, MJC_OBJ_JOINT, MjcEqualityTargetKind
from newton.examples import get_asset
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.tests.utils.mujoco import _add_deprecated_mimic_constraint


class TestMuJoCoConversion(unittest.TestCase):
    def test_cone_is_converted_to_mesh(self):
        """Convert Newton cones to meshes because MuJoCo has no cone primitive."""
        mujoco, _ = SolverMuJoCo.import_mujoco()
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_cone(body, radius=0.25, half_height=0.5)
        joint = builder.add_joint_free(body)
        builder.add_articulation([joint])

        solver = SolverMuJoCo(builder.finalize(device="cpu"))

        self.assertEqual(solver.mj_model.nmesh, 1)
        self.assertEqual(solver.mj_model.geom_type[0], mujoco.mjtGeom.mjGEOM_MESH)
        self.assertEqual(solver.mjw_model.geom_type.numpy()[0], mujoco.mjtGeom.mjGEOM_MESH)

    def test_cone_runtime_resize_is_rejected(self):
        """Reject cone resizing because its MuJoCo mesh is compiled during construction."""
        for use_mujoco_cpu in (False, True):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                builder = newton.ModelBuilder()
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                builder.add_shape_cone(body, radius=0.25, half_height=0.5)
                joint = builder.add_joint_free(body)
                builder.add_articulation([joint])
                model = builder.finalize(device="cpu")
                solver = SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu)

                model.shape_scale.assign([[0.5, 1.0, 0.0]])

                with self.assertRaisesRegex(ValueError, "Recreate the solver after resizing"):
                    solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)

    def test_setup_preserves_shape_scale(self):
        """Preserve model shape scales while converting MuJoCo geometry sizes."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_cylinder(body, radius=0.1, half_height=0.5)
        joint = builder.add_joint_free(body)
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        expected_shape_scale = model.shape_scale.numpy().copy()

        SolverMuJoCo(model, use_mujoco_cpu=True)

        np.testing.assert_array_equal(model.shape_scale.numpy(), expected_shape_scale)

    def test_no_shapes_separate_worlds_false(self):
        """Testing that an articulation without any shapes can be converted successfully when setting separate_worlds=False."""
        builder = newton.ModelBuilder()
        builder.bound_inertia = 0.01
        b0 = builder.add_link(mass=0.01)
        b1 = builder.add_link(mass=0.01)
        j0 = builder.add_joint_revolute(-1, b0)
        j1 = builder.add_joint_revolute(b0, b1)
        builder.add_articulation([j0, j1])
        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=False)
        self.assertEqual(solver.mj_model.nv, 2)

    def test_no_shapes_separate_worlds_true(self):
        """Testing that an articulation without any shapes can be converted successfully when setting separate_worlds=True."""
        builder = newton.ModelBuilder()
        builder.bound_inertia = 0.01
        builder.begin_world()
        b0 = builder.add_link(mass=0.01)
        b1 = builder.add_link(mass=0.01)
        j0 = builder.add_joint_revolute(-1, b0)
        j1 = builder.add_joint_revolute(b0, b1)
        builder.add_articulation([j0, j1])
        builder.end_world()
        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertEqual(solver.mj_model.nv, 2)

    def test_separate_worlds_false_multi_world_validation(self):
        """Test that separate_worlds=False is rejected for multi-world models."""
        # Create a model with 2 worlds
        template_builder = newton.ModelBuilder()
        body = template_builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        template_builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        joint = template_builder.add_joint_revolute(-1, body, axis=(0.0, 0.0, 1.0))
        template_builder.add_articulation([joint])

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for i in range(2):
            world_transform = wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity())
            builder.add_world(template_builder, xform=world_transform)

        model = builder.finalize()
        self.assertEqual(model.world_count, 2, "Model should have 2 worlds")

        # Test that separate_worlds=False raises ValueError
        with self.assertRaises(ValueError) as context:
            SolverMuJoCo(model, separate_worlds=False)

        self.assertIn("separate_worlds=False", str(context.exception))
        self.assertIn("single-world", str(context.exception))
        self.assertIn("world_count=2", str(context.exception))

        # Test that separate_worlds=True works fine
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertIsNotNone(solver)

    def test_separate_worlds_false_single_world_works(self):
        """Test that separate_worlds=False works correctly for single-world models."""
        builder = newton.ModelBuilder()
        b = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1)
        j = builder.add_joint_revolute(-1, b, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([j])
        model = builder.finalize()

        # Should work fine with single world
        solver = SolverMuJoCo(model, separate_worlds=False)
        self.assertIsNotNone(solver)
        self.assertEqual(solver.mj_model.nv, 1)

    def test_joint_transform_composition(self):
        """
        Test that the MuJoCo solver correctly handles joint transform composition,
        including a non-zero joint angle (joint_q) and nonzero joint translations.
        """
        builder = newton.ModelBuilder()

        # Add parent body (root) with identity transform and inertia
        parent_body = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
        )

        # Add child body with identity transform and inertia
        child_body = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
        )

        # Define translations for the joint frames in parent and child
        parent_joint_translation = wp.vec3(0.5, -0.2, 0.3)
        child_joint_translation = wp.vec3(-0.1, 0.4, 0.2)

        # Define orientations for the joint frames
        parent_xform = wp.transform(
            parent_joint_translation,
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi / 3),  # 60 deg about Y
        )
        child_xform = wp.transform(
            child_joint_translation,
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 4),  # 45 deg about X
        )

        # Add free joint to parent
        joint_free = builder.add_joint_free(parent_body)

        # Add revolute joint between parent and child with specified transforms and axis
        joint_revolute = builder.add_joint_revolute(
            parent=parent_body,
            child=child_body,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=(0.0, 0.0, 1.0),  # Revolute about Z
        )

        # Add articulation for the root free joint and the revolute joint
        builder.add_articulation([joint_free, joint_revolute])

        # Add simple box shapes for both bodies (not strictly needed for kinematics)
        builder.add_shape_box(body=parent_body, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=child_body, hx=0.1, hy=0.1, hz=0.1)

        # Set the joint angle (joint_q) for the revolute joint
        joint_angle = 0.5 * wp.pi  # 90 degrees
        builder.joint_q[7] = joint_angle  # Index 7: first dof after 7 root dofs

        model = builder.finalize()

        # Try to create the MuJoCo solver (skip if not available)
        try:
            solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")

        # Run forward kinematics using mujoco_warp
        solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)

        # Extract computed positions and orientations from MuJoCo data
        parent_pos = solver.mjw_data.xpos.numpy()[0, 1]
        parent_quat = solver.mjw_data.xquat.numpy()[0, 1]
        child_pos = solver.mjw_data.xpos.numpy()[0, 2]
        child_quat = solver.mjw_data.xquat.numpy()[0, 2]

        # Expected parent: at origin, identity orientation
        expected_parent_pos = np.array([0.0, 0.0, 0.0])
        expected_parent_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Compose expected child transform:
        #   - parent_xform: parent joint frame in parent
        #   - joint_rot: rotation from joint_q about joint axis
        #   - child_xform: child joint frame in child (inverse)
        joint_rot = wp.transform(
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), joint_angle),
        )
        t0 = wp.transform_multiply(wp.transform_identity(), parent_xform)  # parent to joint frame
        t1 = wp.transform_multiply(t0, joint_rot)  # apply joint rotation
        t2 = wp.transform_multiply(t1, wp.transform_inverse(child_xform))  # to child frame
        expected_child_xform = t2
        expected_child_pos = expected_child_xform.p
        expected_child_quat = expected_child_xform.q
        # Convert to MuJoCo quaternion order (w, x, y, z)
        expected_child_quat_mjc = np.array(
            [expected_child_quat.w, expected_child_quat.x, expected_child_quat.y, expected_child_quat.z]
        )

        # Check parent body pose
        np.testing.assert_allclose(
            parent_pos, expected_parent_pos, atol=1e-6, err_msg="Parent body position should be at origin"
        )
        np.testing.assert_allclose(
            parent_quat, expected_parent_quat, atol=1e-6, err_msg="Parent body quaternion should be identity"
        )

        # Check child body pose matches expected transform composition
        np.testing.assert_allclose(
            child_pos,
            expected_child_pos,
            atol=1e-6,
            err_msg="Child body position should match composed joint transforms (with joint_q and translations)",
        )
        np.testing.assert_allclose(
            child_quat,
            expected_child_quat_mjc,
            atol=1e-6,
            err_msg="Child body quaternion should match composed joint transforms (with joint_q and translations)",
        )

    def test_diagonal_to_full_inertia_preserves_dynamics(self):
        """Runtime inertia coupling matches a model compiled with that coupling."""

        def build_model(inertia):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_link(
                mass=0.1,
                com=wp.vec3(0.0, 0.0, 0.0),
                inertia=wp.mat33(inertia),
            )
            joint = builder.add_joint_free(child=body)
            builder.add_articulation([joint])
            return builder.finalize(), body

        diagonal_inertia = np.diag([0.04, 0.026, 0.05]).astype(np.float32)
        model, body = build_model(diagonal_inertia)
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        angle = np.pi / 6.0
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        full_inertia = rotation @ diagonal_inertia @ rotation.T
        model.body_inertia.assign(full_inertia[None, ...])
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        reference_model, reference_body = build_model(full_inertia)
        reference_solver = SolverMuJoCo(reference_model, iterations=1, disable_contacts=True)

        def step_with_torque(test_model, test_body, test_solver):
            state_in = test_model.state()
            state_out = test_model.state()
            newton.eval_fk(test_model, test_model.joint_q, test_model.joint_qd, state_in)
            state_in.body_f.assign([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
            test_solver.step(state_in, state_out, test_model.control(), None, 0.01)
            return state_out.body_qd.numpy()[test_body]

        actual_qd = step_with_torque(model, body, solver)
        reference_qd = step_with_torque(reference_model, reference_body, reference_solver)
        self.assertGreater(abs(float(reference_qd[4])), 1.0e-4)
        np.testing.assert_allclose(actual_qd, reference_qd, rtol=1.0e-5, atol=1.0e-6)

    def test_runtime_com_edit_preserves_dynamics(self):
        """A runtime COM edit matches a model compiled with that COM on both backends."""

        def build_model(com):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_link(
                mass=0.1,
                com=wp.vec3(*com),
                inertia=wp.mat33(np.diag([0.04, 0.026, 0.05])),
            )
            joint = builder.add_joint_free(child=body)
            builder.add_articulation([joint])
            return builder.finalize(), body

        def step_with_wrench(test_model, test_body, test_solver):
            state_in = test_model.state()
            state_out = test_model.state()
            newton.eval_fk(test_model, test_model.joint_q, test_model.joint_qd, state_in)
            state_in.body_f.assign([1.0, 0.0, 0.0, 0.1, 0.0, 0.0])
            test_solver.step(state_in, state_out, test_model.control(), None, 0.01)
            return state_out.body_qd.numpy()[test_body]

        com = np.array([0.05, 0.02, -0.03], dtype=np.float32)
        for use_mujoco_cpu in (False, True):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                # A body compiled with its COM at the origin is eligible for MuJoCo's simple layout.
                model, body = build_model((0.0, 0.0, 0.0))
                solver = SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu, iterations=1, disable_contacts=True)
                model.body_com.assign(com[None, :])
                solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

                reference_model, reference_body = build_model(com)
                reference_solver = SolverMuJoCo(
                    reference_model, use_mujoco_cpu=use_mujoco_cpu, iterations=1, disable_contacts=True
                )

                actual_qd = step_with_wrench(model, body, solver)
                reference_qd = step_with_wrench(reference_model, reference_body, reference_solver)
                np.testing.assert_allclose(actual_qd, reference_qd, rtol=1.0e-5, atol=1.0e-6)

    def test_global_joint_solver_params(self):
        """Test that global joint solver parameters affect joint limit behavior."""
        # Create a simple pendulum model
        builder = newton.ModelBuilder()

        # Add pendulum body
        mass = 1.0
        length = 1.0
        I_sphere = wp.diag(wp.vec3(2.0 / 5.0 * mass * 0.1**2, 2.0 / 5.0 * mass * 0.1**2, 2.0 / 5.0 * mass * 0.1**2))

        pendulum = builder.add_link(
            mass=mass,
            inertia=I_sphere,
        )

        # Add joint with limits - attach to world (-1)
        joint = builder.add_joint_revolute(
            parent=-1,  # World/ground
            child=pendulum,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, length), wp.quat_identity()),
            axis=newton.Axis.Y,
            limit_lower=0.0,  # Lower limit at 0 degrees
            limit_upper=np.pi / 2,  # Upper limit at 90 degrees
        )

        # Register the articulation containing the joint
        builder.add_articulation([joint])

        model = builder.finalize(requires_grad=False)
        state = model.state()

        # Initialize joint near lower limit with strong negative velocity
        state.joint_q.assign([0.1])  # Start above lower limit
        state.joint_qd.assign([-10.0])  # Very strong velocity towards lower limit

        # Create two models with different joint limit stiffness/damping
        # Soft model - more compliant, should allow more penetration
        model_soft = builder.finalize(requires_grad=False)
        # Set soft joint limits (low stiffness and damping)
        model_soft.joint_limit_ke.assign([100.0])  # Low stiffness
        model_soft.joint_limit_kd.assign([10.0])  # Low damping

        # Stiff model - less compliant, should allow less penetration
        model_stiff = builder.finalize(requires_grad=False)
        # Set stiff joint limits (high stiffness and damping)
        model_stiff.joint_limit_ke.assign([10000.0])  # High stiffness
        model_stiff.joint_limit_kd.assign([100.0])  # High damping

        # Create solvers
        solver_soft = newton.solvers.SolverMuJoCo(model_soft)
        solver_stiff = newton.solvers.SolverMuJoCo(model_stiff)

        dt = 0.005
        num_steps = 50

        # Simulate both systems
        state_soft_in = model_soft.state()
        state_soft_out = model_soft.state()
        state_stiff_in = model_stiff.state()
        state_stiff_out = model_stiff.state()

        # Copy initial state
        state_soft_in.joint_q.assign(state.joint_q.numpy())
        state_soft_in.joint_qd.assign(state.joint_qd.numpy())
        state_stiff_in.joint_q.assign(state.joint_q.numpy())
        state_stiff_in.joint_qd.assign(state.joint_qd.numpy())

        control_soft = model_soft.control()
        control_stiff = model_stiff.control()
        collision_pipeline_soft = newton.CollisionPipeline(model_soft)
        contacts_soft = collision_pipeline_soft.contacts()
        collision_pipeline_soft.collide(state_soft_in, contacts_soft)
        collision_pipeline_stiff = newton.CollisionPipeline(model_stiff)
        contacts_stiff = collision_pipeline_stiff.contacts()
        collision_pipeline_stiff.collide(state_stiff_in, contacts_stiff)

        # Track minimum positions during simulation
        min_q_soft = float("inf")
        min_q_stiff = float("inf")

        # Run simulations
        for _ in range(num_steps):
            solver_soft.step(state_soft_in, state_soft_out, control_soft, contacts_soft, dt)
            min_q_soft = min(min_q_soft, state_soft_out.joint_q.numpy()[0])
            state_soft_in, state_soft_out = state_soft_out, state_soft_in

            solver_stiff.step(state_stiff_in, state_stiff_out, control_stiff, contacts_stiff, dt)
            min_q_stiff = min(min_q_stiff, state_stiff_out.joint_q.numpy()[0])
            state_stiff_in, state_stiff_out = state_stiff_out, state_stiff_in

        # The soft joint should penetrate more (have a lower minimum) than the stiff joint
        self.assertLess(
            min_q_soft,
            min_q_stiff,
            f"Soft joint min ({min_q_soft}) should be lower than stiff joint min ({min_q_stiff})",
        )

    def test_joint_frame_update(self):
        """Test joint frame updates with specific expected values to verify correctness."""
        # Create a simple model with one revolute joint
        builder = newton.ModelBuilder()

        body = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(1.0, 1.0, 1.0)))

        # Add joint with known transforms
        parent_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        child_xform = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())

        joint = builder.add_joint_revolute(
            parent=-1,
            child=body,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=newton.Axis.X,
        )

        builder.add_articulation([joint])

        model = builder.finalize(requires_grad=False)
        solver = newton.solvers.SolverMuJoCo(model)

        # Find MuJoCo body for the Newton body by searching the mapping
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()
        mjc_body = -1
        for b in range(mjc_body_to_newton.shape[1]):
            if mjc_body_to_newton[0, b] == body:
                mjc_body = b
                break
        self.assertGreaterEqual(mjc_body, 0, "Could not find MuJoCo body for Newton body")

        # Check initial joint position and axis
        initial_joint_pos = solver.mjw_model.jnt_pos.numpy()
        initial_joint_axis = solver.mjw_model.jnt_axis.numpy()

        # Joint position should be at child frame position (0, 0, 1)
        np.testing.assert_allclose(
            initial_joint_pos[0, 0],
            [0.0, 0.0, 1.0],
            atol=1e-6,
            err_msg="Initial joint position should match child frame position",
        )

        # Joint axis should be X-axis (1, 0, 0) since child frame has no rotation
        np.testing.assert_allclose(
            initial_joint_axis[0, 0], [1.0, 0.0, 0.0], atol=1e-6, err_msg="Initial joint axis should be X-axis"
        )

        tf = parent_xform * wp.transform_inverse(child_xform)
        np.testing.assert_allclose(solver.mjw_model.body_pos.numpy()[0, mjc_body], tf.p, atol=1e-6)
        np.testing.assert_allclose(
            solver.mjw_model.body_quat.numpy()[0, mjc_body], [tf.q.w, tf.q.x, tf.q.y, tf.q.z], atol=1e-6
        )

        # Update child frame with translation and rotation
        new_child_pos = wp.vec3(1.0, 2.0, 1.0)
        new_child_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2)  # 90° around Z
        new_child_xform = wp.transform(new_child_pos, new_child_rot)

        model.joint_X_c.assign([new_child_xform])
        solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

        # Check updated values
        updated_joint_pos = solver.mjw_model.jnt_pos.numpy()
        updated_joint_axis = solver.mjw_model.jnt_axis.numpy()

        # Joint position should now be at new child frame position
        np.testing.assert_allclose(
            updated_joint_pos[0, 0],
            [1.0, 2.0, 1.0],
            atol=1e-6,
            err_msg="Updated joint position should match new child frame position",
        )

        # Joint axis should be rotated: X-axis rotated 90° around Z becomes Y-axis
        expected_axis = wp.quat_rotate(new_child_rot, wp.vec3(1.0, 0.0, 0.0))
        np.testing.assert_allclose(
            updated_joint_axis[0, 0],
            [expected_axis.x, expected_axis.y, expected_axis.z],
            atol=1e-6,
            err_msg="Updated joint axis should be rotated according to child frame rotation",
        )

        tf = parent_xform * wp.transform_inverse(new_child_xform)
        np.testing.assert_allclose(solver.mjw_model.body_pos.numpy()[0, mjc_body], tf.p, atol=1e-6)
        np.testing.assert_allclose(
            solver.mjw_model.body_quat.numpy()[0, mjc_body], [tf.q.w, tf.q.x, tf.q.y, tf.q.z], atol=1e-6
        )

        # update parent frame
        new_parent_xform = wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())
        model.joint_X_p.assign([new_parent_xform])
        solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

        # check updated values
        updated_joint_pos = solver.mjw_model.jnt_pos.numpy()
        updated_joint_axis = solver.mjw_model.jnt_axis.numpy()

        # joint position, axis should not change
        np.testing.assert_allclose(
            updated_joint_pos[0, 0],
            [1.0, 2.0, 1.0],
            atol=1e-6,
            err_msg="Updated joint position should not change after updating parent frame",
        )
        np.testing.assert_allclose(
            updated_joint_axis[0, 0],
            expected_axis,
            atol=1e-6,
            err_msg="Updated joint axis should not change after updating parent frame",
        )

        # Check updated body positions and orientations
        tf = new_parent_xform * wp.transform_inverse(new_child_xform)
        np.testing.assert_allclose(
            solver.mjw_model.body_pos.numpy()[0, mjc_body],
            tf.p,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            solver.mjw_model.body_quat.numpy()[0, mjc_body],
            [tf.q.w, tf.q.x, tf.q.y, tf.q.z],
            atol=1e-6,
        )

    def test_shape_offset_across_worlds(self):
        """Test that shape offset works correctly across different worlds in MuJoCo solver."""
        # Create a simple model with 2 worlds
        builder = newton.ModelBuilder()

        # Create shapes for world 1 at normal scale
        env1 = newton.ModelBuilder()
        body1 = env1.add_body(label="body1", mass=1.0)  # Add mass to make it dynamic

        # Add two spheres - one at origin, one offset
        env1.add_shape_sphere(
            body=body1,
            radius=0.1,
            xform=wp.transform([0, 0, 0], wp.quat_identity()),
        )
        env1.add_shape_sphere(
            body=body1,
            radius=0.1,
            xform=wp.transform([1.0, 0, 0], wp.quat_identity()),  # offset by 1 unit
        )

        # Add world 0 at normal scale
        builder.add_world(env1, xform=wp.transform_identity())

        # Create shapes for world 2 at 0.5x scale
        env2 = newton.ModelBuilder()
        body2 = env2.add_body(label="body2", mass=1.0)  # Add mass to make it dynamic

        # Add two spheres with manually scaled properties
        env2.add_shape_sphere(
            body=body2,
            radius=0.05,  # scaled radius
            xform=wp.transform([0, 0, 0], wp.quat_identity()),
        )
        env2.add_shape_sphere(
            body=body2,
            radius=0.05,  # scaled radius
            xform=wp.transform([0.5, 0, 0], wp.quat_identity()),  # scaled offset
        )

        # Add world 1 at different location
        builder.add_world(env2, xform=wp.transform([2.0, 0, 0], wp.quat_identity()))

        # Finalize model
        model = builder.finalize()

        # Create MuJoCo solver
        solver = newton.solvers.SolverMuJoCo(model)

        # Check geom positions in MuJoCo model
        # geom_pos stores body-local coordinates
        # World 0: sphere 1 at [0,0,0], sphere 2 at [1,0,0] (unscaled)
        # World 1: sphere 1 at [0,0,0], sphere 2 at [0.5,0,0] (scaled by 0.5)

        # Get geom positions from MuJoCo warp model
        geom_pos = solver.mjw_model.geom_pos.numpy()

        # Check body-local positions
        # World 0, Sphere 2 should be at x=1.0 (local offset)
        world0_sphere2_x = geom_pos[0, 1, 0]
        self.assertAlmostEqual(world0_sphere2_x, 1.0, places=3, msg="World 0 sphere 2 should have local x=1.0")

        # World 1, Sphere 2 should be at x=0.5 (local offset)
        world1_sphere2_x = geom_pos[1, 1, 0]
        expected_x = 0.5

        # Check that the second sphere in world 1 has the correct local position
        self.assertAlmostEqual(
            world1_sphere2_x,
            expected_x,
            places=3,
            msg=f"World 1 sphere 2 should have local x={expected_x} (scaled offset)",
        )

        # Check scaling of the spheres
        radii = solver.mjw_model.geom_size.numpy()[:, :, 0].flatten()
        expected_radii = [0.1, 0.1, 0.05, 0.05]
        np.testing.assert_allclose(radii, expected_radii, atol=1e-3)

    def test_static_worldbody_geoms_support_offset_worlds(self):
        """Offset worlds collide with their own finite static worldbody geometry."""
        world = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(world)
        world.default_shape_cfg.ke = 1.0e5
        world.default_shape_cfg.kd = 1.0e3
        world.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
            hx=1.0,
            hy=1.0,
            hz=0.1,
        )
        body = world.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        joint = world.add_joint_free(child=body)
        world.add_articulation([joint])
        world.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_world(world, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        builder.add_world(
            world,
            xform=wp.transform(
                wp.vec3(4.0, 0.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5),
            ),
        )
        model = builder.finalize()

        solver = SolverMuJoCo(
            model,
            separate_worlds=True,
            use_mujoco_contacts=True,
            integrator="implicitfast",
            iterations=10,
            nconmax=100,
            njmax=200,
        )

        np.testing.assert_allclose(
            solver.mjw_data.geom_xpos.numpy()[:, 0],
            solver.mjw_model.geom_pos.numpy()[:, 0],
            atol=1.0e-6,
            rtol=0.0,
        )
        geom_quat_wxyz = solver.mjw_model.geom_quat.numpy()[:, 0]
        expected_xmat = np.stack(
            [np.asarray(wp.quat_to_matrix(wp.quat(q[1], q[2], q[3], q[0]))).reshape(3, 3) for q in geom_quat_wxyz]
        )
        np.testing.assert_allclose(solver.mjw_data.geom_xmat.numpy()[:, 0], expected_xmat, atol=1.0e-6, rtol=0.0)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        dt = 0.005
        for _ in range(200):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

        np.testing.assert_allclose(
            state_0.body_q.numpy()[:, 2],
            np.full(2, 0.1, dtype=np.float32),
            atol=2.0e-3,
            rtol=0.0,
            err_msg="Every offset world must keep its free box supported by its local static table.",
        )

    def test_mesh_geoms_across_worlds(self):
        """Test that mesh geoms work correctly across different worlds in MuJoCo solver."""
        # Create a simple model with 2 worlds, each containing a mesh
        builder = newton.ModelBuilder()

        # Create a simple box mesh that is NOT centered at origin
        # The mesh center will be at (0.5, 0.5, 0.5)
        vertices = np.array(
            [
                # Bottom face (z=0)
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [1.0, 1.0, 0.0],  # 2
                [0.0, 1.0, 0.0],  # 3
                # Top face (z=1)
                [0.0, 0.0, 1.0],  # 4
                [1.0, 0.0, 1.0],  # 5
                [1.0, 1.0, 1.0],  # 6
                [0.0, 1.0, 1.0],  # 7
            ],
            dtype=np.float32,
        )

        # Define triangular faces (2 triangles per face)
        indices = np.array(
            [
                # Bottom face
                0,
                1,
                2,
                0,
                2,
                3,
                # Top face
                4,
                6,
                5,
                4,
                7,
                6,
                # Front face
                0,
                5,
                1,
                0,
                4,
                5,
                # Back face
                2,
                7,
                3,
                2,
                6,
                7,
                # Left face
                0,
                3,
                7,
                0,
                7,
                4,
                # Right face
                1,
                5,
                6,
                1,
                6,
                2,
            ],
            dtype=np.int32,
        )

        # Create mesh source
        mesh_src = newton.Mesh(vertices=vertices, indices=indices)

        # Create shapes for world 0
        env1 = newton.ModelBuilder()
        body1 = env1.add_body(label="mesh_body1", mass=1.0)

        # Add mesh shape at specific position
        env1.add_shape_mesh(
            body=body1,
            mesh=mesh_src,
            xform=wp.transform([1.0, 0, 0], wp.quat_identity()),  # offset by 1 unit in x
        )

        # Add world 0 at origin
        builder.add_world(env1, xform=wp.transform([0, 0, 0], wp.quat_identity()))

        # Create shapes for world 1
        env2 = newton.ModelBuilder()
        body2 = env2.add_body(label="mesh_body2", mass=1.0)

        # Add mesh shape at different position
        env2.add_shape_mesh(
            body=body2,
            mesh=mesh_src,
            xform=wp.transform([2.0, 0, 0], wp.quat_identity()),  # offset by 2 units in x
        )

        # Add world 1 at different location
        builder.add_world(env2, xform=wp.transform([5.0, 0, 0], wp.quat_identity()))

        # Finalize model
        model = builder.finalize()

        # Create MuJoCo solver
        solver = newton.solvers.SolverMuJoCo(model)

        # Verify that mesh_pos is non-zero (mesh center should be at 0.5, 0.5, 0.5)
        mesh_pos = solver.mjw_model.mesh_pos.numpy()
        self.assertEqual(len(mesh_pos), 1, "Should have exactly one mesh")
        self.assertAlmostEqual(mesh_pos[0][0], 0.5, places=3, msg="Mesh center x should be 0.5")
        self.assertAlmostEqual(mesh_pos[0][1], 0.5, places=3, msg="Mesh center y should be 0.5")
        self.assertAlmostEqual(mesh_pos[0][2], 0.5, places=3, msg="Mesh center z should be 0.5")

        # Check geom positions (body-local coordinates)
        geom_pos = solver.mjw_model.geom_pos.numpy()

        # World 0 mesh should be at x=1.5 (1.0 local offset + 0.5 mesh center)
        world0_mesh_x = geom_pos[0, 0, 0]
        self.assertAlmostEqual(
            world0_mesh_x, 1.5, places=3, msg="World 0 mesh should have local x=1.5 (local offset + mesh_pos)"
        )

        # World 1 mesh should be at x=2.5 (2.0 local offset + 0.5 mesh center)
        world1_mesh_x = geom_pos[1, 0, 0]
        self.assertAlmostEqual(
            world1_mesh_x, 2.5, places=3, msg="World 1 mesh should have local x=2.5 (local offset + mesh_pos)"
        )


class TestMuJoCoArticulationConversion(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_rootless_usd_mechanism_is_rejected(self):
        """A rootless mechanism without a standalone root for each body is unsupported."""
        builder = newton.ModelBuilder()
        builder.add_usd(get_asset("boxes_fourbar.usda"))
        model = builder.finalize(skip_validation_joints=True)

        with self.assertRaisesRegex(ValueError, "outside articulations"):
            SolverMuJoCo(model, disable_contacts=True)

    def test_orphan_world_fixed_body_is_exported_static(self):
        """A world-fixed body outside an articulation remains a static MuJoCo body."""
        builder = newton.ModelBuilder()
        body_xform = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())
        body = builder.add_link(xform=body_xform, mass=0.0, label="world_link")
        joint = builder.add_joint_fixed(
            parent=-1,
            child=body,
            parent_xform=body_xform,
            label="world_link_fixed_joint",
        )

        self.assertEqual(builder.articulation_count, 0)
        self.assertEqual(builder.joint_articulation[joint], -1)

        model = builder.finalize()
        with self.assertWarnsRegex(UserWarning, "standalone world roots"):
            solver = SolverMuJoCo(model, disable_contacts=True)

        self.assertEqual(model.articulation_count, 0)
        self.assertEqual(solver.mj_model.nbody, 2)
        self.assertEqual(solver.mj_model.njnt, 0)
        self.assertEqual(solver.mj_model.nq, 0)
        self.assertEqual(solver.mj_model.nv, 0)
        self.assertEqual(solver.mj_model.nmocap, 0)
        self.assertEqual(int(solver.mj_model.body_parentid[1]), 0)
        self.assertEqual(int(solver.mj_model.body_mocapid[1]), -1)
        self.assertEqual(int(solver.mj_model.body_jntnum[1]), 0)
        np.testing.assert_allclose(solver.mj_model.body_pos[1], [1.0, 2.0, 3.0], atol=1e-7)

    def test_orphan_world_fixed_kinematic_body_is_exported_as_mocap(self):
        """A kinematic world-fixed orphan preserves MuJoCo mocap semantics."""
        builder = newton.ModelBuilder()
        body = builder.add_link(is_kinematic=True, mass=0.0, label="mocap_link")
        joint = builder.add_joint_fixed(parent=-1, child=body, label="mocap_link_fixed_joint")

        self.assertEqual(builder.articulation_count, 0)
        self.assertEqual(builder.joint_articulation[joint], -1)

        model = builder.finalize()
        with self.assertWarnsRegex(UserWarning, "standalone world roots"):
            solver = SolverMuJoCo(model, disable_contacts=True)

        self.assertEqual(solver.mj_model.nbody, 2)
        self.assertEqual(solver.mj_model.njnt, 0)
        self.assertEqual(solver.mj_model.nmocap, 1)
        self.assertEqual(int(solver.mj_model.body_mocapid[1]), 0)

    def test_orphan_world_dynamic_joint_types_are_exported(self):
        """Standalone world joints retain their MuJoCo coordinates and DOFs."""
        cases = {
            "revolute": (1, 1, 1),
            "prismatic": (1, 1, 1),
            "ball": (1, 4, 3),
            "d6": (2, 2, 2),
        }
        for joint_type, (expected_njnt, expected_nq, expected_nv) in cases.items():
            with self.subTest(joint_type=joint_type):
                builder = newton.ModelBuilder()
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                if joint_type == "revolute":
                    joint = builder.add_joint_revolute(parent=-1, child=body)
                elif joint_type == "prismatic":
                    joint = builder.add_joint_prismatic(parent=-1, child=body)
                elif joint_type == "ball":
                    joint = builder.add_joint_ball(parent=-1, child=body)
                else:
                    axis = newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)
                    joint = builder.add_joint_d6(
                        parent=-1,
                        child=body,
                        linear_axes=[axis],
                        angular_axes=[axis],
                    )

                self.assertEqual(builder.joint_articulation[joint], -1)
                model = builder.finalize()
                with self.assertWarnsRegex(UserWarning, "standalone world roots"):
                    solver = SolverMuJoCo(model, disable_contacts=True)

                self.assertEqual(model.articulation_count, 0)
                self.assertEqual(solver.mj_model.nbody, 2)
                self.assertEqual(solver.mj_model.njnt, expected_njnt)
                self.assertEqual(solver.mj_model.nq, expected_nq)
                self.assertEqual(solver.mj_model.nv, expected_nv)
                self.assertEqual(solver.mj_model.neq, 0)

    def test_orphan_world_joints_support_separate_worlds(self):
        """Standalone roots retain body mappings when worlds are replicated."""
        template = newton.ModelBuilder()
        static_body = template.add_link(mass=0.0, label="static_body")
        template.add_joint_fixed(parent=-1, child=static_body, label="static_root")
        dynamic_body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="dynamic_body")
        template.add_joint_revolute(parent=-1, child=dynamic_body, label="dynamic_root")

        world_count = 3
        builder = newton.ModelBuilder()
        builder.replicate(template, world_count)
        model = builder.finalize()
        with self.assertWarnsRegex(UserWarning, "standalone world roots"):
            solver = SolverMuJoCo(model, separate_worlds=True, disable_contacts=True)

        self.assertEqual(model.world_count, world_count)
        self.assertEqual(solver.mj_model.nbody, 3)
        self.assertEqual(solver.mj_model.njnt, 1)
        expected_mapping = np.array([[-1, 0, 1], [-1, 2, 3], [-1, 4, 5]], dtype=np.int32)
        np.testing.assert_array_equal(solver.mjc_body_to_newton.numpy(), expected_mapping)

    def test_orphan_world_fixed_body_can_participate_in_loop_joint(self):
        """A standalone static body remains available to loop constraints."""
        builder = newton.ModelBuilder()
        static_body = builder.add_link(mass=0.0, label="static_body")
        static_root = builder.add_joint_fixed(parent=-1, child=static_body, label="static_root")
        dynamic_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="dynamic_body")
        dynamic_root = builder.add_joint_free(dynamic_body, label="dynamic_root")
        builder.add_articulation([dynamic_root])
        loop_joint = builder.add_joint_fixed(parent=static_body, child=dynamic_body, label="weld")

        self.assertEqual(builder.joint_articulation[static_root], -1)
        self.assertEqual(builder.joint_articulation[loop_joint], -1)

        model = builder.finalize()
        with self.assertWarnsRegex(UserWarning, "standalone world roots"):
            solver = SolverMuJoCo(model, disable_contacts=True)

        self.assertEqual(solver.mj_model.nbody, 3)
        self.assertEqual(solver.mj_model.njnt, 1)
        self.assertEqual(solver.mj_model.neq, 1)
        self.assertEqual(int(solver.mj_model.eq_type[0]), int(solver._mujoco.mjtEq.mjEQ_WELD))

    def test_world_fixed_loop_on_articulated_body_remains_weld(self):
        """A world-fixed joint on an articulated body remains a loop constraint."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        root_joint = builder.add_joint_free(body)
        builder.add_articulation([root_joint])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*FREE joint parallel.*", category=UserWarning)
            loop_joint = builder.add_joint_fixed(parent=-1, child=body)

        self.assertEqual(builder.joint_articulation[loop_joint], -1)

        model = builder.finalize()
        solver = SolverMuJoCo(model, disable_contacts=True)

        self.assertEqual(solver.mj_model.nbody, 2)
        self.assertEqual(solver.mj_model.njnt, 1)
        self.assertEqual(solver.mj_model.neq, 1)
        self.assertEqual(int(solver.mj_model.eq_type[0]), int(solver._mujoco.mjtEq.mjEQ_WELD))

    def test_loop_joints_only(self):
        """Testing that loop joints are converted to equality constraints."""
        builder = newton.ModelBuilder()
        b0 = builder.add_link(mass=0.01)
        b1 = builder.add_link(mass=0.01)
        b2 = builder.add_link(mass=0.01)
        j0 = builder.add_joint_revolute(-1, b0)
        j1 = builder.add_joint_revolute(b0, b1)
        j2 = builder.add_joint_revolute(b1, b2)
        builder.add_articulation([j0, j1, j2])
        # add a loop joint with asymmetric xforms to exercise relpose computation
        builder.add_joint_fixed(
            b2,
            b0,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.45), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.1, -0.3), wp.quat_identity()),
        )
        world_count = 4
        world_builder = newton.ModelBuilder()
        # force the ModelBuilder to correct zero mass/inertia values
        world_builder.bound_inertia = 0.01
        world_builder.bound_mass = 0.01
        world_builder.replicate(builder, world_count=world_count)
        model = world_builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertEqual(solver.mj_model.nv, 3)
        # Fixed loop joint → 1 weld constraint
        self.assertEqual(solver.mj_model.neq, 1)
        self.assertEqual(int(solver.mj_model.eq_type[0]), int(solver._mujoco.mjtEq.mjEQ_WELD))
        # Verify weld constraint data: anchor is set explicitly; relpose (data[3:10])
        # is auto-computed by MuJoCo's spec.compile() from body positions.
        eq_data = solver.mj_model.eq_data[0]
        assert np.allclose(eq_data[0:3], [0.0, 0.0, -0.45], atol=1e-6)
        # Auto-computed relpose: both bodies are at origin (default xforms), so
        # relpose translation equals the anchor offset, quaternion is identity.
        assert np.allclose(eq_data[3:6], [0.0, 0.0, -0.45], atol=1e-6)
        quat = np.array(eq_data[6:10], dtype=np.float64)
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        assert np.allclose(quat, expected, atol=1e-6) or np.allclose(quat, -expected, atol=1e-6)
        # we defined no regular equality constraints, so there is no mapping from MuJoCo to Newton equality constraints
        assert np.allclose(solver.mjc_eq_to_newton_eq.numpy(), np.full_like(solver.mjc_eq_to_newton_eq.numpy(), -1))
        # WELD entries from FIXED loop joints are excluded from mjc_eq_to_newton_jnt
        # (only CONNECT entries are tracked) so the mapping should be all -1
        assert np.allclose(solver.mjc_eq_to_newton_jnt.numpy(), np.full_like(solver.mjc_eq_to_newton_jnt.numpy(), -1))

    def test_mixed_loop_joints_and_equality_constraints(self):
        """Testing that loop joints and regular equality constraints are converted to equality constraints."""
        builder = newton.ModelBuilder()
        b0 = builder.add_link(mass=0.01)
        b1 = builder.add_link(mass=0.01)
        b2 = builder.add_link(mass=0.01)
        j0 = builder.add_joint_revolute(-1, b0)
        j1 = builder.add_joint_revolute(-1, b1)
        j2 = builder.add_joint_revolute(b1, b2)
        builder.add_articulation([j0, j1, j2])
        # add one equality constraint before the loop joint
        _add_equality_constraint(
            builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT,
            body1=b0,
            body2=b1,
            anchor=wp.vec3(0.0, 0.0, 1.0),
        )
        # add a loop joint
        builder.add_joint_fixed(
            b0,
            b2,
            # note these offset transforms here are important to ensure valid anchor points for the equality constraints are used
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.45), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.45), wp.quat_identity()),
        )
        # add one equality constraint after the loop joint
        _add_equality_constraint(
            builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT,
            body1=b0,
            body2=b2,
            anchor=wp.vec3(0.0, 0.0, 1.0),
        )
        world_count = 4
        world_builder = newton.ModelBuilder()
        # force the ModelBuilder to correct zero mass/inertia values
        world_builder.bound_inertia = 0.01
        world_builder.bound_mass = 0.01
        world_builder.replicate(builder, world_count=world_count)
        model = world_builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertEqual(model.joint_count, 4 * world_count)
        self.assertEqual(model.mujoco.equality_constraint_count, 2 * world_count)
        self.assertEqual(solver.mj_model.nv, 3)
        # 2 explicit connect constraints + 1 weld from fixed loop joint
        self.assertEqual(solver.mj_model.neq, 3)
        eq_type_connect = int(solver._mujoco.mjtEq.mjEQ_CONNECT)
        eq_type_weld = int(solver._mujoco.mjtEq.mjEQ_WELD)
        assert np.allclose(solver.mj_model.eq_type, [eq_type_connect, eq_type_connect, eq_type_weld])
        # the two equality constraints we explicitly created are defined first in MuJoCo
        expected_eq_to_newton_eq = np.full((world_count, 3), -1, dtype=np.int32)
        for i in range(world_count):
            expected_eq_to_newton_eq[i, 0] = i * 2
            expected_eq_to_newton_eq[i, 1] = i * 2 + 1
        assert np.allclose(solver.mjc_eq_to_newton_eq.numpy(), expected_eq_to_newton_eq)
        # WELD entries from FIXED loop joints are excluded from mjc_eq_to_newton_jnt
        # (only CONNECT entries are tracked) so the weld slot should remain -1
        assert np.allclose(solver.mjc_eq_to_newton_jnt.numpy(), np.full((world_count, 3), -1, dtype=np.int32))

    def test_loop_connect_mapping_tiles_template_joint_indices(self):
        """Generic CONNECT loop-joint mappings tile template joint indices across worlds."""
        builder = newton.ModelBuilder()
        inertia = wp.mat33(np.eye(3))
        b0 = builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=inertia)
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=inertia)
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=inertia)
        j0 = builder.add_joint_revolute(-1, b0, axis=(0.0, 0.0, 1.0))
        j1 = builder.add_joint_revolute(b0, b1, axis=(0.0, 1.0, 0.0))
        j2 = builder.add_joint_revolute(b1, b2, axis=(1.0, 0.0, 0.0))
        builder.add_articulation([j0, j1, j2])
        builder.add_joint_revolute(b2, b0, axis=(0.0, 0.0, 1.0))

        world_count = 3
        world_builder = newton.ModelBuilder()
        world_builder.replicate(builder, world_count=world_count)
        model = world_builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=True, disable_contacts=True)

        self.assertEqual(solver.mj_model.neq, 2)
        expected = np.array([[3, 3], [7, 7], [11, 11]], dtype=np.int32)
        np.testing.assert_array_equal(solver.mjc_eq_to_newton_jnt.numpy(), expected)

    def test_mjc_equality_target_remaps_in_add_builder(self):
        """The polymorphic MuJoCo equality target remaps joint and mimic indices."""
        main = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(main)
        inertia = wp.mat33(np.eye(3))
        b0 = main.add_link(mass=1.0, com=wp.vec3(0.0), inertia=inertia)
        b1 = main.add_link(mass=1.0, com=wp.vec3(0.0), inertia=inertia)
        j0 = main.add_joint_revolute(-1, b0)
        j1 = main.add_joint_revolute(b0, b1)
        main.add_articulation([j0, j1])
        _add_deprecated_mimic_constraint(main, joint0=j1, joint1=j0)

        source = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(source)
        s0 = source.add_link(mass=1.0, com=wp.vec3(0.0), inertia=inertia)
        s1 = source.add_link(mass=1.0, com=wp.vec3(0.0), inertia=inertia)
        sj0 = source.add_joint_revolute(-1, s0)
        sj1 = source.add_joint_revolute(s0, s1)
        source.add_articulation([sj0, sj1])
        sm = _add_deprecated_mimic_constraint(source, joint0=sj1, joint1=sj0)
        _add_equality_constraint(
            source,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT,
            body1=s0,
            body2=s1,
            anchor=wp.vec3(0.0),
            custom_attributes={
                "mujoco:equality_constraint_target_kind": int(MjcEqualityTargetKind.JOINT),
                "mujoco:equality_constraint_target": sj0,
                "mujoco:equality_constraint_objtype": MJC_OBJ_BODY,
            },
        )
        _add_equality_constraint(
            source,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.JOINT,
            joint1=sj1,
            joint2=sj0,
            custom_attributes={
                "mujoco:equality_constraint_target_kind": int(MjcEqualityTargetKind.MIMIC),
                "mujoco:equality_constraint_target": sm,
                "mujoco:equality_constraint_objtype": MJC_OBJ_JOINT,
            },
        )

        joint_offset = main.joint_count
        mimic_offset = len(main.constraint_mimic_joint0)
        main.add_builder(source)
        model = main.finalize()

        np.testing.assert_array_equal(
            model.mujoco.equality_constraint_target_kind.numpy(),
            np.array([int(MjcEqualityTargetKind.JOINT), int(MjcEqualityTargetKind.MIMIC)], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            model.mujoco.equality_constraint_target.numpy(),
            np.array([joint_offset + sj0, mimic_offset + sm], dtype=np.int32),
        )

    def test_mjc_equality_target_invalidates_removed_projected_joint(self):
        """collapse_fixed_joints clears targets for projected joints that are removed."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        inertia = wp.mat33(np.eye(3))
        root = builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=inertia)
        child = builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=inertia)
        root_joint = builder.add_joint_revolute(-1, root)
        fixed_joint = builder.add_joint_fixed(root, child)
        builder.add_articulation([root_joint, fixed_joint])
        _add_equality_constraint(
            builder,
            constraint_type=newton.solvers.SolverMuJoCo.EqType.WELD,
            body1=root,
            body2=child,
            custom_attributes={
                "mujoco:equality_constraint_target_kind": int(MjcEqualityTargetKind.JOINT),
                "mujoco:equality_constraint_target": fixed_joint,
                "mujoco:equality_constraint_objtype": MJC_OBJ_BODY,
            },
        )

        builder.collapse_fixed_joints()

        target_kind = builder.custom_attributes["mujoco:equality_constraint_target_kind"].values[0]
        target = builder.custom_attributes["mujoco:equality_constraint_target"].values[0]
        self.assertEqual(target_kind, int(MjcEqualityTargetKind.NONE))
        self.assertEqual(target, -1)

    def test_preserved_loop_equality_runtime_update_keeps_per_world_mujoco_data(self):
        """Preserved MuJoCo CONNECT/WELD loop equalities keep per-world metadata."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        connect_anchor = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]], dtype=np.float32)
        weld_anchor = np.array([[0.7, -0.8, 0.9], [-1.0, 1.1, -1.2]], dtype=np.float32)
        weld_relpose_pos = np.array([[0.01, 0.02, 0.03], [-0.04, 0.05, -0.06]], dtype=np.float32)
        weld_torquescale = np.array([0.37, 0.83], dtype=np.float32)
        connect_solref = np.array([[0.04, 1.2], [0.08, 0.9]], dtype=np.float32)
        weld_solref = np.array([[0.03, 1.4], [0.07, 0.6]], dtype=np.float32)
        connect_solimp = np.array(
            [[0.82, 0.91, 0.002, 0.45, 1.8], [0.75, 0.88, 0.004, 0.55, 1.6]],
            dtype=np.float32,
        )
        weld_solimp = np.array(
            [[0.66, 0.86, 0.006, 0.35, 1.4], [0.72, 0.9, 0.008, 0.65, 2.2]],
            dtype=np.float32,
        )
        connect_enabled = np.array([True, False])
        weld_enabled = np.array([False, True])

        for world in range(2):
            builder.begin_world()
            root = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
            mid = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
            tip = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
            j0 = builder.add_joint_revolute(-1, root, axis=(0.0, 0.0, 1.0))
            j1 = builder.add_joint_revolute(root, mid, axis=(0.0, 1.0, 0.0))
            j2 = builder.add_joint_revolute(mid, tip, axis=(1.0, 0.0, 0.0))
            builder.add_shape_box(body=root, hx=0.1, hy=0.1, hz=0.1)
            builder.add_shape_box(body=mid, hx=0.1, hy=0.1, hz=0.1)
            builder.add_shape_box(body=tip, hx=0.1, hy=0.1, hz=0.1)
            builder.add_articulation([j0, j1, j2])

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*undefined semantics.*", category=UserWarning)
                connect_joint = builder.add_joint_ball(
                    parent=root,
                    child=mid,
                    enabled=bool(connect_enabled[world]),
                )
            _add_equality_constraint(
                builder,
                constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT,
                body1=root,
                body2=mid,
                anchor=wp.vec3(*connect_anchor[world]),
                enabled=bool(connect_enabled[world]),
                custom_attributes={
                    "mujoco:eq_solref": wp.vec2(*connect_solref[world]),
                    "mujoco:eq_solimp": vec5(*connect_solimp[world]),
                    "mujoco:equality_constraint_target_kind": int(MjcEqualityTargetKind.JOINT),
                    "mujoco:equality_constraint_target": connect_joint,
                    "mujoco:equality_constraint_objtype": MJC_OBJ_BODY,
                },
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*undefined semantics.*", category=UserWarning)
                weld_joint = builder.add_joint_fixed(
                    parent=mid,
                    child=tip,
                    enabled=bool(weld_enabled[world]),
                )
            _add_equality_constraint(
                builder,
                constraint_type=newton.solvers.SolverMuJoCo.EqType.WELD,
                body1=mid,
                body2=tip,
                anchor=wp.vec3(*weld_anchor[world]),
                relpose=wp.transform(
                    wp.vec3(*weld_relpose_pos[world]),
                    wp.quat_identity(),
                ),
                torquescale=float(weld_torquescale[world]),
                enabled=bool(weld_enabled[world]),
                custom_attributes={
                    "mujoco:eq_solref": wp.vec2(*weld_solref[world]),
                    "mujoco:eq_solimp": vec5(*weld_solimp[world]),
                    "mujoco:equality_constraint_target_kind": int(MjcEqualityTargetKind.JOINT),
                    "mujoco:equality_constraint_target": weld_joint,
                    "mujoco:equality_constraint_objtype": MJC_OBJ_BODY,
                },
            )
            builder.end_world()

        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        self.assertEqual(solver.mj_model.neq, 2)
        self.assertEqual(int(solver.mj_model.eq_type[0]), int(solver._mujoco.mjtEq.mjEQ_CONNECT))
        self.assertEqual(int(solver.mj_model.eq_type[1]), int(solver._mujoco.mjtEq.mjEQ_WELD))
        np.testing.assert_array_equal(solver.mjc_eq_to_newton_eq.numpy(), np.array([[0, 1], [2, 3]]))
        np.testing.assert_array_equal(solver.mjc_eq_to_newton_jnt.numpy(), np.full((2, 2), -1, dtype=np.int32))

        eq_data = solver.mjw_model.eq_data.numpy()
        np.testing.assert_allclose(eq_data[:, 0, :3], connect_anchor, rtol=1e-5)
        np.testing.assert_allclose(eq_data[:, 1, :3], weld_anchor, rtol=1e-5)
        np.testing.assert_allclose(eq_data[:, 1, 3:6], weld_relpose_pos, rtol=1e-5)
        np.testing.assert_allclose(eq_data[:, 1, 6:10], [[1.0, 0.0, 0.0, 0.0]] * 2, rtol=1e-5)
        np.testing.assert_allclose(eq_data[:, 1, 10], weld_torquescale, rtol=1e-5)
        np.testing.assert_allclose(solver.mjw_model.eq_solref.numpy()[:, 0], connect_solref, rtol=1e-5)
        np.testing.assert_allclose(solver.mjw_model.eq_solref.numpy()[:, 1], weld_solref, rtol=1e-5)
        np.testing.assert_allclose(solver.mjw_model.eq_solimp.numpy()[:, 0], connect_solimp, rtol=1e-5)
        np.testing.assert_allclose(solver.mjw_model.eq_solimp.numpy()[:, 1], weld_solimp, rtol=1e-5)
        np.testing.assert_array_equal(solver.mjw_data.eq_active.numpy()[:, 0], connect_enabled)
        np.testing.assert_array_equal(solver.mjw_data.eq_active.numpy()[:, 1], weld_enabled)

        eq_enabled = model.mujoco.equality_constraint_enabled.numpy()
        eq_enabled[0] = False
        eq_enabled[1] = True
        eq_enabled[2] = True
        eq_enabled[3] = False
        model.mujoco.equality_constraint_enabled.assign(eq_enabled)
        solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)

        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy(), eq_data, rtol=1e-5)
        np.testing.assert_array_equal(solver.mjw_data.eq_active.numpy()[:, 0], [False, True])
        np.testing.assert_array_equal(solver.mjw_data.eq_active.numpy()[:, 1], [True, False])

    def test_loop_joint_coordinate_conversion_offset(self):
        """Verify coordinate conversion when revolute loop joints precede other joints.

        When a revolute loop joint (articulation=-1) is added before a free body,
        its DOF creates an offset between Newton's joint_q_start and MuJoCo's
        jnt_qposadr. The conversion must handle this correctly so that the
        free body's coordinates are not corrupted.
        """
        builder = newton.ModelBuilder()

        # 3-link articulation: b1 offset from b0 by (0, 0, 1), with b2
        # colocated with b1, so the loop
        # joint can use asymmetric local anchors that coincide in world space.
        b0 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j0 = builder.add_joint_revolute(-1, b0, axis=(0, 0, 1))
        j1 = builder.add_joint_revolute(
            b0,
            b1,
            axis=(0, 0, 1),
            parent_xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()),
        )
        j2 = builder.add_joint_revolute(b1, b2, axis=(0, 1, 0))
        builder.add_articulation([j0, j1, j2])

        # Revolute loop joint BEFORE the free body — creates q_start offset.
        # Asymmetric anchors: (0, 0, -0.5) on b2 at (0,0,1) → world (0, 0, 0.5)
        #                     (0, 0,  0.5) on b0 at origin   → world (0, 0, 0.5)
        loop_j = builder.add_joint_revolute(
            b2,
            b0,
            parent_xform=wp.transform(wp.vec3(0, 0, -0.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
            axis=(0, 0, 1),
        )
        builder.joint_articulation[loop_j] = -1

        # Free body added AFTER loop joint — its Newton q_start will be offset
        # from MuJoCo's jnt_qposadr by the loop joint's DOF count.
        # Use a distinctive non-zero position AND non-identity quaternion so the
        # roundtrip cannot succeed by accident (e.g. all-zero loop joint q
        # coincidentally matching a default pose).
        free_pos = wp.vec3(2.0, 3.0, 1.0)
        free_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 4.0)
        b_free = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), xform=wp.transform(free_pos, free_rot))

        world_builder = newton.ModelBuilder()
        world_builder.replicate(builder, world_count=1)
        model = world_builder.finalize()

        solver = SolverMuJoCo(model)

        # Revolute loop joint → 2x connect constraints (origin + offset along axis)
        self.assertEqual(solver.mj_model.neq, 2)
        self.assertEqual(int(solver.mj_model.eq_type[0]), int(solver._mujoco.mjtEq.mjEQ_CONNECT))
        self.assertEqual(int(solver.mj_model.eq_type[1]), int(solver._mujoco.mjtEq.mjEQ_CONNECT))
        # First CONNECT: parent anchor on b2 at (0, 0, -0.5)
        assert np.allclose(solver.mj_model.eq_data[0, 0:3], [0, 0, -0.5], atol=1e-6)
        # MuJoCo auto-computes child anchor from body positions at compile time:
        # world point = b2_pos + (0,0,-0.5) = (0,0,1) + (0,0,-0.5) = (0,0,0.5)
        # in b0 frame: (0,0,0.5) - b0_pos = (0,0,0.5)
        assert np.allclose(solver.mj_model.eq_data[0, 3:6], [0, 0, 0.5], atol=1e-6)
        # Second CONNECT: offset along hinge axis (0, 0, 1) by 0.1
        assert np.allclose(solver.mj_model.eq_data[1, 0:3], [0, 0, -0.4], atol=1e-6)

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        # Record the free body's initial position
        body_q_before = state.body_q.numpy().copy()

        # Sanity-check: the free body must have the non-default pose we set,
        # otherwise the roundtrip comparison below is meaningless.
        assert not np.allclose(body_q_before[b_free, 0:3], 0.0, atol=0.1)
        assert not np.allclose(body_q_before[b_free, 3:7], [0, 0, 0, 1], atol=0.1)

        # Round-trip: Newton → MuJoCo → Newton
        solver._update_mjc_data(solver.mjw_data, model, state)
        solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)
        solver._update_newton_state(model, state, solver.mjw_data, state_prev=state)

        body_q_after = state.body_q.numpy()

        # The free body's position must survive the round trip
        assert np.allclose(body_q_after[b_free, 0:3], body_q_before[b_free, 0:3], atol=1e-3), (
            f"Free body position corrupted: {body_q_after[b_free, 0:3]} vs {body_q_before[b_free, 0:3]}"
        )

        # Quaternion check (sign-invariant)
        q_before = body_q_before[b_free, 3:7]
        q_after = body_q_after[b_free, 3:7]
        quat_dist = min(np.linalg.norm(q_after - q_before), np.linalg.norm(q_after + q_before))
        self.assertLess(quat_dist, 1e-3, "Free body orientation corrupted by loop joint q_start offset")

    def test_ball_loop_joint_coordinate_conversion_offset(self):
        """Verify coordinate conversion when a ball loop joint precedes other joints.

        A ball joint has 4 q DOFs (quaternion) and 3 qd DOFs in Newton,
        but 0 in MuJoCo (becomes connect constraint). This creates a larger
        offset than revolute (4 vs 1) and tests the q != qd dimension case.
        """
        builder = newton.ModelBuilder()

        # 3-link articulation
        b0 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j0 = builder.add_joint_revolute(-1, b0, axis=(0, 0, 1))
        j1 = builder.add_joint_revolute(b0, b1, axis=(0, 0, 1))
        j2 = builder.add_joint_revolute(b1, b2, axis=(0, 1, 0))
        builder.add_articulation([j0, j1, j2])

        # Ball loop joint BEFORE the free body — creates 4q/3qd offset
        loop_j = builder.add_joint_ball(
            b2,
            b0,
            parent_xform=wp.transform(wp.vec3(0, 0, -0.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0, 0, -0.5), wp.quat_identity()),
        )
        builder.joint_articulation[loop_j] = -1

        # Free body added AFTER loop joint — use distinctive pose (see revolute variant).
        free_pos = wp.vec3(2.0, 3.0, 1.0)
        free_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 4.0)
        b_free = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), xform=wp.transform(free_pos, free_rot))

        world_builder = newton.ModelBuilder()
        world_builder.replicate(builder, world_count=1)
        model = world_builder.finalize()

        solver = SolverMuJoCo(model)

        # Ball loop joint → single connect constraint
        self.assertEqual(solver.mj_model.neq, 1)
        self.assertEqual(int(solver.mj_model.eq_type[0]), int(solver._mujoco.mjtEq.mjEQ_CONNECT))

        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        body_q_before = state.body_q.numpy().copy()

        # Sanity-check: pose must be non-default
        assert not np.allclose(body_q_before[b_free, 0:3], 0.0, atol=0.1)
        assert not np.allclose(body_q_before[b_free, 3:7], [0, 0, 0, 1], atol=0.1)

        # Round-trip: Newton → MuJoCo → Newton
        solver._update_mjc_data(solver.mjw_data, model, state)
        solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)
        solver._update_newton_state(model, state, solver.mjw_data, state_prev=state)

        body_q_after = state.body_q.numpy()

        assert np.allclose(body_q_after[b_free, 0:3], body_q_before[b_free, 0:3], atol=1e-3), (
            f"Free body position corrupted: {body_q_after[b_free, 0:3]} vs {body_q_before[b_free, 0:3]}"
        )

        q_before = body_q_before[b_free, 3:7]
        q_after = body_q_after[b_free, 3:7]
        quat_dist = min(np.linalg.norm(q_after - q_before), np.linalg.norm(q_after + q_before))
        self.assertLess(quat_dist, 1e-3, "Free body orientation corrupted by ball loop joint q_start offset")

    def test_ball_joint_fk_rotated_child_anchor(self):
        """Newton, MuJoCo, and mujoco_warp FK agree on the child body's world pose and angular velocity for varying (joint_q, joint_qd) under non-identity child_xform rotation."""
        import mujoco

        child_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 6.0)
        # (joint_q, joint_qd) chosen so neither commutes with child_rot — exercises the full bridge.
        cases = {
            "rx90_wz": (wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0), wp.vec3(0.0, 0.0, 1.0)),
            "ry45_wx": (wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 4.0), wp.vec3(1.0, 0.0, 0.0)),
            "compound": (
                wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 1.0)), math.pi / 3.0),
                wp.vec3(0.7, 0.0, -0.5),
            ),
        }

        builder = newton.ModelBuilder()
        child = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        ball_j = builder.add_joint_ball(
            -1,
            child,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(0.1, 0.2, 0.0), child_rot),
        )
        builder.add_articulation([ball_j])
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        q_start = int(model.joint_q_start.numpy()[ball_j])
        qd_start = int(model.joint_qd_start.numpy()[ball_j])
        mj_body_id = solver.mj_model.body(model.body_label[child].replace("/", "_")).id

        def quat_err(a, b):  # quaternion sign ambiguity: q and -q are the same rotation
            return min(np.linalg.norm(a[3:] - b[3:]), np.linalg.norm(a[3:] + b[3:]))

        for name, (r, w) in cases.items():
            with self.subTest(case=name):
                state = model.state()
                q_np = state.joint_q.numpy().copy()
                q_np[q_start : q_start + 4] = [r[0], r[1], r[2], r[3]]
                wp.copy(state.joint_q, wp.array(q_np, dtype=wp.float32))
                qd_np = state.joint_qd.numpy().copy()
                qd_np[qd_start : qd_start + 3] = [w[0], w[1], w[2]]
                wp.copy(state.joint_qd, wp.array(qd_np, dtype=wp.float32))

                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                bq_newton = state.body_q.numpy()[child].copy()
                # body_qd convention: (v_com_world, omega_world).
                w_world_newton = state.body_qd.numpy()[child][3:6].copy()

                solver._update_mjc_data(solver.mj_data, model, state)
                mujoco.mj_forward(solver.mj_model, solver.mj_data)
                mj_pos = np.array(solver.mj_data.xpos[mj_body_id], dtype=np.float64)
                mj_quat_wxyz = np.array(solver.mj_data.xquat[mj_body_id], dtype=np.float64)
                bq_mj = np.concatenate([mj_pos, [mj_quat_wxyz[1], mj_quat_wxyz[2], mj_quat_wxyz[3], mj_quat_wxyz[0]]])
                vel = np.zeros(6, dtype=np.float64)
                mujoco.mj_objectVelocity(solver.mj_model, solver.mj_data, mujoco.mjtObj.mjOBJ_BODY, mj_body_id, vel, 0)
                # mj_objectVelocity with flg_local=0 returns [angular(3), linear(3)] in world.
                w_world_mj = vel[:3]

                solver._update_mjc_data(solver.mjw_data, model, state)
                solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)
                solver._update_newton_state(model, state, solver.mjw_data, state_prev=state)
                bq_mjw = state.body_q.numpy()[child]

                self.assertLess(
                    np.max(np.abs(bq_newton[:3] - bq_mj[:3])), 1e-5, f"newton vs mj pos: {bq_newton}, {bq_mj}"
                )
                self.assertLess(quat_err(bq_newton, bq_mj), 1e-5, f"newton vs mj quat: {bq_newton}, {bq_mj}")
                self.assertLess(
                    np.max(np.abs(bq_newton[:3] - bq_mjw[:3])), 1e-5, f"newton vs mjw pos: {bq_newton}, {bq_mjw}"
                )
                self.assertLess(quat_err(bq_newton, bq_mjw), 1e-5, f"newton vs mjw quat: {bq_newton}, {bq_mjw}")
                self.assertLess(
                    np.max(np.abs(w_world_newton - w_world_mj)),
                    1e-5,
                    f"newton vs mj omega: {w_world_newton}, {w_world_mj}",
                )

    def test_ball_joint_applied_torque_rotated_child_anchor(self):
        """control.joint_f drives both Newton's and MuJoCo's post-step world omega to tau*dt for varying (joint_q, torque) under non-identity child_xform rotation.

        Pushing state_out back through _update_mjc_data and reading MuJoCo's own mj_objectVelocity catches missing factors that would cancel between apply and readback in a Newton-only view.
        """
        import mujoco

        child_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 6.0)
        # (joint_q, torque) chosen so R(r) * tau != tau — exercises the full r^{-1} factor in the apply path.
        cases = {
            "rx90_tz": (wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0), wp.vec3(0.0, 0.0, 1.0)),
            "ry45_tx": (wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 4.0), wp.vec3(1.0, 0.0, 0.0)),
            "compound": (
                wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 1.0)), math.pi / 3.0),
                wp.vec3(0.7, 0.0, -0.5),
            ),
        }

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        child = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        ball_j = builder.add_joint_ball(
            -1,
            child,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), child_rot),
        )
        builder.add_articulation([ball_j])
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        q_start = int(model.joint_q_start.numpy()[ball_j])
        qd_start = int(model.joint_qd_start.numpy()[ball_j])
        mj_body_id = solver.mj_model.body(model.body_label[child].replace("/", "_")).id
        dt = 1.0e-4

        for name, (r, tau) in cases.items():
            with self.subTest(case=name):
                state_in = model.state()
                state_out = model.state()
                control = model.control()

                q_np = state_in.joint_q.numpy().copy()
                q_np[q_start : q_start + 4] = [r[0], r[1], r[2], r[3]]
                wp.copy(state_in.joint_q, wp.array(q_np, dtype=wp.float32))

                joint_f_np = control.joint_f.numpy().copy()
                joint_f_np[qd_start : qd_start + 3] = [tau[0], tau[1], tau[2]]
                wp.copy(control.joint_f, wp.array(joint_f_np, dtype=wp.float32))

                solver.step(state_in, state_out, control, None, dt)
                omega_world_newton = state_out.body_qd.numpy()[child][3:6]

                solver._update_mjc_data(solver.mj_data, model, state_out)
                mujoco.mj_forward(solver.mj_model, solver.mj_data)
                vel = np.zeros(6, dtype=np.float64)
                mujoco.mj_objectVelocity(solver.mj_model, solver.mj_data, mujoco.mjtObj.mjOBJ_BODY, mj_body_id, vel, 0)
                omega_world_mj = vel[:3]

                expected = np.array([tau[0] * dt, tau[1] * dt, tau[2] * dt])
                self.assertLess(
                    np.max(np.abs(omega_world_mj - expected)), 1e-6, f"mj={omega_world_mj}, expected={expected}"
                )
                self.assertLess(
                    np.max(np.abs(omega_world_newton - omega_world_mj)),
                    1e-6,
                    f"newton={omega_world_newton}, mj={omega_world_mj}",
                )

    def test_ball_joint_actuator_readback_rotated_child_anchor(self):
        """convert_qfrc_actuator_from_mj_kernel BALL maps mj_data.qfrc_actuator to state.mujoco.qfrc_actuator in Newton's anchor frame for varying joint_q under non-identity child_xform rotation."""
        child_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 6.0)
        rx90 = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2.0)
        ry45 = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 4.0)
        compound = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 1.0)), math.pi / 3.0)
        cases = {
            "rx90_tx": (rx90, wp.vec3(1.0, 0.0, 0.0)),
            "ry45_tz": (ry45, wp.vec3(0.0, 0.0, 1.0)),
            "compound": (compound, wp.vec3(0.3, -0.5, 0.7)),
        }

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        child = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        ball_j = builder.add_joint_ball(
            -1,
            child,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), child_rot),
        )
        builder.add_articulation([ball_j])
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        observables = solver.observables({solver.ObservableFlags.QFRC_ACTUATOR})

        q_start = int(model.joint_q_start.numpy()[ball_j])
        qd_start = int(model.joint_qd_start.numpy()[ball_j])

        def qmul(a, b):
            ax, ay, az, aw = a
            bx, by, bz, bw = b
            return np.array(
                [
                    aw * bx + ax * bw + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw,
                    aw * bw - ax * bx - ay * by - az * bz,
                ]
            )

        def qinv(q):
            return np.array([-q[0], -q[1], -q[2], q[3]])

        def qrot(q, v):
            return qmul(qmul(q, np.array([v[0], v[1], v[2], 0.0])), qinv(q))[:3]

        c = np.array([child_rot[0], child_rot[1], child_rot[2], child_rot[3]], dtype=np.float64)

        for name, (r, tau_mj) in cases.items():
            with self.subTest(case=name):
                state = model.state()
                q_np = state.joint_q.numpy().copy()
                q_np[q_start : q_start + 4] = [r[0], r[1], r[2], r[3]]
                wp.copy(state.joint_q, wp.array(q_np, dtype=wp.float32))

                solver._update_mjc_data(solver.mj_data, model, state)
                qfrc_np = np.array(solver.mj_data.qfrc_actuator, dtype=np.float32).copy()
                qfrc_np[qd_start : qd_start + 3] = [tau_mj[0], tau_mj[1], tau_mj[2]]
                solver.mj_data.qfrc_actuator[:] = qfrc_np

                solver._update_newton_state(model, state, solver.mj_data, state_prev=state, observables=observables)

                # Expected: tau_newton = R(c^{-1} * q_mj) * tau_mj, where q_mj = c * r * c^{-1}.
                r_np = np.array([r[0], r[1], r[2], r[3]], dtype=np.float64)
                tau_mj_np = np.array([tau_mj[0], tau_mj[1], tau_mj[2]], dtype=np.float64)
                q_mj = qmul(qmul(c, r_np), qinv(c))
                expected = qrot(qmul(qinv(c), q_mj), tau_mj_np)

                tau_newton = observables.qfrc_actuator.numpy()[qd_start : qd_start + 3]
                err = float(np.max(np.abs(tau_newton - expected)))
                self.assertLess(err, 1e-5, f"got={tau_newton}, expected={expected}, err={err:.2e}")
