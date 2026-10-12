# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton import BodyFlags, ModelFlags
from newton._src.solvers.mujoco.constants import (
    DEFAULT_LIMIT_SOLREF,
    KINEMATIC_ARMATURE,
    SOLREF_MODE_FORCE_SPACE,
    SOLREF_MODE_MJCF_DEFAULT,
    SOLREF_MODE_RAW,
)
from newton.solvers import SolverMuJoCo


class TestMuJoCoSolver(unittest.TestCase):
    def test_collision_coloring_uses_all_32_mujoco_mask_bits(self):
        """Verify that graph-color fallback uses bits 0 through 31 before degrading to MuJoCo defaults."""
        clique_size = 33
        isolated_shape_count = 24
        shape_count = clique_size + isolated_shape_count
        builder = newton.ModelBuilder()
        for i in range(shape_count):
            body = builder.add_link(label=f"body_{i}")
            builder.add_shape_sphere(body, radius=0.1, label=f"shape_{i}")
            joint = builder.add_joint_free(body)
            builder.add_articulation([joint])

        # The 33-shape clique requires 33 colors. Isolating another 24 shapes
        # creates more than 1,024 sparse exclusions, which also forces the
        # legacy fallback when the bounded mask compiler from #3714 is used.
        colliding_pairs = set(itertools.combinations(range(clique_size), 2))
        for pair in itertools.combinations(range(shape_count), 2):
            if pair not in colliding_pairs:
                builder.add_shape_collision_filter_pair(*pair)

        model = builder.finalize(device="cpu")
        colors = SolverMuJoCo._color_collision_shapes(model, np.arange(model.shape_count))
        np.testing.assert_array_equal(np.sort(np.unique(colors)), np.arange(33))

        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        expected_masks = {SolverMuJoCo._collision_color_masks(color) for color in range(33)}
        actual_masks = {
            (int(contype), int(conaffinity))
            for contype, conaffinity in zip(
                solver.mj_model.geom_contype,
                solver.mj_model.geom_conaffinity,
                strict=True,
            )
        }
        self.assertEqual(actual_masks, expected_masks)
        self.assertIn((-2147483648, 2147483647), actual_masks)
        self.assertIn((1, 1), actual_masks)

    def test_tolerance_options(self):
        """Test that tolerance and ls_tolerance options are properly set on the MuJoCo Warp model."""
        # Create minimal model with proper inertia
        builder = newton.ModelBuilder()
        link = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_revolute(-1, link)
        builder.add_articulation([joint])
        model = builder.finalize()

        # Test with custom tolerance and ls_tolerance values
        custom_tolerance = 1e-2
        custom_ls_tolerance = 0.001
        solver = SolverMuJoCo(model, tolerance=custom_tolerance, ls_tolerance=custom_ls_tolerance)

        # Check that values made it to the mjw_model
        self.assertAlmostEqual(
            float(solver.mjw_model.opt.tolerance.numpy()[0]),
            custom_tolerance,
            places=5,
            msg=f"tolerance should be {custom_tolerance}",
        )
        self.assertAlmostEqual(
            float(solver.mjw_model.opt.ls_tolerance.numpy()[0]),
            custom_ls_tolerance,
            places=5,
            msg=f"ls_tolerance should be {custom_ls_tolerance}",
        )


class TestMuJoCoSolverGraphCapture(unittest.TestCase):
    def test_raw_solreflimit_validation_is_deferred_during_capture(self):
        """Defer RAW solreflimit warnings until an eager update after capture and replay."""
        if wp.get_cuda_device_count() == 0:
            self.skipTest("CUDA graph capture requires a CUDA device")

        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the CUDA mempool allocator")

        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            SolverMuJoCo.register_custom_attributes(builder)
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_revolute(-1, body, limit_lower=-1.0, limit_upper=1.0)
            builder.add_articulation([joint])
            model = builder.finalize(device=device)
            model.mujoco.solreflimit_mode.fill_(SOLREF_MODE_RAW)
            model.mujoco.solreflimit.fill_(wp.vec2(0.02, 1.0))
            solver = SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True, iterations=1)

            model.mujoco.solreflimit.fill_(wp.vec2(-1.0, 1.0))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with wp.ScopedCapture(device=device) as capture:
                    solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
                wp.capture_launch(capture.graph)
            self.assertFalse(any("invalid components" in str(w.message) for w in caught))
            np.testing.assert_allclose(solver.mjw_model.jnt_solref.numpy()[0, 0], [-1.0, 1.0])

            # Capture leaves validation pending even for a different eager flag.
            with self.assertWarnsRegex(UserWarning, "invalid components"):
                solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
            self.assertFalse(any("invalid components" in str(w.message) for w in caught))

            # Replay cannot change the Python guard; an eager DOF notification
            # must still validate values edited after the preceding warning.
            model.mujoco.solreflimit.fill_(wp.vec2(0.0, 1.0))
            wp.capture_launch(capture.graph)
            with self.assertWarnsRegex(UserWarning, "invalid components"):
                solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
            model.mujoco.solreflimit.fill_(wp.vec2(0.03, 0.7))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
            self.assertFalse(any("invalid components" in str(w.message) for w in caught))

    def test_loop_connect_updates_are_cuda_graph_capture_safe(self):
        """Match eager loop-joint CONNECT anchors after per-world graph-replayed edits."""
        if wp.get_cuda_device_count() == 0:
            self.skipTest("CUDA graph capture requires a CUDA device")

        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the CUDA mempool allocator")

        with wp.ScopedDevice(device):
            template = newton.ModelBuilder()
            bodies = [template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(3)]
            joints = [
                template.add_joint_revolute(-1, bodies[0]),
                template.add_joint_revolute(bodies[0], bodies[1]),
                template.add_joint_revolute(bodies[1], bodies[2]),
            ]
            template.add_articulation(joints)
            loop_joint = template.add_joint_revolute(bodies[2], bodies[0], axis=(0.0, 0.0, 1.0))
            builder = newton.ModelBuilder()
            builder.replicate(template, 2)
            model = builder.finalize(device=device)
            solver = SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True, iterations=1)
            eager_solver = SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True, iterations=1)
            self.assertTrue(solver.has_jnt_connect_constraints)
            self.assertEqual(solver.mj_model.neq, 2)
            initial_eq_data = solver.mjw_model.eq_data.numpy().copy()
            flags = ModelFlags.JOINT_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES
            with wp.ScopedCapture(device=device) as capture:
                solver.notify_model_changed(flags)

            loop_indices = loop_joint + np.arange(2) * template.joint_count
            axis_indices = model.joint_qd_start.numpy()[loop_indices]
            for factor in (1.0, 2.0):
                xforms = model.joint_X_p.numpy()
                xforms[loop_indices, :3] = factor * np.array([[0.1, 0.2, 0.3], [-0.2, 0.4, 0.1]])
                model.joint_X_p.assign(xforms)
                axes = model.joint_axis.numpy()
                axes[axis_indices] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                model.joint_axis.assign(axes)
                wp.capture_launch(capture.graph)
                eager_solver.notify_model_changed(flags)
                eq_data = solver.mjw_model.eq_data.numpy()
                np.testing.assert_allclose(eq_data, eager_solver.mjw_model.eq_data.numpy(), atol=1.0e-6)
                self.assertFalse(np.allclose(eq_data, initial_eq_data))
                self.assertFalse(np.allclose(eq_data[0], eq_data[1]))

    def test_cone_shape_updates_are_cuda_graph_capture_safe(self):
        """Replay shape updates with fixed-size cones and retain eager resize validation."""
        if wp.get_cuda_device_count() == 0:
            self.skipTest("CUDA graph capture requires a CUDA device")

        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the CUDA mempool allocator")

        with wp.ScopedDevice(device):
            template = newton.ModelBuilder()
            body = template.add_link()
            template.add_shape_cone(body, radius=0.25, half_height=0.5)
            template.add_shape_sphere(body, radius=0.1)
            joint = template.add_joint_revolute(parent=-1, child=body)
            template.add_articulation([joint])
            builder = newton.ModelBuilder()
            builder.replicate(template, 2)
            model = builder.finalize(device=device)
            solver = SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True, iterations=1)
            shape_map = solver.mjc_geom_to_newton_shape.numpy()
            cone_indices = np.flatnonzero(model.shape_type.numpy() == newton.GeoType.CONE)
            sphere_mask = model.shape_type.numpy()[shape_map] == newton.GeoType.SPHERE
            initial_scales = model.shape_scale.numpy().copy()
            mesh_vertices = solver.mjw_model.mesh_vert.numpy().copy()

            for flags in (ModelFlags.SHAPE_PROPERTIES, ModelFlags.ALL):
                with self.subTest(flags=flags):
                    solver.notify_model_changed(flags)
                    with wp.ScopedCapture(device=device) as capture:
                        solver.notify_model_changed(flags)

                    for factor in (1.5, 2.0):
                        friction = factor * np.arange(1, model.shape_count + 1, dtype=np.float32) / 10.0
                        scales = initial_scales.copy()
                        scales[model.shape_type.numpy() == newton.GeoType.SPHERE] *= factor
                        model.shape_material_mu.assign(friction)
                        model.shape_scale.assign(scales)
                        wp.capture_launch(capture.graph)
                        np.testing.assert_allclose(solver.mjw_model.geom_friction.numpy()[..., 0], friction[shape_map])
                        np.testing.assert_allclose(
                            solver.mjw_model.geom_size.numpy()[..., 0][sphere_mask],
                            scales[shape_map, 0][sphere_mask],
                        )
                        np.testing.assert_array_equal(solver.mjw_model.mesh_vert.numpy(), mesh_vertices)

                    # Capture must not disable subsequent eager validation, including
                    # edits in worlds other than the compiled template world.
                    scales[cone_indices[-1], 0] *= 2.0
                    model.shape_scale.assign(scales)
                    with self.assertRaisesRegex(ValueError, "Recreate the solver after resizing"):
                        solver.notify_model_changed(flags)
                    model.shape_scale.assign(initial_scales)
                    solver.notify_model_changed(flags)

    def test_joint_dof_updates_are_cuda_graph_capture_safe(self):
        """Replay joint friction, damping, and limit updates from a CUDA graph."""
        if wp.get_cuda_device_count() == 0:
            self.skipTest("CUDA graph capture requires a CUDA device")

        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the CUDA mempool allocator")

        with wp.ScopedDevice(device):
            for kinematic, flags in (
                (kinematic, flags)
                for kinematic in (False, True)
                for flags in (ModelFlags.JOINT_DOF_PROPERTIES, ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
            ):
                with self.subTest(kinematic=kinematic, flags=flags):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    SolverMuJoCo.register_custom_attributes(builder)
                    body = builder.add_link(
                        mass=1.0,
                        com=wp.vec3(0.0, 0.0, 0.0),
                        inertia=wp.mat33(np.eye(3)),
                    )
                    joint = builder.add_joint_revolute(
                        parent=-1,
                        child=body,
                        limit_lower=-1.0,
                        limit_upper=1.0,
                    )
                    builder.add_articulation([joint])
                    model = builder.finalize(device=device)
                    if kinematic:
                        model.body_flags.fill_(int(BodyFlags.KINEMATIC))
                    model.mujoco.solreflimit_mode.fill_(SOLREF_MODE_MJCF_DEFAULT)
                    initial_ke = model.joint_limit_ke.numpy().copy()
                    initial_kd = model.joint_limit_kd.numpy().copy()

                    solver = SolverMuJoCo(
                        model,
                        use_mujoco_cpu=False,
                        disable_contacts=True,
                        iterations=1,
                    )
                    with wp.ScopedCapture(device=device) as capture:
                        solver.notify_model_changed(flags)

                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(model.mujoco.solreflimit_mode.numpy(), SOLREF_MODE_MJCF_DEFAULT)
                    np.testing.assert_allclose(solver.mjw_model.jnt_solref.numpy()[0, 0], DEFAULT_LIMIT_SOLREF)

                    model.joint_friction.fill_(3.25)
                    model.joint_damping.fill_(4.5)
                    model.joint_limit_ke.fill_(1234.0)
                    model.joint_limit_kd.fill_(56.0)
                    wp.capture_launch(capture.graph)

                    np.testing.assert_allclose(solver.mjw_model.dof_frictionloss.numpy(), 3.25)
                    np.testing.assert_allclose(solver.mjw_model.dof_damping.numpy(), 4.5)
                    np.testing.assert_array_equal(
                        model.mujoco.solreflimit_mode.numpy(),
                        np.full(model.joint_dof_count, SOLREF_MODE_FORCE_SPACE, dtype=np.int32),
                    )

                    model.joint_limit_ke.assign(initial_ke)
                    model.joint_limit_kd.assign(initial_kd)
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(model.mujoco.solreflimit_mode.numpy(), SOLREF_MODE_FORCE_SPACE)

                    # An explicit mode reset starts a new baseline; even a one-ULP
                    # gain edit after that must promote on a later graph replay.
                    model.mujoco.solreflimit_mode.fill_(SOLREF_MODE_MJCF_DEFAULT)
                    model.joint_limit_ke.fill_(2345.0)
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(model.mujoco.solreflimit_mode.numpy(), SOLREF_MODE_MJCF_DEFAULT)
                    np.testing.assert_allclose(solver.mjw_model.jnt_solref.numpy()[0, 0], DEFAULT_LIMIT_SOLREF)
                    model.joint_limit_ke.fill_(np.nextafter(np.float32(2345.0), np.float32(np.inf)))
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(model.mujoco.solreflimit_mode.numpy(), SOLREF_MODE_FORCE_SPACE)

    def test_all_model_updates_are_cuda_graph_capture_safe(self):
        """Capture every GPU model-property branch and replay body-flag changes."""
        if wp.get_cuda_device_count() == 0:
            self.skipTest("CUDA graph capture requires a CUDA device")

        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the CUDA mempool allocator")

        mjcf = """
        <mujoco>
          <worldbody>
            <body name="a" pos="0 0 1">
              <joint name="ja" type="hinge" range="-1 1"/>
              <geom name="ga" type="capsule" size="0.05 0.2"/>
              <site name="sa" pos="0 0 0.2" size="0.02"/>
            </body>
            <body name="b" pos="0.4 0 1">
              <joint name="jb" type="hinge" range="-1 1"/>
              <geom name="gb" type="sphere" size="0.08"/>
            </body>
          </worldbody>
          <contact><pair geom1="ga" geom2="gb"/></contact>
          <equality><connect body1="a" body2="b" anchor="0.2 0 1"/></equality>
          <tendon><fixed name="t"><joint joint="ja" coef="1"/><joint joint="jb" coef="-1"/></fixed></tendon>
          <actuator><general name="u" joint="ja" gainprm="1"/></actuator>
        </mujoco>
        """

        world_count = 2

        def per_world(values, count):
            return np.repeat(np.asarray(values), count // world_count, axis=0)

        for enable_sleeping in (False, True):
            with self.subTest(enable_sleeping=enable_sleeping), wp.ScopedDevice(device):
                template = newton.ModelBuilder()
                template.add_mjcf(mjcf, ctrl_direct=True)
                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                builder.replicate(template, world_count)
                model = builder.finalize(device=device)
                self.assertEqual(model.world_count, world_count)
                model.joint_armature.fill_(0.25)
                # Distinct per-world tendon armature makes the mean inertia differ
                # between worlds and must be included like in ``set_const_0``.
                tendon_armature = np.array([0.5, 1.5], dtype=np.float32)
                model.mujoco.tendon_armature.assign(
                    np.repeat(tendon_armature, model.mujoco.tendon_armature.shape[0] // world_count)
                )
                solver = SolverMuJoCo(
                    model,
                    use_mujoco_cpu=False,
                    disable_contacts=True,
                    separate_worlds=True,
                    enable_sleeping=enable_sleeping,
                    iterations=1,
                )

                self.assertGreater(solver.mj_model.nsite, 0)
                self.assertGreater(solver.mj_model.npair, 0)
                self.assertGreater(solver.mj_model.neq, 0)
                self.assertGreater(solver.mj_model.ntendon, 0)
                self.assertGreater(solver.mj_model.nu, 0)
                physical_meaninertia = solver.mjw_model.stat.meaninertia.numpy().copy()
                # World 0 matches MuJoCo's compiled statistic. The fixed tendon has
                # unit coefficients on both DOFs, so its armature adds to each diagonal.
                self.assertAlmostEqual(physical_meaninertia[0], solver.mj_model.stat.meaninertia, places=5)
                self.assertAlmostEqual(
                    physical_meaninertia[1] - physical_meaninertia[0],
                    tendon_armature[1] - tendon_armature[0],
                    places=5,
                )

                # Compare every world against the upstream implementation to catch
                # drift in its mass-matrix assembly or mean-inertia reduction.
                solver._mujoco_warp.set_const_0(solver.mjw_model, solver.mjw_data)
                np.testing.assert_allclose(physical_meaninertia, solver.mjw_model.stat.meaninertia.numpy(), rtol=1.0e-5)

                with wp.ScopedCapture(device=device) as capture:
                    solver.notify_model_changed(ModelFlags.ALL)

                kinematic = int(BodyFlags.KINEMATIC)
                dynamic = int(BodyFlags.DYNAMIC)
                friction = np.array([3.25, 5.5], dtype=np.float32)
                damping = np.array([4.5, 6.75], dtype=np.float32)
                mu = np.array([0.75, 0.25], dtype=np.float32)
                eq_enabled = np.array([False, True])
                tendon_stiffness = np.array([123.0, 321.0], dtype=np.float32)
                gain = np.array([2.0, 7.0], dtype=np.float32)

                model.body_flags.assign(per_world([kinematic, dynamic], model.body_count).astype(np.int32))
                model.joint_friction.assign(per_world(friction, model.joint_dof_count))
                model.joint_damping.assign(per_world(damping, model.joint_dof_count))
                model.shape_material_mu.assign(per_world(mu, model.shape_count))
                model.mujoco.equality_constraint_enabled.assign(
                    per_world(eq_enabled, model.mujoco.equality_constraint_enabled.shape[0])
                )
                model.mujoco.tendon_stiffness.assign(
                    per_world(tendon_stiffness, model.mujoco.tendon_stiffness.shape[0])
                )
                gainprm = model.mujoco.actuator_gainprm.numpy()
                gainprm[:, 0] = per_world(gain, gainprm.shape[0])
                model.mujoco.actuator_gainprm.assign(gainprm)
                wp.capture_launch(capture.graph)

                dof_armature = solver.mjw_model.dof_armature.numpy()
                np.testing.assert_allclose(dof_armature[0], KINEMATIC_ARMATURE)
                np.testing.assert_allclose(dof_armature[1], 0.25)
                np.testing.assert_allclose(solver.mjw_model.stat.meaninertia.numpy(), physical_meaninertia, rtol=1.0e-5)
                dof_frictionloss = solver.mjw_model.dof_frictionloss.numpy()
                np.testing.assert_allclose(dof_frictionloss, np.broadcast_to(friction[:, None], dof_frictionloss.shape))
                dof_damping = solver.mjw_model.dof_damping.numpy()
                np.testing.assert_allclose(dof_damping, np.broadcast_to(damping[:, None], dof_damping.shape))
                geom_friction = solver.mjw_model.geom_friction.numpy()[..., 0]
                np.testing.assert_allclose(geom_friction, np.broadcast_to(mu[:, None], geom_friction.shape))
                eq_active = solver.mjw_data.eq_active.numpy()
                np.testing.assert_array_equal(eq_active, np.broadcast_to(eq_enabled[:, None], eq_active.shape))
                mjw_tendon_stiffness = solver.mjw_model.tendon_stiffness.numpy()
                np.testing.assert_allclose(
                    mjw_tendon_stiffness, np.broadcast_to(tendon_stiffness[:, None], mjw_tendon_stiffness.shape)
                )
                mjw_gain = solver.mjw_model.actuator_gainprm.numpy()[..., 0]
                np.testing.assert_allclose(mjw_gain, np.broadcast_to(gain[:, None], mjw_gain.shape))

                model.body_flags.assign(per_world([dynamic, kinematic], model.body_count).astype(np.int32))
                wp.capture_launch(capture.graph)
                dof_armature = solver.mjw_model.dof_armature.numpy()
                np.testing.assert_allclose(dof_armature[0], 0.25)
                np.testing.assert_allclose(dof_armature[1], KINEMATIC_ARMATURE)
                np.testing.assert_allclose(solver.mjw_model.stat.meaninertia.numpy(), physical_meaninertia, rtol=1.0e-5)
