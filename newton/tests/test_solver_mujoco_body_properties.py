# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp

import newton
from newton import BodyFlags, ModelFlags
from newton._src.solvers.mujoco.constants import KINEMATIC_ARMATURE
from newton.solvers import SolverMuJoCo
from newton.tests.utils.mujoco import _MuJoCoSolverPropertiesFixture


class TestMuJoCoSolverPropertiesSetup(_MuJoCoSolverPropertiesFixture, unittest.TestCase):
    """Tests of the shared property-test scene, run once rather than per property test class."""

    def _run_substeps_for_frame(self, sim_dt, sim_substeps):
        """Helper method to run simulation substeps for one rendered frame."""
        for _ in range(sim_substeps):
            self.solver.step(self.state_in, self.state_out, self.control, self.contacts, sim_dt)
            self.state_in, self.state_out = self.state_out, self.state_in  # Output becomes input for next substep

    def test_setup_completes(self):
        """Verify the shared scene builds two worlds with four bodies each."""
        self.assertEqual(self.model.world_count, 2)
        np.testing.assert_array_equal(np.bincount(self.model.body_world.numpy()), [4, 4])

    @unittest.skip("Trajectory rendering for debugging")
    def test_render_trajectory(self):
        """Simulates and renders a trajectory if solver and viewer are available."""
        print("\nDebug: Starting test_render_trajectory...")

        solver = None
        viewer = None
        substep_graph = None
        use_cuda_graph = wp.get_device().is_cuda

        try:
            print("Debug: Attempting to initialize SolverMuJoCo for trajectory test...")
            solver = SolverMuJoCo(self.model, iterations=10, ls_iterations=10)
            print("Debug: SolverMuJoCo initialized successfully for trajectory test.")
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping trajectory rendering: {e}")
        except Exception as e:
            self.skipTest(f"Error initializing SolverMuJoCo for trajectory test: {e}")

        if self.debug_stage_path:
            try:
                print("Debug: Attempting to initialize ViewerGL...")
                viewer = newton.viewer.ViewerGL()
                viewer.set_model(self.model)
                print("Debug: ViewerGL initialized successfully for trajectory test.")
            except ImportError as e:
                self.skipTest(f"ViewerGL dependencies not met. Skipping trajectory rendering: {e}")
            except Exception as e:
                self.skipTest(f"Error initializing ViewerGL for trajectory test: {e}")
        else:
            self.skipTest("No debug_stage_path set. Skipping trajectory rendering.")

        num_frames = 200
        sim_substeps = 2
        frame_dt = 1.0 / 60.0
        sim_dt = frame_dt / sim_substeps
        sim_time = 0.0

        self.solver = solver  # _run_substeps_for_frame reads self.solver

        if use_cuda_graph:
            print(
                f"Debug: CUDA device detected. Attempting to capture {sim_substeps} substeps with dt={sim_dt:.4f} into a CUDA graph..."
            )
            try:
                with wp.ScopedCapture() as capture:
                    self._run_substeps_for_frame(sim_dt, sim_substeps)
                substep_graph = capture.graph
                print("Debug: CUDA graph captured successfully.")
            except Exception as e:
                print(f"Debug: CUDA graph capture failed: {e}. Falling back to regular execution.")
                substep_graph = None
        else:
            print("Debug: Not using CUDA graph (non-CUDA device or flag disabled).")

        print(f"Debug: Simulating and rendering {num_frames} frames ({sim_substeps} substeps/frame)...")
        print("       Press Ctrl+C in the console to stop early.")

        try:
            for frame_num in range(num_frames):
                if frame_num % 20 == 0:
                    print(f"Debug: Frame {frame_num}/{num_frames}, Sim time: {sim_time:.2f}s")

                viewer.begin_frame(sim_time)
                viewer.log_state(self.state_in)
                viewer.end_frame()

                if use_cuda_graph and substep_graph:
                    wp.capture_launch(substep_graph)
                else:
                    self._run_substeps_for_frame(sim_dt, sim_substeps)

                sim_time += frame_dt
                time.sleep(0.016)

        except KeyboardInterrupt:
            print("\nDebug: Trajectory rendering stopped by user.")
        except Exception as e:
            self.fail(f"Error during trajectory rendering: {e}")
        finally:
            print("Debug: test_render_trajectory finished.")


class TestMuJoCoSolverMassProperties(_MuJoCoSolverPropertiesFixture, unittest.TestCase):
    def test_randomize_body_mass(self):
        """
        Tests if the body mass is randomized correctly and updated properly after simulation steps.
        """
        # Randomize masses for all bodies in all worlds
        new_masses = self.rng.uniform(1.0, 10.0, size=self.model.body_count)
        self.model.body_mass.assign(new_masses)

        # Initialize solver
        solver = SolverMuJoCo(self.model, ls_iterations=1, iterations=1, disable_contacts=True)

        # Check that masses were transferred correctly
        # Iterate over MuJoCo bodies and verify mapping
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()
        nworld = mjc_body_to_newton.shape[0]
        nbody = mjc_body_to_newton.shape[1]
        for world_idx in range(nworld):
            for mjc_body in range(nbody):
                newton_body = mjc_body_to_newton[world_idx, mjc_body]
                if newton_body >= 0:  # Skip unmapped bodies
                    self.assertAlmostEqual(
                        new_masses[newton_body],
                        solver.mjw_model.body_mass.numpy()[world_idx, mjc_body],
                        places=6,
                        msg=f"Mass mismatch for mjc_body {mjc_body} (newton {newton_body}) in world {world_idx}",
                    )

        # Run a simulation step
        solver.step(self.state_in, self.state_out, self.control, self.contacts, 0.01)
        self.state_in, self.state_out = self.state_out, self.state_in

        # Update masses again
        updated_masses = self.rng.uniform(1.0, 10.0, size=self.model.body_count)
        self.model.body_mass.assign(updated_masses)

        # Notify solver of mass changes
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        # Check that updated masses were transferred correctly
        for world_idx in range(nworld):
            for mjc_body in range(nbody):
                newton_body = mjc_body_to_newton[world_idx, mjc_body]
                if newton_body >= 0:  # Skip unmapped bodies
                    self.assertAlmostEqual(
                        updated_masses[newton_body],
                        solver.mjw_model.body_mass.numpy()[world_idx, mjc_body],
                        places=6,
                        msg=f"Updated mass mismatch for mjc_body {mjc_body} (newton {newton_body}) in world {world_idx}",
                    )

    def test_randomize_body_com(self):
        """
        Tests if the body center of mass is randomized correctly and updates properly after simulation steps.
        """
        # Randomize COM for all bodies in all worlds
        new_coms = self.rng.uniform(-1.0, 1.0, size=(self.model.body_count, 3))
        self.model.body_com.assign(new_coms)

        # Initialize solver
        solver = SolverMuJoCo(self.model, ls_iterations=1, iterations=1, disable_contacts=True, njmax=1)

        # Check that COM positions were transferred correctly
        # Iterate over MuJoCo bodies and verify mapping
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()
        nworld = mjc_body_to_newton.shape[0]
        nbody = mjc_body_to_newton.shape[1]
        for world_idx in range(nworld):
            for mjc_body in range(nbody):
                newton_body = mjc_body_to_newton[world_idx, mjc_body]
                if newton_body >= 0:  # Skip unmapped bodies
                    newton_pos = new_coms[newton_body]
                    mjc_pos = solver.mjw_model.body_ipos.numpy()[world_idx, mjc_body]

                    for dim in range(3):
                        self.assertAlmostEqual(
                            newton_pos[dim],
                            mjc_pos[dim],
                            places=6,
                            msg=f"COM position mismatch for mjc_body {mjc_body} (newton {newton_body}) in world {world_idx}, dimension {dim}",
                        )

        # Run a simulation step
        solver.step(self.state_in, self.state_out, self.control, self.contacts, 0.01)
        self.state_in, self.state_out = self.state_out, self.state_in

        # Update COM positions again
        updated_coms = self.rng.uniform(-1.0, 1.0, size=(self.model.body_count, 3))
        self.model.body_com.assign(updated_coms)

        # Notify solver of COM changes
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        # Check that updated COM positions were transferred correctly
        for world_idx in range(nworld):
            for mjc_body in range(nbody):
                newton_body = mjc_body_to_newton[world_idx, mjc_body]
                if newton_body >= 0:  # Skip unmapped bodies
                    newton_pos = updated_coms[newton_body]
                    mjc_pos = solver.mjw_model.body_ipos.numpy()[world_idx, mjc_body]

                    for dim in range(3):
                        self.assertAlmostEqual(
                            newton_pos[dim],
                            mjc_pos[dim],
                            places=6,
                            msg=f"Updated COM position mismatch for mjc_body {mjc_body} (newton {newton_body}) in world {world_idx}, dimension {dim}",
                        )

    def test_randomize_body_inertia(self):
        """
        Tests if the body inertia is randomized correctly.
        """
        # Randomize inertia tensors for all bodies in all worlds
        # Simple inertia tensors that satisfy triangle inequality

        def _make_spd_inertia(a_base, b_base, c_max):
            # Sample principal moments (triangle inequality on principal values)
            l1 = np.float32(a_base + self.rng.uniform(0.0, 0.5))
            l2 = np.float32(b_base + self.rng.uniform(0.0, 0.5))
            l3 = np.float32(min(l1 + l2 - 0.1, c_max))
            lam = np.array(sorted([l1, l2, l3], reverse=True), dtype=np.float32)

            # Random right-handed rotation
            Q, _ = np.linalg.qr(self.rng.normal(size=(3, 3)).astype(np.float32))
            if np.linalg.det(Q) < 0.0:
                Q[:, 2] *= -1.0

            inertia = (Q @ np.diag(lam) @ Q.T).astype(np.float32)
            return inertia

        new_inertias = np.zeros((self.model.body_count, 3, 3), dtype=np.float32)
        bodies_per_world = self.model.body_count // self.model.world_count
        for i in range(self.model.body_count):
            world_idx = i // bodies_per_world
            # Unified inertia generation for all worlds, parameterized by world_idx
            if world_idx == 0:
                a_base, b_base, c_max = 2.5, 3.5, 4.5
            else:
                a_base, b_base, c_max = 3.5, 4.5, 5.5

            new_inertias[i] = _make_spd_inertia(a_base, b_base, c_max)
        self.model.body_inertia.assign(new_inertias)

        # Initialize solver
        solver = SolverMuJoCo(self.model, iterations=1, ls_iterations=1, disable_contacts=True)

        # Get body mapping once outside the loop - iterate over MuJoCo bodies
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()
        nworld = mjc_body_to_newton.shape[0]
        nbody = mjc_body_to_newton.shape[1]

        def _quat_wxyz_to_rotmat(q):
            w, x, y, z = q
            return np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )

        def check_inertias(inertias_to_check, msg_prefix=""):
            for world_idx in range(nworld):
                for mjc_body in range(nbody):
                    newton_body = mjc_body_to_newton[world_idx, mjc_body]
                    if newton_body >= 0:  # Skip unmapped bodies
                        newton_inertia = inertias_to_check[newton_body].astype(np.float32)
                        mjc_principal = solver.mjw_model.body_inertia.numpy()[world_idx, mjc_body]
                        mjc_iquat = solver.mjw_model.body_iquat.numpy()[world_idx, mjc_body]  # wxyz

                        # Reconstruct full tensor from principal + iquat and compare
                        R = _quat_wxyz_to_rotmat(mjc_iquat)
                        reconstructed = R @ np.diag(mjc_principal) @ R.T

                        np.testing.assert_allclose(
                            reconstructed,
                            newton_inertia,
                            atol=1e-4,
                            err_msg=f"{msg_prefix}Inertia tensor mismatch for mjc_body {mjc_body} "
                            f"(newton {newton_body}) in world {world_idx}",
                        )

        # Check initial inertia tensors
        check_inertias(new_inertias, "Initial ")

        # Run a simulation step
        solver.step(self.state_in, self.state_out, self.control, self.contacts, 0.01)
        self.state_in, self.state_out = self.state_out, self.state_in

        # Update inertia tensors again with new random values
        updated_inertias = np.zeros((self.model.body_count, 3, 3), dtype=np.float32)
        for i in range(self.model.body_count):
            world_idx = i // bodies_per_world
            if world_idx == 0:
                a_base, b_base, c_max = 2.5, 3.5, 4.5
            else:
                a_base, b_base, c_max = 3.5, 4.5, 5.5
            updated_inertias[i] = _make_spd_inertia(a_base, b_base, c_max)
        self.model.body_inertia.assign(updated_inertias)

        # Notify solver of inertia changes
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        # Check updated inertia tensors
        check_inertias(updated_inertias, "Updated ")

    def test_body_inertia_eigendecomposition_determinant(self):
        """Verify eigendecomposition handles det=-1 and non-trivial rotations.

        The kernel must ensure det(V) = +1 before calling quat_from_matrix(),
        and convert the resulting xyzw quaternion to wxyz correctly.
        Uses a rotated (non-diagonal) inertia tensor to catch convention errors
        that would be invisible with axis-aligned inertia.
        """
        # Distinct eigenvalues with a non-trivial rotation to expose both
        # det=-1 handling and xyzw/wxyz convention errors.
        principal = np.array([0.06, 0.04, 0.02], dtype=np.float32)
        # 45-degree rotation around y-axis creates off-diagonal terms
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        rotated_inertia = (R @ np.diag(principal) @ R.T).astype(np.float32)

        # Assign this inertia to ALL bodies to ensure the mapped body gets it
        new_inertias = np.zeros((self.model.body_count, 3, 3), dtype=np.float32)
        for i in range(self.model.body_count):
            new_inertias[i] = rotated_inertia
        self.model.body_inertia.assign(new_inertias)

        # Initialize solver
        solver = SolverMuJoCo(self.model, iterations=1, ls_iterations=1, disable_contacts=True)

        # Get the mapping
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()

        # Helper to reconstruct full tensor from principal + iquat
        def quat_to_rotmat(q_wxyz):
            w, x, y, z = q_wxyz
            return np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )

        # Check that all mapped bodies have correct reconstructed inertia
        nworld = mjc_body_to_newton.shape[0]
        nbody = mjc_body_to_newton.shape[1]
        checked_count = 0

        for world_idx in range(nworld):
            for mjc_body in range(nbody):
                newton_body = mjc_body_to_newton[world_idx, mjc_body]
                if newton_body >= 0:
                    # Get principal moments and iquat
                    principal = solver.mjw_model.body_inertia.numpy()[world_idx, mjc_body]
                    iquat = solver.mjw_model.body_iquat.numpy()[world_idx, mjc_body]  # wxyz

                    # Reconstruct full tensor
                    R = quat_to_rotmat(iquat)
                    reconstructed = R @ np.diag(principal) @ R.T

                    # Compare to original (should match within tolerance)
                    np.testing.assert_allclose(
                        reconstructed,
                        rotated_inertia,
                        atol=1e-5,
                        err_msg=f"Reconstructed inertia tensor does not match original for "
                        f"mjc_body {mjc_body} (newton {newton_body}) in world {world_idx}",
                    )
                    checked_count += 1

        self.assertGreater(checked_count, 0, "No bodies were checked")

    def test_mujoco_cpu_inertia_frame_sync(self):
        """CPU MuJoCo must use the full MJWarp inertia representation."""
        mjcf = """
        <mujoco model="cpu_inertia_frame_sync">
          <compiler angle="radian" autolimits="true"/>
          <option gravity="0 0 0" timestep="0.001" integrator="Euler"/>
          <worldbody>
            <body name="base">
              <inertial pos="0 0 0" mass="8.7914" diaginertia="0.115841 0.082211 0.142951"/>
              <joint name="yaw" type="hinge" axis="0 0 1" armature="0.001" damping="0" frictionloss="0"/>
              <geom type="cylinder" size="0.25 0.08" mass="0"/>
            </body>
          </worldbody>
          <actuator>
            <velocity
              name="yaw_velocity"
              joint="yaw"
              kv="1000000"
              ctrllimited="true"
              ctrlrange="-4 4"
              forcelimited="true"
              forcerange="-600000 600000"/>
          </actuator>
        </mujoco>
        """

        mujoco, _ = SolverMuJoCo.import_mujoco()
        native_model = mujoco.MjModel.from_xml_string(mjcf)

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, ctrl_direct=True, ignore_inertial_definitions=False, parse_meshes=False)
        model = builder.finalize()
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True, integrator="euler")

        body_id = 1
        np.testing.assert_allclose(
            solver.mj_model.body_inertia[body_id],
            solver.mjw_model.body_inertia.numpy()[0, body_id],
            rtol=1e-7,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            solver.mj_model.body_iquat[body_id],
            solver.mjw_model.body_iquat.numpy()[0, body_id],
            rtol=1e-7,
            atol=1e-8,
        )
        self.assertEqual(
            int(solver.mj_model.body_sameframe[body_id]),
            int(mujoco.mjtSameFrame.mjSAMEFRAME_NONE),
        )
        self.assertEqual(int(solver.mj_model.body_simple[body_id]), 0)
        np.testing.assert_array_equal(solver.mj_model.dof_simplenum, np.zeros_like(solver.mj_model.dof_simplenum))
        np.testing.assert_allclose(solver.mj_model.dof_M0[0], native_model.dof_M0[0], rtol=1e-6, atol=1e-7)

        native_data = mujoco.MjData(native_model)
        native_data.ctrl[0] = 1.0
        mujoco.mj_step(native_model, native_data)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        control.mujoco.ctrl.assign([1.0])
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver.step(state_in, state_out, control, None, 0.001)

        np.testing.assert_allclose(solver.mj_data.qfrc_actuator[0], native_data.qfrc_actuator[0], rtol=1e-7)
        np.testing.assert_allclose(solver.mj_data.qvel[0], native_data.qvel[0], rtol=1e-6, atol=1e-7)

    def test_body_gravcomp(self):
        """
        Tests if the body gravity compensation is updated properly.
        """
        # Register custom attributes manually since setUp only creates basic builder
        newton.solvers.SolverMuJoCo.register_custom_attributes(self.builder)

        # Re-finalize model to allocate space for custom attributes
        # Note: The bodies are already added by _add_test_robot, so they have default gravcomp=0.0
        self.model = self.builder.finalize()

        # Verify attribute exists
        self.assertTrue(hasattr(self.model, "mujoco"))
        self.assertTrue(hasattr(self.model.mujoco, "gravcomp"))

        # Initialize deterministic gravcomp values based on index
        # Pattern: 0.1 + (i * 0.01) % 0.9
        indices = np.arange(self.model.body_count, dtype=np.float32)
        new_gravcomp = 0.1 + (indices * 0.01) % 0.9
        self.model.mujoco.gravcomp.assign(new_gravcomp)

        # Initialize solver
        solver = SolverMuJoCo(self.model, ls_iterations=1, iterations=1, disable_contacts=True)

        # Check initial values transferred to solver - iterate over MuJoCo bodies
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()
        nworld = mjc_body_to_newton.shape[0]
        nbody = mjc_body_to_newton.shape[1]

        for world_idx in range(nworld):
            for mjc_body in range(nbody):
                newton_body = mjc_body_to_newton[world_idx, mjc_body]
                if newton_body >= 0:
                    self.assertAlmostEqual(
                        new_gravcomp[newton_body],
                        solver.mjw_model.body_gravcomp.numpy()[world_idx, mjc_body],
                        places=6,
                        msg=f"Gravcomp mismatch for mjc_body {mjc_body} (newton {newton_body}) in world {world_idx}",
                    )

        # Step simulation
        solver.step(self.state_in, self.state_out, self.control, self.contacts, 0.01)
        self.state_in, self.state_out = self.state_out, self.state_in

        # Update gravcomp values (shift pattern)
        # Pattern: 0.9 - (i * 0.01) % 0.9
        updated_gravcomp = 0.9 - (indices * 0.01) % 0.9
        self.model.mujoco.gravcomp.assign(updated_gravcomp)

        # Notify solver
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        # Verify updates
        for world_idx in range(nworld):
            for mjc_body in range(nbody):
                newton_body = mjc_body_to_newton[world_idx, mjc_body]
                if newton_body >= 0:
                    self.assertAlmostEqual(
                        updated_gravcomp[newton_body],
                        solver.mjw_model.body_gravcomp.numpy()[world_idx, mjc_body],
                        places=6,
                        msg=f"Updated gravcomp mismatch for mjc_body {mjc_body} (newton {newton_body}) in world {world_idx}",
                    )

    def test_body_subtreemass_update(self):
        """
        Tests if body_subtreemass is correctly computed and updated after mass changes.

        body_subtreemass is a derived quantity that represents the total mass of a body
        and all its descendants in the kinematic tree. It is computed by set_const after
        mass updates.
        """
        # Initialize solver first to get the model structure
        solver = SolverMuJoCo(self.model, ls_iterations=1, iterations=1, disable_contacts=True)

        # Get body mapping - iterate over MuJoCo bodies
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()
        nworld = mjc_body_to_newton.shape[0]
        nbody = mjc_body_to_newton.shape[1]

        # Get initial subtreemass values
        initial_subtreemass = solver.mjw_model.body_subtreemass.numpy().copy()

        # Verify initial subtreemass values are reasonable (should be >= body_mass)
        for world_idx in range(nworld):
            for mjc_body in range(nbody):
                body_mass = solver.mjw_model.body_mass.numpy()[world_idx, mjc_body]
                subtree_mass = initial_subtreemass[world_idx, mjc_body]
                self.assertGreaterEqual(
                    subtree_mass,
                    body_mass - 1e-6,
                    msg=f"Initial subtreemass should be >= body_mass for mjc_body {mjc_body} in world {world_idx}",
                )

        # Update masses - double all masses
        new_masses = self.model.body_mass.numpy() * 2.0
        self.model.body_mass.assign(new_masses)

        # Notify solver of mass changes (this should call set_const to update subtreemass)
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        # Get updated subtreemass values
        updated_subtreemass = solver.mjw_model.body_subtreemass.numpy()

        # Verify subtreemass values are updated (should have roughly doubled for leaf bodies)
        # For the world body (0), subtreemass should be the sum of all body masses
        for world_idx in range(nworld):
            for mjc_body in range(nbody):
                newton_body = mjc_body_to_newton[world_idx, mjc_body]
                if newton_body >= 0:
                    # Subtreemass should have changed after mass update
                    old_subtree = initial_subtreemass[world_idx, mjc_body]
                    new_subtree = updated_subtreemass[world_idx, mjc_body]

                    # For leaf bodies (no children), subtreemass == body_mass
                    # so it should have doubled
                    new_body_mass = solver.mjw_model.body_mass.numpy()[world_idx, mjc_body]
                    self.assertGreaterEqual(
                        new_subtree,
                        new_body_mass - 1e-6,
                        msg=f"Updated subtreemass should be >= body_mass for mjc_body {mjc_body} in world {world_idx}",
                    )

                    # The subtreemass should be different from the initial value
                    # (unless it was originally 0, which shouldn't happen for real bodies)
                    if old_subtree > 1e-6:
                        self.assertNotAlmostEqual(
                            old_subtree,
                            new_subtree,
                            places=4,
                            msg=f"Subtreemass should have changed for mjc_body {mjc_body} in world {world_idx}",
                        )

    def test_derived_fields_updated_correctly(self):
        """
        Tests that derived fields (body_subtreemass, body_invweight0, dof_invweight0) are
        correctly computed after mass changes via Newton's interface.

        This verifies that set_const correctly computes derived quantities for all
        worlds and bodies. Since Newton's body_mass is per-body (not per-world),
        all worlds should have the same derived values.
        """
        # Initialize solver with multiple worlds
        solver = SolverMuJoCo(self.model, ls_iterations=1, iterations=1, disable_contacts=True)

        # Get dimensions
        nworld = self.model.world_count
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()
        nbody = mjc_body_to_newton.shape[1]
        nv = solver.mjw_model.nv

        # Randomize masses per-body through Newton's interface
        new_masses = np.zeros(self.model.body_count, dtype=np.float32)
        for body_idx in range(self.model.body_count):
            new_masses[body_idx] = 1.0 + 0.5 * body_idx  # Different mass per body

        self.model.body_mass.assign(new_masses)

        # Notify solver of mass changes (this calls set_const internally)
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        # Get derived fields (2D arrays: [nworld, nbody] or [nworld, nv])
        body_subtreemass = solver.mjw_model.body_subtreemass.numpy()
        body_invweight0 = solver.mjw_model.body_invweight0.numpy()
        dof_invweight0 = solver.mjw_model.dof_invweight0.numpy()
        mjw_body_mass = solver.mjw_model.body_mass.numpy()

        # Verify body_subtreemass is correctly computed for all worlds and bodies
        for world_idx in range(nworld):
            for mjc_body in range(nbody):
                newton_body = mjc_body_to_newton[world_idx, mjc_body]
                if newton_body >= 0:
                    body_mass = mjw_body_mass[world_idx, mjc_body]
                    subtree_mass = body_subtreemass[world_idx, mjc_body]

                    # subtreemass should be >= body_mass (includes mass of descendants)
                    self.assertGreaterEqual(
                        subtree_mass,
                        body_mass - 1e-6,
                        msg=f"body_subtreemass should be >= body_mass for world {world_idx}, body {mjc_body}",
                    )

        # Verify body_invweight0 is computed for all worlds and bodies
        for world_idx in range(nworld):
            for mjc_body in range(1, nbody):  # Skip world body 0
                newton_body = mjc_body_to_newton[world_idx, mjc_body]
                if newton_body >= 0:
                    # body_invweight0 is vec2 (trans, rot) - should be non-negative
                    invweight = body_invweight0[world_idx, mjc_body]
                    self.assertGreaterEqual(
                        invweight[0],
                        0.0,
                        msg=f"body_invweight0[0] should be >= 0 for world {world_idx}, body {mjc_body}",
                    )
                    self.assertGreaterEqual(
                        invweight[1],
                        0.0,
                        msg=f"body_invweight0[1] should be >= 0 for world {world_idx}, body {mjc_body}",
                    )

        # Verify dof_invweight0 is computed for all worlds and DOFs
        for world_idx in range(nworld):
            for dof_idx in range(nv):
                invweight = dof_invweight0[world_idx, dof_idx]
                # dof_invweight0 should be non-negative
                self.assertGreaterEqual(
                    invweight,
                    0.0,
                    msg=f"dof_invweight0 should be >= 0 for world {world_idx}, dof {dof_idx}",
                )

    def test_body_gravcomp_spec_conversion(self):
        """Test that body gravcomp is correctly written to the MuJoCo spec and saved XML."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        body1 = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
            custom_attributes={"mujoco:gravcomp": 0.5},
        )
        body2 = builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
            custom_attributes={"mujoco:gravcomp": 1.0},
        )

        builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=body2, hx=0.1, hy=0.1, hz=0.1)

        joint1 = builder.add_joint_revolute(-1, body1, axis=(0.0, 0.0, 1.0))
        joint2 = builder.add_joint_revolute(body1, body2, axis=(0.0, 1.0, 0.0))
        builder.add_articulation([joint1, joint2])

        model = builder.finalize()

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
            xml_path = f.name
        try:
            solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, save_to_mjcf=xml_path)

            # Verify compiled mj_model has correct values
            mj_gravcomp = solver.mj_model.body_gravcomp
            self.assertAlmostEqual(float(mj_gravcomp[1]), 0.5, places=5)
            self.assertAlmostEqual(float(mj_gravcomp[2]), 1.0, places=5)

            # Parse the saved XML and verify gravcomp is on the correct bodies
            tree = ET.parse(xml_path)
            bodies = {b.get("name"): b for b in tree.iter("body")}
            self.assertAlmostEqual(float(bodies["body_0"].get("gravcomp")), 0.5, places=5)
            self.assertAlmostEqual(float(bodies["body_1"].get("gravcomp")), 1.0, places=5)
        finally:
            os.unlink(xml_path)


class TestMuJoCoSolverKinematicBodyProperties(unittest.TestCase):
    @staticmethod
    def _build_model(*, root_kinematic: bool) -> tuple[newton.Model, int]:
        builder = newton.ModelBuilder()
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
        root = builder.add_link(mass=1.0, is_kinematic=root_kinematic, label="root")
        child = builder.add_link(mass=1.0, label="child")
        builder.add_shape_box(root, hx=0.05, hy=0.05, hz=0.05, cfg=shape_cfg)
        builder.add_shape_box(child, hx=0.05, hy=0.05, hz=0.05, cfg=shape_cfg)

        j_root = builder.add_joint_free(child=root, parent=-1)
        j_child = builder.add_joint_revolute(parent=root, child=child, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([j_root, j_child])

        model = builder.finalize(requires_grad=False)
        return model, root

    @staticmethod
    def _compute_dof_to_body(model: newton.Model) -> np.ndarray:
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        joint_child = model.joint_child.numpy()

        dof_to_body = np.full(model.joint_dof_count, -1, dtype=np.int32)
        for joint_idx in range(model.joint_count):
            dof_start = int(joint_qd_start[joint_idx])
            dof_count = int(joint_dof_dim[joint_idx, 0] + joint_dof_dim[joint_idx, 1])
            if dof_count > 0:
                dof_to_body[dof_start : dof_start + dof_count] = int(joint_child[joint_idx])
        return dof_to_body

    @staticmethod
    def _get_meaninertia(solver: SolverMuJoCo) -> float:
        if solver.use_mujoco_cpu:
            return float(solver.mj_model.stat.meaninertia)
        return float(solver.mjw_model.stat.meaninertia.numpy()[0])

    def _assert_armature_matches_flags(self, model: newton.Model, solver: SolverMuJoCo):
        dof_to_body = self._compute_dof_to_body(model)
        body_flags = model.body_flags.numpy()
        joint_armature = model.joint_armature.numpy()

        mjc_dof_to_newton_dof = solver.mjc_dof_to_newton_dof.numpy()
        dof_armature = solver.mjw_model.dof_armature.numpy()

        checked_count = 0
        for world_idx in range(mjc_dof_to_newton_dof.shape[0]):
            for mjc_dof in range(mjc_dof_to_newton_dof.shape[1]):
                newton_dof = int(mjc_dof_to_newton_dof[world_idx, mjc_dof])
                if newton_dof < 0:
                    continue

                body_idx = int(dof_to_body[newton_dof])
                is_kinematic = body_idx >= 0 and (int(body_flags[body_idx]) & int(BodyFlags.KINEMATIC)) != 0
                expected_armature = float(KINEMATIC_ARMATURE if is_kinematic else joint_armature[newton_dof])
                actual_armature = float(dof_armature[world_idx, mjc_dof])

                self.assertAlmostEqual(
                    actual_armature,
                    expected_armature,
                    places=6,
                    msg=(
                        f"world={world_idx}, mjc_dof={mjc_dof}, newton_dof={newton_dof}, "
                        f"body={body_idx}, is_kinematic={is_kinematic}"
                    ),
                )
                checked_count += 1

        self.assertGreater(
            checked_count,
            0,
            "No mapped DOFs were validated; armature checks may be passing vacuously.",
        )

    def test_floating_kinematic_body_from_add_body_applies_high_armature(self):
        builder = newton.ModelBuilder()
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
        body = builder.add_body(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
            is_kinematic=True,
            label="floating_kinematic",
        )
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=shape_cfg)
        model = builder.finalize(requires_grad=False)

        initial_armature = np.linspace(0.1, 0.1 * model.joint_dof_count, model.joint_dof_count, dtype=np.float32)
        model.joint_armature.assign(initial_armature)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        self._assert_armature_matches_flags(model, solver)

    def test_kinematic_body_applies_high_armature_on_conversion(self):
        model, _ = self._build_model(root_kinematic=True)
        initial_armature = np.linspace(0.1, 0.1 * model.joint_dof_count, model.joint_dof_count, dtype=np.float32)
        model.joint_armature.assign(initial_armature)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        self._assert_armature_matches_flags(model, solver)

    def test_kinematic_armature_does_not_affect_initial_meaninertia(self):
        for use_mujoco_cpu in (False, True):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                dynamic_model, _ = self._build_model(root_kinematic=False)
                kinematic_model, _ = self._build_model(root_kinematic=True)
                armature = np.linspace(
                    0.1,
                    0.1 * dynamic_model.joint_dof_count,
                    dynamic_model.joint_dof_count,
                    dtype=np.float32,
                )
                dynamic_model.joint_armature.assign(armature)
                kinematic_model.joint_armature.assign(armature)

                dynamic_solver = SolverMuJoCo(
                    dynamic_model,
                    iterations=1,
                    disable_contacts=True,
                    use_mujoco_cpu=use_mujoco_cpu,
                )
                kinematic_solver = SolverMuJoCo(
                    kinematic_model,
                    iterations=1,
                    disable_contacts=True,
                    use_mujoco_cpu=use_mujoco_cpu,
                )

                self.assertAlmostEqual(
                    self._get_meaninertia(kinematic_solver),
                    self._get_meaninertia(dynamic_solver),
                    places=5,
                )

    def test_kinematic_meaninertia_tracks_inertial_updates(self):
        for use_mujoco_cpu in (False, True):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                model, root_body = self._build_model(root_kinematic=False)
                solver = SolverMuJoCo(
                    model,
                    iterations=1,
                    disable_contacts=True,
                    use_mujoco_cpu=use_mujoco_cpu,
                )
                initial_meaninertia = self._get_meaninertia(solver)

                body_flags = model.body_flags.numpy()
                body_flags[root_body] = int(BodyFlags.KINEMATIC)
                model.body_flags.assign(body_flags)
                solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
                self.assertAlmostEqual(self._get_meaninertia(solver), initial_meaninertia, places=5)

                body_mass = model.body_mass.numpy()
                body_mass[root_body] *= 2.0
                model.body_mass.assign(body_mass)
                solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
                mass_meaninertia = self._get_meaninertia(solver)
                self.assertNotAlmostEqual(mass_meaninertia, initial_meaninertia, places=5)

                joint_armature = model.joint_armature.numpy()
                joint_armature += 0.5
                model.joint_armature.assign(joint_armature)
                solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
                updated_meaninertia = self._get_meaninertia(solver)
                self.assertNotAlmostEqual(updated_meaninertia, mass_meaninertia, places=5)

                body_flags[root_body] = int(BodyFlags.DYNAMIC)
                model.body_flags.assign(body_flags)
                solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
                self.assertAlmostEqual(self._get_meaninertia(solver), updated_meaninertia, places=5)

    def test_meaninertia_update_preserves_dampratio_resolution(self):
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="base">
                    <joint name="j1" type="hinge" axis="0 0 1"/>
                    <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1"/>
                </body>
            </worldbody>
            <actuator>
                <position joint="j1" kp="100" dampratio="1"/>
            </actuator>
        </mujoco>
        """

        def build_solver(use_mujoco_cpu: bool) -> SolverMuJoCo:
            builder = newton.ModelBuilder()
            builder.add_mjcf(mjcf, ctrl_direct=True)
            model = builder.finalize()
            model.body_flags.fill_(int(BodyFlags.KINEMATIC))
            return SolverMuJoCo(model, disable_contacts=True, use_mujoco_cpu=use_mujoco_cpu)

        for use_mujoco_cpu in (False, True):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                solver = build_solver(use_mujoco_cpu)
                reference = build_solver(use_mujoco_cpu)

                if use_mujoco_cpu:
                    solver.mj_model.actuator_biasprm[0, 2] = 0.5
                    reference.mj_model.actuator_biasprm[0, 2] = 0.5
                    solver._set_const_0_with_physical_meaninertia()
                    reference._mujoco.mj_setConst(reference.mj_model, reference.mj_data)
                    actual = solver.mj_model.actuator_biasprm[0, 2]
                    expected = reference.mj_model.actuator_biasprm[0, 2]
                else:
                    for current_solver in (solver, reference):
                        biasprm = current_solver.mjw_model.actuator_biasprm.numpy()
                        biasprm[0, 0, 2] = 0.5
                        current_solver.mjw_model.actuator_biasprm.assign(biasprm)
                    solver._set_const_0_with_physical_meaninertia()
                    reference._mujoco_warp.set_const_0(reference.mjw_model, reference.mjw_data)
                    actual = solver.mjw_model.actuator_biasprm.numpy()[0, 0, 2]
                    expected = reference.mjw_model.actuator_biasprm.numpy()[0, 0, 2]

                self.assertAlmostEqual(float(actual), float(expected), places=5)

    def test_body_properties_runtime_update_and_dof_updates(self):
        model, root_body = self._build_model(root_kinematic=False)
        initial_armature = np.linspace(0.2, 0.2 * model.joint_dof_count, model.joint_dof_count, dtype=np.float32)
        model.joint_armature.assign(initial_armature)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        self._assert_armature_matches_flags(model, solver)

        body_flags = model.body_flags.numpy()
        body_flags[root_body] = int(BodyFlags.KINEMATIC)
        model.body_flags.assign(body_flags)
        solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
        self._assert_armature_matches_flags(model, solver)

        updated_armature = initial_armature + 3.0
        model.joint_armature.assign(updated_armature)
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        self._assert_armature_matches_flags(model, solver)

        body_flags[root_body] = int(BodyFlags.DYNAMIC)
        model.body_flags.assign(body_flags)
        solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
        self._assert_armature_matches_flags(model, solver)

    def test_fixed_root_attached_to_world_uses_mocap_and_tracks_pose(self):
        for root_joint_kind, is_kinematic in itertools.product(("fixed", "locked_d6"), (False, True)):
            with self.subTest(root_joint_kind=root_joint_kind, is_kinematic=is_kinematic):
                builder = newton.ModelBuilder()
                root = builder.add_link(
                    mass=1.0,
                    com=wp.vec3(0.0, 0.0, 0.0),
                    inertia=wp.mat33(np.eye(3)),
                    is_kinematic=is_kinematic,
                    label="fixed_root",
                )
                if root_joint_kind == "fixed":
                    root_joint = builder.add_joint_fixed(parent=-1, child=root)
                else:
                    # zero-DOF D6, as imported from a generic USD PhysicsJoint
                    root_joint = builder.add_joint_d6(parent=-1, child=root)
                builder.add_articulation([root_joint])

                model = builder.finalize(requires_grad=False)
                solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

                body_kind = "kinematic" if is_kinematic else "dynamic"

                # Fixed-root links should be exported as MuJoCo mocap bodies.
                self.assertEqual(solver.mj_model.nmocap, 1)
                body_mocapid = solver.mjw_model.body_mocapid.numpy()

                mjc_body_to_newton = solver.mjc_body_to_newton.numpy()
                matching_bodies = np.where(mjc_body_to_newton[0] == root)[0]
                self.assertEqual(len(matching_bodies), 1, "Expected a unique MuJoCo body for the fixed root")
                mjc_root_body = int(matching_bodies[0])
                mocap_idx = int(body_mocapid[mjc_root_body])
                self.assertGreaterEqual(mocap_idx, 0, f"Fixed-root {body_kind} body should be exported as mocap")

                jnt_bodyid = solver.mjw_model.jnt_bodyid.numpy()
                joints_on_root = np.where(jnt_bodyid == mjc_root_body)[0]
                self.assertEqual(
                    len(joints_on_root), 0, f"Fixed-root {body_kind} body should not create a MuJoCo joint"
                )

                initial_mocap_pos = np.array(solver.mjw_data.mocap_pos.numpy()[0, mocap_idx], copy=True)
                initial_mocap_quat = np.array(solver.mjw_data.mocap_quat.numpy()[0, mocap_idx], copy=True)

                new_position = wp.vec3(0.2, -0.1, 0.3)
                new_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.35)
                model.joint_X_p.assign([wp.transform(new_position, new_rotation)])

                solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

                updated_mocap_pos = solver.mjw_data.mocap_pos.numpy()[0, mocap_idx]
                updated_mocap_quat = solver.mjw_data.mocap_quat.numpy()[0, mocap_idx]

                self.assertFalse(np.allclose(updated_mocap_pos, initial_mocap_pos, atol=1e-6))
                self.assertFalse(np.allclose(updated_mocap_quat, initial_mocap_quat, atol=1e-6))

                np.testing.assert_allclose(
                    updated_mocap_pos,
                    [new_position.x, new_position.y, new_position.z],
                    atol=1e-6,
                    err_msg=f"mocap_pos should track the fixed-root {body_kind} transform",
                )

                expected_quat = np.array([new_rotation.w, new_rotation.x, new_rotation.y, new_rotation.z])
                if np.dot(updated_mocap_quat, expected_quat) < 0.0:
                    expected_quat = -expected_quat
                np.testing.assert_allclose(
                    updated_mocap_quat,
                    expected_quat,
                    atol=1e-6,
                    err_msg=f"mocap_quat should track the fixed-root {body_kind} transform",
                )

                solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)
                updated_body_pos = solver.mjw_data.xpos.numpy()[0, mjc_root_body]
                updated_body_quat = solver.mjw_data.xquat.numpy()[0, mjc_root_body]

                np.testing.assert_allclose(
                    updated_body_pos,
                    [new_position.x, new_position.y, new_position.z],
                    atol=1e-6,
                    err_msg=f"xpos should track the fixed-root {body_kind} transform",
                )
                if np.dot(updated_body_quat, expected_quat) < 0.0:
                    expected_quat = -expected_quat
                np.testing.assert_allclose(
                    updated_body_quat,
                    expected_quat,
                    atol=1e-6,
                    err_msg=f"xquat should track the fixed-root {body_kind} transform",
                )

    def test_world_attached_root_multi_world_placement(self):
        """Replicated world-attached roots must sit at each world's own root transform."""
        world_count = 3
        for root_joint_kind in ("fixed", "locked_d6"):
            with self.subTest(root_joint_kind=root_joint_kind):
                template = newton.ModelBuilder()
                root = template.add_link(
                    mass=1.0,
                    com=wp.vec3(0.0, 0.0, 0.0),
                    inertia=wp.mat33(np.eye(3)),
                    label="root",
                )
                root_xform = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
                if root_joint_kind == "fixed":
                    root_joint = template.add_joint_fixed(parent=-1, child=root, parent_xform=root_xform)
                else:
                    # zero-DOF D6, as imported from a generic USD PhysicsJoint
                    root_joint = template.add_joint_d6(parent=-1, child=root, parent_xform=root_xform)
                link = template.add_link(
                    mass=1.0,
                    com=wp.vec3(0.0, 0.0, 0.0),
                    inertia=wp.mat33(np.eye(3)),
                    label="link",
                )
                hinge = template.add_joint_revolute(root, link, axis=wp.vec3(0.0, 1.0, 0.0))
                template.add_articulation([root_joint, hinge])

                builder = newton.ModelBuilder()
                builder.replicate(template, world_count, spacing=(2.0, 0.0, 0.0))
                model = builder.finalize(requires_grad=False)
                solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

                self.assertEqual(solver.mj_model.nmocap, 1, "World-attached root should be exported as mocap")

                # Refresh derived poses from the per-world root placement.
                solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)

                joint_parent = model.joint_parent.numpy()
                joint_world = model.joint_world.numpy()
                joint_child = model.joint_child.numpy()
                joint_X_p = model.joint_X_p.numpy()
                mjc_body_to_newton = solver.mjc_body_to_newton.numpy()
                xpos = solver.mjw_data.xpos.numpy()

                root_joints = np.where(joint_parent == -1)[0]
                self.assertEqual(len(root_joints), world_count)
                for j in root_joints:
                    world = int(joint_world[j])
                    newton_root = int(joint_child[j])
                    matching = np.where(mjc_body_to_newton[world] == newton_root)[0]
                    self.assertEqual(len(matching), 1, "Expected a unique MuJoCo body for the root")
                    np.testing.assert_allclose(
                        xpos[world, matching[0]],
                        joint_X_p[j][:3],
                        atol=1e-6,
                        err_msg=f"world {world} root must sit at its own joint_X_p, not the template world's",
                    )


class TestMuJoCoSolverZeroMassBody(unittest.TestCase):
    def test_zero_mass_body(self):
        """SolverMuJoCo accepts models with zero-mass bodies (e.g. sensor frames).

        Zero-mass bodies keep their zero mass. MuJoCo handles these natively
        when they have fixed joints.
        """
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="robot" pos="0 0 1">
                    <freejoint name="root"/>
                    <geom type="box" size="0.1 0.1 0.1" mass="1.0"/>
                    <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
                </body>
                <body name="sensor_frame" pos="0 0 0"/>
            </worldbody>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        self.assertIsNotNone(solver.mj_model)


class TestMuJoCoSolverQpos0(unittest.TestCase):
    """Tests for qpos0, qpos_spring, ref/springref coordinate conversion, and FK correctness."""

    def test_reference_coordinates_and_roundtrip(self):
        """Convert reference coordinates, spring coordinates, and state in both directions."""
        mjcf = """<mujoco><worldbody>
            <body name="floating" pos="0 0 1.5">
                <joint name="free_initial" type="free"/>
                <geom type="sphere" size="0.1"/>
            </body>
            <body name="hinge_ref_base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="hinge_ref_body" pos="0 0 1">
                    <joint name="hinge_ref" type="hinge" axis="0 1 0" ref="90"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
            <body name="slide_ref_base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="slide_ref_body" pos="0 0 1">
                    <joint name="slide_ref" type="slide" axis="0 0 1" ref="0.1"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
            <body name="hinge_zero_base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="hinge_zero_body" pos="0 0 1">
                    <joint name="hinge_zero" type="hinge" axis="0 1 0"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
            <body name="hinge_spring_base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="hinge_spring_body" pos="0 0 1">
                    <joint name="hinge_spring" type="hinge" axis="0 1 0" springref="30"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
            <body name="free_roundtrip_body" pos="1 2 3">
                <joint name="free_roundtrip" type="free"/>
                <geom type="sphere" size="0.1"/>
            </body>
            <body name="slide_spring_base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="slide_spring_body" pos="0 0 1">
                    <joint name="slide_spring" type="slide" axis="0 0 1" springref="0.25"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
            <body name="slide_roundtrip_base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="slide_roundtrip_body" pos="0 0 1">
                    <joint name="slide_roundtrip" type="slide" axis="0 0 1" ref="0.5"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        builder.joint_label = [label.rsplit("/", 1)[-1] for label in builder.joint_label]
        parent = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(body=parent, hx=0.1, hy=0.1, hz=0.1)
        root = builder.add_joint_fixed(-1, parent)
        child = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(body=child, hx=0.1, hy=0.1, hz=0.1)
        ball = builder.add_joint_ball(parent, child, label="ball")
        builder.add_articulation([root, ball])
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        expected_qpos0 = {
            "free_initial": ([0, 0, 1.5, 1, 0, 0, 0], 1e-6),
            "hinge_ref": ([np.pi / 2], 1e-5),
            "slide_ref": ([0.1], 1e-6),
            "hinge_zero": ([0.0], 1e-6),
            "ball": ([1, 0, 0, 0], 1e-6),
        }
        qpos0 = solver.mjw_model.qpos0.numpy()[0]
        qpos_spring = solver.mjw_model.qpos_spring.numpy()[0]
        for name, (expected, atol) in expected_qpos0.items():
            with self.subTest(property="qpos0", joint=name):
                start = int(solver.mj_model.joint(name).qposadr[0])
                np.testing.assert_allclose(qpos0[start : start + len(expected)], expected, atol=atol)
        for name, expected, atol in (("hinge_spring", np.pi / 6, 1e-5), ("slide_spring", 0.25, 1e-6)):
            with self.subTest(property="qpos_spring", joint=name):
                start = int(solver.mj_model.joint(name).qposadr[0])
                np.testing.assert_allclose(qpos_spring[start], expected, atol=atol)
        free_mj_start = int(solver.mj_model.joint("free_roundtrip").qposadr[0])
        np.testing.assert_allclose(
            qpos_spring[free_mj_start : free_mj_start + 7], qpos0[free_mj_start : free_mj_start + 7], atol=1e-6
        )

        q_start = model.joint_q_start.numpy()
        hinge_start = int(q_start[model.joint_label.index("hinge_ref")])
        slide_start = int(q_start[model.joint_label.index("slide_roundtrip")])
        free_start = int(q_start[model.joint_label.index("free_roundtrip")])
        hinge_mj_start = int(solver.mj_model.joint("hinge_ref").qposadr[0])
        slide_mj_start = int(solver.mj_model.joint("slide_roundtrip").qposadr[0])
        state = model.state()
        original_q = state.joint_q.numpy().copy()
        original_q[slide_start] = 0.3
        state.joint_q.assign(original_q)
        solver._update_mjc_data(solver.mjw_data, model, state)
        qpos = solver.mjw_data.qpos.numpy()
        with self.subTest(direction="Newton to MuJoCo"):
            np.testing.assert_allclose(qpos[0, hinge_mj_start], np.pi / 2, atol=1e-5)
            np.testing.assert_allclose(qpos[0, slide_mj_start], 0.8, atol=1e-5)

        qpos[0, hinge_mj_start] = np.pi / 2 + 0.1
        solver.mjw_data.qpos.assign(qpos)
        solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)
        state_out = model.state()
        solver._update_newton_state(model, state_out, solver.mjw_data, state_prev=state)
        actual_q = state_out.joint_q.numpy()
        with self.subTest(direction="MuJoCo to Newton"):
            np.testing.assert_allclose(actual_q[hinge_start], 0.1, atol=1e-5)
            np.testing.assert_allclose(actual_q[slide_start], 0.3, atol=1e-5)
            np.testing.assert_allclose(
                actual_q[free_start : free_start + 3], original_q[free_start : free_start + 3], atol=1e-5
            )
            q_orig = original_q[free_start + 3 : free_start + 7]
            q_rt = actual_q[free_start + 3 : free_start + 7]
            self.assertLess(min(np.linalg.norm(q_orig - q_rt), np.linalg.norm(q_orig + q_rt)), 1e-5)

    def test_mixed_model_qpos0(self):
        """Verify qpos0 for a model with free, hinge, and slide joints.

        All joint types should produce correct qpos0 values simultaneously:
        free joint from body_q, hinge from ref, slide from ref.
        """
        mjcf = """<mujoco><worldbody>
            <body name="floating" pos="0 0 2">
                <joint name="free_jnt" type="free"/>
                <geom type="sphere" size="0.1"/>
                <body name="hinge_body" pos="0 0 0.5">
                    <joint name="hinge_jnt" type="hinge" axis="0 1 0" ref="45"/>
                    <geom type="box" size="0.05 0.05 0.05" mass="0.1"/>
                    <body name="slide_body" pos="0 0 0.5">
                        <joint name="slide_jnt" type="slide" axis="0 0 1" ref="0.2"/>
                        <geom type="box" size="0.05 0.05 0.05" mass="0.1"/>
                    </body>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        qpos0 = solver.mjw_model.qpos0.numpy()[0]
        # Free joint: 7 coords (pos + quat wxyz)
        np.testing.assert_allclose(qpos0[:3], [0, 0, 2], atol=1e-5)
        np.testing.assert_allclose(qpos0[3:7], [1, 0, 0, 0], atol=1e-5)
        # Hinge with ref=45deg
        np.testing.assert_allclose(qpos0[7], np.deg2rad(45), atol=1e-5)
        # Slide with ref=0.2
        np.testing.assert_allclose(qpos0[8], 0.2, atol=1e-6)

    def test_free_joint_anchor_transform_conversion(self):
        parent_xform = wp.transform(wp.vec3(1.0, -0.5, 0.25), wp.quat_rpy(0.2, -0.1, 0.3))
        child_xform = wp.transform(wp.vec3(-0.2, 0.4, 0.1), wp.quat_rpy(-0.3, 0.2, 0.1))
        joint_xform = wp.transform(wp.vec3(0.5, 1.0, -0.25), wp.quat_rpy(0.1, 0.4, -0.2))
        world_xform = parent_xform * joint_xform * wp.transform_inverse(child_xform)

        builder = newton.ModelBuilder()
        body = builder.add_link(
            xform=world_xform,
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_shape_sphere(body=body, radius=0.1)
        joint = builder.add_joint_free(
            child=body,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )
        builder.add_articulation([joint])
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        state = model.state()
        state.joint_q.assign(np.array(joint_xform))
        solver._update_mjc_data(solver.mjw_data, model, state)

        qpos = solver.mjw_data.qpos.numpy()[0]
        expected_world = np.array(world_xform)
        np.testing.assert_allclose(qpos[:3], expected_world[:3], atol=1.0e-6)
        qpos_quat = qpos[[4, 5, 6, 3]]
        quat_dist = min(
            np.linalg.norm(qpos_quat - expected_world[3:]),
            np.linalg.norm(qpos_quat + expected_world[3:]),
        )
        self.assertLess(quat_dist, 1.0e-6)

        next_joint_xform = wp.transform(wp.vec3(-0.3, 0.2, 0.8), wp.quat_rpy(-0.2, 0.1, 0.5))
        next_world_xform = parent_xform * next_joint_xform * wp.transform_inverse(child_xform)
        next_world = np.array(next_world_xform)
        qpos[:3] = next_world[:3]
        qpos[3:7] = next_world[[6, 3, 4, 5]]
        solver.mjw_data.qpos.assign([qpos])
        solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)

        state_out = model.state()
        solver._update_newton_state(model, state_out, solver.mjw_data, state_prev=state)
        actual_joint = state_out.joint_q.numpy()
        expected_joint = np.array(next_joint_xform)
        np.testing.assert_allclose(actual_joint[:3], expected_joint[:3], atol=1.0e-6)
        quat_dist = min(
            np.linalg.norm(actual_joint[3:7] - expected_joint[3:]),
            np.linalg.norm(actual_joint[3:7] + expected_joint[3:]),
        )
        self.assertLess(quat_dist, 1.0e-6)

    # -- Group D: FK correctness --

    def _compare_body_positions(self, model, solver, state, body_names, atol=0.01):
        """Compare Newton and MuJoCo body positions after FK.

        Runs _update_mjc_data, kinematics, and _update_newton_state, then
        asserts that Newton body positions match MuJoCo xpos for each
        named body.
        """
        solver._update_mjc_data(solver.mjw_data, model, state)
        solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)
        solver._update_newton_state(model, state, solver.mjw_data, state_prev=state)

        newton_body_q = state.body_q.numpy()
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()
        mj_xpos = solver.mjw_data.xpos.numpy()

        for name in body_names:
            newton_idx = next(
                (i for i, lbl in enumerate(model.body_label) if lbl.endswith(f"/{name}")),
                None,
            )
            assert newton_idx is not None, f"Body '{name}' not found in model.body_label"
            mjc_idx = np.where(mjc_body_to_newton[0] == newton_idx)[0][0]
            newton_pos = newton_body_q[newton_idx, :3]
            mj_pos = mj_xpos[0, mjc_idx]
            np.testing.assert_allclose(newton_pos, mj_pos, atol=atol, err_msg=f"Position mismatch for {name}")

    def test_ref_fk_matches_mujoco(self):
        """Verify Newton FK matches MuJoCo FK for joints with ref.

        At joint_q=0 (Newton) / qpos=ref (MuJoCo), both should produce
        the same body positions corresponding to the MJCF configuration.
        """
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child1" pos="0 0 1">
                    <joint type="hinge" axis="0 1 0" ref="90"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                    <body name="child2" pos="0 0 1">
                        <joint type="slide" axis="0 0 1" ref="0.5"/>
                        <geom type="box" size="0.1 0.1 0.1"/>
                    </body>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        state = model.state()
        self._compare_body_positions(model, solver, state, ["child1", "child2"])

    def test_ref_fk_after_stepping(self):
        """Verify Newton and MuJoCo body positions match after stepping with ref."""
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child" pos="0 0 1">
                    <joint type="hinge" axis="0 1 0" ref="45"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        for _ in range(10):
            solver.step(state_in, state_out, control, contacts, 0.01)
            state_in, state_out = state_out, state_in
        self._compare_body_positions(model, solver, state_in, ["child"])

    def test_multi_joint_ref_fk(self):
        """Verify FK matches MuJoCo for a multi-joint chain with mixed ref values."""
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="b1" pos="0 0 1">
                    <joint type="hinge" axis="0 1 0" ref="30"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                    <body name="b2" pos="0 0 1">
                        <joint type="hinge" axis="1 0 0" ref="60"/>
                        <geom type="box" size="0.1 0.1 0.1"/>
                        <body name="b3" pos="0 0 1">
                            <joint type="slide" axis="0 0 1" ref="0.3"/>
                            <geom type="box" size="0.1 0.1 0.1"/>
                        </body>
                    </body>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        state = model.state()
        self._compare_body_positions(model, solver, state, ["b1", "b2", "b3"])

    # -- Group E: Runtime randomization --

    def test_multiworld_free_joint_qpos0_differs(self):
        """Verify per-world qpos0 differs after changing body_q per world."""
        mjcf = """<mujoco><worldbody>
            <body name="b" pos="0 0 1">
                <joint type="free"/>
                <geom type="sphere" size="0.1"/>
            </body>
        </worldbody></mujoco>"""
        template = newton.ModelBuilder()
        template.add_mjcf(mjcf)
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        # Set different body_q for each world
        body_q = model.body_q.numpy()
        body_q[0, :3] = [0, 0, 1.0]  # world 0
        body_q[1, :3] = [0, 0, 2.0]  # world 1
        model.body_q.assign(body_q)

        solver.notify_model_changed(ModelFlags.ALL)

        qpos0 = solver.mjw_model.qpos0.numpy()
        np.testing.assert_allclose(qpos0[0, 2], 1.0, atol=1e-5)
        np.testing.assert_allclose(qpos0[1, 2], 2.0, atol=1e-5)

    def test_multiworld_hinge_ref_qpos0_differs(self):
        """Verify per-world qpos0 differs after setting different dof_ref per world."""
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child" pos="0 0 1">
                    <joint type="hinge" axis="0 1 0" ref="0"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        template = newton.ModelBuilder()
        template.add_mjcf(mjcf)
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        # Set different dof_ref per world
        dof_ref = model.mujoco.dof_ref.numpy()
        dof_ref[0] = 0.5  # world 0
        dof_ref[1] = 1.0  # world 1
        model.mujoco.dof_ref.assign(dof_ref)

        solver.notify_model_changed(ModelFlags.ALL)

        qpos0 = solver.mjw_model.qpos0.numpy()
        np.testing.assert_allclose(qpos0[0, 0], 0.5, atol=1e-5)
        np.testing.assert_allclose(qpos0[1, 0], 1.0, atol=1e-5)

    def test_dof_ref_runtime_change_updates_qpos0(self):
        """Verify qpos0 updates after runtime dof_ref change via notify_model_changed."""
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child" pos="0 0 1">
                    <joint type="hinge" axis="0 1 0" ref="0"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        # Initially ref=0, so qpos0=0
        qpos0 = solver.mjw_model.qpos0.numpy()
        np.testing.assert_allclose(qpos0[0, 0], 0.0, atol=1e-6)

        # Change ref at runtime
        dof_ref = model.mujoco.dof_ref.numpy()
        dof_ref[0] = 0.7
        model.mujoco.dof_ref.assign(dof_ref)
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

        qpos0 = solver.mjw_model.qpos0.numpy()
        np.testing.assert_allclose(qpos0[0, 0], 0.7, atol=1e-5)

    # -- Group F: Edge cases --

    def test_ref_zero_no_offset(self):
        """Verify no offset is applied when ref defaults to zero."""
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child" pos="0 0 1">
                    <joint type="hinge" axis="0 1 0"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        state = model.state()
        # Set a known joint_q
        q = state.joint_q.numpy()
        q[0] = 0.5
        state.joint_q.assign(q)
        solver._update_mjc_data(solver.mjw_data, model, state)
        qpos = solver.mjw_data.qpos.numpy()
        np.testing.assert_allclose(qpos[0, 0], 0.5, atol=1e-6, err_msg="With ref=0, qpos should equal joint_q")

    def test_free_joint_non_identity_orientation_qpos0(self):
        """Verify free joint qpos0 quaternion for non-identity initial orientation."""
        mjcf = """<mujoco><worldbody>
            <body name="b" pos="0 0 1" quat="0.707 0 0.707 0">
                <joint type="free"/>
                <geom type="sphere" size="0.1"/>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        qpos0 = solver.mjw_model.qpos0.numpy()
        # Position
        np.testing.assert_allclose(qpos0[0, :3], [0, 0, 1], atol=1e-3)
        # Quaternion in wxyz - should be approximately [0.707, 0, 0.707, 0]
        q = qpos0[0, 3:7]
        expected = np.array([0.707, 0, 0.707, 0])
        expected = expected / np.linalg.norm(expected)
        quat_dist = min(np.linalg.norm(q - expected), np.linalg.norm(q + expected))
        self.assertLess(quat_dist, 0.01)


class TestMuJoCoRefCoordinates(unittest.TestCase):
    """Verify the joint-reference (MJCF ``ref``) coordinate convention.

    Newton joint coordinates, limits, and position targets are relative to the authored pose; MuJoCo's qpos,
    jnt_range, and position-actuator ctrl are absolute. The importer subtracts ref from authored ranges, and the
    solver adds it back at the MuJoCo boundary.
    """

    MJCF = """<?xml version="1.0" ?>
    <mujoco model="refmodel">
        <compiler angle="radian"/>
        <option gravity="0 0 0"/>
        <worldbody>
            <body name="base" pos="0 0 0">
                <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                <body name="child" pos="0 0 0">
                    <joint name="hinge" type="hinge" axis="0 0 1" ref="0.5" range="0.1 0.9"/>
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <position name="drive" joint="hinge" kp="500" kv="50" inheritrange="1"/>
        </actuator>
    </mujoco>
    """

    def _make_sim(self, *, ctrl_direct=False, **solver_kwargs):
        builder = newton.ModelBuilder()
        builder.add_mjcf(self.MJCF, ctrl_direct=ctrl_direct)
        model = builder.finalize()
        solver = SolverMuJoCo(model, disable_contacts=True, **solver_kwargs)
        return model, solver

    def test_jnt_range_is_absolute(self):
        """Exported jnt_range matches the authored absolute range."""
        model, solver = self._make_sim()

        # Newton stores the range relative to the authored pose.
        self.assertAlmostEqual(float(model.joint_limit_lower.numpy()[0]), 0.1 - 0.5, places=5)
        self.assertAlmostEqual(float(model.joint_limit_upper.numpy()[0]), 0.9 - 0.5, places=5)

        np.testing.assert_allclose(solver.mj_model.jnt_range[0], [0.1, 0.9], atol=1e-5)
        np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[0, 0], [0.1, 0.9], atol=1e-5)

    def test_runtime_limit_and_ref_updates(self):
        """Preserve jnt_range = limits + ref across runtime updates."""
        model, solver = self._make_sim()

        model.joint_limit_lower.assign([-0.3])
        model.joint_limit_upper.assign([0.3])
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[0, 0], [0.2, 0.8], atol=1e-5)

        # Changing ref relabels the MuJoCo coordinates: joint_q and its limits
        # keep their meaning, while jnt_range and qpos0 shift with the new ref.
        model.mujoco.dof_ref.assign([0.2])
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[0, 0], [-0.1, 0.5], atol=1e-5)
        self.assertAlmostEqual(float(solver.mjw_model.qpos0.numpy()[0][0]), 0.2, places=5)

    def test_runtime_ref_update_preserves_compiled_ctrlrange(self):
        """Keep compiled ctrlrange fixed when ref changes."""
        model, solver = self._make_sim(use_mujoco_cpu=True)

        model.mujoco.dof_ref.assign([1.5])
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

        np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[0, 0], [1.1, 1.9], atol=1e-5)
        np.testing.assert_allclose(solver.mjw_model.actuator_ctrlrange.numpy()[0, 0], [0.1, 0.9], atol=1e-5)
        np.testing.assert_allclose(solver.mj_model.actuator_ctrlrange[0], [0.1, 0.9], atol=1e-5)

    def test_runtime_ref_update_preserves_direct_compiled_ctrlrange(self):
        """Keep CTRL_DIRECT compiled ctrlrange fixed when ref changes."""
        model, solver = self._make_sim(ctrl_direct=True, use_mujoco_cpu=True)

        model.mujoco.dof_ref.assign([1.5])
        solver.notify_model_changed(ModelFlags.ALL)

        np.testing.assert_allclose(solver.mjw_model.actuator_ctrlrange.numpy()[0, 0], [0.1, 0.9], atol=1e-5)
        np.testing.assert_allclose(solver.mj_model.actuator_ctrlrange[0], [0.1, 0.9], atol=1e-5)

    def test_multiworld_runtime_refs_preserve_compiled_ctrlranges(self):
        """Keep per-world compiled ctrlranges fixed when refs change."""
        template = newton.ModelBuilder()
        template.add_mjcf(self.MJCF)
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        model = builder.finalize()
        solver = SolverMuJoCo(model, disable_contacts=True)

        model.mujoco.dof_ref.assign([0.5, 1.5])
        solver.notify_model_changed(ModelFlags.ALL)

        np.testing.assert_allclose(solver.mjw_model.jnt_range.numpy()[:, 0], [[0.1, 0.9], [1.1, 1.9]], atol=1e-5)
        np.testing.assert_allclose(
            solver.mjw_model.actuator_ctrlrange.numpy()[:, 0], [[0.1, 0.9], [0.1, 0.9]], atol=1e-5
        )

    def test_multiworld_inheritrange_factors_resolve_independently(self):
        """Resolve each world's inheritrange factor independently."""
        half_range = newton.ModelBuilder()
        half_range.add_mjcf(self.MJCF.replace('inheritrange="1"', 'inheritrange="0.5"'))
        full_range = newton.ModelBuilder()
        full_range.add_mjcf(self.MJCF)

        builder = newton.ModelBuilder()
        builder.add_world(half_range)
        builder.add_world(full_range)
        model = builder.finalize()
        solver = SolverMuJoCo(model, disable_contacts=True)

        expected = [[0.3, 0.7], [0.1, 0.9]]
        np.testing.assert_allclose(model.mujoco.actuator_ctrlrange.numpy(), expected, atol=1e-5)
        np.testing.assert_allclose(solver.mjw_model.actuator_ctrlrange.numpy()[:, 0], expected, atol=1e-5)

    def test_runtime_ctrlrange_overrides_compiled_inheritrange(self):
        """Apply explicit runtime ctrlrange edits after inheritrange compilation."""
        for ctrl_direct in (False, True):
            with self.subTest(ctrl_direct=ctrl_direct):
                model, solver = self._make_sim(ctrl_direct=ctrl_direct, use_mujoco_cpu=True)
                model.mujoco.actuator_ctrlrange.assign([[0.3, 0.7]])

                solver.notify_model_changed(ModelFlags.ACTUATOR_PROPERTIES)

                np.testing.assert_allclose(solver.mjw_model.actuator_ctrlrange.numpy()[0, 0], [0.3, 0.7], atol=1e-5)
                np.testing.assert_allclose(solver.mj_model.actuator_ctrlrange[0], [0.3, 0.7], atol=1e-5)

    def test_position_target_matches_joint_q(self):
        """A position target of x must drive joint_q to x regardless of ref."""
        model, solver = self._make_sim()
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        control.joint_target_q.assign([0.2])
        for _ in range(500):
            solver.step(state_0, state_1, control, None, 0.01)
            state_0, state_1 = state_1, state_0

        self.assertAlmostEqual(float(state_0.joint_q.numpy()[0]), 0.2, delta=1e-3)

    def test_limit_clamps_joint_q(self):
        """Driving past the limit clamps joint_q at its upper bound."""
        model, solver = self._make_sim()
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # Target beyond the upper joint_q limit (0.9 - 0.5 = 0.4).
        control.joint_target_q.assign([0.8])
        for _ in range(500):
            solver.step(state_0, state_1, control, None, 0.01)
            state_0, state_1 = state_1, state_0

        self.assertAlmostEqual(float(state_0.joint_q.numpy()[0]), 0.4, delta=0.02)

    def test_inheritrange_ctrlrange_is_absolute(self):
        """Verify that inheritrange control ranges use absolute MuJoCo coordinates."""
        mjcf = """<?xml version="1.0" ?>
        <mujoco model="inheritrange">
            <compiler angle="radian" autolimits="true"/>
            <worldbody>
                <body name="base" pos="0 0 0">
                    <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                    <body name="child" pos="0 0 0">
                        <joint name="hinge" type="hinge" axis="0 0 1" ref="0.5" range="0.1 0.9"/>
                        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <position name="drive" joint="hinge" kp="500" inheritrange="1"/>
            </actuator>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model, disable_contacts=True)

        # Newton stores limits relative to the authored pose; MuJoCo uses absolute coordinates.
        np.testing.assert_allclose(solver.mj_model.actuator_ctrlrange[0], [0.1, 0.9], atol=1e-5)
        np.testing.assert_allclose(solver.mj_model.jnt_range[0], [0.1, 0.9], atol=1e-5)
