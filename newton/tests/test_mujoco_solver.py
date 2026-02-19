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

import time
import unittest

import numpy as np  # For numerical operations and random values
import warp as wp

import newton
from newton import JointType, Mesh
from newton._src.core.types import vec5
from newton.solvers import SolverMuJoCo, SolverNotifyFlags
from newton.tests.unittest_utils import USD_AVAILABLE


class TestMuJoCoSolver(unittest.TestCase):
    def _run_substeps_for_frame(self, sim_dt, sim_substeps):
        """Helper method to run simulation substeps for one rendered frame."""
        for _ in range(sim_substeps):
            self.solver.step(self.state_in, self.state_out, self.control, self.contacts, sim_dt)
            self.state_in, self.state_out = self.state_out, self.state_in  # Output becomes input for next substep

    def test_setup_completes(self):
        """
        Tests if the setUp method completes successfully.
        This implicitly tests model creation, finalization, solver, and viewer initialization.
        """
        self.assertTrue(True, "setUp method completed.")

    def test_ls_parallel_option(self):
        """Test that ls_parallel option is properly set on the MuJoCo Warp model."""
        # Create minimal model with proper inertia
        builder = newton.ModelBuilder()
        link = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_revolute(-1, link)
        builder.add_articulation([joint])
        model = builder.finalize()

        # Test with ls_parallel=True
        solver = SolverMuJoCo(model, ls_parallel=True)
        self.assertTrue(solver.mjw_model.opt.ls_parallel, "ls_parallel should be True when set to True")

        # Test with ls_parallel=False (default)
        solver_default = SolverMuJoCo(model, ls_parallel=False)
        self.assertFalse(solver_default.mjw_model.opt.ls_parallel, "ls_parallel should be False when set to False")

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
            return
        except Exception as e:
            self.skipTest(f"Error initializing SolverMuJoCo for trajectory test: {e}")
            return

        if self.debug_stage_path:
            try:
                print("Debug: Attempting to initialize ViewerGL...")
                viewer = newton.viewer.ViewerGL()
                viewer.set_model(self.model)
                print("Debug: ViewerGL initialized successfully for trajectory test.")
            except ImportError as e:
                self.skipTest(f"ViewerGL dependencies not met. Skipping trajectory rendering: {e}")
                return
            except Exception as e:
                self.skipTest(f"Error initializing ViewerGL for trajectory test: {e}")
                return
        else:
            self.skipTest("No debug_stage_path set. Skipping trajectory rendering.")
            return

        num_frames = 200
        sim_substeps = 2
        frame_dt = 1.0 / 60.0
        sim_dt = frame_dt / sim_substeps
        sim_time = 0.0

        # Override self.solver for _run_substeps_for_frame if it was defined in setUp
        # However, since we moved initialization here, we pass it directly or use the local var.
        # For simplicity, let _run_substeps_for_frame use self.solver, so we assign the local one to it.
        self.solver = solver  # Make solver accessible to _run_substeps_for_frame via self

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


class TestMuJoCoSolverPropertiesBase(TestMuJoCoSolver):
    """Base class for MuJoCo solver property tests with common setup."""

    def setUp(self):
        """Set up a model with multiple worlds, each with a free body and an articulated tree."""
        self.seed = 123
        self.rng = np.random.default_rng(self.seed)

        world_count = 2
        self.debug_stage_path = "newton/tests/test_mujoco_render.usda"

        template_builder = newton.ModelBuilder()
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)  # Define ShapeConfig

        # --- Free-floating body (e.g., a box) ---
        free_body_initial_pos = wp.transform((0.5, 0.5, 0.0), wp.quat_identity())
        free_body_idx = template_builder.add_body(mass=0.2, xform=free_body_initial_pos)
        template_builder.add_shape_box(
            body=free_body_idx,
            xform=wp.transform(),  # Shape at body's local origin
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=shape_cfg,
        )

        # --- Articulated tree (3 bodies) ---
        link_radius = 0.05
        link_half_length = 0.15
        tree_root_initial_pos_y = link_half_length * 2.0
        tree_root_initial_transform = wp.transform((0.0, tree_root_initial_pos_y, 0.0), wp.quat_identity())

        body1_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=body1_idx,
            xform=wp.transform(),  # Shape at body's local origin
            radius=link_radius,
            half_height=link_half_length,
            cfg=shape_cfg,
        )
        joint1 = template_builder.add_joint_free(child=body1_idx, parent_xform=tree_root_initial_transform)

        body2_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=body2_idx,
            xform=wp.transform(),  # Shape at body's local origin
            radius=link_radius,
            half_height=link_half_length,
            cfg=shape_cfg,
        )
        joint2 = template_builder.add_joint_revolute(
            parent=body1_idx,
            child=body2_idx,
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            axis=(0.0, 0.0, 1.0),
        )

        body3_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=body3_idx,
            xform=wp.transform(),  # Shape at body's local origin
            radius=link_radius,
            half_height=link_half_length,
            cfg=shape_cfg,
        )
        joint3 = template_builder.add_joint_revolute(
            parent=body2_idx,
            child=body3_idx,
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            axis=(1.0, 0.0, 0.0),
        )

        template_builder.add_articulation([joint1, joint2, joint3])

        self.builder = newton.ModelBuilder()
        self.builder.add_shape_plane()

        for i in range(world_count):
            world_transform = wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity())
            self.builder.add_world(template_builder, xform=world_transform)

        try:
            if self.builder.world_count == 0 and world_count > 0:
                self.builder.world_count = world_count
            self.model = self.builder.finalize()
            if self.model.world_count != world_count:
                print(
                    f"Warning: Model.world_count ({self.model.world_count}) does not match expected world_count ({world_count})."
                )
        except Exception as e:
            self.fail(f"Model finalization failed: {e}")

        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.model.collide(self.state_in, self.contacts)


class TestMuJoCoSolverMassProperties(TestMuJoCoSolverPropertiesBase):
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
        solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

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

                    # Convert positions based on up_axis
                    if self.model.up_axis == 1:  # Y-axis up
                        expected_pos = np.array([newton_pos[0], -newton_pos[2], newton_pos[1]])
                    else:  # Z-axis up
                        expected_pos = newton_pos

                    for dim in range(3):
                        self.assertAlmostEqual(
                            expected_pos[dim],
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
        solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

        # Check that updated COM positions were transferred correctly
        for world_idx in range(nworld):
            for mjc_body in range(nbody):
                newton_body = mjc_body_to_newton[world_idx, mjc_body]
                if newton_body >= 0:  # Skip unmapped bodies
                    newton_pos = updated_coms[newton_body]
                    mjc_pos = solver.mjw_model.body_ipos.numpy()[world_idx, mjc_body]

                    # Convert positions based on up_axis
                    if self.model.up_axis == 1:  # Y-axis up
                        expected_pos = np.array([newton_pos[0], -newton_pos[2], newton_pos[1]])
                    else:  # Z-axis up
                        expected_pos = newton_pos

                    for dim in range(3):
                        self.assertAlmostEqual(
                            expected_pos[dim],
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

        def check_inertias(inertias_to_check, msg_prefix=""):
            for world_idx in range(nworld):
                for mjc_body in range(nbody):
                    newton_body = mjc_body_to_newton[world_idx, mjc_body]
                    if newton_body >= 0:  # Skip unmapped bodies
                        newton_inertia = inertias_to_check[newton_body].astype(np.float32)
                        mjc_inertia = solver.mjw_model.body_inertia.numpy()[world_idx, mjc_body].astype(np.float32)

                        # Get eigenvalues of both tensors
                        newton_eigvecs, newton_eigvals = wp.eig3(wp.mat33(newton_inertia))
                        newton_eigvecs = np.array(newton_eigvecs)
                        newton_eigvecs = newton_eigvecs.reshape((3, 3))

                        newton_eigvals = np.array(newton_eigvals)

                        mjc_eigvals = mjc_inertia  # Already in diagonal form
                        mjc_iquat = np.roll(
                            solver.mjw_model.body_iquat.numpy()[world_idx, mjc_body].astype(np.float32), 1
                        )

                        # Sort eigenvalues in descending order and reorder eigenvectors by columns
                        sort_indices = np.argsort(newton_eigvals)[::-1]
                        newton_eigvals = newton_eigvals[sort_indices]
                        newton_eigvecs = newton_eigvecs[:, sort_indices]

                        # Ensure proper rotation (det=+1) before quaternion conversion
                        # This mirrors the fix in update_body_inertia_kernel
                        if np.linalg.det(newton_eigvecs) < 0:
                            newton_eigvecs[:, 2] = -newton_eigvecs[:, 2]

                        newton_quat = wp.quat_from_matrix(
                            wp.matrix_from_cols(
                                wp.vec3(newton_eigvecs[:, 0]),
                                wp.vec3(newton_eigvecs[:, 1]),
                                wp.vec3(newton_eigvecs[:, 2]),
                            )
                        )
                        newton_quat = wp.normalize(newton_quat)

                        for dim in range(3):
                            self.assertAlmostEqual(
                                float(newton_eigvals[dim]),
                                float(mjc_eigvals[dim]),
                                places=4,
                                msg=f"{msg_prefix}Inertia eigenvalue mismatch for mjc_body {mjc_body} (newton {newton_body}) in world {world_idx}, dimension {dim}",
                            )
                        # Handle quaternion sign ambiguity by ensuring dot product is non-negative
                        newton_quat_np = np.array(newton_quat)
                        mjc_iquat_np = np.array(mjc_iquat)
                        if np.dot(newton_quat_np, mjc_iquat_np) < 0:
                            newton_quat_np = -newton_quat_np

                        for dim in range(4):
                            self.assertAlmostEqual(
                                float(mjc_iquat_np[dim]),
                                float(newton_quat_np[dim]),
                                places=5,
                                msg=f"{msg_prefix}Inertia quaternion mismatch for mjc_body {mjc_body} (newton {newton_body}) in world {world_idx}",
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
        solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

        # Check updated inertia tensors
        check_inertias(updated_inertias, "Updated ")

    def test_body_inertia_eigendecomposition_determinant(self):
        """
        Tests that the inertia eigendecomposition correctly handles cases where
        wp.eig3() returns an eigenvector matrix with determinant -1 (a reflection).

        The kernel must ensure det(V) = +1 before calling quat_from_matrix(),
        otherwise the quaternion will be incorrect and the reconstructed inertia
        tensor will not match the original.
        """
        # Create a specific inertia tensor that is known to trigger det=-1 from wp.eig3
        # This is a diagonal tensor where eigendecomposition can return reflections
        diagonal_inertia = np.diag([9.9086283e-05, 1.3213398e-04, 9.9086275e-05]).astype(np.float32)

        # Assign this inertia to ALL bodies to ensure the mapped body gets it
        new_inertias = np.zeros((self.model.body_count, 3, 3), dtype=np.float32)
        for i in range(self.model.body_count):
            new_inertias[i] = diagonal_inertia
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
                    # If determinant fix wasn't applied, this would fail
                    np.testing.assert_allclose(
                        reconstructed,
                        diagonal_inertia,
                        atol=1e-5,
                        err_msg=f"Reconstructed inertia tensor does not match original for "
                        f"mjc_body {mjc_body} (newton {newton_body}) in world {world_idx}. "
                        "This may indicate the determinant fix in update_body_inertia_kernel is not working.",
                    )
                    checked_count += 1

        self.assertGreater(checked_count, 0, "No bodies were checked")

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
        solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

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
        solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

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
        solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

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


class TestMuJoCoSolverJointProperties(TestMuJoCoSolverPropertiesBase):
    def test_joint_attributes_registration_and_updates(self):
        """
        Verify that joint effort limit, velocity limit, armature, and friction:
        1. Are properly set in Newton Model
        2. Are properly registered in MuJoCo
        3. Can be changed during simulation via notify_model_changed()

        Uses different values for each joint and world to catch indexing bugs.

        TODO: We currently don't check velocity_limits because MuJoCo doesn't seem to have
              a matching parameter. The values are set in Newton but not verified in MuJoCo.
        """
        # Skip if no joints
        if self.model.joint_dof_count == 0:
            self.skipTest("No joints in model, skipping joint attributes test")

        # Step 1: Set initial values with different patterns for each attribute
        # Pattern: base_value + dof_idx * increment + world_offset
        dofs_per_world = self.model.joint_dof_count // self.model.world_count
        joints_per_world = self.model.joint_count // self.model.world_count

        initial_effort_limits = np.zeros(self.model.joint_dof_count)
        initial_velocity_limits = np.zeros(self.model.joint_dof_count)
        initial_friction = np.zeros(self.model.joint_dof_count)
        initial_armature = np.zeros(self.model.joint_dof_count)

        # Iterate over joints and set values for each DOF (skip free joints)
        joint_qd_start = self.model.joint_qd_start.numpy()
        joint_dof_dim = self.model.joint_dof_dim.numpy()
        joint_type = self.model.joint_type.numpy()

        for world_idx in range(self.model.world_count):
            world_joint_offset = world_idx * joints_per_world

            for joint_idx in range(joints_per_world):
                global_joint_idx = world_joint_offset + joint_idx

                # Skip free joints
                if joint_type[global_joint_idx] == JointType.FREE:
                    continue

                # Get DOF start and count for this joint
                dof_start = joint_qd_start[global_joint_idx]
                dof_count = joint_dof_dim[global_joint_idx].sum()

                # Set values for each DOF in this joint
                for dof_offset in range(dof_count):
                    global_dof_idx = dof_start + dof_offset

                    # Effort limit: 50 + dof_offset * 10 + joint_idx * 5 + world_idx * 100
                    initial_effort_limits[global_dof_idx] = (
                        50.0 + dof_offset * 10.0 + joint_idx * 5.0 + world_idx * 100.0
                    )
                    # Velocity limit: 10 + dof_offset * 2 + joint_idx * 1 + world_idx * 20
                    initial_velocity_limits[global_dof_idx] = (
                        10.0 + dof_offset * 2.0 + joint_idx * 1.0 + world_idx * 20.0
                    )
                    # Friction: 0.5 + dof_offset * 0.1 + joint_idx * 0.05 + world_idx * 0.5
                    initial_friction[global_dof_idx] = 0.5 + dof_offset * 0.1 + joint_idx * 0.05 + world_idx * 0.5
                    # Armature: 0.01 + dof_offset * 0.005 + joint_idx * 0.002 + world_idx * 0.05
                    initial_armature[global_dof_idx] = 0.01 + dof_offset * 0.005 + joint_idx * 0.002 + world_idx * 0.05

        self.model.joint_effort_limit.assign(initial_effort_limits)
        self.model.joint_velocity_limit.assign(initial_velocity_limits)
        self.model.joint_friction.assign(initial_friction)
        self.model.joint_armature.assign(initial_armature)

        # Step 2: Create solver (this should apply values to MuJoCo)
        solver = SolverMuJoCo(self.model, iterations=1, disable_contacts=True)

        # Check armature: Newton value should appear directly in MuJoCo DOF armature
        for world_idx in range(self.model.world_count):
            for dof_idx in range(min(dofs_per_world, solver.mjw_model.dof_armature.shape[1])):
                global_dof_idx = world_idx * dofs_per_world + dof_idx
                expected_armature = initial_armature[global_dof_idx]
                actual_armature = solver.mjw_model.dof_armature.numpy()[world_idx, dof_idx]
                self.assertAlmostEqual(
                    actual_armature,
                    expected_armature,
                    places=3,
                    msg=f"MuJoCo DOF {dof_idx} in world {world_idx} armature should match Newton value",
                )

        # Check friction: Newton value should appear in MuJoCo DOF friction loss
        for world_idx in range(self.model.world_count):
            for dof_idx in range(min(dofs_per_world, solver.mjw_model.dof_frictionloss.shape[1])):
                global_dof_idx = world_idx * dofs_per_world + dof_idx
                expected_friction = initial_friction[global_dof_idx]
                actual_friction = solver.mjw_model.dof_frictionloss.numpy()[world_idx, dof_idx]
                self.assertAlmostEqual(
                    actual_friction,
                    expected_friction,
                    places=4,
                    msg=f"MuJoCo DOF {dof_idx} in world {world_idx} friction should match Newton value",
                )

        # Step 4: Change all values with different patterns
        updated_effort_limits = np.zeros(self.model.joint_dof_count)
        updated_velocity_limits = np.zeros(self.model.joint_dof_count)
        updated_friction = np.zeros(self.model.joint_dof_count)
        updated_armature = np.zeros(self.model.joint_dof_count)

        # Iterate over joints and set updated values for each DOF (skip free joints)
        for world_idx in range(self.model.world_count):
            world_joint_offset = world_idx * joints_per_world

            for joint_idx in range(joints_per_world):
                global_joint_idx = world_joint_offset + joint_idx

                # Skip free joints
                if joint_type[global_joint_idx] == JointType.FREE:
                    continue

                # Get DOF start and count for this joint
                dof_start = joint_qd_start[global_joint_idx]
                dof_count = joint_dof_dim[global_joint_idx].sum()

                # Set updated values for each DOF in this joint
                for dof_offset in range(dof_count):
                    global_dof_idx = dof_start + dof_offset

                    # Updated effort limit: 100 + dof_offset * 15 + joint_idx * 8 + world_idx * 150
                    updated_effort_limits[global_dof_idx] = (
                        100.0 + dof_offset * 15.0 + joint_idx * 8.0 + world_idx * 150.0
                    )
                    # Updated velocity limit: 20 + dof_offset * 3 + joint_idx * 2 + world_idx * 30
                    updated_velocity_limits[global_dof_idx] = (
                        20.0 + dof_offset * 3.0 + joint_idx * 2.0 + world_idx * 30.0
                    )
                    # Updated friction: 1.0 + dof_offset * 0.2 + joint_idx * 0.1 + world_idx * 1.0
                    updated_friction[global_dof_idx] = 1.0 + dof_offset * 0.2 + joint_idx * 0.1 + world_idx * 1.0
                    # Updated armature: 0.05 + dof_offset * 0.01 + joint_idx * 0.005 + world_idx * 0.1
                    updated_armature[global_dof_idx] = 0.05 + dof_offset * 0.01 + joint_idx * 0.005 + world_idx * 0.1

        self.model.joint_effort_limit.assign(updated_effort_limits)
        self.model.joint_velocity_limit.assign(updated_velocity_limits)
        self.model.joint_friction.assign(updated_friction)
        self.model.joint_armature.assign(updated_armature)

        # Step 5: Notify MuJoCo of changes
        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        # Check updated armature
        for world_idx in range(self.model.world_count):
            for dof_idx in range(min(dofs_per_world, solver.mjw_model.dof_armature.shape[1])):
                global_dof_idx = world_idx * dofs_per_world + dof_idx
                expected_armature = updated_armature[global_dof_idx]
                actual_armature = solver.mjw_model.dof_armature.numpy()[world_idx, dof_idx]
                self.assertAlmostEqual(
                    actual_armature,
                    expected_armature,
                    places=4,
                    msg=f"Updated MuJoCo DOF {dof_idx} in world {world_idx} armature should match Newton value",
                )

        # Check updated friction
        for world_idx in range(self.model.world_count):
            for dof_idx in range(min(dofs_per_world, solver.mjw_model.dof_frictionloss.shape[1])):
                global_dof_idx = world_idx * dofs_per_world + dof_idx
                expected_friction = updated_friction[global_dof_idx]
                actual_friction = solver.mjw_model.dof_frictionloss.numpy()[world_idx, dof_idx]
                self.assertAlmostEqual(
                    actual_friction,
                    expected_friction,
                    places=4,
                    msg=f"Updated MuJoCo DOF {dof_idx} in world {world_idx} friction should match Newton value",
                )

    def test_jnt_solimp_conversion_and_updates(self):
        """
        Verify that custom solimplimit attribute:
        1. Is properly registered in Newton Model
        2. Is properly converted to MuJoCo jnt_solimp
        3. Can be changed during simulation via notify_model_changed()
        4. Is properly expanded for multi-world models

        Uses different values for each joint DOF and world to catch indexing bugs.
        """
        # Skip if no joints
        if self.model.joint_dof_count == 0:
            self.skipTest("No joints in model, skipping jnt_solimp test")

        # Step 1: Create a template builder and register SolverMuJoCo custom attributes
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)

        # Free-floating body
        free_body_initial_pos = wp.transform((0.5, 0.5, 0.0), wp.quat_identity())
        free_body_idx = template_builder.add_body(mass=0.2, xform=free_body_initial_pos)
        template_builder.add_shape_box(body=free_body_idx, xform=wp.transform(), hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)

        # Articulated tree
        link_radius = 0.05
        link_half_length = 0.15
        tree_root_initial_pos_y = link_half_length * 2.0
        tree_root_initial_transform = wp.transform((0.0, tree_root_initial_pos_y, 0.0), wp.quat_identity())

        body1_idx = template_builder.add_link(mass=0.1)
        joint1_idx = template_builder.add_joint_free(child=body1_idx, parent_xform=tree_root_initial_transform)
        template_builder.add_shape_capsule(
            body=body1_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )

        body2_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=body2_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )
        joint2_idx = template_builder.add_joint_revolute(
            parent=body1_idx,
            child=body2_idx,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            limit_lower=-np.pi / 2,
            limit_upper=np.pi / 2,
        )

        body3_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=body3_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )
        joint3_idx = template_builder.add_joint_revolute(
            parent=body2_idx,
            child=body3_idx,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            limit_lower=-np.pi / 3,
            limit_upper=np.pi / 3,
        )

        template_builder.add_articulation([joint1_idx, joint2_idx, joint3_idx])

        # Replicate to create multiple worlds
        world_count = 2
        builder = newton.ModelBuilder()
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        # Step 2: Set initial solimplimit values
        joints_per_world = model.joint_count // model.world_count

        # Create initial solimplimit array
        initial_solimplimit = np.zeros((model.joint_dof_count, 5), dtype=np.float32)

        # Iterate over joints and set values for each DOF (skip free joints)
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        joint_type = model.joint_type.numpy()

        for world_idx in range(model.world_count):
            world_joint_offset = world_idx * joints_per_world

            for joint_idx in range(joints_per_world):
                global_joint_idx = world_joint_offset + joint_idx

                # Skip free joints
                if joint_type[global_joint_idx] == JointType.FREE:
                    continue

                # Get DOF start and count for this joint
                dof_start = joint_qd_start[global_joint_idx]
                dof_count = joint_dof_dim[global_joint_idx].sum()

                # Set values for each DOF in this joint
                for dof_offset in range(dof_count):
                    global_dof_idx = dof_start + dof_offset

                    # Pattern: base values + dof_offset * 0.01 + joint_idx * 0.005 + world_idx * 0.1
                    val0 = 0.89 + dof_offset * 0.01 + joint_idx * 0.005 + world_idx * 0.1
                    val1 = 0.90 + dof_offset * 0.01 + joint_idx * 0.005 + world_idx * 0.1
                    val2 = 0.01 + dof_offset * 0.001 + joint_idx * 0.0005 + world_idx * 0.01
                    val3 = 2.0 + dof_offset * 0.1 + joint_idx * 0.05 + world_idx * 0.5
                    val4 = 1.8 + dof_offset * 0.1 + joint_idx * 0.05 + world_idx * 0.5
                    initial_solimplimit[global_dof_idx] = [val0, val1, val2, val3, val4]

        # Assign to model
        model.mujoco.solimplimit.assign(wp.array(initial_solimplimit, dtype=vec5, device=model.device))

        # Step 3: Create solver (it will read the updated values now)
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Step 4: Verify jnt_solimp is properly expanded for multi-world
        jnt_solimp = solver.mjw_model.jnt_solimp.numpy()
        self.assertEqual(jnt_solimp.shape[0], model.world_count, "jnt_solimp should have one entry per world")

        # Step 5: Verify initial values were converted correctly
        # Iterate over MuJoCo joints and verify values match Newton's
        mjc_jnt_to_newton_dof = solver.mjc_jnt_to_newton_dof.numpy()
        nworld_mjc = mjc_jnt_to_newton_dof.shape[0]
        njnt_mjc = mjc_jnt_to_newton_dof.shape[1]

        for world_idx in range(nworld_mjc):
            for mjc_jnt in range(njnt_mjc):
                newton_dof = mjc_jnt_to_newton_dof[world_idx, mjc_jnt]
                if newton_dof < 0:
                    continue  # Skip unmapped joints

                # Get expected solimplimit from Newton model
                expected_solimp = model.mujoco.solimplimit.numpy()[newton_dof, :]

                # Get actual jnt_solimp from MuJoCo
                actual_solimp = jnt_solimp[world_idx, mjc_jnt, :]

                # Verify they match
                np.testing.assert_allclose(
                    actual_solimp,
                    expected_solimp,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Initial jnt_solimp[{world_idx}, {mjc_jnt}] doesn't match "
                    f"Newton solimplimit[{newton_dof}]",
                )

        # Step 6: Update solimplimit values with different patterns
        updated_solimplimit = np.zeros((model.joint_dof_count, 5), dtype=np.float32)

        # Iterate over joints and set updated values for each DOF (skip free joints)
        for world_idx in range(model.world_count):
            world_joint_offset = world_idx * joints_per_world

            for joint_idx in range(joints_per_world):
                global_joint_idx = world_joint_offset + joint_idx

                # Skip free joints
                if joint_type[global_joint_idx] == JointType.FREE:
                    continue

                # Get DOF start and count for this joint
                dof_start = joint_qd_start[global_joint_idx]
                dof_count = joint_dof_dim[global_joint_idx].sum()

                # Set updated values for each DOF in this joint
                for dof_offset in range(dof_count):
                    global_dof_idx = dof_start + dof_offset

                    # Updated pattern: different from initial
                    val0 = 0.85 + dof_offset * 0.02 + joint_idx * 0.01 + world_idx * 0.15
                    val1 = 0.88 + dof_offset * 0.02 + joint_idx * 0.01 + world_idx * 0.15
                    val2 = 0.005 + dof_offset * 0.0005 + joint_idx * 0.00025 + world_idx * 0.005
                    val3 = 1.5 + dof_offset * 0.15 + joint_idx * 0.08 + world_idx * 0.6
                    val4 = 2.2 + dof_offset * 0.15 + joint_idx * 0.08 + world_idx * 0.6
                    updated_solimplimit[global_dof_idx] = [val0, val1, val2, val3, val4]

        model.mujoco.solimplimit.assign(wp.array(updated_solimplimit, dtype=vec5, device=model.device))

        # Step 7: Notify solver of changes
        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        # Step 8: Verify updated values were converted correctly
        updated_jnt_solimp = solver.mjw_model.jnt_solimp.numpy()

        for world_idx in range(nworld_mjc):
            for mjc_jnt in range(njnt_mjc):
                newton_dof = mjc_jnt_to_newton_dof[world_idx, mjc_jnt]
                if newton_dof < 0:
                    continue  # Skip unmapped joints

                # Get expected solimplimit from updated Newton model
                expected_solimp = model.mujoco.solimplimit.numpy()[newton_dof, :]

                # Get actual jnt_solimp from MuJoCo
                actual_solimp = updated_jnt_solimp[world_idx, mjc_jnt, :]

                # Verify they match
                np.testing.assert_allclose(
                    actual_solimp,
                    expected_solimp,
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Updated jnt_solimp[{world_idx}, {mjc_jnt}] doesn't match "
                    f"Newton solimplimit[{newton_dof}]",
                )

    def test_limit_margin_runtime_update(self):
        """Test multi-world expansion and runtime updates of limit_margin."""
        # Step 1: Create a template builder and register SolverMuJoCo custom attributes
        template_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(template_builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)

        # Free-floating body
        free_body_initial_pos = wp.transform((0.5, 0.5, 0.0), wp.quat_identity())
        free_body_idx = template_builder.add_body(mass=0.2, xform=free_body_initial_pos)
        template_builder.add_shape_box(body=free_body_idx, xform=wp.transform(), hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)

        # Articulated tree
        link_radius = 0.05
        link_half_length = 0.15
        tree_root_initial_pos_y = link_half_length * 2.0
        tree_root_initial_transform = wp.transform((0.0, tree_root_initial_pos_y, 0.0), wp.quat_identity())

        link1_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=link1_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )
        joint1_idx = template_builder.add_joint_free(child=link1_idx, parent_xform=tree_root_initial_transform)

        link2_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=link2_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )
        joint2_idx = template_builder.add_joint_revolute(
            parent=link1_idx,
            child=link2_idx,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            limit_lower=-np.pi / 2,
            limit_upper=np.pi / 2,
            custom_attributes={"mujoco:limit_margin": [0.01]},
        )

        link3_idx = template_builder.add_link(mass=0.1)
        template_builder.add_shape_capsule(
            body=link3_idx, xform=wp.transform(), radius=link_radius, half_height=link_half_length, cfg=shape_cfg
        )
        joint3_idx = template_builder.add_joint_revolute(
            parent=link2_idx,
            child=link3_idx,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            limit_lower=-np.pi / 3,
            limit_upper=np.pi / 3,
            custom_attributes={"mujoco:limit_margin": [0.02]},
        )

        template_builder.add_articulation([joint1_idx, joint2_idx, joint3_idx])

        # Step 2: Replicate to multiple worlds
        world_count = 3
        builder = newton.ModelBuilder()
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        # Step 3: Initialize solver
        solver = SolverMuJoCo(model, separate_worlds=True, iterations=1, disable_contacts=True)

        # Check solver attribute (jnt_margin)
        jnt_margin = solver.mjw_model.jnt_margin.numpy()

        # Retrieve model info
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        joint_type = model.joint_type.numpy()

        joints_per_world = model.joint_count // model.world_count

        # Step 4: Verify initial values - iterate over MuJoCo joints
        limit_margin = model.mujoco.limit_margin.numpy()
        mjc_jnt_to_newton_dof = solver.mjc_jnt_to_newton_dof.numpy()
        nworld_mjc = mjc_jnt_to_newton_dof.shape[0]
        njnt_mjc = mjc_jnt_to_newton_dof.shape[1]

        for world_idx in range(nworld_mjc):
            for mjc_jnt in range(njnt_mjc):
                newton_dof = mjc_jnt_to_newton_dof[world_idx, mjc_jnt]
                if newton_dof < 0:
                    continue

                expected_val = limit_margin[newton_dof]
                actual_val = jnt_margin[world_idx, mjc_jnt]
                self.assertAlmostEqual(actual_val, expected_val, places=6)

        # Step 5: Update limit_margin values at runtime
        new_margins = np.zeros_like(limit_margin)

        for world_idx in range(model.world_count):
            world_joint_offset = world_idx * joints_per_world
            for joint_idx in range(joints_per_world):
                global_joint_idx = world_joint_offset + joint_idx

                if joint_type[global_joint_idx] == JointType.FREE:
                    continue

                newton_dof_start = joint_qd_start[global_joint_idx]
                dof_count = int(joint_dof_dim[global_joint_idx].sum())

                for dof_offset in range(dof_count):
                    newton_dof_idx = newton_dof_start + dof_offset
                    val = 0.1 + world_idx * 0.1 + joint_idx * 0.01
                    new_margins[newton_dof_idx] = val

        model.mujoco.limit_margin.assign(new_margins)

        # Step 6: Notify solver
        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        # Step 7: Verify updates - iterate over MuJoCo joints
        updated_jnt_margin = solver.mjw_model.jnt_margin.numpy()

        for world_idx in range(nworld_mjc):
            for mjc_jnt in range(njnt_mjc):
                newton_dof = mjc_jnt_to_newton_dof[world_idx, mjc_jnt]
                if newton_dof < 0:
                    continue

                expected_val = new_margins[newton_dof]
                actual_val = updated_jnt_margin[world_idx, mjc_jnt]
                self.assertAlmostEqual(actual_val, expected_val, places=6)

    def test_dof_passive_stiffness_damping_multiworld(self):
        """
        Verify that dof_passive_stiffness and dof_passive_damping propagate correctly:
        1. Different per-world values survive conversion to MuJoCo.
        2. notify_model_changed updates all worlds consistently.
        """

        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)

        pendulum = template_builder.add_link(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
        )
        template_builder.add_shape_box(
            body=pendulum,
            xform=wp.transform(),
            hx=0.05,
            hy=0.05,
            hz=0.05,
        )
        joint = template_builder.add_joint_revolute(
            parent=-1,
            child=pendulum,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(),
            child_xform=wp.transform(),
        )
        template_builder.add_articulation([joint])

        world_count = 3
        builder = newton.ModelBuilder()
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        dofs_per_world = model.joint_dof_count // model.world_count

        initial_stiffness = np.zeros(model.joint_dof_count, dtype=np.float32)
        initial_damping = np.zeros(model.joint_dof_count, dtype=np.float32)

        for world_idx in range(model.world_count):
            world_dof_offset = world_idx * dofs_per_world
            for dof_idx in range(dofs_per_world):
                global_idx = world_dof_offset + dof_idx
                initial_stiffness[global_idx] = 0.05 + 0.01 * dof_idx + 0.25 * world_idx
                initial_damping[global_idx] = 0.4 + 0.02 * dof_idx + 0.3 * world_idx

        model.mujoco.dof_passive_stiffness.assign(initial_stiffness)
        model.mujoco.dof_passive_damping.assign(initial_damping)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Get mappings
        mjc_dof_to_newton_dof = solver.mjc_dof_to_newton_dof.numpy()
        mjc_jnt_to_newton_dof = solver.mjc_jnt_to_newton_dof.numpy()
        nworld_mjc = mjc_dof_to_newton_dof.shape[0]
        nv_mjc = mjc_dof_to_newton_dof.shape[1]
        njnt_mjc = mjc_jnt_to_newton_dof.shape[1]

        def assert_passive_values(expected_stiffness: np.ndarray, expected_damping: np.ndarray):
            dof_damping = solver.mjw_model.dof_damping.numpy()
            jnt_stiffness = solver.mjw_model.jnt_stiffness.numpy()

            # Check DOF damping - iterate over MuJoCo DOFs
            for world_idx in range(nworld_mjc):
                for mjc_dof in range(nv_mjc):
                    newton_dof = mjc_dof_to_newton_dof[world_idx, mjc_dof]
                    if newton_dof < 0:
                        continue
                    self.assertAlmostEqual(
                        dof_damping[world_idx, mjc_dof],
                        expected_damping[newton_dof],
                        places=6,
                        msg=f"dof_damping mismatch for world={world_idx}, mjc_dof={mjc_dof}, newton_dof={newton_dof}",
                    )

            # Check joint stiffness - iterate over MuJoCo joints
            for world_idx in range(nworld_mjc):
                for mjc_jnt in range(njnt_mjc):
                    newton_dof = mjc_jnt_to_newton_dof[world_idx, mjc_jnt]
                    if newton_dof < 0:
                        continue
                    self.assertAlmostEqual(
                        jnt_stiffness[world_idx, mjc_jnt],
                        expected_stiffness[newton_dof],
                        places=6,
                        msg=f"jnt_stiffness mismatch for world={world_idx}, mjc_jnt={mjc_jnt}, newton_dof={newton_dof}",
                    )

        assert_passive_values(initial_stiffness, initial_damping)

        updated_stiffness = initial_stiffness + 0.5 + 0.05 * np.arange(model.joint_dof_count, dtype=np.float32)
        updated_damping = initial_damping + 0.3 + 0.03 * np.arange(model.joint_dof_count, dtype=np.float32)

        model.mujoco.dof_passive_stiffness.assign(updated_stiffness)
        model.mujoco.dof_passive_damping.assign(updated_damping)
        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        assert_passive_values(updated_stiffness, updated_damping)

    def test_joint_limit_solref_conversion(self):
        """
        Verify that joint_limit_ke and joint_limit_kd are properly converted to MuJoCo's solref_limit
        using the negative convention: solref_limit = (-stiffness, -damping)
        """
        # Skip if no joints
        if self.model.joint_dof_count == 0:
            self.skipTest("No joints in model, skipping joint limit solref test")

        # Set initial joint limit stiffness and damping values
        dofs_per_world = self.model.joint_dof_count // self.model.world_count

        initial_limit_ke = np.zeros(self.model.joint_dof_count)
        initial_limit_kd = np.zeros(self.model.joint_dof_count)

        # Set different values for each DOF to catch indexing bugs
        for world_idx in range(self.model.world_count):
            world_dof_offset = world_idx * dofs_per_world

            for dof_idx in range(dofs_per_world):
                global_dof_idx = world_dof_offset + dof_idx
                # Stiffness: 1000 + dof_idx * 100 + world_idx * 1000
                initial_limit_ke[global_dof_idx] = 1000.0 + dof_idx * 100.0 + world_idx * 1000.0
                # Damping: 10 + dof_idx * 1 + world_idx * 10
                initial_limit_kd[global_dof_idx] = 10.0 + dof_idx * 1.0 + world_idx * 10.0

        self.model.joint_limit_ke.assign(initial_limit_ke)
        self.model.joint_limit_kd.assign(initial_limit_kd)

        # Create solver (this should convert ke/kd to solref_limit)
        solver = SolverMuJoCo(self.model, iterations=1, disable_contacts=True)

        # Verify initial conversion to jnt_solref
        # Only revolute joints have limits in this model
        # In MuJoCo: joints 0,1 are FREE joints, joints 2,3 are revolute joints
        # Newton DOF mapping: FREE joints use DOFs 0-11, revolute joints use DOFs 12-13
        mjc_revolute_indices = [2, 3]  # MuJoCo joint indices for revolute joints
        newton_revolute_dof_indices = [12, 13]  # Newton DOF indices for revolute joints

        for world_idx in range(self.model.world_count):
            for _i, (mjc_idx, newton_dof_idx) in enumerate(
                zip(mjc_revolute_indices, newton_revolute_dof_indices, strict=False)
            ):
                global_dof_idx = world_idx * dofs_per_world + newton_dof_idx
                expected_ke = -initial_limit_ke[global_dof_idx]
                expected_kd = -initial_limit_kd[global_dof_idx]

                # Get actual values from MuJoCo's jnt_solref array
                actual_solref = solver.mjw_model.jnt_solref.numpy()[world_idx, mjc_idx]
                self.assertAlmostEqual(
                    actual_solref[0],
                    expected_ke,
                    places=3,
                    msg=f"Initial solref stiffness for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                )
                self.assertAlmostEqual(
                    actual_solref[1],
                    expected_kd,
                    places=3,
                    msg=f"Initial solref damping for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                )

        # Test runtime update capability - update joint limit ke/kd values
        updated_limit_ke = initial_limit_ke * 2.0
        updated_limit_kd = initial_limit_kd * 2.0

        self.model.joint_limit_ke.assign(updated_limit_ke)
        self.model.joint_limit_kd.assign(updated_limit_kd)

        # Notify solver of changes - jnt_solref is updated via JOINT_DOF_PROPERTIES
        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        # Verify runtime updates to jnt_solref
        for world_idx in range(self.model.world_count):
            for _i, (mjc_idx, newton_dof_idx) in enumerate(
                zip(mjc_revolute_indices, newton_revolute_dof_indices, strict=False)
            ):
                global_dof_idx = world_idx * dofs_per_world + newton_dof_idx
                expected_ke = -updated_limit_ke[global_dof_idx]
                expected_kd = -updated_limit_kd[global_dof_idx]

                # Get actual values from MuJoCo's jnt_solref array
                actual_solref = solver.mjw_model.jnt_solref.numpy()[world_idx, mjc_idx]
                self.assertAlmostEqual(
                    actual_solref[0],
                    expected_ke,
                    places=3,
                    msg=f"Updated solref stiffness for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                )
                self.assertAlmostEqual(
                    actual_solref[1],
                    expected_kd,
                    places=3,
                    msg=f"Updated solref damping for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                )

    def test_joint_limit_range_conversion(self):
        """
        Verify that joint_limit_lower and joint_limit_upper are properly converted to MuJoCo's jnt_range.
        Test both initial conversion and runtime updates, with different values per world.

        Note: The jnt_limited flag cannot be changed at runtime in MuJoCo.
        """
        # Skip if no joints
        if self.model.joint_dof_count == 0:
            self.skipTest("No joints in model, skipping joint limit range test")

        # Set initial joint limit values
        dofs_per_world = self.model.joint_dof_count // self.model.world_count

        initial_limit_lower = np.zeros(self.model.joint_dof_count)
        initial_limit_upper = np.zeros(self.model.joint_dof_count)

        # Set different values for each DOF and world to catch indexing bugs
        for world_idx in range(self.model.world_count):
            world_dof_offset = world_idx * dofs_per_world

            for dof_idx in range(dofs_per_world):
                global_dof_idx = world_dof_offset + dof_idx
                # Lower limit: -2.0 - dof_idx * 0.1 - world_idx * 0.5
                initial_limit_lower[global_dof_idx] = -2.0 - dof_idx * 0.1 - world_idx * 0.5
                # Upper limit: 2.0 + dof_idx * 0.1 + world_idx * 0.5
                initial_limit_upper[global_dof_idx] = 2.0 + dof_idx * 0.1 + world_idx * 0.5

        self.model.joint_limit_lower.assign(initial_limit_lower)
        self.model.joint_limit_upper.assign(initial_limit_upper)

        # Create solver (this should convert limits to jnt_range)
        solver = SolverMuJoCo(self.model, iterations=1, disable_contacts=True)

        # Verify initial conversion to jnt_range
        # Only revolute joints have limits in this model
        # In MuJoCo: joints 0,1 are FREE joints, joints 2,3 are revolute joints
        # Newton DOF mapping: FREE joints use DOFs 0-11, revolute joints use DOFs 12-13
        mjc_revolute_indices = [2, 3]  # MuJoCo joint indices for revolute joints
        newton_revolute_dof_indices = [12, 13]  # Newton DOF indices for revolute joints

        for world_idx in range(self.model.world_count):
            for _i, (mjc_idx, newton_dof_idx) in enumerate(
                zip(mjc_revolute_indices, newton_revolute_dof_indices, strict=False)
            ):
                global_dof_idx = world_idx * dofs_per_world + newton_dof_idx
                expected_lower = initial_limit_lower[global_dof_idx]
                expected_upper = initial_limit_upper[global_dof_idx]

                # Get actual values from MuJoCo's jnt_range array
                actual_range = solver.mjw_model.jnt_range.numpy()[world_idx, mjc_idx]
                self.assertAlmostEqual(
                    actual_range[0],
                    expected_lower,
                    places=5,
                    msg=f"Initial range lower for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                )
                self.assertAlmostEqual(
                    actual_range[1],
                    expected_upper,
                    places=5,
                    msg=f"Initial range upper for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                )

        # Test runtime update capability - update joint limit values with different values per world
        updated_limit_lower = np.zeros(self.model.joint_dof_count)
        updated_limit_upper = np.zeros(self.model.joint_dof_count)

        for world_idx in range(self.model.world_count):
            world_dof_offset = world_idx * dofs_per_world

            for dof_idx in range(dofs_per_world):
                global_dof_idx = world_dof_offset + dof_idx
                # Different values per world to verify per-world updates
                # Lower limit: -1.5 - dof_idx * 0.2 - world_idx * 1.0
                updated_limit_lower[global_dof_idx] = -1.5 - dof_idx * 0.2 - world_idx * 1.0
                # Upper limit: 1.5 + dof_idx * 0.2 + world_idx * 1.0
                updated_limit_upper[global_dof_idx] = 1.5 + dof_idx * 0.2 + world_idx * 1.0

        self.model.joint_limit_lower.assign(updated_limit_lower)
        self.model.joint_limit_upper.assign(updated_limit_upper)

        # Notify solver of changes - jnt_range is updated via JOINT_PROPERTIES
        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        # Verify runtime updates to jnt_range with different values per world
        for world_idx in range(self.model.world_count):
            for _i, (mjc_idx, newton_dof_idx) in enumerate(
                zip(mjc_revolute_indices, newton_revolute_dof_indices, strict=False)
            ):
                global_dof_idx = world_idx * dofs_per_world + newton_dof_idx
                expected_lower = updated_limit_lower[global_dof_idx]
                expected_upper = updated_limit_upper[global_dof_idx]

                # Get actual values from MuJoCo's jnt_range array
                actual_range = solver.mjw_model.jnt_range.numpy()[world_idx, mjc_idx]
                self.assertAlmostEqual(
                    actual_range[0],
                    expected_lower,
                    places=5,
                    msg=f"Updated range lower for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                )
                self.assertAlmostEqual(
                    actual_range[1],
                    expected_upper,
                    places=5,
                    msg=f"Updated range upper for MuJoCo joint {mjc_idx} (Newton DOF {newton_dof_idx}) in world {world_idx}",
                )

        # Verify that the values changed from initial
        for world_idx in range(self.model.world_count):
            for _i, (_mjc_idx, newton_dof_idx) in enumerate(
                zip(mjc_revolute_indices, newton_revolute_dof_indices, strict=False)
            ):
                global_dof_idx = world_idx * dofs_per_world + newton_dof_idx
                initial_lower = initial_limit_lower[global_dof_idx]
                initial_upper = initial_limit_upper[global_dof_idx]
                updated_lower = updated_limit_lower[global_dof_idx]
                updated_upper = updated_limit_upper[global_dof_idx]

                # Verify values actually changed
                self.assertNotAlmostEqual(
                    initial_lower,
                    updated_lower,
                    places=5,
                    msg=f"Range lower should have changed for Newton DOF {newton_dof_idx} in world {world_idx}",
                )
                self.assertNotAlmostEqual(
                    initial_upper,
                    updated_upper,
                    places=5,
                    msg=f"Range upper should have changed for Newton DOF {newton_dof_idx} in world {world_idx}",
                )

    def test_jnt_actgravcomp_conversion(self):
        """Test that jnt_actgravcomp custom attribute is properly converted to MuJoCo."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        # Add two bodies with revolute joints
        body1 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        body2 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))

        # Add shapes
        builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=body2, hx=0.1, hy=0.1, hz=0.1)

        # Add joints with custom actuatorgravcomp values
        joint1 = builder.add_joint_revolute(
            -1, body1, axis=(0.0, 0.0, 1.0), custom_attributes={"mujoco:jnt_actgravcomp": True}
        )
        joint2 = builder.add_joint_revolute(
            body1, body2, axis=(0.0, 1.0, 0.0), custom_attributes={"mujoco:jnt_actgravcomp": False}
        )

        builder.add_articulation([joint1, joint2])
        model = builder.finalize()

        # Verify the custom attribute exists and has correct values
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "jnt_actgravcomp"))

        jnt_actgravcomp = model.mujoco.jnt_actgravcomp.numpy()
        self.assertEqual(jnt_actgravcomp[0], True)
        self.assertEqual(jnt_actgravcomp[1], False)

        # Create solver and verify it's properly converted to MuJoCo
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify the MuJoCo model has the correct jnt_actgravcomp values
        mjc_actgravcomp = solver.mj_model.jnt_actgravcomp
        self.assertEqual(mjc_actgravcomp[0], 1)  # True -> 1
        self.assertEqual(mjc_actgravcomp[1], 0)  # False -> 0

    def test_solimp_friction_conversion_and_update(self):
        """
        Test validation of solimp_friction custom attribute:
        1. Initial conversion from Model to MuJoCo (multi-world)
        2. Runtime updates (multi-world)
        """
        # Create template with a few joints
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)

        # Body 1
        b1 = template_builder.add_link()
        j1 = template_builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        template_builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)

        # Body 2
        b2 = template_builder.add_link()
        j2 = template_builder.add_joint_revolute(b1, b2, axis=(1, 0, 0))
        template_builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j1, j2])

        # Create main builder with multiple worlds
        world_count = 2
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        # Verify we have the custom attribute
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "solimpfriction"))

        # --- Step 1: Set initial values and verify conversion ---

        # Initialize with unique values for every DOF
        # 2 joints per world -> 2 DOFs per world
        total_dofs = model.joint_dof_count
        initial_values = np.zeros((total_dofs, 5), dtype=np.float32)

        for i in range(total_dofs):
            # Unique pattern: [i, i*2, i*3, i*4, i*5] normalized roughly
            initial_values[i] = [
                0.1 + (i * 0.01) % 0.8,
                0.1 + (i * 0.02) % 0.8,
                0.001 + (i * 0.001) % 0.1,
                0.5 + (i * 0.1) % 0.5,
                1.0 + (i * 0.1) % 2.0,
            ]

        model.mujoco.solimpfriction.assign(wp.array(initial_values, dtype=vec5, device=model.device))

        solver = SolverMuJoCo(model)

        # Check mapping to MuJoCo using mjc_dof_to_newton_dof
        mjc_dof_to_newton_dof = solver.mjc_dof_to_newton_dof.numpy()
        mjw_dof_solimp = solver.mjw_model.dof_solimp.numpy()
        nv = solver.mj_model.nv  # Number of MuJoCo DOFs

        def check_values(expected_values, actual_mjw_values, msg_prefix):
            for w in range(world_count):
                for mjc_dof in range(nv):
                    newton_dof = mjc_dof_to_newton_dof[w, mjc_dof]
                    if newton_dof < 0:
                        continue

                    expected = expected_values[newton_dof]
                    actual = actual_mjw_values[w, mjc_dof]

                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=1e-5,
                        err_msg=f"{msg_prefix} mismatch at World {w}, MuJoCo DOF {mjc_dof}, Newton DOF {newton_dof}",
                    )

        check_values(initial_values, mjw_dof_solimp, "Initial conversion")

        # --- Step 2: Runtime Update ---

        # Generate new unique values
        updated_values = np.zeros((total_dofs, 5), dtype=np.float32)
        for i in range(total_dofs):
            updated_values[i] = [
                0.8 - (i * 0.01) % 0.8,
                0.8 - (i * 0.02) % 0.8,
                0.1 - (i * 0.001) % 0.05,
                0.9 - (i * 0.1) % 0.5,
                2.5 - (i * 0.1) % 1.0,
            ]

        # Update model attribute
        model.mujoco.solimpfriction.assign(wp.array(updated_values, dtype=vec5, device=model.device))

        # Notify solver
        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        # Verify updates
        mjw_dof_solimp_updated = solver.mjw_model.dof_solimp.numpy()

        check_values(updated_values, mjw_dof_solimp_updated, "Runtime update")

        # Check that it is different from initial (sanity check)
        # Just check the first element
        self.assertFalse(
            np.allclose(mjw_dof_solimp_updated[0, 0], initial_values[0]),
            "Value did not change from initial!",
        )

    def test_solref_friction_conversion_and_update(self):
        """
        Test validation of solref_friction custom attribute:
        1. Initial conversion from Model to MuJoCo (multi-world)
        2. Runtime updates (multi-world)
        """
        # Create template with a few joints
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)

        # Body 1
        b1 = template_builder.add_link()
        j1 = template_builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        template_builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)

        # Body 2
        b2 = template_builder.add_link()
        j2 = template_builder.add_joint_revolute(b1, b2, axis=(1, 0, 0))
        template_builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j1, j2])

        # Create main builder with multiple worlds
        world_count = 2
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        # Verify we have the custom attribute
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "solreffriction"))

        # --- Step 1: Set initial values and verify conversion ---

        # Initialize with unique values for every DOF
        # 2 joints per world -> 2 DOFs per world
        total_dofs = model.joint_dof_count
        initial_values = np.zeros((total_dofs, 2), dtype=np.float32)

        for i in range(total_dofs):
            # Unique pattern for 2-element solref
            initial_values[i] = [
                0.01 + (i * 0.005) % 0.05,  # timeconst
                0.5 + (i * 0.1) % 1.5,  # dampratio
            ]

        model.mujoco.solreffriction.assign(initial_values)

        solver = SolverMuJoCo(model)

        # Check mapping to MuJoCo
        mjc_dof_to_newton_dof = solver.mjc_dof_to_newton_dof.numpy()
        mjw_dof_solref = solver.mjw_model.dof_solref.numpy()

        nv = mjc_dof_to_newton_dof.shape[1]  # Number of MuJoCo DOFs

        def check_values(expected_values, actual_mjw_values, msg_prefix):
            for w in range(world_count):
                for mjc_dof in range(nv):
                    newton_dof = mjc_dof_to_newton_dof[w, mjc_dof]
                    if newton_dof < 0:
                        continue

                    expected = expected_values[newton_dof]
                    actual = actual_mjw_values[w, mjc_dof]

                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=1e-5,
                        err_msg=f"{msg_prefix} mismatch at World {w}, MuJoCo DOF {mjc_dof}, Newton DOF {newton_dof}",
                    )

        check_values(initial_values, mjw_dof_solref, "Initial conversion")

        # --- Step 2: Runtime Update ---

        # Generate new unique values
        updated_values = np.zeros((total_dofs, 2), dtype=np.float32)
        for i in range(total_dofs):
            updated_values[i] = [
                0.05 - (i * 0.005) % 0.04,  # timeconst
                2.0 - (i * 0.1) % 1.0,  # dampratio
            ]

        # Update model attribute
        model.mujoco.solreffriction.assign(updated_values)

        # Notify solver
        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        # Verify updates
        mjw_dof_solref_updated = solver.mjw_model.dof_solref.numpy()

        check_values(updated_values, mjw_dof_solref_updated, "Runtime update")

        # Check that it is different from initial (sanity check)
        # Just check the first element
        self.assertFalse(
            np.allclose(mjw_dof_solref_updated[0, 0], initial_values[0]),
            "Value did not change from initial!",
        )


class TestMuJoCoSolverGeomProperties(TestMuJoCoSolverPropertiesBase):
    def test_geom_property_conversion(self):
        """
        Test that ALL Newton shape properties are correctly converted to MuJoCo geom properties.
        This includes: friction, contact parameters (solref), size, position, and orientation.
        Note: geom_rbound is computed by MuJoCo from geom size during conversion.
        """
        # Create solver
        solver = SolverMuJoCo(self.model, iterations=1, disable_contacts=True)

        # Verify mjc_geom_to_newton_shape mapping exists
        self.assertTrue(hasattr(solver, "mjc_geom_to_newton_shape"))

        # Get mappings and arrays
        mjc_geom_to_newton_shape = solver.mjc_geom_to_newton_shape.numpy()
        shape_types = self.model.shape_type.numpy()
        num_geoms = solver.mj_model.ngeom

        # Get all property arrays from Newton
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_sizes = self.model.shape_scale.numpy()
        shape_transforms = self.model.shape_transform.numpy()
        shape_bodies = self.model.shape_body.numpy()

        # Get all property arrays from MuJoCo
        geom_friction = solver.mjw_model.geom_friction.numpy()
        geom_solref = solver.mjw_model.geom_solref.numpy()
        geom_size = solver.mjw_model.geom_size.numpy()
        geom_pos = solver.mjw_model.geom_pos.numpy()
        geom_quat = solver.mjw_model.geom_quat.numpy()

        # Test all properties for each geom in each world
        tested_count = 0
        for world_idx in range(self.model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                if shape_idx < 0:  # No mapping for this geom
                    continue

                tested_count += 1
                shape_type = shape_types[shape_idx]

                # Test 1: Friction conversion
                expected_mu = shape_mu[shape_idx]
                actual_friction = geom_friction[world_idx, geom_idx]

                # Slide friction should match exactly
                self.assertAlmostEqual(
                    float(actual_friction[0]),
                    expected_mu,
                    places=5,
                    msg=f"Slide friction mismatch for shape {shape_idx} (type {shape_type}) in world {world_idx}, geom {geom_idx}",
                )

                # Torsional and rolling friction should be absolute values (not scaled by mu)
                expected_torsional = self.model.shape_material_mu_torsional.numpy()[shape_idx]
                expected_rolling = self.model.shape_material_mu_rolling.numpy()[shape_idx]

                self.assertAlmostEqual(
                    float(actual_friction[1]),
                    expected_torsional,
                    places=5,
                    msg=f"Torsional friction mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_friction[2]),
                    expected_rolling,
                    places=5,
                    msg=f"Rolling friction mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                # Test 2: Contact parameters (solref)
                actual_solref = geom_solref[world_idx, geom_idx]

                # Compute expected solref based on Newton's conversion logic
                ke = shape_ke[shape_idx]
                kd = shape_kd[shape_idx]

                if ke > 0.0 and kd > 0.0:
                    timeconst = 2.0 / kd
                    dampratio = np.sqrt(1.0 / (timeconst * timeconst * ke))
                    expected_solref = (timeconst, dampratio)
                else:
                    expected_solref = (0.02, 1.0)

                self.assertAlmostEqual(
                    float(actual_solref[0]),
                    expected_solref[0],
                    places=5,
                    msg=f"Solref[0] mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_solref[1]),
                    expected_solref[1],
                    places=5,
                    msg=f"Solref[1] mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                # Test 3: Size
                actual_size = geom_size[world_idx, geom_idx]
                expected_size = shape_sizes[shape_idx]
                for dim in range(3):
                    if expected_size[dim] > 0:  # Only check non-zero dimensions
                        self.assertAlmostEqual(
                            float(actual_size[dim]),
                            float(expected_size[dim]),
                            places=5,
                            msg=f"Size mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, dimension {dim}",
                        )

                # Test 4: Position and orientation (body-local coordinates)
                actual_pos = geom_pos[world_idx, geom_idx]
                actual_quat = geom_quat[world_idx, geom_idx]

                # Get expected transform from Newton (body-local coordinates)
                shape_transform = wp.transform(*shape_transforms[shape_idx])
                expected_pos = wp.vec3(*shape_transform.p)
                expected_quat = wp.quat(*shape_transform.q)

                # Apply shape-specific rotations (matching update_geom_properties_kernel logic)
                shape_body = shape_bodies[shape_idx]

                # Handle up-axis conversion if needed
                if self.model.up_axis == 1:  # Y-up to Z-up conversion
                    # For static geoms, position conversion
                    if shape_body == -1:
                        expected_pos = wp.vec3(expected_pos[0], -expected_pos[2], expected_pos[1])
                    rot_y2z = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi * 0.5)
                    expected_quat = rot_y2z * expected_quat

                # Convert expected quaternion to MuJoCo format (wxyz)
                expected_quat_mjc = np.array([expected_quat.w, expected_quat.x, expected_quat.y, expected_quat.z])

                # Test position
                for dim in range(3):
                    self.assertAlmostEqual(
                        float(actual_pos[dim]),
                        float(expected_pos[dim]),
                        places=5,
                        msg=f"Position mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, dimension {dim}",
                    )

                # Test quaternion
                for dim in range(4):
                    self.assertAlmostEqual(
                        float(actual_quat[dim]),
                        float(expected_quat_mjc[dim]),
                        places=5,
                        msg=f"Quaternion mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, component {dim}",
                    )

        # Ensure we tested at least some shapes
        self.assertGreater(tested_count, 0, "Should have tested at least one shape")

    def test_geom_property_update(self):
        """
        Test that geom properties can be dynamically updated during simulation.
        This includes: friction, contact parameters (solref), size, position, and orientation.
        Note: collision radius (rbound) is not updated from Newton's shape_collision_radius as MuJoCo computes it internally.
        """
        # Create solver with initial values
        solver = SolverMuJoCo(self.model, iterations=1, disable_contacts=True)

        # Get mappings
        mjc_geom_to_newton_shape = solver.mjc_geom_to_newton_shape.numpy()
        num_geoms = solver.mj_model.ngeom

        # Run an initial simulation step
        solver.step(self.state_in, self.state_out, self.control, self.contacts, 0.01)
        self.state_in, self.state_out = self.state_out, self.state_in

        # Store initial values for comparison
        initial_friction = solver.mjw_model.geom_friction.numpy().copy()
        initial_solref = solver.mjw_model.geom_solref.numpy().copy()
        initial_size = solver.mjw_model.geom_size.numpy().copy()
        initial_pos = solver.mjw_model.geom_pos.numpy().copy()
        initial_quat = solver.mjw_model.geom_quat.numpy().copy()

        # Update ALL Newton shape properties with new values
        shape_count = self.model.shape_count

        # 1. Update friction (slide, torsional, and rolling)
        new_mu = np.zeros(shape_count)
        new_torsional = np.zeros(shape_count)
        new_rolling = np.zeros(shape_count)
        for i in range(shape_count):
            new_mu[i] = 1.0 + (i + 1) * 0.05  # Pattern: 1.05, 1.10, ...
            new_torsional[i] = 0.6 + (i + 1) * 0.02  # Pattern: 0.62, 0.64, ...
            new_rolling[i] = 0.002 + (i + 1) * 0.0001  # Pattern: 0.0021, 0.0022, ...
        self.model.shape_material_mu.assign(new_mu)
        self.model.shape_material_mu_torsional.assign(new_torsional)
        self.model.shape_material_mu_rolling.assign(new_rolling)

        # 2. Update contact stiffness/damping
        new_ke = np.ones(shape_count) * 1000.0  # High stiffness
        new_kd = np.ones(shape_count) * 10.0  # Some damping
        self.model.shape_material_ke.assign(new_ke)
        self.model.shape_material_kd.assign(new_kd)

        # 3. Update sizes
        new_sizes = []
        for i in range(shape_count):
            old_size = self.model.shape_scale.numpy()[i]
            new_size = wp.vec3(old_size[0] * 1.2, old_size[1] * 1.2, old_size[2] * 1.2)
            new_sizes.append(new_size)
        self.model.shape_scale.assign(wp.array(new_sizes, dtype=wp.vec3, device=self.model.device))

        # 4. Update transforms (position and orientation)
        new_transforms = []
        for i in range(shape_count):
            # New position with offset
            new_pos = wp.vec3(0.5 + i * 0.1, 1.0 + i * 0.1, 1.5 + i * 0.1)
            # New orientation (small rotation)
            angle = 0.1 + i * 0.05
            new_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
            new_transform = wp.transform(new_pos, new_quat)
            new_transforms.append(new_transform)
        self.model.shape_transform.assign(wp.array(new_transforms, dtype=wp.transform, device=self.model.device))

        # Notify solver of all shape property changes
        solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

        # Verify properties were updated
        updated_friction = solver.mjw_model.geom_friction.numpy()
        updated_solref = solver.mjw_model.geom_solref.numpy()
        updated_size = solver.mjw_model.geom_size.numpy()
        updated_pos = solver.mjw_model.geom_pos.numpy()
        updated_quat = solver.mjw_model.geom_quat.numpy()

        tested_count = 0
        for world_idx in range(self.model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                if shape_idx < 0:  # No mapping
                    continue

                tested_count += 1

                # Verify 1: Friction updated (slide, torsional, and rolling)
                expected_mu = new_mu[shape_idx]
                expected_torsional = new_torsional[shape_idx]
                expected_rolling = new_rolling[shape_idx]

                # Verify slide friction
                self.assertAlmostEqual(
                    float(updated_friction[world_idx, geom_idx][0]),
                    expected_mu,
                    places=5,
                    msg=f"Updated slide friction should match new value for shape {shape_idx}",
                )
                # Verify torsional friction
                self.assertAlmostEqual(
                    float(updated_friction[world_idx, geom_idx][1]),
                    expected_torsional,
                    places=5,
                    msg=f"Updated torsional friction should match new value for shape {shape_idx}",
                )
                # Verify rolling friction
                self.assertAlmostEqual(
                    float(updated_friction[world_idx, geom_idx][2]),
                    expected_rolling,
                    places=5,
                    msg=f"Updated rolling friction should match new value for shape {shape_idx}",
                )

                # Verify all friction components changed from initial
                self.assertNotAlmostEqual(
                    float(updated_friction[world_idx, geom_idx][0]),
                    float(initial_friction[world_idx, geom_idx][0]),
                    places=5,
                    msg=f"Slide friction should have changed for shape {shape_idx}",
                )
                self.assertNotAlmostEqual(
                    float(updated_friction[world_idx, geom_idx][1]),
                    float(initial_friction[world_idx, geom_idx][1]),
                    places=5,
                    msg=f"Torsional friction should have changed for shape {shape_idx}",
                )
                self.assertNotAlmostEqual(
                    float(updated_friction[world_idx, geom_idx][2]),
                    float(initial_friction[world_idx, geom_idx][2]),
                    places=5,
                    msg=f"Rolling friction should have changed for shape {shape_idx}",
                )

                # Verify 2: Contact parameters updated (solref)
                # Compute expected values based on new ke/kd using timeconst/dampratio conversion
                ke = new_ke[shape_idx]
                kd = new_kd[shape_idx]

                if ke > 0.0 and kd > 0.0:
                    timeconst = 2.0 / kd
                    dampratio = np.sqrt(1.0 / (timeconst * timeconst * ke))
                    expected_solref = (timeconst, dampratio)
                else:
                    expected_solref = (0.02, 1.0)

                self.assertAlmostEqual(
                    float(updated_solref[world_idx, geom_idx][0]),
                    expected_solref[0],
                    places=5,
                    msg=f"Updated solref[0] should match expected for shape {shape_idx}",
                )

                self.assertAlmostEqual(
                    float(updated_solref[world_idx, geom_idx][1]),
                    expected_solref[1],
                    places=5,
                    msg=f"Updated solref[1] should match expected for shape {shape_idx}",
                )

                # Also verify it changed from initial
                self.assertFalse(
                    np.allclose(updated_solref[world_idx, geom_idx], initial_solref[world_idx, geom_idx]),
                    f"Contact parameters should have changed for shape {shape_idx}",
                )

                # Verify 3: Size updated
                # Verify the size matches the expected new size
                expected_size = new_sizes[shape_idx]
                for dim in range(3):
                    self.assertAlmostEqual(
                        float(updated_size[world_idx, geom_idx][dim]),
                        float(expected_size[dim]),
                        places=5,
                        msg=f"Updated size mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, dimension {dim}",
                    )

                # Also verify at least one dimension changed
                size_changed = False
                for dim in range(3):
                    if not np.isclose(updated_size[world_idx, geom_idx][dim], initial_size[world_idx, geom_idx][dim]):
                        size_changed = True
                        break
                self.assertTrue(size_changed, f"Size should have changed for shape {shape_idx}")

                # Verify 4: Position and orientation updated (body-local coordinates)
                # Compute expected values based on new transforms
                new_transform = wp.transform(*new_transforms[shape_idx])
                expected_pos = new_transform.p
                expected_quat = new_transform.q

                # Apply same transformations as in the kernel
                shape_body = self.model.shape_body.numpy()[shape_idx]

                # Handle up-axis conversion if needed
                if self.model.up_axis == 1:  # Y-up to Z-up conversion
                    if shape_body == -1:
                        expected_pos = wp.vec3(expected_pos[0], -expected_pos[2], expected_pos[1])
                    rot_y2z = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi * 0.5)
                    expected_quat = rot_y2z * expected_quat

                # Convert expected quaternion to MuJoCo format (wxyz)
                expected_quat_mjc = np.array([expected_quat.w, expected_quat.x, expected_quat.y, expected_quat.z])

                # Test position updated correctly
                for dim in range(3):
                    self.assertAlmostEqual(
                        float(updated_pos[world_idx, geom_idx][dim]),
                        float(expected_pos[dim]),
                        places=5,
                        msg=f"Updated position mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, dimension {dim}",
                    )

                # Test quaternion updated correctly
                for dim in range(4):
                    self.assertAlmostEqual(
                        float(updated_quat[world_idx, geom_idx][dim]),
                        float(expected_quat_mjc[dim]),
                        places=5,
                        msg=f"Updated quaternion mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}, component {dim}",
                    )

                # Also verify they changed from initial values
                self.assertFalse(
                    np.allclose(updated_pos[world_idx, geom_idx], initial_pos[world_idx, geom_idx]),
                    f"Position should have changed for shape {shape_idx}",
                )
                self.assertFalse(
                    np.allclose(updated_quat[world_idx, geom_idx], initial_quat[world_idx, geom_idx]),
                    f"Orientation should have changed for shape {shape_idx}",
                )

        # Ensure we tested shapes
        self.assertGreater(tested_count, 0, "Should have tested at least one shape")

        # Run another simulation step to ensure the updated properties work
        solver.step(self.state_in, self.state_out, self.control, self.contacts, 0.01)

    def test_mesh_maxhullvert_attribute(self):
        """Test that Mesh objects can store maxhullvert attribute"""

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=np.int32)

        # Test default maxhullvert
        mesh1 = Mesh(vertices, indices)
        self.assertEqual(mesh1.maxhullvert, 64)

        # Test custom maxhullvert
        mesh2 = Mesh(vertices, indices, maxhullvert=128)
        self.assertEqual(mesh2.maxhullvert, 128)

    def test_mujoco_solver_uses_mesh_maxhullvert(self):
        """Test that MuJoCo solver uses per-mesh maxhullvert values"""

        builder = newton.ModelBuilder()

        # Create meshes with different maxhullvert values
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=np.int32)

        mesh1 = Mesh(vertices, indices, maxhullvert=32)
        mesh2 = Mesh(vertices, indices, maxhullvert=128)

        # Add bodies and shapes with these meshes
        body1 = builder.add_body(mass=1.0)
        builder.add_shape_mesh(body=body1, mesh=mesh1)

        body2 = builder.add_body(mass=1.0)
        builder.add_shape_mesh(body=body2, mesh=mesh2)

        model = builder.finalize()

        # Create MuJoCo solver
        solver = SolverMuJoCo(model)

        # The solver should have used the per-mesh maxhullvert values
        # We can't directly verify this without inspecting MuJoCo internals,
        # but we can at least verify the solver was created successfully
        self.assertIsNotNone(solver)

        # Verify that the meshes retained their maxhullvert values
        self.assertEqual(model.shape_source[0].maxhullvert, 32)
        self.assertEqual(model.shape_source[1].maxhullvert, 128)

    def test_heterogeneous_per_shape_friction(self):
        """Test per-shape friction conversion to MuJoCo and dynamic updates across multiple worlds."""
        # Use per-world iteration to handle potential global shapes correctly
        shape_world = self.model.shape_world.numpy()
        initial_mu = np.zeros(self.model.shape_count)
        initial_torsional = np.zeros(self.model.shape_count)
        initial_rolling = np.zeros(self.model.shape_count)

        # Set unique friction values per shape and world
        for world_idx in range(self.model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                initial_mu[shape_idx] = 0.5 + local_idx * 0.1 + world_idx * 0.3
                initial_torsional[shape_idx] = 0.3 + local_idx * 0.05 + world_idx * 0.2
                initial_rolling[shape_idx] = 0.001 + local_idx * 0.0005 + world_idx * 0.002

        self.model.shape_material_mu.assign(initial_mu)
        self.model.shape_material_mu_torsional.assign(initial_torsional)
        self.model.shape_material_mu_rolling.assign(initial_rolling)

        solver = SolverMuJoCo(self.model, iterations=1, disable_contacts=True)
        mjc_geom_to_newton_shape = solver.mjc_geom_to_newton_shape.numpy()
        num_geoms = solver.mj_model.ngeom

        # Verify initial conversion
        geom_friction = solver.mjw_model.geom_friction.numpy()
        tested_count = 0
        for world_idx in range(self.model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                if shape_idx < 0:
                    continue

                tested_count += 1
                expected_mu = initial_mu[shape_idx]
                expected_torsional_abs = initial_torsional[shape_idx]
                expected_rolling_abs = initial_rolling[shape_idx]

                actual_friction = geom_friction[world_idx, geom_idx]

                self.assertAlmostEqual(
                    float(actual_friction[0]),
                    expected_mu,
                    places=5,
                    msg=f"Initial slide friction mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_friction[1]),
                    expected_torsional_abs,
                    places=5,
                    msg=f"Initial torsional friction mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_friction[2]),
                    expected_rolling_abs,
                    places=5,
                    msg=f"Initial rolling friction mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

        self.assertGreater(tested_count, 0, "Should have tested at least one shape")

        # Update with different values
        updated_mu = np.zeros(self.model.shape_count)
        updated_torsional = np.zeros(self.model.shape_count)
        updated_rolling = np.zeros(self.model.shape_count)

        for world_idx in range(self.model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                updated_mu[shape_idx] = 1.0 + local_idx * 0.15 + world_idx * 0.4
                updated_torsional[shape_idx] = 0.6 + local_idx * 0.08 + world_idx * 0.25
                updated_rolling[shape_idx] = 0.005 + local_idx * 0.001 + world_idx * 0.003

        self.model.shape_material_mu.assign(updated_mu)
        self.model.shape_material_mu_torsional.assign(updated_torsional)
        self.model.shape_material_mu_rolling.assign(updated_rolling)

        solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

        # Verify updates
        updated_geom_friction = solver.mjw_model.geom_friction.numpy()

        for world_idx in range(self.model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                if shape_idx < 0:
                    continue

                expected_mu = updated_mu[shape_idx]
                expected_torsional_abs = updated_torsional[shape_idx]
                expected_rolling_abs = updated_rolling[shape_idx]

                actual_friction = updated_geom_friction[world_idx, geom_idx]

                self.assertAlmostEqual(
                    float(actual_friction[0]),
                    expected_mu,
                    places=5,
                    msg=f"Updated slide friction mismatch for shape {shape_idx} in world {world_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_friction[1]),
                    expected_torsional_abs,
                    places=5,
                    msg=f"Updated torsional friction mismatch for shape {shape_idx} in world {world_idx}",
                )

                self.assertAlmostEqual(
                    float(actual_friction[2]),
                    expected_rolling_abs,
                    places=5,
                    msg=f"Updated rolling friction mismatch for shape {shape_idx} in world {world_idx}",
                )

    def test_geom_priority_conversion(self):
        """Test that geom_priority custom attribute is properly converted to MuJoCo."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        # Add two bodies with shapes
        body1 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        body2 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))

        # Add shapes with custom priority values
        builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1, custom_attributes={"mujoco:geom_priority": 1})
        builder.add_shape_box(body=body2, hx=0.1, hy=0.1, hz=0.1, custom_attributes={"mujoco:geom_priority": 0})

        # Add joints
        joint1 = builder.add_joint_revolute(-1, body1, axis=(0.0, 0.0, 1.0))
        joint2 = builder.add_joint_revolute(body1, body2, axis=(0.0, 1.0, 0.0))

        builder.add_articulation([joint1, joint2])
        model = builder.finalize()

        # Verify the custom attribute exists and has correct values
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "geom_priority"))

        geom_priority = model.mujoco.geom_priority.numpy()
        self.assertEqual(geom_priority[0], 1)
        self.assertEqual(geom_priority[1], 0)

        # Create solver and verify it's properly converted to MuJoCo
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify the MuJoCo model has the correct geom_priority values
        mjc_priority = solver.mjw_model.geom_priority.numpy()
        self.assertEqual(mjc_priority[0], 1)
        self.assertEqual(mjc_priority[1], 0)

    def test_geom_solimp_conversion_and_update(self):
        """Test per-shape geom_solimp conversion to MuJoCo and dynamic updates across multiple worlds."""
        # Create a model with custom attributes registered
        world_count = 2
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)

        # Create bodies with shapes
        body1 = template_builder.add_link(mass=0.1)
        template_builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
        joint1 = template_builder.add_joint_free(child=body1)

        body2 = template_builder.add_link(mass=0.1)
        template_builder.add_shape_sphere(body=body2, radius=0.1, cfg=shape_cfg)
        joint2 = template_builder.add_joint_revolute(parent=body1, child=body2, axis=(0.0, 0.0, 1.0))

        template_builder.add_articulation([joint1, joint2])

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace")
        self.assertTrue(hasattr(model.mujoco, "geom_solimp"), "Model should have geom_solimp attribute")

        # Use per-world iteration to handle potential global shapes correctly
        shape_world = model.shape_world.numpy()
        initial_solimp = np.zeros((model.shape_count, 5), dtype=np.float32)

        # Set unique solimp values per shape and world
        for world_idx in range(model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                initial_solimp[shape_idx] = [
                    0.8 + local_idx * 0.02 + world_idx * 0.05,  # dmin
                    0.9 + local_idx * 0.01 + world_idx * 0.02,  # dmax
                    0.001 + local_idx * 0.0005 + world_idx * 0.001,  # width
                    0.4 + local_idx * 0.05 + world_idx * 0.1,  # midpoint
                    2.0 + local_idx * 0.2 + world_idx * 0.5,  # power
                ]

        model.mujoco.geom_solimp.assign(wp.array(initial_solimp, dtype=vec5, device=model.device))

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        mjc_geom_to_newton_shape = solver.mjc_geom_to_newton_shape.numpy()
        num_geoms = solver.mj_model.ngeom

        # Verify initial conversion
        geom_solimp = solver.mjw_model.geom_solimp.numpy()
        tested_count = 0
        for world_idx in range(model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                if shape_idx < 0:
                    continue

                tested_count += 1
                expected_solimp = initial_solimp[shape_idx]
                actual_solimp = geom_solimp[world_idx, geom_idx]

                for i in range(5):
                    self.assertAlmostEqual(
                        float(actual_solimp[i]),
                        expected_solimp[i],
                        places=5,
                        msg=f"Initial geom_solimp[{i}] mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                    )

        self.assertGreater(tested_count, 0, "Should have tested at least one shape")

        # Update with different values
        updated_solimp = np.zeros((model.shape_count, 5), dtype=np.float32)

        for world_idx in range(model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                updated_solimp[shape_idx] = [
                    0.7 + local_idx * 0.03 + world_idx * 0.06,
                    0.85 + local_idx * 0.02 + world_idx * 0.03,
                    0.002 + local_idx * 0.0003 + world_idx * 0.0005,
                    0.5 + local_idx * 0.06 + world_idx * 0.08,
                    2.5 + local_idx * 0.3 + world_idx * 0.4,
                ]

        model.mujoco.geom_solimp.assign(wp.array(updated_solimp, dtype=vec5, device=model.device))

        solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

        # Verify updates
        updated_geom_solimp = solver.mjw_model.geom_solimp.numpy()

        for world_idx in range(model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = mjc_geom_to_newton_shape[world_idx, geom_idx]
                if shape_idx < 0:
                    continue

                expected_solimp = updated_solimp[shape_idx]
                actual_solimp = updated_geom_solimp[world_idx, geom_idx]

                for i in range(5):
                    self.assertAlmostEqual(
                        float(actual_solimp[i]),
                        expected_solimp[i],
                        places=5,
                        msg=f"Updated geom_solimp[{i}] mismatch for shape {shape_idx} in world {world_idx}",
                    )

    def test_geom_gap_conversion_and_update(self):
        """Test per-shape geom_gap conversion to MuJoCo and dynamic updates across multiple worlds."""

        # Create a model with custom attributes registered
        world_count = 2
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)

        # Create bodies with shapes
        body1 = template_builder.add_link(mass=0.1)
        template_builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
        joint1 = template_builder.add_joint_free(child=body1)

        body2 = template_builder.add_link(mass=0.1)
        template_builder.add_shape_sphere(body=body2, radius=0.1, cfg=shape_cfg)
        joint2 = template_builder.add_joint_revolute(parent=body1, child=body2, axis=(0.0, 0.0, 1.0))

        template_builder.add_articulation([joint1, joint2])

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace")
        self.assertTrue(hasattr(model.mujoco, "geom_gap"), "Model should have geom_gap attribute")

        # Use per-world iteration to handle potential global shapes correctly
        shape_world = model.shape_world.numpy()
        initial_gap = np.zeros(model.shape_count, dtype=np.float32)

        # Set unique gap values per shape and world
        for world_idx in range(model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                initial_gap[shape_idx] = 0.4 + local_idx * 0.2 + world_idx * 0.05

        model.mujoco.geom_gap.assign(wp.array(initial_gap, dtype=wp.float32, device=model.device))

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        to_newton_shape_index = solver.mjc_geom_to_newton_shape.numpy()
        num_geoms = solver.mj_model.ngeom

        # Verify initial conversion
        geom_gap = solver.mjw_model.geom_gap.numpy()
        tested_count = 0
        for world_idx in range(model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = to_newton_shape_index[world_idx, geom_idx]
                if shape_idx < 0:
                    continue

                tested_count += 1
                expected_gap = initial_gap[shape_idx]
                actual_gap = geom_gap[world_idx, geom_idx]

                self.assertAlmostEqual(
                    float(actual_gap),
                    expected_gap,
                    places=5,
                    msg=f"Initial geom_gap mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

        self.assertGreater(tested_count, 0, "Should have tested at least one shape")

        # Update with different values
        updated_gap = np.zeros(model.shape_count, dtype=np.float32)

        # Set unique gap values per shape and world
        for world_idx in range(model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                updated_gap[shape_idx] = 0.7 + local_idx * 0.03 + world_idx * 0.06

        model.mujoco.geom_gap.assign(wp.array(updated_gap, dtype=wp.float32, device=model.device))

        solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

        # Verify updates
        updated_geom_gap = solver.mjw_model.geom_gap.numpy()

        for world_idx in range(model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = to_newton_shape_index[world_idx, geom_idx]
                if shape_idx < 0:
                    continue

                expected_gap = updated_gap[shape_idx]
                actual_gap = updated_geom_gap[world_idx, geom_idx]

                self.assertAlmostEqual(
                    float(actual_gap),
                    expected_gap,
                    places=5,
                    msg=f"Updated geom_gap mismatch for shape {shape_idx} in world {world_idx}",
                )

    def test_geom_margin_from_thickness(self):
        """Test shape_thickness to geom_margin conversion and runtime updates.

        Verifies that shape_thickness [m] values are correctly propagated to
        geom_margin [m] during solver initialization and after runtime updates
        via notify_model_changed across multiple worlds.
        """
        num_worlds = 2
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, thickness=0.005)

        body1 = template_builder.add_link(mass=0.1)
        template_builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
        joint1 = template_builder.add_joint_free(child=body1)

        body2 = template_builder.add_link(mass=0.1)
        shape_cfg2 = newton.ModelBuilder.ShapeConfig(density=1000.0, thickness=0.01)
        template_builder.add_shape_sphere(body=body2, radius=0.1, cfg=shape_cfg2)
        joint2 = template_builder.add_joint_revolute(parent=body1, child=body2, axis=(0.0, 0.0, 1.0))
        template_builder.add_articulation([joint1, joint2])

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, num_worlds)
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        to_newton = solver.mjc_geom_to_newton_shape.numpy()
        num_geoms = solver.mj_model.ngeom

        # Verify initial conversion: geom_margin should match shape_thickness
        shape_thickness = model.shape_thickness.numpy()
        geom_margin = solver.mjw_model.geom_margin.numpy()
        tested_count = 0
        for world_idx in range(model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = to_newton[world_idx, geom_idx]
                if shape_idx < 0:
                    continue
                tested_count += 1
                self.assertAlmostEqual(
                    float(geom_margin[world_idx, geom_idx]),
                    float(shape_thickness[shape_idx]),
                    places=5,
                    msg=f"Initial geom_margin mismatch for shape {shape_idx} in world {world_idx}",
                )
        self.assertGreater(tested_count, 0)

        # Update thickness values at runtime
        new_thickness = np.array([0.02 + i * 0.005 for i in range(model.shape_count)], dtype=np.float32)
        model.shape_thickness.assign(wp.array(new_thickness, dtype=wp.float32, device=model.device))
        solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

        # Verify runtime update
        updated_margin = solver.mjw_model.geom_margin.numpy()
        for world_idx in range(model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = to_newton[world_idx, geom_idx]
                if shape_idx < 0:
                    continue
                self.assertAlmostEqual(
                    float(updated_margin[world_idx, geom_idx]),
                    float(new_thickness[shape_idx]),
                    places=5,
                    msg=f"Updated geom_margin mismatch for shape {shape_idx} in world {world_idx}",
                )

    def test_geom_solmix_conversion_and_update(self):
        """Test per-shape geom_solmix conversion to MuJoCo and dynamic updates across multiple worlds."""

        # Create a model with custom attributes registered
        world_count = 2
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)

        # Create bodies with shapes
        body1 = template_builder.add_link(mass=0.1)
        template_builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
        joint1 = template_builder.add_joint_free(child=body1)

        body2 = template_builder.add_link(mass=0.1)
        template_builder.add_shape_sphere(body=body2, radius=0.1, cfg=shape_cfg)
        joint2 = template_builder.add_joint_revolute(parent=body1, child=body2, axis=(0.0, 0.0, 1.0))

        template_builder.add_articulation([joint1, joint2])

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        self.assertTrue(hasattr(model, "mujoco"), "Model should have mujoco namespace")
        self.assertTrue(hasattr(model.mujoco, "geom_solmix"), "Model should have geom_solmix attribute")

        # Use per-world iteration to handle potential global shapes correctly
        shape_world = model.shape_world.numpy()
        initial_solmix = np.zeros(model.shape_count, dtype=np.float32)

        # Set unique solmix values per shape and world
        for world_idx in range(model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                initial_solmix[shape_idx] = 0.4 + local_idx * 0.2 + world_idx * 0.05

        model.mujoco.geom_solmix.assign(wp.array(initial_solmix, dtype=wp.float32, device=model.device))

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        to_newton_shape_index = solver.mjc_geom_to_newton_shape.numpy()
        num_geoms = solver.mj_model.ngeom

        # Verify initial conversion
        geom_solmix = solver.mjw_model.geom_solmix.numpy()
        tested_count = 0
        for world_idx in range(model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = to_newton_shape_index[world_idx, geom_idx]
                if shape_idx < 0:
                    continue

                tested_count += 1
                expected_solmix = initial_solmix[shape_idx]
                actual_solmix = geom_solmix[world_idx, geom_idx]

                self.assertAlmostEqual(
                    float(actual_solmix),
                    expected_solmix,
                    places=5,
                    msg=f"Initial geom_solmix mismatch for shape {shape_idx} in world {world_idx}, geom {geom_idx}",
                )

        self.assertGreater(tested_count, 0, "Should have tested at least one shape")

        # Update with different values
        updated_solmix = np.zeros(model.shape_count, dtype=np.float32)

        # Set unique solmix values per shape and world
        for world_idx in range(model.world_count):
            world_shape_indices = np.where(shape_world == world_idx)[0]
            for local_idx, shape_idx in enumerate(world_shape_indices):
                updated_solmix[shape_idx] = 0.7 + local_idx * 0.03 + world_idx * 0.06

        model.mujoco.geom_solmix.assign(wp.array(updated_solmix, dtype=wp.float32, device=model.device))

        solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

        # Verify updates
        updated_geom_solmix = solver.mjw_model.geom_solmix.numpy()

        for world_idx in range(model.world_count):
            for geom_idx in range(num_geoms):
                shape_idx = to_newton_shape_index[world_idx, geom_idx]
                if shape_idx < 0:
                    continue

                expected_solmix = updated_solmix[shape_idx]
                actual_solmix = updated_geom_solmix[world_idx, geom_idx]

                self.assertAlmostEqual(
                    float(actual_solmix),
                    expected_solmix,
                    places=5,
                    msg=f"Updated geom_solmix mismatch for shape {shape_idx} in world {world_idx}",
                )


class TestMuJoCoSolverEqualityConstraintProperties(TestMuJoCoSolverPropertiesBase):
    def test_eq_solref_conversion_and_update(self):
        """
        Test validation of eq_solref custom attribute:
        1. Initial conversion from Model to MuJoCo (multi-world)
        2. Runtime updates (multi-world)
        """
        # Create template with two articulations connected by an equality constraint
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)

        # Articulation 1: revolute joint from world
        b1 = template_builder.add_link()
        j1 = template_builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        template_builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j1])

        # Articulation 2: revolute joint from world (separate chain)
        b2 = template_builder.add_link()
        j2 = template_builder.add_joint_revolute(-1, b2, axis=(0, 0, 1))
        template_builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j2])

        # Add a connect constraint between the two bodies
        template_builder.add_equality_constraint_connect(
            body1=b1,
            body2=b2,
            anchor=wp.vec3(0.1, 0.0, 0.0),
        )

        # Create main builder with multiple worlds
        world_count = 2
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        # Verify we have the custom attribute
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "eq_solref"))
        self.assertEqual(model.equality_constraint_count, world_count)  # 1 constraint per world

        # --- Step 1: Set initial values and verify conversion ---

        total_eq = model.equality_constraint_count
        initial_values = np.zeros((total_eq, 2), dtype=np.float32)

        for i in range(total_eq):
            # Unique pattern for 2-element solref
            initial_values[i] = [
                0.01 + (i * 0.005) % 0.05,  # timeconst
                0.5 + (i * 0.2) % 1.5,  # dampratio
            ]

        model.mujoco.eq_solref.assign(initial_values)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Check mapping to MuJoCo
        mjc_eq_to_newton_eq = solver.mjc_eq_to_newton_eq.numpy()
        mjw_eq_solref = solver.mjw_model.eq_solref.numpy()

        neq = mjc_eq_to_newton_eq.shape[1]  # Number of MuJoCo equality constraints

        def check_values(expected_values, actual_mjw_values, msg_prefix):
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

        check_values(initial_values, mjw_eq_solref, "Initial conversion")

        # --- Step 2: Runtime Update ---

        # Generate new unique values
        updated_values = np.zeros((total_eq, 2), dtype=np.float32)
        for i in range(total_eq):
            updated_values[i] = [
                0.05 - (i * 0.005) % 0.04,  # timeconst
                2.0 - (i * 0.2) % 1.0,  # dampratio
            ]

        # Update model attribute
        model.mujoco.eq_solref.assign(updated_values)

        # Notify solver
        solver.notify_model_changed(SolverNotifyFlags.CONSTRAINT_PROPERTIES)

        # Verify updates
        mjw_eq_solref_updated = solver.mjw_model.eq_solref.numpy()

        check_values(updated_values, mjw_eq_solref_updated, "Runtime update")

        # Check that it is different from initial (sanity check)
        self.assertFalse(
            np.allclose(mjw_eq_solref_updated[0, 0], initial_values[0]),
            "Value did not change from initial!",
        )

    def test_eq_solimp_conversion_and_update(self):
        """
        Test validation of eq_solimp custom attribute:
        1. Initial conversion from Model to MuJoCo (multi-world)
        2. Runtime updates (multi-world)
        """
        # Create template with two articulations connected by an equality constraint
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)

        # Articulation 1: revolute joint from world
        b1 = template_builder.add_link()
        j1 = template_builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        template_builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j1])

        # Articulation 2: revolute joint from world (separate chain)
        b2 = template_builder.add_link()
        j2 = template_builder.add_joint_revolute(-1, b2, axis=(0, 0, 1))
        template_builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j2])

        # Add a connect constraint between the two bodies
        template_builder.add_equality_constraint_connect(
            body1=b1,
            body2=b2,
            anchor=wp.vec3(0.1, 0.0, 0.0),
        )

        # Create main builder with multiple worlds
        world_count = 2
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        # Verify we have the custom attribute
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "eq_solimp"))
        self.assertEqual(model.equality_constraint_count, world_count)  # 1 constraint per world

        # --- Step 1: Set initial values and verify conversion ---

        total_eq = model.equality_constraint_count
        initial_values = np.zeros((total_eq, 5), dtype=np.float32)

        for i in range(total_eq):
            # Unique pattern for 5-element solimp (dmin, dmax, width, midpoint, power)
            initial_values[i] = [
                0.85 + (i * 0.02) % 0.1,  # dmin
                0.92 + (i * 0.01) % 0.05,  # dmax
                0.001 + (i * 0.0005) % 0.005,  # width
                0.4 + (i * 0.05) % 0.2,  # midpoint
                1.8 + (i * 0.2) % 1.0,  # power
            ]

        model.mujoco.eq_solimp.assign(wp.array(initial_values, dtype=vec5, device=model.device))

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Check mapping to MuJoCo
        mjc_eq_to_newton_eq = solver.mjc_eq_to_newton_eq.numpy()
        mjw_eq_solimp = solver.mjw_model.eq_solimp.numpy()

        neq = mjc_eq_to_newton_eq.shape[1]  # Number of MuJoCo equality constraints

        def check_values(expected_values, actual_mjw_values, msg_prefix):
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

        check_values(initial_values, mjw_eq_solimp, "Initial conversion")

        # --- Step 2: Runtime Update ---

        # Generate new unique values
        updated_values = np.zeros((total_eq, 5), dtype=np.float32)
        for i in range(total_eq):
            updated_values[i] = [
                0.80 - (i * 0.02) % 0.08,  # dmin
                0.88 - (i * 0.01) % 0.04,  # dmax
                0.005 - (i * 0.0005) % 0.003,  # width
                0.55 - (i * 0.05) % 0.15,  # midpoint
                2.2 - (i * 0.2) % 0.8,  # power
            ]

        # Update model attribute
        model.mujoco.eq_solimp.assign(wp.array(updated_values, dtype=vec5, device=model.device))

        # Notify solver
        solver.notify_model_changed(SolverNotifyFlags.CONSTRAINT_PROPERTIES)

        # Verify updates
        mjw_eq_solimp_updated = solver.mjw_model.eq_solimp.numpy()

        check_values(updated_values, mjw_eq_solimp_updated, "Runtime update")

        # Check that it is different from initial (sanity check)
        self.assertFalse(
            np.allclose(mjw_eq_solimp_updated[0, 0], initial_values[0]),
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
        template_builder.add_equality_constraint_connect(
            body1=b1,
            body2=b2,
            anchor=wp.vec3(0.1, 0.2, 0.3),
        )

        # Add a WELD constraint with specific relpose values
        weld_relpose = wp.transform(wp.vec3(0.01, 0.02, 0.03), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.1))
        template_builder.add_equality_constraint_weld(
            body1=b2,
            body2=b3,
            anchor=wp.vec3(0.5, 0.6, 0.7),
            relpose=weld_relpose,
            torquescale=0.5,
        )

        # Add a JOINT constraint with specific polycoef values
        joint_polycoef = [0.1, 0.2, 0.3, 0.4, 0.5]
        template_builder.add_equality_constraint_joint(
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

        eq_constraint_anchor = model.equality_constraint_anchor.numpy()
        eq_constraint_relpose = model.equality_constraint_relpose.numpy()
        eq_constraint_polycoef = model.equality_constraint_polycoef.numpy()
        eq_constraint_torquescale = model.equality_constraint_torquescale.numpy()
        eq_constraint_type = model.equality_constraint_type.numpy()

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
        model.equality_constraint_anchor.assign(new_anchors[: model.equality_constraint_count])

        # Update torquescale for WELD constraints
        new_torquescale = np.array([0.0, 0.9, 0.0, 0.0, 0.8, 0.0], dtype=np.float32)
        model.equality_constraint_torquescale.assign(new_torquescale[: model.equality_constraint_count])

        # Update relpose for WELD constraints
        new_relpose = np.zeros((model.equality_constraint_count, 7), dtype=np.float32)
        # Set new relpose for WELD constraint (index 1 in template, indices 1 and 4 after replication)
        new_trans = [0.11, 0.22, 0.33]
        new_quat_xyzw = [0.0, 0.0, 0.38268343, 0.92387953]  # 45 degrees around Z
        new_relpose[1] = new_trans + new_quat_xyzw
        new_relpose[4] = new_trans + new_quat_xyzw
        model.equality_constraint_relpose.assign(new_relpose)

        # Update polycoef for JOINT constraints
        new_polycoef = np.zeros((model.equality_constraint_count, 5), dtype=np.float32)
        # Set new polycoef for JOINT constraint (index 2 in template, indices 2 and 5 after replication)
        new_polycoef[2] = [1.1, 1.2, 1.3, 1.4, 1.5]
        new_polycoef[5] = [1.1, 1.2, 1.3, 1.4, 1.5]
        model.equality_constraint_polycoef.assign(new_polycoef)

        # Notify solver
        solver.notify_model_changed(SolverNotifyFlags.CONSTRAINT_PROPERTIES)

        # Verify updates
        mjw_eq_data_updated = solver.mjw_model.eq_data.numpy()
        eq_constraint_anchor_updated = model.equality_constraint_anchor.numpy()
        eq_constraint_relpose_updated = model.equality_constraint_relpose.numpy()
        eq_constraint_polycoef_updated = model.equality_constraint_polycoef.numpy()
        eq_constraint_torquescale_updated = model.equality_constraint_torquescale.numpy()

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

    def test_eq_active_conversion_and_update(self):
        """
        Test validation of eq_active update from Newton equality_constraint_enabled:
        1. Initial conversion from Model to MuJoCo (multi-world)
        2. Runtime updates (multi-world) - toggling constraints on/off
        """
        # Create template with an equality constraint
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)

        # Articulation 1: free joint from world
        b1 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        j1 = template_builder.add_joint_free(child=b1)
        template_builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j1])

        # Articulation 2: free joint from world (separate chain)
        b2 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        j2 = template_builder.add_joint_free(child=b2)
        template_builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j2])

        # Add a connect constraint between the two bodies (enabled by default)
        template_builder.add_equality_constraint_connect(
            body1=b1,
            body2=b2,
            anchor=wp.vec3(0.1, 0.0, 0.0),
            enabled=True,
        )

        # Create main builder with multiple worlds
        world_count = 2
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        self.assertEqual(model.equality_constraint_count, world_count)  # 1 constraint per world

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # --- Step 1: Verify initial conversion - all enabled ---
        mjc_eq_to_newton_eq = solver.mjc_eq_to_newton_eq.numpy()
        mjw_eq_active = solver.mjw_data.eq_active.numpy()
        neq = mjc_eq_to_newton_eq.shape[1]

        eq_enabled = model.equality_constraint_enabled.numpy()

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

        # --- Step 2: Disable some constraints and verify ---
        # Disable constraint in world 0, keep world 1 enabled
        new_enabled = np.array([False, True], dtype=bool)
        model.equality_constraint_enabled.assign(new_enabled)

        # Notify solver
        solver.notify_model_changed(SolverNotifyFlags.CONSTRAINT_PROPERTIES)

        # Verify updates
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

        # --- Step 3: Re-enable all constraints ---
        new_enabled = np.array([True, True], dtype=bool)
        model.equality_constraint_enabled.assign(new_enabled)

        solver.notify_model_changed(SolverNotifyFlags.CONSTRAINT_PROPERTIES)

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

    def test_eq_data_connect_preserves_second_anchor(self):
        """
        Test that updating CONNECT constraint properties does not reset
        data[3:6] (second anchor) to zero.
        """
        # Create template with a CONNECT constraint
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)

        # Articulation 1: free joint from world
        b1 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        j1 = template_builder.add_joint_free(child=b1)
        template_builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j1])

        # Articulation 2: free joint from world (separate chain)
        b2 = template_builder.add_link(mass=1.0, com=wp.vec3(0.0), inertia=wp.mat33(np.eye(3)))
        j2 = template_builder.add_joint_free(child=b2)
        template_builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        template_builder.add_articulation([j2])

        # Add a CONNECT constraint between the two bodies
        template_builder.add_equality_constraint_connect(
            body1=b1,
            body2=b2,
            anchor=wp.vec3(0.1, 0.2, 0.3),
        )

        # Create main builder
        world_count = 2
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Capture the initial eq_data values computed by MuJoCo
        initial_eq_data = solver.mjw_model.eq_data.numpy().copy()

        # Notify solver to trigger the update kernel
        solver.notify_model_changed(SolverNotifyFlags.CONSTRAINT_PROPERTIES)

        # Verify data[3:6] (second anchor) was NOT overwritten
        updated_eq_data = solver.mjw_model.eq_data.numpy()
        for w in range(world_count):
            np.testing.assert_allclose(
                updated_eq_data[w, 0, 3:6],
                initial_eq_data[w, 0, 3:6],
                rtol=1e-5,
                err_msg=f"World {w}: CONNECT second anchor (data[3:6]) was incorrectly overwritten",
            )


class TestMuJoCoSolverFixedTendonProperties(TestMuJoCoSolverPropertiesBase):
    """Test fixed tendon property replication and runtime updates across multiple worlds."""

    def test_tendon_properties_conversion_and_update(self):
        """
        Test validation of fixed tendon custom attributes:
        1. Initial conversion from Model to MuJoCo (multi-world)
        2. Runtime updates via notify_model_changed (multi-world)
        """
        # Create template with tendons using MJCF
        mjcf = """<?xml version="1.0" ?>
<mujoco>
    <worldbody>
        <body name="root" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="link1" pos="0.0 -0.5 0">
                <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
                <geom type="cylinder" size="0.05 0.025"/>
                <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            </body>
            <body name="link2" pos="-0.0 -0.7 0">
                <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
                <geom type="cylinder" size="0.05 0.025"/>
                <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            </body>
        </body>
    </worldbody>
    <tendon>
        <fixed name="coupling_tendon" stiffness="1.0" damping="2.0" frictionloss="0.5">
            <joint joint="joint1" coef="1"/>
            <joint joint="joint2" coef="-1"/>
        </fixed>
    </tendon>
</mujoco>
"""

        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)
        template_builder.add_mjcf(mjcf)

        # Create main builder with multiple worlds
        world_count = 3
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template_builder, world_count)
        model = builder.finalize()

        # Verify we have the custom attributes
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "tendon_stiffness"))

        # Get the total number of tendons (1 per world)
        tendon_count = len(model.mujoco.tendon_stiffness)
        self.assertEqual(tendon_count, world_count)  # 1 tendon per world

        # --- Step 1: Set initial values and verify conversion ---

        # Set different values for each world's tendon
        initial_stiffness = np.array([1.0 + i * 0.5 for i in range(world_count)], dtype=np.float32)
        initial_damping = np.array([2.0 + i * 0.3 for i in range(world_count)], dtype=np.float32)
        initial_frictionloss = np.array([0.5 + i * 0.1 for i in range(world_count)], dtype=np.float32)

        model.mujoco.tendon_stiffness.assign(initial_stiffness)
        model.mujoco.tendon_damping.assign(initial_damping)
        model.mujoco.tendon_frictionloss.assign(initial_frictionloss)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Check mapping exists
        self.assertIsNotNone(solver.mjc_tendon_to_newton_tendon)

        # Get MuJoCo tendon values
        mjc_tendon_to_newton = solver.mjc_tendon_to_newton_tendon.numpy()
        mjw_stiffness = solver.mjw_model.tendon_stiffness.numpy()
        mjw_damping = solver.mjw_model.tendon_damping.numpy()
        mjw_frictionloss = solver.mjw_model.tendon_frictionloss.numpy()

        ntendon = mjc_tendon_to_newton.shape[1]  # Number of MuJoCo tendons per world

        def check_values(
            expected_stiff, expected_damp, expected_friction, actual_stiff, actual_damp, actual_friction, msg_prefix
        ):
            for w in range(world_count):
                for mjc_tendon in range(ntendon):
                    newton_tendon = mjc_tendon_to_newton[w, mjc_tendon]
                    if newton_tendon < 0:
                        continue

                    self.assertAlmostEqual(
                        float(actual_stiff[w, mjc_tendon]),
                        float(expected_stiff[newton_tendon]),
                        places=4,
                        msg=f"{msg_prefix} stiffness mismatch at World {w}, tendon {mjc_tendon}",
                    )
                    self.assertAlmostEqual(
                        float(actual_damp[w, mjc_tendon]),
                        float(expected_damp[newton_tendon]),
                        places=4,
                        msg=f"{msg_prefix} damping mismatch at World {w}, tendon {mjc_tendon}",
                    )
                    self.assertAlmostEqual(
                        float(actual_friction[w, mjc_tendon]),
                        float(expected_friction[newton_tendon]),
                        places=4,
                        msg=f"{msg_prefix} frictionloss mismatch at World {w}, tendon {mjc_tendon}",
                    )

        check_values(
            initial_stiffness,
            initial_damping,
            initial_frictionloss,
            mjw_stiffness,
            mjw_damping,
            mjw_frictionloss,
            "Initial conversion",
        )

        # --- Step 2: Runtime Update ---

        # Generate new unique values
        updated_stiffness = np.array([10.0 + i * 2.0 for i in range(world_count)], dtype=np.float32)
        updated_damping = np.array([5.0 + i * 1.0 for i in range(world_count)], dtype=np.float32)
        updated_frictionloss = np.array([1.0 + i * 0.2 for i in range(world_count)], dtype=np.float32)

        # Update model attributes
        model.mujoco.tendon_stiffness.assign(updated_stiffness)
        model.mujoco.tendon_damping.assign(updated_damping)
        model.mujoco.tendon_frictionloss.assign(updated_frictionloss)

        # Notify solver
        solver.notify_model_changed(SolverNotifyFlags.TENDON_PROPERTIES)

        # Verify updates
        mjw_stiffness_updated = solver.mjw_model.tendon_stiffness.numpy()
        mjw_damping_updated = solver.mjw_model.tendon_damping.numpy()
        mjw_frictionloss_updated = solver.mjw_model.tendon_frictionloss.numpy()

        check_values(
            updated_stiffness,
            updated_damping,
            updated_frictionloss,
            mjw_stiffness_updated,
            mjw_damping_updated,
            mjw_frictionloss_updated,
            "Runtime update",
        )

        # Check that values actually changed (sanity check)
        self.assertFalse(
            np.allclose(mjw_stiffness_updated[0, 0], initial_stiffness[0]),
            "Stiffness value did not change from initial!",
        )


class TestMuJoCoSolverNewtonContacts(unittest.TestCase):
    def setUp(self):
        """Set up a simple model with a sphere and a plane."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.add_ground_plane()

        self.sphere_radius = 0.5
        sphere_body_idx = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_sphere(
            body=sphere_body_idx,
            radius=self.sphere_radius,
        )

        self.model = builder.finalize()
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.model.collide(self.state_in, self.contacts)
        self.sphere_body_idx = sphere_body_idx

    def test_sphere_on_plane_with_newton_contacts(self):
        """Test that a sphere correctly collides with a plane using Newton contacts."""
        try:
            solver = SolverMuJoCo(self.model, use_mujoco_contacts=False)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")
            return

        sim_dt = 1.0 / 240.0
        num_steps = 120  # Simulate for 0.5 seconds to ensure it settles

        self.contacts = self.model.contacts()
        for _ in range(num_steps):
            self.model.collide(self.state_in, self.contacts)
            solver.step(self.state_in, self.state_out, self.control, self.contacts, sim_dt)
            self.state_in, self.state_out = self.state_out, self.state_in

        final_pos = self.state_in.body_q.numpy()[self.sphere_body_idx, :3]
        final_height = final_pos[2]  # Z-up in MuJoCo

        # The sphere should settle on the plane, with its center at its radius's height
        self.assertGreater(
            final_height,
            self.sphere_radius * 0.9,
            f"Sphere fell through the plane. Final height: {final_height}",
        )
        self.assertLess(
            final_height,
            self.sphere_radius * 1.2,
            f"Sphere is floating above the plane. Final height: {final_height}",
        )


class TestMuJoCoValidation(unittest.TestCase):
    """Test cases for SolverMuJoCo._validate_model_for_separate_worlds()."""

    def _create_homogeneous_model(self, world_count=2, with_ground_plane=True):
        """Create a valid homogeneous multi-world model for validation tests."""
        # Create a simple robot template (following pattern from working tests)
        template = newton.ModelBuilder()
        b1 = template.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        b2 = template.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        j1 = template.add_joint_revolute(-1, b1, axis=(0.0, 0.0, 1.0))
        j2 = template.add_joint_revolute(b1, b2, axis=(0.0, 0.0, 1.0))
        template.add_articulation([j1, j2])
        template.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        template.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)

        # Build main model using replicate (pattern from working tests)
        builder = newton.ModelBuilder()
        if with_ground_plane:
            builder.add_ground_plane()  # Global static shape
        builder.replicate(template, world_count)

        return builder.finalize()

    def test_valid_homogeneous_model_passes(self):
        """Test that a valid homogeneous model passes validation."""
        model = self._create_homogeneous_model(world_count=2, with_ground_plane=False)
        # Should not raise
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertIsNotNone(solver)

    def test_valid_model_with_global_shape_passes(self):
        """Test that a model with global static shapes (ground plane) passes validation."""
        model = self._create_homogeneous_model(world_count=2, with_ground_plane=True)
        # Should not raise - global shapes are allowed
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertIsNotNone(solver)

    def test_heterogeneous_body_count_fails(self):
        """Test that different body counts per world raises ValueError."""
        # Create two robots with different body counts
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)
        robot1.add_articulation([j1])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_revolute(-1, b1)
        j2 = robot2.add_joint_revolute(b1, b2)
        robot2.add_articulation([j1, j2])
        robot2.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot2.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)

        main = newton.ModelBuilder()
        main.add_world(robot1)  # 1 body
        main.add_world(robot2)  # 2 bodies
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("world 0 has 1 bodies", str(ctx.exception).lower())

    def test_heterogeneous_shape_count_fails(self):
        """Test that different shape counts per world raises ValueError."""
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)
        robot1.add_articulation([j1])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_revolute(-1, b1)
        robot2.add_articulation([j1])
        robot2.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot2.add_shape_sphere(b1, radius=0.05)  # Extra shape

        main = newton.ModelBuilder()
        main.add_world(robot1)  # 1 shape
        main.add_world(robot2)  # 2 shapes
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("world 0 has 1 shapes", str(ctx.exception).lower())

    def test_mismatched_joint_types_fails(self):
        """Test that different joint types at same position across worlds raises ValueError."""
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)  # Revolute joint
        robot1.add_articulation([j1])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_prismatic(-1, b1)  # Prismatic joint (different type)
        robot2.add_articulation([j1])
        robot2.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)

        main = newton.ModelBuilder()
        main.add_world(robot1)
        main.add_world(robot2)
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("joint types mismatch at position", str(ctx.exception).lower())

    def test_mismatched_shape_types_fails(self):
        """Test that different shape types at same position across worlds raises ValueError."""
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)
        robot1.add_articulation([j1])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)  # Box

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_revolute(-1, b1)
        robot2.add_articulation([j1])
        robot2.add_shape_sphere(b1, radius=0.1)  # Sphere (different type)

        main = newton.ModelBuilder()
        main.add_world(robot1)
        main.add_world(robot2)
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("shape types mismatch at position", str(ctx.exception).lower())

    def test_global_body_fails(self):
        """Test that a body in global world (-1) raises ValueError."""
        builder = newton.ModelBuilder()

        # Add ground plane (allowed)
        builder.add_ground_plane()

        # Manually create a body in global world
        builder.current_world = -1
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        # Need a joint to make this a valid model
        builder.current_world = -1
        j1 = builder.add_joint_free(b1)
        builder.add_articulation([j1])

        # Add normal world content
        builder.begin_world()
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j2 = builder.add_joint_revolute(-1, b2)
        builder.add_articulation([j2])
        builder.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()

        builder.begin_world()
        b3 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j3 = builder.add_joint_revolute(-1, b3)
        builder.add_articulation([j3])
        builder.add_shape_box(b3, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()

        model = builder.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("global world (-1) cannot contain bodies", str(ctx.exception).lower())

    def test_global_joint_fails(self):
        """Test that a joint in global world (-1) raises ValueError."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Add a body in global world with a joint
        builder.current_world = -1
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = builder.add_joint_revolute(-1, b1)
        builder.add_articulation([j1])

        # Add normal world content
        builder.begin_world()
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j2 = builder.add_joint_revolute(-1, b2)
        builder.add_articulation([j2])
        builder.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()

        builder.begin_world()
        b3 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j3 = builder.add_joint_revolute(-1, b3)
        builder.add_articulation([j3])
        builder.add_shape_box(b3, hx=0.1, hy=0.1, hz=0.1)
        builder.end_world()

        model = builder.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        # Fails on global bodies first (bodies are checked before joints)
        self.assertIn("global world (-1) cannot contain", str(ctx.exception).lower())

    def test_single_world_model_skips_validation(self):
        """Test that single-world models skip validation (no homogeneity needed)."""
        model = self._create_homogeneous_model(world_count=1)

        # Should not raise - single world doesn't need homogeneity validation
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertIsNotNone(solver)

    def test_many_worlds_homogeneous_passes(self):
        """Test that a model with many homogeneous worlds passes validation."""
        model = self._create_homogeneous_model(world_count=10)
        # Should not raise
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertIsNotNone(solver)

    def test_heterogeneous_equality_constraint_count_fails(self):
        """Test that different equality constraint counts per world raises ValueError."""
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)
        j2 = robot1.add_joint_revolute(b1, b2)
        robot1.add_articulation([j1, j2])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot1.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        robot1.add_equality_constraint_weld(body1=b1, body2=b2)  # 1 constraint

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_revolute(-1, b1)
        j2 = robot2.add_joint_revolute(b1, b2)
        robot2.add_articulation([j1, j2])
        robot2.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot2.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        # No constraints in robot2

        main = newton.ModelBuilder()
        main.add_world(robot1)  # 1 constraint
        main.add_world(robot2)  # 0 constraints
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("world 0 has 1 equality constraints", str(ctx.exception).lower())

    def test_mismatched_equality_constraint_types_fails(self):
        """Test that different constraint types at same position across worlds raises ValueError."""
        robot1 = newton.ModelBuilder()
        b1 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot1.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot1.add_joint_revolute(-1, b1)
        j2 = robot1.add_joint_revolute(b1, b2)
        robot1.add_articulation([j1, j2])
        robot1.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot1.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        robot1.add_equality_constraint_weld(body1=b1, body2=b2)  # WELD type

        robot2 = newton.ModelBuilder()
        b1 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot2.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot2.add_joint_revolute(-1, b1)
        j2 = robot2.add_joint_revolute(b1, b2)
        robot2.add_articulation([j1, j2])
        robot2.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot2.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)
        robot2.add_equality_constraint_connect(body1=b1, body2=b2)  # CONNECT type (different)

        main = newton.ModelBuilder()
        main.add_world(robot1)
        main.add_world(robot2)
        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("equality constraint types mismatch at position", str(ctx.exception).lower())

    def test_global_equality_constraint_fails(self):
        """Test that an equality constraint in global world (-1) raises ValueError."""
        # Create a model with a global equality constraint
        robot = newton.ModelBuilder()
        b1 = robot.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        b2 = robot.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        j1 = robot.add_joint_revolute(-1, b1)
        j2 = robot.add_joint_revolute(b1, b2)
        robot.add_articulation([j1, j2])
        robot.add_shape_box(b1, hx=0.1, hy=0.1, hz=0.1)
        robot.add_shape_box(b2, hx=0.1, hy=0.1, hz=0.1)

        main = newton.ModelBuilder()
        main.add_world(robot)
        main.add_world(robot)

        # Add a global equality constraint
        main.current_world = -1
        # We need body indices in the main builder - use the first two bodies from world 0
        main.add_equality_constraint_weld(body1=0, body2=1)

        model = main.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverMuJoCo(model, separate_worlds=True)
        self.assertIn("global world (-1) cannot contain equality constraints", str(ctx.exception).lower())

    def test_body_missing_joint(self):
        """Ensure that each body has an incoming joint and these joints are part of an articulation."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        b0 = builder.add_link()
        b1 = builder.add_link()
        j0 = builder.add_joint_revolute(-1, b0)
        builder.add_joint_revolute(b0, b1)
        builder.add_articulation([j0])
        builder.end_world()
        # we forgot to add the second joint to the articulation
        # finalize() should now catch this and raise an error about orphan joints
        with self.assertRaises(ValueError) as ctx:
            builder.finalize()
        self.assertIn("not belonging to any articulation", str(ctx.exception))


class TestMuJoCoConversion(unittest.TestCase):
    def test_no_shapes_separate_worlds_false(self):
        """Testing that an articulation without any shapes can be converted successfully when setting separate_worlds=False."""
        builder = newton.ModelBuilder()
        # force the ModelBuilder to correct zero mass/inertia values
        builder.bound_inertia = 0.01
        builder.bound_mass = 0.01
        b0 = builder.add_link()
        b1 = builder.add_link()
        j0 = builder.add_joint_revolute(-1, b0)
        j1 = builder.add_joint_revolute(b0, b1)
        builder.add_articulation([j0, j1])
        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=False)
        self.assertEqual(solver.mj_model.nv, 2)

    def test_no_shapes_separate_worlds_true(self):
        """Testing that an articulation without any shapes can be converted successfully when setting separate_worlds=True."""
        builder = newton.ModelBuilder()
        # force the ModelBuilder to correct zero mass/inertia values
        builder.bound_inertia = 0.01
        builder.bound_mass = 0.01
        builder.begin_world()
        b0 = builder.add_link()
        b1 = builder.add_link()
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
            return

        # Run forward kinematics using mujoco_warp (skip if not available)
        try:
            import mujoco_warp

            mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)
        except ImportError as e:
            self.skipTest(f"mujoco_warp not installed. Skipping test: {e}")
            return

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
        contacts_soft = model_soft.contacts()
        model_soft.collide(state_soft_in, contacts_soft)
        contacts_stiff = model_stiff.contacts()
        model_stiff.collide(state_stiff_in, contacts_stiff)

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
        solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES)

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
        solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES)

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
        body1 = env1.add_body(key="body1", mass=1.0)  # Add mass to make it dynamic

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
        body2 = env2.add_body(key="body2", mass=1.0)  # Add mass to make it dynamic

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
        body1 = env1.add_body(key="mesh_body1", mass=1.0)

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
        body2 = env2.add_body(key="mesh_body2", mass=1.0)

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


class TestMuJoCoMocapBodies(unittest.TestCase):
    def test_mocap_body_transform_updates_collision_geoms(self):
        """
        Test that mocap bodies (fixed-base articulations) correctly update collision geometry
        when their joint transforms change.

        Setup:
        - Fixed-base (mocap) body at root
        - Welded/fixed descendant body with collision geometry
        - Dynamic ball resting on the descendant body

        Test:
        - Rotate and translate the mocap body (update joint transform)
        - Verify mocap_pos/mocap_quat are correctly updated in MuJoCo arrays
        - Step simulation and verify ball falls (collision geometry moved, contact lost)
        """
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.default_shape_cfg.mu = 0.5

        # Create fixed-base (mocap) body at root (at origin)
        # This body will have a FIXED joint to the world, making it a mocap body in MuJoCo
        mocap_body = builder.add_link(
            mass=10.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )

        # Add FIXED joint to world - this makes it a mocap body
        mocap_joint = builder.add_joint_fixed(
            parent=-1,
            child=mocap_body,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )

        # Create welded/fixed descendant body with collision geometry (platform)
        # Offset horizontally (X direction) from mocap body, at height 0.5m
        platform_body = builder.add_link(
            mass=5.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3)),
        )

        # Add FIXED joint from mocap body to platform (welded connection)
        # Platform is offset in +X direction by 1m and up in +Z by 0.5m
        platform_joint = builder.add_joint_fixed(
            parent=mocap_body,
            child=platform_body,
            parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )

        # Add collision box to platform (thin platform)
        platform_height = 0.1
        builder.add_shape_box(
            body=platform_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            hx=1.0,
            hy=1.0,
            hz=platform_height,
        )

        # Add mocap articulation
        builder.add_articulation([mocap_joint, platform_joint])

        # Create dynamic ball resting on the platform
        # Position it above the platform at (1.0, 0, 0.5 + platform_height + ball_radius)
        ball_radius = 0.2
        ball_body = builder.add_body(
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3) * 0.01),
            xform=wp.transform(wp.vec3(1.0, 0.0, 0.5 + platform_height + ball_radius), wp.quat_identity()),
        )
        builder.add_shape_sphere(
            body=ball_body,
            radius=ball_radius,
        )

        model = builder.finalize()

        # Create MuJoCo solver
        try:
            solver = SolverMuJoCo(model, use_mujoco_contacts=True)
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping test: {e}")
            return

        # Verify mocap body was created using MuJoCo's body_mocapid
        body_mocapid = solver.mjw_model.body_mocapid.numpy()
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()

        # Find MuJoCo body indices for our Newton bodies by searching the mapping
        def find_mjc_body(newton_body):
            for b in range(mjc_body_to_newton.shape[1]):
                if mjc_body_to_newton[0, b] == newton_body:
                    return b
            return -1

        mjc_mocap_body = find_mjc_body(mocap_body)
        mjc_platform_body = find_mjc_body(platform_body)
        mjc_ball_body = find_mjc_body(ball_body)

        # mocap_body should have a valid mocap index (>= 0)
        mocap_index = body_mocapid[mjc_mocap_body]
        self.assertGreaterEqual(mocap_index, 0, f"mocap_body should be a mocap body, got index {mocap_index}")

        # platform_body and ball_body should NOT be mocap bodies (-1)
        self.assertEqual(body_mocapid[mjc_platform_body], -1, "platform_body should not be a mocap body")
        self.assertEqual(body_mocapid[mjc_ball_body], -1, "ball_body should not be a mocap body")

        # Setup simulation
        state_in = model.state()
        state_out = model.state()
        control = model.control()

        sim_dt = 1.0 / 240.0

        # Let ball settle on platform
        for _ in range(5):
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in

        # Verify ball is resting on platform (should have contacts)
        initial_n_contacts = int(solver.mjw_data.nacon.numpy()[0])
        self.assertGreater(initial_n_contacts, 0, "Ball should be in contact with platform initially")

        # Record initial ball state
        initial_ball_height = state_in.body_q.numpy()[ball_body, 2]
        initial_ball_velocity_z = state_in.body_qd.numpy()[ball_body, 2]

        # Verify ball is at rest (vertical velocity near zero)
        self.assertAlmostEqual(
            initial_ball_velocity_z,
            0.0,
            delta=0.001,
            msg=f"Ball should be at rest initially, got Z velocity {initial_ball_velocity_z}",
        )

        # Get initial mocap_pos/mocap_quat for verification
        initial_mocap_pos = solver.mjw_data.mocap_pos.numpy()[0, mocap_index].copy()
        initial_mocap_quat = solver.mjw_data.mocap_quat.numpy()[0, mocap_index].copy()

        # Rotate mocap body by 90 degrees around Z-axis (vertical) and translate slightly
        # Since platform is offset in +X from mocap, after 90° Z rotation it becomes offset in +Y
        # This swings the platform away horizontally, leaving the ball with no support
        # Add small translation to verify mocap_pos is updated correctly
        rotation_angle = wp.pi / 2  # 90 degrees
        rotation_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rotation_angle)
        new_position = wp.vec3(0.1, 0.2, 0.0)  # Small translation for verification
        new_parent_xform = wp.transform(new_position, rotation_quat)

        # Update the mocap body's joint transform
        model.joint_X_p.assign([new_parent_xform])

        # Notify solver that joint properties changed
        solver.notify_model_changed(SolverNotifyFlags.JOINT_PROPERTIES)

        # Verify mocap_pos was updated correctly
        updated_mocap_pos = solver.mjw_data.mocap_pos.numpy()[0, mocap_index]
        updated_mocap_quat = solver.mjw_data.mocap_quat.numpy()[0, mocap_index]

        # Check that position changed
        pos_changed = not np.allclose(initial_mocap_pos, updated_mocap_pos, atol=1e-6)
        self.assertTrue(pos_changed, "mocap_pos should be updated after transform change")

        # Verify position was updated to new position
        np.testing.assert_allclose(
            updated_mocap_pos,
            [new_position.x, new_position.y, new_position.z],
            atol=1e-5,
            err_msg="mocap_pos should match the new position",
        )

        # Check that quaternion changed
        quat_changed = not np.allclose(initial_mocap_quat, updated_mocap_quat, atol=1e-6)
        self.assertTrue(quat_changed, "mocap_quat should be updated after rotation")

        # Verify the rotation is approximately correct (90 degrees around Y)
        expected_quat_mjc = np.array([rotation_quat.w, rotation_quat.x, rotation_quat.y, rotation_quat.z])
        # Account for potential quaternion sign flip
        if np.dot(updated_mocap_quat, expected_quat_mjc) < 0:
            expected_quat_mjc = -expected_quat_mjc
        np.testing.assert_allclose(
            updated_mocap_quat, expected_quat_mjc, atol=1e-5, err_msg="mocap_quat should match the rotation"
        )

        # Simulate and verify ball falls (collision geometry moved with mocap body)
        for _ in range(10):
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in

        # Verify ball has fallen (lost contact and dropped in height)
        final_ball_height = state_in.body_q.numpy()[ball_body, 2]
        final_ball_velocity_z = state_in.body_qd.numpy()[ball_body, 2]
        final_n_contacts = int(solver.mjw_data.nacon.numpy()[0])

        # Ball should have fallen below initial height
        self.assertLess(
            final_ball_height,
            initial_ball_height,
            f"Ball should have fallen after platform rotated. Initial: {initial_ball_height:.3f}, Final: {final_ball_height:.3f}",
        )

        # Ball should have significant downward (negative Z) velocity
        self.assertLess(
            final_ball_velocity_z,
            -0.2,
            f"Ball should be falling with downward velocity, got {final_ball_velocity_z:.3f} m/s",
        )

        # Ball should have zero contacts (platform moved away)
        self.assertEqual(
            final_n_contacts,
            0,
            f"Ball should have no contacts after platform rotated away, got {final_n_contacts} contacts",
        )


class TestMuJoCoAttributes(unittest.TestCase):
    def test_custom_attributes_from_code(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        b0 = builder.add_link()
        j0 = builder.add_joint_revolute(-1, b0, axis=(0.0, 0.0, 1.0))
        builder.add_shape_box(body=b0, hx=0.1, hy=0.1, hz=0.1, custom_attributes={"mujoco:condim": 6})
        b1 = builder.add_link()
        j1 = builder.add_joint_revolute(b0, b1, axis=(0.0, 0.0, 1.0))
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1, custom_attributes={"mujoco:condim": 4})
        b2 = builder.add_link()
        j2 = builder.add_joint_revolute(b1, b2, axis=(0.0, 0.0, 1.0))
        builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([j0, j1, j2])
        model = builder.finalize()

        # Should work fine with single world
        solver = SolverMuJoCo(model, separate_worlds=False)

        assert hasattr(model, "mujoco")
        assert hasattr(model.mujoco, "condim")
        assert np.allclose(model.mujoco.condim.numpy(), [6, 4, 3])
        assert np.allclose(solver.mjw_model.geom_condim.numpy(), [6, 4, 3])

    def test_custom_attributes_from_mjcf(self):
        mjcf = """
        <mujoco>
            <worldbody>
                <body>
                    <joint type="hinge" axis="0 0 1" />
                    <geom type="box" size="0.1 0.1 0.1" condim="6" />
                </body>
                <body>
                    <joint type="hinge" axis="0 0 1" />
                    <geom type="box" size="0.1 0.1 0.1" condim="4" />
                </body>
                <body>
                    <joint type="hinge" axis="0 0 1" />
                    <geom type="box" size="0.1 0.1 0.1" />
                </body>
            </worldbody>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=False)
        assert hasattr(model, "mujoco")
        assert hasattr(model.mujoco, "condim")
        assert np.allclose(model.mujoco.condim.numpy(), [6, 4, 3])
        assert np.allclose(solver.mjw_model.geom_condim.numpy(), [6, 4, 3])

    def test_custom_attributes_from_urdf(self):
        urdf = """
        <robot name="test_robot">
            <link name="body1">
                <joint type="revolute" axis="0 0 1" />
                <collision>
                    <geometry condim="6">
                        <box size="0.1 0.1 0.1" />
                    </geometry>
                </collision>
            </link>
            <link name="body2">
                <joint type="revolute" axis="0 0 1" />
                <collision>
                    <geometry condim="4">
                        <box size="0.1 0.1 0.1" />
                    </geometry>
                </collision>
            </link>
            <link name="body3">
                <joint type="revolute" axis="0 0 1" />
                <collision>
                    <geometry>
                        <box size="0.1 0.1 0.1" />
                    </geometry>
                </collision>
            </link>
            <joint name="joint1" type="revolute">
                <parent link="body1" />
                <child link="body2" />
            </joint>
            <joint name="joint2" type="revolute">
                <parent link="body2" />
                <child link="body3" />
            </joint>
        </robot>
        """
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_urdf(urdf)
        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=False)
        assert hasattr(model, "mujoco")
        assert hasattr(model.mujoco, "condim")
        assert np.allclose(model.mujoco.condim.numpy(), [6, 4, 3])
        assert np.allclose(solver.mjw_model.geom_condim.numpy(), [6, 4, 3])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_custom_attributes_from_usd(self):
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        self.assertTrue(stage)

        body_path = "/body"
        shape = UsdGeom.Cube.Define(stage, body_path)
        prim = shape.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int, True).Set(6)

        joint_path = "/joint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateAxisAttr().Set("Z")
        joint.CreateBody0Rel().SetTargets([body_path])

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=False)
        assert hasattr(model, "mujoco")
        assert hasattr(model.mujoco, "condim")
        assert np.allclose(model.mujoco.condim.numpy(), [6])
        assert np.allclose(solver.mjw_model.geom_condim.numpy(), [6])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_mjc_damping_from_usd_via_schema_resolver(self):
        """Test mjc:damping attributes are parsed via SchemaResolverMjc."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        from newton._src.usd.schemas import SchemaResolverMjc  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        # Create root body
        root_path = "/robot"
        root_shape = UsdGeom.Cube.Define(stage, root_path)
        root_prim = root_shape.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        UsdPhysics.CollisionAPI.Apply(root_prim)

        # Create child body
        child_path = "/robot/child"
        child_shape = UsdGeom.Cube.Define(stage, child_path)
        child_prim = child_shape.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(child_prim)
        UsdPhysics.CollisionAPI.Apply(child_prim)

        # Create joint with mjc:damping
        joint_path = "/robot/child/joint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateAxisAttr().Set("Z")
        joint.CreateBody0Rel().SetTargets([root_path])
        joint.CreateBody1Rel().SetTargets([child_path])
        joint_prim = joint.GetPrim()
        joint_prim.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Double, True).Set(5.0)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage, schema_resolvers=[SchemaResolverMjc()])
        model = builder.finalize()

        assert hasattr(model, "mujoco")
        assert hasattr(model.mujoco, "dof_passive_damping")
        damping_values = model.mujoco.dof_passive_damping.numpy()
        # 6 DOFs from floating base (all 0.0) + 1 DOF from revolute joint (5.0)
        assert damping_values[-1] == 5.0, f"Expected last DOF damping to be 5.0, got {damping_values}"

    def test_ref_coordinate_conversion(self):
        """Verify ref offset in coordinate conversion.

        With a hinge joint at ref=90 degrees, setting joint_q=0 in Newton
        should produce qpos=pi/2 in MuJoCo after update_mjc_data.
        """
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_ref">
    <worldbody>
        <body name="base">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child" pos="0 0 1">
                <joint name="hinge" type="hinge" axis="0 1 0" ref="90"/>
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        # joint_q=0 should map to qpos=ref (pi/2)
        state = model.state()
        solver.update_mjc_data(solver.mjw_data, model, state)
        qpos = solver.mjw_data.qpos.numpy()
        np.testing.assert_allclose(qpos[0, 0], np.pi / 2, atol=1e-5, err_msg="joint_q=0 should map to qpos=ref")


class TestMuJoCoOptions(unittest.TestCase):
    """Tests for MuJoCo solver options (impratio, etc.) with WORLD frequency."""

    def _create_multiworld_model(self, world_count=3):
        """Helper to create a multi-world model with MuJoCo custom attributes registered."""
        template_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template_builder)

        pendulum = template_builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        template_builder.add_shape_box(body=pendulum, hx=0.05, hy=0.05, hz=0.05)
        joint = template_builder.add_joint_revolute(parent=-1, child=pendulum, axis=(0.0, 0.0, 1.0))
        template_builder.add_articulation([joint])

        builder = newton.ModelBuilder()
        builder.replicate(template_builder, world_count)
        return builder.finalize()

    def test_impratio_multiworld_conversion(self):
        """
        Verify that impratio custom attribute with WORLD frequency:
        1. Is properly registered and exists on the model.
        2. The array has correct shape (one value per world).
        3. Different per-world values are stored correctly in the Newton model.
        4. Solver expands per-world values to MuJoCo Warp.
        """
        world_count = 3
        model = self._create_multiworld_model(world_count)

        # Verify the custom attribute is registered and exists on the model
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "impratio"))

        # Verify the array has correct shape (one value per world)
        impratio = model.mujoco.impratio.numpy()
        self.assertEqual(len(impratio), world_count, "impratio array should have one entry per world")

        # Set different impratio values per world
        initial_impratio = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        model.mujoco.impratio.assign(initial_impratio)

        # Verify all per-world values are stored correctly in Newton model
        updated_impratio = model.mujoco.impratio.numpy()
        for world_idx in range(world_count):
            self.assertAlmostEqual(
                updated_impratio[world_idx],
                initial_impratio[world_idx],
                places=4,
                msg=f"Newton model impratio[{world_idx}] should be {initial_impratio[world_idx]}",
            )

        # Create solver without constructor override
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify MuJoCo Warp model has per-world impratio_invsqrt values
        mjw_impratio_invsqrt = solver.mjw_model.opt.impratio_invsqrt.numpy()
        self.assertEqual(
            len(mjw_impratio_invsqrt),
            world_count,
            f"MuJoCo Warp opt.impratio_invsqrt should have {world_count} values (one per world)",
        )

        # Verify each world has the correct impratio_invsqrt value (1/sqrt(impratio))
        for world_idx in range(world_count):
            expected_invsqrt = 1.0 / np.sqrt(initial_impratio[world_idx])
            self.assertAlmostEqual(
                mjw_impratio_invsqrt[world_idx],
                expected_invsqrt,
                places=4,
                msg=f"MuJoCo Warp impratio_invsqrt[{world_idx}] should be {expected_invsqrt}",
            )

    def test_impratio_invalid_values_guarded(self):
        """
        Verify that zero or negative impratio values are guarded against
        to prevent NaN/Inf in opt_impratio_invsqrt computation.
        """
        world_count = 3
        model = self._create_multiworld_model(world_count)

        # Set impratio with invalid values: 0, negative, and positive
        initial_impratio = np.array([0.0, -1.0, 2.0], dtype=np.float32)
        model.mujoco.impratio.assign(initial_impratio)

        # Create solver - should not crash or produce NaN/Inf
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify MuJoCo Warp model has valid impratio_invsqrt values
        mjw_impratio_invsqrt = solver.mjw_model.opt.impratio_invsqrt.numpy()
        self.assertEqual(len(mjw_impratio_invsqrt), world_count)

        # World 0 (impratio=0): should keep MuJoCo default (not update)
        self.assertFalse(
            np.isnan(mjw_impratio_invsqrt[0]),
            "impratio=0 should not produce NaN",
        )
        self.assertFalse(
            np.isinf(mjw_impratio_invsqrt[0]),
            "impratio=0 should not produce Inf",
        )

        # World 1 (impratio=-1): should keep MuJoCo default (not update)
        self.assertFalse(
            np.isnan(mjw_impratio_invsqrt[1]),
            "impratio=-1 should not produce NaN",
        )
        self.assertFalse(
            np.isinf(mjw_impratio_invsqrt[1]),
            "impratio=-1 should not produce Inf",
        )

        # World 2 (impratio=2): should compute correctly
        expected_invsqrt = 1.0 / np.sqrt(2.0)
        self.assertAlmostEqual(
            mjw_impratio_invsqrt[2],
            expected_invsqrt,
            places=4,
            msg=f"impratio=2.0 should produce valid impratio_invsqrt={expected_invsqrt}",
        )

    def test_scalar_options_constructor_override(self):
        """
        Verify that passing scalar options (impratio, tolerance, ls_tolerance, ccd_tolerance, density, viscosity)
        to the SolverMuJoCo constructor overrides any per-world values from custom attributes.
        """
        world_count = 2
        model = self._create_multiworld_model(world_count)

        # Set custom attribute values per world
        model.mujoco.impratio.assign(np.array([1.5, 1.5], dtype=np.float32))
        model.mujoco.tolerance.assign(np.array([1e-6, 1e-7], dtype=np.float32))
        model.mujoco.ls_tolerance.assign(np.array([0.01, 0.02], dtype=np.float32))
        model.mujoco.ccd_tolerance.assign(np.array([1e-6, 1e-7], dtype=np.float32))
        model.mujoco.density.assign(np.array([0.0, 0.0], dtype=np.float32))
        model.mujoco.viscosity.assign(np.array([0.0, 0.0], dtype=np.float32))

        # Create solver WITH constructor overrides
        # NOTE: density and viscosity must be 0 to avoid triggering MuJoCo Warp's
        # "fluid model not implemented" error. Non-zero values enable fluid dynamics.
        solver = SolverMuJoCo(
            model,
            impratio=3.0,
            tolerance=1e-5,
            ls_tolerance=0.001,
            ccd_tolerance=1e-4,
            density=0.0,
            viscosity=0.0,
            iterations=1,
            disable_contacts=True,
        )

        # Verify MuJoCo Warp uses constructor-provided values (tiled to all worlds)
        mjw_impratio_invsqrt = solver.mjw_model.opt.impratio_invsqrt.numpy()
        mjw_tolerance = solver.mjw_model.opt.tolerance.numpy()
        mjw_ls_tolerance = solver.mjw_model.opt.ls_tolerance.numpy()
        mjw_ccd_tolerance = solver.mjw_model.opt.ccd_tolerance.numpy()
        mjw_density = solver.mjw_model.opt.density.numpy()
        mjw_viscosity = solver.mjw_model.opt.viscosity.numpy()

        self.assertEqual(len(mjw_impratio_invsqrt), world_count)
        self.assertEqual(len(mjw_tolerance), world_count)
        self.assertEqual(len(mjw_ls_tolerance), world_count)
        self.assertEqual(len(mjw_ccd_tolerance), world_count)
        self.assertEqual(len(mjw_density), world_count)
        self.assertEqual(len(mjw_viscosity), world_count)

        # All worlds should have the same constructor-provided values
        expected_impratio_invsqrt = 1.0 / np.sqrt(3.0)
        for world_idx in range(world_count):
            self.assertAlmostEqual(
                mjw_impratio_invsqrt[world_idx],
                expected_impratio_invsqrt,
                places=4,
                msg=f"impratio_invsqrt[{world_idx}] should be {expected_impratio_invsqrt}",
            )
            self.assertAlmostEqual(
                mjw_tolerance[world_idx], 1e-5, places=10, msg=f"tolerance[{world_idx}] should be 1e-5"
            )
            self.assertAlmostEqual(
                mjw_ls_tolerance[world_idx], 0.001, places=6, msg=f"ls_tolerance[{world_idx}] should be 0.001"
            )
            self.assertAlmostEqual(
                mjw_ccd_tolerance[world_idx], 1e-4, places=10, msg=f"ccd_tolerance[{world_idx}] should be 1e-4"
            )
            self.assertAlmostEqual(mjw_density[world_idx], 0.0, places=6, msg=f"density[{world_idx}] should be 0.0")
            self.assertAlmostEqual(
                mjw_viscosity[world_idx], 0.0, places=10, msg=f"viscosity[{world_idx}] should be 0.0"
            )

    def test_vector_options_multiworld_conversion(self):
        """
        Verify that vector options (wind, magnetic) with WORLD frequency:
        1. Are properly registered and exist on the model.
        2. Arrays have correct shape (one vec3 per world).
        3. Different per-world vector values are stored correctly.
        4. Solver expands per-world vectors to MuJoCo Warp.
        """
        world_count = 3
        model = self._create_multiworld_model(world_count)

        # Verify arrays have correct shape
        wind = model.mujoco.wind.numpy()
        magnetic = model.mujoco.magnetic.numpy()
        self.assertEqual(len(wind), world_count, "wind array should have one entry per world")
        self.assertEqual(len(magnetic), world_count, "magnetic array should have one entry per world")

        # Set different vector values per world
        initial_wind = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        initial_magnetic = np.array([[0.0, -0.5, 0.0], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
        model.mujoco.wind.assign(initial_wind)
        model.mujoco.magnetic.assign(initial_magnetic)

        # Verify values stored correctly
        updated_wind = model.mujoco.wind.numpy()
        updated_magnetic = model.mujoco.magnetic.numpy()
        for world_idx in range(world_count):
            self.assertTrue(
                np.allclose(updated_wind[world_idx], initial_wind[world_idx]),
                msg=f"Newton model wind[{world_idx}] should be {initial_wind[world_idx]}",
            )
            self.assertTrue(
                np.allclose(updated_magnetic[world_idx], initial_magnetic[world_idx]),
                msg=f"Newton model magnetic[{world_idx}] should be {initial_magnetic[world_idx]}",
            )

        # Create solver without constructor override
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify MuJoCo Warp has per-world vector values
        mjw_wind = solver.mjw_model.opt.wind.numpy()
        mjw_magnetic = solver.mjw_model.opt.magnetic.numpy()
        self.assertEqual(len(mjw_wind), world_count, f"MuJoCo Warp opt.wind should have {world_count} values")
        self.assertEqual(len(mjw_magnetic), world_count, f"MuJoCo Warp opt.magnetic should have {world_count} values")

        # Verify each world has correct values
        for world_idx in range(world_count):
            self.assertTrue(
                np.allclose(mjw_wind[world_idx], initial_wind[world_idx]),
                msg=f"MuJoCo Warp wind[{world_idx}] should be {initial_wind[world_idx]}",
            )
            self.assertTrue(
                np.allclose(mjw_magnetic[world_idx], initial_magnetic[world_idx]),
                msg=f"MuJoCo Warp magnetic[{world_idx}] should be {initial_magnetic[world_idx]}",
            )

    def test_once_numeric_options_shared_across_worlds(self):
        """
        Verify that ONCE frequency numeric options (ccd_iterations, sdf_iterations, sdf_initpoints)
        are shared across all worlds (not per-world arrays).
        """
        world_count = 3
        model = self._create_multiworld_model(world_count)

        # ONCE frequency: single value, not per-world array
        ccd_iterations = model.mujoco.ccd_iterations.numpy()
        sdf_iterations = model.mujoco.sdf_iterations.numpy()
        sdf_initpoints = model.mujoco.sdf_initpoints.numpy()
        self.assertEqual(len(ccd_iterations), 1, "ONCE frequency should have single value")
        self.assertEqual(len(sdf_iterations), 1, "ONCE frequency should have single value")
        self.assertEqual(len(sdf_initpoints), 1, "ONCE frequency should have single value")

        # Set values
        model.mujoco.ccd_iterations.assign(np.array([25], dtype=np.int32))
        model.mujoco.sdf_iterations.assign(np.array([20], dtype=np.int32))
        model.mujoco.sdf_initpoints.assign(np.array([50], dtype=np.int32))

        # Create solver without constructor override
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify MuJoCo model uses the custom attribute values
        self.assertEqual(solver.mj_model.opt.ccd_iterations, 25)
        self.assertEqual(solver.mj_model.opt.sdf_iterations, 20)
        self.assertEqual(solver.mj_model.opt.sdf_initpoints, 50)

    def test_once_numeric_options_constructor_override(self):
        """
        Verify that constructor parameters override custom attribute values
        for ONCE frequency numeric options.
        """
        model = self._create_multiworld_model(world_count=2)

        # Set custom attribute values
        model.mujoco.ccd_iterations.assign(np.array([25], dtype=np.int32))
        model.mujoco.sdf_iterations.assign(np.array([20], dtype=np.int32))
        model.mujoco.sdf_initpoints.assign(np.array([50], dtype=np.int32))

        # Create solver WITH constructor overrides
        solver = SolverMuJoCo(
            model,
            ccd_iterations=100,
            sdf_iterations=30,
            sdf_initpoints=80,
            iterations=1,
            disable_contacts=True,
        )

        # Verify MuJoCo model uses constructor-provided values
        self.assertEqual(solver.mj_model.opt.ccd_iterations, 100, "Constructor should override custom attribute")
        self.assertEqual(solver.mj_model.opt.sdf_iterations, 30, "Constructor should override custom attribute")
        self.assertEqual(solver.mj_model.opt.sdf_initpoints, 80, "Constructor should override custom attribute")

    def test_jacobian_from_custom_attribute(self):
        """
        Verify that jacobian option is read from custom attribute when not provided to constructor.
        """
        model = self._create_multiworld_model(world_count=2)

        # Set jacobian to sparse (1)
        model.mujoco.jacobian.assign(np.array([1], dtype=np.int32))

        # Create solver
        import mujoco

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify MuJoCo model uses custom attribute value
        self.assertEqual(solver.mj_model.opt.jacobian, mujoco.mjtJacobian.mjJAC_SPARSE)

    def test_jacobian_constructor_override(self):
        """
        Verify that jacobian constructor parameter overrides custom attribute value.
        """
        model = self._create_multiworld_model(world_count=2)

        # Set jacobian custom attribute to sparse (1)
        model.mujoco.jacobian.assign(np.array([1], dtype=np.int32))

        # Create solver with constructor override to dense (0)
        import mujoco

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, jacobian="dense")

        # Verify MuJoCo model uses constructor parameter, not custom attribute
        self.assertEqual(solver.mj_model.opt.jacobian, mujoco.mjtJacobian.mjJAC_DENSE)

    def test_enum_options_use_custom_attributes_when_not_provided(self):
        """
        Verify that solver, integrator, cone, and jacobian options use custom attribute
        values when no constructor parameter is provided.

        This tests the resolution priority:
        1. Constructor parameter (if provided)
        2. Custom attribute (if exists)
        3. Default value
        """
        import mujoco

        model = self._create_multiworld_model(world_count=2)

        # Set custom attributes to non-default values
        # Newton defaults: solver=2 (Newton), integrator=3 (implicitfast), cone=0 (pyramidal), jacobian=2 (auto)
        # Set to: solver=1 (CG), integrator=0 (Euler), cone=1 (elliptic), jacobian=1 (sparse)
        model.mujoco.solver.assign(np.array([1], dtype=np.int32))  # CG
        model.mujoco.integrator.assign(np.array([0], dtype=np.int32))  # Euler
        model.mujoco.cone.assign(np.array([1], dtype=np.int32))  # elliptic
        model.mujoco.jacobian.assign(np.array([1], dtype=np.int32))  # sparse

        # Create solver WITHOUT specifying these options - should use custom attributes
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify MuJoCo model uses custom attribute values, not Newton defaults
        self.assertEqual(
            solver.mj_model.opt.solver, mujoco.mjtSolver.mjSOL_CG, "Should use custom attribute CG, not Newton default"
        )
        self.assertEqual(
            solver.mj_model.opt.integrator,
            mujoco.mjtIntegrator.mjINT_EULER,
            "Should use custom attribute Euler, not Newton default implicitfast",
        )
        self.assertEqual(
            solver.mj_model.opt.cone,
            mujoco.mjtCone.mjCONE_ELLIPTIC,
            "Should use custom attribute elliptic, not Newton default pyramidal",
        )
        self.assertEqual(
            solver.mj_model.opt.jacobian,
            mujoco.mjtJacobian.mjJAC_SPARSE,
            "Should use custom attribute sparse, not Newton default auto",
        )

    def test_enum_options_use_defaults_when_no_custom_attribute(self):
        """
        Verify that solver, integrator, cone, and jacobian use Newton defaults
        when no constructor parameter or custom attribute is provided.
        """
        import mujoco

        # Create model WITHOUT registering custom attributes
        builder = newton.ModelBuilder()
        pendulum = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(body=pendulum, hx=0.05, hy=0.05, hz=0.05)
        joint = builder.add_joint_revolute(parent=-1, child=pendulum, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([joint])
        model = builder.finalize()

        # Create solver without specifying enum options - should use Newton defaults
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify Newton defaults are used
        # Newton defaults: solver=Newton(2), integrator=implicitfast(3), cone=pyramidal(0), jacobian=auto(2)
        self.assertEqual(
            solver.mj_model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON, "Should use Newton default (Newton solver)"
        )
        self.assertEqual(
            solver.mj_model.opt.integrator,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            "Should use Newton default (implicitfast)",
        )
        self.assertEqual(
            solver.mj_model.opt.cone, mujoco.mjtCone.mjCONE_PYRAMIDAL, "Should use Newton default (pyramidal)"
        )
        self.assertEqual(
            solver.mj_model.opt.jacobian, mujoco.mjtJacobian.mjJAC_AUTO, "Should use Newton default (auto)"
        )

    def test_iterations_use_custom_attributes_when_not_provided(self):
        """
        Verify that iterations and ls_iterations use custom attribute values
        when no constructor parameter is provided.

        This tests the resolution priority:
        1. Constructor parameter (if provided)
        2. Custom attribute (if exists)
        3. Default value
        """
        model = self._create_multiworld_model(world_count=2)

        # Set custom attributes to non-default values
        # MuJoCo defaults: iterations=100, ls_iterations=50
        # Set to: iterations=150, ls_iterations=75
        model.mujoco.iterations.assign(np.array([150], dtype=np.int32))
        model.mujoco.ls_iterations.assign(np.array([75], dtype=np.int32))

        # Create solver WITHOUT specifying these options - should use custom attributes
        solver = SolverMuJoCo(model, disable_contacts=True)

        # Verify MuJoCo model uses custom attribute values, not defaults
        self.assertEqual(solver.mj_model.opt.iterations, 150, "Should use custom attribute 150, not default 100")
        self.assertEqual(solver.mj_model.opt.ls_iterations, 75, "Should use custom attribute 75, not default 50")

    def test_iterations_use_defaults_when_no_custom_attribute(self):
        """
        Verify that iterations and ls_iterations use MuJoCo defaults when no
        constructor parameter or custom attribute is provided.
        """
        # Create model WITHOUT registering custom attributes
        builder = newton.ModelBuilder()
        pendulum = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(body=pendulum, hx=0.05, hy=0.05, hz=0.05)
        joint = builder.add_joint_revolute(parent=-1, child=pendulum, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([joint])
        model = builder.finalize()

        # Create solver without specifying iterations - should use MuJoCo defaults
        solver = SolverMuJoCo(model, disable_contacts=True)

        # Verify MuJoCo defaults are used: iterations=100, ls_iterations=50
        self.assertEqual(solver.mj_model.opt.iterations, 100, "Should use MuJoCo default (100)")
        self.assertEqual(solver.mj_model.opt.ls_iterations, 50, "Should use MuJoCo default (50)")

    def test_iterations_constructor_override(self):
        """
        Verify that constructor parameters override custom attributes for iterations.
        """
        model = self._create_multiworld_model(world_count=2)

        # Set custom attributes
        model.mujoco.iterations.assign(np.array([150], dtype=np.int32))
        model.mujoco.ls_iterations.assign(np.array([75], dtype=np.int32))

        # Create solver with explicit constructor values - should override custom attributes
        solver = SolverMuJoCo(model, iterations=5, ls_iterations=3, disable_contacts=True)

        # Verify constructor values override custom attributes
        self.assertEqual(solver.mj_model.opt.iterations, 5, "Constructor value should override custom attribute")
        self.assertEqual(solver.mj_model.opt.ls_iterations, 3, "Constructor value should override custom attribute")


class TestMuJoCoArticulationConversion(unittest.TestCase):
    def test_loop_joints_only(self):
        """Testing that loop joints are converted to equality constraints."""
        import mujoco

        builder = newton.ModelBuilder()
        b0 = builder.add_link()
        b1 = builder.add_link()
        j0 = builder.add_joint_revolute(-1, b0)
        j1 = builder.add_joint_revolute(b0, b1)
        builder.add_articulation([j0, j1])
        # add a loop joint
        loop_joint = builder.add_joint_fixed(
            b1,
            b0,
            # note these offset transforms here are important to ensure valid anchor points for the equality constraints are used
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.45), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.45), wp.quat_identity()),
        )
        world_count = 4
        world_builder = newton.ModelBuilder()
        # force the ModelBuilder to correct zero mass/inertia values
        world_builder.bound_inertia = 0.01
        world_builder.bound_mass = 0.01
        world_builder.replicate(builder, world_count=world_count)
        model = world_builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertEqual(solver.mj_model.nv, 2)
        # 2 equality constraints per loop joint
        self.assertEqual(solver.mj_model.neq, 2)
        eq_type = int(mujoco.mjtEq.mjEQ_CONNECT)
        assert np.allclose(solver.mj_model.eq_type, [eq_type, eq_type])
        # we defined no regular equality constraints, so there is no mapping from MuJoCo to Newton equality constraints
        assert np.allclose(solver.mjc_eq_to_newton_eq.numpy(), np.full_like(solver.mjc_eq_to_newton_eq.numpy(), -1))
        # but we converted the loop joints to equality constraints, so there is a mapping from MuJoCo to Newton joints
        assert np.allclose(
            solver.mjc_eq_to_newton_jnt.numpy(),
            [[loop_joint + i * builder.joint_count, loop_joint + i * builder.joint_count] for i in range(world_count)],
        )

    def test_mixed_loop_joints_and_equality_constraints(self):
        """Testing that loop joints and regular equality constraints are converted to equality constraints."""
        import mujoco

        builder = newton.ModelBuilder()
        b0 = builder.add_link()
        b1 = builder.add_link()
        b2 = builder.add_link()
        j0 = builder.add_joint_revolute(-1, b0)
        j1 = builder.add_joint_revolute(-1, b1)
        j2 = builder.add_joint_revolute(b1, b2)
        builder.add_articulation([j0, j1, j2])
        # add one equality constraint before the loop joint
        builder.add_equality_constraint_connect(body1=b0, body2=b1, anchor=wp.vec3(0.0, 0.0, 1.0))
        # add a loop joint
        loop_joint = builder.add_joint_fixed(
            b0,
            b2,
            # note these offset transforms here are important to ensure valid anchor points for the equality constraints are used
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.45), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.45), wp.quat_identity()),
        )
        # add one equality constraint after the loop joint
        builder.add_equality_constraint_connect(body1=b0, body2=b2, anchor=wp.vec3(0.0, 0.0, 1.0))
        world_count = 4
        world_builder = newton.ModelBuilder()
        # force the ModelBuilder to correct zero mass/inertia values
        world_builder.bound_inertia = 0.01
        world_builder.bound_mass = 0.01
        world_builder.replicate(builder, world_count=world_count)
        model = world_builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=True)
        self.assertEqual(model.joint_count, 4 * world_count)
        self.assertEqual(model.equality_constraint_count, 2 * world_count)
        self.assertEqual(solver.mj_model.nv, 3)
        # 2 equality constraints per loop joint
        self.assertEqual(solver.mj_model.neq, 4)
        eq_type = int(mujoco.mjtEq.mjEQ_CONNECT)
        assert np.allclose(solver.mj_model.eq_type, [eq_type] * 4)
        # the two equality constraints we explicitly created are defined first in MuJoCo
        expected_eq_to_newton_eq = np.full((world_count, 4), -1, dtype=np.int32)
        for i in range(world_count):
            expected_eq_to_newton_eq[i, 0] = i * 2
            expected_eq_to_newton_eq[i, 1] = i * 2 + 1
        assert np.allclose(solver.mjc_eq_to_newton_eq.numpy(), expected_eq_to_newton_eq)
        # after those two explicit equality constraints come the 2 equality constraints per loop joint
        expected_eq_to_newton_jnt = np.full((world_count, 4), -1, dtype=np.int32)
        for i in range(world_count):
            # joint 3 is the loop joint, we have 4 joints per world
            expected_eq_to_newton_jnt[i, 2] = i * 4 + loop_joint
            expected_eq_to_newton_jnt[i, 3] = i * 4 + loop_joint
        assert np.allclose(solver.mjc_eq_to_newton_jnt.numpy(), expected_eq_to_newton_jnt)


class TestMuJoCoSolverPairProperties(unittest.TestCase):
    """Test contact pair property conversion and runtime updates across multiple worlds."""

    def test_pair_properties_conversion_and_update(self):
        """
        Test validation of contact pair custom attributes:
        1. Initial conversion from Model to MuJoCo (multi-world)
        2. Runtime updates (multi-world)

        Tests: pair_solref, pair_solreffriction, pair_solimp, pair_margin, pair_gap, pair_friction
        """
        world_count = 3
        pairs_per_world = 2

        # Create a simple model with geoms that we can create pairs between
        template_builder = newton.ModelBuilder()

        # Add a body with three shapes for creating pairs
        body_idx = template_builder.add_body()
        shape1_idx = template_builder.add_shape_sphere(
            body=body_idx,
            xform=wp.transform(wp.vec3(-0.5, 0.0, 0.5), wp.quat_identity()),
            radius=0.1,
        )
        shape2_idx = template_builder.add_shape_sphere(
            body=body_idx,
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.5), wp.quat_identity()),
            radius=0.1,
        )
        shape3_idx = template_builder.add_shape_sphere(
            body=body_idx,
            xform=wp.transform(wp.vec3(0.0, 0.5, 0.5), wp.quat_identity()),
            radius=0.1,
        )

        # Build multi-world model
        builder = newton.ModelBuilder()
        builder.add_shape_plane()

        # Register MuJoCo custom attributes (including pair attributes)
        SolverMuJoCo.register_custom_attributes(builder)

        # Replicate template across worlds
        for i in range(world_count):
            world_transform = wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity())
            builder.add_world(template_builder, xform=world_transform)

        # Add contact pairs for each world
        # Each world gets pairs_per_world pairs
        total_pairs = world_count * pairs_per_world
        shapes_per_world = template_builder.shape_count

        for w in range(world_count):
            world_shape_offset = w * shapes_per_world + 1  # +1 for ground plane

            # Pair 1: shape1 <-> shape2
            builder.add_custom_values(
                **{
                    "mujoco:pair_world": w,
                    "mujoco:pair_geom1": world_shape_offset + shape1_idx,
                    "mujoco:pair_geom2": world_shape_offset + shape2_idx,
                    "mujoco:pair_condim": 3,
                    "mujoco:pair_solref": wp.vec2(0.02 + w * 0.01, 1.0 + w * 0.1),
                    "mujoco:pair_solreffriction": wp.vec2(0.03 + w * 0.01, 1.1 + w * 0.1),
                    "mujoco:pair_solimp": vec5(0.9 - w * 0.01, 0.95, 0.001, 0.5, 2.0),
                    "mujoco:pair_margin": 0.01 + w * 0.005,
                    "mujoco:pair_gap": 0.002 + w * 0.001,
                    "mujoco:pair_friction": vec5(1.0 + w * 0.1, 1.0, 0.005, 0.0001, 0.0001),
                }
            )

            # Pair 2: shape2 <-> shape3
            builder.add_custom_values(
                **{
                    "mujoco:pair_world": w,
                    "mujoco:pair_geom1": world_shape_offset + shape2_idx,
                    "mujoco:pair_geom2": world_shape_offset + shape3_idx,
                    "mujoco:pair_condim": 3,
                    "mujoco:pair_solref": wp.vec2(0.025 + w * 0.01, 1.2 + w * 0.1),
                    "mujoco:pair_solreffriction": wp.vec2(0.035 + w * 0.01, 1.3 + w * 0.1),
                    "mujoco:pair_solimp": vec5(0.85 - w * 0.01, 0.92, 0.002, 0.6, 2.5),
                    "mujoco:pair_margin": 0.015 + w * 0.005,
                    "mujoco:pair_gap": 0.003 + w * 0.001,
                    "mujoco:pair_friction": vec5(1.1 + w * 0.1, 1.1, 0.006, 0.0002, 0.0002),
                }
            )

        model = builder.finalize()

        # Verify custom attribute counts
        self.assertEqual(model.custom_frequency_counts.get("mujoco:pair", 0), total_pairs)

        # Create solver
        solver = SolverMuJoCo(model, separate_worlds=True, iterations=1)

        # Verify MuJoCo has the pairs (only from template world, which is world 0)
        npair = solver.mj_model.npair
        self.assertEqual(npair, pairs_per_world)

        # --- Step 1: Verify initial conversion ---
        # Use .copy() to ensure we capture the values, not a view (important for CPU mode)
        mjw_pair_solref = solver.mjw_model.pair_solref.numpy().copy()
        mjw_pair_solreffriction = solver.mjw_model.pair_solreffriction.numpy().copy()
        mjw_pair_solimp = solver.mjw_model.pair_solimp.numpy().copy()
        mjw_pair_margin = solver.mjw_model.pair_margin.numpy().copy()
        mjw_pair_gap = solver.mjw_model.pair_gap.numpy().copy()
        mjw_pair_friction = solver.mjw_model.pair_friction.numpy().copy()

        # Get expected values from Newton custom attributes (outside loop for performance)
        expected_solref_all = model.mujoco.pair_solref.numpy()
        expected_solreffriction_all = model.mujoco.pair_solreffriction.numpy()
        expected_solimp_all = model.mujoco.pair_solimp.numpy()
        expected_margin_all = model.mujoco.pair_margin.numpy()
        expected_gap_all = model.mujoco.pair_gap.numpy()
        expected_friction_all = model.mujoco.pair_friction.numpy()

        # Check values for each world and pair
        for w in range(world_count):
            newton_pair_base = w * pairs_per_world
            for p in range(pairs_per_world):
                newton_pair = newton_pair_base + p

                np.testing.assert_allclose(
                    mjw_pair_solref[w, p],
                    expected_solref_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_solref mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_solreffriction[w, p],
                    expected_solreffriction_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_solreffriction mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_solimp[w, p],
                    expected_solimp_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_solimp mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_margin[w, p],
                    expected_margin_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_margin mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_gap[w, p],
                    expected_gap_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_gap mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_friction[w, p],
                    expected_friction_all[newton_pair],
                    rtol=1e-5,
                    err_msg=f"pair_friction mismatch at world {w}, pair {p}",
                )

        # --- Step 2: Runtime Update ---
        # Generate new values (different pattern)
        new_solref = np.zeros((total_pairs, 2), dtype=np.float32)
        new_solreffriction = np.zeros((total_pairs, 2), dtype=np.float32)
        new_solimp = np.zeros((total_pairs, 5), dtype=np.float32)
        new_margin = np.zeros(total_pairs, dtype=np.float32)
        new_gap = np.zeros(total_pairs, dtype=np.float32)
        new_friction = np.zeros((total_pairs, 5), dtype=np.float32)

        for i in range(total_pairs):
            new_solref[i] = [0.05 - i * 0.002, 2.0 - i * 0.1]
            new_solreffriction[i] = [0.06 - i * 0.002, 2.1 - i * 0.1]
            new_solimp[i] = [0.8 + i * 0.01, 0.9, 0.003, 0.4, 1.5]
            new_margin[i] = 0.02 + i * 0.003
            new_gap[i] = 0.005 + i * 0.001
            new_friction[i] = [1.5 + i * 0.05, 1.2, 0.007, 0.0003, 0.0003]

        # Update Newton model attributes
        model.mujoco.pair_solref.assign(wp.array(new_solref, dtype=wp.vec2, device=model.device))
        model.mujoco.pair_solreffriction.assign(wp.array(new_solreffriction, dtype=wp.vec2, device=model.device))
        model.mujoco.pair_solimp.assign(wp.array(new_solimp, dtype=vec5, device=model.device))
        model.mujoco.pair_margin.assign(wp.array(new_margin, dtype=wp.float32, device=model.device))
        model.mujoco.pair_gap.assign(wp.array(new_gap, dtype=wp.float32, device=model.device))
        model.mujoco.pair_friction.assign(wp.array(new_friction, dtype=vec5, device=model.device))

        # Notify solver of property change (pair properties are under SHAPE_PROPERTIES)
        solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

        # Verify updates
        mjw_pair_solref_updated = solver.mjw_model.pair_solref.numpy()
        mjw_pair_solreffriction_updated = solver.mjw_model.pair_solreffriction.numpy()
        mjw_pair_solimp_updated = solver.mjw_model.pair_solimp.numpy()
        mjw_pair_margin_updated = solver.mjw_model.pair_margin.numpy()
        mjw_pair_gap_updated = solver.mjw_model.pair_gap.numpy()
        mjw_pair_friction_updated = solver.mjw_model.pair_friction.numpy()

        for w in range(world_count):
            for p in range(pairs_per_world):
                newton_pair = w * pairs_per_world + p

                np.testing.assert_allclose(
                    mjw_pair_solref_updated[w, p],
                    new_solref[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_solref mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_solreffriction_updated[w, p],
                    new_solreffriction[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_solreffriction mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_solimp_updated[w, p],
                    new_solimp[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_solimp mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_margin_updated[w, p],
                    new_margin[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_margin mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_gap_updated[w, p],
                    new_gap[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_gap mismatch at world {w}, pair {p}",
                )
                np.testing.assert_allclose(
                    mjw_pair_friction_updated[w, p],
                    new_friction[newton_pair],
                    rtol=1e-5,
                    err_msg=f"Updated pair_friction mismatch at world {w}, pair {p}",
                )

        # Sanity check: values actually changed
        self.assertFalse(
            np.allclose(mjw_pair_solref_updated[0, 0], mjw_pair_solref[0, 0]),
            "pair_solref should have changed after update!",
        )


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
        builder.add_constraint_mimic(joint0=j2, joint1=j1, coef0=coef0, coef1=coef1, enabled=enabled)
        return builder.finalize()

    def test_mimic_constraint_conversion(self):
        """Test that mimic constraints are converted to MuJoCo mjEQ_JOINT constraints."""
        import mujoco

        model = self._make_two_revolute_model(coef0=0.5, coef1=2.0)
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Verify MuJoCo has 1 equality constraint of type JOINT
        self.assertEqual(solver.mj_model.neq, 1)
        self.assertEqual(solver.mj_model.eq_type[0], mujoco.mjtEq.mjEQ_JOINT)

        # Verify polycoef data: [coef0, coef1, 0, 0, 0]
        eq_data = solver.mjw_model.eq_data.numpy()
        np.testing.assert_allclose(eq_data[0, 0, :5], [0.5, 2.0, 0.0, 0.0, 0.0], rtol=1e-5)

        # Verify mapping exists
        self.assertIsNotNone(solver.mjc_eq_to_newton_mimic)
        mimic_map = solver.mjc_eq_to_newton_mimic.numpy()
        self.assertEqual(mimic_map[0, 0], 0)

    def test_mimic_constraint_runtime_update(self):
        """Test that mimic constraint properties can be updated at runtime."""
        model = self._make_two_revolute_model(coef0=0.5, coef1=2.0)
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Modify coefficients via assign
        model.constraint_mimic_coef0.assign(np.array([1.0], dtype=np.float32))
        model.constraint_mimic_coef1.assign(np.array([3.0], dtype=np.float32))
        model.constraint_mimic_enabled.assign(np.array([False], dtype=bool))

        # Trigger update
        solver.notify_model_changed(SolverNotifyFlags.CONSTRAINT_PROPERTIES)

        # Verify updated values
        eq_data = solver.mjw_model.eq_data.numpy()
        np.testing.assert_allclose(eq_data[0, 0, 0], 1.0, rtol=1e-5)
        np.testing.assert_allclose(eq_data[0, 0, 1], 3.0, rtol=1e-5)
        eq_active = solver.mjw_data.eq_active.numpy()
        self.assertFalse(eq_active[0, 0])

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
        import mujoco

        builder = newton.ModelBuilder()
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=wp.mat33(np.eye(3)))
        j1 = builder.add_joint_revolute(-1, b1, axis=(0, 0, 1))
        j2 = builder.add_joint_revolute(-1, b2, axis=(0, 0, 1))
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([j1, j2])

        # Add a regular JOINT equality constraint
        builder.add_equality_constraint_joint(joint1=j1, joint2=j2, polycoef=[0.0, 1.0, 0.0, 0.0, 0.0])
        # Add a mimic constraint
        builder.add_constraint_mimic(joint0=j2, joint1=j1, coef0=0.0, coef1=1.0)

        model = builder.finalize()
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # 1 regular eq + 1 mimic = 2 MuJoCo eq constraints
        self.assertEqual(solver.mj_model.neq, 2)
        self.assertEqual(solver.mj_model.eq_type[0], mujoco.mjtEq.mjEQ_JOINT)
        self.assertEqual(solver.mj_model.eq_type[1], mujoco.mjtEq.mjEQ_JOINT)

    def test_mimic_constraint_simulation(self):
        """Test that mimic constraint enforces joint tracking during simulation."""
        # Use coef1=2.0 so the relationship is non-trivial: j2 = 2.0 * j1
        model = self._make_two_revolute_model(coef0=0.0, coef1=2.0)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()

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
        template_builder.add_constraint_mimic(joint0=j2, joint1=j1, coef0=0.0, coef1=1.0)

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

        solver.notify_model_changed(SolverNotifyFlags.CONSTRAINT_PROPERTIES)

        # Verify each world got its own coefficients
        eq_data = solver.mjw_model.eq_data.numpy()
        for w in range(world_count):
            np.testing.assert_allclose(
                eq_data[w, 0, 0], new_coef0[w], rtol=1e-5, err_msg=f"coef0 mismatch in world {w}"
            )
            np.testing.assert_allclose(
                eq_data[w, 0, 1], new_coef1[w], rtol=1e-5, err_msg=f"coef1 mismatch in world {w}"
            )


class TestMuJoCoSolverFreeJointBodyPos(unittest.TestCase):
    """Verify free joint bodies preserve their initial position in qpos0."""

    def test_free_joint_body_pos(self):
        """Verify free joint qpos0 contains the MJCF body position.

        A free joint body placed at pos="0 0 1.5" should produce
        ``solver.mj_model.qpos0`` whose first three elements match
        the body's initial z-offset ``[0, 0, 1.5]``.
        """
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="robot" pos="0 0 1.5">
                    <freejoint name="root"/>
                    <geom type="sphere" size="0.1" mass="1.0"/>
                    <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
                </body>
            </worldbody>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        # qpos0 should have the body's initial z position
        qpos0 = np.array(solver.mj_model.qpos0)
        np.testing.assert_allclose(
            qpos0[:3],
            [0.0, 0.0, 1.5],
            atol=1e-6,
            err_msg="Free joint qpos0 position should match body pos from MJCF",
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
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        self.assertIsNotNone(solver.mj_model)


class TestMuJoCoSolverQpos0(unittest.TestCase):
    """Tests for qpos0, qpos_spring, ref/springref coordinate conversion, and FK correctness."""

    # -- Group A: qpos0 initial values per joint type --

    def test_free_joint_qpos0(self):
        """Verify free joint qpos0 contains body position and identity quaternion.

        A free joint body at pos="0 0 1.5" should produce qpos0 with
        position [0, 0, 1.5] and identity quaternion [1, 0, 0, 0] (wxyz).
        """
        mjcf = """<mujoco><worldbody>
            <body name="b" pos="0 0 1.5">
                <joint type="free"/>
                <geom type="sphere" size="0.1"/>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        qpos0 = solver.mjw_model.qpos0.numpy()
        np.testing.assert_allclose(qpos0[0, :3], [0, 0, 1.5], atol=1e-6)
        np.testing.assert_allclose(qpos0[0, 3:7], [1, 0, 0, 0], atol=1e-6)  # wxyz identity

    def test_hinge_with_ref_qpos0(self):
        """Verify hinge joint qpos0 equals ref in radians.

        A hinge with ref=90 degrees should produce qpos0 approximately
        equal to pi/2 radians.
        """
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child" pos="0 0 1">
                    <joint type="hinge" axis="0 1 0" ref="90"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        qpos0 = solver.mjw_model.qpos0.numpy()
        np.testing.assert_allclose(qpos0[0, 0], np.pi / 2, atol=1e-5)

    def test_slide_with_ref_qpos0(self):
        """Verify slide joint qpos0 equals ref value.

        A slide joint with ref=0.1 should produce qpos0 equal to 0.1.
        """
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child" pos="0 0 1">
                    <joint type="slide" axis="0 0 1" ref="0.1"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        qpos0 = solver.mjw_model.qpos0.numpy()
        np.testing.assert_allclose(qpos0[0, 0], 0.1, atol=1e-6)

    def test_ball_joint_qpos0(self):
        """Verify ball joint qpos0 is an identity quaternion.

        A ball joint should produce qpos0 equal to [1, 0, 0, 0] (wxyz).
        """
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        parent = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(body=parent, hx=0.1, hy=0.1, hz=0.1)
        j0 = builder.add_joint_fixed(-1, parent)
        child = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(body=child, hx=0.1, hy=0.1, hz=0.1)
        j1 = builder.add_joint_ball(parent, child)
        builder.add_articulation([j0, j1])
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        qpos0 = solver.mjw_model.qpos0.numpy()
        np.testing.assert_allclose(qpos0[0, :4], [1, 0, 0, 0], atol=1e-6)

    def test_hinge_no_ref_qpos0(self):
        """Verify hinge joint without ref has qpos0 of zero."""
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
        qpos0 = solver.mjw_model.qpos0.numpy()
        np.testing.assert_allclose(qpos0[0, 0], 0.0, atol=1e-6)

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

    # -- Group B: qpos_spring values --

    def test_hinge_springref_qpos_spring(self):
        """Verify hinge qpos_spring equals springref in radians.

        A hinge with springref=30 degrees should produce qpos_spring
        approximately equal to pi/6 radians.
        """
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child" pos="0 0 1">
                    <joint type="hinge" axis="0 1 0" springref="30"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        qpos_spring = solver.mjw_model.qpos_spring.numpy()
        np.testing.assert_allclose(qpos_spring[0, 0], np.deg2rad(30), atol=1e-5)

    def test_free_joint_qpos_spring_matches_qpos0(self):
        """Verify free joint qpos_spring equals qpos0."""
        mjcf = """<mujoco><worldbody>
            <body name="b" pos="1 2 3">
                <joint type="free"/>
                <geom type="sphere" size="0.1"/>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        qpos0 = solver.mjw_model.qpos0.numpy()
        qpos_spring = solver.mjw_model.qpos_spring.numpy()
        np.testing.assert_allclose(qpos_spring, qpos0, atol=1e-6)

    def test_slide_springref_qpos_spring(self):
        """Verify slide qpos_spring equals springref value."""
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child" pos="0 0 1">
                    <joint type="slide" axis="0 0 1" springref="0.25"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        qpos_spring = solver.mjw_model.qpos_spring.numpy()
        np.testing.assert_allclose(qpos_spring[0, 0], 0.25, atol=1e-6)

    # -- Group C: Coordinate conversion with ref offset --

    def test_hinge_ref_newton_to_mujoco(self):
        """Verify Newton-to-MuJoCo conversion adds ref offset.

        With ref=90 degrees, joint_q=0 should map to qpos=pi/2.
        """
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child" pos="0 0 1">
                    <joint type="hinge" axis="0 1 0" ref="90"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        state = model.state()
        # joint_q defaults to 0 for hinge
        solver.update_mjc_data(solver.mjw_data, model, state)
        qpos = solver.mjw_data.qpos.numpy()
        np.testing.assert_allclose(qpos[0, 0], np.pi / 2, atol=1e-5)

    def test_hinge_ref_mujoco_to_newton(self):
        """Verify MuJoCo-to-Newton conversion subtracts ref offset.

        With ref=90 degrees, qpos=pi/2+0.1 should map to joint_q=0.1.
        """
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child" pos="0 0 1">
                    <joint type="hinge" axis="0 1 0" ref="90"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        # Set qpos = ref + 0.1
        qpos = solver.mjw_data.qpos.numpy()
        qpos[0, 0] = np.pi / 2 + 0.1
        solver.mjw_data.qpos.assign(qpos)
        state = model.state()
        solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)
        solver.update_newton_state(model, state, solver.mjw_data)
        joint_q = state.joint_q.numpy()
        np.testing.assert_allclose(joint_q[0], 0.1, atol=1e-5)

    def test_slide_ref_roundtrip(self):
        """Verify slide joint_q survives Newton-MuJoCo-Newton roundtrip with ref.

        Sets joint_q=0.3 with ref=0.5, converts to MuJoCo (expecting qpos=0.8),
        then back to Newton (expecting joint_q=0.3).
        """
        mjcf = """<mujoco><worldbody>
            <body name="base"><geom type="box" size="0.1 0.1 0.1"/>
                <body name="child" pos="0 0 1">
                    <joint type="slide" axis="0 0 1" ref="0.5"/>
                    <geom type="box" size="0.1 0.1 0.1"/>
                </body>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        state = model.state()

        # Set a known joint_q value
        test_q = 0.3
        q = state.joint_q.numpy()
        q[0] = test_q
        state.joint_q.assign(q)

        # Newton → MuJoCo
        solver.update_mjc_data(solver.mjw_data, model, state)
        qpos = solver.mjw_data.qpos.numpy()
        np.testing.assert_allclose(qpos[0, 0], test_q + 0.5, atol=1e-5)

        # MuJoCo → Newton
        solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)
        state2 = model.state()
        solver.update_newton_state(model, state2, solver.mjw_data)
        np.testing.assert_allclose(state2.joint_q.numpy()[0], test_q, atol=1e-5)

    def test_free_joint_position_roundtrip(self):
        """Verify free joint position survives Newton-MuJoCo-Newton roundtrip.

        Free joints have no ref offset, so joint_q should be preserved
        exactly through the coordinate conversion cycle.
        """
        mjcf = """<mujoco><worldbody>
            <body name="b" pos="1 2 3">
                <joint type="free"/>
                <geom type="sphere" size="0.1"/>
            </body>
        </worldbody></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model)
        state = model.state()
        original_q = state.joint_q.numpy().copy()

        # Newton → MuJoCo → Newton
        solver.update_mjc_data(solver.mjw_data, model, state)
        solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)
        solver.update_newton_state(model, state, solver.mjw_data)
        roundtrip_q = state.joint_q.numpy()

        np.testing.assert_allclose(roundtrip_q[:3], original_q[:3], atol=1e-5)
        # Quaternion comparison (sign-invariant)
        q_orig = original_q[3:7]
        q_rt = roundtrip_q[3:7]
        quat_dist = min(np.linalg.norm(q_orig - q_rt), np.linalg.norm(q_orig + q_rt))
        self.assertLess(quat_dist, 1e-5)

    # -- Group D: FK correctness --

    def _compare_body_positions(self, model, solver, state, body_names, atol=0.01):
        """Compare Newton and MuJoCo body positions after FK.

        Runs update_mjc_data, kinematics, and update_newton_state, then
        asserts that Newton body positions match MuJoCo xpos for each
        named body.
        """
        solver.update_mjc_data(solver.mjw_data, model, state)
        solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)
        solver.update_newton_state(model, state, solver.mjw_data)

        newton_body_q = state.body_q.numpy()
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()
        mj_xpos = solver.mjw_data.xpos.numpy()

        for name in body_names:
            newton_idx = model.body_key.index(name)
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
        contacts = model.contacts()
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

        solver.notify_model_changed(SolverNotifyFlags.ALL)

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

        solver.notify_model_changed(SolverNotifyFlags.ALL)

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
        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

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
        solver.update_mjc_data(solver.mjw_data, model, state)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
