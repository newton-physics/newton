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

import os  # For path manipulation
import time  # Added for the interactive loop
import unittest

import numpy as np  # For numerical operations and random values
import warp as wp

import newton
from newton.core import types
from newton.solvers import MuJoCoSolver

# Import the kernels for coordinate conversion
from newton.utils import SimRendererOpenGL


class TestMuJoCoSolver(unittest.TestCase):
    def setUp(self):
        "Hook method for setting up the test fixture before exercising it."
        pass

    def _run_substeps_for_frame(self, sim_dt, sim_substeps):
        """Helper method to run simulation substeps for one rendered frame."""
        for _ in range(sim_substeps):
            self.solver.step(self.model, self.state_in, self.state_out, self.control, self.contacts, sim_dt)
            self.state_in, self.state_out = self.state_out, self.state_in  # Output becomes input for next substep

    def test_setup_completes(self):
        """
        Tests if the setUp method completes successfully.
        This implicitly tests model creation, finalization, solver, and renderer initialization.
        """
        self.assertTrue(True, "setUp method completed.")

    @unittest.skip("Trajectory rendering for debugging")
    def test_render_trajectory(self):
        """Simulates and renders a trajectory if solver and renderer are available."""
        print("\nDebug: Starting test_render_trajectory...")

        solver = None
        renderer = None
        substep_graph = None
        use_cuda_graph = wp.get_device().is_cuda

        try:
            print("Debug: Attempting to initialize MuJoCoSolver for trajectory test...")
            solver = MuJoCoSolver(self.model, iterations=10, ls_iterations=10)
            print("Debug: MuJoCoSolver initialized successfully for trajectory test.")
        except ImportError as e:
            self.skipTest(f"MuJoCo or deps not installed. Skipping trajectory rendering: {e}")
            return
        except Exception as e:
            self.skipTest(f"Error initializing MuJoCoSolver for trajectory test: {e}")
            return

        if self.debug_stage_path:
            try:
                print(f"Debug: Attempting to initialize SimRendererOpenGL (stage: {self.debug_stage_path})...")
                stage_dir = os.path.dirname(self.debug_stage_path)
                if stage_dir and not os.path.exists(stage_dir):
                    os.makedirs(stage_dir)
                    print(f"Debug: Created directory for stage: {stage_dir}")
                renderer = SimRendererOpenGL(
                    path=self.debug_stage_path, model=self.model, scaling=1.0, show_joints=True
                )
                print("Debug: SimRendererOpenGL initialized successfully for trajectory test.")
            except ImportError as e:
                self.skipTest(f"SimRendererOpenGL dependencies not met. Skipping trajectory rendering: {e}")
                return
            except Exception as e:
                self.skipTest(f"Error initializing SimRendererOpenGL for trajectory test: {e}")
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

                renderer.begin_frame(sim_time)
                renderer.render(self.state_in)
                renderer.end_frame()

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


class TestMuJoCoSolverMassProperties(TestMuJoCoSolver):
    def setUp(self):
        """Set up a model with multiple environments, each with a free body and an articulated tree."""
        self.seed = 123
        self.rng = np.random.default_rng(self.seed)

        num_envs = 2
        self.debug_stage_path = "newton/tests/test_mujoco_render.usda"

        template_builder = newton.ModelBuilder()
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)  # Define ShapeConfig

        # --- Free-floating body (e.g., a box) ---
        free_body_initial_pos = wp.transform((0.5, 0.5, 0.0), wp.quat_identity())
        free_body_idx = template_builder.add_body(mass=0.2)
        template_builder.add_joint_free(child=free_body_idx, parent_xform=free_body_initial_pos)
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

        body1_idx = template_builder.add_body(mass=0.1)
        template_builder.add_joint_free(child=body1_idx, parent_xform=tree_root_initial_transform)
        template_builder.add_shape_capsule(
            body=body1_idx,
            xform=wp.transform(),  # Shape at body's local origin
            radius=link_radius,
            half_height=link_half_length,
            cfg=shape_cfg,
        )

        body2_idx = template_builder.add_body(mass=0.1)
        template_builder.add_shape_capsule(
            body=body2_idx,
            xform=wp.transform(),  # Shape at body's local origin
            radius=link_radius,
            half_height=link_half_length,
            cfg=shape_cfg,
        )
        template_builder.add_joint_revolute(
            parent=body1_idx,
            child=body2_idx,
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            axis=(0.0, 0.0, 1.0),
        )

        body3_idx = template_builder.add_body(mass=0.1)
        template_builder.add_shape_capsule(
            body=body3_idx,
            xform=wp.transform(),  # Shape at body's local origin
            radius=link_radius,
            half_height=link_half_length,
            cfg=shape_cfg,
        )
        template_builder.add_joint_revolute(
            parent=body2_idx,
            child=body3_idx,
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            axis=(1.0, 0.0, 0.0),
        )

        self.builder = newton.ModelBuilder()
        self.builder.add_shape_plane()

        for i in range(num_envs):
            env_transform = wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity())
            self.builder.add_builder(
                template_builder, xform=env_transform, update_num_env_count=True, separate_collision_group=True
            )

        try:
            if self.builder.num_envs == 0 and num_envs > 0:
                self.builder.num_envs = num_envs
            self.model = self.builder.finalize()
            if self.model.num_envs != num_envs:
                print(f"Warning: Model.num_envs ({self.model.num_envs}) does not match expected num_envs ({num_envs}).")
        except Exception as e:
            self.fail(f"Model finalization failed: {e}")

        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contact()

    def test_randomize_body_mass(self):
        """
        Tests if the body mass is randomized correctly and updates properly after simulation steps.
        """
        # Randomize masses for all bodies in all environments
        new_masses = self.rng.uniform(1.0, 10.0, size=self.model.body_count)
        self.model.body_mass.assign(new_masses)

        # Initialize solver
        solver = MuJoCoSolver(self.model, ls_iterations=1, iterations=1, disable_contacts=True)

        # Check that masses were transferred correctly
        bodies_per_env = self.model.body_count // self.model.num_envs
        body_mapping = self.model.to_mjc_body_index.numpy()
        for env_idx in range(self.model.num_envs):
            for body_idx in range(bodies_per_env):
                newton_idx = env_idx * bodies_per_env + body_idx
                mjc_idx = body_mapping[body_idx]
                if mjc_idx != -1:  # Skip unmapped bodies
                    self.assertAlmostEqual(
                        new_masses[newton_idx],
                        solver.mjw_model.body_mass.numpy()[env_idx, mjc_idx],
                        places=6,
                        msg=f"Mass mismatch for body {body_idx} in environment {env_idx}",
                    )

        # Run a simulation step
        solver.step(self.model, self.state_in, self.state_out, self.control, self.contacts, 0.01)
        self.state_in, self.state_out = self.state_out, self.state_in

        # Update masses again
        updated_masses = self.rng.uniform(1.0, 10.0, size=self.model.body_count)
        self.model.body_mass.assign(updated_masses)

        # Notify solver of mass changes
        solver.notify_model_changed(types.NOTIFY_FLAG_BODY_INERTIAL_PROPERTIES)

        # Check that updated masses were transferred correctly
        for env_idx in range(self.model.num_envs):
            for body_idx in range(bodies_per_env):
                newton_idx = env_idx * bodies_per_env + body_idx
                mjc_idx = body_mapping[body_idx]
                if mjc_idx != -1:  # Skip unmapped bodies
                    self.assertAlmostEqual(
                        updated_masses[newton_idx],
                        solver.mjw_model.body_mass.numpy()[env_idx, mjc_idx],
                        places=6,
                        msg=f"Updated mass mismatch for body {body_idx} in environment {env_idx}",
                    )

    def test_randomize_body_com(self):
        """
        Tests if the body center of mass is randomized correctly and updates properly after simulation steps.
        """
        # Randomize COM for all bodies in all environments
        new_coms = self.rng.uniform(-1.0, 1.0, size=(self.model.body_count, 3))
        self.model.body_com.assign(new_coms)

        # Initialize solver
        solver = MuJoCoSolver(self.model, ls_iterations=1, iterations=1, disable_contacts=True, nefc_per_env=1)

        # Check that COM positions were transferred correctly
        bodies_per_env = self.model.body_count // self.model.num_envs
        body_mapping = self.model.to_mjc_body_index.numpy()
        for env_idx in range(self.model.num_envs):
            for body_idx in range(bodies_per_env):
                newton_idx = env_idx * bodies_per_env + body_idx
                mjc_idx = body_mapping[body_idx]
                if mjc_idx != -1:  # Skip unmapped bodies
                    newton_pos = new_coms[newton_idx]
                    mjc_pos = solver.mjw_model.body_ipos.numpy()[env_idx, mjc_idx]

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
                            msg=f"COM position mismatch for body {body_idx} in environment {env_idx}, dimension {dim}",
                        )

        # Run a simulation step
        solver.step(self.model, self.state_in, self.state_out, self.control, self.contacts, 0.01)
        self.state_in, self.state_out = self.state_out, self.state_in

        # Update COM positions again
        updated_coms = self.rng.uniform(-1.0, 1.0, size=(self.model.body_count, 3))
        self.model.body_com.assign(updated_coms)

        # Notify solver of COM changes
        solver.notify_model_changed(types.NOTIFY_FLAG_BODY_INERTIAL_PROPERTIES)

        # Check that updated COM positions were transferred correctly
        for env_idx in range(self.model.num_envs):
            for body_idx in range(bodies_per_env):
                newton_idx = env_idx * bodies_per_env + body_idx
                mjc_idx = body_mapping[body_idx]
                if mjc_idx != -1:  # Skip unmapped bodies
                    newton_pos = updated_coms[newton_idx]
                    mjc_pos = solver.mjw_model.body_ipos.numpy()[env_idx, mjc_idx]

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
                            msg=f"Updated COM position mismatch for body {body_idx} in environment {env_idx}, dimension {dim}",
                        )

    def test_randomize_body_inertia(self):
        """
        Tests if the body inertia is randomized correctly.
        """
        # Randomize inertia tensors for all bodies in all environments
        # Simple inertia tensors that satisfy triangle inequality

        new_inertias = np.zeros((self.model.body_count, 3, 3))
        bodies_per_env = self.model.body_count // self.model.num_envs
        for i in range(self.model.body_count):
            env_idx = i // bodies_per_env
            if env_idx == 0:
                # First environment: ensure a + b > c with random values
                a = 2.0 + self.rng.uniform(0.0, 0.5)
                b = 3.0 + self.rng.uniform(0.0, 0.5)
                c = min(a + b - 0.1, 4.0)  # Ensure a + b > c
                new_inertias[i] = np.diag([a, b, c])
            else:
                # Second environment: ensure a + b > c with random values
                a = 3.0 + self.rng.uniform(0.0, 0.5)
                b = 4.0 + self.rng.uniform(0.0, 0.5)
                c = min(a + b - 0.1, 5.0)  # Ensure a + b > c
                new_inertias[i] = np.diag([a, b, c])
        self.model.body_inertia.assign(new_inertias)

        # Initialize solver
        solver = MuJoCoSolver(self.model, iterations=1, ls_iterations=1, disable_contacts=True)

        # Get body mapping once outside the loop
        body_mapping = self.model.to_mjc_body_index.numpy()

        def check_inertias(inertias_to_check, msg_prefix=""):
            for env_idx in range(self.model.num_envs):
                for body_idx in range(bodies_per_env):
                    newton_idx = env_idx * bodies_per_env + body_idx
                    mjc_idx = body_mapping[body_idx]
                    if mjc_idx != -1:  # Skip unmapped bodies
                        newton_inertia = inertias_to_check[newton_idx]
                        mjc_inertia = solver.mjw_model.body_inertia.numpy()[env_idx, mjc_idx]

                        # Get eigenvalues of both tensors
                        newton_eigvals = np.linalg.eigvalsh(newton_inertia)
                        mjc_eigvals = mjc_inertia  # Already in diagonal form

                        # Sort eigenvalues in descending order
                        newton_eigvals.sort()
                        newton_eigvals = newton_eigvals[::-1]

                        for dim in range(3):
                            self.assertAlmostEqual(
                                newton_eigvals[dim],
                                mjc_eigvals[dim],
                                places=6,
                                msg=f"{msg_prefix}Inertia eigenvalue mismatch for body {body_idx} in environment {env_idx}, dimension {dim}",
                            )

        # Check initial inertia tensors
        check_inertias(new_inertias, "Initial ")

        # Run a simulation step
        solver.step(self.model, self.state_in, self.state_out, self.control, self.contacts, 0.01)
        self.state_in, self.state_out = self.state_out, self.state_in

        # Update inertia tensors again with new random values
        updated_inertias = np.zeros((self.model.body_count, 3, 3))
        for i in range(self.model.body_count):
            env_idx = i // bodies_per_env
            if env_idx == 0:
                a = 2.5 + self.rng.uniform(0.0, 0.5)
                b = 3.5 + self.rng.uniform(0.0, 0.5)
                c = min(a + b - 0.1, 4.5)
                updated_inertias[i] = np.diag([a, b, c])
            else:
                a = 3.5 + self.rng.uniform(0.0, 0.5)
                b = 4.5 + self.rng.uniform(0.0, 0.5)
                c = min(a + b - 0.1, 5.5)
                updated_inertias[i] = np.diag([a, b, c])
        self.model.body_inertia.assign(updated_inertias)

        # Notify solver of inertia changes
        solver.notify_model_changed(types.NOTIFY_FLAG_BODY_INERTIAL_PROPERTIES)

        # Check updated inertia tensors
        check_inertias(updated_inertias, "Updated ")


class TestMuJoCoSolverJointProperties(TestMuJoCoSolverMassProperties):
    def test_joint_attributes_registration_and_updates(self):
        """
        Verify that joint effort limit, velocity limit, armature, and friction:
        1. Are properly set in Newton Model
        2. Are properly registered in MuJoCo
        3. Can be changed during simulation via notify_model_changed()

        Uses different values for each joint and environment to catch indexing bugs.

        TODO: We currently don't check velocity_limits because MuJoCo doesn't seem to have
              a matching parameter. The values are set in Newton but not verified in MuJoCo.
        """
        # Skip if no joints
        if self.model.joint_axis_count == 0:
            self.skipTest("No joints in model, skipping joint attributes test")

        # Step 1: Set initial values with different patterns for each attribute
        # Pattern: base_value + axis_idx * increment + env_offset
        axes_per_env = self.model.joint_axis_count // self.model.num_envs
        dofs_per_env = self.model.joint_dof_count // self.model.num_envs

        initial_effort_limits = np.zeros(self.model.joint_axis_count)
        initial_velocity_limits = np.zeros(self.model.joint_axis_count)
        initial_friction = np.zeros(self.model.joint_axis_count)
        initial_armature = np.zeros(self.model.joint_dof_count)

        # Set different values for each axis and environment
        for env_idx in range(self.model.num_envs):
            env_axis_offset = env_idx * axes_per_env
            env_dof_offset = env_idx * dofs_per_env

            for axis_idx in range(axes_per_env):
                global_axis_idx = env_axis_offset + axis_idx
                # Effort limit: 50 + axis_idx * 10 + env_idx * 100
                initial_effort_limits[global_axis_idx] = 50.0 + axis_idx * 10.0 + env_idx * 100.0
                # Velocity limit: 10 + axis_idx * 2 + env_idx * 20
                initial_velocity_limits[global_axis_idx] = 10.0 + axis_idx * 2.0 + env_idx * 20.0
                # Friction: 0.5 + axis_idx * 0.1 + env_idx * 0.5
                initial_friction[global_axis_idx] = 0.5 + axis_idx * 0.1 + env_idx * 0.5

            for dof_idx in range(dofs_per_env):
                global_dof_idx = env_dof_offset + dof_idx
                # Armature: 0.01 + dof_idx * 0.005 + env_idx * 0.05
                initial_armature[global_dof_idx] = 0.01 + dof_idx * 0.005 + env_idx * 0.05

        self.model.joint_effort_limit.assign(initial_effort_limits)
        self.model.joint_velocity_limit.assign(initial_velocity_limits)
        self.model.joint_friction.assign(initial_friction)
        self.model.joint_armature.assign(initial_armature)

        # Step 2: Create solver (this should apply values to MuJoCo)
        solver = MuJoCoSolver(self.model, iterations=1, disable_contacts=True)

        # Step 3: Verify initial values were applied to MuJoCo

        # Check effort limits: Newton value should appear as MuJoCo actuator force range
        for env_idx in range(self.model.num_envs):
            for axis_idx in range(axes_per_env):
                global_axis_idx = env_idx * axes_per_env + axis_idx
                actuator_idx = solver.model.mjc_axis_to_actuator.numpy()[axis_idx]

                if actuator_idx >= 0:  # This axis has an actuator
                    force_range = solver.mjw_model.actuator_forcerange.numpy()[env_idx, actuator_idx]
                    expected_limit = initial_effort_limits[global_axis_idx]
                    self.assertAlmostEqual(
                        force_range[0],
                        -expected_limit,
                        places=3,
                        msg=f"MuJoCo actuator {actuator_idx} in env {env_idx} min force should match negative Newton effort limit",
                    )
                    self.assertAlmostEqual(
                        force_range[1],
                        expected_limit,
                        places=3,
                        msg=f"MuJoCo actuator {actuator_idx} in env {env_idx} max force should match Newton effort limit",
                    )

        # Check armature: Newton value should appear directly in MuJoCo DOF armature
        for env_idx in range(self.model.num_envs):
            for dof_idx in range(min(dofs_per_env, solver.mjw_model.dof_armature.shape[1])):
                global_dof_idx = env_idx * dofs_per_env + dof_idx
                expected_armature = initial_armature[global_dof_idx]
                actual_armature = solver.mjw_model.dof_armature.numpy()[env_idx, dof_idx]
                self.assertAlmostEqual(
                    actual_armature,
                    expected_armature,
                    places=4,
                    msg=f"MuJoCo DOF {dof_idx} in env {env_idx} armature should match Newton value",
                )

        # Check friction: Newton value should appear in MuJoCo DOF friction loss
        for env_idx in range(self.model.num_envs):
            for dof_idx in range(min(dofs_per_env, solver.mjw_model.dof_frictionloss.shape[1])):
                global_dof_idx = env_idx * dofs_per_env + dof_idx
                axis_idx = self.model.dof_to_axis_map.numpy()[global_dof_idx]
                if axis_idx >= 0:  # Only check DOFs that have axis mappings
                    expected_friction = initial_friction[axis_idx]
                    actual_friction = solver.mjw_model.dof_frictionloss.numpy()[env_idx, dof_idx]
                    self.assertAlmostEqual(
                        actual_friction,
                        expected_friction,
                        places=4,
                        msg=f"MuJoCo DOF {dof_idx} in env {env_idx} friction should match Newton value",
                    )

        # Step 4: Change all values with different patterns
        updated_effort_limits = np.zeros(self.model.joint_axis_count)
        updated_velocity_limits = np.zeros(self.model.joint_axis_count)
        updated_friction = np.zeros(self.model.joint_axis_count)
        updated_armature = np.zeros(self.model.joint_dof_count)

        # Set different updated values for each axis and environment
        for env_idx in range(self.model.num_envs):
            env_axis_offset = env_idx * axes_per_env
            env_dof_offset = env_idx * dofs_per_env

            for axis_idx in range(axes_per_env):
                global_axis_idx = env_axis_offset + axis_idx
                # Updated effort limit: 100 + axis_idx * 15 + env_idx * 150
                updated_effort_limits[global_axis_idx] = 100.0 + axis_idx * 15.0 + env_idx * 150.0
                # Updated velocity limit: 20 + axis_idx * 3 + env_idx * 30
                updated_velocity_limits[global_axis_idx] = 20.0 + axis_idx * 3.0 + env_idx * 30.0
                # Updated friction: 1.0 + axis_idx * 0.2 + env_idx * 1.0
                updated_friction[global_axis_idx] = 1.0 + axis_idx * 0.2 + env_idx * 1.0

            for dof_idx in range(dofs_per_env):
                global_dof_idx = env_dof_offset + dof_idx
                # Updated armature: 0.05 + dof_idx * 0.01 + env_idx * 0.1
                updated_armature[global_dof_idx] = 0.05 + dof_idx * 0.01 + env_idx * 0.1

        self.model.joint_effort_limit.assign(updated_effort_limits)
        self.model.joint_velocity_limit.assign(updated_velocity_limits)
        self.model.joint_friction.assign(updated_friction)
        self.model.joint_armature.assign(updated_armature)

        # Step 5: Notify MuJoCo of changes
        solver.notify_model_changed(types.NOTIFY_FLAG_JOINT_AXIS_PROPERTIES | types.NOTIFY_FLAG_DOF_PROPERTIES)

        # Step 6: Verify all changes were applied

        # Check updated effort limits
        for env_idx in range(self.model.num_envs):
            for axis_idx in range(axes_per_env):
                global_axis_idx = env_idx * axes_per_env + axis_idx
                actuator_idx = solver.model.mjc_axis_to_actuator.numpy()[axis_idx]

                if actuator_idx >= 0:
                    force_range = solver.mjw_model.actuator_forcerange.numpy()[env_idx, actuator_idx]
                    expected_limit = updated_effort_limits[global_axis_idx]
                    self.assertAlmostEqual(
                        force_range[0],
                        -expected_limit,
                        places=3,
                        msg=f"Updated MuJoCo actuator {actuator_idx} in env {env_idx} min force should match negative Newton effort limit",
                    )
                    self.assertAlmostEqual(
                        force_range[1],
                        expected_limit,
                        places=3,
                        msg=f"Updated MuJoCo actuator {actuator_idx} in env {env_idx} max force should match Newton effort limit",
                    )

        # Check updated armature
        for env_idx in range(self.model.num_envs):
            for dof_idx in range(min(dofs_per_env, solver.mjw_model.dof_armature.shape[1])):
                global_dof_idx = env_idx * dofs_per_env + dof_idx
                expected_armature = updated_armature[global_dof_idx]
                actual_armature = solver.mjw_model.dof_armature.numpy()[env_idx, dof_idx]
                self.assertAlmostEqual(
                    actual_armature,
                    expected_armature,
                    places=4,
                    msg=f"Updated MuJoCo DOF {dof_idx} in env {env_idx} armature should match Newton value",
                )

        # Check updated friction
        for env_idx in range(self.model.num_envs):
            for dof_idx in range(min(dofs_per_env, solver.mjw_model.dof_frictionloss.shape[1])):
                global_dof_idx = env_idx * dofs_per_env + dof_idx
                axis_idx = self.model.dof_to_axis_map.numpy()[global_dof_idx]
                if axis_idx >= 0:  # Only check DOFs that have axis mappings
                    expected_friction = updated_friction[axis_idx]
                    actual_friction = solver.mjw_model.dof_frictionloss.numpy()[env_idx, dof_idx]
                    self.assertAlmostEqual(
                        actual_friction,
                        expected_friction,
                        places=4,
                        msg=f"Updated MuJoCo DOF {dof_idx} in env {env_idx} friction should match Newton value",
                    )


if __name__ == "__main__":
    unittest.main()
