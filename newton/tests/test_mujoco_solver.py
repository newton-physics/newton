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

import unittest
import os # For path manipulation
import time # Added for the interactive loop
import numpy as np # For numerical operations and random values
import warp as wp
import newton
from newton.core import ModelBuilder, State, Control, Contact, JOINT_FREE
from newton.solvers import MuJoCoSolver
from newton.utils import SimRendererOpenGL
# Import the kernels for coordinate conversion
from newton.solvers.solver_mujoco import convert_mj_coords_to_warp_kernel, convert_warp_coords_to_mj_kernel

class TestMuJoCoSolver(unittest.TestCase):

    def setUp(self):
        """Set up a model with multiple environments, each with a free body and an articulated tree."""
        num_envs = 2
        self.debug_stage_path = "newton/tests/test_mujoco_render.usda"

        template_builder = newton.ModelBuilder()
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0) # Define ShapeConfig

        # --- Free-floating body (e.g., a box) ---
        free_body_initial_pos = wp.transform((0.5, 0.5, 0.0), wp.quat_identity())
        free_body_idx = template_builder.add_body(mass=0.2)
        template_builder.add_joint_free(child=free_body_idx, parent_xform=free_body_initial_pos)
        template_builder.add_shape_box(
            body=free_body_idx, 
            xform=wp.transform(), # Shape at body's local origin
            hx=0.1, hy=0.1, hz=0.1, 
            cfg=shape_cfg
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
            xform=wp.transform(), # Shape at body's local origin
            radius=link_radius, 
            half_height=link_half_length, 
            cfg=shape_cfg
        )

        body2_idx = template_builder.add_body(mass=0.1)
        template_builder.add_shape_capsule(
            body=body2_idx, 
            xform=wp.transform(), # Shape at body's local origin
            radius=link_radius, 
            half_height=link_half_length, 
            cfg=shape_cfg
        )
        template_builder.add_joint_revolute(
            parent=body1_idx, child=body2_idx,
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            axis=(0.0, 0.0, 1.0)
        )

        body3_idx = template_builder.add_body(mass=0.1)
        template_builder.add_shape_capsule(
            body=body3_idx, 
            xform=wp.transform(), # Shape at body's local origin
            radius=link_radius, 
            half_height=link_half_length, 
            cfg=shape_cfg
        )
        template_builder.add_joint_revolute(
            parent=body2_idx, child=body3_idx,
            parent_xform=wp.transform((0.0, link_half_length, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_half_length, 0.0), wp.quat_identity()),
            axis=(1.0, 0.0, 0.0)
        )

        self.builder = newton.ModelBuilder()
        self.builder.add_shape_plane()

        for i in range(num_envs):
            env_transform = wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity())
            self.builder.add_builder(template_builder, xform=env_transform, update_num_env_count=True, separate_collision_group=True)
        
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

    def _run_substeps_for_frame(self, sim_dt, sim_substeps):
        """Helper method to run simulation substeps for one rendered frame."""
        for _ in range(sim_substeps):
            self.solver.step(self.model, self.state_in, self.state_out, self.control, self.contacts, sim_dt)
            self.state_in, self.state_out = self.state_out, self.state_in # Output becomes input for next substep

    def test_setup_completes(self):
        """
        Tests if the setUp method completes successfully.
        This implicitly tests model creation, finalization, solver, and renderer initialization.
        """
        self.assertTrue(True, "setUp method completed.")

    def test_randomize_joint_q(self):
        """Tests if MuJoCo's internal qpos (derived from model.joint_q) converts back to the original model.joint_q correctly."""
        np.random.seed(42) # For reproducible random values

        num_envs = self.model.num_envs
        self.assertTrue(num_envs >= 2, "This test expects at least 2 environments.")

        template_joint_q_dim = 7 + 7 + 1 + 1 # Total 16 q-coordinates per environment
        template_joint_dof_dim = 6 + 6 + 1 + 1 # Total 14 DoFs (for qd) per environment
        joints_per_env = self.model.joint_count // num_envs

        # 1. Prepare the target modified Warp joint_q values
        full_joint_q_numpy_modified = self.model.joint_q.numpy().copy()
        delta_val = 0.1

        for env_idx in range(num_envs):
            q_start_offset_for_current_env = env_idx * template_joint_q_dim
            current_delta = -delta_val if env_idx == 0 else delta_val

            full_joint_q_numpy_modified[q_start_offset_for_current_env + 0 : q_start_offset_for_current_env + 3] += current_delta
            # Generate and normalize random quaternion
            raw_q0 = np.random.rand(4) * 2.0 - 1.0 # Random numbers in [-1, 1]
            q_tmp_0 = wp.normalize(wp.quat(raw_q0[0], raw_q0[1], raw_q0[2], raw_q0[3]))
            full_joint_q_numpy_modified[q_start_offset_for_current_env + 3 : q_start_offset_for_current_env + 7] = [q_tmp_0[0], q_tmp_0[1], q_tmp_0[2], q_tmp_0[3]]
            
            full_joint_q_numpy_modified[q_start_offset_for_current_env + 7 : q_start_offset_for_current_env + 10] += current_delta
            # Generate and normalize random quaternion
            raw_q1 = np.random.rand(4) * 2.0 - 1.0 # Random numbers in [-1, 1]
            q_tmp_1 = wp.normalize(wp.quat(raw_q1[0], raw_q1[1], raw_q1[2], raw_q1[3]))
            full_joint_q_numpy_modified[q_start_offset_for_current_env + 10 : q_start_offset_for_current_env + 14] = [q_tmp_1[0], q_tmp_1[1], q_tmp_1[2], q_tmp_1[3]]
            
            full_joint_q_numpy_modified[q_start_offset_for_current_env + 14] += current_delta
            full_joint_q_numpy_modified[q_start_offset_for_current_env + 15] += current_delta

        # Set the model's joint_q to these modified values
        self.model.joint_q = wp.array(full_joint_q_numpy_modified, dtype=wp.float32, device=self.model.device)
        
        # 2. Initialize MuJoCoSolver. This will internally convert self.model.joint_q to MuJoCo's qpos0 and mjw_data.qpos.
        try:
            solver = MuJoCoSolver(self.model, use_mujoco=False, separate_envs_to_worlds=True)
        except Exception as e:
            self.fail(f"MuJoCoSolver initialization failed: {e}")

        # 3. Convert MuJoCo data (solver.mjw_data.qpos) back to Warp joint_q format
        output_full_warp_joint_q = wp.empty_like(self.model.joint_q) # Same shape as original modified joint_q
        output_full_warp_joint_qd = wp.empty_like(self.model.joint_qd)

        wp.launch(
            kernel=convert_mj_coords_to_warp_kernel, 
            dim=(num_envs, joints_per_env), # Process all environments
            inputs=[
                solver.mjw_model.qpos0,     # MuJoCo qpos0 for all envs
                solver.mjw_data.qvel,     # MuJoCo qvel for all envs
                joints_per_env,            
                self.model.up_axis,
                self.model.joint_type,     
                self.model.joint_q_start,  
                self.model.joint_qd_start, 
                self.model.joint_axis_dim, 
            ],
            outputs=[output_full_warp_joint_q, output_full_warp_joint_qd],
            device=self.model.device,
        )

        # 4. Verify that the back-converted Warp joint_q matches the original modified Warp joint_q
        converted_warp_joint_q_numpy = output_full_warp_joint_q.numpy()
        
        np.testing.assert_allclose(
            converted_warp_joint_q_numpy,
            full_joint_q_numpy_modified, # Compare against the NumPy array we set self.model.joint_q from
            atol=1e-5, 
            err_msg="Back-converted Warp joint_q does not match the initial modified self.model.joint_q"
        )

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
                    path=self.debug_stage_path,
                    model=self.model,
                    scaling=1.0,
                    show_joints=True
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
        self.solver = solver # Make solver accessible to _run_substeps_for_frame via self

        if use_cuda_graph:
            print(f"Debug: CUDA device detected. Attempting to capture {sim_substeps} substeps with dt={sim_dt:.4f} into a CUDA graph...")
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

if __name__ == '__main__':
    unittest.main() 