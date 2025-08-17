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

import copy
import unittest

import mujoco
import numpy as np
import warp as wp

import newton
import newton.utils
from newton.utils.selection import ArticulationView


class TestAnymalReset(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device()
        self.num_envs = 1000
        self.headless = True

    def _setup_simulation(self, cone_type):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.01,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        articulation_builder.default_shape_cfg.ke = 5.0e4
        articulation_builder.default_shape_cfg.kd = 5.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e3
        articulation_builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anymal_usd")
        stage_path = str(asset_path / "anymal_d.usda")
        newton.utils.parse_usd(
            stage_path,
            articulation_builder,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )

        articulation_builder.joint_q[:3] = [0.0, 0.0, 0.8]

        for i in range(len(articulation_builder.joint_dof_mode)):
            articulation_builder.joint_dof_mode[i] = newton.JOINT_MODE_TARGET_POSITION

        for i in range(len(articulation_builder.joint_target_ke)):
            articulation_builder.joint_target_ke[i] = 0
            articulation_builder.joint_target_kd[i] = 0

        builder.add_builder(articulation_builder, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))

        robots_per_row = int(np.sqrt(self.num_envs))
        spacing = 3.0

        for i in range(1, self.num_envs):
            row = i // robots_per_row
            col = i % robots_per_row
            x = col * spacing
            y = row * spacing

            builder.add_builder(articulation_builder, xform=wp.transform(wp.vec3(x, y, 0.0), wp.quat_identity()))

        builder.add_ground_plane()

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 100
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.model = builder.finalize()
        self.solver = newton.solvers.MuJoCoSolver(
            self.model, solver=2, cone=cone_type, impratio=100.0, iterations=100, ls_iterations=50, nefc_per_env=200
        )

        self.renderer = None if self.headless else newton.utils.SimRendererOpenGL(self.model, stage_path)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.sim.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.anymal = ArticulationView(
            self.model, "/World/envs/env_0/Robot/base", verbose=False, exclude_joint_types=[newton.JOINT_FREE]
        )
        self.default_root_transforms = wp.to_torch(self.anymal.get_root_transforms(self.model)).clone()
        self.default_root_velocities = wp.to_torch(self.anymal.get_root_velocities(self.model)).clone()

        self.initial_dof_positions = wp.to_torch(self.anymal.get_dof_positions(self.state_0)).clone()
        self.initial_dof_velocities = wp.to_torch(self.anymal.get_dof_velocities(self.state_0)).clone()
        self.simulate()
        self.save_initial_mjw_data()

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def _cone_type_name(self, cone_type):
        if cone_type == mujoco.mjtCone.mjCONE_ELLIPTIC:
            return "ELLIPTIC"
        elif cone_type == mujoco.mjtCone.mjCONE_PYRAMIDAL:
            return "PYRAMIDAL"
        else:
            return f"UNKNOWN({cone_type})"

    def simulate(self):
        self.contacts = None
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render(self.state_0)
        self.renderer.end_frame()

    def save_initial_mjw_data(self):
        self.initial_mjw_data = {}
        mjw_data = self.solver.mjw_data

        all_attributes = [attr for attr in dir(mjw_data) if not attr.startswith("_")]

        skip_attributes = {
            "time",
            "solver_niter",
            "ncollision",
            "nsolving",
            "collision_pair",
            "collision_pairid",
            "solver_nisland",
            "nefc",
            "ncon",
        }

        for attr_name in all_attributes:
            if attr_name in skip_attributes:
                continue
            attr_value = getattr(mjw_data, attr_name)

            if hasattr(attr_value, "numpy"):
                self.initial_mjw_data[attr_name] = attr_value.numpy().copy()
            elif isinstance(attr_value, np.ndarray):
                self.initial_mjw_data[attr_name] = attr_value.copy()
            elif isinstance(attr_value, (int, float, bool)):
                self.initial_mjw_data[attr_name] = copy.deepcopy(attr_value)

    def compare_mjw_data_with_initial(self):
        mjw_data = self.solver.mjw_data
        differences = []
        identical_count = 0

        for attr_name, initial_value in self.initial_mjw_data.items():
            current_attr = getattr(mjw_data, attr_name)

            if hasattr(current_attr, "numpy"):
                current_value = current_attr.numpy()
            elif isinstance(current_attr, np.ndarray):
                current_value = current_attr
            else:
                current_value = current_attr

            if isinstance(initial_value, np.ndarray) and isinstance(current_value, np.ndarray):
                if initial_value.dtype == bool and current_value.dtype == bool:
                    if not np.array_equal(initial_value, current_value):
                        diff_mask = np.logical_xor(initial_value, current_value)
                        diff_indices = np.where(diff_mask)
                        num_different = len(diff_indices[0])
                        percent_different = (num_different / initial_value.size) * 100
                        differences.append(
                            f"{attr_name}: {num_different}/{initial_value.size} boolean values differ ({percent_different:.2f}%)"
                        )
                    else:
                        identical_count += 1
                else:
                    if not np.array_equal(initial_value, current_value):
                        max_diff = np.max(np.abs(initial_value - current_value))
                        mean_diff = np.mean(np.abs(initial_value - current_value))
                        tolerance = 1e-4
                        diff_mask = ~np.isclose(initial_value, current_value, atol=tolerance)

                        diff_indices = np.where(diff_mask)
                        num_different = len(diff_indices[0])
                        percent_different = (num_different / initial_value.size) * 100
                        if num_different > 0:
                            differences.append(
                                f"{attr_name}: max_diff={max_diff:.10f}, mean_diff={mean_diff:.10f}, shape={initial_value.shape}, {num_different}/{initial_value.size} different values ({percent_different:.2f}%)"
                            )
                        else:
                            identical_count += 1
                    else:
                        identical_count += 1
            else:
                if initial_value != current_value:
                    differences.append(f"{attr_name}: {initial_value} -> {current_value}")
                    print()
                else:
                    identical_count += 1

        if differences:
            for i, diff in enumerate(differences, 1):
                print(f"  {i}. {diff}")
            return False
        else:
            return True

    def render_frame(self):
        if self.renderer is None:
            return
        self.renderer.begin_frame(self.sim_time)
        self.renderer.render(self.state_0)
        self.renderer.end_frame()

    def reset_robot_state(self):
        self.anymal.set_root_transforms(self.state_0, self.default_root_transforms)

        initial_dof_positions = wp.from_torch(self.initial_dof_positions, dtype=wp.float32)
        self.anymal.set_dof_positions(self.state_0, initial_dof_positions)

        initial_root_velocities = wp.from_torch(self.default_root_velocities, dtype=wp.float32)
        self.anymal.set_root_velocities(self.state_0, initial_root_velocities)

        zero_dof_velocities = wp.from_torch(self.initial_dof_velocities, dtype=wp.float32)
        self.anymal.set_dof_velocities(self.state_0, zero_dof_velocities)

        self.sim_time = 0.0

    def propagate_reset_state(self):
        if self.use_cuda_graph and self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def get_iteration_difference(self):
        current_iterations = self.solver.mjw_data.solver_niter
        current_iter_numpy = current_iterations.numpy()
        max_iterations = int(current_iter_numpy.max())
        opt_iterations = int(self.solver.mjw_model.opt.iterations)
        return opt_iterations - max_iterations

    def _run_reset_test(self, cone_type):
        self._setup_simulation(cone_type)
        for i in range(100):
            if self.use_cuda_graph and self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
            self.sim_time += self.frame_dt
            if not self.headless:
                self.render_frame()
            if i % 10 == 0:
                iteration_diff = self.get_iteration_difference()
                self.assertGreaterEqual(
                    iteration_diff,
                    50,
                    f"Iteration difference ({iteration_diff}) is below 50, "
                    "indicating solver is approaching maximum iteration limit",
                )

        self.reset_robot_state()
        self.propagate_reset_state()
        mjw_data_matches = self.compare_mjw_data_with_initial()

        for _ in range(50):
            if self.use_cuda_graph and self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
            self.sim_time += self.frame_dt
            if not self.headless:
                self.render_frame()
            iteration_diff = self.get_iteration_difference()

            self.assertGreaterEqual(
                iteration_diff,
                50,
                f"Iteration difference ({iteration_diff}) is below 50, "
                "indicating solver is approaching maximum iteration limit",
            )

        self.assertTrue(
            mjw_data_matches,
            f"mjw_data after reset does not match initial state with {self._cone_type_name(cone_type)} cone",
        )

        if self.renderer:
            self.renderer.save()

    def test_reset_functionality_elliptic(self):
        """Test reset functionality with ELLIPTIC cone"""
        self._run_reset_test(mujoco.mjtCone.mjCONE_ELLIPTIC)

    def test_reset_functionality_pyramidal(self):
        """Test reset functionality with PYRAMIDAL cone"""
        self._run_reset_test(mujoco.mjtCone.mjCONE_PYRAMIDAL)


if __name__ == "__main__":
    wp.init()
    unittest.main()
