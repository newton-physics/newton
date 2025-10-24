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

"""Tests that reset results in the same data and converge of solver is preserved."""

import copy
import unittest

import mujoco
import numpy as np
import warp as wp

import newton
import newton.utils
from newton.selection import ArticulationView
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestAnymalReset(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device()
        self.num_worlds = 1
        self.headless = True

    def _setup_simulation(self, cone_type):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e2,
            limit_kd=1.0e0,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        stage_path = str(asset_path / "usd" / "anymal_d.usda")
        builder.add_usd(
            stage_path,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )

        builder.add_ground_plane()

        self.sim_time = 0.0
        fps = 50
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder.joint_q[:3] = [0.0, 0.0, 0.92]

        builder.joint_q[3:7] = [
            0.0,
            0.0,
            0.7071,
            0.7071,
        ]

        builder.joint_q[7:] = [
            0.0,
            -0.4,
            0.8,
            0.0,
            -0.4,
            0.8,
            0.0,
            0.4,
            -0.8,
            0.0,
            0.4,
            -0.8,
        ]

        for i in range(len(builder.joint_dof_mode)):
            builder.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION

        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 0
            builder.joint_target_kd[i] = 0

        self.model = builder.finalize()

        if cone_type == mujoco.mjtCone.mjCONE_PYRAMIDAL:
            impratio = 1.0
        else:
            impratio = 100.0

        self.solver = newton.solvers.SolverMuJoCo(
            self.model, solver=2, cone=cone_type, impratio=impratio, iterations=100, ls_iterations=50, njmax=300
        )

        if self.headless:
            self.viewer = None
        else:
            self.viewer = newton.viewer.ViewerGL()
            self.viewer.set_model(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.anymal = ArticulationView(
            self.model, "*/anymal/base", verbose=False, exclude_joint_types=[newton.JointType.FREE]
        )
        self.default_root_transforms = wp.clone(self.anymal.get_root_transforms(self.model))
        self.default_root_velocities = wp.clone(self.anymal.get_root_velocities(self.model))

        self.initial_dof_positions = wp.clone(self.anymal.get_dof_positions(self.state_0))
        self.initial_dof_velocities = wp.clone(self.anymal.get_dof_velocities(self.state_0))
        self.simulate()
        self.save_initial_mjw_data()

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(self.device)
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
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

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
            "cfrc_int",
            "collision_worldid",
            "epa_face",
            "epa_horizon",
            "epa_index",
            "epa_map",
            "epa_norm2",
            "epa_pr",
            "epa_vert",
            "epa_vert1",
            "epa_vert2",
            "epa_vert_index1",
            "epa_vert_index2",
        }

        for attr_name in all_attributes:
            if attr_name in skip_attributes:
                continue
            attr_value = getattr(mjw_data, attr_name)

            if hasattr(attr_value, "numpy"):
                self.initial_mjw_data[attr_name] = attr_value.numpy().copy()
            elif isinstance(attr_value, np.ndarray):
                self.initial_mjw_data[attr_name] = attr_value.copy()
            elif isinstance(attr_value, int | float | bool):
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
                        tolerance = 1e-3
                        diff_mask = ~np.isclose(initial_value, current_value, atol=tolerance, equal_nan=True)

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
                else:
                    identical_count += 1

        if differences:
            for i, diff in enumerate(differences, 1):
                print(f"  {i}. {diff}")
            return False
        else:
            return True

    def reset_robot_state(self):
        self.anymal.set_root_transforms(self.state_0, self.default_root_transforms)
        self.anymal.set_dof_positions(self.state_0, self.initial_dof_positions)
        self.anymal.set_root_velocities(self.state_0, self.default_root_velocities)
        self.anymal.set_dof_velocities(self.state_0, self.initial_dof_velocities)

        self.sim_time = 0.0

    def propagate_reset_state(self):
        if self.use_cuda_graph and self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def get_current_iterations(self):
        current_iterations = self.solver.mjw_data.solver_niter
        current_iter_numpy = current_iterations.numpy()
        return int(current_iter_numpy.max())

    def get_max_iterations(self):
        return int(self.solver.mjw_model.opt.iterations)

    def _run_reset_test(self, cone_type):
        self._setup_simulation(cone_type)
        num_steps = 50 if self.device.is_cuda else 5
        check_interval = 10 if self.device.is_cuda else 4
        for i in range(num_steps):
            self.step()
            if not self.headless:
                self.render()
            if i % check_interval == 0:
                current_iters = self.get_current_iterations()
                max_iters = self.get_max_iterations()
                self.assertLess(
                    current_iters,
                    max_iters * 0.9,
                    f"Solver iterations ({current_iters}) are too high (>{max_iters * 0.9:.0f}), "
                    f"max allowed is {max_iters}. Simulation is unstable!",
                )

        self.reset_robot_state()
        self.propagate_reset_state()
        mjw_data_matches = self.compare_mjw_data_with_initial()

        for i in range(num_steps):
            self.step()
            if not self.headless:
                self.render()
            if i % check_interval == 0:
                current_iters = self.get_current_iterations()
                max_iters = self.get_max_iterations()
                self.assertLess(
                    current_iters,
                    max_iters * 0.9,
                    f"Solver iterations ({current_iters}) are too high (>{max_iters * 0.9:.0f}), "
                    f"max allowed is {max_iters}. Simulation is unstable!",
                )

        self.assertTrue(
            mjw_data_matches,
            f"mjw_data after reset does not match initial state with {self._cone_type_name(cone_type)} cone",
        )


def test_reset_functionality(test: TestAnymalReset, device, cone_type):
    test.device = device
    with wp.ScopedDevice(device):
        test._run_reset_test(cone_type)


devices = get_test_devices()
add_function_test(
    TestAnymalReset,
    "test_reset_functionality_elliptic",
    test_reset_functionality,
    devices=devices,
    cone_type=mujoco.mjtCone.mjCONE_ELLIPTIC,
    check_output=False,
)
add_function_test(
    TestAnymalReset,
    "test_reset_functionality_pyramidal",
    test_reset_functionality,
    devices=devices,
    cone_type=mujoco.mjtCone.mjCONE_PYRAMIDAL,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main()
