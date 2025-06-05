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

# import math

import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.utils.isaaclab import replicate_environment
from newton.utils.selection import ArticulationView

USE_HELPER_API = True
COLLAPSE_FIXED_JOINTS = True
VERBOSE = False


class Example:
    def __init__(self, stage_path=None, num_envs=8):
        self.num_envs = num_envs

        builder, stage_info = replicate_environment(
            newton.examples.get_asset("envs/humanoid_env.usd"),
            "/World/envs/env_0",
            "/World/envs/env_{}",
            num_envs,
            (4.0, 4.0, 0.0),
            # USD importer args
            collapse_fixed_joints=True,
            joint_ordering="dfs",
        )

        up_axis = stage_info.get("up_axis") or newton.Axis.Z

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.solver = newton.solvers.MuJoCoSolver(self.model)

        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(
                path=stage_path,
                model=self.model,
                scaling=2.0,
                up_axis=str(up_axis),
                screen_width=1280,
                screen_height=720,
                camera_pos=(0, 4, 30),
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.next_reset = 0.0

        # ===========================================================
        # create articulation view
        # ===========================================================
        self.humanoids = ArticulationView(self.model, "/World/envs/*/Robot/torso", verbose=VERBOSE)

        print(f"articulation count: {self.humanoids.count}")
        print(f"link_count:         {self.humanoids.link_count}")
        print(f"joint_count:        {self.humanoids.joint_count}")
        print(f"joint_axis_count:   {self.humanoids.joint_axis_count}")
        print(f"joint_coord_count:  {self.humanoids.joint_coord_count}")
        print(f"joint_dof_count:    {self.humanoids.joint_dof_count}")

        if USE_HELPER_API:
            # separate root and dof transforms
            self.default_root_transforms = wp.to_torch(self.humanoids.get_root_transforms(self.model)).clone()
            self.default_root_transforms[:, 2] = 1.5
            self.default_dof_positions = wp.to_torch(self.humanoids.get_dof_positions(self.model)).clone()
            # separate root and dof velocities
            self.default_root_velocities = wp.to_torch(self.humanoids.get_root_velocities(self.model)).clone()
            # self.default_root_velocities[:, 2] = 1.0 * math.pi  # rotate about z-axis
            # self.default_root_velocities[:, 5] = 5.0  # move up z-axis
            self.default_dof_velocities = wp.to_torch(self.humanoids.get_dof_velocities(self.model)).clone()
        else:
            # combined root and dof transforms
            self.default_transforms = wp.to_torch(self.humanoids.get_attribute("joint_q", self.model)).clone()
            self.default_transforms[:, 2] = 1.5  # z-coordinate of articulation root
            # combined root and dof velocities
            self.default_velocities = wp.to_torch(self.humanoids.get_attribute("joint_qd", self.model)).clone()
            # self.default_velocities[:, 2] = 1.0 * math.pi  # rotate about z-axis
            # self.default_velocities[:, 5] = 5.0  # move up z-axis

        # create disjoint index groups to alternate between
        all_indices = torch.arange(num_envs, dtype=torch.int32)
        self.indices_0 = all_indices[::2]
        self.indices_1 = all_indices[1::2]

        # reset all
        self.reset()
        self.next_reset = self.sim_time + 2.0

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # explicit collisions needed without MuJoCo solver
            if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
                newton.collision.collide(self.model, self.state_0)

            self.solver.step(self.model, self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time >= self.next_reset:
            self.reset(self.indices_0)
            self.next_reset = self.sim_time + 2.0
            self.indices_0, self.indices_1 = self.indices_1, self.indices_0

        # =========================
        # apply random controls
        # =========================
        dof_forces = 20.0 - 40.0 * torch.rand((self.num_envs, self.humanoids.joint_axis_count))
        if USE_HELPER_API:
            self.humanoids.set_dof_forces(self.control, dof_forces)
        else:
            # include the root free joint
            forces = torch.cat([torch.zeros((self.num_envs, 6)), dof_forces], axis=1)
            self.humanoids.set_attribute("joint_f", self.control, forces)

        with wp.ScopedTimer("step", active=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def reset(self, indices=None):
        # ==============================
        # set transforms and velocities
        # ==============================
        if USE_HELPER_API:
            # set root and dof states separately
            self.humanoids.set_root_transforms(self.state_0, self.default_root_transforms, indices=indices)
            self.humanoids.set_root_velocities(self.state_0, self.default_root_velocities, indices=indices)
            self.humanoids.set_dof_positions(self.state_0, self.default_dof_positions, indices=indices)
            self.humanoids.set_dof_velocities(self.state_0, self.default_dof_velocities, indices=indices)
        else:
            # set root and dof states together
            self.humanoids.set_attribute("joint_q", self.state_0, self.default_transforms, indices=indices)
            self.humanoids.set_attribute("joint_qd", self.state_0, self.default_velocities, indices=indices)

        if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
            self.humanoids.eval_fk(self.state_0, indices=indices)

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_selection_humanoid.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1200, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=16, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device), torch.device(wp.device_to_torch(args.device)):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            example.step()
            example.render()

            # import time
            # time.sleep(0.5)

        if example.renderer:
            example.renderer.save()
