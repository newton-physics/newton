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

import math

import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.utils.isaaclab import replicate_environment
from newton.utils.selection import ArticulationView


class Example:
    def __init__(self, stage_path="example_ant.usd", num_envs=8):
        self.num_envs = num_envs

        builder, stage_info, env_offsets = replicate_environment(
            newton.examples.get_asset("ant_prototype.usda"),
            "/World/envs/env_0",
            "/World/envs/env_{}",
            num_envs,
            (5.0, 5.0, 0.0),
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
        self.ants = ArticulationView(self.model, "/World/envs/*/Robot/torso", include_free_joint=True)

        print(f"articulation count: {self.ants.count}")
        print(f"link_count:         {self.ants.link_count}")
        print(f"joint_count:        {self.ants.joint_count}")
        print(f"joint_axis_count:   {self.ants.joint_axis_count}")

        print(f"joint_q shape:      {self.ants.get_attribute_shape('joint_q')}")
        print(f"joint_qd shape:     {self.ants.get_attribute_shape('joint_qd')}")
        print(f"joint_f shape:      {self.ants.get_attribute_shape('joint_f')}")
        print(f"joint_target shape: {self.ants.get_attribute_shape('joint_target')}")
        print(f"body_q shape:       {self.ants.get_attribute_shape('body_q')}")
        print(f"body_qd shape:      {self.ants.get_attribute_shape('body_qd')}")

        # set all dofs to the middle of their range by default
        dof_limit_lower = wp.to_torch(self.ants.get_attribute("joint_limit_lower", self.model))
        dof_limit_upper = wp.to_torch(self.ants.get_attribute("joint_limit_upper", self.model))
        default_dof_transforms = 0.5 * (dof_limit_lower + dof_limit_upper)

        if self.ants.include_free_joint:
            # combined root and dof transforms
            self.default_transforms = wp.to_torch(self.ants.get_attribute("joint_q", self.model)).clone()
            self.default_transforms[:, 2] = 0.8  # z-coordinate of articulation root
            self.default_transforms[:, 7:] = default_dof_transforms
            # combined root and dof velocities
            self.default_velocities = wp.to_torch(self.ants.get_attribute("joint_qd", self.model)).clone()
            self.default_velocities[:, 2] = 0.5 * math.pi  # rotate about z-axis
            self.default_velocities[:, 5] = 5.0  # move up z-axis
        else:
            # root transforms
            self.default_root_transforms = wp.to_torch(self.ants.get_root_transforms(self.model)).clone()
            self.default_root_transforms[:, 2] = 0.8
            # dof transforms
            self.default_dof_transforms = default_dof_transforms
            # root velocities
            self.default_root_velocities = wp.to_torch(self.ants.get_root_velocities(self.model)).clone()
            self.default_root_velocities[:, 2] = 0.5 * math.pi  # rotate about z-axis
            self.default_root_velocities[:, 5] = 5.0  # move up z-axis
            # dof velocities
            self.default_dof_velocities = wp.to_torch(self.ants.get_attribute("joint_qd", self.model)).clone()

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
        joint_forces = 300.0 - 600.0 * torch.rand((self.num_envs, 8))
        if self.ants.include_free_joint:
            # include the leading root joint (pad with zeros)
            joint_forces = torch.cat([torch.zeros((self.num_envs, 6)), joint_forces], axis=1)
        self.ants.set_attribute("joint_f", self.control, joint_forces)

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
        if self.ants.include_free_joint:
            # set root and dof transforms together
            self.ants.set_attribute("joint_q", self.state_0, self.default_transforms, indices=indices)
            # set root and dof velocities together
            self.ants.set_attribute("joint_qd", self.state_0, self.default_velocities, indices=indices)
        else:
            # set root and dof transforms separately
            self.ants.set_root_transforms(self.state_0, self.default_root_transforms, indices=indices)
            self.ants.set_attribute("joint_q", self.state_0, self.default_dof_transforms, indices=indices)
            # set root and dof velocities separately
            self.ants.set_root_velocities(self.state_0, self.default_root_velocities, indices=indices)
            self.ants.set_attribute("joint_qd", self.state_0, self.default_dof_velocities, indices=indices)

        if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
            self.ants.eval_fk(self.state_0, indices=indices)

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
        default="example_ant.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1200, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=16, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            example.step()
            example.render()

            # import time
            # time.sleep(0.2)

        if example.renderer:
            example.renderer.save()
