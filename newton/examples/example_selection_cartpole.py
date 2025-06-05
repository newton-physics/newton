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

USE_HELPER_API = True
COLLAPSE_FIXED_JOINTS = True
VERBOSE = False


class Example:
    def __init__(self, stage_path=None, num_envs=8):
        self.num_envs = num_envs

        builder, stage_info = replicate_environment(
            newton.examples.get_asset("envs/cartpole_env.usda"),
            "/World/envs/env_0",
            "/World/envs/env_{}",
            num_envs,
            (2.0, 3.0, 0.0),
            # USD importer args
            collapse_fixed_joints=COLLAPSE_FIXED_JOINTS,
            joint_ordering="dfs",
        )

        up_axis = stage_info.get("up_axis") or newton.Axis.Z

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.solver = newton.solvers.MuJoCoSolver(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # =======================
        # get cartpole view
        # =======================
        self.cartpoles = ArticulationView(self.model, "/World/envs/*/Robot", verbose=VERBOSE)

        # print(self.cartpoles.get_attribute("body_q", self.state_0))
        # print(self.cartpoles.get_attribute("body_qd", self.state_0))
        # print(self.cartpoles.get_attribute("joint_q", self.state_0))
        # print(self.cartpoles.get_attribute("joint_qd", self.state_0))
        # print(self.cartpoles.get_attribute("joint_f", self.control))

        # =========================
        # randomize initial state
        # =========================
        cart_positions = 2.0 - 4.0 * torch.rand(num_envs)
        pole_angles = math.pi / 16.0 - math.pi / 8.0 * torch.rand(num_envs)
        axis_transforms = torch.stack([cart_positions, pole_angles], dim=1)
        height = 2.0
        if USE_HELPER_API:
            # root transforms
            root_transforms = wp.to_torch(self.cartpoles.get_root_transforms(self.state_0))
            root_transforms[:, 2] = height
            self.cartpoles.set_root_transforms(self.state_0, root_transforms)
            # dof transforms
            self.cartpoles.set_dof_positions(self.state_0, axis_transforms)
        else:
            # root transforms (we need to use joint_X_p for fixed joints)
            root_transforms = wp.to_torch(self.cartpoles.get_attribute("joint_X_p", self.model))
            root_transforms[:, 0, 2] = height
            self.cartpoles.set_attribute("joint_X_p", self.model, root_transforms)
            # dof transforms
            self.cartpoles.set_attribute("joint_q", self.state_0, axis_transforms)

        if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
            self.cartpoles.eval_fk(self.state_0)

        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(
                path=stage_path,
                model=self.model,
                scaling=1.0,
                up_axis=str(up_axis),
                screen_width=1280,
                screen_height=720,
                camera_pos=(0, 3, 10),
            )

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.model, self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # =========================
        # get observations
        # =========================
        if USE_HELPER_API:
            dof_positions = wp.to_torch(self.cartpoles.get_dof_positions(self.state_0))
        else:
            dof_positions = wp.to_torch(self.cartpoles.get_attribute("joint_q", self.state_0))

        # =========================
        # apply controls
        # =========================
        dof_forces = torch.zeros((self.num_envs, self.cartpoles.joint_axis_count))
        dof_forces[:, 0] = torch.where(dof_positions[:, 0] > 0, -100, 100)
        if USE_HELPER_API:
            self.cartpoles.set_dof_forces(self.control, dof_forces)
        else:
            self.cartpoles.set_attribute("joint_f", self.control, dof_forces)

        # simulate
        with wp.ScopedTimer("step", active=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

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
        default="example_selection_cartpole.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=12000, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=16, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device), torch.device(wp.device_to_torch(args.device)):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
