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

###########################################################################
# Example Anymal C walk
#
# Shows how to control Anymal C with a policy pretrained in physx.
#
# Example usage:
# uv run --extra cu12 newton/examples/example_anymal_c_walk_physx_policy.py
#
###########################################################################

import warp as wp

wp.config.enable_backward = False

import newton
import newton.utils
from newton import State


class Example:
    def __init__(self, stage_path=None, headless=False):
        self.device = wp.get_device()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75


        if stage_path is None:
            asset_path = newton.utils.download_asset("anymal_usd")
            stage_path = str(asset_path /  "anymal_d.usda")
        newton.utils.parse_usd(
            stage_path,
            builder,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_non_physics_prims=False,
        )

        builder.add_ground_plane()

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 50
        self.frame_dt = 1.0e0 / fps

        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder.joint_q[:3] = [0.0, 0.0, 0.62]

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
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        self.renderer = None if headless else newton.viewer.RendererOpenGL(self.model, stage_path)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            warp_array = wp.zeros(18, dtype=wp.float32, device=self.device)
            self.control.joint_target = warp_array
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        help="Path to the output URDF file.",
    )
    parser.add_argument("--num-frames", type=int, default=1000, help="Total number of frames.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, headless=args.headless)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
