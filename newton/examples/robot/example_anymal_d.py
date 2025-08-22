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
# Example Anymal D walk
#
# Shows how to control Anymal D with multiple environments.
#
# Example usage:
# uv run newton/examples/example_anymal_d.py --num-envs 4
#
###########################################################################

import warp as wp

wp.config.enable_backward = False
import mujoco

import newton
import newton.utils


class Example:
    def __init__(self, stage_path="example_anymal_d.usd", headless=False, num_envs=8, use_cuda_graph=True):
        self.device = wp.get_device()
        self.num_envs = num_envs

        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
        )
        articulation_builder.default_shape_cfg.ke = 5.0e4
        articulation_builder.default_shape_cfg.kd = 5.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e3
        articulation_builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")
        newton.utils.parse_usd(
            asset_file,
            articulation_builder,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            load_non_physics_prims=False,
        )

        articulation_builder.joint_q[:3] = [0.0, 0.0, 0.62]
        if len(articulation_builder.joint_q) > 6:
            articulation_builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]

        for i in range(len(articulation_builder.joint_dof_mode)):
            articulation_builder.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
            articulation_builder.joint_target_ke[i] = 150
            articulation_builder.joint_target_kd[i] = 5

        spacing = 3.0
        sqn = int(wp.ceil(wp.sqrt(float(self.num_envs))))

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for i in range(self.num_envs):
            pos = wp.vec3((i % sqn) * spacing, (i // sqn) * spacing, 0)
            builder.add_builder(articulation_builder, xform=wp.transform(pos, wp.quat_identity()))

        builder.add_ground_plane()

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 50
        self.frame_dt = 1.0e0 / fps

        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, cone=mujoco.mjtCone.mjCONE_ELLIPTIC, impratio=100, iterations=100, ls_iterations=50
        )

        self.renderer = None
        if not headless and stage_path:
            self.renderer = newton.viewer.RendererOpenGL(self.model, stage_path)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device()) and use_cuda_graph

        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = None
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
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
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_anymal_d.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=1000, help="Total number of frames.")
    parser.add_argument("--num-envs", type=int, default=8, help="Total number of simulated environments.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-cuda-graph", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            headless=args.headless,
            num_envs=args.num_envs,
            use_cuda_graph=args.use_cuda_graph,
        )

        for frame_idx in range(args.num_frames):
            example.step()
            example.render()

            if example.renderer is None:
                print(f"[{frame_idx:4d}/{args.num_frames}]")

        if example.renderer:
            example.renderer.save()
