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
# Example G1
#
# Shows how to set up a simulation of a G1 articulation
# from a USD file using the newton.ModelBuilder().
# Note this example does not include a trained policy.
#
#
###########################################################################

import warp as wp

wp.config.enable_backward = False

import newton
import newton.utils


class Example:
    def __init__(self, num_envs=8, use_cuda_graph=True, headless=False):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.num_envs = num_envs

        self.device = wp.get_device()
        self.headless = headless

        # build model
        articulation_builder = newton.ModelBuilder()
        articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
        )
        asset_path = newton.utils.download_asset("unitree_g1")

        newton.utils.parse_usd(
            str(asset_path / "usd" / "g1_isaac.usd"),
            articulation_builder,
            xform=wp.transform(wp.vec3f(0, 0, 0.8), wp.quat_identity()),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
        )
        articulation_builder.approximate_meshes()
        spacing = 2.0
        sqn = int(wp.ceil(wp.sqrt(float(self.num_envs))))

        builder = newton.ModelBuilder()
        for i in range(self.num_envs):
            pos = wp.vec3((i % sqn) * spacing, (i // sqn) * spacing, 0.0)
            builder.add_builder(articulation_builder, xform=wp.transform(pos, wp.quat_identity()))
        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()

        self.sim_substeps = 6
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="euler",
            nefc_per_env=300,
            ncon_per_env=150,
            cone="elliptic",
            impratio=100,
            iterations=100,
            ls_iterations=50,
        )

        self.sim_dt = self.frame_dt / self.sim_substeps

        if not self.headless:
            self.renderer = newton.viewer.RendererOpenGL(
                model=self.model,
                scaling=1.0,
                screen_width=1280,
                screen_height=720,
                camera_pos=(0, 1, 4),
            )
        else:
            self.renderer = None

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = None
        self.use_cuda_graph = (
            not getattr(self.solver, "use_mujoco_cpu", False) and wp.get_device().is_cuda and use_cuda_graph
        )

        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.renderer and hasattr(self.renderer, "apply_picking_force"):
                self.renderer.apply_picking_force(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step", active=True):
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
    parser.add_argument("--num-frames", type=int, default=1000, help="Total number of frames.")
    parser.add_argument("--num-envs", type=int, default=1, help="Total number of simulated environments.")
    parser.add_argument(
        "--show-mujoco-viewer",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Toggle MuJoCo viewer next to Newton renderer when SolverMuJoCo is active.",
    )
    parser.add_argument("--use-cuda-graph", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(num_envs=args.num_envs, use_cuda_graph=args.use_cuda_graph, headless=args.headless)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
