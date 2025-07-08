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
# Loads a MuJoCo model from MJCF into Newton and simulates it using the
# MuJoCo solver.
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

wp.config.enable_backward = False


class Example:
    def __init__(self, stage_path="example_humanoid.usd", num_envs=8, use_cuda_graph=True):
        self.num_envs = num_envs

        use_mujoco = False

        # set numpy random seed
        self.seed = 123
        self.rng = np.random.default_rng(self.seed)

        start_rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*self.rng.uniform(-1.0, 1.0, size=3))), -wp.pi * 0.5)

        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")

        articulation_builder = newton.ModelBuilder()

        newton.utils.parse_mjcf(
            mjcf_filename,
            articulation_builder,
            ignore_names=["floor", "ground"],
            up_axis="Z",
        )

        # joint initial positions
        articulation_builder.joint_q[:7] = [0.0, 0.0, 1.5, *start_rot]

        spacing = 3.0
        sqn = int(wp.ceil(wp.sqrt(float(self.num_envs))))

        builder = newton.ModelBuilder()
        for i in range(self.num_envs):
            pos = wp.vec3((i % sqn) * spacing, (i // sqn) * spacing, 0.0)
            articulation_builder.joint_q[7:] = self.rng.uniform(
                -1.0, 1.0, size=(len(articulation_builder.joint_q) - 7,)
            ).tolist()
            builder.add_builder(articulation_builder, xform=wp.transform(pos, wp.quat_identity()))
        builder.add_ground_plane()

        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # finalize model
        self.model = builder.finalize()

        self.control = self.model.control()

        self.solver = newton.solvers.MuJoCoSolver(
            self.model,
            use_mujoco=use_mujoco,
            solver="newton",
            integrator="euler",
            iterations=10,
            ls_iterations=5,
        )

        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(
                path=stage_path, model=self.model, scaling=1.0, show_joints=True
            )

        self.state_0, self.state_1 = self.model.state(), self.model.state()

        self.use_cuda_graph = (
            not getattr(self.solver, "use_mujoco", False) and wp.get_device().is_cuda and use_cuda_graph
        )

        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
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
        default="example_humanoid.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=12000, help="Total number of frames.")
    parser.add_argument("--num-envs", type=int, default=9, help="Total number of simulated environments.")
    parser.add_argument("--use-cuda-graph", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs, use_cuda_graph=args.use_cuda_graph)

        for frame_idx in range(args.num_frames):
            example.step()
            example.render()

            if example.renderer is None:
                print(f"[{frame_idx:4d}/{args.num_frames}]")

        if example.renderer:
            example.renderer.save()
