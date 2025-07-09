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
# Example Sim Rigid Force
#
# Shows how to apply an external force (torque) to a rigid body causing
# it to roll.
#
###########################################################################

from __future__ import annotations

import warp as wp

import newton


class Example:
    def __init__(self, stage_path: str | None = "example_rigid_force.usd", headless: bool = False):
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder(up_axis="Y")

        b = builder.add_body(xform=wp.transform((0.0, 10.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=b, hx=1.0, hy=1.0, hz=1.0, cfg=newton.ModelBuilder.ShapeConfig(density=100.0))
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.ground = True

        self.solver = newton.solvers.XPBDSolver(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if not headless:
            self.renderer = newton.utils.SimRendererOpenGL(self.model, stage_path)
        elif stage_path:
            self.renderer = newton.utils.SimRendererUsd(self.model, stage_path)
        else:
            self.renderer = None

        # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)

            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.state_0.body_f.assign([[0.0, 0.0, -7000.0, 0.0, 0.0, 0.0]])

            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

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
        default="example_rigid_force.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Toggles the opening of an interactive window to play back animations in real time. Ignores --num-frames if used.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, headless=args.headless)

        if not args.headless:
            while example.renderer.is_running():
                example.step()
                example.render()
        else:
            for _ in range(args.num_frames):
                example.step()
                example.render()

        if example.renderer:
            example.renderer.save()
