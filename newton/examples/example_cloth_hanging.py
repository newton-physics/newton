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
# Example Sim Cloth Hanging
#
# This simulation demonstrates a simple cloth hanging behavior. A planar cloth mesh is fixed on one
# side and hangs under gravity, colliding with the ground.
#
###########################################################################

from enum import Enum

import warp as wp

import newton
import newton.geometry.kernels
import newton.solvers.vbd.solver_vbd
import newton.utils


class SolverType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    VBD = "vbd"

    def __str__(self):
        return self.value


class Example:
    def __init__(
        self,
        stage_path="example_cloth_hanging.usd",
        solver_type: SolverType = SolverType.VBD,
        height=32,
        width=64,
        num_frames=300,
    ):
        self.solver_type = solver_type

        self.sim_height = height
        self.sim_width = width

        fps = 60
        self.frame_dt = 1.0 / fps

        if self.solver_type == SolverType.EULER:
            self.num_substeps = 32
        else:
            self.num_substeps = 10

        self.iterations = 10
        self.dt = self.frame_dt / self.num_substeps

        self.num_frames = num_frames
        self.sim_time = 0.0
        self.profiler = {}
        self.use_cuda_graph = wp.get_device().is_cuda

        self.renderer_scale_factor = 1.0

        builder = newton.ModelBuilder()

        if self.solver_type == SolverType.EULER:
            ground_cfg = builder.default_shape_cfg.copy()
            ground_cfg.ke = 1.0e2
            ground_cfg.kd = 5.0e1
            builder.add_ground_plane(cfg=ground_cfg)
        else:
            builder.add_ground_plane()

        # common cloth properties
        common_params = {
            "pos": wp.vec3(0.0, 0.0, 4.0),
            "rot": wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            "vel": wp.vec3(0.0, 0.0, 0.0),
            "dim_x": self.sim_width,
            "dim_y": self.sim_height,
            "cell_x": 0.1,
            "cell_y": 0.1,
            "mass": 0.1,
            "fix_left": True,
            "edge_ke": 1.0e1,
            "edge_kd": 0.0,
            "particle_radius": 0.05,
        }

        solver_params = {}
        if self.solver_type == SolverType.EULER:
            solver_params = {
                "tri_ke": 1.0e3,
                "tri_ka": 1.0e3,
                "tri_kd": 1.0e1,
            }

        elif self.solver_type == SolverType.XPBD:
            solver_params = {
                "add_springs": True,
                "spring_ke": 1.0e3,
                "spring_kd": 1.0e1,
            }

        else:  # self.solver_type == SolverType.VBD
            solver_params = {
                "tri_ke": 1.0e3,
                "tri_ka": 1.0e3,
                "tri_kd": 1.0e-1,
            }

        builder.add_cloth_grid(**common_params, **solver_params)

        if self.solver_type == SolverType.VBD:
            builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0

        if self.solver_type == SolverType.EULER:
            self.solver = newton.solvers.SemiImplicitSolver(model=self.model)
        elif self.solver_type == SolverType.XPBD:
            self.solver = newton.solvers.XPBDSolver(
                model=self.model,
                iterations=self.iterations,
            )
        else:  # self.solver_type == SolverType.VBD
            self.solver = newton.solvers.VBDSolver(model=self.model, iterations=self.iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if stage_path is not None:
            self.renderer = newton.utils.SimRendererOpenGL(
                path=stage_path,
                model=self.model,
                scaling=self.renderer_scale_factor,
                enable_backface_culling=False,
            )
        else:
            self.renderer = None

        self.cuda_graph = None
        if self.use_cuda_graph:
            # Initial graph launch, load modules (necessary for drivers prior to CUDA 12.3)
            if self.solver_type == SolverType.VBD:
                wp.set_module_options({"block_dim": 256}, newton.solvers.vbd.solver_vbd)
                wp.load_module(newton.solvers.vbd.solver_vbd, device=wp.get_device())
            wp.set_module_options({"block_dim": 256}, newton.geometry.kernels)
            wp.load_module(newton.geometry.kernels, device=wp.get_device())

            with wp.ScopedCapture() as capture:
                self.simulate_substeps()
            self.cuda_graph = capture.graph

    def simulate_substeps(self):
        for _ in range(self.num_substeps):
            contacts = self.model.collide(self.state_0)
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, contacts, self.dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step", print=False, dict=self.profiler):
            if self.use_cuda_graph:
                wp.capture_launch(self.cuda_graph)
            else:
                self.simulate_substeps()
            self.sim_time += self.frame_dt

    def run(self):
        for i in range(self.num_frames):
            self.step()
            self.render()
            print(f"[{i:4d}/{self.num_frames}]")

    def render(self):
        if self.renderer is None:
            return

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
        default="example_cloth_hanging.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument(
        "--solver",
        help="Type of solver",
        type=SolverType,
        choices=list(SolverType),
        default=SolverType.VBD,
    )
    parser.add_argument("--width", type=int, default=64, help="Cloth resolution in x.")
    parser.add_argument("--height", type=int, default=32, help="Cloth resolution in y.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path,
            solver_type=args.solver,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
        example.run()

        frame_times = example.profiler["step"]
        print(f"\nAverage frame sim time: {sum(frame_times) / len(frame_times):.2f} ms")

        if example.renderer:
            example.renderer.save()
