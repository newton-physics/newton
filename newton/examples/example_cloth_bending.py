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
# Example Sim Cloth Bending
#
# This simulation demonstrates cloth bending behavior using the Vertex Block
# Descent (VBD) integrator. A cloth mesh, initially curved, is dropped on
# the ground. The cloth maintains its curved shape due to bending stiffness,
# controlled by edge_ke and edge_kd parameters.
#
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.geometry.kernels
import newton.solvers.vbd.solver_vbd
import newton.utils


class Example:
    def __init__(self, stage_path="example_cloth_bending.usd", num_frames=300):
        fps = 60
        self.frame_dt = 1.0 / fps

        self.num_substeps = 10
        self.iterations = 10
        self.dt = self.frame_dt / self.num_substeps

        self.num_frames = num_frames
        self.sim_time = 0.0
        self.profiler = {}
        self.use_cuda_graph = wp.get_device().is_cuda

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("curvedSurface.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/cloth"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        self.input_scale_factor = 1.0
        self.renderer_scale_factor = 1.0
        vertices = [wp.vec3(v) * self.input_scale_factor for v in mesh_points]
        self.faces = mesh_indices.reshape(-1, 3)

        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 10.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 6.0),
            scale=1.0,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            tri_ke=5.0e1,
            tri_ka=5.0e1,
            tri_kd=1.0e-1,
            edge_ke=1.0e1,
            edge_kd=1.0e0,
        )

        builder.color(include_bending=True)
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.VBDSolver(
            self.model,
            self.iterations,
            handle_self_contact=True,
            self_contact_radius=0.2,
            self_contact_margin=0.35,
        )

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
            wp.set_module_options({"block_dim": 256}, newton.solvers.vbd.solver_vbd)
            wp.load_module(newton.solvers.vbd.solver_vbd, device=wp.get_device())
            wp.set_module_options({"block_dim": 16}, newton.geometry.kernels)
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
        default="example_cloth_bending.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_frames=args.num_frames)
        example.run()

        frame_times = example.profiler["step"]
        print(f"\nAverage frame sim time: {sum(frame_times) / len(frame_times):.2f} ms")

        if example.renderer:
            example.renderer.save()
