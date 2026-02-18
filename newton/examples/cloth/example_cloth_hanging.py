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
# This simulation demonstrates a simple cloth hanging behavior. A planar cloth
# mesh is fixed on one side and hangs under gravity, colliding with the ground.
#
# Command: python -m newton.examples cloth_hanging (--solver [semi_implicit, style3d, xpbd, vbd])
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton.solvers import style3d


class Example:
    def __init__(
        self,
        viewer,
        args=None,
        solver_type: str = "vbd",
        height=32,
        width=64,
    ):
        # setup simulation parameters first
        self.solver_type = solver_type

        self.sim_height = height
        self.sim_width = width
        self.sim_time = 0.0

        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        if self.solver_type == "semi_implicit":
            self.sim_substeps = 32
        elif self.solver_type == "style3d":
            self.sim_substeps = 2
        else:
            self.sim_substeps = 10

        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        if self.solver_type == "style3d":
            builder = newton.ModelBuilder()
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        else:
            builder = newton.ModelBuilder()

        if self.solver_type == "semi_implicit":
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
        if self.solver_type == "semi_implicit":
            solver_params = {
                "tri_ke": 1.0e3,
                "tri_ka": 1.0e3,
                "tri_kd": 1.0e1,
            }

        elif self.solver_type == "style3d":
            common_params.pop("edge_ke")
            solver_params = {
                "tri_aniso_ke": wp.vec3(1.0e4, 1.0e4, 1.0e3),
                "edge_aniso_ke": wp.vec3(2.0e-6, 1.0e-6, 5.0e-6),
            }

        elif self.solver_type == "xpbd":
            solver_params = {
                "add_springs": True,
                "spring_ke": 1.0e3,
                "spring_kd": 1.0e0,
            }

        else:  # self.solver_type == "vbd"
            solver_params = {
                "tri_ke": 1.0e3,
                "tri_ka": 1.0e3,
                "tri_kd": 1.0e-1,
            }

        if self.solver_type == "style3d":
            style3d.add_cloth_grid(builder, **common_params, **solver_params)
        else:
            builder.add_cloth_grid(**common_params, **solver_params)

        if self.solver_type == "vbd":
            builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0

        if self.solver_type == "semi_implicit":
            self.solver = newton.solvers.SolverSemiImplicit(model=self.model)
        elif self.solver_type == "style3d":
            self.solver = newton.solvers.SolverStyle3D(
                model=self.model,
                iterations=self.iterations,
            )
            self.solver.precompute(builder)
        elif self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                model=self.model,
                iterations=self.iterations,
            )
        else:  # self.solver_type == "vbd"
            self.solver = newton.solvers.SolverVBD(
                model=self.model,
                iterations=self.iterations,
                particle_enable_self_contact=True,
                particle_self_contact_radius=0.02,
                particle_self_contact_margin=0.03,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create collision pipeline (default)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        if self.solver_type != "style3d":
            newton.examples.test_particle_state(
                self.state_0,
                "particles are above the ground",
                lambda q, qd: q[2] > 0.0,
            )

        min_x = -float(self.sim_width) * 0.11
        p_lower = wp.vec3(min_x, -4.0, -1.8)
        p_upper = wp.vec3(0.1, 7.0, 4.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Create parser with base arguments
    parser = newton.examples.create_parser()

    # Add solver-specific arguments
    parser.add_argument(
        "--solver",
        help="Type of solver",
        type=str,
        choices=["semi_implicit", "style3d", "xpbd", "vbd"],
        default="vbd",
    )
    parser.add_argument("--width", type=int, default=64, help="Cloth resolution in x.")
    parser.add_argument("--height", type=int, default=32, help="Cloth resolution in y.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(
        viewer=viewer,
        args=args,
        solver_type=args.solver,
        height=args.height,
        width=args.width,
    )

    newton.examples.run(example, args)
