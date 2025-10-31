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

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


class Example:
    def __init__(self, viewer, options):
        # setup simulation parameters first
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        # save a reference to the viewer
        self.viewer = viewer
        builder = newton.ModelBuilder()
        Example.emit_particles(builder, options)

        if options.collider and options.collider != "none":
            extents = (0.5, 2.0, 0.6)
            if options.collider == "cube":
                xform = wp.transform(wp.vec3(0.75, 0.0, 0.9), wp.quat_identity())
            elif options.collider == "wedge":
                xform = wp.transform(
                    wp.vec3(0.0, 0.0, 0.9), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 4.0)
                )

            builder.add_shape_box(
                body=-1,
                cfg=newton.ModelBuilder.ShapeConfig(mu=0.1),
                xform=xform,
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
            )

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        self.model = builder.finalize()
        self.model.particle_mu = options.friction_coeff
        self.model.set_gravity(options.gravity)

        # Disable model's particle material parameters,
        # we want to read directly from MPM options instead
        self.model.particle_ke = None
        self.model.particle_kd = None
        self.model.particle_cohesion = None
        self.model.particle_adhesion = None

        # Copy all remaining CLI arguments to MPM options
        mpm_options = SolverImplicitMPM.Options()
        for key in vars(options):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(options, key))

        # Create MPM model from Newton model
        mpm_model = SolverImplicitMPM.Model(self.model, mpm_options)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Initialize MPM solver and add supplemental state variables
        self.solver = SolverImplicitMPM(mpm_model, mpm_options)

        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.solver.grid_type == "fixed":
            if self.sim_substeps % 2 != 0:
                wp.utils.warn("Sim substeps must be even for graph capture of MPM step")
            else:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test(self):
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -0.05,
        )
        cube_extents = wp.vec3(0.5, 2.0, 0.6) * 0.9
        cube_center = wp.vec3(0.75, 0, 0.9)
        cube_lower = cube_center - cube_extents
        cube_upper = cube_center + cube_extents
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are outside the cube",
            lambda q, qd: not newton.utils.vec_inside_limits(q, cube_lower, cube_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        density = args.density
        voxel_size = args.voxel_size

        particles_per_cell = 3
        particle_lo = np.array(args.emit_lo)
        particle_hi = np.array(args.emit_hi)
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = np.prod(cell_size)

        radius = np.max(cell_size) * 0.5
        mass = np.prod(cell_volume) * density

        builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=particle_res[0] + 1,
            dim_y=particle_res[1] + 1,
            dim_z=particle_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
        )


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()

    # Scene configuration
    parser.add_argument("--collider", default="cube", choices=["cube", "wedge", "none"], type=str)
    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-1, -1, 1.5])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[1, 1, 3.5])
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)

    # Add MPM-specific arguments
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--air-drag", type=float, default=1.0)
    parser.add_argument("--critical-fraction", "-cf", type=float, default=0.0)

    parser.add_argument("--young-modulus", "-ym", type=float, default=1.0e15)
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.68)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--yield-pressure", "-yp", type=float, default=1.0e12)
    parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.0)
    parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
    parser.add_argument("--hardening", type=float, default=0.0)

    parser.add_argument("--grid-type", "-gt", type=str, default="sparse", choices=["sparse", "fixed", "dense"])
    parser.add_argument("--grid-padding", "-gp", type=int, default=0)
    parser.add_argument("--max-active-cell-count", "-mac", type=int, default=-1)
    parser.add_argument("--solver", "-s", type=str, default="gauss-seidel", choices=["gauss-seidel", "jacobi"])
    parser.add_argument("--transfer-scheme", "-ts", type=str, default="apic", choices=["apic", "pic"])

    parser.add_argument("--strain-basis", "-sb", type=str, default="P0", choices=["P0", "Q1"])

    parser.add_argument("--max-iterations", "-it", type=int, default=250)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.1)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
