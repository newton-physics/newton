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
        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        # save a reference to the viewer
        self.viewer = viewer
        builder = newton.ModelBuilder()

        # Register MPM custom attributes before adding particles
        SolverImplicitMPM.register_custom_attributes(builder)

        Example.emit_particles(builder, options)
        builder.add_ground_plane()
        self.model = builder.finalize()

        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = options.voxel_size

        # Initialize MPM solver
        self.solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Setup grain rendering

        self.grains = self.solver.sample_render_grains(self.state_0, options.points_per_particle)
        grain_radius = options.voxel_size / (3 * options.points_per_particle)
        self.grain_radii = wp.full(self.grains.size, value=grain_radius, dtype=float, device=self.model.device)
        self.grain_colors = wp.full(
            self.grains.size, value=wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=self.model.device
        )

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)

            # update grains
            self.solver.update_particle_frames(self.state_0, self.state_1, self.sim_dt)
            self.solver.update_render_grains(self.state_0, self.state_1, self.grains, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -0.05,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "grains", points=self.grains.flatten(), radii=self.grain_radii, colors=self.grain_colors, hidden=False
        )
        self.viewer.end_frame()

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        voxel_size = args.voxel_size

        particles_per_cell = 3
        particle_lo = np.array([-0.5, -0.5, 0.0])
        particle_hi = np.array([0.5, 0.5, 2.0])
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        Example._spawn_particles(builder, particle_res, particle_lo, particle_hi, density=2500)

    @staticmethod
    def _spawn_particles(
        builder: newton.ModelBuilder,
        res,
        bounds_lo,
        bounds_hi,
        density,
    ):
        cell_size = (bounds_hi - bounds_lo) / res
        cell_volume = np.prod(cell_size)
        radius = np.max(cell_size) * 0.5
        mass = np.prod(cell_volume) * density

        builder.add_particle_grid(
            pos=wp.vec3(bounds_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=res[0] + 1,
            dim_y=res[1] + 1,
            dim_z=res[2] + 1,
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

    parser.add_argument("--voxel-size", "-dx", type=float, default=0.1)
    parser.add_argument("--points-per-particle", "-ppp", type=float, default=8)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
