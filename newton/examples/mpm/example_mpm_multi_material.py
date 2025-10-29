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
        sand_particles, snow_particles, mud_particles = Example.emit_particles(builder, voxel_size=options.voxel_size)

        builder.add_ground_plane()
        self.model = builder.finalize()

        sand_particles = wp.array(sand_particles, dtype=int, device=self.model.device)
        snow_particles = wp.array(snow_particles, dtype=int, device=self.model.device)
        mud_particles = wp.array(mud_particles, dtype=int, device=self.model.device)

        self.model.particle_ke = 1.0e15  # non-compliant particles
        self.model.particle_kd = 0.0
        self.model.particle_mu = 0.5

        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = options.voxel_size
        mpm_options.tolerance = options.tolerance
        mpm_options.max_iterations = options.max_iterations

        # Create MPM model from Newton model
        mpm_model = SolverImplicitMPM.Model(self.model, mpm_options)

        # multi-material setup
        # some properties like elastic stiffness, damping, can be adjusted directly on the model,
        # but not all yet. here we directly adjust the MPM model's material parameters

        mpm_model.material_parameters.yield_pressure[snow_particles].fill_(2.0e4)
        mpm_model.material_parameters.yield_stress[snow_particles].fill_(1.0e3)
        mpm_model.material_parameters.tensile_yield_ratio[snow_particles].fill_(0.05)
        mpm_model.material_parameters.friction[snow_particles].fill_(0.1)
        mpm_model.material_parameters.hardening[snow_particles].fill_(10.0)

        mpm_model.material_parameters.yield_pressure[mud_particles].fill_(1.0e10)
        mpm_model.material_parameters.yield_stress[mud_particles].fill_(3.0e2)
        mpm_model.material_parameters.tensile_yield_ratio[mud_particles].fill_(1.0)
        mpm_model.material_parameters.hardening[mud_particles].fill_(2.0)
        mpm_model.material_parameters.friction[mud_particles].fill_(0.0)

        mpm_model.notify_particle_material_changed()

        # Initialize MPM solver
        self.solver = SolverImplicitMPM(mpm_model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        # Assign different colors to each particle type
        self.particle_colors = wp.full(
            shape=self.model.particle_count, value=wp.vec3(0.1, 0.1, 0.2), device=self.model.device
        )
        self.particle_colors[sand_particles].fill_(wp.vec3(0.7, 0.6, 0.4))
        self.particle_colors[snow_particles].fill_(wp.vec3(0.75, 0.75, 0.8))
        self.particle_colors[mud_particles].fill_(wp.vec3(0.4, 0.25, 0.25))

        self.viewer.set_model(self.model)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test(self):
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -0.05,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_points(
            name="/model/particles",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=False,
        )
        self.viewer.end_frame()

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, voxel_size: float):
        # inactive particles
        Example._spawn_particles(
            builder,
            voxel_size,
            bounds_lo=np.array([-0.5, -0.5, 0.0]),
            bounds_hi=np.array([0.5, 0.5, 0.25]),
            density=1000.0,
            flags=0,
        )

        # sand particles
        sand_particles = Example._spawn_particles(
            builder,
            voxel_size,
            bounds_lo=np.array([0.25, -0.5, 0.5]),
            bounds_hi=np.array([0.75, 0.5, 0.75]),
            density=2500.0,
            flags=newton.ParticleFlags.ACTIVE,
        )

        # snow particles
        snow_particles = Example._spawn_particles(
            builder,
            voxel_size,
            bounds_lo=np.array([-0.75, -0.5, 0.5]),
            bounds_hi=np.array([-0.25, 0.5, 0.75]),
            density=300,
            flags=newton.ParticleFlags.ACTIVE,
        )

        # mud particles
        mud_particles = Example._spawn_particles(
            builder,
            voxel_size,
            bounds_lo=np.array([-0.5, -0.25, 1.0]),
            bounds_hi=np.array([0.5, 0.25, 1.5]),
            density=1000.0,
            flags=newton.ParticleFlags.ACTIVE,
        )

        return sand_particles, snow_particles, mud_particles

    @staticmethod
    def _spawn_particles(builder: newton.ModelBuilder, voxel_size, bounds_lo, bounds_hi, density, flags):
        particles_per_cell = 3
        res = np.array(
            np.ceil(particles_per_cell * (bounds_hi - bounds_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (bounds_hi - bounds_lo) / res
        cell_volume = np.prod(cell_size)
        radius = np.max(cell_size) * 0.5
        mass = np.prod(cell_volume) * density

        begin_id = len(builder.particle_q)
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
            flags=flags,
        )

        end_id = len(builder.particle_q)
        return np.arange(begin_id, end_id, dtype=int)


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()

    parser.add_argument("--max-iterations", "-it", type=int, default=250)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.05)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
