# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Viscous fluid flowing through a funnel using MPM.

A funnel-shaped mesh collider is filled with viscous fluid particles that
flow out through a narrow aperture at the bottom, demonstrating viscoplastic
material behavior with mesh collisions.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


class Example:
    def __init__(self, viewer, options):
        self.fps = options.fps
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.funnel_offset_z = options.funnel_offset_z

        self.viewer = viewer
        builder = newton.ModelBuilder()

        SolverImplicitMPM.register_custom_attributes(builder)

        # Create funnel mesh collider
        vertices, indices = Example.create_funnel_mesh(
            aperture_radius=options.funnel_aperture / 2.0,
            top_radius=options.funnel_top_radius,
            height=options.funnel_height,
            z_offset=options.funnel_offset_z,
        )
        mesh = newton.Mesh(vertices, indices, compute_inertia=False, is_solid=False)
        builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            cfg=newton.ModelBuilder.ShapeConfig(mu=options.funnel_friction),
        )

        # Fill funnel with particles
        Example.emit_particles(builder, options)

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=options.ground_friction))

        self.model = builder.finalize()
        self.model.set_gravity(options.gravity)

        # Set per-particle material properties
        self.model.mpm.viscosity.fill_(options.viscosity)
        self.model.mpm.tensile_yield_ratio.fill_(options.tensile_yield_ratio)
        self.model.mpm.friction.fill_(options.friction)

        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = options.voxel_size
        mpm_options.tolerance = options.tolerance
        mpm_options.max_iterations = options.max_iterations
        mpm_options.strain_basis = options.strain_basis
        mpm_options.collider_basis = options.collider_basis

        self.solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.viewer.show_particles = True
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "camera"):
            self.viewer.set_camera(pos=wp.vec3(0.45, -0.15, 0.25), pitch=-20.0, yaw=160.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        voxel_size = self.solver.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )
        # Check that some particles flowed through the funnel aperture
        positions = self.state_0.particle_q.numpy()
        below_funnel = np.sum(positions[:, 2] < self.funnel_offset_z)
        if below_funnel == 0:
            raise ValueError("No particles flowed through the funnel")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Scene configuration
        parser.add_argument("--funnel-aperture", type=float, default=0.02, help="Diameter of the narrow opening [m]")
        parser.add_argument("--funnel-top-radius", type=float, default=0.05, help="Radius of the wide opening [m]")
        parser.add_argument("--funnel-height", type=float, default=0.2, help="Vertical extent of the funnel [m]")
        parser.add_argument("--funnel-offset-z", type=float, default=0.2, help="Z position of the funnel bottom [m]")
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
        parser.add_argument("--fps", type=float, default=240.0)
        parser.add_argument("--substeps", type=int, default=1)

        # Material parameters
        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--viscosity", type=float, default=50.0)
        parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=1.0)
        parser.add_argument("--friction", "-mu", type=float, default=0.0)
        parser.add_argument("--ground-friction", type=float, default=0.5)
        parser.add_argument("--funnel-friction", type=float, default=0.0)

        # Solver parameters
        parser.add_argument("--max-iterations", "-it", type=int, default=250)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.005)
        parser.add_argument("--strain-basis", "-sb", type=str, default="P0")
        parser.add_argument("--collider-basis", "-cb", type=str, default="S2")

        return parser

    @staticmethod
    def create_funnel_mesh(aperture_radius, top_radius, height, z_offset, thickness=0.005, num_segments=64):
        """Generate a thick-walled funnel mesh open at both ends.

        The funnel is a truncated cone extruded radially outward by *thickness*
        to form a closed shell.  Four rings of vertices define the inner and
        outer surfaces, connected by top and bottom rims.  All face normals
        point outward from the solid wall so the collider works correctly.

        Args:
            aperture_radius: Radius of the narrow opening at the bottom [m].
            top_radius: Radius of the wide opening at the top [m].
            height: Vertical extent of the funnel [m].
            z_offset: Z position of the funnel bottom [m].
            thickness: Radial wall thickness [m].
            num_segments: Number of segments around the circumference.

        Returns:
            Tuple of (vertices, indices) suitable for :class:`newton.Mesh`.
        """
        theta = np.linspace(0.0, 2.0 * np.pi, num_segments, endpoint=False)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        n = num_segments

        def ring(radius, z):
            return np.column_stack([radius * cos_t, radius * sin_t, np.full(n, z)])

        # Ring 0: inner bottom (aperture)
        # Ring 1: inner top
        # Ring 2: outer top
        # Ring 3: outer bottom (aperture)
        vertices = np.vstack(
            [
                ring(aperture_radius, z_offset),
                ring(top_radius, z_offset + height),
                ring(top_radius + thickness, z_offset + height),
                ring(aperture_radius + thickness, z_offset),
            ]
        ).astype(np.float32)

        indices = []
        for i in range(n):
            j = (i + 1) % n
            r0_i, r0_j = i, j
            r1_i, r1_j = i + n, j + n
            r2_i, r2_j = i + 2 * n, j + 2 * n
            r3_i, r3_j = i + 3 * n, j + 3 * n

            # Inner wall (ring0 -> ring1): normals face inward (toward axis)
            indices.extend([r0_i, r1_i, r0_j])
            indices.extend([r0_j, r1_i, r1_j])

            # Outer wall (ring3 -> ring2): normals face outward (away from axis)
            indices.extend([r3_i, r3_j, r2_i])
            indices.extend([r2_i, r3_j, r2_j])

            # Top rim (ring1 -> ring2): normals face up
            indices.extend([r1_i, r2_i, r1_j])
            indices.extend([r1_j, r2_i, r2_j])

            # Bottom rim (ring3 -> ring0): normals face down
            indices.extend([r3_i, r0_i, r3_j])
            indices.extend([r3_j, r0_i, r0_j])

        indices = np.array(indices, dtype=np.int32)
        return vertices, indices

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        """Fill the funnel interior with particles on a jittered grid."""
        voxel_size = args.voxel_size
        density = args.density
        particles_per_cell = 3.0

        aperture_radius = args.funnel_aperture / 2.0
        top_radius = args.funnel_top_radius
        height = args.funnel_height
        z_offset = args.funnel_offset_z

        # Bounding box of the funnel interior
        particle_lo = np.array([-top_radius, -top_radius, z_offset])
        particle_hi = np.array([top_radius, top_radius, z_offset + height])

        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = np.prod(cell_size)
        radius = np.max(cell_size) * 0.5
        mass = cell_volume * density

        dim_x = particle_res[0] + 1
        dim_y = particle_res[1] + 1
        dim_z = particle_res[2] + 1

        px = np.arange(dim_x) * cell_size[0]
        py = np.arange(dim_y) * cell_size[1]
        pz = np.arange(dim_z) * cell_size[2]
        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

        # Add jitter
        jitter = 2.0 * np.max(cell_size)
        rng = np.random.default_rng(422)
        points += (rng.random(points.shape) - 0.5) * jitter

        # Shift to funnel bounding box origin
        points += particle_lo

        # Cone filter: keep points inside the funnel with an inward margin
        margin = voxel_size
        z_frac = np.clip((points[:, 2] - z_offset) / height, 0.0, 1.0)
        r_max = aperture_radius + z_frac * (top_radius - aperture_radius) - margin
        r_xy = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        inside = (r_xy < r_max) & (points[:, 2] > z_offset + margin) & (points[:, 2] < z_offset + height - margin)
        points = points[inside]

        builder.add_particles(
            pos=points.tolist(),
            vel=np.zeros_like(points).tolist(),
            mass=[mass] * points.shape[0],
            radius=[radius] * points.shape[0],
        )


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
