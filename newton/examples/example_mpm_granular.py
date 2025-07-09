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

"""Common definitions for types and constants."""

import argparse

import numpy as np
import warp as wp

import newton
from newton.solvers.implicit_mpm import ImplicitMPMSolver


class Example:
    def __init__(self, options: argparse.Namespace):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        Example.emit_particles(builder, options)

        if options.collider is not None:
            collider = _create_collider_mesh(options.collider)
            builder.add_shape_mesh(
                body=-1, mesh=newton.geometry.Mesh(collider.points.numpy(), collider.indices.numpy())
            )
            colliders = [collider]
        else:
            colliders = []

        builder.add_ground_plane()

        builder.gravity = wp.vec3(options.gravity)

        options.grid_padding = 0 if options.dynamic_grid else 5
        options.yield_stresses = wp.vec3(
            options.yield_stress,
            -options.stretching_yield_stress,
            options.compression_yield_stress,
        )

        model: newton.Model = builder.finalize()
        model.particle_mu = options.friction_coeff

        self.frame_dt = 1.0 / options.fps
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.model = model
        self.state_0: newton.State = model.state()
        self.state_1: newton.State = model.state()

        self.sim_time = 0.0
        self.solver = ImplicitMPMSolver(model, options)
        self.solver.setup_collider(model, colliders=colliders)

        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        if options.headless:
            self.renderer = None
        else:
            self.renderer = newton.utils.SimRendererOpenGL(self.model, "MPM Granular")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("simulate", synchronize=True):
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", synchronize=True):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

    @staticmethod
    def emit_particles(builder: newton.ModelBuilder, args):
        max_fraction = args.max_fraction
        voxel_size = args.voxel_size

        particles_per_cell = 3
        particle_lo = np.array(args.emit_lo)
        particle_hi = np.array(args.emit_hi)
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        Example._spawn_particles(builder, particle_res, particle_lo, particle_hi, max_fraction)

    @staticmethod
    def _spawn_particles(builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, packing_fraction):
        Nx = res[0]
        Ny = res[1]
        Nz = res[2]

        px = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
        py = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
        pz = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

        cell_size = (bounds_hi - bounds_lo) / res
        cell_volume = np.prod(cell_size)

        radius = np.max(cell_size) * 0.5
        volume = np.prod(cell_volume) * packing_fraction

        rng = np.random.default_rng()
        points += 2.0 * radius * (rng.random(points.shape) - 0.5)
        vel = np.zeros_like(points)

        builder.particle_q = points
        builder.particle_qd = vel
        builder.particle_mass = np.full(points.shape[0], volume)
        builder.particle_radius = np.full(points.shape[0], radius)
        builder.particle_flags = np.zeros(points.shape[0], dtype=int)


@wp.kernel
def _fill_triangle_indices(
    face_offsets: wp.array(dtype=int),
    face_vertex_indices: wp.array(dtype=int),
    tri_vertex_indices: wp.array(dtype=int),
):
    fid = wp.tid()

    if fid == 0:
        beg = 0
    else:
        beg = face_offsets[fid - 1]
    end = face_offsets[fid]

    for t in range(beg, end - 2):
        tri_index = t - 2 * fid
        tri_vertex_indices[3 * tri_index + 0] = face_vertex_indices[beg]
        tri_vertex_indices[3 * tri_index + 1] = face_vertex_indices[t + 1]
        tri_vertex_indices[3 * tri_index + 2] = face_vertex_indices[t + 2]


def mesh_triangle_indices(face_index_counts, face_indices):
    """Zero-vertex triangulates a polynomial mesh"""

    face_count = len(face_index_counts)

    face_offsets = np.cumsum(face_index_counts)
    tot_index_count = int(face_offsets[-1])

    tri_count = tot_index_count - 2 * face_count
    tri_index_count = 3 * tri_count

    face_offsets = wp.array(face_offsets, dtype=int)
    face_indices = wp.array(face_indices, dtype=int)

    tri_indices = wp.empty(tri_index_count, dtype=int)

    wp.launch(
        kernel=_fill_triangle_indices,
        dim=face_count,
        inputs=[face_offsets, face_indices, tri_indices],
    )

    return tri_indices


def _create_collider_mesh(collider: str):
    """Create a collider mesh from a string; either load from a file or create a simple predefined shape."""

    if collider == "wedge":
        cube_faces = np.array(
            [
                [0, 2, 6, 4],
                [1, 5, 7, 3],
                [0, 4, 5, 1],
                [2, 3, 7, 6],
                [0, 1, 3, 2],
                [4, 6, 7, 5],
            ]
        )

        # Generate cube vertex positions and rotate them by 45 degrees along z
        cube_points = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        cube_points = (cube_points * [1, 1, 2.5]) @ np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        )
        cube_points = cube_points + np.array([-0.9, 1, -1.2])
        cube_indices = mesh_triangle_indices(np.full(6, 4), cube_faces.flatten())

        return wp.Mesh(wp.array(cube_points, dtype=wp.vec3), wp.array(cube_indices, dtype=int))

    mesh_points, mesh_indices = newton.utils.load_mesh(collider)
    return wp.Mesh(wp.array(mesh_points, dtype=wp.vec3), wp.array(mesh_indices, dtype=int))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")

    parser.add_argument("--collider", type=str)

    parser.add_argument("--emit-lo", type=float, nargs=3, default=[-1, 0, -1])
    parser.add_argument("--emit-hi", type=float, nargs=3, default=[1, 2, 1])
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, -10, 0])
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)

    parser.add_argument("--max-fraction", type=float, default=1.0)

    parser.add_argument("--compliance", type=float, default=0.0)
    parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.3)
    parser.add_argument("--friction-coeff", "-mu", type=float, default=0.48)
    parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
    parser.add_argument("--compression-yield-stress", "-cys", type=float, default=1.0e8)
    parser.add_argument("--stretching-yield-stress", "-sys", type=float, default=1.0e8)
    parser.add_argument("--unilateral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gauss-seidel", "-gs", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-iterations", "-it", type=int, default=250)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.1)
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(args)

        for _ in range(args.num_frames):
            example.step()
            example.render()
