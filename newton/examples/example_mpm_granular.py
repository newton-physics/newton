import argparse
from typing import Optional

import numpy as np
import warp as wp
import warp.examples
import warp.sim.render
from warp.sim import Model, State

import newton
from newton.solvers.solver_implicit_mpm import ImplicitMPMSolver


class Example:
    def __init__(self, options: argparse.Namespace, collider_mesh_path: Optional[str] = None):
        builder = newton.ModelBuilder()
        Example.emit_particles(builder, options)

        if collider_mesh_path is not None:
            collider = load_collider_mesh(collider_mesh_path)
            builder.set_ground_plane(offset=np.min(collider.points.numpy()[:, 1]))
            colliders = [collider]
        else:
            builder.set_ground_plane(offset=np.min(builder.particle_q[:, 1]))
            colliders = []

        builder.gravity = wp.vec3(options.gravity)

        model: Model = builder.finalize()

        self.frame_dt = 1.0 / options.fps
        self.sim_substeps = options.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.model = model
        self.state_0: State = model.state()
        self.state_1: State = model.state()

        self.sim_time = 0.0
        self.solver = ImplicitMPMSolver(model, options, colliders=colliders)

        self.solver.enrich_state(self.state_0)
        self.solver.enrich_state(self.state_1)

        self.particle_radius = self.solver.voxel_size / 6

        if options.grains:
            self.grains = self.solver.sample_grains(
                self.state_0,
                particle_radius=self.particle_radius,
                grains_per_particle=10,
            )
        else:
            self.grains = None

        self.renderer = newton.utils.SimRendererOpenGL(self.model, "MPM Granular")

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.model, self.state_0, self.state_1, None, None, self.sim_dt)
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
    def emit_particles(builder: wp.sim.ModelBuilder, args):
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
    def _spawn_particles(builder: wp.sim.ModelBuilder, res, bounds_lo, bounds_hi, packing_fraction):
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

        points += 2.0 * radius * (np.random.rand(*points.shape) - 0.5)
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


def load_collider_mesh(stage_path, prim_path):
    # Create collider mesh
    from pxr import Usd, UsdGeom

    collider_stage = Usd.Stage.Open(stage_path)
    usd_mesh = UsdGeom.Mesh(collider_stage.GetPrimAtPath(prim_path))
    usd_counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get())
    usd_indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get())

    collider_points = wp.array(usd_mesh.GetPointsAttr().Get(), dtype=wp.vec3)
    collider_indices = mesh_triangle_indices(usd_counts, usd_indices)
    return wp.Mesh(collider_points, collider_indices)


def _create_collider_mesh(kind: str):
    if kind == "wedge":
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
        cube_points = (cube_points * [10, 10, 25]) @ np.array(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        )
        cube_points = cube_points + np.array([-9, 0, -12])

        cube_indices = mesh_triangle_indices(np.full(6, 4), cube_faces.flatten())

        return wp.Mesh(wp.array(cube_points, dtype=wp.vec3), wp.array(cube_indices, dtype=int))

    return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--emit_lo", type=float, nargs=3, default=[-10, 0, -10])
    parser.add_argument("--emit_hi", type=float, nargs=3, default=[10, 20, 10])
    parser.add_argument("--gravity", type=float, nargs=3, default=[0, -10, 0])
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--grains", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--max_fraction", type=float, default=1.0)

    parser.add_argument("--compliance", type=float, default=0.0)
    parser.add_argument("--poisson", type=float, default=0.3)
    parser.add_argument("--friction", type=float, default=0.48)
    parser.add_argument("--yield_stress", "-ys", type=float, default=0.0)
    parser.add_argument(
        "--compression_yield_stress", "-cys", type=float, default=1.0e8
    )
    parser.add_argument(
        "--stretching_yield_stress", "-sys", type=float, default=1.0e8
    )
    parser.add_argument(
        "--unilateral", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--pad_grid", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--gs", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max_iters", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1.0e-5)
    parser.add_argument("--voxel_size", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")

    options = parser.parse_args()

    example = Example(options)

    for _ in range(options.num_frames):
        example.step()
        example.render()

    if example.renderer:
        example.renderer.save()
