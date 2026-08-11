# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a deformable LIMX ARAP bunny onto a four-corner-pinned cloth."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils


class Example:
    """Couple one volumetric ARAP bunny to one membrane cloth."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 0.01
        self.sim_time = 0.0
        self.cloth_cells = 40
        self.cloth_height = 0.45

        mesh_path = Path(__file__).resolve().parents[1] / "assets" / "bunny_tet.npz"
        bunny_mesh = newton.TetMesh.create_from_file(str(mesh_path))
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))

        self.bunny_particle_start = builder.particle_count
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.78),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi),
            scale=0.15,
            vel=wp.vec3(0.0),
            mesh=bunny_mesh,
            density=1000.0,
            k_mu=0.0,
            k_lambda=0.0,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )
        self.bunny_particle_stop = builder.particle_count
        self.cloth_particle_start = builder.particle_count

        cloth_side = self.cloth_cells + 1
        cloth_particle_count = cloth_side * cloth_side
        cloth_positions = [
            wp.vec3(-0.4 + 0.8 * x / self.cloth_cells, -0.4 + 0.8 * y / self.cloth_cells, self.cloth_height)
            for y in range(cloth_side)
            for x in range(cloth_side)
        ]
        builder.add_particles(
            pos=cloth_positions,
            vel=[wp.vec3(0.0)] * cloth_particle_count,
            mass=[0.3 / cloth_particle_count] * cloth_particle_count,
            radius=[0.0025] * cloth_particle_count,
        )
        self.cloth_particle_stop = builder.particle_count
        self.cloth_particle_count = cloth_particle_count

        cloth_triangles_local = []
        for y in range(self.cloth_cells):
            for x in range(self.cloth_cells):
                lower_left = y * cloth_side + x
                lower_right = lower_left + 1
                upper_left = lower_left + cloth_side
                upper_right = upper_left + 1
                if (x + y) % 2 == 0:
                    cloth_triangles_local.extend(
                        ((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left))
                    )
                else:
                    cloth_triangles_local.extend(
                        ((lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left))
                    )
        cloth_triangles_local = np.asarray(cloth_triangles_local, dtype=np.int32)
        self.cloth_triangles = cloth_triangles_local + self.cloth_particle_start
        self.cloth_triangle_start = builder.tri_count
        builder.add_triangles(
            self.cloth_triangles[:, 0],
            self.cloth_triangles[:, 1],
            self.cloth_triangles[:, 2],
        )
        self.cloth_triangle_stop = builder.tri_count

        self.model = builder.finalize()
        self.bunny_tetrahedra = self.model.tet_indices.numpy()
        bunny_inverse_rest_matrices = self.model.tet_poses.numpy()
        cloth_inverse_rest_matrices = self.model.tri_poses.numpy()[self.cloth_triangle_start : self.cloth_triangle_stop]
        cloth_areas = self.model.tri_areas.numpy()[self.cloth_triangle_start : self.cloth_triangle_stop]

        cloth_edges = newton.utils.MeshAdjacency(cloth_triangles_local).edge_indices
        cloth_interior_edges = cloth_edges[cloth_edges[:, 1] >= 0]
        self.cloth_dihedral_indices = (cloth_interior_edges[:, [2, 3, 0, 1]] + self.cloth_particle_start).astype(
            np.int32
        )
        self.cloth_anchor_indices = [
            self.cloth_particle_start,
            self.cloth_particle_start + self.cloth_cells,
            self.cloth_particle_start + self.cloth_cells * cloth_side,
            self.cloth_particle_stop - 1,
        ]
        self.cloth_center_index = self.cloth_particle_start + cloth_particle_count // 2

        constraints = [
            newton.solvers.ConstraintTetrahedronARAP(
                self.bunny_tetrahedra.tolist(),
                [wp.mat33(*matrix.reshape(-1)) for matrix in bunny_inverse_rest_matrices],
                [1.0e6] * self.model.tet_count,
                self.model.particle_count,
                self.model.device,
            ),
            newton.solvers.ConstraintAnchor(
                self.cloth_anchor_indices,
                [self.model.particle_q.numpy()[index] for index in self.cloth_anchor_indices],
                [1.0e7] * len(self.cloth_anchor_indices),
                self.model.particle_count,
                self.model.device,
            ),
            newton.solvers.ConstraintTriangleElastic(
                self.cloth_triangles.tolist(),
                cloth_inverse_rest_matrices,
                cloth_areas,
                [wp.vec3(1.0e4, 1.0e4, 1.0e3)] * len(self.cloth_triangles),
                self.model.particle_count,
                self.model.device,
            ),
            newton.solvers.ConstraintDihedralBending(
                self.cloth_dihedral_indices,
                self.model.particle_q.numpy(),
                1.0e-4,
                self.model.particle_count,
                self.model.device,
            ),
        ]
        self.self_collision = newton.solvers.ConstraintSelfCollision(
            self.model,
            thickness=None,
            stiffness=None,
            max_contacts=262144,
            stiffness_factors=(0.5, 0.3, 1.5),
            geometry_radius_scale=0.25,
            geometry_radius_topology_local_only=True,
            friction=0.0,
            enable_edge_face=True,
            use_outward_normals=False,
        )
        self.solver = newton.solvers.SolverLIMX(
            self.model,
            constraints,
            nonlinear_iterations=1,
            linear_iterations=50,
            velocity_damping=1.0,
            dynamic_operator=self.self_collision,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.15, -1.45, 1.05), -12.0, 140.0)
        self.capture()

    def capture(self):
        """Capture one CUDA simulation step when supported."""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Advance one LIMX step."""
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        self.state_0.assign(self.state_1)

    def step(self):
        """Advance one rendered frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the bunny and cloth."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Keep the coupled state finite."""
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("LIMX bunny-cloth state must remain finite")

    def test_final(self):
        """Keep the final coupled state finite."""
        self.test_post_step()

    @staticmethod
    def create_parser():
        """Create the standard Newton example parser."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=300)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
