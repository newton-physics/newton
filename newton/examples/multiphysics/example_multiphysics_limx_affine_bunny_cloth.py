# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop one high-rigidity affine bunny onto a four-corner-pinned LIMX cloth."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils


class Example:
    """Couple one ABD bunny to a particle cloth in one mixed Newton solve."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 0.01
        self.sim_time = 0.0
        self.cloth_cells = 100
        self.cloth_height = 0.45

        cloth_side = self.cloth_cells + 1
        cloth_particle_count = cloth_side * cloth_side
        cloth_positions = [
            wp.vec3(
                -0.4 + 0.8 * x / self.cloth_cells,
                -0.4 + 0.8 * y / self.cloth_cells,
                self.cloth_height,
            )
            for y in range(cloth_side)
            for x in range(cloth_side)
        ]
        cloth_triangles = []
        for y in range(self.cloth_cells):
            for x in range(self.cloth_cells):
                lower_left = y * cloth_side + x
                lower_right = lower_left + 1
                upper_left = lower_left + cloth_side
                upper_right = upper_left + 1
                if (x + y) % 2 == 0:
                    cloth_triangles.extend(
                        ((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left))
                    )
                else:
                    cloth_triangles.extend(
                        ((lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left))
                    )
        self.cloth_triangles = np.asarray(cloth_triangles, dtype=np.int32)
        cloth_positions_np = np.asarray(cloth_positions, dtype=np.float32)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_particles(
            pos=cloth_positions,
            vel=[wp.vec3(0.0)] * cloth_particle_count,
            mass=[0.3 / cloth_particle_count] * cloth_particle_count,
            radius=[0.0025] * cloth_particle_count,
        )
        builder.add_triangles(
            self.cloth_triangles[:, 0],
            self.cloth_triangles[:, 1],
            self.cloth_triangles[:, 2],
        )
        self.model = builder.finalize()

        cloth_adjacency = newton.utils.MeshAdjacency(self.cloth_triangles)
        interior_edges = cloth_adjacency.edge_indices[cloth_adjacency.edge_indices[:, 1] >= 0]
        self.cloth_dihedral_indices = interior_edges[:, [2, 3, 0, 1]].astype(np.int32)
        self.cloth_anchor_indices = [
            0,
            self.cloth_cells,
            self.cloth_cells * cloth_side,
            cloth_particle_count - 1,
        ]
        self.cloth_anchor_targets = cloth_positions_np[self.cloth_anchor_indices].copy()
        self.cloth_center_index = cloth_particle_count // 2

        constraints = [
            newton.solvers.ConstraintAnchor(
                self.cloth_anchor_indices,
                [cloth_positions[index] for index in self.cloth_anchor_indices],
                [1.0e7] * len(self.cloth_anchor_indices),
                cloth_particle_count,
                self.model.device,
            ),
            newton.solvers.ConstraintTriangleElastic(
                self.cloth_triangles,
                self.model.tri_poses.numpy(),
                self.model.tri_areas.numpy(),
                [wp.vec3(1.0e4, 1.0e4, 1.0e3)] * len(self.cloth_triangles),
                cloth_particle_count,
                self.model.device,
            ),
            newton.solvers.ConstraintDihedralBending(
                self.cloth_dihedral_indices,
                cloth_positions_np,
                1.0e-4,
                cloth_particle_count,
                self.model.device,
            ),
        ]

        mesh_path = Path(__file__).resolve().parents[1] / "assets" / "bunny_tet.npz"
        bunny_mesh = newton.TetMesh.create_from_file(str(mesh_path))
        upright = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
        self.body_model = newton.solvers.AffineBodyModel(
            rest_vertices=0.15 * np.asarray(bunny_mesh.vertices),
            tetrahedron_indices=np.asarray(bunny_mesh.tet_indices).reshape(-1, 4),
            surface_triangle_indices=np.asarray(bunny_mesh.surface_tri_indices).reshape(-1, 3),
            density=1000.0,
            rigidity=1.0e8,
            initial_transform=wp.transform(wp.vec3(0.0), upright),
            device=self.model.device,
        )
        initial_affine_state = self.body_model.q.numpy()
        initial_affine_state[0, :3] = (0.0, 0.0, 0.55)
        self.body_model.q.assign(initial_affine_state)

        self.contact = newton.solvers.ConstraintAffineParticleContact(
            self.model,
            self.body_model,
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.0,
            friction=0.01,
            friction_epsilon=1.0e-4,
            max_contacts=262144,
        )
        self.solver = newton.solvers.SolverLIMXCoupled(
            self.model,
            constraints,
            self.body_model,
            nonlinear_iterations=1,
            linear_iterations=50,
            velocity_damping=1.0,
            dynamic_operator=self.contact,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.surface_positions = wp.empty(
            self.body_model.surface_vertex_count,
            dtype=wp.vec3,
            device=self.body_model.device,
        )
        self.solver.update_surface_positions(self.surface_positions)
        self.render_indices = wp.array(
            self.body_model.surface_triangle_indices.numpy().reshape(-1),
            dtype=wp.int32,
            device=self.body_model.device,
        )

        self.initial_cloth_center_height = float(cloth_positions_np[self.cloth_center_index, 2])
        self.minimum_cloth_center_height = self.initial_cloth_center_height
        self.minimum_affine_determinant = float(np.linalg.det(initial_affine_state[:, 3:].reshape(-1, 3, 3)).min())
        self.maximum_contact_overflow = 0
        self.maximum_contact_depth = 0.0
        self.contact_observed = False

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.15, -1.45, 1.05), -12.0, 140.0)
        self.capture()

    def capture(self) -> None:
        """Capture one complete mixed solve and surface reconstruction on CUDA."""
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        """Advance one mixed particle-affine step."""
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        self.state_0.assign(self.state_1)
        self.solver.update_surface_positions(self.surface_positions)

    def step(self) -> None:
        """Advance one 0.01-second rendered frame."""
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        """Render the cloth and reconstructed affine bunny surface."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/affine_bunny",
            self.surface_positions,
            self.render_indices,
            color=(0.82, 0.36, 0.30),
            backface_culling=False,
        )
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        """Keep both domains finite, orientation preserving, and within contact capacity."""
        cloth_positions = self.state_0.particle_q.numpy()
        cloth_velocities = self.state_0.particle_qd.numpy()
        affine_states = self.solver.q.numpy()
        affine_velocities = self.solver.qd.numpy()
        if not all(
            np.isfinite(values).all()
            for values in (cloth_positions, cloth_velocities, affine_states, affine_velocities)
        ):
            raise AssertionError("Coupled affine bunny-cloth state must remain finite")

        anchor_error = np.linalg.norm(
            cloth_positions[self.cloth_anchor_indices] - self.cloth_anchor_targets,
            axis=1,
        )
        if float(anchor_error.max()) >= 1.0e-4:
            raise AssertionError("A cloth corner anchor drifted by at least 0.1 mm")

        determinant = float(np.linalg.det(affine_states[:, 3:].reshape(-1, 3, 3)).min())
        if determinant <= 0.0:
            raise AssertionError("The affine bunny lost orientation")

        maximum_depth = 0.0
        overflow = 0
        active_contacts = 0
        for buffer in (
            self.contact.cloth_vertex_face_contacts,
            self.contact.affine_vertex_face_contacts,
            self.contact.edge_edge_contacts,
        ):
            count = min(int(buffer.count.numpy()[0]), buffer.capacity)
            active_contacts += count
            overflow += int(buffer.overflow_count.numpy()[0])
            if count:
                maximum_depth = max(maximum_depth, float(buffer.depths.numpy()[:count].max()))
        if overflow != 0:
            raise AssertionError("A mixed contact buffer overflowed")
        if maximum_depth >= 0.012:
            raise AssertionError("Mixed contact depth reached at least 12 mm")

        self.minimum_cloth_center_height = min(
            self.minimum_cloth_center_height,
            float(cloth_positions[self.cloth_center_index, 2]),
        )
        self.minimum_affine_determinant = min(self.minimum_affine_determinant, determinant)
        self.maximum_contact_overflow = max(self.maximum_contact_overflow, overflow)
        self.maximum_contact_depth = max(self.maximum_contact_depth, maximum_depth)
        self.contact_observed = self.contact_observed or active_contacts > 0

    def test_final(self) -> None:
        """Support the bunny above the deflected four-corner cloth."""
        self.test_post_step()
        if not self.contact_observed:
            raise AssertionError("The affine bunny never contacted the cloth")
        if self.initial_cloth_center_height - self.minimum_cloth_center_height <= 0.02:
            raise AssertionError("The cloth center did not deflect by more than 2 cm")
        cloth_center_height = float(self.state_0.particle_q.numpy()[self.cloth_center_index, 2])
        if float(self.solver.q.numpy()[0, 2]) - cloth_center_height <= 0.03:
            raise AssertionError("The cloth did not support the affine bunny")

    @staticmethod
    def create_parser():
        """Create the standard parser with a 300-frame default rollout."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=300)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
