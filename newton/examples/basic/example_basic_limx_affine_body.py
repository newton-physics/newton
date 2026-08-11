# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic LIMX Affine Body
#
# Shows one collision-free affine body falling under gravity while a visible
# initial shear relaxes toward a rigid shape through PSD ARAP dynamics.
#
# Command: uv run -m newton.examples basic_limx_affine_body
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    """Relax a sheared affine cube while it falls under gravity."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 0.01
        self.sim_time = 0.0

        rest_vertices = np.asarray(
            [
                [-0.45, -0.45, -0.45],
                [0.45, -0.45, -0.45],
                [0.45, 0.45, -0.45],
                [-0.45, 0.45, -0.45],
                [-0.45, -0.45, 0.45],
                [0.45, -0.45, 0.45],
                [0.45, 0.45, 0.45],
                [-0.45, 0.45, 0.45],
            ],
            dtype=np.float32,
        )
        tetrahedron_indices = np.asarray(
            [
                [0, 1, 3, 4],
                [1, 2, 3, 6],
                [1, 3, 4, 6],
                [1, 4, 5, 6],
                [3, 4, 6, 7],
            ],
            dtype=np.int32,
        )
        surface_triangle_indices = np.asarray(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int32,
        )

        body_rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.7, 0.3)), 0.35)
        self.body_model = newton.solvers.AffineBodyModel(
            rest_vertices=rest_vertices,
            tetrahedron_indices=tetrahedron_indices,
            surface_triangle_indices=surface_triangle_indices,
            density=6.0,
            rigidity=100.0,
            initial_transform=wp.transform(wp.vec3(0.0, 0.0, 2.4), body_rotation),
            device=wp.get_device(),
        )

        initial_state = self.body_model.q.numpy()
        rotation = initial_state[0, 3:].reshape(3, 3)
        shear = np.asarray(
            [
                [1.0, 0.60, 0.12],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        initial_state[0, 3:] = (rotation @ shear).reshape(-1)
        self.body_model.q.assign(initial_state)

        self.solver = newton.solvers.SolverLIMXAffine(
            self.body_model,
            nonlinear_iterations=4,
            linear_iterations=16,
            velocity_damping=0.995,
        )

        initial_surface_positions = wp.empty(
            self.body_model.surface_vertex_count,
            dtype=wp.vec3,
            device=self.body_model.device,
        )
        self.solver.update_surface_positions(initial_surface_positions)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_particles(
            pos=[wp.vec3(*position) for position in initial_surface_positions.numpy()],
            vel=[wp.vec3(0.0)] * self.body_model.surface_vertex_count,
            mass=[1.0] * self.body_model.surface_vertex_count,
            radius=[0.035] * self.body_model.surface_vertex_count,
        )
        triangles = self.body_model.surface_triangle_indices.numpy()
        builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
        self.model = builder.finalize(device=self.body_model.device)
        self.state_0 = self.model.state()

        initial_matrix = self.solver.q.numpy()[0, 3:].reshape(3, 3)
        self.initial_center_height = float(initial_surface_positions.numpy()[:, 2].mean())
        self.initial_singular_value_error = float(np.linalg.norm(np.linalg.svd(initial_matrix, compute_uv=False) - 1.0))
        self.minimum_determinant = float(np.linalg.det(initial_matrix))
        self.center_heights: list[float] = []
        self.singular_value_errors: list[float] = []

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.4, -3.5, 3.0), -8.0, 124.0)

    def step(self):
        """Advance one collision-free affine-body step and reconstruct its surface."""
        self.solver.step(self.frame_dt)
        self.solver.update_surface_positions(self.state_0.particle_q)
        self.sim_time += self.frame_dt

    def render(self):
        """Render the reconstructed affine surface through a particle state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Keep the reconstructed surface finite and the affine map orientation preserving."""
        positions = self.state_0.particle_q.numpy()
        state = self.solver.q.numpy()[0]
        velocity = self.solver.qd.numpy()[0]
        if not np.isfinite(positions).all() or not np.isfinite(state).all() or not np.isfinite(velocity).all():
            raise AssertionError("Affine LIMX state and reconstructed surface must remain finite")

        matrix = state[3:].reshape(3, 3)
        determinant = float(np.linalg.det(matrix))
        if determinant <= 0.0:
            raise AssertionError("Affine LIMX body must remain orientation preserving")

        self.minimum_determinant = min(self.minimum_determinant, determinant)
        self.center_heights.append(float(positions[:, 2].mean()))
        self.singular_value_errors.append(float(np.linalg.norm(np.linalg.svd(matrix, compute_uv=False) - 1.0)))

    def test_final(self):
        """Fall downward and reduce the initial affine singular-value error."""
        self.test_post_step()
        if self.center_heights[-1] >= self.initial_center_height:
            raise AssertionError("Affine LIMX body must fall under gravity")
        if self.singular_value_errors[-1] >= self.initial_singular_value_error:
            raise AssertionError("Affine LIMX body must relax its initial shear")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
