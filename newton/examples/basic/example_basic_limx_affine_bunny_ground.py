# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic LIMX Affine Bunny Ground
#
# Drops one high-rigidity affine bunny onto a static plane using penalty
# normal contact and regularized Coulomb friction.
#
# Command: uv run -m newton.examples basic_limx_affine_bunny_ground
#
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    """Drop a high-rigidity affine bunny onto frictional ground."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 0.01
        self.sim_time = 0.0
        self.ground_top = 0.0

        mesh_path = Path(__file__).resolve().parents[1] / "assets" / "bunny_tet.npz"
        mesh = newton.TetMesh.create_from_file(str(mesh_path))
        upright = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
        tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(15.0))
        rotation = wp.mul(tilt, upright)
        self.body_model = newton.solvers.AffineBodyModel(
            rest_vertices=0.15 * np.asarray(mesh.vertices),
            tetrahedron_indices=np.asarray(mesh.tet_indices).reshape(-1, 4),
            surface_triangle_indices=np.asarray(mesh.surface_tri_indices).reshape(-1, 3),
            density=1000.0,
            rigidity=1.0e8,
            initial_transform=wp.transform(wp.vec3(0.0), rotation),
            device=wp.get_device(),
        )
        initial_state = self.body_model.q.numpy()
        initial_state[0, :3] = [0.0, 0.0, 0.65]
        self.body_model.q.assign(initial_state)
        self.body_mass = float(self.body_model.mass_matrices.numpy()[0, 0, 0])

        self.ground_contact = newton.solvers.ConstraintAffineStaticPlaneContact(
            self.body_model,
            normal=(0.0, 0.0, 1.0),
            offset=self.ground_top,
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.5,
            friction_epsilon=1.0e-4,
        )
        self.solver = newton.solvers.SolverLIMXAffine(
            self.body_model,
            nonlinear_iterations=1,
            linear_iterations=50,
            velocity_damping=1.0,
            dynamic_operator=self.ground_contact,
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
            radius=[0.002] * self.body_model.surface_vertex_count,
        )
        triangles = self.body_model.surface_triangle_indices.numpy()
        builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.03), wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=0.03,
            color=wp.vec3(0.42, 0.46, 0.52),
        )
        self.model = builder.finalize(device=self.body_model.device)
        self.state_0 = self.model.state()
        self.solver.update_surface_positions(self.state_0.particle_q)

        self.initial_center_height = float(self.solver.q.numpy()[0, 2])
        self.minimum_height = float(initial_surface_positions.numpy()[:, 2].min())
        self.minimum_determinant = float(np.linalg.det(initial_state[0, 3:].reshape(3, 3)))
        self.maximum_singular_value_error = float(
            np.max(np.abs(np.linalg.svd(initial_state[0, 3:].reshape(3, 3), compute_uv=False) - 1.0))
        )
        self.contact_activated = False
        self.center_heights: list[float] = []
        self.tangential_speeds: list[float] = []

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.62, -0.82, 0.48), -8.0, 143.0)
        self.capture()

    def capture(self) -> None:
        """Capture one affine solve and surface reconstruction on CUDA."""
        if self.body_model.device.is_cuda:
            with wp.ScopedCapture(device=self.body_model.device) as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        """Advance the affine state and reconstruct its render surface."""
        self.solver.step(self.frame_dt)
        self.solver.update_surface_positions(self.state_0.particle_q)

    def step(self) -> None:
        """Advance one 0.01-second affine-body frame."""
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        """Render the reconstructed bunny and static ground box."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        """Keep the affine bunny finite, rigid, oriented, and near the ground."""
        positions = self.state_0.particle_q.numpy()
        state = self.solver.q.numpy()[0]
        velocity = self.solver.qd.numpy()[0]
        if not np.isfinite(positions).all() or not np.isfinite(state).all() or not np.isfinite(velocity).all():
            raise AssertionError("Affine bunny state and reconstructed surface must remain finite")

        matrix = state[3:].reshape(3, 3)
        determinant = float(np.linalg.det(matrix))
        if determinant <= 0.0:
            raise AssertionError("Affine bunny must remain orientation preserving")
        singular_value_error = float(np.max(np.abs(np.linalg.svd(matrix, compute_uv=False) - 1.0)))
        if singular_value_error >= 0.02:
            raise AssertionError("Affine bunny singular values deviated by at least two percent")

        minimum_height = float(positions[:, 2].min())
        if minimum_height < -0.006:
            raise AssertionError("Affine bunny penetrated more than 6 mm below the ground")

        self.minimum_height = min(self.minimum_height, minimum_height)
        self.minimum_determinant = min(self.minimum_determinant, determinant)
        self.maximum_singular_value_error = max(self.maximum_singular_value_error, singular_value_error)
        self.contact_activated = self.contact_activated or minimum_height <= self.ground_contact.thickness
        self.center_heights.append(float(state[2]))
        self.tangential_speeds.append(float(np.linalg.norm(velocity[:2])))

    def test_final(self) -> None:
        """Reach the ground and settle without continued tangential sliding."""
        self.test_post_step()
        if not self.contact_activated:
            raise AssertionError("Affine bunny did not enter the ground contact band")
        if self.initial_center_height - self.center_heights[-1] <= 0.20:
            raise AssertionError("Affine bunny center did not fall by at least 0.20 m")
        if len(self.tangential_speeds) >= 30 and float(np.mean(self.tangential_speeds[-30:])) >= 0.05:
            raise AssertionError("Affine bunny retained excessive tangential sliding")

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
