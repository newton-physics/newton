# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic LIMX Affine Bunnies Ground
#
# Drops eight high-rigidity affine bunnies onto frictional ground with
# mutually coupled VF and strict-interior EE penalty contact.
#
# Command: uv run -m newton.examples basic_limx_affine_bunnies_ground
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
    """Drop a staggered pile of frictional affine bunnies."""

    _COLORS = (
        (0.82, 0.36, 0.30),
        (0.28, 0.56, 0.86),
        (0.36, 0.72, 0.48),
        (0.88, 0.67, 0.25),
        (0.66, 0.42, 0.82),
        (0.26, 0.72, 0.76),
        (0.90, 0.48, 0.68),
        (0.55, 0.68, 0.28),
    )

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 0.01
        self.sim_time = 0.0
        self.ground_top = 0.0

        mesh_path = Path(__file__).resolve().parents[1] / "assets" / "bunny_tet.npz"
        mesh = newton.TetMesh.create_from_file(str(mesh_path))
        rest_vertices = 0.15 * np.asarray(mesh.vertices)
        tetrahedra = np.asarray(mesh.tet_indices).reshape(-1, 4)
        surface_triangles = np.asarray(mesh.surface_tri_indices).reshape(-1, 3)
        upright = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
        tilt_axes_angles = (
            ((0.0, 0.0, 1.0), -4.0),
            ((0.0, 1.0, 0.0), 3.0),
            ((1.0, 0.0, 0.0), -3.0),
            ((0.0, 0.0, 1.0), 4.0),
            ((0.0, 1.0, 0.0), -5.0),
            ((1.0, 0.0, 0.0), 4.0),
            ((0.0, 0.0, 1.0), 5.0),
            ((0.0, 1.0, 0.0), -3.0),
        )
        transforms = []
        for axis, angle_degrees in tilt_axes_angles:
            tilt = wp.quat_from_axis_angle(wp.vec3(*axis), math.radians(angle_degrees))
            transforms.append(wp.transform(wp.vec3(0.0), wp.mul(tilt, upright)))

        self.body_model = newton.solvers.AffineBodyModel.from_instances(
            rest_vertices=rest_vertices,
            tetrahedron_indices=tetrahedra,
            surface_triangle_indices=surface_triangles,
            density=1000.0,
            rigidity=1.0e8,
            initial_transforms=transforms,
            device=wp.get_device(),
        )
        centers = np.asarray(
            [
                [-0.18, -0.22, 0.26],
                [0.18, -0.22, 0.26],
                [-0.18, 0.22, 0.26],
                [0.18, 0.22, 0.26],
                [-0.14, -0.18, 0.56],
                [0.22, -0.18, 0.56],
                [-0.14, 0.26, 0.56],
                [0.22, 0.26, 0.56],
            ],
            dtype=np.float32,
        )
        translation_velocities = np.asarray(
            [
                [0.0015, 0.0010, 0.0],
                [-0.0015, 0.0010, 0.0],
                [0.0015, -0.0010, 0.0],
                [-0.0015, -0.0010, 0.0],
                [-0.0012, -0.0008, 0.0],
                [0.0012, -0.0008, 0.0],
                [-0.0012, 0.0008, 0.0],
                [0.0012, 0.0008, 0.0],
            ],
            dtype=np.float32,
        )
        initial_states = self.body_model.q.numpy()
        initial_states[:, :3] = centers
        self.body_model.q.assign(initial_states)
        initial_velocities = self.body_model.qd.numpy()
        initial_velocities[:, :3] = translation_velocities
        self.body_model.qd.assign(initial_velocities)

        self.body_contact = newton.solvers.ConstraintAffineBodyContact(
            self.body_model,
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.5,
            friction=0.5,
            friction_epsilon=1.0e-4,
        )
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
        self.dynamic_constraints = newton.solvers.ConstraintGroupAffine([self.body_contact, self.ground_contact])
        self.solver = newton.solvers.SolverLIMXAffine(
            self.body_model,
            nonlinear_iterations=1,
            linear_iterations=50,
            velocity_damping=1.0,
            dynamic_operator=self.dynamic_constraints,
        )

        self.surface_positions = wp.empty(
            self.body_model.surface_vertex_count,
            dtype=wp.vec3,
            device=self.body_model.device,
        )
        self.solver.update_surface_positions(self.surface_positions)
        self._validate_initial_aabbs()

        triangles = self.body_model.surface_triangle_indices.numpy()
        triangles_per_body = self.body_model.surface_triangle_count // self.body_model.body_count
        self.render_indices = [
            wp.array(
                triangles[body * triangles_per_body : (body + 1) * triangles_per_body].reshape(-1),
                dtype=wp.int32,
                device=self.body_model.device,
            )
            for body in range(self.body_model.body_count)
        ]
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.03, 0.03, -0.03), wp.quat_identity()),
            hx=0.55,
            hy=0.62,
            hz=0.03,
            color=wp.vec3(0.42, 0.46, 0.52),
        )
        self.model = builder.finalize(device=self.body_model.device)
        self.render_state = self.model.state()

        initial_matrices = initial_states[:, 3:].reshape(-1, 3, 3)
        self.initial_center_heights = initial_states[:, 2].copy()
        self.minimum_height = float(self.surface_positions.numpy()[:, 2].min())
        self.minimum_determinant = float(np.linalg.det(initial_matrices).min())
        self.maximum_singular_value_error = float(
            np.max(np.abs(np.linalg.svd(initial_matrices, compute_uv=False) - 1.0))
        )
        self.maximum_vf_overflow = 0
        self.maximum_ee_overflow = 0
        self.maximum_contact_depth = 0.0
        self.cross_body_contact_observed = False
        self.center_heights: list[np.ndarray] = []
        self.support_margins: list[float] = []

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.25, -1.55, 1.05), -10.0, 140.0)
        self.capture()

    def _validate_initial_aabbs(self) -> None:
        """Reject an initial layout with overlapping body AABBs."""
        positions = self.surface_positions.numpy().reshape(self.body_model.body_count, -1, 3)
        lower = positions.min(axis=1)
        upper = positions.max(axis=1)
        for body_0 in range(self.body_model.body_count):
            for body_1 in range(body_0 + 1, self.body_model.body_count):
                overlap = np.minimum(upper[body_0], upper[body_1]) - np.maximum(lower[body_0], lower[body_1])
                if np.all(overlap > 0.0):
                    raise ValueError(f"Initial affine body AABBs {body_0} and {body_1} overlap")

    def capture(self) -> None:
        """Capture one coupled affine solve and reconstruction on CUDA."""
        if self.body_model.device.is_cuda:
            with wp.ScopedCapture(device=self.body_model.device) as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        """Advance every affine body and reconstruct the shared surface."""
        self.solver.step(self.frame_dt)
        self.solver.update_surface_positions(self.surface_positions)

    def step(self) -> None:
        """Advance one 0.01-second coupled affine-body frame."""
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        """Render every bunny with a distinct color and the static ground."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.render_state)
        for body, (indices, color) in enumerate(zip(self.render_indices, self._COLORS, strict=True)):
            self.viewer.log_mesh(
                f"/affine_bunny_{body}",
                self.surface_positions,
                indices,
                color=color,
                backface_culling=False,
            )
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        """Keep the affine pile finite, rigid, shallow, and within capacity."""
        positions = self.surface_positions.numpy()
        states = self.solver.q.numpy()
        velocities = self.solver.qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(states).all() or not np.isfinite(velocities).all():
            raise AssertionError("Affine bunny states and reconstructed surfaces must remain finite")

        matrices = states[:, 3:].reshape(-1, 3, 3)
        determinant = float(np.linalg.det(matrices).min())
        if determinant <= 0.0:
            raise AssertionError("Every affine bunny must remain orientation preserving")
        singular_value_error = float(np.max(np.abs(np.linalg.svd(matrices, compute_uv=False) - 1.0)))
        if singular_value_error >= 0.02:
            raise AssertionError("Affine bunny singular values deviated by at least two percent")

        minimum_height = float(positions[:, 2].min())
        if minimum_height < -0.006:
            raise AssertionError("An affine bunny penetrated more than 6 mm below the ground")

        vf_count = min(
            int(self.body_contact.vertex_face_contacts.count.numpy()[0]),
            self.body_contact.vertex_face_contacts.capacity,
        )
        ee_count = min(
            int(self.body_contact.edge_edge_contacts.count.numpy()[0]),
            self.body_contact.edge_edge_contacts.capacity,
        )
        vf_overflow = int(self.body_contact.vertex_face_contacts.overflow_count.numpy()[0])
        ee_overflow = int(self.body_contact.edge_edge_contacts.overflow_count.numpy()[0])
        if vf_overflow != 0 or ee_overflow != 0:
            raise AssertionError("Affine body contact buffers overflowed")
        maximum_contact_depth = 0.0
        if vf_count:
            maximum_contact_depth = max(
                maximum_contact_depth,
                float(self.body_contact.vertex_face_contacts.depths.numpy()[:vf_count].max()),
            )
        if ee_count:
            maximum_contact_depth = max(
                maximum_contact_depth,
                float(self.body_contact.edge_edge_contacts.depths.numpy()[:ee_count].max()),
            )
        if maximum_contact_depth >= 0.012:
            raise AssertionError("Affine body contact depth reached at least 12 mm")

        center_heights = states[:, 2].copy()
        support_margin = float(np.mean(center_heights[4:]) - np.mean(center_heights[:4]))
        self.minimum_height = min(self.minimum_height, minimum_height)
        self.minimum_determinant = min(self.minimum_determinant, determinant)
        self.maximum_singular_value_error = max(self.maximum_singular_value_error, singular_value_error)
        self.maximum_vf_overflow = max(self.maximum_vf_overflow, vf_overflow)
        self.maximum_ee_overflow = max(self.maximum_ee_overflow, ee_overflow)
        self.maximum_contact_depth = max(self.maximum_contact_depth, maximum_contact_depth)
        self.cross_body_contact_observed = self.cross_body_contact_observed or vf_count > 0 or ee_count > 0
        self.center_heights.append(center_heights)
        self.support_margins.append(support_margin)

    def test_final(self) -> None:
        """Fall, contact another body, and retain an upper supported layer."""
        self.test_post_step()
        if not np.all(self.initial_center_heights - self.center_heights[-1] > 0.03):
            raise AssertionError("Every affine bunny center must fall by more than 3 cm")
        if not self.cross_body_contact_observed:
            raise AssertionError("The affine bunny pile never generated cross-body contact")
        if len(self.support_margins) >= 30 and float(np.mean(self.support_margins[-30:])) <= 0.10:
            raise AssertionError("The upper affine bunny layer lost its support margin")

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
