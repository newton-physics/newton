# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Twist a LIMX cloth strip through matrix-free self-contact."""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt
        self.sim_time = 0.0
        self.drive_duration = 4.0
        self.target_angle = 4.0 * math.pi

        length_cells = 32
        width_cells = 12
        length_side = length_cells + 1
        width_side = width_cells + 1
        particle_count = length_side * width_side
        strip_length = 1.6
        strip_width = 0.6
        self.strip_center_z = 1.2

        positions = [
            wp.vec3(
                -0.5 * strip_length + strip_length * x / length_cells,
                -0.5 * strip_width + strip_width * y / width_cells,
                self.strip_center_z,
            )
            for y in range(width_side)
            for x in range(length_side)
        ]
        triangles = []
        for y in range(width_cells):
            for x in range(length_cells):
                lower_left = y * length_side + x
                lower_right = lower_left + 1
                upper_left = lower_left + length_side
                upper_right = upper_left + 1
                if (x + y) % 2 == 0:
                    triangles.extend([(lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)])
                else:
                    triangles.extend([(lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left)])

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_particles(
            pos=positions,
            vel=[wp.vec3(0.0)] * particle_count,
            mass=[0.3 / particle_count] * particle_count,
            radius=[0.006] * particle_count,
        )
        triangle_indices = np.asarray(triangles, dtype=np.int32)
        builder.add_triangles(triangle_indices[:, 0], triangle_indices[:, 1], triangle_indices[:, 2])
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        positions_np = np.asarray(positions, dtype=np.float32)
        self.triangle_indices = triangle_indices
        self.inverse_rest_matrices = self.model.tri_poses.numpy()
        self.left_boundary_indices = [y * length_side for y in range(width_side)]
        self.right_boundary_indices = [y * length_side + length_cells for y in range(width_side)]
        self.anchor_indices = self.left_boundary_indices + self.right_boundary_indices
        self.boundary_particle_count = width_side
        self.anchor_rest_targets = positions_np[self.anchor_indices].copy()

        edge_rows = newton.utils.MeshAdjacency(triangle_indices).edge_indices
        interior_edge_rows = edge_rows[edge_rows[:, 1] >= 0]
        self.dihedral_indices = interior_edge_rows[:, [2, 3, 0, 1]]

        self.anchor_constraint = newton.solvers.ConstraintAnchor(
            self.anchor_indices,
            [positions[index] for index in self.anchor_indices],
            [1.0e7] * len(self.anchor_indices),
            particle_count,
            self.model.device,
        )
        constraints = [
            self.anchor_constraint,
            newton.solvers.ConstraintTriangleElastic(
                triangles,
                self.inverse_rest_matrices,
                self.model.tri_areas.numpy(),
                [wp.vec3(1.0e4, 1.0e4, 1.0e3)] * len(triangles),
                particle_count,
                self.model.device,
            ),
            newton.solvers.ConstraintDihedralBending(
                self.dihedral_indices,
                positions_np,
                0.01,
                particle_count,
                self.model.device,
            ),
        ]
        self.self_collision = newton.solvers.ConstraintSelfCollision(
            self.model,
            thickness=0.012,
            stiffness=1.0e4,
            untangle_stiffness=3.0e4,
            max_contacts=32768,
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
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.1, -2.2, 1.65), 8.0, 132.0)
        self.capture()

    def _compute_anchor_targets(self, angle: float) -> np.ndarray:
        targets = self.anchor_rest_targets.copy()
        cosine = math.cos(angle)
        sine = math.sin(angle)
        targets[:, 1] = cosine * self.anchor_rest_targets[:, 1]
        boundary_count = self.boundary_particle_count
        targets[:boundary_count, 2] = self.strip_center_z + sine * self.anchor_rest_targets[:boundary_count, 1]
        targets[boundary_count:, 2] = self.strip_center_z - sine * self.anchor_rest_targets[boundary_count:, 1]
        return targets

    def _drive_angle(self) -> float:
        phase = min(self.sim_time / self.drive_duration, 1.0)
        smooth_phase = phase * phase * (3.0 - 2.0 * phase)
        return self.target_angle * smooth_phase

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.state_0.assign(self.state_1)

    def step(self):
        self.anchor_constraint.targets.assign(self._compute_anchor_targets(self._drive_angle()))
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("LIMX twist cloth state contains non-finite values")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
