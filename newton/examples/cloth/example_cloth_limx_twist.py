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
        self.rot_angular_velocity = 1.0
        self.rot_end_time = 25.0

        cloth_size_z = 200
        cloth_size_y = 100
        particle_count = cloth_size_z * cloth_size_y
        edge_length = 0.5 / (cloth_size_y - 1)
        positions_np = np.asarray(
            [
                (0.0, -0.25 + edge_length * y, -0.5 + z / (cloth_size_z - 1))
                for z in range(cloth_size_z)
                for y in range(cloth_size_y)
            ],
            dtype=np.float32,
        )
        triangles = []
        for z in range(cloth_size_z - 1):
            for y in range(cloth_size_y - 1):
                lower_left = z * cloth_size_y + y
                upper_left = lower_left + 1
                lower_right = lower_left + cloth_size_y
                upper_right = lower_right + 1
                if (z + y) % 2 == 0:
                    triangles.extend([(lower_left, upper_right, upper_left), (lower_left, lower_right, upper_right)])
                else:
                    triangles.extend([(lower_left, lower_right, upper_left), (upper_left, lower_right, upper_right)])

        triangle_indices = np.asarray(triangles, dtype=np.int32)
        triangle_positions = positions_np[triangle_indices]
        triangle_areas = 0.5 * np.linalg.norm(
            np.cross(
                triangle_positions[:, 1] - triangle_positions[:, 0],
                triangle_positions[:, 2] - triangle_positions[:, 0],
            ),
            axis=1,
        )
        masses = np.zeros(particle_count, dtype=np.float32)
        np.add.at(
            masses,
            triangle_indices.reshape(-1),
            np.repeat(0.1 * triangle_areas / 3.0, 3),
        )
        positions = [wp.vec3(float(x), float(y), float(z)) for x, y, z in positions_np]

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_particles(
            pos=positions,
            vel=[wp.vec3(0.0)] * particle_count,
            mass=masses.tolist(),
            radius=[0.2 * edge_length] * particle_count,
        )
        builder.add_triangles(triangle_indices[:, 0], triangle_indices[:, 1], triangle_indices[:, 2])
        self.model = builder.finalize()
        self.model.set_gravity((0.0, -9.8, 0.0))

        self.triangle_indices = triangle_indices
        self.inverse_rest_matrices = self.model.tri_poses.numpy()
        self.negative_z_boundary_indices = list(range(cloth_size_y))
        positive_z_start = (cloth_size_z - 1) * cloth_size_y
        self.positive_z_boundary_indices = list(range(positive_z_start, positive_z_start + cloth_size_y))
        self.anchor_indices = self.negative_z_boundary_indices + self.positive_z_boundary_indices
        self.boundary_particle_count = cloth_size_y
        self.anchor_rest_targets = positions_np[self.anchor_indices].copy()

        edge_rows = newton.utils.MeshAdjacency(triangle_indices).edge_indices
        interior_edge_rows = edge_rows[edge_rows[:, 1] >= 0]
        self.dihedral_indices = interior_edge_rows[:, [2, 3, 0, 1]]

        self.anchor_constraint = newton.solvers.ConstraintAnchor(
            self.anchor_indices,
            [positions[index] for index in self.anchor_indices],
            [1.0e9] * len(self.anchor_indices),
            particle_count,
            self.model.device,
        )
        constraints = [
            self.anchor_constraint,
            newton.solvers.ConstraintTriangleElastic(
                triangles,
                self.inverse_rest_matrices,
                self.model.tri_areas.numpy(),
                [wp.vec3(500.0, 500.0, 500.0)] * len(triangles),
                particle_count,
                self.model.device,
            ),
            newton.solvers.ConstraintDihedralBending(
                self.dihedral_indices,
                positions_np,
                5.0e-5,
                particle_count,
                self.model.device,
            ),
        ]
        self.self_collision = newton.solvers.ConstraintSelfCollision(
            self.model,
            thickness=None,
            stiffness=1.0e3,
            untangle_stiffness=2.0e3,
            max_contacts=131072,
            geometry_radius_scale=0.25,
            geometry_radius_topology_local_only=True,
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
        self.viewer.set_camera(wp.vec3(2.25, 0.0, 0.0), 0.0, -180.0)
        self.capture()

    def _compute_anchor_targets(self, angle: float) -> np.ndarray:
        targets = self.anchor_rest_targets.copy()
        cosine = math.cos(angle)
        sine = math.sin(angle)
        boundary_count = self.boundary_particle_count
        targets[:boundary_count, 0] = sine * self.anchor_rest_targets[:boundary_count, 1]
        targets[boundary_count:, 0] = -sine * self.anchor_rest_targets[boundary_count:, 1]
        targets[:, 1] = cosine * self.anchor_rest_targets[:, 1]
        return targets

    def _drive_angle(self) -> float:
        return self.rot_angular_velocity * min(self.sim_time, self.rot_end_time)

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
    parser.set_defaults(num_frames=1000)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
