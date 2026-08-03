# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mass-spring cloth solved by the LIMX projected-Newton pipeline."""

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        grid_cells = 20
        grid_side = grid_cells + 1
        particle_count = grid_side * grid_side
        positions = [
            wp.vec3(-0.5 + x / grid_cells, -0.5 + y / grid_cells, 2.0)
            for y in range(grid_side)
            for x in range(grid_side)
        ]

        triangles = []
        for y in range(grid_cells):
            for x in range(grid_cells):
                lower_left = y * grid_side + x
                lower_right = lower_left + 1
                upper_left = lower_left + grid_side
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
            radius=[0.005] * particle_count,
        )
        triangle_array = np.asarray(triangles, dtype=np.int32)
        builder.add_triangles(triangle_array[:, 0], triangle_array[:, 1], triangle_array[:, 2])
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.edges = sorted(
            {
                tuple(sorted(edge))
                for triangle in triangles
                for edge in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))
            }
        )
        positions_np = np.asarray(positions, dtype=np.float32)
        self.rest_lengths = np.asarray(
            [np.linalg.norm(positions_np[j] - positions_np[i]) for i, j in self.edges], dtype=np.float32
        )
        self.anchor_indices = [0, grid_cells]
        self.anchor_targets = positions_np[self.anchor_indices].copy()
        self.anchor_y = float(self.anchor_targets[0, 1])
        self.center_index = particle_count // 2
        self.initial_center_height = float(positions_np[self.center_index, 2])

        constraints = [
            newton.solvers.ConstraintAnchor(
                self.anchor_indices,
                [positions[index] for index in self.anchor_indices],
                [1.0e7] * len(self.anchor_indices),
                particle_count,
                self.model.device,
            ),
            newton.solvers.ConstraintDistance(
                self.edges,
                self.rest_lengths,
                [1.0e4] * len(self.edges),
                particle_count,
                self.model.device,
            ),
        ]
        self.solver = newton.solvers.SolverLIMX(
            self.model,
            constraints,
            nonlinear_iterations=1,
            linear_iterations=50,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.35, -1.75, 1.25), 10.0, 128.0)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            if self.sim_substeps % 2 == 1 and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        current_edge_lengths = np.asarray([np.linalg.norm(positions[j] - positions[i]) for i, j in self.edges])

        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("LIMX cloth state contains non-finite values")
        np.testing.assert_allclose(positions[self.anchor_indices], self.anchor_targets, atol=1.0e-3)
        if positions[self.center_index, 2] >= self.initial_center_height - 5.0e-2:
            raise AssertionError("LIMX cloth center did not sag under gravity")
        if positions[self.center_index, 1] >= self.anchor_y:
            raise AssertionError("LIMX cloth center did not swing past the anchor line")
        if float(np.max(current_edge_lengths)) >= 2.0 * float(np.max(self.rest_lengths)):
            raise AssertionError("LIMX cloth springs stretched beyond the expected bound")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
