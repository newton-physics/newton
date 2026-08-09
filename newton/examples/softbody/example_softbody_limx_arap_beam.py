# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody LIMX ARAP Beam
#
# A tetrahedral cantilever uses the analytical ARAP energy and Hessian in
# SolverLIMX. The left particle layer is anchored while gravity bends the
# free end. Every 0.01 s frame uses one full Newton increment and no damping.
#
# Command: uv run -m newton.examples softbody_limx_arap_beam
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    """Simulate a fixed tetrahedral beam with LIMX ARAP elasticity."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 0.01
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_soft_grid(
            pos=wp.vec3(0.0, -0.05, 0.75),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=12,
            dim_y=2,
            dim_z=2,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=1000.0,
            k_mu=0.0,
            k_lambda=0.0,
            k_damp=0.0,
            fix_left=False,
        )
        self.model = builder.finalize()
        self.rest_positions = self.model.particle_q.numpy()
        self.tetrahedra = self.model.tet_indices.numpy()
        inverse_rest_matrices = self.model.tet_poses.numpy()

        minimum_x = float(np.min(self.rest_positions[:, 0]))
        maximum_x = float(np.max(self.rest_positions[:, 0]))
        self.anchor_indices = np.flatnonzero(np.isclose(self.rest_positions[:, 0], minimum_x))
        self.free_end_indices = np.flatnonzero(np.isclose(self.rest_positions[:, 0], maximum_x))
        self.initial_free_end_z = float(np.mean(self.rest_positions[self.free_end_indices, 2]))
        self.minimum_free_end_z = self.initial_free_end_z

        self.anchor_constraint = newton.solvers.ConstraintAnchor(
            self.anchor_indices.tolist(),
            [wp.vec3(*position) for position in self.rest_positions[self.anchor_indices]],
            [1.0e8] * len(self.anchor_indices),
            self.model.particle_count,
            self.model.device,
        )
        self.arap_constraint = newton.solvers.ConstraintTetrahedronARAP(
            self.tetrahedra.tolist(),
            [wp.mat33(*matrix.reshape(-1)) for matrix in inverse_rest_matrices],
            [1.0e6] * self.model.tet_count,
            self.model.particle_count,
            self.model.device,
        )
        self.solver = newton.solvers.SolverLIMX(
            self.model,
            [self.anchor_constraint, self.arap_constraint],
            nonlinear_iterations=1,
            linear_iterations=128,
            velocity_damping=1.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.3, -0.9, 0.95), -10.0, 90.0)

    def step(self):
        """Advance one undamped 0.01-second Newton step."""
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current tetrahedral-beam surface."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Keep every simulated tetrahedron finite and positively oriented."""
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("ARAP beam state must remain finite")

        self.minimum_free_end_z = min(
            self.minimum_free_end_z,
            float(np.mean(positions[self.free_end_indices, 2])),
        )
        for tetrahedron in self.tetrahedra:
            deformation_edges = np.column_stack(
                (
                    positions[tetrahedron[1]] - positions[tetrahedron[0]],
                    positions[tetrahedron[2]] - positions[tetrahedron[0]],
                    positions[tetrahedron[3]] - positions[tetrahedron[0]],
                )
            )
            if float(np.linalg.det(deformation_edges)) <= 0.0:
                raise AssertionError("ARAP beam tetrahedra must remain positive-volume")

    def test_final(self):
        """Keep the fixed end anchored while the free end visibly sags."""
        positions = self.state_0.particle_q.numpy()
        np.testing.assert_allclose(
            positions[self.anchor_indices],
            self.rest_positions[self.anchor_indices],
            atol=2.0e-3,
        )
        if self.minimum_free_end_z >= self.initial_free_end_z - 2.0e-3:
            raise AssertionError("ARAP beam free end must sag under gravity")

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
