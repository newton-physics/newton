# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody LIMX ARAP Bunny Table Drop
#
# A volumetric bunny falls onto a static table using tetrahedral ARAP
# elasticity and penalty contact. Every 0.01 s frame uses one full Newton
# increment without line search, substeps, or velocity/contact damping.
#
# Command: uv run -m newton.examples softbody_limx_arap_bunny_table
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
    """Drop a volumetric bunny onto a table with LIMX ARAP elasticity."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 0.01
        self.sim_time = 0.0
        self.table_top = 0.0
        self.minimum_height = np.inf
        self.maximum_speeds: list[float] = []
        self.center_heights: list[float] = []

        mesh_path = Path(__file__).resolve().parents[1] / "assets" / "bunny_tet.npz"
        mesh = newton.TetMesh.create_from_file(str(mesh_path))

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.25),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi),
            scale=0.15,
            vel=wp.vec3(0.0),
            mesh=mesh,
            density=1000.0,
            k_mu=0.0,
            k_lambda=0.0,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.03), wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=0.03,
            color=wp.vec3(0.42, 0.46, 0.52),
        )
        self.model = builder.finalize()

        self.rest_positions = self.model.particle_q.numpy()
        self.tetrahedra = self.model.tet_indices.numpy()
        self.particle_masses = self.model.particle_mass.numpy()
        self.initial_center_height = float(np.average(self.rest_positions[:, 2], weights=self.particle_masses))
        self.initial_minimum_height = float(self.rest_positions[:, 2].min())

        inverse_rest_matrices = self.model.tet_poses.numpy()
        self.arap_constraint = newton.solvers.ConstraintTetrahedronARAP(
            self.tetrahedra.tolist(),
            [wp.mat33(*matrix.reshape(-1)) for matrix in inverse_rest_matrices],
            [1.0e5] * self.model.tet_count,
            self.model.particle_count,
            self.model.device,
        )
        self.table_contact = newton.solvers.ConstraintStaticPlaneContact(
            normal=(0.0, 0.0, 1.0),
            offset=self.table_top,
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.0,
            friction=0.05,
            friction_epsilon=1.0e-4,
            particle_count=self.model.particle_count,
            device=self.model.device,
        )
        self.solver = newton.solvers.SolverLIMX(
            self.model,
            [self.arap_constraint],
            nonlinear_iterations=1,
            linear_iterations=128,
            velocity_damping=1.0,
            dynamic_operator=self.table_contact,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.62, -0.82, 0.48), -8.0, 143.0)

    def step(self):
        """Advance one undamped 0.01-second Newton step."""
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        """Render the deforming bunny and visualization table."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Keep the falling bunny finite, positive-volume, and near the scene."""
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("LIMX ARAP bunny state must remain finite")

        edges = np.stack(
            (
                positions[self.tetrahedra[:, 1]] - positions[self.tetrahedra[:, 0]],
                positions[self.tetrahedra[:, 2]] - positions[self.tetrahedra[:, 0]],
                positions[self.tetrahedra[:, 3]] - positions[self.tetrahedra[:, 0]],
            ),
            axis=2,
        )
        if float(np.linalg.det(edges).min()) <= 0.0:
            raise AssertionError("LIMX ARAP bunny tetrahedra must remain positive-volume")

        minimum_height = float(positions[:, 2].min())
        if minimum_height < -0.015:
            raise AssertionError("LIMX ARAP bunny penetrated more than 15 mm below the table")

        center_height = float(np.average(positions[:, 2], weights=self.particle_masses))
        if not -0.05 <= center_height <= 1.0:
            raise AssertionError("LIMX ARAP bunny center left the bounded scene")

        self.minimum_height = min(self.minimum_height, minimum_height)
        self.maximum_speeds.append(float(np.linalg.norm(velocities, axis=1).max()))
        self.center_heights.append(center_height)

    def test_final(self):
        """Reach the table without inversion or increasing-amplitude motion."""
        self.test_post_step()
        if self.sim_time >= 0.25:
            if self.minimum_height > 0.03:
                raise AssertionError("LIMX ARAP bunny did not reach the table")
            if self.center_heights[-1] >= self.initial_center_height - 0.05:
                raise AssertionError("LIMX ARAP bunny did not fall toward the table")

        if len(self.maximum_speeds) >= 100:
            previous_peak = max(self.maximum_speeds[-100:-50])
            late_peak = max(self.maximum_speeds[-50:])
            if late_peak > max(1.5 * previous_peak, 5.0):
                raise AssertionError("LIMX ARAP bunny motion grows after impact")

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
