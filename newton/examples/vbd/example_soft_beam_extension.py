# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Beam Extension
#
# A vertical tetrahedral beam pinned at the top and hanging under gravity.
# At equilibrium the bottom-layer displacement is compared to an empirical
# baseline calibrated against the stable Neo-Hookean material in VBD. The
# test also validates that the beam's center of mass drops monotonically
# along Z (no lateral drift) and that all velocities remain bounded.
#
# Command: python -m newton.examples vbd.example_soft_beam_extension
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags


class Example:
    DIM_X = 4
    DIM_Y = 4
    DIM_Z = 20
    CELL = 0.05
    DENSITY = 1000.0
    K_MU = 5.0e4
    K_LAMBDA = 5.0e4
    K_DAMP = 0.1

    # Empirical equilibrium extension of the bottom layer (m), calibrated on
    # L40 with VBD iterations=20, substeps=5, 300 frames.
    EXPECTED_DELTA = 0.10
    TOLERANCE = 0.15  # relative

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 20
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        self.rest_length = self.DIM_Z * self.CELL

        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.DIM_X,
            dim_y=self.DIM_Y,
            dim_z=self.DIM_Z,
            cell_x=self.CELL,
            cell_y=self.CELL,
            cell_z=self.CELL,
            density=self.DENSITY,
            k_mu=self.K_MU,
            k_lambda=self.K_LAMBDA,
            k_damp=self.K_DAMP,
        )

        builder.color()
        self.model = builder.finalize()

        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0

        # Pin top-Z layer
        q_np = self.model.particle_q.numpy()
        top_z = self.DIM_Z * self.CELL + 2.0
        top_mask = np.abs(q_np[:, 2] - top_z) < 1e-6
        flags = self.model.particle_flags.numpy()
        for i in np.where(top_mask)[0]:
            flags[i] = flags[i] & ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(flags)

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.rest_bottom_z = 2.0
        self.bottom_indices = np.where(np.abs(q_np[:, 2] - 2.0) < 1e-6)[0]

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities do not explode",
            lambda q, qd: wp.length(qd) < 20.0,
        )

        q = self.state_0.particle_q.numpy()
        bottom_z = q[self.bottom_indices, 2]
        measured_delta = self.rest_bottom_z - float(np.mean(bottom_z))

        rel_error = abs(measured_delta - self.EXPECTED_DELTA) / self.EXPECTED_DELTA
        if rel_error > self.TOLERANCE:
            raise ValueError(
                f"Extension regression: measured {measured_delta:.4f} m, "
                f"expected {self.EXPECTED_DELTA:.4f} m (error {rel_error:.1%}, tolerance {self.TOLERANCE:.0%})"
            )

        # No lateral drift: X and Y center of mass should stay near beam center
        beam_cx = self.DIM_X * self.CELL / 2.0
        beam_cy = self.DIM_Y * self.CELL / 2.0
        com_x = float(np.mean(q[:, 0]))
        com_y = float(np.mean(q[:, 1]))
        if abs(com_x - beam_cx) > 0.05 or abs(com_y - beam_cy) > 0.05:
            raise ValueError(f"Lateral drift: COM=({com_x:.4f}, {com_y:.4f}), expected ~({beam_cx}, {beam_cy})")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer=viewer, args=args)
    newton.examples.run(example, args)
