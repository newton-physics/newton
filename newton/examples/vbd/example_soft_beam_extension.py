# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Beam Extension
#
# A vertical tetrahedral beam pinned at the top and hanging under gravity.
# At equilibrium the bottom-layer extension is compared to the analytical
# linear-elastic self-weight prediction delta = W*L/(2*A*E), derived from the
# material parameters (the stable Neo-Hookean material is calibrated to the
# linear Lame parameters). The test also checks there is no lateral drift
# (center of mass stays centered in X/Y) and that velocities stay bounded.
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

    # Tolerance on the relative error between the measured tip extension and the
    # analytical linear-elastic prediction (see test_final). The small residual is
    # finite-strain softening at ~7% strain.
    TOLERANCE = 0.10  # relative

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        # Even substep count keeps the CUDA-graph capture parity-safe: with an
        # odd count the state_0/state_1 ping-pong leaves the graph replaying
        # from the wrong start buffer, under-integrating each replayed frame.
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps
        # High iteration count so the stiff beam converges to its static equilibrium;
        # this lets the analytical linear-elastic check in test_final be tight (an
        # under-converged solve leaves the beam too soft and over-extended).
        self.iterations = 100
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
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
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

        # Analytical expectation from linear elasticity: a prismatic bar of length L
        # and cross-section A, fixed at the top and hanging under its own weight,
        # extends by W * L / (2 * A * E), where W = M * g is the total weight. The
        # stable Neo-Hookean material is calibrated to the linear Lame parameters
        # (mu_NH = mu, lambda_NH = lambda + mu; Smith et al. 2018), so its Young's
        # modulus is E = mu (3*lambda + 2*mu) / (lambda + mu). Mass is read from the
        # model because add_soft_grid lumps a full cell's mass onto every grid vertex.
        total_mass = float(np.sum(self.model.particle_mass.numpy()))
        gravity_z = abs(float(self.model.gravity.numpy().reshape(-1)[2]))
        young = self.K_MU * (3.0 * self.K_LAMBDA + 2.0 * self.K_MU) / (self.K_LAMBDA + self.K_MU)
        area = (self.DIM_X * self.CELL) * (self.DIM_Y * self.CELL)
        expected_delta = total_mass * gravity_z * self.rest_length / (2.0 * area * young)

        rel_error = abs(measured_delta - expected_delta) / expected_delta
        if rel_error > self.TOLERANCE:
            raise ValueError(
                f"Extension does not match linear-elastic theory: measured {measured_delta:.4f} m, "
                f"analytical {expected_delta:.4f} m (error {rel_error:.1%}, tolerance {self.TOLERANCE:.0%})"
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
