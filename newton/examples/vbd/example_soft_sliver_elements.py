# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Sliver Elements
#
# A tetrahedral beam with intentionally poor aspect-ratio elements:
# cells are 10:1 in X vs Z, creating highly elongated tetrahedra. The
# solver should handle these degenerate rest shapes without NaN or
# explosion when the beam hangs under gravity.
#
# Command: python -m newton.examples vbd.example_soft_sliver_elements
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags


class Example:
    DIM_X = 2
    DIM_Y = 2
    DIM_Z = 10
    CELL_X = 0.2
    CELL_Y = 0.2
    CELL_Z = 0.02  # 10:1 aspect ratio
    DENSITY = 1000.0
    K_MU = 5.0e4
    K_LAMBDA = 5.0e4
    K_DAMP = 0.1

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

        self.beam_height = self.DIM_Z * self.CELL_Z

        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.DIM_X,
            dim_y=self.DIM_Y,
            dim_z=self.DIM_Z,
            cell_x=self.CELL_X,
            cell_y=self.CELL_Y,
            cell_z=self.CELL_Z,
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

        q_np = self.model.particle_q.numpy()
        top_z = 2.0 + self.beam_height
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
        q = self.state_0.particle_q.numpy()

        if np.any(np.isnan(q)):
            raise ValueError("NaN detected in particle positions with sliver elements")

        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities do not explode with sliver elements",
            lambda q, qd: wp.length(qd) < 20.0,
        )

        newton.examples.test_particle_state(
            self.state_0,
            "particles stay within reasonable bounds",
            lambda q, qd: newton.math.vec_inside_limits(q, wp.vec3(-2.0, -2.0, -0.5), wp.vec3(2.0, 2.0, 3.5)),
        )

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
