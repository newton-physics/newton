# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX Free Fall
#
# A flat cloth patch is dropped 2 m above the ground and integrated by the
# toy CUDA backend in `ChysX/`.  This is the smallest possible Newton
# example exercising a custom external physics engine: no collisions, no
# elasticity, no spring/bending forces — just `v += g*dt; x += v*dt`
# running on the GPU.
#
# Because the integrator is purely free-fall, every cloth particle receives
# the same gravity and starts from rest, so the whole sheet drops as a
# rigid plane.  This is exactly what we should see when the solver ignores
# all internal forces — a deliberate visual baseline before plugging in
# elasticity later.
#
# Command: python -m newton.examples chysx_freefall
#
###########################################################################

from __future__ import annotations

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        z0 = 2.0
        self._initial_z = z0

        # A flat 10x10 cloth patch (1 m x 1 m) hovering at z = z0.  We pass
        # zero stiffness for every spring/bending term — chysx wouldn't
        # read them anyway, but this also keeps Newton from trying to
        # initialise unused material attributes.
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, z0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=10,
            dim_y=10,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.05,
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.02,
        )

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverChysX(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(2.5, 2.5, 1.5),
            pitch=-15.0,
            yaw=-135.0,
        )
        self.viewer._paused = True  # start paused so the user can see the initial grid before it falls

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Free fall: v_z = -g t, z = z0 - 0.5 g t^2.  After self.sim_time
        # seconds, every particle should be at the same height (no
        # interaction between particles), strictly below the start height
        # and falling downward.
        z0 = self._initial_z
        g = 9.81
        t = self.sim_time
        z_expected = z0 - 0.5 * g * t * t
        vz_expected = -g * t

        def check(q, qd):
            # Numerical drift from semi-implicit Euler at dt=1/60 is small;
            # accept 5% relative error plus an absolute floor.
            tol_z = max(0.05 * abs(z_expected - z0), 0.05)
            tol_v = max(0.05 * abs(vz_expected), 0.05)
            return (
                abs(q[2] - z_expected) < tol_z
                and abs(qd[2] - vz_expected) < tol_v
                and abs(q[0]) < 2.0
                and abs(q[1]) < 2.0
                and abs(qd[0]) < 1e-4
                and abs(qd[1]) < 1e-4
            )

        newton.examples.test_particle_state(
            self.state_0,
            "free-fall trajectory matches analytic z(t) and v_z(t)",
            check,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=120)

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
