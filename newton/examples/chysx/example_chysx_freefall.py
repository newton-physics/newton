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
# To showcase chysx's `PinConstraint`, particle 0 (a corner of the grid)
# is pinned in place.  In the current freefall integrator the pin is
# enforced by hard-clamping the particle's position at the end of every
# step; once the chysx PCG step lands, the same `PinConstraint` will
# contribute a large diagonal block to the linear system instead.
#
# Without elastic forces the rest of the sheet still falls as a rigid
# plane (no inter-particle coupling), so visually you see particle 0
# floating in mid-air while the other 120 particles drop together.
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
        # Pin particle 0 (a grid corner).  Its initial position is
        # captured from `model.particle_q` and stays fixed forever.
        self._pinned_indices = [0]
        self.solver = newton.solvers.SolverChysX(
            self.model,
            pin_indices=self._pinned_indices,
        )

        # Snapshot the pinned particles' initial positions so test_final
        # can verify the clamp held.
        q0 = self.model.particle_q.numpy().reshape(-1, 3)
        self._pinned_targets = q0[self._pinned_indices].copy()

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
        # After self.sim_time seconds:
        #   * pinned particles are still exactly at their initial
        #     position with zero velocity (hard-clamped every step);
        #   * every other particle followed free-fall analytic
        #     z(t) = z0 - 0.5 g t^2, v_z(t) = -g t.
        z0 = self._initial_z
        g = 9.81
        t = self.sim_time
        z_expected = z0 - 0.5 * g * t * t
        vz_expected = -g * t

        # Pinned particles: position is exactly their initial location.
        # Verify by reading the state on the host (this is a single
        # particle, so the round-trip cost is negligible).
        q_np = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd_np = self.state_0.particle_qd.numpy().reshape(-1, 3)
        for k, idx in enumerate(self._pinned_indices):
            tx, ty, tz = self._pinned_targets[k]
            qx, qy, qz = q_np[idx]
            vx, vy, vz = qd_np[idx]
            pos_ok = abs(qx - tx) < 1e-5 and abs(qy - ty) < 1e-5 and abs(qz - tz) < 1e-5
            vel_ok = abs(vx) < 1e-5 and abs(vy) < 1e-5 and abs(vz) < 1e-5
            if not (pos_ok and vel_ok):
                raise ValueError(
                    f"Pinned particle {idx} drifted: "
                    f"pos=({qx:.4g},{qy:.4g},{qz:.4g}) target=({tx:.4g},{ty:.4g},{tz:.4g}) "
                    f"vel=({vx:.4g},{vy:.4g},{vz:.4g})"
                )

        # Free-fall particles: 5% relative error plus an absolute floor.
        tol_z = max(0.05 * abs(z_expected - z0), 0.05)
        tol_v = max(0.05 * abs(vz_expected), 0.05)

        def check(q, qd):
            return (
                abs(q[2] - z_expected) < tol_z
                and abs(qd[2] - vz_expected) < tol_v
                and abs(q[0]) < 2.0
                and abs(q[1]) < 2.0
                and abs(qd[0]) < 1e-4
                and abs(qd[1]) < 1e-4
            )

        # Test all particles except the pinned ones.
        free_indices = [
            i for i in range(self.model.particle_count)
            if i not in set(self._pinned_indices)
        ]
        newton.examples.test_particle_state(
            self.state_0,
            "free-fall particles match z(t) / v_z(t)",
            check,
            indices=free_indices,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=120)

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
