# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""VBD dry joint friction: pendulums swing, stop, or hold against gravity.

Four identical 1 kg, 1 m rods start horizontal with friction bounds of
0, 0.5, 2, and 6 N*m. The maximum gravity torque is 4.905 N*m, so the last
rod stays horizontal. The middle rods dissipate energy and stop away from
the downward equilibrium; the first has no joint friction (but still has
time-integration dissipation). There are no contacts or passive dampers.

Run with:
    uv run --extra examples -m newton.examples vbd_joint_friction_pendulum
"""

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    """Compare gravitational motion with several dry joint-friction bounds."""

    FRICTIONS = (0.0, 0.5, 2.0, 6.0)
    MASS = 1.0
    LENGTH = 1.0
    GRAVITY = 9.81
    FPS = 60
    SUBSTEPS = 8
    ITERATIONS = 8

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / self.FPS
        self.sim_dt = self.frame_dt / self.SUBSTEPS
        self.sim_time = 0.0
        self._late_free_angle_min = np.inf
        self._late_free_angle_max = -np.inf

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -self.GRAVITY))
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
        colors = ((0.25, 0.55, 0.95), (0.35, 0.75, 0.55), (0.95, 0.65, 0.25), (0.95, 0.35, 0.20))
        inertia = self.MASS * self.LENGTH**2 / 12.0
        for index, (friction, color) in enumerate(zip(self.FRICTIONS, colors, strict=True)):
            pivot = wp.vec3((index - 1.5) * 1.5, 0.0, 1.5)
            body = builder.add_link(
                xform=wp.transform(pivot, wp.quat_identity()),
                mass=self.MASS,
                com=wp.vec3(self.LENGTH / 2.0, 0.0, 0.0),
                inertia=wp.mat33(np.eye(3) * inertia),
                lock_inertia=True,
                label=f"pendulum_{friction:g}",
            )
            builder.add_shape_box(
                body,
                xform=wp.transform((self.LENGTH / 2.0, 0.0, 0.0), wp.quat_identity()),
                hx=self.LENGTH / 2.0,
                hy=0.035,
                hz=0.035,
                cfg=visual,
                color=color,
            )
            builder.add_shape_sphere(-1, xform=wp.transform(pivot, wp.quat_identity()), radius=0.07, cfg=visual)
            joint = builder.add_joint_revolute(
                -1,
                body,
                parent_xform=wp.transform(pivot, wp.quat_identity()),
                axis=newton.Axis.Y,
                target_ke=0.0,
                target_kd=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
                friction=friction,
                label=f"hinge_{friction:g}",
            )
            builder.add_articulation([joint], label=f"friction_{friction:g}")

        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.ITERATIONS,
            rigid_compliant_alm=True,
            # Stiff but finite hinges approximate the ideal-pivot comparison.
            rigid_joint_linear_ke=1.0e8,
            rigid_joint_angular_ke=1.0e8,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.joint_q = wp.empty_like(self.model.joint_q)
        self.joint_qd = wp.empty_like(self.model.joint_qd)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.5, -8.0, 2.5), pitch=-10.0, yaw=90.0)
        if not args.quiet:
            print(f"Joint friction: {self.FRICTIONS} N*m; maximum gravity torque: 4.905 N*m")
            print("Blue swings; green and yellow slow and stop; red stays horizontal.")

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Advance one frame using an even number of capturable substeps."""
        for _ in range(self.SUBSTEPS):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        newton.eval_ik(self.model, self.state_0, self.joint_q, self.joint_qd)

    def step(self):
        """Advance the captured or uncaptured simulation by one display frame."""
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current pendulum poses."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Check trajectory invariants and record late friction-free motion."""
        angle, velocity = self.joint_q.numpy(), self.joint_qd.numpy()
        if not np.isfinite(angle).all() or not np.isfinite(velocity).all():
            raise ValueError("Non-finite pendulum state")
        if abs(angle[-1]) > 1.0e-5 or abs(velocity[-1]) > 1.0e-4:
            raise ValueError(f"Gravity exceeded the holding budget: q={angle[-1]}, qd={velocity[-1]}")
        if 0.25 <= self.sim_time <= 0.5 and np.min(angle[:3]) <= 0.05:
            raise ValueError(f"Below-gravity friction prevented release: q={angle[:3]}")
        if 6.0 <= self.sim_time <= 8.0 + self.frame_dt / 2.0:
            # This motion range cannot be reconstructed from the final state.
            self._late_free_angle_min = min(self._late_free_angle_min, float(angle[0]))
            self._late_free_angle_max = max(self._late_free_angle_max, float(angle[0]))

    def test_final(self):
        """Verify continued free motion and admissible frictional rest."""
        if self.sim_time < 8.0 - self.frame_dt / 2.0:
            raise ValueError(f"Pendulum test requires at least {8 * self.FPS} frames (8 s)")

        angle, velocity = self.joint_q.numpy(), self.joint_qd.numpy()
        if self._late_free_angle_max - self._late_free_angle_min < 0.5:
            raise ValueError("Friction-free pendulum failed to keep swinging")
        if np.max(np.abs(velocity[1:3])) > 1.0e-3:
            raise ValueError(f"Frictional pendulums failed to stop: qd={velocity[1:3]}")
        # Independent stick/slip work-energy reference for these rods.
        # Allow 0.03 rad for the finite-step integrator, not near-rest creep.
        np.testing.assert_allclose(angle[1:3], [1.5120913431, 1.8759852498], atol=0.03, rtol=0.0)
        gravity_torque = self.MASS * self.GRAVITY * self.LENGTH / 2.0 * np.cos(angle[1:3])
        if np.any(np.abs(gravity_torque) > np.array(self.FRICTIONS[1:3])):
            raise ValueError(f"Stopped outside the static-friction interval: torque={gravity_torque}")


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
