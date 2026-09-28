# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""VBD dry joint friction: applied torques below and above the static threshold.

Four identical unit-inertia hinges receive -8, -2, +2, and +8 N*m while each
has 4 N*m of Coulomb friction. The inner pair sticks because its applied torque
is below the friction bound. The outer pair slides in opposite directions with
4 N*m net torque. Dry joint friction requires the compliant-ALM rigid path.

Run with:
    uv run --extra examples -m newton.examples vbd_joint_friction
"""

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    """Compare sticking and sliding hinges under constant applied torque."""

    TORQUES = (-8.0, -2.0, 2.0, 8.0)
    FRICTION = 4.0
    FPS = 60
    SUBSTEPS = 4
    ITERATIONS = 10

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / self.FPS
        self.sim_dt = self.frame_dt / self.SUBSTEPS
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
        joints = []
        colors = ((0.25, 0.55, 0.95), (0.35, 0.75, 0.95), (0.95, 0.65, 0.25), (0.95, 0.35, 0.20))
        for index, (torque, color) in enumerate(zip(self.TORQUES, colors, strict=True)):
            pivot = wp.vec3((index - 1.5) * 1.25, 0.0, 1.0)
            body = builder.add_link(
                xform=wp.transform(pivot, wp.quat_identity()),
                mass=1.0,
                com=wp.vec3(0.0),
                inertia=wp.mat33(np.eye(3)),
                lock_inertia=True,
                label=f"torque_{torque:+.0f}",
            )
            builder.add_shape_box(
                body,
                xform=wp.transform((0.38, 0.0, 0.0), wp.quat_identity()),
                hx=0.38,
                hy=0.06,
                hz=0.06,
                cfg=visual,
                color=color,
            )
            builder.add_shape_cylinder(
                -1,
                xform=wp.transform(pivot, wp.quat_identity()),
                radius=0.09,
                half_height=0.08,
                cfg=visual,
                color=(0.25, 0.27, 0.30),
            )
            joints.append(
                builder.add_joint_revolute(
                    -1,
                    body,
                    parent_xform=wp.transform(pivot, wp.quat_identity()),
                    axis=newton.Axis.Z,
                    target_ke=0.0,
                    target_kd=0.0,
                    limit_ke=0.0,
                    limit_kd=0.0,
                    friction=self.FRICTION,
                    label=f"hinge_{torque:+.0f}",
                )
            )

        builder.add_articulation(joints, label="dry_friction_thresholds")
        builder.color()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.ITERATIONS,
            rigid_compliant_alm=True,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.control.joint_f.assign(self.TORQUES)
        self.joint_q = wp.empty_like(self.model.joint_q)
        self.joint_qd = wp.empty_like(self.model.joint_qd)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -7.5, 4.0), pitch=-20.0, yaw=90.0)
        if not args.quiet:
            print(f"Applied torques: {self.TORQUES} N*m; dry-friction bound: {self.FRICTION} N*m")
            print("The inner hinges stick; the outer hinges slide in opposite directions.")

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
        """Render the current hinge poses."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite state, static sticking, and analytic sliding speed."""
        angle, velocity = self.joint_q.numpy(), self.joint_qd.numpy()
        if not np.isfinite(angle).all() or not np.isfinite(velocity).all():
            raise ValueError("Non-finite joint-friction state")
        if np.max(np.abs(angle[1:3])) > 1.0e-5 or np.max(np.abs(velocity[1:3])) > 1.0e-5:
            raise ValueError(f"Sub-threshold hinges failed to stick: q={angle[1:3]}, qd={velocity[1:3]}")

        sliding_indices = [0, len(self.TORQUES) - 1]
        applied_torque = np.asarray(self.TORQUES)[sliding_indices]
        net_torque = applied_torque - np.sign(applied_torque) * self.FRICTION
        # Each hinge has unit inertia. Allow the finite-step rotation
        # integrator's error, not a changed dry-friction law.
        expected_velocity = net_torque * self.sim_time
        np.testing.assert_allclose(velocity[sliding_indices], expected_velocity, rtol=0.01, atol=2.0e-4)


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
