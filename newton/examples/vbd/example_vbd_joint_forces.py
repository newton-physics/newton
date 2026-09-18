# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure a driven pendulum's hinge load with VBD's experimental force query.

Run with:
    uv run --extra examples -m newton.examples vbd_joint_forces

The Viewer plots the force magnitude, net torque about the hinge axis, and
motor-effort estimate. Wrenches act from parent to child, in the child joint
frame, about the joint origin. The net torque includes friction and damping;
it is not the motor torque alone. The motor estimate includes the drive effort
and an armature * acceleration correction. The local VBD solve does not simulate armature
inertia; that correction estimates extra effort for the observed motion.

Each plotted sample is from the last substep, not an average over the frame.
Right-drag the pendulum to apply an external load and see the readings change.
"""

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def update_target(time: wp.array[float], target: wp.array[float], dt: float):
    time[0] += dt
    target[0] = 0.8 * wp.sin(1.5 * time[0])


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
        link = builder.add_link(label="pendulum")
        builder.add_shape_box(
            link,
            xform=wp.transform((0.0, 0.0, -0.3), wp.quat_identity()),
            hx=0.025,
            hy=0.025,
            hz=0.3,
            cfg=cfg,
            color=(0.2, 0.6, 0.95),
        )
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            axis=newton.Axis.Y,
            target_ke=100.0,
            target_kd=10.0,
            friction=0.3,
            damping=0.5,
            armature=0.08,
            label="driven_hinge",
        )
        builder.add_articulation([joint])
        builder.add_shape_sphere(
            -1,
            xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            radius=0.045,
            cfg=cfg,
            color=(0.95, 0.5, 0.15),
        )
        builder.color()
        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.solver = newton.solvers.SolverVBD(self.model, iterations=12, rigid_compliant_alm=True)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.control = self.model.control()
        self.time = wp.zeros(1, device=self.model.device)

        # The query writes into caller-owned buffers; snapshots must not alias
        # states because VBD also updates its input poses and velocities.
        self.body_q_prev = wp.empty_like(self.state_0.body_q)
        self.joint_qd_prev = wp.empty_like(self.state_0.joint_qd)
        self.joint_wrench = wp.zeros(self.model.joint_count, dtype=wp.spatial_vector, device=self.model.device)
        self.joint_effort = wp.zeros_like(self.state_0.joint_qd)
        self.wrench = np.zeros(6)
        self.motor_effort = 0.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -2.8, 1.25), pitch=-10.0, yaw=90.0)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for substep in range(self.sim_substeps):
            wp.launch(update_target, dim=1, inputs=[self.time, self.control.joint_target_q, self.sim_dt])
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if substep == self.sim_substeps - 1:
                wp.copy(self.body_q_prev, self.state_0.body_q)
                wp.copy(self.joint_qd_prev, self.state_0.joint_qd)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Query immediately after the final step, with that step's dt/control.
        # The optional motor estimate needs the previous joint velocities too.
        self.solver.eval_joint_forces(
            self.state_0,
            self.joint_wrench,
            body_q_prev=self.body_q_prev,
            dt=self.sim_dt,
            control=self.control,
            joint_effort=self.joint_effort,
            joint_qd_prev=self.joint_qd_prev,
        )

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        # Only plotting needs a device-to-host copy, once per displayed frame.
        self.wrench = self.joint_wrench.numpy()[0]
        self.motor_effort = float(self.joint_effort.numpy()[0])

    def gui(self, ui):
        ui.text_wrapped("Blue pendulum: one driven hinge with friction and viscous damping. Right-drag to add a load.")
        ui.text_wrapped("Force and net torque act on the pendulum at the orange hinge, in its child joint frame.")
        ui.text_wrapped(
            "Motor estimate = drive effort + armature x acceleration. The local solve ignores armature inertia."
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("Hinge force magnitude [N]", np.linalg.norm(self.wrench[:3]))
        # This hinge rotates about child-joint Y: spatial_vector[4] is Ty.
        self.viewer.log_scalar("Hinge net torque Y [N*m]", self.wrench[4])
        self.viewer.log_scalar("Motor effort estimate [N*m]", self.motor_effort)
        self.viewer.end_frame()

    def test_final(self):
        """Check finite joint loads and a nonzero hinge support force."""
        if not np.isfinite(self.joint_wrench.numpy()).all() or not np.isfinite(self.joint_effort.numpy()).all():
            raise ValueError("Non-finite joint force readout")
        if np.linalg.norm(self.wrench[:3]) < 1.0:
            raise ValueError("Expected the hinge to support the pendulum against gravity")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600, render_fps=60)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
