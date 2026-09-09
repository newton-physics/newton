# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Passive VBD gripper: drag the left finger against friction on the right.

Run with the example dependencies installed:
    uv run --extra examples -m newton.examples vbd_mimic_gripper

Right-drag either fingertip. The viewer draws the picking spring. Compare
friction = 0 with friction = 0.15 N*m; release the mouse to observe stopping.
Expand "Example Options" in the viewer sidebar for the friction slider and plots.
The plots show mirrored angles and speeds. VBD regularizes friction near
zero speed, so creep is normal.
The fingers are 30 cm long and about 0.8 kg each. The 16 substeps resolve the
friction smoothing region; much lighter fingers need a smaller step or friction.

For a reproducible coast-down without picking:
    uv run --extra examples -m newton.examples vbd_mimic_gripper --initial-speed 0.5
    uv run --extra examples -m newton.examples vbd_mimic_gripper --initial-speed 0.5 --friction 0
"""

import math
from collections import deque

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    FPS = 60
    # Resolve the narrow friction smoothing region for these ~0.8 kg fingers.
    SUBSTEPS = 16
    HISTORY_SECONDS = 8.0

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / self.FPS
        self.sim_dt = self.frame_dt / self.SUBSTEPS
        self.sim_time = 0.0
        self.friction = float(args.friction)
        if not math.isfinite(self.friction) or not 0.0 <= self.friction <= 0.3:
            raise ValueError("friction must be in [0, 0.3] N*m")
        if not math.isfinite(args.initial_speed) or abs(args.initial_speed) > 1.0:
            raise ValueError("initial-speed must be in [-1, 1] rad/s")

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        # With no contacts, motors, or passive damping, resistance comes from
        # hinge friction and, at the ends of travel, the angular limits.
        finger_cfg = newton.ModelBuilder.ShapeConfig(
            density=2200.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        visual_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        joints = []
        colors = ((0.22, 0.58, 0.95), (0.95, 0.48, 0.18))
        for i, side in enumerate((-1.0, 1.0)):
            body = builder.add_link(label=("left_finger", "right_finger")[i])
            # Body origin is the hinge; shape mass properties supply the COM.
            builder.add_shape_box(
                body,
                xform=wp.transform((0.0, 0.15, 0.0), wp.quat_identity()),
                hx=0.016,
                hy=0.15,
                hz=0.016,
                cfg=finger_cfg,
                color=colors[i],
            )
            builder.add_shape_box(
                body,
                xform=wp.transform((-side * 0.040, 0.284, 0.0), wp.quat_identity()),
                hx=0.024,
                hy=0.016,
                hz=0.016,
                cfg=finger_cfg,
                color=colors[i],
            )
            pivot = wp.transform((side * 0.09, 0.0, 0.08), wp.quat_identity())
            joint = builder.add_joint_revolute(
                parent=-1,
                child=body,
                parent_xform=pivot,
                axis=newton.Axis.Z,
                target_ke=0.0,
                target_kd=0.0,
                damping=0.0,
                armature=0.0,
                friction=0.0 if i == 0 else self.friction,
                limit_lower=0.0 if i == 0 else -math.radians(55.0),
                limit_upper=math.radians(55.0) if i == 0 else 0.0,
                limit_ke=100.0,
                limit_kd=0.1,
                label=("left_hinge", "right_hinge_with_friction")[i],
            )
            joints.append(joint)
            builder.add_shape_cylinder(
                body=-1,
                xform=pivot,
                radius=0.026,
                half_height=0.024,
                cfg=visual_cfg,
                color=colors[i],
            )

        builder.add_articulation(joints, label="passive_gripper")
        builder.set_joint_mimic(joints[1], joints[0], coeffs=(0.0, -1.0))
        builder.joint_q[:] = [math.radians(20.0), -math.radians(20.0)]
        builder.joint_qd[:] = [args.initial_speed, -args.initial_speed]
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, -0.020, 0.038), wp.quat_identity()),
            hx=0.13,
            hy=0.05,
            hz=0.020,
            cfg=visual_cfg,
            color=(0.25, 0.28, 0.32),
        )
        builder.color()
        self.model = builder.finalize()
        # VBD uses the model pose as its angular rest reference.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=12,
            rigid_compliant_alm=True,
            rigid_joint_linear_ke=1.0e6,
            rigid_joint_angular_ke=1.0e6,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.joint_q = wp.empty_like(self.model.joint_q)
        self.joint_qd = wp.empty_like(self.model.joint_qd)
        self.history = deque(maxlen=int(self.HISTORY_SECONDS * self.FPS))
        self.max_mimic_error = 0.0
        self.reset()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -0.40, 1.04), pitch=-62.0, yaw=90.0)
        self.graph = None
        if self.model.device.is_cuda:
            # Capture after set_model() so mouse-picking buffers are included.
            # The even substep count returns both state buffers to their starting roles.
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph

    def reset(self):
        """Restore the initial pose and velocity and clear VBD history."""
        self.solver.reset(self.state_0)
        for state in (self.state_0, self.state_1):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
            state.clear_forces()
        self.control.joint_f.zero_()
        self.sim_time = 0.0
        self.history.clear()
        self.q = self.model.joint_q.numpy()
        self.qd = self.model.joint_qd.numpy()
        self.max_mimic_error = 0.0

    def set_friction(self, friction):
        """Change only the follower's friction; VBD reads this array live."""
        self.friction = float(friction)
        self.model.joint_friction[1:2].fill_(self.friction)

    def simulate(self):
        """Advance one frame with GPU operations that can be captured and replayed."""
        for _ in range(self.SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # VBD advances body poses, so reconstruct joint coordinates for plots.
        newton.eval_ik(self.model, self.state_0, self.joint_q, self.joint_qd)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.q, self.qd = self.joint_q.numpy(), self.joint_qd.numpy()
        self.history.append(
            (
                self.sim_time,
                math.degrees(self.q[0]),
                -math.degrees(self.q[1]),
                math.degrees(self.qd[0]),
                -math.degrees(self.qd[1]),
            )
        )
        self.max_mimic_error = max(self.max_mimic_error, abs(float(self.q[0] + self.q[1])))

    def gui(self, ui):
        """Adjust follower friction live and plot mirrored angles and speeds."""
        from imgui_bundle import implot  # noqa: PLC0415 - optional GUI dependency

        ui.text_wrapped("Right-drag a fingertip; release to let go.")
        ui.text_wrapped("Blue: left hinge. Orange: frictional right hinge.")
        ui.text("Right hinge friction [N*m]")
        ui.set_next_item_width(-1)
        changed, friction = ui.slider_float("##right_friction", self.friction, 0.0, 0.3, "%.3f")
        if changed:
            self.set_friction(friction)
        if ui.button("Reset"):
            self.reset()
        ui.text(f"Left speed: {math.degrees(self.qd[0]):+.2f} deg/s")
        ui.text(f"qL + qR: {math.degrees(self.q[0] + self.q[1]):+.4f} deg")
        ui.text_wrapped("Near-zero friction is smoothed; slow creep is expected.")
        if len(self.history) < 2:
            return
        if implot.get_current_context() is None:
            implot.create_context()
        data = np.asarray(self.history, dtype=np.float64)
        time = data[:, 0].copy()
        plots = (
            ("Coupled motion", "Angle [deg]", (("qL", data[:, 1]), ("-qR", data[:, 2]))),
            ("Coupled speed", "Speed [deg/s]", (("qdL", data[:, 3]), ("-qdR", data[:, 4]))),
        )
        for title, ylabel, lines in plots:
            if implot.begin_plot(title, ui.ImVec2(-1, 175)):
                implot.setup_axes("Time [s]", ylabel, 0, implot.AxisFlags_.auto_fit)
                implot.setup_axis_limits(
                    implot.ImAxis_.x1,
                    max(0.0, self.sim_time - self.HISTORY_SECONDS),
                    max(self.HISTORY_SECONDS, self.sim_time),
                    implot.Cond_.always,
                )
                if title == "Coupled motion":
                    # Keep tiny resting drift from filling the angle plot.
                    implot.setup_axis_limits(implot.ImAxis_.y1, -2.0, 57.0, implot.Cond_.always)
                for label, values in lines:
                    implot.plot_line(label, time, values.copy())
                implot.end_plot()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Check finite state and coupling when run with --test."""
        if not np.isfinite(self.state_0.body_q.numpy()).all() or not np.isfinite(self.qd).all():
            raise ValueError("Non-finite gripper state")
        if self.max_mimic_error > math.radians(0.5):
            raise ValueError(f"Mimic error exceeded 0.5 degrees: {math.degrees(self.max_mimic_error):.4f}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--friction", type=float, default=0.15, help="Right hinge friction [N*m].")
        parser.add_argument("--initial-speed", type=float, default=0.0, help="Initial opening speed [rad/s].")
        parser.set_defaults(num_frames=600, render_fps=60)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
