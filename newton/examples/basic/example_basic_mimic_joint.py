# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Mimic Joint
#
# Models a lead screw by making a prismatic nut joint mimic a revolute screw
# joint. The relationship converts screw rotation to linear travel:
#
#     q_nut = pitch / (2 * pi) * q_screw
#
# Command: python -m newton.examples basic_mimic_joint
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples

__all__ = ["Example"]


class Example:
    FPS = 60
    DEFAULT_SIM_SUBSTEPS = 4
    CYCLE_TIME = 6.0
    TRAVEL_AMPLITUDE = 0.45
    SCREW_CENTER_Z = 1.5
    SCREW_HALF_LENGTH = 1.15
    SCREW_RADIUS = 0.11
    THREAD_SEGMENTS = 192
    SOLVERS = ("featherstone", "semi_implicit", "xpbd", "mujoco", "vbd")

    def __init__(self, viewer, args):
        """Build and actuate a lead screw with a joint-owned mimic relationship."""
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.solver_name = args.solver
        self.frame_dt = 1.0 / self.FPS
        # VBD projects the mimic relationship inside each solver iteration, so
        # this simple mechanism does not benefit from additional substeps.
        self.sim_substeps = 1 if self.solver_name == "vbd" else self.DEFAULT_SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.pitch = float(args.pitch)
        if not math.isfinite(self.pitch) or self.pitch == 0.0:
            raise ValueError(f"pitch must be finite and nonzero, got {self.pitch}")
        self.coupling_ratio = self.pitch / (2.0 * math.pi)
        self.angle_amplitude = self.TRAVEL_AMPLITUDE / abs(self.coupling_ratio)
        if self.angle_amplitude >= math.pi:
            raise ValueError(
                "pitch is too small for this example: the screw motion must remain below half a revolution"
            )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        screw_cfg = newton.ModelBuilder.ShapeConfig(
            density=200.0,
            collision_group=0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        nut_cfg = newton.ModelBuilder.ShapeConfig(
            density=50.0,
            collision_group=0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        visual_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            collision_group=0,
            has_shape_collision=False,
            has_particle_collision=False,
        )

        screw_body = builder.add_link(label="lead_screw")
        builder.add_shape_cylinder(
            screw_body,
            radius=self.SCREW_RADIUS,
            half_height=self.SCREW_HALF_LENGTH,
            cfg=screw_cfg,
            color=(0.28, 0.42, 0.58),
            label="screw_shaft",
        )
        builder.add_shape_box(
            screw_body,
            xform=wp.transform(p=(0.24, 0.0, self.SCREW_HALF_LENGTH - 0.08), q=wp.quat_identity()),
            hx=0.24,
            hy=0.045,
            hz=0.045,
            cfg=visual_cfg,
            color=(0.95, 0.62, 0.16),
            label="rotation_indicator",
        )
        builder.add_shape_sphere(
            screw_body,
            xform=wp.transform(p=(0.47, 0.0, self.SCREW_HALF_LENGTH - 0.08), q=wp.quat_identity()),
            radius=0.075,
            cfg=visual_cfg,
            color=(0.95, 0.62, 0.16),
            label="rotation_indicator_tip",
        )
        self.screw_joint = builder.add_joint_revolute(
            parent=-1,
            child=screw_body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(p=(0.0, 0.0, self.SCREW_CENTER_Z), q=wp.quat_identity()),
            target_ke=80.0,
            target_kd=10.0,
            effort_limit=200.0,
            label="screw_rotation",
        )

        nut_body = builder.add_link(label="traveling_nut")
        builder.add_shape_box(
            nut_body,
            hx=0.4,
            hy=0.3,
            hz=0.13,
            cfg=nut_cfg,
            color=(0.24, 0.68, 0.42),
            label="traveling_nut",
        )
        self.nut_joint = builder.add_joint_prismatic(
            parent=-1,
            child=nut_body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(p=(0.0, 0.0, self.SCREW_CENTER_Z), q=wp.quat_identity()),
            limit_lower=-0.6,
            limit_upper=0.6,
            limit_ke=1.0e4,
            limit_kd=100.0,
            damping=0.2,
            label="nut_translation",
        )

        builder.add_articulation([self.screw_joint, self.nut_joint], label="lead_screw")
        builder.set_joint_mimic(self.nut_joint, self.screw_joint, coeffs=(0.0, self.coupling_ratio))

        for rail_x in (-0.58, 0.58):
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(p=(rail_x, 0.0, self.SCREW_CENTER_Z), q=wp.quat_identity()),
                hx=0.045,
                hy=0.045,
                hz=1.3,
                cfg=visual_cfg,
                color=(0.32, 0.34, 0.38),
                label="guide_rail",
            )
        for support_z in (0.2, 2.8):
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(p=(0.0, 0.0, support_z), q=wp.quat_identity()),
                hx=0.72,
                hy=0.36,
                hz=0.08,
                cfg=visual_cfg,
                color=(0.18, 0.2, 0.24),
                label="lead_screw_support",
            )

        builder.color()
        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.solver = self._create_solver()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        joint_q_start = self.model.joint_q_start.numpy()
        joint_target_q_start = self.model.joint_target_q_start.numpy()
        self.screw_q_index = int(joint_q_start[self.screw_joint])
        self.nut_q_index = int(joint_q_start[self.nut_joint])
        self.screw_target_index = int(joint_target_q_start[self.screw_joint])
        self.joint_q = wp.empty_like(self.model.joint_q)
        self.joint_qd = wp.empty_like(self.model.joint_qd)

        self.target_angle = 0.0
        self.screw_angle = 0.0
        self.nut_travel = 0.0
        self.coupling_error = 0.0
        self.max_abs_screw_angle = 0.0
        self.max_abs_coupling_error = 0.0

        thread_z = np.linspace(
            -self.SCREW_HALF_LENGTH,
            self.SCREW_HALF_LENGTH,
            self.THREAD_SEGMENTS + 1,
            dtype=np.float32,
        )
        self.thread_z = thread_z + self.SCREW_CENTER_Z
        self.thread_angle = thread_z * (2.0 * math.pi / self.pitch)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.8, -3.6, 2.4), pitch=-11.0, yaw=128.0)

    def _create_solver(self):
        """Create the selected solver with settings suitable for the mechanism."""
        if self.solver_name == "featherstone":
            return newton.solvers.SolverFeatherstone(self.model)
        if self.solver_name == "semi_implicit":
            return newton.solvers.SolverSemiImplicit(self.model)
        if self.solver_name == "xpbd":
            return newton.solvers.SolverXPBD(self.model, iterations=10)
        if self.solver_name == "mujoco":
            return newton.solvers.SolverMuJoCo(
                self.model,
                iterations=5,
                ls_iterations=10,
                disable_contacts=True,
                use_mujoco_cpu=wp.get_device().is_cpu,
            )
        if self.solver_name == "vbd":
            return newton.solvers.SolverVBD(
                self.model,
                iterations=2,
                rigid_compliant_alm=True,
                rigid_joint_linear_ke=1.0e6,
                rigid_joint_angular_ke=1.0e6,
            )
        raise ValueError(f"Unknown solver: {self.solver_name}")

    def simulate(self):
        """Advance the lead-screw mechanism by one rendered frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Drive the screw through a smooth reversing motion."""
        next_time = self.sim_time + self.frame_dt
        self.target_angle = self.angle_amplitude * math.sin(2.0 * math.pi * next_time / self.CYCLE_TIME)
        self.control.joint_target_q[self.screw_target_index : self.screw_target_index + 1].fill_(self.target_angle)

        self.simulate()
        self.sim_time = next_time
        self._read_joint_state()

        self.viewer.log_scalar("Screw angle [rev]", self.screw_angle / (2.0 * math.pi))
        self.viewer.log_scalar("Nut travel [m]", self.nut_travel)
        self.viewer.log_scalar("Mimic error [mm]", 1000.0 * self.coupling_error)

    def _read_joint_state(self):
        """Read generalized coordinates and update coupling diagnostics."""
        newton.eval_ik(self.model, self.state_0, self.joint_q, self.joint_qd)
        joint_q = self.joint_q.numpy()
        self.screw_angle = float(joint_q[self.screw_q_index])
        self.nut_travel = float(joint_q[self.nut_q_index])
        expected_travel = self.coupling_ratio * self.screw_angle
        self.coupling_error = self.nut_travel - expected_travel
        self.max_abs_screw_angle = max(self.max_abs_screw_angle, abs(self.screw_angle))
        self.max_abs_coupling_error = max(self.max_abs_coupling_error, abs(self.coupling_error))

    def _render_thread(self):
        """Render a helical line that rotates with the screw body."""
        angle = self.thread_angle + self.screw_angle
        radius = self.SCREW_RADIUS * 1.08
        points = np.column_stack(
            (
                radius * np.cos(angle),
                radius * np.sin(angle),
                self.thread_z,
            )
        ).astype(np.float32)
        starts = wp.array(points[:-1], dtype=wp.vec3, device=self.model.device)
        ends = wp.array(points[1:], dtype=wp.vec3, device=self.model.device)
        self.viewer.log_lines("/lead_screw/thread", starts, ends, (0.95, 0.72, 0.2))

    def render(self):
        """Render the mechanism and its helical thread guide."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._render_thread()
        self.viewer.end_frame()

    def gui(self, ui):
        """Display the selected solver and measured joint coordinates."""
        ui.text(f"Solver: {self.solver_name}")
        ui.text("q_nut = pitch / (2*pi) * q_screw")
        ui.separator()
        ui.text(f"Pitch: {self.pitch:.3f} m/rev")
        ui.text(f"Screw: {self.screw_angle / (2.0 * math.pi):.3f} rev")
        ui.text(f"Nut: {self.nut_travel:.3f} m")
        ui.text(f"Mimic error: {1000.0 * self.coupling_error:.2f} mm")

    def test_final(self):
        """Verify the screw moves and the nut follows its pitch ratio."""
        if self.max_abs_screw_angle < 0.5:
            raise ValueError("Lead-screw actuator did not produce enough rotation")
        if self.max_abs_coupling_error > 0.01:
            raise ValueError(f"Lead-screw mimic error exceeded 10 mm: {1000.0 * self.max_abs_coupling_error:.3f} mm")

    @staticmethod
    def create_parser():
        """Create the command-line parser for the lead-screw example."""
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            choices=Example.SOLVERS,
            default="xpbd",
            help="Solver backend to use.",
        )
        parser.add_argument(
            "--pitch",
            type=float,
            default=1.2,
            help="Signed lead-screw travel per revolution [m/rev].",
        )
        parser.set_defaults(num_frames=360)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
