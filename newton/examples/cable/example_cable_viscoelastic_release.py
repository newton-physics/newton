# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Viscoelastic Release
#
# Demonstrates standard-linear-solid (SLS) cable bending in SolverVBD. The
# left endpoint is clamped while a temporary world joint holds the raised tip,
# then releases it without resetting the cable material history.
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def update_release_state(
    joint_index: int,
    release_time: float,
    dt: float,
    joint_enabled: wp.array[bool],
    sim_time: wp.array[float],
):
    """Disable the temporary tip joint at the release time and advance time."""
    if wp.tid() == 0:
        if sim_time[0] >= release_time:
            joint_enabled[joint_index] = False
        sim_time[0] = sim_time[0] + dt


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 200
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_iterations = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.release_time = float(args.release_time)

        # Illustrative cable geometry and mass distribution.
        self.cable_length = 0.4
        self.cable_radius = 0.0032
        self.linear_density = 0.01714
        self.num_segments = 20
        segment_length = self.cable_length / self.num_segments

        # Illustrative material parameters; applications should identify these
        # values from their own material data.
        relaxed_bend_stiffness = float(args.relaxed_bend_stiffness)
        transient_bend_stiffness = 0.0 if args.no_visco else float(args.transient_bend_stiffness)
        visco_tau = 0.0 if args.no_visco else float(args.visco_tau)
        bend_damping = float(args.bend_damping)
        stretch_stiffness = 2.5e5
        self.with_visco = transient_bend_stiffness > 0.0 and visco_tau > 0.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverVBD.register_custom_attributes(builder)

        # Set the requested linear mass using Newton's per-segment capsule volume.
        segment_volume = math.pi * self.cable_radius**2 * segment_length + 4.0 / 3.0 * math.pi * self.cable_radius**3
        cable_density = self.linear_density * segment_length / segment_volume
        cable_cfg = builder.default_shape_cfg.copy()
        cable_cfg.density = cable_density
        cable_cfg.has_shape_collision = False

        # The held cable is nearly horizontal and raised slightly at the free end.
        held_angle = math.radians(8.0)
        start = wp.vec3(-0.2, 0.0, 0.65)
        direction = wp.vec3(math.cos(held_angle), 0.0, math.sin(held_angle))
        rod = newton.Rod.create_straight(
            start=start,
            direction=direction,
            length=self.cable_length,
            segment_count=self.num_segments,
            radius=self.cable_radius,
            twist_total=0.0,
        )

        points, quaternions = rod.points, rod.quaternions
        rod_bodies, rod_joints = builder.add_rod(
            rod=rod,
            cfg=cable_cfg,
            stretch_stiffness=stretch_stiffness,
            bend_stiffness=relaxed_bend_stiffness,
            bend_damping=bend_damping,
            wrap_in_articulation=False,
            color=wp.vec3(0.22, 0.68, 0.95),
            label="viscoelastic_cable",
            body_frame_origin="com",
        )
        self.rod_bodies = rod_bodies
        self.rod_joints = rod_joints

        # A visible kinematic clamp fixes the cable's first endpoint and frame.
        clamp = builder.add_link(xform=wp.transform(start, quaternions[0]))
        clamp_cfg = builder.default_shape_cfg.copy()
        clamp_cfg.density = 0.0
        clamp_cfg.has_shape_collision = False
        builder.add_shape_box(
            clamp,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.012), wp.quat_identity()),
            hx=0.012,
            hy=0.012,
            hz=0.018,
            cfg=clamp_cfg,
        )
        builder.body_mass[clamp] = 0.0
        builder.body_inv_mass[clamp] = 0.0
        builder.body_inertia[clamp] = wp.mat33(0.0)
        builder.body_inv_inertia[clamp] = wp.mat33(0.0)

        endpoint_local_start = wp.vec3(0.0, 0.0, -0.5 * segment_length)
        clamp_joint = builder.add_joint_fixed(
            parent=clamp,
            child=rod_bodies[0],
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform(endpoint_local_start, wp.quat_identity()),
            label="cable_clamp",
        )
        builder.add_articulation([*rod_joints, clamp_joint])

        # This standalone world joint represents the robot gripper. Disabling it
        # releases the tip while preserving the internal cable material state.
        endpoint_local_end = wp.vec3(0.0, 0.0, 0.5 * segment_length)
        self.tip_hold_joint = builder.add_joint_ball(
            parent=-1,
            child=rod_bodies[-1],
            parent_xform=wp.transform(points[-1], quaternions[-1]),
            child_xform=wp.transform(endpoint_local_end, wp.quat_identity()),
            label="temporary_tip_hold",
        )

        builder.color(balance_colors=False)
        sim_device = wp.get_device(args.device) if args.device else None
        self.model = builder.finalize(device=sim_device)
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.model.vbd.visco_bend_ke.fill_(transient_bend_stiffness)
        self.model.vbd.visco_bend_tau.fill_(visco_tau)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_compliant_alm=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.sim_time_array = wp.zeros(1, dtype=float, device=self.solver.device)

        self.initial_tip_body_z = 0.5 * (float(points[-2][2]) + float(points[-1][2]))
        self.expected_cable_mass = self.linear_density * self.cable_length

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.0, -0.65, 0.65),
            pitch=-10.0,
            yaw=90.0,
        )
        self.capture()

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                update_release_state,
                dim=1,
                inputs=[
                    self.tip_hold_joint,
                    self.release_time,
                    self.sim_dt,
                ],
                outputs=[
                    self.model.joint_enabled,
                    self.sim_time_array,
                ],
                device=self.solver.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify release, cable mass, and a nonzero enabled SLS response."""
        if self.state_0.body_q is None or self.state_0.body_qd is None:
            raise RuntimeError("Body state is not available.")

        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(body_qd)):
            raise ValueError("Non-finite cable state after release.")

        joint_enabled = self.model.joint_enabled.numpy()
        if joint_enabled[self.tip_hold_joint]:
            raise ValueError("Temporary tip joint did not release.")

        tip_body_z = float(body_q[self.rod_bodies[-1], 2])
        if tip_body_z >= self.initial_tip_body_z - 0.01:
            raise ValueError("Released cable tip did not move downward.")

        cable_mass = float(np.sum(self.model.body_mass.numpy()[self.rod_bodies]))
        if not np.isclose(cable_mass, self.expected_cable_mass, rtol=0.02):
            raise ValueError(f"Cable mass {cable_mass:.6f} kg does not match the configured linear density.")

        if self.with_visco:
            moments = self.solver.joint_visco_moment_prev.numpy()[self.rod_joints]
            if np.max(np.linalg.norm(moments, axis=1)) <= 1.0e-6:
                raise ValueError("Viscoelastic cable branch produced no bending moment.")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--release-time", type=float, default=0.5, help="Tip release time [s]")
        parser.add_argument("--no-visco", action="store_true", help="Disable the transient SLS branch")
        parser.add_argument(
            "--relaxed-bend-stiffness",
            type=float,
            default=0.6423,
            help="Relaxed per-joint bending stiffness [N*m/rad]",
        )
        parser.add_argument(
            "--transient-bend-stiffness",
            type=float,
            default=6.3639,
            help="Transient per-joint bending stiffness [N*m/rad]",
        )
        parser.add_argument(
            "--bend-damping",
            type=float,
            default=0.025,
            help="Effective per-joint bending damping [N*m*s/rad]",
        )
        parser.add_argument("--visco-tau", type=float, default=3.5, help="SLS relaxation time [s]")
        parser.set_defaults(num_frames=360)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
