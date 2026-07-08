# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Surface Gripper Kinematic Drive
#
# A single-pad surface gripper (box A) is sealed onto the top of a larger box (B). Box A is a
# kinematic gripper whose full 6-DOF motion is prescribed by a velocity time series (see
# velocity_profile()), and box B is a free body pulled up by the suction seal -- exercising the
# surface-gripper hold wrench in tension. Seal detection is still trivial (the pad is always
# engaged); the force model lives in surface_gripper.py.
#
# Command: python -m newton.examples surface_gripper_kinematic_drive
###########################################################################

import argparse
import math

import warp as wp

import newton
import newton.examples
from newton.examples.suctioncup.surface_gripper import (
    PadShape,
    SurfaceGripper,
    SurfaceGripperBuilder,
    evaluate_gripper_force,
    latch_engagement,
)

FPS = 60  # fixed render frame rate; the run length (num_frames) is derived from --sim-time
DEFAULT_SIM_TIME = 10.0  # default total simulation time [s]
DEFAULT_SIM_DT = 1.0 / 240.0  # default physics timestep [s]


class _SetNumFramesFromSimTime(argparse.Action):
    """argparse action: setting ``--sim-time`` also derives ``num_frames = round(sim_time * FPS)``."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.num_frames = round(values * FPS)


@wp.func
def eval_twist_at_current_time(times: wp.array[float], twists: wp.array[wp.spatial_vector], n: int, t: float):
    """Piecewise-linear lookup of a 6-DOF velocity time series at time ``t`` (clamped at ends)."""
    if t <= times[0]:
        return twists[0]
    if t >= times[n - 1]:
        return twists[n - 1]
    v = twists[n - 1]
    for i in range(n - 1):
        if times[i] <= t and t < times[i + 1]:
            frac = (t - times[i]) / (times[i + 1] - times[i])
            v = twists[i] * (1.0 - frac) + twists[i + 1] * frac
    return v


@wp.kernel
def apply_twist_at_current_time(
    times: wp.array[float],
    twists: wp.array[wp.spatial_vector],
    n: int,
    sim_dt: float,
    q_start: int,
    qd_start: int,
    # in/out
    sim_time: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Kinematically drive the gripper (box A) from a 6-DOF velocity time series.

    Samples the prescribed world-frame twist ``(linear, angular)`` at the current time and
    applies it to the free joint: sets the joint velocity and integrates the joint pose by one
    substep (position linearly, orientation by the quaternion derivative). ``sim_time`` is
    advanced by one substep. Free-joint layout is ``joint_q = [px, py, pz, qx, qy, qz, qw]`` and
    ``joint_qd = [vx, vy, vz, wx, wy, wz]``.
    """
    velocity = eval_twist_at_current_time(times, twists, n, sim_time[0])
    v_lin = wp.spatial_top(velocity)
    v_ang = wp.spatial_bottom(velocity)

    p = wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2])
    q = wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6])

    p = p + v_lin * sim_dt
    # quaternion derivative for a world-frame angular velocity: q_dot = 0.5 * omega_quat * q
    q = wp.normalize(q + (wp.quat(v_ang[0], v_ang[1], v_ang[2], 0.0) * q) * (0.5 * sim_dt))

    joint_q[q_start + 0] = p[0]
    joint_q[q_start + 1] = p[1]
    joint_q[q_start + 2] = p[2]
    joint_q[q_start + 3] = q[0]
    joint_q[q_start + 4] = q[1]
    joint_q[q_start + 5] = q[2]
    joint_q[q_start + 6] = q[3]

    joint_qd[qd_start + 0] = v_lin[0]
    joint_qd[qd_start + 1] = v_lin[1]
    joint_qd[qd_start + 2] = v_lin[2]
    joint_qd[qd_start + 3] = v_ang[0]
    joint_qd[qd_start + 4] = v_ang[1]
    joint_qd[qd_start + 5] = v_ang[2]

    sim_time[0] = sim_time[0] + sim_dt


class Example:
    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        # this example derives the frame count from the total sim time, so remove --num-frames.
        for action in list(parser._actions):
            if "--num-frames" in action.option_strings:
                parser._remove_action(action)
                for opt in action.option_strings:
                    parser._option_string_actions.pop(opt, None)
        parser.add_argument(
            "--sim-time",
            type=float,
            action=_SetNumFramesFromSimTime,
            help="Total simulation time in seconds (num_frames = sim_time * 60).",
        )
        parser.add_argument(
            "--sim-dt",
            type=float,
            default=DEFAULT_SIM_DT,
            help="Physics timestep in seconds; sim_substeps = round((1 / 60) / sim_dt).",
        )
        parser.set_defaults(sim_time=DEFAULT_SIM_TIME, num_frames=round(DEFAULT_SIM_TIME * FPS))
        return parser

    def __init__(self, viewer, args):
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # run length and physics timestep come from the CLI (--sim-time, --sim-dt); fall back to
        # sensible defaults when args is not provided (e.g. instantiated directly in a test).
        # num_frames = sim_time * fps. sim_substeps is chosen so an integer number of substeps
        # fills each frame, and sim_dt is snapped to match exactly. The velocity profile spans
        # the whole run (total_time = num_frames / fps).
        self.num_frames = getattr(args, "num_frames", round(DEFAULT_SIM_TIME * self.fps))
        requested_sim_dt = getattr(args, "sim_dt", DEFAULT_SIM_DT)
        self.sim_substeps = max(1, round(self.frame_dt / requested_sim_dt))
        self.sim_dt = self.frame_dt / self.sim_substeps

        # box half-extents: A is the small gripper body, B is the larger gripped box
        LA = 0.1
        LB = 0.2

        def box_inertia(mass, half):
            s = (1.0 / 6.0) * mass * (2.0 * half) ** 2  # solid cube about its centre
            return wp.diag(wp.vec3(s, s, s))

        m_a, m_b = 0.5, 1.0
        gravity = 10.0

        # box B (free, dynamic) rests on the ground; box A (the kinematic gripper) sits on top
        # with its pad sealing B's top face (A_center = B_center + LA + LB).
        pose_b = wp.transform(wp.vec3(0.0, 0.0, LB), wp.quat_identity())
        pose_a = wp.transform(wp.vec3(0.0, 0.0, LB + LA + LB), wp.quat_identity())

        builder = newton.ModelBuilder(gravity=-gravity)

        # zero-density box shapes so the builder honours the explicit mass/inertia passed to
        # add_link/add_body; a nonzero-density shape would add its own computed mass on top.
        box_cfg = builder.default_shape_cfg.copy()
        box_cfg.density = 0.0

        # box A is the gripper: kinematic (does not respond to forces) and driven entirely by a
        # prescribed 6-DOF velocity through a free joint to the world -- every DOF is prescribed.
        self.box_a = builder.add_link(
            xform=pose_a,
            mass=m_a,
            inertia=box_inertia(m_a, LA),
            is_kinematic=True,
            label="gripper_box_A",
        )
        builder.add_shape_box(self.box_a, hx=LA, hy=LA, hz=LA, cfg=box_cfg)
        self.gripper_joint = builder.add_joint_free(child=self.box_a, label="gripper_free")
        builder.add_articulation([self.gripper_joint], label="gripper_articulation")

        # box B is the gripped object: a free body pulled up by the suction seal (no joint drive).
        self.box_b = builder.add_body(
            xform=pose_b, mass=m_b, inertia=box_inertia(m_b, LB), label="gripped_box_B"
        )
        builder.add_shape_box(self.box_b, hx=LB, hy=LB, hz=LB, cfg=box_cfg)

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # surface gripper on box A: one pad covering A's bottom (-z) face
        gripper = SurfaceGripper(
            body_id=self.box_a,
            xform=wp.transform_identity(),  # gripper frame == box A body frame
            k_normal=5000.0,
            d_normal=100.0,
            f_normal_max=200.0,
            f_grip_max=1000.0,
            # initial run: normal forces only. Zero shear stiffness (also zeros the derived
            # torsion stiffness) and zero peel damping so only the normal DOF is active.
            k_shear_x=0.0,
            k_shear_y=0.0,
            mu_x=1.0,
            mu_y=1.0,
            d_peel_x=0.0,
            d_peel_y=0.0,
            shape=int(PadShape.RECTANGLE),
            dim_a=LA,
            dim_b=LA,
        )
        # pad at the bottom face: origin (0,0,-LA), +z (suction dir) rotated to point down (-z of A)
        gripper.add_pad(wp.transform(wp.vec3(0.0, 0.0, -LA), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)))

        gbuilder = SurfaceGripperBuilder()
        gbuilder.add_gripper(gripper)
        self.gripper_model = gbuilder.finalize(device=self.model.device)
        self.gripper_state = self.gripper_model.state()
        self.gripper_control = self.gripper_model.control()
        self.gripper_control.pad_grip_control.fill_(1.0)  # full grip command

        # trivial Phase-1 seal decision for now: the single pad is always engaged to box B
        self.seal_engaged = wp.full(1, True, dtype=wp.bool, device=self.model.device)
        self.seal_body_b = wp.full(1, self.box_b, dtype=wp.int32, device=self.model.device)

        # velocity time series driving the gripper (box A). Override velocity_profile() in a
        # subclass to run a different series per test.
        profile_times, profile_twists = self.velocity_profile()
        self.vel_times = wp.array(profile_times, dtype=wp.float32, device=self.model.device)
        self.vel_twists = wp.array(profile_twists, dtype=wp.spatial_vector, device=self.model.device)
        self.num_vel = len(profile_times)
        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)
        self.gripper_q_start = int(self.model.joint_q_start.numpy()[self.gripper_joint])
        self.gripper_qd_start = int(self.model.joint_qd_start.numpy()[self.gripper_joint])

        self.viewer.set_model(self.model)
        self.capture()

    def velocity_profile(self):
        """Return the gripper's velocity time series as ``(times [s], twists)``.

        Each twist is a world-frame 6-DOF spatial velocity ``(linear, angular)``, sampled by
        piecewise-linear interpolation. Override this in a subclass to drive box A with a
        different series per test. The default is a constant upward motion: 0.4 m/s along +z for
        the whole run (``total_time = num_frames / fps``).
        """
        total_time = self.num_frames / self.fps
        up = wp.spatial_vector(0.0, 0.0, 0.4, 0.0, 0.0, 0.0)
        num_entries = 11  # keyframes evenly spaced over the run, all the same 0.4 m/s upward
        times = []
        twists = []
        for i in range(num_entries):
            times.append(i * total_time / (num_entries - 1))
            twists.append(up)
        return times, twists

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            # kinematically drive the gripper (box A) at its prescribed 6-DOF velocity, then
            # propagate the new joint state to the kinematic body's maximal coordinates.
            wp.launch(
                apply_twist_at_current_time,
                dim=1,
                inputs=[
                    self.vel_times,
                    self.vel_twists,
                    self.num_vel,
                    self.sim_dt,
                    self.gripper_q_start,
                    self.gripper_qd_start,
                ],
                outputs=[self.sim_time_wp, self.state_0.joint_q, self.state_0.joint_qd],
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )

            self.model.collide(self.state_0, self.contacts)

            # surface gripper -- Phase 1 (trivial: always engaged) then Phase 2 (apply wrench).
            # clear body_f right before writing the seal wrench, since eval_pad_force accumulates.
            self.state_0.clear_forces()
            latch_engagement(self.state_0, self.gripper_model, self.gripper_state, self.seal_engaged, self.seal_body_b)
            evaluate_gripper_force(
                self.model, self.state_0, self.gripper_model, self.gripper_state, self.gripper_control
            )

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # box A (kinematic) rises at a constant 0.4 m/s; box B is carried by the suction seal.
        # Sanity bounds only (the exact height depends on the run length): both rise off the
        # start, stay on-axis, and don't fly off. The per-step seal check lives in test_surface_gripper.
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "gripper A lifts box B upward and both stay on-axis",
            lambda q, qd: 0.1 < q[2] < 5.0 and abs(q[0]) < 0.1 and abs(q[1]) < 0.1,
            [self.box_a, self.box_b],
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
