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
# kinematic gripper whose full 6-DOF motion is prescribed by a velocity time series (one of the
# named VelocityProfile options, selected with --profile), and box B is a free body pulled up by
# the suction seal -- exercising the
# surface-gripper hold wrench in tension. Seal detection is still trivial (the pad is always
# engaged); the force model lives in surface_gripper.py.
#
# Command: python -m newton.examples surface_gripper_kinematic_drive
###########################################################################

import argparse
import math
from enum import Enum

import warp as wp

import newton
import newton.examples
from newton.examples.suctioncup.surface_gripper import (
    PadShape,
    SurfaceGripper,
    SurfaceGripperBuilder,
    evaluate_gripper_force,
    evaluate_seal,
    latch_engagement,
)

FPS = 60  # fixed render frame rate; the run length (num_frames) is derived from the profile
DEFAULT_SIM_DT = 1.0 / 240.0  # default physics timestep [s]


class VelocityProfile(Enum):
    """Named velocity time series for the kinematic gripper (box A).

    Select one with ``--profile`` (or pass it via ``args`` in a test). The profile's end time
    sets the total simulation length. See :func:`make_velocity_profile` for the keyframes.
    """

    CONSTANT_UP = 1
    LIFT_AND_LOWER = 2
    UP_AND_HOLD = 3
    CONSTANT_UP_AND_DROP = 4


DEFAULT_PROFILE = VelocityProfile.CONSTANT_UP


def make_profile(profile: VelocityProfile):
    """Return ``(vel_times, vel_twists, seal_times, seal_values)`` keyframes for ``profile``.

    ``vel_*`` is a 6-DOF velocity time series (world-frame ``(linear, angular)`` twists, sampled
    by piecewise-linear interpolation). ``seal_*`` is a boolean engaged/disengaged time series
    (sampled as a step function). ``vel_times[-1]`` is the profile's end time and sets the sim
    length (``num_frames = round(end_time * FPS)``). For now every seal profile is engaged for
    the whole run; per-profile seal series come later.
    """
    up = wp.spatial_vector(0.0, 0.0, 0.4, 0.0, 0.0, 0.0)
    down = wp.spatial_vector(0.0, 0.0, -0.4, 0.0, 0.0, 0.0)
    rest = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    seal_times = [0.0]
    seal_values = [True]  # engaged for the whole run
    if profile == VelocityProfile.CONSTANT_UP:
        # rise at 0.4 m/s for 10 s
        return [0.0, 10.0], [up, up], seal_times, seal_values
    if profile == VelocityProfile.LIFT_AND_LOWER:
        # up, hold at the top, then back down to the start over 10 s (net-zero displacement)
        return [0.0, 1.0, 4.0, 5.0, 6.0, 9.0, 10.0], [rest, up, up, rest, down, down, rest], seal_times, seal_values
    if profile == VelocityProfile.UP_AND_HOLD:
        # rise for 3 s, then hold in place for the rest of the 5 s run
        return [0.0, 3.0, 5.0], [up, up, rest], seal_times, seal_values
    if profile == VelocityProfile.CONSTANT_UP_AND_DROP:
        # same upward motion as CONSTANT_UP, but the seal releases at t = 4 s (box B drops)
        return [0.0, 10.0], [up, up], [0.0, 4.0], [True, False]
    raise ValueError(f"unknown velocity profile: {profile}")


class _SelectProfile(argparse.Action):
    """argparse action: selecting ``--profile`` also derives ``num_frames`` from its end time."""

    def __call__(self, parser, namespace, values, option_string=None):
        profile = VelocityProfile[values]  # member name -> enum
        setattr(namespace, self.dest, profile)
        vel_times = make_profile(profile)[0]
        namespace.num_frames = round(vel_times[-1] * FPS)


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


@wp.func
def eval_seal_at_current_time(times: wp.array[float], values: wp.array[int], n: int, t: float):
    """Step (piecewise-constant) lookup of a seal on/off series: the value of the latest
    keyframe at or before ``t``. The first value is held before the series starts, and the last
    value is held for any ``t`` past the final keyframe (so a short series still covers a longer
    run)."""
    if t <= times[0]:
        return values[0]
    if t >= times[n - 1]:
        return values[n - 1]
    v = values[0]
    for i in range(n):
        if times[i] <= t:
            v = values[i]
    return v


@wp.kernel
def update_seal(
    times: wp.array[float],
    values: wp.array[int],
    n: int,
    sim_time: wp.array[float],
    # out
    seal_engaged: wp.array[wp.bool],
):
    """Set each pad's engaged flag from the seal time series at the current time."""
    pad = wp.tid()
    seal_engaged[pad] = eval_seal_at_current_time(times, values, n, sim_time[0]) != 0


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
            "--profile",
            choices=[p.name for p in VelocityProfile],
            action=_SelectProfile,
            help="Velocity profile driving the gripper; its end time sets the sim length.",
        )
        parser.add_argument(
            "--sim-dt",
            type=float,
            default=DEFAULT_SIM_DT,
            help="Physics timestep in seconds; sim_substeps = round((1 / 60) / sim_dt).",
        )
        default_vel_times = make_profile(DEFAULT_PROFILE)[0]
        parser.set_defaults(profile=DEFAULT_PROFILE, num_frames=round(default_vel_times[-1] * FPS))
        return parser

    def __init__(self, viewer, args):
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # the velocity profile (--profile, or args.profile in a test) sets the run length: its
        # end time gives num_frames = round(end_time * fps). the physics timestep comes from
        # --sim-dt; sim_substeps is chosen so an integer number of substeps fills each frame and
        # sim_dt is snapped to match exactly. Fall back to defaults when args is not provided.
        self.velocity_profile = getattr(args, "profile", DEFAULT_PROFILE)
        vel_times, vel_twists, seal_times, seal_values = make_profile(self.velocity_profile)
        self.num_frames = getattr(args, "num_frames", round(vel_times[-1] * self.fps))
        requested_sim_dt = getattr(args, "sim_dt", DEFAULT_SIM_DT)
        self.sim_substeps = max(1, round(self.frame_dt / requested_sim_dt))
        self.sim_dt = self.frame_dt / self.sim_substeps

        # --- scene constants ---
        # geometry: box half-extents (A is the small gripper body, B the larger gripped box)
        LA = 0.1
        LB = 0.2
        # masses and gravity
        m_a, m_b = 0.5, 1.0
        gravity = 10.0
        # surface-gripper stiffness, damping and force limits. Shear stiffness (which also sets the
        # derived torsion stiffness) and peel damping are non-zero so all 6 DOF of the seal are
        # active: normal, shear (x, y), peel (x, y) and twist.
        k_normal = 5000.0
        d_normal = 100.0
        f_normal_max = 200.0
        f_grip_max = 1000.0
        k_shear_x = 2000.0
        k_shear_y = 2000.0
        mu_x = 1.0
        mu_y = 1.0
        d_peel_x = 10.0
        d_peel_y = 10.0
        pad_shape = PadShape.RECTANGLE

        # Create a scene with two bodies A and B.
        # A is kinematically controlled and has an attached surface gripper.
        # B is gripped by the surface gripper on A.
        builder = self._create_builder(LA, LB, m_a, m_b, gravity)
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Now create the gripper model
        gbuilder = self._create_gripper_builder(
            LA,
            k_normal,
            d_normal,
            f_normal_max,
            f_grip_max,
            k_shear_x,
            k_shear_y,
            mu_x,
            mu_y,
            d_peel_x,
            d_peel_y,
            pad_shape,
        )
        self.gripper_model = gbuilder.finalize(device=self.model.device)
        self.gripper_state = self.gripper_model.state()
        self.gripper_control = self.gripper_model.control()
        self.gripper_control.pad_grip_control.fill_(1.0)  # full grip command

        # Phase-1 seal decision: which body each pad seals against (fixed to box B here).
        self.seal_engaged = wp.full(1, True, dtype=wp.bool, device=self.model.device)
        self.seal_body_b = wp.full(1, self.box_b, dtype=wp.int32, device=self.model.device)

        # velocity time series driving the gripper (box A), from the selected profile.
        self.vel_times = wp.array(vel_times, dtype=wp.float32, device=self.model.device)
        self.vel_twists = wp.array(vel_twists, dtype=wp.spatial_vector, device=self.model.device)
        self.num_vel = len(vel_times)

        # seal on/off time series (engaged True/False), sampled per substep into seal_engaged.
        # an empty series means "no scripted seal": the physics decides via evaluate_seal instead.
        self.num_seal = len(seal_times)
        if self.num_seal > 0:
            self.seal_times = wp.array(seal_times, dtype=wp.float32, device=self.model.device)
            self.seal_values = wp.array([int(v) for v in seal_values], dtype=wp.int32, device=self.model.device)
        else:
            self.seal_times = None
            self.seal_values = None

        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)
        self.gripper_q_start = int(self.model.joint_q_start.numpy()[self.gripper_joint])
        self.gripper_qd_start = int(self.model.joint_qd_start.numpy()[self.gripper_joint])

        self.viewer.set_model(self.model)
        self.capture()

    def _create_builder(self, LA, LB, m_a, m_b, gravity):
        """Build the two-box scene: kinematic gripper A stacked on free box B, plus a ground plane.

        Sets ``self.box_a``, ``self.box_b`` and ``self.gripper_joint``, and returns the builder.
        """
        builder = newton.ModelBuilder(gravity=-gravity)

        # zero-density box shapes so the builder honours the explicit mass/inertia passed to
        # add_link/add_body; a nonzero-density shape would add its own computed mass on top.
        box_cfg = builder.default_shape_cfg.copy()
        box_cfg.density = 0.0

        def box_inertia(mass, half):
            s = (1.0 / 6.0) * mass * (2.0 * half) ** 2  # solid cube about its centre
            return wp.diag(wp.vec3(s, s, s))

        # box B (free, dynamic) rests on the ground; box A (the kinematic gripper) sits on top
        # with its pad sealing B's top face (A_center = B_center + LA + LB).
        pose_b = wp.transform(wp.vec3(0.0, 0.0, LB), wp.quat_identity())
        pose_a = wp.transform(wp.vec3(0.0, 0.0, LB + LA + LB), wp.quat_identity())

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
        self.box_b = builder.add_body(xform=pose_b, mass=m_b, inertia=box_inertia(m_b, LB), label="gripped_box_B")
        builder.add_shape_box(self.box_b, hx=LB, hy=LB, hz=LB, cfg=box_cfg)

        builder.add_ground_plane()
        return builder

    def _create_gripper_builder(
        self,
        LA,
        k_normal,
        d_normal,
        f_normal_max,
        f_grip_max,
        k_shear_x,
        k_shear_y,
        mu_x,
        mu_y,
        d_peel_x,
        d_peel_y,
        pad_shape,
    ):
        """Build the single-pad surface gripper on box A's bottom (-z) face; return its builder."""
        gripper = SurfaceGripper(
            body_id=self.box_a,
            xform=wp.transform_identity(),  # gripper frame == box A body frame
            k_normal=k_normal,
            d_normal=d_normal,
            f_normal_max=f_normal_max,
            f_grip_max=f_grip_max,
            k_shear_x=k_shear_x,
            k_shear_y=k_shear_y,
            mu_x=mu_x,
            mu_y=mu_y,
            d_peel_x=d_peel_x,
            d_peel_y=d_peel_y,
            shape=int(pad_shape),
            dim_a=LA,
            dim_b=LA,
        )
        # pad at the bottom face: origin (0,0,-LA), +z (suction dir) rotated to point down (-z of A)
        gripper.add_pad(wp.transform(wp.vec3(0.0, 0.0, -LA), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)))

        gbuilder = SurfaceGripperBuilder()
        gbuilder.add_gripper(gripper)
        return gbuilder

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):

            if self.num_seal > 0:
                # We have a pre-arranged seal temporal profile.
                # Evaluate the seal state from the pre-arranged temporal profile.
                wp.launch(
                    update_seal,
                    dim=self.seal_engaged.shape[0],
                    inputs=[self.seal_times, self.seal_values, self.num_seal, self.sim_time_wp],
                    outputs=[self.seal_engaged],
                )
            else:
                # We do not have a pre-arranged seal temporal profile.
                # Evaluate the current seal if we have one or evaluate
                # the current geometric state to determine if a seal can be formed.
                evaluate_seal(self.gripper_model, self.gripper_state, self.seal_engaged)

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

            # surface gripper -- Phase 1 (seal series -> engaged flag, above) then Phase 2 (wrench).
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
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
