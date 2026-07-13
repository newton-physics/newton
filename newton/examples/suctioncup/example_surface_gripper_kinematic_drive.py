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
from dataclasses import dataclass
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
    UP_LEFT_DOWN = 5
    UP_LEFT_ROLL_DOWN = 6
    SEAL_YANK_BREAK = 7
    SEAL_SWING_BREAK_X = 8
    SEAL_SWING_BREAK_Y = 9
    SEAL_SWING_BREAK_XY = 10


DEFAULT_PROFILE = VelocityProfile.UP_LEFT_ROLL_DOWN
ALL_PROFILES = "ALL"  # "--profile ALL": play every profile back-to-back (a viewer demo reel)


@dataclass
class GripperConfig:
    """Surface-gripper parameters for a velocity profile (defaults shown; profiles may override).

    Non-zero shear stiffness (which also sets the derived torsion stiffness) and peel damping keep
    all 6 seal DOF active: normal, shear (x, y), peel (x, y) and twist. The physics-break profiles
    override these -- e.g. a low ``grip_command`` so the seal reads intact at rest, or a huge
    ``f_normal_max`` to zero out the normal term of the break metric and isolate peel.
    """

    k_normal: float = 5000.0  # normal spring stiffness [N/m]
    d_normal: float = 100.0  # normal damping [N.s/m]
    f_normal_max: float = 200.0  # break-threshold normal force [N]
    f_grip_max: float = 1000.0  # max suction preload [N]
    k_shear_x: float = 2000.0  # shear stiffness about x [N/m]
    k_shear_y: float = 2000.0  # shear stiffness about y [N/m]
    mu_x: float = 1.0  # shear/twist friction coefficient (x)
    mu_y: float = 1.0  # shear/twist friction coefficient (y)
    d_peel_x: float = 10.0  # peel damping about x [N.m.s/rad]
    d_peel_y: float = 10.0  # peel damping about y [N.m.s/rad]
    pad_shape: PadShape = PadShape.RECTANGLE  # pad cross-section
    grip_command: float = 1.0  # constant grip command in [0, 1] (f_min = grip_command * f_grip_max)


def make_profile(profile: VelocityProfile):
    """Return ``(vel_times, vel_twists, seal_times, seal_values, gripper)`` for ``profile``.

    ``vel_*`` is a 6-DOF velocity time series (world-frame ``(linear, angular)`` twists, sampled
    by piecewise-linear interpolation). ``seal_*`` is a boolean engaged/disengaged time series
    (sampled as a step function); an empty series hands engagement to the physics (break metric).
    ``gripper`` is the :class:`GripperConfig` for this profile (default seal unless overridden).
    ``vel_times[-1]`` is the profile's end time and sets the sim length
    (``num_frames = round(end_time * FPS)``).
    """
    gripper = GripperConfig()  # default seal; physics-break profiles below override specific values
    speed = 0.8  # magnitude of the prescribed gripper velocity [m/s]
    up = wp.spatial_vector(0.0, 0.0, speed, 0.0, 0.0, 0.0)
    down = wp.spatial_vector(0.0, 0.0, -speed, 0.0, 0.0, 0.0)
    left = wp.spatial_vector(0.0, -speed, 0.0, 0.0, 0.0, 0.0)
    rest = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # rotating legs for UP_LEFT_ROLL_DOWN. Angular velocity is world-frame; while A is near its
    # start orientation these axes match the pad-frame seal DOF -- twist about z (the suction axis),
    # swing (peel) about x and y. spin gives ~90 deg per leg, spread over the ~1.25 s each leg is
    # active (its flat span plus half of each adjoining 0.2 s ramp).
    spin = (math.pi / 2.0) / 1.25
    up_twist = wp.spatial_vector(0.0, 0.0, speed, 0.0, 0.0, spin)  # up + twist (z)
    left_swing_y = wp.spatial_vector(0.0, -speed, 0.0, 0.0, spin, 0.0)  # left + swing Y
    down_swing_x_twist = wp.spatial_vector(0.0, 0.0, -speed, spin, 0.0, spin)  # down + swing X + twist
    yank_up = wp.spatial_vector(0.0, 0.0, 25.0, 0.0, 0.0, 0.0)  # hard upward yank (high accel -> normal break)
    seal_times = [0.0]
    seal_values = [True]  # engaged for the whole run
    if profile == VelocityProfile.CONSTANT_UP:
        # rise at 0.8 m/s for 10 s
        return [0.0, 10.0], [up, up], seal_times, seal_values, gripper
    if profile == VelocityProfile.LIFT_AND_LOWER:
        # up, hold at the top, then back down to the start over 10 s (net-zero displacement)
        return [0.0, 1.0, 4.0, 5.0, 6.0, 9.0, 10.0], [rest, up, up, rest, down, down, rest], seal_times, seal_values, gripper
    if profile == VelocityProfile.UP_AND_HOLD:
        # rise for 3 s, then hold in place for the rest of the 5 s run
        return [0.0, 3.0, 5.0], [up, up, rest], seal_times, seal_values, gripper
    if profile == VelocityProfile.CONSTANT_UP_AND_DROP:
        # same upward motion as CONSTANT_UP, but the seal releases at t = 4 s (box B drops)
        return [0.0, 10.0], [up, up], [0.0, 4.0], [True, False], gripper
    if profile == VelocityProfile.UP_LEFT_DOWN:
        # lift 1 m (+z), shift 1 m left (-y), then lower 1 m (-z), all at 0.8 m/s with 0.2 s ramps
        # between legs and a final 0.2 s ramp to rest. Flat times (1.15/1.05/1.05 s) plus each
        # leg's share of the ramps make every leg exactly 1 m. Ending at rest holds A in place
        # afterwards (the sampler clamps past the end to the last twist, so a trailing 'down' would
        # keep dragging B down if playback runs past the profile). Seal engaged throughout.
        return (
            [0.0, 1.15, 1.35, 2.40, 2.60, 3.65, 3.85],
            [up, up, left, left, down, down, rest],
            seal_times,
            seal_values,
            gripper,
        )
    if profile == VelocityProfile.UP_LEFT_ROLL_DOWN:
        # like UP_LEFT_DOWN (1 m legs at 0.8 m/s, 0.2 s ramps), but each leg adds a rotation: twist
        # about z on the way up, swing about y while shifting 1 m left, then swing about x plus
        # twist about z on the way down. Rotations ramp in/out with their leg (~90 deg each). Seal
        # engaged throughout (exercises the seal's twist and swing/peel resistance).
        return (
            [0.0, 1.15, 1.35, 2.40, 2.60, 3.65, 3.85],
            [up_twist, up_twist, left_swing_y, left_swing_y, down_swing_x_twist, down_swing_x_twist, rest],
            seal_times,
            seal_values,
            gripper,
        )
    if profile == VelocityProfile.SEAL_YANK_BREAK:
        # lift to establish the grip, then a brief hard yank straight up and back to rest. The seal
        # series is EMPTY, so the physics owns engagement: the yank's high acceleration drives box B's
        # normal tension past the break threshold, the break metric exceeds 1 and the seal releases on
        # its own. B is flung up, detaches, and falls to the ground while the gripper holds still. The
        # low grip command keeps the seal intact at rest (full grip would put f_min above f_normal_max);
        # the full-strength break threshold is what the yank has to overcome.
        return (
            [0.0, 0.3, 1.3, 1.4, 1.5, 4.0],
            [rest, up, up, yank_up, rest, rest],
            [],
            [],
            GripperConfig(grip_command=0.05),
        )
    if profile == VelocityProfile.SEAL_SWING_BREAK_X:
        # Peel break by swinging (rolling) about x. Lift to establish the grip, then swing slowly
        # about x for ~1.4 s -- box B follows, so the gripper visibly rolls it through most of a turn.
        # Then a brief hard swing spike: the high angular acceleration makes B lag rotationally, the
        # elastic peel moment builds past its capacity, the break metric exceeds 1, and the seal
        # releases (B flies off). f_normal_max is sky-high (see GripperConfig) so the normal term of
        # the break metric is ~0, proving the break is purely peel.
        swing_slow = wp.spatial_vector(0.0, 0.0, 0.0, 4.0, 0.0, 0.0)
        swing_spike = wp.spatial_vector(0.0, 0.0, 0.0, 30.0, 0.0, 0.0)
        return (
            [0.0, 0.3, 1.3, 1.6, 3.0, 3.1, 3.2, 5.5],
            [rest, up, up, swing_slow, swing_slow, swing_spike, rest, rest],
            [],
            [],
            GripperConfig(grip_command=0.05, f_normal_max=1.0e9),
        )
    if profile == VelocityProfile.SEAL_SWING_BREAK_Y:
        # like SEAL_SWING_BREAK_X, but the slow swing and the break spike are about y (peel about y).
        swing_slow = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 4.0, 0.0)
        swing_spike = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 30.0, 0.0)
        return (
            [0.0, 0.3, 1.3, 1.6, 3.0, 3.1, 3.2, 5.5],
            [rest, up, up, swing_slow, swing_slow, swing_spike, rest, rest],
            [],
            [],
            GripperConfig(grip_command=0.05, f_normal_max=1.0e9),
        )
    if profile == VelocityProfile.SEAL_SWING_BREAK_XY:
        # like SEAL_SWING_BREAK_X, but the swing and spike are about x and y together (peel about both).
        swing_slow = wp.spatial_vector(0.0, 0.0, 0.0, 4.0, 4.0, 0.0)
        swing_spike = wp.spatial_vector(0.0, 0.0, 0.0, 30.0, 30.0, 0.0)
        return (
            [0.0, 0.3, 1.3, 1.6, 3.0, 3.1, 3.2, 5.5],
            [rest, up, up, swing_slow, swing_slow, swing_spike, rest, rest],
            [],
            [],
            GripperConfig(grip_command=0.05, f_normal_max=1.0e9),
        )
    raise ValueError(f"unknown velocity profile: {profile}")


class _SelectProfile(argparse.Action):
    """argparse action: selecting ``--profile`` also derives ``num_frames`` from its end time.

    ``--profile ALL`` plays every profile back-to-back, so num_frames is the sum over all profiles.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if values == ALL_PROFILES:
            setattr(namespace, self.dest, ALL_PROFILES)
            namespace.num_frames = sum(round(make_profile(p)[0][-1] * FPS) for p in VelocityProfile)
            return
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
            choices=[p.name for p in VelocityProfile] + [ALL_PROFILES],
            action=_SelectProfile,
            help="Velocity profile driving the gripper (its end time sets the sim length); ALL plays every profile in turn.",
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

        # --profile selects one profile, or ALL to play every profile back-to-back (a demo reel that
        # rebuilds the scene between profiles so B is re-gripped for each). num_frames is the single
        # profile's frame count, or the sum over all profiles for ALL. The physics timestep comes
        # from --sim-dt; sim_substeps is chosen so an integer number of substeps fills each frame.
        selected = getattr(args, "profile", DEFAULT_PROFILE)
        self.profiles = list(VelocityProfile) if selected == ALL_PROFILES else [selected]
        default_frames = sum(round(make_profile(p)[0][-1] * self.fps) for p in self.profiles)
        self.num_frames = getattr(args, "num_frames", default_frames)
        requested_sim_dt = getattr(args, "sim_dt", DEFAULT_SIM_DT)
        self.sim_substeps = max(1, round(self.frame_dt / requested_sim_dt))
        self.sim_dt = self.frame_dt / self.sim_substeps

        # play the first profile; step() advances to the next once a profile's frames are exhausted.
        self.profile_index = 0
        self.profile_frame = 0
        self._load_profile(self.profiles[0])

    def _load_profile(self, profile):
        """(Re)build the scene and gripper for ``profile``, reset its clock, and (re)capture the graph.

        Called at start-up, and in ALL mode at each profile boundary (so box B is re-gripped for the
        next profile). ``sim_time_wp`` (the per-profile velocity clock) is reset; ``self.sim_time``
        stays monotonic for the viewer.
        """
        self.velocity_profile = profile
        vel_times, vel_twists, seal_times, seal_values, gripper_cfg = make_profile(profile)
        self.profile_num_frames = round(vel_times[-1] * self.fps)

        # --- scene constants (shared by every profile) ---
        # geometry: box half-extents. K is the kinematic controller, A the gripper body K drives, B
        # the gripped free body. start_height_b is B's initial clearance above the ground; it must
        # exceed a profile's largest downward excursion (e.g. UP_LEFT_DOWN lowers 1 m) so B never
        # hits the ground while still sealed.
        LK = 0.1
        LA = 0.1
        LB = 0.2
        start_height_b = 1.5
        m_k, m_a, m_b = 0.5, 0.5, 1.0  # masses; K is kinematic, so its mass is nominal
        gravity = 10.0
        # PD gains for the 6-DOF drive coupling kinematic body K to gripper body A (target 0 = A at
        # its rest offset below K). Armature adds artificial inertia so the stiff gains give a low,
        # timestep-resolvable natural frequency (the gripper's own inertia is tiny). Tunable.
        drive_ke_lin = 8000.0
        drive_kd_lin = 300.0
        drive_armature_lin = 1.0
        drive_ke_ang = 500.0
        drive_kd_ang = 30.0
        drive_armature_ang = 0.2

        # Scene: the velocity profile drives kinematic body K; a 6-DOF PD drive couples K to dynamic
        # gripper body A so A follows K compliantly; the gripper on A seals onto free box B.
        builder = self._create_builder(
            LK,
            LA,
            LB,
            m_k,
            m_a,
            m_b,
            gravity,
            start_height_b,
            drive_ke_lin,
            drive_kd_lin,
            drive_armature_lin,
            drive_ke_ang,
            drive_kd_ang,
            drive_armature_ang,
        )
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model, use_mujoco_contacts=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Create the gripper model from the profile's GripperConfig (defaults unless the profile
        # overrides them -- e.g. physics-break profiles weaken the grip and/or raise f_normal_max).
        gbuilder = self._create_gripper_builder(LA, gripper_cfg)
        self.gripper_model = gbuilder.finalize(device=self.model.device)
        self.gripper_state = self.gripper_model.state()
        self.gripper_control = self.gripper_model.control()
        self.gripper_control.pad_grip_control.fill_(gripper_cfg.grip_command)

        # Phase-1 seal decision: which body each pad seals against (fixed to box B here).
        self.seal_engaged = wp.full(1, True, dtype=wp.bool, device=self.model.device)
        self.seal_body_b = wp.full(1, self.box_b, dtype=wp.int32, device=self.model.device)

        # velocity time series driving kinematic body K, from the selected profile.
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

        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)  # per-profile velocity clock
        self.box_k_q_start = int(self.model.joint_q_start.numpy()[self.box_k_joint])
        self.box_k_qd_start = int(self.model.joint_qd_start.numpy()[self.box_k_joint])

        self.viewer.set_model(self.model)
        self.capture()

    def _create_builder(
        self,
        LK,
        LA,
        LB,
        m_k,
        m_a,
        m_b,
        gravity,
        start_height_b,
        drive_ke_lin,
        drive_kd_lin,
        drive_armature_lin,
        drive_ke_ang,
        drive_kd_ang,
        drive_armature_ang,
    ):
        """Build the scene: kinematic body K drives dynamic gripper body A through a 6-DOF PD drive;
        A's gripper seals onto free box B, above a ground plane. Returns the builder."""
        builder = newton.ModelBuilder(gravity=-gravity)
        # enable the per-shape MuJoCo condim attribute (used below to make the A-B contact frictionless)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # zero-density box shapes so the builder honours the explicit mass/inertia passed to
        # add_link/add_body; a nonzero-density shape would add its own computed mass on top.
        box_cfg = builder.default_shape_cfg.copy()
        box_cfg.density = 0.0

        def box_inertia(mass, half):
            s = (1.0 / 6.0) * mass * (2.0 * half) ** 2  # solid cube about its centre
            return wp.diag(wp.vec3(s, s, s))

        # box B (free, dynamic) starts start_height_b above the ground; gripper body A sits on top
        # with its pad sealing B's top face (A_center = B_center + LA + LB). Kinematic body K is a
        # visible box a small gap above A; the drive's rest state (target 0) holds A drive_gap below K.
        drive_gap = LK + LA + 0.05
        pose_b = wp.transform(wp.vec3(0.0, 0.0, start_height_b + LB), wp.quat_identity())
        pose_a = wp.transform(wp.vec3(0.0, 0.0, start_height_b + LB + LA + LB), wp.quat_identity())
        pose_k = wp.transform(wp.vec3(0.0, 0.0, start_height_b + LB + LA + LB + drive_gap), wp.quat_identity())

        # K: the kinematic controller -- a box driven by the velocity profile through a free joint to
        # the world (we write its joint_qd every substep). It ignores forces and does not collide.
        self.box_k = builder.add_link(
            xform=pose_k, mass=m_k, inertia=box_inertia(m_k, LK), is_kinematic=True, label="kinematic_K"
        )
        kinematic_cfg = box_cfg.copy()
        kinematic_cfg.has_shape_collision = False  # visual only; never collides
        builder.add_shape_box(self.box_k, hx=LK, hy=LK, hz=LK, cfg=kinematic_cfg)
        self.box_k_joint = builder.add_joint_free(child=self.box_k, label="kinematic_free")

        # A: the dynamic gripper body carrying the surface gripper. A 6-DOF D6 joint with per-axis PD
        # drives (target 0) couples A to K, so A follows K but seal loads deflect it against the drive
        # springs. Armature adds artificial inertia so the stiff gains stay timestep-stable. The joint
        # anchor is offset drive_gap down from K so the rest pose puts A at pose_a.
        self.box_a = builder.add_link(xform=pose_a, mass=m_a, inertia=box_inertia(m_a, LA), label="gripper_box_A")
        shape_a = builder.add_shape_box(self.box_a, hx=LA, hy=LA, hz=LA, cfg=box_cfg)
        JointDof = newton.ModelBuilder.JointDofConfig
        axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self.drive_joint = builder.add_joint_d6(
            parent=self.box_k,
            child=self.box_a,
            linear_axes=[
                JointDof(axis=ax, target_ke=drive_ke_lin, target_kd=drive_kd_lin, armature=drive_armature_lin)
                for ax in axes
            ],
            angular_axes=[
                JointDof(axis=ax, target_ke=drive_ke_ang, target_kd=drive_kd_ang, armature=drive_armature_ang)
                for ax in axes
            ],
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -drive_gap), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            label="gripper_drive",
        )
        builder.add_articulation([self.box_k_joint, self.drive_joint], label="gripper_articulation")

        # box B is the gripped object: a free body pulled up by the suction seal (no joint drive).
        self.box_b = builder.add_body(xform=pose_b, mass=m_b, inertia=box_inertia(m_b, LB), label="gripped_box_B")
        shape_b = builder.add_shape_box(self.box_b, hx=LB, hy=LB, hz=LB, cfg=box_cfg)

        # Make the A-B contact frictionless: condim=1 (normal only, no friction dimension) on both
        # boxes, so the contact carries no tangential force -- the seal's shear DOF is the sole
        # tangential model (a condim=3 contact would double-count it). condim=1 is stable, whereas
        # mu=0 on a condim=3 contact NaNs (degenerate friction cone). condim combines as max, so
        # B-ground stays frictional: max(1, ground=3) = 3.
        condim = builder.custom_attributes["mujoco:condim"]
        if condim.values is None:
            condim.values = {}
        condim.values[shape_a] = 1
        condim.values[shape_b] = 1

        builder.add_ground_plane()
        return builder

    def _create_gripper_builder(self, LA, cfg):
        """Build the single-pad surface gripper on box A's bottom (-z) face from ``cfg``
        (:class:`GripperConfig`); return its builder."""
        gripper = SurfaceGripper(
            body_id=self.box_a,
            xform=wp.transform_identity(),  # gripper frame == box A body frame
            k_normal=cfg.k_normal,
            d_normal=cfg.d_normal,
            f_normal_max=cfg.f_normal_max,
            f_grip_max=cfg.f_grip_max,
            k_shear_x=cfg.k_shear_x,
            k_shear_y=cfg.k_shear_y,
            mu_x=cfg.mu_x,
            mu_y=cfg.mu_y,
            d_peel_x=cfg.d_peel_x,
            d_peel_y=cfg.d_peel_y,
            shape=int(cfg.pad_shape),
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

            # kinematically drive body K at its prescribed 6-DOF velocity, then propagate the new
            # joint state to the kinematic body's maximal coordinates (A follows via the drive).
            wp.launch(
                apply_twist_at_current_time,
                dim=1,
                inputs=[
                    self.vel_times,
                    self.vel_twists,
                    self.num_vel,
                    self.sim_dt,
                    self.box_k_q_start,
                    self.box_k_qd_start,
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
        self.profile_frame += 1
        # ALL mode: once the current profile's frames are exhausted, rebuild the scene for the next.
        if self.profile_frame >= self.profile_num_frames and self.profile_index + 1 < len(self.profiles):
            self.profile_index += 1
            self.profile_frame = 0
            self._load_profile(self.profiles[self.profile_index])

    def test_final(self):
        # Default profile UP_LEFT_ROLL_DOWN: A lifts 1 m, shifts 1 m left (-y) while rolling 90 deg
        # about +x, then lowers 1 m, ending ~1 m left near the start height (box B swings to sit
        # beside A as the seal rolls). Sanity bounds only: both boxes stay near x=0, end shifted
        # along -y, and don't fly off or hit the ground. The per-step seal check lives in
        # test_surface_gripper.
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "gripper A carries box B up, left with a roll, then down",
            lambda q, qd: 0.1 < q[2] < 4.0 and abs(q[0]) < 0.2 and -1.6 < q[1] < -0.4,
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
