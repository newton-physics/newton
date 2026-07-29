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
# A single kinematic body K drives up to four surface grippers (bodies A[i]) arranged at the corners
# of a square, each coupled to K by a 6-DOF PD drive so it follows K compliantly. Each gripper seals
# onto its own free box B[i] resting on the ground. K's full 6-DOF motion is prescribed by a velocity
# time series (one of the named VelocityProfile options, selected with --profile), lifting the boxes
# off the ground and exercising the surface-gripper hold wrench in tension. --num-pads (1-4) selects
# how many pads start engaged; the rest leave their box on the ground. The seal is always physics-
# computed (a pad stays engaged until its brittle break metric exceeds 1); the force model lives in
# surface_gripper.py.
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
    attach_seal,
    evaluate_gripper_force,
    evaluate_seal,
)

FPS = 60  # fixed render frame rate; the run length (num_frames) is derived from the profile
DEFAULT_SIM_DT = 1.0 / 240.0  # default physics timestep [s]


class VelocityProfile(Enum):
    """Named velocity time series for the kinematic gripper (box A).

    Select one with ``--profile`` (or pass it via ``args`` in a test). The profile's end time
    sets the total simulation length. See :func:`make_velocity_profile` for the keyframes.
    """

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
    all 6 seal DOF active: normal, shear (x, y), peel (x, y) and twist. Every profile commands full
    suction (``grip_command = 1``), so the preload is just ``f_min = f_grip_max`` -- ``f_grip_max`` is
    the literal preload force. Since the seal is always physics-computed, the default keeps
    ``f_grip_max`` below ``f_normal_max`` so the seal reads intact at rest; break profiles weaken the
    cup (lower ``f_grip_max``) and/or the break threshold (lower ``f_normal_max``, or raise it to a
    huge value to zero out the normal term of the break metric and isolate peel).
    """

    k_normal: float = 5000.0  # normal spring stiffness [N/m]
    d_normal: float = 100.0  # normal damping [N.s/m]
    f_normal_max: float = 200.0  # break-threshold normal force [N]
    f_grip_max: float = 150.0  # suction preload [N] (= f_min, since grip_command = 1); kept < f_normal_max
    k_shear_x: float = 2000.0  # shear stiffness about x [N/m]
    k_shear_y: float = 2000.0  # shear stiffness about y [N/m]
    mu_x: float = 1.0  # shear/twist friction coefficient (x)
    mu_y: float = 1.0  # shear/twist friction coefficient (y)
    d_peel_x: float = 10.0  # peel damping about x [N.m.s/rad]
    d_peel_y: float = 10.0  # peel damping about y [N.m.s/rad]
    pad_shape: PadShape = PadShape.RECTANGLE  # pad cross-section
    grip_command: float = 1.0  # constant grip command in [0, 1]; f_min = grip_command * f_grip_max


def make_profile(profile: VelocityProfile):
    """Return ``(vel_times, vel_twists, gripper)`` for ``profile``.

    ``vel_*`` is a 6-DOF velocity time series (world-frame ``(linear, angular)`` twists, sampled by
    piecewise-linear interpolation). ``gripper`` is the :class:`GripperConfig` for this profile. The
    seal is always physics-computed (:func:`evaluate_seal`): each pad stays engaged until its break
    metric exceeds 1, so aggressive profiles peel/yank the seal off on their own. ``vel_times[-1]``
    is the profile's end time and sets the sim length (``num_frames = round(end_time * FPS)``).
    """
    gripper = GripperConfig()  # default seal; break profiles below override specific values
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
    if profile == VelocityProfile.UP_LEFT_DOWN:
        # lift 1 m (+z), shift 1 m left (-y), then lower 1 m (-z), all at 0.8 m/s with 0.2 s ramps
        # between legs and a final 0.2 s ramp to rest. Flat times (1.15/1.05/1.05 s) plus each leg's
        # share of the ramps make every leg exactly 1 m. Ending at rest holds A in place afterwards
        # (the sampler clamps past the end to the last twist, so a trailing 'down' would keep dragging
        # the payload down if playback runs past the profile).
        return (
            [0.0, 1.15, 1.35, 2.40, 2.60, 3.65, 3.85],
            [up, up, left, left, down, down, rest],
            gripper,
        )
    if profile == VelocityProfile.UP_LEFT_ROLL_DOWN:
        # like UP_LEFT_DOWN (1 m legs at 0.8 m/s, 0.2 s ramps), but each leg adds a rotation: twist
        # about z on the way up, swing about y while shifting 1 m left, then swing about x plus twist
        # about z on the way down. Rotations ramp in/out with their leg (~90 deg each), exercising the
        # seal's twist and swing/peel resistance.
        return (
            [0.0, 1.15, 1.35, 2.40, 2.60, 3.65, 3.85],
            [up_twist, up_twist, left_swing_y, left_swing_y, down_swing_x_twist, down_swing_x_twist, rest],
            gripper,
        )
    if profile == VelocityProfile.SEAL_YANK_BREAK:
        # lift to establish the grip, then a brief hard yank straight up and back to rest. The yank's
        # high acceleration drives the payload's normal tension past the break threshold, the break
        # metric exceeds 1 and the seal releases on its own; the payload is flung up, detaches, and
        # falls to the ground while the gripper holds still. Full grip (f_min = f_grip_max) but a weak
        # cup: f_grip_max stays below f_normal_max so the seal holds at rest, and the low f_normal_max
        # is what the yank overcomes.
        return (
            [0.0, 0.3, 1.3, 1.4, 1.5, 4.0],
            [rest, up, up, yank_up, rest, rest],
            GripperConfig(f_grip_max=100.0, f_normal_max=150.0),
        )
    if profile == VelocityProfile.SEAL_SWING_BREAK_X:
        # Peel break by swinging (rolling) about x. Lift to establish the grip, then swing slowly
        # about x for ~1.4 s -- the payload follows, so the gripper visibly rolls it through most of a
        # turn. Then a brief hard swing spike: the high angular acceleration makes it lag rotationally,
        # the elastic peel moment builds past its capacity, the break metric exceeds 1, and the seal
        # releases. Full grip but a weak cup (low f_grip_max) so the peel capacity, which scales with
        # the holding force, is small; f_normal_max is sky-high so the normal term of the break metric
        # is ~0, proving the break is purely peel.
        swing_slow = wp.spatial_vector(0.0, 0.0, 0.0, 4.0, 0.0, 0.0)
        swing_spike = wp.spatial_vector(0.0, 0.0, 0.0, 30.0, 0.0, 0.0)
        return (
            [0.0, 0.3, 1.3, 1.6, 3.0, 3.1, 3.2, 5.5],
            [rest, up, up, swing_slow, swing_slow, swing_spike, rest, rest],
            GripperConfig(f_grip_max=50.0, f_normal_max=1.0e9),
        )
    if profile == VelocityProfile.SEAL_SWING_BREAK_Y:
        # like SEAL_SWING_BREAK_X, but the slow swing and the break spike are about y (peel about y).
        swing_slow = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 4.0, 0.0)
        swing_spike = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 30.0, 0.0)
        return (
            [0.0, 0.3, 1.3, 1.6, 3.0, 3.1, 3.2, 5.5],
            [rest, up, up, swing_slow, swing_slow, swing_spike, rest, rest],
            GripperConfig(f_grip_max=50.0, f_normal_max=1.0e9),
        )
    if profile == VelocityProfile.SEAL_SWING_BREAK_XY:
        # like SEAL_SWING_BREAK_X, but the swing and spike are about x and y together (peel about both).
        swing_slow = wp.spatial_vector(0.0, 0.0, 0.0, 4.0, 4.0, 0.0)
        swing_spike = wp.spatial_vector(0.0, 0.0, 0.0, 30.0, 30.0, 0.0)
        return (
            [0.0, 0.3, 1.3, 1.6, 3.0, 3.1, 3.2, 5.5],
            [rest, up, up, swing_slow, swing_slow, swing_spike, rest, rest],
            GripperConfig(f_grip_max=50.0, f_normal_max=1.0e9),
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
        parser.add_argument(
            "--num-pads",
            type=int,
            default=1,
            choices=[1, 2, 3, 4],
            help="Number of suction pads that start engaged (1-4); the rest leave their box on the ground.",
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
        # ALL mode loops forever (an interactive demo reel): step() wraps back to the first profile
        # after the last. A single profile plays once and then holds.
        self.cycle_profiles = selected == ALL_PROFILES
        default_frames = sum(round(make_profile(p)[0][-1] * self.fps) for p in self.profiles)
        self.num_frames = getattr(args, "num_frames", default_frames)
        requested_sim_dt = getattr(args, "sim_dt", DEFAULT_SIM_DT)
        self.sim_substeps = max(1, round(self.frame_dt / requested_sim_dt))
        self.sim_dt = self.frame_dt / self.sim_substeps

        # --num-pads (1-4): how many of the four corner pads start engaged. The rest leave their box
        # on the ground. Used by _load_profile when it builds the per-pad seal_engaged array.
        self.num_pads = min(4, max(1, int(getattr(args, "num_pads", 1))))

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
        vel_times, vel_twists, gripper_cfg = make_profile(profile)
        self.profile_num_frames = round(vel_times[-1] * self.fps)

        # --- scene constants (shared by every profile) ---
        # geometry: box half-extents. K is the single kinematic controller; A[i] are the four gripper
        # bodies K drives (one per corner of a square of half-side `spacing`); B[i] are the four
        # gripped free boxes, one under each gripper, resting on the ground. --num-pads selects how
        # many of the four pads start engaged (the rest leave their box on the ground).
        LK = 0.1
        LA = 0.1
        LB = 0.2
        spacing = 0.4  # half-side of the square the four grippers sit on [m]
        m_k, m_a, m_b = 0.5, 0.5, 1.0  # masses; K is kinematic, so its mass is nominal
        gravity = 10.0
        # PD gains for the 6-DOF drives coupling kinematic body K to each gripper body A[i] (target 0
        # = A at its rest offset below K). Armature adds artificial inertia so the stiff gains give a
        # low, timestep-resolvable natural frequency (the gripper's own inertia is tiny). Tunable.
        drive_ke_lin = 8000.0
        drive_kd_lin = 300.0
        drive_armature_lin = 1.0
        drive_ke_ang = 500.0
        drive_kd_ang = 30.0
        drive_armature_ang = 0.2

        # Scene: the velocity profile drives kinematic body K; a 6-DOF PD drive couples K to each
        # dynamic gripper body A[i] so it follows K compliantly; the gripper on each A[i] seals onto
        # its free box B[i].
        builder = self._create_builder(
            LK,
            LA,
            LB,
            spacing,
            m_k,
            m_a,
            m_b,
            gravity,
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

        # Seal decision: pad i seals against box B[i]. The first --num-pads pads start engaged, the
        # rest start (and stay) disengaged so their box is left on the ground. Engagement is always
        # physics-driven (evaluate_seal): an engaged pad stays engaged until its break metric > 1.
        engaged = [i < self.num_pads for i in range(len(self.box_a_list))]
        self.seal_engaged = wp.array(engaged, dtype=wp.bool, device=self.model.device)
        self.seal_body_b = wp.array(self.box_b_list, dtype=wp.int32, device=self.model.device)

        # velocity time series driving kinematic body K, from the selected profile.
        self.vel_times = wp.array(vel_times, dtype=wp.float32, device=self.model.device)
        self.vel_twists = wp.array(vel_twists, dtype=wp.spatial_vector, device=self.model.device)
        self.num_vel = len(vel_times)

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
        spacing,
        m_k,
        m_a,
        m_b,
        gravity,
        drive_ke_lin,
        drive_kd_lin,
        drive_armature_lin,
        drive_ke_ang,
        drive_kd_ang,
        drive_armature_ang,
    ):
        """Build the 4-corner rig: kinematic body K drives four dynamic gripper bodies A[i] (one per
        corner of a square of half-side ``spacing``) through 6-DOF PD drives; each gripper seals onto
        a free box B[i] resting on the ground plane. Returns the builder. All four grippers and boxes
        are always built; ``--num-pads`` only selects how many pads start engaged."""
        builder = newton.ModelBuilder(gravity=-gravity)
        # enable the per-shape MuJoCo condim attribute (used below to make each A-B contact frictionless)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # zero-density box shapes so the builder honours the explicit mass/inertia passed to
        # add_link/add_body; a nonzero-density shape would add its own computed mass on top.
        box_cfg = builder.default_shape_cfg.copy()
        box_cfg.density = 0.0

        def box_inertia(mass, half):
            s = (1.0 / 6.0) * mass * (2.0 * half) ** 2  # solid cube about its centre
            return wp.diag(wp.vec3(s, s, s))

        # z-stack (ground at z=0): each box B rests on the ground, its gripper body A sits on top of
        # it (A's pad sealing B's top face, A_center = B_center + LA + LB), and the shared kinematic
        # body K sits a small drive_gap above the A layer. The drive rest state holds each A at
        # drive_gap below K.
        drive_gap = LK + LA + 0.05
        z_b = LB  # box bottom on the ground
        z_a = 2.0 * LB + LA  # A's bottom face flush with B's top
        z_k = z_a + drive_gap

        # Four grippers at the corners of a square. Right column (x > 0) first so --num-pads selects
        # whole x-columns; the default -y shift then never drags an active box over an inactive one in
        # the same column (they are in different x-columns).
        self.corners = [(spacing, spacing), (spacing, -spacing), (-spacing, spacing), (-spacing, -spacing)]

        # K: the kinematic controller -- driven by the velocity profile through a free joint to the
        # world (we write its joint_qd every substep). It ignores forces and does not collide. Its
        # shape is a thin plate spanning all four corner pads (like a manifold the pads hang from):
        # half-extents reach the outer edge of the corner grippers (spacing + LA) and are thin in z.
        plate_hxy = spacing + LA
        plate_hz = 0.02
        pose_k = wp.transform(wp.vec3(0.0, 0.0, z_k), wp.quat_identity())
        self.box_k = builder.add_link(
            xform=pose_k, mass=m_k, inertia=box_inertia(m_k, LK), is_kinematic=True, label="kinematic_K"
        )
        kinematic_cfg = box_cfg.copy()
        kinematic_cfg.has_shape_collision = False  # visual only; never collides
        builder.add_shape_box(self.box_k, hx=plate_hxy, hy=plate_hxy, hz=plate_hz, cfg=kinematic_cfg)
        self.box_k_joint = builder.add_joint_free(child=self.box_k, label="kinematic_free")

        JointDof = newton.ModelBuilder.JointDofConfig
        axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        condim = builder.custom_attributes["mujoco:condim"]
        if condim.values is None:
            condim.values = {}

        # First pass: build the articulation (K's free joint plus the four drive joints). The
        # articulation's joints must be contiguous, so create every A[i] and its drive here, before
        # adding the free boxes B[i] (each of which adds its own implicit free joint).
        self.box_a_list = []
        shape_a_list = []
        drive_joints = []
        for i, (cx, cy) in enumerate(self.corners):
            # A[i]: dynamic gripper body carrying the surface gripper. A 6-DOF D6 joint with per-axis
            # PD drives (target 0) couples it to K, so A follows K but seal loads deflect it against
            # the drive springs. Armature keeps the stiff gains timestep-stable. The joint anchor is
            # offset (cx, cy, -drive_gap) from K so the rest pose puts A[i] at this corner.
            pose_a = wp.transform(wp.vec3(cx, cy, z_a), wp.quat_identity())
            box_a = builder.add_link(xform=pose_a, mass=m_a, inertia=box_inertia(m_a, LA), label=f"gripper_box_A{i}")
            shape_a = builder.add_shape_box(box_a, hx=LA, hy=LA, hz=LA, cfg=box_cfg)
            drive_joint = builder.add_joint_d6(
                parent=self.box_k,
                child=box_a,
                linear_axes=[
                    JointDof(axis=ax, target_ke=drive_ke_lin, target_kd=drive_kd_lin, armature=drive_armature_lin)
                    for ax in axes
                ],
                angular_axes=[
                    JointDof(axis=ax, target_ke=drive_ke_ang, target_kd=drive_kd_ang, armature=drive_armature_ang)
                    for ax in axes
                ],
                parent_xform=wp.transform(wp.vec3(cx, cy, -drive_gap), wp.quat_identity()),
                child_xform=wp.transform_identity(),
                label=f"gripper_drive_{i}",
            )
            drive_joints.append(drive_joint)
            self.box_a_list.append(box_a)
            shape_a_list.append(shape_a)
        builder.add_articulation([self.box_k_joint, *drive_joints], label="gripper_articulation")

        # Second pass: the gripped boxes B[i] -- free bodies resting on the ground, pulled up by the
        # suction seal (no joint drive).
        self.box_b_list = []
        for i, (cx, cy) in enumerate(self.corners):
            pose_b = wp.transform(wp.vec3(cx, cy, z_b), wp.quat_identity())
            box_b = builder.add_body(xform=pose_b, mass=m_b, inertia=box_inertia(m_b, LB), label=f"gripped_box_B{i}")
            shape_b = builder.add_shape_box(box_b, hx=LB, hy=LB, hz=LB, cfg=box_cfg)

            # Make each A-B contact frictionless: condim=1 (normal only, no friction dimension) on both
            # boxes, so the contact carries no tangential force -- the seal's shear DOF is the sole
            # tangential model (a condim=3 contact would double-count it). condim=1 is stable, whereas
            # mu=0 on a condim=3 contact NaNs (degenerate friction cone). condim combines as max, so
            # B-ground stays frictional: max(1, ground=3) = 3.
            condim.values[shape_a_list[i]] = 1
            condim.values[shape_b] = 1
            self.box_b_list.append(box_b)

        builder.add_ground_plane()
        return builder

    def _create_gripper_builder(self, LA, cfg):
        """Build four single-pad surface grippers (one per corner body A[i]), each with its pad on
        the body's bottom (-z) face, from ``cfg`` (:class:`GripperConfig`); return the builder."""
        gbuilder = SurfaceGripperBuilder()
        for box_a in self.box_a_list:
            gripper = SurfaceGripper(
                body_id=box_a,
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
            gripper.add_pad(
                wp.transform(wp.vec3(0.0, 0.0, -LA), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi))
            )
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
            # Seal engagement is always physics-driven: an engaged pad stays engaged until its break
            # metric exceeds 1, then releases; a disengaged pad stays disengaged.
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
            attach_seal(self.state_0, self.gripper_model, self.gripper_state, self.seal_engaged, self.seal_body_b)
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
        # Once the current profile's frames are exhausted, advance to the next and rebuild the scene
        # (re-gripping the boxes). In ALL mode this wraps back to the first profile so the demo reel
        # loops forever; a single profile has nothing to advance to and just holds at its end state.
        if self.profile_frame >= self.profile_num_frames:
            if self.profile_index + 1 < len(self.profiles):
                self.profile_index += 1
            elif self.cycle_profiles:
                self.profile_index = 0
            else:
                return
            self.profile_frame = 0
            self._load_profile(self.profiles[self.profile_index])

    def test_final(self):
        # Default profile UP_LEFT_ROLL_DOWN: K (and each gripper A[i]) lifts 1 m, shifts 1 m left
        # (-y) while rolling 90 deg about +x, then lowers 1 m -- a pick-and-place that carries each
        # engaged box up and sets it back down ~1 m to the -y. Sanity bounds only (the per-pad seal
        # check lives in test_surface_gripper): every gripped box stays within the swept region and
        # neither flies off nor falls through the ground.
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "each engaged gripper carries its box up, left with a roll, then down",
            lambda q, qd: 0.05 < q[2] < 4.0 and abs(q[0]) < 1.0 and -1.8 < q[1] < 0.8,
            self.box_b_list,
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
