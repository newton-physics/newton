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
# Example Suction Cup Isaac Sim Repro
#
# Reproduction scene for the suction-cup gripper on a robot arm. Loads the robot arm from a USD stage
# (Assets/robot_only_newton_flattened.usda) with a fixed base on a ground plane, then plays back a
# recorded FANUC palletizer cycle (Assets/robot_recording_truncated.jsonl -- the leading idle removed
# from robot_recording.jsonl so the arm moves right away). Playback is time-accurate: the six
# arm joint position targets are interpolated from the recorded timestamps at the current simulation
# time (J3 coupled to J2, degrees -> radians) and updated before every physics sub-step, so the arm
# follows the recording at its true speed. The recording's suction-cup engagement command (ro[0]) is
# extracted per frame; the suction gripper itself is wired up and added in later steps.
#
# Command: python -m newton.examples suction_cup_isaac_sim_repro
###########################################################################

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.suctioncup.robot_playback import load_recording, recorded_times
from newton.examples.suctioncup.surface_gripper import (
    PadShape,
    SurfaceGripper,
    SurfaceGripperBuilder,
    evaluate_gripper_force,
    latch_engagement,
)

# assets live alongside this example
ASSETS = Path(__file__).parent / "Assets"
# robot USD with convex-hull collision added to the suction-gripper (EOAT) meshes
ROBOT_USD = ASSETS / "fanuc_arm_flattened_collision.usda"
# recording with the leading idle removed (truncated from robot_recording.jsonl); it starts just
# before the first joint motion, so the arm moves right away.
RECORDING_JSONL = ASSETS / "robot_recording_truncated.jsonl"

# Gaussian smoothing of the recorded drive targets [s]. The recording is a coarse waypoint staircase
# (values held, then stepped ~17 deg), so smoothing recovers a continuous motion. 0 = raw recording;
# larger = smoother (and further from the exact recorded knots). ~1 waypoint interval (~0.08 s) is a
# reasonable start.
SMOOTHING_SIGMA = 0.06

FPS = 60  # rendered frames per second
SIM_HZ = 240  # target physics rate; sim_substeps = SIM_HZ / FPS physics steps per render frame
NUM_ARM_DOFS = 6  # J1-J6; recorded joints 6-8 are unused finger DOFs
BOX_HALF = 0.5  # half-extent of the static support box (size [1, 1, 1]) [m]
# Two pick boxes to choose between; set PICK_BOX to 0 or 1. The gripper parameters are fixed (a
# statement of the tool -- only the payload changes), so this exercises whether the one real gripper
# holds different boxes. The box + pallet placement is auto-derived by FK, so it adapts to each size.
PICK_BOX = 1  # 0: the crate (deep); 1: a wide, shallow panel
_PICK_BOXES = (
    ((0.242, 0.166, 0.137), 12.0),  # 0: crate, size [0.484, 0.332, 0.274] m, 12 kg
    ((0.5, 0.5, 0.04), 30.0),  # 1: wide shallow panel, size [1.0, 1.0, 0.08] m, 30 kg
)
PICK_BOX_HALF, PICK_BOX_MASS = _PICK_BOXES[PICK_BOX]  # half-extents [m], mass [kg] (shape density is 0)

# Deepest reach of the finger collision geometry along the suction axis, in the J6_link (flange)
# frame [m] -- the point that would first penetrate the box. The box top is seated here so the fingers
# rest on the box without sinking in. Resolved from the USD (max +x over the Finger_0x meshes).
FINGER_HULL_DEEPEST_X = 0.3109

# The pick box and its pallet are NOT hard-coded: their positions are derived at build time from the
# flange pose at the first-engagement time (forward kinematics), so the box always seats itself at the
# cups regardless of SMOOTHING_SIGMA, the cup geometry, or the recording. See Example.__init__.

# Set False to disable the suction cup: the seal wrench is never applied, so the arm plays back the
# recorded trajectory and the pick box just sits on the pallet (useful for inspecting the bare arm
# motion). Read at graph-capture time, so set it before constructing the example.
ENABLE_GRIPPER = True

# Enable the pick-box <-> gripper-geometry rigid contact. On: the box presses flush against the tool
# and tipping is resisted by contact (matches a real vacuum grip on rigid tooling); pair with a
# tension-only seal (surface_gripper.SEAL_TENSION_ONLY). Off: the soft seal alone owns the hold.
ENABLE_PAD_BOX_CONTACT = False

# Draw a small non-colliding disk at each suction cup (GRIPPER_PADS) so the cup layout is visible in
# the viewer. Purely visual (has_shape_collision off); does not affect the physics.
SHOW_CUP_MARKERS = True

# Set False to disable the debug CSV recording -- the end-effector acceleration and the smoothed
# runtime drive targets (see EndEffectorAccelerationRecorder / DriveTargetRecorder). Recording is
# host-side, so it only takes effect on CPU regardless.
RECORD_DEBUG = False

# Seal fractures (releases) once its brittle break metric exceeds this. 1.0 = nominal capacity; a
# value > 1 is a capacity safety factor (the seal tolerates sqrt(threshold)x the nominal elastic peel
# before breaking). Set to 5 (~2.2x): the wide panel (box 1) overhangs the cups, so the arm's
# reorientations spike its peel to ~2x nominal -- the holding force still carries it, so a strict 1.0
# would drop it ~60 frames before the recorded release. The crate (box 0) stays far below either way.
BREAK_THRESHOLD = 5.0

# The break metric must stay over BREAK_THRESHOLD for at least this long before the seal fractures.
# Debounces lone transient spikes (a genuine overload is sustained), so a held box is not dropped by a
# brief sub-step spike. Expressed as a time so it is independent of the sim rate; the sub-step count is
# round(BREAK_HOLD_TIME / sim_dt), floored at 1.
BREAK_HOLD_TIME = 0.033  # [s]

# Suction gripper on the end-effector (body EE_BODY / J6_link). Four cups at the suction vents at the
# tips of Finger_01..04, resolved into the J6_link (flange) frame from the USD. Each finger is an
# L-shaped arm whose vent faces the box; the vents are wide-set at the crate edges (a cross ~+/-6 cm on
# one axis, ~+/-13 cm on the other), giving the tilt leverage a real palletizer needs. x=0.309 is the
# vent (suction-face) plane. The suction axis is the flange +x (world-down at the pick), so each pad's
# local +z is rotated onto +x. Positions in the EE body frame [m].
GRIPPER_PADS = (
    (0.2880, 0.1286, 0.0035),
    (0.2885, -0.0024, -0.2022),
    (0.2880, -0.1323, 0.0035),
    (0.2886, -0.0025, 0.2085),
)


@dataclass(frozen=True)
class GripperParams:
    """Per-pad suction-seal tuning (see :class:`~newton.examples.suctioncup.surface_gripper.SurfaceGripper`).

    Field-for-field the ``SurfaceGripper`` keyword arguments except the runtime ``body_id`` / ``xform``,
    so it unpacks straight into the constructor (``SurfaceGripper(body_id=..., xform=..., **asdict(...))``).
    """

    k_normal: float  # normal stiffness [N/m]
    d_normal: float  # normal damping [N.s/m]
    f_normal_max: float  # per-pad break threshold [N]
    f_grip_max: float  # per-pad suction preload [N]
    k_shear_x: float  # shear stiffness [N/m]
    k_shear_y: float
    mu_x: float  # shear friction coefficient
    mu_y: float
    d_peel_x: float  # peel damping [N.m.s/rad]
    d_peel_y: float
    shape: int  # PadShape
    dim_a: float  # pad radius (CIRCLE) [m]
    dim_b: float
    d_shear_x: float = 0.0  # shear damping [N.s/m]; not in gripper.pdf, kept at 0
    d_shear_y: float = 0.0
    peel_capacity_scale: float = 1.0  # multiplies the geometric peel capacity (peel-limited lifts)


# Tuned for the light pick box (~1 kg, weight ~10 N). Preload ~= box weight so the box rests against
# the pads (constant contact); the break threshold is well above the carry loads so the seal holds.
# Damped springs so the four redundant pads settle, not ring. Stiff seal so the box tracks the flange
# rigidly (a soft seal lets it swing like a pendulum under the fast arm). The seal forces are applied
# explicitly, so with the small box (m = 1, I ~ 4e-3) at 240 Hz: near-critical damping keeps k stable
# up to ~m/dt^2, but the angular d_peel must stay tiny (dt < 2*I/d_peel) or it diverges. Seal
# stiffness is bounded by explicit stability at 240 Hz (omega*dt must stay well below 2, or the seal
# rings): k ~ 6000 tracks the box with ~mm lag while staying smooth.
GRIPPER_PARAMS = GripperParams(

    # Normal - translation - z
    k_normal=96000.0,  # stiff, like a vacuum cup on rigid tooling; sets the tilt tracking (peel angle ~0.2deg)
    d_normal=40.0,  # low: the wide-cup couple amplifies damping into the explicit-integrator limit at 240 Hz
    f_normal_max=2000.0,  # normal break threshold [N]; sized for the 30 kg panel's weight + lift accel
    f_grip_max=50.0,  # per-pad suction preload [N]; ~box weight / 4 so the panel rests on the pads

    # Shear - translation - x,y
    k_shear_x=6000.0,
    k_shear_y=6000.0,
    # High friction: when the arm holds the flange with the suction axis near-horizontal, the box
    # weight is a pure shear load, and shear capacity = mu * |holding force|. High mu keeps ample
    # margin through the arm's fast reorientation so the box doesn't slip and dangle.
    mu_x=16.0,
    mu_y=16.0,
    # Shear damping must stay small with the wide-set cups: the far cups' tangential velocity scales
    # with the cup spread, so d_shear's effective twist damping is amplified by ~Sigma r^2 (75x vs a
    # tight cluster) and overshoots the explicit-integrator limit at 240 Hz if raised much above ~30.
    d_shear_x=20.0,
    d_shear_y=20.0,

    # peel rotation- x,y. Small, for the same explicit-integrator reason as the shear/normal damping
    # above (the wide cup couple amplifies it); the wide-set cups need little peel damping to stay put.
    d_peel_x=0.5,
    d_peel_y=0.5,
    peel_capacity_scale=1.0,  # geometric capacity only; the real wide-set cups should hold without a fudge
    shape=int(PadShape.CIRCLE),
    dim_a=0.03,
    dim_b=0.03,
)


def arm_targets_rad(frame) -> np.ndarray:
    """Recorded :class:`Frame` -> the six arm joint position targets [rad].

    Takes J1-J6, applies the J3-relative-to-J2 coupling (real J3 = recorded J3 + J2), and converts
    degrees to radians. This is where the coupling/units are applied -- the recording stores raw.
    """
    j = list(frame.joints_deg[:NUM_ARM_DOFS])
    j[2] += j[1]  # J3 is recorded relative to J2
    return np.deg2rad(np.asarray(j, dtype=np.float32))


def load_playback(path):
    """Load a recording and extract the arrays the sim consumes.

    Returns ``(rec_times, rec_targets, rec_engaged, rec_duration)``:
        - ``rec_times``: sample times [s], shape [N], starting at 0 (:func:`recorded_times`).
        - ``rec_targets``: coupled arm joint targets [rad], shape [N, NUM_ARM_DOFS] (:func:`arm_targets_rad`).
        - ``rec_engaged``: suction-cup engagement command (robot output ro[0]) per frame, shape [N] bool.
        - ``rec_duration``: recording length [s] (``rec_times[-1]``).
    """
    frames = load_recording(path)
    rec_times = np.asarray(recorded_times(frames), dtype=np.float64)
    rec_targets = np.stack([arm_targets_rad(f) for f in frames]).astype(np.float64)  # [N, NUM_ARM_DOFS]
    rec_engaged = np.array([f.ro[0] for f in frames], dtype=bool)  # [N]
    return rec_times, rec_targets, rec_engaged, float(rec_times[-1])


def gaussian_smooth(times, values, sigma):
    """Gaussian-smooth ``values`` ([N, D]) over the non-uniform sample ``times`` ([N]).

    Each output sample is a Gaussian-weighted average of all samples by *time* distance
    (``w_ij = exp(-((t_i - t_j) / sigma)^2 / 2)``), so it correctly handles the non-uniform sample
    rate. ``sigma <= 0`` returns the input unchanged.
    """
    if sigma <= 0.0:
        return values
    dt = times[:, None] - times[None, :]  # [N, N] pairwise time differences [s]
    weights = np.exp(-0.5 * (dt / sigma) ** 2)  # Gaussian weights by time distance
    weights /= weights.sum(axis=1, keepdims=True)  # normalize per output sample
    return weights @ values  # [N, D] smoothed targets


@wp.kernel
def sample_playback_kernel(
    rec_times: wp.array[float],  # [N] recorded sample times [s], monotonic
    rec_targets: wp.array2d[float],  # [N, num_dofs] coupled arm targets [rad]
    rec_engaged: wp.array[wp.bool],  # [N] suction-cup engagement command (ro[0]) per frame
    sim_step_count: wp.array[
        int
    ],  # in/out: device sub-step counter (current time = sim_step_count[0] * dt); advanced in place
    last_lo: wp.array[int],  # in/out: cached lower sample index; the forward search resumes from here
    dt: float,  # physics sub-step [s]; sim time = sim_step_count * dt
    # outputs
    joint_target_q: wp.array[float],  # [num_dofs] interpolated position targets [rad]
    engaged: wp.array[wp.bool],  # [1] engagement command sampled at the current time
):
    """Interpolate the recorded joint position targets and sample the engagement command at the
    current time (one thread per DOF); advance the sub-step counter for the next sub-step.

    The time is the integer sub-step count times ``dt`` (exact, no float accumulation). Since sim time
    only advances, the bracketing samples are found by a forward search resumed from the cached
    ``last_lo`` (usually 0-1 steps) rather than a fresh binary search, and the new index is cached
    back. Engagement is a step signal, so its value at ``t`` is ``rec_engaged[lo]``. Clamps to the
    last sample past the end, so the arm holds the final recorded pose.
    """
    dof = wp.tid()
    n = rec_times.shape[0]

    # every thread reads the shared scratch (counter, index) into a local first; then a single thread
    # writes them back, so the reads and writes don't race (this launch is one warp, dim = NUM_ARM_DOFS).
    step = sim_step_count[0]
    if dof == 0:
        sim_step_count[0] = step + 1
    t = float(step) * dt

    # forward search from the cached index for the largest lo with rec_times[lo] <= t
    lo = last_lo[0]
    while lo < n - 1 and rec_times[lo + 1] <= t:
        lo += 1

    if dof == 0:
        last_lo[0] = lo
        engaged[0] = rec_engaged[lo]  # step signal: value of the most recent frame at or before t

    if lo >= n - 1:
        joint_target_q[dof] = rec_targets[n - 1, dof]  # past the end: hold the last recorded pose
        return

    # Cubic (Catmull-Rom) interpolation through the four surrounding knots. It passes through the two
    # bracketing knots (p1, p2) with tangents estimated from their neighbors, so the target is
    # C1-continuous (no slope kink at the knots) -- unlike linear interpolation, whose kinks inject
    # jerk that the stiff drives ring on.
    frac = (t - rec_times[lo]) / (rec_times[lo + 1] - rec_times[lo])
    p0 = rec_targets[wp.max(lo - 1, 0), dof]
    p1 = rec_targets[lo, dof]
    p2 = rec_targets[lo + 1, dof]
    p3 = rec_targets[wp.min(lo + 2, n - 1), dof]
    f2 = frac * frac
    f3 = f2 * frac
    joint_target_q[dof] = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * frac
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f3
    )


@wp.kernel
def command_seal_kernel(
    engaged: wp.array[wp.bool],  # [1] recorded engagement command (ro[0])
    pad_break_metric: wp.array[float],  # [pads] brittle break envelope from the previous force eval
    pad_engaged: wp.array[wp.bool],  # [pads] whether the pad held last sub-step (from latch_engagement)
    break_threshold: float,  # break metric above this counts as over-capacity (1.0 = nominal capacity)
    break_hold_steps: int,  # sub-steps the metric must stay over threshold before the seal fractures
    seal_break_count: wp.array[int],  # [pads] in/out: consecutive over-threshold sub-steps
    seal_broken: wp.array[wp.bool],  # [pads] in/out: latched physics break within an engaged window
    seal_engaged: wp.array[wp.bool],  # [pads] out: seal command fed to latch_engagement
):
    """Break a pad's seal on either a recorded release (ro[0] low) or a *sustained* physics break.

    The recorded command is the master enable. Within an engaged window a pad also breaks -- and
    stays broken -- once its brittle break metric (see
    :func:`~newton.examples.suctioncup.surface_gripper.eval_break_metric`) exceeds ``break_threshold``
    for ``break_hold_steps`` consecutive sub-steps. The debounce ignores lone transient spikes (brief
    sub-step spikes that survive even the capacity floor) so only a genuine sustained overload fractures
    the seal. The over-threshold test is gated on ``pad_engaged``
    so a pad that was actually holding is what breaks, and a stale metric cannot veto a fresh grip. A
    recorded release clears the latch and counter so the next engage cycle can re-seal.
    """
    pad = wp.tid()
    cmd = engaged[0]
    if not cmd:
        seal_broken[pad] = False  # recorded release resets the break latch for the next cycle
        seal_break_count[pad] = 0
    elif pad_engaged[pad] and pad_break_metric[pad] > break_threshold:
        seal_break_count[pad] = seal_break_count[pad] + 1
        if seal_break_count[pad] >= break_hold_steps:
            seal_broken[pad] = True  # sustained overload: latched off until the recording releases
    else:
        seal_break_count[pad] = 0  # dipped back under -> not a sustained overload
    if cmd and not seal_broken[pad]:
        seal_engaged[pad] = True
    else:
        seal_engaged[pad] = False


class EndEffectorAccelerationRecorder:
    """Records the end-effector acceleration each sim step while the suction is engaged, then writes it
    to CSV at the first disengagement.

    Gated by the caller (see the record calls in ``simulate``). Recording is host-side (reads state
    each sub-step), so it only runs on CPU -- on CUDA it is skipped to avoid breaking graph capture.
    """

    def __init__(self, ee_body, sim_dt):
        self.ee_body = ee_body
        self.sim_dt = sim_dt
        self.accel_log = []  # [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z], EE frame
        self.time_log = []  # matching sim time [s]
        self.done = False  # set True at the first disengagement -> record no more

    def record(self, prev_state, curr_state, engaged, sim_step_count):
        """Record one sub-step. Call after the solver step, before the state swap.

        ``prev_state`` / ``curr_state`` are the pre-/post-step states (finite-difference acceleration);
        ``engaged`` / ``sim_step_count`` are the device arrays for the engagement command and sub-step
        counter.
        """
        if self.done:
            return
        if not bool(engaged.numpy()[0]):
            if self.accel_log:  # was engaged, now disengaged -> stop recording and dump
                self.done = True
                self._dump()
            return

        prev_v = prev_state.body_qd.numpy()[self.ee_body]  # [ang, lin] world, before the step
        curr_v = curr_state.body_qd.numpy()[self.ee_body]  # after the step
        accel = (curr_v - prev_v) / self.sim_dt  # world-frame spatial acceleration
        # rotate into the end-effector frame (inverse of the current EE orientation)
        quat = wp.quat(*curr_state.body_q.numpy()[self.ee_body][3:7])  # EE orientation [x, y, z, w]
        ang = wp.quat_rotate_inv(quat, wp.vec3(*accel[0:3]))  # angular accel, EE frame [rad/s^2]
        lin = wp.quat_rotate_inv(quat, wp.vec3(*accel[3:6]))  # linear accel, EE frame [m/s^2]
        self.accel_log.append([ang[0], ang[1], ang[2], lin[0], lin[1], lin[2]])
        self.time_log.append(float(sim_step_count.numpy()[0]) * self.sim_dt)

    def _dump(self, path="ee_accelerations.csv"):
        """Write the accel log (time + 6 EE-frame components) to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "ang_x", "ang_y", "ang_z", "lin_x", "lin_y", "lin_z"])
            for t, accel in zip(self.time_log, self.accel_log, strict=True):
                writer.writerow([t, *accel])
        print(f"wrote {len(self.time_log)} rows to {path}")


class DriveTargetRecorder:
    """Records the smoothed runtime arm drive targets (the interpolated ``control.joint_target_q``
    applied each sim step) while the suction is engaged, then writes them to CSV at the first
    disengagement.

    Gated by the caller (see the record calls in ``simulate``). Host-side (reads state each sub-step),
    so CPU only.
    """

    def __init__(self, sim_dt, num_arm_dofs):
        self.sim_dt = sim_dt
        self.num_arm_dofs = num_arm_dofs
        self.target_log = []  # applied arm drive targets [rad], J1..J6
        self.time_log = []  # matching sim time [s]
        self.done = False  # set True at the first disengagement -> record no more

    def record(self, engaged, joint_target_q, sim_step_count):
        """Record one sub-step. ``engaged`` / ``joint_target_q`` / ``sim_step_count`` are device arrays
        for the engagement command, the applied drive targets, and the sub-step counter.
        """
        if self.done:
            return
        if not bool(engaged.numpy()[0]):
            if self.target_log:  # was engaged, now disengaged -> stop recording and dump
                self.done = True
                self._dump()
            return
        self.target_log.append(list(joint_target_q.numpy()[: self.num_arm_dofs]))
        self.time_log.append(float(sim_step_count.numpy()[0]) * self.sim_dt)

    def _dump(self, path="drive_targets.csv"):
        """Write the drive-target log (time + J1..J6 [rad]) to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s"] + [f"J{i + 1}" for i in range(self.num_arm_dofs)])
            for t, q in zip(self.time_log, self.target_log, strict=True):
                writer.writerow([t, *q])
        print(f"wrote {len(self.time_log)} rows to {path}")


class Example:
    def __init__(self, viewer, args):

        # Cache the viewer
        self.viewer = viewer

        # FPS and sim step dt
        self.fps = FPS  # rendered frames per second
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt * SIM_HZ))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.break_hold_steps = max(1, round(BREAK_HOLD_TIME / self.sim_dt))  # debounce span in sub-steps

        # Device scratch for sample_playback_kernel: the sub-step counter (drives the sim clock),
        # the cached lower sample index for the forward time search, and the engagement command
        # sampled at the current time (kernel output, for the seal wired up later).
        self.sim_step_count_wp = wp.zeros(1, dtype=wp.int32)
        self.last_lo_wp = wp.zeros(1, dtype=wp.int32)
        self.engaged_wp = wp.zeros(1, dtype=wp.bool)

        # RECORDING_JSONL contains time-stamped joint drive target positions and suction pad engagement
        # states. Load and extract the time-stamps, the joint drive target positions and the
        # suction pad engagement states.
        # Apply gaussian smoothing to the raw drive target after loading.
        rec_times, rec_targets, rec_engaged, self.rec_duration = load_playback(RECORDING_JSONL)
        rec_targets = gaussian_smooth(rec_times, rec_targets, SMOOTHING_SIGMA)  # smooth the coarse waypoints
        self.rec_times_wp = wp.array(rec_times, dtype=wp.float32)
        self.rec_targets_wp = wp.array(rec_targets, dtype=wp.float32)  # 2d [N, NUM_ARM_DOFS]
        self.rec_engaged_wp = wp.array(rec_engaged, dtype=wp.bool)  # [N]; suction engagement command per frame

        # Load the Fanuc robot arm on a ground plane.
        initial_arm_q = rec_targets[0].astype(np.float32)  # drive target at t=0, the start pose
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.restitution = 0.0  # low restitution: the held box shouldn't bounce
        builder.add_usd(str(ROBOT_USD), floating=False, collapse_fixed_joints=True)
        ee_body = builder.body_count - 1  # last arm link (J6_link) is the end-effector flange
        builder.add_ground_plane()

        # Auto-place the pick box + pallet from the flange pose at first engagement (forward kinematics).
        # 1) engagement time -> the smoothed arm targets there; 2) eval_fk -> the flange world pose;
        # 3) project the finger-hull deepest point -> the box-top world position; 4) seat the box (and
        # pallet under it) there. Robust to SMOOTHING_SIGMA / cup geometry / recording (no hard-coded pose).
        engage_idx = int(np.argmax(rec_engaged))  # first frame the suction command (ro[0]) is on
        fk_model = builder.finalize()  # arm-only model, solely to run the FK placement probe
        fk_state = fk_model.state()
        fk_q = fk_state.joint_q.numpy()
        fk_q[:NUM_ARM_DOFS] = rec_targets[engage_idx]  # smoothed arm targets at engagement
        fk_state.joint_q.assign(fk_q)
        newton.eval_fk(fk_model, fk_state.joint_q, fk_state.joint_qd, fk_state)
        t_flange = fk_state.body_q.numpy()[ee_body]  # flange world pose at engagement [px,py,pz,qx,qy,qz,qw]
        cup_c = np.mean(GRIPPER_PADS, axis=0)  # box top seats at the finger-hull deepest, under the cups
        box_top = wp.vec3(*t_flange[:3]) + wp.quat_rotate(
            wp.quat(*t_flange[3:7]), wp.vec3(FINGER_HULL_DEEPEST_X, float(cup_c[1]), float(cup_c[2]))
        )
        box_top_x, box_top_y, box_top_z = float(box_top[0]), float(box_top[1]), float(box_top[2])
        static_box_center = (box_top_x, box_top_y, box_top_z - 2.0 * PICK_BOX_HALF[2] - BOX_HALF)  # pallet
        pick_box_center = (box_top_x, box_top_y, box_top_z - PICK_BOX_HALF[2])  # box top at box_top

        # Static support box (1x1x1, collidable) at the pick pose -- the pallet the pick box sits on.
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(*static_box_center), wp.quat_identity()),
            hx=BOX_HALF,
            hy=BOX_HALF,
            hz=BOX_HALF,
        )
        # Dynamic pick box (the object to pick) resting on the static box. Mass and inertia are set
        # directly on the body (solid-box formula); the shape density is 0 so the shape adds no mass.
        hx, hy, hz = PICK_BOX_HALF
        ixx = PICK_BOX_MASS / 3.0 * (hy * hy + hz * hz)
        iyy = PICK_BOX_MASS / 3.0 * (hx * hx + hz * hz)
        izz = PICK_BOX_MASS / 3.0 * (hx * hx + hy * hy)
        pick_box = builder.add_body(
            xform=wp.transform(wp.vec3(*pick_box_center), wp.quat_identity()),
            mass=PICK_BOX_MASS,
            inertia=wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz),
            label="pick_box",
        )
        pick_cfg = builder.default_shape_cfg.copy()
        pick_cfg.density = 0.0
        pick_box_shape = builder.add_shape_box(
            pick_box, hx=PICK_BOX_HALF[0], hy=PICK_BOX_HALF[1], hz=PICK_BOX_HALF[2], cfg=pick_cfg
        )

        # Pick-box <-> gripper-geometry contact. Enabled: the box presses flush against the rigid tool,
        # so tipping is resisted by contact (like the real gripper). Filtered: the seal alone owns the
        # hold. When the contact is on, the seal should be tension-only so the seal and contact don't
        # both react compression (see SEAL_TENSION_ONLY in surface_gripper.py). The box always collides
        # with the pallet and ground.
        if not ENABLE_PAD_BOX_CONTACT:
            for shape in range(len(builder.shape_body)):
                if builder.shape_body[shape] == ee_body:
                    builder.add_shape_collision_filter_pair(pick_box_shape, shape)

        # Cup markers: a thin non-colliding disk at each suction cup so the cup layout is visible in the
        # viewer (radius = the modeled cup radius dim_a; oriented so the disk faces along the suction axis).
        if SHOW_CUP_MARKERS:
            marker_down = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)  # disk axis -> flange +x
            marker_cfg = builder.default_shape_cfg.copy()
            marker_cfg.density = 0.0
            marker_cfg.has_shape_collision = False
            for px, py, pz in GRIPPER_PADS:
                builder.add_shape_cylinder(
                    ee_body,
                    xform=wp.transform(wp.vec3(px, py, pz), marker_down),
                    radius=GRIPPER_PARAMS.dim_a,
                    half_height=0.004,
                    cfg=marker_cfg,
                )

        self.model = builder.finalize()
        # njmax: MuJoCo's per-world constraint-row buffer. Its auto-estimate from the initial (resting)
        # state is too small once the arm's self-contacts and the box/pallet/ground contacts are all
        # active mid-cycle, which overflows nefc. Give ample headroom.
        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=512, iterations=10)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Suction gripper on the end-effector: one SurfaceGripper on the flange with four pads at the
        # recorded finger offsets, suction axis along the flange +x (pad local +z rotated onto +x).
        # Driven by the recorded ro[0] command -- all four pads engage/release together, sealing the
        # dynamic pick box.
        pad_down = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)  # pad +z -> flange +x
        gripper = SurfaceGripper(
            body_id=ee_body,
            xform=wp.transform_identity(),  # gripper frame == flange body frame
            **asdict(GRIPPER_PARAMS),
        )
        for px, py, pz in GRIPPER_PADS:
            gripper.add_pad(wp.transform(wp.vec3(px, py, pz), pad_down))
        # Characterize the seal's three spring-damper modes for this picked box (constant; shown in the
        # side panel). ixx is the box tilt inertia about a horizontal grip axis; hz is the COM depth
        # below the top-face grip. Stored as (name, omega_n [rad/s], zeta) per mode.
        self.seal_modes = (
            (
                "peel",
                gripper.peel_natural_frequency(ixx, PICK_BOX_MASS, hz),
                gripper.peel_damping_ratio(ixx, PICK_BOX_MASS, GRIPPER_PARAMS.d_peel_x, hz),
            ),
            (
                "normal",
                gripper.normal_natural_frequency(PICK_BOX_MASS),
                gripper.normal_damping_ratio(PICK_BOX_MASS, GRIPPER_PARAMS.d_normal),
            ),
            (
                "shear",
                gripper.shear_natural_frequency(PICK_BOX_MASS),
                gripper.shear_damping_ratio(PICK_BOX_MASS, GRIPPER_PARAMS.d_shear_x),
            ),
        )
        gripper_builder = SurfaceGripperBuilder()
        gripper_builder.add_gripper(gripper)
        self.gripper_model = gripper_builder.finalize(device=self.model.device)
        self.gripper_state = self.gripper_model.state()
        self.gripper_control = self.gripper_model.control()
        self.gripper_control.pad_grip_control.fill_(1.0)  # full suction command
        self.seal_engaged = wp.zeros(len(GRIPPER_PADS), dtype=wp.bool)
        self.seal_broken = wp.zeros(len(GRIPPER_PADS), dtype=wp.bool)  # latched physics break per pad
        self.seal_break_count = wp.zeros(len(GRIPPER_PADS), dtype=wp.int32)  # consecutive over-threshold steps
        self.seal_body_b = wp.full(len(GRIPPER_PADS), pick_box, dtype=wp.int32)

        # Start the arm at the first recorded pose. Set only the arm DOFs; the pick box's free-joint
        # DOFs keep their built-in rest pose (from add_body), so it starts resting on the static box.
        joint_q = self.state_0.joint_q.numpy()
        joint_q[:NUM_ARM_DOFS] = initial_arm_q
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Capture one frame of physics into a CUDA graph, then restore the clean start pose (capturing
        # runs the frame for real, advancing the state).
        self.capture()
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.viewer.set_model(self.model)

        # Record the EE acceleration and the smoothed drive targets over the 1st engaged window to CSV
        if RECORD_DEBUG and not wp.get_device().is_cuda:
            self.accel_recorder = EndEffectorAccelerationRecorder(ee_body, self.sim_dt)
            self.drive_target_recorder = DriveTargetRecorder(self.sim_dt, NUM_ARM_DOFS)

    def capture(self):
        # capturing runs one frame for real, which advances the device sub-step counter and search
        # index, so reset both to 0 afterwards.
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            self.sim_step_count_wp.zero_()
            self.last_lo_wp.zero_()

    def simulate(self):
        for _ in range(self.sim_substeps):
            # Compute the joint drive target positions (joint_target_q)
            # at current sim time (sim_step_count_wp*sim_dt) from the
            # corresponding time series (rec_targets_wp, rec_times_wp).
            # Compute the gripper engagement state (engaged_wp) at current sim
            # time (sim_step_count_wp*sim_dt) from the time series
            # (rec_engaged_wp, rec_times_wp).
            # Advance the progress (last_lo_wp) through the time series
            # (rec_times_wp) and cache it for the next simulate step.
            # Advance sim time for the next call to sample_playback_kernel
            # by incrementing sim_step_count_wp.
            wp.launch(
                sample_playback_kernel,
                dim=NUM_ARM_DOFS,
                inputs=[
                    self.rec_times_wp,
                    self.rec_targets_wp,
                    self.rec_engaged_wp,
                    self.sim_step_count_wp,  # in/out: read as the current time, then advanced in place
                    self.last_lo_wp,  # in/out: forward-search index, resumed and cached
                    float(self.sim_dt),
                ],
                outputs=[self.control.joint_target_q, self.engaged_wp],
            )
            self.state_0.clear_forces()  # zero body_f each sub-step (the suction cup accumulates into it)

            # Suction seal: command all pads from the recorded ro[0] (engaged_wp) OR-ed with a physics
            # break (break metric from the previous sub-step's force eval), latch onto the pick box on
            # the rising edge, then accumulate the seal wrench into body_f before stepping.
            wp.launch(
                command_seal_kernel,
                dim=self.seal_engaged.shape[0],
                inputs=[
                    self.engaged_wp,
                    self.gripper_state.pad_break_metric,
                    self.gripper_state.pad_engaged,
                    float(BREAK_THRESHOLD),
                    int(self.break_hold_steps),
                    self.seal_break_count,
                    self.seal_broken,
                ],
                outputs=[self.seal_engaged],
            )
            latch_engagement(self.state_0, self.gripper_model, self.gripper_state, self.seal_engaged, self.seal_body_b)
            if ENABLE_GRIPPER:
                evaluate_gripper_force(
                    self.model, self.state_0, self.gripper_model, self.gripper_state, self.gripper_control
                )

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if RECORD_DEBUG and not wp.get_device().is_cuda:
                self.accel_recorder.record(self.state_0, self.state_1, self.engaged_wp, self.sim_step_count_wp)
                self.drive_target_recorder.record(self.engaged_wp, self.control.joint_target_q, self.sim_step_count_wp)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # the target kernel interpolates and applies the drive targets and advances the sub-step
        # counter before each physics sub-step, so step() just runs one frame.
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

    def render(self):
        # wall-clock time = physics sub-steps elapsed (read back from the device) * sim_dt
        sim_time = int(self.sim_step_count_wp.numpy()[0]) * self.sim_dt
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        # show the recorded suction-cup command (sampled per sub-step by sample_playback_kernel)
        engaged = bool(self.engaged_wp.numpy()[0])
        ui.text(f"Suction: {'On' if engaged else 'Off'}")
        # seal spring-damper modes for the picked box (constant): natural frequency and damping ratio
        ui.text("Seal modes:")
        for name, omega_n, zeta in self.seal_modes:
            ui.text(f"  {name:6s} f_n={omega_n / (2.0 * np.pi):5.2f} Hz  zeta={zeta:.2f}")

    def test_final(self):
        # the fixed-base arm should hold together on its stiff joint drives: bodies stay at or above
        # the ground (no explosion, no fall-through).
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "robot arm bodies stay at or above the ground",
            lambda q, qd: q[2] > -0.05,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
