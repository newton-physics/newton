# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Diffsim Anymal Backflip

An example of how differentiable physics can be used to perform a backflip on
a modified ANYmal-D quadruped robot dog. Trajectory optimization split into
3 phases:

1. Crouch (300 steps + a 100-step zero-amp push preview): learned cosine
   servo ramps; the loss shapes crouch depth, stillness and knee clearance.
2. Jump (1500 steps): per-joint-group servo amplitudes produce the launch
   and the flip; the loss prices the flip error and attitude at touchdown,
   an apex height band and knee clearance.
3. Landing (first 300 steps alone, then 1600): per-joint target splines
   absorb the touchdown impact; the loss combines knee ordering,
   anti-bounce, paw-glue (every paw pinned to the ground) and an end-state
   stance term.

How the optimization runs:
- every phase shares one optimizer (``_descent``: steepest descent with a
  backtracking line search) and one rollout path (``_simulate``) - the loss
  the optimizer differentiates is exactly the loss it accepts steps on;
- each phase window is a ``Rollout``: buffers allocated once, forward and
  forward+backward passes captured into CUDA graphs;
- a missed gate retries the chain from a different crouch warm start
  (``CROUCH_WARM_SCALES``);
- a full run takes a few minutes on a single GPU, advancing a little every
  frame so an interactive viewer stays live.

Writing differentiable loss kernels: Warp does not replay dynamic
(runtime-bound) loops in the backward pass (see "Dynamic Loops" in Warp's
differentiability guide), so loop-carried values - a running min/max, a
prefix sum, a scanned segment index - get wrong adjoints. The loss kernels
avoid them with four patterns (the loop-sensitive ones FD-checked in
``newton/tests/test_diffsim_anymal_backflip_losses.py``):
- spline segments are bracketed on the host (``_bracket``);
- fixed paw sets are statically unrolled;
- prefix sums/maxima are chained per-element launches
  (``cum_sum_kernel`` / ``cum_max_kernel``);
- the knee minimum is an integer argmin pre-pass + a differentiable gather.

Success criteria:
- Crouch phase: a clean crouch (depth ~0.32 m, upright, knees clear of the ground).
- Jump phase: the dog leaves the ground, completes a full backflip and touches
  down on its paws: touchdown pitch within 10 degrees of -360 and tilt
  below 10 degrees.
- Landing phase: the dog settles standing after the impact (settled gate: tilt
  <= 15 deg, |vz| <= 0.15 m/s, |z - stand| <= 0.10 m, |w| <= 0.6 rad/s).
- When ``--test`` runs, numeric and analytic gradients agree within the
  check tolerance.

Runs:
- `python -m newton.examples diffsim_anymal_backflip`
- `python -m newton.examples diffsim_anymal_backflip --test` (gradient check only)
- `python -m newton.examples diffsim_anymal_backflip --rec backflip.bin`

Table of Contents
1.  Header docstring
2.  Imports + constants
3.  Host-side spline-segment lookup
4.  Servo kernels
5.  Sample kernels
6.  Loss kernels
7.  Replay helpers
8.  Rollout
9.  Example class
10. Execution helpers
11. Optimizer
12. Replays
13. The chain
14. Runner interface
15. Gradient check
16. CLI entry point
"""

# ------------------------------------------------------------------------------
# 2. Imports + constants - every tunable in one place: the simulation
#    parameters and objective weights (single-source: the configuration the
#    demonstrated behavior was achieved with)
# ------------------------------------------------------------------------------

import os
import time

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import GeoType, ShapeFlags

# 2.1 Simulation + model indices
SIM_DT = 1.0e-3  # simulation timestep (s)
STAND_LEG = (0.0, 0.2, -0.4)  # stance pose per leg (HAA, HFE, KFE) in rad
STAND = np.tile(np.array(STAND_LEG, dtype=np.float32), 4)
HINGE_DOFS = np.arange(6, 18, dtype=np.int32)  # the 12 hinge DOFs, after the root's 6
FOOT_BODIES = (3, 6, 9, 12)  # body indices of the four paws
PAWS = len(FOOT_BODIES)  # fixed paw count: paw minima may unroll statically
KE_SCALE = 50.0  # contact stiffness realism multiplier
KD_SCALE = 4.0  # contact damping multiplier (ke * sqrt(KE_SCALE) * KD_SCALE)

# 2.2 Objective weights (found by trial and error and bilevel optimization)
Z_GOOD = 1.7463734824840804  # apex ceiling of the height band (m)
WEIGHT_FLIP = 0.956137871751204  # flip-error weight at touchdown attention
KNEE_MARGIN_GROUND = 0.0217610642328222  # ground-based knee clearance bar (m)
Z_TARGET = 0.32296679716115667  # crouch depth target (m)
WEIGHT_ARRIVAL = 0.4918151418204104  # arrival quality weight
WEIGHT_ATT_JUMP = 3.0 * WEIGHT_ARRIVAL  # jump attitude weight: 3 x the arrival weight
Z_LOW = Z_GOOD - 0.35  # apex floor of the height band (m)
WEIGHT_BAND = 30.0  # apex height-band penalty weight
Z_TOUCHDOWN = 0.68  # touchdown attention: root height the replay calls touchdown (m)
TOUCHDOWN_SIG = 0.015  # touchdown attention sharpness (m)
TOUCHDOWN_LEN = 10.0  # touchdown attention spans ~this many steps after first touchdown
LIFT_HEIGHT = 0.15  # airborne gate: prefix-max paw height for "has flown" (m)
FLIP_SIGN = -1.0  # backflip sign about world +y
WEIGHT_ORDER_CROUCH = 10.0  # crouch per-state knee-ordering weight
WEIGHT_ORDER_JUMP = 40.0  # jump per-state knee-ordering weight
WEIGHT_ORDER_LAND = 20.0  # land per-state knee-ordering weight
WEIGHT_PAWS = 40.0  # paw-glue weight (the yielding-knee mechanism)
WEIGHT_UP = 1.0  # mid-window upright weight
# crouch end terms
WEIGHT_CROUCH_DEPTH, WEIGHT_CROUCH_STILL, WEIGHT_CROUCH_UP = 30.0, 2.0, 6.0
# land end terms (scaled by WEIGHT_ARRIVAL) + per-state anti-bounce
WEIGHT_LAND_DEPTH, WEIGHT_LAND_SPIN = 100.0, 0.1
WEIGHT_LAND_UP, WEIGHT_LAND_VZ = 6.0, 0.5
WEIGHT_BOUNCE = 15.0
TWO_PI = 2.0 * np.pi

# 2.3 Crouch phase
CROUCH_STEPS = 300  # crouch steps (learned part)
CROUCH_PREVIEW = 100  # zero-amp push preview steps
CROUCH_ROLLOUT_STEPS = CROUCH_STEPS + CROUCH_PREVIEW  # full crouch rollout
CROUCH_ITERS = 64  # optimizer iterations
# crouch warm-start scale per chain attempt, alternating around the default;
# the next one is tried when the crouch, the flip or the landing misses its gate
CROUCH_WARM_SCALES = (1.0, 0.98, 1.02, 0.96, 1.04, 0.94)
# _descent step lengths to try, largest first
CROUCH_RADII = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
CROUCH_N_PARAMS = 8  # cosine amps (4 channels) + ramp times (4 channels)
# cosine ramp duration (steps) at multiplier 1; the learned ramp times scale it
CROUCH_RAMP_REF = 249.0
# warm-start params, hip-dominant
CROUCH_WARM = np.array([5.2, 3.8, 4.5, 4.5, 1.0, 1.0, 1.0, 1.0])
# per-channel bounds: amps, then ramp times
CROUCH_AMP_LOW = np.array([-1.0, -1.0, -1.0, -1.0, 0.1, 0.1, 0.1, 0.1])
CROUCH_AMP_HIGH = np.array([5.5, 5.5, 5.5, 5.5, 1.2, 1.2, 1.2, 1.2])
VZ_LAG_STEPS = 5  # vz metric: finite-difference window (steps) at the crouch end
INV_VZ_LAG_DT = 1.0 / (VZ_LAG_STEPS * SIM_DT)

# 2.4 Jump phase
JUMP_STEPS = 1500  # jump rollout steps
JUMP_SEGMENTS = 15  # trajectory segments
JUMP_SEG_STEPS = JUMP_STEPS // JUMP_SEGMENTS  # steps per segment
JUMP_ITERS = 16  # optimizer iterations
JUMP_RADII = (0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005)
# per-segment servo amplitude bounds
JUMP_AMP_LOW, JUMP_AMP_HIGH = -0.9, 3.0
TAU_FRONT, TAU_HIND = 450.0, 300.0  # jump torque caps
TAU_MAX = 300.0  # land servo torque clamp
GROUP_OF_JOINT = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.int32)  # LF,RF front; LH,RH hind
# amp channel per joint: 0/1 front hip pair/knee, 2/3 hind hip pair/knee
CHANNEL_OF_JOINT = np.array([GROUP_OF_JOINT[j] * 2 + (1 if j % 3 == 2 else 0) for j in range(12)], dtype=np.int32)

# 2.5 Landing phase (per-joint direct-target splines, C1 smoothstep, dense
# around touchdown; past the last knot the final values are held)
LAND_STEPS = 1600  # landing rollout steps
LAND_SHORT_WINDOWS = (300, 0, 600)  # first-stage window per landing attempt (steps; 0 = none)
LAND_ITERS = 10  # optimizer iterations
LAND_RADII = (0.6, 0.4, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01)
# knot times of the target splines
LAND_KNOTS_MS = [0, 20, 40, 60, 90, 120, 160, 200, 260, 320, 400, 500, 650, 850, 1150, 1600]
LAND_N_KNOTS = len(LAND_KNOTS_MS)
LAND_N_PARAMS = 12 * (LAND_N_KNOTS - 1)
LAND_JOINT_LOW = (-0.7, -1.2, -2.4)  # spline target bounds per joint type (HAA, HFE, KFE)
LAND_JOINT_HIGH = (0.7, 0.9, -0.05)  # knee high bound kept from locking straight
# per-joint knot bounds, tiled over the four legs
LAND_LOW = np.tile(np.repeat(LAND_JOINT_LOW, LAND_N_KNOTS - 1).astype(np.float64), 4)
LAND_HIGH = np.tile(np.repeat(LAND_JOINT_HIGH, LAND_N_KNOTS - 1).astype(np.float64), 4)
# initial profile: hold the stance pose
LAND_INIT = np.tile(np.repeat(STAND_LEG, LAND_N_KNOTS - 1).astype(np.float64), 4)

# 2.6 Shared rollout + replay constants
STEP_BUDGET_S = 0.05  # optimization time per viewer frame (keeps the window responsive)
REPLAY_YIELD_EVERY = 100  # replays hand control back to the viewer every this many steps
COLLIDE_EVERY = 2  # contact-truthful tapes; replays use the SAME schedule
PAW_MASK_HEIGHT = 0.02  # "grounded" mask: lowest paw within 2 cm of ground
FRAME_EVERY = 8  # one playback frame per this many simulated steps

# 2.7 Servo PD gains (shared by the four servo kernels)
SERVO_KP = 2000.0  # position gain (Nm/rad)
SERVO_KD = 15.0  # velocity damping gain (Nm s/rad)


# -----------------------------------------------------------------------------
# 3. Host-side spline-segment lookup - _bracket: the adjoint-safe spline
#    segment index
# -----------------------------------------------------------------------------


def _bracket(t_ms):
    """Landing-spline segment index: the last knot with LAND_KNOTS_MS[k] <= t_ms,
    clamped to [0, LAND_N_KNOTS - 2] so k + 1 is always a valid knot. Computed on
    the HOST per launch - Warp does not replay dynamic loops in the backward pass,
    so a scanned segment index would get a wrong adjoint (see module docstring)."""
    k = int(np.searchsorted(np.asarray(LAND_KNOTS_MS, dtype=np.float64), float(t_ms), side="right")) - 1
    return max(0, min(k, LAND_N_KNOTS - 2))


# ------------------------------------------------------------------------------
# 4. Servo kernels - the four control laws writing joint torques: settle hold,
#    crouch cosine ramp, jump per-group trajectory, landing spline
# ------------------------------------------------------------------------------


@wp.kernel
def servo_hold_kernel(
    params: wp.array[float],  # one element: the mode-1 pulse amplitude (ignored in mode 0)
    joint_q: wp.array[float],  # State's joint positions: root's 7 pose entries (pos + quat) first, hinges after
    joint_qd: wp.array[float],  # State's joint velocities: root's 6 DOFs first, hinges after
    joint_f: wp.array[float],  # Control's joint-force buffer, written per DOF
    stand: wp.array[float],  # stance angle per hinge joint (rad)
    hinge_dofs: wp.array[int],  # velocity/force buffer index of each hinge joint
    mode: int,  # 0 = hold stand, 1 = stand scaled by (1 + params[0])
):
    """Settle servo: grounds the robot before the rollouts (hold, then one pulse)."""
    j = wp.tid()
    dof = hinge_dofs[j]
    q = joint_q[7 + j]
    qd = joint_qd[dof]
    if mode == 0:
        target = stand[j]
    else:
        target = stand[j] * (1.0 + params[0])
    joint_f[dof] = SERVO_KP * (target - q) - SERVO_KD * qd


@wp.kernel
def servo_crouch_cos_kernel(
    params: wp.array[float],  # 8 learned values: 4 amplitudes, then 4 ramp-time multipliers
    joint_q: wp.array[float],  # State's joint positions: root's 7 pose entries (pos + quat) first, hinges after
    joint_qd: wp.array[float],  # State's joint velocities: root's 6 DOFs first, hinges after
    joint_f: wp.array[float],  # Control's joint-force buffer, written per DOF
    stand: wp.array[float],  # stance angle per hinge joint (rad)
    hinge_dofs: wp.array[int],  # velocity/force buffer index of each hinge joint
    channel_of_joint: wp.array[int],  # learned channel per joint (0/1 = front hips/knees, 2/3 = hind)
    t: int,  # step within the crouch rollout
    ramp_ref: float,  # ramp duration in steps at multiplier 1 (CROUCH_RAMP_REF)
    tau_cap: float,  # torque clamp (Nm; TAU_MAX)
):
    """Crouch ramp: the target glides from stand to stand * (1 + amp) along a
    half-cosine with zero target velocity at both ends."""
    j = wp.tid()
    dof = hinge_dofs[j]
    q = joint_q[7 + j]
    qd = joint_qd[dof]
    ch = channel_of_joint[j]
    amp_a = params[ch]
    tau_p = params[4 + ch]
    ramp_t = ramp_ref * wp.max(tau_p, 0.05)
    u = wp.min(float(t) / ramp_t, 1.0)
    target = stand[j] * (1.0 + amp_a * 0.5 * (1.0 - wp.cos(3.14159265 * u)))
    tau = SERVO_KP * (target - q) - SERVO_KD * qd
    joint_f[dof] = wp.clamp(tau, -tau_cap, tau_cap)


@wp.kernel
def servo_jump_trajectory_caps_kernel(
    seg_amp: wp.array[float],  # learned amplitude per segment and group (15 x 2, index seg * 2 + g)
    joint_q: wp.array[float],  # State's joint positions: root's 7 pose entries (pos + quat) first, hinges after
    joint_qd: wp.array[float],  # State's joint velocities: root's 6 DOFs first, hinges after
    joint_f: wp.array[float],  # Control's joint-force buffer, written per DOF
    stand: wp.array[float],  # stance angle per hinge joint (rad)
    hinge_dofs: wp.array[int],  # velocity/force buffer index of each hinge joint
    group_of_joint: wp.array[int],  # front/hind group per joint (0 = LF + RF, 1 = LH + RH)
    seg: int,  # current segment index (0-14)
    tau_caps: wp.array[float],  # torque cap per group (Nm): [front, hind]
):
    """Jump trajectory: the target pose is piecewise-constant, one learned
    amplitude per (segment, group)."""
    j = wp.tid()
    dof = hinge_dofs[j]
    q = joint_q[7 + j]
    qd = joint_qd[dof]
    g = group_of_joint[j]
    target = stand[j] * (1.0 + seg_amp[seg * 2 + g])
    tau = SERVO_KP * (target - q) - SERVO_KD * qd
    cap = tau_caps[g]
    joint_f[dof] = wp.clamp(tau, -cap, cap)


@wp.kernel
def servo_land_spline_kernel(
    seg_amp: wp.array[float],  # learned target per joint and knot 1..n_knots - 1 (knot 0 pinned to stand)
    joint_q: wp.array[float],  # State's joint positions: root's 7 pose entries (pos + quat) first, hinges after
    joint_qd: wp.array[float],  # State's joint velocities: root's 6 DOFs first, hinges after
    joint_f: wp.array[float],  # Control's joint-force buffer, written per DOF
    stand: wp.array[float],  # stance angle per hinge joint (rad)
    hinge_dofs: wp.array[int],  # velocity/force buffer index of each hinge joint
    knots_ms: wp.array[float],  # the knot times (ms), device copy of LAND_KNOTS_MS
    n_knots: int,  # knot count (len(LAND_KNOTS_MS))
    dt_ms: float,  # milliseconds per simulation step (SIM_DT * 1000)
    t: int,  # step within the landing rollout
    tau_max: float,  # torque clamp (Nm; TAU_MAX)
    k: int,  # spline segment index, precomputed on the host by _bracket
):
    """Per-joint target spline through the landing window. The segment index
    k arrives precomputed from the host: a loop-carried index variable inside
    a differentiable kernel breaks Warp's auto-adjoint (see module docstring)."""
    j = wp.tid()
    dof = hinge_dofs[j]
    q = joint_q[7 + j]
    qd = joint_qd[dof]
    t_ms = float(t) * dt_ms
    t0 = knots_ms[k]
    t1 = knots_ms[k + 1]
    stride = n_knots - 1
    a0 = stand[j]  # knot 0 pinned: hold stand targets at contact
    if k > 0:
        a0 = seg_amp[j * stride + (k - 1)]
    a1 = seg_amp[j * stride + k]
    s = wp.clamp((t_ms - t0) / wp.max(t1 - t0, 1.0e-6), 0.0, 1.0)
    target = a0 + (a1 - a0) * s * s * (3.0 - 2.0 * s)
    tau = SERVO_KP * (target - q) - SERVO_KD * qd
    joint_f[dof] = wp.clamp(tau, -tau_max, tau_max)


# ------------------------------------------------------------------------------
# 5. Sample kernels - per-state captures + chained prefix launches (each
#    sample index written exactly once)
# ------------------------------------------------------------------------------


@wp.kernel
def wy_state_kernel(
    body_qd: wp.array[wp.spatial_vector],  # State's body velocities (world frame); body 0 is the root
    wy_arr: wp.array[float],  # flip-progress increment per state (rad): y angular rate x dt
    idx: int,  # sample index this launch writes (one slot per rollout state)
    dt: float,  # integration window for the velocity sample (SIM_DT)
):
    """Flip-progress increment at one state (``wy_arr[idx]``): the root's
    world-frame y angular velocity integrated over dt. A cum_sum chain over
    these samples is the flip angle the jump loss reads."""
    w = wp.spatial_bottom(body_qd[0])
    wy_arr[idx] = w[1] * dt


@wp.kernel
def zraw_state_kernel(
    body_q: wp.array[wp.transform],  # State's body transforms (world frame); body 0 is the root
    z_arr: wp.array[float],  # root height per state (m)
    idx: int,  # sample index this launch writes (one slot per rollout state)
):
    """Root height at one state (``z_arr[idx]``): read by the touchdown
    attention and by the apex prefix-max the jump height band uses."""
    z = wp.transform_get_translation(body_q[0])[2]
    z_arr[idx] = z


@wp.kernel
def pawraw_state_kernel(
    body_q: wp.array[wp.transform],  # State's body transforms (world frame); body 0 is the root
    paw_body: wp.array[int],  # body index of each paw
    paw_center: wp.array[wp.vec3],  # paw sphere center (body frame)
    paw_r: wp.array[float],  # paw sphere radius (m)
    paw_arr: wp.array[float],  # lowest paw bottom per state (m)
    idx: int,  # sample index this launch writes (one slot per rollout state)
):
    """Lowest paw bottom at one state (``paw_arr[idx]``). The paw count is the
    fixed constant ``PAWS``, so the minimum is a static straight-line unroll
    with an exact adjoint - a runtime-bound loop must not carry it (see the
    module docstring)."""
    min_p = float(1.0e9)
    for k in range(PAWS):
        c = wp.transform_point(body_q[paw_body[k]], paw_center[k])
        min_p = wp.min(min_p, c[2] - paw_r[k])
    paw_arr[idx] = min_p


@wp.kernel
def knee_argmin_kernel(
    body_q: wp.array[wp.transform],  # State's body transforms (world frame); body 0 is the root
    pts_body: wp.array[int],  # body index per knee point
    pts_local: wp.array[wp.vec3],  # knee point position (body frame)
    n_pts: int,  # knee point count
    argmin_arr: wp.array[int],  # lowest knee point index per state (no adjoint)
    idx: int,  # sample index this launch writes (one slot per rollout state)
):
    """Index of the lowest knee point at one state. NON-differentiated pre-pass
    (int output, no adjoint): a runtime-bound loop is fine here precisely
    because nothing differentiable flows through it - the gradient reaches
    the winning knee point through knee_mingz_kernel's gather instead."""
    best = int(0)
    min_k = float(1.0e9)
    for k in range(n_pts):
        p = wp.transform_point(body_q[pts_body[k]], pts_local[k])
        if p[2] < min_k:
            min_k = p[2]
            best = k
    argmin_arr[idx] = best


@wp.kernel
def knee_mingz_kernel(
    body_q: wp.array[wp.transform],  # State's body transforms (world frame); body 0 is the root
    pts_body: wp.array[int],  # body index per knee point
    pts_local: wp.array[wp.vec3],  # knee point position (body frame)
    argmin_arr: wp.array[int],  # lowest knee point index per state (no adjoint)
    min_knee_arr: wp.array[float],  # lowest knee point world z per state (m)
    idx: int,  # sample index this launch writes (one slot per rollout state)
):
    """World z of the lowest knee point at one state (``min_knee_arr[idx]``):
    a loop-free gather of the argmin pre-pass, so the adjoint flows exactly
    to the winning point's body transform (same pattern as the host-side
    ``_bracket`` spline segment, moved onto the device)."""
    k = argmin_arr[idx]
    p = wp.transform_point(body_q[pts_body[k]], pts_local[k])
    min_knee_arr[idx] = p[2]


@wp.kernel
def cum_sum_kernel(
    x_arr: wp.array[float],  # per-state samples to accumulate (already filled)
    cum_arr: wp.array[float],  # running sum written per link
    idx: int,  # this link's index (launch in order 0..n)
):
    """One link of a chained per-element prefix sum: launch for idx = 0..n in
    order and ``cum_arr`` holds x's running sum. Loop-free, so the adjoint is
    exact - the runtime-bound prefix loops this replaces measured with zero
    or last-index-garbage gradients (module docstring)."""
    prev = float(0.0)
    if idx > 0:
        prev = cum_arr[idx - 1]
    cum_arr[idx] = prev + x_arr[idx]


@wp.kernel
def cum_max_kernel(
    x_arr: wp.array[float],  # per-state samples to accumulate (already filled)
    cum_arr: wp.array[float],  # running max written per link
    idx: int,  # this link's index (launch in order 0..n)
):
    """One link of a chained per-element running maximum: launch for
    idx = 0..n in order and ``cum_arr`` holds x's running max (same adjoint
    rationale as cum_sum_kernel)."""
    prev = float(-1.0e9)
    if idx > 0:
        prev = cum_arr[idx - 1]
    cum_arr[idx] = wp.max(prev, x_arr[idx])


@wp.kernel
def touchdown_att_kernel(
    z_arr: wp.array[float],  # root height per state (m; from zraw_state_kernel)
    cum_paw_arr: wp.array[float],  # running max of paw-bottom samples (chained cum_max_kernel)
    idx: int,  # sample index this launch writes (one slot per rollout state)
    z_touchdown: float,  # touchdown attention root height (m)
    touchdown_sig: float,  # attention sharpness (m)
    lift_height: float,  # airborne gate: prefix paw height to call it flown (m)
    att_arr: wp.array[float],  # touchdown attention per state (0..1)
):
    """Touchdown attention at one state (``att_arr[idx]``): ~1 once the dog
    has flown (prefix-max paw height above lift_height) and its root has come
    back down to the touchdown height z_touchdown, ~0 otherwise. The airborne gate
    keeps the push-off out: there the root rises through z_touchdown while the
    paws are still on the ground and the flip error is still a full turn."""
    # sigmoid arguments clamped to +/-30: wp.exp overflows f32 above ~88
    a = wp.clamp((z_touchdown - z_arr[idx]) / touchdown_sig, -30.0, 30.0)
    # airborne-gate sharpness (m): full attention weight 1 cm past lift_height
    ag = wp.clamp((cum_paw_arr[idx] - lift_height) / 0.01, -30.0, 30.0)
    att_arr[idx] = 1.0 / ((1.0 + wp.exp(-a)) * (1.0 + wp.exp(-ag)))


# -----------------------------------------------------------------------------
# 6. Loss kernels - the three phases' loss terms, assembled read-only from
#    the samples
# -----------------------------------------------------------------------------


@wp.kernel
def order_state_ground_kernel(
    min_knee_arr: wp.array[float],  # lowest knee point z per state (m; from knee_mingz_kernel)
    min_paw_arr: wp.array[float],  # lowest paw bottom per state (m; from pawraw_state_kernel)
    idx: int,  # sample index this launch prices (one slot per rollout state)
    weight_order: float,  # crouch knee-ordering weight (WEIGHT_ORDER_CROUCH)
    margin_ground: float,  # knee clearance bar above ground (m; KNEE_MARGIN_GROUND)
    mask_height: float,  # grounded-mask scale (m; PAW_MASK_HEIGHT)
    loss: wp.array[float],  # shared window accumulator (read-modify-write)
):
    """Ground-based knee ordering: while paws are grounded, the lowest knee
    point must stay above the GROUND margin. The minima arrive precomputed
    per state (knee_mingz_kernel / pawraw_state_kernel); the kernel itself is
    loop-free."""
    mask = wp.max(1.0 - min_paw_arr[idx] / mask_height, 0.0)
    viol = wp.max(margin_ground - min_knee_arr[idx], 0.0)
    loss[0] = loss[0] + weight_order * mask * viol * viol


@wp.kernel
def loss_crouch_end_kernel(
    body_q_n: wp.array[wp.transform],  # root transform at the window's last state (states[CROUCH_STEPS])
    body_q_m: wp.array[wp.transform],  # root transform VZ_LAG_STEPS states earlier (the vz finite-difference base)
    loss: wp.array[float],  # shared window accumulator (read-modify-write)
    z_target: float,  # crouch depth target (m; Z_TARGET)
    weight_depth: float,  # depth weight (WEIGHT_CROUCH_DEPTH)
    weight_still: float,  # stillness weight (WEIGHT_CROUCH_STILL)
    weight_up: float,  # upright weight (WEIGHT_CROUCH_UP)
    inv_vz_lag_dt: float,  # 1 / (VZ_LAG_STEPS * SIM_DT): scales the height delta into m/s
):
    """Crouch-end terms at the window's last state: depth (root z vs
    z_target), stillness (root vz over the last VZ_LAG_STEPS steps, a finite
    difference against body_q_m) and upright (the root up-vector's z). Reads
    the root body transforms directly - crouch builds no z sample chain - and
    adds into ``loss[0]``, the shared accumulator zeroed before the window."""
    zn = wp.transform_get_translation(body_q_n[0])
    zm = wp.transform_get_translation(body_q_m[0])
    vz = (zn[2] - zm[2]) * inv_vz_lag_dt
    q = wp.transform_get_rotation(body_q_n[0])
    up = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    loss[0] = (
        loss[0] + weight_depth * wp.pow(zn[2] - z_target, 2.0) + weight_still * vz * vz + weight_up * (1.0 - up[2])
    )


@wp.kernel
def loss_jump_state_kernel(
    body_q: wp.array[wp.transform],  # State's body transforms (world frame); body 0 is the root
    cum_wy_arr: wp.array[float],  # accumulated flip angle per state (chained cum_sum_kernel over wy_arr)
    min_knee_arr: wp.array[float],  # lowest knee point z per state (m; from knee_mingz_kernel)
    min_paw_arr: wp.array[float],  # lowest paw bottom per state (m; the jump window passes the raw paw sample)
    att_arr: wp.array[float],  # touchdown attention per state (from touchdown_att_kernel)
    cum_att_arr: wp.array[float],  # attention prefix sum (chained cum_sum_kernel); read at idx - 1
    idx: int,  # sample index this launch prices (one slot per rollout state)
    loss: wp.array[float],  # shared window accumulator (read-modify-write)
    weight_flip: float,  # flip-error weight at full attention (WEIGHT_FLIP)
    weight_att: float,  # attitude weight (3 x WEIGHT_ARRIVAL at the launch site)
    weight_order: float,  # per-state knee-ordering weight (WEIGHT_ORDER_JUMP)
    margin_ground: float,  # knee clearance bar above ground (m; KNEE_MARGIN_GROUND)
    mask_height: float,  # grounded-mask scale (m; PAW_MASK_HEIGHT)
    flip_sign: float,  # backflip direction about world +y (FLIP_SIGN)
    two_pi: float,  # the wanted flip angle is flip_sign * two_pi (TWO_PI)
    touchdown_len: float,  # attention decay scale in steps (TOUCHDOWN_LEN)
):
    """Per-state jump terms: flip error and attitude at touchdown, knee
    clearance while grounded. pitch (the accumulated roll through this
    state) and the attention prefix sum arrive precomputed by chained
    per-element launches, the knee/paw minima by the gather kernels - the
    kernel itself is loop-free (module docstring).

    Every term is weighted by first = exp(-attention so far / touchdown_len), so
    only the first ~touchdown_len steps after touchdown are priced: past that the
    landing is phase C's job, and the stiff post-touchdown contacts would
    only add chaotic gradient noise."""
    pitch = cum_wy_arr[idx]
    min_k = min_knee_arr[idx]
    min_p = min_paw_arr[idx]
    q = wp.transform_get_rotation(body_q[0])
    up = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    prev = float(0.0)
    if idx > 0:
        prev = cum_att_arr[idx - 1]
    first = wp.exp(-prev / touchdown_len)
    w = att_arr[idx] * first
    flip_err = pitch - flip_sign * two_pi
    mask_g = wp.max(1.0 - min_p / mask_height, 0.0)
    viol_g = wp.max(margin_ground - min_k, 0.0)
    loss[0] = (
        loss[0]
        + weight_flip * w * flip_err * flip_err
        + weight_att * w * (1.0 - up[2])
        + weight_order * first * mask_g * viol_g * viol_g
    )


@wp.kernel
def loss_jump_band_kernel(
    cum_z_arr: wp.array[float],  # running max of root heights (chained cum_max_kernel over z_arr)
    n: int,  # last state index: cum_z_arr[n] is the window's apex height
    loss: wp.array[float],  # shared window accumulator (read-modify-write)
    weight_band: float,  # apex band penalty weight (WEIGHT_BAND)
    z_good: float,  # apex ceiling of the band (m; Z_GOOD)
    z_low: float,  # apex floor of the band (m; Z_LOW)
):
    """Apex band on the window's highest root height. The maximum arrives
    precomputed (a cum_max_kernel chain over the z state samples): a
    runtime-bound running max here measured as a wrong
    ``-2*weight_band*(z_low + 1e9)``-scale adjoint deposited on the last element
    (module docstring)."""
    zmax = cum_z_arr[n]
    loss[0] = loss[0] + weight_band * (wp.pow(wp.max(zmax - z_good, 0.0), 2.0) + wp.pow(wp.max(z_low - zmax, 0.0), 2.0))


@wp.kernel
def loss_land_state_kernel(
    body_q: wp.array[wp.transform],  # State's body transforms (world frame); body 0 is the root
    body_qd: wp.array[wp.spatial_vector],  # State's body velocities (top 3 = linear, bottom 3 = angular)
    min_knee_arr: wp.array[float],  # lowest knee point z per state (m; from knee_mingz_kernel)
    paw_body: wp.array[int],  # body index of each paw
    paw_center: wp.array[wp.vec3],  # paw sphere center (body frame)
    paw_r: wp.array[float],  # paw sphere radius (m)
    idx: int,  # sample index this launch prices (one slot per rollout state)
    loss: wp.array[float],  # shared window accumulator (read-modify-write)
    weight_order: float,  # per-state knee-ordering weight (WEIGHT_ORDER_LAND)
    margin_ground: float,  # knee clearance bar above ground (m; KNEE_MARGIN_GROUND)
    mask_height: float,  # grounded-mask scale (m; PAW_MASK_HEIGHT)
    weight_bounce: float,  # anti-bounce weight on upward root velocity (WEIGHT_BOUNCE)
    weight_paws: float,  # paw-glue weight (WEIGHT_PAWS): pulls paws to the ground
    weight_up: float,  # upright weight (WEIGHT_UP)
):
    """Per-state land terms: ground-based knee ordering + anti-bounce +
    paw-glue + upright. Holding the paw-glue at zero REQUIRES flexed knees -
    the yielding-squat is the only feasible solution, it is not prescribed.

    The knee minimum arrives precomputed per state (knee_mingz_kernel); the
    paw minimum and the paw-glue sum unroll statically over the fixed PAWS
    count - straight-line code with an exact adjoint (module docstring).

    The terms accumulate over every state of the window. (A variant that
    starts the window before touchdown would need to gate them on
    t >= first-contact so airborne states contribute no loss; this example
    starts the window at touchdown, so every state is post-contact.)"""
    min_k = min_knee_arr[idx]
    min_p = float(1.0e9)
    pp = float(0.0)
    for k in range(PAWS):
        c = wp.transform_point(body_q[paw_body[k]], paw_center[k])
        h = c[2] - paw_r[k]
        min_p = wp.min(min_p, h)
        hu = wp.max(h, 0.0)
        pp = pp + hu * hu
    mask_g = wp.max(1.0 - min_p / mask_height, 0.0)
    viol_g = wp.max(margin_ground - min_k, 0.0)
    vz = wp.spatial_top(body_qd[0])[2]
    q = wp.transform_get_rotation(body_q[0])
    up = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    loss[0] = (
        loss[0]
        + weight_order * mask_g * viol_g * viol_g
        + weight_bounce * wp.pow(wp.max(vz, 0.0), 2.0)
        + weight_paws * pp
        + weight_up * (1.0 - up[2])
    )


@wp.kernel
def loss_land_end_kernel(
    body_q_n: wp.array[wp.transform],  # root transform at the window's last state (states[n])
    body_qd_n: wp.array[wp.spatial_vector],  # root velocity there (top 3 = linear, bottom 3 = angular)
    loss: wp.array[float],  # shared window accumulator (read-modify-write)
    z_target: float,  # final depth target (m; the settled height self.z_stand at the launch site)
    weight_arrival: float,  # outer scale for the whole arrival sum (WEIGHT_ARRIVAL)
    weight_depth: float,  # depth weight inside the arrival sum (WEIGHT_LAND_DEPTH)
    weight_spin: float,  # angular-velocity (spin) weight (WEIGHT_LAND_SPIN)
    weight_up: float,  # upright weight (WEIGHT_LAND_UP)
    weight_vz: float,  # vertical-velocity weight (WEIGHT_LAND_VZ)
):
    """Land-end terms at the window's last state: arrival quality - depth
    (root z vs z_target), spin, upright and vz, summed and scaled by
    weight_arrival. Reads the root transform and velocity directly (no
    sample chain) and adds into ``loss[0]``, the shared accumulator zeroed
    before the window."""
    zn = wp.transform_get_translation(body_q_n[0])
    q = wp.transform_get_rotation(body_q_n[0])
    up = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    v = wp.spatial_top(body_qd_n[0])  # linear velocity part
    w = wp.spatial_bottom(body_qd_n[0])  # angular velocity part
    loss[0] = loss[0] + weight_arrival * (
        weight_depth * wp.pow(zn[2] - z_target, 2.0)
        + weight_spin * wp.dot(w, w)
        + weight_up * (1.0 - up[2])
        + weight_vz * v[2] * v[2]
    )


# ------------------------------------------------------------------------------
# 7. Replay helpers - host-side metrics (tilt, up-z) + snapshot loading
# ------------------------------------------------------------------------------


def _up_z(body_q):
    """Up-axis z component of a body transform row (1 upright, -1 inverted)."""
    # a transform row is (pos, quat): [3:7] is the quaternion
    return float(wp.quat_rotate(wp.quat(body_q[3:7]), wp.vec3(0.0, 0.0, 1.0))[2])


def _tilt_of(body_q):
    """Tilt angle (radians) of a body transform row away from upright."""
    return float(np.arccos(np.clip(_up_z(body_q), -1.0, 1.0)))  # clip: arccos NaNs on |up_z| > 1 (fp drift)


def _load_snapshot(st, snap):
    """Copy a host-side snapshot tuple (from _snapshot) back into a state's device arrays."""
    st.joint_q.assign(snap[0])
    st.joint_qd.assign(snap[1])
    st.body_q.assign(snap[2])
    st.body_qd.assign(snap[3])


# ------------------------------------------------------------------------------
# 8. Rollout - one phase window's preallocated buffers + its two CUDA graphs
# ------------------------------------------------------------------------------


class Rollout:
    """All buffers of one phase window, allocated once, plus two CUDA graphs
    captured over them: the forward pass alone (loss evaluations for the line
    search, and replays) and forward + backward (loss and gradient).

    Every per-step buffer is written at most once inside the tape: one state
    per step, one control per step, one contact buffer per collide. Sharing
    one buffer across steps would leave it holding a LATER step's values by
    the time an earlier step's adjoint re-derives its primal locals from it.
    The same holds for the solver's mass-matrix scratch, which it attaches to
    each gradient-recording output state.

    Launching a captured graph replays the whole window with a single call,
    which removes the Python launch overhead of the thousands of kernels a
    1000+ step window issues. Without CUDA graph support (e.g. on CPU) the
    same code runs uncaptured."""

    def __init__(self, example, mode, n, n_params):
        """Buffers for an n-step window (n + 1 states: every step produces
        one) and n_params learned parameters (mode-specific)."""
        model = example.model
        self.example = example
        self.mode = mode
        self.n = n
        self.params = wp.zeros(n_params, dtype=wp.float32, requires_grad=True)
        self.loss_arr = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.states = [model.state(requires_grad=True) for _ in range(n + 1)]
        self.ctrls = [model.control(requires_grad=True) for _ in range(n)]
        self.contacts = [example.pipe.contacts() for _ in range((n + COLLIDE_EVERY - 1) // COLLIDE_EVERY)]
        # per-state loss-stage samples: knee/paw minima as adjoint-exact
        # gathers (module docstring - why these are not in-kernel
        # runtime-bound loops)
        self.argmin_arr = wp.zeros(n + 1, dtype=wp.int32)
        self.min_knee_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
        self.min_paw_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
        if mode == "jump":
            # per-state samples the jump loss reads; the prefix sum / maxima
            # are chained per-element launches into the cum_* arrays
            self.wy_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
            self.z_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
            self.cum_wy_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
            self.cum_z_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
            self.cum_paw_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
            # touchdown attention per state and its prefix sum
            self.att_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
            self.cum_att_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
        self.tape = wp.Tape()
        self.fwd_graph = None
        self.grad_graph = None
        device = model.device
        # capture needs the device memory pool: raw cudaMalloc is not capture-safe
        if device.is_cuda and device.is_mempool_enabled:
            # warm-up pass outside the capture: the solver allocates its
            # per-state scratch on the first step into each state, and those
            # buffers must exist before (not inside) the graphs that share them
            example._simulate(self)
            with wp.ScopedCapture(device) as cap:
                example._simulate(self)
            self.fwd_graph = cap.graph
            with wp.ScopedCapture(device) as cap:
                with self.tape:
                    example._simulate(self)
                self.tape.backward(self.loss_arr)
            self.grad_graph = cap.graph

    def _load(self, amps, snap):
        """Load the learned parameters and the window's start state (settle
        snapshot for crouch, crouch end for jump, touchdown for land)."""
        self.params.assign(np.asarray(amps, dtype=np.float32))
        _load_snapshot(self.states[0], snap)

    def loss(self, amps, snap):
        """Forward pass only: the window's loss for these parameters."""
        self._load(amps, snap)
        if self.fwd_graph is not None:
            wp.capture_launch(self.fwd_graph)
        else:
            self.example._simulate(self)
        return float(self.loss_arr.numpy()[0])

    def loss_and_grad(self, amps, snap):
        """Forward + backward pass: the loss and its gradient w.r.t. the parameters."""
        self._load(amps, snap)
        if self.grad_graph is not None:
            self.tape.zero()  # gradients accumulate: clear the previous call's
            wp.capture_launch(self.grad_graph)
        else:
            self.tape.reset()
            with self.tape:
                self.example._simulate(self)
            self.tape.backward(self.loss_arr)
        # the gradient is promoted to float64: _descent's step arithmetic is f64
        return float(self.loss_arr.numpy()[0]), self.params.grad.numpy().astype(np.float64)


# -----------------------------------------------------------------------------
# 9. Example class - setup: build ANYmal-D, classify paw/knee contact shapes,
#    settle the stance
# -----------------------------------------------------------------------------


class Example:
    """ANYmal-D standing backflip: crouch -> -360 deg jump -> landing."""

    def __init__(self, viewer, args):
        """Build the ANYmal-D model, classify paw and knee contact shapes, settle the
        standing stance, and open the optional --rec recorder."""
        self.viewer = viewer
        self.args = args
        self.frame = 0
        self.chain_done = False
        self._chain = None  # the run_chain() generator, advanced by step()
        # runs that stop after a fixed frame count have no window to keep
        # responsive and must not stop mid-optimization
        self._finish_in_one_frame = isinstance(viewer, (newton.viewer.ViewerNull, newton.viewer.ViewerUSD)) or bool(
            getattr(args, "headless", False)
        )
        self.playback = []  # (phase, snapshot) frames of the final run
        self.result = {}

        # ---- model: ANYmal-D on a ground plane, differentiable ----
        asset = newton.utils.download_asset("anybotics_anymal_d")
        robot = newton.ModelBuilder(up_axis=newton.Axis.Z)
        robot.default_joint_cfg.armature = 0.1
        robot.default_shape_cfg.mu = 1.0
        robot.add_usd(
            str(asset / "usd" / "anymal_d.usda"),
            force_position_velocity_actuation=True,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
        )
        # slightly inflated inertias stabilize the stiff contact transients
        for body in range(robot.body_count):
            inertia_np = np.asarray(robot.body_inertia[body], dtype=np.float32).reshape(3, 3)
            inertia_np += np.eye(3, dtype=np.float32) * 0.01
            robot.body_inertia[body] = wp.mat33(inertia_np)
        robot.joint_q[0:3] = (0.0, 0.0, 0.66)  # root at 0.66 m
        robot.joint_q[6] = 1.0  # identity quaternion (w)
        robot.joint_q[7:19] = STAND.tolist()  # hinge targets = stance pose
        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        scene.replicate(robot, 1)
        scene.add_ground_plane(cfg=robot.default_shape_cfg)
        # requires_grad=True: the whole model's downstream quantities carry adjoints
        self.model = scene.finalize(requires_grad=True)

        # contact realism: stiffen/damp every shape material
        ke0 = float(self.model.shape_material_ke.numpy()[0])
        kd0 = float(self.model.shape_material_kd.numpy()[0])
        n_shapes = len(self.model.shape_material_ke.numpy())
        self.model.shape_material_ke.assign(np.full(n_shapes, ke0 * KE_SCALE, dtype=np.float32))
        self.model.shape_material_kd.assign(np.full(n_shapes, kd0 * np.sqrt(KE_SCALE) * KD_SCALE, dtype=np.float32))

        # use_tile_gemm: the joint-space mass matrix H = J^T M J is formed with
        # Warp's tile API (one thread block per articulation) instead of a
        # single-threaded dense GEMM - for one 18-DOF robot this is ~30x less
        # GPU time per differentiable step
        self.solver = newton.solvers.SolverFeatherstone(self.model, use_tile_gemm=True)
        # differentiable contacts: contact-force adjoints flow through the pipeline into the loss
        self.pipe = newton.CollisionPipeline(self.model, requires_grad=True)
        self.cts = self.pipe.contacts()
        # ---- classify collision shapes: paw spheres vs true knee shapes ----
        shape_body = np.array(self.model.shape_body.numpy(), dtype=np.int32)
        shape_type = np.array(self.model.shape_type.numpy(), dtype=np.int32)
        shape_flags = np.array(self.model.shape_flags.numpy(), dtype=np.int32)
        shape_scale = np.array(self.model.shape_scale.numpy(), dtype=np.float64)
        shape_xf = np.array(self.model.shape_transform.numpy(), dtype=np.float64)
        self.foot_shapes = frozenset(
            int(i)
            for i in range(n_shapes)
            if shape_body[i] in FOOT_BODIES
            and shape_type[i] == int(GeoType.SPHERE)
            and (shape_flags[i] & int(ShapeFlags.COLLIDE_SHAPES))
        )
        # TRUE knee shapes only: colliding shapes on the leg bodies. The feet
        # carry a second non-sphere shape whose corners ride ~1 cm above the
        # ground when planted - baking those as knee points creates a phantom
        # violation no optimizer can clear.
        knee_shapes = sorted(
            int(i)
            for i in range(n_shapes)
            if 1 <= shape_body[i] <= 12  # bodies 1-12 are the 12 leg links (0 is the root)
            and shape_body[i] not in FOOT_BODIES
            and (shape_flags[i] & int(ShapeFlags.COLLIDE_SHAPES))
        )

        knee_pts_local = []
        knee_pts_body = []
        for i in knee_shapes:
            b = int(shape_body[i])
            stype = int(shape_type[i])
            sc = shape_scale[i]
            xf = shape_xf[i]
            spos, sq = xf[:3], xf[3:]
            pts = []
            if stype == int(GeoType.BOX):
                hx, hy, hz = sc[:3]
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            pts.append(np.array([sx * hx, sy * hy, sz * hz]))
            elif stype == int(GeoType.CYLINDER):
                r, hh = sc[0], sc[1]
                for sz in (-1, 1):
                    for ang in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
                        pts.append(np.array([r * np.cos(ang), r * np.sin(ang), sz * hh]))
            else:
                continue
            for p in pts:
                knee_pts_local.append(wp.transform_point(wp.transform(spos, sq), wp.vec3(p)))
                knee_pts_body.append(b)
        self.knee_pts_body_arr = wp.array(knee_pts_body, dtype=wp.int32)
        self.knee_pts_local_arr = wp.array(np.array(knee_pts_local, dtype=np.float32), dtype=wp.vec3)
        self.n_knee_pts = len(knee_pts_body)

        # paw points: the four foot spheres' lowest points (center - radius)
        paw_pts = [
            (int(shape_body[i]), wp.vec3(shape_xf[i][:3]), float(shape_scale[i][0])) for i in sorted(self.foot_shapes)
        ]
        self.paw_pts_py = paw_pts
        self.paw_body_arr = wp.array([p[0] for p in paw_pts], dtype=wp.int32)
        self.paw_center_arr = wp.array([p[1] for p in paw_pts], dtype=wp.vec3)
        self.paw_r_arr = wp.array([p[2] for p in paw_pts], dtype=wp.float32)
        self.n_paw = len(paw_pts)
        if self.n_paw != PAWS:
            # the paw-min kernels unroll statically over PAWS
            raise RuntimeError(f"expected {PAWS} paw collision shapes, found {self.n_paw}")

        # constant device arrays
        self.stand_arr = wp.array(STAND, dtype=wp.float32)
        self.dofs_arr = wp.array(HINGE_DOFS, dtype=wp.int32)
        self.group_arr = wp.array(GROUP_OF_JOINT, dtype=wp.int32)
        self.channel_arr = wp.array(CHANNEL_OF_JOINT, dtype=wp.int32)
        self.zero_amp_arr = wp.zeros(JUMP_SEGMENTS * 2, dtype=wp.float32)
        self.tau_caps_arr = wp.array([TAU_FRONT, TAU_HIND], dtype=wp.float32)
        self.land_knots_arr = wp.array(np.array(LAND_KNOTS_MS, dtype=np.float32), dtype=wp.float32)
        self.ctrl = self.model.control()

        # forward kinematics: derive the body transforms from the posed joint_q,
        # so the viewer and the settle start from the standing pose
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.viewer.set_model(self.model)

        # optional recording of the optimized playback (see --rec): the
        # cached playback is written as one full ViewerFile pass when the
        # recording is closed at exit, independent of the viewer's frame
        # limit (a `--viewer null` run defaults to 100 frames, fewer than
        # one playback pass contains)
        self.recorder = None
        self.rec_done = False
        if self.args.rec:
            self.recorder = newton.viewer.ViewerFile(self.args.rec, auto_save=False)
            self.recorder.set_model(self.model)

        # ---- settle to a grounded stance (hold stand, then one pulse) ----
        # hold for 1200 steps (contact transients die), then a 100-step pulse
        # at -12% stand compresses the suspension once and seats the contacts
        s0 = self.model.state()
        s1 = self.model.state()
        s0 = self._settle_segment(s0, s1, 1200, mode=0)
        s0 = self._settle_segment(s0, s1, 100, mode=1, amp=-0.12)
        self.snap = self._snapshot(s0)
        self.z_stand = float(self.snap[2][0][2])
        print(f"settle done: z_stand={self.z_stand:.4f}", flush=True)

    # --------------------------------------------------------------------------
    # 10. Execution helpers - snapshots, settling, contact counting, gate
    #     metrics, _simulate (the one rollout+loss path all consumers share)
    # --------------------------------------------------------------------------
    def _snapshot(self, st):
        """Copy a state's joint/body arrays to a host-side numpy snapshot tuple."""
        return tuple(arr.numpy().copy() for arr in (st.joint_q, st.joint_qd, st.body_q, st.body_qd))

    def _fresh(self, snap):
        """Materialize a fresh model state from a snapshot tuple."""
        st = self.model.state()
        _load_snapshot(st, snap)
        return st

    def _settle_segment(self, st0, st1, n, mode, amp=0.0):
        """Run n solver steps under the hold servo with the given mode and amplitude."""
        params = wp.array([amp], dtype=wp.float32)
        for _ in range(n):
            self.pipe.collide(st0, self.cts)
            wp.launch(
                servo_hold_kernel,
                dim=12,
                inputs=[params, st0.joint_q, st0.joint_qd, self.ctrl.joint_f, self.stand_arr, self.dofs_arr, mode],
            )
            st0.clear_forces()
            self.solver.step(st0, st1, self.ctrl, self.cts, SIM_DT)
            st0, st1 = st1, st0
        return st0

    def _foot_row_count(self, contacts):
        """Number of live contacts involving a paw collision shape."""
        n_c = int(contacts.rigid_contact_count.numpy()[0])
        if n_c == 0:
            return 0
        s0 = contacts.rigid_contact_shape0.numpy()[:n_c]
        s1 = contacts.rigid_contact_shape1.numpy()[:n_c]
        return int(sum(1 for k in range(n_c) if s0[k] in self.foot_shapes or s1[k] in self.foot_shapes))

    def _state_metrics(self, body_q0, body_qd0):
        """Root-body gate metrics: height, tilt, vertical and angular speed, as a dict."""
        return {
            "z": float(body_q0[2]),
            "tilt": _tilt_of(body_q0),
            "vz": float(body_qd0[2]),
            "w": float(np.linalg.norm(body_qd0[3:6])),
        }

    def _launch_servo(self, ro, t):
        """Write the phase's servo torques for step t into the rollout's control buffer."""
        st, ctrl = ro.states[t], ro.ctrls[t]
        if ro.mode == "crouch" and t < CROUCH_STEPS:
            wp.launch(
                servo_crouch_cos_kernel,
                dim=12,
                inputs=[
                    ro.params,
                    st.joint_q,
                    st.joint_qd,
                    ctrl.joint_f,
                    self.stand_arr,
                    self.dofs_arr,
                    self.channel_arr,
                    t,
                    CROUCH_RAMP_REF,
                    TAU_MAX,
                ],
            )
        elif ro.mode in ("crouch", "jump"):
            # jump: per-segment trajectory amplitudes; crouch preview
            # (t >= CROUCH_STEPS): zero-amp rebound with jump torque caps - the order
            # term prices the rebound over these states too
            jump = ro.mode == "jump"
            wp.launch(
                servo_jump_trajectory_caps_kernel,
                dim=12,
                inputs=[
                    ro.params if jump else self.zero_amp_arr,
                    st.joint_q,
                    st.joint_qd,
                    ctrl.joint_f,
                    self.stand_arr,
                    self.dofs_arr,
                    self.group_arr,
                    min(t // JUMP_SEG_STEPS, JUMP_SEGMENTS - 1) if jump else 0,
                    self.tau_caps_arr,
                ],
            )
        else:
            wp.launch(
                servo_land_spline_kernel,
                dim=12,
                inputs=[
                    ro.params,
                    st.joint_q,
                    st.joint_qd,
                    ctrl.joint_f,
                    self.stand_arr,
                    self.dofs_arr,
                    self.land_knots_arr,
                    LAND_N_KNOTS,
                    SIM_DT * 1000.0,
                    t,
                    TAU_MAX,
                    _bracket(t * SIM_DT * 1000.0),
                ],
            )

    def _launch_knee_min(self, st, ro, t):
        """Record the state's lowest knee-point world z into ``ro.min_knee_arr[t]``:
        a non-differentiated argmin pre-pass followed by a differentiable
        gather of the winning point (see knee_argmin_kernel)."""
        wp.launch(
            knee_argmin_kernel,
            dim=1,
            inputs=[st.body_q, self.knee_pts_body_arr, self.knee_pts_local_arr, self.n_knee_pts, ro.argmin_arr, t],
        )
        wp.launch(
            knee_mingz_kernel,
            dim=1,
            inputs=[st.body_q, self.knee_pts_body_arr, self.knee_pts_local_arr, ro.argmin_arr, ro.min_knee_arr, t],
        )

    def _simulate(self, ro):
        """Roll out one phase window and accumulate its loss into ro.loss.

        This is the ONLY place the optimization windows are simulated: the
        gradient (recorded on a tape), the line-search loss evaluations and
        the replays all run these exact kernels on the same buffers, so the
        loss the optimizer differentiates is the loss it accepts steps on."""
        n = ro.n
        ro.loss_arr.zero_()
        contacts = None
        for t in range(n):
            if t % COLLIDE_EVERY == 0:
                contacts = ro.contacts[t // COLLIDE_EVERY]
                self.pipe.collide(ro.states[t], contacts)
            self._launch_servo(ro, t)
            ro.states[t].clear_forces()
            self.solver.step(ro.states[t], ro.states[t + 1], ro.ctrls[t], contacts, SIM_DT)

        states = ro.states
        if ro.mode == "crouch":
            for t in range(n + 1):
                self._launch_knee_min(states[t], ro, t)
                wp.launch(
                    pawraw_state_kernel,
                    dim=1,
                    inputs=[
                        states[t].body_q,
                        self.paw_body_arr,
                        self.paw_center_arr,
                        self.paw_r_arr,
                        ro.min_paw_arr,
                        t,
                    ],
                )
                wp.launch(
                    order_state_ground_kernel,
                    dim=1,
                    inputs=[
                        ro.min_knee_arr,
                        ro.min_paw_arr,
                        t,
                        WEIGHT_ORDER_CROUCH,
                        KNEE_MARGIN_GROUND,
                        PAW_MASK_HEIGHT,
                        ro.loss_arr,
                    ],
                )
            wp.launch(
                loss_crouch_end_kernel,
                dim=1,
                inputs=[
                    states[CROUCH_STEPS].body_q,
                    states[CROUCH_STEPS - VZ_LAG_STEPS].body_q,
                    ro.loss_arr,
                    Z_TARGET,
                    WEIGHT_CROUCH_DEPTH,
                    WEIGHT_CROUCH_STILL,
                    WEIGHT_CROUCH_UP,
                    INV_VZ_LAG_DT,
                ],
            )
        elif ro.mode == "jump":
            for t in range(n + 1):
                wp.launch(wy_state_kernel, dim=1, inputs=[states[t].body_qd, ro.wy_arr, t, SIM_DT])
                wp.launch(zraw_state_kernel, dim=1, inputs=[states[t].body_q, ro.z_arr, t])
                wp.launch(
                    pawraw_state_kernel,
                    dim=1,
                    inputs=[
                        states[t].body_q,
                        self.paw_body_arr,
                        self.paw_center_arr,
                        self.paw_r_arr,
                        ro.min_paw_arr,
                        t,
                    ],
                )
                # prefix values as chained per-element launches (adjoint-exact;
                # the in-kernel runtime-bound loops they replace measured with
                # zero or garbage gradients - see the module docstring)
                wp.launch(cum_sum_kernel, dim=1, inputs=[ro.wy_arr, ro.cum_wy_arr, t])
                wp.launch(cum_max_kernel, dim=1, inputs=[ro.z_arr, ro.cum_z_arr, t])
                wp.launch(cum_max_kernel, dim=1, inputs=[ro.min_paw_arr, ro.cum_paw_arr, t])
                wp.launch(
                    touchdown_att_kernel,
                    dim=1,
                    inputs=[ro.z_arr, ro.cum_paw_arr, t, Z_TOUCHDOWN, TOUCHDOWN_SIG, LIFT_HEIGHT, ro.att_arr],
                )
                wp.launch(cum_sum_kernel, dim=1, inputs=[ro.att_arr, ro.cum_att_arr, t])
                self._launch_knee_min(states[t], ro, t)
                wp.launch(
                    loss_jump_state_kernel,
                    dim=1,
                    inputs=[
                        states[t].body_q,
                        ro.cum_wy_arr,
                        ro.min_knee_arr,
                        # the state sample IS the state's lowest paw bottom:
                        # the kernel reads its min_p from it
                        ro.min_paw_arr,
                        ro.att_arr,
                        ro.cum_att_arr,
                        t,
                        ro.loss_arr,
                        WEIGHT_FLIP,
                        WEIGHT_ATT_JUMP,
                        WEIGHT_ORDER_JUMP,
                        KNEE_MARGIN_GROUND,
                        PAW_MASK_HEIGHT,
                        FLIP_SIGN,
                        TWO_PI,
                        TOUCHDOWN_LEN,
                    ],
                )
            wp.launch(loss_jump_band_kernel, dim=1, inputs=[ro.cum_z_arr, n, ro.loss_arr, WEIGHT_BAND, Z_GOOD, Z_LOW])
        else:
            for t in range(n + 1):
                self._launch_knee_min(states[t], ro, t)
                # read by the replay only (the landing loss takes its paw minimum in-kernel)
                wp.launch(
                    pawraw_state_kernel,
                    dim=1,
                    inputs=[
                        states[t].body_q,
                        self.paw_body_arr,
                        self.paw_center_arr,
                        self.paw_r_arr,
                        ro.min_paw_arr,
                        t,
                    ],
                )
                wp.launch(
                    loss_land_state_kernel,
                    dim=1,
                    inputs=[
                        states[t].body_q,
                        states[t].body_qd,
                        ro.min_knee_arr,
                        self.paw_body_arr,
                        self.paw_center_arr,
                        self.paw_r_arr,
                        t,
                        ro.loss_arr,
                        WEIGHT_ORDER_LAND,
                        KNEE_MARGIN_GROUND,
                        PAW_MASK_HEIGHT,
                        WEIGHT_BOUNCE,
                        WEIGHT_PAWS,
                        WEIGHT_UP,
                    ],
                )
            wp.launch(
                loss_land_end_kernel,
                dim=1,
                inputs=[
                    states[n].body_q,
                    states[n].body_qd,
                    ro.loss_arr,
                    self.z_stand,
                    WEIGHT_ARRIVAL,
                    WEIGHT_LAND_DEPTH,
                    WEIGHT_LAND_SPIN,
                    WEIGHT_LAND_UP,
                    WEIGHT_LAND_VZ,
                ],
            )

    # --------------------------------------------------------------------------
    # 11. Optimizer - _descent: shared steepest descent + backtracking line search
    # --------------------------------------------------------------------------
    def _descent(self, tag, ro, snap, amps_init, low, high, iters, radii):
        """Steepest descent with a backtracking line search, shared by all
        three phases: each iteration takes a fresh tape gradient and tries
        the step lengths in ``radii`` (largest first) along the normalized
        descent direction, projected onto the [low, high] box, and accepts the
        first one that lowers the loss. It stops once no step length helps.
        Trying the large steps first every iteration (instead of shrinking a
        trust radius on each rejection) keeps the search from stalling on
        the kinks that contact events put into these losses.

        A generator: it yields after every rollout so ``step()`` can hand
        control back to the viewer, and returns the optimized amplitudes."""
        amps = np.clip(np.asarray(amps_init, dtype=np.float64), low, high)
        L_init = ro.loss(amps, snap)
        L = L_init
        accepts = 0
        for it in range(iters):
            yield
            _, g = ro.loss_and_grad(amps, snap)
            ng = float(np.linalg.norm(g))
            if not np.isfinite(ng) or ng < 1e-14:
                print(f"  {tag} it{it}: |g|={ng:.3g} - stop", flush=True)
                break
            d = -g / ng
            moved = False
            for r in radii:
                yield
                trial = np.clip(amps + r * d, low, high)
                L_trial = ro.loss(trial, snap)
                if np.isfinite(L_trial) and L_trial < L - 1e-4 * abs(L):
                    amps, L = trial, L_trial
                    accepts += 1
                    moved = True
                    print(f"  {tag} it{it}: L={L:.4g}  (step {r:g})", flush=True)
                    break
            if not moved:
                print(f"  {tag} it{it}: L={L:.4g}  no step length improved - stop", flush=True)
                break
        print(f"  {tag}: L {L_init:.4g} -> {L:.4g} ({accepts} steps accepted)", flush=True)
        return amps

    # A replay re-runs the window's forward pass and reads the resulting
    # trajectory back from the rollout buffers. Contacts at step t are the
    # ones collided at the start of that step: ro.contacts[t // COLLIDE_EVERY].
    # Replays are generators like ``_descent`` (they yield every
    # REPLAY_YIELD_EVERY steps) and return their results.

    # --------------------------------------------------------------------------
    # 12. Replays - re-run a window, read back gate metrics + snapshots
    # --------------------------------------------------------------------------
    def _crouch_replay(self, ro, amps):
        """Replay the crouch phase for the given amplitudes; returns the end snapshot and metrics."""
        ro.loss(amps, self.snap)
        # knee and paw minima over the full replay window (crouch + preview),
        # as the loss computed them on the device
        mink, minp = ro.min_knee_arr.numpy()[1:], ro.min_paw_arr.numpy()[1:]
        grounded = minp < PAW_MASK_HEIGHT
        rep = {
            "min_foot_rows": 10**9,
            "knee_min": float(mink.min()),
            "push_margin": float(mink[grounded].min()) if grounded.any() else 1e9,
        }
        snap_end = None
        for t in range(CROUCH_ROLLOUT_STEPS):
            if t % REPLAY_YIELD_EVERY == 0:
                yield
            rows = self._foot_row_count(ro.contacts[t // COLLIDE_EVERY])
            rep["min_foot_rows"] = min(rep["min_foot_rows"], rows)
            if t == CROUCH_STEPS - 1:
                st = ro.states[t + 1]
                m = self._state_metrics(st.body_q.numpy()[0], st.body_qd.numpy()[0])
                snap_end = self._snapshot(st)
                rep["z_end"] = m["z"]
                rep["tilt_end_deg"] = float(np.degrees(m["tilt"]))
                rep["vz_end"] = m["vz"]
                rep["foot_rows_end"] = rows
        return snap_end, rep

    def _jump_replay(self, ro, amps, snap, record=False):
        """Fly the jump FROM THE CROUCH END (the ascent's anchor); graze-
        filtered touchdown detection (airborne = 5 foot-row-free steps; td =
        2 grounded steps AND descending near ground). Returns (td_snap or
        None, report)."""
        ro.loss(amps, snap)
        mink, minp = ro.min_knee_arr.numpy(), ro.min_paw_arr.numpy()
        q_prev = wp.quat(snap[2][0][3:])
        pitch_cum = 0.0
        airborne_steps = 0
        foot_steps = 0
        td_snap = None
        td_t = None
        rep = {"apex": -1e9, "paw_min": 1e9, "blew_up": False, "knee_margin_min": 1e9, "min_foot_rows": 10**9}
        airborne = False
        for t in range(JUMP_STEPS):
            if t % REPLAY_YIELD_EVERY == 0:
                yield
            rows = self._foot_row_count(ro.contacts[t // COLLIDE_EVERY])
            st = ro.states[t + 1]
            bqd = st.body_q.numpy()
            m = self._state_metrics(bqd[0], st.body_qd.numpy()[0])
            # pitch from orientation deltas, NOT angular-velocity
            # integration (the two estimates differ slightly)
            q_cur = wp.quat(bqd[0][3:])
            pitch_cum += float(newton.math.quat_velocity(q_cur, q_prev, 1.0)[1])
            q_prev = q_cur
            min_k, mp = float(mink[t + 1]), float(minp[t + 1])
            rep["apex"] = max(rep["apex"], m["z"])
            rep["paw_min"] = min(rep["paw_min"], mp)
            rep["min_foot_rows"] = min(rep["min_foot_rows"], rows)
            if mp < PAW_MASK_HEIGHT:
                rep["knee_margin_min"] = min(rep["knee_margin_min"], min_k)
            if not np.isfinite(m["z"]) or m["z"] > 3.5:
                rep["blew_up"] = True
                break
            if record and t % FRAME_EVERY == 0:
                self.playback.append(("jump", self._snapshot(st)))
            # graze-filtered touchdown (physical gates: descending + low)
            if rows == 0:
                airborne_steps += 1
                foot_steps = 0
            elif airborne_steps >= 5:
                airborne = True
                if m["vz"] < 3.0 and m["z"] < Z_TOUCHDOWN:
                    foot_steps += 1
                    if foot_steps >= 2:
                        td_snap = self._snapshot(st)
                        td_t = t + 1
                        break
            else:
                airborne_steps = 0
        rep["pitch_end_deg"] = float(np.degrees(pitch_cum))
        if td_snap is not None:
            bqd0 = td_snap[2][0]
            bqd_d0 = td_snap[3][0]
            rep["td_pitch_deg"] = rep["pitch_end_deg"]
            rep["td_step"] = td_t
            rep["td_found"] = True
            rep["td_z"] = float(bqd0[2])
            rep["td_tilt_deg"] = float(np.degrees(_tilt_of(bqd0)))
            rep["td_vz"] = float(bqd_d0[2])
        else:
            rep["td_found"] = False
        rep["airborne"] = airborne
        return td_snap, rep

    def _paw_max_z(self, body_q_np):
        """HIGHEST paw bottom - the numeric proxy for the paw rebound."""
        max_p = float("-inf")
        for b, c, r in self.paw_pts_py:
            xf = body_q_np[b]
            z = wp.transform_point(wp.transform(xf[:3], xf[3:]), c)[2] - r
            if z > max_p:
                max_p = z
        return max_p

    def _land_replay(self, ro, amps, snap, record=False, frame_offset=0):
        """Landing replay; the settled gate is evaluated at window end.
        Recorded playback frames are sampled every FRAME_EVERY steps counted
        from the jump phase's touchdown step (``frame_offset`` =
        ``rep["td_step"]``) so the frames stay evenly spaced across the
        touchdown boundary."""
        ro.loss(amps, snap)
        mink, minp = ro.min_knee_arr.numpy(), ro.min_paw_arr.numpy()
        rep = {"knee_margin_min": 1e9, "bounce": 0.0, "blew_up": False, "min_foot_rows": 10**9, "paw_lift_max": 0.0}
        m = None
        bqd = None
        for t in range(LAND_STEPS):
            if t % REPLAY_YIELD_EVERY == 0:
                yield
            rows = self._foot_row_count(ro.contacts[t // COLLIDE_EVERY])
            st = ro.states[t + 1]
            bqd = st.body_q.numpy()
            m = self._state_metrics(bqd[0], st.body_qd.numpy()[0])
            min_k, min_p = float(mink[t + 1]), float(minp[t + 1])
            rep["min_foot_rows"] = min(rep["min_foot_rows"], rows)
            if min_p < PAW_MASK_HEIGHT:
                rep["knee_margin_min"] = min(rep["knee_margin_min"], min_k)
            rep["paw_lift_max"] = max(rep["paw_lift_max"], self._paw_max_z(bqd))
            rep["bounce"] += max(m["vz"], 0.0) ** 2 * SIM_DT
            if record and (t + frame_offset) % FRAME_EVERY == 0:
                self.playback.append(("land", self._snapshot(st)))
            if not np.isfinite(m["z"]):
                rep["blew_up"] = True
                break
        rep["final_z"] = m["z"]
        rep["final_tilt_deg"] = float(np.degrees(m["tilt"]))
        rep["final_vz"] = m["vz"]
        rep["final_w"] = m["w"]
        rep["final_up_z"] = _up_z(bqd[0])
        rep["settled"] = bool(
            rep["final_tilt_deg"] <= 15.0
            and abs(rep["final_vz"]) <= 0.15
            and abs(rep["final_z"] - self.z_stand) <= 0.10
            and rep["final_w"] <= 0.6
        )
        return rep

    # --------------------------------------------------------------------------
    # 13. The chain - run_chain retries over crouch warm starts; the three
    #     phases and their gates
    # --------------------------------------------------------------------------
    def run_chain(self):
        """Phase A -> B -> C. Prints progress; stores the final result.

        A generator that yields after every rollout (see ``step()``).

        Contact makes the takeoff and the landing chaotic: two crouches that
        differ only in the last decimals can launch measurably different
        spins (the push is torque-saturated, so the jump amplitudes cannot
        undo that), and some touchdowns are much harder to catch than
        others. When an attempt misses a gate, the chain is retried from
        the next crouch warm start in ``CROUCH_WARM_SCALES``, and the best
        attempt is kept."""
        best = None
        for attempt, scale in enumerate(CROUCH_WARM_SCALES):
            if attempt:
                print(f"== retrying the chain, crouch warm start x{scale:g} (attempt {attempt + 1}) ==", flush=True)
            res = yield from self._attempt(CROUCH_WARM * scale)
            if best is None or res["score"] > best["score"]:
                best = res
            if res["crouch_ok"] and res["flip_ok"] and res["land_ok"]:
                break
        self.playback = best["playback"]
        self.result = best["result"]
        self.chain_done = True
        print(
            f"== chain done: crouch {'PASS' if best['crouch_ok'] else 'FAIL'}, "
            f"flip {'PASS' if best['flip_ok'] else 'FAIL'}, landing {'PASS' if best['land_ok'] else 'FAIL'} ==",
            flush=True,
        )

    def _attempt(self, crouch_warm):
        """One A -> B -> C attempt; returns its gates, score, result and playback.

        Each phase runs only when the previous one passed its gate, so a
        missed gate costs little before the next attempt. The score ranks
        attempts by the number of gates passed, then by the flip error."""
        res = {"crouch_ok": False, "flip_ok": False, "land_ok": False, "playback": [], "score": (0, -np.inf)}
        res["crouch_ok"], crouch_snap = yield from self._crouch_phase(crouch_warm)
        res["result"] = {"crouch_ok": res["crouch_ok"], "jump_ok": False}
        if not res["crouch_ok"]:
            return res
        td_snap, rep_b = yield from self._jump_phase(crouch_snap)
        res["playback"] = self.playback
        if td_snap is None:
            print("  jump phase never achieved a clean touchdown - skipping the landing phase", flush=True)
            res["score"] = (1, -np.inf)
            return res
        flip_err = abs(rep_b["td_pitch_deg"] - FLIP_SIGN * 360.0)
        res["flip_ok"] = flip_err <= 10.0 and rep_b["td_tilt_deg"] <= 10.0
        res["result"] = {"crouch_ok": True, "jump_ok": True, "td_pitch_deg": rep_b["td_pitch_deg"]}
        res["score"] = (1 + int(res["flip_ok"]), -flip_err)
        if not res["flip_ok"]:
            return res
        rep_c = yield from self._landing_phase(td_snap, rep_b["td_step"])
        res["land_ok"] = rep_c["settled"]
        res["result"] = {
            "crouch_ok": True,
            "jump_ok": True,
            "td_pitch_deg": rep_b["td_pitch_deg"],
            "td_tilt_deg": rep_b["td_tilt_deg"],
            "land_settled": rep_c["settled"],
            "land_bounce": float(rep_c["bounce"]),
            "land_paw_lift": float(rep_c["paw_lift_max"]),
        }
        res["score"] = (2 + int(res["land_ok"]), -flip_err)
        return res

    def _crouch_phase(self, warm):
        """Phase A: optimize the crouch from the settled stance, starting at
        the ``warm`` amplitudes; returns (gates passed, crouch-end snapshot)."""
        print("== crouch phase ==", flush=True)
        ro = Rollout(self, "crouch", CROUCH_ROLLOUT_STEPS, CROUCH_N_PARAMS)
        amps = yield from self._descent(
            "crouch", ro, self.snap, warm, CROUCH_AMP_LOW, CROUCH_AMP_HIGH, CROUCH_ITERS, CROUCH_RADII
        )
        end_snap, rep = yield from self._crouch_replay(ro, amps)
        ok = (
            rep["z_end"] <= 0.375
            and rep["tilt_end_deg"] < 20.0
            and rep["min_foot_rows"] > 0
            and rep["foot_rows_end"] > 0
            and rep["knee_min"] >= KNEE_MARGIN_GROUND - 0.005
            and rep["push_margin"] >= -0.002
        )
        print(
            f"  => crouch: z_end={rep['z_end']:.4f} (target {Z_TARGET:.3f})  tilt={rep['tilt_end_deg']:.1f}deg  "
            f"rows_min={rep['min_foot_rows']}  {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
        return ok, end_snap

    def _jump_phase(self, crouch_snap):
        """Phase B: optimize the jump from the crouch end; records the flight
        playback and returns (touchdown snapshot or None, replay report)."""
        print("== jump phase ==", flush=True)
        ro = Rollout(self, "jump", JUMP_STEPS, JUMP_SEGMENTS * 2)
        low = np.full(JUMP_SEGMENTS * 2, JUMP_AMP_LOW)
        high = np.full(JUMP_SEGMENTS * 2, JUMP_AMP_HIGH)
        amps = yield from self._descent(
            "jump", ro, crouch_snap, np.zeros(JUMP_SEGMENTS * 2), low, high, JUMP_ITERS, JUMP_RADII
        )
        self.playback = []  # the flight frames; phase C appends the landing
        td_snap, rep = yield from self._jump_replay(ro, amps, crouch_snap, record=True)
        print(
            f"  => jump: apex={rep['apex']:.3f}  pitch_end={rep['pitch_end_deg']:.1f}deg  "
            f"td={'YES' if td_snap is not None else 'NO'}"
            + (f"  td tilt={rep['td_tilt_deg']:.1f}deg vz={rep['td_vz']:+.2f}" if td_snap is not None else ""),
            flush=True,
        )
        return td_snap, rep

    def _landing_phase(self, td_snap, td_step):
        """Phase C: optimize the landing from the touchdown state; records the
        landing playback and returns the replay report.

        Curriculum: the dog still carries the flip's pitch rate at touchdown,
        and whether it catches itself or tumbles onto its back is decided in
        the first few hundred milliseconds. Over the full window, a tumbling
        start gives a gradient dominated by the chaotic tumble, so a first
        stage optimizes a short window alone and a second refines the result
        over the full window. Which window length works best depends on the
        touchdown, so the attempts in ``LAND_SHORT_WINDOWS`` (0 = no short
        stage) are tried in order until one settles; the best one is kept."""
        print("== landing phase ==", flush=True)
        ro = Rollout(self, "land", LAND_STEPS, LAND_N_PARAMS)
        best = None
        for n_short in LAND_SHORT_WINDOWS:
            amps = LAND_INIT
            if n_short:
                ro_short = Rollout(self, "land", n_short, LAND_N_PARAMS)
                amps = yield from self._descent(
                    f"land-{n_short}", ro_short, td_snap, amps, LAND_LOW, LAND_HIGH, LAND_ITERS, LAND_RADII
                )
                del ro_short
                # hold the pose of the last knot inside the short window for
                # the rest of the landing, instead of springing back to the
                # untouched stand targets the later knots still hold
                last = int(np.searchsorted(LAND_KNOTS_MS, n_short * SIM_DT * 1000.0)) - 1
                amps = amps.reshape(12, LAND_N_KNOTS - 1)
                amps[:, last + 1 :] = amps[:, last : last + 1]
                amps = amps.reshape(-1)
            amps = yield from self._descent("land", ro, td_snap, amps, LAND_LOW, LAND_HIGH, LAND_ITERS, LAND_RADII)
            rep = yield from self._land_replay(ro, amps, td_snap)
            score = (rep["settled"], -rep["bounce"])
            if best is None or score > best[0]:
                best = (score, amps)
            if rep["settled"]:
                break
            print("  landing did not settle - retrying with the next curriculum window", flush=True)
        rep = yield from self._land_replay(ro, best[1], td_snap, record=True, frame_offset=td_step)
        print(
            f"  => landing: z={rep['final_z']:.3f} tilt={rep['final_tilt_deg']:.1f}deg "
            f"bounce={rep['bounce']:.4f} pawlift={rep['paw_lift_max']:.3f} "
            f"{'SETTLED' if rep['settled'] else 'NOT SETTLED'}",
            flush=True,
        )
        return rep

    # --------------------------------------------------------------------------
    # 14. Runner interface - step (frame budget), render, --rec close, CLI
    #     arguments
    # --------------------------------------------------------------------------
    def step(self):
        """Framework hook: advance the optimization (skipped when done or in --test mode).

        Each frame runs rollouts for about ``STEP_BUDGET_S`` and then returns,
        so an interactive viewer keeps rendering (the current playback, or
        the settled stance) and can pause the optimization. Viewers that stop
        after a fixed number of frames finish it in a single frame."""
        if self.chain_done or self.args.test:
            # --test runs ONLY the gradient check (test_final); the full
            # optimization is the interactive run's purpose
            return
        if self._chain is None:
            self._chain = self.run_chain()
        start = time.perf_counter()
        for _ in self._chain:
            if not self._finish_in_one_frame and time.perf_counter() - start >= STEP_BUDGET_S:
                return

    def _rec_close(self):
        """Write one full playback pass to the recording and close it; safe to call repeatedly.

        The recording is fed from the cached playback at close time instead of
        mirroring render(), so it always holds exactly one full pass and never
        the settled stance - regardless of how many frames the viewer loop
        reached (a `--viewer null` run defaults to 100 frames, fewer than one
        pass contains) or how the run exited."""
        if self.recorder is None or self.rec_done:
            return
        if not self.playback:
            print("recording: no optimized playback was produced; nothing saved", flush=True)
        for k, (_phase, snap) in enumerate(self.playback):
            st = self._fresh(snap)
            self.recorder.begin_frame(k * SIM_DT * FRAME_EVERY)
            self.recorder.log_state(st)
            self.recorder.end_frame()
        self.recorder.close()
        self.rec_done = True

    def render(self):
        """Framework hook: draw the settled stance or the optimized playback, looping it."""
        t = self.frame * SIM_DT * FRAME_EVERY
        if not self.playback:
            # nothing optimized yet (or --test): render the settled stance
            st = self._fresh(self.snap)
            self.viewer.begin_frame(t)
            self.viewer.log_state(st)
            self.viewer.end_frame()
            self.frame += 1
            return
        _ph, snap = self.playback[min(self.frame, len(self.playback) - 1)]
        st = self._fresh(snap)
        self.viewer.begin_frame(t)
        self.viewer.log_state(st)
        self.viewer.end_frame()
        self.frame += 1
        if self.frame >= len(self.playback):
            self.frame = 0  # loop the achieved behavior

    @staticmethod
    def create_parser():
        """Register this example's CLI arguments on top of the shared newton example parser."""
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--rec",
            type=str,
            default=None,
            help="Save the optimized playback to this ViewerFile recording (.bin or .json).",
        )
        return parser

    # --------------------------------------------------------------------------
    # 15. Gradient check - test_final: the proof the differentiability claims
    #     hold (FD vs tape probe, _probe_loss_kernel)
    # --------------------------------------------------------------------------
    def test_final(self):
        """Gradient check: a 2-step landing-servo rollout from the settled
        stance, linear probe loss; tape gradient vs central finite differences
        on a spread of joint channels (skips the optimization)."""
        model = self.model
        solver = self.solver
        cts = self.cts
        snap = self.snap
        channels = [(0, 5), (0, 9), (0, 16), (1, 3), (1, 4), (1, 16)]
        amp = 0.07  # off zero so the spline path is exercised

        def rollout(kick_arr, kick_idx, kick):
            """Two-step landing-servo rollout from the settled stance, with an optional kick."""
            amps = np.full(LAND_N_PARAMS, amp, dtype=np.float32)
            amps_arr = wp.array(amps, dtype=wp.float32)
            st0 = self._fresh(snap)
            st1 = model.state()
            if kick_arr == 0:
                q = snap[0].copy()
                q[kick_idx] += kick
                st0.joint_q.assign(q)
            else:
                qd = snap[1].copy()
                qd[kick_idx] += kick
                st0.joint_qd.assign(qd)
            ctrl = model.control()
            for t in range(2):
                if t % COLLIDE_EVERY == 0:
                    self.pipe.collide(st0, cts)
                wp.launch(
                    servo_land_spline_kernel,
                    dim=12,
                    inputs=[
                        amps_arr,
                        st0.joint_q,
                        st0.joint_qd,
                        ctrl.joint_f,
                        self.stand_arr,
                        self.dofs_arr,
                        self.land_knots_arr,
                        LAND_N_KNOTS,
                        SIM_DT * 1000.0,
                        t,
                        TAU_MAX,
                        _bracket(t * SIM_DT * 1000.0),
                    ],
                )
                st0.clear_forces()
                solver.step(st0, st1, ctrl, cts, SIM_DT)
                st0, st1 = st1, st0
            jq = st0.joint_q.numpy().astype(np.float64)
            return float(1.0 * jq[9] + 0.75 * jq[13] - 0.5 * jq[7] + 0.25 * jq[16])

        # analytic: 2-step tape from the same state
        amps = np.full(LAND_N_PARAMS, amp, dtype=np.float32)
        params = wp.array(amps, dtype=wp.float32, requires_grad=True)
        states = [model.state(requires_grad=True) for _ in range(3)]
        states[0].joint_q.assign(snap[0])
        states[0].joint_qd.assign(snap[1])
        states[0].body_q.assign(snap[2])
        states[0].body_qd.assign(snap[3])
        # same freshness discipline as the phase tapes (see Rollout): the
        # servo writes per-step controls and collide writes its own buffer
        ctrls = [model.control(requires_grad=True) for _ in range(2)]
        cts_tape = self.pipe.contacts()
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        tape = wp.Tape()
        with tape:
            for t in range(2):
                if t % COLLIDE_EVERY == 0:
                    self.pipe.collide(states[t], cts_tape)
                wp.launch(
                    servo_land_spline_kernel,
                    dim=12,
                    inputs=[
                        params,
                        states[t].joint_q,
                        states[t].joint_qd,
                        ctrls[t].joint_f,
                        self.stand_arr,
                        self.dofs_arr,
                        self.land_knots_arr,
                        LAND_N_KNOTS,
                        SIM_DT * 1000.0,
                        t,
                        TAU_MAX,
                        _bracket(t * SIM_DT * 1000.0),
                    ],
                )
                states[t].clear_forces()
                solver.step(states[t], states[t + 1], ctrls[t], cts_tape, SIM_DT)
            jq = states[2].joint_q
            wp.launch(
                _probe_loss_kernel,
                dim=1,
                inputs=[jq, loss],
            )
        tape.backward(loss)
        g_q = states[0].joint_q.grad.numpy().astype(np.float64)
        g_qd = states[0].joint_qd.grad.numpy().astype(np.float64)
        tape.zero()

        eps = 3e-4
        for arr_idx, c in channels:
            fd = (rollout(arr_idx, c, +eps) - rollout(arr_idx, c, -eps)) / (2 * eps)
            gv = (g_q if arr_idx == 0 else g_qd)[c]
            # 15% relative bar; the 5e-5 absolute bar covers channels at the
            # float32 noise floor of a 2-step rollout
            err = abs(gv - fd)
            rel = err / max(abs(fd), 1e-12)
            assert rel <= 0.15 or err <= 5e-5, (
                f"grad check {'q' if arr_idx == 0 else 'qd'}[{c}]: tape={gv:.5e} fd={fd:.5e} rel={rel:.2e} abs={err:.2e}"
            )
            print(
                f"grad check {'q' if arr_idx == 0 else 'qd'}[{c}]: tape={gv:.5e} fd={fd:.5e} rel={rel:.2e}", flush=True
            )

        print("gradient check passed", flush=True)

        if self.result.get("land_settled"):
            print(f"chain result: SETTLED landing, bounce={self.result['land_bounce']:.4f}", flush=True)


@wp.kernel
def _probe_loss_kernel(joint_q: wp.array[float], loss: wp.array[float]):
    loss[0] = 1.0 * joint_q[9] + 0.75 * joint_q[13] - 0.5 * joint_q[7] + 0.25 * joint_q[16]


# -----------------------------------------------------------------------------
# 16. CLI entry point - __main__: --rec validation, construction, guaranteed
#     recording close
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    if args.test and args.rec:
        # nothing is optimized in test mode, so the recording could only ever
        # hold the settled stance - reject the combination before any work
        parser.error("--rec is not supported together with --test")
    if args.rec:
        # mirror ViewerFile's save-time validation (_get_serialization_format
        # in newton/_src/viewer/viewer_file.py) here, so an unusable path is
        # rejected before the long run instead of silently producing no
        # recording at save time
        ext = os.path.splitext(args.rec)[1].lower()
        if ext not in (".bin", ".json"):
            parser.error(f"--rec {args.rec}: unsupported extension '{ext}' (ViewerFile supports .bin and .json)")
        if ext == ".bin":
            try:
                import cbor2  # noqa: F401
            except ImportError:
                parser.error("--rec: .bin recordings require the cbor2 package (pip install 'cbor2>=5.7.0')")
    ex = Example(viewer, args)
    try:
        newton.examples.run(ex, args)
    finally:
        ex._rec_close()  # write the full playback pass and close on any exit path
