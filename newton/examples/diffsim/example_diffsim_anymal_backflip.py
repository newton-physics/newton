# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Diffsim Anymal Backflip

An example of how differentiable physics can be used in Newton to achieve
things: an ANYmal-D quadruped robot dog learns a complete standing backflip
- crouch, 360 degree flip in the air, and a controlled landing - through
gradient-based optimization, with every gradient computed by backpropagating
through the raw physics simulation via ``wp.Tape()``.

Three skills are optimized in sequence over single rollout windows
("one window, one skill" - no rolling windows):

1. Crouch (300 steps + a 100-step zero-amp push preview): PD-servo targets
   follow learned cosine ramps; the loss shapes crouch depth, stillness and
   a ground-based knee-clearance margin so the dog squats cleanly without
   scraping its knees.
2. Jump (1500 steps): per-joint-group servo trajectory amplitudes produce
   the launch and the full -360 degree pitch rotation. The loss prices the
   flip error, mid-air attitude, an apex height band and knee clearance,
   with touchdown-attention and airborne gates so terms switch on exactly
   when they are physically meaningful.
3. Landing (1600 steps): per-joint target SPLINES (non-uniform knots, dense
   around touchdown) absorb the ~4.7 m/s impact. The loss combines
   ground-based knee ordering, anti-bounce, a paw-glue term (every paw
   pinned to the ground - holding it at zero REQUIRES flexed knees, so the
   yielding-squat posture emerges instead of being prescribed) and an
   end-state stance term.

Gradients flow through SolverFeatherstone AND the differentiable contact
pipeline (soft_contact_gap margin; the collide schedule is mirrored between
the tape world and the flown world so both are bit-identical).

This example is LONG: one full run optimizes all three skills end to end
and takes roughly 45-60 minutes on a single GPU. That is the honest cost of
the behavior; nothing about the method is abbreviated. ``--test`` skips the
optimization entirely and runs only the gradient check (seconds).

Warp auto-adjoint pitfall: an in-kernel linear scan
(``k = 0; for kk in range(n): if t >= knots[kk]: k = kk``) is correct in the
forward, but the auto-adjoint mishandles loop-carried variables mutated
inside conditionals (k stays 0 in the backward here) and concentrates the
whole adjoint on one parameter. The active segment is therefore precomputed
on the host per launch (``_bracket``), which makes the adjoint exact. Avoid
loop-carried index variables inside differentiable kernels.

Success criteria:
- Phase A: a clean crouch (depth ~0.34 m, upright, knees clear of the ground).
- Phase B: the dog leaves the ground, completes a full -360 degree pitch
  rotation and touches down on its paws.
- Phase C: the dog settles standing after the impact (settled gate: tilt
  <= 15 deg, |vz| <= 0.15 m/s, |z - stand| <= 0.10 m, |w| <= 0.6 rad/s).
- When ``--test`` runs, numeric and analytic gradients agree within the
  check tolerance.

Runs:
- `python -m newton.examples diffsim_anymal_backflip`           (full run, ~45-60 min)
- `python -m newton.examples diffsim_anymal_backflip --test`    (gradient check only)
"""

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import GeoType, ShapeFlags

# ------------------------------------------------------------------------------
# simulation parameters and objective weights (single-source constants - the
# configuration the demonstrated behavior was achieved with)
# ------------------------------------------------------------------------------

SIM_DT = 1.0e-3
STAND_LEG = (0.0, 0.2, -0.4)
STAND = np.tile(np.array(STAND_LEG, dtype=np.float32), 4)
HINGE_DOFS = np.arange(6, 18, dtype=np.int32)
FOOT_BODIES = (3, 6, 9, 12)
KE_SCALE = 50.0  # contact stiffness realism multiplier
KD_SCALE = 4.0  # contact damping multiplier (ke * sqrt(KE_SCALE) * KD_SCALE)

# objective weights (the configuration the demonstrated behavior was
# achieved with; a search over these weights is possible but out of scope)
Z_GOOD = 1.7463734824840804  # apex ceiling of the height band (m)
C_FLIP = 0.956137871751204  # flip-error weight at touchdown attention
KNEE_MARGIN_G = 0.0217610642328222  # ground-based knee clearance bar (m)
Z_TARGET = 0.32296679716115667  # crouch depth target (m)
C_ARR = 0.4918151418204104  # arrival quality weight
Z_LOW = Z_GOOD - 0.35  # apex floor of the height band (m)
C_BAND = 30.0
FOOT_TD = 0.04  # touchdown attention center (paw height, m)
TD_SIG = 0.015  # touchdown attention sharpness (m)
LIFT_H = 0.05  # airborne gate: prefix-max paw height for "has flown"
FLIP_SIGN = -1.0  # backflip sign about world +y
W_ORDER_A = 10.0  # crouch per-state knee-ordering weight
W_ORDER_B = 40.0  # jump per-state knee-ordering weight
W_ORDER_C = 20.0  # land per-state knee-ordering weight
W_PAWS = 40.0  # paw-glue weight (the yielding-knee mechanism)
W_UP = 1.0  # mid-window upright weight
A1, B1, C1 = 30.0, 2.0, 6.0  # crouch end terms
A2, B2, C2, D2, D3 = 100.0, 0.1, 6.0, 0.5, 15.0  # land end + anti-bounce
TWO_PI = 2.0 * np.pi

# phase A: crouch
CR_N = 300
CR_PREVIEW = 100
CR_N_EXT = CR_N + CR_PREVIEW
CR_ROUNDS, CR_ITERS = 4, 16
CR_NP = 8  # cosine amps (4 channels) + ramp times (4 channels)
CR_T_REF = 249.0
CR_WARM = np.array([5.2, 3.8, 4.5, 4.5, 1.0, 1.0, 1.0, 1.0])  # hip-dominant
CR_AMP_LO = np.array([-1.0, -1.0, -1.0, -1.0, 0.1, 0.1, 0.1, 0.1])
CR_AMP_HI = np.array([5.5, 5.5, 5.5, 5.5, 1.2, 1.2, 1.2, 1.2])

# phase B: jump
N_B, NSEG_B = 1500, 15
B_SPS = N_B // NSEG_B
JB_ITERS, JB_ROUNDS = 16, 2
JB_R0, JB_R_MAX = 5.0e-3, 8.0e-2
B_AMP_LO, B_AMP_HI = -0.9, 3.0
TAUF, TAUH = 450.0, 300.0  # jump torque caps (front/hind)
TAU_MAX = 300.0  # land servo torque clamp
GROUP_OF_JOINT = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.int32)  # LF,RF front; LH,RH hind
CHAN_OF_JOINT = np.array([GROUP_OF_JOINT[j] * 2 + (1 if j % 3 == 2 else 0) for j in range(12)], dtype=np.int32)

# phase C: landing (per-joint direct-target splines, C1 smoothstep, dense
# around touchdown; past the last knot the final values are held)
N_C, NSEG_C = 1600, 8
C_SPS = N_C // NSEG_C
LC_ITERS, LC_ROUNDS = 10, 2
PROF_KNOTS_MS = [0, 20, 40, 60, 90, 120, 160, 200, 260, 320, 400, 500, 650, 850, 1150, 1600]
PROF_N_KNOTS = len(PROF_KNOTS_MS)
LAND_NPAR = 12 * (PROF_N_KNOTS - 1)
_JOINT_LO = (-0.7, -1.2, -2.4)  # HAA, HFE, KFE
_JOINT_HI = (0.7, 0.9, -0.05)  # knee kept from locking straight
LAND_LO = np.tile(np.repeat(_JOINT_LO, PROF_N_KNOTS - 1).astype(np.float64), 4)
LAND_HI = np.tile(np.repeat(_JOINT_HI, PROF_N_KNOTS - 1).astype(np.float64), 4)
LAND_INIT = np.tile(np.repeat((0.0, 0.2, -0.4), PROF_N_KNOTS - 1).astype(np.float64), 4)

COLLIDE_EVERY = 2  # contact-truthful tapes; replays use the SAME schedule
R_MIN, R_MAX = 1.0e-9, 5.0e-2
PAW_MASK_H = 0.02  # "grounded" mask: lowest paw within 2 cm of ground
VZ_LAG_M = 5
INV_MDT = 1.0 / (VZ_LAG_M * SIM_DT)
FRAME_EVERY = 8  # one playback frame per this many simulated steps


def _bracket(t_ms):
    """Landing-spline segment index: LAST knot with knots_ms[kk] <= t_ms,
    clamped to [0, n_knots-2]. Computed on the HOST per launch - see the
    module docstring for why this must not live inside the kernel."""
    k = int(np.searchsorted(np.asarray(PROF_KNOTS_MS, dtype=np.float64), float(t_ms), side="right")) - 1
    return max(0, min(k, PROF_N_KNOTS - 2))


# ------------------------------------------------------------------------------
# Warp kernels: servo laws
# ------------------------------------------------------------------------------


@wp.kernel
def servo_hold_kernel(
    params: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_f: wp.array[float],
    stand: wp.array[float],
    hinge_dofs: wp.array[int],
    mode: int,
):
    """Settle servo: mode 0 = hold stand, mode 1 = stand + params[0] pulse."""
    j = wp.tid()
    dof = hinge_dofs[j]
    q = joint_q[7 + j]
    qd = joint_qd[dof]
    if mode == 0:
        target = stand[j]
    else:
        target = stand[j] * (1.0 + params[0])
    joint_f[dof] = 2000.0 * (target - q) - 15.0 * qd


@wp.kernel
def servo_crouch_cos_kernel(
    params: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_f: wp.array[float],
    stand: wp.array[float],
    hinge_dofs: wp.array[int],
    chan_of_joint: wp.array[int],
    t: int,
    ramp_ref: float,
    tau_cap: float,
):
    j = wp.tid()
    dof = hinge_dofs[j]
    q = joint_q[7 + j]
    qd = joint_qd[dof]
    ch = chan_of_joint[j]
    amp_a = params[ch]
    tau_p = params[4 + ch]
    ramp_t = ramp_ref * wp.max(tau_p, 0.05)
    u = wp.min(float(t) / ramp_t, 1.0)
    target = stand[j] * (1.0 + amp_a * 0.5 * (1.0 - wp.cos(3.14159265 * u)))
    tau = 2000.0 * (target - q) - 15.0 * qd
    joint_f[dof] = wp.clamp(tau, -tau_cap, tau_cap)


@wp.kernel
def servo_traj_group_caps_kernel(
    seg_amp: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_f: wp.array[float],
    stand: wp.array[float],
    hinge_dofs: wp.array[int],
    group_of_joint: wp.array[int],
    seg: int,
    tau_caps: wp.array[float],
):
    j = wp.tid()
    dof = hinge_dofs[j]
    q = joint_q[7 + j]
    qd = joint_qd[dof]
    g = group_of_joint[j]
    target = stand[j] * (1.0 + seg_amp[seg * 2 + g])
    tau = 2000.0 * (target - q) - 15.0 * qd
    cap = tau_caps[g]
    joint_f[dof] = wp.clamp(tau, -cap, cap)


@wp.kernel
def servo_profile_kernel(
    seg_amp: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_f: wp.array[float],
    stand: wp.array[float],
    hinge_dofs: wp.array[int],
    knots_ms: wp.array[float],
    n_knots: int,
    dt_ms: float,
    t: int,
    tau_max: float,
    k: int,
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
    tau = 2000.0 * (target - q) - 15.0 * qd
    joint_f[dof] = wp.clamp(tau, -tau_max, tau_max)


# ------------------------------------------------------------------------------
# Warp kernels: per-state sample passes + losses (each sample index written
# exactly once; loss kernels read read-only with internal prefix sums)
# ------------------------------------------------------------------------------


@wp.kernel
def wy_state_kernel(body_qd: wp.array[wp.spatial_vector], wy_arr: wp.array[float], idx: int, dt: float):
    w = wp.spatial_bottom(body_qd[0])
    wy_arr[idx] = w[1] * dt


@wp.kernel
def zraw_state_kernel(body_q: wp.array[wp.transform], z_arr: wp.array[float], idx: int):
    z = wp.transform_get_translation(body_q[0])[2]
    z_arr[idx] = z


@wp.kernel
def pawraw_state_kernel(
    body_q: wp.array[wp.transform],
    paw_body: wp.array[int],
    paw_center: wp.array[wp.vec3],
    paw_r: wp.array[float],
    n_paw: int,
    paw_arr: wp.array[float],
    idx: int,
):
    min_p = float(1.0e9)
    for k in range(n_paw):
        c = wp.transform_point(body_q[paw_body[k]], paw_center[k])
        if c[2] - paw_r[k] < min_p:
            min_p = c[2] - paw_r[k]
    paw_arr[idx] = min_p


@wp.kernel
def order_state_g_kernel(
    body_q: wp.array[wp.transform],
    pts_body: wp.array[int],
    pts_local: wp.array[wp.vec3],
    n_pts: int,
    paw_body: wp.array[int],
    paw_center: wp.array[wp.vec3],
    paw_r: wp.array[float],
    n_paw: int,
    w_order: float,
    margin_g: float,
    mask_h: float,
    loss: wp.array[float],
):
    """Ground-based knee ordering: while paws are grounded, the lowest knee
    point must stay above the GROUND margin."""
    min_k = float(1.0e9)
    for k in range(n_pts):
        p = wp.transform_point(body_q[pts_body[k]], pts_local[k])
        if p[2] < min_k:
            min_k = p[2]
    min_p = float(1.0e9)
    for k in range(n_paw):
        c = wp.transform_point(body_q[paw_body[k]], paw_center[k])
        if c[2] - paw_r[k] < min_p:
            min_p = c[2] - paw_r[k]
    mask = wp.max(1.0 - min_p / mask_h, 0.0)
    viol = wp.max(margin_g - min_k, 0.0)
    loss[0] = loss[0] + w_order * mask * viol * viol


@wp.kernel
def loss_crouch_end_kernel(
    body_q_n: wp.array[wp.transform],
    body_q_m: wp.array[wp.transform],
    loss: wp.array[float],
    z_target: float,
    a1: float,
    b1: float,
    c1: float,
    inv_mdt: float,
):
    zn = wp.transform_get_translation(body_q_n[0])
    zm = wp.transform_get_translation(body_q_m[0])
    vz = (zn[2] - zm[2]) * inv_mdt
    q = wp.transform_get_rotation(body_q_n[0])
    up = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    loss[0] = loss[0] + a1 * wp.pow(zn[2] - z_target, 2.0) + b1 * vz * vz + c1 * (1.0 - up[2])


@wp.kernel
def loss_jump_state_kernel(
    body_q: wp.array[wp.transform],
    z_arr: wp.array[float],
    wy_arr: wp.array[float],
    paw_arr: wp.array[float],
    idx: int,
    pts_body: wp.array[int],
    pts_local: wp.array[wp.vec3],
    n_pts: int,
    paw_body: wp.array[int],
    paw_center: wp.array[wp.vec3],
    paw_r: wp.array[float],
    n_paw: int,
    loss: wp.array[float],
    c_flip: float,
    c_att: float,
    w_order: float,
    foot_td: float,
    td_sig: float,
    margin_g: float,
    mask_h: float,
    flip_sign: float,
    two_pi: float,
    lift_h: float,
):
    pitch = float(0.0)
    zmax = float(-1.0e9)
    paw_hi = float(-1.0e9)
    for s in range(idx + 1):
        pitch += wy_arr[s]
        if z_arr[s] > zmax:
            zmax = z_arr[s]
        if paw_arr[s] > paw_hi:
            paw_hi = paw_arr[s]
    min_k = float(1.0e9)
    for k in range(n_pts):
        p = wp.transform_point(body_q[pts_body[k]], pts_local[k])
        if p[2] < min_k:
            min_k = p[2]
    min_p = float(1.0e9)
    for k in range(n_paw):
        c = wp.transform_point(body_q[paw_body[k]], paw_center[k])
        if c[2] - paw_r[k] < min_p:
            min_p = c[2] - paw_r[k]
    q = wp.transform_get_rotation(body_q[0])
    up = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    a = wp.clamp((foot_td - min_p) / td_sig, -30.0, 30.0)
    w_att = 1.0 / (1.0 + wp.exp(-a))
    # airborne gate: the flip error at pre-liftoff grounded states is
    # physically unfixable and its tax drowns the gradient; post-touchdown
    # states stay gated ON (paw_hi keeps its max), so over-rotation keeps
    # being priced through the recovery
    ag = wp.clamp((paw_hi - lift_h) / 0.01, -30.0, 30.0)
    air = 1.0 / (1.0 + wp.exp(-ag))
    flip_err = pitch - flip_sign * two_pi
    mask_g = wp.max(1.0 - min_p / mask_h, 0.0)
    viol_g = wp.max(margin_g - min_k, 0.0)
    loss[0] = (
        loss[0]
        + c_flip * w_att * air * flip_err * flip_err
        + c_att * w_att * air * (1.0 - up[2])
        + w_order * mask_g * viol_g * viol_g
    )


@wp.kernel
def loss_jump_band_kernel(
    z_arr: wp.array[float],
    n: int,
    loss: wp.array[float],
    c_band: float,
    z_good: float,
    z_low: float,
):
    zmax = float(-1.0e9)
    for s in range(n + 1):
        if z_arr[s] > zmax:
            zmax = z_arr[s]
    loss[0] = loss[0] + c_band * (wp.pow(wp.max(zmax - z_good, 0.0), 2.0) + wp.pow(wp.max(z_low - zmax, 0.0), 2.0))


@wp.kernel
def loss_land_state_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    pts_body: wp.array[int],
    pts_local: wp.array[wp.vec3],
    n_pts: int,
    paw_body: wp.array[int],
    paw_center: wp.array[wp.vec3],
    paw_r: wp.array[float],
    n_paw: int,
    loss: wp.array[float],
    w_order: float,
    margin_g: float,
    mask_h: float,
    d3: float,
    w_paws: float,
    w_up: float,
):
    """Per-state land terms: ground-based knee ordering + anti-bounce +
    paw-glue + upright. Holding the paw-glue at zero REQUIRES flexed knees -
    the yielding-squat is the only feasible solution, it is not prescribed.

    The terms accumulate over every state of the window. (A variant that
    starts the window before touchdown would need to gate them on
    t >= first-contact so airborne states contribute no loss; this example
    starts the window at touchdown, so every state is post-contact.)"""
    min_k = float(1.0e9)
    for k in range(n_pts):
        p = wp.transform_point(body_q[pts_body[k]], pts_local[k])
        if p[2] < min_k:
            min_k = p[2]
    min_p = float(1.0e9)
    pp = float(0.0)
    for k in range(n_paw):
        c = wp.transform_point(body_q[paw_body[k]], paw_center[k])
        h = c[2] - paw_r[k]
        if h < min_p:
            min_p = h
        hu = wp.max(h, 0.0)
        pp = pp + hu * hu
    mask_g = wp.max(1.0 - min_p / mask_h, 0.0)
    viol_g = wp.max(margin_g - min_k, 0.0)
    vz = wp.spatial_top(body_qd[0])[2]
    q = wp.transform_get_rotation(body_q[0])
    up = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    loss[0] = (
        loss[0]
        + w_order * mask_g * viol_g * viol_g
        + d3 * wp.pow(wp.max(vz, 0.0), 2.0)
        + w_paws * pp
        + w_up * (1.0 - up[2])
    )


@wp.kernel
def loss_land_end_kernel(
    body_q_n: wp.array[wp.transform],
    body_qd_n: wp.array[wp.spatial_vector],
    loss: wp.array[float],
    z_target: float,
    c_arr: float,
    a2: float,
    b2: float,
    c2: float,
    d2: float,
):
    zn = wp.transform_get_translation(body_q_n[0])
    q = wp.transform_get_rotation(body_q_n[0])
    up = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    v = wp.spatial_top(body_qd_n[0])
    w = wp.spatial_bottom(body_qd_n[0])
    loss[0] = loss[0] + c_arr * (
        a2 * wp.pow(zn[2] - z_target, 2.0) + b2 * wp.dot(w, w) + c2 * (1.0 - up[2]) + d2 * v[2] * v[2]
    )


# ------------------------------------------------------------------------------
# numpy twins of the kernels (eager losses, metrics)
# ------------------------------------------------------------------------------


def _quat_rot_np(q, v):
    """Rotate v by quaternion (xyzw order)."""
    qx, qy, qz = q[:3]
    t = 2.0 * np.cross(np.array([qx, qy, qz]), v)
    return v + q[3] * t + np.cross(np.array([qx, qy, qz]), t)


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_conj(a):
    return np.array([-a[0], -a[1], -a[2], a[3]])


def _rotvec_incr(q_prev, q_cur):
    """Rotation-vector increment between two quaternions. The jump replay
    integrates pitch from ORIENTATION deltas rather than angular velocity x
    dt - the two estimates differ slightly and the round score prices the
    accumulated pitch."""
    dq = _quat_mul(q_cur, _quat_conj(q_prev))
    v = dq[:3]
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    return 2.0 * float(np.arctan2(n, dq[3])) * v / n


def _tilt_of(b):
    qx, qy = float(b[3]), float(b[4])
    return float(np.arccos(np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)))


def _up_z_of_np(b):
    return 1.0 - 2.0 * (float(b[3]) ** 2 + float(b[4]) ** 2)


# ------------------------------------------------------------------------------
# Example
# ------------------------------------------------------------------------------


class Example:
    """ANYmal-D standing backflip: crouch -> -360 deg jump -> landing."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame = 0
        self.chain_done = False
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
        robot.joint_q[0:3] = (0.0, 0.0, 0.66)
        robot.joint_q[6] = 1.0  # identity quaternion (w)
        robot.joint_q[7:19] = STAND.tolist()
        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        scene.replicate(robot, 1)
        scene.add_ground_plane(cfg=robot.default_shape_cfg)
        self.model = scene.finalize(requires_grad=True)

        # contact realism: stiffen/damp every shape material
        ke0 = float(self.model.shape_material_ke.numpy()[0])
        kd0 = float(self.model.shape_material_kd.numpy()[0])
        n_shapes = len(self.model.shape_material_ke.numpy())
        self.model.shape_material_ke.assign(np.full(n_shapes, ke0 * KE_SCALE, dtype=np.float32))
        self.model.shape_material_kd.assign(np.full(n_shapes, kd0 * np.sqrt(KE_SCALE) * KD_SCALE, dtype=np.float32))

        self.solver = newton.solvers.SolverFeatherstone(self.model)
        self.pipe = newton.CollisionPipeline(self.model, soft_contact_gap=10.0, requires_grad=True)
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
            if 1 <= shape_body[i] <= 12
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
                knee_pts_local.append(_quat_rot_np(sq, p) + spos)
                knee_pts_body.append(b)
        self.knee_pts_body_arr = wp.array(knee_pts_body, dtype=wp.int32)
        self.knee_pts_local_arr = wp.array(np.array(knee_pts_local, dtype=np.float32), dtype=wp.vec3)
        self.n_knee_pts = len(knee_pts_body)
        self.kp_body = np.array(knee_pts_body, dtype=np.int32)
        self.kp_local = np.array(knee_pts_local, dtype=np.float64)

        # paw points: the four foot spheres' lowest points (center - radius)
        paw_pts = [
            (int(shape_body[i]), shape_xf[i][:3].copy(), float(shape_scale[i][0])) for i in sorted(self.foot_shapes)
        ]
        self.paw_pts_py = paw_pts
        self.paw_body_arr = wp.array([p[0] for p in paw_pts], dtype=wp.int32)
        self.paw_center_arr = wp.array(np.array([p[1] for p in paw_pts], dtype=np.float32), dtype=wp.vec3)
        self.paw_r_arr = wp.array([p[2] for p in paw_pts], dtype=wp.float32)
        self.n_paw = len(paw_pts)

        # constant device arrays
        self.stand_arr = wp.array(STAND, dtype=wp.float32)
        self.dofs_arr = wp.array(HINGE_DOFS, dtype=wp.int32)
        self.group_arr = wp.array(GROUP_OF_JOINT, dtype=wp.int32)
        self.chan_arr = wp.array(CHAN_OF_JOINT, dtype=wp.int32)
        self.zero_amp_arr = wp.zeros(NSEG_B * 2, dtype=wp.float32)
        self.caps_j_arr = wp.array([TAUF, TAUH], dtype=wp.float32)
        self.prof_knots_arr = wp.array(np.array(PROF_KNOTS_MS, dtype=np.float32), dtype=wp.float32)
        self.ctrl = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.viewer.set_model(self.model)

        # ---- settle to a grounded stance (hold stand, then one pulse) ----
        s0 = self.model.state()
        s1 = self.model.state()
        s0 = self._settle_segment(s0, s1, 1200, mode=0)
        s0 = self._settle_segment(s0, s1, 100, mode=1, amp=-0.12)
        self.snap = self._snapshot(s0)
        self.z_stand = float(self.snap[2][0][2])
        print(f"settle done: z_stand={self.z_stand:.4f}", flush=True)

        self.loss_history = {"crouch": [], "jump": [], "land": []}

    # ---------------- execution helpers ----------------

    def _snapshot(self, st):
        return tuple(arr.numpy().copy() for arr in (st.joint_q, st.joint_qd, st.body_q, st.body_qd))

    def _fresh(self, snap, requires_grad=None):
        st = self.model.state(requires_grad=requires_grad)
        st.joint_q.assign(snap[0])
        st.joint_qd.assign(snap[1])
        st.body_q.assign(snap[2])
        st.body_qd.assign(snap[3])
        return st

    def _settle_segment(self, st0, st1, n, mode, amp=0.0):
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

    def _foot_row_count(self):
        n_c = int(self.cts.rigid_contact_count.numpy()[0])
        if n_c == 0:
            return 0
        s0 = self.cts.rigid_contact_shape0.numpy()[:n_c]
        s1 = self.cts.rigid_contact_shape1.numpy()[:n_c]
        return int(sum(1 for k in range(n_c) if s0[k] in self.foot_shapes or s1[k] in self.foot_shapes))

    def _min_zs(self, body_q_np):
        """(min knee-point z, min paw z) - numpy twin of the loss kernels."""
        out = np.empty(len(self.kp_local))
        for b in np.unique(self.kp_body):
            sel = np.where(self.kp_body == b)[0]
            xf = body_q_np[int(b)]
            x, y, z, w = xf[3:]
            R = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            out[sel] = (R @ self.kp_local[sel].T).T[:, 2] + xf[2]
        min_p = float("inf")
        for b, c, r in self.paw_pts_py:
            xf = body_q_np[b]
            pz = float(_quat_rot_np(xf[3:], c)[2]) + float(xf[2]) - r
            min_p = min(min_p, pz)
        return float(out.min()), min_p

    def _state_metrics(self, body_q0, body_qd0):
        return {
            "z": float(body_q0[2]),
            "tilt": _tilt_of(body_q0),
            "vz": float(body_qd0[2]),
            "w": float(np.linalg.norm(body_qd0[3:6])),
        }

    def _unroll(self, mode, n, amps, sps, snap):
        """Eager rollout (the eager twin of the tapes: same collide schedule).
        Returns per-step (body_q, body_qd) lists and the final state."""
        st0 = self._fresh(snap)
        st1 = self.model.state()
        amps_arr = wp.array(np.asarray(amps, dtype=np.float32), dtype=wp.float32)
        bqs = [st0.body_q.numpy().copy()]
        qds = [st0.body_qd.numpy().copy()]
        for t in range(n):
            if t % COLLIDE_EVERY == 0:
                self.pipe.collide(st0, self.cts)
            if mode == "crouch":
                if t < CR_N:
                    wp.launch(
                        servo_crouch_cos_kernel,
                        dim=12,
                        inputs=[
                            amps_arr,
                            st0.joint_q,
                            st0.joint_qd,
                            self.ctrl.joint_f,
                            self.stand_arr,
                            self.dofs_arr,
                            self.chan_arr,
                            t,
                            CR_T_REF,
                            TAU_MAX,
                        ],
                    )
                else:
                    # push preview: zero-amp rebound with jump torque caps
                    wp.launch(
                        servo_traj_group_caps_kernel,
                        dim=12,
                        inputs=[
                            self.zero_amp_arr,
                            st0.joint_q,
                            st0.joint_qd,
                            self.ctrl.joint_f,
                            self.stand_arr,
                            self.dofs_arr,
                            self.group_arr,
                            0,
                            self.caps_j_arr,
                        ],
                    )
            elif mode == "jump":
                nseg = len(amps) // 2
                wp.launch(
                    servo_traj_group_caps_kernel,
                    dim=12,
                    inputs=[
                        amps_arr,
                        st0.joint_q,
                        st0.joint_qd,
                        self.ctrl.joint_f,
                        self.stand_arr,
                        self.dofs_arr,
                        self.group_arr,
                        min(t // sps, nseg - 1),
                        self.caps_j_arr,
                    ],
                )
            else:
                wp.launch(
                    servo_profile_kernel,
                    dim=12,
                    inputs=[
                        amps_arr,
                        st0.joint_q,
                        st0.joint_qd,
                        self.ctrl.joint_f,
                        self.stand_arr,
                        self.dofs_arr,
                        self.prof_knots_arr,
                        PROF_N_KNOTS,
                        SIM_DT * 1000.0,
                        t,
                        TAU_MAX,
                        _bracket(t * SIM_DT * 1000.0),
                    ],
                )
            st0.clear_forces()
            self.solver.step(st0, st1, self.ctrl, self.cts, SIM_DT)
            st0, st1 = st1, st0
            bqs.append(st0.body_q.numpy().copy())
            qds.append(st0.body_qd.numpy().copy())
        return bqs, qds, st0

    def _eager_loss(self, mode):
        if mode == "crouch":

            def fn(n, amps, sps, snap):
                bqs, _qds, _ = self._unroll(mode, n, amps, sps, snap)
                total = 0.0
                for bq in bqs:
                    min_k, min_p = self._min_zs(bq)
                    mask = max(1.0 - min_p / PAW_MASK_H, 0.0)
                    viol = max(KNEE_MARGIN_G - min_k, 0.0)
                    total += W_ORDER_A * mask * viol * viol
                z_end = float(bqs[CR_N][0][2])
                vz = (float(bqs[CR_N][0][2]) - float(bqs[CR_N - VZ_LAG_M][0][2])) * INV_MDT
                tilt = _tilt_of(bqs[CR_N][0])
                return total + A1 * (z_end - Z_TARGET) ** 2 + B1 * vz * vz + C1 * (1.0 - np.cos(tilt))

            return fn
        if mode == "jump":

            def fn(n, amps, sps, snap):
                bqs, qds, _ = self._unroll(mode, n, amps, sps, snap)
                total = 0.0
                pitch = 0.0
                zmax = -1.0e9
                paw_hi = -1.0e9
                for t in range(n + 1):
                    pitch += float(qds[t][0][4]) * SIM_DT
                    z = float(bqs[t][0][2])
                    zmax = max(zmax, z)
                    min_k, min_p = self._min_zs(bqs[t])
                    paw_hi = max(paw_hi, min_p)
                    a = min(max((FOOT_TD - min_p) / TD_SIG, -30.0), 30.0)
                    w_att = 1.0 / (1.0 + np.exp(-a))
                    ag = min(max((paw_hi - LIFT_H) / 0.01, -30.0), 30.0)
                    air = 1.0 / (1.0 + np.exp(-ag))
                    flip_err = pitch - FLIP_SIGN * TWO_PI
                    upz = _up_z_of_np(bqs[t][0])
                    mask_g = max(1.0 - min_p / PAW_MASK_H, 0.0)
                    viol_g = max(KNEE_MARGIN_G - min_k, 0.0)
                    total += (
                        C_FLIP * w_att * air * flip_err * flip_err
                        + 3.0 * C_ARR * w_att * air * (1.0 - upz)
                        + W_ORDER_B * mask_g * viol_g * viol_g
                    )
                total += C_BAND * (max(zmax - Z_GOOD, 0.0) ** 2 + max(Z_LOW - zmax, 0.0) ** 2)
                return total

            return fn

        def fn(n, amps, sps, snap):
            bqs, qds, _ = self._unroll(mode, n, amps, sps, snap)
            total = 0.0
            for t in range(n + 1):
                min_k, min_p = self._min_zs(bqs[t])
                mask_g = max(1.0 - min_p / PAW_MASK_H, 0.0)
                viol_g = max(KNEE_MARGIN_G - min_k, 0.0)
                vz = float(qds[t][0][2])
                paws_up = 0.0
                for b, c, r in self.paw_pts_py:
                    xf = bqs[t][b]
                    pz = float(_quat_rot_np(xf[3:], c)[2]) + float(xf[2]) - r
                    paws_up += max(pz, 0.0) ** 2
                upz = _up_z_of_np(bqs[t][0])
                total += (
                    W_ORDER_C * mask_g * viol_g * viol_g
                    + D3 * max(vz, 0.0) ** 2
                    + W_PAWS * paws_up
                    + W_UP * (1.0 - upz)
                )
            bend = bqs[n][0]
            vzend = float(qds[n][0][2])
            upz = _up_z_of_np(bend)
            wend = float(np.linalg.norm(qds[n][0][3:6]))
            total += C_ARR * (
                A2 * (float(bend[2]) - self.z_stand) ** 2 + B2 * wend**2 + C2 * (1.0 - upz) + D2 * vzend**2
            )
            return total

        return fn

    def _tape_loss(self, mode):
        def fn(n, amps0, sps, snap):
            loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            params = wp.array(np.asarray(amps0, dtype=np.float32), dtype=wp.float32, requires_grad=True)
            states = [self.model.state(requires_grad=True) for _ in range(n + 1)]
            states[0].joint_q.assign(snap[0])
            states[0].joint_qd.assign(snap[1])
            states[0].body_q.assign(snap[2])
            states[0].body_qd.assign(snap[3])
            if mode == "jump":
                wy_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
                z_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
                paw_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
            # per-step controls and per-collide contacts: every buffer is
            # written at most once inside the tape. Sharing one buffer across
            # steps leaves the array holding a LATER step's values by the time
            # an earlier step's adjoint replay re-derives its primal locals
            # from it — the same staleness class the solver's shared J/M/P/H/L
            # scratch had (fixed in this PR, pinned by test_multistep_gradient);
            # wp.config.verify_autograd_array_access also flags the pattern.
            ctrls = [self.model.control(requires_grad=True) for _ in range(n)]
            cts_list = [self.pipe.contacts() for _ in range((n + COLLIDE_EVERY - 1) // COLLIDE_EVERY)]
            nseg = len(amps0) // (2 if mode == "jump" else 1)
            tape = wp.Tape()
            with tape:
                for t in range(n):
                    if t % COLLIDE_EVERY == 0:
                        cur_cts = cts_list[t // COLLIDE_EVERY]
                        self.pipe.collide(states[t], cur_cts)
                    if mode == "crouch":
                        if t < CR_N:
                            wp.launch(
                                servo_crouch_cos_kernel,
                                dim=12,
                                inputs=[
                                    params,
                                    states[t].joint_q,
                                    states[t].joint_qd,
                                    ctrls[t].joint_f,
                                    self.stand_arr,
                                    self.dofs_arr,
                                    self.chan_arr,
                                    t,
                                    CR_T_REF,
                                    TAU_MAX,
                                ],
                            )
                        else:
                            # push preview: zero-amp rebound with jump torque
                            # caps - the order term prices the rebound over
                            # these states too
                            wp.launch(
                                servo_traj_group_caps_kernel,
                                dim=12,
                                inputs=[
                                    self.zero_amp_arr,
                                    states[t].joint_q,
                                    states[t].joint_qd,
                                    ctrls[t].joint_f,
                                    self.stand_arr,
                                    self.dofs_arr,
                                    self.group_arr,
                                    0,
                                    self.caps_j_arr,
                                ],
                            )
                    elif mode == "jump":
                        wp.launch(
                            servo_traj_group_caps_kernel,
                            dim=12,
                            inputs=[
                                params,
                                states[t].joint_q,
                                states[t].joint_qd,
                                ctrls[t].joint_f,
                                self.stand_arr,
                                self.dofs_arr,
                                self.group_arr,
                                min(t // sps, nseg - 1),
                                self.caps_j_arr,
                            ],
                        )
                    else:
                        wp.launch(
                            servo_profile_kernel,
                            dim=12,
                            inputs=[
                                params,
                                states[t].joint_q,
                                states[t].joint_qd,
                                ctrls[t].joint_f,
                                self.stand_arr,
                                self.dofs_arr,
                                self.prof_knots_arr,
                                PROF_N_KNOTS,
                                SIM_DT * 1000.0,
                                t,
                                TAU_MAX,
                                _bracket(t * SIM_DT * 1000.0),
                            ],
                        )
                    states[t].clear_forces()
                    self.solver.step(states[t], states[t + 1], ctrls[t], cur_cts, SIM_DT)
                if mode == "crouch":
                    for t in range(n + 1):
                        wp.launch(
                            order_state_g_kernel,
                            dim=1,
                            inputs=[
                                states[t].body_q,
                                self.knee_pts_body_arr,
                                self.knee_pts_local_arr,
                                self.n_knee_pts,
                                self.paw_body_arr,
                                self.paw_center_arr,
                                self.paw_r_arr,
                                self.n_paw,
                                W_ORDER_A,
                                KNEE_MARGIN_G,
                                PAW_MASK_H,
                                loss,
                            ],
                        )
                    wp.launch(
                        loss_crouch_end_kernel,
                        dim=1,
                        inputs=[
                            states[CR_N].body_q,
                            states[CR_N - VZ_LAG_M].body_q,
                            loss,
                            Z_TARGET,
                            A1,
                            B1,
                            C1,
                            INV_MDT,
                        ],
                    )
                elif mode == "jump":
                    for t in range(n + 1):
                        wp.launch(wy_state_kernel, dim=1, inputs=[states[t].body_qd, wy_arr, t, SIM_DT])
                        wp.launch(zraw_state_kernel, dim=1, inputs=[states[t].body_q, z_arr, t])
                        wp.launch(
                            pawraw_state_kernel,
                            dim=1,
                            inputs=[
                                states[t].body_q,
                                self.paw_body_arr,
                                self.paw_center_arr,
                                self.paw_r_arr,
                                self.n_paw,
                                paw_arr,
                                t,
                            ],
                        )
                    for t in range(n + 1):
                        wp.launch(
                            loss_jump_state_kernel,
                            dim=1,
                            inputs=[
                                states[t].body_q,
                                z_arr,
                                wy_arr,
                                paw_arr,
                                t,
                                self.knee_pts_body_arr,
                                self.knee_pts_local_arr,
                                self.n_knee_pts,
                                self.paw_body_arr,
                                self.paw_center_arr,
                                self.paw_r_arr,
                                self.n_paw,
                                loss,
                                C_FLIP,
                                3.0 * C_ARR,
                                W_ORDER_B,
                                FOOT_TD,
                                TD_SIG,
                                KNEE_MARGIN_G,
                                PAW_MASK_H,
                                FLIP_SIGN,
                                TWO_PI,
                                LIFT_H,
                            ],
                        )
                    wp.launch(loss_jump_band_kernel, dim=1, inputs=[z_arr, n, loss, C_BAND, Z_GOOD, Z_LOW])
                else:
                    for t in range(n + 1):
                        wp.launch(
                            loss_land_state_kernel,
                            dim=1,
                            inputs=[
                                states[t].body_q,
                                states[t].body_qd,
                                self.knee_pts_body_arr,
                                self.knee_pts_local_arr,
                                self.n_knee_pts,
                                self.paw_body_arr,
                                self.paw_center_arr,
                                self.paw_r_arr,
                                self.n_paw,
                                loss,
                                W_ORDER_C,
                                KNEE_MARGIN_G,
                                PAW_MASK_H,
                                D3,
                                W_PAWS,
                                W_UP,
                            ],
                        )
                    wp.launch(
                        loss_land_end_kernel,
                        dim=1,
                        inputs=[states[n].body_q, states[n].body_qd, loss, self.z_stand, C_ARR, A2, B2, C2, D2],
                    )
            tape.backward(loss)
            L = float(loss.numpy()[0])
            g = params.grad.numpy().astype(np.float64).copy()
            tape.zero()
            return L, g

        return fn

    # ---------------- optimizers ----------------

    @staticmethod
    def _clipped_step(amps, d, r, lo, hi, r_min):
        # project onto the feasible box; accept any real move so warm starts
        # stay usable
        while r >= r_min:
            trial = np.clip(amps + r * d, lo, hi)
            if not np.array_equal(trial, amps):
                return trial, r
            r *= 0.5
        return amps.copy(), r

    def _tr_ascent(
        self,
        tag,
        tape_fn,
        eager_fn,
        r0,
        snap,
        amp_hi,
        iters,
        amps_init=None,
        nseg=4,
        lo=None,
        n_steps=100,
        sps=1,
        r_max=R_MAX,
        strict_rho=False,
    ):
        """Backtracking trust-region ascent (crouch + jump phases): ONE
        projected step per iteration (radius halved once per rejection,
        FRESH gradient every iteration - a stale gradient with an inner
        shrink loop is a different, broken search). Accepts ANY improvement
        on the chaotic tapes (rho only steers the radius); replay-based
        round selection guards result quality."""
        if lo is None:
            lo = np.full(nseg, -0.9)
        amps = np.array(amps_init, dtype=np.float64) if amps_init is not None else np.zeros(nseg, dtype=np.float64)
        L_init = eager_fn(n_steps, amps, sps, snap)
        L_cur = L_init
        r = r0
        accepts = 0
        for it in range(iters):
            L, g = tape_fn(n_steps, amps, sps, snap)
            if it == 0 and abs(L - L_cur) > 1e-3 * max(abs(L_cur), 1e-9):
                print(f"  [guard] {tag}: tape {L:.4f} vs eager {L_cur:.4f}", flush=True)
            ng = float(np.linalg.norm(g))
            if not np.isfinite(ng) or ng < 1e-14:
                print(f"  {tag} it{it}: |g|={ng:.3g} - stop", flush=True)
                break
            d = -g / ng
            trial, r_use = self._clipped_step(amps, d, r, lo, amp_hi, R_MIN)
            if r_use < R_MIN:
                break
            pred = r_use * ng
            L_new = eager_fn(n_steps, trial, sps, snap)
            if not np.isfinite(L_new):
                imp, rho = -np.inf, None
            else:
                imp = L_cur - L_new
                rho = imp / pred if pred > 0 else None
            # rho is meaningless on the chaotic jump/land tapes - only the
            # crouch phase gates on it
            ok = imp > 0 and (rho is None or rho >= 0.1 or not strict_rho)
            if ok:
                amps, L_cur = trial, L_new
                accepts += 1
                print(f"  {tag} it{it}: L={L_cur:.1f}", flush=True)
                if rho is not None and rho > 0.75:
                    r = min(r * 1.5, r_max)
            else:
                r *= 0.25 if strict_rho else 0.5
                if r < R_MIN:
                    break
        return {"amps": amps, "L_init": L_init, "L_cur": L_cur, "accepts": accepts}

    def _land_descent(self, tag, tape_fn, eager_fn, snap, iters, amps_init=None):
        """Landing optimizer: steepest direction from the tape with a
        backtracking line search over a dimension-scaled radius grid. This is
        the phase-C optimizer BY DESIGN (a trust-region ascent variant
        starves on the 180-dimensional landing box - it accepted 2-3 tiny
        steps and never reached the demonstrated landing)."""
        lo_d, hi_d = LAND_LO, LAND_HI
        amps = LAND_INIT.copy() if amps_init is None else np.clip(np.asarray(amps_init, dtype=np.float64), lo_d, hi_d)
        Le = eager_fn(N_C, amps, C_SPS, snap)
        L = Le
        radius_grid = (0.6, 0.4, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01)
        accepts = 0
        for it in range(iters):
            L, g = tape_fn(N_C, amps, C_SPS, snap)
            if it == 0 and abs(L - Le) > 1e-3 * max(abs(Le), 1e-9):
                print(f"  [guard] {tag}: tape {L:.4f} vs eager {Le:.4f}", flush=True)
            ng = float(np.linalg.norm(g))
            if not np.isfinite(ng) or ng < 1e-14:
                break
            d = -g / ng
            moved = False
            for r in radius_grid:
                trial = np.clip(amps + r * d, lo_d, hi_d)
                L2 = eager_fn(N_C, trial, C_SPS, snap)
                if np.isfinite(L2) and L2 < L - 1e-4 * abs(L):
                    amps, L = trial, L2
                    accepts += 1
                    print(f"  {tag} it{it}: L={L:.1f}  (r={r:g} accepted)", flush=True)
                    moved = True
                    break
            if not moved:
                print(f"  {tag} it{it}: L={L:.1f}  no radius improved - stop", flush=True)
                break
        return amps, {"L0": Le, "L1": L, "accepts": accepts}

    # ---------------- replays + metrics ----------------

    def _crouch_replay(self, amps):
        snap_end = None
        rep = {"min_foot_rows": 10**9, "knee_front_min": 1e9, "knee_hind_min": 1e9, "push_margin": 1e9}
        st = self._fresh(self.snap)
        st2 = self.model.state()
        amps_arr = wp.array(amps.astype(np.float32), dtype=wp.float32)
        m = None
        for t in range(CR_N_EXT):
            if t % COLLIDE_EVERY == 0:
                self.pipe.collide(st, self.cts)
            # foot rows read BEFORE the servo launch/step: the contact
            # state at the START of step t, not after it
            rows = self._foot_row_count()
            rep["min_foot_rows"] = min(rep["min_foot_rows"], rows)
            if t < CR_N:
                wp.launch(
                    servo_crouch_cos_kernel,
                    dim=12,
                    inputs=[
                        amps_arr,
                        st.joint_q,
                        st.joint_qd,
                        self.ctrl.joint_f,
                        self.stand_arr,
                        self.dofs_arr,
                        self.chan_arr,
                        t,
                        CR_T_REF,
                        TAU_MAX,
                    ],
                )
            else:
                # push preview: zero-amp rebound with jump torque caps
                wp.launch(
                    servo_traj_group_caps_kernel,
                    dim=12,
                    inputs=[
                        self.zero_amp_arr,
                        st.joint_q,
                        st.joint_qd,
                        self.ctrl.joint_f,
                        self.stand_arr,
                        self.dofs_arr,
                        self.group_arr,
                        0,
                        self.caps_j_arr,
                    ],
                )
            st.clear_forces()
            self.solver.step(st, st2, self.ctrl, self.cts, SIM_DT)
            st, st2 = st2, st
            bqd = st.body_q.numpy()
            fz, hz = self._leg_knee_zs(bqd)
            _, mp = self._min_zs(bqd)
            m = self._state_metrics(bqd[0], st.body_qd.numpy()[0])
            # knee mins over the FULL replay window (crouch + preview)
            rep["knee_front_min"] = min(rep["knee_front_min"], fz)
            rep["knee_hind_min"] = min(rep["knee_hind_min"], hz)
            if mp < PAW_MASK_H:
                rep["push_margin"] = min(rep["push_margin"], fz, hz)
            if t == CR_N - 1:
                snap_end = self._snapshot(st)
                rep["z_end"] = m["z"]
                rep["tilt_end_deg"] = float(np.degrees(m["tilt"]))
                rep["vz_end"] = m["vz"]
                rep["foot_rows_end"] = self._foot_row_count()
        return snap_end, rep

    def _leg_knee_zs(self, body_q_np):
        """(min front knee z, min hind knee z) - absolute heights."""
        out = np.empty(len(self.kp_local))
        front = np.zeros(len(self.kp_local), dtype=bool)
        for i, b in enumerate(self.kp_body):
            xf = body_q_np[int(b)]
            x, y, z, w = xf[3:]
            R = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            out[i] = (R @ self.kp_local[i].T)[2] + xf[2]
            front[i] = int(b) in (1, 2, 7, 8)  # LF/RF thigh+shank
        return float(out[front].min()), float(out[~front].min())

    def _jump_replay(self, amps, snap, record=False):
        """Fly the jump FROM THE CROUCH END (the ascent's anchor); graze-
        filtered touchdown detection (airborne = 5 foot-row-free steps; td =
        2 grounded steps AND descending near ground). Returns (td_snap or
        None, report)."""
        amps_arr = wp.array(amps.astype(np.float32), dtype=wp.float32)
        st = self._fresh(snap)
        st2 = self.model.state()
        q_prev = snap[2][0][3:].copy()
        pitch_cum = 0.0
        airborne_steps = 0
        foot_steps = 0
        td_snap = None
        td_t = None
        rep = {"apex": -1e9, "paw_min": 1e9, "blew_up": False, "knee_margin_min": 1e9, "min_foot_rows": 10**9}
        airborne = False
        t = -1
        for t in range(N_B):
            if t % COLLIDE_EVERY == 0:
                self.pipe.collide(st, self.cts)
            rows = self._foot_row_count()
            wp.launch(
                servo_traj_group_caps_kernel,
                dim=12,
                inputs=[
                    amps_arr,
                    st.joint_q,
                    st.joint_qd,
                    self.ctrl.joint_f,
                    self.stand_arr,
                    self.dofs_arr,
                    self.group_arr,
                    min(t // B_SPS, NSEG_B - 1),
                    self.caps_j_arr,
                ],
            )
            st.clear_forces()
            self.solver.step(st, st2, self.ctrl, self.cts, SIM_DT)
            st, st2 = st2, st
            bqd = st.body_q.numpy()
            bqd_d = st.body_qd.numpy()
            m = self._state_metrics(bqd[0], bqd_d[0])
            # pitch from orientation deltas, NOT angular-velocity
            # integration (the two estimates differ slightly)
            drv = _rotvec_incr(q_prev, bqd[0][3:])
            q_prev = bqd[0][3:].copy()
            pitch_cum += float(drv[1])
            min_k, mp = self._min_zs(bqd)
            rep["apex"] = max(rep["apex"], m["z"])
            rep["paw_min"] = min(rep["paw_min"], mp)
            rep["min_foot_rows"] = min(rep["min_foot_rows"], rows)
            if mp < PAW_MASK_H:
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
                if m["vz"] < 3.0 and m["z"] < 0.68:
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
            z = float(_quat_rot_np(xf[3:], c)[2]) + float(xf[2]) - r
            if z > max_p:
                max_p = z
        return max_p

    def _land_replay(self, amps, snap, record=False, extra=0, frame_offset=0):
        """Landing replay. extra > 0 = settle-check tail: keep simulating on
        with the final spline values held so a delayed bounce-back cannot
        hide past the window end. The settled gate is evaluated both at
        window end (settled_win) and at the true end of the run (settled);
        the chain itself runs extra=0. Recorded playback frames are sampled
        every FRAME_EVERY steps counted from the jump phase's touchdown step
        (``frame_offset`` = ``rep["td_step"]``) so the frames stay evenly
        spaced across the touchdown boundary."""
        rep = {
            "knee_margin_min": 1e9,
            "bounce": 0.0,
            "blew_up": False,
            "min_foot_rows": 10**9,
            "bounce_win": 0.0,
            "tail_max_vz": 0.0,
            "settled_win": False,
            "paw_lift_win": 0.0,
            "paw_lift_max": 0.0,
        }
        amps_arr = wp.array(amps.astype(np.float32), dtype=wp.float32)
        st = self._fresh(snap)
        st2 = self.model.state()
        m = None
        for t in range(N_C + extra):
            if t % COLLIDE_EVERY == 0:
                self.pipe.collide(st, self.cts)
            rows = self._foot_row_count()
            wp.launch(
                servo_profile_kernel,
                dim=12,
                inputs=[
                    amps_arr,
                    st.joint_q,
                    st.joint_qd,
                    self.ctrl.joint_f,
                    self.stand_arr,
                    self.dofs_arr,
                    self.prof_knots_arr,
                    PROF_N_KNOTS,
                    SIM_DT * 1000.0,
                    t,
                    TAU_MAX,
                    _bracket(t * SIM_DT * 1000.0),
                ],
            )
            st.clear_forces()
            self.solver.step(st, st2, self.ctrl, self.cts, SIM_DT)
            st, st2 = st2, st
            bqd = st.body_q.numpy()
            bqd_d = st.body_qd.numpy()
            m = self._state_metrics(bqd[0], bqd_d[0])
            min_k, min_p = self._min_zs(bqd)
            rep["min_foot_rows"] = min(rep["min_foot_rows"], rows)
            if min_p < PAW_MASK_H:
                rep["knee_margin_min"] = min(rep["knee_margin_min"], min_k)
            pmx = self._paw_max_z(bqd)
            rep["paw_lift_max"] = max(rep["paw_lift_max"], pmx)
            rep["bounce"] += max(m["vz"], 0.0) ** 2 * SIM_DT
            if record and t < N_C and (t + frame_offset) % FRAME_EVERY == 0:
                self.playback.append(("land", self._snapshot(st)))
            if t == N_C - 1:
                rep["bounce_win"] = rep["bounce"]
                rep["paw_lift_win"] = rep["paw_lift_max"]
                rep["final_z"] = m["z"]
                rep["final_tilt_deg"] = float(np.degrees(m["tilt"]))
                rep["final_vz"] = m["vz"]
                rep["final_w"] = m["w"]
                rep["final_up_z"] = _up_z_of_np(bqd[0])
                rep["settled_win"] = bool(
                    rep["final_tilt_deg"] <= 15.0
                    and abs(rep["final_vz"]) <= 0.15
                    and abs(rep["final_z"] - self.z_stand) <= 0.10
                    and m["w"] <= 0.6
                )
            if not np.isfinite(m["z"]):
                rep["blew_up"] = True
                break
            if t >= N_C:
                rep["tail_max_vz"] = max(rep["tail_max_vz"], m["vz"])
        rep["final_z"] = float(st.body_q.numpy()[0][2])
        rep["final_tilt_deg"] = float(np.degrees(_tilt_of(st.body_q.numpy()[0])))
        rep["final_vz"] = m["vz"]
        rep["final_w"] = m["w"]
        rep["final_up_z"] = _up_z_of_np(st.body_q.numpy()[0])
        rep["settled"] = bool(
            rep["final_tilt_deg"] <= 15.0
            and abs(rep["final_vz"]) <= 0.15
            and abs(rep["final_z"] - self.z_stand) <= 0.10
            and rep["final_w"] <= 0.6
        )
        return rep

    def _land_round_score(self, rep):
        return (
            0.0 if rep["blew_up"] else 1.0 if rep["settled"] else 0.5,
            -abs(rep["final_vz"]) - 0.01 * rep["final_tilt_deg"] - 0.5 * abs(rep["final_z"] - self.z_stand),
            -rep["bounce"],
        )

    # ---------------- the chain ----------------

    def run_chain(self):
        """Phase A -> B -> C. Prints progress; stores the final result."""
        tape_crouch = self._tape_loss("crouch")
        eager_crouch = self._eager_loss("crouch")

        # ---- phase A: crouch (rounds differ only in warm-start amps) ----
        print("== phase A: crouch ==", flush=True)
        rounds_a = []
        for rnd in range(CR_ROUNDS):
            warm = CR_WARM if rnd == 0 else rounds_a[-1]["res"]["amps"]
            res = self._tr_ascent(
                f"crouch-r{rnd}",
                tape_crouch,
                eager_crouch,
                3.0e-3,
                self.snap,
                CR_AMP_HI,
                CR_ITERS,
                amps_init=warm,
                nseg=CR_NP,
                lo=CR_AMP_LO,
                n_steps=CR_N_EXT,
                sps=1,
                strict_rho=True,
            )
            self.loss_history["crouch"].append(res["L_cur"])
            end_snap, rep = self._crouch_replay(res["amps"])
            ok = (
                rep["z_end"] <= 0.375
                and rep["tilt_end_deg"] < 20.0
                and rep["min_foot_rows"] > 0
                and rep["foot_rows_end"] > 0
                and rep["knee_front_min"] >= KNEE_MARGIN_G - 0.005
                and rep["knee_hind_min"] >= KNEE_MARGIN_G - 0.005
                and rep["push_margin"] >= -0.002
            )
            print(
                f"  r{rnd}: plan {res['L_init']:+.3f}->{res['L_cur']:+.3f} (acc {res['accepts']:2d})  z_end={rep['z_end']:.4f}  "
                f"tilt={rep['tilt_end_deg']:.1f}deg  rows_min={rep['min_foot_rows']}  {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
            rounds_a.append({"res": res, "end_snap": end_snap, "rep": rep, "ok": ok})
        cands_ok = [r for r in rounds_a if r["ok"]]
        # round selection: among gate-passing rounds, the one
        # whose crouch DEPTH is closest to the target (not the lowest loss -
        # depth is what the jump phase needs)
        if cands_ok:
            best_a = min(cands_ok, key=lambda r: abs(r["rep"]["z_end"] - Z_TARGET))
        else:
            best_a = max(
                rounds_a, key=lambda r: (round(r["rep"]["push_margin"], 4), -abs(r["rep"]["z_end"] - Z_TARGET))
            )
            print("  (no round passed all gates - chose cleanest-push/closest-to-depth candidate)", flush=True)
        print(
            f"  => chosen round: z_end={best_a['rep']['z_end']:.4f} (target {Z_TARGET:.3f})  "
            f"tilt={best_a['rep']['tilt_end_deg']:.1f}deg  rows_min={best_a['rep']['min_foot_rows']}  "
            f"{'PASS' if best_a['ok'] else 'FAIL'}",
            flush=True,
        )
        # ---- phase B: jump (from the chosen crouch end) ----
        print("== phase B: jump ==", flush=True)
        crouch_snap = best_a["end_snap"]
        tape_jump = self._tape_loss("jump")
        eager_jump = self._eager_loss("jump")
        b_amp_lo_arr = np.full(NSEG_B * 2, B_AMP_LO)
        warm_b = None
        best_b = None
        for rnd in range(JB_ROUNDS):
            res = self._tr_ascent(
                f"jump-r{rnd}",
                tape_jump,
                eager_jump,
                JB_R0,
                crouch_snap,
                B_AMP_HI,
                JB_ITERS,
                amps_init=warm_b,
                nseg=NSEG_B * 2,
                lo=b_amp_lo_arr,
                n_steps=N_B,
                sps=B_SPS,
                r_max=JB_R_MAX,
            )
            self.loss_history["jump"].append(res["L_cur"])
            same = best_b is not None and np.array_equal(res["amps"], best_b["res"]["amps"])
            print(f"  r{rnd}: plan {res['L_init']:+.3f}->{res['L_cur']:+.3f} (acc {res['accepts']:2d})", flush=True)
            if not same:
                td_r, rep_r = self._jump_replay(res["amps"], crouch_snap, record=False)
                # lexicographic tuple: a blown-up round loses to everything
                if rep_r["blew_up"]:
                    sc = (-1e9, 0.0, 0.0)
                else:
                    flip_pen = abs(rep_r.get("td_pitch_deg", rep_r["pitch_end_deg"]) - FLIP_SIGN * 360.0)
                    att_pen = rep_r.get("td_tilt_deg", 0.0)
                    km = rep_r["knee_margin_min"] if rep_r["knee_margin_min"] < 1e8 else KNEE_MARGIN_G
                    sc = (1.0 if td_r is not None else 0.0, -flip_pen - 0.02 * att_pen, km)
                print(
                    f"      replay: apex={rep_r['apex']:.3f}  pitch_end={rep_r['pitch_end_deg']:.1f}deg  "
                    f"td={'YES' if td_r is not None else 'NO'}"
                    + (f"  td tilt={rep_r['td_tilt_deg']:.1f}deg vz={rep_r['td_vz']:+.2f}" if td_r is not None else ""),
                    flush=True,
                )
                if best_b is None or sc > best_b["score"]:
                    best_b = {"res": res, "td_snap": td_r, "rep": rep_r, "score": sc}
            warm_b = res["amps"]
            if res["accepts"] == 0 and rnd > 0:
                break
        if best_b is None or best_b["td_snap"] is None:
            # last-resort: replay the last amps for the record; abort C
            print("phase B never achieved a clean touchdown - aborting before C", flush=True)
            self.result = {"crouch_ok": best_a["ok"], "jump_ok": False}
            self.chain_done = True
            return
        self.playback = []  # the final run's frames: chosen flight + landing
        td_snap, rep_b = self._jump_replay(best_b["res"]["amps"], crouch_snap, record=True)
        print(
            f"  => phase B best: apex={rep_b['apex']:.3f}  td pitch={rep_b.get('td_pitch_deg', 0):.1f}deg", flush=True
        )

        # ---- phase C: landing (descent line search from the TD anchor) ----
        print("== phase C: landing ==", flush=True)
        tape_land = self._tape_loss("land")
        eager_land = self._eager_loss("land")
        warm_c = None
        best_c = None
        for rnd in range(LC_ROUNDS):
            amps_r, res_r = self._land_descent(
                f"land-r{rnd}", tape_land, eager_land, td_snap, LC_ITERS, amps_init=warm_c
            )
            self.loss_history["land"].append(res_r["L1"])
            same = best_c is not None and np.array_equal(amps_r, best_c["amps"])
            print(f"  r{rnd}: plan {res_r['L0']:+.1f}->{res_r['L1']:+.1f} (acc {res_r['accepts']:2d})", flush=True)
            if not same:
                rep_r = self._land_replay(amps_r, td_snap, record=False)
                sc = self._land_round_score(rep_r)
                print(
                    f"      replay: z={rep_r['final_z']:.3f} tilt={rep_r['final_tilt_deg']:.1f}deg "
                    f"bounce(win)={rep_r['bounce_win']:.4f} pawlift={rep_r['paw_lift_max']:.3f} "
                    f"{'SETTLED' if rep_r['settled'] else 'NOT settled'}",
                    flush=True,
                )
                if best_c is None or sc > best_c["score"]:
                    best_c = {"amps": amps_r, "rep": rep_r, "score": sc}
            warm_c = amps_r
            if res_r["accepts"] == 0 and rnd > 0:
                break
        # final recorded replay of the chosen landing (bit-identical execution;
        # the playback frames come from THIS replay)
        rep_c = self._land_replay(
            best_c["amps"] if best_c is not None else warm_c,
            td_snap,
            record=True,
            frame_offset=rep_b["td_step"],
        )
        land_ok = rep_c.get("settled", False)
        print(
            f"  => phase C final: z={rep_c['final_z']:.3f} tilt={rep_c['final_tilt_deg']:.1f}deg "
            f"bounce(win)={rep_c['bounce_win']:.4f} pawlift={rep_c['paw_lift_max']:.3f} "
            f"{'SETTLED' if land_ok else 'NOT SETTLED'}",
            flush=True,
        )
        self.result = {
            "crouch_ok": best_a["ok"],
            "jump_ok": True,
            "td_pitch_deg": rep_b.get("td_pitch_deg"),
            "land_settled": bool(land_ok),
            "land_bounce": float(rep_c["bounce_win"]),
            "land_paw_lift": float(rep_c["paw_lift_max"]),
        }
        self.chain_done = True
        print("== chain done ==", flush=True)

    # ---------------- runner interface ----------------

    def step(self):
        if self.chain_done or self.args.test:
            # --test runs ONLY the gradient check (test_final); the full
            # optimization is the interactive run's purpose
            return
        self.run_chain()

    def render(self):
        if not self.playback:
            # nothing optimized yet (or --test): render the settled stance
            st = self._fresh(self.snap)
            self.viewer.begin_frame(self.frame * SIM_DT * FRAME_EVERY)
            self.viewer.log_state(st)
            self.viewer.end_frame()
            self.frame += 1
            return
        _ph, snap = self.playback[min(self.frame, len(self.playback) - 1)]
        st = self._fresh(snap)
        self.viewer.begin_frame(self.frame * SIM_DT * FRAME_EVERY)
        self.viewer.log_state(st)
        self.viewer.end_frame()
        self.frame += 1
        if self.frame >= len(self.playback):
            self.frame = 0  # loop the achieved behavior

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        return parser

    def test_final(self):
        """Gradient check: a 2-step landing-servo rollout from the settled
        stance, linear probe loss; tape gradient vs central finite differences
        on a spread of joint channels (skips the ~1 h optimization)."""
        model = self.model
        solver = self.solver
        cts = self.cts
        snap = self.snap
        channels = [(0, 5), (0, 9), (0, 16), (1, 3), (1, 4), (1, 16)]
        amp = 0.07  # off zero so the spline path is exercised

        def rollout(kick_arr, kick_idx, kick):
            amps = np.full(LAND_NPAR, amp, dtype=np.float32)
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
                    servo_profile_kernel,
                    dim=12,
                    inputs=[
                        amps_arr,
                        st0.joint_q,
                        st0.joint_qd,
                        ctrl.joint_f,
                        self.stand_arr,
                        self.dofs_arr,
                        self.prof_knots_arr,
                        PROF_N_KNOTS,
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
        amps = np.full(LAND_NPAR, amp, dtype=np.float32)
        params = wp.array(amps, dtype=wp.float32, requires_grad=True)
        states = [model.state(requires_grad=True) for _ in range(3)]
        states[0].joint_q.assign(snap[0])
        states[0].joint_qd.assign(snap[1])
        states[0].body_q.assign(snap[2])
        states[0].body_qd.assign(snap[3])
        # same freshness discipline as the phase tapes (see _tape_loss): the
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
                    servo_profile_kernel,
                    dim=12,
                    inputs=[
                        params,
                        states[t].joint_q,
                        states[t].joint_qd,
                        ctrls[t].joint_f,
                        self.stand_arr,
                        self.dofs_arr,
                        self.prof_knots_arr,
                        PROF_N_KNOTS,
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


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    def _build():
        return Example(viewer, args)

    newton.examples.run(_build(), args)
