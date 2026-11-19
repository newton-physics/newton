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
   flip error and attitude AT TOUCHDOWN (the first ~10 steps after the root
   comes back down, once the paws have flown), an apex height band and knee
   clearance. Pricing only the touchdown keeps the loss aligned with the
   outcome: the liftoff, where the flip error is still a full turn, and the
   stiff post-touchdown contacts, whose gradients are chaotic, stay out.
3. Landing (300 steps, then 1600 steps): per-joint target SPLINES
   (non-uniform knots, dense around touchdown) absorb the touchdown impact.
   The splines are first optimized over the first 300 ms alone, where the
   dog either catches the flip's leftover rotation or tumbles, and then over
   the full window (other first-stage windows are retried if it does not
   settle). The loss combines ground-based knee ordering, anti-bounce, a
   paw-glue term (every paw pinned to the ground - holding it at zero
   REQUIRES flexed knees, so the yielding-squat posture emerges instead of
   being prescribed) and an end-state stance term.

Gradients flow through SolverFeatherstone AND the differentiable contact
pipeline (soft_contact_gap margin). Every phase uses the same optimizer:
steepest descent on the tape gradient with a backtracking line search over
a fixed list of step lengths (``_descent``). Takeoff and touchdown over
contact are chaotic, so when the crouch, the flip or the landing misses its
gate, the chain is retried from a slightly shallower or deeper crouch warm
start (``CR_WARM_SCALES``).

Each phase window is a ``Rollout``: its states, controls and contact buffers
are allocated once, and on CUDA the forward pass and the forward + backward
pass are each captured into a CUDA graph. The optimizer's gradients, its
line-search loss evaluations and the replays all run the same kernels on the
same buffers, so the loss being differentiated is exactly the loss being
accepted. Together with the solver's tile-based mass-matrix assembly
(``use_tile_gemm=True``) a full run of all three skills takes a few minutes
on a single GPU. ``--test`` skips the optimization entirely and runs only the
gradient check (seconds); ``--rec`` saves the optimized playback to a
ViewerFile recording. The optimization advances a little every frame, so
an interactive viewer stays live (and can pause it) while it runs.

Writing differentiable loss kernels: Warp does not replay dynamic
(runtime-bound) loops in the backward pass (see "Dynamic Loops" in Warp's
differentiability guide), so loop-carried values such as a running min/max,
a prefix sum or a scanned spline-segment index get wrong adjoints. The loss
kernels here avoid them: spline segments are bracketed on the host
(``_bracket``), fixed paw sets are statically unrolled, prefix values are
built by chained per-element launches (``cum_sum_kernel`` /
``cum_max_kernel``), and the knee minimum is an integer argmin pre-pass
followed by a differentiable gather. ``test_final`` finite-difference-checks
these patterns.

Success criteria:
- Phase A: a clean crouch (depth ~0.32 m, upright, knees clear of the ground).
- Phase B: the dog leaves the ground, completes a full backflip and touches
  down on its paws: touchdown pitch within 10 degrees of -360 and tilt
  below 10 degrees.
- Phase C: the dog settles standing after the impact (settled gate: tilt
  <= 15 deg, |vz| <= 0.15 m/s, |z - stand| <= 0.10 m, |w| <= 0.6 rad/s).
- When ``--test`` runs, numeric and analytic gradients agree within the
  check tolerance.

Runs:
- `python -m newton.examples diffsim_anymal_backflip`
- `python -m newton.examples diffsim_anymal_backflip --rec backflip.bin`
"""

import os
import time

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
PAWS = len(FOOT_BODIES)  # fixed paw count: paw minima may unroll statically
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
Z_TD = 0.68  # touchdown attention: root height the replay calls touchdown (m)
TD_SIG = 0.015  # touchdown attention sharpness (m)
TD_LEN = 10.0  # touchdown attention spans ~this many steps after first touchdown
LIFT_H = 0.15  # airborne gate: prefix-max paw height for "has flown" (m)
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
CR_ITERS = 64
STEP_BUDGET_S = 0.05  # optimization time per viewer frame (keeps the window responsive)
REPLAY_YIELD_EVERY = 100  # replays hand control back to the viewer every this many steps
# crouch warm-start scale per chain attempt, alternating around the default;
# the next one is tried when the crouch, the flip or the landing misses its gate
CR_WARM_SCALES = (1.0, 0.98, 1.02, 0.96, 1.04, 0.94)
CR_RADII = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
CR_NP = 8  # cosine amps (4 channels) + ramp times (4 channels)
CR_T_REF = 249.0
CR_WARM = np.array([5.2, 3.8, 4.5, 4.5, 1.0, 1.0, 1.0, 1.0])  # hip-dominant
CR_AMP_LO = np.array([-1.0, -1.0, -1.0, -1.0, 0.1, 0.1, 0.1, 0.1])
CR_AMP_HI = np.array([5.5, 5.5, 5.5, 5.5, 1.2, 1.2, 1.2, 1.2])

# phase B: jump
N_B, NSEG_B = 1500, 15
B_SPS = N_B // NSEG_B
JB_ITERS = 16
JB_RADII = (0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005)
B_AMP_LO, B_AMP_HI = -0.9, 3.0
TAUF, TAUH = 450.0, 300.0  # jump torque caps (front/hind)
TAU_MAX = 300.0  # land servo torque clamp
GROUP_OF_JOINT = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=np.int32)  # LF,RF front; LH,RH hind
CHAN_OF_JOINT = np.array([GROUP_OF_JOINT[j] * 2 + (1 if j % 3 == 2 else 0) for j in range(12)], dtype=np.int32)

# phase C: landing (per-joint direct-target splines, C1 smoothstep, dense
# around touchdown; past the last knot the final values are held)
N_C, NSEG_C = 1600, 8
LAND_SHORT_WINDOWS = (300, 0, 600)  # first-stage window per landing attempt (steps; 0 = none)
C_SPS = N_C // NSEG_C
LC_ITERS = 10
LC_RADII = (0.6, 0.4, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01)
PROF_KNOTS_MS = [0, 20, 40, 60, 90, 120, 160, 200, 260, 320, 400, 500, 650, 850, 1150, 1600]
PROF_N_KNOTS = len(PROF_KNOTS_MS)
LAND_NPAR = 12 * (PROF_N_KNOTS - 1)
_JOINT_LO = (-0.7, -1.2, -2.4)  # HAA, HFE, KFE
_JOINT_HI = (0.7, 0.9, -0.05)  # knee kept from locking straight
LAND_LO = np.tile(np.repeat(_JOINT_LO, PROF_N_KNOTS - 1).astype(np.float64), 4)
LAND_HI = np.tile(np.repeat(_JOINT_HI, PROF_N_KNOTS - 1).astype(np.float64), 4)
LAND_INIT = np.tile(np.repeat((0.0, 0.2, -0.4), PROF_N_KNOTS - 1).astype(np.float64), 4)

COLLIDE_EVERY = 2  # contact-truthful tapes; replays use the SAME schedule
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
    paw_arr: wp.array[float],
    idx: int,
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
    body_q: wp.array[wp.transform],
    pts_body: wp.array[int],
    pts_local: wp.array[wp.vec3],
    n_pts: int,
    argmin_arr: wp.array[int],
    idx: int,
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
    body_q: wp.array[wp.transform],
    pts_body: wp.array[int],
    pts_local: wp.array[wp.vec3],
    argmin_arr: wp.array[int],
    mink_arr: wp.array[float],
    idx: int,
):
    """World z of the lowest knee point at one state (``mink_arr[idx]``):
    a loop-free gather of the argmin pre-pass, so the adjoint flows exactly
    to the winning point's body transform (same pattern as the host-side
    ``_bracket`` spline segment, moved onto the device)."""
    k = argmin_arr[idx]
    p = wp.transform_point(body_q[pts_body[k]], pts_local[k])
    mink_arr[idx] = p[2]


@wp.kernel
def cum_sum_kernel(x_arr: wp.array[float], cum_arr: wp.array[float], idx: int):
    """One link of a chained per-element prefix sum: launch for idx = 0..n in
    order and ``cum_arr`` holds x's running sum. Loop-free, so the adjoint is
    exact - the runtime-bound prefix loops this replaces measured with zero
    or last-index-garbage gradients (module docstring)."""
    prev = float(0.0)
    if idx > 0:
        prev = cum_arr[idx - 1]
    cum_arr[idx] = prev + x_arr[idx]


@wp.kernel
def cum_max_kernel(x_arr: wp.array[float], cum_arr: wp.array[float], idx: int):
    """One link of a chained per-element running maximum (see cum_sum_kernel)."""
    prev = float(-1.0e9)
    if idx > 0:
        prev = cum_arr[idx - 1]
    cum_arr[idx] = wp.max(prev, x_arr[idx])


@wp.kernel
def order_state_g_kernel(
    mink_arr: wp.array[float],
    minp_arr: wp.array[float],
    idx: int,
    w_order: float,
    margin_g: float,
    mask_h: float,
    loss: wp.array[float],
):
    """Ground-based knee ordering: while paws are grounded, the lowest knee
    point must stay above the GROUND margin. The minima arrive precomputed
    per state (knee_mingz_kernel / pawraw_state_kernel); the kernel itself is
    loop-free."""
    mask = wp.max(1.0 - minp_arr[idx] / mask_h, 0.0)
    viol = wp.max(margin_g - mink_arr[idx], 0.0)
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
def touchdown_att_kernel(
    z_arr: wp.array[float],
    cum_paw_arr: wp.array[float],
    idx: int,
    z_td: float,
    td_sig: float,
    lift_h: float,
    att_arr: wp.array[float],
):
    """Touchdown attention at one state (``att_arr[idx]``): ~1 once the dog
    has flown (prefix-max paw height above lift_h) and its root has come
    back down to the touchdown height z_td, ~0 otherwise. The airborne gate
    keeps the push-off out: there the root rises through z_td while the
    paws are still on the ground and the flip error is still a full turn."""
    a = wp.clamp((z_td - z_arr[idx]) / td_sig, -30.0, 30.0)
    ag = wp.clamp((cum_paw_arr[idx] - lift_h) / 0.01, -30.0, 30.0)
    att_arr[idx] = 1.0 / ((1.0 + wp.exp(-a)) * (1.0 + wp.exp(-ag)))


@wp.kernel
def loss_jump_state_kernel(
    body_q: wp.array[wp.transform],
    cum_wy_arr: wp.array[float],
    mink_arr: wp.array[float],
    minp_arr: wp.array[float],
    att_arr: wp.array[float],
    cum_att_arr: wp.array[float],
    idx: int,
    loss: wp.array[float],
    c_flip: float,
    c_att: float,
    w_order: float,
    margin_g: float,
    mask_h: float,
    flip_sign: float,
    two_pi: float,
    td_len: float,
):
    """Per-state jump terms: flip error and attitude at touchdown, knee
    clearance while grounded. pitch (the accumulated roll through this
    state) and the attention prefix sum arrive precomputed by chained
    per-element launches, the knee/paw minima by the gather kernels - the
    kernel itself is loop-free (module docstring).

    Every term is weighted by first = exp(-attention so far / td_len), so
    only the first ~td_len steps after touchdown are priced: past that the
    landing is phase C's job, and the stiff post-touchdown contacts would
    only add chaotic gradient noise."""
    pitch = cum_wy_arr[idx]
    min_k = mink_arr[idx]
    min_p = minp_arr[idx]
    q = wp.transform_get_rotation(body_q[0])
    up = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    prev = float(0.0)
    if idx > 0:
        prev = cum_att_arr[idx - 1]
    first = wp.exp(-prev / td_len)
    w = att_arr[idx] * first
    flip_err = pitch - flip_sign * two_pi
    mask_g = wp.max(1.0 - min_p / mask_h, 0.0)
    viol_g = wp.max(margin_g - min_k, 0.0)
    loss[0] = (
        loss[0]
        + c_flip * w * flip_err * flip_err
        + c_att * w * (1.0 - up[2])
        + w_order * first * mask_g * viol_g * viol_g
    )


@wp.kernel
def loss_jump_band_kernel(
    cum_z_arr: wp.array[float],
    n: int,
    loss: wp.array[float],
    c_band: float,
    z_good: float,
    z_low: float,
):
    """Apex band on the window's highest root height. The maximum arrives
    precomputed (a cum_max_kernel chain over the z state samples): a
    runtime-bound running max here measured as a wrong
    ``-2*c_band*(z_low + 1e9)``-scale adjoint deposited on the last element
    (module docstring)."""
    zmax = cum_z_arr[n]
    loss[0] = loss[0] + c_band * (wp.pow(wp.max(zmax - z_good, 0.0), 2.0) + wp.pow(wp.max(z_low - zmax, 0.0), 2.0))


@wp.kernel
def loss_land_state_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    mink_arr: wp.array[float],
    paw_body: wp.array[int],
    paw_center: wp.array[wp.vec3],
    paw_r: wp.array[float],
    idx: int,
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

    The knee minimum arrives precomputed per state (knee_mingz_kernel); the
    paw minimum and the paw-glue sum unroll statically over the fixed PAWS
    count - straight-line code with an exact adjoint (module docstring).

    The terms accumulate over every state of the window. (A variant that
    starts the window before touchdown would need to gate them on
    t >= first-contact so airborne states contribute no loss; this example
    starts the window at touchdown, so every state is post-contact.)"""
    min_k = mink_arr[idx]
    min_p = float(1.0e9)
    pp = float(0.0)
    for k in range(PAWS):
        c = wp.transform_point(body_q[paw_body[k]], paw_center[k])
        h = c[2] - paw_r[k]
        min_p = wp.min(min_p, h)
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
# numpy helpers (replay metrics)
# ------------------------------------------------------------------------------


def _up_z(body_q):
    """Up-axis z component of a body transform row (1 upright, -1 inverted)."""
    return float(wp.quat_rotate(wp.quat(body_q[3:7]), wp.vec3(0.0, 0.0, 1.0))[2])


def _tilt_of(body_q):
    """Tilt angle (radians) of a body transform row away from upright."""
    return float(np.arccos(np.clip(_up_z(body_q), -1.0, 1.0)))


def _load_snapshot(st, snap):
    """Copy a host-side (joint_q, joint_qd, body_q, body_qd) snapshot into a state."""
    st.joint_q.assign(snap[0])
    st.joint_qd.assign(snap[1])
    st.body_q.assign(snap[2])
    st.body_qd.assign(snap[3])


# ------------------------------------------------------------------------------
# Rollout: preallocated, CUDA-graph-captured optimization window
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
        self.mink_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
        self.minp_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
        if mode == "jump":
            # per-state samples the jump loss reads; the prefix sum / maxima
            # are chained per-element launches into the cum_* arrays
            self.wy_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
            self.z_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
            self.paw_arr = wp.zeros(n + 1, dtype=wp.float32, requires_grad=True)
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
        return float(self.loss_arr.numpy()[0]), self.params.grad.numpy().astype(np.float64)


# ------------------------------------------------------------------------------
# Example
# ------------------------------------------------------------------------------


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

        # use_tile_gemm: the joint-space mass matrix H = J^T M J is formed with
        # Warp's tile API (one thread block per articulation) instead of a
        # single-threaded dense GEMM - for one 18-DOF robot this is ~30x less
        # GPU time per differentiable step
        self.solver = newton.solvers.SolverFeatherstone(self.model, use_tile_gemm=True)
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
                knee_pts_local.append(wp.transform_point(wp.transform(spos, sq), wp.vec3(p)))
                knee_pts_body.append(b)
        self.knee_pts_body_arr = wp.array(knee_pts_body, dtype=wp.int32)
        self.knee_pts_local_arr = wp.array(np.array(knee_pts_local, dtype=np.float32), dtype=wp.vec3)
        self.n_knee_pts = len(knee_pts_body)
        self.kp_body = np.array(knee_pts_body, dtype=np.int32)
        self.kp_local = np.array(knee_pts_local, dtype=np.float64)

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
        self.chan_arr = wp.array(CHAN_OF_JOINT, dtype=wp.int32)
        self.zero_amp_arr = wp.zeros(NSEG_B * 2, dtype=wp.float32)
        self.caps_j_arr = wp.array([TAUF, TAUH], dtype=wp.float32)
        self.prof_knots_arr = wp.array(np.array(PROF_KNOTS_MS, dtype=np.float32), dtype=wp.float32)
        self.ctrl = self.model.control()

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
        s0 = self.model.state()
        s1 = self.model.state()
        s0 = self._settle_segment(s0, s1, 1200, mode=0)
        s0 = self._settle_segment(s0, s1, 100, mode=1, amp=-0.12)
        self.snap = self._snapshot(s0)
        self.z_stand = float(self.snap[2][0][2])
        print(f"settle done: z_stand={self.z_stand:.4f}", flush=True)

    # ---------------- execution helpers ----------------

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
            pz = wp.transform_point(wp.transform(xf[:3], xf[3:]), c)[2] - r
            min_p = min(min_p, pz)
        return float(out.min()), min_p

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
        if ro.mode == "crouch" and t < CR_N:
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
                    self.chan_arr,
                    t,
                    CR_T_REF,
                    TAU_MAX,
                ],
            )
        elif ro.mode in ("crouch", "jump"):
            # jump: per-segment trajectory amplitudes; crouch preview
            # (t >= CR_N): zero-amp rebound with jump torque caps - the order
            # term prices the rebound over these states too
            jump = ro.mode == "jump"
            wp.launch(
                servo_traj_group_caps_kernel,
                dim=12,
                inputs=[
                    ro.params if jump else self.zero_amp_arr,
                    st.joint_q,
                    st.joint_qd,
                    ctrl.joint_f,
                    self.stand_arr,
                    self.dofs_arr,
                    self.group_arr,
                    min(t // B_SPS, NSEG_B - 1) if jump else 0,
                    self.caps_j_arr,
                ],
            )
        else:
            wp.launch(
                servo_profile_kernel,
                dim=12,
                inputs=[
                    ro.params,
                    st.joint_q,
                    st.joint_qd,
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

    def _launch_knee_min(self, st, ro, t):
        """Record the state's lowest knee-point world z into ``ro.mink_arr[t]``:
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
            inputs=[st.body_q, self.knee_pts_body_arr, self.knee_pts_local_arr, ro.argmin_arr, ro.mink_arr, t],
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
                    inputs=[states[t].body_q, self.paw_body_arr, self.paw_center_arr, self.paw_r_arr, ro.minp_arr, t],
                )
                wp.launch(
                    order_state_g_kernel,
                    dim=1,
                    inputs=[ro.mink_arr, ro.minp_arr, t, W_ORDER_A, KNEE_MARGIN_G, PAW_MASK_H, ro.loss_arr],
                )
            wp.launch(
                loss_crouch_end_kernel,
                dim=1,
                inputs=[
                    states[CR_N].body_q,
                    states[CR_N - VZ_LAG_M].body_q,
                    ro.loss_arr,
                    Z_TARGET,
                    A1,
                    B1,
                    C1,
                    INV_MDT,
                ],
            )
        elif ro.mode == "jump":
            for t in range(n + 1):
                wp.launch(wy_state_kernel, dim=1, inputs=[states[t].body_qd, ro.wy_arr, t, SIM_DT])
                wp.launch(zraw_state_kernel, dim=1, inputs=[states[t].body_q, ro.z_arr, t])
                wp.launch(
                    pawraw_state_kernel,
                    dim=1,
                    inputs=[states[t].body_q, self.paw_body_arr, self.paw_center_arr, self.paw_r_arr, ro.paw_arr, t],
                )
                # prefix values as chained per-element launches (adjoint-exact;
                # the in-kernel runtime-bound loops they replace measured with
                # zero or garbage gradients - see the module docstring)
                wp.launch(cum_sum_kernel, dim=1, inputs=[ro.wy_arr, ro.cum_wy_arr, t])
                wp.launch(cum_max_kernel, dim=1, inputs=[ro.z_arr, ro.cum_z_arr, t])
                wp.launch(cum_max_kernel, dim=1, inputs=[ro.paw_arr, ro.cum_paw_arr, t])
                wp.launch(
                    touchdown_att_kernel,
                    dim=1,
                    inputs=[ro.z_arr, ro.cum_paw_arr, t, Z_TD, TD_SIG, LIFT_H, ro.att_arr],
                )
                wp.launch(cum_sum_kernel, dim=1, inputs=[ro.att_arr, ro.cum_att_arr, t])
                self._launch_knee_min(states[t], ro, t)
                wp.launch(
                    loss_jump_state_kernel,
                    dim=1,
                    inputs=[
                        states[t].body_q,
                        ro.cum_wy_arr,
                        ro.mink_arr,
                        # the state sample IS the state's lowest paw bottom:
                        # the kernel reads its min_p from it
                        ro.paw_arr,
                        ro.att_arr,
                        ro.cum_att_arr,
                        t,
                        ro.loss_arr,
                        C_FLIP,
                        3.0 * C_ARR,
                        W_ORDER_B,
                        KNEE_MARGIN_G,
                        PAW_MASK_H,
                        FLIP_SIGN,
                        TWO_PI,
                        TD_LEN,
                    ],
                )
            wp.launch(loss_jump_band_kernel, dim=1, inputs=[ro.cum_z_arr, n, ro.loss_arr, C_BAND, Z_GOOD, Z_LOW])
        else:
            for t in range(n + 1):
                self._launch_knee_min(states[t], ro, t)
                wp.launch(
                    loss_land_state_kernel,
                    dim=1,
                    inputs=[
                        states[t].body_q,
                        states[t].body_qd,
                        ro.mink_arr,
                        self.paw_body_arr,
                        self.paw_center_arr,
                        self.paw_r_arr,
                        t,
                        ro.loss_arr,
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
                inputs=[states[n].body_q, states[n].body_qd, ro.loss_arr, self.z_stand, C_ARR, A2, B2, C2, D2],
            )

    # ---------------- optimizer ----------------

    def _descent(self, tag, ro, snap, amps_init, lo, hi, iters, radii):
        """Steepest descent with a backtracking line search, shared by all
        three phases: each iteration takes a fresh tape gradient and tries
        the step lengths in ``radii`` (largest first) along the normalized
        descent direction, projected onto the [lo, hi] box, and accepts the
        first one that lowers the loss. It stops once no step length helps.
        Trying the large steps first every iteration (instead of shrinking a
        trust radius on each rejection) keeps the search from stalling on
        the kinks that contact events put into these losses.

        A generator: it yields after every rollout so ``step()`` can hand
        control back to the viewer, and returns the optimized amplitudes."""
        amps = np.clip(np.asarray(amps_init, dtype=np.float64), lo, hi)
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
                trial = np.clip(amps + r * d, lo, hi)
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

    # ---------------- replays + metrics ----------------
    #
    # A replay re-runs the window's forward pass and reads the resulting
    # trajectory back from the rollout buffers. Contacts at step t are the
    # ones collided at the start of that step: ro.contacts[t // COLLIDE_EVERY].
    # Replays are generators like ``_descent`` (they yield every
    # REPLAY_YIELD_EVERY steps) and return their results.

    def _crouch_replay(self, ro, amps):
        """Replay the crouch phase for the given amplitudes; returns the end snapshot and metrics."""
        ro.loss(amps, self.snap)
        snap_end = None
        rep = {"min_foot_rows": 10**9, "knee_front_min": 1e9, "knee_hind_min": 1e9, "push_margin": 1e9}
        for t in range(CR_N_EXT):
            if t % REPLAY_YIELD_EVERY == 0:
                yield
            rows = self._foot_row_count(ro.contacts[t // COLLIDE_EVERY])
            rep["min_foot_rows"] = min(rep["min_foot_rows"], rows)
            st = ro.states[t + 1]
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
                rep["foot_rows_end"] = rows
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

    def _jump_replay(self, ro, amps, snap, record=False):
        """Fly the jump FROM THE CROUCH END (the ascent's anchor); graze-
        filtered touchdown detection (airborne = 5 foot-row-free steps; td =
        2 grounded steps AND descending near ground). Returns (td_snap or
        None, report)."""
        ro.loss(amps, snap)
        q_prev = wp.quat(snap[2][0][3:])
        pitch_cum = 0.0
        airborne_steps = 0
        foot_steps = 0
        td_snap = None
        td_t = None
        rep = {"apex": -1e9, "paw_min": 1e9, "blew_up": False, "knee_margin_min": 1e9, "min_foot_rows": 10**9}
        airborne = False
        for t in range(N_B):
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
                if m["vz"] < 3.0 and m["z"] < Z_TD:
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
        rep = {"knee_margin_min": 1e9, "bounce": 0.0, "blew_up": False, "min_foot_rows": 10**9, "paw_lift_max": 0.0}
        m = None
        bqd = None
        for t in range(N_C):
            if t % REPLAY_YIELD_EVERY == 0:
                yield
            rows = self._foot_row_count(ro.contacts[t // COLLIDE_EVERY])
            st = ro.states[t + 1]
            bqd = st.body_q.numpy()
            m = self._state_metrics(bqd[0], st.body_qd.numpy()[0])
            min_k, min_p = self._min_zs(bqd)
            rep["min_foot_rows"] = min(rep["min_foot_rows"], rows)
            if min_p < PAW_MASK_H:
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

    # ---------------- the chain ----------------

    def run_chain(self):
        """Phase A -> B -> C. Prints progress; stores the final result.

        A generator that yields after every rollout (see ``step()``).

        Contact makes the takeoff and the landing chaotic: two crouches that
        differ only in the last decimals can launch measurably different
        spins (the push is torque-saturated, so the jump amplitudes cannot
        undo that), and some touchdowns are much harder to catch than
        others. When an attempt misses a gate, the chain is retried from
        the next crouch warm start in ``CR_WARM_SCALES``, and the best
        attempt is kept."""
        best = None
        for attempt, scale in enumerate(CR_WARM_SCALES):
            if attempt:
                print(f"== retrying the chain, crouch warm start x{scale:g} (attempt {attempt + 1}) ==", flush=True)
            res = yield from self._attempt(CR_WARM * scale)
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
        res["crouch_ok"], crouch_snap = yield from self._phase_a(crouch_warm)
        res["result"] = {"crouch_ok": res["crouch_ok"], "jump_ok": False}
        if not res["crouch_ok"]:
            return res
        td_snap, rep_b = yield from self._phase_b(crouch_snap)
        res["playback"] = self.playback
        if td_snap is None:
            print("  phase B never achieved a clean touchdown - skipping phase C", flush=True)
            res["score"] = (1, -np.inf)
            return res
        flip_err = abs(rep_b["td_pitch_deg"] - FLIP_SIGN * 360.0)
        res["flip_ok"] = flip_err <= 10.0 and rep_b["td_tilt_deg"] <= 10.0
        res["result"] = {"crouch_ok": True, "jump_ok": True, "td_pitch_deg": rep_b["td_pitch_deg"]}
        res["score"] = (1 + int(res["flip_ok"]), -flip_err)
        if not res["flip_ok"]:
            return res
        rep_c = yield from self._phase_c(td_snap, rep_b["td_step"])
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

    def _phase_a(self, warm):
        """Phase A: optimize the crouch from the settled stance, starting at
        the ``warm`` amplitudes; returns (gates passed, crouch-end snapshot)."""
        print("== phase A: crouch ==", flush=True)
        ro = Rollout(self, "crouch", CR_N_EXT, CR_NP)
        amps = yield from self._descent("crouch", ro, self.snap, warm, CR_AMP_LO, CR_AMP_HI, CR_ITERS, CR_RADII)
        end_snap, rep = yield from self._crouch_replay(ro, amps)
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
            f"  => crouch: z_end={rep['z_end']:.4f} (target {Z_TARGET:.3f})  tilt={rep['tilt_end_deg']:.1f}deg  "
            f"rows_min={rep['min_foot_rows']}  {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
        return ok, end_snap

    def _phase_b(self, crouch_snap):
        """Phase B: optimize the jump from the crouch end; records the flight
        playback and returns (touchdown snapshot or None, replay report)."""
        print("== phase B: jump ==", flush=True)
        ro = Rollout(self, "jump", N_B, NSEG_B * 2)
        lo = np.full(NSEG_B * 2, B_AMP_LO)
        hi = np.full(NSEG_B * 2, B_AMP_HI)
        amps = yield from self._descent("jump", ro, crouch_snap, np.zeros(NSEG_B * 2), lo, hi, JB_ITERS, JB_RADII)
        self.playback = []  # the flight frames; phase C appends the landing
        td_snap, rep = yield from self._jump_replay(ro, amps, crouch_snap, record=True)
        print(
            f"  => jump: apex={rep['apex']:.3f}  pitch_end={rep['pitch_end_deg']:.1f}deg  "
            f"td={'YES' if td_snap is not None else 'NO'}"
            + (f"  td tilt={rep['td_tilt_deg']:.1f}deg vz={rep['td_vz']:+.2f}" if td_snap is not None else ""),
            flush=True,
        )
        return td_snap, rep

    def _phase_c(self, td_snap, td_step):
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
        print("== phase C: landing ==", flush=True)
        ro = Rollout(self, "land", N_C, LAND_NPAR)
        best = None
        for n_short in LAND_SHORT_WINDOWS:
            amps = LAND_INIT
            if n_short:
                ro_short = Rollout(self, "land", n_short, LAND_NPAR)
                amps = yield from self._descent(
                    f"land-{n_short}", ro_short, td_snap, amps, LAND_LO, LAND_HI, LC_ITERS, LC_RADII
                )
                del ro_short
                # hold the pose of the last knot inside the short window for
                # the rest of the landing, instead of springing back to the
                # untouched stand targets the later knots still hold
                last = int(np.searchsorted(PROF_KNOTS_MS, n_short * SIM_DT * 1000.0)) - 1
                amps = amps.reshape(12, PROF_N_KNOTS - 1)
                amps[:, last + 1 :] = amps[:, last : last + 1]
                amps = amps.reshape(-1)
            amps = yield from self._descent("land", ro, td_snap, amps, LAND_LO, LAND_HI, LC_ITERS, LC_RADII)
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

    # ---------------- runner interface ----------------

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

        # ---- loss-stage adjoint guard (module docstring: dynamic loops). The
        # 2-step probe above never runs the losses' running minima and prefix
        # patterns, so they are finite-difference-checked on a one-body scene.
        pts_body_g = wp.array([0, 0, 0], dtype=wp.int32)
        pts_local_g = wp.array(
            np.array([[0.0, 0.0, -0.55], [0.0, 0.0, -0.5], [0.0, 0.0, -0.45]], dtype=np.float32), dtype=wp.vec3
        )
        paw_body_g = wp.array([0, 0, 0, 0], dtype=wp.int32)
        paw_center_g = wp.array(
            np.array([[0.0, 0.0, -0.495], [0.0, 0.0, -0.3], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            dtype=wp.vec3,
        )
        paw_r_g = wp.array([0.0, 0.0, 0.0, 0.0], dtype=wp.float32)

        def order_scene(dz):
            """Guard scene: knee points 5/0/5 cm below the body origin, one paw
            5 mm above ground - knee-margin violation and grounded mask both
            active, so the order-term gradient is genuinely nonzero."""
            body_q = wp.array(
                np.array([[0.0, 0.0, 0.5 + dz, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                dtype=wp.transform,
                requires_grad=True,
            )
            argmin_arr = wp.zeros(1, dtype=wp.int32)
            mink_arr = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            minp_arr = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            wp.launch(knee_argmin_kernel, dim=1, inputs=[body_q, pts_body_g, pts_local_g, 3, argmin_arr, 0])
            wp.launch(knee_mingz_kernel, dim=1, inputs=[body_q, pts_body_g, pts_local_g, argmin_arr, mink_arr, 0])
            wp.launch(pawraw_state_kernel, dim=1, inputs=[body_q, paw_body_g, paw_center_g, paw_r_g, minp_arr, 0])
            loss_g = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            wp.launch(
                order_state_g_kernel,
                dim=1,
                inputs=[mink_arr, minp_arr, 0, W_ORDER_A, KNEE_MARGIN_G, PAW_MASK_H, loss_g],
            )
            return body_q, loss_g

        eps_g = 1.0e-4
        _, loss_p = order_scene(eps_g)
        _, loss_m = order_scene(-eps_g)
        fd_pz = (float(loss_p.numpy()[0]) - float(loss_m.numpy()[0])) / (2 * eps_g)
        tape_g = wp.Tape()
        with tape_g:
            body_q_g, loss_0 = order_scene(0.0)
        tape_g.backward(loss_0)
        g_pz = float(body_q_g.grad.numpy()[0][2])
        tape_g.reset()
        err_g = abs(g_pz - fd_pz)
        rel_g = err_g / max(abs(fd_pz), 1e-12)
        assert err_g <= 5e-5 or rel_g <= 0.15, (
            f"loss-stage guard (order term): tape={g_pz:.5e} fd={fd_pz:.5e} rel={rel_g:.2e} - zero/garbage adjoint?"
        )

        def band_loss(z_max_val):
            z_np = np.full(17, 1.5, dtype=np.float32)
            z_np[9] = z_max_val
            z_arr = wp.array(z_np)
            cum_z = wp.zeros(17, dtype=wp.float32)
            loss_b = wp.zeros(1, dtype=wp.float32)
            for t in range(17):
                wp.launch(cum_max_kernel, dim=1, inputs=[z_arr, cum_z, t])
            wp.launch(loss_jump_band_kernel, dim=1, inputs=[cum_z, 16, loss_b, C_BAND, Z_GOOD, Z_LOW])
            return float(loss_b.numpy()[0])

        z_np_g = np.full(17, 1.5, dtype=np.float32)
        z_np_g[9] = 2.0
        z_arr_g = wp.array(z_np_g, requires_grad=True)
        cum_z_g = wp.zeros(17, dtype=wp.float32, requires_grad=True)
        tape_b = wp.Tape()
        with tape_b:
            loss_b = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            for t in range(17):
                wp.launch(cum_max_kernel, dim=1, inputs=[z_arr_g, cum_z_g, t])
            wp.launch(loss_jump_band_kernel, dim=1, inputs=[cum_z_g, 16, loss_b, C_BAND, Z_GOOD, Z_LOW])
        tape_b.backward(loss_b)
        g_z = z_arr_g.grad.numpy().copy()
        tape_b.reset()
        eps_b = 1.0e-3
        fd_b = (band_loss(2.0 + eps_b) - band_loss(2.0 - eps_b)) / (2 * eps_b)
        err_b = abs(float(g_z[9]) - fd_b)
        rel_b = err_b / max(abs(fd_b), 1e-12)
        leak_b = float(np.abs(np.where(np.arange(17) == 9, 0.0, g_z)).max())
        assert err_b <= 5e-5 or rel_b <= 0.15, (
            f"loss-stage guard (apex band): tape={g_z[9]:.5e} fd={fd_b:.5e} rel={rel_b:.2e} - zero/garbage adjoint?"
        )
        assert leak_b <= 1e-6 * max(abs(fd_b), 1.0), (
            f"loss-stage guard (apex band): adjoint leaked off the max element (max|g| elsewhere = {leak_b:.2e})"
        )
        print(
            f"loss-stage adjoint guard: order tape={g_pz:.5e} fd={fd_pz:.5e}; band tape={float(g_z[9]):.5e} fd={fd_b:.5e}",
            flush=True,
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
