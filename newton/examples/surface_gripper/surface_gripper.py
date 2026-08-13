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

"""Surface-gripper model: authoring types, the finalized model/state/control,
and the per-pad force kernel. Imported by ``example_surface_gripper``; not a runnable example.

Mirrors Newton's Builder -> Model -> State/Control layout::

    SurfaceGripper (authoring) -> SurfaceGripperBuilder.finalize() -> SurfaceGripperModel
    SurfaceGripperModel.state_input() / .state_output() / .control() -> SurfaceGripperStateInput / SurfaceGripperStateOutput / SurfaceGripperControl

Frame nomenclature
------------------
All transforms are rigid-body poses (position + quaternion).  ``X_in_Y`` means frame X
expressed relative to frame Y; ``X^-1`` is its inverse.

GA    World pose of the end-effector (robot flange / body A).
      ``body_q[gripper_body_id]``

sgA   Pose of the surface-gripper assembly in the end-effector frame.
      ``gripper_xform``

padA  Pose of an individual suction pad in the surface-gripper frame.
      ``pad_xform``

SA    World pose of the pad seal frame: ``SA = GA * sgA * padA``.
      Called ``seal_world`` in the kernels.

GB    World pose of the gripped body (body B).
      ``body_q[body_b]``

T_bs  Pose of the gripped body's SDF mesh in body B's local frame (mesh -> body B).
      ``model.shape_transform[shape_b]`` where ``shape_b`` is the gripped collision shape ID.
      The world pose of the mesh is ``GB * T_bs``.

SB    Pose of the seal frame in body B's local frame, cached at engagement:
      ``SB = GB0^-1 * SA0`` where the subscript 0 denotes values at engagement time.
      Stored in ``pad_anchor_b``.  The force kernel reconstructs the world seal pose
      on body B as ``GB * SB`` and measures the bias ``SA^-1 * GB * SB`` against identity.
"""

import math

import warp as wp

import newton
from newton.geometry import sdf_mesh

# 6-vector / 6x6 matrix types: seal DOF wrenches (fx, fy, fz, mx, my, mz) and the Gauss-Newton normal equations.
_mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
_vec6 = wp.types.vector(length=6, dtype=wp.float32)


# --------------- internal helpers and kernels ---------------


def _nat_freq_damping_ratio_to_stiffness_damping(mu: float, zeta: float, m_eff: float) -> tuple[float, float]:
    """``(k, d)`` for a 1-DOF spring-damper of effective mass/inertia ``m_eff`` tuned to angular natural
    frequency ``mu`` [rad/s] and damping ratio ``zeta``: ``k = m_eff*mu^2``, ``d = 2*zeta*mu*m_eff``.
    ``m_eff`` is a mass [kg] for a translation DOF, an inertia [kg.m^2] for a rotation DOF.
    """
    return m_eff * mu * mu, 2.0 * zeta * mu * m_eff


def _perimeter_circle(radius: float, half_height: float, n_samples: int) -> list[wp.vec3]:
    """The ``n_samples`` perimeter sample points of one pad, in the pad frame: evenly spaced on the circle of
    ``radius`` at height ``half_height`` (the perimeter-plane offset along the grip axis). Returns a list of
    ``wp.vec3``; empty when ``n_samples`` is 0."""
    points = []
    if n_samples <= 0:
        return points
    d_th = 2.0 * math.pi / float(n_samples)
    for s in range(n_samples):
        th = d_th * float(s)
        points.append(wp.vec3(radius * math.cos(th), radius * math.sin(th), half_height))
    return points


@wp.func
def _eval_pad_separation(
    t_seal_a: wp.transform, t_seal_b: wp.transform
) -> tuple[float, float, float, float, float, float]:
    """Per-DOF separation of the two seal frames accumulated since engagement.

    Args:
        t_seal_a: Current world pose of SA (see Frame nomenclature).
        t_seal_b: Current world pose of GB*SB; equals SA at engagement.
    Returns:
        ``(px, py, pz, theta_x, theta_y, theta_z)`` [m, m, m, rad, rad, rad]: shear
        along x/y, normal along z, peel about x/y, and twist about z.
    """

    # bias = SA^-1 * GB * SB (see Frame nomenclature; equals identity at engagement)
    t_rel = wp.transform_inverse(t_seal_a) * t_seal_b
    p = wp.transform_get_translation(t_rel)
    q = wp.transform_get_rotation(t_rel)
    # small-angle rotation vector: theta ~ 2*(qx, qy, qz) for a near-identity quaternion
    return p[0], p[1], p[2], 2.0 * q[0], 2.0 * q[1], 2.0 * q[2]


@wp.func
def _eval_pad_relative_velocity(
    twist_a_world: wp.spatial_vector,
    twist_b_world: wp.spatial_vector,
    r_a_world: wp.vec3,
    r_b_world: wp.vec3,
    q_seal: wp.quat,
) -> tuple[float, float, float, float, float, float]:
    """Relative velocity of the seal point (B minus A), expressed in the seal frame.

    Args:
        twist_a_world: Spatial velocity of body A, (linear, angular) in the world frame.
        twist_b_world: Spatial velocity of body B, (linear, angular) in the world frame.
        r_a_world: World-frame offset from body A's COM to the seal point [m].
        r_b_world: World-frame offset from body B's COM to the seal point [m].
        q_seal: World orientation of the seal frame.

    Returns:
        ``(vx, vy, vz, omega_x, omega_y, omega_z)`` [m/s x3, rad/s x3]: the velocity of
        B's seal point relative to A's, rotated into the seal frame. ``vz`` feeds the
        normal damper; ``omega_x``/``omega_y`` feed the peel dampers.
    """
    # split each twist into its linear (at the COM) and angular parts.
    # all four are expressed in the WORLD frame (Newton stores body twists in world).
    vLinA = wp.spatial_top(twist_a_world)  # world-frame linear velocity of body A's COM
    vAngA = wp.spatial_bottom(twist_a_world)  # world-frame angular velocity of body A
    vLinB = wp.spatial_top(twist_b_world)  # world-frame linear velocity of body B's COM
    vAngB = wp.spatial_bottom(twist_b_world)  # world-frame angular velocity of body B

    # velocity of each seal point in world = v_com + omega x r (r = world offset from COM)
    vSealA = vLinA + wp.cross(vAngA, r_a_world)
    vSealB = vLinB + wp.cross(vAngB, r_b_world)

    # relative velocity of B's seal point w.r.t. A's, taken into the seal frame
    v = wp.quat_rotate_inv(q_seal, vSealB - vSealA)
    w = wp.quat_rotate_inv(q_seal, vAngB - vAngA)
    return v[0], v[1], v[2], w[0], w[1], w[2]


@wp.func
def _apparent_mass(axis_w: wp.vec3, r: wp.vec3, q_b: wp.quat, inv_m: float, inv_inertia: wp.mat33) -> float:
    """Effective mass a rigid body presents to a force at offset ``r`` (world, COM->point) along the
    world unit direction ``axis_w``: ``1/m_app = 1/m + (r x n).I^-1.(r x n)`` -- translational compliance
    plus the rotational compliance from the off-COM spin. The rotational term is evaluated in the body
    frame (``q_b``, ``inv_inertia``), where the inertia tensor is stored.
    """
    c_b = wp.quat_rotate_inv(q_b, wp.cross(r, axis_w))  # (r x n) in the body frame
    return 1.0 / (inv_m + wp.dot(c_b, inv_inertia * c_b))


@wp.func
def _apparent_inertia(axis_w: wp.vec3, q_b: wp.quat, inv_inertia: wp.mat33) -> float:
    """Effective inertia a rigid body presents to a pure moment about the world unit axis ``axis_w``:
    ``1/(n.I^-1.n)`` -- the free-body angular admittance about the axis, inverted. The axis is first
    rotated into the body frame (``q_b``, where ``inv_inertia`` lives). No translation term: a couple
    does not move the COM.
    """
    axis_b = wp.quat_rotate_inv(q_b, axis_w)  # seal axis in the body frame
    return 1.0 / wp.dot(axis_b, inv_inertia * axis_b)


@wp.func
def _effective_damping(d: float, m_eff: float, dt: float) -> float:
    """Backward-Euler (implicit) damping coefficient: the explicit ``d`` rescaled so the applied force
    lands the pad-point velocity at the implicit value ``v*m_eff/(m_eff + d*dt)`` in one step. Bounded by
    ``m_eff/dt`` for any ``d`` (the damper can't overshoot the velocity). ``m_eff`` is the DOF's effective
    mass (translation) or effective inertia (rotation).
    """
    return d / (1.0 + d * dt / m_eff)


@wp.func
def _eval_effective_damping(
    q_seal: wp.quat,  # seal frame world orientation
    q_body_b: wp.quat,  # gripped body world orientation
    r_body_b: wp.vec3,  # gripped body COM -> seal point (world)
    mass_b: float,  # gripped body mass
    inertia_b: wp.mat33,  # gripped body inertia tensor (body frame)
    d_normal: float,
    d_shear_x: float,
    d_shear_y: float,
    d_peel_x: float,
    d_peel_y: float,
    d_torsion: float,
    dt: float,
) -> tuple[float, float, float, float, float, float]:
    """Implicit (backward-Euler) damping coefficient for each of the six damped seal DOFs. Each raw
    damping is rescaled via :func:`_effective_damping` using that DOF's effective mass at the seal point
    (:func:`_apparent_mass`, translation) or effective inertia about the seal axis, 1/(n.I^-1.n) (peel
    and twist). Returns ``(d_normal_eff, d_shear_x_eff, d_shear_y_eff, d_peel_x_eff, d_peel_y_eff,
    d_torsion_eff)``. Pass ``d_torsion = 0`` when the caller's twist DOF is not viscously damped.
    """
    # Step 1.
    # Implicit (backward-Euler) damping.
    # F = M * [v(t+dt) - v(t)] / dt = -gamma * v(t+dt)
    # Solver for v(t+dt)
    # v(t+dt) = v(t) * [M / (M + gamma * dt)] * v(t)
    # Now compute the force required to move from v(t) to v(t+dt) in dt.
    # F = (M/dt) *  {[M / (M + gamma * dt)] - 1} * v(t)
    # Rearrange
    # F = (M/dt) * [-gamma*dt]/[M + gamma*dt]
    # Some more algebra reveals
    # F = - gamma/[1 + gamma*dt/M] * v(t)
    # This looks like a damping equation with simple substitutions
    # F = -gamma_effective * v(t)
    # with
    # gamma_effective = gamma/[1 + gamma*dt/M]

    # Step 2.
    # For translational dofs.
    # We apply the force at a vector r from the COM of the picked body.
    # The goal is to prevent the dof speeds of the picked body measured
    # at its COM from changing sign under damping.
    # Compute the acceleration a at the COM that arises from a force F applied
    # along a vector n.
    # a = a_com + alpha X r
    # with a_com = (F/M)n
    # and alpha the angular acceleration.
    # Now project a along n
    # a.n = (F/M) + (alpha X r).n
    # Now apply the triple product rule:
    # (alpha X r).n = alpha.(r X n)
    # and we now have
    # a.n = (F/M) + alpha.(r X n)
    # We can compute alpha:
    # I * alpha = F*(r X n) so alpha = F*[I^-1 * (r X n)]
    # a.n = (F/M) + F*[I^-1.(r X n)].(r X n)
    # I is symmetric so we have
    # a.n = (F/M) + F*[(r X n).I^-1].(r X n)
    # This reveals an effective mass
    # m_eff = 1/M + (r X n).I^-1.(r X n)

    # Step 2 for rotational dofs.
    # Applying a torque tau around an axis n follows:
    # I*alpha = tau*n
    # Solve for alpha
    # alpha = tau (I^-1.n)
    # Now project alpha onto n
    # alpha.n = tau (n.I^-1.n)
    # We can now say that the effective inertia I_eff obeys
    # I_eff^-1  = n.I^-1.n
    # Rearrange
    # I_eff = 1/(n.I^-1.n)

    # Compute the seal axes in the world frame.
    axes = wp.matrix_from_rows(
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
    )  # the seal-frame basis axes x, y, z
    # rotate each seal axis into world once, reused by translation and peel below
    axes_w = wp.mat33(0.0)
    for i in range(3):
        axes_w[i] = wp.quat_rotate(q_seal, axes[i])

    inv_m = wp.where(mass_b > 0.0, 1.0 / mass_b, 0.0)
    inv_inertia = wp.inverse(inertia_b)

    # Combine Step 1 and 2 for translational dofs
    # gamma_effective = gamma/[1 + gamma*dt/m_eff]
    d_trans_eff = wp.vec3(0.0, 0.0, 0.0)
    d_trans = wp.vec3(d_shear_x, d_shear_y, d_normal)
    for i in range(3):
        m_app = _apparent_mass(axes_w[i], r_body_b, q_body_b, inv_m, inv_inertia)
        d_trans_eff[i] = _effective_damping(d_trans[i], m_app, dt)

    # Combine Step 1 and 2 for rotational dofs (peel about x/y, twist about z)
    # gamma_effective = gamma/[1 + gamma*dt/I_eff]
    d_rot_eff = wp.vec3(0.0, 0.0, 0.0)
    d_rot = wp.vec3(d_peel_x, d_peel_y, d_torsion)
    for i in range(3):
        I_app = _apparent_inertia(axes_w[i], q_body_b, inv_inertia)
        d_rot_eff[i] = _effective_damping(d_rot[i], I_app, dt)

    return d_trans_eff[2], d_trans_eff[0], d_trans_eff[1], d_rot_eff[0], d_rot_eff[1], d_rot_eff[2]


@wp.func
def _clamp_symmetric(f: float, f_max: float) -> float:
    """Clamp ``f`` to ``[-f_max, f_max]``. ``f_max <= 0`` means no cap (``f`` returned unchanged)."""
    if f_max > 0.0:
        return wp.clamp(f, -f_max, f_max)
    return f


@wp.func
def _clamp_magnitude_2d(fx: float, fy: float, f_max: float) -> tuple[float, float]:
    """Scale the pair ``(fx, fy)`` onto the disk of radius ``f_max``: if ``sqrt(fx^2+fy^2) > f_max``
    both components are scaled by ``f_max / mag``. ``f_max <= 0`` means no cap (returned unchanged)."""
    if f_max > 0.0:
        mag = wp.sqrt(fx * fx + fy * fy)
        if mag > f_max:
            s = f_max / mag
            fx = fx * s
            fy = fy * s
    return fx, fy


@wp.kernel
def _attach_seal_kernel(
    pad_engaged_bs_curr: wp.array[wp.vec2i],  # [pads] current step's gripped body/shape (``[0] < 0`` = released)
    pad_engaged_bs_prev: wp.array[
        wp.vec2i
    ],  # [pads] previous step's gripped body/shape (``[0] < 0`` = was released, for rising-edge detection)
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],  # world pose of body A (the gripper body)
    hold_pose_body_b: wp.array[
        wp.transform
    ],  # GB0 per body: raw body_q (attach_seal) or the fitted seated pose (attach_seal_seated)
    # outputs
    pad_anchor_b: wp.array[wp.transform],
):
    """On a disengaged->engaged rising edge, cache SB = GB0^-1 * SA0 into pad_anchor_b (see Frame nomenclature)."""
    pad = wp.tid()
    body_b = pad_engaged_bs_curr[pad][0]  # the body this pad is engaging to; the gate below keeps it valid
    if body_b >= 0 and pad_engaged_bs_prev[pad][0] < 0:
        gripper_id = pad_gripper[pad]
        seal_world = body_q[gripper_body_id[gripper_id]] * gripper_xform[gripper_id] * pad_xform[pad]  # SA
        pad_anchor_b[pad] = wp.transform_inverse(hold_pose_body_b[body_b]) * seal_world


@wp.kernel
def _eval_pad_force_linear_kernel(
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    gripper_k_normal: wp.array[float],
    gripper_d_normal: wp.array[float],
    gripper_f_grip_max: wp.array[float],
    gripper_k_shear_x: wp.array[float],
    gripper_d_shear_x: wp.array[float],
    gripper_k_shear_y: wp.array[float],
    gripper_d_shear_y: wp.array[float],
    gripper_f_shear_max: wp.array[float],
    gripper_k_peel_x: wp.array[float],
    gripper_d_peel_x: wp.array[float],
    gripper_k_peel_y: wp.array[float],
    gripper_d_peel_y: wp.array[float],
    gripper_f_peel_max: wp.array[float],
    gripper_k_torsion: wp.array[float],
    gripper_d_torsion: wp.array[float],
    gripper_f_torsion_max: wp.array[float],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    pad_grip_control: wp.array[float],
    pad_engaged_bs: wp.array[wp.vec2i],
    pad_anchor_b: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    dt: float,
    # outputs (mutated in place)
    pad_seal_load: wp.array[wp.vec4],  # out: (normal, shear, peel, torsion) after the caps
    pad_seal_load_unclamped: wp.array[wp.vec4],  # out: same four groups before the caps
    body_f: wp.array[wp.spatial_vector],
):
    """Per-pad linear spring-damper seal wrench with fixed magnitude caps (see the section header).

    Seal kinematics (seal-frame separation, seal-point relative velocity, implicit damping) drive a
    plain linear force per DOF with four caps (normal, combined shear, combined peel, twist). Writes the
    equal-and-opposite wrench into ``body_f``.
    """
    pad = wp.tid()
    if pad_engaged_bs[pad][0] < 0:
        # A released pad carries nothing. Clear the reported loads so they do not read as the last
        # value the seal held before it let go.
        pad_seal_load[pad] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        pad_seal_load_unclamped[pad] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        return
    engaged_body_b = pad_engaged_bs[pad][0]

    gripper_id = pad_gripper[pad]
    body_a = gripper_body_id[gripper_id]

    # SA and GB*SB in world frame; separation and relative velocity per DOF (see Frame nomenclature)
    t_a_seal = body_q[body_a] * gripper_xform[gripper_id] * pad_xform[pad]
    q_a_seal = wp.transform_get_rotation(t_a_seal)
    p_a_seal = wp.transform_get_translation(t_a_seal)
    t_b_seal = body_q[engaged_body_b] * pad_anchor_b[pad]
    p_b_seal = wp.transform_get_translation(t_b_seal)

    px, py, pz, theta_x, theta_y, theta_z = _eval_pad_separation(t_a_seal, t_b_seal)

    com_a = wp.transform_point(body_q[body_a], body_com[body_a])
    com_b = wp.transform_point(body_q[engaged_body_b], body_com[engaged_body_b])
    r_a = p_a_seal - com_a
    r_b = p_b_seal - com_b
    vx, vy, vz, omega_x, omega_y, omega_z = _eval_pad_relative_velocity(
        body_qd[body_a], body_qd[engaged_body_b], r_a, r_b, q_a_seal
    )

    # implicit (backward-Euler) damping for all six DOFs (normal, shear x/y, peel x/y, twist)
    q_body_b = wp.transform_get_rotation(body_q[engaged_body_b])
    d_normal_eff, d_shear_x_eff, d_shear_y_eff, d_peel_x_eff, d_peel_y_eff, d_torsion_eff = _eval_effective_damping(
        q_a_seal,
        q_body_b,
        r_b,
        body_mass[engaged_body_b],
        body_inertia[engaged_body_b],
        gripper_d_normal[gripper_id],
        gripper_d_shear_x[gripper_id],
        gripper_d_shear_y[gripper_id],
        gripper_d_peel_x[gripper_id],
        gripper_d_peel_y[gripper_id],
        gripper_d_torsion[gripper_id],
        dt,
    )

    # normal (z): a pure spring pull capped at the vacuum grip (control * f_grip_max) -- no preload
    # baseline, so the spring has authority from zero stretch up to the vacuum, and the cap is the
    # pull-off limit (pull harder than the vacuum and the pad lets go). Compression (push) is left
    # uncapped (contact-like). Damping is applied on top of the cap.
    f_vac = pad_grip_control[pad] * gripper_f_grip_max[gripper_id]
    fz_elastic_raw = gripper_k_normal[gripper_id] * pz
    fz_elastic = fz_elastic_raw
    if fz_elastic > f_vac:
        fz_elastic = f_vac
    fz = fz_elastic + d_normal_eff * vz
    fz_unclamped = fz_elastic_raw + d_normal_eff * vz

    # shear (x, y): spring-damper per axis, combined magnitude capped at f_shear_max
    fx_raw = gripper_k_shear_x[gripper_id] * px + d_shear_x_eff * vx
    fy_raw = gripper_k_shear_y[gripper_id] * py + d_shear_y_eff * vy
    fx, fy = _clamp_magnitude_2d(fx_raw, fy_raw, gripper_f_shear_max[gripper_id])

    # peel (about x, y): spring-damper per axis, combined magnitude capped at f_peel_max
    m_peel_x_raw = gripper_k_peel_x[gripper_id] * theta_x + d_peel_x_eff * omega_x
    m_peel_y_raw = gripper_k_peel_y[gripper_id] * theta_y + d_peel_y_eff * omega_y
    m_peel_x, m_peel_y = _clamp_magnitude_2d(m_peel_x_raw, m_peel_y_raw, gripper_f_peel_max[gripper_id])

    # twist (about z): linear spring-damper, clamped to +/-f_torsion_max
    m_twist_raw = gripper_k_torsion[gripper_id] * theta_z + d_torsion_eff * omega_z
    m_twist = _clamp_symmetric(m_twist_raw, gripper_f_torsion_max[gripper_id])

    # assemble the seal-frame wrench, rotate to world, and accumulate equal-and-opposite on A and B
    force = wp.quat_rotate(q_a_seal, wp.vec3(fx, fy, fz))
    torque = wp.quat_rotate(q_a_seal, wp.vec3(m_peel_x, m_peel_y, m_twist))
    wp.atomic_add(body_f, body_a, wp.spatial_vector(force, torque + wp.cross(r_a, force)))
    wp.atomic_add(body_f, engaged_body_b, wp.spatial_vector(-force, -torque + wp.cross(r_b, -force)))

    # Group the six seal-frame components into the four DOF loads the caps act on.
    # Shear and peel become magnitudes because their caps are direction-independent disks.
    shear_mag = wp.sqrt(fx * fx + fy * fy)
    peel_mag = wp.sqrt(m_peel_x * m_peel_x + m_peel_y * m_peel_y)
    shear_mag_raw = wp.sqrt(fx_raw * fx_raw + fy_raw * fy_raw)
    peel_mag_raw = wp.sqrt(m_peel_x_raw * m_peel_x_raw + m_peel_y_raw * m_peel_y_raw)

    pad_seal_load[pad] = wp.vec4(fz, shear_mag, peel_mag, m_twist)
    pad_seal_load_unclamped[pad] = wp.vec4(fz_unclamped, shear_mag_raw, peel_mag_raw, m_twist_raw)


@wp.func
def _solve6(a: _mat66, b: _vec6) -> _vec6:
    """Solve the 6x6 system ``a x = b`` by Gaussian elimination with partial pivoting (``a`` assumed
    non-singular, e.g. damped normal equations). Operates on local copies."""
    m = a
    x = b
    for c in range(6):
        piv = c
        big = wp.abs(m[c, c])
        for rr in range(c + 1, 6):
            if wp.abs(m[rr, c]) > big:
                big = wp.abs(m[rr, c])
                piv = rr
        if piv != c:
            for k in range(6):
                tmp = m[c, k]
                m[c, k] = m[piv, k]
                m[piv, k] = tmp
            t = x[c]
            x[c] = x[piv]
            x[piv] = t
        for rr in range(c + 1, 6):
            f = m[rr, c] / m[c, c]
            for k in range(c, 6):
                m[rr, k] = m[rr, k] - f * m[c, k]
            x[rr] = x[rr] - f * x[c]
    y = _vec6()
    for i in range(5, -1, -1):
        s = x[i]
        for k in range(i + 1, 6):
            s = s - m[i, k] * y[k]
        y[i] = s / m[i, i]
    return y


@wp.func
def _sdf_grad(mesh_id: wp.uint64, q: wp.vec3, max_dist: float, h: float) -> wp.vec3:
    """Central-difference gradient of :func:`newton.geometry.sdf_mesh` at ``q`` (object frame)."""
    gx = sdf_mesh(mesh_id, q + wp.vec3(h, 0.0, 0.0), max_dist) - sdf_mesh(mesh_id, q - wp.vec3(h, 0.0, 0.0), max_dist)
    gy = sdf_mesh(mesh_id, q + wp.vec3(0.0, h, 0.0), max_dist) - sdf_mesh(mesh_id, q - wp.vec3(0.0, h, 0.0), max_dist)
    gz = sdf_mesh(mesh_id, q + wp.vec3(0.0, 0.0, h), max_dist) - sdf_mesh(mesh_id, q - wp.vec3(0.0, 0.0, h), max_dist)
    return wp.vec3(gx, gy, gz) / (2.0 * h)


@wp.func
def _seat_body_pose(
    body_b: int,
    pad_lo: int,  # scan only [pad_lo, pad_hi): the pads of body body_b's world (see pad_world_start)
    pad_hi: int,
    pad_target_bs: wp.array[wp.vec2i],  # membership: a pad joins the fit when pad_target_bs[p][0] == body_b
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],  # shape_transform[shape_b] = T_bs for the gripped shape
    shape_b: int,  # global shape ID of the gripped shape (indexes shape_transform)
    mesh_id: wp.uint64,
    pad_perimeter_local: wp.array[
        wp.vec3
    ],  # precomputed perimeter points in the pad frame, indexed by pad_perimeter_start (start indices per pad)
    pad_perimeter_start: wp.array[
        int
    ],  # [n_pads+1] start indices: pad p's perimeter points are pad_perimeter_local[pad_perimeter_start[p] : pad_perimeter_start[p+1]]
    max_dist: float,
    grad_h: float,
    damping: float,
    iters: int,
) -> wp.transform:
    """
    Compute the pose of body b that minimises the rms of the signed distances of sample points
    arranged around the contact perimeters of all pads gripping (or preparing to grip) body b.
    Using the Frame nomenclature defined in the module docstring, with the additional symbols:
    perimeterA = sample point in the pad frame
    perimeterB = same point in the gripped mesh frame: perimeterB = (GB * T_bs)^-1 * SA * perimeterA
    It makes sense to cache (GB * T_bs)^-1 * SA for each pad being considered.
    For each sample point perimeterB we compute the signed distance using the sdf mesh.
    A least squares algorithm is employed to compute the pose of body b that
    minimises the rms of the signed distances of all sample points of all pads
    that are gripping (or preparing to grip) body b.
    The least square algorithm imagines a twist (v,w) applied to body b and
    computes the change to the signed distance that the twist applies to each sample
    point. This requires knowledge of the gradient of the signed distance at each sample point.
    The twist (v,w) applied to body b moves the sample point by -(v + w X q) in
    the frame of the mesh associated with body b.
    The change to the signed distance is dSdf = -grad.(v + w X q) with grad denoting
    the per dof gradient of the signed distance meausured around the sample point.
    The triple product rule may be applied as follows:
    dSdf = -grad.(v + w X q) = -grad.V - (q X grad).w = [grad, q X grad].[v, w]
    We seek the twist that results in Sdf + dSdf = 0.
    We have more equations than unknowns so a least squares algorithm is employed.
    (J^T*J) * (v, w) = J^T * residual
    Note: it is necessary to compute the signed distance and its gradient in the frame
    of the mesh.  The term (v + w X q) is, however, in the frame of body b.  It is
    therefore necessary to compute [grad, q X grad] in the frame of body b as well.
    We compute grad_mesh and sample_point_in_mesh and need to transform these into the
    frame of body b.
    [grad, q X grad] = [T_bs.rotate_only(grad_mesh),  (T_bs.transform(sample_point_in_mesh) X T_bs.rotate_only(grad_mesh)]
    """
    # T_bs and its rotation are constant for this body; precompute once before the Gauss-Newton loop.
    T_bs = shape_transform[shape_b]
    R_bs = wp.transform_get_rotation(T_bs)
    tb = body_q[body_b]  # current pose estimate; refined by each Gauss-Newton iteration below
    for _ in range(iters):
        # (GB * T_bs)^-1
        inv_mesh_world = wp.transform_inverse(tb * T_bs)
        jtj = _mat66()
        rhs = _vec6()
        for p in range(pad_lo, pad_hi):
            # a pad participates in the fit when it is targeting this body (engaged or preparing)
            if pad_target_bs[p][0] == body_b:
                gripper_id = pad_gripper[p]
                seal_world_pad = body_q[gripper_body_id[gripper_id]] * gripper_xform[gripper_id] * pad_xform[p]  # SA
                t_rel_mesh = inv_mesh_world * seal_world_pad  # (GB · T_bs)^-1 · SA: seal frame in mesh-local
                for k in range(pad_perimeter_start[p], pad_perimeter_start[p + 1]):
                    # perimeterB = (GB · T_bs)^-1 · SA · perimeterA: sample point in mesh frame
                    sample_point_in_mesh = wp.transform_point(t_rel_mesh, pad_perimeter_local[k])
                    sdf = sdf_mesh(mesh_id, sample_point_in_mesh, max_dist)
                    grad_mesh = _sdf_grad(mesh_id, sample_point_in_mesh, max_dist, grad_h)
                    # Jacobian uses body-frame quantities: rotate point and gradient from mesh to body frame.
                    x_body = wp.transform_point(T_bs, sample_point_in_mesh)
                    grad_body = wp.quat_rotate(R_bs, grad_mesh)
                    cr = wp.cross(x_body, grad_body)
                    a = _vec6(
                        grad_body[0], grad_body[1], grad_body[2], cr[0], cr[1], cr[2]
                    )  # Jacobian row (sdf grows as -a . xi)
                    jtj = jtj + wp.outer(a, a)
                    rhs = rhs + a * sdf
        for k in range(6):
            jtj[k, k] = jtj[k, k] + damping  # steadies the fit; without it a flat face could slide/spin freely
        dxi = _solve6(jtj, rhs)
        v = wp.vec3(dxi[0], dxi[1], dxi[2])
        om = wp.vec3(dxi[3], dxi[4], dxi[5])
        ang = wp.length(om)
        axis = wp.vec3(0.0, 0.0, 1.0)
        if ang > 1.0e-6:
            axis = om / ang
        tb = tb * wp.transform(v, wp.quat_from_axis_angle(axis, ang))  # apply the step, then re-linearize
    return tb


@wp.kernel
def _attach_seal_seated_kernel(
    pad_engaged_bs_curr: wp.array[wp.vec2i],  # [pads] current step's gripped body/shape (``[0] < 0`` = released)
    pad_engaged_bs_prev: wp.array[
        wp.vec2i
    ],  # [pads] previous step's gripped body/shape (``[0] < 0`` = was released, for rising-edge detection)
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    shape_mesh_id: wp.array[wp.uint64],  # shape_id -> SDF mesh id (for the inline seat fit)
    shape_transform: wp.array[wp.transform],  # shape_transform[shape_b] = T_bs for the gripped shape
    pad_world: wp.array[int],  # world of each pad (see SurfaceGripperModel)
    pad_world_start: wp.array[
        int
    ],  # start indices: world w's pads are pad-ids [pad_world_start[w] : pad_world_start[w+1]]
    pad_perimeter_local: wp.array[
        wp.vec3
    ],  # precomputed perimeter points in the pad frame, indexed by pad_perimeter_start (start indices per pad)
    pad_perimeter_start: wp.array[
        int
    ],  # [n_pads+1] start indices: pad p's perimeter points are pad_perimeter_local[pad_perimeter_start[p] : pad_perimeter_start[p+1]]
    max_dist: float,  # SDF search radius [m]
    grad_h: float,  # SDF central-difference step [m]
    damping: float,  # small stabiliser: steadies the fit when the pads don't fully pin the gripped object
    iters: int,  # Gauss-Newton iterations for the seat fit (1 for planar faces, more for curved objects)
    # outputs
    pad_anchor_b: wp.array[wp.transform],
    pad_perimeter_sdf0: wp.array[
        float
    ],  # seated perimeter signed distances cached at engagement (indexed by pad_perimeter_start)
):
    """Seated variant of :func:`_attach_seal_kernel`: on each pad's rising edge, compute the gripped body's
    seated pose inline (:func:`_seat_body_pose`, scanning only this pad's world's pads), cache
    ``pad_anchor_b`` against it, and cache the seated perimeter signed distances in ``pad_perimeter_sdf0``. The seat
    fit only runs on the rising edge. Does not write ``pad_engaged_bs``."""
    pad = wp.tid()
    body_b = pad_engaged_bs_curr[pad][0]  # the body this pad is engaging to; the gate below keeps it valid
    shape_b = pad_engaged_bs_curr[pad][1]  # its collision shape, used to index shape_mesh_id below
    # Both ids must be valid: shape_b indexes shape_mesh_id and shape_transform, so a negative one
    # would read out of bounds even when body_b is a real body.
    if body_b >= 0 and shape_b >= 0 and pad_engaged_bs_prev[pad][0] < 0:  # rising edge: seat and cache SB
        w = pad_world[pad]  # this pad's world; scan only that world's pads for body_b
        hold_pose_body_b = _seat_body_pose(
            body_b,
            pad_world_start[w],
            pad_world_start[w + 1],
            pad_engaged_bs_curr,  # membership: every pad engaging this body joins the same seat fit
            gripper_body_id,
            gripper_xform,
            pad_gripper,
            pad_xform,
            body_q,
            shape_transform,
            shape_b,
            shape_mesh_id[shape_b],
            pad_perimeter_local,
            pad_perimeter_start,
            max_dist,
            grad_h,
            damping,
            iters,
        )

        # Cache SB = GB0^-1 * SA (see Frame nomenclature)
        gripper_id = pad_gripper[pad]
        seal_world = body_q[gripper_body_id[gripper_id]] * gripper_xform[gripper_id] * pad_xform[pad]  # SA
        pad_anchor_b[pad] = wp.transform_inverse(hold_pose_body_b) * seal_world
        # Cache pad_perimeter_sdf0: SDF at each perimeter point placed by SB (in mesh frame). Seal-quality metric measures deviation from these.
        mesh_b = shape_mesh_id[shape_b]
        pad_anchor_in_mesh = wp.transform_inverse(shape_transform[shape_b]) * pad_anchor_b[pad]
        for k in range(pad_perimeter_start[pad], pad_perimeter_start[pad + 1]):
            perimeter_mesh = wp.transform_point(pad_anchor_in_mesh, pad_perimeter_local[k])
            pad_perimeter_sdf0[k] = sdf_mesh(mesh_b, perimeter_mesh, max_dist)


@wp.kernel
def _seal_quality_kernel(
    pad_engaged_bs: wp.array[wp.vec2i],  # [pads] latched gripped body/shape while engaged (``[0] < 0`` = released)
    pad_preparing_bs: wp.array[
        wp.vec2i
    ],  # [pads] body/shape a preparing pad is approaching (``[0] >= 0`` means preparing)
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    shape_mesh_id: wp.array[wp.uint64],  # shape_id -> SDF mesh id
    shape_transform: wp.array[wp.transform],  # shape_transform[shape_b] = T_bs for the gripped shape
    pad_world: wp.array[int],
    pad_world_start: wp.array[int],
    pad_perimeter_local: wp.array[
        wp.vec3
    ],  # precomputed perimeter points in the pad frame, indexed by pad_perimeter_start (start indices per pad)
    pad_perimeter_sdf0: wp.array[float],  # seated perimeter SDFs cached at engagement (used only for engaged pads)
    pad_perimeter_start: wp.array[
        int
    ],  # [n_pads+1] start indices: pad p's perimeter points are pad_perimeter_local[pad_perimeter_start[p] : pad_perimeter_start[p+1]]
    max_dist: float,
    grad_h: float,  # seat-fit params (preparing case only)
    damping: float,
    iters: int,
    # output: one root-mean-square value per pad. Each thread writes only its own pad, so no atomics.
    pad_rms: wp.array[float],  # [pads] this pad's RMS perimeter-gap deviation [m] (-1 if not gripping/preparing)
):
    pad = wp.tid()
    pad_rms[pad] = -1.0  # sentinel: "not evaluated" (overwritten below when engaged or preparing)
    # The three modes are exclusive, and engaged wins: once a seal has formed, its quality is the
    # drift from the seated pose, not the live fit error of an approach that is already over.
    engaged = pad_engaged_bs[pad][0] >= 0
    preparing = (not engaged) and pad_preparing_bs[pad][0] >= 0
    if not (engaged or preparing):
        return  # disengaged mode: pad_rms stays -1
    if engaged:
        body_b = pad_engaged_bs[pad][0]  # engaged: the latched gripped body
        shape_b = pad_engaged_bs[pad][1]
    else:
        body_b = pad_preparing_bs[pad][0]  # preparing: the crate being approached
        shape_b = pad_preparing_bs[pad][1]
    if shape_b < 0:
        return  # shape_b indexes shape_mesh_id and shape_transform below; pad_rms stays -1
    gripper_id = pad_gripper[pad]
    seal_world = body_q[gripper_body_id[gripper_id]] * gripper_xform[gripper_id] * pad_xform[pad]
    mesh_b = shape_mesh_id[shape_b]
    # world pose of the mesh: body world pose composed with the mesh-in-body pose (GB * T_bs)
    mesh_world = body_q[body_b] * shape_transform[shape_b]
    # seal frame in mesh-local coordinates: (GB * T_bs)^-1 * SA
    t_seal_mesh = wp.transform_inverse(mesh_world) * seal_world
    preparing_anchor_mesh = (
        wp.transform_identity()
    )  # seated seal frame in mesh frame (preparing only; recomputed below)
    if preparing:
        w = pad_world[pad]
        seated = _seat_body_pose(  # recompute the seated pose of the approached crate
            body_b,
            pad_world_start[w],
            pad_world_start[w + 1],
            pad_preparing_bs,  # membership: matches body_b above, which is preparing-only in this branch
            gripper_body_id,
            gripper_xform,
            pad_gripper,
            pad_xform,
            body_q,
            shape_transform,
            shape_b,
            mesh_b,
            pad_perimeter_local,
            pad_perimeter_start,
            max_dist,
            grad_h,
            damping,
            iters,
        )
        preparing_anchor_body = wp.transform_inverse(seated) * seal_world  # SB_preparing = GB0^-1 * SA (body frame)
        preparing_anchor_mesh = (
            wp.transform_inverse(shape_transform[shape_b]) * preparing_anchor_body
        )  # T_bs^-1 * SB_preparing (mesh frame)
    dev_sq = float(0.0)  # this pad's running totals (this thread owns pad_rms[pad], so no atomics needed)
    count = int(0)
    for k in range(pad_perimeter_start[pad], pad_perimeter_start[pad + 1]):
        perimeter_point = pad_perimeter_local[k]
        sdf0 = pad_perimeter_sdf0[k]  # engaged: cached seated sdf0
        if preparing:
            sdf0 = sdf_mesh(
                mesh_b, wp.transform_point(preparing_anchor_mesh, perimeter_point), max_dist
            )  # preparing: recomputed live
        if sdf0 >= max_dist:
            pad_rms[pad] = -1.0  # perimeter point outside SDF search radius: result is unreliable
            return
        sdf_now = sdf_mesh(mesh_b, wp.transform_point(t_seal_mesh, perimeter_point), max_dist)
        if sdf_now >= max_dist:
            pad_rms[pad] = -1.0  # perimeter point outside SDF search radius: result is unreliable
            return
        dev = sdf_now - sdf0
        dev_sq += dev * dev
        count += 1
    if count > 0:
        pad_rms[pad] = wp.sqrt(dev_sq / float(count))  # root-mean-square over this pad's perimeter points


# --------------- public API ---------------


# --------------------------------------------------------------------------------------------------
# Simple linear seal model (SurfaceGripper)
#
# Every DOF is an independent linear spring-damper, F = k*delta + d*deltadot, with its own stiffness
# and damping. Four caps limit the result, one per DOF group: the normal pull at control * f_grip_max
# (the vacuum), the two shear components together at f_shear_max (combined magnitude), the two peel
# moments together at f_peel_max, and the twist at +/-f_torsion_max (0 => uncapped). Stiffness/damping
# are set directly -- no shape/geometry factors, friction cones or stick-slip. Damping uses an
# implicit (backward-Euler) rescale (_effective_damping). Both the capped and uncapped loads are
# reported (pad_seal_load / pad_seal_load_unclamped); deciding when a seal fractures is left to the caller.
# --------------------------------------------------------------------------------------------------


class SurfaceGripper:
    """An individual linear surface gripper (authoring object); see the section comment above.

    Construct with the target ``body_id`` and gripper ``xform`` only, then set the seal parameters with
    exactly one of :meth:`set_stiffness_damping` (per-axis stiffness/damping directly) or
    :meth:`set_natural_frequency_damping_ratio` (per-axis natural frequency / damping ratio, converted
    against a reference solid). Add pads with :meth:`add_pad` and flatten with :class:`SurfaceGripperBuilder`.
    """

    def __init__(self, body_id: int, xform: wp.transform, world: int = 0, n_perimeter_samples: int = 0):
        self.body_id = body_id
        self.xform = xform
        if world < 0:
            raise ValueError(f"world must be >= 0, got {world}: a gripper follows the body it is attached to")
        # Replicated environment this gripper belongs to. Pads are laid out grouped by world, so the
        # joint seat fit only ever scans pads from the same environment.
        self.world = world
        self.n_perimeter_samples = (
            n_perimeter_samples  # perimeter sample points per pad for this gripper's seat fit / seal metric
        )
        self.pads: list[wp.transform] = []  # pad poses in the gripper frame
        self.pad_radii: list[float] = []  # per-pad perimeter circle radius [m], parallel to self.pads
        self.pad_half_heights: list[
            float
        ] = []  # per-pad perimeter plane offset along the grip axis [m], parallel to self.pads
        # Seal parameters -- zero (no seal force) until set via one of the two setters below.
        self.k_normal = 0.0
        self.d_normal = 0.0
        self.f_grip_max = 0.0
        self.k_shear_x = 0.0
        self.d_shear_x = 0.0
        self.k_shear_y = 0.0
        self.d_shear_y = 0.0
        self.f_shear_max = 0.0
        self.k_peel_x = 0.0
        self.d_peel_x = 0.0
        self.k_peel_y = 0.0
        self.d_peel_y = 0.0
        self.f_peel_max = 0.0
        self.k_torsion = 0.0
        self.d_torsion = 0.0
        self.f_torsion_max = 0.0

    def set_stiffness_damping(
        self,
        k_normal: float,
        d_normal: float,
        f_grip_max: float,
        k_shear_x: float,
        d_shear_x: float,
        k_shear_y: float,
        d_shear_y: float,
        f_shear_max: float,
        k_peel_x: float,
        d_peel_x: float,
        k_peel_y: float,
        d_peel_y: float,
        f_peel_max: float,
        k_torsion: float,
        d_torsion: float,
        f_torsion_max: float,
    ) -> "SurfaceGripper":
        """Set the seal from per-axis stiffness and damping directly; returns ``self``.

        ``f_grip_max`` is the vacuum grip [N] (the normal pull cap = ``control * f_grip_max``); the
        ``f_*_max`` are the per DOF-group force caps (0 => uncapped).
        """
        self.k_normal = k_normal
        self.d_normal = d_normal
        self.f_grip_max = f_grip_max
        self.k_shear_x = k_shear_x
        self.d_shear_x = d_shear_x
        self.k_shear_y = k_shear_y
        self.d_shear_y = d_shear_y
        self.f_shear_max = f_shear_max
        self.k_peel_x = k_peel_x
        self.d_peel_x = d_peel_x
        self.k_peel_y = k_peel_y
        self.d_peel_y = d_peel_y
        self.f_peel_max = f_peel_max
        self.k_torsion = k_torsion
        self.d_torsion = d_torsion
        self.f_torsion_max = f_torsion_max
        return self

    def set_natural_frequency_damping_ratio(
        self,
        mass: float,
        inertia: wp.mat33,
        f_grip_max: float,
        normal_mode: tuple[float, float],
        shear_x_mode: tuple[float, float],
        shear_y_mode: tuple[float, float],
        peel_x_mode: tuple[float, float],
        peel_y_mode: tuple[float, float],
        torsion_mode: tuple[float, float],
        f_shear_max: float = 0.0,
        f_peel_max: float = 0.0,
        f_torsion_max: float = 0.0,
    ) -> "SurfaceGripper":
        """Set the seal from per-axis modes ``(angular natural frequency [rad/s], damping ratio)``,
        converted to stiffness/damping (:func:`_nat_freq_damping_ratio_to_stiffness_damping`) against a
        design body of ``mass`` [kg] and ``inertia`` [kg.m^2, body frame]. Translation DOFs use the mass;
        peel/twist use the inertia about that axis (the diagonal terms). Returns ``self``.
        """
        m = mass
        ixx = inertia[0, 0]  # inertia about x (peel-x)
        iyy = inertia[1, 1]  # about y (peel-y)
        izz = inertia[2, 2]  # about z (twist)
        to = _nat_freq_damping_ratio_to_stiffness_damping
        k_normal, d_normal = to(*normal_mode, m)
        k_shear_x, d_shear_x = to(*shear_x_mode, m)
        k_shear_y, d_shear_y = to(*shear_y_mode, m)
        k_peel_x, d_peel_x = to(*peel_x_mode, ixx)
        k_peel_y, d_peel_y = to(*peel_y_mode, iyy)
        k_torsion, d_torsion = to(*torsion_mode, izz)
        return self.set_stiffness_damping(
            k_normal,
            d_normal,
            f_grip_max,
            k_shear_x,
            d_shear_x,
            k_shear_y,
            d_shear_y,
            f_shear_max,
            k_peel_x,
            d_peel_x,
            k_peel_y,
            d_peel_y,
            f_peel_max,
            k_torsion,
            d_torsion,
            f_torsion_max,
        )

    def add_pad(self, xform: wp.transform, radius: float, half_height: float) -> int:
        """Add a pad at ``xform`` (gripper frame) with perimeter circle ``radius`` [m] and ``half_height`` [m]
        (the perimeter-plane offset along the grip axis). Returns its index within this gripper."""
        self.pads.append(xform)
        self.pad_radii.append(radius)
        self.pad_half_heights.append(half_height)
        return len(self.pads) - 1


class SurfaceGripperBuilder:
    """Collects :class:`SurfaceGripper` objects and finalizes them into a
    :class:`SurfaceGripperModel` (mirrors :class:`newton.ModelBuilder`).
    """

    def __init__(self):
        self.grippers: list[SurfaceGripper] = []

    def add_gripper(self, gripper: SurfaceGripper) -> int:
        """Add an authored gripper. Returns its gripper index."""
        self.grippers.append(gripper)
        return len(self.grippers) - 1

    def finalize(self, device: str | wp.Device | None = None) -> "SurfaceGripperModel":
        """Flatten all grippers into a device-resident :class:`SurfaceGripperModel`.

        Each gripper's ``n_perimeter_samples`` (perimeter sample points per pad, for the seat fit / seal-quality metric)
        is stored per gripper (``n_perimeter_samples``). The (constant, pad-frame) perimeter positions are precomputed once
        into ``pad_perimeter_local`` so the kernels don't rebuild cos/sin each step; because the count can differ per
        gripper, each pad's points are addressed by the start-index array ``pad_perimeter_start`` (``pad_perimeter_sdf0`` shares it).
        """
        g = self.grippers
        m = SurfaceGripperModel()
        # per-gripper arrays (indexed by gripper id)
        m.gripper_body_id = wp.array([x.body_id for x in g], dtype=wp.int32, device=device)
        m.gripper_xform = wp.array([x.xform for x in g], dtype=wp.transform, device=device)
        m.gripper_k_normal = wp.array([x.k_normal for x in g], dtype=wp.float32, device=device)
        m.gripper_d_normal = wp.array([x.d_normal for x in g], dtype=wp.float32, device=device)
        m.gripper_f_grip_max = wp.array([x.f_grip_max for x in g], dtype=wp.float32, device=device)
        m.gripper_k_shear_x = wp.array([x.k_shear_x for x in g], dtype=wp.float32, device=device)
        m.gripper_d_shear_x = wp.array([x.d_shear_x for x in g], dtype=wp.float32, device=device)
        m.gripper_k_shear_y = wp.array([x.k_shear_y for x in g], dtype=wp.float32, device=device)
        m.gripper_d_shear_y = wp.array([x.d_shear_y for x in g], dtype=wp.float32, device=device)
        m.gripper_f_shear_max = wp.array([x.f_shear_max for x in g], dtype=wp.float32, device=device)
        m.gripper_k_peel_x = wp.array([x.k_peel_x for x in g], dtype=wp.float32, device=device)
        m.gripper_d_peel_x = wp.array([x.d_peel_x for x in g], dtype=wp.float32, device=device)
        m.gripper_k_peel_y = wp.array([x.k_peel_y for x in g], dtype=wp.float32, device=device)
        m.gripper_d_peel_y = wp.array([x.d_peel_y for x in g], dtype=wp.float32, device=device)
        m.gripper_f_peel_max = wp.array([x.f_peel_max for x in g], dtype=wp.float32, device=device)
        m.gripper_k_torsion = wp.array([x.k_torsion for x in g], dtype=wp.float32, device=device)
        m.gripper_d_torsion = wp.array([x.d_torsion for x in g], dtype=wp.float32, device=device)
        m.gripper_f_torsion_max = wp.array([x.f_torsion_max for x in g], dtype=wp.float32, device=device)
        m.gripper_world = wp.array([x.world for x in g], dtype=wp.int32, device=device)
        # world_count = highest world index + 1 (SurfaceGripper rejects a negative world)
        m.world_count = 0
        for gi in range(len(g)):
            w = g[gi].world
            if w + 1 > m.world_count:
                m.world_count = w + 1

        # Flatten every gripper's pads into per-pad arrays, ordered by world: world 0's pads first, then
        # world 1's, and so on. So each world's pads are one contiguous range,
        # whose first index is recorded in pad_world_start. For each pad we also lay its perimeter sample points
        # (constant, pad frame) into pad_perimeter_local; pad_perimeter_start[p] marks where pad p's points start (a
        # gripper's n_perimeter_samples can differ, hence a start-index array, not a fixed stride). pad_perimeter_sdf0 shares pad_perimeter_start.
        pad_gripper = []
        pad_xform = []
        pad_world = []
        pad_perimeter_local = []
        pad_perimeter_start = [
            0
        ]  # [n_pads + 1]: pad p's perimeter points are pad_perimeter_local[pad_perimeter_start[p] : pad_perimeter_start[p+1]]
        pad_world_start = [0] * (m.world_count + 1)  # per-world starts, then the total

        for w in range(m.world_count):
            pad_world_start[w] = len(pad_gripper)  # first pad of world w
            for gi in range(len(g)):
                gripper = g[gi]
                if gripper.world != w:
                    continue
                for pi in range(len(gripper.pads)):
                    pad_gripper.append(gi)
                    pad_xform.append(gripper.pads[pi])
                    pad_world.append(w)
                    perimeter = _perimeter_circle(
                        gripper.pad_radii[pi], gripper.pad_half_heights[pi], gripper.n_perimeter_samples
                    )
                    for li in range(len(perimeter)):
                        pad_perimeter_local.append(perimeter[li])
                    pad_perimeter_start.append(len(pad_perimeter_local))  # end of this pad's perimeter points
        pad_world_start[m.world_count] = len(pad_gripper)  # total pad count

        gripper_n_perimeter_samples = []
        for gi in range(len(g)):
            gripper_n_perimeter_samples.append(g[gi].n_perimeter_samples)

        m.n_perimeter_samples = wp.array(gripper_n_perimeter_samples, dtype=wp.int32, device=device)  # [grippers]
        m.pad_gripper = wp.array(pad_gripper, dtype=wp.int32, device=device)
        m.pad_xform = wp.array(pad_xform, dtype=wp.transform, device=device)
        m.pad_perimeter_local = wp.array(
            pad_perimeter_local, dtype=wp.vec3, device=device
        )  # indexed by pad_perimeter_start
        m.pad_perimeter_start = wp.array(
            pad_perimeter_start, dtype=wp.int32, device=device
        )  # [n_pads+1] start indices into pad_perimeter_local
        m.pad_world = wp.array(pad_world, dtype=wp.int32, device=device)
        m.pad_world_start = wp.array(pad_world_start, dtype=wp.int32, device=device)
        return m


class SurfaceGripperStateInput:
    """Per-pad inputs to the gripper simulation, set by the caller. Per-pad arrays.

    The caller writes these fields each step; ``surface_gripper.py`` kernels only read them.
    Engagement and preparing state are encoded in ``[0]`` of each pair: a non-negative value means the
    pad is engaged or preparing to engage (and identifies the gripped body), while ``[0] < 0`` means
    released or idle. ``[1]`` carries the global shape ID of the gripped collision shape, giving direct
    access to ``model.shape_transform`` (T_bs) and the SDF mesh via ``shape_mesh_id``.
    """

    pad_engaged_bs: wp.array[wp.vec2i]  # ``[0]`` = body id, ``[1]`` = shape id; ``[0] < 0`` = not engaged
    pad_preparing_bs: wp.array[wp.vec2i]  # ``[0]`` = body id, ``[1]`` = shape id; ``[0] < 0`` = not preparing


class SurfaceGripperStateOutput:
    """Per-pad outputs written by the gripper simulation kernels. Per-pad arrays.

    These fields are computed each step by the gripper kernels; the caller reads them for telemetry,
    break detection, and GUI, but should not write them.
    """

    # Component indices into the seal-load vec4 fields below. Usable from Warp kernels as
    # compile-time constants, e.g. ``load[SurfaceGripperStateOutput.SEAL_LOAD_PEEL]``.
    SEAL_LOAD_NORMAL = wp.constant(0)
    SEAL_LOAD_SHEAR = wp.constant(1)
    SEAL_LOAD_PEEL = wp.constant(2)
    SEAL_LOAD_TORSION = wp.constant(3)
    SEAL_LOAD_COUNT = wp.constant(4)

    # Per-pad seal loads grouped by DOF, as (normal, shear, peel, torsion):
    #   normal  = fz, the pull along the seal axis [N] (signed; > 0 pulls the body onto the pad)
    #   shear   = |(fx, fy)|, the in-plane force magnitude [N]
    #   peel    = |(mx, my)|, the out-of-plane moment magnitude [N.m]
    #   torsion = mz, the twist about the seal axis [N.m] (signed)
    # Shear and peel are magnitudes because their caps are direction-independent disks.
    pad_seal_load: wp.array[wp.vec4]  # after the per-group caps: what was applied to the bodies
    pad_seal_load_unclamped: wp.array[
        wp.vec4
    ]  # before the caps; feeds the break metric (clamped values can never exceed their cap)
    pad_anchor_b: wp.array[wp.transform]  # SB -- see Frame nomenclature
    pad_perimeter_sdf0: wp.array[
        wp.float32
    ]  # seated perimeter signed distances cached at engagement (indexed by model.pad_perimeter_start)
    pad_seal_quality_rms: wp.array[
        wp.float32
    ]  # RMS perimeter-gap deviation from seated pose per pad [m]; -1 if not engaged or preparing


class SurfaceGripperControl:
    """Gripper control inputs. Per-pad arrays."""

    pad_grip_control: wp.array[float]  # per-pad grip command [0, 1]; f_min = pad_grip_control * f_grip_max


class SurfaceGripperModel:
    """Finalized simple-gripper model (mirrors :class:`newton.Model`).

    Constant device arrays; ``gripper_*`` indexed by gripper id, ``pad_*`` by pad id. Use
    :meth:`state_input` / :meth:`state_output` to allocate the matching per-step state objects.
    """

    def state_input(self) -> SurfaceGripperStateInput:
        """Allocate a fresh per-pad :class:`SurfaceGripperStateInput` for this model."""
        si = SurfaceGripperStateInput()
        n = self.pad_xform.shape[0]
        si.pad_engaged_bs = wp.full(n, wp.vec2i(-1, -1), dtype=wp.vec2i, device=self.pad_xform.device)
        si.pad_preparing_bs = wp.full(n, wp.vec2i(-1, -1), dtype=wp.vec2i, device=self.pad_xform.device)
        return si

    def state_output(self) -> SurfaceGripperStateOutput:
        """Allocate a fresh per-pad :class:`SurfaceGripperStateOutput` for this model. ``pad_perimeter_sdf0`` shares
        the start-index scheme of ``pad_perimeter_local`` (one entry per perimeter sample point across all pads)."""
        so = SurfaceGripperStateOutput()
        n = self.pad_xform.shape[0]
        so.pad_seal_load = wp.zeros(n, dtype=wp.vec4, device=self.pad_xform.device)
        so.pad_seal_load_unclamped = wp.zeros(n, dtype=wp.vec4, device=self.pad_xform.device)
        so.pad_anchor_b = wp.zeros(n, dtype=wp.transform, device=self.pad_xform.device)
        so.pad_perimeter_sdf0 = wp.zeros(
            self.pad_perimeter_local.shape[0], dtype=wp.float32, device=self.pad_xform.device
        )
        so.pad_seal_quality_rms = wp.zeros(n, dtype=wp.float32, device=self.pad_xform.device)
        return so

    def control(self) -> SurfaceGripperControl:
        """Allocate a fresh per-pad :class:`SurfaceGripperControl` for this model."""
        c = SurfaceGripperControl()
        n = self.pad_xform.shape[0]
        c.pad_grip_control = wp.zeros(n, dtype=wp.float32, device=self.pad_xform.device)
        return c


def attach_seal(
    state: newton.State,
    gripper_model: "SurfaceGripperModel",
    gripper_state_input_prev: SurfaceGripperStateInput,
    gripper_state_output: SurfaceGripperStateOutput,
    gripper_state_input_curr: SurfaceGripperStateInput,
) -> None:
    """Latch ``pad_anchor_b`` for pads that just engaged, then commit the seal state.

    On a disengaged->engaged rising edge (``gripper_state_input_prev.pad_engaged_bs[..][0] < 0`` and
    ``gripper_state_input_curr.pad_engaged_bs[..][0] >= 0``), cache SB into ``pad_anchor_b`` against
    the body's raw pose (``state.body_q``).
    Cf. :func:`attach_seal_seated`, which seats against the inline-fitted pose.

    Args:
        state: Simulation state; source of ``body_q`` (world body poses).
        gripper_model: Finalized gripper holding the pad/gripper layout arrays.
        gripper_state_input_prev: Previous sub-step's input state; ``pad_engaged_bs[..][0]``
            (< 0 = released last step) detects the rising edge.
        gripper_state_output: Per-pad output state; ``pad_anchor_b`` is written on each rising edge.
        gripper_state_input_curr: Current sub-step's input state; ``pad_engaged_bs[..][0]``
            (>= 0 = engaged this step) detects the rising edge and identifies the target body.
    """
    n_pads = gripper_model.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        _attach_seal_kernel,
        dim=n_pads,
        inputs=[
            gripper_state_input_curr.pad_engaged_bs,
            gripper_state_input_prev.pad_engaged_bs,
            gripper_model.gripper_body_id,
            gripper_model.gripper_xform,
            gripper_model.pad_gripper,
            gripper_model.pad_xform,
            state.body_q,
            state.body_q,  # hold pose = the body's raw pose
            gripper_state_output.pad_anchor_b,
        ],
        device=gripper_model.pad_xform.device,
    )


def attach_seal_seated(
    model: newton.Model,
    state: newton.State,
    gripper_model: SurfaceGripperModel,
    gripper_state_input_prev: SurfaceGripperStateInput,
    gripper_state_output: SurfaceGripperStateOutput,
    gripper_state_input_curr: SurfaceGripperStateInput,
    shape_mesh_id: wp.array[wp.uint64],
    max_dist: float = 1.0,
    grad_h: float = 1.0e-4,
    damping: float = 1.0e-3,
    iters: int = 8,
) -> None:
    """On each pad's rising edge (``gripper_state_input_prev.pad_engaged_bs[..][0] < 0`` and
    ``gripper_state_input_curr.pad_engaged_bs[..][0] >= 0``), compute the seated body pose and
    cache the seal anchor SB (the seal frame expressed in body B's local frame) into
    ``gripper_state_output.pad_anchor_b``.

    The seated pose is the world pose of body B (GB0, see Frame nomenclature) that minimises the signed distances of the contact perimeter
    sample points of **all pads gripping or preparing to grip body B** to its surface -- i.e., the
    pose where all those perimeters sit flush simultaneously. After the fit, ``SB = GB0^-1 * SA0`` is cached (see Frame
    nomenclature) and ``pad_perimeter_sdf0`` is written with the seated SDF baseline.

    Seated variant of :func:`attach_seal`, which anchors to the raw (unfit) body pose instead.

    Args:
        model: Finalized Newton model; source of ``shape_transform`` (T_bs per shape).
        state: Simulation state; source of ``body_q`` for the seat fit and seal frames.
        gripper_model: Finalized gripper holding the pad/gripper layout arrays.
        gripper_state_input_prev: Previous sub-step's input state; ``pad_engaged_bs[..][0]``
            (< 0 = released last step) detects the rising edge.
        gripper_state_output: Per-pad output state; ``pad_anchor_b`` and ``pad_perimeter_sdf0``
            are written on each rising edge.
        gripper_state_input_curr: Current sub-step's input state; ``pad_engaged_bs[..][0]``
            (>= 0 = engaged this step) detects the rising edge and identifies the body for the
            seat fit, while ``[..][1]`` identifies the gripped collision shape. Every pad engaging
            the same body is fitted together, so they seat flush simultaneously.
        shape_mesh_id: shape id -> SDF mesh id (:class:`warp.Mesh`), shape [n_shapes].
        max_dist: SDF search radius [m].
        grad_h: SDF central-difference step [m] for gradient estimation.
        damping: Stabiliser for the Gauss-Newton fit; prevents drift in unconstrained directions.
        iters: Gauss-Newton iterations (1 suffices for planar faces).
    """
    gm = gripper_model
    n_pads = gm.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        _attach_seal_seated_kernel,
        dim=n_pads,
        inputs=[
            gripper_state_input_curr.pad_engaged_bs,
            gripper_state_input_prev.pad_engaged_bs,
            gm.gripper_body_id,
            gm.gripper_xform,
            gm.pad_gripper,
            gm.pad_xform,
            state.body_q,
            shape_mesh_id,
            model.shape_transform,
            gm.pad_world,
            gm.pad_world_start,
            gm.pad_perimeter_local,
            gm.pad_perimeter_start,
            max_dist,
            grad_h,
            damping,
            iters,
            gripper_state_output.pad_anchor_b,
            gripper_state_output.pad_perimeter_sdf0,
        ],
        device=gm.pad_xform.device,
    )


def evaluate_gripper_force(
    model: newton.Model,
    state: newton.State,
    gripper_model: SurfaceGripperModel,
    gripper_state_input: SurfaceGripperStateInput,
    gripper_state_output: SurfaceGripperStateOutput,
    gripper_control: SurfaceGripperControl,
    dt: float,
) -> None:
    """Accumulate the seal wrench for each engaged pad into ``state.body_f``.

    For each engaged pad the bias ``SA^-1 * GB * SB`` (see Frame nomenclature) is decomposed into
    six scalar DOF displacements (normal, shear x/y, peel x/y, twist) and a matching velocity.
    A linear spring-damper acts on each DOF with a fixed magnitude cap; the resulting wrench is
    expressed in the world frame (with the body COM as the reference point, matching Newton's
    ``state.body_f`` convention) and applied equal-and-opposite to body A (end-effector) and body B
    (gripped body).

    Args:
        model: Finalized Newton model; source of ``body_com``, ``body_mass``, ``body_inertia``.
        state: Simulation state; ``body_q`` and ``body_qd`` are read, ``body_f`` is accumulated into.
        gripper_model: Finalized gripper holding pad/gripper layout and seal stiffness/damping arrays.
        gripper_state_input: Per-pad input state; ``pad_engaged_bs[..][0]`` selects engaged pads and
            ``pad_anchor_b`` (SB) provides the cached seal reference frame.
        gripper_state_output: Per-pad output state; ``pad_seal_load`` and
            ``pad_seal_load_unclamped`` are written.
        gripper_control: Per-pad control; ``pad_grip_control`` scales the normal pull cap.
        dt: Physics sub-step duration [s]; used for implicit damping rescaling.
    """
    n_pads = gripper_model.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        _eval_pad_force_linear_kernel,
        dim=n_pads,
        inputs=[
            gripper_model.gripper_body_id,
            gripper_model.gripper_xform,
            gripper_model.gripper_k_normal,
            gripper_model.gripper_d_normal,
            gripper_model.gripper_f_grip_max,
            gripper_model.gripper_k_shear_x,
            gripper_model.gripper_d_shear_x,
            gripper_model.gripper_k_shear_y,
            gripper_model.gripper_d_shear_y,
            gripper_model.gripper_f_shear_max,
            gripper_model.gripper_k_peel_x,
            gripper_model.gripper_d_peel_x,
            gripper_model.gripper_k_peel_y,
            gripper_model.gripper_d_peel_y,
            gripper_model.gripper_f_peel_max,
            gripper_model.gripper_k_torsion,
            gripper_model.gripper_d_torsion,
            gripper_model.gripper_f_torsion_max,
            gripper_model.pad_gripper,
            gripper_model.pad_xform,
            gripper_control.pad_grip_control,
            gripper_state_input.pad_engaged_bs,
            gripper_state_output.pad_anchor_b,
            model.body_com,
            state.body_q,
            state.body_qd,
            model.body_mass,
            model.body_inertia,
            dt,
            # outputs (mutated in place)
            gripper_state_output.pad_seal_load,
            gripper_state_output.pad_seal_load_unclamped,
            state.body_f,
        ],
        device=gripper_model.pad_xform.device,
    )


def evaluate_seal_quality(
    model: newton.Model,
    state: newton.State,
    gripper_model: SurfaceGripperModel,
    gripper_state_input: SurfaceGripperStateInput,
    gripper_state_output: SurfaceGripperStateOutput,
    shape_mesh_id: wp.array[wp.uint64],
    max_dist: float = 1.0,
    grad_h: float = 1.0e-4,
    damping: float = 1.0e-3,
    iters: int = 8,
) -> None:
    """
    Compute a geometric seal quality per pad (pad_rms[pad]).
    Three mutually exclusive modes of operation: preparing (to grip), engaged (currently gripping)
    and disengaged. A pad is in preparing mode when ``pad_preparing_bs[pad][0] >= 0``.
    A pad is in engaged mode when ``pad_engaged_bs[pad][0] >= 0``.
    Disengaged mode occurs when neither preparing nor engaged mode applies; pad_rms is set to -1.
    In preparing and engaged modes, we compute the rms error per pad as follows:
    pad_rms = sqrt{ [sum_i (sdf_now(i) - sdf_baseline(i))^2]/n_sample_points_per_pad}
    i spans the sample points of the pad.
    In engaged mode, sdf_baseline(i) is the signed distance of the ith sample point
    that was computed and cached at initial engagement.
    In preparation mode, sdf_baseline(i) is the signed distance of the ith sample point
    using the seated pose computed from the current pose. This provides an estimate of the
    error that would immediately occur in the event that the pad state would be set to engaged.
    In preparing and engaged modes, sdf_now(i) is the signed distance of the ith sample point at the
    current pose.


    Args:
        model: Finalized Newton model; source of ``shape_transform`` (T_bs per shape).
        state: Simulation state; source of ``body_q`` (world body poses).
        gripper_model: Finalized gripper holding the pad/gripper layout arrays.
        gripper_state_input: Per-pad input state; ``pad_engaged_bs[..][0]`` and
            ``pad_preparing_bs[..][0]`` select which mode each pad is in; ``[1]`` of each
            identifies the gripped collision shape.
        gripper_state_output: Per-pad output state; ``pad_perimeter_sdf0`` provides the cached
            seated baseline; ``pad_seal_quality_rms`` receives the result.
        shape_mesh_id: shape id -> gripped-object SDF mesh id, shape [n_shapes].
        max_dist: SDF search radius [m].
        grad_h: SDF central-difference step [m] for the seat fit in preparing mode.
        damping: Stabiliser for the seat fit (prevents drift in unconstrained directions).
        iters: Gauss-Newton iterations for the seat fit in preparing mode.
    """

    gm = gripper_model
    n_pads = gm.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        _seal_quality_kernel,
        dim=n_pads,
        inputs=[
            gripper_state_input.pad_engaged_bs,
            gripper_state_input.pad_preparing_bs,
            gm.gripper_body_id,
            gm.gripper_xform,
            gm.pad_gripper,
            gm.pad_xform,
            state.body_q,
            shape_mesh_id,
            model.shape_transform,
            gm.pad_world,
            gm.pad_world_start,
            gm.pad_perimeter_local,
            gripper_state_output.pad_perimeter_sdf0,
            gm.pad_perimeter_start,
            max_dist,
            grad_h,
            damping,
            iters,
            gripper_state_output.pad_seal_quality_rms,
        ],
        device=gm.pad_xform.device,
    )
