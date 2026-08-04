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
and the per-pad force kernel. Imported by ``example_surface_gripper_repro``; not a runnable example.

Mirrors Newton's Builder -> Model -> State/Control layout::

    SurfaceGripper (authoring) -> SurfaceGripperBuilder.finalize() -> SurfaceGripperModel
    SurfaceGripperModel.state() / .control() -> SurfaceGripperState / SurfaceGripperControl
"""

import math

import numpy as np
import warp as wp

import newton
from newton.geometry import sdf_mesh


class SurfaceGripperState:
    """Mutable per-step gripper state (mirrors :class:`newton.State`). Per-pad arrays."""

    pad_break_metric: wp.array  # brittle break envelope; > 1 => seal exceeded capacity (float)
    pad_dof_force: wp.array  # per-DOF force telemetry (normal, shear mag, peel mag, twist), per pad (wp.vec4)
    pad_engaged: wp.array  # per-pad engaged/disengaged flag (bool)
    pad_body_b: wp.array  # gripped body index, valid when engaged (int)
    pad_anchor_b: wp.array  # TBS: seal frame in the gripped body's frame, cached at engagement (wp.transform)


class SurfaceGripperControl:
    """Gripper control inputs (mirrors :class:`newton.Control`). Per-pad arrays."""

    pad_grip_control: wp.array  # per-pad grip command [0, 1]; f_min = pad_grip_control * f_grip_max


@wp.func
def evaluate_current_seal(pad: int, pad_break_metric: wp.array[float]):
    """Decide whether an already-engaged pad keeps its seal; returns the new engaged flag.

    Physics-based break: the seal survives while the brittle break metric (from the most recent
    force evaluation) is within capacity, and breaks once it exceeds 1.
    """
    return pad_break_metric[pad] <= 1.0


@wp.func
def evaluate_potential_seal():
    """Decide whether a disengaged pad forms a new seal; returns the new engaged flag.

    Placeholder: a real implementation would run the geometric seal test (does the pad lip seal
    against a nearby surface). For now no new seal forms once broken.
    """
    return False


@wp.kernel
def evaluate_seal_kernel(
    pad_break_metric: wp.array[float],
    # in/out
    seal_engaged: wp.array[wp.bool],
):
    """Per-pad seal decision used when no seal series is scripted: keep or break an engaged seal
    (:func:`evaluate_current_seal`), otherwise try to form a new one (:func:`evaluate_potential_seal`).
    """
    pad = wp.tid()
    if seal_engaged[pad]:
        seal_engaged[pad] = evaluate_current_seal(pad, pad_break_metric)
    else:
        seal_engaged[pad] = evaluate_potential_seal()


def evaluate_seal(gripper_model: "SurfaceGripperModel", gripper_state: "SurfaceGripperState", seal_engaged) -> None:
    """Update the per-pad seal decision from the physics (used when no seal series is scripted).

    For each pad: if currently engaged, evaluate whether the seal survives; otherwise evaluate
    whether a new seal forms. Writes the decision into ``seal_engaged``, which then feeds
    :func:`attach_seal`.
    """
    n_pads = gripper_model.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        evaluate_seal_kernel,
        dim=n_pads,
        inputs=[gripper_state.pad_break_metric],
        outputs=[seal_engaged],
    )


@wp.kernel
def attach_seal_kernel(
    pad_seal_engaged: wp.array[wp.bool],  # [pads] fresh per-pad seal decision (from the seal logic)
    pad_body_b_id: wp.array[int],  # body each pad seals against this step
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],  # world pose of body A (the gripper body)
    hold_pose_body_b: wp.array[wp.transform],  # per-body pose B is held at: raw body_q, or the fitted (seated) pose
    # outputs
    pad_engaged: wp.array[wp.bool],  # stored state, updated in place
    pad_body_b: wp.array[int],
    pad_anchor_b: wp.array[wp.transform],
):
    pad = wp.tid()
    if pad_seal_engaged[pad] and not pad_engaged[pad]:
        # disengaged -> engaged: latch the gripped body id and cache TBS = TB^-1 * TA(t0) * TAS (seal frame
        # in body B). TB is the hold pose from ``hold_pose_body_b`` -- the raw body pose (plain latch) or the
        # fitted pose (seated latch), so the seal's rest state is whichever the caller chose.
        pad_body_b[pad] = pad_body_b_id[
            pad
        ]  # stays fixed while gripped (matches the body pad_anchor_b is cached against)
        gripper_id = pad_gripper[pad]
        seal_world = body_q[gripper_body_id[gripper_id]] * gripper_xform[gripper_id] * pad_xform[pad]  # TA * TAS
        pad_anchor_b[pad] = wp.transform_inverse(hold_pose_body_b[pad_body_b_id[pad]]) * seal_world

    pad_engaged[pad] = pad_seal_engaged[pad]


def attach_seal(
    state,
    gripper_model: "SurfaceGripperModel",
    gripper_state: SurfaceGripperState,
    pad_seal_engaged,
    pad_body_b_id,
) -> None:
    """Latch ``pad_anchor_b`` for pads that just engaged, then commit the seal state.

    ``pad_seal_engaged`` / ``pad_body_b_id`` are this step's fresh seal decision. On a disengaged ->
    engaged transition, TBS (the seal frame in body B's frame) is cached into
    ``pad_anchor_b``; ``pad_engaged`` / ``pad_body_b`` are then updated to the decision. The seal seats
    against the body's raw pose (``state.body_q``); cf. :func:`attach_seal_seated`, which seats against the
    inline-fitted pose.
    """
    n_pads = gripper_model.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        attach_seal_kernel,
        dim=n_pads,
        inputs=[
            pad_seal_engaged,
            pad_body_b_id,
            gripper_model.gripper_body_id,
            gripper_model.gripper_xform,
            gripper_model.pad_gripper,
            gripper_model.pad_xform,
            state.body_q,
            state.body_q,  # hold pose = the body's raw pose
            gripper_state.pad_engaged,
            gripper_state.pad_body_b,
            gripper_state.pad_anchor_b,
        ],
    )


@wp.func
def eval_pad_separation(
    t_seal_a: wp.transform, t_seal_b: wp.transform
) -> tuple[float, float, float, float, float, float]:
    """Per-DOF separation of the two seal frames accumulated since engagement.

    Args:
        t_seal_a: World pose of the seal frame carried by body A, TA(t)*TAS.
        t_seal_b: World pose of the seal frame carried by body B, TB(t)*TBS, where
            TBS was cached when the pad engaged (so the two frames then coincided).
    Returns:
        ``(px, py, pz, theta_x, theta_y, theta_z)`` [m, m, m, rad, rad, rad]: shear
        along x/y, normal along z, peel about x/y, and twist about z.
    """

    # TA(t)*TAS* TRel(t) = TB(t)*TBS
    # t_seal_b*TRel(t) = t_seal_b
    # TRel(t) = t_seal_b^-1 * t_seal_b
    t_rel = wp.transform_inverse(t_seal_a) * t_seal_b
    p = wp.transform_get_translation(t_rel)
    q = wp.transform_get_rotation(t_rel)
    # small-angle rotation vector: theta ~ 2*(qx, qy, qz) for a near-identity quaternion
    return p[0], p[1], p[2], 2.0 * q[0], 2.0 * q[1], 2.0 * q[2]


@wp.func
def eval_pad_relative_velocity(
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
def apparent_mass(axis_w: wp.vec3, r: wp.vec3, q_b: wp.quat, inv_m: float, inv_inertia: wp.mat33) -> float:
    """Effective mass a rigid body presents to a force at offset ``r`` (world, COM->point) along the
    world unit direction ``axis_w``: ``1/m_app = 1/m + (r x n).I^-1.(r x n)`` -- translational compliance
    plus the rotational compliance from the off-COM spin. The rotational term is evaluated in the body
    frame (``q_b``, ``inv_inertia``), where the inertia tensor is stored.
    """
    c_b = wp.quat_rotate_inv(q_b, wp.cross(r, axis_w))  # (r x n) in the body frame
    return 1.0 / (inv_m + wp.dot(c_b, inv_inertia * c_b))


@wp.func
def apparent_inertia(axis_w: wp.vec3, q_b: wp.quat, inv_inertia: wp.mat33) -> float:
    """Effective inertia a rigid body presents to a pure moment about the world unit axis ``axis_w``:
    ``1/(n.I^-1.n)`` -- the free-body angular admittance about the axis, inverted. The axis is first
    rotated into the body frame (``q_b``, where ``inv_inertia`` lives). No translation term: a couple
    does not move the COM.
    """
    axis_b = wp.quat_rotate_inv(q_b, axis_w)  # seal axis in the body frame
    return 1.0 / wp.dot(axis_b, inv_inertia * axis_b)


@wp.func
def effective_damping(d: float, m_eff: float, dt: float) -> float:
    """Backward-Euler (implicit) damping coefficient: the explicit ``d`` rescaled so the applied force
    lands the pad-point velocity at the implicit value ``v*m_eff/(m_eff + d*dt)`` in one step. Bounded by
    ``m_eff/dt`` for any ``d`` (the damper can't overshoot the velocity). ``m_eff`` is the DOF's effective
    mass (translation) or effective inertia (rotation).
    """
    return d / (1.0 + d * dt / m_eff)


@wp.func
def eval_effective_damping(
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
    damping is rescaled via :func:`effective_damping` using that DOF's effective mass at the seal point
    (:func:`apparent_mass`, translation) or effective inertia about the seal axis, 1/(n.I^-1.n) (peel
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
        m_app = apparent_mass(axes_w[i], r_body_b, q_body_b, inv_m, inv_inertia)
        d_trans_eff[i] = effective_damping(d_trans[i], m_app, dt)

    # Combine Step 1 and 2 for rotational dofs (peel about x/y, twist about z)
    # gamma_effective = gamma/[1 + gamma*dt/I_eff]
    d_rot_eff = wp.vec3(0.0, 0.0, 0.0)
    d_rot = wp.vec3(d_peel_x, d_peel_y, d_torsion)
    for i in range(3):
        I_app = apparent_inertia(axes_w[i], q_body_b, inv_inertia)
        d_rot_eff[i] = effective_damping(d_rot[i], I_app, dt)

    return d_trans_eff[2], d_trans_eff[0], d_trans_eff[1], d_rot_eff[0], d_rot_eff[1], d_rot_eff[2]


# --------------------------------------------------------------------------------------------------
# Simple linear seal model (SurfaceGripper)
#
# Every DOF is an independent linear spring-damper, F = k*delta + d*deltadot, with the normal DOF
# adding a controllable preload F += control * f_grip_max. Each axis has its own stiffness and damping.
# Four caps limit the result: normal to +/-f_normal_max, the two shear components together to
# f_shear_max (combined magnitude), the two peel moments together to f_peel_max, and the twist to
# +/-f_torsion_max (0 => uncapped). Stiffness/damping are set directly (no shape/geometry factors,
# friction cones or stick-slip). Damping uses an implicit (backward-Euler) rescale
# (:func:`effective_damping`). The brittle break metric is not evaluated (left at 0, so the seal never
# fractures) -- to add later. Mirrors the Builder -> Model -> State/Control layout; the state/control are
# the shared :class:`SurfaceGripperState` / :class:`SurfaceGripperControl`, so the engagement helper
# (:func:`attach_seal`) works unchanged.
# --------------------------------------------------------------------------------------------------


def nat_freq_damping_ratio_to_stiffness_damping(mu: float, zeta: float, m_eff: float) -> tuple[float, float]:
    """``(k, d)`` for a 1-DOF spring-damper of effective mass/inertia ``m_eff`` tuned to angular natural
    frequency ``mu`` [rad/s] and damping ratio ``zeta``: ``k = m_eff*mu^2``, ``d = 2*zeta*mu*m_eff``.
    ``m_eff`` is a mass [kg] for a translation DOF, an inertia [kg.m^2] for a rotation DOF.
    """
    return m_eff * mu * mu, 2.0 * zeta * mu * m_eff


class SurfaceGripper:
    """An individual linear surface gripper (authoring object); see the section header.

    Construct with the target ``body_id`` and gripper ``xform`` only, then set the seal parameters with
    exactly one of :meth:`set_stiffness_damping` (per-axis stiffness/damping directly) or
    :meth:`set_natural_frequency_damping_ratio` (per-axis natural frequency / damping ratio, converted
    against a reference solid). Add pads with :meth:`add_pad` and flatten with :class:`SurfaceGripperBuilder`.
    """

    def __init__(self, body_id: int, xform: wp.transform, world: int = 0):
        self.body_id = body_id
        self.xform = xform
        self.world = world  # world (environment) this gripper lives in; -1 for a global gripper
        self.pads: list[wp.transform] = []  # pad poses in the gripper frame
        # Seal parameters -- zero (no seal force) until set via one of the two setters below.
        self.f_grip_max = 0.0
        self.k_normal = 0.0
        self.d_normal = 0.0
        self.f_normal_max = 0.0
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
        f_grip_max: float,
        k_normal: float,
        d_normal: float,
        f_normal_max: float,
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
        self.f_grip_max = f_grip_max
        self.k_normal = k_normal
        self.d_normal = d_normal
        self.f_normal_max = f_normal_max
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
        reference_solid: tuple,
        f_grip_max: float,
        normal_mode: tuple,
        shear_x_mode: tuple,
        shear_y_mode: tuple,
        peel_x_mode: tuple,
        peel_y_mode: tuple,
        torsion_mode: tuple,
        f_normal_max: float = 0.0,
        f_shear_max: float = 0.0,
        f_peel_max: float = 0.0,
        f_torsion_max: float = 0.0,
    ) -> "SurfaceGripper":
        """Set the seal from per-axis modes ``(angular natural frequency [rad/s], damping ratio)``,
        converted to stiffness/damping (:func:`nat_freq_damping_ratio_to_stiffness_damping`) against a
        design ``reference_solid`` = ``((hx, hy, hz) half-extents [m], mass [kg])``. Translation DOFs use
        its mass; peel/twist use its solid-cuboid inertia about that axis. Returns ``self``.
        """
        (hx, hy, hz), m = reference_solid
        ixx = m / 3.0 * (hy * hy + hz * hz)  # solid-cuboid inertia about x (peel-x)
        iyy = m / 3.0 * (hx * hx + hz * hz)  # about y (peel-y)
        izz = m / 3.0 * (hx * hx + hy * hy)  # about z (twist)
        to = nat_freq_damping_ratio_to_stiffness_damping
        k_normal, d_normal = to(*normal_mode, m)
        k_shear_x, d_shear_x = to(*shear_x_mode, m)
        k_shear_y, d_shear_y = to(*shear_y_mode, m)
        k_peel_x, d_peel_x = to(*peel_x_mode, ixx)
        k_peel_y, d_peel_y = to(*peel_y_mode, iyy)
        k_torsion, d_torsion = to(*torsion_mode, izz)
        return self.set_stiffness_damping(
            f_grip_max,
            k_normal,
            d_normal,
            f_normal_max,
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

    def add_pad(self, xform: wp.transform) -> int:
        """Add a pad at ``xform`` (gripper frame). Returns its index within this gripper."""
        self.pads.append(xform)
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

    def finalize(self, device=None) -> "SurfaceGripperModel":
        """Flatten all grippers into a device-resident :class:`SurfaceGripperModel`."""
        g = self.grippers
        m = SurfaceGripperModel()
        # per-gripper arrays (indexed by gripper id)
        m.gripper_body_id = wp.array([x.body_id for x in g], dtype=wp.int32, device=device)
        m.gripper_xform = wp.array([x.xform for x in g], dtype=wp.transform, device=device)
        m.gripper_f_grip_max = wp.array([x.f_grip_max for x in g], dtype=wp.float32, device=device)
        m.gripper_k_normal = wp.array([x.k_normal for x in g], dtype=wp.float32, device=device)
        m.gripper_d_normal = wp.array([x.d_normal for x in g], dtype=wp.float32, device=device)
        m.gripper_f_normal_max = wp.array([x.f_normal_max for x in g], dtype=wp.float32, device=device)
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
        # per-pad arrays: flatten each gripper's pads, recording the owning gripper id and world. Pads are
        # grouped by world (each world's pads contiguous) so a world's pads form a range addressable by the
        # CSR ``pad_world_start`` -- mirrors newton.Model's body_world / body_world_start layout.
        m.world_count = (max((x.world for x in g), default=-1) + 1) if g else 0
        pad_gripper: list[int] = []
        pad_xform: list[wp.transform] = []
        pad_world: list[int] = []
        pad_world_start = [0] * (m.world_count + 2)  # CSR: [per-world starts..., global start, total]
        for w in [*range(m.world_count), -1]:  # each world in order, then global (world -1) pads last
            if w >= 0:
                pad_world_start[w] = len(pad_gripper)
            else:
                pad_world_start[m.world_count] = len(pad_gripper)  # index -2: start of global pads
            for gi, x in enumerate(g):
                if x.world == w:
                    for p in x.pads:
                        pad_gripper.append(gi)
                        pad_xform.append(p)
                        pad_world.append(w)
        pad_world_start[m.world_count + 1] = len(pad_gripper)  # index -1: total pad count
        m.pad_gripper = wp.array(pad_gripper, dtype=wp.int32, device=device)
        m.pad_xform = wp.array(pad_xform, dtype=wp.transform, device=device)
        m.pad_world = wp.array(pad_world, dtype=wp.int32, device=device)
        m.pad_world_start = wp.array(pad_world_start, dtype=wp.int32, device=device)
        return m


class SurfaceGripperModel:
    """Finalized simple-gripper model (mirrors :class:`newton.Model`).

    Constant device arrays; ``gripper_*`` indexed by gripper id, ``pad_*`` by pad id. Reuses the shared
    :class:`SurfaceGripperState` / :class:`SurfaceGripperControl` so the engagement/attach helpers apply.
    """

    def state(self) -> "SurfaceGripperState":
        """Allocate a fresh per-pad :class:`SurfaceGripperState` for this model."""
        s = SurfaceGripperState()
        n = self.pad_xform.shape[0]
        s.pad_break_metric = wp.zeros(n, dtype=wp.float32, device=self.pad_xform.device)
        s.pad_dof_force = wp.zeros(n, dtype=wp.vec4, device=self.pad_xform.device)
        s.pad_engaged = wp.zeros(n, dtype=wp.bool, device=self.pad_xform.device)
        s.pad_body_b = wp.full(n, -1, dtype=wp.int32, device=self.pad_xform.device)
        s.pad_anchor_b = wp.zeros(n, dtype=wp.transform, device=self.pad_xform.device)
        return s

    def control(self) -> "SurfaceGripperControl":
        """Allocate a fresh per-pad :class:`SurfaceGripperControl` for this model."""
        c = SurfaceGripperControl()
        n = self.pad_xform.shape[0]
        c.pad_grip_control = wp.zeros(n, dtype=wp.float32, device=self.pad_xform.device)
        return c


@wp.func
def clamp_symmetric(f: float, f_max: float) -> float:
    """Clamp ``f`` to ``[-f_max, f_max]``. ``f_max <= 0`` means no cap (``f`` returned unchanged)."""
    if f_max > 0.0:
        return wp.clamp(f, -f_max, f_max)
    return f


@wp.func
def clamp_magnitude_2d(fx: float, fy: float, f_max: float) -> tuple[float, float]:
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
def eval_pad_force_linear_kernel(
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    gripper_f_grip_max: wp.array[float],
    gripper_k_normal: wp.array[float],
    gripper_d_normal: wp.array[float],
    gripper_f_normal_max: wp.array[float],
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
    pad_engaged: wp.array[wp.bool],
    pad_body_b: wp.array[int],
    pad_anchor_b: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    dt: float,
    # outputs (mutated in place)
    pad_break_metric: wp.array[float],
    pad_dof_force: wp.array[wp.vec4],
    body_f: wp.array[wp.spatial_vector],
):
    """Per-pad linear spring-damper seal wrench with fixed magnitude caps (see the section header).

    Seal kinematics (seal-frame separation, seal-point relative velocity, implicit damping) drive a
    plain linear force per DOF with four caps (normal, combined shear, combined peel, twist). Writes the
    equal-and-opposite wrench into ``body_f``.
    """
    pad = wp.tid()
    if not pad_engaged[pad]:
        return
    pad_body_b_id = pad_body_b[pad]

    gripper_id = pad_gripper[pad]
    body_a = gripper_body_id[gripper_id]

    # world seal frames on A (TA*TAS) and B (TB*TBS); separation and relative velocity per DOF
    t_a_seal = body_q[body_a] * gripper_xform[gripper_id] * pad_xform[pad]
    q_a_seal = wp.transform_get_rotation(t_a_seal)
    p_a_seal = wp.transform_get_translation(t_a_seal)
    t_b_seal = body_q[pad_body_b_id] * pad_anchor_b[pad]
    p_b_seal = wp.transform_get_translation(t_b_seal)

    px, py, pz, theta_x, theta_y, theta_z = eval_pad_separation(t_a_seal, t_b_seal)

    com_a = wp.transform_point(body_q[body_a], body_com[body_a])
    com_b = wp.transform_point(body_q[pad_body_b_id], body_com[pad_body_b_id])
    r_a = p_a_seal - com_a
    r_b = p_b_seal - com_b
    vx, vy, vz, omega_x, omega_y, omega_z = eval_pad_relative_velocity(
        body_qd[body_a], body_qd[pad_body_b_id], r_a, r_b, q_a_seal
    )

    # implicit (backward-Euler) damping for all six DOFs (normal, shear x/y, peel x/y, twist)
    q_body_b = wp.transform_get_rotation(body_q[pad_body_b_id])
    d_normal_eff, d_shear_x_eff, d_shear_y_eff, d_peel_x_eff, d_peel_y_eff, d_torsion_eff = eval_effective_damping(
        q_a_seal,
        q_body_b,
        r_b,
        body_mass[pad_body_b_id],
        body_inertia[pad_body_b_id],
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
    fz_elastic = gripper_k_normal[gripper_id] * pz
    if fz_elastic > f_vac:
        fz_elastic = f_vac
    fz = fz_elastic + d_normal_eff * vz

    # shear (x, y): spring-damper per axis, combined magnitude capped at f_shear_max
    fx = gripper_k_shear_x[gripper_id] * px + d_shear_x_eff * vx
    fy = gripper_k_shear_y[gripper_id] * py + d_shear_y_eff * vy
    fx, fy = clamp_magnitude_2d(fx, fy, gripper_f_shear_max[gripper_id])

    # peel (about x, y): spring-damper per axis, combined magnitude capped at f_peel_max
    m_peel_x = gripper_k_peel_x[gripper_id] * theta_x + d_peel_x_eff * omega_x
    m_peel_y = gripper_k_peel_y[gripper_id] * theta_y + d_peel_y_eff * omega_y
    m_peel_x, m_peel_y = clamp_magnitude_2d(m_peel_x, m_peel_y, gripper_f_peel_max[gripper_id])

    # twist (about z): linear spring-damper, clamped to +/-f_torsion_max
    m_twist = clamp_symmetric(
        gripper_k_torsion[gripper_id] * theta_z + d_torsion_eff * omega_z, gripper_f_torsion_max[gripper_id]
    )

    # assemble the seal-frame wrench, rotate to world, and accumulate equal-and-opposite on A and B
    force = wp.quat_rotate(q_a_seal, wp.vec3(fx, fy, fz))
    torque = wp.quat_rotate(q_a_seal, wp.vec3(m_peel_x, m_peel_y, m_twist))
    wp.atomic_add(body_f, body_a, wp.spatial_vector(force, torque + wp.cross(r_a, force)))
    wp.atomic_add(body_f, pad_body_b_id, wp.spatial_vector(-force, -torque + wp.cross(r_b, -force)))

    # per-DOF magnitudes for telemetry: (normal fz, shear |(fx,fy)|, peel |(mx,my)|, twist mz) -- for
    # shear and peel this is the combined magnitude compared against the cap (sqrt(x^2 + y^2)).
    pad_dof_force[pad] = wp.vec4(
        fz, wp.sqrt(fx * fx + fy * fy), wp.sqrt(m_peel_x * m_peel_x + m_peel_y * m_peel_y), m_twist
    )

    pad_break_metric[pad] = 0.0  # break not evaluated on this path yet


def evaluate_gripper_force(
    model,
    state,
    gripper_model: SurfaceGripperModel,
    gripper_state: SurfaceGripperState,
    gripper_control: SurfaceGripperControl,
    dt: float,
) -> None:
    """Accumulate the linear per-DOF spring-damper seal wrench (:func:`eval_pad_force_linear_kernel`) into
    ``state.body_f``. No stick-slip anchors and no break metric -- each DOF is a plain spring-damper
    with a fixed magnitude cap. Uses the engagement state (``pad_engaged``, ``pad_body_b``,
    ``pad_anchor_b``).
    """
    n_pads = gripper_model.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        eval_pad_force_linear_kernel,
        dim=n_pads,
        inputs=[
            gripper_model.gripper_body_id,
            gripper_model.gripper_xform,
            gripper_model.gripper_f_grip_max,
            gripper_model.gripper_k_normal,
            gripper_model.gripper_d_normal,
            gripper_model.gripper_f_normal_max,
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
            gripper_state.pad_engaged,
            gripper_state.pad_body_b,
            gripper_state.pad_anchor_b,
            model.body_com,
            state.body_q,
            state.body_qd,
            model.body_mass,
            model.body_inertia,
            dt,
            # outputs (mutated in place)
            gripper_state.pad_break_metric,
            gripper_state.pad_dof_force,
            state.body_f,
        ],
    )


@wp.kernel
def _points_sdf_kernel(
    points: wp.array[wp.vec3],  # query points in world
    obj_xform: wp.transform,  # object world pose
    mesh_id: wp.uint64,  # object surface mesh (the SDF source; not used for collision)
    max_dist: float,  # SDF search radius [m]
    out: wp.array[float],  # signed distances [m]: >0 outside, ~0 on the surface, <0 inside
):
    i = wp.tid()
    p_local = wp.transform_point(wp.transform_inverse(obj_xform), points[i])
    out[i] = sdf_mesh(mesh_id, p_local, max_dist)


def sample_object_sdf(points, obj_xform: wp.transform, mesh: wp.Mesh, max_dist: float = 1.0):
    """Signed distances [m] from world ``points`` (iterable of ``wp.vec3``) to the surface of an object,
    whose geometry is given by the triangle mesh ``mesh`` (used only for the SDF query, not collision),
    posed at ``obj_xform``. Points are rotated into the object frame and sampled with
    :func:`newton.geometry.sdf_mesh`; > 0 outside, ~0 on the surface, < 0 inside. Returns a numpy array.
    """
    pts = list(points)
    if not pts:
        return np.zeros(0)
    dev = mesh.points.device
    pts_wp = wp.array(pts, dtype=wp.vec3, device=dev)
    out = wp.zeros(len(pts), dtype=float, device=dev)
    wp.launch(_points_sdf_kernel, len(pts), inputs=[pts_wp, obj_xform, mesh.id, max_dist], outputs=[out], device=dev)
    return out.numpy()


_TWO_PI = wp.constant(2.0 * math.pi)


# --------------------------------------------------------------------------------------------------
# Inline (graph-capturable) Gauss-Newton seat fit of a gripped body's pose to the pad lips.
#
# Used by attach_seal_seated_kernel on a pad's rising edge (:func:`_seat_body_pose`). Per lip point the
# analytic Jacobian row of its SDF w.r.t. a body-frame pose twist xi = (v, omega) is -[grad; q x grad]
# (q = point in the object frame, grad = SDF gradient there). The normal equations (JtJ, b) are summed
# over the lips of every pad latching the body; a damped 6x6 solve gives the step dxi and the pose is
# updated on-manifold: TB <- TB * exp(dxi). Repeated for a fixed ``iters`` (re-sampling the SDF at the
# updated pose each step, so it converges for a curved object -- 1 step suffices for planar faces),
# all in one kernel thread -- no scratch arrays, so it is graph-capturable.
# --------------------------------------------------------------------------------------------------

_mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
_vec6 = wp.types.vector(length=6, dtype=wp.float32)


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
    bdy: int,
    pad_lo: int,  # scan only [pad_lo, pad_hi): the pads of body bdy's world (see pad_world_start)
    pad_hi: int,
    pad_body_b_id: wp.array[int],
    pad_seal_engaged: wp.array[wp.bool],
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    mesh_id: wp.uint64,
    pad_radius: float,
    pad_face_offset: float,
    n_samples_per_pad: int,
    max_dist: float,
    grad_h: float,
    damping: float,
    iters: int,
) -> wp.transform:
    """Seated world pose of gripped body ``bdy``: ``iters`` Gauss-Newton steps (from its current pose
    ``body_q``) minimizing the SDF of the lips of every pad sealing ``bdy`` after this step
    (``pad_seal_engaged``, i.e. already-engaged or newly-engaging pads on ``bdy``). Each step
    re-samples the SDF at the updated pose, so it converges for a general (curved) object, not just planar
    faces -- a flat face needs 1 step, a curved surface a few. Only ``[pad_lo, pad_hi)`` (``bdy``'s world's pads)
    is scanned, since pads only seal bodies in their own world. See the section header for the Jacobian
    ``a = [grad; q x grad]`` and the on-manifold update ``TB <- TB * exp(dxi)``."""
    tb = body_q[bdy]  # current pose estimate; refined by each Gauss-Newton iteration below
    d_th = _TWO_PI / float(n_samples_per_pad)  # angular step between lip samples (loop-invariant)
    for _it in range(iters):
        inv_tb = wp.transform_inverse(tb)  # re-linearize: sample the lip SDFs at the *current* pose
        jtj = _mat66()
        rhs = _vec6()
        for p in range(pad_lo, pad_hi):
            if pad_body_b_id[p] == bdy and pad_seal_engaged[p]:  # a pad sealing this body after this step
                gripper_id = pad_gripper[p]
                t_rel = inv_tb * (
                    body_q[gripper_body_id[gripper_id]] * gripper_xform[gripper_id] * pad_xform[p]
                )  # seal frame in body
                for s in range(n_samples_per_pad):
                    th = d_th * float(s)
                    # this lip point, expressed in the gripped body's frame (where its SDF is defined)
                    sample_point_in_body_b_frame = wp.transform_point(
                        t_rel, wp.vec3(pad_radius * wp.cos(th), pad_radius * wp.sin(th), pad_face_offset)
                    )
                    sdf = sdf_mesh(mesh_id, sample_point_in_body_b_frame, max_dist)
                    grad = _sdf_grad(mesh_id, sample_point_in_body_b_frame, max_dist, grad_h)
                    cr = wp.cross(sample_point_in_body_b_frame, grad)
                    # Apply a twist (v,w) to the pose of body b and compute the effect on the sdf of the sample point.
                    # The sample point moves in the frame of body b by -(v + w X q).
                    # The difference to the sdf is -grad.(v + w X q)
                    # dSdf = -grad.(v + w X q) = -grad.V - (q X grad).w = [grad, q X grad].[v, w]
                    # We seek Sdf + dSdf = 0.
                    a = _vec6(grad[0], grad[1], grad[2], cr[0], cr[1], cr[2])  # Jacobian row (sdf grows as -a . xi)
                    # Least squares solution: (J^T*J)*(v.w) = J^T * residual
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
def attach_seal_seated_kernel(
    pad_seal_engaged: wp.array[wp.bool],  # [pads] fresh per-pad seal decision
    pad_body_b_id: wp.array[int],  # gripped body each pad seals against this step (< 0 = none)
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_b_mesh_id: wp.array[wp.uint64],  # gripped-body SDF mesh (for the inline seat fit)
    pad_world: wp.array[int],  # world of each pad (see SurfaceGripperModel)
    pad_world_start: wp.array[int],  # CSR: world w's pads are [pad_world_start[w], pad_world_start[w+1])
    pad_radius: float,  # lip circle radius [m]
    pad_face_offset: float,  # lip plane offset along the grip axis (pad local z) [m]
    n_samples_per_pad: int,  # lip points per pad
    max_dist: float,  # SDF search radius [m]
    grad_h: float,  # SDF central-difference step [m]
    damping: float,  # small stabiliser: steadies the fit when the pads don't fully pin the gripped object
    iters: int,  # Gauss-Newton iterations for the seat fit (1 for planar faces, more for curved objects)
    # outputs
    pad_engaged: wp.array[wp.bool],
    pad_body_b: wp.array[int],
    pad_anchor_b: wp.array[wp.transform],
):
    """Seated variant of :func:`attach_seal_kernel`: on each pad's rising edge, compute the gripped body's
    seated pose inline (:func:`_seat_body_pose`, scanning only this pad's world's pads) and latch
    ``pad_anchor_b`` against it (so the seal seats the body on the pads). The seat fit only runs on the
    rising edge -- every other sub-step just early-outs and commits state. Pads of one body redundantly
    recompute its (identical) seated pose; that fires only at engagement, which is rare."""
    pad = wp.tid()
    if pad_seal_engaged[pad] and not pad_engaged[pad]:  # rising edge: latch the gripped body, seat, cache TBS
        pad_body_b[pad] = pad_body_b_id[pad]  # latch the gripped body id at engagement (stays fixed while gripped)
        bdy = pad_body_b_id[pad]
        if bdy >= 0:
            w = pad_world[pad]  # this pad's world; scan only that world's pads for bdy
            hold_pose_body_b = _seat_body_pose(
                bdy,
                pad_world_start[w],
                pad_world_start[w + 1],
                pad_body_b_id,
                pad_seal_engaged,
                gripper_body_id,
                gripper_xform,
                pad_gripper,
                pad_xform,
                body_q,
                body_b_mesh_id[bdy],
                pad_radius,
                pad_face_offset,
                n_samples_per_pad,
                max_dist,
                grad_h,
                damping,
                iters,
            )
            gripper_id = pad_gripper[pad]
            seal_world = body_q[gripper_body_id[gripper_id]] * gripper_xform[gripper_id] * pad_xform[pad]  # TA * TAS
            pad_anchor_b[pad] = wp.transform_inverse(hold_pose_body_b) * seal_world
    pad_engaged[pad] = pad_seal_engaged[pad]


def attach_seal_seated(
    state: newton.State,
    gripper_model: SurfaceGripperModel,
    gripper_state: SurfaceGripperState,
    pad_seal_engaged: wp.array[wp.bool],
    pad_body_b_id: wp.array[int],
    body_b_mesh_id: wp.array[wp.uint64],
    pad_radius: float,
    pad_face_offset: float,
    n_samples_per_pad: int,
    max_dist: float = 1.0,
    grad_h: float = 1.0e-4,
    damping: float = 1.0e-3,
    iters: int = 8,
):
    """For each pad that switches engagement False (``gripper_state.pad_engaged``) -> True
    (``pad_seal_engaged``):

    1. update ``gripper_state.pad_engaged`` and latch ``gripper_state.pad_body_b`` (the gripped body id);
    2. for each gripped body affected by the state change, compute the pose that minimizes the signed
       distance between the gripped body and all pads gripping it;
    3. use that pose to compute ``gripper_state.pad_anchor_b`` (the cached seal frame TBS).

    For each pad that switches engagement True (``gripper_state.pad_engaged``) -> False
    (``pad_seal_engaged``): update ``gripper_state.pad_engaged``.

    Done inline on the device (:func:`attach_seal_seated_kernel`), so it is graph-capturable. Seated
    variant of :func:`attach_seal`, which instead anchors to the gripped body's raw pose.

    Args:
        state: Simulation state; source of ``body_q`` (world body poses) for the seal frames and the fit.
        gripper_model: Finalized gripper holding the pad/gripper layout arrays.
        gripper_state: Gripper state; ``pad_engaged`` / ``pad_body_b`` / ``pad_anchor_b`` are updated in place.
        pad_seal_engaged: This step's fresh per-pad seal decision, shape [n_pads].
        pad_body_b_id: Gripped body each pad seals against this step (< 0 = none), shape [n_pads].
        body_b_mesh_id: Body id -> gripped-object SDF mesh id (a :class:`warp.Mesh` id), shape [n_bodies].
        pad_radius: Pad lip circle radius [m].
        pad_face_offset: Lip-plane offset along the pad's z axis (pad local +z) [m].
        n_samples_per_pad: Number of lip sample points placed around each pad's lip.
        max_dist: SDF search radius [m].
        grad_h: SDF central-difference step [m].
        damping: A small stabiliser for the fit. When the pads don't fully pin the gripped object down --
            e.g. a flat face lets it slide sideways or spin without changing any lip distance -- the fit
            would drift in those free directions. Damping holds them still. Keep it small (too large just
            slows the fit).
        iters: Gauss-Newton iterations for the seat fit. 1 suffices for planar faces; increase for
            curved gripped objects (each iteration re-samples the SDF at the updated pose).
    """
    gm = gripper_model
    n_pads = gm.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        attach_seal_seated_kernel,
        dim=n_pads,
        inputs=[
            pad_seal_engaged,
            pad_body_b_id,
            gm.gripper_body_id,
            gm.gripper_xform,
            gm.pad_gripper,
            gm.pad_xform,
            state.body_q,
            body_b_mesh_id,
            gm.pad_world,
            gm.pad_world_start,
            pad_radius,
            pad_face_offset,
            n_samples_per_pad,
            max_dist,
            grad_h,
            damping,
            iters,
            gripper_state.pad_engaged,
            gripper_state.pad_body_b,
            gripper_state.pad_anchor_b,
        ],
        device=gm.pad_xform.device,
    )
