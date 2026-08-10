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
    SurfaceGripperModel.state_input() / .state_output() / .control() -> SurfaceGripperStateInput / SurfaceGripperStateOutput / SurfaceGripperControl
"""

import math

import numpy as np
import warp as wp

import newton
from newton.geometry import sdf_mesh


class SurfaceGripperStateInput:
    """Per-pad inputs to the gripper simulation, set by the caller. Per-pad arrays.

    The caller writes these fields each step; ``surface_gripper.py`` kernels only read them.
    Engagement and preparing state are encoded directly in the body-ID fields: a non-negative value
    means the pad is engaged or preparing to engage (and identifies the gripped body), while ``< 0`` means
    released or idle.
    """

    pad_engaged_body_b_id: wp.array[wp.int32]  
    pad_preparing_body_b_id: wp.array[wp.int32]  


class SurfaceGripperStateOutput:
    """Per-pad outputs written by the gripper simulation kernels. Per-pad arrays.

    These fields are computed each step by the gripper kernels; the caller reads them for telemetry,
    break detection, and GUI, but should not write them.
    """

    pad_break_metric: wp.array[wp.float32]      # brittle break envelope; > 1 => seal exceeded capacity
    pad_dof_force: wp.array[wp.vec4]            # per-DOF force telemetry (normal, shear mag, peel mag, twist), per pad
    pad_anchor_b: wp.array[wp.transform]        # TBS: seal frame in the gripped body's frame, cached at engagement
    pad_lip_sdf0: wp.array[wp.float32]          # seated lip signed distances cached at engagement (indexed by model.pad_lip_start)
    pad_seal_quality_rms: wp.array[wp.float32]  # RMS lip-gap deviation from seated pose per pad [m]; -1 if not engaged or preparing


class SurfaceGripperControl:
    """Gripper control inputs. Per-pad arrays."""

    pad_grip_control: wp.array  # per-pad grip command [0, 1]; f_min = pad_grip_control * f_grip_max


@wp.kernel
def attach_seal_kernel(
    pad_engaged_body_b_id_curr: wp.array[int],  # [pads] current step's gripped body (< 0 = released)
    pad_preparing_body_b_id: wp.array[int],  # body each pad seals against this step
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],  # world pose of body A (the gripper body)
    hold_pose_body_b: wp.array[wp.transform],  # per-body pose B is held at: raw body_q, or the fitted (seated) pose
    pad_engaged_body_b_id_prev: wp.array[int],  # [pads] previous step's gripped body (< 0 = was released, for rising-edge detection)
    # outputs
    pad_anchor_b: wp.array[wp.transform],
):
    """On a disengaged->engaged rising edge, cache TBS = TB^-1 * TA(t0) * TAS (seal frame in body B).
    Does not write pad_engaged_body_b_id -- that is managed by the caller."""
    pad = wp.tid()
    if pad_engaged_body_b_id_curr[pad] >= 0 and pad_engaged_body_b_id_prev[pad] < 0:
        gripper_id = pad_gripper[pad]
        seal_world = body_q[gripper_body_id[gripper_id]] * gripper_xform[gripper_id] * pad_xform[pad]  # TA * TAS
        pad_anchor_b[pad] = wp.transform_inverse(hold_pose_body_b[pad_preparing_body_b_id[pad]]) * seal_world


def attach_seal(
    state,
    gripper_model: "SurfaceGripperModel",
    gripper_state_input: SurfaceGripperStateInput,
    gripper_state_output: SurfaceGripperStateOutput,
    pad_engaged_body_b_id_curr,
) -> None:
    """Latch ``pad_anchor_b`` for pads that just engaged, then commit the seal state.

    On a disengaged->engaged rising edge, cache ``pad_anchor_b`` (TBS) against the body's raw pose
    (``state.body_q``). Does not write ``pad_engaged_body_b_id`` -- the caller manages that.
    Cf. :func:`attach_seal_seated`, which seats against the inline-fitted pose.
    """
    n_pads = gripper_model.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        attach_seal_kernel,
        dim=n_pads,
        inputs=[
            pad_engaged_body_b_id_curr,
            gripper_state_input.pad_preparing_body_b_id,
            gripper_model.gripper_body_id,
            gripper_model.gripper_xform,
            gripper_model.pad_gripper,
            gripper_model.pad_xform,
            state.body_q,
            state.body_q,  # hold pose = the body's raw pose
            gripper_state_input.pad_engaged_body_b_id,
            gripper_state_output.pad_anchor_b,
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
# the shared :class:`SurfaceGripperStateInput` / :class:`SurfaceGripperStateOutput` / :class:`SurfaceGripperControl`,
# so the engagement helper (:func:`attach_seal`) works unchanged.
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

    def __init__(self, body_id: int, xform: wp.transform, world: int = 0, n_lip_samples: int = 0):
        self.body_id = body_id
        self.xform = xform
        self.world = world  # world (environment) this gripper lives in; -1 for a global gripper
        self.n_lip_samples = n_lip_samples  # lip sample points per pad for this gripper's seat fit / seal metric
        self.pads: list[wp.transform] = []  # pad poses in the gripper frame
        self.pad_radii: list[float] = []  # per-pad lip circle radius [m], parallel to self.pads
        self.pad_half_heights: list[float] = []  # per-pad lip plane offset along the grip axis [m], parallel to self.pads
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
        mass: float,
        inertia: wp.mat33,
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
        design body of ``mass`` [kg] and ``inertia`` [kg.m^2, body frame]. Translation DOFs use the mass;
        peel/twist use the inertia about that axis (the diagonal terms). Returns ``self``.
        """
        m = mass
        ixx = inertia[0, 0]  # inertia about x (peel-x)
        iyy = inertia[1, 1]  # about y (peel-y)
        izz = inertia[2, 2]  # about z (twist)
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

    def add_pad(self, xform: wp.transform, radius: float, half_height: float) -> int:
        """Add a pad at ``xform`` (gripper frame) with lip circle ``radius`` [m] and ``half_height`` [m]
        (the lip-plane offset along the grip axis). Returns its index within this gripper."""
        self.pads.append(xform)
        self.pad_radii.append(radius)
        self.pad_half_heights.append(half_height)
        return len(self.pads) - 1


def _lip_circle(radius, half_height, n_samples):
    """The ``n_samples`` lip sample points of one pad, in the pad frame: evenly spaced on the circle of
    ``radius`` at height ``half_height`` (the lip-plane offset along the grip axis). Returns a list of
    ``wp.vec3``; empty when ``n_samples`` is 0."""
    points = []
    if n_samples <= 0:
        return points
    d_th = 2.0 * math.pi / float(n_samples)
    for s in range(n_samples):
        th = d_th * float(s)
        points.append(wp.vec3(radius * math.cos(th), radius * math.sin(th), half_height))
    return points


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
        """Flatten all grippers into a device-resident :class:`SurfaceGripperModel`.

        Each gripper's ``n_lip_samples`` (lip sample points per pad, for the seat fit / seal-quality metric)
        is stored per gripper (``n_lip_samples``). The (constant, pad-frame) lip positions are precomputed once
        into ``pad_lip_local`` so the kernels don't rebuild cos/sin each step; because the count can differ per
        gripper, each pad's points are addressed by the start-index array ``pad_lip_start`` (``pad_lip_sdf0`` shares it).
        """
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
        # world_count = highest world index + 1 (the global world -1 does not count)
        m.world_count = 0
        for gi in range(len(g)):
            w = g[gi].world
            if w >= 0 and w + 1 > m.world_count:
                m.world_count = w + 1

        # Flatten every gripper's pads into per-pad arrays, ordered by world: world 0's pads first, then
        # world 1's, ..., then the global world (-1) last. So each world's pads are one contiguous range,
        # whose first index is recorded in pad_world_start. For each pad we also lay its lip sample points
        # (constant, pad frame) into pad_lip_local; pad_lip_start[p] marks where pad p's points start (a
        # gripper's n_lip_samples can differ, hence a start-index array, not a fixed stride). pad_lip_sdf0 shares pad_lip_start.
        pad_gripper = []
        pad_xform = []
        pad_world = []
        pad_lip_local = []
        pad_lip_start = [0]  # [n_pads + 1]: pad p's lip points are pad_lip_local[pad_lip_start[p] : pad_lip_start[p+1]]
        pad_world_start = [0] * (m.world_count + 2)  # per-world starts, then the global start, then the total

        # the order of worlds to visit: 0, 1, ..., world_count-1, then -1 (global)
        world_order = []
        for w in range(m.world_count):
            world_order.append(w)
        world_order.append(-1)

        for oi in range(len(world_order)):
            w = world_order[oi]
            if w >= 0:
                pad_world_start[w] = len(pad_gripper)  # first pad of world w
            else:
                pad_world_start[m.world_count] = len(pad_gripper)  # first global (world -1) pad
            for gi in range(len(g)):
                gripper = g[gi]
                if gripper.world != w:
                    continue
                for pi in range(len(gripper.pads)):
                    pad_gripper.append(gi)
                    pad_xform.append(gripper.pads[pi])
                    pad_world.append(w)
                    lip = _lip_circle(gripper.pad_radii[pi], gripper.pad_half_heights[pi], gripper.n_lip_samples)
                    for li in range(len(lip)):
                        pad_lip_local.append(lip[li])
                    pad_lip_start.append(len(pad_lip_local))  # end of this pad's lip points
        pad_world_start[m.world_count + 1] = len(pad_gripper)  # total pad count

        gripper_n_lip_samples = []
        for gi in range(len(g)):
            gripper_n_lip_samples.append(g[gi].n_lip_samples)

        m.n_lip_samples = wp.array(gripper_n_lip_samples, dtype=wp.int32, device=device)  # [grippers]
        m.pad_gripper = wp.array(pad_gripper, dtype=wp.int32, device=device)
        m.pad_xform = wp.array(pad_xform, dtype=wp.transform, device=device)
        m.pad_lip_local = wp.array(pad_lip_local, dtype=wp.vec3, device=device)  # indexed by pad_lip_start
        m.pad_lip_start = wp.array(pad_lip_start, dtype=wp.int32, device=device)  # [n_pads+1] start indices into pad_lip_local
        m.pad_world = wp.array(pad_world, dtype=wp.int32, device=device)
        m.pad_world_start = wp.array(pad_world_start, dtype=wp.int32, device=device)
        return m


class SurfaceGripperModel:
    """Finalized simple-gripper model (mirrors :class:`newton.Model`).

    Constant device arrays; ``gripper_*`` indexed by gripper id, ``pad_*`` by pad id. Use
    :meth:`state_input` / :meth:`state_output` to allocate the matching per-step state objects.
    """

    def state_input(self) -> SurfaceGripperStateInput:
        """Allocate a fresh per-pad :class:`SurfaceGripperStateInput` for this model."""
        si = SurfaceGripperStateInput()
        n = self.pad_xform.shape[0]
        si.pad_engaged_body_b_id = wp.full(n, -1, dtype=wp.int32, device=self.pad_xform.device)
        si.pad_preparing_body_b_id = wp.full(n, -1, dtype=wp.int32, device=self.pad_xform.device)
        return si

    def state_output(self) -> SurfaceGripperStateOutput:
        """Allocate a fresh per-pad :class:`SurfaceGripperStateOutput` for this model. ``pad_lip_sdf0`` shares
        the start-index scheme of ``pad_lip_local`` (one entry per lip sample point across all pads)."""
        so = SurfaceGripperStateOutput()
        n = self.pad_xform.shape[0]
        so.pad_break_metric = wp.zeros(n, dtype=wp.float32, device=self.pad_xform.device)
        so.pad_dof_force = wp.zeros(n, dtype=wp.vec4, device=self.pad_xform.device)
        so.pad_anchor_b = wp.zeros(n, dtype=wp.transform, device=self.pad_xform.device)
        so.pad_lip_sdf0 = wp.zeros(self.pad_lip_local.shape[0], dtype=wp.float32, device=self.pad_xform.device)
        so.pad_seal_quality_rms = wp.zeros(n, dtype=wp.float32, device=self.pad_xform.device)
        return so

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
    pad_engaged_body_b_id: wp.array[int],
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
    if pad_engaged_body_b_id[pad] < 0:
        return
    engaged_body_b = pad_engaged_body_b_id[pad]

    gripper_id = pad_gripper[pad]
    body_a = gripper_body_id[gripper_id]

    # world seal frames on A (TA*TAS) and B (TB*TBS); separation and relative velocity per DOF
    t_a_seal = body_q[body_a] * gripper_xform[gripper_id] * pad_xform[pad]
    q_a_seal = wp.transform_get_rotation(t_a_seal)
    p_a_seal = wp.transform_get_translation(t_a_seal)
    t_b_seal = body_q[engaged_body_b] * pad_anchor_b[pad]
    p_b_seal = wp.transform_get_translation(t_b_seal)

    px, py, pz, theta_x, theta_y, theta_z = eval_pad_separation(t_a_seal, t_b_seal)

    com_a = wp.transform_point(body_q[body_a], body_com[body_a])
    com_b = wp.transform_point(body_q[engaged_body_b], body_com[engaged_body_b])
    r_a = p_a_seal - com_a
    r_b = p_b_seal - com_b
    vx, vy, vz, omega_x, omega_y, omega_z = eval_pad_relative_velocity(
        body_qd[body_a], body_qd[engaged_body_b], r_a, r_b, q_a_seal
    )

    # implicit (backward-Euler) damping for all six DOFs (normal, shear x/y, peel x/y, twist)
    q_body_b = wp.transform_get_rotation(body_q[engaged_body_b])
    d_normal_eff, d_shear_x_eff, d_shear_y_eff, d_peel_x_eff, d_peel_y_eff, d_torsion_eff = eval_effective_damping(
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
    wp.atomic_add(body_f, engaged_body_b, wp.spatial_vector(-force, -torque + wp.cross(r_b, -force)))

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
    gripper_state_input: SurfaceGripperStateInput,
    gripper_state_output: SurfaceGripperStateOutput,
    gripper_control: SurfaceGripperControl,
    dt: float,
) -> None:
    """Accumulate the linear per-DOF spring-damper seal wrench (:func:`eval_pad_force_linear_kernel`) into
    ``state.body_f``. No stick-slip anchors and no break metric -- each DOF is a plain spring-damper
    with a fixed magnitude cap. Uses the engagement state (``pad_engaged``, ``pad_engaged_body_b_id``,
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
            gripper_state_input.pad_engaged_body_b_id,
            gripper_state_output.pad_anchor_b,
            model.body_com,
            state.body_q,
            state.body_qd,
            model.body_mass,
            model.body_inertia,
            dt,
            # outputs (mutated in place)
            gripper_state_output.pad_break_metric,
            gripper_state_output.pad_dof_force,
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
    body_b: int,
    pad_lo: int,  # scan only [pad_lo, pad_hi): the pads of body body_b's world (see pad_world_start)
    pad_hi: int,
    pad_preparing_body_b_id: wp.array[int],
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_b_mesh_xform: wp.array[wp.transform],  # body_b_mesh_xform[b] = T_bs for body b
    mesh_id: wp.uint64,
    pad_lip_local: wp.array[wp.vec3],  # precomputed lip points in the pad frame, indexed by pad_lip_start (start indices per pad)
    pad_lip_start: wp.array[int],  # [n_pads+1] start indices: pad p's lip points are pad_lip_local[pad_lip_start[p] : pad_lip_start[p+1]]
    max_dist: float,
    grad_h: float,
    damping: float,
    iters: int,
) -> wp.transform:
    """
    Compute the pose of body b that minimises the rms of the signed distances of sample points 
    arranged around the lips of all pads gripping (or preparing to grip) body b.
    With the following nomenclature:
    GA = pose of end effector body in world frame
    sgA = pose of surface gripper in end effector body frame
    padA = pose of a pad in the surface grippe frame
    lipA = pose of sample point in pad frame
    GB = pose of body b (gripped body) in world frame
    T_bs = pose of mesh being gripped in body b frame
    lipB = pose of sample point in frame of mesh b
    we may compute the pose of lipA in the frame of the mesh of body b
    lipB = (GB * T_bs)⁻¹ * (GA * sgA * padA) * lipA.
    Introduce SA = (GA * sgA * padA)
    and we have 
    lipB = (GB * T_bs)⁻¹ * SA * lipA.
    It makes sense to cache (GB * T_bs)⁻¹ * SA for each pad being considered.
    For each sample point lipB we compute the signed distance using the sdf mesh.
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
    (J^T*J) * (v, w) = J^T * residual"""

    # T_bs and its rotation are constant for this body; precompute once before the Gauss-Newton loop.
    T_bs = body_b_mesh_xform[body_b]
    R_bs = wp.transform_get_rotation(T_bs)
    tb = body_q[body_b]  # current pose estimate; refined by each Gauss-Newton iteration below
    for _ in range(iters):
        # (GB · T_bs)^-1
        inv_mesh_world = wp.transform_inverse(tb * T_bs)
        jtj = _mat66()
        rhs = _vec6()
        for p in range(pad_lo, pad_hi):
            # a pad participates in the fit when it is targeting this body (engaged or preparing)
            if pad_preparing_body_b_id[p] == body_b:
                gripper_id = pad_gripper[p]
                seal_world_pad = body_q[gripper_body_id[gripper_id]] * gripper_xform[gripper_id] * pad_xform[p]  # SA
                t_rel_mesh = inv_mesh_world * seal_world_pad  # (GB · T_bs)^-1 · SA: seal frame in mesh-local
                for k in range(pad_lip_start[p], pad_lip_start[p + 1]):
                    # lipB = (GB · T_bs)^-1 · SA · lipA: sample point in mesh frame
                    sample_point_in_mesh = wp.transform_point(t_rel_mesh, pad_lip_local[k])
                    sdf = sdf_mesh(mesh_id, sample_point_in_mesh, max_dist)
                    grad_mesh = _sdf_grad(mesh_id, sample_point_in_mesh, max_dist, grad_h)
                    # Jacobian uses body-frame quantities: rotate point and gradient from mesh to body frame.
                    x_body = wp.transform_point(T_bs, sample_point_in_mesh)
                    grad_body = wp.quat_rotate(R_bs, grad_mesh)
                    cr = wp.cross(x_body, grad_body)
                    a = _vec6(grad_body[0], grad_body[1], grad_body[2], cr[0], cr[1], cr[2])  # Jacobian row (sdf grows as -a . xi)
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
    pad_engaged_body_b_id_curr: wp.array[int],  # [pads] current step's gripped body (< 0 = released)
    pad_preparing_body_b_id: wp.array[int],  # gripped body each pad seals against this step (< 0 = none)
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_b_mesh_id: wp.array[wp.uint64],  # gripped-body SDF mesh (for the inline seat fit)
    body_b_mesh_xform: wp.array[wp.transform],  # body_b_mesh_xform[b] = T_bs for body b
    pad_world: wp.array[int],  # world of each pad (see SurfaceGripperModel)
    pad_world_start: wp.array[int],  # start indices: world w's pads are pad-ids [pad_world_start[w] : pad_world_start[w+1]]
    pad_lip_local: wp.array[wp.vec3],  # precomputed lip points in the pad frame, indexed by pad_lip_start (start indices per pad)
    pad_lip_start: wp.array[int],  # [n_pads+1] start indices: pad p's lip points are pad_lip_local[pad_lip_start[p] : pad_lip_start[p+1]]
    max_dist: float,  # SDF search radius [m]
    grad_h: float,  # SDF central-difference step [m]
    damping: float,  # small stabiliser: steadies the fit when the pads don't fully pin the gripped object
    iters: int,  # Gauss-Newton iterations for the seat fit (1 for planar faces, more for curved objects)
    pad_engaged_body_b_id_prev: wp.array[int],  # [pads] previous step's gripped body (< 0 = was released, for rising-edge detection)
    # outputs
    pad_anchor_b: wp.array[wp.transform],
    pad_lip_sdf0: wp.array[float],  # seated lip signed distances cached at engagement (indexed by pad_lip_start)
):
    """Seated variant of :func:`attach_seal_kernel`: on each pad's rising edge, compute the gripped body's
    seated pose inline (:func:`_seat_body_pose`, scanning only this pad's world's pads), cache
    ``pad_anchor_b`` against it, and cache the seated lip signed distances in ``pad_lip_sdf0``. The seat
    fit only runs on the rising edge. Does not write ``pad_engaged_body_b_id``."""
    pad = wp.tid()
    if pad_engaged_body_b_id_curr[pad] >= 0 and pad_engaged_body_b_id_prev[pad] < 0:  # rising edge: seat and cache TBS
        body_b = pad_preparing_body_b_id[pad]
        if body_b >= 0:
            w = pad_world[pad]  # this pad's world; scan only that world's pads for body_b
            hold_pose_body_b = _seat_body_pose(
                body_b,
                pad_world_start[w],
                pad_world_start[w + 1],
                pad_preparing_body_b_id,
                gripper_body_id,
                gripper_xform,
                pad_gripper,
                pad_xform,
                body_q,
                body_b_mesh_xform,
                body_b_mesh_id[body_b],
                pad_lip_local,
                pad_lip_start,
                max_dist,
                grad_h,
                damping,
                iters,
            )

            # GA = pose of end effector body in world frame
            # sgA = pose of surface gripper in end effector body frame
            # padA = pose of a pad in the surface grippe frame
            # lipA = pose of sample point in pad frame
            # GB = pose of body b (gripped body) in world frame
            # T_bs = pose of mesh being gripped in body b frame
            # lipB = pose of sample point in frame of mesh b
            # we may compute the pose of lipA in the frame of the mesh of body b
            # lipB = (GB * T_bs)⁻¹ * (GA * sgA * padA) * lipA.
            # Introduce SA = (GA * sgA * padA)
            # and we have 
            # lipB = (GB * T_bs)⁻¹ * SA * lipA.
            # Expand to produce
            # lipB = T_bs⁻¹ * GB⁻¹ * SA * lipA.
            gripper_id = pad_gripper[pad]
            seal_world = body_q[gripper_body_id[gripper_id]] * gripper_xform[gripper_id] * pad_xform[pad]  # TA * TAS
            pad_anchor_b[pad] = wp.transform_inverse(hold_pose_body_b) * seal_world
            # Cache the seated lip signed distances: the lip circle (precomputed pad-frame points) placed by
            # the seated seal frame (pad_anchor_b, in the body frame). The seal-quality metric measures deviation.
            mesh_b = body_b_mesh_id[body_b]
            pad_anchor_in_mesh = wp.transform_inverse(body_b_mesh_xform[body_b]) * pad_anchor_b[pad]
            for k in range(pad_lip_start[pad], pad_lip_start[pad + 1]):
                lip_mesh = wp.transform_point(pad_anchor_in_mesh, pad_lip_local[k])
                pad_lip_sdf0[k] = sdf_mesh(mesh_b, lip_mesh, max_dist)


def attach_seal_seated(
    state: newton.State,
    gripper_model: SurfaceGripperModel,
    gripper_state_input: SurfaceGripperStateInput,
    gripper_state_output: SurfaceGripperStateOutput,
    pad_engaged_body_b_id_curr: wp.array[int],
    body_b_mesh_id: wp.array[wp.uint64],
    body_b_mesh_xform: wp.array[wp.transform],
    max_dist: float = 1.0,
    grad_h: float = 1.0e-4,
    damping: float = 1.0e-3,
    iters: int = 8,
):
    """For each pad that switches engagement (``pad_engaged_body_b_id_prev < 0``) -> engaged
    (``pad_engaged_body_b_id_curr >= 0``):

    1. latch ``gripper_state_input.pad_engaged_body_b_id`` (the gripped body id);
    2. for each gripped body affected by the state change, compute the pose that minimizes the signed
       distance between the gripped body and all pads gripping it;
    3. use that pose to compute ``gripper_state_output.pad_anchor_b`` (the cached seal frame TBS).

    Done inline on the device (:func:`attach_seal_seated_kernel`), so it is graph-capturable. Seated
    variant of :func:`attach_seal`, which instead anchors to the gripped body's raw pose.

    Args:
        state: Simulation state; source of ``body_q`` (world body poses) for the seal frames and the fit.
        gripper_model: Finalized gripper holding the pad/gripper layout arrays.
        gripper_state_input: Caller-controlled per-pad input state; ``pad_engaged_body_b_id`` is read as
            the previous step's engagement (< 0 = released). ``pad_preparing_body_b_id`` (the target body,
            < 0 = none) encodes preparing state (``>= 0`` means preparing).
        gripper_state_output: Gripper output state; ``pad_anchor_b`` and ``pad_lip_sdf0`` are written on
            each engagement rising edge.
        pad_engaged_body_b_id_curr: This step's fresh per-pad gripped body id (< 0 = released), shape [n_pads].
        body_b_mesh_id: Body id -> gripped-object SDF mesh id (a :class:`warp.Mesh` id), shape [n_bodies].
        body_b_mesh_xform: Body id -> body-to-mesh-local transform, shape [n_bodies]. Use
            :func:`warp.transform_identity` for each body whose mesh is centred at the body origin.
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
            pad_engaged_body_b_id_curr,
            gripper_state_input.pad_preparing_body_b_id,
            gm.gripper_body_id,
            gm.gripper_xform,
            gm.pad_gripper,
            gm.pad_xform,
            state.body_q,
            body_b_mesh_id,
            body_b_mesh_xform,
            gm.pad_world,
            gm.pad_world_start,
            gm.pad_lip_local,
            gm.pad_lip_start,
            max_dist,
            grad_h,
            damping,
            iters,
            gripper_state_input.pad_engaged_body_b_id,
            gripper_state_output.pad_anchor_b,
            gripper_state_output.pad_lip_sdf0,
        ],
        device=gm.pad_xform.device,
    )


@wp.kernel
def detect_pad_engagement_rising_edge_kernel(
    pad_engaged_body_b_id_prev: wp.array[int],  # [pads] gripped body from the previous sub-step (< 0 = released)
    pad_engaged_body_b_id_curr: wp.array[int],  # [pads] gripped body from the current sub-step (< 0 = released)
    pad_rising_edge: wp.array[wp.bool],      # [pads] out: True where released->engaged transition occurred
):
    pad = wp.tid()
    pad_rising_edge[pad] = pad_engaged_body_b_id_curr[pad] >= 0 and pad_engaged_body_b_id_prev[pad] < 0


def detect_pad_engagement_rising_edge(
    gripper_state_input_prev: SurfaceGripperStateInput,
    gripper_state_input_curr: SurfaceGripperStateInput,
    pad_rising_edge: wp.array[wp.bool],
) -> None:
    """Per-pad rising-edge detection for the engaged flag: True where ``pad_engaged`` transitioned
    False -> True between two consecutive sub-step input states.

    Compare ``gripper_state_input_0`` (previous sub-step) with ``gripper_state_input_1`` (current
    sub-step, after the swap) to find pads that newly engaged this step.

    Args:
        gripper_state_input_prev: Input state from the previous sub-step.
        gripper_state_input_curr: Input state from the current sub-step.
        pad_rising_edge: Per-pad output array [n_pads], bool.
    """
    n_pads = pad_rising_edge.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        detect_pad_engagement_rising_edge_kernel,
        dim=n_pads,
        inputs=[gripper_state_input_prev.pad_engaged_body_b_id, gripper_state_input_curr.pad_engaged_body_b_id],
        outputs=[pad_rising_edge],
        device=pad_rising_edge.device,
    )



@wp.kernel
def seal_quality_kernel(
    pad_engaged_body_b_id: wp.array[int],  # [pads] latched gripped body while engaged (< 0 = released)
    pad_preparing_body_b_id: wp.array[int],  # [pads] body a preparing pad is approaching (>= 0 means preparing)
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_b_mesh_id: wp.array[wp.uint64],
    body_b_mesh_xform: wp.array[wp.transform],  # body_b_mesh_xform[b] = T_bs for body b
    pad_world: wp.array[int],
    pad_world_start: wp.array[int],
    pad_lip_local: wp.array[wp.vec3],  # precomputed lip points in the pad frame, indexed by pad_lip_start (start indices per pad)
    pad_lip_sdf0: wp.array[float],  # seated lip SDFs cached at engagement (used only for engaged pads)
    pad_lip_start: wp.array[int],  # [n_pads+1] start indices: pad p's lip points are pad_lip_local[pad_lip_start[p] : pad_lip_start[p+1]]
    max_dist: float,
    grad_h: float,  # seat-fit params (preparing case only)
    damping: float,
    iters: int,
    # output: one root-mean-square value per pad. Each thread writes only its own pad, so no atomics.
    pad_rms: wp.array[float],  # [pads] this pad's RMS lip-gap deviation [m] (-1 if not gripping/preparing)
):
    pad = wp.tid()
    pad_rms[pad] = -1.0  # sentinel: "not evaluated" (overwritten below when engaged or preparing)
    engaged = pad_engaged_body_b_id[pad] >= 0
    preparing = pad_preparing_body_b_id[pad] >= 0
    if not (engaged or preparing):
        return  # released / not gripping and not preparing -> contributes nothing
    if preparing:
        body_b = pad_preparing_body_b_id[pad]  # preparing: the crate being approached
    else:
        body_b = pad_engaged_body_b_id[pad]  # engaged: the latched gripped body
    if body_b < 0:
        return
    gripper_id = pad_gripper[pad]
    seal_world = body_q[gripper_body_id[gripper_id]] * gripper_xform[gripper_id] * pad_xform[pad]
    mesh_b = body_b_mesh_id[body_b]
    # world pose of the mesh: body world pose composed with the mesh-in-body pose (GB * T_bs)
    mesh_world = body_q[body_b] * body_b_mesh_xform[body_b]
    # seal frame in mesh-local coordinates: (GB * T_bs)^-1 * SA
    t_seal_mesh = wp.transform_inverse(mesh_world) * seal_world
    preparing_anchor_mesh = wp.transform_identity()  # seated seal frame in mesh frame (preparing only; recomputed below)
    if preparing:
        w = pad_world[pad]
        seated = _seat_body_pose(  # recompute the seated pose of the approached crate
            body_b,
            pad_world_start[w],
            pad_world_start[w + 1],
            pad_preparing_body_b_id,
            gripper_body_id,
            gripper_xform,
            pad_gripper,
            pad_xform,
            body_q,
            body_b_mesh_xform,
            mesh_b,
            pad_lip_local,
            pad_lip_start,
            max_dist,
            grad_h,
            damping,
            iters,
        )
        preparing_anchor_body = wp.transform_inverse(seated) * seal_world  # seated seal frame in the body frame
        preparing_anchor_mesh = wp.transform_inverse(body_b_mesh_xform[body_b]) * preparing_anchor_body  # body frame -> mesh frame
    dev_sq = float(0.0)  # this pad's running totals (this thread owns pad_rms[pad], so no atomics needed)
    count = int(0)
    for k in range(pad_lip_start[pad], pad_lip_start[pad + 1]):
        lip = pad_lip_local[k]
        sdf0 = pad_lip_sdf0[k]  # engaged: cached seated sdf0
        if preparing:
            sdf0 = sdf_mesh(mesh_b, wp.transform_point(preparing_anchor_mesh, lip), max_dist)  # preparing: recomputed live
        if sdf0 >= max_dist:
            pad_rms[pad] = -1.0  # lip point outside SDF search radius: result is unreliable
            return
        sdf_now = sdf_mesh(mesh_b, wp.transform_point(t_seal_mesh, lip), max_dist)
        if sdf_now >= max_dist:
            pad_rms[pad] = -1.0  # lip point outside SDF search radius: result is unreliable
            return
        dev = sdf_now - sdf0
        dev_sq += dev * dev
        count += 1
    if count > 0:
        pad_rms[pad] = wp.sqrt(dev_sq / float(count))  # root-mean-square over this pad's lips


def evaluate_seal_quality(
    state: newton.State,
    gripper_model: SurfaceGripperModel,
    gripper_state_input: SurfaceGripperStateInput,
    gripper_state_output: SurfaceGripperStateOutput,
    body_b_mesh_id: wp.array[wp.uint64],
    body_b_mesh_xform: wp.array[wp.transform],
    pad_rms: wp.array[float],
    max_dist: float = 1.0,
    grad_h: float = 1.0e-4,
    damping: float = 1.0e-3,
    iters: int = 8,
):
    """
    Compute a geometric seal quality per pad (pad_rms[pad]).
    Three mutually exclusive modes of operation: preparing (to grip), engaged (currently gripping)
    and disengaged. A pad is in preparing mode when ``pad_preparing_body_b_id[pad] >= 0``.
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
    In disengaged mode, pad_rms is set to -1.

    Args:
        state: Simulation state; source of ``body_q`` (world body poses).
        gripper_model: Finalized gripper holding the pad/gripper layout arrays.
        gripper_state_input: Per-pad input state; ``pad_engaged_body_b_id`` and
            ``pad_preparing_body_b_id`` select which mode each pad is in.
        gripper_state_output: Per-pad output state; ``pad_lip_sdf0`` provides the cached
            seated baseline; ``pad_seal_quality_rms`` receives the result.
        body_b_mesh_id: Body id -> gripped-object SDF mesh id, shape [n_bodies].
        body_b_mesh_xform: Body id -> body-to-mesh-local transform, shape [n_bodies]. Use
            :func:`warp.transform_identity` for each body whose mesh is centred at the body origin.
        pad_rms: Per-pad output array [n_pads]; RMS lip-gap deviation [m], or ``-1`` if idle.
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
        seal_quality_kernel,
        dim=n_pads,
        inputs=[
            gripper_state_input.pad_engaged_body_b_id,
            gripper_state_input.pad_preparing_body_b_id,
            gm.gripper_body_id,
            gm.gripper_xform,
            gm.pad_gripper,
            gm.pad_xform,
            state.body_q,
            body_b_mesh_id,
            body_b_mesh_xform,
            gm.pad_world,
            gm.pad_world_start,
            gm.pad_lip_local,
            gripper_state_output.pad_lip_sdf0,
            gm.pad_lip_start,
            max_dist,
            grad_h,
            damping,
            iters,
            pad_rms,
        ],
        device=gm.pad_xform.device,
    )
