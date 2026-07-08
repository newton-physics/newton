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

"""Surface-gripper (suction-cup) model: authoring types, the finalized model/state/control,
and the per-pad force kernel. Imported by ``example_suction_cup``; not a runnable example.

Mirrors Newton's Builder -> Model -> State/Control layout::

    SurfaceGripper (authoring) -> SurfaceGripperBuilder.finalize() -> SurfaceGripperModel
    SurfaceGripperModel.state() / .control() -> SurfaceGripperState / SurfaceGripperControl
"""

import math
from enum import IntEnum

import warp as wp


class PadShape(IntEnum):
    CIRCLE = 0
    ELLIPSE = 1
    RECTANGLE = 2


def _pad_geometry_factors(shape: int, a: float, b: float):
    """Geometry-derived force factors for one pad (see gripper.pdf tables/appendices).

    Returns ``(peel_ratio_x, peel_ratio_y, capacity_x, capacity_y, tw_x, tw_y)`` such that::

        k_peel_x = k_normal * peel_ratio_x
        k_peel_y = k_normal * peel_ratio_y
        k_torsion = k_shear_x * peel_ratio_x + k_shear_y * peel_ratio_y
        F_peel_x_max = N_f * capacity_x
        F_peel_y_max = N_f * capacity_y
        M_twist_max = F_sucker * (mu_x * tw_x + mu_y * tw_y)

    Here ``peel_ratio = I / A`` (second area moment over area; sets peel/torsion stiffness),
    ``capacity = (I / A) / c`` where ``c`` is the pad half-extent perpendicular to the tilt
    axis -- the pull-off pressure vanishes at that trailing edge (Appendix 3) -- and ``tw`` is
    the torsional-capacity factor derived from the first moment ``int r dA`` (Appendices 4-5).
    """
    if shape == int(PadShape.CIRCLE):
        # circle of radius R = a; fully symmetric, so the x and y factors are identical.
        radius = a
        # peel/torsion stiffness factor I / A, with I = pi R^4 / 4 and A = pi R^2.
        peel_ratio_x = radius * radius / 4.0
        peel_ratio_y = radius * radius / 4.0
        # peel capacity factor (I/A)/c with c = R (edge where the pull-off pressure vanishes,
        # Appendix 3): (R^2/4) / R = R/4.
        capacity_x = radius / 4.0
        capacity_y = radius / 4.0
        # twist factor: int r dA / A = 2R/3, split per axis by the mu_x/mu_y cos^2/sin^2
        # integration, giving M_twist_max = (mu_x + mu_y)/3 * N_f * R -> R/3 per axis (Appendix 5).
        tw_x = radius / 3.0
        tw_y = radius / 3.0
        return peel_ratio_x, peel_ratio_y, capacity_x, capacity_y, tw_x, tw_y

    if shape == int(PadShape.ELLIPSE):
        # ellipse with semi-axis a along x and b along y.
        # stiffness factor I / A: I_x = pi a b^3 / 4 over A = pi a b gives b^2 / 4 (and a^2 / 4).
        peel_ratio_x = b * b / 4.0
        peel_ratio_y = a * a / 4.0
        # peel capacity factor: b/4 about x, a/4 about y (Appendix 3).
        capacity_x = b / 4.0
        capacity_y = a / 4.0
        # twist factor tw = J / (3 pi a b) with J_x = int R(phi)^3 sin^2(phi) dphi (Appendix 5).
        # No closed form, so integrate numerically over phi in [0, 2 pi) with the midpoint rule.
        num_samples = 256
        d_phi = 2.0 * math.pi / float(num_samples)
        j_x = 0.0
        j_y = 0.0
        for i in range(num_samples):
            phi = (float(i) + 0.5) * d_phi  # sample the middle of each interval
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            # distance from the centre to the ellipse edge at angle phi
            r_edge = a * b / math.sqrt(b * b * cos_phi * cos_phi + a * a * sin_phi * sin_phi)
            j_x += r_edge**3 * sin_phi * sin_phi * d_phi
            j_y += r_edge**3 * cos_phi * cos_phi * d_phi
        twist_denom = 3.0 * math.pi * a * b
        tw_x = j_x / twist_denom
        tw_y = j_y / twist_denom
        return peel_ratio_x, peel_ratio_y, capacity_x, capacity_y, tw_x, tw_y

    # RECTANGLE with half-length a along x and b along y.
    # stiffness factor I / A: I_x = 4 a b^3 / 3 over A = 4 a b gives b^2 / 3 (and a^2 / 3).
    peel_ratio_x = b * b / 3.0
    peel_ratio_y = a * a / 3.0
    # peel capacity factor: b/3 about x, a/3 about y (Appendix 3).
    capacity_x = b / 3.0
    capacity_y = a / 3.0
    # twist factor tw = U / (4 a b), with the closed-form integrals U_x, U_y (Appendix 5).
    diagonal = math.sqrt(a * a + b * b)
    u_x = (2.0 * a * b * diagonal) / 3.0 + (4.0 * b**3 / 3.0) * math.asinh(a / b) - (2.0 * a**3 / 3.0) * math.asinh(b / a)
    u_y = (2.0 * a * b * diagonal) / 3.0 + (4.0 * a**3 / 3.0) * math.asinh(b / a) - (2.0 * b**3 / 3.0) * math.asinh(a / b)
    area = 4.0 * a * b
    tw_x = u_x / area
    tw_y = u_y / area
    return peel_ratio_x, peel_ratio_y, capacity_x, capacity_y, tw_x, tw_y


class SurfaceGripper:
    """An individual surface gripper (authoring object).

    Holds one gripper's parameters and its pads. Author it directly and add pads
    with :meth:`add_pad` -- no global indexing to worry about.
    :meth:`SurfaceGripperBuilder.finalize` flattens a set of these into the model
    arrays and resolves all the indexing.
    """

    def __init__(
        self,
        body_id: int,
        xform: wp.transform,
        k_normal: float,
        d_normal: float,
        f_normal_max: float,
        f_grip_max: float,
        k_shear_x: float,
        k_shear_y: float,
        mu_x: float,
        mu_y: float,
        d_peel_x: float,
        d_peel_y: float,
        shape: int,
        dim_a: float,
        dim_b: float,
    ):
        self.body_id = body_id
        self.xform = xform
        self.k_normal = k_normal
        self.d_normal = d_normal
        self.f_normal_max = f_normal_max
        self.f_grip_max = f_grip_max
        self.k_shear_x = k_shear_x
        self.k_shear_y = k_shear_y
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.d_peel_x = d_peel_x
        self.d_peel_y = d_peel_y
        self.shape = shape
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.pads: list[wp.transform] = []  # pad poses in the gripper frame

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
        m.gripper_k_normal = wp.array([x.k_normal for x in g], dtype=wp.float32, device=device)
        m.gripper_d_normal = wp.array([x.d_normal for x in g], dtype=wp.float32, device=device)
        m.gripper_f_normal_max = wp.array([x.f_normal_max for x in g], dtype=wp.float32, device=device)
        m.gripper_f_grip_max = wp.array([x.f_grip_max for x in g], dtype=wp.float32, device=device)
        m.gripper_k_shear_x = wp.array([x.k_shear_x for x in g], dtype=wp.float32, device=device)
        m.gripper_k_shear_y = wp.array([x.k_shear_y for x in g], dtype=wp.float32, device=device)
        m.gripper_mu_x = wp.array([x.mu_x for x in g], dtype=wp.float32, device=device)
        m.gripper_mu_y = wp.array([x.mu_y for x in g], dtype=wp.float32, device=device)
        m.gripper_d_peel_x = wp.array([x.d_peel_x for x in g], dtype=wp.float32, device=device)
        m.gripper_d_peel_y = wp.array([x.d_peel_y for x in g], dtype=wp.float32, device=device)
        m.gripper_shape = wp.array([x.shape for x in g], dtype=wp.int32, device=device)
        m.gripper_dim_a = wp.array([x.dim_a for x in g], dtype=wp.float32, device=device)
        m.gripper_dim_b = wp.array([x.dim_b for x in g], dtype=wp.float32, device=device)
        # geometry-derived force factors (see gripper.pdf), precomputed per gripper
        factors = [_pad_geometry_factors(x.shape, x.dim_a, x.dim_b) for x in g]
        # peel stiffness and torsional stiffness are constant per gripper (k * I/A), precompute
        m.gripper_k_peel_x = wp.array([x.k_normal * f[0] for x, f in zip(g, factors, strict=True)], dtype=wp.float32, device=device)
        m.gripper_k_peel_y = wp.array([x.k_normal * f[1] for x, f in zip(g, factors, strict=True)], dtype=wp.float32, device=device)
        m.gripper_k_torsion = wp.array(
            [x.k_shear_x * f[0] + x.k_shear_y * f[1] for x, f in zip(g, factors, strict=True)], dtype=wp.float32, device=device
        )
        m.gripper_peel_capacity_x = wp.array([f[2] for f in factors], dtype=wp.float32, device=device)
        m.gripper_peel_capacity_y = wp.array([f[3] for f in factors], dtype=wp.float32, device=device)
        m.gripper_tw_x = wp.array([f[4] for f in factors], dtype=wp.float32, device=device)
        m.gripper_tw_y = wp.array([f[5] for f in factors], dtype=wp.float32, device=device)
        # per-pad arrays: flatten each gripper's pads, recording the owning gripper id
        pad_gripper: list[int] = []
        pad_xform: list[wp.transform] = []
        for gi, x in enumerate(g):
            for p in x.pads:
                pad_gripper.append(gi)
                pad_xform.append(p)
        m.pad_gripper = wp.array(pad_gripper, dtype=wp.int32, device=device)
        m.pad_xform = wp.array(pad_xform, dtype=wp.transform, device=device)
        return m


class SurfaceGripperModel:
    """Finalized surface-gripper model (mirrors :class:`newton.Model`).

    Constant device arrays. ``gripper_*`` are indexed by gripper id; ``pad_*`` by
    pad id, with ``pad_gripper`` mapping each pad to its owning gripper.
    """

    def state(self) -> "SurfaceGripperState":
        """Allocate a fresh per-pad :class:`SurfaceGripperState` for this model."""
        s = SurfaceGripperState()
        n = self.pad_xform.shape[0]
        s.pad_shear_stick_point = wp.zeros(n, dtype=wp.vec2, device=self.pad_xform.device)
        s.pad_theta_anchor = wp.zeros(n, dtype=wp.float32, device=self.pad_xform.device)
        s.pad_break_metric = wp.zeros(n, dtype=wp.float32, device=self.pad_xform.device)
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


class SurfaceGripperState:
    """Mutable per-step gripper state (mirrors :class:`newton.State`). Per-pad arrays."""

    pad_shear_stick_point: wp.array  # shear stick point [m], per pad (wp.vec2)
    pad_theta_anchor: wp.array  # twist stick anchor [rad], per pad (float)
    pad_break_metric: wp.array  # brittle break envelope; > 1 => seal exceeded capacity (float)
    pad_engaged: wp.array  # per-pad engaged/disengaged flag (bool)
    pad_body_b: wp.array  # gripped body index, valid when engaged (int)
    pad_anchor_b: wp.array  # TBS: seal frame in the gripped body's frame, cached at engagement (wp.transform)


class SurfaceGripperControl:
    """Gripper control inputs (mirrors :class:`newton.Control`). Per-pad arrays."""

    pad_grip_control: wp.array  # per-pad grip command [0, 1]; f_min = pad_grip_control * f_grip_max


def evaluate_seal() -> bool:
    """Return whether the suction cup has formed a seal."""
    return False


@wp.kernel
def latch_engagement_kernel(
    engaged: wp.array[wp.bool],  # fresh seal decision (from the seal logic)
    body_b: wp.array[int],  # body each pad seals against this step
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    pad_engaged: wp.array[wp.bool],  # stored state, updated in place
    pad_body_b: wp.array[int],
    pad_anchor_b: wp.array[wp.transform],
):
    pad = wp.tid()
    if engaged[pad] and not pad_engaged[pad]:
        # disengaged -> engaged: cache TBS = TB(t0)^-1 * TA(t0) * TAS  (seal frame in body B)
        g = pad_gripper[pad]
        b = body_b[pad]
        t_as = gripper_xform[g] * pad_xform[pad]  # TAS: seal frame in body A
        pad_anchor_b[pad] = wp.transform_inverse(body_q[b]) * (body_q[gripper_body_id[g]] * t_as)
    pad_body_b[pad] = body_b[pad]
    pad_engaged[pad] = engaged[pad]


def latch_engagement(
    state,
    gripper_model: SurfaceGripperModel,
    gripper_state: SurfaceGripperState,
    engaged,
    body_b,
) -> None:
    """Latch ``pad_anchor_b`` for pads that just engaged, then commit the seal state.

    ``engaged`` / ``body_b`` are this step's fresh seal decision. On a disengaged ->
    engaged transition, TBS (the seal frame in body B's frame) is cached into
    ``pad_anchor_b``; ``pad_engaged`` / ``pad_body_b`` are then updated to the decision.
    """
    n_pads = gripper_model.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        latch_engagement_kernel,
        dim=n_pads,
        inputs=[
            engaged,
            body_b,
            gripper_model.gripper_body_id,
            gripper_model.gripper_xform,
            gripper_model.pad_gripper,
            gripper_model.pad_xform,
            state.body_q,
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
def eval_normal_force(
    grip_control: float,
    f_grip_max: float,
    k_normal: float,
    d_normal: float,
    f_normal_max: float,
    pz: float,
    vz: float,
) -> tuple[float, float]:
    """Normal (z) suction force: controllable preload + spring-damper, clamped tension-only.

    See gripper.pdf, Normal Force: ``F_normal = clamp(f_min + k_normal*pz + d_normal*vz, 0,
    F_normal_max)`` with the controllable preload ``f_min = grip_control * f_grip_max`` (the
    commanded fraction of the max suction force dP*A).

    Args:
        grip_control: Grip command in [0, 1] (the control value).
        f_grip_max: Maximum suction (preload) force [N].
        k_normal: Normal stiffness [N/m].
        d_normal: Normal damping [N.s/m].
        f_normal_max: Maximum (break-threshold) normal force [N].
        pz: Normal separation of the seal frames since engagement [m].
        vz: Normal relative velocity of the seal point [m/s].

    Returns:
        ``(fz, fz_unclamped)``: the applied normal force [N] (clamped to ``[0, f_normal_max]``)
        and the raw unclamped value [N] -- the latter feeds the brittle break envelope.
    """
    f_min = grip_control * f_grip_max
    fz_unclamped = f_min + k_normal * pz + d_normal * vz
    fz = wp.clamp(fz_unclamped, 0.0, f_normal_max)
    return fz, fz_unclamped


@wp.func
def eval_shear_friction(
    px: float,
    py: float,
    stick_x: float,
    stick_y: float,
    k_shear_x: float,
    k_shear_y: float,
    mu_x: float,
    mu_y: float,
    fz: float,
) -> tuple[float, float, wp.vec2]:
    """Anisotropic Coulomb shear friction with stick-slip re-anchor.

    Builds the trial (elastic) shear force from the spring between the current offset
    ``(px, py)`` and the stick point ``(stick_x, stick_y)``, then clamps it onto the elliptical cone
    ``(fx/(mu_x*fz))^2 + (fy/(mu_y*fz))^2 <= 1``. Outside the cone the pad slips: the force is
    scaled back and the stick point slides to the offset that reproduces the clamped force.
    With no normal force there is no friction to hold, so the force is zero and the stick
    follows the current offset.

    Args:
        px: Current shear offset along seal x [m].
        py: Current shear offset along seal y [m].
        stick_x: Stick point along seal x [m].
        stick_y: Stick point along seal y [m].
        k_shear_x: Shear stiffness along seal x [N/m].
        k_shear_y: Shear stiffness along seal y [N/m].
        mu_x: Friction coefficient along seal x.
        mu_y: Friction coefficient along seal y.
        fz: Normal (holding) force [N].

    Returns:
        ``(fx, fy, stick)``: the friction-limited shear force [N] and updated stick point [m].
    """
    fx = k_shear_x * (px - stick_x)
    fy = k_shear_y * (py - stick_y)
    stick = wp.vec2(stick_x, stick_y)  # unchanged while sticking
    mux_n = mu_x * fz
    muy_n = mu_y * fz
    if mux_n > 0.0 and muy_n > 0.0:
        e = wp.sqrt((fx / mux_n) * (fx / mux_n) + (fy / muy_n) * (fy / muy_n))
        if e > 1.0:  # slip: scale onto the cone and re-anchor to the clamped force
            fx = fx / e
            fy = fy / e
            new_ax = px
            new_ay = py
            if k_shear_x > 0.0:
                new_ax = px - fx / k_shear_x
            if k_shear_y > 0.0:
                new_ay = py - fy / k_shear_y
            stick = wp.vec2(new_ax, new_ay)
    else:  # no normal force -> no friction to hold; anchor follows current offset
        fx = 0.0
        fy = 0.0
        stick = wp.vec2(px, py)
    return fx, fy, stick


@wp.func
def eval_peel_moment(
    k_peel_x: float,
    k_peel_y: float,
    d_peel_x: float,
    d_peel_y: float,
    theta_x: float,
    theta_y: float,
    omega_x: float,
    omega_y: float,
    m_peel_x_max: float,
    m_peel_y_max: float,
) -> tuple[float, float]:
    """Applied peel moments (spring + damper), capped onto the peel envelope.

    The restoring peel moment about each tilt axis is ``k_peel*theta + d_peel*omega``. The
    combined moment is then scaled onto the elliptical capacity envelope
    ``(M_x/M_x_max)^2 + (M_y/M_y_max)^2 <= 1`` (gripper.pdf, Peel Force) so the applied moment
    never exceeds capacity. Zero capacity (no holding force) supports no peel moment.

    Args:
        k_peel_x: Peel stiffness about x [N.m/rad].
        k_peel_y: Peel stiffness about y [N.m/rad].
        d_peel_x: Peel damping about x [N.m.s/rad].
        d_peel_y: Peel damping about y [N.m.s/rad].
        theta_x: Tilt angle about x [rad].
        theta_y: Tilt angle about y [rad].
        omega_x: Tilt rate about x [rad/s].
        omega_y: Tilt rate about y [rad/s].
        m_peel_x_max: Peel capacity about x, ``N_f * capacity_x`` [N.m].
        m_peel_y_max: Peel capacity about y, ``N_f * capacity_y`` [N.m].

    Returns:
        ``(m_peel_x, m_peel_y)``: the capped peel moments to apply [N.m].
    """
    m_peel_x = k_peel_x * theta_x + d_peel_x * omega_x
    m_peel_y = k_peel_y * theta_y + d_peel_y * omega_y
    rx = float(0.0)
    ry = float(0.0)
    if m_peel_x_max > 0.0:
        rx = m_peel_x / m_peel_x_max
    else:
        m_peel_x = 0.0
    if m_peel_y_max > 0.0:
        ry = m_peel_y / m_peel_y_max
    else:
        m_peel_y = 0.0
    scale = wp.sqrt(rx * rx + ry * ry)
    if scale > 1.0:
        m_peel_x = m_peel_x / scale
        m_peel_y = m_peel_y / scale
    return m_peel_x, m_peel_y


@wp.func
def eval_break_limit(
    fz_unclamped: float,
    f_normal_max: float,
    k_peel_x: float,
    k_peel_y: float,
    theta_x: float,
    theta_y: float,
    m_peel_x_max: float,
    m_peel_y_max: float,
) -> float:
    """Brittle break envelope of the seal (gripper.pdf, Break Forces).

    ``env = (F_normal/F_max)^2 + (M_peel_x/M_peel_x_max)^2 + (M_peel_y/M_peel_y_max)^2``; ``env
    > 1`` means the seal exceeded its brittle capacity. Only the brittle DOFs contribute --
    shear/twist yield (return-mapping) rather than break. The normal term uses the *unclamped*
    demand and only in tension; the peel terms use the *elastic* (spring-only) moments, since
    fracture is driven by elastic stress, not transient damping.

    Args:
        fz_unclamped: Unclamped normal force demand [N].
        f_normal_max: Maximum (break-threshold) normal force [N].
        k_peel_x: Peel stiffness about x [N.m/rad].
        k_peel_y: Peel stiffness about y [N.m/rad].
        theta_x: Tilt angle about x [rad].
        theta_y: Tilt angle about y [rad].
        m_peel_x_max: Peel capacity about x [N.m].
        m_peel_y_max: Peel capacity about y [N.m].

    Returns:
        The break envelope ``env`` (dimensionless); ``> 1`` signals a broken seal.
    """
    env = float(0.0)
    if f_normal_max > 0.0 and fz_unclamped > 0.0:  # only tension loads the seal
        rn = fz_unclamped / f_normal_max
        env += rn * rn
    mpx_elastic = k_peel_x * theta_x
    mpy_elastic = k_peel_y * theta_y
    if m_peel_x_max > 0.0:
        env += (mpx_elastic / m_peel_x_max) * (mpx_elastic / m_peel_x_max)
    if m_peel_y_max > 0.0:
        env += (mpy_elastic / m_peel_y_max) * (mpy_elastic / m_peel_y_max)
    return env


@wp.func
def eval_twist_friction(
    theta_z: float,
    twist_stick: float,
    k_torsion: float,
    m_twist_max: float,
) -> tuple[float, float]:
    """Torsional Coulomb friction about z, with stick-slip re-anchor (rotational analog of shear).

    The trial moment is ``k_torsion * (theta_z - twist_stick)``. If it exceeds the capacity
    ``m_twist_max`` the pad slips: the moment is clamped and the stick angle slides to the angle
    that reproduces the clamped moment (gripper.pdf, Torsional Force).

    Args:
        theta_z: Twist angle about z since engagement [rad].
        twist_stick: Twist stick angle [rad].
        k_torsion: Torsional stiffness [N.m/rad].
        m_twist_max: Torsional friction capacity [N.m].

    Returns:
        ``(m_twist, twist_stick)``: the applied torsional moment [N.m] and updated stick angle [rad].
    """
    m_twist = k_torsion * (theta_z - twist_stick)
    if wp.abs(m_twist) > m_twist_max:  # slip: clamp and re-anchor
        m_twist = wp.sign(m_twist) * m_twist_max
        if k_torsion > 0.0:
            twist_stick = theta_z - m_twist / k_torsion
    return m_twist, twist_stick


@wp.kernel
def eval_pad_force(
    gripper_body_id: wp.array[int],
    gripper_xform: wp.array[wp.transform],
    gripper_k_normal: wp.array[float],
    gripper_d_normal: wp.array[float],
    gripper_f_normal_max: wp.array[float],
    gripper_f_grip_max: wp.array[float],
    gripper_k_shear_x: wp.array[float],
    gripper_k_shear_y: wp.array[float],
    gripper_mu_x: wp.array[float],
    gripper_mu_y: wp.array[float],
    gripper_d_peel_x: wp.array[float],
    gripper_d_peel_y: wp.array[float],
    gripper_k_peel_x: wp.array[float],
    gripper_k_peel_y: wp.array[float],
    gripper_k_torsion: wp.array[float],
    gripper_peel_capacity_x: wp.array[float],
    gripper_peel_capacity_y: wp.array[float],
    gripper_tw_x: wp.array[float],
    gripper_tw_y: wp.array[float],
    pad_gripper: wp.array[int],
    pad_xform: wp.array[wp.transform],
    pad_grip_control: wp.array[float],
    pad_engaged: wp.array[wp.bool],
    pad_body_b: wp.array[int],
    pad_anchor_b: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    # outputs (mutated in place)
    pad_shear_stick_point: wp.array[wp.vec2],
    pad_theta_anchor: wp.array[float],
    pad_break_metric: wp.array[float],
    body_f: wp.array[wp.spatial_vector],
):
    pad = wp.tid()
    if not pad_engaged[pad]:
        return  # pad not engaged
    body_b = pad_body_b[pad]

    g = pad_gripper[pad]
    body_a = gripper_body_id[g]

    # world pose of body A's seal frame: TA(t) * TAS, with TAS = gripper_xform * pad_xform
    t_a_seal = body_q[body_a] * gripper_xform[g] * pad_xform[pad]
    q_a_seal = wp.transform_get_rotation(t_a_seal)
    p_a_seal = wp.transform_get_translation(t_a_seal)  # seal point on A (world)

    # matching seal frame on body B: TB(t)*TBS (TBS = pad_anchor_b, cached at engagement)
    t_b_seal = body_q[body_b] * pad_anchor_b[pad]
    p_b_seal = wp.transform_get_translation(t_b_seal)  # seal point on B (world)

    # accumulated separation since engagement, one value per DOF (in the seal frame):
    # shear px, py; normal pz; peel theta_x, theta_y; twist theta_z
    px, py, pz, theta_x, theta_y, theta_z = eval_pad_separation(t_a_seal, t_b_seal)

    # world-frame offset from each body's COM to the seal point. this is both the lever
    # arm for v_com + omega x r (velocity) and the moment arm for the wrench below.
    com_a = wp.transform_point(body_q[body_a], body_com[body_a])  # A's COM (world)
    com_b = wp.transform_point(body_q[body_b], body_com[body_b])  # B's COM (world)
    r_a = p_a_seal - com_a
    r_b = p_b_seal - com_b
    _vx, _vy, vz, omega_x, omega_y, _omega_z = eval_pad_relative_velocity(
        body_qd[body_a], body_qd[body_b], r_a, r_b, q_a_seal
    )

    # --- normal (z): controllable preload + spring-damper, clamped (tension only) ---
    f_normal_max = gripper_f_normal_max[g]
    fz, fz_unclamped = eval_normal_force(
        pad_grip_control[pad],
        gripper_f_grip_max[g],
        gripper_k_normal[g],
        gripper_d_normal[g],
        f_normal_max,
        pz,
        vz,
    )

    # --- shear (x, y): tangential spring from stick anchor, elliptical friction cone ---
    k_shear_x = gripper_k_shear_x[g]
    k_shear_y = gripper_k_shear_y[g]
    shear_stick = pad_shear_stick_point[pad]
    fx, fy, shear_stick = eval_shear_friction(
        px, py, shear_stick[0], shear_stick[1], k_shear_x, k_shear_y, gripper_mu_x[g], gripper_mu_y[g], fz
    )
    pad_shear_stick_point[pad] = shear_stick

    # --- peel (rotation about x, y): capped torsional spring-damper ---
    k_peel_x = gripper_k_peel_x[g]
    k_peel_y = gripper_k_peel_y[g]
    m_peel_x_max = fz * gripper_peel_capacity_x[g]  # capacity scales with the holding force
    m_peel_y_max = fz * gripper_peel_capacity_y[g]
    m_peel_x, m_peel_y = eval_peel_moment(
        k_peel_x,
        k_peel_y,
        gripper_d_peel_x[g],
        gripper_d_peel_y[g],
        theta_x,
        theta_y,
        omega_x,
        omega_y,
        m_peel_x_max,
        m_peel_y_max,
    )

    # --- twist (rotation about z): torsional friction from a stick anchor ---
    m_twist_max = fz * (gripper_mu_x[g] * gripper_tw_x[g] + gripper_mu_y[g] * gripper_tw_y[g])
    m_twist, twist_stick = eval_twist_friction(theta_z, pad_theta_anchor[pad], gripper_k_torsion[g], m_twist_max)
    pad_theta_anchor[pad] = twist_stick

    # assemble the seal-frame wrench and rotate into the world frame
    force = wp.quat_rotate(q_a_seal, wp.vec3(fx, fy, fz))
    torque = wp.quat_rotate(q_a_seal, wp.vec3(m_peel_x, m_peel_y, m_twist))

    # accumulate equal-and-opposite wrenches; r_a / r_b are the COM-to-seal arms (world)
    wp.atomic_add(body_f, body_a, wp.spatial_vector(force, torque + wp.cross(r_a, force)))
    wp.atomic_add(body_f, body_b, wp.spatial_vector(-force, -torque + wp.cross(r_b, -force)))

    # --- brittle break envelope: reported for the external disengage policy (see gripper.pdf) ---
    pad_break_metric[pad] = eval_break_limit(
        fz_unclamped,
        f_normal_max,
        k_peel_x,
        k_peel_y,
        theta_x,
        theta_y,
        m_peel_x_max,
        m_peel_y_max,
    )


def evaluate_gripper_force(
    model,
    state,
    gripper_model: SurfaceGripperModel,
    gripper_state: SurfaceGripperState,
    gripper_control: SurfaceGripperControl,
) -> None:
    """Accumulate the full per-pad suction wrench (all six DOF) into ``state.body_f``.

    Normal (preload + spring-damper, clamped tension-only), shear (Coulomb friction
    with an elliptical cone and stick anchor), peel (torsional spring-damper) and twist
    (torsional friction with a stick anchor). Uses the engagement recorded in
    ``gripper_state`` (``pad_engaged``, ``pad_body_b`` and the attach-time
    ``pad_anchor_b``) and mutates the stick anchors in place; the break/seal logic
    comes later.
    """
    n_pads = gripper_model.pad_xform.shape[0]
    if n_pads == 0:
        return
    wp.launch(
        eval_pad_force,
        dim=n_pads,
        inputs=[
            gripper_model.gripper_body_id,
            gripper_model.gripper_xform,
            gripper_model.gripper_k_normal,
            gripper_model.gripper_d_normal,
            gripper_model.gripper_f_normal_max,
            gripper_model.gripper_f_grip_max,
            gripper_model.gripper_k_shear_x,
            gripper_model.gripper_k_shear_y,
            gripper_model.gripper_mu_x,
            gripper_model.gripper_mu_y,
            gripper_model.gripper_d_peel_x,
            gripper_model.gripper_d_peel_y,
            gripper_model.gripper_k_peel_x,
            gripper_model.gripper_k_peel_y,
            gripper_model.gripper_k_torsion,
            gripper_model.gripper_peel_capacity_x,
            gripper_model.gripper_peel_capacity_y,
            gripper_model.gripper_tw_x,
            gripper_model.gripper_tw_y,
            gripper_model.pad_gripper,
            gripper_model.pad_xform,
            gripper_control.pad_grip_control,
            gripper_state.pad_engaged,
            gripper_state.pad_body_b,
            gripper_state.pad_anchor_b,
            model.body_com,
            state.body_q,
            state.body_qd,
            # outputs (mutated in place)
            gripper_state.pad_shear_stick_point,
            gripper_state.pad_theta_anchor,
            gripper_state.pad_break_metric,
            state.body_f,
        ],
    )
