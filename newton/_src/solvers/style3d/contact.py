# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...geometry import ParticleFlags
from ...sim import Contacts, Model
from ...sim.contacts import contact_surface_separation


@wp.struct
class ContactTangent:
    """Cache one contact's fixed linearization for the PCG solve."""

    active: int
    body0: int
    body1: int
    weights0: wp.vec4
    weights1: wp.vec4
    friction_weights0: wp.vec4
    friction_weights1: wp.vec4
    normal_hessian: wp.mat33
    friction_hessian: wp.mat33


@wp.struct
class SoftContactTangent:
    """Cache the particle and affine Jacobian with its combined contact tangent."""

    active: int
    particle: int
    body: int
    weights: wp.vec4
    hessian: wp.mat33


@wp.func
def contact_response(
    gap: float,
    n: wp.vec3,
    vn: float,
    vt: wp.vec3,
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    dt: float,
) -> tuple[wp.vec3, wp.mat33, wp.vec3, wp.mat33]:
    """Return normal/friction gradients and PSD tangents for a penetrating contact."""
    normal_gradient = ke * gap + kd * wp.min(vn, 0.0)
    normal_stiffness = ke
    if vn < 0.0:
        normal_stiffness += kd / dt
    Hn = normal_stiffness * wp.outer(n, n)
    gn = normal_gradient * n
    speed = wp.length(vt)
    limit = mu * (-normal_gradient)
    gt = wp.vec3(0.0)
    Ht = wp.mat33(0.0)
    if limit > 0.0 and kf > 0.0:
        projector = wp.identity(3, float) - wp.outer(n, n)
        if kf * speed <= limit:
            gt = kf * vt
            Ht = (kf / dt) * projector
        else:
            gt = limit * (vt / speed)
            # Freeze the secant gain, retaining stiffness along the slip direction.
            # The exact sliding Hessian has zero curvature there: an unrestricted
            # Newton step can reverse slip, repeatedly overshoot, and inject energy.
            Ht = (limit / (dt * speed)) * projector
    return gn, Hn, gt, Ht


@wp.func
def contact_weights(body: int, point: wp.vec3, body_com: wp.array[wp.vec3]) -> wp.vec4:
    weights = wp.vec4(0.0)
    if body >= 0:
        local = point - body_com[body]
        weights = wp.vec4(1.0, local[0], local[1], local[2])
    return weights


@wp.func
def contact_point(body: int, weights: wp.vec4, static_point: wp.vec3, x: wp.array[wp.vec3]) -> wp.vec3:
    point = static_point
    if body >= 0:
        start = 4 * body
        point = x[start] + weights[1] * x[start + 1] + weights[2] * x[start + 2] + weights[3] * x[start + 3]
    return point


@wp.func
def current_contact_point(
    body: int,
    weights: wp.vec4,
    static_point: wp.vec3,
    x: wp.array[wp.vec3],
    x_inertia: wp.array[wp.vec3],
    solver_flags: wp.array[int],
) -> wp.vec3:
    # Prescribed bodies affect contacts through their inertial target, without solver increments.
    if body >= 0:
        if (solver_flags[4 * body] & ParticleFlags.ACTIVE) == 0:
            return contact_point(body, weights, static_point, x_inertia)
    return contact_point(body, weights, static_point, x)


@wp.kernel
def eval_contact_kernel(
    x: wp.array[wp.vec3],
    x_prev: wp.array[wp.vec3],
    x_inertia: wp.array[wp.vec3],
    solver_flags: wp.array[int],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_ke: wp.array[float],
    shape_kd: wp.array[float],
    shape_kf: wp.array[float],
    shape_mu: wp.array[float],
    contact_count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    offset0: wp.array[wp.vec3],
    offset1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    contact_ke: wp.array[float],
    contact_kd: wp.array[float],
    contact_mu: wp.array[float],
    surface_velocity: wp.array[wp.vec3],
    dt: float,
    # outputs
    rhs: wp.array[wp.vec3],
    diag: wp.array[wp.mat33],
    tangents: wp.array[ContactTangent],
):
    """Accumulate penalty contact forces and block diagonals, and cache the tangent.

    E_n = ke/2 min(gap, 0)^2. Closing-only damping adds kd/dt to its
    normal tangent. Friction is min(kf |v_t|, mu |f_n|), with a symmetric
    positive-semidefinite tangent obtained by holding the normal load and
    friction gain fixed. The sliding tangent majorizes the friction Hessian.
    Normal gaps use skeleton points and margins; friction uses the shifted
    surface anchors. Each point Jacobian consists of four scalar weights.
    """
    tid = wp.tid()
    tangent = ContactTangent()
    tangents[tid] = tangent
    if tid >= wp.min(contact_count[0], shape0.shape[0]):
        return
    s0 = shape0[tid]
    s1 = shape1[tid]
    if s0 < 0 or s1 < 0:
        return
    b0 = shape_body[s0]
    b1 = shape_body[s1]
    if b0 == b1:
        return

    w0 = contact_weights(b0, point0[tid], body_com)
    w1 = contact_weights(b1, point1[tid], body_com)
    p0 = current_contact_point(b0, w0, point0[tid], x, x_inertia, solver_flags)
    p1 = current_contact_point(b1, w1, point1[tid], x, x_inertia, solver_flags)
    n = normal[tid]
    gap = contact_surface_separation(p0, p1, n, margin0[tid], margin1[tid])
    if gap >= 0.0:
        return

    ke = wp.max(0.0, 0.5 * (shape_ke[s0] + shape_ke[s1]))
    kd = wp.max(0.0, 0.5 * (shape_kd[s0] + shape_kd[s1]))
    kf = wp.max(0.0, 0.5 * (shape_kf[s0] + shape_kf[s1]))
    mu = wp.sqrt(wp.max(0.0, shape_mu[s0] * shape_mu[s1]))
    if contact_ke:
        # Collision writers use zero for an unspecified override and store a friction scale.
        ke = contact_ke[tid] if contact_ke[tid] > 0.0 else ke
        kd = contact_kd[tid] if contact_kd[tid] > 0.0 else kd
        mu *= contact_mu[tid] if contact_mu[tid] > 0.0 else 1.0

    previous0 = contact_point(b0, w0, point0[tid], x_prev)
    previous1 = contact_point(b1, w1, point1[tid], x_prev)
    surface = wp.vec3(0.0)
    if surface_velocity:
        surface = surface_velocity[tid]
    vn = wp.dot(n, ((p1 - previous1) - (p0 - previous0)) / dt)
    anchor0 = point0[tid] + offset0[tid]
    anchor1 = point1[tid] + offset1[tid]
    t0 = contact_weights(b0, anchor0, body_com)
    t1 = contact_weights(b1, anchor1, body_com)
    current0 = current_contact_point(b0, t0, anchor0, x, x_inertia, solver_flags)
    current1 = current_contact_point(b1, t1, anchor1, x, x_inertia, solver_flags)
    previous0 = contact_point(b0, t0, anchor0, x_prev)
    previous1 = contact_point(b1, t1, anchor1, x_prev)
    velocity = ((current1 - previous1) - (current0 - previous0)) / dt + surface
    vt = velocity - wp.dot(n, velocity) * n
    gn, Hn, gt, Ht = contact_response(gap, n, vn, vt, ke, kd, kf, mu, dt)

    for side in range(2):
        body = b0
        weights = w0
        friction_weights = t0
        sign = float(1.0)
        if side == 1:
            body = b1
            weights = w1
            friction_weights = t1
            sign = -1.0
        if body >= 0:
            for i in range(4):
                row = 4 * body + i
                wp.atomic_add(rhs, row, sign * (weights[i] * gn + friction_weights[i] * gt))
                wp.atomic_add(diag, row, weights[i] * weights[i] * Hn + friction_weights[i] * friction_weights[i] * Ht)

    tangent.active = 1
    tangent.body0 = b0
    tangent.body1 = b1
    tangent.weights0 = w0
    tangent.weights1 = w1
    tangent.friction_weights0 = t0
    tangent.friction_weights1 = t1
    tangent.normal_hessian = Hn
    tangent.friction_hessian = Ht
    tangents[tid] = tangent


@wp.kernel
def contact_off_diag_multiply_kernel(
    x: wp.array[wp.vec3],
    tangents: wp.array[ContactTangent],
    # outputs
    Ax: wp.array[wp.vec3],
):
    """Add J^T H J x excluding the 3x3 diagonal blocks already assembled."""
    tangent = tangents[wp.tid()]
    if tangent.active == 0:
        return
    dn = contact_point(tangent.body1, tangent.weights1, wp.vec3(0.0), x) - contact_point(
        tangent.body0, tangent.weights0, wp.vec3(0.0), x
    )
    dt = contact_point(tangent.body1, tangent.friction_weights1, wp.vec3(0.0), x) - contact_point(
        tangent.body0, tangent.friction_weights0, wp.vec3(0.0), x
    )

    for side in range(2):
        body = tangent.body0
        weights = -tangent.weights0
        friction_weights = -tangent.friction_weights0
        if side == 1:
            body = tangent.body1
            weights = tangent.weights1
            friction_weights = tangent.friction_weights1
        if body >= 0:
            for i in range(4):
                row = 4 * body + i
                normal_product = weights[i] * (tangent.normal_hessian * (dn - weights[i] * x[row]))
                friction_product = friction_weights[i] * (
                    tangent.friction_hessian * (dt - friction_weights[i] * x[row])
                )
                wp.atomic_add(Ax, row, normal_product + friction_product)


@wp.kernel
def eval_soft_contact_kernel(
    x: wp.array[wp.vec3],
    x_prev: wp.array[wp.vec3],
    affine_x: wp.array[wp.vec3],
    affine_prev: wp.array[wp.vec3],
    affine_inertia: wp.array[wp.vec3],
    affine_flags: wp.array[int],
    body_com: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    shape_mu: wp.array[float],
    ke: float,
    kd: float,
    kf: float,
    mu: float,
    contact_count: wp.array[int],
    contact_particle: wp.array[int],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    particle_count: int,
    dt: float,
    # outputs
    rhs: wp.array[wp.vec3],
    diag: wp.array[wp.mat33],
    tangents: wp.array[SoftContactTangent],
):
    """Assemble both sides of particle-shape contact in the shared linear system.

    The normal points from the shape to the particle. The shape point is raw,
    so gap = n dot (particle - shape_point) - radius - shape_margin.
    J = [Id, -w_0 Id, ..., -w_3 Id] maps solver increments to relative motion.
    Use model.soft_contact_ke/kd/kf and geometrically mix mu with the shape.
    """
    tid = wp.tid()
    tangent = SoftContactTangent()
    tangents[tid] = tangent
    if tid >= wp.min(contact_count[0], contact_particle.shape[0]):
        return
    particle = contact_particle[tid]
    shape = contact_shape[tid]
    if particle < 0 or particle >= particle_count or shape < 0:
        return
    body = shape_body[shape]
    point = contact_body_pos[tid]
    weights = contact_weights(body, point, body_com)
    current = current_contact_point(body, weights, point, affine_x, affine_inertia, affine_flags)
    previous = contact_point(body, weights, point, affine_prev)
    n = contact_normal[tid]
    gap = wp.dot(n, x[particle] - current) - particle_radius[particle] - shape_margin[shape]
    if gap >= 0.0:
        return

    # Mesh surface velocity is body-local and excludes rigid-body motion.
    # Freeze this prescribed velocity in the contact tangent for this iteration.
    surface = contact_body_vel[tid]
    if body >= 0:
        start = 4 * body
        if (affine_flags[start] & ParticleFlags.ACTIVE) != 0:
            surface = (
                surface[0] * affine_x[start + 1] + surface[1] * affine_x[start + 2] + surface[2] * affine_x[start + 3]
            )
        else:
            surface = (
                surface[0] * affine_inertia[start + 1]
                + surface[1] * affine_inertia[start + 2]
                + surface[2] * affine_inertia[start + 3]
            )
    velocity = ((x[particle] - x_prev[particle]) - (current - previous)) / dt - surface
    vn = wp.dot(n, velocity)
    vt = velocity - vn * n
    gn, Hn, gt, Ht = contact_response(
        gap,
        n,
        vn,
        vt,
        wp.max(ke, 0.0),
        wp.max(kd, 0.0),
        wp.max(kf, 0.0),
        wp.sqrt(wp.max(0.0, mu * shape_mu[shape])),
        dt,
    )
    gradient = gn + gt
    hessian = Hn + Ht
    wp.atomic_sub(rhs, particle, gradient)
    wp.atomic_add(diag, particle, hessian)
    if body >= 0:
        for i in range(4):
            row = particle_count + 4 * body + i
            wp.atomic_add(rhs, row, weights[i] * gradient)
            wp.atomic_add(diag, row, weights[i] * weights[i] * hessian)

    tangent.active = 1
    tangent.particle = particle
    tangent.body = body
    tangent.weights = weights
    tangent.hessian = hessian
    tangents[tid] = tangent


@wp.kernel
def soft_contact_off_diag_multiply_kernel(
    x: wp.array[wp.vec3],
    affine_x: wp.array[wp.vec3],
    tangents: wp.array[SoftContactTangent],
    particle_count: int,
    # outputs
    Ax: wp.array[wp.vec3],
):
    """Add particle-body and intra-body coupling, excluding assembled diagonals."""
    tangent = tangents[wp.tid()]
    if tangent.active == 0 or tangent.body < 0:
        return
    body_dx = contact_point(tangent.body, tangent.weights, wp.vec3(0.0), affine_x)
    relative_dx = x[tangent.particle] - body_dx
    wp.atomic_sub(Ax, tangent.particle, tangent.hessian * body_dx)
    for i in range(4):
        row = particle_count + 4 * tangent.body + i
        weight = tangent.weights[i]
        wp.atomic_sub(Ax, row, weight * (tangent.hessian * (relative_dx + weight * x[row])))


class Contact:
    """Assemble contact forces and retain per-iteration matrix-free tangents.

    Handle rigid-rigid and particle-rigid contacts in [particles, affine bodies].
    Bodies outside the active solve contribute prescribed motion through their
    inertial targets. Particle-only solves retain the cloth collision path.
    """

    def __init__(self, model: Model, particle_count: int) -> None:
        self.model = model
        self.particle_count = particle_count
        self._tangents = wp.zeros(model.rigid_contact_max or 0, dtype=ContactTangent, device=model.device)
        self._soft_tangents = wp.zeros(0, dtype=SoftContactTangent, device=model.device)

    def eval_contacts(
        self,
        contacts: Contacts | None,
        x: wp.array[wp.vec3],
        x_prev: wp.array[wp.vec3],
        x_inertia: wp.array[wp.vec3],
        solver_flags: wp.array[int],
        rhs: wp.array[wp.vec3],
        diag: wp.array[wp.mat33],
        dt: float,
    ) -> None:
        """Accumulate forces and diagonal blocks after orthogonality evaluation."""
        if contacts is None:
            self._tangents.zero_()
            self._soft_tangents.zero_()
            return

        if contacts.rigid_contact_max > len(self._tangents):
            self._tangents = wp.zeros(contacts.rigid_contact_max, dtype=ContactTangent, device=self.model.device)

        offset = self.particle_count
        wp.launch(
            eval_contact_kernel,
            dim=len(self._tangents),
            inputs=[
                x[offset:],
                x_prev[offset:],
                x_inertia[offset:],
                solver_flags[offset:],
                self.model.body_com,
                self.model.shape_body,
                self.model.shape_material_ke,
                self.model.shape_material_kd,
                self.model.shape_material_kf,
                self.model.shape_material_mu,
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_stiffness,
                contacts.rigid_contact_damping,
                contacts.rigid_contact_friction,
                contacts.rigid_contact_surface_velocity,
                dt,
            ],
            outputs=[rhs[offset:], diag[offset:], self._tangents],
            device=self.model.device,
        )
        if self.particle_count:
            contacts._assert_particle_only_soft_contacts("SolverStyle3D")
            if contacts.soft_contact_max > len(self._soft_tangents):
                self._soft_tangents = wp.zeros(
                    contacts.soft_contact_max, dtype=SoftContactTangent, device=self.model.device
                )
            wp.launch(
                eval_soft_contact_kernel,
                dim=len(self._soft_tangents),
                inputs=[
                    x,
                    x_prev,
                    x[offset:],
                    x_prev[offset:],
                    x_inertia[offset:],
                    solver_flags[offset:],
                    self.model.body_com,
                    self.model.particle_radius,
                    self.model.shape_body,
                    self.model.shape_margin,
                    self.model.shape_material_mu,
                    self.model.soft_contact_ke,
                    self.model.soft_contact_kd,
                    self.model.soft_contact_kf,
                    self.model.soft_contact_mu,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    self.particle_count,
                    dt,
                ],
                outputs=[rhs, diag, self._soft_tangents],
                device=self.model.device,
            )

    def off_diag_multiply(self, x: wp.array[wp.vec3], Ax: wp.array[wp.vec3]) -> None:
        """Add the cached contact off-diagonal product to caller-owned storage."""
        wp.launch(
            contact_off_diag_multiply_kernel,
            dim=len(self._tangents),
            inputs=[x[self.particle_count :], self._tangents],
            outputs=[Ax[self.particle_count :]],
            device=self.model.device,
        )
        if self.particle_count:
            wp.launch(
                soft_contact_off_diag_multiply_kernel,
                dim=len(self._soft_tangents),
                inputs=[x, x[self.particle_count :], self._soft_tangents, self.particle_count],
                outputs=[Ax],
                device=self.model.device,
            )
