# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Force-based conveyor model for Newton.

A conveyor belt is modeled as a *static* shape with an attached velocity field
that defines the desired surface velocity at every point. No geometry moves;
instead, after each solver step the model:

1. reads the per-contact points, normals, and normal forces reported by the solver,
2. keeps only the contacts between a registered belt and a resting body,
3. evaluates the belt's velocity field at each contact point to get a target velocity,
4. computes the tangential force that drives the contact-point velocity toward that
   target, clamped by a Coulomb friction limit (``friction * normal_force``), and
5. accumulates the resulting force and torque per body into ``State.body_f``.

Each contact uses the velocity field of its belt. Contact contributions use a per-body
``1 / contact_count`` mass-splitting factor.
"""

import warp as wp

import newton

# Velocity field types: a constant world-space velocity, or a rotation about a pivot.
VELOCITY_FIELD_TYPE_CONSTANT = 0
VELOCITY_FIELD_TYPE_PIVOT = 1


# ---------------------------------------------------------------------------
# Friction force model: target velocity -> Coulomb-clamped tangential force/torque
# ---------------------------------------------------------------------------
@wp.struct
class Vec3Pair:
    v0: wp.vec3
    v1: wp.vec3


@wp.func
def compute_basis_vectors(dir: wp.vec3) -> Vec3Pair:
    """Two unit vectors orthogonal to ``dir`` and to each other. ``dir`` must be normalized."""
    basis = Vec3Pair()
    if wp.abs(dir[1]) <= 0.9999:
        basis.v0 = wp.normalize(wp.vec3(dir[2], 0.0, -dir[0]))
        basis.v1 = wp.vec3(
            dir[1] * basis.v0[2],
            (dir[2] * basis.v0[0]) - (dir[0] * basis.v0[2]),
            -dir[1] * basis.v0[0],
        )
    else:
        basis.v0 = wp.vec3(1.0, 0.0, 0.0)
        basis.v1 = wp.normalize(wp.vec3(0.0, dir[2], -dir[1]))
    return basis


@wp.func
def compute_axis_impulse(
    constraint_axis: wp.vec3,
    relative_velocity: wp.vec3,
    response_linear: wp.float32,
    inv_inertia_world: wp.mat33,
    center_of_mass_to_point: wp.vec3,
    mass_splitting_scale: wp.float32,
) -> wp.float32:
    """Impulse along ``constraint_axis`` to null the relative velocity along it."""
    delta_cross_axis = wp.cross(center_of_mass_to_point, constraint_axis)
    response_angular = wp.dot(delta_cross_axis, wp.mul(inv_inertia_world, delta_cross_axis))
    response = response_linear + response_angular
    if response <= 0.0:
        return 0.0
    vel_multiplier = (1.0 / response) * mass_splitting_scale
    rel_vel_proj = wp.dot(constraint_axis, relative_velocity)
    return rel_vel_proj * vel_multiplier


@wp.func
def compute_point_impulse(
    normal: wp.vec3,
    normal_impulse: wp.float32,
    current_vel: wp.vec3,
    target_vel: wp.vec3,
    response_linear: wp.float32,
    inv_inertia_world: wp.mat33,
    center_of_mass_to_point: wp.vec3,
    friction_coefficient: wp.float32,
    mass_splitting_scale: wp.float32,
) -> wp.vec3:
    """Coulomb-clamped tangential impulse driving the contact-point velocity toward ``target_vel``."""
    rel_vel = target_vel - current_vel
    basis = compute_basis_vectors(normal)

    i0 = compute_axis_impulse(
        basis.v0, rel_vel, response_linear, inv_inertia_world, center_of_mass_to_point, mass_splitting_scale
    )
    i1 = compute_axis_impulse(
        basis.v1, rel_vel, response_linear, inv_inertia_world, center_of_mass_to_point, mass_splitting_scale
    )

    friction_impulse_max = normal_impulse * friction_coefficient
    zero_err_magn = wp.sqrt((i0 * i0) + (i1 * i1))
    impulse_magn = wp.min(friction_impulse_max, zero_err_magn)
    if zero_err_magn > 0.0:
        ratio = impulse_magn / zero_err_magn
    else:
        ratio = 0.0
    return (basis.v0 * (i0 * ratio)) + (basis.v1 * (i1 * ratio))


@wp.func
def compute_point_force(
    dt: wp.float32,
    inverse_dt: wp.float32,
    com_world: wp.vec3,
    body_inverse_mass: wp.float32,
    body_inverse_inertia_world: wp.mat33,
    body_linear_velocity: wp.vec3,
    body_angular_velocity: wp.vec3,
    contact_position: wp.vec3,
    contact_normal: wp.vec3,
    contact_force: wp.float32,
    mass_splitting_scale: wp.float32,
    target_vel: wp.vec3,
    friction_coefficient: wp.float32,
) -> wp.spatial_vector:
    """Friction-limited (force, torque about COM) for one contact point, world frame."""
    contact_impulse = contact_force * dt
    center_of_mass_to_point = contact_position - com_world
    current_point_vel = body_linear_velocity + wp.cross(body_angular_velocity, center_of_mass_to_point)

    tangential_impulse = compute_point_impulse(
        contact_normal,
        contact_impulse,
        current_point_vel,
        target_vel,
        body_inverse_mass,
        body_inverse_inertia_world,
        center_of_mass_to_point,
        friction_coefficient,
        mass_splitting_scale,
    )

    force = tangential_impulse * inverse_dt
    torque = wp.cross(center_of_mass_to_point, force)
    return wp.spatial_vector(force, torque)


# ---------------------------------------------------------------------------
# Newton coupling kernels
# ---------------------------------------------------------------------------
@wp.struct
class BeltContact:
    valid: wp.int32
    body: wp.int32
    conv: wp.int32
    point: wp.vec3
    normal: wp.vec3  # points from belt toward the body
    normal_force: wp.float32


@wp.func
def identify_belt_contact(
    i: int,
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    contact_force_vec: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_conveyor: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    conv_surface_normal: wp.array[wp.vec3],
    conv_threshold: wp.array[wp.float32],
) -> BeltContact:
    """Decide whether contact ``i`` is a belt/body contact and extract its inputs for the force model."""
    out = BeltContact()
    out.valid = 0
    if i >= rigid_contact_count[0]:
        return out

    shape0 = rigid_contact_shape0[i]
    shape1 = rigid_contact_shape1[i]
    if shape0 < 0 or shape1 < 0:
        return out

    conv0 = shape_conveyor[shape0]
    conv1 = shape_conveyor[shape1]

    # Exactly one side must be a belt; the other must be a dynamic body.
    normal = rigid_contact_normal[i]  # shape0 -> shape1
    if conv0 >= 0 and conv1 < 0:
        conv = conv0
        body = shape_body[shape1]
        body_point_local = rigid_contact_point1[i]
        normal_toward_body = normal
    elif conv1 >= 0 and conv0 < 0:
        conv = conv1
        body = shape_body[shape0]
        body_point_local = rigid_contact_point0[i]
        normal_toward_body = -normal
    else:
        return out

    if body < 0:
        return out

    # Reject contacts whose normal is not aligned with the belt surface normal.
    acceptance = wp.dot(normal_toward_body, conv_surface_normal[conv])
    if acceptance < conv_threshold[conv]:
        return out

    # Normal-force magnitude from the reported per-contact force vector.
    nforce = wp.abs(wp.dot(contact_force_vec[i], normal))
    if nforce <= 0.0:
        return out

    out.valid = 1
    out.body = body
    out.conv = conv
    out.point = wp.transform_point(body_q[body], body_point_local)
    out.normal = normal_toward_body
    out.normal_force = nforce
    return out


@wp.kernel
def count_belt_contacts(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    contact_force_vec: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_conveyor: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    conv_surface_normal: wp.array[wp.vec3],
    conv_threshold: wp.array[wp.float32],
    # output
    body_contact_count: wp.array[wp.int32],
):
    i = wp.tid()
    c = identify_belt_contact(
        i,
        rigid_contact_count,
        rigid_contact_shape0,
        rigid_contact_shape1,
        rigid_contact_normal,
        rigid_contact_point0,
        rigid_contact_point1,
        contact_force_vec,
        shape_body,
        shape_conveyor,
        body_q,
        conv_surface_normal,
        conv_threshold,
    )
    if c.valid == 1:
        wp.atomic_add(body_contact_count, c.body, 1)


@wp.kernel
def accumulate_conveyor_forces(
    dt: wp.float32,
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    contact_force_vec: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_conveyor: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia: wp.array[wp.mat33],
    body_contact_count: wp.array[wp.int32],
    conv_field_type: wp.array[wp.int32],
    conv_const_vel: wp.array[wp.vec3],
    conv_pivot_point: wp.array[wp.vec3],
    conv_pivot_angvel: wp.array[wp.vec3],
    conv_surface_normal: wp.array[wp.vec3],
    conv_threshold: wp.array[wp.float32],
    conv_friction: wp.array[wp.float32],
    global_velocity_scale: wp.array[wp.float32],
    # output
    conveyor_body_f: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    c = identify_belt_contact(
        i,
        rigid_contact_count,
        rigid_contact_shape0,
        rigid_contact_shape1,
        rigid_contact_normal,
        rigid_contact_point0,
        rigid_contact_point1,
        contact_force_vec,
        shape_body,
        shape_conveyor,
        body_q,
        conv_surface_normal,
        conv_threshold,
    )
    if c.valid == 0:
        return

    count = body_contact_count[c.body]
    if count <= 0:
        return
    mass_splitting_scale = 1.0 / float(count)

    # Target surface velocity from the belt's velocity field, evaluated at the contact point.
    conv = c.conv
    if conv_field_type[conv] == VELOCITY_FIELD_TYPE_CONSTANT:
        target_vel = conv_const_vel[conv]
    else:
        delta = c.point - conv_pivot_point[conv]
        target_vel = wp.cross(conv_pivot_angvel[conv], delta)
    target_vel = target_vel * global_velocity_scale[0]

    # World-space body quantities.
    q = body_q[c.body]
    com_world = wp.transform_point(q, body_com[c.body])
    r = wp.quat_to_matrix(wp.transform_get_rotation(q))
    inv_inertia_world = r * body_inv_inertia[c.body] * wp.transpose(r)
    qd = body_qd[c.body]
    lin_vel = wp.spatial_top(qd)
    ang_vel = wp.spatial_bottom(qd)

    ft = compute_point_force(
        dt,
        1.0 / dt,
        com_world,
        body_inv_mass[c.body],
        inv_inertia_world,
        lin_vel,
        ang_vel,
        c.point,
        c.normal,
        c.normal_force,
        mass_splitting_scale,
        target_vel,
        conv_friction[conv],
    )

    wp.atomic_add(conveyor_body_f, c.body, ft)


@wp.kernel
def add_spatial(dst: wp.array[wp.spatial_vector], src: wp.array[wp.spatial_vector]):
    i = wp.tid()
    dst[i] = dst[i] + src[i]


@wp.kernel
def extract_linear(spatial_force: wp.array[wp.spatial_vector], out_vec: wp.array[wp.vec3]):
    """Copy the linear part of a per-contact spatial wrench into a plain force vector."""
    i = wp.tid()
    out_vec[i] = wp.spatial_top(spatial_force[i])


class ConveyorForceModel:
    """Force-based conveyor driver for a Newton :class:`~newton.Model`.

    Register belts (a static shape + a velocity field) with :meth:`add_constant_belt` or
    :meth:`add_pivot_belt`, call :meth:`finalize`, then drive it each substep::

        conveyor.apply(state_0)  # add the conveyor wrench from the last step
        conveyor.snapshot_prev(state_0)
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        conveyor.update(solver, contacts, state_1, dt)  # read forces, recompute wrench
    """

    def __init__(self, model: newton.Model, solver_type: str = "xpbd"):
        if solver_type not in {"xpbd", "vbd", "mujoco"}:
            raise ValueError(f"Unsupported solver type: {solver_type!r}")

        self.model = model
        self.solver_type = solver_type
        self.device = model.device
        self._field_type: list[int] = []
        self._const_vel: list[wp.vec3] = []
        self._pivot_point: list[wp.vec3] = []
        self._pivot_angvel: list[wp.vec3] = []
        self._surface_normal: list[wp.vec3] = []
        self._threshold: list[float] = []
        self._friction: list[float] = []
        self._shape_conveyor = [-1] * model.shape_count
        self._finalized = False

    def _add_belt(
        self, shape_index, field_type, const_vel, pivot_point, pivot_angvel, surface_normal, friction, threshold
    ):
        if self._finalized:
            raise RuntimeError("Cannot register belts after finalize().")
        if shape_index < 0 or shape_index >= self.model.shape_count:
            raise IndexError(f"Belt shape index {shape_index} is out of range.")
        if self._shape_conveyor[shape_index] >= 0:
            raise ValueError(f"Shape {shape_index} is already registered as a belt.")
        if int(self.model.shape_body.numpy()[shape_index]) >= 0:
            raise ValueError("Conveyor belts must be static shapes.")
        if wp.length(surface_normal) == 0.0:
            raise ValueError("surface_normal must be nonzero.")
        if friction < 0.0:
            raise ValueError("friction must be nonnegative.")
        if threshold < -1.0 or threshold > 1.0:
            raise ValueError("threshold must be in [-1, 1].")

        conv = len(self._field_type)
        self._shape_conveyor[shape_index] = conv
        self._field_type.append(field_type)
        self._const_vel.append(const_vel)
        self._pivot_point.append(pivot_point)
        self._pivot_angvel.append(pivot_angvel)
        self._surface_normal.append(wp.normalize(surface_normal))
        self._friction.append(friction)
        self._threshold.append(threshold)
        return conv

    def add_constant_belt(
        self,
        shape_index: int,
        velocity: wp.vec3,
        surface_normal: wp.vec3 | None = None,
        friction: float = 0.5,
        threshold: float = 0.9,
    ) -> int:
        """Register a belt with a constant world-space surface velocity.

        Args:
            shape_index: Index of the static belt shape.
            velocity: Target surface velocity [m/s].
            surface_normal: Belt-facing unit normal in world space. If ``None``, uses +Z.
            friction: Coulomb friction coefficient used by the force model.
            threshold: Minimum dot product between the contact and belt normals.

        Returns:
            Index of the registered belt.
        """
        if surface_normal is None:
            surface_normal = wp.vec3(0.0, 0.0, 1.0)
        return self._add_belt(
            shape_index,
            VELOCITY_FIELD_TYPE_CONSTANT,
            velocity,
            wp.vec3(),
            wp.vec3(),
            surface_normal,
            friction,
            threshold,
        )

    def add_pivot_belt(
        self,
        shape_index: int,
        pivot_point: wp.vec3,
        angular_velocity: wp.vec3,
        surface_normal: wp.vec3 | None = None,
        friction: float = 0.5,
        threshold: float = 0.9,
    ) -> int:
        """Register a belt with a pivoting world-space surface velocity.

        Args:
            shape_index: Index of the static belt shape.
            pivot_point: World-space center of rotation [m].
            angular_velocity: World-space angular velocity [rad/s].
            surface_normal: Belt-facing unit normal in world space. If ``None``, uses +Z.
            friction: Coulomb friction coefficient used by the force model.
            threshold: Minimum dot product between the contact and belt normals.

        Returns:
            Index of the registered belt.
        """
        if surface_normal is None:
            surface_normal = wp.vec3(0.0, 0.0, 1.0)
        return self._add_belt(
            shape_index,
            VELOCITY_FIELD_TYPE_PIVOT,
            wp.vec3(),
            pivot_point,
            angular_velocity,
            surface_normal,
            friction,
            threshold,
        )

    def finalize(self, contacts) -> None:
        """Allocate device buffers. Call once after registering all belts.

        Args:
            contacts: The :class:`~newton.Contacts` the step loop will populate; used to size the
                per-contact force buffer.
        """
        if self._finalized:
            raise RuntimeError("ConveyorForceModel is already finalized.")
        if not self._field_type:
            raise RuntimeError("Register at least one belt before finalize().")
        if contacts.rigid_contact_max <= 0:
            raise ValueError("Contacts must have nonzero rigid-contact capacity.")
        if self.solver_type == "vbd":
            force_buffer = contacts.rigid_contact_force
        else:
            force_buffer = contacts.force
        if force_buffer is None:
            raise ValueError("Call model.request_contact_attributes('force') before creating Contacts.")

        d = self.device
        self.conv_field_type = wp.array(self._field_type, dtype=wp.int32, device=d)
        self.conv_const_vel = wp.array(self._const_vel, dtype=wp.vec3, device=d)
        self.conv_pivot_point = wp.array(self._pivot_point, dtype=wp.vec3, device=d)
        self.conv_pivot_angvel = wp.array(self._pivot_angvel, dtype=wp.vec3, device=d)
        self.conv_surface_normal = wp.array(self._surface_normal, dtype=wp.vec3, device=d)
        self.conv_threshold = wp.array(self._threshold, dtype=wp.float32, device=d)
        self.conv_friction = wp.array(self._friction, dtype=wp.float32, device=d)
        self.shape_conveyor = wp.array(self._shape_conveyor, dtype=wp.int32, device=d)
        self.global_velocity_scale = wp.array([1.0], dtype=wp.float32, device=d)

        self.body_contact_count = wp.zeros(self.model.body_count, dtype=wp.int32, device=d)
        self.conveyor_body_f = wp.zeros(self.model.body_count, dtype=wp.spatial_vector, device=d)
        self.contact_force_vec = wp.zeros(contacts.rigid_contact_max, dtype=wp.vec3, device=d)
        self.body_q_prev = wp.zeros(self.model.body_count, dtype=wp.transform, device=d)
        self._finalized = True

    def set_speed_scale(self, scale: float) -> None:
        """Set a global multiplier on all belt target velocities (e.g. a start-up ramp)."""
        self.global_velocity_scale.fill_(scale)

    def apply(self, state: newton.State) -> None:
        """Add the conveyor wrench computed by the previous :meth:`update` into ``state.body_f``."""
        wp.launch(
            add_spatial,
            dim=self.model.body_count,
            inputs=[state.body_f, self.conveyor_body_f],
            device=self.device,
        )

    def snapshot_prev(self, state: newton.State) -> None:
        """Store the pre-step body poses used for contact-force reporting."""
        wp.copy(self.body_q_prev, state.body_q)

    def _report_contact_forces(self, solver, contacts, state_post, dt: float) -> None:
        if self.solver_type == "vbd":
            solver.collect_rigid_contact_forces(state_post.body_q, self.body_q_prev, contacts, dt)
            wp.copy(self.contact_force_vec, contacts.rigid_contact_force)
        else:
            solver.update_contacts(contacts)
            wp.launch(
                extract_linear,
                dim=contacts.rigid_contact_max,
                inputs=[contacts.force, self.contact_force_vec],
                device=self.device,
            )

    def update(self, solver, contacts, state_post: newton.State, dt: float) -> None:
        """Read the solver's per-contact forces and recompute the per-body conveyor wrench."""
        self._report_contact_forces(solver, contacts, state_post, dt)
        self.conveyor_body_f.zero_()
        self.body_contact_count.zero_()
        wp.launch(
            count_belt_contacts,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                self.contact_force_vec,
                self.model.shape_body,
                self.shape_conveyor,
                state_post.body_q,
                self.conv_surface_normal,
                self.conv_threshold,
            ],
            outputs=[self.body_contact_count],
            device=self.device,
        )
        wp.launch(
            accumulate_conveyor_forces,
            dim=contacts.rigid_contact_max,
            inputs=[
                dt,
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                self.contact_force_vec,
                self.model.shape_body,
                self.shape_conveyor,
                state_post.body_q,
                state_post.body_qd,
                self.model.body_com,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
                self.body_contact_count,
                self.conv_field_type,
                self.conv_const_vel,
                self.conv_pivot_point,
                self.conv_pivot_angvel,
                self.conv_surface_normal,
                self.conv_threshold,
                self.conv_friction,
                self.global_velocity_scale,
            ],
            outputs=[self.conveyor_body_f],
            device=self.device,
        )
