# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Conveyor Forces
#
# A multi-belt conveyor circuit transports rigid boxes with a velocity field
# rather than belt friction: every belt is a static, frictionless surface with
# an attached constant or pivot velocity field. Each step the per-contact
# normal forces are read back from the solver and converted into Coulomb-limited
# tangential body forces, so boxes are carried around the loop, through the
# 180-degree turn, up the incline, across the differential pair, and back.
#
# Command: python -m newton.examples basic_conveyor_forces
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples

# Velocity field types: a constant world-space velocity, or a rotation about a pivot.
VELOCITY_FIELD_TYPE_CONSTANT = 0
VELOCITY_FIELD_TYPE_PIVOT = 1


# ---------------------------------------------------------------------------
# Friction force model: target velocity -> Coulomb-clamped tangential force/torque
# ---------------------------------------------------------------------------
@wp.struct
class Vec3Pair:
    """Two orthonormal vectors spanning the plane tangent to a contact normal."""

    v0: wp.vec3
    v1: wp.vec3


@wp.func
def compute_basis_vectors(direction: wp.vec3) -> Vec3Pair:
    """Two unit vectors orthogonal to normalized ``direction`` and to each other."""
    basis = Vec3Pair()
    if wp.abs(direction[1]) <= 0.9999:
        basis.v0 = wp.normalize(wp.vec3(direction[2], 0.0, -direction[0]))
        basis.v1 = wp.vec3(
            direction[1] * basis.v0[2],
            (direction[2] * basis.v0[0]) - (direction[0] * basis.v0[2]),
            -direction[1] * basis.v0[0],
        )
    else:
        basis.v0 = wp.vec3(1.0, 0.0, 0.0)
        basis.v1 = wp.normalize(wp.vec3(0.0, direction[2], -direction[1]))
    return basis


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

    r_cross_t0 = wp.cross(center_of_mass_to_point, basis.v0)
    r_cross_t1 = wp.cross(center_of_mass_to_point, basis.v1)
    k00 = response_linear + wp.dot(r_cross_t0, wp.mul(inv_inertia_world, r_cross_t0))
    k11 = response_linear + wp.dot(r_cross_t1, wp.mul(inv_inertia_world, r_cross_t1))
    k01 = wp.dot(r_cross_t0, wp.mul(inv_inertia_world, r_cross_t1))
    det = (k00 * k11) - (k01 * k01)

    i0 = wp.float32(0.0)
    i1 = wp.float32(0.0)
    if det > 0.0:
        v0 = wp.dot(basis.v0, rel_vel)
        v1 = wp.dot(basis.v1, rel_vel)
        i0 = ((k11 * v0) - (k01 * v1)) * mass_splitting_scale / det
        i1 = ((k00 * v1) - (k01 * v0)) * mass_splitting_scale / det

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
# Contact processing kernels
# ---------------------------------------------------------------------------
@wp.struct
class BeltContact:
    """One belt-to-body contact, reduced to what the force model needs."""

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
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    contact_force_vec: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_conveyor: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    conv_surface_normal: wp.array[wp.vec3],
    conv_threshold: wp.array[wp.float32],
    use_contact_offsets: wp.int32,
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
        if use_contact_offsets != 0:
            body_point_local += rigid_contact_offset1[i]
        normal_toward_body = normal
    elif conv1 >= 0 and conv0 < 0:
        conv = conv1
        body = shape_body[shape0]
        body_point_local = rigid_contact_point0[i]
        if use_contact_offsets != 0:
            body_point_local += rigid_contact_offset0[i]
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
def classify_belt_contacts(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    contact_force_vec: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_conveyor: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    conv_surface_normal: wp.array[wp.vec3],
    conv_threshold: wp.array[wp.float32],
    use_contact_offsets: wp.int32,
    # output
    belt_contacts: wp.array[BeltContact],
    body_contact_count: wp.array[wp.int32],
):
    """Classify belt contacts and count the accepted contacts per dynamic body."""
    i = wp.tid()
    c = identify_belt_contact(
        i,
        rigid_contact_count,
        rigid_contact_shape0,
        rigid_contact_shape1,
        rigid_contact_normal,
        rigid_contact_point0,
        rigid_contact_point1,
        rigid_contact_offset0,
        rigid_contact_offset1,
        contact_force_vec,
        shape_body,
        shape_conveyor,
        body_q,
        conv_surface_normal,
        conv_threshold,
        use_contact_offsets,
    )
    belt_contacts[i] = c
    if c.valid == 1:
        wp.atomic_add(body_contact_count, c.body, 1)


@wp.kernel
def accumulate_conveyor_forces(
    dt: wp.float32,
    belt_contacts: wp.array[BeltContact],
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
    """Accumulate Coulomb-limited belt-contact wrenches by body."""
    i = wp.tid()
    c = belt_contacts[i]
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
    """Add one spatial-vector array into another."""
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
        conveyor.snapshot_prev(solver)
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        conveyor.update(solver, contacts, state_1, dt)  # read forces, recompute wrench
    """

    def __init__(self, model: newton.Model, solver_type: str = "xpbd"):
        """Initialize a conveyor driver for a model and solver type."""
        if solver_type not in {"xpbd", "vbd", "mujoco"}:
            raise ValueError(f"Unsupported solver type: {solver_type!r}")

        self.model = model
        self.solver_type = solver_type
        self.device = model.device
        # MuJoCo reporting replaces Newton support points with geometry-surface points.
        self._use_contact_offsets = solver_type != "mujoco"
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
        """Validate and store one belt registration."""
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
        if self.solver_type != "vbd" and contacts.force is None:
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
        self.belt_contacts = wp.empty(contacts.rigid_contact_max, dtype=BeltContact, device=d)
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

    def snapshot_prev(self, solver) -> None:
        """Store the pre-step body poses used for contact-force reporting.

        Only VBD reports forces from a pose history, so this is a no-op for the other solvers.

        Args:
            solver: Solver that is about to advance the state.
        """
        if self.solver_type == "vbd":
            wp.copy(self.body_q_prev, solver.body_q_prev)

    def _report_contact_forces(self, solver, contacts, state_post, dt: float) -> None:
        """Copy solver-specific contact forces into a common linear-force buffer."""
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
            classify_belt_contacts,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                self.contact_force_vec,
                self.model.shape_body,
                self.shape_conveyor,
                state_post.body_q,
                self.conv_surface_normal,
                self.conv_threshold,
                int(self._use_contact_offsets),
            ],
            outputs=[self.belt_contacts, self.body_contact_count],
            device=self.device,
        )
        wp.launch(
            accumulate_conveyor_forces,
            dim=contacts.rigid_contact_max,
            inputs=[
                dt,
                self.belt_contacts,
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


# ---------------------------------------------------------------------------
# Example scene configuration
# ---------------------------------------------------------------------------
# A small positive collision margin smooths the belt-to-belt seam transitions. VBD's
# rigid-contact handling needs a larger margin than XPBD to keep bodies on the belts.
SOLVER_MARGIN = {"xpbd": 0.015, "vbd": 0.05, "mujoco": 0.015}
XPBD_ITERATIONS = 4
VBD_ITERATIONS = 15
# A frictional-to-normal impedance ratio below 1 softens the friction
# constraints so boxes slip through the tight 180 degree turn instead of
# wedging against the guide walls.
MUJOCO_IMPRATIO = 0.1
# Just above mujoco.mjMINMU, below which mujoco_warp warns about NaN-prone contacts.
MUJOCO_MIN_FRICTION = 1.1e-5

# Test thresholds distinguish active transport from stationary bodies while allowing
# temporary slip at belt seams and turns.
MIN_TRANSPORT_SPEED = 0.3  # [m/s]
MIN_TRAVEL = 0.5  # [m]

STARTUP_DURATION = 1.0  # belts ramp from rest to full speed over this time [s]
BELT_HALF_THICKNESS = 0.1  # [m]
GUARD_HALF_THICKNESS = 0.2  # [m]
GUARD_WALL_THICKNESS = 0.1  # radial thickness of the curved turn guards [m]
TURN_SEGMENTS = 48

# A contact only drives a body if its normal is within ~4.4 degrees of the belt's
# surface normal, so a box pressed against a guard wall is not carried sideways.
CONTACT_PROCESSING_THRESHOLD = 0.997

# Coulomb limit on the drive: the tangential force on a box never exceeds this
# times the contact normal force.
BELT_DRIVE_FRICTION = 0.5

BOX_FRICTION = 0.5
GUARD_FRICTION = 0.2

BELT_COLOR = (0.09, 0.09, 0.09)  # dark rubber
GUARD_COLOR = (0.66, 0.69, 0.74)  # brushed metal
BOX_COLOR = (0.72, 0.55, 0.35)  # cardboard

# Straight belts, each a static slab driven along its own +Y axis.
# (label, center [m], yaw [deg], tilt about the belt's X axis [deg], half extents [m], surface speed [m/s])
STRAIGHT_BELTS = (
    ("infeed", (0.0, 0.0, 0.5), 0.0, 0.0, (0.5, 5.05, BELT_HALF_THICKNESS), -2.0),
    ("incline", (-5.1, -3.0075, 0.6757), 0.0, 5.0, (0.5, 2.05, BELT_HALF_THICKNESS), 2.0),
    ("differential_fast", (-4.84, 4.0849, 0.8514), 0.0, 0.0, (0.25, 5.05, BELT_HALF_THICKNESS), 4.0),
    ("differential_slow", (-5.36, 4.0849, 0.8514), 0.0, 0.0, (0.25, 5.05, BELT_HALF_THICKNESS), 2.0),
    ("crossover", (-3.08, 10.1349, 0.8514), 90.0, 0.0, (1.0, 2.58, BELT_HALF_THICKNESS), -2.0),
    ("decline", (0.0, 8.0925, 0.6757), 0.0, 3.305, (0.5, 3.0475, BELT_HALF_THICKNESS), -2.0),
)

# The 180 degree turn, built from two annular sectors spinning about their own pivot.
# (label, pivot [m], start angle [deg], end angle [deg], (inner, outer) radius [m], angular speed [rad/s])
TURN_BELTS = (
    ("turn_outer", (-2.5, -5.1, 0.5), 270.0, 360.0, (2.0, 3.0), -0.8),
    ("turn_inner", (-2.6, -5.1, 0.5), 180.0, 270.0, (2.0, 3.0), -0.8),
)

# Static guard walls that keep boxes on the belts through the turns and the incline.
# (label, center [m], yaw [deg], tilt [deg], half extents [m])
GUARDS = (
    ("incline_guard", (-5.6, -3.0075, 0.6757), -15.0, 5.0, (0.1, 0.5, GUARD_HALF_THICKNESS)),
    ("crossover_guard", (-3.08, 11.2349, 0.8514), 0.0, 0.0, (2.53, 0.1, GUARD_HALF_THICKNESS)),
    ("decline_guard", (0.5, 9.3094, 0.746), -5.0, 3.305, (0.1, 1.5, GUARD_HALF_THICKNESS)),
)

# Transported boxes: (label, center [m], half extents [m], mass [kg])
BOXES = (
    ("box_0", (0.0, 0.0, 0.804), (0.2, 0.3, 0.2), 2.0),
    ("box_1", (-0.2, -1.0, 0.704), (0.1, 0.1, 0.1), 0.5),
    ("box_2", (0.1, -2.0, 0.804), (0.2, 0.3, 0.2), 2.0),
    ("box_3", (0.0, 4.0, 0.704), (0.25, 0.25, 0.1), 1.0),
    ("box_4", (-0.75, -6.85, 0.704), (0.1, 0.1, 0.1), 0.5),
    ("box_5", (-4.35, -6.85, 0.804), (0.2, 0.3, 0.2), 2.0),
    ("box_6", (-5.1, 4.0849, 1.1554), (0.2, 0.3, 0.2), 2.0),
    ("box_7", (-4.84, 4.5349, 1.0554), (0.1, 0.1, 0.1), 0.5),
    ("box_8", (-5.36, 3.6349, 1.0554), (0.1, 0.1, 0.1), 0.5),
    ("box_9", (-5.025, 1.3849, 1.1554), (0.2, 0.3, 0.2), 2.0),
    ("box_10", (-5.1, 7.5849, 1.0554), (0.25, 0.25, 0.1), 1.0),
)

# A two-link parcel hinged about Z, straddling the differential pair: the speed
# difference between the two belts folds it as it travels.
HINGED_PARCEL = (
    ("parcel_a", (-4.975, -0.7151, 1.0054), (0.125, 0.05, 0.05), 1.0),
    ("parcel_b", (-5.225, -0.7151, 1.0054), (0.125, 0.05, 0.05), 1.0),
)
HINGE_LIMIT = math.radians(45.0)
HINGE_LIMIT_KE = 100.0  # [N·m/rad]
HINGE_LIMIT_KD = 1.0  # [N·m·s/rad]


def belt_rotation(yaw_deg: float, tilt_deg: float) -> wp.quat:
    """Return the belt frame: yaw about world Z, then tilt about the belt's own X axis."""
    yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(yaw_deg))
    tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(tilt_deg))
    return yaw * tilt


def create_annular_sector_mesh(
    inner_radius: float,
    outer_radius: float,
    half_thickness: float,
    start_angle: float,
    end_angle: float,
    segments: int,
) -> newton.Mesh:
    """Create a closed annular sector prism about the Z axis, angles in degrees."""
    angles = np.radians(np.linspace(start_angle, end_angle, segments + 1, dtype=np.float32))
    cos_theta, sin_theta = np.cos(angles), np.sin(angles)
    n = segments + 1

    def ring(radius, z):
        return np.stack((radius * cos_theta, radius * sin_theta, np.full(n, z, dtype=np.float32)), axis=1)

    vertices = np.vstack(
        (
            ring(inner_radius, half_thickness),
            ring(outer_radius, half_thickness),
            ring(inner_radius, -half_thickness),
            ring(outer_radius, -half_thickness),
        )
    ).astype(np.float32)

    top_in, top_out, bot_in, bot_out = 0, n, 2 * n, 3 * n
    indices: list[int] = []
    for i in range(segments):
        j = i + 1
        # top (+Z), bottom (-Z), outer and inner walls
        indices.extend((top_in + i, top_out + i, top_out + j, top_in + i, top_out + j, top_in + j))
        indices.extend((bot_in + i, bot_in + j, bot_out + j, bot_in + i, bot_out + j, bot_out + i))
        indices.extend((bot_out + i, bot_out + j, top_out + j, bot_out + i, top_out + j, top_out + i))
        indices.extend((bot_in + i, top_in + i, top_in + j, bot_in + i, top_in + j, bot_in + j))
    # end caps
    e = segments
    indices.extend((top_in, bot_in, bot_out, top_in, bot_out, top_out))
    indices.extend((top_in + e, top_out + e, bot_out + e, top_in + e, bot_out + e, bot_in + e))

    return newton.Mesh(vertices=vertices, indices=np.asarray(indices, dtype=np.int32), compute_inertia=False)


class Example:
    """Conveyor circuit whose static belts carry rigid boxes with contact forces."""

    def __init__(self, viewer, args=None):
        self.solver_type = getattr(args, "solver", "xpbd") if args is not None else "xpbd"

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        straight_belts, turn_belts, self.tracked_bodies = self._build_scene(builder)

        transported = set(self.tracked_bodies)
        if self.solver_type == "mujoco":
            # The conveyor supplies tangential contact forces explicitly, so native
            # friction on transported bodies is reduced to MuJoCo's positive minimum.
            # The conveyor force remains bounded by BELT_DRIVE_FRICTION.
            for shape, body in enumerate(builder.shape_body):
                builder.shape_material_mu[shape] = max(builder.shape_material_mu[shape], MUJOCO_MIN_FRICTION)
                if body in transported:
                    builder.shape_material_mu[shape] = MUJOCO_MIN_FRICTION
                    builder.shape_material_mu_torsional[shape] = 0.0
                    builder.shape_material_mu_rolling[shape] = 0.0
        elif self.solver_type == "xpbd":
            # XPBD averages the two shape coefficients, so both sides of a belt
            # contact must be frictionless to leave tangential drive to the conveyor.
            for shape, body in enumerate(builder.shape_body):
                if body in transported:
                    builder.shape_material_mu[shape] = 0.0
                    builder.shape_material_mu_torsional[shape] = 0.0
                    builder.shape_material_mu_rolling[shape] = 0.0

        margin = SOLVER_MARGIN[self.solver_type]
        mesh_types = (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH, newton.GeoType.HFIELD)
        shape_type = builder.shape_type
        for i in range(len(builder.shape_margin)):
            if shape_type[i] not in mesh_types:
                builder.shape_margin[i] = max(builder.shape_margin[i], margin)

        builder.color()
        self.model = builder.finalize()

        # The conveyor consumes the per-contact normal force reported by the solver.
        self.model.request_contact_attributes("force")

        if self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                cone="elliptic",
                impratio=MUJOCO_IMPRATIO,
                use_mujoco_contacts=False,
                njmax=2000,
                nconmax=1000,
            )
        elif self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(
                self.model, iterations=VBD_ITERATIONS, rigid_body_contact_buffer_size=2048
            )
        else:
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=XPBD_ITERATIONS)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="explicit")
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.conveyor = ConveyorForceModel(self.model, solver_type=self.solver_type)
        for shape, velocity, surface_normal in straight_belts:
            self.conveyor.add_constant_belt(
                shape,
                velocity=velocity,
                surface_normal=surface_normal,
                friction=BELT_DRIVE_FRICTION,
                threshold=CONTACT_PROCESSING_THRESHOLD,
            )
        for shape, pivot_point, angular_velocity in turn_belts:
            self.conveyor.add_pivot_belt(
                shape,
                pivot_point=pivot_point,
                angular_velocity=angular_velocity,
                friction=BELT_DRIVE_FRICTION,
                threshold=CONTACT_PROCESSING_THRESHOLD,
            )
        self.conveyor.finalize(self.contacts)

        self.tracked_start_pos = self.state_0.body_q.numpy()[self.tracked_bodies, :3].copy()
        self.max_travel = np.zeros(len(self.tracked_bodies))

        self.viewer.set_model(self.model)
        pos, pitch, yaw = _look_at(eye=(2.0, -14.0, 12.0), target=(-2.0, -4.0, 0.5))
        self.viewer.set_camera(pos, pitch, yaw)

    def _build_scene(self, builder):
        """Build the conveyor circuit.

        Returns the belt registrations for :class:`ConveyorForceModel` — straight belts as
        ``(shape, velocity, surface_normal)`` and turns as ``(shape, pivot_point, angular_velocity)``
        — plus the body indices of the transported boxes.
        """
        # The explicit conveyor forces provide all belt traction.
        belt_cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, mu_torsional=0.0, mu_rolling=0.0)
        guard_cfg = newton.ModelBuilder.ShapeConfig(mu=GUARD_FRICTION)

        straight_belts = []
        for label, pos, yaw, tilt, half_extents, speed in STRAIGHT_BELTS:
            rot = belt_rotation(yaw, tilt)
            shape = builder.add_shape_box(
                body=-1,
                xform=wp.transform(p=wp.vec3(*pos), q=rot),
                hx=half_extents[0],
                hy=half_extents[1],
                hz=half_extents[2],
                cfg=belt_cfg,
                color=BELT_COLOR,
                label=label,
            )
            # The belt carries its surface along its own +Y axis; its normal is its own +Z.
            straight_belts.append(
                (shape, wp.quat_rotate(rot, wp.vec3(0.0, speed, 0.0)), wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0)))
            )

        turn_belts = []
        for label, pivot, start_angle, end_angle, radii, angular_speed in TURN_BELTS:
            inner_radius, outer_radius = radii
            xform = wp.transform(p=wp.vec3(*pivot), q=wp.quat_identity())
            shape = builder.add_shape_mesh(
                body=-1,
                xform=xform,
                mesh=create_annular_sector_mesh(
                    inner_radius, outer_radius, BELT_HALF_THICKNESS, start_angle, end_angle, TURN_SEGMENTS
                ),
                cfg=belt_cfg,
                color=BELT_COLOR,
                label=label,
            )
            builder.add_shape_mesh(
                body=-1,
                xform=xform,
                mesh=create_annular_sector_mesh(
                    outer_radius,
                    outer_radius + GUARD_WALL_THICKNESS,
                    GUARD_HALF_THICKNESS,
                    start_angle,
                    end_angle,
                    TURN_SEGMENTS,
                ),
                cfg=guard_cfg,
                color=GUARD_COLOR,
                label=f"{label}_guard",
            )
            turn_belts.append((shape, wp.vec3(*pivot), wp.vec3(0.0, 0.0, angular_speed)))

        for label, pos, yaw, tilt, half_extents in GUARDS:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(p=wp.vec3(*pos), q=belt_rotation(yaw, tilt)),
                hx=half_extents[0],
                hy=half_extents[1],
                hz=half_extents[2],
                cfg=guard_cfg,
                color=GUARD_COLOR,
                label=label,
            )

        bodies = []
        for label, pos, half_extents, mass in BOXES:
            body = self._add_box_body(builder, label, pos, half_extents, mass)
            builder.add_articulation([builder.add_joint_free(body)], label=label)
            bodies.append(body)

        (label_a, pos_a, half_a, mass_a), (label_b, pos_b, half_b, mass_b) = HINGED_PARCEL
        link_a = self._add_box_body(builder, label_a, pos_a, half_a, mass_a)
        link_b = self._add_box_body(builder, label_b, pos_b, half_b, mass_b)
        free = builder.add_joint_free(link_a)
        hinge = builder.add_joint_revolute(
            parent=link_a,
            child=link_b,
            parent_xform=wp.transform(p=wp.vec3(-half_a[0], 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(half_b[0], 0.0, 0.0), q=wp.quat_identity()),
            axis=newton.Axis.Z,
            limit_lower=-HINGE_LIMIT,
            limit_upper=HINGE_LIMIT,
            limit_ke=HINGE_LIMIT_KE,
            limit_kd=HINGE_LIMIT_KD,
            label="parcel_hinge",
        )
        builder.add_articulation([free, hinge], label="hinged_parcel")
        bodies += [link_a, link_b]

        return straight_belts, turn_belts, bodies

    @staticmethod
    def _add_box_body(builder, label, pos, half_extents, mass):
        """Add a transported box, sizing its density so the shape carries the authored mass."""
        hx, hy, hz = half_extents
        body = builder.add_link(xform=wp.transform(p=wp.vec3(*pos), q=wp.quat_identity()), label=label)
        cfg = newton.ModelBuilder.ShapeConfig(mu=BOX_FRICTION, density=mass / (8.0 * hx * hy * hz))
        builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg, color=BOX_COLOR, label=label)
        return body

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Add the wrench the conveyor computed from the previous step's contacts.
            self.conveyor.apply(self.state_0)
            self.conveyor.snapshot_prev(self.solver)

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.conveyor.update(self.solver, self.contacts, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Ramp the belts up from rest so the boxes are not kicked at t = 0.
        self.conveyor.set_speed_scale(min(1.0, self.sim_time / STARTUP_DURATION))
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Record each box's furthest displacement from its spawn pose."""
        # Track how far each box has ever been from its spawn, so test_final can tell a
        # box that is genuinely carried from one that never left the belt it started on.
        travel = np.linalg.norm(self.state_0.body_q.numpy()[self.tracked_bodies, :3] - self.tracked_start_pos, axis=1)
        np.maximum(self.max_travel, travel, out=self.max_travel)

    def test_final(self):
        """Verify every box stayed on the circuit and was actually carried by the belts."""
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(body_q)), "non-finite body pose"
        for b in self.tracked_bodies:
            z = float(body_q[b][2])
            assert z > -0.5, f"transported body {b} fell through the floor: z={z:.4f}"

        # The transport checks below are the point of this test, so a run too short to
        # reach full belt speed must fail rather than silently skip them.
        assert self.sim_time > STARTUP_DURATION + 0.5, (
            f"run too short to verify transport: {self.sim_time:.2f} s of simulation, "
            f"need more than {STARTUP_DURATION + 0.5:.2f} s"
        )

        # Pose validity alone does not establish transport, so require measurable displacement.
        stalled = int(np.argmin(self.max_travel))
        assert self.max_travel[stalled] > MIN_TRAVEL, (
            f"transported body {self.tracked_bodies[stalled]} was never carried off its start pose: "
            f"{self.max_travel[stalled]:.3f} m"
        )
        # Individual boxes briefly slow at belt seams and in the turn, so this is a
        # fleet average rather than a per-box floor.
        speed = np.linalg.norm(self.state_0.body_qd.numpy()[self.tracked_bodies, :3], axis=1)
        assert speed.mean() > MIN_TRANSPORT_SPEED, f"boxes are not moving with the belts: {speed.mean():.3f} m/s"


def _look_at(eye, target):
    """Return (pos, pitch_deg, yaw_deg) for a Z-up camera at ``eye`` looking at ``target``."""
    d = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    d /= np.linalg.norm(d)
    pitch = np.degrees(np.arcsin(d[2]))
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    return wp.vec3(*eye), float(pitch), float(yaw)


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--solver", type=str, choices=["xpbd", "vbd", "mujoco"], default="xpbd", help="Solver backend to use."
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
