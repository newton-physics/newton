# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Define Warp kernels and actuator classes for conveyor belt contact forces."""

import math

import warp as wp

from newton.examples.basic import conveyor_forces_utils

# not needed for the purpose of this sample
wp.config.enable_backward = False


VELOCITY_FIELD_TYPE_CONSTANT_VELOCITY = 0
VELOCITY_FIELD_TYPE_PIVOT = 1


@wp.func
def compute_axis_impulse(
    constraint_axis: wp.vec3,
    relative_velocity: wp.vec3,
    response_linear: wp.float32,
    inv_inertia_world: wp.mat33,
    center_of_mass_to_point: wp.vec3,
    mass_splitting_scale: wp.float32,
) -> wp.float32:
    """Compute the impulse along a specified constraint axis to get the relative velocity along that axis to zero.

    Args:
        constraint_axis: Constraint axis along which to apply an impulse to reach a zero relative velocity
            (must be normalized).
        relative_velocity: Delta world-space velocity (target minus current velocity) at the contact point.
            Will get projected onto the constraint axis.
        response_linear: Linear response coefficient (inverse mass of the body).
        inv_inertia_world: World-space inverse inertia tensor of the body.
        center_of_mass_to_point: Vector from the body's center of mass to the contact point (world space).
        mass_splitting_scale: Scale factor applied to the effective mass to distribute impulse across
            multiple contact points.

    Returns:
        Impulse (scalar) along the constraint axis to get relative velocity to zero.
    """
    delta_cross_constraint_axis = wp.cross(center_of_mass_to_point, constraint_axis)

    # angular response: (r x t)^T * I^-1 * (r x t)
    #
    # r: center_of_mass_to_point
    # t: constraint_axis

    response_angular = wp.dot(delta_cross_constraint_axis, wp.mul(inv_inertia_world, delta_cross_constraint_axis))

    response = response_linear + response_angular

    vel_multiplier = (1.0 / response) * mass_splitting_scale

    rel_vel_proj = wp.dot(constraint_axis, relative_velocity)

    axis_zero_vel_impulse = rel_vel_proj * vel_multiplier

    return axis_zero_vel_impulse


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
    """Compute the friction impulse at a single contact point needed to bring the point velocity toward ``target_vel``.

    The impulse is projected onto the plane orthogonal to ``normal`` and clamped by
    ``normal_impulse * friction_coefficient``.

    Args:
        normal: Contact surface normal (must be normalized).
        normal_impulse: Normal impulse magnitude at the contact point.
        current_vel: Current world-space velocity of the contact point on the rigid body.
        target_vel: Desired world-space velocity at the contact point.
        response_linear: Linear response coefficient (inverse mass of the body).
        inv_inertia_world: World-space inverse inertia tensor of the body.
        center_of_mass_to_point: Vector from the body's center of mass to the contact point (world space).
        friction_coefficient: Coulomb friction coefficient for this contact.
        mass_splitting_scale: Scale factor applied to the effective mass to distribute force across
            multiple contact points.

    Returns:
        Friction impulse vector (world space, tangential to the contact normal).
    """
    rel_vel = target_vel - current_vel

    basis_vectors = conveyor_forces_utils.compute_basis_vectors(normal)

    zero_err_impulse_0 = compute_axis_impulse(
        basis_vectors.v0,
        rel_vel,
        response_linear,
        inv_inertia_world,
        center_of_mass_to_point,
        mass_splitting_scale,
    )

    zero_err_impulse_1 = compute_axis_impulse(
        basis_vectors.v1,
        rel_vel,
        response_linear,
        inv_inertia_world,
        center_of_mass_to_point,
        mass_splitting_scale,
    )

    friction_impulse_max = normal_impulse * friction_coefficient

    zero_err_impulse_magn = wp.sqrt(
        (zero_err_impulse_0 * zero_err_impulse_0) + (zero_err_impulse_1 * zero_err_impulse_1)
    )

    impulse_magn = wp.min(friction_impulse_max, zero_err_impulse_magn)

    if zero_err_impulse_magn > 0.0:
        ratio = impulse_magn / zero_err_impulse_magn
    else:
        ratio = 0.0

    impulse = (basis_vectors.v0 * (zero_err_impulse_0 * ratio)) + (basis_vectors.v1 * (zero_err_impulse_1 * ratio))

    return impulse


@wp.func
def compute_point_force(
    dt: wp.float32,
    inverse_dt: wp.float32,
    body_to_world_transform: wp.transform,
    body_inverse_mass: wp.float32,
    body_inverse_inertia: wp.mat33,
    body_linear_velocity: wp.vec3,
    body_angular_velocity: wp.vec3,
    contact_position: wp.vec3,
    contact_normal: wp.vec3,
    contact_force: wp.float32,
    mass_splitting_scale: wp.float32,
    target_vel: wp.vec3,
    friction_coefficient: wp.float32,
) -> wp.spatial_vector:
    """Compute force and torque for a rigid body contact point.

    The force is clamped by friction using a simple Coulomb friction model:
    ``contact_force * friction_coefficient``.

    Args:
        dt: Simulation time-step in seconds.
        inverse_dt: Precomputed reciprocal of ``dt`` (1 / dt).
        body_to_world_transform: Center-of-mass-to-world transform of the rigid body.
        body_inverse_mass: Inverse mass of the rigid body.
        body_inverse_inertia: World-space inverse inertia tensor of the rigid body.
        body_linear_velocity: World-space linear velocity at the body's center of mass.
        body_angular_velocity: World-space angular velocity at the body's center of mass.
        contact_position: World-space position of the contact point.
        contact_normal: World-space contact normal (pointing towards the rigid body).
        contact_force: Normal force magnitude at the contact point.
        mass_splitting_scale: Scale factor to distribute force across multiple contact points.
        target_vel: Desired world-space velocity at the contact point.
        friction_coefficient: Coulomb friction coefficient for this contact.

    Returns:
        Spatial vector whose first three components are the linear force and last three are the
        torque, both expressed in world space and intended to be applied at the center of mass.
    """
    contact_impulse = contact_force * dt

    center_of_mass_to_point = contact_position - body_to_world_transform.p

    current_point_vel = body_linear_velocity + wp.cross(body_angular_velocity, center_of_mass_to_point)

    response_linear = body_inverse_mass

    tangential_impulse = compute_point_impulse(
        contact_normal,
        contact_impulse,
        current_point_vel,
        target_vel,
        response_linear,
        body_inverse_inertia,
        center_of_mass_to_point,
        friction_coefficient,
        mass_splitting_scale,
    )

    force = tangential_impulse * inverse_dt

    torque = wp.cross(center_of_mass_to_point, force)

    force_torque = wp.spatial_vector(force, torque)

    return force_torque


@wp.kernel
def velocity_field_compute_force(
    dt: wp.float32,
    parallel_contact_processing_count: wp.uint32,
    batch_size: wp.uint32,
    constant_velocity_field_target_velocity_buffer: wp.array[wp.vec3],
    pivot_velocity_field_pivot_point_buffer: wp.array[wp.vec3],
    pivot_velocity_field_angular_velocity_buffer: wp.array[wp.vec3],
    body_to_world_transform_buffer: wp.array[wp.transform],
    body_inverse_mass_buffer: wp.indexedarray2d(dtype=wp.float32),
    body_inverse_inertia_buffer: wp.array[wp.mat33],
    body_linear_velocities: wp.indexedarray2d(dtype=wp.float32),
    body_angular_velocities: wp.indexedarray2d(dtype=wp.float32),
    contact_point_buffer: wp.array2d[wp.float32],
    contact_normal_buffer: wp.array2d[wp.float32],
    contact_force_buffer: wp.array2d[wp.float32],
    point_to_indices_map: wp.array2d[wp.uint32],
    mass_splitting_scale_buffer: wp.array[wp.float32],
    friction_coefficient_buffer: wp.array[wp.float32],
    total_contact_count_single_element_array: wp.array[wp.uint32],
    global_velocity_scale: wp.array[wp.float32],
    # output
    per_point_force_torque_buffer: wp.array[wp.spatial_vector],
):
    """Compute the conveyor-belt friction force/torque at every active contact point.

    Each thread processes a contiguous batch of contact points, computing the desired
    target velocity based on the assigned velocity field and then using the target velocity to
    derive the force necessary to reach that velocity. A simple Coulomb friction model is
    used to clamp the resulting forces. Since the contact points are processed independently
    of each other, a mass splitting approach is taken to assign a fraction of the total
    rigid body mass to each point when computing the forces.

    Args:
        dt: Simulation time-step in seconds.
        parallel_contact_processing_count: Total number of threads launched; used as the stride
            when a thread loops over more than ``batch_size`` contact points.
        batch_size: Number of contact points each thread processes per loop iteration.
        constant_velocity_field_target_velocity_buffer: Per-field constant target velocity vectors.
        pivot_velocity_field_pivot_point_buffer: Per-field world-space pivot points for pivot fields.
        pivot_velocity_field_angular_velocity_buffer: Per-field angular velocity vectors for pivot fields.
        body_to_world_transform_buffer: Per-body center-of-mass-to-world transforms.
        body_inverse_mass_buffer: Per-body inverse mass values (indexed, shape (N, 1)).
        body_inverse_inertia_buffer: Per-body world-space inverse inertia tensors.
        body_linear_velocities: Per-body world-space linear velocities at the body's center of mass (indexed, shape (N, 3)).
        body_angular_velocities: Per-body world-space angular velocities at the body's center of mass (indexed, shape (N, 3)).
        contact_point_buffer: (C, 3) world-space positions of all contact points.
        contact_normal_buffer: (C, 3) contact normals for all contact points.
        contact_force_buffer: (C, 1) normal force per contact point.
        point_to_indices_map: (C, 3) per-contact mapping to [body_index, velocity_field_type, velocity_field_id].
        mass_splitting_scale_buffer: Per-contact mass-splitting scale factors (0 for filtered-out contacts).
        friction_coefficient_buffer: Per-contact Coulomb friction coefficients.
        total_contact_count_single_element_array: Single-element array holding the total active contact count.
        global_velocity_scale: Single-element array holding a global velocity scale to apply to the velocity field
            target velocities.
        per_point_force_torque_buffer: Output - spatial force/torque written for each contact point.
    """
    inverse_dt = 1.0 / dt

    start_index = wp.uint32(wp.tid()) * batch_size

    total_contact_count = total_contact_count_single_element_array[0]

    while start_index < total_contact_count:
        end_index_plus_one = start_index + batch_size

        i = start_index

        while (i < end_index_plus_one) and (i < total_contact_count):
            mass_splitting_scale = mass_splitting_scale_buffer[i]

            if mass_splitting_scale != 0.0:
                contact_position = wp.vec3(
                    contact_point_buffer[i, 0],
                    contact_point_buffer[i, 1],
                    contact_point_buffer[i, 2],
                )

                #
                # Compute the target velocity according to the assigned velocity field
                #

                velocity_field_type = point_to_indices_map[i, 1]
                velocity_field_index = point_to_indices_map[i, 2]

                friction_coefficient = friction_coefficient_buffer[i]

                if velocity_field_type == VELOCITY_FIELD_TYPE_CONSTANT_VELOCITY:
                    target_vel = constant_velocity_field_target_velocity_buffer[velocity_field_index]

                elif velocity_field_type == VELOCITY_FIELD_TYPE_PIVOT:
                    pivot_point = pivot_velocity_field_pivot_point_buffer[velocity_field_index]
                    angular_velocity = pivot_velocity_field_angular_velocity_buffer[velocity_field_index]

                    delta = contact_position - pivot_point
                    target_vel = wp.cross(angular_velocity, delta)

                else:
                    wp.printf(
                        "Error: unknown velocity field type %d at body %d\n",
                        velocity_field_type,
                        point_to_indices_map[i, 0],
                    )
                    mass_splitting_scale = 0.0
                    target_vel = wp.vec3(0.0)

                target_vel = target_vel * global_velocity_scale[0]

                #
                # Compute the force to reach the target velocity
                #

                body_index = point_to_indices_map[i, 0]

                force_torque = compute_point_force(
                    dt,
                    inverse_dt,
                    body_to_world_transform_buffer[body_index],
                    body_inverse_mass_buffer[body_index, 0],
                    body_inverse_inertia_buffer[body_index],
                    wp.vec3(
                        body_linear_velocities[body_index, 0],
                        body_linear_velocities[body_index, 1],
                        body_linear_velocities[body_index, 2],
                    ),
                    wp.vec3(
                        body_angular_velocities[body_index, 0],
                        body_angular_velocities[body_index, 1],
                        body_angular_velocities[body_index, 2],
                    ),
                    contact_position,
                    wp.vec3(
                        contact_normal_buffer[i, 0],
                        contact_normal_buffer[i, 1],
                        contact_normal_buffer[i, 2],
                    ),
                    contact_force_buffer[i, 0],
                    mass_splitting_scale,
                    target_vel,
                    friction_coefficient,
                )

            else:
                force_torque = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            per_point_force_torque_buffer[i] = force_torque

            i += wp.uint32(1)

        start_index += batch_size * parallel_contact_processing_count


class VelocityFieldActuator:
    """Velocity-field actuator for conveyor contact forces.

    The computed forces are clamped based on a simple Coulomb friction model (friction coefficient
    times normal force).
    """

    def __init__(
        self,
    ) -> None:

        #
        # constant velocity fields
        #
        self.constant_velocity_field_target_velocity_list = []

        self.constant_velocity_field_target_velocity_buffer = None

        #
        # pivot velocity fields
        #
        self.pivot_velocity_field_pivot_point_list = []
        self.pivot_velocity_field_angular_velocity_list = []

        self.pivot_velocity_field_pivot_point_buffer = None
        self.pivot_velocity_field_angular_velocity_buffer = None

    def add_constant_velocity_field(
        self,
        target_velocity: wp.vec3,
    ) -> int:
        """Register a constant velocity field and return its index.

        Args:
            target_velocity: Target velocity a point in this field should reach.

        Returns:
            Integer index of the newly registered velocity field instance.
        """
        index = len(self.constant_velocity_field_target_velocity_list)

        self.constant_velocity_field_target_velocity_list.append(target_velocity)

        return index

    def add_pivot_velocity_field(
        self,
        pivot_point: wp.vec3,
        angular_velocity: wp.vec3,
    ) -> int:
        """Register a pivot (rotational) velocity field and return its index.

        Args:
            pivot_point: World-space point to rotate around.
            angular_velocity: Target angular velocity to rotate around the pivot point
                (direction is the rotation axis, magnitude is the angular speed).

        Returns:
            Integer index of the newly registered velocity field instance.
        """
        index = len(self.pivot_velocity_field_pivot_point_list)

        self.pivot_velocity_field_pivot_point_list.append(pivot_point)
        self.pivot_velocity_field_angular_velocity_list.append(angular_velocity)

        return index

    def create_buffers(
        self,
        device: str | None = None,
    ) -> None:
        """Allocate Warp arrays for all registered constant and pivot velocity fields.

        Args:
            device: Warp device string. Uses the default device when ``None``.
        """
        #
        # constant velocity fields
        #
        self.constant_velocity_field_target_velocity_buffer = wp.array(
            self.constant_velocity_field_target_velocity_list,
            dtype=wp.vec3,
            device=device,
        )

        #
        # pivot velocity fields
        #
        self.pivot_velocity_field_pivot_point_buffer = wp.array(
            self.pivot_velocity_field_pivot_point_list,
            dtype=wp.vec3,
            device=device,
        )

        self.pivot_velocity_field_angular_velocity_buffer = wp.array(
            self.pivot_velocity_field_angular_velocity_list,
            dtype=wp.vec3,
            device=device,
        )

    def step(
        self,
        dt: float,
        max_contact_count: int,
        body_to_world_transform_buffer: wp.array[wp.transform],
        body_inverse_mass_buffer: wp.indexedarray2d(dtype=wp.float32),
        body_inverse_inertia_buffer: wp.array[wp.mat33],
        body_linear_velocities: wp.indexedarray2d(dtype=wp.float32),
        body_angular_velocities: wp.indexedarray2d(dtype=wp.float32),
        contact_point_buffer: wp.array2d[wp.float32],
        contact_normal_buffer: wp.array2d[wp.float32],
        contact_force_buffer: wp.array2d[wp.float32],
        point_to_indices_map: wp.array2d[wp.uint32],
        mass_splitting_scale_buffer: wp.array[wp.float32],
        friction_coefficient_buffer: wp.array[wp.float32],
        total_contact_count: wp.array[wp.uint32],
        global_velocity_scale: wp.array[wp.float32],
        # output
        per_point_force_torque_buffer: wp.array[wp.spatial_vector],
        # input
        max_thread_count: int = 1000,
        batch_size: int = 5,
        device: str | None = None,
    ) -> None:
        """Compute per-contact forces for one simulation step.

        Args:
            dt: Simulation time-step in seconds.
            max_contact_count: Upper bound on the number of contact points (used to determine parallelism).
            body_to_world_transform_buffer: Per-body center-of-mass-to-world transforms.
            body_inverse_mass_buffer: Per-body inverse masses.
            body_inverse_inertia_buffer: Per-body world-space inverse inertia tensors.
            body_linear_velocities: Per-body world-space linear velocities at the body's center of mass.
            body_angular_velocities: Per-body world-space angular velocities at the body's center of mass.
            contact_point_buffer: (C, 3) world-space contact point positions.
            contact_normal_buffer: (C, 3) contact normals.
            contact_force_buffer: (C, 1) contact normal forces.
            point_to_indices_map: (C, 3) mapping from contact index to body index, velocity field type, and
                velocity field ID.
            mass_splitting_scale_buffer: Per-contact mass-splitting scale factors.
            friction_coefficient_buffer: Per-contact friction coefficients.
            total_contact_count: Single-element array holding the active contact count.
            global_velocity_scale: Single-element array holding a global velocity scale to apply to the velocity
                field target velocities.
            per_point_force_torque_buffer: Output buffer; receives the spatial force/torque for each contact.
            max_thread_count: Maximum number of parallel Warp threads to launch.
            batch_size: Number of contact points processed per thread per iteration.
            device: Warp device string. Uses the default device when ``None``.
        """
        parallel_contact_processing_count = min(max_thread_count, int(math.ceil(float(max_contact_count) / batch_size)))

        wp.launch(
            kernel=velocity_field_compute_force,
            dim=parallel_contact_processing_count,
            inputs=[
                dt,
                parallel_contact_processing_count,
                batch_size,
                self.constant_velocity_field_target_velocity_buffer,
                self.pivot_velocity_field_pivot_point_buffer,
                self.pivot_velocity_field_angular_velocity_buffer,
                body_to_world_transform_buffer,
                body_inverse_mass_buffer,
                body_inverse_inertia_buffer,
                body_linear_velocities,
                body_angular_velocities,
                contact_point_buffer,
                contact_normal_buffer,
                contact_force_buffer,
                point_to_indices_map,
                mass_splitting_scale_buffer,
                friction_coefficient_buffer,
                total_contact_count,
                global_velocity_scale,
            ],
            outputs=[
                per_point_force_torque_buffer,
            ],
            device=device,
        )
