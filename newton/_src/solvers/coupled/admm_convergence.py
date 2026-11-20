# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-side fixed-point checks for the coupled ADMM solver."""

import warp as wp


@wp.struct
class AdmmConvergenceGroup:
    """References to one ADMM row group used by convergence checks.

    Attributes:
        W: Interface weights, shape [capacity].
        u: Projected interface velocities [m/s or rad/s], shape [capacity].
        lambda_: Scaled dual variables, shape [capacity].
        Jv: Mechanics interface velocities [m/s or rad/s], shape [capacity].
        active_count: Populated contact count, shape [1], or null for static rows.
        offset: First row in the flattened convergence buffers.
        capacity: Number of allocated rows in this group.
        angular: Whether rows use angular velocity and torque tolerances.
        revolute: Whether to omit the free local x axis.
    """

    W: wp.array[float]
    u: wp.array[wp.vec3]
    lambda_: wp.array[wp.vec3]
    Jv: wp.array[wp.vec3]
    active_count: wp.array[int]
    offset: int
    capacity: int
    angular: bool
    revolute: bool


@wp.func
def _constrained(value: wp.vec3, revolute: bool) -> wp.vec3:
    """Omit the free hinge component for revolute angular rows."""
    if revolute:
        return wp.vec3(0.0, value[1], value[2])
    return value


@wp.func
def _finite(value: wp.vec3) -> bool:
    """Check whether every vector component is finite."""
    return wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])


@wp.func
def _norm(value: wp.vec3) -> float:
    """Compute a scaled Euclidean norm, returning infinity for nonfinite input."""
    if not _finite(value):
        return wp.inf
    # Scaling avoids squaring very small or large residuals in single precision.
    scale = wp.max(wp.abs(value[0]), wp.max(wp.abs(value[1]), wp.abs(value[2])))
    if scale == 0.0:
        return 0.0
    return scale * wp.length(value / scale)


@wp.kernel(enable_backward=False)
def snapshot_forces_kernel(
    groups: wp.array[AdmmConvergenceGroup],
    group_indices: wp.array[int],
    rho: float,
    force_used: wp.array[wp.vec3],
    failed: wp.array[int],
):
    """Snapshot coupling forces before mechanics integration and clear failure status.

    Launch over the total allocated row capacity. Unpopulated contact rows are
    skipped; nonfinite inputs produce an infinite snapshot to fail the next check.

    Args:
        groups: Descriptors referencing the current ADMM row buffers.
        group_indices: Group index for each flattened row.
        rho: ADMM penalty parameter.
        force_used: Output forces [N or N*m], one vector per flattened row.
        failed: Output failure flag, shape [1], cleared to zero.
    """
    tid = wp.tid()
    if tid == 0:
        failed[0] = 0
    group = groups[group_indices[tid]]
    row = tid - group.offset
    if group.active_count and row >= group.active_count[0]:
        return
    weight = group.W[row]
    if (
        not wp.isfinite(weight)
        or not _finite(group.lambda_[row])
        or not _finite(group.u[row])
        or not _finite(group.Jv[row])
    ):
        force_used[tid] = wp.vec3(wp.inf)
        return
    force_used[tid] = _constrained(
        weight * (group.lambda_[row] + rho * weight * (group.u[row] - group.Jv[row])), group.revolute
    )


@wp.kernel(enable_backward=False)
def check_convergence_kernel(
    groups: wp.array[AdmmConvergenceGroup],
    group_indices: wp.array[int],
    rho: float,
    linear_velocity_tolerance: float,
    angular_velocity_tolerance: float,
    force_tolerance: float,
    torque_tolerance: float,
    force_relative_tolerance: float,
    force_used: wp.array[wp.vec3],
    failed: wp.array[int],
):
    """Check velocity agreement and force change after projection and dual update.

    Launch over the total allocated row capacity. Every populated row must pass
    both tests. A failed test or invalid input sets the shared failure flag.

    Args:
        groups: Descriptors referencing the updated ADMM row buffers.
        group_indices: Group index for each flattened row.
        rho: ADMM penalty parameter.
        linear_velocity_tolerance: Absolute linear velocity tolerance [m/s].
        angular_velocity_tolerance: Absolute angular velocity tolerance [rad/s].
        force_tolerance: Absolute force change tolerance [N].
        torque_tolerance: Absolute torque change tolerance [N*m].
        force_relative_tolerance: Relative force and torque change tolerance.
        force_used: Forces before mechanics integration [N or N*m], one vector per row.
        failed: Failure flag, shape [1], initialized to zero by the snapshot kernel.
    """
    tid = wp.tid()
    group = groups[group_indices[tid]]
    row = tid - group.offset
    if group.active_count:
        count = group.active_count[0]
        if count < 0 or count > group.capacity:
            wp.atomic_max(failed, 0, 1)
            return
        if row >= count:
            return
    weight = group.W[row]
    if (
        not wp.isfinite(weight)
        or not _finite(group.lambda_[row])
        or not _finite(group.u[row])
        or not _finite(group.Jv[row])
    ):
        wp.atomic_max(failed, 0, 1)
        return
    velocity = _constrained(group.u[row] - group.Jv[row], group.revolute)
    force_next = _constrained(
        weight * (group.lambda_[row] + rho * weight * (group.u[row] - group.Jv[row])), group.revolute
    )
    previous = force_used[tid]
    velocity_limit = linear_velocity_tolerance
    force_limit = force_tolerance
    if group.angular:
        velocity_limit = angular_velocity_tolerance
        force_limit = torque_tolerance
    force_limit += force_relative_tolerance * wp.max(_norm(previous), _norm(force_next))
    velocity_error = _norm(velocity)
    force_error = _norm(force_next - previous)
    if (
        not wp.isfinite(velocity_error)
        or not wp.isfinite(force_error)
        or not wp.isfinite(force_limit)
        or velocity_error > velocity_limit
        or force_error > force_limit
    ):
        wp.atomic_max(failed, 0, 1)


@wp.func
def _proximal_factor(lump: float, augmented_mass: float) -> float:
    """Compute the ratio of added proximal mass to base mass.

    The base mass is ``augmented_mass - lump``. The same ratio applies to
    scalar inertia values. Return zero when the velocity shift is inactive
    and infinity for invalid inputs that must fail the convergence check.
    """
    if not wp.isfinite(lump) or lump < 0.0:
        return wp.inf
    if lump == 0.0:
        return 0.0
    if not wp.isfinite(augmented_mass):
        return wp.inf
    base_mass = augmented_mass - lump
    # The shift kernels leave v_n unchanged when the base mass is nonpositive.
    if base_mass <= 0.0:
        return 0.0
    return lump / base_mass


@wp.func
def _proximal_passes(delta_norm: float, factor: float, tolerance: float) -> bool:
    """Check the velocity change and its product with the proximal factor against the tolerance.

    A zero factor disables the check. Otherwise, both quantities must be
    finite and no greater than the tolerance.
    """
    if not wp.isfinite(factor) or factor < 0.0:
        return False
    if factor == 0.0:
        return True
    error = wp.max(1.0, factor) * delta_norm
    return wp.isfinite(error) and error <= tolerance


@wp.kernel(enable_backward=False)
def check_particle_proximal_convergence_kernel(
    previous: wp.array[wp.vec3],
    current: wp.array[wp.vec3],
    proximal_mass: wp.array[float],
    augmented_mass: wp.array[float],
    velocity_tolerance: float,
    failed: wp.array[int],
):
    """Flag particle velocity changes that fail the proximal convergence check.

    Args:
        previous: Velocities from the previous ADMM iteration [m/s], shape [particle_count].
        current: Velocities after the participant solve [m/s], shape [particle_count].
        proximal_mass: Added proximal masses [kg], shape [particle_count].
        augmented_mass: Base masses plus proximal masses [kg], shape [particle_count].
        velocity_tolerance: Absolute linear velocity tolerance [m/s].
        failed: Shared failure flag, shape [1], set to one when a check fails.
    """
    i = wp.tid()
    factor = _proximal_factor(proximal_mass[i], augmented_mass[i])
    if not _proximal_passes(_norm(current[i] - previous[i]), factor, velocity_tolerance):
        wp.atomic_max(failed, 0, 1)


@wp.kernel(enable_backward=False)
def check_body_proximal_convergence_kernel(
    previous: wp.array[wp.spatial_vector],
    current: wp.array[wp.spatial_vector],
    proximal_mass: wp.array[float],
    proximal_inertia: wp.array[float],
    augmented_mass: wp.array[float],
    augmented_inertia: wp.array[wp.mat33],
    linear_velocity_tolerance: float,
    angular_velocity_tolerance: float,
    failed: wp.array[int],
):
    """Check body velocity changes using separate linear and angular tolerances.

    Args:
        previous: Velocities from the previous ADMM iteration, linear [m/s] then
            angular [rad/s], shape [body_count].
        current: Velocities after the participant solve, linear [m/s] then
            angular [rad/s], shape [body_count].
        proximal_mass: Added proximal masses [kg], shape [body_count].
        proximal_inertia: Added scalar proximal inertias [kg*m^2], shape [body_count].
        augmented_mass: Base masses plus proximal masses [kg], shape [body_count].
        augmented_inertia: Base inertia tensors plus the proximal diagonal
            contribution [kg*m^2], shape [body_count].
        linear_velocity_tolerance: Absolute linear velocity tolerance [m/s].
        angular_velocity_tolerance: Absolute angular velocity tolerance [rad/s].
        failed: Shared failure flag, shape [1], set to one when a check fails.
    """
    i = wp.tid()
    delta = current[i] - previous[i]
    linear_factor = _proximal_factor(proximal_mass[i], augmented_mass[i])
    angular_factor = _proximal_factor(proximal_inertia[i], wp.trace(augmented_inertia[i]) / 3.0)
    if not _proximal_passes(
        _norm(wp.spatial_top(delta)), linear_factor, linear_velocity_tolerance
    ) or not _proximal_passes(_norm(wp.spatial_bottom(delta)), angular_factor, angular_velocity_tolerance):
        wp.atomic_max(failed, 0, 1)


@wp.kernel(enable_backward=False)
def check_joint_proximal_convergence_kernel(
    previous: wp.array[float],
    current: wp.array[float],
    proximal_factor: wp.array[float],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    linear_velocity_tolerance: float,
    angular_velocity_tolerance: float,
    failed: wp.array[int],
):
    """Check each joint velocity change using its linear or angular tolerance.

    Args:
        previous: Velocities from the previous ADMM iteration [m/s or rad/s],
            shape [joint_dof_count].
        current: Velocities after the participant solve [m/s or rad/s],
            shape [joint_dof_count].
        proximal_factor: Dimensionless proximal factors, shape [joint_dof_count].
        joint_qd_start: First velocity index for each joint, followed by the total
            velocity count, shape [joint_count + 1].
        joint_dof_dim: Linear and angular velocity counts per joint, shape [joint_count, 2].
        linear_velocity_tolerance: Absolute linear velocity tolerance [m/s].
        angular_velocity_tolerance: Absolute angular velocity tolerance [rad/s].
        failed: Shared failure flag, shape [1], set to one when a check fails.
    """
    joint = wp.tid()
    start = joint_qd_start[joint]
    linear_count = joint_dof_dim[joint, 0]
    count = linear_count + joint_dof_dim[joint, 1]
    for axis in range(count):
        dof = start + axis
        tolerance = linear_velocity_tolerance
        if axis >= linear_count:
            tolerance = angular_velocity_tolerance
        if not _proximal_passes(wp.abs(current[dof] - previous[dof]), proximal_factor[dof], tolerance):
            wp.atomic_max(failed, 0, 1)


@wp.kernel(enable_backward=False)
def update_convergence_status_kernel(
    iterations: int,
    failed: wp.array[int],
    iteration_count: wp.array[int],
    converged: wp.array[int],
    continuing: wp.array[int],
):
    """Publish a completed check and update the device continuation condition.

    Args:
        iterations: Number of completed ADMM iterations.
        failed: Reduced row failure flag (0 or 1), shape [1].
        iteration_count: Output completed iteration count, shape [1].
        converged: Output convergence flag (0 or 1), shape [1].
        continuing: Output continuation flag (0 or 1), shape [1].
    """
    iteration_count[0] = iterations
    converged[0] = 1 - failed[0]
    continuing[0] = failed[0]
