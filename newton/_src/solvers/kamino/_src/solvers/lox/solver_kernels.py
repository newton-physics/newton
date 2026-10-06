# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kernels of the LOX splitting iterations: initialization, dual updates, convergence
reductions, terminal status, and integrator inputs.
"""

import warp as wp

from ...core.math import compute_gyroscopic_torque
from ...core.types import mat66f, vec6f
from .system_kernels import apply_body_weight, compute_body_explicit_wrench, unpack_body_solution
from .types import LOXStatus, make_status

###
# Module interface
###

__all__ = [
    "_blend_accepted_twist",
    "_finalize_residual_iteration",
    "_initialize_bodies",
    "_initialize_fixed_iteration",
    "_initialize_iteration_residuals",
    "_initialize_world_convergence",
    "_prepare_projection",
    "_reduce_body_residuals",
    "_reduce_final_body_residuals",
    "_reset_dual_wrench",
    "_restore_dual_from_wrench",
    "_store_dual_wrench",
    "_update_splitting_dual",
    "_write_integrator_body_inputs",
    "_write_world_status",
    "atomic_max_nonnegative",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Functions
###


@wp.func_native("""
if (value > 0.0f) {
#if defined(__CUDA_ARCH__)
    atomicMax(reinterpret_cast<int*>(wp::address(values, index)), __float_as_int(value));
#else
    float* address = wp::address(values, index);
    if (value > *address)
        *address = value;
#endif
}
""")
def atomic_max_nonnegative(values: wp.array[wp.float32], index: wp.int32, value: wp.float32):
    """Raise the nonnegative ``values[index]`` to ``value``, ignoring NaN like :func:`warp.atomic_max`.

    Nonnegative floats order like their bit patterns, so one integer ``atomicMax`` replaces the
    compare-and-swap loop that emulates the float ``atomic_max``, which retries whenever another
    thread raises the same entry, e.g. the residual of a world shared by all its bodies or rows.
    """
    ...


@wp.func
def _reduce_physical_residuals(
    bid: wp.int32,
    wid: wp.int32,
    weight: wp.array[mat66f],
    previous: vec6f,
    current: vec6f,
    projected: vec6f,
    primal_residual: wp.array[wp.float32],
    dual_residual: wp.array[wp.float32],
):
    """Reduce the consensus gap ``|W (v - p)|`` and the change ``|v - v_prev|`` of one body into its world."""
    primal = apply_body_weight(weight[bid], current - projected)
    dual = current - previous
    primal_max = wp.float32(0.0)
    dual_max = wp.float32(0.0)
    for axis in range(6):
        primal_max = wp.max(primal_max, wp.abs(primal[axis]))
        dual_max = wp.max(dual_max, wp.abs(dual[axis]))
    atomic_max_nonnegative(primal_residual, wid, primal_max)
    atomic_max_nonnegative(dual_residual, wid, dual_max)


###
# Kernels
###


@wp.kernel
def _blend_accepted_twist(
    # Inputs:
    projected_fraction: wp.float32,
    projected_twist: wp.array[vec6f],
    # Outputs:
    global_twist: wp.array[vec6f],
):
    """Blend the projected twist into the smooth twist, ``v <- f p + (1 - f) v``, as the accepted twist."""
    bid = wp.tid()
    global_twist[bid] = projected_fraction * projected_twist[bid] + (1.0 - projected_fraction) * global_twist[bid]


@wp.kernel
def _initialize_bodies(
    # Inputs:
    body_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    body_velocity_begin: wp.array[vec6f],
    body_block: wp.array[wp.int32],
    model_bodies_m_i: wp.array[wp.float32],
    model_bodies_inv_m_i: wp.array[wp.float32],
    data_bodies_I_i: wp.array[wp.mat33f],
    data_bodies_inv_I_i: wp.array[wp.mat33f],
    data_bodies_w_e_i: wp.array[vec6f],
    data_bodies_w_a_i: wp.array[vec6f],
    model_gravity_vector: wp.array[wp.vec3f],
    model_time_dt: wp.array[wp.float32],
    inertial_fraction: wp.float32,
    dual_wrench: wp.array[vec6f],
    # Outputs:
    projected_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    global_twist_previous: wp.array[vec6f],
):
    """Seed the splitting iterates with the begin-step velocity advanced by a fraction of the predicted step.

    A unit fraction seeds the predicted velocity ``v + dt M^-1 (f + f_u)``, where ``f`` is the explicit
    wrench of the smooth system (external, actuation, gravity and gyroscopic terms) and ``f_u`` the
    warm-started wrench of the unilateral rows. The splitting dual is restored from the same ``f_u``, so
    both iterates agree, and the splitting converges in one iteration when ``f_u`` is unchanged.
    """
    bid = wp.tid()
    wid = body_world[bid]
    if world_mask and not world_mask[wid]:
        return
    velocity = body_velocity_begin[bid]
    value = velocity
    if body_block[bid] >= 0 and inertial_fraction > 0.0:
        wrench = compute_body_explicit_wrench(
            model_bodies_m_i[bid],
            data_bodies_I_i[bid],
            velocity,
            data_bodies_w_e_i[bid],
            data_bodies_w_a_i[bid],
            model_gravity_vector[wid],
            model_time_dt[wid],
        )
        if dual_wrench:
            wrench += dual_wrench[bid]
        linear_acceleration = model_bodies_inv_m_i[bid] * wp.vec3f(wrench[0], wrench[1], wrench[2])
        angular_acceleration = data_bodies_inv_I_i[bid] @ wp.vec3f(wrench[3], wrench[4], wrench[5])
        step = inertial_fraction * model_time_dt[wid]
        for axis in range(3):
            value[axis] += step * linear_acceleration[axis]
            value[axis + 3] += step * angular_acceleration[axis]
    projected_twist[bid] = value
    projected_twist_previous[bid] = value
    global_twist[bid] = value
    global_twist_previous[bid] = value


@wp.kernel
def _initialize_world_convergence(
    # Inputs:
    world_mask: wp.array[wp.bool],
    keep_failed: wp.bool,
    # Outputs:
    world_active: wp.array[wp.bool],
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
    residual_unilateral: wp.array[wp.float32],
    residual_lagged: wp.array[wp.float32],
    primal_residual: wp.array[wp.float32],
    dual_residual: wp.array[wp.float32],
    box_residual_max: wp.array[wp.float32],
    contact_residual_max: wp.array[wp.float32],
    structural_residual: wp.array[wp.float32],
    effort_residual: wp.array[wp.float32],
    solver_status: wp.array[LOXStatus],
):
    """Reset the per-world convergence state: iteration flags and count, residuals and reported status.

    With ``keep_failed``, a failed world stays failed and inactive until a reset clears it.
    """
    wid = wp.tid()
    if world_mask and not world_mask[wid]:
        return
    failed = keep_failed and world_failed[wid]
    world_active[wid] = not failed
    world_converged[wid] = False
    world_failed[wid] = failed
    iteration_count[wid] = 0
    residual_change[wid] = 0.0
    residual_split[wid] = 0.0
    residual_cross_iterate[wid] = 0.0
    residual_unilateral[wid] = 0.0
    residual_lagged[wid] = 0.0
    primal_residual[wid] = 0.0
    dual_residual[wid] = 0.0
    box_residual_max[wid] = 0.0
    contact_residual_max[wid] = 0.0
    structural_residual[wid] = 0.0
    if effort_residual:
        effort_residual[wid] = 0.0
    solver_status[wid] = make_status(False, 0, False)


@wp.kernel
def _restore_dual_from_wrench(
    # Inputs:
    body_world: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
    dt: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    dual_wrench: wp.array[vec6f],
    # Outputs:
    splitting_dual: wp.array[vec6f],
):
    """Convert the generalized wrench ``W u / dt`` back to the scaled splitting dual ``u``.

    The splitting dual of bodies without unilateral rows is zero.
    """
    bid = wp.tid()
    splitting_dual[bid] = vec6f(0.0)
    if body_has_unilateral[bid] != 0:
        splitting_dual[bid] = dt[body_world[bid]] * apply_body_weight(inverse_weight[bid], dual_wrench[bid])


@wp.kernel
def _prepare_projection(
    # Inputs:
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    body_vector_index: wp.array[wp.int32],
    packed_solution: wp.array[wp.float32],
    prescribed_twist: wp.array[vec6f],
    # Outputs:
    splitting_dual: wp.array[vec6f],
    global_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    bid = wp.tid()
    wid = body_world[bid]
    if not world_active[wid]:
        return
    global_solution = unpack_body_solution(bid, body_vector_index, packed_solution, prescribed_twist)
    global_twist_previous[bid] = global_twist[bid]
    global_twist[bid] = global_solution
    projected_twist_previous[bid] = projected_twist[bid]
    projected_twist[bid] = global_solution - splitting_dual[bid]


@wp.kernel
def _update_splitting_dual(
    # Inputs:
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    global_twist: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    # Outputs:
    splitting_dual: wp.array[vec6f],
):
    """Accumulate the consensus gap ``p - v`` into the scaled splitting dual ``u``."""
    bid = wp.tid()
    wid = body_world[bid]
    if world_active[wid] and not world_failed[wid]:
        splitting_dual[bid] += projected_twist[bid] - global_twist[bid]


@wp.kernel
def _initialize_iteration_residuals(
    # Inputs:
    world_failed: wp.array[wp.bool],
    # Outputs:
    world_active: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
    residual_unilateral: wp.array[wp.float32],
    residual_lagged: wp.array[wp.float32],
    box_residual_max: wp.array[wp.float32],
    contact_residual_max: wp.array[wp.float32],
    iteration_condition: wp.array[wp.int32],
):
    wid = wp.tid()
    # Cleared at the start of each iteration and set again by
    # _finalize_residual_iteration for the worlds that keep iterating.
    if iteration_condition and wid == 0:
        iteration_condition[0] = 0
    if not world_active[wid]:
        return
    iteration_count[wid] += 1
    residual_change[wid] = 0.0
    residual_split[wid] = 0.0
    residual_cross_iterate[wid] = 0.0
    residual_unilateral[wid] = 0.0
    residual_lagged[wid] = 0.0
    box_residual_max[wid] = 0.0
    contact_residual_max[wid] = 0.0
    if world_failed[wid]:
        world_active[wid] = False


@wp.kernel
def _reduce_body_residuals(
    # Inputs:
    model_time_dt: wp.array[wp.float32],
    position_tolerance: wp.float32,
    rotation_tolerance: wp.float32,
    velocity_tolerance: wp.float32,
    body_world: wp.array[wp.int32],
    global_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist_previous: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    world_active: wp.array[wp.bool],
    # Outputs:
    splitting_dual: wp.array[vec6f],
    world_failed: wp.array[wp.bool],
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
):
    """Advance the splitting dual, fail the worlds with non-finite iterates, and reduce the convergence
    residuals of the body twists.

    This fuses :func:`_update_splitting_dual` with the reduction, which reads the same twists.
    """
    bid = wp.tid()
    wid = body_world[bid]
    dt = model_time_dt[wid]
    if not world_active[wid]:
        return

    previous = global_twist_previous[bid]
    current = global_twist[bid]
    projected_previous = projected_twist_previous[bid]
    projected = projected_twist[bid]
    dual = splitting_dual[bid] + projected - current
    splitting_dual[bid] = dual
    # The previous twists were checked as the current ones of the last iteration
    if not (wp.isfinite(current) and wp.isfinite(projected) and wp.isfinite(dual)):
        world_failed[wid] = True
        return

    linear_change = wp.float32(0.0)
    angular_change = wp.float32(0.0)
    linear_split = wp.float32(0.0)
    angular_split = wp.float32(0.0)
    linear_cross_iterate = wp.float32(0.0)
    angular_cross_iterate = wp.float32(0.0)
    for axis in range(3):
        linear_change = wp.max(linear_change, wp.abs(current[axis] - previous[axis]))
        angular_change = wp.max(angular_change, wp.abs(current[axis + 3] - previous[axis + 3]))
        linear_split = wp.max(linear_split, wp.abs(current[axis] - projected[axis]))
        angular_split = wp.max(angular_split, wp.abs(current[axis + 3] - projected[axis + 3]))
        linear_cross_iterate = wp.max(
            linear_cross_iterate,
            wp.abs(current[axis] - projected_previous[axis]),
        )
        angular_cross_iterate = wp.max(
            angular_cross_iterate,
            wp.abs(current[axis + 3] - projected_previous[axis + 3]),
        )
    change = wp.max(
        dt * linear_change / position_tolerance,
        dt * angular_change / rotation_tolerance,
    )
    split = wp.max(linear_split, angular_split) / velocity_tolerance
    cross_iterate = wp.max(
        dt * linear_cross_iterate / position_tolerance,
        dt * angular_cross_iterate / rotation_tolerance,
    )
    atomic_max_nonnegative(residual_change, wid, change)
    atomic_max_nonnegative(residual_split, wid, split)
    atomic_max_nonnegative(residual_cross_iterate, wid, cross_iterate)


@wp.kernel
def _initialize_fixed_iteration(
    # Inputs:
    world_failed: wp.array[wp.bool],
    # Outputs:
    world_active: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
):
    wid = wp.tid()
    if not world_active[wid]:
        return
    iteration_count[wid] += 1
    if world_failed[wid]:
        world_active[wid] = False


@wp.kernel
def _reduce_final_body_residuals(
    # Inputs:
    body_world: wp.array[wp.int32],
    weight: wp.array[mat66f],
    global_twist_previous: wp.array[vec6f],
    global_twist: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
    splitting_dual: wp.array[vec6f],
    # Outputs:
    world_failed: wp.array[wp.bool],
    primal_residual: wp.array[wp.float32],
    dual_residual: wp.array[wp.float32],
):
    """Fail the worlds with non-finite twists and reduce the reported residuals of the others, after the iterations.

    The iterates of a world stop changing once it converges, so this matches its last iteration.
    """
    bid = wp.tid()
    wid = body_world[bid]
    if world_failed[wid]:
        return
    current = global_twist[bid]
    projected = projected_twist[bid]
    dual = splitting_dual[bid]
    finite = wp.isfinite(current) and wp.isfinite(projected) and wp.isfinite(dual)
    if finite:
        _reduce_physical_residuals(
            bid, wid, weight, global_twist_previous[bid], current, projected, primal_residual, dual_residual
        )
    else:
        world_failed[wid] = True


@wp.kernel
def _finalize_residual_iteration(
    # Inputs:
    residual_change: wp.array[wp.float32],
    residual_split: wp.array[wp.float32],
    residual_cross_iterate: wp.array[wp.float32],
    lagged_residual: wp.array[wp.float32],
    unilateral_residual: wp.array[wp.float32],
    iteration_count: wp.array[wp.int32],
    world_failed: wp.array[wp.bool],
    max_iterations: wp.int32,
    # Outputs:
    structural_residual: wp.array[wp.float32],
    effort_residual: wp.array[wp.float32],
    world_active: wp.array[wp.bool],
    world_converged: wp.array[wp.bool],
    iteration_condition: wp.array[wp.int32],
):
    """Deactivate the converged and failed worlds, and request another iteration for the others.

    The structural and effort residuals accumulate during the multiplier updates of an iteration;
    this check consumes and clears them.
    """
    wid = wp.tid()
    if not world_active[wid]:
        return
    # Failures of the projections, residuals, and body updates of this iteration
    if world_failed[wid]:
        world_active[wid] = False
        return

    change = residual_change[wid]
    split = residual_split[wid]
    cross_iterate = residual_cross_iterate[wid]
    structural = structural_residual[wid]
    structural_residual[wid] = 0.0
    total = wp.max(wp.max(change, split), wp.max(structural, cross_iterate))
    total = wp.max(total, lagged_residual[wid])
    if effort_residual:
        total = wp.max(total, effort_residual[wid])
        effort_residual[wid] = 0.0
    total = wp.max(total, unilateral_residual[wid])
    if total <= 1.0:
        world_active[wid] = False
        world_converged[wid] = True
    elif iteration_condition and iteration_count[wid] < max_iterations:
        iteration_condition[0] = 1


@wp.kernel
def _store_dual_wrench(
    # Inputs:
    body_world: wp.array[wp.int32],
    body_has_unilateral: wp.array[wp.int32],
    world_failed: wp.array[wp.bool],
    inv_dt: wp.array[wp.float32],
    weight: wp.array[mat66f],
    splitting_dual: wp.array[vec6f],
    # Outputs:
    dual_wrench: wp.array[vec6f],
):
    """Convert the scaled splitting dual ``u`` to the generalized wrench ``W u / dt``.

    Failed worlds store zero, so that their next time step starts from a zero dual.
    """
    bid = wp.tid()
    dual_wrench[bid] = vec6f(0.0)
    if body_has_unilateral[bid] != 0 and not world_failed[body_world[bid]]:
        dual_wrench[bid] = inv_dt[body_world[bid]] * apply_body_weight(weight[bid], splitting_dual[bid])


@wp.kernel
def _reset_dual_wrench(
    # Inputs:
    body_world: wp.array[wp.int32],
    world_mask: wp.array[wp.bool],
    # Outputs:
    dual_wrench: wp.array[vec6f],
):
    """Clear the warm-start wrench of the bodies of the masked worlds."""
    bid = wp.tid()
    if world_mask[body_world[bid]]:
        dual_wrench[bid] = vec6f(0.0)


@wp.kernel
def _write_world_status(
    # Inputs:
    world_converged: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    iteration_count: wp.array[wp.int32],
    primal_residual: wp.array[wp.float32],
    dual_residual: wp.array[wp.float32],
    box_residual_max: wp.array[wp.float32],
    contact_residual_max: wp.array[wp.float32],
    # Outputs:
    solver_status: wp.array[LOXStatus],
):
    wid = wp.tid()
    failed = world_failed[wid]
    status = make_status(world_converged[wid] and not failed, iteration_count[wid], failed)
    if not failed:
        status.r_p = primal_residual[wid]
        status.r_d = dual_residual[wid]
        # The squared metric-scaled natural-map residuals [sqrt(J)] measure the
        # unilateral feasibility and complementarity violation in energy units.
        unilateral_residual = wp.max(box_residual_max[wid], contact_residual_max[wid])
        status.r_c = unilateral_residual * unilateral_residual
    solver_status[wid] = status


@wp.kernel
def _write_integrator_body_inputs(
    # Inputs:
    body_vector_index: wp.array[wp.int32],
    model_bodies_wid: wp.array[wp.int32],
    world_failed: wp.array[wp.bool],
    model_time_dt: wp.array[wp.float32],
    model_bodies_m_i: wp.array[wp.float32],
    data_bodies_I_i: wp.array[wp.mat33f],
    model_bodies_inv_m_i: wp.array[wp.float32],
    model_bodies_inv_i_I_i: wp.array[wp.mat33f],
    model_gravity_vector: wp.array[wp.vec3f],
    velocity_begin: wp.array[vec6f],
    velocity_projected: wp.array[vec6f],
    # Outputs:
    data_bodies_w_i: wp.array[wp.spatial_vectorf],
    data_bodies_u_i: wp.array[wp.spatial_vectorf],
):
    """Encode the accepted LOX velocity as inputs to Kamino integration."""
    bid = wp.tid()
    wid = model_bodies_wid[bid]
    velocity_end = velocity_begin[bid]
    if body_vector_index[bid] >= 0 and not world_failed[wid]:
        velocity_end = velocity_projected[bid]
    linear_velocity_begin = wp.vec3f(
        velocity_begin[bid][0],
        velocity_begin[bid][1],
        velocity_begin[bid][2],
    )
    angular_velocity_begin = wp.vec3f(velocity_begin[bid][3], velocity_begin[bid][4], velocity_begin[bid][5])
    linear_velocity_end = wp.vec3f(velocity_end[0], velocity_end[1], velocity_end[2])
    angular_velocity_end = wp.vec3f(velocity_end[3], velocity_end[4], velocity_end[5])
    inv_dt = 1.0 / model_time_dt[wid]
    force = model_bodies_m_i[bid] * (inv_dt * (linear_velocity_end - linear_velocity_begin) - model_gravity_vector[wid])
    inertia = data_bodies_I_i[bid]
    # Cancel the gyroscopic torque that the Kamino integrators add to the input torque
    torque = inertia @ (inv_dt * (angular_velocity_end - angular_velocity_begin)) - compute_gyroscopic_torque(
        model_time_dt[wid], inertia, angular_velocity_begin
    )
    data_bodies_w_i[bid] = wp.spatial_vectorf(
        force[0],
        force[1],
        force[2],
        torque[0],
        torque[1],
        torque[2],
    )
    if body_vector_index[bid] >= 0 and (
        model_bodies_inv_m_i[bid] == 0.0 or wp.determinant(model_bodies_inv_i_I_i[bid]) == 0.0
    ):
        data_bodies_u_i[bid] = wp.spatial_vectorf(
            velocity_end[0],
            velocity_end[1],
            velocity_end[2],
            velocity_end[3],
            velocity_end[4],
            velocity_end[5],
        )
