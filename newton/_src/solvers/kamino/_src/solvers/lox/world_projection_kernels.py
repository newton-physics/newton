# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-world LOX projection, which runs every sweep of a world inside one thread block.

One launch warm starts and sweeps the rows of every world, with block barriers between the
sweeps. See :mod:`.projection_kernels`
for the row updates and :mod:`.projection` for the schedule that selects this path.
"""

from functools import cache

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from .anderson import (
    AndersonWorldState,
    finalize_anderson_world,
    mix_box_anderson_row,
    mix_contact_anderson_row,
    store_box_anderson_sample,
    store_contact_anderson_sample,
    store_contact_map_input,
)
from .apgd import (
    APGDRowState,
    apgd_momentum,
    box_restart_dot,
    contact_restart_dot,
    extrapolate_box_apgd_row,
    extrapolate_contact_apgd_row,
    initialize_box_apgd_row,
    initialize_contact_apgd_row,
)
from .contact import (
    angular_contact_reaction,
    clamp_contact_reaction,
    linear_contact_reaction,
    mat55f,
    pack_contact_reaction,
    pack_spatial_friction,
    scatter_spatial_impulse,
    vec5f,
)
from .projection_kernels import (
    _project_box_row,
    _project_contact_row,
    _project_spatial_contact_row,
    scatter_box_impulse,
    scatter_contact_impulse,
)
from .types import ProjectionAcceleration

###
# Module interface
###

__all__ = [
    "NoAccelerationState",
    "can_project_by_world",
    "make_project_by_world_kernel",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_WORLD_MIN_BLOCKS_PER_SM = 4


_WORLD_ROWS_PER_THREAD = 16
"""Largest number of rows per thread of a world block for which the sweeps run per world."""


###
# Types
###


@wp.struct
class NoAccelerationState:
    """Empty acceleration state of the unaccelerated per-world projection."""

    unused: wp.int32


###
# Functions
###


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _sync_threads(): ...


@wp.func
def _sweep_world_box_rows(
    wid: wp.int32,
    tid: wp.int32,
    color: wp.int32,
    world_box_offset: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_body_a: wp.array[wp.int32],
    box_body_b: wp.array[wp.int32],
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    box_bias: wp.array[wp.float32],
    box_lower: wp.array[wp.float32],
    box_upper: wp.array[wp.float32],
    box_delassus: wp.array[wp.float32],
    box_reaction: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    occupancy: wp.array2d[wp.int32],
    projected_twist: wp.array[vec6f],
    twist_delta: wp.array[vec6f],
):
    """Project every box row of one world with the threads of one block."""
    local = tid
    while local < world_box_count[wid]:
        _project_box_row(
            world_box_offset[wid] + local,
            color,
            box_body_a,
            box_body_b,
            box_jacobian_a,
            box_jacobian_b,
            box_bias,
            box_lower,
            box_upper,
            box_delassus,
            box_reaction,
            inverse_weight,
            occupancy,
            projected_twist,
            twist_delta,
        )
        local += wp.block_dim()


@wp.func
def _apply_world_delta(
    wid: wp.int32,
    tid: wp.int32,
    world_body_offset: wp.array[wp.int32],
    world_body_count: wp.array[wp.int32],
    world_failed: wp.array[wp.bool],
    twist_delta: wp.array[vec6f],
    projected_twist: wp.array[vec6f],
):
    """Apply the accumulated body deltas of one world between block barriers."""
    _sync_threads()
    valid = not world_failed[wid]
    local = tid
    while local < world_body_count[wid]:
        bid = world_body_offset[wid] + local
        if valid:
            projected_twist[bid] += twist_delta[bid]
        twist_delta[bid] = vec6f(0.0)
        local += wp.block_dim()
    _sync_threads()


###
# Kernels
###


@cache
def make_project_by_world_kernel(spatial: bool, colored: bool, acceleration: ProjectionAcceleration):
    """Build the kernel that projects one world per block.

    Args:
        spatial: Whether the contacts are spatial contacts with angular friction.
        colored: Whether the rows are swept by colored Gauss--Seidel; Jacobi otherwise.
        acceleration: Acceleration of the projection maps, whose state the kernel takes as one struct:
            :class:`.AndersonWorldState`, :class:`.APGDRowState`, or :class:`NoAccelerationState`.
    """
    anderson = acceleration == ProjectionAcceleration.ANDERSON
    apgd = acceleration == ProjectionAcceleration.APGD
    AccelerationState = AndersonWorldState if anderson else APGDRowState if apgd else NoAccelerationState

    @wp.func
    def project_contact_row(
        cid: wp.int32,
        wid: wp.int32,
        color: wp.int32,
        contact_body_a: wp.array[wp.int32],
        contact_body_b: wp.array[wp.int32],
        contact_jacobian_a: wp.array[mat36f],
        contact_jacobian_b: wp.array[mat36f],
        contact_bias: wp.array[wp.vec3f],
        contact_friction: wp.array[wp.float32],
        contact_delassus: wp.array[wp.mat33f],
        contact_compliance: wp.array[wp.float32],
        contact_restitution: wp.array[wp.vec4f],
        contact_frame: wp.array[wp.mat33f],
        angular_friction: wp.array[wp.vec2f],
        spatial_delassus: wp.array[mat66f],
        spatial_eigenvectors: wp.array[mat55f],
        spatial_eigenvalues: wp.array[vec5f],
        contact_reaction: wp.array[wp.vec3f],
        angular_reaction: wp.array[wp.vec3f],
        inverse_weight: wp.array[mat66f],
        occupancy: wp.array2d[wp.int32],
        projected_twist: wp.array[vec6f],
        twist_delta: wp.array[vec6f],
        world_failed: wp.array[wp.bool],
    ):
        if wp.static(spatial):
            _project_spatial_contact_row(
                cid,
                wid,
                color,
                contact_body_a,
                contact_body_b,
                contact_jacobian_a,
                contact_jacobian_b,
                contact_reaction,
                contact_frame,
                contact_friction,
                angular_friction,
                contact_bias,
                contact_compliance,
                contact_restitution,
                spatial_delassus,
                spatial_eigenvectors,
                spatial_eigenvalues,
                angular_reaction,
                inverse_weight,
                occupancy,
                projected_twist,
                twist_delta,
                world_failed,
            )
        else:
            _project_contact_row(
                cid,
                color,
                contact_body_a,
                contact_body_b,
                contact_jacobian_a,
                contact_jacobian_b,
                contact_bias,
                contact_friction,
                contact_delassus,
                contact_compliance,
                contact_restitution,
                contact_reaction,
                inverse_weight,
                occupancy,
                projected_twist,
                twist_delta,
            )

    @wp.func
    def sweep_world_contacts(
        wid: wp.int32,
        tid: wp.int32,
        color: wp.int32,
        world_contact_offset: wp.array[wp.int32],
        world_contact_count: wp.array[wp.int32],
        contact_body_a: wp.array[wp.int32],
        contact_body_b: wp.array[wp.int32],
        contact_jacobian_a: wp.array[mat36f],
        contact_jacobian_b: wp.array[mat36f],
        contact_bias: wp.array[wp.vec3f],
        contact_friction: wp.array[wp.float32],
        contact_delassus: wp.array[wp.mat33f],
        contact_compliance: wp.array[wp.float32],
        contact_restitution: wp.array[wp.vec4f],
        contact_frame: wp.array[wp.mat33f],
        angular_friction: wp.array[wp.vec2f],
        spatial_delassus: wp.array[mat66f],
        spatial_eigenvectors: wp.array[mat55f],
        spatial_eigenvalues: wp.array[vec5f],
        contact_reaction: wp.array[wp.vec3f],
        angular_reaction: wp.array[wp.vec3f],
        inverse_weight: wp.array[mat66f],
        occupancy: wp.array2d[wp.int32],
        projected_twist: wp.array[vec6f],
        twist_delta: wp.array[vec6f],
        world_failed: wp.array[wp.bool],
    ):
        """Project every contact of one world with the threads of one block."""
        local = tid
        while local < world_contact_count[wid]:
            project_contact_row(
                world_contact_offset[wid] + local,
                wid,
                color,
                contact_body_a,
                contact_body_b,
                contact_jacobian_a,
                contact_jacobian_b,
                contact_bias,
                contact_friction,
                contact_delassus,
                contact_compliance,
                contact_restitution,
                contact_frame,
                angular_friction,
                spatial_delassus,
                spatial_eigenvectors,
                spatial_eigenvalues,
                contact_reaction,
                angular_reaction,
                inverse_weight,
                occupancy,
                projected_twist,
                twist_delta,
                world_failed,
            )
            local += wp.block_dim()

    @wp.func
    def sweep_world_color(
        wid: wp.int32,
        tid: wp.int32,
        color: wp.int32,
        box_capacity: wp.int32,
        world_box_offset: wp.array[wp.int32],
        world_contact_offset: wp.array[wp.int32],
        world_color_count: wp.array2d[wp.int32],
        world_order: wp.array[wp.int32],
        box_body_a: wp.array[wp.int32],
        box_body_b: wp.array[wp.int32],
        box_jacobian_a: wp.array[vec6f],
        box_jacobian_b: wp.array[vec6f],
        box_bias: wp.array[wp.float32],
        box_lower: wp.array[wp.float32],
        box_upper: wp.array[wp.float32],
        box_delassus: wp.array[wp.float32],
        box_reaction: wp.array[wp.float32],
        contact_body_a: wp.array[wp.int32],
        contact_body_b: wp.array[wp.int32],
        contact_jacobian_a: wp.array[mat36f],
        contact_jacobian_b: wp.array[mat36f],
        contact_bias: wp.array[wp.vec3f],
        contact_friction: wp.array[wp.float32],
        contact_delassus: wp.array[wp.mat33f],
        contact_compliance: wp.array[wp.float32],
        contact_restitution: wp.array[wp.vec4f],
        contact_frame: wp.array[wp.mat33f],
        angular_friction: wp.array[wp.vec2f],
        spatial_delassus: wp.array[mat66f],
        spatial_eigenvectors: wp.array[mat55f],
        spatial_eigenvalues: wp.array[vec5f],
        contact_reaction: wp.array[wp.vec3f],
        angular_reaction: wp.array[wp.vec3f],
        inverse_weight: wp.array[mat66f],
        occupancy: wp.array2d[wp.int32],
        projected_twist: wp.array[vec6f],
        twist_delta: wp.array[vec6f],
        world_failed: wp.array[wp.bool],
    ):
        """Project the rows of one color of one world, read from its color-sorted rows, with the threads of one block."""
        start = world_box_offset[wid] + world_contact_offset[wid]
        for previous in range(color):
            start += world_color_count[wid, previous]
        local = tid
        while local < world_color_count[wid, color]:
            row = world_order[start + local]
            if row < box_capacity:
                _project_box_row(
                    row,
                    color,
                    box_body_a,
                    box_body_b,
                    box_jacobian_a,
                    box_jacobian_b,
                    box_bias,
                    box_lower,
                    box_upper,
                    box_delassus,
                    box_reaction,
                    inverse_weight,
                    occupancy,
                    projected_twist,
                    twist_delta,
                )
            else:
                project_contact_row(
                    row - box_capacity,
                    wid,
                    color,
                    contact_body_a,
                    contact_body_b,
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_bias,
                    contact_friction,
                    contact_delassus,
                    contact_compliance,
                    contact_restitution,
                    contact_frame,
                    angular_friction,
                    spatial_delassus,
                    spatial_eigenvectors,
                    spatial_eigenvalues,
                    contact_reaction,
                    angular_reaction,
                    inverse_weight,
                    occupancy,
                    projected_twist,
                    twist_delta,
                    world_failed,
                )
            local += wp.block_dim()

    @wp.func
    def store_world_map_input(
        wid: wp.int32,
        tid: wp.int32,
        world_box_offset: wp.array[wp.int32],
        world_box_count: wp.array[wp.int32],
        world_contact_offset: wp.array[wp.int32],
        world_contact_count: wp.array[wp.int32],
        box_reaction: wp.array[wp.float32],
        contact_reaction: wp.array[wp.vec3f],
        angular_reaction: wp.array[wp.vec3f],
        box_input: wp.array[wp.float32],
        contact_input: wp.array[wp.vec3f],
        angular_input: wp.array[wp.vec3f],
    ):
        """Store the reactions of one world as the input of its next Anderson map."""
        local = tid
        while local < world_box_count[wid]:
            row = world_box_offset[wid] + local
            box_input[row] = box_reaction[row]
            local += wp.block_dim()
        local = tid
        while local < world_contact_count[wid]:
            store_contact_map_input(
                world_contact_offset[wid] + local, contact_reaction, angular_reaction, contact_input, angular_input
            )
            local += wp.block_dim()

    @wp.func
    def finish_world_map(
        wid: wp.int32,
        tid: wp.int32,
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        world_box_offset: wp.array[wp.int32],
        world_box_count: wp.array[wp.int32],
        world_contact_offset: wp.array[wp.int32],
        world_contact_count: wp.array[wp.int32],
        box_body_a: wp.array[wp.int32],
        box_body_b: wp.array[wp.int32],
        box_jacobian_a: wp.array[vec6f],
        box_jacobian_b: wp.array[vec6f],
        box_lower: wp.array[wp.float32],
        box_upper: wp.array[wp.float32],
        box_reaction: wp.array[wp.float32],
        contact_body_a: wp.array[wp.int32],
        contact_body_b: wp.array[wp.int32],
        contact_jacobian_a: wp.array[mat36f],
        contact_jacobian_b: wp.array[mat36f],
        contact_friction: wp.array[wp.float32],
        contact_frame: wp.array[wp.mat33f],
        angular_friction: wp.array[wp.vec2f],
        contact_reaction: wp.array[wp.vec3f],
        angular_reaction: wp.array[wp.vec3f],
        state: AndersonWorldState,
        inverse_weight: wp.array[mat66f],
        twist_delta: wp.array[vec6f],
    ):
        """Store the samples of one Anderson map of one world, mix its rows, and accumulate their body deltas.

        Every thread of the block must call this function, which reduces the secant of the world over the block.
        """
        had_sample = state.sample_valid[wid] == 1
        had_direction = state.direction_valid[wid] == 1
        term = wp.vec3f(0.0)
        local = tid
        while local < world_box_count[wid]:
            term += store_box_anderson_sample(
                world_box_offset[wid] + local,
                had_sample,
                had_direction,
                box_reaction,
                state.box_input,
                state.box_sample_residual,
                state.box_sample_image,
                state.box_direction_residual,
                state.box_direction_image,
            )
            local += wp.block_dim()
        local = tid
        while local < world_contact_count[wid]:
            term += store_contact_anderson_sample(
                world_contact_offset[wid] + local,
                had_sample,
                had_direction,
                contact_reaction,
                state.contact_input,
                state.contact_sample_residual,
                state.contact_sample_image,
                state.contact_direction_residual,
                state.contact_direction_image,
                angular_reaction,
                state.angular_input,
                state.angular_sample_residual,
                state.angular_sample_image,
                state.angular_direction_residual,
                state.angular_direction_image,
                angular_friction,
            )
            local += wp.block_dim()
        secant = wp.tile_sum(wp.tile(term), axis=1)
        if tid == 0:
            finalize_anderson_world(
                wid,
                world_active,
                world_failed,
                secant[0],
                secant[1],
                secant[2],
                state.sample_valid,
                state.direction_valid,
                state.guard_pending,
                state.previous_residual_norm,
                state.coefficient,
            )
        _sync_threads()
        mixing = state.coefficient[wid]
        local = tid
        while local < world_box_count[wid]:
            mix_box_anderson_row(
                world_box_offset[wid] + local,
                mixing,
                box_body_a,
                box_body_b,
                box_jacobian_a,
                box_jacobian_b,
                box_lower,
                box_upper,
                state.box_direction_image,
                inverse_weight,
                state.box_input,
                box_reaction,
                twist_delta,
            )
            local += wp.block_dim()
        local = tid
        while local < world_contact_count[wid]:
            mix_contact_anderson_row(
                world_contact_offset[wid] + local,
                mixing,
                contact_body_a,
                contact_body_b,
                contact_jacobian_a,
                contact_jacobian_b,
                contact_friction,
                state.contact_direction_image,
                angular_friction,
                contact_frame,
                state.angular_direction_image,
                inverse_weight,
                state.contact_input,
                contact_reaction,
                state.angular_input,
                angular_reaction,
                twist_delta,
            )
            local += wp.block_dim()

    @wp.func
    def initialize_world_apgd(
        wid: wp.int32,
        tid: wp.int32,
        world_box_offset: wp.array[wp.int32],
        world_box_count: wp.array[wp.int32],
        world_contact_offset: wp.array[wp.int32],
        world_contact_count: wp.array[wp.int32],
        box_lower: wp.array[wp.float32],
        box_upper: wp.array[wp.float32],
        contact_body_a: wp.array[wp.int32],
        contact_body_b: wp.array[wp.int32],
        contact_friction: wp.array[wp.float32],
        angular_friction: wp.array[wp.vec2f],
        box_reaction: wp.array[wp.float32],
        contact_reaction: wp.array[wp.vec3f],
        angular_reaction: wp.array[wp.vec3f],
        state: APGDRowState,
    ):
        """Clamp the warm-start reactions of one world and reset their momentum.

        The rows are visited in the order of the warm start, so that each thread applies its own clamped rows.
        """
        local = tid
        while local < world_box_count[wid]:
            initialize_box_apgd_row(
                world_box_offset[wid] + local, box_lower, box_upper, box_reaction, state.box_trial, state.box_previous
            )
            local += wp.block_dim()
        local = tid
        while local < world_contact_count[wid]:
            initialize_contact_apgd_row(
                world_contact_offset[wid] + local,
                contact_body_a,
                contact_body_b,
                contact_friction,
                angular_friction,
                contact_reaction,
                state.contact_trial,
                state.contact_previous,
                angular_reaction,
                state.angular_trial,
                state.angular_previous,
            )
            local += wp.block_dim()

    @wp.func
    def extrapolate_world(
        wid: wp.int32,
        tid: wp.int32,
        theta: wp.float32,
        world_box_offset: wp.array[wp.int32],
        world_box_count: wp.array[wp.int32],
        world_contact_offset: wp.array[wp.int32],
        world_contact_count: wp.array[wp.int32],
        box_body_a: wp.array[wp.int32],
        box_body_b: wp.array[wp.int32],
        box_jacobian_a: wp.array[vec6f],
        box_jacobian_b: wp.array[vec6f],
        contact_body_a: wp.array[wp.int32],
        contact_body_b: wp.array[wp.int32],
        contact_jacobian_a: wp.array[mat36f],
        contact_jacobian_b: wp.array[mat36f],
        contact_frame: wp.array[wp.mat33f],
        contact_friction: wp.array[wp.float32],
        angular_friction: wp.array[wp.vec2f],
        box_reaction: wp.array[wp.float32],
        contact_reaction: wp.array[wp.vec3f],
        angular_reaction: wp.array[wp.vec3f],
        state: APGDRowState,
        inverse_weight: wp.array[mat66f],
        twist_delta: wp.array[vec6f],
    ) -> wp.float32:
        """Update the momentum of one world, extrapolate its rows, and return the next ``theta``.

        Every thread of the block must call this function, which reduces the restart test over the block.
        """
        dot = wp.float32(0.0)
        local = tid
        while local < world_box_count[wid]:
            dot += box_restart_dot(world_box_offset[wid] + local, box_reaction, state.box_trial, state.box_previous)
            local += wp.block_dim()
        local = tid
        while local < world_contact_count[wid]:
            dot += contact_restart_dot(
                world_contact_offset[wid] + local,
                contact_reaction,
                state.contact_trial,
                state.contact_previous,
                angular_reaction,
                state.angular_trial,
                state.angular_previous,
                contact_friction,
                angular_friction,
            )
            local += wp.block_dim()
        restart_dot = wp.tile_sum(wp.tile(dot))
        next_theta, beta = apgd_momentum(theta, restart_dot[0])
        local = tid
        while local < world_box_count[wid]:
            extrapolate_box_apgd_row(
                world_box_offset[wid] + local,
                beta,
                box_body_a,
                box_body_b,
                box_jacobian_a,
                box_jacobian_b,
                inverse_weight,
                box_reaction,
                state.box_trial,
                state.box_previous,
                twist_delta,
            )
            local += wp.block_dim()
        local = tid
        while local < world_contact_count[wid]:
            extrapolate_contact_apgd_row(
                world_contact_offset[wid] + local,
                beta,
                contact_body_a,
                contact_body_b,
                contact_jacobian_a,
                contact_jacobian_b,
                contact_frame,
                inverse_weight,
                contact_reaction,
                state.contact_trial,
                state.contact_previous,
                angular_reaction,
                state.angular_trial,
                state.angular_previous,
                twist_delta,
            )
            local += wp.block_dim()
        return next_theta

    @wp.kernel(module="unique", enable_backward=False)
    def project_by_world(
        # Inputs:
        iterations: wp.int32,
        color_count: wp.int32,
        world_body_offset: wp.array[wp.int32],
        world_body_count: wp.array[wp.int32],
        box_capacity: wp.int32,
        world_box_offset: wp.array[wp.int32],
        world_box_count: wp.array[wp.int32],
        box_body_a: wp.array[wp.int32],
        box_body_b: wp.array[wp.int32],
        box_jacobian_a: wp.array[vec6f],
        box_jacobian_b: wp.array[vec6f],
        box_bias: wp.array[wp.float32],
        box_lower: wp.array[wp.float32],
        box_upper: wp.array[wp.float32],
        box_delassus: wp.array[wp.float32],
        box_smoothing_delassus: wp.array[wp.float32],
        world_contact_offset: wp.array[wp.int32],
        world_contact_count: wp.array[wp.int32],
        contact_body_a: wp.array[wp.int32],
        contact_body_b: wp.array[wp.int32],
        contact_jacobian_a: wp.array[mat36f],
        contact_jacobian_b: wp.array[mat36f],
        contact_bias: wp.array[wp.vec3f],
        contact_friction: wp.array[wp.float32],
        contact_delassus: wp.array[wp.mat33f],
        contact_smoothing_delassus: wp.array[wp.mat33f],
        contact_compliance: wp.array[wp.float32],
        contact_restitution: wp.array[wp.vec4f],
        # Spatial contacts (with angular friction):
        contact_frame: wp.array[wp.mat33f],
        angular_friction: wp.array[wp.vec2f],
        spatial_delassus: wp.array[mat66f],
        spatial_eigenvectors: wp.array[mat55f],
        spatial_eigenvalues: wp.array[vec5f],
        spatial_smoothing_delassus: wp.array[mat66f],
        spatial_smoothing_eigenvectors: wp.array[mat55f],
        spatial_smoothing_eigenvalues: wp.array[vec5f],
        acceleration_state: AccelerationState,
        # Colored sweeps:
        world_color_count: wp.array2d[wp.int32],
        world_order: wp.array[wp.int32],
        occupancy: wp.array2d[wp.int32],
        # Common:
        body_incidence: wp.array2d[wp.int32],
        world_active: wp.array[wp.bool],
        inverse_weight: wp.array[mat66f],
        # Outputs:
        box_reaction: wp.array[wp.float32],
        contact_reaction: wp.array[wp.vec3f],
        angular_reaction: wp.array[wp.vec3f],
        projected_twist: wp.array[vec6f],
        twist_delta: wp.array[vec6f],
        world_failed: wp.array[wp.bool],
    ):
        """Warm start and sweep one world per block.

        Colored worlds sweep their colors in order and finish with one Jacobi
        smoothing sweep; Jacobi worlds sweep all their rows at once. With
        Anderson acceleration, each iteration is one map whose
        image is mixed along the stored direction; with APGD, every Jacobi sweep
        but the last is extrapolated.
        """
        wid, tid = wp.tid()
        if not world_active[wid]:
            return
        if world_failed[wid]:
            return

        if wp.static(apgd):
            initialize_world_apgd(
                wid,
                tid,
                world_box_offset,
                world_box_count,
                world_contact_offset,
                world_contact_count,
                box_lower,
                box_upper,
                contact_body_a,
                contact_body_b,
                contact_friction,
                angular_friction,
                box_reaction,
                contact_reaction,
                angular_reaction,
                acceleration_state,
            )

        # Apply the existing reactions before the first sweep, clamping the spatial reactions into their cone.
        local = tid
        while local < world_box_count[wid]:
            row = world_box_offset[wid] + local
            scatter_box_impulse(
                row,
                box_body_a,
                box_body_b,
                box_jacobian_a,
                box_jacobian_b,
                box_reaction[row],
                inverse_weight,
                twist_delta,
            )
            local += wp.block_dim()
        local = tid
        while local < world_contact_count[wid]:
            cid = world_contact_offset[wid] + local
            if wp.static(spatial):
                reaction = clamp_contact_reaction(
                    pack_contact_reaction(contact_reaction[cid], angular_reaction[cid]),
                    pack_spatial_friction(cid, contact_friction, angular_friction),
                )
                contact_reaction[cid] = linear_contact_reaction(reaction)
                angular_reaction[cid] = angular_contact_reaction(reaction)
                scatter_spatial_impulse(
                    cid,
                    contact_body_a,
                    contact_body_b,
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_frame,
                    reaction,
                    inverse_weight,
                    twist_delta,
                )
            else:
                scatter_contact_impulse(
                    cid,
                    contact_body_a,
                    contact_body_b,
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_reaction[cid],
                    inverse_weight,
                    twist_delta,
                )
            local += wp.block_dim()
        _apply_world_delta(wid, tid, world_body_offset, world_body_count, world_failed, twist_delta, projected_twist)
        if wp.static(anderson):
            store_world_map_input(
                wid,
                tid,
                world_box_offset,
                world_box_count,
                world_contact_offset,
                world_contact_count,
                box_reaction,
                contact_reaction,
                angular_reaction,
                acceleration_state.box_input,
                acceleration_state.contact_input,
                acceleration_state.angular_input,
            )

        theta = wp.float32(1.0)
        for iteration in range(iterations):
            if wp.static(colored):
                for color in range(color_count):
                    if world_color_count[wid, color] == 0:
                        continue
                    sweep_world_color(
                        wid,
                        tid,
                        color,
                        box_capacity,
                        world_box_offset,
                        world_contact_offset,
                        world_color_count,
                        world_order,
                        box_body_a,
                        box_body_b,
                        box_jacobian_a,
                        box_jacobian_b,
                        box_bias,
                        box_lower,
                        box_upper,
                        box_delassus,
                        box_reaction,
                        contact_body_a,
                        contact_body_b,
                        contact_jacobian_a,
                        contact_jacobian_b,
                        contact_bias,
                        contact_friction,
                        contact_delassus,
                        contact_compliance,
                        contact_restitution,
                        contact_frame,
                        angular_friction,
                        spatial_delassus,
                        spatial_eigenvectors,
                        spatial_eigenvalues,
                        contact_reaction,
                        angular_reaction,
                        inverse_weight,
                        occupancy,
                        projected_twist,
                        twist_delta,
                        world_failed,
                    )
                    _apply_world_delta(
                        wid, tid, world_body_offset, world_body_count, world_failed, twist_delta, projected_twist
                    )
                    if world_failed[wid]:
                        return
            else:
                _sweep_world_box_rows(
                    wid,
                    tid,
                    0,
                    world_box_offset,
                    world_box_count,
                    box_body_a,
                    box_body_b,
                    box_jacobian_a,
                    box_jacobian_b,
                    box_bias,
                    box_lower,
                    box_upper,
                    box_delassus,
                    box_reaction,
                    inverse_weight,
                    body_incidence,
                    projected_twist,
                    twist_delta,
                )
                sweep_world_contacts(
                    wid,
                    tid,
                    0,
                    world_contact_offset,
                    world_contact_count,
                    contact_body_a,
                    contact_body_b,
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_bias,
                    contact_friction,
                    contact_delassus,
                    contact_compliance,
                    contact_restitution,
                    contact_frame,
                    angular_friction,
                    spatial_delassus,
                    spatial_eigenvectors,
                    spatial_eigenvalues,
                    contact_reaction,
                    angular_reaction,
                    inverse_weight,
                    body_incidence,
                    projected_twist,
                    twist_delta,
                    world_failed,
                )
                _apply_world_delta(
                    wid, tid, world_body_offset, world_body_count, world_failed, twist_delta, projected_twist
                )
                if world_failed[wid]:
                    return
            if wp.static(anderson):
                finish_world_map(
                    wid,
                    tid,
                    world_active,
                    world_failed,
                    world_box_offset,
                    world_box_count,
                    world_contact_offset,
                    world_contact_count,
                    box_body_a,
                    box_body_b,
                    box_jacobian_a,
                    box_jacobian_b,
                    box_lower,
                    box_upper,
                    box_reaction,
                    contact_body_a,
                    contact_body_b,
                    contact_jacobian_a,
                    contact_jacobian_b,
                    contact_friction,
                    contact_frame,
                    angular_friction,
                    contact_reaction,
                    angular_reaction,
                    acceleration_state,
                    inverse_weight,
                    twist_delta,
                )
                _apply_world_delta(
                    wid, tid, world_body_offset, world_body_count, world_failed, twist_delta, projected_twist
                )
            if wp.static(apgd):
                if iteration + 1 < iterations:
                    theta = extrapolate_world(
                        wid,
                        tid,
                        theta,
                        world_box_offset,
                        world_box_count,
                        world_contact_offset,
                        world_contact_count,
                        box_body_a,
                        box_body_b,
                        box_jacobian_a,
                        box_jacobian_b,
                        contact_body_a,
                        contact_body_b,
                        contact_jacobian_a,
                        contact_jacobian_b,
                        contact_frame,
                        contact_friction,
                        angular_friction,
                        box_reaction,
                        contact_reaction,
                        angular_reaction,
                        acceleration_state,
                        inverse_weight,
                        twist_delta,
                    )
                    _apply_world_delta(
                        wid, tid, world_body_offset, world_body_count, world_failed, twist_delta, projected_twist
                    )

        if wp.static(colored):
            # Finish with one order-independent Jacobi sweep, which restores the
            # symmetry lost to the color ordering.
            _sweep_world_box_rows(
                wid,
                tid,
                0,
                world_box_offset,
                world_box_count,
                box_body_a,
                box_body_b,
                box_jacobian_a,
                box_jacobian_b,
                box_bias,
                box_lower,
                box_upper,
                box_smoothing_delassus,
                box_reaction,
                inverse_weight,
                body_incidence,
                projected_twist,
                twist_delta,
            )
            sweep_world_contacts(
                wid,
                tid,
                0,
                world_contact_offset,
                world_contact_count,
                contact_body_a,
                contact_body_b,
                contact_jacobian_a,
                contact_jacobian_b,
                contact_bias,
                contact_friction,
                contact_smoothing_delassus,
                contact_compliance,
                contact_restitution,
                contact_frame,
                angular_friction,
                spatial_smoothing_delassus,
                spatial_smoothing_eigenvectors,
                spatial_smoothing_eigenvalues,
                contact_reaction,
                angular_reaction,
                inverse_weight,
                body_incidence,
                projected_twist,
                twist_delta,
                world_failed,
            )
            _apply_world_delta(
                wid, tid, world_body_offset, world_body_count, world_failed, twist_delta, projected_twist
            )

    return project_by_world


###
# Utilities
###


def can_project_by_world(
    device: wp.Device,
    world_count: int,
    rows_per_world_pass: int,
    block_dim: int,
    min_blocks_per_sm: int = _WORLD_MIN_BLOCKS_PER_SM,
) -> bool:
    """Return whether one block per world beats row-parallel launches.

    The worlds must fill the device, or their row capacity must stay within a few rows per thread
    of their blocks. The capacity bounds the active rows, often by far, e.g. for hydroelastic contacts.

    Args:
        device: The device running the projection.
        world_count: Number of worlds, one block each.
        rows_per_world_pass: Rows processed in parallel by one pass over all worlds.
        block_dim: Threads per world block.
        min_blocks_per_sm: World blocks per SM that keep the device occupied.
    """
    if not device.is_cuda or world_count == 0:
        return False
    enough_world_blocks = world_count >= device.sm_count * min_blocks_per_sm
    small_world_work = rows_per_world_pass <= world_count * block_dim * _WORLD_ROWS_PER_THREAD
    return enough_world_blocks or small_world_work
