# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Nesterov (APGD) extrapolation of the LOX Jacobi projection sweeps.

After every sweep but the last, the reactions are extrapolated along their
last change, ``lambda <- lambda + beta (lambda - lambda_previous)``, with the
usual ``theta`` recursion. The momentum restarts per world whenever the last
sweep moved away from the previous extrapolation.

The kernels run over the stacked reaction vector ``[box rows | contacts]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...core.types import mat36f, mat66f, vec6f
from ..padmm.math import project_to_coulomb_cone
from .contact import (
    angular_contact_reaction,
    clamp_contact_reaction,
    linear_contact_reaction,
    pack_contact_reaction,
    pack_spatial_friction,
    scatter_spatial_impulse,
    spatial_friction_scaling,
)
from .projection_kernels import is_active_row, scatter_box_impulse, scatter_contact_impulse

if TYPE_CHECKING:
    from .types import APGDData, LOXProblemData

###
# Module interface
###

__all__ = [
    "APGDAcceleration",
    "APGDRowState",
    "apgd_momentum",
    "box_restart_dot",
    "contact_restart_dot",
    "extrapolate_box_apgd_row",
    "extrapolate_contact_apgd_row",
    "initialize_box_apgd_row",
    "initialize_contact_apgd_row",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@wp.struct
class APGDRowState:
    """Trial and previous reactions of the APGD extrapolation, for the per-world projection."""

    box_trial: wp.array[wp.float32]
    box_previous: wp.array[wp.float32]
    contact_trial: wp.array[wp.vec3f]
    contact_previous: wp.array[wp.vec3f]
    angular_trial: wp.array[wp.vec3f]
    angular_previous: wp.array[wp.vec3f]


###
# Functions
###


@wp.func
def apgd_momentum(theta: wp.float32, restart_dot: wp.float32):
    """Return the next ``theta`` and the extrapolation ``beta``, restarting when the last sweep moved away.

    ``theta`` starts at one and the recursion keeps it in ``(0, 1]``.
    """
    # The negated comparison also restarts on a NaN dot product
    if not (restart_dot > 0.0):
        return wp.float32(1.0), wp.float32(0.0)
    next_theta = 2.0 * theta / (wp.sqrt(theta * theta + 4.0) + theta)
    return next_theta, theta * (1.0 - theta) / (theta * theta + next_theta)


@wp.func
def initialize_box_apgd_row(
    row: wp.int32,
    box_lower: wp.array[wp.float32],
    box_upper: wp.array[wp.float32],
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
):
    """Clamp the warm-start reaction of one box row and reset its momentum."""
    value = wp.clamp(box_reaction[row], box_lower[row], box_upper[row])
    box_reaction[row] = value
    box_trial[row] = value
    box_previous[row] = value


@wp.func
def initialize_contact_apgd_row(
    cid: wp.int32,
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
):
    """Project the warm-start reaction of one contact onto its cone and reset its momentum."""
    value = wp.vec3f(0.0)
    if contact_body_a[cid] >= 0 or contact_body_b[cid] >= 0:
        if angular_reaction:
            spatial = clamp_contact_reaction(
                pack_contact_reaction(contact_reaction[cid], angular_reaction[cid]),
                pack_spatial_friction(cid, contact_friction, angular_friction),
            )
            value = linear_contact_reaction(spatial)
            angular = angular_contact_reaction(spatial)
            angular_reaction[cid] = angular
            angular_trial[cid] = angular
            angular_previous[cid] = angular
        else:
            value = project_to_coulomb_cone(contact_reaction[cid], contact_friction[cid])
    contact_reaction[cid] = value
    contact_trial[cid] = value
    contact_previous[cid] = value


@wp.func
def box_restart_dot(
    row: wp.int32,
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
) -> wp.float32:
    """Return ``(lambda - trial) (lambda - previous)`` of one box row."""
    current = box_reaction[row]
    return (current - box_trial[row]) * (current - box_previous[row])


@wp.func
def contact_restart_dot(
    cid: wp.int32,
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
) -> wp.float32:
    """Return ``(lambda - trial) . (lambda - previous)`` of one contact, in units of the normal impulse."""
    current = contact_reaction[cid]
    if not angular_reaction:
        return wp.dot(current - contact_trial[cid], current - contact_previous[cid])
    spatial = pack_contact_reaction(current, angular_reaction[cid])
    trial = pack_contact_reaction(contact_trial[cid], angular_trial[cid])
    previous = pack_contact_reaction(contact_previous[cid], angular_previous[cid])
    scale = spatial_friction_scaling(pack_spatial_friction(cid, contact_friction, angular_friction))
    dot = wp.float32(0.0)
    for axis in range(6):
        if scale[axis] > 0.0:
            dot += ((spatial[axis] - trial[axis]) / scale[axis]) * ((spatial[axis] - previous[axis]) / scale[axis])
    return dot


@wp.func
def extrapolate_box_apgd_row(
    row: wp.int32,
    beta: wp.float32,
    box_body_a: wp.array[wp.int32],
    box_body_b: wp.array[wp.int32],
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    inverse_weight: wp.array[mat66f],
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
    twist_delta: wp.array[vec6f],
):
    """Extrapolate the reaction of one box row and accumulate its weighted body correction."""
    current = box_reaction[row]
    extrapolated = current + beta * (current - box_previous[row])
    box_previous[row] = current
    box_trial[row] = extrapolated
    box_reaction[row] = extrapolated
    scatter_box_impulse(
        row,
        box_body_a,
        box_body_b,
        box_jacobian_a,
        box_jacobian_b,
        extrapolated - current,
        inverse_weight,
        twist_delta,
    )


@wp.func
def extrapolate_contact_apgd_row(
    cid: wp.int32,
    beta: wp.float32,
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    contact_frame: wp.array[wp.mat33f],
    inverse_weight: wp.array[mat66f],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
    twist_delta: wp.array[vec6f],
):
    """Extrapolate the reaction of one contact and accumulate its weighted body correction."""
    current = contact_reaction[cid]
    extrapolated = current + beta * (current - contact_previous[cid])
    contact_previous[cid] = current
    contact_trial[cid] = extrapolated
    contact_reaction[cid] = extrapolated
    if angular_reaction:
        angular = angular_reaction[cid]
        angular_extrapolated = angular + beta * (angular - angular_previous[cid])
        angular_previous[cid] = angular
        angular_trial[cid] = angular_extrapolated
        angular_reaction[cid] = angular_extrapolated
        # One spatial scatter applies the linear and angular extrapolations together
        scatter_spatial_impulse(
            cid,
            contact_body_a,
            contact_body_b,
            contact_jacobian_a,
            contact_jacobian_b,
            contact_frame,
            pack_contact_reaction(extrapolated - current, angular_extrapolated - angular),
            inverse_weight,
            twist_delta,
        )
        return
    scatter_contact_impulse(
        cid,
        contact_body_a,
        contact_body_b,
        contact_jacobian_a,
        contact_jacobian_b,
        extrapolated - current,
        inverse_weight,
        twist_delta,
    )


###
# Kernels
###


@wp.kernel
def _initialize_worlds(
    # Inputs:
    world_active: wp.array[wp.bool],
    # Outputs:
    theta: wp.array[wp.float32],
    beta: wp.array[wp.float32],
    restart_dot: wp.array[wp.float32],
):
    wid = wp.tid()
    if world_active[wid]:
        theta[wid] = 1.0
        beta[wid] = 0.0
        restart_dot[wid] = 0.0


@wp.kernel
def _initialize_reactions(
    # Inputs:
    box_capacity: wp.int32,
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_lower: wp.array[wp.float32],
    box_upper: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
    world_active: wp.array[wp.bool],
    # Outputs:
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
):
    """Clamp the warm-start reactions into their feasible sets and reset the momentum."""
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        if box_local[row] < world_box_count[wid] and world_active[wid]:
            initialize_box_apgd_row(row, box_lower, box_upper, box_reaction, box_trial, box_previous)
        return
    cid = tid - box_capacity
    wid = contact_world[cid]
    if contact_local[cid] < world_contact_count[wid] and world_active[wid]:
        initialize_contact_apgd_row(
            cid,
            contact_body_a,
            contact_body_b,
            contact_friction,
            angular_friction,
            contact_reaction,
            contact_trial,
            contact_previous,
            angular_reaction,
            angular_trial,
            angular_previous,
        )


@wp.kernel
def _accumulate_restart_dot(
    # Inputs:
    box_capacity: wp.int32,
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
    contact_friction: wp.array[wp.float32],
    angular_friction: wp.array[wp.vec2f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    # Outputs:
    restart_dot: wp.array[wp.float32],
):
    """Accumulate ``(lambda - trial) . (lambda - previous)`` per world."""
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        if is_active_row(box_local[row], wid, world_box_count, world_active, world_failed):
            wp.atomic_add(restart_dot, wid, box_restart_dot(row, box_reaction, box_trial, box_previous))
        return
    cid = tid - box_capacity
    wid = contact_world[cid]
    if is_active_row(contact_local[cid], wid, world_contact_count, world_active, world_failed):
        wp.atomic_add(
            restart_dot,
            wid,
            contact_restart_dot(
                cid,
                contact_reaction,
                contact_trial,
                contact_previous,
                angular_reaction,
                angular_trial,
                angular_previous,
                contact_friction,
                angular_friction,
            ),
        )


@wp.kernel
def _update_momentum(
    # Inputs:
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    # Outputs:
    restart_dot: wp.array[wp.float32],
    theta: wp.array[wp.float32],
    beta: wp.array[wp.float32],
):
    wid = wp.tid()
    if not world_active[wid]:
        return
    value = restart_dot[wid]
    restart_dot[wid] = 0.0
    if world_failed[wid]:
        value = 0.0
    next_theta, next_beta = apgd_momentum(theta[wid], value)
    theta[wid] = next_theta
    beta[wid] = next_beta


@wp.kernel
def _extrapolate_reactions(
    # Inputs:
    box_capacity: wp.int32,
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    box_body_a: wp.array[wp.int32],
    box_body_b: wp.array[wp.int32],
    box_jacobian_a: wp.array[vec6f],
    box_jacobian_b: wp.array[vec6f],
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    contact_body_a: wp.array[wp.int32],
    contact_body_b: wp.array[wp.int32],
    contact_jacobian_a: wp.array[mat36f],
    contact_jacobian_b: wp.array[mat36f],
    contact_frame: wp.array[wp.mat33f],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    beta: wp.array[wp.float32],
    inverse_weight: wp.array[mat66f],
    # Outputs:
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
    twist_delta: wp.array[vec6f],
):
    """Extrapolate each reaction and accumulate the weighted body correction."""
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        if is_active_row(box_local[row], wid, world_box_count, world_active, world_failed):
            extrapolate_box_apgd_row(
                row,
                beta[wid],
                box_body_a,
                box_body_b,
                box_jacobian_a,
                box_jacobian_b,
                inverse_weight,
                box_reaction,
                box_trial,
                box_previous,
                twist_delta,
            )
        return
    cid = tid - box_capacity
    wid = contact_world[cid]
    if is_active_row(contact_local[cid], wid, world_contact_count, world_active, world_failed):
        extrapolate_contact_apgd_row(
            cid,
            beta[wid],
            contact_body_a,
            contact_body_b,
            contact_jacobian_a,
            contact_jacobian_b,
            contact_frame,
            inverse_weight,
            contact_reaction,
            contact_trial,
            contact_previous,
            angular_reaction,
            angular_trial,
            angular_previous,
            twist_delta,
        )


###
# Interfaces
###


class APGDAcceleration:
    """Own the momentum state of APGD-accelerated Jacobi projection."""

    def __init__(self, problem_data: LOXProblemData, data: APGDData):
        self.problem_data = problem_data
        self._data = data
        self._capacity = problem_data.box_rows.capacity + problem_data.contact_rows.capacity
        self.row_state = APGDRowState()
        """Trial and previous reactions, for the per-world projection."""
        for name in ("box_trial", "box_previous", "contact_trial", "contact_previous"):
            setattr(self.row_state, name, getattr(data, name))
        if data.angular_trial is not None:
            self.row_state.angular_trial = data.angular_trial
            self.row_state.angular_previous = data.angular_previous

    ###
    # Public API
    ###

    def begin(self, world_active: wp.array[wp.bool]) -> None:
        """Clamp the warm-start reactions and restart the momentum of active worlds."""
        p = self.problem_data
        angular_reaction, angular_friction, _frame = self._spatial_arrays()
        wp.launch(
            _initialize_worlds,
            dim=p.num_worlds,
            inputs=[world_active],
            outputs=[self._data.theta, self._data.beta, self._data.restart_dot],
            device=p.device,
        )
        if self._capacity == 0:
            return
        wp.launch(
            _initialize_reactions,
            dim=self._capacity,
            inputs=[
                p.box_rows.capacity,
                p.box_rows.world,
                p.box_rows.local,
                p.box_rows.world_count,
                p.box_rows.lower,
                p.box_rows.upper,
                p.contact_rows.world,
                p.contact_rows.local,
                p.contact_rows.world_count,
                p.contact_rows.body_a,
                p.contact_rows.body_b,
                p.contact_rows.friction,
                angular_friction,
                world_active,
            ],
            outputs=[
                p.box_rows.reaction,
                self._data.box_trial,
                self._data.box_previous,
                p.contact_rows.reaction,
                self._data.contact_trial,
                self._data.contact_previous,
                angular_reaction,
                self._data.angular_trial,
                self._data.angular_previous,
            ],
            device=p.device,
        )

    def extrapolate(
        self,
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        inverse_weight: wp.array[mat66f],
        twist_delta: wp.array[vec6f],
    ) -> None:
        """Update the momentum and accumulate the extrapolation into ``twist_delta``."""
        p = self.problem_data
        angular_reaction, angular_friction, frame = self._spatial_arrays()
        if self._capacity > 0:
            wp.launch(
                _accumulate_restart_dot,
                dim=self._capacity,
                inputs=[
                    p.box_rows.capacity,
                    p.box_rows.world,
                    p.box_rows.local,
                    p.box_rows.world_count,
                    p.box_rows.reaction,
                    self._data.box_trial,
                    self._data.box_previous,
                    p.contact_rows.world,
                    p.contact_rows.local,
                    p.contact_rows.world_count,
                    p.contact_rows.reaction,
                    self._data.contact_trial,
                    self._data.contact_previous,
                    angular_reaction,
                    self._data.angular_trial,
                    self._data.angular_previous,
                    p.contact_rows.friction,
                    angular_friction,
                    world_active,
                    world_failed,
                ],
                outputs=[self._data.restart_dot],
                device=p.device,
            )
        wp.launch(
            _update_momentum,
            dim=p.num_worlds,
            inputs=[world_active, world_failed],
            outputs=[self._data.restart_dot, self._data.theta, self._data.beta],
            device=p.device,
        )
        if self._capacity == 0:
            return
        wp.launch(
            _extrapolate_reactions,
            dim=self._capacity,
            inputs=[
                p.box_rows.capacity,
                p.box_rows.world,
                p.box_rows.local,
                p.box_rows.world_count,
                p.box_rows.body_a,
                p.box_rows.body_b,
                p.box_rows.jacobian_a,
                p.box_rows.jacobian_b,
                p.contact_rows.world,
                p.contact_rows.local,
                p.contact_rows.world_count,
                p.contact_rows.body_a,
                p.contact_rows.body_b,
                p.contact_rows.jacobian_a,
                p.contact_rows.jacobian_b,
                frame,
                world_active,
                world_failed,
                self._data.beta,
                inverse_weight,
            ],
            outputs=[
                p.box_rows.reaction,
                self._data.box_trial,
                self._data.box_previous,
                p.contact_rows.reaction,
                self._data.contact_trial,
                self._data.contact_previous,
                angular_reaction,
                self._data.angular_trial,
                self._data.angular_previous,
                twist_delta,
            ],
            device=p.device,
        )

    ###
    # Internals
    ###

    def _spatial_arrays(self):
        """Return the angular reaction, angular friction, and frame of spatial contacts, or ``None``."""
        rows = self.problem_data.contact_rows
        return rows.angular_reaction, rows.angular_friction, rows.frame
