# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Nesterov extrapolation of the LOX projection sweeps.

After every sweep but the last, the reactions are extrapolated along their
last change, ``lambda <- lambda + beta (lambda - lambda_previous)``, with the
usual ``theta`` recursion. The momentum restarts per world whenever the last
sweep moved away from the previous extrapolation. When recycled across
splitting iterations, the extrapolation of the last sweep is applied at the
start of the next projection, after its warm start.

The body twists are affine in the reactions, with weights fixed over a LOX
solve, so each body twist is extrapolated by the same momentum,
``v <- v + beta (v - v_previous)``, with no impulse scatter. Across splitting
iterations, the change ``v - v_previous`` of the last sweep carries over to
the warm start of the next projection, whose candidate twist differs.

The kernels run over the stacked reaction vector ``[box rows | contacts]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...core.types import vec6f
from ..padmm.math import project_to_coulomb_cone
from .contact import (
    angular_contact_reaction,
    clamp_contact_reaction,
    linear_contact_reaction,
    pack_contact_reaction,
    pack_spatial_friction,
    spatial_friction_scaling,
)
from .projection_kernels import is_active_row

if TYPE_CHECKING:
    from .types import LOXProblemData, NesterovData

###
# Module interface
###

__all__ = [
    "NesterovAcceleration",
    "NesterovRowState",
    "box_restart_dot",
    "contact_restart_dot",
    "extrapolate_box_nesterov_row",
    "extrapolate_contact_nesterov_row",
    "initialize_box_nesterov_row",
    "initialize_contact_nesterov_row",
    "nesterov_momentum",
]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Types
###


@wp.struct
class NesterovRowState:
    """Trial and previous reactions of the Nesterov extrapolation, for the per-world projection."""

    box_trial: wp.array[wp.float32]
    box_previous: wp.array[wp.float32]
    contact_trial: wp.array[wp.vec3f]
    contact_previous: wp.array[wp.vec3f]
    angular_trial: wp.array[wp.vec3f]
    angular_previous: wp.array[wp.vec3f]
    theta: wp.array[wp.float32]
    history: wp.array[wp.int32]
    twist_previous: wp.array[vec6f]


###
# Functions
###


@wp.func
def nesterov_momentum(theta: wp.float32, restart_dot: wp.float32):
    """Return the next ``theta`` and the extrapolation ``beta``, restarting when the last sweep moved away.

    ``theta`` starts at one and the recursion keeps it in ``(0, 1]``.
    """
    # The negated comparison also restarts on a NaN dot product
    if not (restart_dot > 0.0):
        return wp.float32(1.0), wp.float32(0.0)
    next_theta = 2.0 * theta / (wp.sqrt(theta * theta + 4.0) + theta)
    return next_theta, theta * (1.0 - theta) / (theta * theta + next_theta)


@wp.func
def initialize_box_nesterov_row(
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
def initialize_contact_nesterov_row(
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
def extrapolate_box_nesterov_row(
    row: wp.int32,
    beta: wp.float32,
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
):
    """Extrapolate the reaction of one box row."""
    current = box_reaction[row]
    extrapolated = current + beta * (current - box_previous[row])
    box_previous[row] = current
    box_trial[row] = extrapolated
    box_reaction[row] = extrapolated


@wp.func
def extrapolate_contact_nesterov_row(
    cid: wp.int32,
    beta: wp.float32,
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
):
    """Extrapolate the linear and, with spatial friction, angular reaction of one contact."""
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


@wp.func
def extrapolate_body_twist(
    bid: wp.int32,
    beta: wp.float32,
    twist: wp.array[vec6f],
    twist_previous: wp.array[vec6f],
):
    """Extrapolate the twist of one body, which follows the extrapolation of the reactions of its world."""
    current = twist[bid]
    twist[bid] = current + beta * (current - twist_previous[bid])
    twist_previous[bid] = current


###
# Kernels
###


@wp.kernel
def _initialize_worlds(
    # Inputs:
    world_active: wp.array[wp.bool],
    recycle: wp.bool,
    first_buffer: wp.int32,
    # Outputs:
    theta: wp.array2d[wp.float32],
    restart_dot: wp.array2d[wp.float32],
    history: wp.array[wp.int32],
):
    """Load the momentum of each world into the buffer of the first extrapolation and clear the restart dots.

    The first buffer holds the momentum between projections, where the last extrapolation writes it.
    """
    wid = wp.tid()
    if world_active[wid]:
        value = wp.float32(1.0)
        if recycle:
            value = theta[0, wid]
        theta[first_buffer, wid] = value
        restart_dot[0, wid] = 0.0
        restart_dot[1, wid] = 0.0
        history[wid] = wp.int32(recycle)


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
    recycle: wp.bool,
    history: wp.array[wp.int32],
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
    """Clamp the warm-start reactions into their feasible sets and reset the momentum.

    With ``recycle``, the worlds that hold the row history of the previous splitting iteration keep it.
    """
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        if box_local[row] < world_box_count[wid] and world_active[wid] and not (recycle and history[wid] == 1):
            initialize_box_nesterov_row(row, box_lower, box_upper, box_reaction, box_trial, box_previous)
        return
    cid = tid - box_capacity
    wid = contact_world[cid]
    if contact_local[cid] < world_contact_count[wid] and world_active[wid] and not (recycle and history[wid] == 1):
        initialize_contact_nesterov_row(
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
    read: wp.int32,
    restart_dot: wp.array2d[wp.float32],
):
    """Accumulate ``(lambda - trial) . (lambda - previous)`` per world, into the ``read`` buffer."""
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        if is_active_row(box_local[row], wid, world_box_count, world_active, world_failed):
            wp.atomic_add(restart_dot, read, wid, box_restart_dot(row, box_reaction, box_trial, box_previous))
        return
    cid = tid - box_capacity
    wid = contact_world[cid]
    if is_active_row(contact_local[cid], wid, world_contact_count, world_active, world_failed):
        wp.atomic_add(
            restart_dot,
            read,
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
def _extrapolate_reactions_and_twists(
    # Inputs:
    box_capacity: wp.int32,
    box_world: wp.array[wp.int32],
    box_local: wp.array[wp.int32],
    world_box_count: wp.array[wp.int32],
    contact_capacity: wp.int32,
    contact_world: wp.array[wp.int32],
    contact_local: wp.array[wp.int32],
    world_contact_count: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    world_body_offset: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    read: wp.int32,
    # Outputs:
    theta: wp.array2d[wp.float32],
    restart_dot: wp.array2d[wp.float32],
    box_reaction: wp.array[wp.float32],
    box_trial: wp.array[wp.float32],
    box_previous: wp.array[wp.float32],
    contact_reaction: wp.array[wp.vec3f],
    contact_trial: wp.array[wp.vec3f],
    contact_previous: wp.array[wp.vec3f],
    angular_reaction: wp.array[wp.vec3f],
    angular_trial: wp.array[wp.vec3f],
    angular_previous: wp.array[wp.vec3f],
    projected_twist: wp.array[vec6f],
    twist_previous: wp.array[vec6f],
):
    """Extrapolate each reaction and each body twist by the momentum of its world.

    Every thread derives the momentum of its world from the ``read`` buffers of ``theta`` and ``restart_dot``;
    the first body of each world writes the next ``theta`` and clears the restart dot of the next sweep, in the
    other buffers.
    """
    tid = wp.tid()
    if tid < box_capacity:
        row = tid
        wid = box_world[row]
        if is_active_row(box_local[row], wid, world_box_count, world_active, world_failed):
            _next_theta, beta = nesterov_momentum(theta[read, wid], restart_dot[read, wid])
            extrapolate_box_nesterov_row(row, beta, box_reaction, box_trial, box_previous)
        return
    cid = tid - box_capacity
    if cid < contact_capacity:
        wid = contact_world[cid]
        if is_active_row(contact_local[cid], wid, world_contact_count, world_active, world_failed):
            _next_theta, beta = nesterov_momentum(theta[read, wid], restart_dot[read, wid])
            extrapolate_contact_nesterov_row(
                cid,
                beta,
                contact_reaction,
                contact_trial,
                contact_previous,
                angular_reaction,
                angular_trial,
                angular_previous,
            )
        return
    bid = cid - contact_capacity
    wid = body_world[bid]
    if world_active[wid] and not world_failed[wid]:
        next_theta, beta = nesterov_momentum(theta[read, wid], restart_dot[read, wid])
        extrapolate_body_twist(bid, beta, projected_twist, twist_previous)
        if bid == world_body_offset[wid]:
            theta[1 - read, wid] = next_theta
            restart_dot[1 - read, wid] = 0.0


@wp.kernel
def _difference_twists(
    # Inputs:
    body_world: wp.array[wp.int32],
    world_active: wp.array[wp.bool],
    world_failed: wp.array[wp.bool],
    projected_twist: wp.array[vec6f],
    # Outputs:
    twist_previous: wp.array[vec6f],
):
    """Replace each previous twist by its difference with the projected twist."""
    bid = wp.tid()
    wid = body_world[bid]
    if world_active[wid] and not world_failed[wid]:
        twist_previous[bid] = projected_twist[bid] - twist_previous[bid]


###
# Interfaces
###


class NesterovAcceleration:
    """Own the momentum state of Nesterov-accelerated projection."""

    def __init__(self, problem_data: LOXProblemData, data: NesterovData):
        self.problem_data = problem_data
        self._data = data
        self._capacity = problem_data.box_rows.capacity + problem_data.contact_rows.capacity
        self.row_state = NesterovRowState()
        """Trial and previous reactions, for the per-world projection."""
        for name in (
            "box_trial",
            "box_previous",
            "contact_trial",
            "contact_previous",
            "history",
            "twist_previous",
        ):
            setattr(self.row_state, name, getattr(data, name))
        # The per-world projection keeps theta in registers and stores it in the first buffer
        self.row_state.theta = data.theta[0]
        if data.angular_trial is not None:
            self.row_state.angular_trial = data.angular_trial
            self.row_state.angular_previous = data.angular_previous

    ###
    # Public API
    ###

    def reset(self) -> None:
        """Restart the momentum and the row history of every world, once per LOX solve."""
        self._data.theta.fill_(1.0)
        self._data.history.zero_()
        self._data.twist_previous.zero_()

    def begin(self, world_active: wp.array[wp.bool], extrapolations: int, recycle: bool = False) -> None:
        """Clamp the warm-start reactions and restart the row history.

        With ``recycle``, the worlds keep ``theta`` and the row history of the previous splitting iteration,
        whose deferred extrapolation :meth:`extrapolate` then applies after the warm start and
        :meth:`begin_twists`.

        Args:
            world_active: Worlds to project.
            extrapolations: Number of :meth:`extrapolate` calls of the projection.
            recycle: Whether the worlds keep the momentum and row history of the previous splitting iteration.
        """
        # The buffers alternate at each extrapolation, starting so that the last one writes the first buffer
        self._extrapolations = extrapolations
        self._extrapolation = 0
        p = self.problem_data
        angular_reaction, angular_friction, _frame = self._spatial_arrays()
        if self._capacity > 0:
            self._initialize_reactions(world_active, recycle, angular_reaction, angular_friction)
        # After the rows, which read the history flags of the previous splitting iteration
        wp.launch(
            _initialize_worlds,
            dim=p.num_worlds,
            inputs=[world_active, recycle, extrapolations % 2],
            outputs=[self._data.theta, self._data.restart_dot, self._data.history],
            device=p.device,
        )

    def _initialize_reactions(self, world_active, recycle, angular_reaction, angular_friction) -> None:
        p = self.problem_data
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
                recycle,
                self._data.history,
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

    def begin_twists(
        self,
        body_world: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        recycle: bool,
        projected_twist: wp.array[vec6f],
    ) -> None:
        """Set the previous twists of the warm-started reactions, after the warm start.

        With ``recycle``, the previous twists hold the change of the last sweep of the previous splitting
        iteration (:meth:`end_twists`), and become the warm-started twists minus that change.
        """
        if not recycle:
            wp.copy(self._data.twist_previous, projected_twist)
            return
        self._difference_twists(body_world, world_active, world_failed, projected_twist)

    def end_twists(
        self,
        body_world: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        projected_twist: wp.array[vec6f],
    ) -> None:
        """Keep the twist change of the last sweep, for the deferred extrapolation of a recycled projection."""
        self._difference_twists(body_world, world_active, world_failed, projected_twist)

    def extrapolate(
        self,
        body_world: wp.array[wp.int32],
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        projected_twist: wp.array[vec6f],
    ) -> None:
        """Update the momentum and extrapolate the reactions and the body twists of the last sweep."""
        p = self.problem_data
        read = (self._extrapolations + self._extrapolation) % 2
        self._extrapolation += 1
        angular_reaction, angular_friction, _frame = self._spatial_arrays()
        if self._capacity > 0:
            self._accumulate_restart_dot(world_active, world_failed, read, angular_reaction, angular_friction)
        wp.launch(
            _extrapolate_reactions_and_twists,
            dim=self._capacity + projected_twist.shape[0],
            inputs=[
                p.box_rows.capacity,
                p.box_rows.world,
                p.box_rows.local,
                p.box_rows.world_count,
                p.contact_rows.capacity,
                p.contact_rows.world,
                p.contact_rows.local,
                p.contact_rows.world_count,
                body_world,
                p.world_body_offset,
                world_active,
                world_failed,
                read,
            ],
            outputs=[
                self._data.theta,
                self._data.restart_dot,
                p.box_rows.reaction,
                self._data.box_trial,
                self._data.box_previous,
                p.contact_rows.reaction,
                self._data.contact_trial,
                self._data.contact_previous,
                angular_reaction,
                self._data.angular_trial,
                self._data.angular_previous,
                projected_twist,
                self._data.twist_previous,
            ],
            device=p.device,
        )

    def _accumulate_restart_dot(self, world_active, world_failed, read, angular_reaction, angular_friction) -> None:
        p = self.problem_data
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
                read,
            ],
            outputs=[self._data.restart_dot],
            device=p.device,
        )

    ###
    # Internals
    ###

    def _difference_twists(self, body_world, world_active, world_failed, projected_twist) -> None:
        wp.launch(
            _difference_twists,
            dim=projected_twist.shape[0],
            inputs=[body_world, world_active, world_failed, projected_twist],
            outputs=[self._data.twist_previous],
            device=self.problem_data.device,
        )

    def _spatial_arrays(self):
        """Return the angular reaction, angular friction, and frame of spatial contacts, or ``None``."""
        rows = self.problem_data.contact_rows
        return rows.angular_reaction, rows.angular_friction, rows.frame
