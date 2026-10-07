# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Jacobi and colored Gauss--Seidel schedules of the LOX projection sweeps.

Both schedules share the same row updates (see :mod:`.projection_kernels`).
Jacobi projects every row in each sweep; colored Gauss--Seidel projects one
color at a time, applying the body corrections between colors, and finishes
with one Jacobi smoothing sweep. The ordered color sweeps favor the rows of the
last colors; the order-independent smoothing sweep restores the symmetry of the
result, e.g. between identical contacts of a symmetric body.

Each sweep is one row-parallel launch. When the worlds fill the device or their
rows are few, one thread block runs every sweep of a world, in a single launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...core.types import mat66f, vec6f
from .anderson import AndersonProjection
from .apgd import APGDAcceleration
from .coloring import RowColoring
from .projection_kernels import (
    _apply_twist_delta,
    _prepare_box_rows,
    _prepare_contacts,
    _prepare_physical_box_rows,
    _prepare_physical_contacts,
    _prepare_physical_spatial_contacts,
    _prepare_spatial_contacts,
    _warmstart_box_rows,
    _warmstart_contacts,
    _warmstart_spatial_contacts,
    make_sweep_kernel,
)
from .types import ProjectionAcceleration, ProjectionData
from .world_projection_kernels import NoAccelerationState, can_project_by_world, make_project_by_world_kernel

if TYPE_CHECKING:
    from .types import LOXProblemData

###
# Module interface
###

__all__ = [
    "ProjectionSchedule",
]

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Constants
###


_SWEEP_BLOCK_DIM = 128


_JACOBI_BLOCKS_PER_SM = 5


_COLORED_BLOCKS_PER_SM = 2


_JACOBI_WORLD_BLOCK_DIM = 128


_COLORED_WORLD_BLOCK_DIM = 64


###
# Interfaces
###


class ProjectionSchedule:
    """Prepare and run the projection sweeps of one LOX problem.

    Args:
        problem_data: Box rows, contacts, and body incidence of the LOX problem.
        color_count: Maximum number of Gauss--Seidel colors; one selects Jacobi.
        acceleration: Acceleration of the projection sweeps; APGD requires the Jacobi schedule.
    """

    def __init__(
        self,
        problem_data: LOXProblemData,
        color_count: int,
        acceleration: ProjectionAcceleration = ProjectionAcceleration.NONE,
    ):
        self.problem_data = problem_data
        self.device = problem_data.device
        self._data = ProjectionData(problem_data, color_count, acceleration)
        self._acceleration = acceleration
        data = self._data
        self.coloring = RowColoring(problem_data, data.coloring) if data.coloring is not None else None
        self.color_count = data.coloring.color_count if data.coloring is not None else 1
        colored = self.coloring is not None
        self.apgd = APGDAcceleration(problem_data, data.apgd) if data.apgd is not None else None
        self.anderson = AndersonProjection(problem_data, data.anderson) if data.anderson is not None else None

        row_capacity = problem_data.box_rows.capacity + problem_data.contact_rows.capacity
        blocks_per_sm = _COLORED_BLOCKS_PER_SM if colored else _JACOBI_BLOCKS_PER_SM
        self._worker_count = row_capacity
        if self.device.is_cuda:
            max_workers = max(_SWEEP_BLOCK_DIM, self.device.sm_count * blocks_per_sm * _SWEEP_BLOCK_DIM)
            self._worker_count = min(row_capacity, max_workers)
        if colored:
            self._world_block_dim = _COLORED_WORLD_BLOCK_DIM
            self._by_world = can_project_by_world(
                self.device,
                problem_data.num_worlds,
                (row_capacity + self.color_count - 1) // self.color_count,
                _COLORED_WORLD_BLOCK_DIM,
                min_blocks_per_sm=1,
            )
        else:
            self._world_block_dim = _JACOBI_WORLD_BLOCK_DIM
            self._by_world = can_project_by_world(
                self.device, problem_data.num_worlds, row_capacity, _JACOBI_WORLD_BLOCK_DIM
            )
        if data.coloring is not None and not self._by_world:
            # Only the per-world sweeps read the rows of each color of each world
            data.coloring.world_color_count = None
            data.coloring.world_color_cursor = None
            data.coloring.world_order = None

    ###
    # Properties
    ###

    @property
    def data(self) -> ProjectionData:
        """Returns the local blocks, the sweep buffer, and the coloring or acceleration state."""
        return self._data

    ###
    # Public API
    ###

    def prepare(self, inverse_weight: wp.array[mat66f], world_failed: wp.array[wp.bool]) -> None:
        """Color the rows and prepare the local blocks for the current body weights.

        Args:
            inverse_weight: Inverse body weight ``W^-1`` of each body.
            world_failed: Failure flag of each world, set for worlds with an invalid local block.
        """
        p = self.problem_data
        if self.coloring is not None:
            self.coloring.build()
            self._prepare_blocks(
                inverse_weight,
                self._data.coloring.occupancy,
                self._data.coloring.box.color,
                self._data.coloring.contact.color,
                self._data.box_delassus,
                self._data.contact_delassus,
                self._data.spatial_blocks,
                world_failed,
            )
            self._prepare_blocks(
                inverse_weight,
                p.body_incidence,
                None,
                None,
                self._data.box_smoothing_delassus,
                self._data.contact_smoothing_delassus,
                self._data.spatial_smoothing_blocks,
                world_failed,
            )
        else:
            self._prepare_blocks(
                inverse_weight,
                p.body_incidence,
                None,
                None,
                self._data.box_delassus,
                self._data.contact_delassus,
                self._data.spatial_blocks,
                world_failed,
            )
        self._prepare_physical_blocks(inverse_weight)
        if self.anderson is not None:
            self.anderson.reset()

    def project(
        self,
        iterations: int,
        recycle: bool,
        world_active: wp.array[wp.bool],
        world_failed: wp.array[wp.bool],
        body_world: wp.array[wp.int32],
        inverse_weight: wp.array[mat66f],
        projected_twist: wp.array[vec6f],
    ) -> None:
        """Warm start with the current reactions and run the projection sweeps.

        Worlds that already failed are skipped, and any failed local update marks
        its world in ``world_failed``.

        Args:
            iterations: Number of sweeps; a colored sweep visits every color once.
            recycle: Whether Anderson retains its direction from the previous splitting iteration.
            world_active: Worlds to project.
            world_failed: Failure flag of each world.
            body_world: World of each body.
            inverse_weight: Inverse body weight ``W^-1`` of each body.
            projected_twist: Projected body twists, updated in place.
        """
        self._begin_outer_iteration(world_active, recycle=recycle)
        if self._by_world:
            self._project_by_world(iterations, world_active, inverse_weight, projected_twist, world_failed)
            return

        if self.apgd is not None:
            self.apgd.begin(world_active)
        self._warmstart(world_active, inverse_weight, world_failed)
        self._apply(body_world, world_active, world_failed, projected_twist)

        if self.anderson is not None:
            self.anderson.begin_map(world_active, world_failed)
        for iteration in range(iterations):
            for color in range(self.color_count):
                self._sweep(color, world_active, inverse_weight, projected_twist, world_failed)
                self._apply(body_world, world_active, world_failed, projected_twist)
            if self.anderson is not None:
                self.anderson.finish_map(world_active, world_failed, inverse_weight, self._data.twist_delta)
                self._apply(body_world, world_active, world_failed, projected_twist)
            if self.apgd is not None and iteration + 1 < iterations:
                self.apgd.extrapolate(world_active, world_failed, inverse_weight, self._data.twist_delta)
                self._apply(body_world, world_active, world_failed, projected_twist)
        if self.coloring is not None:
            # An order-independent Jacobi sweep restores the symmetry lost to the color ordering.
            self._sweep(0, world_active, inverse_weight, projected_twist, world_failed, smoothing=True)
            self._apply(body_world, world_active, world_failed, projected_twist)

    ###
    # Internals
    ###

    def _prepare_physical_blocks(self, inverse_weight: wp.array[mat66f]) -> None:
        """Prepare the unsplit Delassus blocks that scale the residuals of the convergence check."""
        p = self.problem_data
        if p.box_rows.capacity > 0:
            rows = p.box_rows
            wp.launch(
                _prepare_physical_box_rows,
                dim=rows.capacity,
                inputs=[
                    rows.world,
                    rows.local,
                    rows.world_count,
                    rows.body_a,
                    rows.body_b,
                    rows.jacobian_a,
                    rows.jacobian_b,
                    inverse_weight,
                ],
                outputs=[rows.physical_delassus],
                device=self.device,
            )
        if p.contact_rows.capacity == 0:
            return
        rows = p.contact_rows
        if self._data.spatial_blocks is None:
            wp.launch(
                _prepare_physical_contacts,
                dim=rows.capacity,
                inputs=[
                    rows.world,
                    rows.local,
                    rows.world_count,
                    rows.body_a,
                    rows.body_b,
                    rows.jacobian_a,
                    rows.jacobian_b,
                    rows.normal_compliance,
                    inverse_weight,
                ],
                outputs=[rows.physical_delassus],
                device=self.device,
            )
        else:
            wp.launch(
                _prepare_physical_spatial_contacts,
                dim=rows.capacity,
                inputs=[
                    rows.world,
                    rows.local,
                    rows.world_count,
                    rows.body_a,
                    rows.body_b,
                    rows.jacobian_a,
                    rows.jacobian_b,
                    rows.frame,
                    rows.normal_compliance,
                    inverse_weight,
                ],
                outputs=[rows.physical_delassus],
                device=self.device,
            )

    def _begin_outer_iteration(self, world_active: wp.array[wp.bool], recycle: bool) -> None:
        """Restart the Anderson samples of a new splitting iteration."""
        if self.anderson is not None:
            self.anderson.begin_outer_iteration(world_active, recycle=recycle)

    def _prepare_blocks(
        self,
        inverse_weight,
        occupancy,
        box_color,
        contact_color,
        box_delassus,
        contact_delassus,
        spatial_blocks,
        world_failed,
    ):
        p = self.problem_data
        if p.box_rows.capacity > 0:
            wp.launch(
                _prepare_box_rows,
                dim=p.box_rows.capacity,
                inputs=[
                    p.box_rows.world,
                    p.box_rows.local,
                    p.box_rows.world_count,
                    box_color,
                    p.box_rows.body_a,
                    p.box_rows.body_b,
                    p.box_rows.jacobian_a,
                    p.box_rows.jacobian_b,
                    occupancy,
                    inverse_weight,
                ],
                outputs=[box_delassus, world_failed],
                device=self.device,
            )
        if spatial_blocks is not None and p.contact_rows.capacity > 0:
            rows = p.contact_rows
            wp.launch(
                _prepare_spatial_contacts,
                dim=rows.capacity,
                inputs=[
                    rows.world,
                    rows.local,
                    rows.world_count,
                    contact_color,
                    rows.body_a,
                    rows.body_b,
                    rows.jacobian_a,
                    rows.jacobian_b,
                    rows.frame,
                    rows.friction,
                    rows.angular_friction,
                    rows.normal_compliance,
                    occupancy,
                    inverse_weight,
                ],
                outputs=[
                    spatial_blocks.delassus,
                    spatial_blocks.eigenvectors,
                    spatial_blocks.eigenvalues,
                    world_failed,
                ],
                device=self.device,
            )
        elif p.contact_rows.capacity > 0:
            wp.launch(
                _prepare_contacts,
                dim=p.contact_rows.capacity,
                inputs=[
                    p.contact_rows.world,
                    p.contact_rows.local,
                    p.contact_rows.world_count,
                    contact_color,
                    p.contact_rows.body_a,
                    p.contact_rows.body_b,
                    p.contact_rows.jacobian_a,
                    p.contact_rows.jacobian_b,
                    p.contact_rows.bias,
                    p.contact_rows.friction,
                    p.contact_rows.normal_compliance,
                    occupancy,
                    inverse_weight,
                ],
                outputs=[contact_delassus, world_failed],
                device=self.device,
            )

    def _warmstart(self, world_active, inverse_weight, world_failed) -> None:
        p = self.problem_data
        if p.box_rows.capacity > 0:
            wp.launch(
                _warmstart_box_rows,
                dim=p.box_rows.capacity,
                inputs=[
                    p.box_rows.world,
                    p.box_rows.local,
                    p.box_rows.world_count,
                    p.box_rows.body_a,
                    p.box_rows.body_b,
                    p.box_rows.jacobian_a,
                    p.box_rows.jacobian_b,
                    p.box_rows.reaction,
                    world_active,
                    world_failed,
                    inverse_weight,
                ],
                outputs=[self._data.twist_delta],
                device=self.device,
            )
        if p.contact_rows.capacity == 0:
            return
        rows = p.contact_rows
        if self._data.spatial_blocks is not None:
            wp.launch(
                _warmstart_spatial_contacts,
                dim=rows.capacity,
                inputs=[
                    rows.world,
                    rows.local,
                    rows.world_count,
                    rows.body_a,
                    rows.body_b,
                    rows.jacobian_a,
                    rows.jacobian_b,
                    rows.frame,
                    rows.friction,
                    rows.angular_friction,
                    world_active,
                    world_failed,
                    inverse_weight,
                ],
                outputs=[rows.reaction, rows.angular_reaction, self._data.twist_delta],
                device=self.device,
            )
        else:
            wp.launch(
                _warmstart_contacts,
                dim=p.contact_rows.capacity,
                inputs=[
                    p.contact_rows.world,
                    p.contact_rows.local,
                    p.contact_rows.world_count,
                    p.contact_rows.body_a,
                    p.contact_rows.body_b,
                    p.contact_rows.jacobian_a,
                    p.contact_rows.jacobian_b,
                    p.contact_rows.reaction,
                    world_active,
                    world_failed,
                    inverse_weight,
                ],
                outputs=[self._data.twist_delta],
                device=self.device,
            )

    def _sweep(self, color, world_active, inverse_weight, projected_twist, world_failed, smoothing=False) -> None:
        p = self.problem_data
        if self._worker_count == 0:
            return
        colored = self.coloring is not None and not smoothing
        box = self._data.coloring.box if colored else None
        contact = self._data.coloring.contact if colored else None
        spatial = self._data.spatial_smoothing_blocks if smoothing else self._data.spatial_blocks
        wp.launch(
            make_sweep_kernel(spatial is not None),
            dim=self._worker_count,
            inputs=[
                self._worker_count,
                color,
                box.order if colored else None,
                box.color_offset if colored else None,
                box.color_count if colored else None,
                p.box_rows.world,
                p.box_rows.local,
                p.box_rows.world_count,
                p.box_rows.body_a,
                p.box_rows.body_b,
                p.box_rows.jacobian_a,
                p.box_rows.jacobian_b,
                p.box_rows.bias,
                p.box_rows.lower,
                p.box_rows.upper,
                self._data.box_smoothing_delassus if smoothing else self._data.box_delassus,
                contact.order if colored else None,
                contact.color_offset if colored else None,
                contact.color_count if colored else None,
                p.contact_rows.world,
                p.contact_rows.local,
                p.contact_rows.world_count,
                p.contact_rows.body_a,
                p.contact_rows.body_b,
                p.contact_rows.jacobian_a,
                p.contact_rows.jacobian_b,
                p.contact_rows.bias,
                p.contact_rows.friction,
                self._data.contact_smoothing_delassus if smoothing else self._data.contact_delassus,
                p.contact_rows.normal_compliance,
                p.contact_rows.restitution,
                p.contact_rows.frame,
                p.contact_rows.angular_friction,
                spatial.delassus if spatial is not None else None,
                spatial.eigenvectors if spatial is not None else None,
                spatial.eigenvalues if spatial is not None else None,
                world_active,
                inverse_weight,
                self._data.coloring.occupancy if colored else p.body_incidence,
                projected_twist,
            ],
            outputs=[
                p.box_rows.reaction,
                p.contact_rows.reaction,
                p.contact_rows.angular_reaction,
                self._data.twist_delta,
                world_failed,
            ],
            device=self.device,
            block_dim=_SWEEP_BLOCK_DIM,
        )

    def _apply(self, body_world, world_active, world_failed, projected_twist) -> None:
        wp.launch(
            _apply_twist_delta,
            dim=projected_twist.shape[0],
            inputs=[body_world, world_active, world_failed],
            outputs=[self._data.twist_delta, projected_twist],
            device=self.device,
        )

    def _project_by_world(self, iterations, world_active, inverse_weight, projected_twist, world_failed) -> None:
        p = self.problem_data
        coloring = self._data.coloring
        spatial = self._data.spatial_blocks
        smoothing = self._data.spatial_smoothing_blocks
        acceleration_state = NoAccelerationState()
        if self.anderson is not None:
            acceleration_state = self.anderson.world_state
        elif self.apgd is not None:
            acceleration_state = self.apgd.row_state
        wp.launch(
            make_project_by_world_kernel(spatial is not None, self.coloring is not None, self._acceleration),
            dim=(p.num_worlds, self._world_block_dim),
            block_dim=self._world_block_dim,
            inputs=[
                iterations,
                self.color_count,
                p.world_body_offset,
                p.world_body_count,
                p.box_rows.capacity,
                p.box_rows.world_offset,
                p.box_rows.world_count,
                p.box_rows.body_a,
                p.box_rows.body_b,
                p.box_rows.jacobian_a,
                p.box_rows.jacobian_b,
                p.box_rows.bias,
                p.box_rows.lower,
                p.box_rows.upper,
                self._data.box_delassus,
                self._data.box_smoothing_delassus,
                p.contact_rows.world_offset,
                p.contact_rows.world_count,
                p.contact_rows.body_a,
                p.contact_rows.body_b,
                p.contact_rows.jacobian_a,
                p.contact_rows.jacobian_b,
                p.contact_rows.bias,
                p.contact_rows.friction,
                self._data.contact_delassus,
                self._data.contact_smoothing_delassus,
                p.contact_rows.normal_compliance,
                p.contact_rows.restitution,
                p.contact_rows.frame,
                p.contact_rows.angular_friction,
                spatial.delassus if spatial is not None else None,
                spatial.eigenvectors if spatial is not None else None,
                spatial.eigenvalues if spatial is not None else None,
                smoothing.delassus if smoothing is not None else None,
                smoothing.eigenvectors if smoothing is not None else None,
                smoothing.eigenvalues if smoothing is not None else None,
                acceleration_state,
                coloring.world_color_count if coloring is not None else None,
                coloring.world_order if coloring is not None else None,
                coloring.occupancy if coloring is not None else None,
                p.body_incidence,
                world_active,
                inverse_weight,
            ],
            outputs=[
                p.box_rows.reaction,
                p.contact_rows.reaction,
                p.contact_rows.angular_reaction,
                projected_twist,
                self._data.twist_delta,
                world_failed,
            ],
            device=self.device,
        )
