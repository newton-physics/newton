# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent state for LOX splitting iterations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from ...core.types import mat66f, vec6f
from .kernels import (
    _finalize_residual_iteration,
    _initialize_bodies,
    _initialize_fixed_iteration,
    _initialize_iteration_residuals,
    _initialize_worlds,
    _mark_iteration_limit,
    _prepare_projection,
    _restore_dual_from_impulse,
    _store_dual_impulse,
    _update_bodies_and_reduce_residuals,
    _update_fixed_iteration_bodies,
)

__all__ = ["SplittingState"]


class SplittingState:
    """Persistent body/world state for a batched LOX solve."""

    def __init__(self, body_counts: Sequence[int], device: wp.DeviceLike = None):
        if len(body_counts) == 0:
            raise ValueError("At least one world is required.")
        if any(not isinstance(count, int) or count < 0 for count in body_counts) or sum(body_counts) == 0:
            raise ValueError("Body counts must be non-negative and include at least one active body.")

        self.device = wp.get_device(device)
        self.body_counts = tuple(body_counts)
        self.num_worlds = len(body_counts)
        self.num_bodies = sum(body_counts)
        body_world = np.repeat(
            np.arange(self.num_worlds, dtype=np.int32),
            np.asarray(body_counts, dtype=np.int32),
        )

        self.body_world = wp.array(body_world, dtype=wp.int32, device=self.device)
        self.projected_twist = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.projected_twist_previous = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.global_twist = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.global_twist_previous = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.splitting_dual = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.splitting_dual_impulse = wp.zeros(self.num_bodies, dtype=vec6f, device=self.device)
        self.world_active = wp.ones(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_converged = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_failed = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.world_iteration_limit = wp.zeros(self.num_worlds, dtype=wp.bool, device=self.device)
        self.iteration_count = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)
        self.residual_change = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_split = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_structural = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_structural_projected = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_cross_iterate = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_lagged_velocity = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self.residual_total = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)
        self._iteration_failed = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.device)

    def _initialize(
        self,
        world_mask: wp.array[wp.bool] | None,
        initial_twist: wp.array[vec6f] | None,
        reset_dual: bool,
    ) -> None:
        wp.launch(
            _initialize_bodies,
            dim=self.num_bodies,
            inputs=[self.body_world, world_mask, initial_twist, reset_dual],
            outputs=[
                self.projected_twist,
                self.projected_twist_previous,
                self.global_twist,
                self.global_twist_previous,
                self.splitting_dual,
                self.splitting_dual_impulse,
            ],
            device=self.device,
        )
        wp.launch(
            _initialize_worlds,
            dim=self.num_worlds,
            inputs=[world_mask],
            outputs=[
                self.world_active,
                self.world_converged,
                self.world_failed,
                self.world_iteration_limit,
                self.iteration_count,
                self.residual_change,
                self.residual_split,
                self.residual_structural,
                self.residual_structural_projected,
                self.residual_cross_iterate,
                self.residual_lagged_velocity,
                self.residual_total,
                self._iteration_failed,
            ],
            device=self.device,
        )

    def reset(self, world_mask: wp.array[wp.bool] | None = None) -> None:
        """Reset body-space warm starts and world diagnostics."""
        if world_mask is not None and world_mask.shape[0] != self.num_worlds:
            raise ValueError("world_mask must contain one entry per world.")
        self._initialize(world_mask, initial_twist=None, reset_dual=True)

    def begin(self, initial_twist: wp.array[vec6f], reset_dual: bool = False) -> None:
        """Initialize one nonlinear solve while optionally retaining its impulse warm start."""
        if initial_twist.shape[0] != self.num_bodies:
            raise ValueError("initial_twist must contain one entry per packed active body.")
        self._initialize(world_mask=None, initial_twist=initial_twist, reset_dual=reset_dual)

    def store_dual_impulse(
        self,
        weight: wp.array[mat66f],
        body_has_unilateral: wp.array[wp.int32],
    ) -> None:
        """Store the physical body impulse ``W u`` for later warm starting."""
        if weight.shape[0] != self.num_bodies or body_has_unilateral.shape[0] != self.num_bodies:
            raise ValueError("Weight and unilateral mask arrays must contain one entry per body.")
        wp.launch(
            _store_dual_impulse,
            dim=self.num_bodies,
            inputs=[body_has_unilateral, weight],
            outputs=[self.splitting_dual, self.splitting_dual_impulse],
            device=self.device,
        )

    def restore_dual_from_impulse(
        self,
        inverse_weight: wp.array[mat66f],
        body_has_unilateral: wp.array[wp.int32],
    ) -> None:
        """Recover the scaled dual ``u = W^-1 (W u)`` for the current weight."""
        if inverse_weight.shape[0] != self.num_bodies or body_has_unilateral.shape[0] != self.num_bodies:
            raise ValueError("Inverse weight and unilateral mask arrays must contain one entry per body.")
        wp.launch(
            _restore_dual_from_impulse,
            dim=self.num_bodies,
            inputs=[body_has_unilateral, inverse_weight],
            outputs=[self.splitting_dual_impulse, self.splitting_dual],
            device=self.device,
        )

    def prepare_projection(
        self,
        global_solution: wp.array[vec6f],
        body_split_enabled: wp.array[wp.int32] | None = None,
    ) -> None:
        """Store the global solution and initialize ``p = v - u``."""
        if global_solution.shape[0] != self.num_bodies:
            raise ValueError("global_solution must contain one entry per packed active body.")
        if body_split_enabled is not None and body_split_enabled.shape[0] != self.num_bodies:
            raise ValueError("body_split_enabled must contain one entry per packed active body.")
        wp.launch(
            _prepare_projection,
            dim=self.num_bodies,
            inputs=[self.body_world, self.world_active, body_split_enabled, global_solution],
            outputs=[
                self.splitting_dual,
                self.global_twist_previous,
                self.global_twist,
                self.projected_twist_previous,
                self.projected_twist,
            ],
            device=self.device,
        )

    def finish_iteration(
        self,
        projection_status: wp.array[wp.int32],
        time_step: wp.array[wp.float32],
        position_tolerance: float,
        rotation_tolerance: float,
        velocity_tolerance: float,
        effort_residual: wp.array[wp.float32] | None = None,
        structural_residual: wp.array[wp.float32] | None = None,
        projected_structural_residual: wp.array[wp.float32] | None = None,
        lagged_velocity_residual: wp.array[wp.float32] | None = None,
        lagged_velocity_required: wp.array[wp.int32] | None = None,
    ) -> None:
        """Update duals and residuals, then test per-world convergence."""
        if any(
            not np.isfinite(tolerance) or tolerance <= 0.0
            for tolerance in (position_tolerance, rotation_tolerance, velocity_tolerance)
        ):
            raise ValueError("Convergence tolerances must be finite and positive.")
        wp.launch(
            _initialize_iteration_residuals,
            dim=self.num_worlds,
            inputs=[
                projection_status,
            ],
            outputs=[
                self.world_active,
                self.world_failed,
                self.iteration_count,
                self._iteration_failed,
                self.residual_change,
                self.residual_split,
                self.residual_cross_iterate,
            ],
            device=self.device,
        )
        wp.launch(
            _update_bodies_and_reduce_residuals,
            dim=self.num_bodies,
            inputs=[
                time_step,
                position_tolerance,
                rotation_tolerance,
                velocity_tolerance,
                self.body_world,
                self.global_twist_previous,
                self.global_twist,
                self.projected_twist_previous,
                self.projected_twist,
                self.world_active,
            ],
            outputs=[
                self.splitting_dual,
                self._iteration_failed,
                self.residual_change,
                self.residual_split,
                self.residual_cross_iterate,
            ],
            device=self.device,
        )
        wp.launch(
            _finalize_residual_iteration,
            dim=self.num_worlds,
            inputs=[
                structural_residual,
                projected_structural_residual,
                lagged_velocity_residual,
                lagged_velocity_required,
                effort_residual,
                self.iteration_count,
                self._iteration_failed,
            ],
            outputs=[
                self.world_active,
                self.world_converged,
                self.world_failed,
                self.residual_change,
                self.residual_split,
                self.residual_structural,
                self.residual_structural_projected,
                self.residual_cross_iterate,
                self.residual_lagged_velocity,
                self.residual_total,
            ],
            device=self.device,
        )

    def finish_fixed_iteration(
        self,
        projection_status: wp.array[wp.int32],
    ) -> None:
        """Update the dual state and failures for a fixed-count iteration."""
        wp.launch(
            _initialize_fixed_iteration,
            dim=self.num_worlds,
            inputs=[projection_status],
            outputs=[self.world_active, self.world_failed, self.iteration_count],
            device=self.device,
        )
        wp.launch(
            _update_fixed_iteration_bodies,
            dim=self.num_bodies,
            inputs=[self.body_world, self.global_twist, self.projected_twist],
            outputs=[self.world_active, self.world_failed, self.splitting_dual],
            device=self.device,
        )

    def mark_iteration_limit(self) -> None:
        """Deactivate worlds that remain unconverged at the iteration limit."""
        wp.launch(
            _mark_iteration_limit,
            dim=self.num_worlds,
            inputs=[],
            outputs=[self.world_active, self.world_iteration_limit],
            device=self.device,
        )
