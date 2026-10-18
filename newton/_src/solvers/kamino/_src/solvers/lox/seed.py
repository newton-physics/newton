# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint penalty seed of the LOX structural rows.

The seed sets the structural ALM penalty scale of each joint-connected component from the
extreme eigenvalues of its effective-mass-normalized structural Delassus. A float32 Golub-Kahan
bidiagonalization estimates them for every component at once on the device; its kernels live in
:mod:`.seed_kernels`.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import to_warp_int32_array
from ...utils import logger as msg
from .seed_kernels import (
    _CERTIFICATION_CANDIDATES,
    SEED_STATUS_NO_POSITIVE_EIGENVALUE,
    SEED_STATUS_NONFINITE,
    SEED_STATUS_NOT_REPRESENTABLE,
    SEED_STATUS_VALID,
    _apply_structural_transpose,
    _bidiagonal_step,
    _estimate_block_scales,
    _exclude_inertia_free_dofs,
    _fill_random_vector,
    _normalize_force,
    _start_bidiagonalization,
    _stop_bounded_blocks,
    _write_body_scales,
)
from .system import LOXSystem
from .types import StructuralRows

###
# Module interface
###

__all__ = ["estimate_joint_penalty_scale"]


###
# Constants
###

_LANCZOS_STEPS_PER_ROW = 3
"""Bidiagonalization steps per structural row of a component.

Without reorthogonalization, float32 Lanczos vectors lose orthogonality, and the smallest
eigenvalue takes more steps than the dimension to converge. Three passes over the rows
resolve the seed of the robot models to about 1e-3.
"""

_BOUND_CHECK_INTERVAL = 8
"""Fewest bidiagonalization steps between the checks that stop the components whose seed takes the ratio cap."""

_STEP_BLOCK_DIM = 256
"""Threads that share the rows and DOFs of one component in a bidiagonalization step."""


###
# Functions
###


def estimate_joint_penalty_scale(
    system: LOXSystem,
    structural_rows: StructuralRows,
    scales: wp.array[wp.float32],
    min_ratio: float,
    max_ratio: float,
) -> None:
    """Set the penalty scale of each body from the structural spectrum of its joint-connected component.

    For each body component, the structural Delassus ``D = J A^+ J^T`` uses
    the pseudo-inverse of the assembled smooth matrix ``A`` and is normalized by
    the structural effective masses as ``S = M^1/2 D M^1/2``. Massless bodies leave
    DOFs without inertia in ``A``, which the estimate excludes. Each component takes the
    scale ``s = clamp(mu_max / mu_min, min_ratio, max_ratio) / mu_max`` of the smallest positive
    eigenvalue ``mu_min`` and the largest eigenvalue ``mu_max`` of ``S``. The spectrum of the
    penalized matrix relative to ``A`` is ``1 + s mu``, where ``s mu`` is the ratio of the penalty
    to the effective mass of the structural mode of eigenvalue ``mu``. Within the bounds, the
    penalty matches the effective mass of the heaviest mode, and the ratio ``s mu_max`` of the
    lightest mode is the conditioning ``mu_max / mu_min`` of ``S``; the penalty adds at most
    ``1 + max_ratio`` to the conditioning. Components whose estimate fails take the scale
    ``min_ratio``, with a warning. Bodies of components without structural rows keep their scale.

    A float32 Golub-Kahan bidiagonalization of ``J^T M^1/2``, with the ``A^-1`` inner product on
    the body forces, estimates the spectrum of every component at once on the device. Each step
    applies ``A^-1`` once with the batched factorization of the smooth system. The singular values
    ``sigma = sqrt(mu)`` of the bidiagonal carry rounding errors of about ``eps * sigma_max``, so
    the redundant structural rows, which form the null space of ``S``, stay near
    ``eps^2 * mu_max``. ``mu_min`` is the smallest bound ``(sigma + r)^2`` that a Ritz value with
    residual ``r`` certifies for a positive eigenvalue, so it bounds the smallest positive
    eigenvalue from above, and the seed approaches its exact value from below as the Ritz values
    converge. A component stops once a certified bound sets its scale to the cap, and after
    :data:`_LANCZOS_STEPS_PER_ROW` steps per structural row otherwise.

    The smooth system and the structural rows must be assembled, e.g. after
    :meth:`.LOXProblem.prepare_joint_penalty_scale_seed` and :meth:`.LOXProblem.build_smooth_system`.
    The estimate overwrites the smooth matrix and its factorization.

    Args:
        system: Smooth body system, with the assembled matrix of the seed.
        structural_rows: Structural rows, with their Jacobians and effective masses.
        scales: Dimensionless penalty scale of each body, updated in place.
        min_ratio: Smallest ratio of the penalty to the effective mass of the lightest structural mode, positive.
        max_ratio: Largest ratio of the penalty to the effective mass of the lightest structural mode, at least
            ``min_ratio``.
    """
    device = system.device
    system_data = system.data
    info = system_data.info

    # Rows attach to the component of their dynamic endpoints, or to no component between
    # prescribed bodies.
    body_block = np.asarray(system.body_block_host, dtype=np.int32)
    body_a = structural_rows.body_a.numpy()
    body_b = structural_rows.body_b.numpy()
    block_a = np.where(body_a >= 0, body_block[body_a.clip(min=0)], -1)
    block_b = np.where(body_b >= 0, body_block[body_b.clip(min=0)], -1)
    row_block = np.where(block_a >= 0, block_a, block_b)
    block_rows_host = np.argsort(row_block, kind="stable")
    block_rows_host = block_rows_host[row_block[block_rows_host] >= 0]
    block_row_counts = np.bincount(row_block[block_rows_host], minlength=system.num_blocks)
    max_steps = _LANCZOS_STEPS_PER_ROW * int(block_row_counts.max(initial=0))
    if max_steps == 0:
        return
    block_row_offset = to_warp_int32_array(np.concatenate(([0], np.cumsum(block_row_counts))), device)
    block_rows = to_warp_int32_array(block_rows_host, device)

    # Incident rows of each dynamic body, encoded as 2 * row + side, sorted by body and then by row
    side_bodies = np.concatenate((body_a[block_rows_host], body_b[block_rows_host]))
    side_entries = np.concatenate((2 * block_rows_host, 2 * block_rows_host + 1))
    dynamic = (side_bodies >= 0) & (body_block[side_bodies.clip(min=0)] >= 0)
    side_bodies = side_bodies[dynamic]
    side_entries = side_entries[dynamic]
    order = np.lexsort((side_entries, side_bodies))
    body_row_counts = np.bincount(side_bodies, minlength=system.num_bodies)
    body_row_offset = to_warp_int32_array(np.concatenate(([0], np.cumsum(body_row_counts))), device)
    body_row_sides = to_warp_int32_array(side_entries[order], device)

    num_blocks = system.num_blocks
    dof_excluded = wp.zeros(info.total_vec_size, dtype=wp.int32, device=device)
    wp.launch(
        _exclude_inertia_free_dofs,
        dim=(num_blocks, 6 * max(system.block_body_counts)),
        inputs=[info.dim, info.mio, info.vio],
        outputs=[system_data.matrix, dof_excluded],
        device=device,
    )
    system.factorize()

    right_hand_side = wp.zeros(info.total_vec_size, dtype=wp.float32, device=device)
    solution = wp.zeros_like(right_hand_side)
    force_previous = wp.zeros_like(right_hand_side)
    lanczos_vector = wp.zeros(structural_rows.count, dtype=wp.float32, device=device)
    alpha = wp.zeros((num_blocks, max_steps), dtype=wp.float32, device=device)
    beta = wp.zeros_like(alpha)
    steps = wp.zeros(num_blocks, dtype=wp.int32, device=device)
    active = wp.zeros_like(steps)
    any_active = wp.zeros(1, dtype=wp.int32, device=device)
    largest = wp.zeros(num_blocks, dtype=wp.float64, device=device)
    next_check = wp.full(num_blocks, _BOUND_CHECK_INTERVAL, dtype=wp.int32, device=device)
    block_scale = wp.zeros(num_blocks, dtype=wp.float32, device=device)
    status = wp.zeros_like(steps)
    row_inputs = [
        block_row_offset,
        block_rows,
        structural_rows.body_a,
        structural_rows.body_b,
        system_data.body_vector_index,
        structural_rows.jacobian_a,
        structural_rows.jacobian_b,
        structural_rows.effective_mass,
        dof_excluded,
    ]

    def solve_structural_transpose() -> None:
        """Write ``A^-1 p`` of the force ``p = J^T M^1/2 v - beta p_previous`` of every component into ``solution``."""
        wp.launch(
            _apply_structural_transpose,
            dim=system.num_bodies,
            inputs=[
                body_row_offset,
                body_row_sides,
                system_data.body_vector_index,
                system_data.body_block,
                structural_rows.jacobian_a,
                structural_rows.jacobian_b,
                structural_rows.effective_mass,
                dof_excluded,
                lanczos_vector,
                steps,
                beta,
                force_previous,
            ],
            outputs=[right_hand_side],
            device=device,
        )
        system.linear_solver.solve(right_hand_side, solution)

    # Start from S v of a pseudo-random v, which lies in the range of S
    wp.launch(
        _fill_random_vector,
        dim=num_blocks,
        inputs=[block_row_offset, block_rows, 0],
        outputs=[lanczos_vector],
        device=device,
    )
    solve_structural_transpose()
    wp.launch(
        _start_bidiagonalization,
        dim=num_blocks,
        inputs=[*row_inputs, solution],
        outputs=[lanczos_vector, steps, active, any_active],
        device=device,
    )

    def bidiagonalization_step() -> None:
        """Advance the bidiagonalization of every active component by one step."""
        any_active.zero_()
        wp.launch_tiled(
            _stop_bounded_blocks,
            dim=num_blocks,
            inputs=[block_row_offset, alpha, beta, steps, max_ratio, _BOUND_CHECK_INTERVAL],
            outputs=[largest, next_check, active],
            device=device,
            block_dim=_CERTIFICATION_CANDIDATES,
        )
        solve_structural_transpose()
        wp.launch_tiled(
            _normalize_force,
            dim=num_blocks,
            inputs=[info.dim, info.vio, steps, beta, right_hand_side],
            outputs=[solution, force_previous, alpha, active],
            device=device,
            block_dim=_STEP_BLOCK_DIM,
        )
        wp.launch_tiled(
            _bidiagonal_step,
            dim=num_blocks,
            inputs=[*row_inputs, solution, alpha, _LANCZOS_STEPS_PER_ROW],
            outputs=[lanczos_vector, beta, steps, active, any_active],
            device=device,
            block_dim=_STEP_BLOCK_DIM,
        )

    # Iterate while any component is active
    wp.capture_while(any_active, bidiagonalization_step)
    wp.launch_tiled(
        _estimate_block_scales,
        dim=num_blocks,
        inputs=[block_row_offset, alpha, beta, steps, min_ratio, max_ratio],
        outputs=[block_scale, status],
        device=device,
        block_dim=_CERTIFICATION_CANDIDATES,
    )

    fallback_scale = min_ratio
    status_host = status.numpy()
    failed = np.flatnonzero(status_host != SEED_STATUS_VALID)
    if failed.size > 0:
        block = int(failed[0])
        reason = {
            SEED_STATUS_NONFINITE: "structural Delassus is not finite",
            SEED_STATUS_NO_POSITIVE_EIGENVALUE: "structural Delassus has no resolvable positive eigenvalue",
            SEED_STATUS_NOT_REPRESENTABLE: "structural ALM penalty seed is not representable in float32",
        }[int(status_host[block])]
        msg.warning(
            f"LOX joint penalty seed failed for {failed.size} component(s), which take the scale "
            f"{fallback_scale:g}. First: world {system.block_world_host[block]} component {block}: {reason}."
        )
    wp.launch(
        _write_body_scales,
        dim=system.num_bodies,
        inputs=[system_data.body_block, block_row_offset, block_scale, status, fallback_scale],
        outputs=[scales],
        device=device,
    )
    # Each time step orders its factorization from its own matrix
    system.linear_solver.reset()
