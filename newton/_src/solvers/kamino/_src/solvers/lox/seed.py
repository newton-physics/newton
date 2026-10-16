# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint penalty seed of the LOX structural rows.

The seed sets the structural ALM penalty scale of each joint-connected component from the
extreme eigenvalues of its effective-mass-normalized structural Delassus. A float32 Lanczos
recurrence estimates them for every component at once on the device; its kernels live in
:mod:`.seed_kernels`.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import to_warp_int32_array
from ...utils import logger as msg
from .seed_kernels import (
    SEED_STATUS_NO_POSITIVE_EIGENVALUE,
    SEED_STATUS_NONFINITE,
    SEED_STATUS_NOT_REPRESENTABLE,
    SEED_STATUS_VALID,
    _apply_structural_transpose,
    _estimate_block_scales,
    _exclude_inertia_free_dofs,
    _fill_random_vector,
    _lanczos_step,
    _start_lanczos,
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
"""Lanczos steps per structural row of a component.

Without reorthogonalization, float32 Lanczos vectors lose orthogonality, and the smallest
eigenvalue takes more steps than the dimension to converge. Three passes over the rows
resolve the seed of the robot models to about 1e-3.
"""


###
# Functions
###


def estimate_joint_penalty_scale(
    system: LOXSystem,
    structural_rows: StructuralRows,
    scales: wp.array[wp.float32],
    condition: float,
    min_scale: float,
) -> None:
    """Set the penalty scale of each body from the structural spectrum of its joint-connected component.

    For each body component, the structural Delassus ``D = J A^+ J^T`` uses
    the pseudo-inverse of the assembled smooth matrix ``A`` and is normalized by
    the structural effective masses as ``S = M^1/2 D M^1/2``. Massless bodies leave
    DOFs without inertia in ``A``, which the estimate excludes. Each component takes the
    scale of its positive eigenvalues ``mu``: the reciprocal of the smallest eigenvalue,
    capped by ``(condition - 1) / mu_max``. The spectrum of the penalized matrix relative
    to ``A`` is ``1 + s mu``, so the cap bounds the conditioning added by the penalty by
    ``condition``. The scale of a component is at least ``min_scale``, which components whose
    estimate fails also take, with a warning. Bodies of components without structural rows keep their scale.

    A float32 Lanczos recurrence on ``S`` estimates the extreme eigenvalues of every
    component at once on the device. It runs for :data:`_LANCZOS_STEPS_PER_ROW` steps per
    structural row of the component, and each step applies ``A^-1`` with the batched
    factorization of the smooth system.

    The smooth system and the structural rows must be assembled, e.g. after
    :meth:`.LOXProblem.prepare_joint_penalty_scale_seed` and :meth:`.LOXProblem.build_smooth_system`.
    The estimate overwrites the smooth matrix and its factorization.

    Args:
        system: Smooth body system, with the assembled matrix of the seed.
        structural_rows: Structural rows, with their Jacobians and effective masses.
        scales: Dimensionless penalty scale of each body, updated in place.
        condition: Largest conditioning added by the structural penalty, greater than one.
        min_scale: Smallest penalty scale of a component with structural rows.
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
    lanczos_vector = wp.zeros(structural_rows.count, dtype=wp.float32, device=device)
    previous_vector = wp.zeros_like(lanczos_vector)
    work_vector = wp.zeros_like(lanczos_vector)
    alpha = wp.zeros((num_blocks, max_steps), dtype=wp.float32, device=device)
    beta = wp.zeros_like(alpha)
    steps = wp.zeros(num_blocks, dtype=wp.int32, device=device)
    active = wp.zeros_like(steps)
    any_active = wp.zeros(1, dtype=wp.int32, device=device)
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

    def solve_structural_transpose(row_vector: wp.array[wp.float32]) -> None:
        """Write ``A^-1 J^T M^1/2 v`` of every component into ``solution``."""
        wp.launch(
            _apply_structural_transpose,
            dim=num_blocks,
            inputs=[
                block_row_offset,
                block_rows,
                info.dim,
                info.vio,
                structural_rows.body_a,
                structural_rows.body_b,
                system_data.body_vector_index,
                structural_rows.jacobian_a,
                structural_rows.jacobian_b,
                structural_rows.effective_mass,
                dof_excluded,
                row_vector,
            ],
            outputs=[right_hand_side],
            device=device,
        )
        system.linear_solver.solve(right_hand_side, solution)

    wp.launch(
        _fill_random_vector,
        dim=num_blocks,
        inputs=[block_row_offset, block_rows, 0],
        outputs=[work_vector],
        device=device,
    )
    solve_structural_transpose(work_vector)
    wp.launch(
        _start_lanczos,
        dim=num_blocks,
        inputs=[*row_inputs, solution],
        outputs=[lanczos_vector, previous_vector, steps, active, any_active],
        device=device,
    )

    def lanczos_step() -> None:
        """Advance the Lanczos recurrence of every active component by one step."""
        any_active.zero_()
        solve_structural_transpose(lanczos_vector)
        wp.launch(
            _lanczos_step,
            dim=num_blocks,
            inputs=[*row_inputs, solution, _LANCZOS_STEPS_PER_ROW],
            outputs=[lanczos_vector, previous_vector, work_vector, alpha, beta, steps, active, any_active],
            device=device,
        )

    # Iterate while any component is active
    wp.capture_while(any_active, lanczos_step)
    wp.launch(
        _estimate_block_scales,
        dim=num_blocks,
        inputs=[block_row_offset, alpha, beta, steps, condition],
        outputs=[block_scale, status],
        device=device,
    )

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
            f"LOX joint penalty seed failed for {failed.size} component(s), which keep the minimum scale "
            f"{min_scale:g}. First: world {system.block_world_host[block]} component {block}: {reason}."
        )
    wp.launch(
        _write_body_scales,
        dim=system.num_bodies,
        inputs=[system_data.body_block, block_row_offset, block_scale, status, min_scale],
        outputs=[scales],
        device=device,
    )
