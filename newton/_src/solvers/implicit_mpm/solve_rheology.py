# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import gc
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.fem.linalg import array_axpy
from warp.optim.linear import LinearOperator, cg, cr, gmres

from .contact_solver_kernels import (
    apply_nodal_impulse_warmstart,
    apply_subgrid_impulse,
    apply_subgrid_impulse_warmstart,
    compute_collider_delassus_diagonal,
    compute_collider_inv_mass,
    solve_nodal_friction,
    solve_subgrid_friction,
)
from .rheology_solver_kernels import (
    YieldParamVec,
    apply_stress_delta_jacobi,
    apply_stress_gs,
    apply_velocity_delta,
    batched_scatter,
    build_batch_transpose_offsets,
    build_flat_color_offsets,
    build_flat_offsets,
    build_strain_to_batch,
    compute_batch_base_offsets,
    compute_delassus_diagonal,
    compute_vel_node_multiplicity,
    evaluate_strain_residual,
    expand_flat_ids,
    fill_batch_transpose,
    globalize_batch_offsets,
    jacobi_preconditioner,
    make_batched_solve_kernel,
    make_gs_solve_kernel,
    make_jacobi_solve_kernel,
    make_reordered_gs_solve_kernel,
    mat13,
    mat55,
    postprocess_stress_and_strain,
    preprocess_stress_and_strain,
    reorder_strain_mat,
    vec6,
)

_TILED_SUM_BLOCK_DIM = 512


@wp.kernel
def _tiled_sum_kernel(
    data: wp.array2d[float],
    partial_sums: wp.array2d[float],
):
    block_id, _ = wp.tid()

    tile = wp.tile_load(data[0], shape=_TILED_SUM_BLOCK_DIM, offset=block_id * _TILED_SUM_BLOCK_DIM)
    wp.tile_store(partial_sums[0], wp.tile_sum(tile), offset=block_id)
    tile = wp.tile_load(data[1], shape=_TILED_SUM_BLOCK_DIM, offset=block_id * _TILED_SUM_BLOCK_DIM)
    wp.tile_store(partial_sums[1], wp.tile_max(tile), offset=block_id)


class ArraySquaredNorm:
    """Utility to compute squared L2 norm of a large array via tiled reductions."""

    def __init__(self, max_length: int, device=None, temporary_store=None):
        self.tile_size = _TILED_SUM_BLOCK_DIM
        self.device = device

        num_blocks = (max_length + self.tile_size - 1) // self.tile_size
        self.partial_sums_a = fem.borrow_temporary(
            temporary_store, shape=(2, num_blocks), dtype=float, device=self.device
        )
        self.partial_sums_b = fem.borrow_temporary(
            temporary_store, shape=(2, num_blocks), dtype=float, device=self.device
        )
        self.partial_sums_a.zero_()
        self.partial_sums_b.zero_()

        self.sum_launch: wp.Launch = wp.launch(
            _tiled_sum_kernel,
            dim=(num_blocks, self.tile_size),
            inputs=(self.partial_sums_a,),
            outputs=(self.partial_sums_b,),
            block_dim=self.tile_size,
            record_cmd=True,
        )

    # Result contains a single value, the sum of the array (will get updated by this function)
    def compute_squared_norm(self, data: wp.array[Any]):
        # cast vector types to float
        if data.ndim != 2:
            data = wp.array(
                ptr=data.ptr,
                shape=(2, data.shape[0]),
                dtype=data.dtype,
                strides=(0, data.strides[0]),
                device=data.device,
            )

        array_length = data.shape[1]

        flip_flop = False
        while True:
            num_blocks = (array_length + self.tile_size - 1) // self.tile_size
            partial_sums = (self.partial_sums_a if flip_flop else self.partial_sums_b)[:, :num_blocks]

            self.sum_launch.set_param_at_index(0, data[:, :array_length])
            self.sum_launch.set_param_at_index(1, partial_sums)
            self.sum_launch.set_dim((num_blocks, self.tile_size))
            self.sum_launch.launch()

            array_length = num_blocks
            data = partial_sums

            flip_flop = not flip_flop

            if num_blocks == 1:
                break

        return data[:, :1]

    def release(self):
        """Return borrowed temporaries to their pool."""
        for attr in ("partial_sums_a", "partial_sums_b"):
            temporary = getattr(self, attr, None)
            if temporary is not None:
                temporary.release()
                setattr(self, attr, None)

    def __del__(self):
        self.release()


@wp.kernel
def update_condition(
    residual_threshold: float,
    l2_scale: float,
    solve_granularity: int,
    max_iterations: int,
    residual: wp.array2d[float],
    iteration: wp.array[int],
    condition: wp.array[int],
):
    cur_it = iteration[0] + solve_granularity
    stop = (
        residual[0, 0] < residual_threshold * l2_scale and residual[1, 0] < residual_threshold
    ) or cur_it > max_iterations

    iteration[0] = cur_it
    condition[0] = wp.where(stop, 0, 1)


def apply_rigidity_operator(rigidity_operator, delta_collider_impulse, collider_velocity, delta_body_qd):
    """Apply collider rigidity feedback to the current collider velocities.

    Computes and applies a velocity correction induced by the rigid coupling
    operator according to the relation::

        delta_body_qd = -IJtm @ delta_collider_impulse
        collider_velocity += J @ delta_body_qd

    where ``(J, IJtm) = rigidity_operator`` are the block-sparse matrices
    returned by ``build_rigidity_operator``.

    Args:
        rigidity_operator: Pair ``(J, IJtm)`` of block-sparse matrices returned
            by ``build_rigidity_operator``.
        delta_collider_impulse: Change in collider impulse to be applied.
        collider_velocity: Current collider velocity vector to be corrected in place.
        delta_body_qd: Change in body velocity to be applied.
    """

    J, IJtm = rigidity_operator
    sp.bsr_mv(IJtm, x=delta_collider_impulse, y=delta_body_qd, alpha=-1.0, beta=0.0)
    sp.bsr_mv(J, x=delta_body_qd, y=collider_velocity, alpha=1.0, beta=1.0)


class _ScopedDisableGC:
    """Context manager to disable automatic garbage collection during graph capture.
    Avoids capturing deallocations of arrays exterior to the capture scope.
    """

    def __enter__(self):
        self.was_enabled = gc.isenabled()
        gc.disable()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.was_enabled:
            gc.enable()


@dataclass
class MomentumData:
    """Per-node momentum quantities used by the rheology solver.

    Attributes:
        inv_volume: Inverse volume (or inverse mass scaling) per velocity
            node, shape ``[node_count]``.
        velocity: Grid velocity DOFs to be updated in place [m/s],
            shape ``[node_count, 3]``.
    """

    inv_volume: wp.array
    velocity: wp.array[wp.vec3]


@dataclass
class RheologyData:
    """Strain, compliance, yield, and coloring data for the rheology solve.

    Attributes:
        strain_mat: Strain-to-velocity block-sparse matrix (B).
        transposed_strain_mat: BSR container for B^T, used by the Jacobi
            solver path.
        compliance_mat: Compliance (inverse stiffness) block-sparse matrix.
        strain_node_volume: Volume associated with each strain node [m^3],
            shape ``[strain_count]``.
        yield_params: Yield-surface parameters per strain node,
            shape ``[strain_count]``.
        unilateral_strain_offset: Per-node offset enforcing unilateral
            incompressibility (void/critical fraction),
            shape ``[strain_count]``.
        color_offsets: Coloring offsets for Gauss-Seidel iteration,
            shape ``[num_colors + 1]``.
        color_blocks: Per-color strain-node indices for Gauss-Seidel,
            shape ``[num_colors, max_block_size]``.
        elastic_strain_delta: Output elastic strain increment per strain
            node, shape ``[strain_count, 6]``.
        plastic_strain_delta: Output plastic strain increment per strain
            node, shape ``[strain_count, 6]``.
        stress: In/out stress per strain node (rotated internally),
            shape ``[strain_count, 6]``.
    """

    strain_mat: sp.BsrMatrix
    transposed_strain_mat: sp.BsrMatrix
    compliance_mat: sp.BsrMatrix
    strain_node_volume: wp.array[float]
    yield_params: wp.array[YieldParamVec]
    unilateral_strain_offset: wp.array[float]

    color_offsets: wp.array[int]
    color_blocks: wp.array2d[int]

    elastic_strain_delta: wp.array[vec6]
    plastic_strain_delta: wp.array[vec6]
    stress: wp.array[vec6]

    has_viscosity: bool = False
    has_dilatancy: bool = False
    strain_velocity_node_count: int = -1


@dataclass
class CollisionData:
    """Collider contact data consumed by the rheology solver.

    Attributes:
        collider_mat: Block-sparse matrix mapping velocity nodes to
            collider DOFs.
        transposed_collider_mat: Transpose of ``collider_mat``.
        collider_friction: Per-node friction coefficients; negative values
            disable contact at that node, shape ``[node_count]``.
        collider_adhesion: Per-node adhesion coefficients [N s / V0],
            shape ``[node_count]``.
        collider_normals: Per-node contact normals,
            shape ``[node_count, 3]``.
        collider_velocities: Per-node collider rigid-body velocities [m/s],
            shape ``[node_count, 3]``.
        rigidity_operator: Optional pair of BSR matrices coupling velocity
            nodes to collider DOFs. ``None`` when unused.
        collider_impulse: In/out stored collider impulses for warm-starting
            [N s / V0], shape ``[node_count, 3]``.
        has_colliders: True when at least one collider mesh is present in the
            scene; used to reject linear-only solvers that do not support
            contact.
    """

    collider_mat: sp.BsrMatrix
    transposed_collider_mat: sp.BsrMatrix
    collider_friction: wp.array[float]
    collider_adhesion: wp.array[float]
    collider_normals: wp.array[wp.vec3]
    collider_velocities: wp.array[wp.vec3]
    rigidity_operator: tuple[sp.BsrMatrix, sp.BsrMatrix] | None
    collider_impulse: wp.array[wp.vec3]
    has_colliders: bool = False


class _DelassusOperator:
    def __init__(
        self,
        rheology: RheologyData,
        momentum: MomentumData,
        temporary_store: fem.TemporaryStore | None = None,
    ):
        self.rheology = rheology
        self.momentum = momentum

        self.delassus_rotation = fem.borrow_temporary(temporary_store, shape=self.size, dtype=mat55)
        self.delassus_diagonal = fem.borrow_temporary(temporary_store, shape=self.size, dtype=vec6)

        self._computed = False
        self._split_mass = False
        self._mass_multiplicity_used = False

        self._has_strain_mat_transpose = False

        self.preprocess_stress_and_strain()

    def compute_diagonal_factorization(
        self,
        split_mass: bool = False,
        strain_batch: wp.array | None = None,
        mass_multiplicity: wp.array | None = None,
    ):
        """Compute or recompute the Delassus diagonal eigendecomposition.

        Args:
            split_mass: If ``True`` and no *mass_multiplicity* is provided,
                compute per-velocity-node multiplicity from the transposed
                strain matrix (standard Jacobi mass splitting with n_batches=1).
            strain_batch: Per-strain-node batch assignment
                (int array, length n_strain).  Required when
                *mass_multiplicity* is provided.
            mass_multiplicity: Pre-computed per-batch per-velocity-node
                multiplicity (float 2D array, shape ``[n_batches, n_vel]``).
                Overrides *split_mass* when provided.
        """
        if (
            mass_multiplicity is None
            and self._computed
            and not self._mass_multiplicity_used
            and self._split_mass == split_mass
        ):
            return

        device = self.momentum.velocity.device

        if mass_multiplicity is not None:
            # Caller-provided multiplicity (batched mode)
            batch_map = strain_batch
            mult = mass_multiplicity
        elif split_mass:
            # Jacobi: n_batches=1, all strain nodes in batch 0
            self.require_strain_mat_transpose()
            n_vel = self.momentum.velocity.shape[0]
            batch_map = wp.zeros(shape=(self.size,), dtype=int, device=device)  # all zeros = batch 0
            mult = wp.zeros(shape=(1, n_vel), dtype=float, device=device)
            wp.launch(
                kernel=compute_vel_node_multiplicity,
                dim=n_vel,
                inputs=[
                    self.rheology.transposed_strain_mat.offsets,
                    self.rheology.transposed_strain_mat.columns,
                    batch_map,
                    1,
                ],
                outputs=[mult],
            )
        else:
            # GS mode: empty arrays → multiplicity of 1
            batch_map = wp.zeros(shape=(0,), dtype=int, device=device)
            mult = wp.zeros(shape=(0, 0), dtype=float, device=device)

        strain_mat_values = self.rheology.strain_mat.values.view(dtype=mat13)
        wp.launch(
            kernel=compute_delassus_diagonal,
            dim=self.size,
            inputs=[
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                strain_mat_values,
                self.momentum.inv_volume,
                self.rheology.compliance_mat.offsets,
                self.rheology.compliance_mat.columns,
                self.rheology.compliance_mat.values,
                batch_map,
                mult,
            ],
            outputs=[
                self.delassus_rotation,
                self.delassus_diagonal,
            ],
        )

        self._computed = True
        self._split_mass = split_mass
        self._mass_multiplicity_used = mass_multiplicity is not None

    def require_strain_mat_transpose(self):
        if not self._has_strain_mat_transpose:
            sp.bsr_set_transpose(dest=self.rheology.transposed_strain_mat, src=self.rheology.strain_mat)
            self._has_strain_mat_transpose = True

    def preprocess_stress_and_strain(self):
        # Project initial stress on yield surface
        wp.launch(
            kernel=preprocess_stress_and_strain,
            dim=self.size,
            inputs=[
                self.rheology.unilateral_strain_offset,
                self.rheology.elastic_strain_delta,
                self.rheology.stress,
                self.rheology.yield_params,
            ],
        )

    @property
    def size(self):
        return self.rheology.stress.shape[0]

    def release(self):
        self.delassus_rotation.release()
        self.delassus_diagonal.release()

    def apply_stress_delta(self, stress_delta: wp.array[vec6], velocity: wp.array[wp.vec3], record_cmd: bool = False):
        return wp.launch(
            kernel=apply_stress_delta_jacobi,
            dim=self.momentum.velocity.shape[0],
            inputs=[
                self.rheology.transposed_strain_mat.offsets,
                self.rheology.transposed_strain_mat.columns,
                self.rheology.transposed_strain_mat.values.view(dtype=mat13),
                self.momentum.inv_volume,
                stress_delta,
            ],
            outputs=[velocity],
            record_cmd=record_cmd,
        )

    def apply_velocity_delta(
        self,
        velocity_delta: wp.array[wp.vec3],
        strain_prev: wp.array[vec6],
        strain: wp.array[vec6],
        alpha: float = 1.0,
        beta: float = 1.0,
        record_cmd: bool = False,
    ):
        return wp.launch(
            kernel=apply_velocity_delta,
            dim=self.size,
            inputs=[
                alpha,
                beta,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                velocity_delta,
                strain_prev,
            ],
            outputs=[
                strain,
            ],
            record_cmd=record_cmd,
        )

    def postprocess_stress_and_strain(self):
        # Convert stress back to world space,
        # and compute final elastic strain
        wp.launch(
            kernel=postprocess_stress_and_strain,
            dim=self.size,
            inputs=[
                self.rheology.compliance_mat.offsets,
                self.rheology.compliance_mat.columns,
                self.rheology.compliance_mat.values,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                self.delassus_diagonal,
                self.delassus_rotation,
                self.rheology.unilateral_strain_offset,
                self.rheology.yield_params,
                self.rheology.strain_node_volume,
                self.rheology.elastic_strain_delta,
                self.rheology.stress,
                self.momentum.velocity,
            ],
            outputs=[
                self.rheology.elastic_strain_delta,
                self.rheology.plastic_strain_delta,
            ],
        )


class _RheologySolver:
    def __init__(
        self,
        delassus_operator: _DelassusOperator,
        split_mass: bool,
        temporary_store: fem.TemporaryStore | None = None,
        skip_factorization: bool = False,
    ):
        self.delassus_operator = delassus_operator
        self.momentum = delassus_operator.momentum
        self.rheology = delassus_operator.rheology
        self.device = self.momentum.velocity.device

        self.delta_stress = fem.borrow_temporary_like(self.rheology.stress, temporary_store)
        self.strain_residual = fem.borrow_temporary(
            temporary_store, shape=(self.size,), dtype=float, device=self.device
        )
        self.strain_residual.zero_()

        if not skip_factorization:
            self.delassus_operator.compute_diagonal_factorization(split_mass)

        self._evaluate_strain_residual_launch = wp.launch(
            kernel=evaluate_strain_residual,
            dim=self.size,
            inputs=[
                self.delta_stress,
                self.delassus_operator.delassus_diagonal,
                self.delassus_operator.delassus_rotation,
            ],
            outputs=[
                self.strain_residual,
            ],
            record_cmd=True,
        )

        # Utility to compute the squared norm of the residual
        self._residual_squared_norm_computer = ArraySquaredNorm(
            max_length=self.size,
            device=self.device,
            temporary_store=temporary_store,
        )

    @property
    def size(self):
        return self.rheology.stress.shape[0]

    def eval_residual(self):
        self._evaluate_strain_residual_launch.launch()
        return self._residual_squared_norm_computer.compute_squared_norm(self.strain_residual)

    def release(self):
        self.delta_stress.release()
        self.strain_residual.release()
        self._residual_squared_norm_computer.release()


class _GaussSeidelSolver(_RheologySolver):
    def __init__(
        self,
        delassus_operator: _DelassusOperator,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        super().__init__(delassus_operator, split_mass=False, temporary_store=temporary_store)

        self.color_count = self.rheology.color_offsets.shape[0] - 1

        if self.device.is_cuda:
            color_block_count = self.device.sm_count * 2
        else:
            color_block_count = 1
        color_block_dim = 64
        color_launch_dim = color_block_count * color_block_dim

        self.apply_stress_launch = wp.launch(
            kernel=apply_stress_gs,
            dim=color_launch_dim,
            inputs=[
                0,  # color
                color_launch_dim,
                self.rheology.color_offsets,
                self.rheology.color_blocks,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                self.momentum.inv_volume,
                self.rheology.stress,
            ],
            outputs=[
                self.momentum.velocity,
            ],
            block_dim=color_block_dim,
            max_blocks=color_block_count,
            record_cmd=True,
        )

        # Solve kernel
        gs_kernel = make_gs_solve_kernel(
            has_viscosity=self.rheology.has_viscosity,
            has_dilatancy=self.rheology.has_dilatancy,
            has_compliance_mat=self.rheology.compliance_mat.nnz > 0,
            strain_velocity_node_count=self.rheology.strain_velocity_node_count,
        )
        self.solve_local_launch = wp.launch(
            kernel=gs_kernel,
            dim=color_launch_dim,
            inputs=[
                0,  # color
                color_launch_dim,
                self.rheology.color_offsets,
                self.rheology.color_blocks,
                self.rheology.yield_params,
                self.rheology.strain_node_volume,
                self.rheology.compliance_mat.offsets,
                self.rheology.compliance_mat.columns,
                self.rheology.compliance_mat.values,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                self.delassus_operator.delassus_diagonal,
                self.delassus_operator.delassus_rotation,
                self.momentum.inv_volume,
                self.rheology.elastic_strain_delta,
            ],
            outputs=[
                self.momentum.velocity,
                self.rheology.stress,
                self.delta_stress,
            ],
            block_dim=color_block_dim,
            max_blocks=color_block_count,
            record_cmd=True,
        )

    @property
    def name(self):
        return "Gauss-Seidel"

    @property
    def solve_granularity(self):
        return 25

    def apply_initial_guess(self):
        for color in range(self.color_count):
            self.apply_stress_launch.set_param_at_index(0, color)
            self.apply_stress_launch.launch()

    def solve(self):
        for color in range(self.color_count):
            self.solve_local_launch.set_param_at_index(0, color)
            self.solve_local_launch.launch()


class _ReorderedGaussSeidelSolver(_RheologySolver):
    """Gauss-Seidel solver with entry-major SoA strain matrix layout.

    Reorders the BSR strain matrix into a flat, entry-major SoA layout at
    construction time.  The solve kernel statically unrolls the velocity
    gather loop for coalesced memory access, giving significant speedups
    on higher-order bases at the cost of increased memory usage.
    """

    def __init__(
        self,
        delassus_operator: _DelassusOperator,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        super().__init__(delassus_operator, split_mass=False, temporary_store=temporary_store)

        self.color_count = self.rheology.color_offsets.shape[0] - 1

        # ── Determine max_entries without synchronization ────────────────
        # Use strain_velocity_node_count if known; otherwise fall back to
        # the number of colors (upper bound for regular grids).

        svnc = self.rheology.strain_velocity_node_count
        if svnc > 0:
            max_entries = svnc
        else:
            max_entries = self.color_count

        # ── Build flat ordering + SoA reordered buffers ──────────────────
        # All operations are GPU kernel launches — no host synchronization.

        n_total = self.size
        # color_blocks.shape[1] is pre-allocated capacity; the actual valid
        # block count is color_offsets[-1], read on device to avoid sync.
        num_blocks_capacity = self.rheology.color_blocks.shape[1]

        # Flat offsets: prefix sum over color-block sizes (fully written by kernel)
        block_flat_offsets = fem.borrow_temporary(
            temporary_store, shape=(num_blocks_capacity + 1,), dtype=int, device=self.device
        )
        wp.launch(
            kernel=build_flat_offsets,
            dim=1,
            inputs=[self.rheology.color_blocks, self.rheology.color_offsets],
            outputs=[block_flat_offsets],
            device=self.device,
        )

        # Flat color offsets (fully written by kernel)
        self._flat_color_offsets = fem.borrow_temporary(
            temporary_store, shape=(self.color_count + 1,), dtype=int, device=self.device
        )
        wp.launch(
            kernel=build_flat_color_offsets,
            dim=self.color_count + 1,
            inputs=[self.rheology.color_offsets, block_flat_offsets],
            outputs=[self._flat_color_offsets],
            device=self.device,
        )

        # Expand color blocks into flat constraint IDs (fully written by kernel)
        self._flat_constraint_ids = fem.borrow_temporary(
            temporary_store, shape=(n_total,), dtype=int, device=self.device
        )
        wp.launch(
            kernel=expand_flat_ids,
            dim=num_blocks_capacity,
            inputs=[self.rheology.color_blocks, self.rheology.color_offsets, block_flat_offsets],
            outputs=[self._flat_constraint_ids],
            device=self.device,
        )

        # Reorder strain matrix into entry-major SoA.
        # cols/vals are zero-padded: excess entries must be zero so the
        # statically-unrolled gather loop contributes nothing for them.
        self._reordered_n_entries = fem.borrow_temporary(
            temporary_store, shape=(n_total,), dtype=int, device=self.device
        )
        self._reordered_cols = fem.borrow_temporary(
            temporary_store, shape=(max_entries, n_total), dtype=int, device=self.device
        )
        self._reordered_cols.zero_()
        self._reordered_vals_x = fem.borrow_temporary(
            temporary_store, shape=(max_entries, n_total), dtype=float, device=self.device
        )
        self._reordered_vals_x.zero_()
        self._reordered_vals_y = fem.borrow_temporary(
            temporary_store, shape=(max_entries, n_total), dtype=float, device=self.device
        )
        self._reordered_vals_y.zero_()
        self._reordered_vals_z = fem.borrow_temporary(
            temporary_store, shape=(max_entries, n_total), dtype=float, device=self.device
        )
        self._reordered_vals_z.zero_()
        wp.launch(
            kernel=reorder_strain_mat,
            dim=n_total,
            inputs=[
                self._flat_constraint_ids,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                self._reordered_cols,
                self._reordered_vals_x,
                self._reordered_vals_y,
                self._reordered_vals_z,
                self._reordered_n_entries,
            ],
            device=self.device,
        )

        # ── Launch config ────────────────────────────────────────────────

        if self.device.is_cuda:
            color_block_count = self.device.sm_count * 2
        else:
            color_block_count = 1
        color_block_dim = 64
        color_launch_dim = color_block_count * color_block_dim

        # Initial guess uses the existing AoS apply_stress_gs kernel (runs once)
        self.apply_stress_launch = wp.launch(
            kernel=apply_stress_gs,
            dim=color_launch_dim,
            inputs=[
                0,
                color_launch_dim,
                self.rheology.color_offsets,
                self.rheology.color_blocks,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                self.momentum.inv_volume,
                self.rheology.stress,
            ],
            outputs=[self.momentum.velocity],
            block_dim=color_block_dim,
            max_blocks=color_block_count,
            record_cmd=True,
        )

        # Solve kernel: reordered SoA layout
        gs_kernel = make_reordered_gs_solve_kernel(
            has_viscosity=self.rheology.has_viscosity,
            has_dilatancy=self.rheology.has_dilatancy,
            has_compliance_mat=self.rheology.compliance_mat.nnz > 0,
            max_entries=max_entries,
        )
        self.solve_local_launch = wp.launch(
            kernel=gs_kernel,
            dim=color_launch_dim,
            inputs=[
                0,
                color_launch_dim,
                self._flat_color_offsets,
                self._flat_constraint_ids,
                self._reordered_n_entries,
                self._reordered_cols,
                self._reordered_vals_x,
                self._reordered_vals_y,
                self._reordered_vals_z,
                self.rheology.yield_params,
                self.rheology.strain_node_volume,
                self.rheology.compliance_mat.offsets,
                self.rheology.compliance_mat.columns,
                self.rheology.compliance_mat.values,
                self.delassus_operator.delassus_diagonal,
                self.delassus_operator.delassus_rotation,
                self.momentum.inv_volume,
                self.rheology.elastic_strain_delta,
                self.momentum.velocity,
                self.rheology.stress,
            ],
            outputs=[
                self.delta_stress,
            ],
            block_dim=color_block_dim,
            max_blocks=color_block_count,
            record_cmd=True,
        )

    @property
    def name(self):
        return "Gauss-Seidel (reordered)"

    @property
    def solve_granularity(self):
        return 25

    def apply_initial_guess(self):
        for color in range(self.color_count):
            self.apply_stress_launch.set_param_at_index(0, color)
            self.apply_stress_launch.launch()

    def solve(self):
        for color in range(self.color_count):
            self.solve_local_launch.set_param_at_index(0, color)
            self.solve_local_launch.launch()

    def release(self):
        super().release()
        self._flat_color_offsets.release()
        self._flat_constraint_ids.release()
        self._reordered_n_entries.release()
        self._reordered_cols.release()
        self._reordered_vals_x.release()
        self._reordered_vals_y.release()
        self._reordered_vals_z.release()


class _BatchedGaussSeidelSolver(_RheologySolver):
    """Batched GS-Jacobi solver with batch grouping.

    Merges the original colors into fewer batches.  Within each
    batch, constraints are solved in parallel (Jacobi-like) with a
    mass-split Delassus diagonal and atomic velocity scatter.  Between
    batches, GS ordering applies.
    """

    def __init__(
        self,
        delassus_operator: _DelassusOperator,
        temporary_store: fem.TemporaryStore | None = None,
        n_batches: int | None = None,
    ) -> None:
        # split_mass=False, skip_factorization=True — we compute the diagonal ourselves
        # with per-batch mass splitting after building the batch structures below
        super().__init__(delassus_operator, split_mass=False, temporary_store=temporary_store, skip_factorization=True)

        self.color_count = self.rheology.color_offsets.shape[0] - 1
        self.n_batches, self.colors_per_batch = _resolve_batched_gs_batching(self.color_count, n_batches)

        # ── Determine max_entries ────────────────────────────────────────

        svnc = self.rheology.strain_velocity_node_count
        if svnc > 0:
            max_entries = svnc
        else:
            max_entries = self.color_count

        # ── Build flat ordering + SoA reordered buffers ──────────────────
        # (same as _ReorderedGaussSeidelSolver)

        n_total = self.size
        num_blocks_capacity = self.rheology.color_blocks.shape[1]

        block_flat_offsets = fem.borrow_temporary(
            temporary_store, shape=(num_blocks_capacity + 1,), dtype=int, device=self.device
        )
        wp.launch(
            kernel=build_flat_offsets,
            dim=1,
            inputs=[self.rheology.color_blocks, self.rheology.color_offsets],
            outputs=[block_flat_offsets],
            device=self.device,
        )

        self._flat_color_offsets = fem.borrow_temporary(
            temporary_store, shape=(self.color_count + 1,), dtype=int, device=self.device
        )
        wp.launch(
            kernel=build_flat_color_offsets,
            dim=self.color_count + 1,
            inputs=[self.rheology.color_offsets, block_flat_offsets],
            outputs=[self._flat_color_offsets],
            device=self.device,
        )

        self._flat_constraint_ids = fem.borrow_temporary(
            temporary_store, shape=(n_total,), dtype=int, device=self.device
        )
        wp.launch(
            kernel=expand_flat_ids,
            dim=num_blocks_capacity,
            inputs=[self.rheology.color_blocks, self.rheology.color_offsets, block_flat_offsets],
            outputs=[self._flat_constraint_ids],
            device=self.device,
        )

        self._reordered_n_entries = fem.borrow_temporary(
            temporary_store, shape=(n_total,), dtype=int, device=self.device
        )
        self._reordered_cols = fem.borrow_temporary(
            temporary_store, shape=(max_entries, n_total), dtype=int, device=self.device
        )
        self._reordered_cols.zero_()
        self._reordered_vals_x = fem.borrow_temporary(
            temporary_store, shape=(max_entries, n_total), dtype=float, device=self.device
        )
        self._reordered_vals_x.zero_()
        self._reordered_vals_y = fem.borrow_temporary(
            temporary_store, shape=(max_entries, n_total), dtype=float, device=self.device
        )
        self._reordered_vals_y.zero_()
        self._reordered_vals_z = fem.borrow_temporary(
            temporary_store, shape=(max_entries, n_total), dtype=float, device=self.device
        )
        self._reordered_vals_z.zero_()
        wp.launch(
            kernel=reorder_strain_mat,
            dim=n_total,
            inputs=[
                self._flat_constraint_ids,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                self._reordered_cols,
                self._reordered_vals_x,
                self._reordered_vals_y,
                self._reordered_vals_z,
                self._reordered_n_entries,
            ],
            device=self.device,
        )

        # ── Build per-batch mass-split Delassus diagonal ───────────

        # Step 1: strain → batch mapping
        self._strain_batch = fem.borrow_temporary(temporary_store, shape=(n_total,), dtype=int, device=self.device)
        self._strain_batch.fill_(-1)
        wp.launch(
            kernel=build_strain_to_batch,
            dim=n_total,
            inputs=[self._flat_color_offsets, self._flat_constraint_ids, self.colors_per_batch, self.n_batches],
            outputs=[self._strain_batch],
            device=self.device,
        )

        # Step 2: per-velocity-node per-batch sharing counts (accumulator: must be zero)
        self.delassus_operator.require_strain_mat_transpose()
        n_vel = self.momentum.velocity.shape[0]
        batch_sharing = fem.borrow_temporary(
            temporary_store, shape=(self.n_batches, n_vel), dtype=float, device=self.device
        )
        batch_sharing.zero_()
        wp.launch(
            kernel=compute_vel_node_multiplicity,
            dim=n_vel,
            inputs=[
                self.rheology.transposed_strain_mat.offsets,
                self.rheology.transposed_strain_mat.columns,
                self._strain_batch,
                self.n_batches,
            ],
            outputs=[batch_sharing],
            device=self.device,
        )

        # Step 3: compute Delassus diagonal with per-batch mass splitting
        self.delassus_operator.compute_diagonal_factorization(
            strain_batch=self._strain_batch,
            mass_multiplicity=batch_sharing,
        )

        # ── Launch config ────────────────────────────────────────────────

        if self.device.is_cuda:
            color_block_count = self.device.sm_count * 2
        else:
            color_block_count = 1
        color_block_dim = 64
        color_launch_dim = color_block_count * color_block_dim

        # (apply_stress_launch is set up after the per-batch transposed matrices)

        # Phase 1: solve kernel
        solve_kernel = make_batched_solve_kernel(
            has_viscosity=self.rheology.has_viscosity,
            has_dilatancy=self.rheology.has_dilatancy,
            has_compliance_mat=self.rheology.compliance_mat.nnz > 0,
            max_entries=max_entries,
        )
        self._solve_launch = wp.launch(
            kernel=solve_kernel,
            dim=color_launch_dim,
            inputs=[
                0,
                color_launch_dim,
                self._flat_color_offsets,
                self.colors_per_batch,
                self._flat_constraint_ids,
                self._reordered_cols,
                self._reordered_vals_x,
                self._reordered_vals_y,
                self._reordered_vals_z,
                self.rheology.yield_params,
                self.rheology.strain_node_volume,
                self.rheology.compliance_mat.offsets,
                self.rheology.compliance_mat.columns,
                self.rheology.compliance_mat.values,
                self.delassus_operator.delassus_diagonal,
                self.delassus_operator.delassus_rotation,
                self.rheology.elastic_strain_delta,
                self.momentum.velocity,
                self.rheology.stress,
            ],
            outputs=[
                self.delta_stress,
            ],
            block_dim=color_block_dim,
            max_blocks=color_block_count,
            record_cmd=True,
        )

        # ── Precompute per-batch transposed matrices (all on device) ─

        n_vel = self.momentum.velocity.shape[0]
        t_mat = self.rheology.transposed_strain_mat
        total_nnz = t_mat.nnz  # total entries across all batches

        # Step 1: count entries per (batch, velocity-node) (accumulator: must be zero)
        batch_counts = fem.borrow_temporary(
            temporary_store, shape=(self.n_batches, n_vel), dtype=int, device=self.device
        )
        batch_counts.zero_()
        wp.launch(
            kernel=build_batch_transpose_offsets,
            dim=n_vel,
            inputs=[t_mat.offsets, t_mat.columns, self._strain_batch, self.n_batches],
            outputs=[batch_counts],
            device=self.device,
        )

        # Step 2: per-row exclusive prefix scan (local offsets per batch; fully written)
        batch_local_offsets = fem.borrow_temporary(
            temporary_store, shape=(self.n_batches, n_vel), dtype=int, device=self.device
        )
        for bi in range(self.n_batches):
            wp.utils.array_scan(batch_counts[bi], batch_local_offsets[bi], inclusive=False)

        # Step 3: compute per-batch base offsets (single-threaded kernel; fully written)
        sc_bases = fem.borrow_temporary(temporary_store, shape=(self.n_batches,), dtype=int, device=self.device)
        wp.launch(
            kernel=compute_batch_base_offsets,
            dim=1,
            inputs=[batch_counts, batch_local_offsets],
            outputs=[sc_bases],
            device=self.device,
        )

        # Step 4: globalize local offsets → _batch_global_offsets[n_batches, n_vel+1]
        self._batch_global_offsets = fem.borrow_temporary(
            temporary_store, shape=(self.n_batches, n_vel + 1), dtype=int, device=self.device
        )
        wp.launch(
            kernel=globalize_batch_offsets,
            dim=n_vel,
            inputs=[batch_counts, batch_local_offsets, sc_bases],
            outputs=[self._batch_global_offsets],
            device=self.device,
        )

        # Step 5: allocate flat arrays and fill all SCs in one pass
        self._batch_columns = fem.borrow_temporary(temporary_store, shape=(total_nnz,), dtype=int, device=self.device)
        self._batch_values = fem.borrow_temporary(temporary_store, shape=(total_nnz,), dtype=mat13, device=self.device)
        batch_write_cursors = fem.borrow_temporary(
            temporary_store, shape=(self.n_batches, n_vel + 1), dtype=int, device=self.device
        )
        wp.copy(dest=batch_write_cursors, src=self._batch_global_offsets)
        wp.launch(
            kernel=fill_batch_transpose,
            dim=n_vel,
            inputs=[
                t_mat.offsets,
                t_mat.columns,
                t_mat.values.view(dtype=mat13),
                self._strain_batch,
                batch_write_cursors,
            ],
            outputs=[self._batch_columns, self._batch_values],
            device=self.device,
        )
        batch_write_cursors.release()

        # Phase 2: scatter launches (one recorded launch per batch)
        # Each batch uses its row of batch_global_offsets as CSR offsets into
        # the shared flat columns/values arrays.
        self._scatter_launches = []
        self._initial_guess_launches = []
        for bi in range(self.n_batches):
            scatter_inputs = [
                self._batch_global_offsets[bi],
                self._batch_columns,
                self._batch_values,
                self.momentum.inv_volume,
            ]
            self._scatter_launches.append(
                wp.launch(
                    kernel=batched_scatter,
                    dim=n_vel,
                    inputs=[*scatter_inputs, self.delta_stress],
                    outputs=[self.momentum.velocity],
                    record_cmd=True,
                )
            )
            # Initial guess: same scatter but reads stress instead of delta_stress
            self._initial_guess_launches.append(
                wp.launch(
                    kernel=batched_scatter,
                    dim=n_vel,
                    inputs=[*scatter_inputs, self.rheology.stress],
                    outputs=[self.momentum.velocity],
                    record_cmd=True,
                )
            )

    @property
    def name(self):
        return "Gauss-Seidel (batched)"

    @property
    def solve_granularity(self):
        return 25

    def apply_initial_guess(self):
        for bi in range(self.n_batches):
            self._initial_guess_launches[bi].launch()

    def solve(self):
        for bi in range(self.n_batches):
            self._solve_launch.set_param_at_index(0, bi)
            self._solve_launch.launch()
            self._scatter_launches[bi].launch()

    def release(self):
        super().release()
        self._flat_color_offsets.release()
        self._flat_constraint_ids.release()
        self._reordered_n_entries.release()
        self._reordered_cols.release()
        self._reordered_vals_x.release()
        self._reordered_vals_y.release()
        self._reordered_vals_z.release()
        self._strain_batch.release()
        self._batch_global_offsets.release()
        self._batch_columns.release()
        self._batch_values.release()


class _JacobiSolver(_RheologySolver):
    def __init__(
        self,
        delassus_operator: _DelassusOperator,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        super().__init__(delassus_operator, split_mass=True, temporary_store=temporary_store)

        self.apply_stress_launch = self.delassus_operator.apply_stress_delta(
            self.delta_stress,
            self.momentum.velocity,
            record_cmd=True,
        )

        # Solve kernel
        jacobi_kernel = make_jacobi_solve_kernel(
            has_viscosity=self.rheology.has_viscosity,
            has_dilatancy=self.rheology.has_dilatancy,
            has_compliance_mat=self.rheology.compliance_mat.nnz > 0,
            strain_velocity_node_count=self.rheology.strain_velocity_node_count,
        )
        self.solve_local_launch = wp.launch(
            kernel=jacobi_kernel,
            dim=self.size,
            inputs=[
                self.rheology.yield_params,
                self.rheology.strain_node_volume,
                self.rheology.compliance_mat.offsets,
                self.rheology.compliance_mat.columns,
                self.rheology.compliance_mat.values,
                self.rheology.strain_mat.offsets,
                self.rheology.strain_mat.columns,
                self.rheology.strain_mat.values.view(dtype=mat13),
                self.delassus_operator.delassus_diagonal,
                self.delassus_operator.delassus_rotation,
                self.rheology.elastic_strain_delta,
                self.momentum.velocity,
                self.rheology.stress,
            ],
            outputs=[
                self.delta_stress,
            ],
            record_cmd=True,
        )

    @property
    def name(self):
        return "Jacobi"

    @property
    def solve_granularity(self):
        return 50

    def apply_initial_guess(self):
        # Apply initial guess
        self.delta_stress.assign(self.rheology.stress)
        self.apply_stress_launch.launch()

    def solve(self):
        self.solve_local_launch.launch()
        # Add jacobi delta
        self.apply_stress_launch.launch()
        array_axpy(x=self.delta_stress, y=self.rheology.stress, alpha=1.0, beta=1.0)


_ITERATIVE_LINEAR_SOLVERS = {
    "cg": cg,
    "cr": cr,
    "gmres": gmres,
}

_RHEOLOGY_SOLVERS = {
    "gauss-seidel": _GaussSeidelSolver,
    "gauss-seidel-soa": _ReorderedGaussSeidelSolver,
    "gauss-seidel-batched": _BatchedGaussSeidelSolver,
    "jacobi": _JacobiSolver,
    # short aliases
    "gs": _GaussSeidelSolver,
    "gs-soa": _ReorderedGaussSeidelSolver,
    "gs-batched": _BatchedGaussSeidelSolver,
}


def _resolve_batched_gs_batching(color_count: int, n_batches: int | None) -> tuple[int, int]:
    """Resolve batch count and colors per batch for batched Gauss-Seidel."""
    if color_count <= 0:
        raise ValueError("Batched Gauss-Seidel requires at least one color.")

    if n_batches is None:
        n_batches = 9 if color_count == 27 else 16

    if n_batches <= 0:
        raise ValueError(f"Batched Gauss-Seidel requires a positive batch count, got {n_batches}.")

    n_batches = min(n_batches, color_count)
    if color_count % n_batches != 0:
        raise ValueError(
            "Batched Gauss-Seidel requires the color count to be divisible by the batch count, "
            f"got color_count={color_count} and n_batches={n_batches}."
        )

    return n_batches, color_count // n_batches


class _LinearSolver:
    def __init__(
        self,
        delassus_operator: _DelassusOperator,
        method: str = "cr",
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        self.momentum = delassus_operator.momentum
        self.rheology = delassus_operator.rheology
        self.delassus_operator = delassus_operator
        self._method_name = method
        self._method_fn = _ITERATIVE_LINEAR_SOLVERS[method]

        self.delassus_operator.require_strain_mat_transpose()
        self.delassus_operator.compute_diagonal_factorization(split_mass=False)

        self.delta_velocity = fem.borrow_temporary_like(self.momentum.velocity, temporary_store)

        shape = self.rheology.compliance_mat.shape
        dtype = self.rheology.compliance_mat.dtype
        device = self.rheology.compliance_mat.device

        self.linear_operator = LinearOperator(shape=shape, dtype=dtype, device=device, matvec=self._delassus_matvec)
        self.preconditioner = LinearOperator(
            shape=shape, dtype=dtype, device=device, matvec=self._preconditioner_matvec
        )

    def _delassus_matvec(self, x: wp.array[vec6], y: wp.array[vec6], z: wp.array[vec6], alpha: float, beta: float):
        # dv = B^T x
        self.delta_velocity.zero_()
        self.delassus_operator.apply_stress_delta(x, self.delta_velocity)
        # z = alpha B dv + beta * y
        self.delassus_operator.apply_velocity_delta(self.delta_velocity, y, z, alpha, beta)

        # z += C x
        sp.bsr_mv(self.rheology.compliance_mat, x, z, alpha=alpha, beta=1.0)

    def _preconditioner_matvec(self, x, y, z, alpha, beta):
        wp.launch(
            kernel=jacobi_preconditioner,
            dim=self.delassus_operator.size,
            inputs=[
                self.delassus_operator.delassus_diagonal,
                self.delassus_operator.delassus_rotation,
                x,
                y,
                z,
                alpha,
                beta,
            ],
        )

    def solve(self, tol: float, tolerance_scale: float, max_iterations: int, use_graph: bool, verbose: bool):
        self.delassus_operator.apply_velocity_delta(
            self.momentum.velocity,
            self.rheology.elastic_strain_delta,
            self.rheology.plastic_strain_delta,
            alpha=-1.0,
            beta=-1.0,
        )

        with _ScopedDisableGC():
            end_iter, residual, atol = self._method_fn(
                A=self.linear_operator,
                M=self.preconditioner,
                b=self.rheology.plastic_strain_delta,
                x=self.rheology.stress,
                atol=tol * tolerance_scale,
                tol=tol,
                maxiter=max_iterations,
                check_every=0 if use_graph else 10,
                use_cuda_graph=use_graph,
            )

        if use_graph:
            end_iter = end_iter.numpy()[0]
            residual = residual.numpy()[0]
            atol = atol.numpy()[0]

        if verbose:
            res = math.sqrt(residual) / tolerance_scale
            print(f"{self.name} terminated after {end_iter} iterations with residual {res}")

    @property
    def name(self):
        return self._method_name.upper()

    def release(self):
        self.delta_velocity.release()


class _ContactSolver:
    def __init__(
        self,
        momentum: MomentumData,
        collision: CollisionData,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        self.momentum = momentum
        self.collision = collision

        self.delta_impulse = fem.borrow_temporary_like(self.collision.collider_impulse, temporary_store)
        self.collider_inv_mass = fem.borrow_temporary_like(self.collision.collider_friction, temporary_store)

        # Setup rigidity correction
        if self.collision.rigidity_operator is not None:
            J, IJtm = self.collision.rigidity_operator
            self.delta_body_qd = fem.borrow_temporary(temporary_store, shape=J.shape[1], dtype=float)

            wp.launch(
                compute_collider_inv_mass,
                dim=self.collision.collider_impulse.shape[0],
                inputs=[
                    J.offsets,
                    J.columns,
                    J.values,
                    IJtm.offsets,
                    IJtm.columns,
                    IJtm.values,
                ],
                outputs=[
                    self.collider_inv_mass,
                ],
            )

        else:
            self.collider_inv_mass.zero_()

    def release(self):
        self.delta_impulse.release()
        self.collider_inv_mass.release()
        if self.collision.rigidity_operator is not None:
            self.delta_body_qd.release()

    def apply_rigidity_operator(self):
        if self.collision.rigidity_operator is not None:
            apply_rigidity_operator(
                self.collision.rigidity_operator,
                self.delta_impulse,
                self.collision.collider_velocities,
                self.delta_body_qd,
            )


class _NodalContactSolver(_ContactSolver):
    def __init__(
        self,
        momentum: MomentumData,
        collision: CollisionData,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        super().__init__(momentum, collision, temporary_store)

        # define solve operation
        self.solve_collider_launch = wp.launch(
            kernel=solve_nodal_friction,
            dim=self.collision.collider_impulse.shape[0],
            inputs=[
                self.momentum.inv_volume,
                self.collision.collider_friction,
                self.collision.collider_adhesion,
                self.collision.collider_normals,
                self.collider_inv_mass,
                self.momentum.velocity,
                self.collision.collider_velocities,
                self.collision.collider_impulse,
                self.delta_impulse,
            ],
            record_cmd=True,
        )

    def apply_initial_guess(self):
        # Apply initial impulse guess
        wp.launch(
            kernel=apply_nodal_impulse_warmstart,
            dim=self.collision.collider_impulse.shape[0],
            inputs=[
                self.collision.collider_impulse,
                self.collision.collider_friction,
                self.collision.collider_normals,
                self.collision.collider_adhesion,
                self.momentum.inv_volume,
                self.momentum.velocity,
                self.delta_impulse,
            ],
        )
        self.apply_rigidity_operator()

    def solve(self):
        self.solve_collider_launch.launch()
        self.apply_rigidity_operator()


class _SubgridContactSolver(_ContactSolver):
    def __init__(
        self,
        momentum: MomentumData,
        collision: CollisionData,
        temporary_store: fem.TemporaryStore | None = None,
    ) -> None:
        super().__init__(momentum, collision, temporary_store)

        self.collider_delassus_diagonal = fem.borrow_temporary_like(self.collider_inv_mass, temporary_store)

        sp.bsr_set_transpose(dest=self.collision.transposed_collider_mat, src=self.collision.collider_mat)

        wp.launch(
            compute_collider_delassus_diagonal,
            dim=self.collision.collider_impulse.shape[0],
            inputs=[
                self.collision.collider_mat.offsets,
                self.collision.collider_mat.columns,
                self.collision.collider_mat.values,
                self.collider_inv_mass,
                self.collision.transposed_collider_mat.offsets,
                self.momentum.inv_volume,
            ],
            outputs=[
                self.collider_delassus_diagonal,
            ],
        )

        # define solve operation
        self.apply_collider_impulse_launch = wp.launch(
            apply_subgrid_impulse,
            dim=self.momentum.velocity.shape[0],
            inputs=[
                self.collision.transposed_collider_mat.offsets,
                self.collision.transposed_collider_mat.columns,
                self.collision.transposed_collider_mat.values,
                self.momentum.inv_volume,
                self.delta_impulse,
                self.momentum.velocity,
            ],
            record_cmd=True,
        )

        self.solve_collider_launch = wp.launch(
            kernel=solve_subgrid_friction,
            dim=self.collision.collider_impulse.shape[0],
            inputs=[
                self.momentum.velocity,
                self.collision.collider_mat.offsets,
                self.collision.collider_mat.columns,
                self.collision.collider_mat.values,
                self.collision.collider_friction,
                self.collision.collider_adhesion,
                self.collision.collider_normals,
                self.collider_delassus_diagonal,
                self.collision.collider_velocities,
                self.collision.collider_impulse,
                self.delta_impulse,
            ],
            record_cmd=True,
        )

    def apply_initial_guess(self):
        wp.launch(
            apply_subgrid_impulse_warmstart,
            dim=self.delta_impulse.shape[0],
            inputs=[
                self.collision.collider_friction,
                self.collision.collider_normals,
                self.collision.collider_adhesion,
                self.collision.collider_impulse,
                self.delta_impulse,
            ],
        )
        self.apply_collider_impulse_launch.launch()
        self.apply_rigidity_operator()

    def solve(self):
        self.solve_collider_launch.launch()
        self.apply_collider_impulse_launch.launch()
        self.apply_rigidity_operator()

    def release(self):
        self.collider_delassus_diagonal.release()
        super().release()


def _run_solver_loop(
    rheology_solver: _RheologySolver,
    contact_solver: _ContactSolver,
    max_iterations: int,
    tolerance: float,
    l2_tolerance_scale: float,
    use_graph: bool,
    verbose: bool,
    temporary_store: fem.TemporaryStore,
):
    solve_graph = None
    if use_graph:
        solve_granularity = 5

        iteration_and_condition = fem.borrow_temporary(temporary_store, shape=(2,), dtype=int)
        iteration_and_condition.fill_(1)

        iteration = iteration_and_condition[:1]
        condition = iteration_and_condition[1:]

        def do_iteration_with_condition():
            for _k in range(solve_granularity):
                contact_solver.solve()
                rheology_solver.solve()
            residual = rheology_solver.eval_residual()
            wp.launch(
                update_condition,
                dim=1,
                inputs=[
                    tolerance * tolerance,
                    l2_tolerance_scale * l2_tolerance_scale,
                    solve_granularity,
                    max_iterations,
                    residual,
                    iteration,
                    condition,
                ],
            )

        device = rheology_solver.device
        if device.is_capturing:
            with _ScopedDisableGC():
                wp.capture_while(condition, do_iteration_with_condition)
        else:
            with _ScopedDisableGC():
                with wp.ScopedCapture(force_module_load=False) as capture:
                    wp.capture_while(condition, do_iteration_with_condition)
            solve_graph = capture.graph
            wp.capture_launch(solve_graph)

            if verbose:
                residual = rheology_solver.eval_residual().numpy()
                res_l2, res_linf = math.sqrt(residual[0, 0]) / l2_tolerance_scale, math.sqrt(residual[1, 0])
                print(
                    f"{rheology_solver.name} terminated after {iteration_and_condition.numpy()[0]} iterations with residuals {res_l2}, {res_linf}"
                )

        iteration_and_condition.release()
    else:
        solve_granularity = rheology_solver.solve_granularity

        for batch in range(max_iterations // solve_granularity):
            for _k in range(solve_granularity):
                contact_solver.solve()
                rheology_solver.solve()

            residual = rheology_solver.eval_residual().numpy()
            res_l2, res_linf = math.sqrt(residual[0, 0]) / l2_tolerance_scale, math.sqrt(residual[1, 0])

            if verbose:
                print(
                    f"{rheology_solver.name} iteration #{(batch + 1) * solve_granularity} \t res(l2)={res_l2}, res(linf)={res_linf}"
                )
            if res_l2 < tolerance and res_linf < tolerance:
                break

    return solve_graph


def solve_rheology(
    solver: str | Sequence[str],
    max_iterations: int,
    tolerance: float,
    momentum: MomentumData,
    rheology: RheologyData,
    collision: CollisionData,
    jacobi_warmstart_smoother_iterations: int = 5,
    temporary_store: fem.TemporaryStore | None = None,
    use_graph: bool = True,
    verbose: bool = wp.config.verbose,
):
    """Solve coupled plasticity and collider contact to compute grid velocities.

    This function executes the implicit rheology loop that couples plastic
    stress update and nodal frictional contact with colliders:

    - Builds the Delassus operator diagonal blocks and rotates all local
      quantities into the decoupled eigenbasis (normal vs tangential).
    - Runs either Gauss-Seidel (with coloring) or Jacobi iterations to solve
      the local stress projection problem per strain node.
    - Applies collider impulses and, when provided, a rigidity coupling step on
      collider velocities each iteration.
    - Iterates until the residual on the stress update falls below
      ``tolerance`` or ``max_iterations`` is reached. Optionally records and
      executes CUDA graphs to reduce CPU overhead.

    On exit, the stress field is rotated back to world space and the elastic
    strain increment and plastic strain delta fields are produced.

    Args:
        solver: Solver type string or ordered sequence of solver type strings.
            Base solvers: ``"gauss-seidel"`` (or ``"gs"``),
            ``"gauss-seidel-soa"`` (or ``"gs-soa"``),
            ``"gauss-seidel-batched"`` (or ``"gs-batched"``),
            ``"jacobi"``, ``"cg"``, ``"cr"``, ``"gmres"``.
            Chained solvers run left-to-right as warmstarts for the
            final solver, e.g. ``("cr", "gs")`` runs CR then Gauss-Seidel,
            ``("cg", "jacobi", "gs-batched")`` runs CG, then a Jacobi smoother,
            then batched Gauss-Seidel.
            ``"gauss-seidel-soa"`` uses an entry-major SoA strain
            matrix layout for improved memory coalescing.
            ``"gauss-seidel-batched"`` additionally merges colors into
            batches with Jacobi-style mass splitting within each
            batch. Good for wide velocity stencils (B2/B3).
            The iterative linear solvers (``"cg"``, ``"cr"``, ``"gmres"``)
            only support solid materials without contacts.
        max_iterations: Maximum number of nonlinear iterations.
        tolerance: Solver tolerance for the stress residual (L2 norm).
        momentum: :class:`MomentumData` containing per-node inverse volume
            and velocity DOFs.
        rheology: :class:`RheologyData` containing strain/compliance matrices,
            yield parameters, coloring data, and output stress/strain arrays.
        collision: :class:`CollisionData` containing collider matrices, friction,
            adhesion, normals, velocities, rigidity operator, and impulse arrays.
        jacobi_warmstart_smoother_iterations: Number of Jacobi smoother
            iterations to run before the main Gauss-Seidel solve (ignored
            for Jacobi solver).
        temporary_store: Temporary storage arena for intermediate arrays.
        use_graph: If True, uses conditional CUDA graphs for the iteration loop.
        verbose: If True, prints residuals/iteration counts.

    Returns:
        A captured execution graph handle when ``use_graph`` is True and the
        device supports it; otherwise ``None``.
    """

    subgrid_collisions = collision.collider_mat.nnz > 0
    if subgrid_collisions:
        contact_solver = _SubgridContactSolver(momentum, collision, temporary_store)
    else:
        contact_solver = _NodalContactSolver(momentum, collision, temporary_store)

    contact_solver.apply_initial_guess()

    delassus_operator = _DelassusOperator(rheology, momentum, temporary_store)
    tolerance_scale = math.sqrt(1 + delassus_operator.size)

    solvers = (solver,) if isinstance(solver, str) else tuple(solver)
    if len(solvers) == 0:
        raise ValueError("Solver sequence must contain at least one solver.")

    if len(solvers) == 1 and solvers[0] in _ITERATIVE_LINEAR_SOLVERS:
        if collision.has_colliders:
            raise ValueError(
                f"Solver {solvers[0]!r} does not support contact; use a GS or Jacobi solver when contacts are active."
            )

    if solvers[0] in _ITERATIVE_LINEAR_SOLVERS:
        rheology_solver = _LinearSolver(delassus_operator, method=solvers[0], temporary_store=temporary_store)
        rheology_solver.solve(tolerance, tolerance_scale, max_iterations, use_graph, verbose)
        rheology_solver.release()

        if len(solvers) == 1:
            # linear solver only
            delassus_operator.apply_stress_delta(rheology.stress, momentum.velocity)
            delassus_operator.postprocess_stress_and_strain()
            delassus_operator.release()
            contact_solver.release()
            return None

        # linear solver as warmstart
        solvers = solvers[1:]

    if len(solvers) > 1 and solvers[0] == "jacobi":
        # jacobi warmstart smoother
        old_v = wp.clone(momentum.velocity)
        warmstart_solver = _JacobiSolver(delassus_operator, temporary_store)
        warmstart_solver.apply_initial_guess()
        for _ in range(jacobi_warmstart_smoother_iterations):
            warmstart_solver.solve()
        warmstart_solver.release()
        momentum.velocity.assign(old_v)

        # continue with next solver
        solvers = solvers[1:]

    if len(solvers) != 1:
        raise ValueError(
            f"Invalid solver sequence {solver!r}: unexpected tokens {solvers[1:]!r}. "
            f"Accepted form: [linear, ][jacobi, ]<final>, where linear is one of "
            f"{list(_ITERATIVE_LINEAR_SOLVERS)} and final is one of {list(_RHEOLOGY_SOLVERS)}."
        )
    rheology_solver_class = _RHEOLOGY_SOLVERS.get(solvers[0])
    if rheology_solver_class is None:
        raise ValueError(f"Invalid solver {solvers[0]!r}. Accepted values: {list(_RHEOLOGY_SOLVERS)}.")

    rheology_solver = rheology_solver_class(delassus_operator, temporary_store)
    rheology_solver.apply_initial_guess()

    solve_graph = _run_solver_loop(
        rheology_solver, contact_solver, max_iterations, tolerance, tolerance_scale, use_graph, verbose, temporary_store
    )

    # release temporary storage
    rheology_solver.release()
    contact_solver.release()

    delassus_operator.postprocess_stress_and_strain()
    delassus_operator.release()

    return solve_graph
