# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cache bilateral-unilateral coupling as structurally nonzero joint blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ...core.types import vec6f
from .sparse_kernels import _subgroup_sum, _unilateral_nzb_offsets

if TYPE_CHECKING:
    from ...dynamics.dual import DualProblem
    from .joint_blocks import JointBlockSolver
    from .sparse import SparseDVIPath

wp.set_module_options({"enable_backward": False})


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __popcll((unsigned long long)value);
#else
return __builtin_popcountll((unsigned long long)value);
#endif
""")
def _count_bits(value: wp.uint64) -> wp.int32: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __ffsll((long long)value) - 1;
#else
return __builtin_ffsll((long long)value) - 1;
#endif
""")
def _first_bit(value: wp.uint64) -> wp.int32: ...


@wp.func
def read_packed_coupling(
    values: wp.array[wp.float32],
    masks: wp.array[wp.uint64],
    w: int,
    col: int,
    group: int,
    local: int,
    width: int,
    slots: int,
) -> wp.float32:
    """Read a structural coefficient, mapping its group to a packed slot."""
    mask = masks[w * width + col]
    bit = wp.uint64(1) << wp.uint64(group)
    result = wp.float32(0)
    if (mask & bit) != wp.uint64(0):
        slot = _count_bits(mask & (bit - wp.uint64(1)))
        result = values[((w * slots + slot) * 6 + local) * width + col]
    return result


@wp.kernel
def _assemble(
    dim: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    nbc: wp.array[wp.int32],
    nl: wp.array[wp.int32],
    bcio: wp.array[wp.int32],
    lio: wp.array[wp.int32],
    cio: wp.array[wp.int32],
    vio: wp.array[wp.int32],
    P: wp.array[wp.float32],
    num_nzb: wp.array[wp.int32],
    nzb_start: wp.array[wp.int32],
    coords: wp.array2d[wp.int32],
    weighted: wp.array[vec6f],
    raw: wp.array[vec6f],
    limits: wp.array[wp.int32],
    contacts: wp.array[wp.int32],
    bounded_offsets: wp.array[wp.vec2i],
    limit_offsets: wp.array[wp.int32],
    contact_offsets: wp.array[wp.int32],
    row_origins: wp.array[wp.int32],
    row_starts: wp.array[wp.int32],
    row_indices: wp.array[wp.int32],
    order: wp.array[wp.int32],
    body_seeds: wp.array[wp.uint64],
    width: int,
    slots: int,
    masks: wp.array[wp.uint64],
    values: wp.array[wp.float32],
):
    """Cache exact shared-body coefficients without materializing zero blocks."""
    w, col = wp.tid()
    if col >= dim[w] - njc[w]:
        return
    offsets = _unilateral_nzb_offsets(
        col,
        njc[w],
        nbc[w],
        nbc[w] + nl[w],
        bcio[w],
        lio[w],
        cio[w],
        nzb_start[w] + num_nzb[w],
        coords,
        limits,
        contacts,
        bounded_offsets,
        limit_offsets,
        contact_offsets,
    )
    body0, body1 = wp.int32(-1), wp.int32(-1)
    jac0, jac1 = vec6f(), vec6f()
    mask = wp.uint64(0)
    if offsets[0] >= 0:
        body0 = coords[offsets[0], 1]
        jac0 = raw[offsets[0]]
        mask = mask | body_seeds[body0 // 6]
    if offsets[1] >= 0:
        body1 = coords[offsets[1], 1]
        jac1 = raw[offsets[1]]
        mask = mask | body_seeds[body1 // 6]
    masks[w * width + col] = mask
    remaining = mask
    slot = wp.int32(0)
    scale = P[vio[w] + njc[w] + col]
    while remaining != wp.uint64(0):
        group = _first_bit(remaining)
        for local in range(6):
            row = order[group * 6 + local]
            value = wp.float32(0)
            if row >= 0:
                index = row_origins[w] + row
                for entry in range(row_starts[index], row_starts[index + 1]):
                    block = row_indices[entry]
                    body = coords[block, 1]
                    weighted_j = weighted[block]
                    if offsets[0] >= 0 and body == body0:
                        for component in range(6):
                            value += weighted_j[component] * jac0[component]
                    if offsets[1] >= 0 and body == body1:
                        for component in range(6):
                            value += weighted_j[component] * jac1[component]
            values[((w * slots + slot) * 6 + local) * width + col] = value * scale
        slot += 1
        remaining = remaining & (remaining - wp.uint64(1))


@wp.kernel
def _build_rhs(
    pvio: wp.array[wp.int32],
    njc: wp.array[wp.int32],
    vf: wp.array[wp.float32],
    dim: wp.array[wp.int32],
    lambdas: wp.array[wp.float32],
    bvio: wp.array[wp.int32],
    scale: wp.array[wp.float32],
    order: wp.array[wp.int32],
    masks: wp.array[wp.uint64],
    values: wp.array[wp.float32],
    width: int,
    slots: int,
    rhs: wp.array[wp.float32],
):
    """Reuse each sparse lookup and impulse across a joint's six RHS rows."""
    w, group, lane = wp.tid()
    value = vec6f(0.0)
    bit = wp.uint64(1) << wp.uint64(group)
    for col in range(lane, dim[w] - njc[w], 16):
        mask = masks[w * width + col]
        impulse = lambdas[pvio[w] + njc[w] + col]
        slot = int(0)
        present = (mask & bit) != wp.uint64(0)
        if present:
            slot = _count_bits(mask & (bit - wp.uint64(1)))
        for local in range(6):
            coefficient = wp.float32(0.0)
            if present:
                coefficient = values[((w * slots + slot) * 6 + local) * width + col]
            value[local] += coefficient * impulse
    for local in range(6):
        total = _subgroup_sum(value[local], 16)
        row = order[group * 6 + local]
        if lane == 0 and row >= 0 and row < njc[w]:
            rhs[bvio[w] + row] = -scale[bvio[w] + row] * (total + vf[pvio[w] + row])


class PackedCoupling:
    """Cache each coupling coefficient once for RHS and response solves.

    A column's two bodies determine its structural joint-group mask. Slots
    follow set-bit order, with six rows per slot including ragged-group padding.
    The original dense cache remains available to the caller's fallback path.
    """

    @classmethod
    def create(cls, path: SparseDVIPath, blocks: JointBlockSolver | None) -> PackedCoupling | None:
        """Select compact storage only for supported saturated joint-block solves."""
        if path.size.num_worlds < 2048 or blocks is None or blocks._persistent or blocks._body_seeds is None:
            return None
        seeds = set(blocks._body_seeds.numpy().tolist())
        slots = max((int(a) | int(b)).bit_count() for a in seeds for b in seeds)
        entries = path.size.num_worlds * blocks._n * slots * 6
        if slots == 0 or entries > 2**31 - 1:
            return None
        return cls(path, blocks, slots=slots)

    def __init__(self, path: SparseDVIPath, blocks: JointBlockSolver, *, slots: int | None = None):
        """Allocate topology-bounded cache storage before graph capture."""
        self._device = path.device
        self._worlds = path.size.num_worlds
        self._blocks = blocks
        # The joint-block dimension guard selects fallback whenever nu > n.
        self.width = blocks._n
        self._body_seeds = blocks._body_seeds
        if slots is None:
            seeds = set(self._body_seeds.numpy().tolist())
            slots = max((int(a) | int(b)).bit_count() for a in seeds for b in seeds)
        self.slots = slots
        self.masks = wp.empty(self._worlds * self.width, dtype=wp.uint64, device=self._device)
        self.values = wp.empty(self._worlds * self.width * self.slots * 6, dtype=wp.float32, device=self._device)

    def assemble(self, path: SparseDVIPath, problem: DualProblem) -> None:
        """Overwrite all active masks and coefficients after inequality mapping."""
        data, state, jacobian = problem.data, path.data.state, path.jacobians
        operator = problem.delassus
        if operator._needs_update:
            operator.update()
        bsm = operator.bsm
        wp.launch(
            _assemble,
            dim=(self._worlds, self.width),
            inputs=[
                data.dim,
                data.njc,
                data.nbc,
                data.nl,
                data.bcio,
                data.lio,
                data.cio,
                data.vio,
                data.P,
                bsm.num_nzb,
                bsm.nzb_start,
                bsm.nzb_coords,
                bsm.nzb_values,
                operator.constraint_jacobian.nzb_values,
                state.limit_indices,
                state.contact_indices,
                jacobian.bounded_constraint_nzb_offsets,
                jacobian.limit_constraint_nzb_offsets,
                jacobian.contact_constraint_nzb_offsets,
                *path.bilateral_row_nzb_topology,
                self._blocks._schedule["order"],
                self._body_seeds,
                self.width,
                self.slots,
                self.masks,
                self.values,
            ],
            device=self._device,
        )

    def build_rhs(self, path: SparseDVIPath, problem: DualProblem) -> None:
        """Reuse cached coefficients for either initial or final bilateral solve."""
        data, state = problem.data, path.data.state
        wp.launch(
            _build_rhs,
            dim=(self._worlds, self._blocks._groups, 16),
            inputs=[
                data.vio,
                data.njc,
                data.v_f,
                data.dim,
                path.data.solution.lambdas,
                path.data.bilateral_operator.info.vio,
                state.bilateral_preconditioner,
                self._blocks._schedule["order"],
                self.masks,
                self.values,
                self.width,
                self.slots,
                state.bilateral_rhs,
            ],
            block_dim=128,
            device=self._device,
        )
