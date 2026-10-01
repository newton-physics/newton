# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Topology-derived, leaf-first mass factors for small articulated forests.

The sparse coordinate ``i`` names physical DOF ``permutation[i]``. With that
permutation, ``P H P.T = L L.T`` and ``Linv = inverse(L)`` have the same lower
ancestor pattern. Neither assembly nor factorization allocates a dense H.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
import warp as wp


@wp.struct
class _SparseMassMatrixIndices:
    dof_count: int
    nonzero_count: int
    factor_level_count: int
    permutation: wp.array[int]
    inverse_permutation: wp.array[int]
    row_offsets: wp.array[int]
    columns: wp.array[int]
    entry_rows: wp.array[int]
    lookup: wp.array2d[int]
    dof_joint: wp.array[int]
    ancestor_mask: wp.array[wp.uint64]
    column_offsets: wp.array[int]
    column_entries: wp.array[int]
    factor_level_offsets: wp.array[int]
    factor_level_columns: wp.array[int]
    factor_level_entry_offsets: wp.array[int]
    factor_level_entries: wp.array[int]


@dataclass(frozen=True, eq=False)
class _SparseMassMatrixPlan:
    """Immutable sparsity for one articulation topology, independent of pose.

    Parent indices are local joint indices, not body indices. Fixed joints have
    zero DOFs; physical DOFs are consecutive in input joint order. Mask bits use
    sparse coordinates, including every coordinate in an ancestor joint block.
    The initial device path is bounded to 64 DOFs, not to a robot or task name.
    """

    joint_parent: np.ndarray
    """Local parent joint indices, with -1 for each root."""
    joint_dof_count: np.ndarray
    """Number of physical coordinates in each joint block."""
    permutation: np.ndarray
    """Sparse-coordinate to physical-coordinate permutation."""
    inverse_permutation: np.ndarray
    """Physical-coordinate to sparse-coordinate permutation."""
    row_offsets: np.ndarray
    """CSR offsets into the packed lower factor."""
    columns: np.ndarray
    """Sparse-coordinate column index for each packed entry."""
    entry_rows: np.ndarray
    """Sparse-coordinate row index for each packed entry."""
    lookup: np.ndarray
    """Dense index table containing a packed entry index or -1."""
    dof_joint: np.ndarray
    """Local joint owning each physical coordinate."""
    ancestor_mask: np.ndarray
    """Sparse-coordinate ancestor support for each physical coordinate."""
    joint_ancestor_mask: np.ndarray
    """Sparse-coordinate endpoint support including fixed-joint endpoints."""
    column_offsets: np.ndarray
    """CSC offsets for each factor column's ancestor chain."""
    column_entries: np.ndarray
    """Packed factor indices ordered by column, then increasing row."""
    factor_level_offsets: np.ndarray
    """Offsets of independent Cholesky pivots at each dependency level."""
    factor_level_columns: np.ndarray
    """Sparse-coordinate pivot columns grouped by dependency level."""
    factor_level_entry_offsets: np.ndarray
    """Offsets of off-diagonal factor entries at each dependency level."""
    factor_level_entries: np.ndarray
    """Packed off-diagonal entries grouped by their column's dependency level."""

    @property
    def dof_count(self) -> int:
        return len(self.permutation)

    @property
    def nonzero_count(self) -> int:
        return len(self.columns)

    @property
    def factor_level_count(self) -> int:
        return len(self.factor_level_offsets) - 1

    @classmethod
    def build(cls, joint_parent, joint_dof_count) -> "_SparseMassMatrixPlan":
        """Build a fill-free scalar pattern from a checked joint forest."""
        parents = np.asarray(joint_parent)
        counts = np.asarray(joint_dof_count)
        if parents.ndim != 1 or counts.shape != parents.shape or not len(parents):
            raise ValueError("Joint parents and DOF counts must be nonempty one-dimensional arrays of equal length")
        if not np.issubdtype(parents.dtype, np.integer) or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("Joint parents and DOF counts must be integers")
        if np.any(parents < -1) or np.any(parents >= len(parents)) or np.any(counts < 0):
            raise ValueError("Invalid joint parent or negative DOF count")
        if np.any(counts > 64):
            raise ValueError("Sparse mass-matrix factors currently require 1 to 64 DOFs")
        parents = parents.astype(np.int32)
        counts = counts.astype(np.int32)
        n = int(np.sum(counts, dtype=np.int64))
        if n < 1 or n > 64:
            raise ValueError("Sparse mass-matrix factors currently require 1 to 64 DOFs")
        children = [[] for _ in parents]
        roots = []
        for joint, parent in enumerate(parents):
            if parent == -1:
                roots.append(joint)
            else:
                children[parent].append(joint)
        preorder = []
        pending = roots[::-1]
        while pending:
            joint = pending.pop()
            preorder.append(joint)
            pending.extend(children[joint][::-1])
        if len(preorder) != len(parents):
            raise ValueError("Joint parents must form an acyclic forest")
        starts = np.concatenate(([0], np.cumsum(counts)))
        permutation = np.array(
            [dof for joint in reversed(preorder) for dof in range(starts[joint + 1] - 1, starts[joint] - 1, -1)],
            dtype=np.int32,
        )
        inverse_permutation = np.argsort(permutation).astype(np.int32)
        dof_joint = np.repeat(np.arange(len(parents), dtype=np.int32), counts)
        joint_masks = [0] * len(parents)
        for joint in range(len(parents)):
            ancestor = joint
            while ancestor != -1:
                for dof in range(starts[ancestor], starts[ancestor + 1]):
                    joint_masks[joint] |= 1 << int(inverse_permutation[dof])
                ancestor = int(parents[ancestor])
        row_offsets = [0]
        columns = []
        entry_rows = []
        lookup = np.full((n, n), -1, dtype=np.int32)
        for row in range(n):
            for col in range(row + 1):
                col_joint = int(dof_joint[permutation[col]])
                if joint_masks[col_joint] & (1 << row):
                    lookup[row, col] = len(columns)
                    entry_rows.append(row)
                    columns.append(col)
            row_offsets.append(len(columns))
        column_offsets = [0]
        column_entries = []
        for col in range(n):
            column_entries.extend(int(lookup[row, col]) for row in range(col, n) if lookup[row, col] >= 0)
            column_offsets.append(len(column_entries))

        # Reverse DFS places every descendant subtree in one contiguous row
        # interval. A pivot depends only on those earlier descendants; sibling
        # branches can therefore be factored at the same dependency level.
        levels = np.zeros(n, dtype=np.int32)
        for row in range(n):
            first = row - (row_offsets[row + 1] - row_offsets[row]) + 1
            if first < row:
                levels[row] = 1 + int(np.max(levels[first:row]))
        level_count = int(np.max(levels)) + 1
        level_columns = np.argsort(levels, kind="stable").astype(np.int32)
        level_offsets = np.concatenate(([0], np.cumsum(np.bincount(levels, minlength=level_count))))
        level_entries = sorted(
            (entry for entry, (row, col) in enumerate(zip(entry_rows, columns, strict=True)) if row != col),
            key=lambda entry: (levels[columns[entry]], columns[entry], entry_rows[entry]),
        )
        entry_counts = np.bincount([levels[columns[entry]] for entry in level_entries], minlength=level_count)
        values = {
            "joint_parent": parents,
            "joint_dof_count": counts,
            "permutation": permutation,
            "inverse_permutation": inverse_permutation,
            "row_offsets": np.asarray(row_offsets, dtype=np.int32),
            "columns": np.asarray(columns, dtype=np.int32),
            "entry_rows": np.asarray(entry_rows, dtype=np.int32),
            "lookup": lookup,
            "dof_joint": dof_joint,
            "ancestor_mask": np.asarray([joint_masks[joint] for joint in dof_joint], dtype=np.uint64),
            "joint_ancestor_mask": np.asarray(joint_masks, dtype=np.uint64),
            "column_offsets": np.asarray(column_offsets, dtype=np.int32),
            "column_entries": np.asarray(column_entries, dtype=np.int32),
            "factor_level_offsets": np.asarray(level_offsets, dtype=np.int32),
            "factor_level_columns": level_columns,
            "factor_level_entry_offsets": np.asarray(np.concatenate(([0], np.cumsum(entry_counts))), dtype=np.int32),
            "factor_level_entries": np.asarray(level_entries, dtype=np.int32),
        }
        for value in values.values():
            value.setflags(write=False)
        return cls(**values)

    def endpoint_support(self, joint_a: int, joint_b: int = -1) -> np.ndarray:
        """Return sparse DOFs supporting either endpoint; -1 denotes the world."""
        mask = 0
        for joint in (joint_a, joint_b):
            if joint < -1 or joint >= len(self.joint_parent):
                raise ValueError("Endpoint must be a local joint index or -1")
            if joint != -1:
                mask |= int(self.joint_ancestor_mask[joint])
        return np.asarray([dof for dof in range(self.dof_count) if mask & (1 << dof)], dtype=np.int32)

    def to_device(self, device) -> _SparseMassMatrixIndices:
        """Upload the topology once; all per-step values remain caller-owned."""
        indices = _SparseMassMatrixIndices()
        indices.dof_count = self.dof_count
        indices.nonzero_count = self.nonzero_count
        indices.factor_level_count = self.factor_level_count
        for name in (
            "permutation",
            "inverse_permutation",
            "row_offsets",
            "columns",
            "entry_rows",
            "lookup",
            "dof_joint",
            "ancestor_mask",
            "column_offsets",
            "column_entries",
            "factor_level_offsets",
            "factor_level_columns",
            "factor_level_entry_offsets",
            "factor_level_entries",
        ):
            dtype = wp.uint64 if name == "ancestor_mask" else int
            setattr(indices, name, wp.array(getattr(self, name), dtype=dtype, device=device))
        return indices


@cache
def _get_crba_sparse_factor_kernel(dof_count: int, nonzero_count: int, *, warps_per_block: int = 4) -> wp.Kernel:
    """Assemble CRBA and produce packed Linv, one warp per articulation.

    Launch ``group_count * 32`` threads with ``warps_per_block * 32`` threads per
    block. All value arrays must be contiguous. A zero articulation update mask
    retains the inverse factor and status. Status is zero after a successful refresh,
    or one for a nonpositive/nonfinite pivot; no pivot floor is introduced.
    """
    n, nnz, warps = int(dof_count), int(nonzero_count), int(warps_per_block)
    if not 1 <= n <= 64 or not n <= nnz <= n * (n + 1) // 2 or not 1 <= warps <= 32:
        raise ValueError("Invalid sparse factor dimensions or warp count")
    snippet = f"""
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
    const int art = group_to_art.data[group];
    if (mass_update_mask.data[art] == 0) return;
#if defined(__CUDA_ARCH__)
    const int lane = tid & 31;
    const int stride = 32;
    const int warp = threadIdx.x >> 5;
    __shared__ float factors[{warps * nnz}];
    __shared__ float inverses[{warps * nnz}];
    __shared__ float forces[{warps * n * 6}];
    float* factor = factors + warp * {nnz};
    float* inverse = inverses + warp * {nnz};
    float* force = forces + warp * {n * 6};
#else
    if ((tid & 31) != 0) return;
    const int lane = 0;
    const int stride = 1;
    float factor[{nnz}];
    float inverse[{nnz}];
    float force[{n * 6}];
#endif
    const int start = articulation_dof_start.data[art];
    for (int physical = lane; physical < {n}; physical += stride) {{
        const int joint = articulation_start.data[art] + indices.dof_joint.data[physical];
        const auto value = wp::mul(body_I_c.data[joint_child.data[joint]], joint_S_s.data[start + physical]);
        for (int k = 0; k < 6; ++k) force[physical * 6 + k] = value.c[k];
    }}
    if (lane == 0) status.data[group] = 0;
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
    for (int entry = lane; entry < {nnz}; entry += stride) {{
        const int row = indices.entry_rows.data[entry];
        const int col = indices.columns.data[entry];
        const int physical_row = indices.permutation.data[row];
        const int physical_col = indices.permutation.data[col];
        const auto motion = joint_S_s.data[start + physical_row];
        float value = 0.0f;
        for (int k = 0; k < 6; ++k) value += motion.c[k] * force[physical_col * 6 + k];
        if (row == col) {{
            value += R_group.data[group * {n} + physical_row];
            if (fused_augmented_drive != 0) {{
                const int drive = drive_row_by_dof.data[start + physical_row];
                if (drive >= 0 && row_K.data[drive] > 0.0f) value += row_K.data[drive];
            }}
        }}
        factor[entry] = value;
    }}
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
    for (int level = 0; level < indices.factor_level_count; ++level) {{
        const int level_begin = indices.factor_level_offsets.data[level];
        const int level_end = indices.factor_level_offsets.data[level + 1];
        for (int pivot = level_begin + lane; pivot < level_end; pivot += stride) {{
            const int col = indices.factor_level_columns.data[pivot];
            const int begin = indices.row_offsets.data[col];
            const int diagonal = indices.row_offsets.data[col + 1] - 1;
            float value = factor[diagonal];
            for (int entry = begin; entry < diagonal; ++entry) value -= factor[entry] * factor[entry];
            if (!(value > 0.0f) || !wp::isfinite(value)) {{
#if defined(__CUDA_ARCH__)
                atomicExch(status.data + group, 1);
#else
                status.data[group] = 1;
#endif
            }}
            factor[diagonal] = sqrtf(value);
        }}
#if defined(__CUDA_ARCH__)
        __syncwarp();
#endif
        const int entry_begin = indices.factor_level_entry_offsets.data[level];
        const int entry_end = indices.factor_level_entry_offsets.data[level + 1];
        for (int work = entry_begin + lane; work < entry_end; work += stride) {{
            const int target = indices.factor_level_entries.data[work];
            const int row = indices.entry_rows.data[target];
            const int col = indices.columns.data[target];
            const int begin = indices.row_offsets.data[col];
            const int diagonal = indices.row_offsets.data[col + 1] - 1;
            // Ancestor closure makes the column's whole prefix present in this
            // row. Both prefixes are contiguous under the reversed DFS order.
            const int row_begin = target - (diagonal - begin);
            float value = factor[target];
            for (int entry = begin; entry < diagonal; ++entry) {{
                value -= factor[row_begin + entry - begin] * factor[entry];
            }}
            factor[target] = value / factor[diagonal];
        }}
#if defined(__CUDA_ARCH__)
        __syncwarp();
#endif
    }}
    // Each inverse column follows only its ancestor chain. In particular, a
    // broad root row no longer scans every unrelated descendant for each RHS.
    for (int col = lane; col < {n}; col += stride) {{
        const int begin = indices.column_offsets.data[col];
        const int end = indices.column_offsets.data[col + 1];
        for (int position = begin; position < end; ++position) {{
            const int target = indices.column_entries.data[position];
            const int row = indices.entry_rows.data[target];
            const int diagonal = indices.row_offsets.data[row + 1] - 1;
            float value = row == col ? 1.0f : 0.0f;
            for (int previous = begin; previous < position; ++previous) {{
                const int other = indices.column_entries.data[previous];
                const int factor_entry = diagonal - row + indices.entry_rows.data[other];
                value -= factor[factor_entry] * inverse[other];
            }}
            inverse[target] = value / factor[diagonal];
        }}
    }}
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
    const bool valid = status.data[group] == 0;
    for (int entry = lane; entry < {nnz}; entry += stride) {{
        Linv_group.data[group * {nnz} + entry] = valid ? inverse[entry] : NAN;
    }}
"""

    @wp.func_native(snippet)
    def factor_native(
        tid: int,
        group_to_art: wp.array[int],
        mass_update_mask: wp.array[int],
        articulation_start: wp.array[int],
        articulation_dof_start: wp.array[int],
        joint_child: wp.array[int],
        joint_S_s: wp.array[wp.spatial_vector],
        body_I_c: wp.array[wp.spatial_matrix],
        R_group: wp.array2d[float],
        fused_augmented_drive: int,
        drive_row_by_dof: wp.array[int],
        row_K: wp.array[float],
        indices: _SparseMassMatrixIndices,
        Linv_group: wp.array2d[float],
        status: wp.array[int],
    ): ...

    def factor_kernel(
        group_to_art: wp.array[int],
        mass_update_mask: wp.array[int],
        articulation_start: wp.array[int],
        articulation_dof_start: wp.array[int],
        joint_child: wp.array[int],
        joint_S_s: wp.array[wp.spatial_vector],
        body_I_c: wp.array[wp.spatial_matrix],
        R_group: wp.array2d[float],
        fused_augmented_drive: int,
        drive_row_by_dof: wp.array[int],
        row_K: wp.array[float],
        indices: _SparseMassMatrixIndices,
        Linv_group: wp.array2d[float],
        status: wp.array[int],
    ):
        factor_native(
            wp.tid(),
            group_to_art,
            mass_update_mask,
            articulation_start,
            articulation_dof_start,
            joint_child,
            joint_S_s,
            body_I_c,
            R_group,
            fused_augmented_drive,
            drive_row_by_dof,
            row_K,
            indices,
            Linv_group,
            status,
        )

    factor_kernel.__name__ = f"crba_sparse_factor_{n}_{nnz}_{warps}"
    factor_kernel.__qualname__ = factor_kernel.__name__
    return wp.kernel(enable_backward=False, module="unique")(factor_kernel)


@wp.func_native("""
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
#if defined(__CUDA_ARCH__)
    const int lane = tid & 31;
    const int stride = 32;
#else
    if ((tid & 31) != 0) return;
    const int lane = 0;
    const int stride = 1;
#endif
    const int n = indices.dof_count;
    const int base = group * indices.nonzero_count;
    const int start = articulation_dof_start.data[group_to_art.data[group]];
    for (int row = lane; row < n; row += stride) {
        float value = 0.0f;
        for (int entry = indices.row_offsets.data[row]; entry < indices.row_offsets.data[row + 1]; ++entry)
            value += Linv_group.data[base + entry] * tau.data[start + indices.permutation.data[indices.columns.data[entry]]];
        scratch.data[group * n + row] = value;
    }
#if defined(__CUDA_ARCH__)
    __syncwarp();
#endif
    for (int col = lane; col < n; col += stride) {
        float value = 0.0f;
        for (int position = indices.column_offsets.data[col]; position < indices.column_offsets.data[col + 1]; ++position) {
            const int entry = indices.column_entries.data[position];
            const int row = indices.entry_rows.data[entry];
            value += Linv_group.data[base + entry] * scratch.data[group * n + row];
        }
        qdd.data[start + indices.permutation.data[col]] = value;
    }
""")
def _solve_native(
    tid: int,
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    indices: _SparseMassMatrixIndices,
    Linv_group: wp.array2d[float],
    tau: wp.array[float],
    scratch: wp.array2d[float],
    qdd: wp.array[float],
): ...


@wp.kernel(enable_backward=False)
def solve_sparse_mass_matrix(
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    indices: _SparseMassMatrixIndices,
    Linv_group: wp.array2d[float],
    tau: wp.array[float],
    articulation_active: wp.array[int],
    scratch: wp.array2d[float],
    qdd: wp.array[float],
):
    """Apply physical H inverse as P.T Linv.T Linv P; launch 32 threads/group."""
    # Each group is one warp, so skipping a sleeping articulation is warp-uniform.
    if articulation_active[group_to_art[wp.tid() // 32]] == 0:
        return
    _solve_native(wp.tid(), group_to_art, articulation_dof_start, indices, Linv_group, tau, scratch, qdd)
