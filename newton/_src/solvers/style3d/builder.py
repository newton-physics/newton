# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

from .linear_solver import NonZeroEntry

########################################################################################################################
###########################################    PD Matrix Builder Kernels    ############################################
########################################################################################################################


@wp.func
def add_connection(v0: int, v1: int, counts: wp.array[int], neighbors: wp.array2d[int], overflow: wp.array[int]):
    """
    Adds a connection from vertex v0 to vertex v1 in a sparse matrix format.
    If the connection already exists, returns its slot index.
    Otherwise, adds the connection in the next available slot and returns the new slot index.

    Args:
        v0: Index of the source vertex.
        v1: Index of the target vertex (to be added as neighbor of v0).
        counts: 1D array storing how many neighbors each vertex currently has.
        neighbors: 2D array storing the neighbor list for each vertex.
        overflow: One-element array recording an overflowing vertex index plus one.

    Returns:
        int: The slot index in `neighbors[v0]` where `v1` is stored,
             or -1 if there is no more space to store the new neighbor.
    """
    for slot in range(counts[v0]):
        if neighbors[v0, slot] == v1:
            return slot  # Connection already exists

    slot = counts[v0]
    if slot < neighbors.shape[1]:
        neighbors[v0, slot] = v1  # Add new neighbor
        counts[v0] += 1
        return slot
    else:
        wp.atomic_max(overflow, 0, v0 + 1)
        return -1


@wp.kernel
def add_bend_constraints_kernel(
    num_edge: int,
    edge_inds: wp.array2d[int],
    bend_hess: wp.array3d[float],
    # outputs
    neighbors: wp.array2d[int],
    neighbor_counts: wp.array[int],
    nz_values: wp.array2d[float],
    diags: wp.array[float],
    overflow: wp.array[int],
):
    """Accumulate contributions from bending constraints into a sparse matrix structure."""
    for eid in range(num_edge):
        edge = edge_inds[eid]
        if edge[0] < 0 or edge[1] < 0:
            continue  # Skip invalid edge

        tmp_bend_hess = bend_hess[eid]
        for i in range(4):
            for j in range(i, 4):
                weight = tmp_bend_hess[i][j]
                if i != j:
                    # Add off-diagonal symmetric entries
                    slot_ij = add_connection(edge[i], edge[j], neighbor_counts, neighbors, overflow)
                    slot_ji = add_connection(edge[j], edge[i], neighbor_counts, neighbors, overflow)
                    if slot_ij >= 0:
                        nz_values[edge[i], slot_ij] += weight
                    if slot_ji >= 0:
                        nz_values[edge[j], slot_ji] += weight
                else:
                    # Diagonal contribution
                    diags[edge[i]] += weight


@wp.kernel
def add_stretch_constraints_kernel(
    num_tri: int,
    tri_indices: wp.array2d[int],
    tri_areas: wp.array[float],
    tri_poses: wp.array3d[float],
    tri_aniso_ke: wp.array2d[float],
    # outputs
    neighbors: wp.array2d[int],
    neighbor_counts: wp.array[int],
    nz_values: wp.array2d[float],
    diags: wp.array[float],
    overflow: wp.array[int],
):
    """Accumulate contributions from stretch constraints into the sparse matrix."""
    for fid in range(num_tri):
        area = tri_areas[fid]
        inv_dm = tri_poses[fid]
        ku = tri_aniso_ke[fid][0]
        kv = tri_aniso_ke[fid][1]
        ks = tri_aniso_ke[fid][2]
        face = wp.vec3i(tri_indices[fid][0], tri_indices[fid][1], tri_indices[fid][2])

        # Derivatives of deformation gradient components
        dFu_dx = wp.vec3(-inv_dm[0][0] - inv_dm[1][0], inv_dm[0][0], inv_dm[1][0])
        dFv_dx = wp.vec3(-inv_dm[0][1] - inv_dm[1][1], inv_dm[0][1], inv_dm[1][1])

        for i in range(3):
            for j in range(i, 3):
                # Weight is a combination of anisotropic stiffness components
                weight = area * ((ku + ks) * dFu_dx[i] * dFu_dx[j] + (kv + ks) * dFv_dx[i] * dFv_dx[j])
                if i != j:
                    # Off-diagonal symmetric terms
                    slot_ij = add_connection(face[i], face[j], neighbor_counts, neighbors, overflow)
                    slot_ji = add_connection(face[j], face[i], neighbor_counts, neighbors, overflow)
                    if slot_ij >= 0:
                        nz_values[face[i], slot_ij] += weight
                    if slot_ji >= 0:
                        nz_values[face[j], slot_ji] += weight
                else:
                    # Diagonal contribution
                    diags[face[i]] += weight


@wp.kernel
def assemble_nz_ell_kernel(
    neighbors: wp.array2d[int],
    nz_values: wp.array2d[float],
    neighbor_counts: wp.array[int],
    # outputs
    nz_ell: wp.array2d[NonZeroEntry],
):
    tid = wp.tid()
    for k in range(neighbor_counts[tid]):
        nz_entry = NonZeroEntry()
        nz_entry.value = nz_values[tid, k]
        nz_entry.column_index = neighbors[tid, k]
        nz_ell[k, tid] = nz_entry


########################################################################################################################
###############################################    PD Matrix Builder    ################################################
########################################################################################################################


def _compute_max_neighbors(num_verts, tri_indices, edge_indices, tet_indices=None):
    """Count the unique target slots inserted by surface and tetrahedral stencils."""
    pairs = []
    stencils_and_widths = [(tri_indices, 3), (edge_indices, 4)]
    if tet_indices is not None:
        stencils_and_widths.append((tet_indices, 4))
    for indices, width in stencils_and_widths:
        stencils = np.asarray(indices, dtype=np.int64).reshape(-1, width)
        if indices is edge_indices:
            stencils = stencils[(stencils[:, 0] >= 0) & (stencils[:, 1] >= 0)]
        if len(stencils) == 0:
            continue
        # Kernels exclude equal stencil positions, not equal vertex IDs.
        rows, cols = np.nonzero(~np.eye(width, dtype=bool))
        pairs.append((stencils[:, rows] * num_verts + stencils[:, cols]).ravel())
    if not pairs:
        return 0
    connections = np.unique(np.concatenate(pairs))
    return int(np.bincount(connections // num_verts).max(initial=0))


class PDMatrixBuilder:
    """Helper class for building Projective Dynamics (PD) matrix in sparse ELL format."""

    def __init__(self, num_verts: int, max_neighbor: int = 32):
        self.num_verts = num_verts
        self.max_neighbors = max_neighbor
        self.counts = wp.zeros(num_verts, dtype=wp.int32, device="cpu")
        self.diags = wp.zeros(num_verts, dtype=wp.float32, device="cpu")
        self.values = wp.zeros(shape=(num_verts, max_neighbor), dtype=wp.float32, device="cpu")
        self.neighbors = wp.zeros(shape=(num_verts, max_neighbor), dtype=wp.int32, device="cpu")
        self._overflow = wp.zeros(1, dtype=wp.int32, device="cpu")

    def _check_overflow(self):
        """Reject incomplete matrices instead of silently dropping connections."""
        vertex = int(self._overflow.numpy()[0]) - 1
        if vertex >= 0:
            raise ValueError(f"PD matrix capacity {self.max_neighbors} exceeded at vertex {vertex}.")

    def add_stretch_constraints(
        self,
        tri_indices: list[list[int]],
        tri_poses: list[list[list[float]]],
        tri_aniso_ke: list[list[float]],
        tri_areas: list[float],
    ):
        if len(tri_indices) == 0:
            return

        # Convert inputs to Warp arrays
        tri_inds_wp = wp.array2d(tri_indices, dtype=int, device="cpu").reshape((-1, 3))
        tri_poses_wp = wp.array3d(tri_poses, dtype=float, device="cpu").reshape((-1, 2, 2))
        tri_aniso_ke_wp = wp.array2d(tri_aniso_ke, dtype=float, device="cpu").reshape((-1, 3))
        tri_areas_wp = wp.array(tri_areas, dtype=float, device="cpu")

        # Launch kernel to compute stretch contributions
        wp.launch(
            add_stretch_constraints_kernel,
            dim=1,
            inputs=[
                len(tri_indices),
                tri_inds_wp,
                tri_areas_wp,
                tri_poses_wp,
                tri_aniso_ke_wp,
            ],
            outputs=[self.neighbors, self.counts, self.values, self.diags, self._overflow],
            device="cpu",
        )
        self._check_overflow()

    def add_bend_constraints(
        self,
        edge_indices: list[list[int]],
        edge_bending_properties: list[list[float]],
        edge_rest_area: list[float],
        edge_bending_cot: list[list[float]],
    ):
        if len(edge_indices) == 0:
            return

        num_edge = len(edge_indices)
        edge_inds = np.array(edge_indices).reshape(-1, 4)
        edge_area = np.array(edge_rest_area)
        edge_prop = np.array(edge_bending_properties).reshape(-1, 2)
        edge_stiff = np.zeros(num_edge, dtype=np.float32)
        active = edge_prop[:, 0] != 0.0
        edge_stiff[active] = edge_prop[active, 0] * (3.0 / edge_area[active])

        bend_cot = np.array(edge_bending_cot).reshape(-1, 4)
        bend_weight = np.zeros(shape=(num_edge, 4), dtype=np.float32)

        # Compute per-vertex weights using cotangent terms
        bend_weight[:, 2] = bend_cot[:, 2] + bend_cot[:, 3]
        bend_weight[:, 3] = bend_cot[:, 0] + bend_cot[:, 1]
        bend_weight[:, 0] = -bend_cot[:, 0] - bend_cot[:, 2]
        bend_weight[:, 1] = -bend_cot[:, 1] - bend_cot[:, 3]

        # Construct Hessian matrix per edge (outer product)
        # Hessian = k * (3 / area) * w^T w
        bend_hess = (
            bend_weight[:, :, np.newaxis] * bend_weight[:, np.newaxis, :] * edge_stiff[:, np.newaxis, np.newaxis]
        )  # shape is num_edge,4,4

        # Convert to Warp arrays
        edge_inds_wp = wp.array2d(edge_inds, dtype=int, device="cpu")
        bend_hess_wp = wp.array3d(bend_hess, dtype=float, device="cpu")

        # Launch kernel to accumulate bend constraints
        wp.launch(
            add_bend_constraints_kernel,
            dim=1,
            inputs=[num_edge, edge_inds_wp, bend_hess_wp],
            outputs=[self.neighbors, self.counts, self.values, self.diags, self._overflow],
            device="cpu",
        )
        self._check_overflow()

    def add_tet_constraints(self, tet_indices, tet_poses, tet_materials, *, volume_ratio: float = 1.0):
        """Add the scalar operator of the projective tetrahedral energy.

        Each spatial block is ``K_ij * I`` with
        ``K_ij = V * (2 * mu + 3 * lambda * volume_ratio) * grad(N_i).grad(N_j)``.
        Consequently the existing float ELL matrix stores the complete operator.
        The volume target has determinant one. At ``volume_ratio=1`` its
        converged projection matches the Lamé volume response near rest.
        """
        if not np.isfinite(volume_ratio) or volume_ratio < 0.0:
            raise ValueError("Tetrahedral PD volume ratio must be finite and nonnegative.")

        indices = np.asarray(tet_indices, dtype=np.int32).reshape(-1, 4)
        poses = np.asarray(tet_poses, dtype=np.float32).reshape(-1, 3, 3)
        materials = np.asarray(tet_materials, dtype=np.float32).reshape(-1, 3)
        if not (len(indices) == len(poses) == len(materials)):
            raise ValueError("Tetrahedral indices, poses, and materials must have the same length.")
        if len(indices) == 0:
            return

        determinants = np.linalg.det(poses)
        if np.any(~np.isfinite(determinants)) or np.any(determinants <= 0.0):
            raise ValueError("Tetrahedral PD requires finite, positively oriented rest tetrahedra.")
        if np.any(~np.isfinite(materials)) or np.any(materials < 0.0):
            raise ValueError("Tetrahedral PD material coefficients must be finite and nonnegative.")
        if np.any(indices < 0) or np.any(indices >= self.num_verts):
            raise ValueError("Tetrahedral PD indices must reference existing particles.")

        counts = self.counts.numpy()
        neighbors = self.neighbors.numpy()
        values = self.values.numpy()
        diags = self.diags.numpy()
        gradients = np.concatenate((-poses.sum(axis=1, keepdims=True), poses), axis=1)
        volume = 1.0 / (6.0 * determinants)
        weight = 2.0 * materials[:, 0] + 3.0 * materials[:, 1] * volume_ratio
        stiffness = (volume * weight)[:, None, None] * (gradients @ gradients.transpose(0, 2, 1))
        np.add.at(diags, indices.ravel(), np.diagonal(stiffness, axis1=1, axis2=2).ravel())

        # Merge surface and tet entries together so shared edges occupy one
        # slot. Stable sorting preserves the order of tet contributions.
        local_i, local_j = np.nonzero(~np.eye(4, dtype=bool))
        old_rows, old_slots = np.nonzero(np.arange(self.max_neighbors)[None, :] < counts[:, None])
        rows = np.concatenate((old_rows, indices[:, local_i].ravel())).astype(np.int64)
        cols = np.concatenate((neighbors[old_rows, old_slots], indices[:, local_j].ravel())).astype(np.int64)
        entries = np.concatenate((values[old_rows, old_slots], stiffness[:, local_i, local_j].ravel()))
        keys = rows * self.num_verts + cols
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        unique_keys = keys[starts]
        rows = unique_keys // self.num_verts
        cols = unique_keys % self.num_verts
        entries = np.add.reduceat(entries[order], starts)
        counts = np.bincount(rows, minlength=self.num_verts).astype(np.int32)
        overflow = np.flatnonzero(counts > self.max_neighbors)
        if len(overflow):
            raise ValueError(f"PD matrix capacity {self.max_neighbors} exceeded at vertex {overflow[0]}.")
        row_starts = np.cumsum(counts, dtype=np.int64) - counts
        slots = np.arange(len(rows)) - row_starts[rows]
        neighbors.fill(0)
        values.fill(0.0)
        neighbors[rows, slots] = cols
        values[rows, slots] = entries

        self.counts.assign(counts)
        self.neighbors.assign(neighbors)
        self.values.assign(values)
        self.diags.assign(diags)

    def finalize(self, device):
        """Assembles final sparse matrix in ELL format.

        Returns:
                diag: wp.array of diagonal entries
                num_nz: wp.array of non-zero count per row
                nz_ell: wp.array2d of NonZeroEntry (value + column)
        """
        self._check_overflow()
        diag = wp.array(self.diags, dtype=float, device=device)
        num_nz = wp.array(self.counts, dtype=int, device=device)
        nz_ell = wp.array2d(shape=(self.max_neighbors, self.num_verts), dtype=NonZeroEntry, device=device)

        nz_values = wp.array2d(self.values, dtype=float, device=device)
        neighbors = wp.array2d(self.neighbors, dtype=int, device=device)

        wp.launch(
            assemble_nz_ell_kernel,
            dim=self.num_verts,
            inputs=[neighbors, nz_values, num_nz],
            outputs=[nz_ell],
            device=device,
        )
        return diag, num_nz, nz_ell
