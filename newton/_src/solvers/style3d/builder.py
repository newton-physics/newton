# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warp as wp

from .linear_solver import NonZeroEntry

########################################################################################################################
###########################################    PD Matrix Builder Kernels    ############################################
########################################################################################################################


@wp.func
def add_connection(v0: int, v1: int, counts: wp.array(dtype=int), neighbors: wp.array2d(dtype=int)):
    """
    Adds a connection from vertex v0 to vertex v1 in a sparse matrix format.
    If the connection already exists, returns its slot index.
    Otherwise, adds the connection in the next available slot and returns the new slot index.

    Args:
        v0 (int): Index of the source vertex.
        v1 (int): Index of the target vertex (to be added as neighbor of v0).
        counts (wp.array): 1D array storing how many neighbors each vertex currently has.
        neighbors (wp.array2d): 2D array storing the neighbor list for each vertex.

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
        wp.printf("Error: Too many neighbors for vertex %d (max %d)\n", v0, neighbors.shape[1])
        return -1


@wp.kernel
def add_bend_constraints_kernel(
    num_edge: int,
    edge_inds: wp.array2d(dtype=int),
    bend_hess: wp.array3d(dtype=float),
    # outputs
    neighbors: wp.array2d(dtype=int),
    neighbor_counts: wp.array(dtype=int),
    nz_values: wp.array2d(dtype=float),
    diags: wp.array(dtype=float),
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
                    slot_ij = add_connection(edge[i], edge[j], neighbor_counts, neighbors)
                    slot_ji = add_connection(edge[j], edge[i], neighbor_counts, neighbors)
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
    tri_indices: wp.array2d(dtype=int),
    tri_areas: wp.array(dtype=float),
    tri_poses: wp.array3d(dtype=float),
    tri_aniso_ke: wp.array2d(dtype=float),
    # outputs
    neighbors: wp.array2d(dtype=int),
    neighbor_counts: wp.array(dtype=int),
    nz_values: wp.array2d(dtype=float),
    diags: wp.array(dtype=float),
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
                    slot_ij = add_connection(face[i], face[j], neighbor_counts, neighbors)
                    slot_ji = add_connection(face[j], face[i], neighbor_counts, neighbors)
                    if slot_ij >= 0:
                        nz_values[face[i], slot_ij] += weight
                    if slot_ji >= 0:
                        nz_values[face[j], slot_ji] += weight
                else:
                    # Diagonal contribution
                    diags[face[i]] += weight


@wp.kernel
def assemble_nz_ell_kernel(
    neighbors: wp.array2d(dtype=int),
    nz_values: wp.array2d(dtype=float),
    neighbor_counts: wp.array(dtype=int),
    # outputs
    nz_ell: wp.array2d(dtype=NonZeroEntry),
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


class PDMatrixBuilder:
    """Helper class for building Projective Dynamics (PD) matrix in sparse ELL format."""

    def __init__(self, num_verts: int, max_neighbor: int = 32):
        self.num_verts = num_verts
        self.max_neighbors = max_neighbor
        self.counts = wp.zeros(num_verts, dtype=wp.int32, device="cpu")
        self.diags = wp.zeros(num_verts, dtype=wp.float32, device="cpu")
        self.values = wp.zeros(shape=(num_verts, max_neighbor), dtype=wp.float32, device="cpu")
        self.neighbors = wp.zeros(shape=(num_verts, max_neighbor), dtype=wp.int32, device="cpu")

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
            outputs=[self.neighbors, self.counts, self.values, self.diags],
            device="cpu",
        )

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
        edge_stiff = edge_prop[:, 0] / edge_area

        bend_cot = np.array(edge_bending_cot).reshape(-1, 4)
        bend_weight = np.zeros(shape=(num_edge, 4), dtype=np.float32)

        # Compute per-vertex weights using cotangent terms
        bend_weight[:, 2] = bend_cot[:, 2] + bend_cot[:, 3]
        bend_weight[:, 3] = bend_cot[:, 0] + bend_cot[:, 1]
        bend_weight[:, 0] = -bend_cot[:, 0] - bend_cot[:, 2]
        bend_weight[:, 1] = -bend_cot[:, 1] - bend_cot[:, 3]

        # Construct Hessian matrix per edge (outer product)
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
            outputs=[self.neighbors, self.counts, self.values, self.diags],
            device="cpu",
        )

    def finalize(self, device):
        """Assembles final sparse matrix in ELL format.

        Returns:
                diag: wp.array of diagonal entries
                num_nz: wp.array of non-zero count per row
                nz_ell: wp.array2d of NonZeroEntry (value + column)
        """
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
