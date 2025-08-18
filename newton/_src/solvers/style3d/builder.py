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
###############################################    PD Matrix Builder    ################################################
########################################################################################################################


@wp.func
def add_connection_wp(v0: int, v1: int, counts: wp.array(dtype=int), neighbors: wp.array2d(dtype=int)):
    for slot in range(counts[v0]):
        if neighbors[v0, slot] == v1:
            return slot
    slot = counts[v0]
    neighbors[v0, slot] = v1
    counts[v0] += 1
    return slot


@wp.kernel
def add_bend_constraints_wp(
    num_edge: int,
    edge_inds: wp.array2d(dtype=int),
    bend_hess: wp.array3d(dtype=float),
    neighbors: wp.array2d(dtype=int),
    neighbor_counts: wp.array(dtype=int),
    # outputs
    nz_values: wp.array2d(dtype=float),
    diags: wp.array(dtype=float),
):
    for eid in range(num_edge):
        edge = edge_inds[eid]
        if edge[0] < 0 or edge[1] < 0:
            continue
        tmp_bend_hess = bend_hess[eid]
        for i in range(4):
            for j in range(i, 4):
                weight = tmp_bend_hess[i][j]
                if i != j:
                    slot_ij = add_connection_wp(edge[i], edge[j], neighbor_counts, neighbors)
                    slot_ji = add_connection_wp(edge[j], edge[i], neighbor_counts, neighbors)
                    nz_values[edge[i], slot_ij] += weight
                    nz_values[edge[j], slot_ji] += weight
                else:
                    diags[edge[i]] += weight


@wp.kernel
def add_stretch_constraints_wp(
    num_tri: int,
    tri_indices: wp.array2d(dtype=int),
    tri_areas: wp.array(dtype=float),
    tri_poses: wp.array3d(dtype=float),
    tri_aniso_ke: wp.array2d(dtype=float),
    neighbors: wp.array2d(dtype=int),
    neighbor_counts: wp.array(dtype=int),
    # outputs
    nz_values: wp.array2d(dtype=float),
    diags: wp.array(dtype=float),
):
    for fid in range(num_tri):
        area = tri_areas[fid]
        inv_dm = tri_poses[fid]
        ku = tri_aniso_ke[fid][0]
        kv = tri_aniso_ke[fid][1]
        ks = tri_aniso_ke[fid][2]
        face = wp.vec3i(tri_indices[fid][0], tri_indices[fid][1], tri_indices[fid][2])
        dFu_dx = wp.vec3(-inv_dm[0][0] - inv_dm[1][0], inv_dm[0][0], inv_dm[1][0])
        dFv_dx = wp.vec3(-inv_dm[0][1] - inv_dm[1][1], inv_dm[0][1], inv_dm[1][1])
        for i in range(3):
            for j in range(i, 3):
                weight = area * ((ku + ks) * dFu_dx[i] * dFu_dx[j] + (kv + ks) * dFv_dx[i] * dFv_dx[j])
                if i != j:
                    slot_ij = add_connection_wp(face[i], face[j], neighbor_counts, neighbors)
                    slot_ji = add_connection_wp(face[j], face[i], neighbor_counts, neighbors)
                    nz_values[face[i], slot_ij] += weight
                    nz_values[face[j], slot_ji] += weight
                else:
                    diags[face[i]] += weight


class PDMatrixBuilder:
    """Helper class for building PD Matrix"""

    def __init__(self, num_verts: int, max_neighbor: int = 32):
        self.num_verts = num_verts
        self.max_neighbors = max_neighbor
        self.counts = np.zeros(num_verts, dtype=np.int32)
        self.diags = np.zeros(num_verts, dtype=np.float32)
        self.values = np.zeros(shape=(num_verts, max_neighbor), dtype=np.float32)
        self.neighbors = np.zeros(shape=(num_verts, max_neighbor), dtype=np.int32)
        self.slot_map = {}

    def add_connection(self, v0: int, v1: int) -> int:
        if v0 >= self.num_verts:
            raise ValueError(f"Vertex index{v0} out of range {self.num_verts}")
        if v1 >= self.num_verts:
            raise ValueError(f"Vertex index{v1} out of range {self.num_verts}")

        key = (v0, v1)
        if key in self.slot_map:
            return self.slot_map[key]

        # for slot in range(self.counts[v0]):
        #     if self.neighbors[v0, slot] == v1:
        #         return slot

        if self.counts[v0] >= self.max_neighbors:
            raise ValueError(f"Exceeds max neighbors limit {self.max_neighbors}")

        slot = self.counts[v0]
        self.neighbors[v0, slot] = v1
        self.counts[v0] += 1
        self.slot_map[key] = slot
        return slot

    def add_stretch_constraints(
        self,
        tri_indices: list[list[int]],
        tri_poses: list[list[list[float]]],
        tri_aniso_ke: list[list[float]],
        tri_areas: list[float],
    ):
        num_tri = len(tri_indices)
        use_kernel = True
        if use_kernel:
            tri_inds_wp = wp.array2d(tri_indices, dtype=int, device="cpu").reshape((-1, 3))
            tri_poses_wp = wp.array3d(tri_poses, dtype=float, device="cpu").reshape((-1, 2, 2))
            tri_aniso_ke_wp = wp.array2d(tri_aniso_ke, dtype=float, device="cpu").reshape((-1, 3))
            tri_areas_wp = wp.array(tri_areas, dtype=float, device="cpu")

            neighbors_wp = wp.array2d(self.neighbors, dtype=int, device="cpu")
            neighbor_counts_wp = wp.array(self.counts, dtype=int, device="cpu")
            nz_values_wp = wp.array2d(self.values, dtype=float, device="cpu")
            diags_wp = wp.array(self.diags, dtype=float, device="cpu")

            wp.launch(
                add_stretch_constraints_wp,
                dim=1,
                inputs=[
                    num_tri,
                    tri_inds_wp,
                    tri_areas_wp,
                    tri_poses_wp,
                    tri_aniso_ke_wp,
                    neighbors_wp,
                    neighbor_counts_wp,
                ],
                outputs=[nz_values_wp, diags_wp],
                device="cpu",
            )

            self.neighbors = neighbors_wp.numpy()
            self.counts = neighbor_counts_wp.numpy()
            self.values = nz_values_wp.numpy()
            self.diags = diags_wp.numpy()

        else:
            for fid in range(num_tri):
                area = tri_areas[fid]
                inv_dm = tri_poses[fid]
                ku, kv, ks = tri_aniso_ke[fid]
                face = wp.vec3i(tri_indices[fid])
                dFu_dx = wp.vec3(-inv_dm[0][0] - inv_dm[1][0], inv_dm[0][0], inv_dm[1][0])
                dFv_dx = wp.vec3(-inv_dm[0][1] - inv_dm[1][1], inv_dm[0][1], inv_dm[1][1])
                for i in range(3):
                    for j in range(i, 3):
                        weight = area * ((ku + ks) * dFu_dx[i] * dFu_dx[j] + (kv + ks) * dFv_dx[i] * dFv_dx[j])
                        if i != j:
                            slot_ij = self.add_connection(face[i], face[j])
                            slot_ji = self.add_connection(face[j], face[i])
                            self.values[face[i], slot_ij] += weight
                            self.values[face[j], slot_ji] += weight
                        else:
                            self.diags[face[i]] += weight

    def add_bend_constraints(
        self,
        edge_indices: list[list[int]],
        edge_bending_properties: list[list[float]],
        edge_rest_area: list[float],
        edge_bending_cot: list[list[float]],
    ):
        num_edge = len(edge_indices)
        edge_inds = np.array(edge_indices).reshape(-1, 4)
        edge_area = np.array(edge_rest_area)
        edge_prop = np.array(edge_bending_properties).reshape(-1, 2)
        edge_stiff = edge_prop[:, 0] / edge_area

        bend_cot = np.array(edge_bending_cot).reshape(-1, 4)
        bend_weight = np.zeros(shape=(num_edge, 4), dtype=np.float32)
        bend_weight[:, 2] = bend_cot[:, 2] + bend_cot[:, 3]
        bend_weight[:, 3] = bend_cot[:, 0] + bend_cot[:, 1]
        bend_weight[:, 0] = -bend_cot[:, 0] - bend_cot[:, 2]
        bend_weight[:, 1] = -bend_cot[:, 1] - bend_cot[:, 3]
        bend_hess = (
            bend_weight[:, :, np.newaxis] * bend_weight[:, np.newaxis, :] * edge_stiff[:, np.newaxis, np.newaxis]
        )  # shape is num_edge,4,4

        use_kernel = True
        if use_kernel:
            edge_inds_wp = wp.array2d(edge_inds, dtype=int, device="cpu")
            bend_hess_wp = wp.array3d(bend_hess, dtype=float, device="cpu")
            neighbors_wp = wp.array2d(self.neighbors, dtype=int, device="cpu")
            neighbor_counts_wp = wp.array(self.counts, dtype=int, device="cpu")

            nz_values_wp = wp.array2d(self.values, dtype=float, device="cpu")
            diags_wp = wp.array(self.diags, dtype=float, device="cpu")

            wp.launch(
                add_bend_constraints_wp,
                dim=1,
                inputs=[num_edge, edge_inds_wp, bend_hess_wp, neighbors_wp, neighbor_counts_wp],
                outputs=[nz_values_wp, diags_wp],
                device="cpu",
            )
            self.neighbors = neighbors_wp.numpy()
            self.counts = neighbor_counts_wp.numpy()
            self.values = nz_values_wp.numpy()
            self.diags = diags_wp.numpy()
        else:
            for eid in range(num_edge):
                edge = edge_inds[eid]
                if edge[0] < 0 or edge[1] < 0:
                    continue
                tmp_bend_hess = bend_hess[eid]
                for i in range(4):
                    for j in range(i, 4):
                        weight = tmp_bend_hess[i][j]
                        if i != j:
                            slot_ij = self.add_connection(edge[i], edge[j])
                            slot_ji = self.add_connection(edge[j], edge[i])
                            self.values[edge[i], slot_ij] += weight
                            self.values[edge[j], slot_ji] += weight
                        else:
                            self.diags[edge[i]] += weight

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

    def finalize(self, device):
        diag = wp.array(self.diags, dtype=float, device=device)
        num_nz = wp.array(self.counts, dtype=int, device=device)
        nz_ell = wp.array2d(shape=(32, self.num_verts), dtype=NonZeroEntry, device=device)

        nz_values = wp.array2d(self.values, dtype=float, device=device)
        neighbors = wp.array2d(self.neighbors, dtype=int, device=device)

        wp.launch(
            self.assemble_nz_ell_kernel,
            dim=self.num_verts,
            inputs=[neighbors, nz_values, num_nz],
            outputs=[nz_ell],
            device=device,
        )
        return diag, num_nz, nz_ell
