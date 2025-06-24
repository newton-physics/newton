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


class PDMatrixBuilder:
    """Helper class for building PD Matrix"""

    def __init__(self, num_verts: int, max_neighbor: int = 32):
        self.num_verts = num_verts
        self.max_neighbors = max_neighbor
        self.counts = np.zeros(num_verts, dtype=np.int32)
        self.diags = np.zeros(num_verts, dtype=np.float32)
        self.values = np.zeros(shape=(num_verts, max_neighbor), dtype=np.float32)
        self.neighbors = np.zeros(shape=(num_verts, max_neighbor), dtype=np.int32)

    def add_connection(self, v0: int, v1: int) -> int:
        if v0 >= self.num_verts:
            raise ValueError(f"Vertex index{v0} out of range {self.num_verts}")
        if v1 >= self.num_verts:
            raise ValueError(f"Vertex index{v1} out of range {self.num_verts}")

        for slot in range(self.counts[v0]):
            if self.neighbors[v0, slot] == v1:
                return slot

        if self.counts[v0] >= self.max_neighbors:
            raise ValueError(f"Exceeds max neighbors limit {self.max_neighbors}")

        slot = self.counts[v0]
        self.neighbors[v0, slot] = v1
        self.counts[v0] += 1
        return slot

    def add_stretch_constraints(
        self,
        tri_indices: list[list[int]],
        tri_poses: list[list[list[float]]],
        tri_aniso_ke: list[list[float]],
        tri_areas: list[float],
    ):
        for fid in range(len(tri_indices)):
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
        edge_rest_angle: list[float],
        edge_rest_length: list[float],
        edge_bending_properties: list[list[float]],
        edge_rest_area: list[float],
        edge_bending_cot: list[list[float]],
    ):
        for eid in range(len(edge_indices)):
            if edge_indices[eid][0] < 0 or edge_indices[eid][1] < 0:
                continue
            # reorder as qbend order
            edge = wp.vec4i(edge_indices[eid][2], edge_indices[eid][3], edge_indices[eid][0], edge_indices[eid][1])
            bend_weight = wp.vec4(0.0)
            bend_weight[0] = edge_bending_cot[eid][2] + edge_bending_cot[eid][3]
            bend_weight[1] = edge_bending_cot[eid][0] + edge_bending_cot[eid][1]
            bend_weight[2] = -edge_bending_cot[eid][0] - edge_bending_cot[eid][2]
            bend_weight[3] = -edge_bending_cot[eid][1] - edge_bending_cot[eid][3]
            bend_weight = (edge_bending_properties[eid][0] / wp.sqrt(edge_rest_area[eid])) * bend_weight
            for i in range(4):
                for j in range(i, 4):
                    weight = bend_weight[i] * bend_weight[j]
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

    def finialize(self, device):
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
