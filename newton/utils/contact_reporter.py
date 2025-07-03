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

import itertools
from typing import Any

import numpy as np
import warp as wp

from newton import Model
from newton.sim.contacts import ContactInfo
from newton.solvers import MuJoCoSolver, SolverBase

NUM_THREADS = 8192


def _normalize_pair(pair: tuple[int, int]) -> tuple[bool, tuple[int, int]]:
    """Normalize a pair by sorting the values, and return if the pair was flipped along with the new pair"""
    return (pair, False) if pair[0] <= pair[1] else ((pair[1], pair[0]), True)


def _sort_into(directory: dict[tuple], iterable, normalize_fn=None, filter_fn=None):
    """Given a directory and an iterable, for each element:
    - normalize it with `normalize_fn`
    - test if it passes `filter_fn`
    - find it in the directory, or create a new entry in the directory
    - return it along with extra output from the normalize function
    """
    n = len(directory)
    for element in iterable:
        norm_info = []
        if normalize_fn is not None:
            element, *norm_info = normalize_fn(element)  # noqa: PLW2901
        if filter_fn is not None and not (filter_fn(element)):
            continue
        pair_idx = directory.get(element, None)
        if pair_idx is None:
            pair_idx = n
            directory[element] = n
            n += 1
        yield pair_idx, *norm_info


@wp.func
def bisect_shape_pairs(
    # inputs
    shape_pairs_sorted: wp.array(dtype=wp.vec2i),
    n_shape_pairs: wp.int32,
    value: wp.vec2i,
):
    lo = wp.int32(0)
    hi = n_shape_pairs
    while lo < hi:
        mid = (lo + hi) // 2
        pair_mid = shape_pairs_sorted[mid]
        if pair_mid[0] < value[0] or (pair_mid[0] == value[0] and pair_mid[1] < value[1]):
            lo = mid + 1
        else:
            hi = mid
    return lo


# Binning function determines the existence and index of appropriate bin
@wp.func
def bin_contact_shape_pair(
    contact_geom: wp.array(dtype=wp.vec2i),
    num_shape_pairs: wp.int32,
    shape_pairs_sorted: wp.array(dtype=wp.vec2i),
    contact_idx: wp.int32,
) -> tuple[bool, wp.int32]:
    """Find the bin index, if it exists, based on the contact's shape_pair."""
    geom = contact_geom[contact_idx]

    if geom[0] == -1 or geom[1] == -1:
        return False, 0

    ga = wp.min(geom[0], geom[1])
    gb = wp.max(geom[0], geom[1])

    normalized_pair = wp.vec2i(ga, gb)
    pair_ord = bisect_shape_pairs(shape_pairs_sorted, num_shape_pairs, normalized_pair)
    if pair_ord < num_shape_pairs and shape_pairs_sorted[pair_ord] == normalized_pair:
        return True, pair_ord
    return False, 0


@wp.kernel
def select_bin_contacts(
    # inputs
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_dist: wp.array(dtype=wp.float32),
    shape_pairs: wp.array(dtype=wp.vec2i),
    num_shape_pairs: wp.int32,
    num_contacts: wp.array(dtype=wp.int32),
    bin_start: wp.array(dtype=wp.int32),
    # outputs
    bin_contacts: wp.array(dtype=wp.int32),
    bin_count: wp.array(dtype=wp.int32),
    bin_contacts_dist: wp.array(dtype=wp.float32),
):
    n_contacts = num_contacts[0]
    n_blocks = (n_contacts + NUM_THREADS - 1) // NUM_THREADS
    for b in range(n_blocks):
        contact_idx = wp.tid() + NUM_THREADS * b
        if contact_idx >= n_contacts:
            return

        # TODO: implement binning based on entity pair
        keep, bin_idx = bin_contact_shape_pair(contact_geom, num_shape_pairs, shape_pairs, contact_idx)

        if not keep:
            continue

        i = wp.atomic_add(bin_count, bin_idx, 1)
        contact_start = bin_start[bin_idx]

        bin_contacts[contact_start + i] = contact_idx
        bin_contacts_dist[contact_start + i] = contact_dist[contact_idx]


@wp.kernel
def aggregate_entity_pair_maxdepth(
    # inputs
    contact_normal: wp.array(dtype=wp.vec3f),
    n_entity_pairs: wp.int32,
    ep_sp_ord: wp.array(dtype=wp.int32),
    ep_sp_start: wp.array(dtype=wp.int32),
    ep_sp_num: wp.array(dtype=wp.int32),
    ep_sp_flip: wp.array(dtype=wp.int32),
    bin_start: wp.array(dtype=wp.int32),
    bin_contacts: wp.array(dtype=wp.int32),
    bin_dist: wp.array(dtype=wp.float32),
    bin_count: wp.array(dtype=wp.int32),
    # outputs
    entity_pair_contact: wp.array(dtype=wp.int32),
    entity_pair_dist: wp.array(dtype=wp.float32),
    entity_pair_normal: wp.array(dtype=wp.vec3),
):
    entity_pair_idx = wp.tid()
    if entity_pair_idx >= n_entity_pairs:
        return

    start = ep_sp_start[entity_pair_idx]
    n_shape_pairs = ep_sp_num[entity_pair_idx]

    # track deepest contact
    best_dist = wp.float32(wp.inf)
    best_contact = wp.int32(-1)
    best_flip = wp.int32(0)

    for i in range(n_shape_pairs):
        sp_ord = ep_sp_ord[start + i]
        n_bin_contacts = bin_count[sp_ord]
        contacts_start = bin_start[sp_ord]

        for j in range(n_bin_contacts):
            contact_idx = bin_contacts[contacts_start + j]
            dist = bin_dist[contacts_start + j]

            if dist < best_dist:
                best_dist = dist
                best_contact = contact_idx
                best_flip = ep_sp_flip[start + i]

    entity_pair_normal[entity_pair_idx] = wp.where(best_flip, -1.0, 1.0) * contact_normal[best_contact]
    entity_pair_contact[entity_pair_idx] = best_contact
    entity_pair_dist[entity_pair_idx] = best_dist


@wp.kernel
def aggregate_entity_pair_net_force(
    # inputs
    contact_normal: wp.array(dtype=wp.vec3f),
    n_entity_pairs: wp.int32,
    ep_sp_ord: wp.array(dtype=wp.int32),
    ep_sp_start: wp.array(dtype=wp.int32),
    ep_sp_num: wp.array(dtype=wp.int32),
    ep_sp_flip: wp.array(dtype=wp.int32),
    bin_start: wp.array(dtype=wp.int32),
    bin_contacts: wp.array(dtype=wp.int32),
    contact_force: wp.array(dtype=wp.float32),
    bin_count: wp.array(dtype=wp.int32),
    # outputs
    entity_pair_net_force: wp.array(dtype=wp.vec3),
):
    entity_pair_idx = wp.tid()
    if entity_pair_idx >= n_entity_pairs:
        return

    start = ep_sp_start[entity_pair_idx]
    n_shape_pairs = ep_sp_num[entity_pair_idx]

    # track deepest contact
    net_force = wp.vec3(0.0)

    for i in range(n_shape_pairs):
        sp_ord = ep_sp_ord[start + i]
        n_bin_contacts = bin_count[sp_ord]
        contacts_start = bin_start[sp_ord]

        flip = ep_sp_flip[start + i]
        for j in range(n_bin_contacts):
            contact_idx = bin_contacts[contacts_start + j]
            force = contact_force[contact_idx]
            net_force += wp.where(flip, -1.0, 1.0) * contact_normal[contact_idx] * force

    entity_pair_net_force[entity_pair_idx] = net_force


@wp.kernel
def fill_contact_matrix(
    # inputs
    q_ep: wp.array(dtype=wp.int32),
    q_ep_mat_idx: wp.array(dtype=wp.vec2i),
    data: wp.array(dtype=Any),
    query_flip: wp.array(dtype=wp.int32),
    flip: bool,
    matrix: wp.array2d(dtype=Any),
):
    """Fill a contact matrix with data for a query.
    Args:
        q_ep: Entity pair indices for the query.
        q_ep_mat_idx: Matrix indices for the query, shape [n_entity_pairs].
        data: Data to fill the matrix with, shape [n_entity_pairs].
        query_flip: flatterned array with flags whether the enditites were flipped.
        flip: for force and normal reporting do the flip if entities were flipped
        matrix: Matrix to fill from data.
    """
    qep_idx = wp.tid()
    if qep_idx >= q_ep.shape[0]:
        return
    ep_idx = q_ep[qep_idx]

    mat_idx = q_ep_mat_idx[qep_idx]
    row, col = mat_idx.x, mat_idx.y
    if flip and query_flip[qep_idx] == 1:
        matrix[row, col] = -data[ep_idx]
    else:
        matrix[row, col] = data[ep_idx]


class ContactReporter:
    """Filter and aggregate contacts by the pair of entities between which they occur.
    An entity is a set of shapes.
    Currently, a shape pair may not appear in multiple entity pairs.

    Initialized with a list of entities and a list of entity pairs.
    """

    # class is concerned only with shapes, collections of shapes, and their contacts.

    def __init__(self, model: Model):
        self.model = model

        # results for deepest contact query
        self.entity_pair_contact = None
        """Index of deepest contact for each entity pair, shape [n_entity_pairs], int"""
        self.entity_pair_dist = None
        """Distance of deepest contact for each entity pair, shape [n_entity_pairs], float"""
        self.entity_pair_normal = None
        """Normal of deepest contact for each entity pair, shape [n_entity_pairs], vec3"""

        # results for net contact force query
        self.entity_pair_force = None
        """Net contact force between each entity pair, shape [n_entity_pairs], int"""

        # intermediate, static data
        self.n_entity_pairs = 0
        """Number of entity pairs, int"""
        self.entity_pairs = None
        """Pairs of entities whose contacts are of interest, shape [n_entity_pairs], vec2i"""
        self.n_shape_pairs = 0
        """Number of shape pairs, int"""
        self.shape_pairs = None  # sorted lexicographically
        """Pairs of shapes whose contacts are of interest, shape [n_entity_pairs], vec2i"""
        self.bin_start = None
        """Start index of contacts for each bin, shape [n_shape_pairs], int"""
        self.bin_start_sorted = None
        """Start index of contacts for each bin, sorted by shape pair ordinal, shape [n_shape_pairs], int"""

        # entity and shape pair mapping
        self.entities = None
        """Dictionary mapping entity tuples to entity indices"""
        self.query_to_entity_pair = None
        """List mapping query indices to entity pair data"""
        self.entity_p_shape_p_map = None
        """Dictionary mapping entity pairs to their shape pairs"""

        # entity pair to shape pair mapping arrays
        self.ep_sp_start = None
        """Start index of shape pairs for each entity pair, shape [n_entity_pairs], int"""
        self.ep_sp_num = None
        """Number of shape pairs for each entity pair, shape [n_entity_pairs], int"""
        self.ep_sp_ord = None
        """Shape pair ordinals for entity pairs, shape [total_shape_pairs], int"""
        self.ep_sp_flip = None
        """Shape pair flip flags for entity pairs, shape [total_shape_pairs], int"""

        # intermediate, variable data

        self.bin_contacts = None
        """Contact index component of bins (shape pair), shape [n_shape_pairs, n_pair_contact_max], int"""

        self.bin_count = None
        """Number of contacts in each bin, shape [n_shape_pairs], int"""

        # inputs to aggregation
        self.bin_contacts_dist = None
        """Contact distance component of bins, shape [n_shape_pairs, n_pair_contact_max], float"""

        # query result matrices
        self.query_dist_matrices = None
        """List of distance matrices for each query"""
        self.query_idx_matrices = None
        """List of contact index matrices for each query"""
        self.query_entities = None
        """List of entity mappings for each query"""

        # setup data
        self.entity_group_pairs: list[tuple[list[tuple[int, ...]], list[tuple[int, ...]]]] = []
        self.query_keys: list[tuple[list[str], list[str]]] = []

    def add_entity_group_pair(self, entity_group_a: list[tuple[int, ...]], entity_group_b: list[tuple[int, ...]]):
        """Add a pair of entity groups (aka query) to the contact reporter."""
        self.entity_group_pairs.append((entity_group_a, entity_group_b))

    def add_query_keys(self, entity_a_keys: list[str], entity_b_keys: list[str]):
        """Add entity keys (names) for a contact query."""
        self.query_keys.append((entity_a_keys, entity_b_keys))

    def finalize(self):
        # TODO: speed up entity pair filtering by finding collision groups per entity
        # simplify entity groups

        def normalize_entity(entity_shapes):
            return tuple(sorted(set(entity_shapes))), None

        # create a directory of entities and reference entities by id from the queries
        entities = {}
        self.entities = entities
        entity_group_pairs = []
        for entity_group_a, entity_group_b in self.entity_group_pairs:
            group_a = tuple(e for e, _ in _sort_into(entities, entity_group_a, normalize_entity))
            group_b = tuple(e for e, _ in _sort_into(entities, entity_group_b, normalize_entity))
            entity_group_pairs.append((group_a, group_b))

        # create a directory of entity pairs and reference them by id from the queries
        entity_pairs = {}
        query_to_entity_pair = []
        for e_group_a, e_group_b in entity_group_pairs:
            query_entity_pairs = _sort_into(
                entity_pairs,
                itertools.product(e_group_a, e_group_b),
                normalize_fn=_normalize_pair,
                filter_fn=lambda pair: pair[0] != pair[1],
            )
            query_to_entity_pair.append(list(query_entity_pairs))

        self.query_to_entity_pair = query_to_entity_pair
        self.entity_pairs = list(entity_pairs)

        self.n_entity_pairs = len(entity_pairs)

        # TODO: use natural ordering of shape pairs

        # create a directory of shape pairs and reference them by id from the entity pairs
        shape_pairs = {}
        entity_p_shape_p_map = {}
        entity_list = list(entities.keys())

        for e_a, e_b in entity_pairs:
            # TODO: ensure that two entities don't overlap
            if e_a == e_b:
                continue
            entity_shape_pairs = itertools.product(entity_list[e_a], entity_list[e_b])
            entity_p_shape_p_map[e_a, e_b] = list(
                _sort_into(
                    shape_pairs,
                    entity_shape_pairs,
                    normalize_fn=_normalize_pair,
                    filter_fn=lambda pair: pair[0] != pair[1],
                )
            )

        self.n_shape_pairs = len(shape_pairs)

        # for each bin(shape pair), allocate the necessary space
        def shape_pair_maxcontacts(shape_pair):
            return 24  # TODO

        bin_size = list(map(shape_pair_maxcontacts, shape_pairs))
        bin_start = np.cumsum([0] + bin_size[:-1])  # Cumulative sum for start indices

        total_bin_size = sum(bin_size)  # Total size for linearized arrays

        self.entity_p_shape_p_map = entity_p_shape_p_map

        # sort shape pairs, bin start and bin size according to shape pair ordinal
        shape_pairs_sorted, shape_pairs_position, bin_size, bin_start_sorted = zip(
            *sorted(zip(shape_pairs, itertools.count(), bin_size, bin_start))
        )

        shape_pairs_id_to_ord = np.argsort(shape_pairs_position)

        ep_sp_ord = []
        ep_sp_num = []
        ep_sp_flip = []

        for _ep, sps in entity_p_shape_p_map.items():
            for sp_idx, flip in sps:
                ep_sp_ord.append(shape_pairs_id_to_ord[sp_idx])
                ep_sp_flip.append(flip)
            ep_sp_num.append(len(sps))
        ep_sp_start = np.cumsum([0] + ep_sp_num[:-1])

        n_bins = self.n_shape_pairs

        with wp.ScopedDevice(self.model.device):
            self.shape_pairs = wp.array(shape_pairs_sorted, dtype=wp.vec2i)
            self.bin_start = wp.array(bin_start, dtype=wp.int32)
            self.bin_start_sorted = wp.array(bin_start_sorted, dtype=wp.int32)
            self.bin_count = wp.zeros(n_bins, dtype=wp.int32)

            # TODO: skip reordering of shape pairs
            self.ep_sp_num = wp.array(ep_sp_num, dtype=wp.int32)
            self.ep_sp_start = wp.array(ep_sp_start, dtype=wp.int32)
            self.ep_sp_ord = wp.array(ep_sp_ord, dtype=wp.int32)
            self.ep_sp_flip = wp.array(ep_sp_flip, dtype=wp.int32)

            self.bin_contacts = wp.empty(total_bin_size, dtype=wp.int32)
            self.bin_contacts_dist = wp.empty(total_bin_size, dtype=wp.float32)

            self.entity_pair_contact = wp.empty(self.n_entity_pairs, dtype=wp.int32)
            self.entity_pair_dist = wp.empty(self.n_entity_pairs, dtype=wp.float32)
            self.entity_pair_normal = wp.empty(self.n_entity_pairs, dtype=wp.vec3)
            self.entity_pair_force = wp.empty(self.n_entity_pairs, dtype=wp.vec3)

            # Pre-initialize contact matrices for each query
            self.query_entities = []

            self.query_entity_pairs = []
            self.query_entity_pair_mat_idx = []

            self.query_dist_matrix = []
            self.query_force_matrix = []
            self.query_normal_matrix = []
            self.query_idx_matrix = []
            self.query_flip = []

            for query_idx in range(len(self.entity_group_pairs)):
                query_pairs = self.query_to_entity_pair[query_idx]
                entity_pairs = [
                    self.entity_pairs[query_pair][::-1] if flip else self.entity_pairs[query_pair]
                    for query_pair, flip in query_pairs
                ]
                row_entities, col_entities = zip(*entity_pairs)
                row_indices = {row: i for i, row in enumerate(sorted(set(row_entities)))}
                col_indices = {col: i for i, col in enumerate(sorted(set(col_entities)))}

                # Store entities mapping for this query
                entities = (row_indices, col_indices)
                self.query_entities.append(entities)

                query_ep_idx, _query_flip = zip(*query_pairs)

                self.query_entity_pairs.append(wp.array(query_ep_idx, dtype=wp.int32))
                self.query_flip.append(wp.array(_query_flip, dtype=wp.int32))
                q_ep_mat_idx = [(row_indices[row], col_indices[col]) for row, col in entity_pairs]

                self.query_entity_pair_mat_idx.append(wp.array(q_ep_mat_idx, dtype=wp.vec2i))
                m, n = len(row_indices), len(col_indices)
                self.query_dist_matrix.append(wp.full((m, n), wp.inf, dtype=wp.float32))
                self.query_force_matrix.append(wp.full((m, n), 0, dtype=wp.vec3))
                self.query_normal_matrix.append(wp.full((m, n), 0, dtype=wp.vec3))
                self.query_idx_matrix.append(wp.full((m, n), -1, dtype=wp.int32))

    def reset(self):
        """Clear intermediate data"""
        self.bin_count.zero_()

    def select_aggregate(
        self,
        contact: ContactInfo,
        num_contacts: wp.array(dtype=wp.int32),
        solver: SolverBase | None = None,
    ):
        self.reset()

        if solver is not None:
            if isinstance(solver, MuJoCoSolver):
                solver.update_newton_contacts(self.model, solver.mjw_data, contact)

        wp.launch(
            select_bin_contacts,
            dim=NUM_THREADS,
            inputs=[
                contact.pair,
                contact.separation,
                self.shape_pairs,
                self.n_shape_pairs,
                num_contacts,
                self.bin_start_sorted,
            ],
            outputs=[
                self.bin_contacts,
                self.bin_count,
                self.bin_contacts_dist,
            ],
        )

        wp.launch(
            aggregate_entity_pair_maxdepth,
            dim=self.n_entity_pairs,
            inputs=[
                contact.normal,
                self.n_entity_pairs,
                self.ep_sp_ord,
                self.ep_sp_start,
                self.ep_sp_num,
                self.ep_sp_flip,
                self.bin_start_sorted,
                self.bin_contacts,
                self.bin_contacts_dist,
                self.bin_count,
            ],
            outputs=[
                self.entity_pair_contact,
                self.entity_pair_dist,
                self.entity_pair_normal,
            ],
        )

        if contact.force is not None:
            wp.launch(
                aggregate_entity_pair_net_force,
                dim=self.n_entity_pairs,
                inputs=[
                    contact.normal,
                    self.n_entity_pairs,
                    self.ep_sp_ord,
                    self.ep_sp_start,
                    self.ep_sp_num,
                    self.ep_sp_flip,
                    self.bin_start_sorted,
                    self.bin_contacts,
                    contact.force,
                    self.bin_count,
                ],
                outputs=[
                    self.entity_pair_force,
                ],
            )

    def fill_contact_matrix(self, query_idx: int, data, matrix, query_flip, flip):
        wp.launch(
            fill_contact_matrix,
            dim=self.query_entity_pairs[query_idx].shape[0],
            inputs=[
                self.query_entity_pairs[query_idx],
                self.query_entity_pair_mat_idx[query_idx],
                data,
                query_flip[query_idx],
                flip,
            ],
            outputs=[matrix],
        )

    def get_dist(self, query_idx: int):
        matrix = self.query_dist_matrix[query_idx]
        self.fill_contact_matrix(query_idx, self.entity_pair_dist, matrix, self.query_flip, False)
        return self.query_entities[query_idx], matrix

    def get_force(self, query_idx: int):
        matrix = self.query_force_matrix[query_idx]
        self.fill_contact_matrix(query_idx, self.entity_pair_force, matrix, self.query_flip, True)
        return self.query_entities[query_idx], matrix

    def get_normal(self, query_idx: int):
        matrix = self.query_normal_matrix[query_idx]
        self.fill_contact_matrix(query_idx, self.entity_pair_normal, matrix, self.query_flip, True)
        return self.query_entities[query_idx], matrix

    def get_idx(self, query_idx: int):
        matrix = self.query_idx_matrix[query_idx]
        self.fill_contact_matrix(query_idx, self.entity_pair_contact, matrix, self.query_flip, False)
        return self.query_entities[query_idx], matrix

    def get_query_keys(self, query_idx: int):
        return self.query_keys[query_idx]
