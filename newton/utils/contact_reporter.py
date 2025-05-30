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

import numpy as np
import warp as wp

from newton import Contact, Model
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


# Binning function determines the existence and index of appropriate bin
@wp.func
def bin_contact_shape_pair(
    contact_geom: wp.array(dtype=wp.vec2i),
    num_shape_pairs: wp.int32,
    shape_pairs_sorted: wp.array(dtype=wp.vec2i),
    contact_idx: wp.int32,
) -> tuple[bool, wp.int32]:
    """Find the bin index, if it exists, based on the contact's shape_pair."""
    # FIXME: use binary search or hash lookup for shape pairs

    geom = contact_geom[contact_idx]

    if geom[0] == -1 or geom[1] == -1:
        return False, 0

    ga = wp.min(geom[0], geom[1])
    gb = wp.max(geom[0], geom[1])

    for pair_ord in range(num_shape_pairs):
        sp = shape_pairs_sorted[pair_ord]
        if sp[0] > ga:
            return False, 0

        if ga == sp[0] and gb == sp[1]:
            return True, pair_ord
    return False, 0


# TODO: move into MuJoCoSolver
@wp.kernel
def remap_contact_geom_mjw(
    # inputs
    contact_geom_mapping: wp.array2d(dtype=wp.int32),
    contact_geom_mjw: wp.array(dtype=wp.vec2i),
    contact_worldid_mjw: wp.array(dtype=wp.int32),
    num_contacts: wp.array(dtype=wp.int32),
    # outputs
    contact_geom: wp.array(dtype=wp.vec2i),
):
    n_contacts = num_contacts[0]
    n_blocks = (n_contacts + NUM_THREADS - 1) // NUM_THREADS
    for b in range(n_blocks):
        contact_idx = wp.tid() + NUM_THREADS * b
        if contact_idx >= n_contacts:
            return

        worldid = contact_worldid_mjw[contact_idx]
        geoms_mjw = contact_geom_mjw[contact_idx]

        geoms = wp.vec2i()
        for i in range(2):
            geoms[i] = contact_geom_mapping[worldid, geoms_mjw[i]]
        contact_geom[contact_idx] = geoms


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
    bin_contacts: wp.array2d(dtype=wp.int32),
    bin_count: wp.array(dtype=wp.int32),
    bin_contacts_dist: wp.array2d(dtype=wp.float32),
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
        # FIXME: linearize
        bin_contacts[contact_start, i] = contact_idx
        bin_contacts_dist[contact_start, i] = contact_dist[contact_idx]


@wp.kernel
def aggregate_entity_pair_maxdepth(
    # inputs
    contact_frame: wp.array(dtype=wp.mat33f),
    n_entity_pairs: wp.int32,
    ep_sp_ord: wp.array(dtype=wp.int32),
    ep_sp_start: wp.array(dtype=wp.int32),
    ep_sp_num: wp.array(dtype=wp.int32),
    ep_sp_flip: wp.array(dtype=wp.int32),
    bin_start: wp.array(dtype=wp.int32),
    bin_contacts: wp.array2d(dtype=wp.int32),
    bin_dist: wp.array2d(dtype=wp.float32),
    bin_count: wp.array(dtype=wp.int32),
    # outputs
    entity_pair_contact: wp.array(dtype=wp.int32),
    entity_pair_dist: wp.array(dtype=wp.float32),
    entity_pair_normal: wp.array(dtype=wp.vec3f),
):
    # TODO: Add indirection to allow queries to share a shape pair between entity pairs
    # FIXME: invert normal/frame when entity pair order doesn't match shape pair order
    entity_pair_idx = wp.tid()
    if entity_pair_idx >= n_entity_pairs:
        return

    start = ep_sp_start[entity_pair_idx]
    n_shape_pairs = ep_sp_num[entity_pair_idx]

    # track deepest contact
    best_dist = wp.float32(wp.inf)
    best_contact = wp.int32(-1)

    for i in range(n_shape_pairs):
        sp_ord = ep_sp_ord[start + i]
        n_bin_contacts = bin_count[sp_ord]
        contacts_start = bin_start[sp_ord]

        for j in range(n_bin_contacts):
            # FIXME: invert normal/frame when entity pair order doesn't match shape pair order
            contact_idx = bin_contacts[contacts_start, j]  # FIXME: linearize
            dist = bin_dist[contacts_start, j]  # FIXME: linearize

            if dist < best_dist:
                best_dist = dist
                best_contact = contact_idx

    entity_pair_contact[entity_pair_idx] = best_contact
    entity_pair_dist[entity_pair_idx] = best_dist
    # if best_contact != -1:
    #     entity_pair_normal[entity_pair_idx] = wp.transpose(contact_frame[best_contact])[0]


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
        """Normal of deepest contact for each entity pair, shape [n_entity_pairs], vec3f"""

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
        # TODO: linearize
        self.bin_contacts = None
        """Contact index component of bins (shape pair), shape [n_shape_pairs, n_pair_contact_max], int"""

        self.bin_count = None
        """Number of contacts in each bin, shape [n_shape_pairs], int"""

        # inputs to aggregation
        self.bin_contacts_dist = None
        """Contact distance component of bins, shape [n_shape_pairs, n_pair_contact_max], float"""

        # MuJoCo-specific contact mapping
        self.contact_geom_tmp = None
        """Temporary array for remapped contact geometry, shape [n_contacts], vec2i"""
        self.contact_geom_mapping = None
        """Mapping from MuJoCo geometry IDs to shape IDs, shape [n_worlds, max_mj_geom_id], int"""

        # query result matrices
        self.query_dist_matrices = None
        """List of distance matrices for each query"""
        self.query_idx_matrices = None
        """List of contact index matrices for each query"""
        self.query_entities = None
        """List of entity mappings for each query"""

        # setup data
        self.entity_group_pairs: list[tuple[list[tuple[int, ...]], list[tuple[int, ...]]]] = []

    def add_entity_group_pair(self, entity_group_a: list[tuple[int, ...]], entity_group_b: list[tuple[int, ...]]):
        """Add a pair of entity groups to the contact reporter."""
        self.entity_group_pairs.append((entity_group_a, entity_group_b))

    def finalize(self, solver: SolverBase):
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
            # return 24
            return 1  # FIXME: linearize

        bin_size = list(map(shape_pair_maxcontacts, shape_pairs))
        bin_start = np.cumulative_sum(bin_size[:-1], include_initial=True)

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
        ep_sp_start = np.cumulative_sum(ep_sp_num[:-1], include_initial=True)

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

            # FIXME: linearize
            bin_array_shape = (n_bins, 24)
            self.bin_contacts = wp.empty(bin_array_shape, dtype=wp.int32)
            self.bin_contacts_dist = wp.empty(bin_array_shape, dtype=wp.float32)

            if isinstance(solver, MuJoCoSolver):
                max_mj_shape_id = max(idx for w, idx in solver.shape_map.values() if idx is not None)
                geom_mapping = np.full((self.model.num_envs, max_mj_shape_id + 1), -1, dtype=np.int32)

                for shape, (worldid, mj_geom) in solver.shape_map.items():
                    if mj_geom is None:
                        continue
                    if worldid == -1:
                        worldid = slice(None)  # noqa
                    geom_mapping[worldid, mj_geom] = shape

                self.contact_geom_tmp = wp.empty_like(solver.mjw_data.contact.geom)
                self.contact_geom_mapping = wp.array(geom_mapping, dtype=wp.int32)

            self.entity_pair_contact = wp.empty(self.n_entity_pairs, dtype=wp.int32)
            self.entity_pair_dist = wp.empty(self.n_entity_pairs, dtype=wp.float32)
            self.entity_pair_normal = wp.empty(self.n_entity_pairs, dtype=wp.vec3f)

            # Pre-initialize contact matrices for each query
            self.query_dist_matrices = []
            self.query_idx_matrices = []
            self.query_entities = []

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

                # Initialize empty matrices that will be populated during select_aggregate
                contact_dist_matrix = np.zeros((len(row_indices), len(col_indices)))
                contact_idx_matrix = np.zeros((len(row_indices), len(col_indices)), dtype=np.int32)

                self.query_dist_matrices.append(contact_dist_matrix)
                self.query_idx_matrices.append(contact_idx_matrix)

    def reset(self):
        """Clear intermediate data"""
        self.bin_count.zero_()

    def select_aggregate(
        self,
        contacts: Contact,
        num_contacts: wp.array(dtype=wp.int32),
    ):
        self.reset()

        wp.launch(
            remap_contact_geom_mjw,
            dim=NUM_THREADS,
            inputs=[
                self.contact_geom_mapping,
                contacts.geom,
                contacts.worldid,
                num_contacts,
            ],
            outputs=[
                self.contact_geom_tmp,
            ],
        )

        wp.launch(
            select_bin_contacts,
            dim=NUM_THREADS,
            inputs=[
                self.contact_geom_tmp,
                contacts.dist,
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
                contacts.frame,
                self.n_entity_pairs,
                self.ep_sp_ord,
                self.ep_sp_start,
                self.ep_sp_num,
                self.ep_sp_flip,
                self.bin_start,
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

    def fill_contact_matrix(self, query_idx: int, data, matrix):
        query_pairs = self.query_to_entity_pair[query_idx]
        entity_pairs = [
            self.entity_pairs[query_pair][::-1] if flip else self.entity_pairs[query_pair]
            for query_pair, flip in query_pairs
        ]
        row_entities, col_entities = zip(*entity_pairs)
        row_indices = {row: i for i, row in enumerate(sorted(set(row_entities)))}
        col_indices = {col: i for i, col in enumerate(sorted(set(col_entities)))}
        for (pair_idx, _), (row, col) in zip(query_pairs, entity_pairs):
            matrix[row_indices[row], col_indices[col]] = data.numpy()[pair_idx]
        return self.query_entities[query_idx], matrix

    def get_dist(self, query_idx: int):
        entities, contact_matrix = self.fill_contact_matrix(
            query_idx, self.entity_pair_dist, self.query_dist_matrices[query_idx]
        )
        return entities, contact_matrix

    def get_idx(self, query_idx: int):
        entities, contact_matrix = self.fill_contact_matrix(
            query_idx, self.entity_pair_contact, self.query_idx_matrices[query_idx]
        )
        return entities, contact_matrix
