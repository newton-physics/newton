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

from collections import defaultdict
from collections.abc import Iterable
import itertools

import numpy as np
import warp as wp

from newton import Model
from newton.sim.contacts import ContactInfo, Contacts
from newton.solvers import MuJoCoSolver, SolverBase

NUM_THREADS = 8192


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


@wp.kernel
def select_aggregate_net_force(
    num_contacts: wp.array(dtype=wp.int32),
    sp_sorted: wp.array(dtype=wp.vec2i),
    num_sp: int,
    sp_ep: wp.array(dtype=wp.vec2i),
    sp_ep_offset: wp.array(dtype=wp.int32),
    sp_ep_count: wp.array(dtype=wp.int32),
    contact_pair: wp.array(dtype=wp.vec2i),
    contact_normal: wp.array(dtype=wp.vec3f),
    contact_force: wp.array(dtype=wp.float32),
    # output
    net_force: wp.array(dtype=wp.vec3),
):
    ncon = num_contacts[0]
    n_blocks = (ncon + NUM_THREADS - 1) // NUM_THREADS
    for b in range(n_blocks):
        con_idx = wp.tid() + NUM_THREADS * b
        if con_idx >= ncon:
            return

        pair = contact_pair[con_idx]

        # Find the entity pairs
        smin, smax = wp.min(pair[0], pair[1]), wp.max(pair[0], pair[1])

        # add contribution for shape pair
        normalized_pair = wp.vec2i(smin, smax)
        sp_flip = not (normalized_pair == pair)
        # sp_flip = normalized_pair[0] != pair[0]
        sp_ord = bisect_shape_pairs(sp_sorted, num_sp, normalized_pair)

        force = contact_force[con_idx] * contact_normal[con_idx]
        if sp_ord < num_sp and sp_sorted[sp_ord] == normalized_pair:
            # add the force to the pair's force accumulators
            offset = sp_ep_offset[sp_ord]
            for i in range(sp_ep_count[sp_ord]):
                ep = sp_ep[offset + i]
                force_acc, flip = ep[0], ep[1]
                wp.atomic_add(net_force, force_acc, wp.where(sp_flip != flip, -force, force))

        # add contribution for shape a and b
        for i in range(2):
            mono_sp = wp.vec2i(-1, pair[i])
            mono_ord = bisect_shape_pairs(sp_sorted, num_sp, mono_sp)

            # for shape vs all, only one accumulator is supported and flip is trivially true
            if mono_ord < num_sp and sp_sorted[mono_ord] == mono_sp:
                force_acc = sp_ep[sp_ep_offset[mono_ord]][0]
                wp.atomic_add(net_force, force_acc, wp.where(bool(i), -force, force))


def _lol_to_arrays(list_of_lists: list[list], dtype) -> tuple[wp.array, wp.array, wp.array]:
    """Convert a list of lists to three warp arrays containing the values, offsets and counts.
    Does nothing and returns None, None, None if the list is empty.
    """
    if not list_of_lists:
        return None, None, None
    a = wp.array([el for l in list_of_lists for el in l], dtype=dtype)
    count_list = list(map(len, list_of_lists))
    offset = wp.array(np.cumsum([0, *count_list[:-1]]), dtype=wp.int32)
    count = wp.array(count_list, dtype=wp.int32)
    return a, offset, count


class ContactView:
    """A view for querying contacts between entities in the simulation."""

    def __init__(self, query_id: int, args: dict):
        # self.contact_reporter = contact_reporter
        self.query_id = query_id
        self.args = args
        self.finalized = False
        self.shape = None

        self.net_force = None  # force matrix, aliased to contact reducer
        self.entity_pairs = None  # entity pair matrix


def convert_contact_info(
    model: Model,
    contact_info: ContactInfo,
    solver: SolverBase | None = None,
    contacts: Contacts | None = None,
):
    """Populate ContactInfo object from the solver or from the Contacts object."""
    if solver is not None:
        if isinstance(solver, MuJoCoSolver):
            solver.update_newton_contacts(model, solver.mjw_data, contact_info)
        else:
            raise NotImplementedError("Contact conversion not yet implemented this solver")


class ContactSensorManager:
    def __init__(self, model):
        self.sensors = []
        self.contact_queries = []
        self.contact_views = []
        self.contact_reporter = None
        self.model = model

    def add_contact_query(
        self,
        contact_view: ContactView,
        sensor_entities: list[tuple[int, ...]],
        select_entities: list[tuple[int, ...] | None],
        sensor_matrix: list[tuple[int, tuple[int, ...]]],
    ):
        self.contact_queries.append((sensor_entities, select_entities, sensor_matrix))
        self.contact_views.append(contact_view)

    def eval_contact_sensors(self, contact_info):
        self.contact_reporter._select_aggregate_net_force(contact_info)

    def build_entity_pair_list(self):
        query_to_eps = []
        entity_pair_list = []
        query_len = []
        query_shape = []

        # TODO: generalize to support partial sensor matrix
        for sensor_entities, select_entities, sensor_matrix in self.contact_queries:
            n_query_sensors = len(sensor_matrix)
            n_query_selects = max(len(partners) for _, partners in sensor_matrix)
            print(f"Sensors in query: {n_query_sensors}\t Max selects: {n_query_selects}")
            query_eps = [
                (sensor_entities[sensor], select_entities[select])
                for sensor, selects in sensor_matrix
                for select in selects
            ]
            query_to_eps.append(query_to_eps)
            query_len.append(n_query_sensors * n_query_selects)
            query_shape.append((n_query_sensors, n_query_selects))
            print(f"query_to_eps: {query_to_eps}")
            entity_pair_list.extend(query_eps)

        self.query_count = query_len
        self.query_offset = [0, *np.cumsum(query_len[:-1]).tolist()]
        self.query_shape = query_shape
        return entity_pair_list

    def finalize(self):
        self.entity_pairs = self.build_entity_pair_list()
        self.contact_reporter = ContactReporter(self.model, self.entity_pairs)
        for offset, count, view, shape in zip(
            self.query_offset, self.query_count, self.contact_views, self.query_shape
        ):
            view.net_force = self.contact_reporter.net_force[offset : offset + count].reshape(shape)


class ContactReporter:
    """Aggregates contacts per entity pair"""

    def __init__(self, model: Model, entity_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]]):
        self.model = model

        # initialize mapping from sp to eps & flips
        self.n_entity_pairs = len(entity_pairs)
        self._create_sp_ep_arrays(entity_pairs)
        # net force (1 vec3 per entity pair)
        self.net_force = wp.zeros(self.n_entity_pairs, dtype=wp.vec3)

        return

    def _create_sp_ep_arrays(self, entity_pairs: Iterable[tuple[tuple[int, ...], tuple[int, ...] | None]]):
        """Build a mapping from shape pairs to entity pairs ordered by shape pair."""
        sp_ep_map = defaultdict(list)
        for ep_idx, (e1, e2) in enumerate(entity_pairs):
            assert e1 is not None, "Sensor cannot be attached to wildcard entity"

            # pair e1 shapes with e2 shapes, or with -1 if e2 is None
            shape_pairs = itertools.product(e1, (-1,) if e2 is None else e2)
            for sp in shape_pairs:
                flip_pair = sp[0] > sp[1]
                if flip_pair:
                    sp = sp[1], sp[0]  # noqa
                # print(f"Adding shape pair {sp} to entity pair {ep_idx} (flip={flip_pair})")

                sp_ep_map[sp].append((ep_idx, flip_pair))

        # sort by shape pair for fast retrieval
        sp_sorted = sorted(sp_ep_map)
        sp_ep = [sp_ep_map[sp] for sp in sp_sorted]

        # store for debugging
        self.sp_sorted_list = sp_sorted
        self.sp_ep_list = sp_ep

        self.n_shape_pairs = len(sp_sorted)

        # TODO: ensure no symmetric pairs

        # initialize warp arrays
        self.sp_sorted = wp.array(sp_sorted, dtype=wp.vec2i)
        self.sp_ep, self.sp_ep_offset, self.sp_ep_count = _lol_to_arrays(sp_ep, wp.vec2i)

    def _select_aggregate_net_force(self, contact: ContactInfo):
        self.net_force.zero_()
        wp.launch(
            select_aggregate_net_force,
            dim=NUM_THREADS,
            inputs=[
                contact.n_contacts,
                self.sp_sorted,
                self.n_shape_pairs,
                self.sp_ep,
                self.sp_ep_offset,
                self.sp_ep_count,
                contact.pair,
                contact.normal,
                contact.force,
            ],
            outputs=[self.net_force],
        )
