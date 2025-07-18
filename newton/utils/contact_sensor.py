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
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
import warp as wp

from newton import Model
from newton.sim.contacts import ContactInfo, Contacts
from newton.solvers import MuJoCoSolver, SolverBase

NUM_THREADS = 8192


class SentinelMeta(type):
    def __repr__(cls):
        return f"<{cls.__name__}>"


class MatchAny(metaclass=SentinelMeta):
    """Sentinel class matching all contact partners."""


EntityKind = Enum("EntityKind", [("SHAPE", 1), ("BODY", 1)])


Entity: TypeAlias = tuple[int, ...]


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


class ContactView:
    """A view for querying contacts between entities in the simulation.
    This class stores the parameters of the query and provides a view of the results.
    """

    def __init__(self, query_id: int, args: dict):
        self.query_id = query_id
        self.args: dict[str, Any] = args
        self.finalized: bool = False
        self.shape: tuple[int] = None

        self.net_force: wp.array(dtype=wp.vec3) = None  # force matrix, aliased to contact reporter
        """Net force matrix, shape (n_sensors, n_contact_partners [+1 if total included])"""

        self.sensor_keys: list[str] = None
        """Keys for the sensors in the query, n_sensors"""
        self.contact_partner_keys: list[str] = None
        """Keys for the contact partners in the query, n_contact_partners"""
        self.sensor_entities: list[tuple[int, ...]] = None
        """Entities for the sensors in the query, n_sensors"""
        self.contact_partner_entities: list[tuple[int, ...]] = None
        """Entities for the contact partners in the query, n_contact_partners"""
        self.entity_pairs: np.ndarray = None  # entity pair matrix
        """Pairs of sensor and contact partner indices for the query, shape (n_sensors, n_contact_partners, 2)"""

    def finalize(
        self,
        net_force: wp.array(dtype=wp.vec3),
        sensor_keys: list[str],
        contact_partner_keys: list[str],
        sensor_entities: list[tuple[int, ...]],
        contact_partner_entities: list[tuple[int, ...]],
        entity_pairs: np.ndarray,
    ):
        assert not self.finalized
        self.net_force = net_force
        self.shape = self.net_force.shape
        self.sensor_keys = sensor_keys
        self.contact_partner_keys = contact_partner_keys
        self.sensor_entities = sensor_entities
        self.contact_partner_entities = contact_partner_entities
        self.entity_pairs = entity_pairs

        self.finalized = True


@dataclass
class ContactQuery:
    """Contact Query data
    sensor_entities: List of entity tuples representing the sensor entities.
        These are indexed by the first element of sensor_matrix tuples.
    select_entities: List of entity tuples representing the contact partners.
        These are indexed by the values in the second element of the sensor_matrix tuples.
    sensor_matrix: List of tuples defining which sensors interact with which select entities.
        Each tuple is (sensor_index, tuple of select_indices) where:
        - sensor_index: Index into sensor_entities list.
        - select_indices: Tuple of indices into select_entities list.
        The resulting contact matrix will have shape (len(sensor_matrix), max_selects)
        where max_selects is the maximum length of select_indices tuples.
    """

    sensor_entities: list[Entity]
    select_entities: list[Entity]

    sensor_list: list[tuple[int, tuple[int, ...]]] | None = None
    """List of sensors. Indexes sensor_entities and select_entities"""

    sensor_keys: list[str] | None = None
    select_keys: list[str] | None = None

    colliding_shape_pairs: set[tuple[int, int]] | None = None
    # sensor_matrix: list[list[tuple[int, int]]]


class ContactSensorManager:
    def __init__(self, model):
        self.sensors = []
        self.contact_queries: list[ContactQuery] = []
        self.contact_views = []
        self.contact_reporter = None
        self.model = model

    def add_contact_query(
        self,
        sensor_entities: list[tuple[int, ...]],
        select_entities: list[tuple[int, ...] | MatchAny],
        sensor_keys: list[tuple],
        select_keys: list[tuple],
        contact_view: ContactView,
        colliding_shape_pairs: set[tuple[int, int]] | None = None,
    ):
        """Add a contact query to track contact forces between specified entities.

        Args:
            query: ContactQuery object containing sensor and select indices and keys.
            contact_view: ContactView object that will reference the results of this query.
        """
        query = ContactQuery(
            sensor_entities=sensor_entities,
            select_entities=select_entities,
            sensor_keys=sensor_keys,
            select_keys=select_keys,
            colliding_shape_pairs=colliding_shape_pairs,
        )

        query.sensor_list = self._build_sensor_list(query)

        self.contact_queries.append(query)
        self.contact_views.append(contact_view)

    def eval_contact_sensors(self, contact_info):
        self.contact_reporter._select_aggregate_net_force(contact_info)

    def finalize(self):
        self._build_entity_pair_list()
        self.contact_reporter = ContactReporter(self.model, [ep for query in self.entity_pairs for ep in query])

        for offset, count, view, shape, query, entity_pairs_idx in zip(
            self.query_offset,
            self.query_count,
            self.contact_views,
            self.query_shape,
            self.contact_queries,
            self.entity_pairs_idx,
        ):
            net_force = self.contact_reporter.net_force[offset : offset + count].reshape(shape)
            n_sens, n_sel = shape
            entity_pair_matrix = [
                [entity_pairs_idx[sensor * n_sel + sel] or (None, None) for sel in range(n_sel)]
                for sensor in range(n_sens)
            ]
            view_entity_pairs = np.array(entity_pair_matrix)

            view.finalize(
                net_force,
                query.sensor_keys,
                query.select_keys,
                query.sensor_entities,
                query.select_entities,
                view_entity_pairs,
            )
            breakpoint()

    @staticmethod
    def _build_sensor_list(
        query: ContactQuery,
    ) -> list[tuple[int, tuple[int, ...]]]:
        """Build the list of sensor - select combinations, as tuples of (sensor_idx, (select_indices...)).
        If colliding_shape_pairs is provided, for each sensor, keep only valid select indices."""
        sensor_list = []

        def check_ep_can_collide(a: Entity, b: Entity) -> bool:
            ep_sps = {(min(pair), max(pair)) for pair in itertools.product(a, b)}
            return not query.colliding_shape_pairs.isdisjoint(ep_sps)

        for sensor_idx, sensor_entity in enumerate(query.sensor_entities):
            select_indices = tuple(
                select_idx
                for select_idx, select_entity in enumerate(query.select_entities)
                if query.colliding_shape_pairs is None or check_ep_can_collide(sensor_entity, select_entity)
            )
            sensor_list.append((sensor_idx, select_indices))
        return sensor_list

    def _build_entity_pair_list(self):
        self.entity_pairs = []
        self.entity_pairs_idx = []
        self.query_shape = []

        for query in self.contact_queries:
            n_query_sensors = len(query.sensor_list)
            n_query_selects = max(len(selects) for _, selects in query.sensor_list)
            query_eps = []
            query_eps_idx = []

            for sensor_idx, selects in query.sensor_list:
                for select_idx in selects:
                    query_eps.append((query.sensor_entities[sensor_idx], query.select_entities[select_idx]))
                    query_eps_idx.append((sensor_idx, select_idx))
                padding = (None,) * (n_query_selects - len(selects))  # fill up with None
                query_eps.extend(padding)
                query_eps_idx.extend(padding)

            assert n_query_selects * n_query_sensors == len(query_eps)
            self.query_shape.append((n_query_sensors, n_query_selects))
            self.entity_pairs.append(query_eps)
            self.entity_pairs_idx.append(query_eps_idx)

        self.query_count = [n_sens * n_sel for n_sens, n_sel in self.query_shape]
        self.query_offset = [0, *np.cumsum(self.query_count[:-1]).tolist()]


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

    def _create_sp_ep_arrays(self, entity_pairs: Iterable[tuple[tuple[int, ...], tuple[int, ...] | MatchAny] | None]):
        """Build a mapping from shape pairs to entity pairs ordered by shape pair.
        None is accepted as a filler value."""
        sp_ep_map = defaultdict(list)
        for ep_idx, entity_pair in enumerate(entity_pairs):
            if entity_pair is None:
                continue
            e1, e2 = entity_pair
            assert e1 is not MatchAny, "Sensor cannot be attached to wildcard entity"

            # pair e1 shapes with e2 shapes, or with -1 if e2 is MatchAny
            shape_pairs = itertools.product(e1, ((-1,) if e2 is MatchAny else e2))
            for sp in shape_pairs:
                flip_pair = sp[0] > sp[1]
                if flip_pair:
                    sp = sp[1], sp[0]  # noqa

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
