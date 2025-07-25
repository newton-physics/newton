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

import unittest

import warp as wp

import newton
from newton.tests.unittest_utils import assert_np_equal
from newton.utils.contact_sensor import ContactReporter, ContactSensorManager, ContactView, MatchAny


class MockModel:
    """Minimal mock model for testing ContactReporter"""

    def __init__(self, device=None):
        self.device = device or wp.get_device()


def create_contact_info(device, pairs, nconmax, positions=None, normals=None, separations=None, forces=None):
    """Helper to create ContactInfo with specified contacts"""
    contact_info = newton.sim.contacts.ContactInfo()

    n_contacts = len(pairs)

    if positions is None:
        positions = [[0.0, 0.0, 0.0]] * n_contacts
    if normals is None:
        normals = [[0.0, 0.0, 1.0]] * n_contacts
    if separations is None:
        separations = [-0.1] * n_contacts
    if forces is None:
        forces = [0.1] * n_contacts

    pairs_padded = pairs + [(-1, -1)] * (nconmax - n_contacts)
    positions_padded = positions + [[0.0, 0.0, 0.0]] * (nconmax - n_contacts)
    normals_padded = normals + [[0.0, 0.0, 0.0]] * (nconmax - n_contacts)
    separations_padded = separations + [0.0] * (nconmax - n_contacts)
    forces_padded = forces + [0.0] * (nconmax - n_contacts)

    with wp.ScopedDevice(device):
        contact_info.pair = wp.array(pairs_padded, dtype=wp.vec2i)
        contact_info.position = wp.array(positions_padded, dtype=wp.vec3f)
        contact_info.normal = wp.array(normals_padded, dtype=wp.vec3f)
        contact_info.separation = wp.array(separations_padded, dtype=wp.float32)
        contact_info.force = wp.array(forces_padded, dtype=wp.float32)

        contact_info.n_contacts = wp.array([n_contacts], dtype=wp.int32)

    return contact_info


class TestContactSensor(unittest.TestCase):
    def test_net_force_aggregation(self):
        """Test net force aggregation across different contact subsets"""
        device = wp.get_device()

        # Define entities: Entity A = (0,1), Entity B = (2)
        entity_A = (0, 1)
        entity_B = (2,)

        entity_pairs = [
            (entity_A, entity_B),
            (entity_B, entity_A),
            (entity_A, MatchAny),
            (entity_B, MatchAny),
        ]
        contact_reporter = ContactReporter(entity_pairs)

        contacts = [
            {
                "pair": (0, 2),
                "position": [0.0, 0.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
                "separation": -0.01,
                "force": 1.0,
            },
            {
                "pair": (1, 2),
                "position": [0.1, 0.0, 0.0],
                "normal": [1.0, 0.0, 0.0],
                "separation": -0.02,
                "force": 2.0,
            },
            {
                "pair": (2, 1),
                "position": [0.2, 0.0, 0.0],
                "normal": [0.0, 1.0, 0.0],
                "separation": -0.015,
                "force": 1.5,
            },
            {
                "pair": (0, 3),
                "position": [0.3, 0.0, 0.0],
                "normal": [0.0, 0.0, -1.0],
                "separation": -0.005,
                "force": 0.5,
            },
        ]

        pairs = [contact["pair"] for contact in contacts]
        positions = [contact["position"] for contact in contacts]
        normals = [contact["normal"] for contact in contacts]
        separations = [contact["separation"] for contact in contacts]
        forces = [contact["force"] for contact in contacts]

        test_scenarios = [
            {
                "name": "no_contacts",
                "pairs": [],
                "positions": [],
                "normals": [],
                "separations": [],
                "forces": [],
                "force_A_vs_B": (0.0, 0.0, 0.0),
                "force_B_vs_A": (0.0, 0.0, 0.0),
                "force_A_vs_All": (0.0, 0.0, 0.0),
                "force_B_vs_All": (0.0, 0.0, 0.0),
            },
            {
                "name": "only_contact_0",
                "pairs": pairs[:1],
                "positions": positions[:1],
                "normals": normals[:1],
                "separations": separations[:1],
                "forces": forces[:1],
                "force_A_vs_B": (0.0, 0.0, 1.0),
                "force_B_vs_A": (0.0, 0.0, -1.0),
                "force_A_vs_All": (0.0, 0.0, 1.0),
                "force_B_vs_All": (0.0, 0.0, -1.0),
            },
            {
                "name": "only 1",
                "pairs": pairs[1:2],
                "positions": positions[1:2],
                "normals": normals[1:2],
                "separations": separations[1:2],
                "forces": forces[1:2],
                "force_A_vs_B": (2.0, 0.0, 0.0),
                "force_B_vs_A": (-2.0, 0.0, 0.0),
                "force_A_vs_All": (2.0, 0.0, 0.0),
                "force_B_vs_All": (-2.0, 0.0, 0.0),
            },
            {
                "name": "only 2",
                "pairs": pairs[2:3],
                "positions": positions[2:3],
                "normals": normals[2:3],
                "separations": separations[2:3],
                "forces": forces[2:3],
                "force_A_vs_B": (0.0, -1.5, 0.0),
                "force_B_vs_A": (0.0, 1.5, 0.0),
                "force_A_vs_All": (0.0, -1.5, 0.0),
                "force_B_vs_All": (0.0, 1.5, 0.0),
            },
            {
                "name": "all_contacts",
                "pairs": pairs,
                "positions": positions,
                "normals": normals,
                "separations": separations,
                "forces": forces,
                "force_A_vs_B": (2.0, -1.5, 1.0),
                "force_B_vs_A": (-2.0, 1.5, -1.0),
                "force_A_vs_All": (2.0, -1.5, 0.5),
                "force_B_vs_All": (-2.0, 1.5, -1.0),
            },
        ]

        for scenario in test_scenarios:
            with self.subTest(scenario=scenario["name"]):
                contact_info = create_contact_info(
                    device,
                    scenario["pairs"],
                    nconmax=10,
                    positions=scenario["positions"],
                    normals=scenario["normals"],
                    separations=scenario["separations"],
                    forces=scenario["forces"],
                )

                contact_reporter._select_aggregate_net_force(contact_info)

                self.assertIsNotNone(contact_reporter.net_force)
                self.assertEqual(contact_reporter.net_force.shape[0], contact_reporter.n_entity_pairs)

                self.assertTrue(contact_reporter.net_force.dtype == wp.vec3)

                net_forces = contact_reporter.net_force.numpy()

                assert_np_equal(net_forces[0], scenario["force_A_vs_B"])
                assert_np_equal(net_forces[1], scenario["force_B_vs_A"])
                assert_np_equal(net_forces[2], scenario["force_A_vs_All"])
                assert_np_equal(net_forces[3], scenario["force_B_vs_All"])


class TestContactSensorManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.device = wp.get_device()
        self.model = MockModel(self.device)
        self.cm = ContactSensorManager(self.model)

    def test_add_contact_query(self):
        """Test sensor list building"""
        contact_view = ContactView(query_id=0, args={})
        sensor_entities = [(0,), (1,)]
        select_entities = [(2,), (3,)]

        colliding_shape_pairs = {(0, 2)}

        self.cm.add_contact_query(
            sensor_entities=sensor_entities,
            select_entities=select_entities,
            sensor_keys=["sensor_0", "sensor_1"],
            select_keys=["target_2", "target_3"],
            contact_view=contact_view,
            colliding_shape_pairs=colliding_shape_pairs,
        )

        query = self.cm.contact_queries[0]
        sensor_list = query.sensor_list

        self.assertEqual(sensor_list[0], (0, (0,)))
        self.assertEqual(sensor_list[1], (1, ()))

    def test_finalize(self):
        """Test finalizing ContactSensorManager"""
        views = [ContactView(query_id=0, args={}), ContactView(query_id=1, args={})]
        queries = [
            {
                "sensor_entities": [(0,)],
                "select_entities": [(1,)],
                "sensor_keys": ["sensor_1"],
                "select_keys": ["target_1"],
                "contact_view": views[0],
            },
            {
                "sensor_entities": [(2,), (3,)],
                "select_entities": [(4,), MatchAny],
                "sensor_keys": ["sensor_2", "sensor_3"],
                "select_keys": ["target_4", "all"],
                "contact_view": views[1],
            },
        ]
        for q in queries:
            self.cm.add_contact_query(**q)
        self.cm.finalize()
        self.assertTrue(all(v.finalized for v in views))
        self.assertEqual(views[0].sensor_keys, ["sensor_1"])
        self.assertEqual(views[0].contact_partner_keys, ["target_1"])
        self.assertEqual(views[0].shape, (1, 1))
        self.assertEqual(views[1].sensor_keys, ["sensor_2", "sensor_3"])
        self.assertEqual(views[1].contact_partner_keys, ["target_4", "all"])
        self.assertEqual(views[1].shape, (2, 2))

    def test_eval_contact_sensors(self):
        """Test evaluating contact sensors"""
        contact_view = ContactView(query_id=0, args={})

        self.cm.add_contact_query(
            sensor_entities=[(0,)],
            select_entities=[(1,)],
            sensor_keys=["sensor_A"],
            select_keys=["entity_B"],
            contact_view=contact_view,
        )

        self.cm.finalize()

        contact_info = create_contact_info(
            self.device,
            [(0, 1)],
            nconmax=10,
            forces=[2.0],
            normals=[[1.0, 0.0, 0.0]],
        )

        self.cm.eval_contact_sensors(contact_info)

        net_forces = self.cm.contact_reporter.net_force.numpy()
        expected_force = [2.0, 0.0, 0.0]
        assert_np_equal(net_forces[0], expected_force)

    def test_build_entity_pair_list(self):
        """Test building entity pair list"""
        contact_view = ContactView(query_id=0, args={})

        self.cm.add_contact_query(
            sensor_entities=[(0,), (1, 2)],
            select_entities=[(3,), MatchAny],
            sensor_keys=["sensor_0", "sensor_12"],
            select_keys=["target_3", "all"],
            contact_view=contact_view,
        )

        self.cm._build_entity_pair_list()

        self.assertEqual(self.cm.query_shape[0], (2, 2))

        entity_pairs = self.cm.entity_pairs[0]
        expected_pairs = [
            ((0,), (3,)),
            ((0,), MatchAny),
            ((1, 2), (3,)),
            ((1, 2), MatchAny),
        ]
        self.assertEqual(entity_pairs, expected_pairs)

    def test_empty_query_handling(self):
        """Test handling of queries with no valid sensor-select pairs"""
        contact_view = ContactView(query_id=0, args={})

        colliding_shape_pairs = set()

        self.cm.add_contact_query(
            sensor_entities=[(0,)],
            select_entities=[(1,)],
            sensor_keys=["sensor_0"],
            select_keys=["target_1"],
            contact_view=contact_view,
            colliding_shape_pairs=colliding_shape_pairs,
        )

        query = self.cm.contact_queries[0]
        sensor_list = query.sensor_list

        self.assertEqual(sensor_list[0], (0, ()))


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
