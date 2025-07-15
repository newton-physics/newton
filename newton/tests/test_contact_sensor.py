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

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import assert_np_equal
from newton.utils.contact_reporter import ContactReporter, ContactSensorManager


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
        model = MockModel(device)

        # Define entities: Entity A = (0,1), Entity B = (2)
        entity_A = (0, 1)
        entity_B = (2,)

        entity_pairs = [
            (entity_A, entity_B),
            (entity_B, entity_A),
            (entity_A, None),
            (entity_B, None),
        ]
        contact_reporter = ContactReporter(model, entity_pairs)

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
    def test_add_sensor(self):
        device = wp.get_device()
        model = MockModel(device)
        cm = ContactSensorManager(model)

        contact_info = create_contact_info(
            device,
            [
                (0, 1),
                (2, 3),
            ],
            nconmax=10,
        )

        entity_a = (0,)
        entity_b = (1,)
        cm.add_contact_query([(entity_a, None), ])


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
