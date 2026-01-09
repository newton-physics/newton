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
from newton.sensors import SensorContact, populate_contacts
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import assert_np_equal


class MockModel:
    """Minimal mock model for testing SensorContact"""

    def __init__(self, device=None):
        self.device = device or wp.get_device()


def create_contacts(device, pairs, naconmax, positions=None, normals=None, separations=None, forces=None):
    """Helper to create Contacts with specified contacts"""
    contacts = newton.Contacts(0, 0)

    n_contacts = len(pairs)

    if positions is None:
        positions = [[0.0, 0.0, 0.0]] * n_contacts
    if normals is None:
        normals = [[0.0, 0.0, 1.0]] * n_contacts
    if separations is None:
        separations = [-0.1] * n_contacts
    if forces is None:
        forces = [0.1] * n_contacts

    pairs_padded = pairs + [(-1, -1)] * (naconmax - n_contacts)
    positions_padded = positions + [[0.0, 0.0, 0.0]] * (naconmax - n_contacts)
    normals_padded = normals + [[0.0, 0.0, 0.0]] * (naconmax - n_contacts)
    separations_padded = separations + [0.0] * (naconmax - n_contacts)
    forces_padded = forces + [0.0] * (naconmax - n_contacts)

    with wp.ScopedDevice(device):
        contacts.pair = wp.array(pairs_padded, dtype=wp.vec2i)
        contacts.position = wp.array(positions_padded, dtype=wp.vec3f)
        contacts.normal = wp.array(normals_padded, dtype=wp.vec3f)
        contacts.separation = wp.array(separations_padded, dtype=wp.float32)
        contacts.force = wp.array(forces_padded, dtype=wp.float32)

        contacts.rigid_contact_count = wp.array([n_contacts], dtype=wp.int32)
        contacts.rigid_contact_max = naconmax

    return contacts


class TestSensorContact(unittest.TestCase):
    def test_net_force_aggregation(self):
        """Test net force aggregation across different contact subsets"""
        device = wp.get_device()

        # Define entities: Entity A = (0,1), Entity B = (2)
        entity_A = (0, 1)
        entity_B = (2,)

        model = MockModel()
        model.body_key = ["A", "B"]
        model.body_shapes = [entity_A, entity_B]

        contact_sensor = SensorContact(model, sensing_obj_bodies="*", counterpart_bodies="*")

        test_contacts = [
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

        pairs = [contact["pair"] for contact in test_contacts]
        positions = [contact["position"] for contact in test_contacts]
        normals = [contact["normal"] for contact in test_contacts]
        separations = [contact["separation"] for contact in test_contacts]
        forces = [contact["force"] for contact in test_contacts]

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
                contacts = create_contacts(
                    device,
                    scenario["pairs"],
                    naconmax=10,
                    positions=scenario["positions"],
                    normals=scenario["normals"],
                    separations=scenario["separations"],
                    forces=scenario["forces"],
                )

                contact_sensor.eval(contacts)

                self.assertIsNotNone(contact_sensor.net_force)
                self.assertEqual(contact_sensor.net_force.shape, contact_sensor.shape)

                self.assertTrue(contact_sensor.net_force.dtype == wp.vec3)

                net_forces = contact_sensor.net_force.numpy()

                assert_np_equal(net_forces[0, 2], scenario["force_A_vs_B"])
                assert_np_equal(net_forces[1, 1], scenario["force_B_vs_A"])
                assert_np_equal(net_forces[0, 0], scenario["force_A_vs_All"])
                assert_np_equal(net_forces[1, 0], scenario["force_B_vs_All"])


class TestSensorContactMuJoCo(unittest.TestCase):
    """End-to-end tests for contact sensors using MuJoCo solver."""

    def test_stacking_scenario(self):
        """Test contact forces with b stacked on a on base."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.default_shape_cfg.density = 1000.0

        builder.add_shape_box(body=-1, hx=1.0, hy=1.0, hz=0.25, key="base")
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.8), wp.quat_identity()), key="a")
        builder.add_shape_box(body_a, hx=0.15, hy=0.15, hz=0.25)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 1.15), wp.quat_identity()), key="b")
        builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.05)

        model = builder.finalize()
        mass_a, mass_b = 45.0, 4.0  # kg (from density * volume)

        try:
            solver = SolverMuJoCo(model, njmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo not available: {e}")

        sensor = SensorContact(model, sensing_obj_bodies=["a", "b"])
        contacts = newton.Contacts(0, 0)

        # Simulate 2s
        state_in, state_out, control = model.state(), model.state(), model.control()
        for _ in range(240 * 2):
            solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
            state_in, state_out = state_out, state_in
        populate_contacts(contacts, solver)
        sensor.eval(contacts)

        forces = sensor.net_force.numpy()
        g = 9.81
        self.assertAlmostEqual(forces[0, 0, 2], mass_a * g, delta=mass_a * g * 0.01)
        self.assertAlmostEqual(forces[1, 0, 2], mass_b * g, delta=mass_b * g * 0.01)

    def test_parallel_scenario(self):
        """Test contact forces with a, b, c side-by-side on base."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.default_shape_cfg.density = 1000.0

        builder.add_shape_box(body=-1, hx=2.0, hy=2.0, hz=0.25, key="base")
        body_a = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0, 0.8), wp.quat_identity()), key="a")
        builder.add_shape_box(body_a, hx=0.15, hy=0.15, hz=0.25)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.6), wp.quat_identity()), key="b")
        builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.05)
        body_c = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0, 0.8), wp.quat_identity()), key="c")
        builder.add_shape_box(body_c, hx=0.1, hy=0.1, hz=0.25)

        model = builder.finalize()
        mass_a, mass_b, mass_c = 45.0, 4.0, 20.0  # kg

        try:
            solver = SolverMuJoCo(model, njmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo not available: {e}")

        sensor_abc = SensorContact(model, sensing_obj_bodies=["a", "b", "c"])
        sensor_base = SensorContact(model, sensing_obj_shapes=["base"])
        contacts = newton.Contacts(0, 0)

        # Simulate 2s
        state_in, state_out, control = model.state(), model.state(), model.control()
        for _ in range(240 * 2):
            solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
            state_in, state_out = state_out, state_in
        populate_contacts(contacts, solver)
        sensor_abc.eval(contacts)
        sensor_base.eval(contacts)

        forces = sensor_abc.net_force.numpy()
        g = 9.81
        self.assertAlmostEqual(forces[0, 0, 2], mass_a * g, delta=mass_a * g * 0.01)
        self.assertAlmostEqual(forces[1, 0, 2], mass_b * g, delta=mass_b * g * 0.01)
        self.assertAlmostEqual(forces[2, 0, 2], mass_c * g, delta=mass_c * g * 0.01)

        total_weight = (mass_a + mass_b + mass_c) * g
        self.assertAlmostEqual(sensor_base.net_force.numpy()[0, 0, 2], -total_weight, delta=total_weight * 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
