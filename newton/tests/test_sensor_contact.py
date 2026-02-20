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
from newton.sensors import SensorContact
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import assert_np_equal


class MockModel:
    """Minimal mock model for testing SensorContact"""

    def __init__(self, device=None):
        self.device = device or wp.get_device()

    def request_contact_attributes(self, *args):
        pass


def create_contacts(device, pairs, naconmax, normals=None, forces=None):
    """Helper to create Contacts with specified contacts.

    The force spatial vectors are computed as (magnitude * normal, 0, 0, 0) to match
    the convention that contacts.force stores the force on body0 from body1.
    """
    contacts = newton.Contacts(naconmax, 0, device=device, requested_attributes={"force"})
    n_contacts = len(pairs)

    if normals is None:
        normals = [[0.0, 0.0, 1.0]] * n_contacts
    if forces is None:
        forces = [0.1] * n_contacts

    padding = naconmax - n_contacts
    shapes0 = [p[0] for p in pairs] + [-1] * padding
    shapes1 = [p[1] for p in pairs] + [-1] * padding
    normals_padded = normals + [[0.0, 0.0, 0.0]] * padding

    # Build spatial force vectors: linear force = magnitude * normal, angular = 0
    forces_spatial = [(f * n[0], f * n[1], f * n[2], 0.0, 0.0, 0.0) for f, n in zip(forces, normals, strict=True)] + [
        (0.0,) * 6
    ] * padding

    with wp.ScopedDevice(device):
        contacts.rigid_contact_shape0 = wp.array(shapes0, dtype=wp.int32)
        contacts.rigid_contact_shape1 = wp.array(shapes1, dtype=wp.int32)
        contacts.rigid_contact_normal = wp.array(normals_padded, dtype=wp.vec3f)
        contacts.rigid_contact_count = wp.array([n_contacts], dtype=wp.int32)
        contacts.force = wp.array(forces_spatial, dtype=wp.spatial_vector)

    return contacts


class TestSensorContact(unittest.TestCase):
    def test_net_force_aggregation(self):
        """Test net force aggregation across different contact subsets"""
        device = wp.get_device()

        # Define entities: Entity A = (0,1), Entity B = (2)
        entity_A = (0, 1)
        entity_B = (2,)

        model = MockModel()
        model.body_label = ["A", "B"]
        model.body_shapes = [entity_A, entity_B]

        contact_sensor = SensorContact(model, sensing_obj_bodies="*", counterpart_bodies="*")

        test_contacts = [
            {"pair": (0, 2), "normal": [0.0, 0.0, -1.0], "force": 1.0},
            {"pair": (1, 2), "normal": [-1.0, 0.0, 0.0], "force": 2.0},
            {"pair": (2, 1), "normal": [0.0, -1.0, 0.0], "force": 1.5},
            {"pair": (0, 3), "normal": [0.0, 0.0, 1.0], "force": 0.5},
        ]

        pairs = [contact["pair"] for contact in test_contacts]
        normals = [contact["normal"] for contact in test_contacts]
        forces = [contact["force"] for contact in test_contacts]

        test_scenarios = [
            {
                "name": "no_contacts",
                "pairs": [],
                "normals": [],
                "forces": [],
                "force_A_vs_B": (0.0, 0.0, 0.0),
                "force_B_vs_A": (0.0, 0.0, 0.0),
                "force_A_vs_All": (0.0, 0.0, 0.0),
                "force_B_vs_All": (0.0, 0.0, 0.0),
            },
            {
                "name": "only_contact_0",
                "pairs": pairs[:1],
                "normals": normals[:1],
                "forces": forces[:1],
                "force_A_vs_B": (0.0, 0.0, 1.0),
                "force_B_vs_A": (0.0, 0.0, -1.0),
                "force_A_vs_All": (0.0, 0.0, 1.0),
                "force_B_vs_All": (0.0, 0.0, -1.0),
            },
            {
                "name": "only 1",
                "pairs": pairs[1:2],
                "normals": normals[1:2],
                "forces": forces[1:2],
                "force_A_vs_B": (2.0, 0.0, 0.0),
                "force_B_vs_A": (-2.0, 0.0, 0.0),
                "force_A_vs_All": (2.0, 0.0, 0.0),
                "force_B_vs_All": (-2.0, 0.0, 0.0),
            },
            {
                "name": "only 2",
                "pairs": pairs[2:3],
                "normals": normals[2:3],
                "forces": forces[2:3],
                "force_A_vs_B": (0.0, -1.5, 0.0),
                "force_B_vs_A": (0.0, 1.5, 0.0),
                "force_A_vs_All": (0.0, -1.5, 0.0),
                "force_B_vs_All": (0.0, 1.5, 0.0),
            },
            {
                "name": "all_contacts",
                "pairs": pairs,
                "normals": normals,
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
                    normals=scenario["normals"],
                    forces=scenario["forces"],
                )

                contact_sensor.update(contacts)

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

        builder.add_shape_box(body=-1, hx=1.0, hy=1.0, hz=0.25, label="base")
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.8), wp.quat_identity()), label="a")
        builder.add_shape_box(body_a, hx=0.15, hy=0.15, hz=0.25)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 1.15), wp.quat_identity()), label="b")
        builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.05)

        model = builder.finalize()
        mass_a, mass_b = 45.0, 4.0  # kg (from density * volume)

        try:
            solver = SolverMuJoCo(model, njmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo not available: {e}")

        sensor = SensorContact(model, sensing_obj_bodies=["a", "b"])
        contacts = newton.Contacts(
            solver.get_max_contact_count(),
            0,
            device=model.device,
            requested_attributes=model.get_requested_contact_attributes(),
        )

        # Simulate 2s
        state_in, state_out, control = model.state(), model.state(), model.control()
        sim_dt = 1.0 / 240.0
        num_steps = 240 * 2

        device = model.device
        use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
        if use_cuda_graph:
            # warmup (2 steps to allocate both buffers)
            solver.step(state_in, state_out, control, None, sim_dt)
            solver.step(state_out, state_in, control, None, sim_dt)
            with wp.ScopedCapture(device) as capture:
                solver.step(state_in, state_out, control, None, sim_dt)
                solver.step(state_out, state_in, control, None, sim_dt)
            graph = capture.graph

        remaining = num_steps - (4 if use_cuda_graph else 0)
        for _ in range(remaining // 2 if use_cuda_graph else remaining):
            if use_cuda_graph:
                wp.capture_launch(graph)
            else:
                solver.step(state_in, state_out, control, None, sim_dt)
                state_in, state_out = state_out, state_in
        if use_cuda_graph and remaining % 2 == 1:
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in
        solver.update_contacts(contacts, state_in)
        sensor.update(contacts)

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

        builder.add_shape_box(body=-1, hx=2.0, hy=2.0, hz=0.25, label="base")
        body_a = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0, 0.8), wp.quat_identity()), label="a")
        builder.add_shape_box(body_a, hx=0.15, hy=0.15, hz=0.25)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.6), wp.quat_identity()), label="b")
        builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.05)
        body_c = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0, 0.8), wp.quat_identity()), label="c")
        builder.add_shape_box(body_c, hx=0.1, hy=0.1, hz=0.25)

        model = builder.finalize()
        mass_a, mass_b, mass_c = 45.0, 4.0, 20.0  # kg

        try:
            solver = SolverMuJoCo(model, njmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo not available: {e}")

        sensor_abc = SensorContact(model, sensing_obj_bodies=["a", "b", "c"])
        sensor_base = SensorContact(model, sensing_obj_shapes=["base"])
        contacts = newton.Contacts(
            solver.get_max_contact_count(),
            0,
            device=model.device,
            requested_attributes=model.get_requested_contact_attributes(),
        )

        # Simulate 2s
        state_in, state_out, control = model.state(), model.state(), model.control()
        sim_dt = 1.0 / 240.0
        num_steps = 240 * 2

        device = model.device
        use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
        if use_cuda_graph:
            # warmup (2 steps to allocate both buffers)
            solver.step(state_in, state_out, control, None, sim_dt)
            solver.step(state_out, state_in, control, None, sim_dt)
            with wp.ScopedCapture(device) as capture:
                solver.step(state_in, state_out, control, None, sim_dt)
                solver.step(state_out, state_in, control, None, sim_dt)
            graph = capture.graph

        remaining = num_steps - (4 if use_cuda_graph else 0)
        for _ in range(remaining // 2 if use_cuda_graph else remaining):
            if use_cuda_graph:
                wp.capture_launch(graph)
            else:
                solver.step(state_in, state_out, control, None, sim_dt)
                state_in, state_out = state_out, state_in
        if use_cuda_graph and remaining % 2 == 1:
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in
        solver.update_contacts(contacts, state_in)
        sensor_abc.update(contacts)
        sensor_base.update(contacts)

        forces = sensor_abc.net_force.numpy()
        g = 9.81
        self.assertAlmostEqual(forces[0, 0, 2], mass_a * g, delta=mass_a * g * 0.01)
        self.assertAlmostEqual(forces[1, 0, 2], mass_b * g, delta=mass_b * g * 0.01)
        self.assertAlmostEqual(forces[2, 0, 2], mass_c * g, delta=mass_c * g * 0.01)

        total_weight = (mass_a + mass_b + mass_c) * g
        self.assertAlmostEqual(sensor_base.net_force.numpy()[0, 0, 2], -total_weight, delta=total_weight * 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
