# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import types
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.geometry import ContactAggregation
from newton._src.utils import is_graph_capture_allocation_enabled
from newton.sensors import SensorContact
from newton.solvers import SolverKamino, SolverMuJoCo
from newton.tests.unittest_utils import assert_np_equal
from newton.tests.utils import basics


def _make_two_world_model(device=None, include_ground=False):
    """Build a 2-world model with bodies A (world 0) and B (world 1).

    Each body owns one shape.  When *include_ground* is True, a global ground
    shape (body=-1) is appended.
    """
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_body(label="A")
    builder.add_shape_box(0, hx=0.1, hy=0.1, hz=0.1, label="s0")
    builder.end_world()
    builder.begin_world()
    builder.add_body(label="B")
    builder.add_shape_box(1, hx=0.1, hy=0.1, hz=0.1, label="s1")
    builder.end_world()
    if include_ground:
        builder.add_shape_box(body=-1, hx=0.1, hy=0.1, hz=0.1, label="ground")
    return builder.finalize(device=device)


def create_contacts(device, pairs, naconmax, normals=None, forces=None):
    """Helper to create Contacts with specified contacts.

    The force spatial vectors are computed as (magnitude * normal, 0, 0, 0) to match
    the convention that contacts.force stores the force on shape0 from shape1.
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

        # Body A owns shapes 0,1; body B owns shape 2; shape 3 is ground
        builder = newton.ModelBuilder()
        body_a = builder.add_body(label="A")
        builder.add_shape_box(body_a, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body_a, hx=0.1, hy=0.1, hz=0.1)
        body_b = builder.add_body(label="B")
        builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=-1, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)

        contact_sensor = SensorContact(model, sensing_bodies="*", counterpart_bodies="*")

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
                "force_on_A_from_B": (0.0, 0.0, 0.0),
                "force_on_B_from_A": (0.0, 0.0, 0.0),
                "force_on_A_from_all": (0.0, 0.0, 0.0),
                "force_on_B_from_all": (0.0, 0.0, 0.0),
            },
            {
                "name": "only_contact_0",
                "pairs": pairs[:1],
                "normals": normals[:1],
                "forces": forces[:1],
                "force_on_A_from_B": (0.0, 0.0, -1.0),
                "force_on_B_from_A": (0.0, 0.0, 1.0),
                "force_on_A_from_all": (0.0, 0.0, -1.0),
                "force_on_B_from_all": (0.0, 0.0, 1.0),
            },
            {
                "name": "only 1",
                "pairs": pairs[1:2],
                "normals": normals[1:2],
                "forces": forces[1:2],
                "force_on_A_from_B": (-2.0, 0.0, 0.0),
                "force_on_B_from_A": (2.0, 0.0, 0.0),
                "force_on_A_from_all": (-2.0, 0.0, 0.0),
                "force_on_B_from_all": (2.0, 0.0, 0.0),
            },
            {
                "name": "only 2",
                "pairs": pairs[2:3],
                "normals": normals[2:3],
                "forces": forces[2:3],
                "force_on_A_from_B": (0.0, 1.5, 0.0),
                "force_on_B_from_A": (0.0, -1.5, 0.0),
                "force_on_A_from_all": (0.0, 1.5, 0.0),
                "force_on_B_from_all": (0.0, -1.5, 0.0),
            },
            {
                "name": "all_contacts",
                "pairs": pairs,
                "normals": normals,
                "forces": forces,
                "force_on_A_from_B": (-2.0, 1.5, -1.0),
                "force_on_B_from_A": (2.0, -1.5, 1.0),
                "force_on_A_from_all": (-2.0, 1.5, -0.5),
                "force_on_B_from_all": (2.0, -1.5, 1.0),
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

                contact_sensor.update(None, contacts)

                self.assertIsNotNone(contact_sensor.force_matrix)
                self.assertIsNotNone(contact_sensor.total_force)

                net_forces = contact_sensor.force_matrix.numpy()
                total_forces = contact_sensor.total_force.numpy()

                assert_np_equal(net_forces[0, 1], scenario["force_on_A_from_B"])
                assert_np_equal(net_forces[1, 0], scenario["force_on_B_from_A"])
                assert_np_equal(total_forces[0], scenario["force_on_A_from_all"])
                assert_np_equal(total_forces[1], scenario["force_on_B_from_all"])

    def test_sensing_transforms(self):
        """Test that sensing object transforms are computed correctly."""
        device = wp.get_device()

        builder = newton.ModelBuilder()
        builder.add_body(label="A")
        builder.add_shape_box(0, hx=0.1, hy=0.1, hz=0.1)
        builder.add_body(label="B")
        builder.add_shape_box(1, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)

        sensor = SensorContact(model, sensing_bodies="*")

        body_pos_a = wp.vec3(1.0, 2.0, 3.0)
        body_pos_b = wp.vec3(4.0, 5.0, 6.0)
        body_q = wp.array(
            [wp.transform(body_pos_a, wp.quat_identity()), wp.transform(body_pos_b, wp.quat_identity())],
            dtype=wp.transform,
            device=device,
        )
        # lightweight stand-in for State
        state = types.SimpleNamespace(body_q=body_q)

        contacts = create_contacts(device, [], naconmax=1)
        sensor.update(state, contacts)

        transforms = sensor.sensing_transforms.numpy()
        assert_np_equal(transforms[0][:3], [1.0, 2.0, 3.0])
        assert_np_equal(transforms[1][:3], [4.0, 5.0, 6.0])

    def test_sensing_transforms_shapes(self):
        """Test transforms for shape-type sensing objects, including ground shapes."""
        device = wp.get_device()

        shape0_xform = (wp.vec3(0.5, 0.25, 0.125), wp.quat_identity())
        shape1_xform = (wp.vec3(10.0, 20.0, 30.0), wp.quat_identity())
        builder = newton.ModelBuilder()
        builder.add_body(label="A")
        builder.add_shape_box(0, xform=shape0_xform, hx=0.1, hy=0.1, hz=0.1, label="s0")
        builder.add_shape_box(body=-1, xform=shape1_xform, hx=0.1, hy=0.1, hz=0.1, label="ground")
        model = builder.finalize(device=device)

        sensor = SensorContact(model, sensing_shapes="*")

        body_pos = wp.vec3(1.0, 2.0, 3.0)
        body_q = wp.array(
            [wp.transform(body_pos, wp.quat_identity())],
            dtype=wp.transform,
            device=device,
        )
        state = types.SimpleNamespace(body_q=body_q)

        contacts = create_contacts(device, [], naconmax=1)
        sensor.update(state, contacts)

        transforms = sensor.sensing_transforms.numpy()
        # shape on a body: body_q * shape_transform -> (1+0.5, 2+0.25, 3+0.125)
        assert_np_equal(transforms[0][:3], [1.5, 2.25, 3.125])
        # ground shape (body index == -1): shape_transform only -> (10, 20, 30)
        assert_np_equal(transforms[1][:3], [10.0, 20.0, 30.0])

    def test_per_world_attributes(self):
        """sensing_indices and counterpart_indices are flat lists."""
        model = _make_two_world_model()

        sensor = SensorContact(model, sensing_bodies="*")

        self.assertEqual(sensor.sensing_indices, [0, 1])
        self.assertEqual(len(sensor.counterpart_indices), 2)
        # No explicit counterparts — each row has an empty counterpart list
        self.assertEqual(sensor.counterpart_indices[0], [])
        self.assertEqual(sensor.counterpart_indices[1], [])

    def test_multi_world_no_cross_world_pairs(self):
        """Per-world construction produces no cross-world counterpart columns."""
        model = _make_two_world_model(include_ground=True)

        sensor = SensorContact(model, sensing_bodies="*", counterpart_shapes="*")

        counterpart_col = sensor._counterpart_shape_to_col.numpy()
        # Ground (shape 2, global) should have a counterpart column
        self.assertGreaterEqual(counterpart_col[2], 0, "Ground shape should be a counterpart")
        # Shape 0 (world 0) and shape 1 (world 1) should both be counterparts
        self.assertGreaterEqual(counterpart_col[0], 0, "Shape 0 should be a counterpart")
        self.assertGreaterEqual(counterpart_col[1], 0, "Shape 1 should be a counterpart")
        # Per-world counterparts reuse the same column (different worlds, no cross-world contacts)
        self.assertEqual(
            counterpart_col[0],
            counterpart_col[1],
            "Per-world counterparts should share column indices",
        )

    def test_multi_world_total_force(self):
        """Total force accumulates correctly with per-world pair tables."""
        device = wp.get_device()
        model = _make_two_world_model(device=device)

        sensor = SensorContact(model, sensing_bodies="*")

        contacts = create_contacts(device, [(0, 1)], naconmax=4, forces=[3.0])
        sensor.update(None, contacts)

        self.assertIsNone(sensor.force_matrix)
        total = sensor.total_force.numpy()
        np.testing.assert_allclose(total[0], [0, 0, 3.0], atol=1e-5)
        np.testing.assert_allclose(total[1], [0, 0, -3.0], atol=1e-5)

    def test_global_sensing_object_raises(self):
        """Global entities as sensing objects raise ValueError."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        builder.add_body(label="A")
        builder.add_shape_box(0, hx=0.1, hy=0.1, hz=0.1, label="s0")
        builder.end_world()
        builder.begin_world()
        builder.end_world()
        builder.add_shape_box(body=-1, hx=0.1, hy=0.1, hz=0.1, label="ground")
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorContact(model, sensing_shapes="*")  # "*" matches ground too

    def test_order_preservation(self):
        """Sensing objects preserve caller's order for list[int] inputs."""
        model = _make_two_world_model()
        # Pass indices in reverse order: [1, 0]
        sensor = SensorContact(model, sensing_bodies=[1, 0])
        self.assertEqual(sensor.sensing_indices, [1, 0])

        contacts = create_contacts(model.device, [(0, 1)], naconmax=4, forces=[3.0])
        sensor.update(None, contacts)
        total = sensor.total_force.numpy()
        # Row 0 is body 1, row 1 is body 0
        np.testing.assert_allclose(total[0], [0, 0, -3.0], atol=1e-5)
        np.testing.assert_allclose(total[1], [0, 0, 3.0], atol=1e-5)

    def test_deprecated_sensing_object_aliases(self):
        """Deprecated sensing object aliases warn and return the new attributes."""
        model = _make_two_world_model()
        sensor = SensorContact(model, sensing_bodies=[1, 0])

        with self.assertWarnsRegex(DeprecationWarning, "sensing_indices"):
            legacy_indices = sensor.sensing_obj_idx
        self.assertIs(legacy_indices, sensor.sensing_indices)

        with self.assertWarnsRegex(DeprecationWarning, "sensing_type"):
            legacy_type = sensor.sensing_obj_type
        self.assertEqual(legacy_type, sensor.sensing_type)

        with self.assertWarnsRegex(DeprecationWarning, "sensing_transforms"):
            legacy_transforms = sensor.sensing_obj_transforms
        self.assertIs(legacy_transforms, sensor.sensing_transforms)

    def test_deprecated_sensing_constructor_aliases(self):
        """Deprecated sensing constructor keywords warn and map to the new keywords."""
        model = _make_two_world_model()

        with self.assertWarnsRegex(DeprecationWarning, "sensing_bodies"):
            body_sensor = SensorContact(model, sensing_obj_bodies=[1, 0])
        self.assertEqual(body_sensor.sensing_indices, [1, 0])

        with self.assertWarnsRegex(DeprecationWarning, "sensing_shapes"):
            shape_sensor = SensorContact(model, sensing_obj_shapes=["s0"])
        self.assertEqual(shape_sensor.sensing_indices, [0])

    def test_measure_total_false(self):
        """measure_total=False produces total_force=None and populates force_matrix."""
        model = _make_two_world_model(include_ground=True)
        sensor = SensorContact(model, sensing_bodies="*", counterpart_shapes="*", measure_total=False)
        self.assertIsNone(sensor.total_force)
        self.assertIsNotNone(sensor.position_matrix)
        self.assertEqual(sensor.position_matrix.shape, sensor.force_matrix.shape)

        contacts = create_contacts(model.device, [(0, 2)], naconmax=4, forces=[5.0])
        sensor.update(None, contacts)
        self.assertIsNotNone(sensor.force_matrix)
        net = sensor.force_matrix.numpy()
        ground_col = sensor.counterpart_indices[0].index(2)
        np.testing.assert_allclose(net[0, ground_col], [0, 0, 5.0], atol=1e-5)
        # no state passed, so positions are reset and never populated
        np.testing.assert_array_equal(sensor.position_matrix.numpy(), 0.0)

    def test_position_matrix(self):
        """Contact positions are force-weighted world-space midpoints grouped by counterpart."""
        device = wp.get_device()

        builder = newton.ModelBuilder()
        body_a = builder.add_body(label="A")
        shape_a = builder.add_shape_box(body_a, hx=0.1, hy=0.1, hz=0.1)
        body_b = builder.add_body(label="B")
        shape_b = builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.1)
        ground = builder.add_shape_box(body=-1, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)

        sensor = SensorContact(
            model,
            sensing_bodies="*",
            counterpart_shapes="*",
            measure_total=False,
        )
        self.assertIsNotNone(sensor.position_matrix)
        self.assertEqual(sensor.position_matrix.shape, sensor.force_matrix.shape)

        state = types.SimpleNamespace(
            body_q=wp.array(
                [
                    wp.transform(
                        wp.vec3(10.0, 0.0, 0.0),
                        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
                    ),
                    wp.transform(wp.vec3(0.0, 20.0, 0.0), wp.quat_identity()),
                ],
                dtype=wp.transform,
                device=device,
            )
        )

        # Contact 4 (b vs ground) is stored with the static shape as shape0 to exercise
        # the matched1-only path with the identity transform on point0.
        contacts = newton.Contacts(5, 0, device=device, requested_attributes={"force"})
        with wp.ScopedDevice(device):
            contacts.rigid_contact_shape0 = wp.array([shape_a, shape_a, shape_a, shape_b, ground], dtype=wp.int32)
            contacts.rigid_contact_shape1 = wp.array([shape_b, shape_b, ground, ground, shape_b], dtype=wp.int32)
            contacts.rigid_contact_point0 = wp.array(
                [
                    (1.0, 0.0, 0.0),
                    (0.0, 2.0, 0.0),
                    (0.0, 0.0, 2.0),
                    (1.0, 1.0, 1.0),
                    (4.0, 24.0, 0.0),
                ],
                dtype=wp.vec3,
            )
            contacts.rigid_contact_point1 = wp.array(
                [
                    (2.0, 0.0, 0.0),
                    (0.0, 4.0, 0.0),
                    (14.0, 6.0, 2.0),
                    (99.0, 98.0, 97.0),
                    (2.0, 0.0, 0.0),
                ],
                dtype=wp.vec3,
            )
            contacts.rigid_contact_offset0 = wp.array([(1.0, 0.0, 0.0)] * 5, dtype=wp.vec3)
            contacts.rigid_contact_offset1 = wp.array([(0.0, 2.0, 0.0)] * 5, dtype=wp.vec3)
            contacts.rigid_contact_normal = wp.array([(0.0, 0.0, 1.0)] * 5, dtype=wp.vec3)
            contacts.rigid_contact_count = wp.array([5], dtype=wp.int32)
            contacts.force = wp.array(
                [
                    (0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
                    (3.0, 4.0, 0.0, 0.0, 0.0, 0.0),
                    (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 9.0, 8.0, 7.0),
                    (-1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0),
                ],
                dtype=wp.spatial_vector,
            )

        sensor.update(state, contacts)

        row_a = sensor.sensing_indices.index(body_a)
        row_b = sensor.sensing_indices.index(body_b)
        col_a = sensor.counterpart_indices[row_b].index(shape_a)
        col_b = sensor.counterpart_indices[row_a].index(shape_b)
        col_ground_a = sensor.counterpart_indices[row_a].index(ground)
        col_ground_b = sensor.counterpart_indices[row_b].index(ground)
        positions = sensor.position_matrix.numpy()
        # surface points are the offset-shifted contact points in world space
        # A-B: weights 2 and 5 over surface midpoints (6, 12, 0) and (4, 13.5, 0)
        expected_ab = [32.0 / 7.0, 91.5 / 7.0, 0.0]
        np.testing.assert_allclose(positions[row_a, col_b], expected_ab, atol=1e-5)
        np.testing.assert_allclose(positions[row_b, col_a], expected_ab, atol=1e-5)
        # A-ground: single contact, midpoint of surface points (10, 1, 2) and (14, 8, 2)
        np.testing.assert_allclose(positions[row_a, col_ground_a], [12.0, 4.5, 2.0], atol=1e-5)
        # B-ground: single tiny-force contact, midpoint of surface points (5, 24, 0) and (2, 22, 0)
        np.testing.assert_allclose(positions[row_b, col_ground_b], [3.5, 23.0, 0.0], atol=1e-5)
        # entries with no contributing contacts (here: each body's own shape) are zero
        col_self_a = sensor.counterpart_indices[row_a].index(shape_a)
        np.testing.assert_array_equal(positions[row_a, col_self_a], 0.0)

        # a second update with the same inputs must give identical results
        # (catches stale weight accumulation across updates)
        sensor.update(state, contacts)
        np.testing.assert_allclose(sensor.position_matrix.numpy(), positions, rtol=1e-6)

        # updates without body transforms reset positions to zero so they cannot pair with the new forces
        forces_before = sensor.force_matrix.numpy().copy()
        changed_contacts = create_contacts(device, [(shape_a, shape_b)], naconmax=4, forces=[11.0])
        sensor.update(None, changed_contacts)
        np.testing.assert_array_equal(sensor.position_matrix.numpy(), 0.0)
        self.assertFalse(np.array_equal(sensor.force_matrix.numpy(), forces_before))

        sensor.update(state, contacts)  # repopulate positions
        sensor.update(types.SimpleNamespace(body_q=None), changed_contacts)
        np.testing.assert_array_equal(sensor.position_matrix.numpy(), 0.0)

        zero_force_contacts = create_contacts(device, [(shape_a, shape_b)], naconmax=4, forces=[0.0])
        sensor.update(state, zero_force_contacts)
        positions = sensor.position_matrix.numpy()
        np.testing.assert_array_equal(positions, 0.0)

    def test_duplicate_sensing_objects_raises(self):
        """Duplicate sensing object indices raise ValueError."""
        model = _make_two_world_model()
        with self.assertRaises(ValueError):
            SensorContact(model, sensing_bodies=[0, 0])

    def test_unmatched_pattern_raises(self):
        """Sensing or counterpart patterns that match nothing raise ValueError."""
        model = _make_two_world_model()
        with self.assertRaises(ValueError):
            SensorContact(model, sensing_bodies="nonexistent")
        with self.assertRaises(ValueError):
            SensorContact(model, sensing_shapes="nonexistent")
        with self.assertRaises(ValueError):
            SensorContact(model, sensing_bodies="*", counterpart_bodies="nonexistent")
        with self.assertRaises(ValueError):
            SensorContact(model, sensing_bodies="*", counterpart_shapes="nonexistent")

    def test_global_counterpart_in_all_worlds(self):
        """Global counterparts (e.g., ground) appear in every sensing object's counterpart list."""
        model = _make_two_world_model(include_ground=True)

        sensor = SensorContact(
            model,
            sensing_bodies="*",
            counterpart_shapes=["ground"],
            measure_total=False,
        )

        # Both sensing objects should have the ground as a counterpart
        for i in range(2):
            self.assertIn(2, sensor.counterpart_indices[i], f"Sensing row {i} missing ground counterpart")

    def test_friction_force_orthogonal_to_normal(self):
        """Friction force is orthogonal to the contact normal."""
        device = wp.get_device()

        builder = newton.ModelBuilder()
        body_a = builder.add_body(label="A")
        builder.add_shape_box(body_a, hx=0.1, hy=0.1, hz=0.1)
        body_b = builder.add_body(label="B")
        builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)

        sensor = SensorContact(model, sensing_bodies="*")

        # Force has normal component (z) and tangential component (x)
        # Normal is [0,0,1], force spatial vector is (3, 0, 5, 0, 0, 0)
        contacts = newton.Contacts(4, 0, device=device, requested_attributes={"force"})
        with wp.ScopedDevice(device):
            contacts.rigid_contact_shape0 = wp.array([0, -1, -1, -1], dtype=wp.int32)
            contacts.rigid_contact_shape1 = wp.array([1, -1, -1, -1], dtype=wp.int32)
            contacts.rigid_contact_normal = wp.array(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=wp.vec3,
            )
            contacts.rigid_contact_count = wp.array([1], dtype=wp.int32)
            contacts.force = wp.array(
                [(3.0, 0.0, 5.0, 0.0, 0.0, 0.0), (0.0,) * 6, (0.0,) * 6, (0.0,) * 6],
                dtype=wp.spatial_vector,
            )

        sensor.update(None, contacts)

        friction = sensor.total_force_friction.numpy()
        # Friction on A should be (3, 0, 0) — the tangential part
        np.testing.assert_allclose(friction[0], [3.0, 0.0, 0.0], atol=1e-5)
        # Friction on B should be (-3, 0, 0) — Newton's third law
        np.testing.assert_allclose(friction[1], [-3.0, 0.0, 0.0], atol=1e-5)
        # Verify orthogonality: dot(friction, normal) == 0
        normal = np.array([0.0, 0.0, 1.0])
        self.assertAlmostEqual(np.dot(friction[0], normal), 0.0, places=5)

    def test_friction_force_multi_contact(self):
        """Friction forces accumulate correctly across contacts with different normals."""
        device = wp.get_device()

        builder = newton.ModelBuilder()
        body_a = builder.add_body(label="A")
        builder.add_shape_box(body_a, hx=0.1, hy=0.1, hz=0.1)
        body_b = builder.add_body(label="B")
        builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(body=-1, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)

        sensor = SensorContact(model, sensing_bodies="*")

        # Contact 0: shape0=0(A), shape1=1(B), normal=[0,0,1], force=(1,2,3)
        #   normal_comp = dot((1,2,3),(0,0,1))*(0,0,1) = (0,0,3)
        #   friction = (1,2,3)-(0,0,3) = (1,2,0)
        #   A gets +(1,2,0), B gets -(1,2,0)
        #
        # Contact 1: shape0=1(B), shape1=2(ground), normal=[0,1,0], force=(4,5,6)
        #   normal_comp = dot((4,5,6),(0,1,0))*(0,1,0) = (0,5,0)
        #   friction = (4,5,6)-(0,5,0) = (4,0,6)
        #   B gets +(4,0,6), ground is not sensed
        #
        # Expected friction: A = (1,2,0), B = (-1,-2,0)+(4,0,6) = (3,-2,6)
        contacts = newton.Contacts(4, 0, device=device, requested_attributes={"force"})
        with wp.ScopedDevice(device):
            contacts.rigid_contact_shape0 = wp.array([0, 1, -1, -1], dtype=wp.int32)
            contacts.rigid_contact_shape1 = wp.array([1, 2, -1, -1], dtype=wp.int32)
            contacts.rigid_contact_normal = wp.array(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=wp.vec3,
            )
            contacts.rigid_contact_count = wp.array([2], dtype=wp.int32)
            contacts.force = wp.array(
                [
                    (1.0, 2.0, 3.0, 0.0, 0.0, 0.0),
                    (4.0, 5.0, 6.0, 0.0, 0.0, 0.0),
                    (0.0,) * 6,
                    (0.0,) * 6,
                ],
                dtype=wp.spatial_vector,
            )

        sensor.update(None, contacts)

        friction = sensor.total_force_friction.numpy()
        np.testing.assert_allclose(friction[0], [1.0, 2.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(friction[1], [3.0, -2.0, 6.0], atol=1e-5)

    def test_force_matrix_friction(self):
        """force_matrix_friction mirrors force_matrix structure."""
        device = wp.get_device()

        builder = newton.ModelBuilder()
        body_a = builder.add_body(label="A")
        builder.add_shape_box(body_a, hx=0.1, hy=0.1, hz=0.1)
        body_b = builder.add_body(label="B")
        builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)

        sensor = SensorContact(model, sensing_bodies="*", counterpart_bodies="*")

        # Force with tangential component: normal=[0,0,1], force=(2, 3, 7, 0, 0, 0)
        contacts = newton.Contacts(4, 0, device=device, requested_attributes={"force"})
        with wp.ScopedDevice(device):
            contacts.rigid_contact_shape0 = wp.array([0, -1, -1, -1], dtype=wp.int32)
            contacts.rigid_contact_shape1 = wp.array([1, -1, -1, -1], dtype=wp.int32)
            contacts.rigid_contact_normal = wp.array(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=wp.vec3,
            )
            contacts.rigid_contact_count = wp.array([1], dtype=wp.int32)
            contacts.force = wp.array(
                [(2.0, 3.0, 7.0, 0.0, 0.0, 0.0), (0.0,) * 6, (0.0,) * 6, (0.0,) * 6],
                dtype=wp.spatial_vector,
            )

        sensor.update(None, contacts)

        self.assertIsNotNone(sensor.force_matrix_friction)
        fmat = sensor.force_matrix_friction.numpy()
        self.assertEqual(fmat.shape, sensor.force_matrix.numpy().shape)
        # A's friction from B: tangential part = (2, 3, 0)
        np.testing.assert_allclose(fmat[0, 1], [2.0, 3.0, 0.0], atol=1e-5)
        # B's friction from A: (-2, -3, 0)
        np.testing.assert_allclose(fmat[1, 0], [-2.0, -3.0, 0.0], atol=1e-5)

    def test_friction_force_measure_total_false(self):
        """measure_total=False produces total_force_friction=None."""
        model = _make_two_world_model(include_ground=True)
        sensor = SensorContact(model, sensing_bodies="*", counterpart_shapes="*", measure_total=False)
        self.assertIsNone(sensor.total_force_friction)
        self.assertIsNotNone(sensor.force_matrix_friction)

    def test_friction_force_no_counterparts(self):
        """No counterparts produces force_matrix_friction=None."""
        model = _make_two_world_model()
        sensor = SensorContact(model, sensing_bodies="*")
        self.assertIsNone(sensor.force_matrix_friction)
        self.assertIsNone(sensor.position_matrix)
        self.assertIsNotNone(sensor.total_force_friction)

    def test_purely_normal_force_has_zero_friction(self):
        """Purely normal contact forces produce zero friction."""
        device = wp.get_device()
        model = _make_two_world_model(device=device)

        sensor = SensorContact(model, sensing_bodies="*")

        # create_contacts builds force = magnitude * normal, so purely normal
        contacts = create_contacts(device, [(0, 1)], naconmax=4, forces=[5.0])
        sensor.update(None, contacts)

        friction = sensor.total_force_friction.numpy()
        np.testing.assert_allclose(friction[0], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(friction[1], [0.0, 0.0, 0.0], atol=1e-5)

    def test_friction_force_diagonal_normal(self):
        """Friction decomposition is correct for a non-axis-aligned normal."""
        device = wp.get_device()

        builder = newton.ModelBuilder()
        body_a = builder.add_body(label="A")
        builder.add_shape_box(body_a, hx=0.1, hy=0.1, hz=0.1)
        body_b = builder.add_body(label="B")
        builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)

        sensor = SensorContact(model, sensing_bodies="*")

        # 30-degree incline normal: n = (0, -sin(30), cos(30)) = (0, -0.5, sqrt(3)/2)
        s30 = 0.5
        c30 = 3.0**0.5 / 2.0
        n = np.array([0.0, -s30, c30])
        force_vec = np.array([1.0, 2.0, 3.0])
        # Expected: normal_comp = dot(f,n)*n, friction = f - normal_comp
        d = float(np.dot(force_vec, n))  # 1*0 + 2*(-0.5) + 3*(sqrt(3)/2) = -1 + 2.598 = 1.598
        expected_friction = force_vec - d * n

        contacts = newton.Contacts(4, 0, device=device, requested_attributes={"force"})
        with wp.ScopedDevice(device):
            contacts.rigid_contact_shape0 = wp.array([0, -1, -1, -1], dtype=wp.int32)
            contacts.rigid_contact_shape1 = wp.array([1, -1, -1, -1], dtype=wp.int32)
            contacts.rigid_contact_normal = wp.array(
                [n.tolist(), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=wp.vec3,
            )
            contacts.rigid_contact_count = wp.array([1], dtype=wp.int32)
            contacts.force = wp.array(
                [(*force_vec.tolist(), 0.0, 0.0, 0.0), (0.0,) * 6, (0.0,) * 6, (0.0,) * 6],
                dtype=wp.spatial_vector,
            )

        sensor.update(None, contacts)

        friction = sensor.total_force_friction.numpy()
        np.testing.assert_allclose(friction[0], expected_friction, atol=1e-5)
        # Verify orthogonality
        self.assertAlmostEqual(np.dot(friction[0], n), 0.0, places=5)


class TestSensorContactMuJoCo(unittest.TestCase):
    """End-to-end tests for contact sensors using MuJoCo solver."""

    def test_stacking_scenario(self):
        """Test contact forces with b stacked on a on base."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 2000.0
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

        sensor = SensorContact(model, sensing_bodies=["a", "b"], counterpart_shapes="*")
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
        use_graph = is_graph_capture_allocation_enabled(device)
        if use_graph:
            # warmup (2 steps to allocate both buffers)
            solver.step(state_in, state_out, control, None, sim_dt)
            solver.step(state_out, state_in, control, None, sim_dt)
            with wp.ScopedCapture(device) as capture:
                solver.step(state_in, state_out, control, None, sim_dt)
                solver.step(state_out, state_in, control, None, sim_dt)
            graph = capture.graph

        avg_steps = 10  # average forces over last few steps for stability
        remaining = num_steps - avg_steps - (4 if use_graph else 0)
        for _ in range(remaining // 2 if use_graph else remaining):
            if use_graph:
                wp.capture_launch(graph)
            else:
                solver.step(state_in, state_out, control, None, sim_dt)
                state_in, state_out = state_out, state_in
        if use_graph and remaining % 2 == 1:
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in

        forces_acc = np.zeros((2, 3))
        for _ in range(avg_steps):
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in
            solver.update_contacts(contacts, state_in)
            sensor.update(state_in, contacts)
            forces_acc += sensor.total_force.numpy()
        total = forces_acc / avg_steps

        g = 9.81
        self.assertAlmostEqual(total[0, 2], mass_a * g, delta=mass_a * g * 0.01)
        self.assertAlmostEqual(total[1, 2], mass_b * g, delta=mass_b * g * 0.01)

        # Contact positions from solver-generated contacts pin the body-frame convention of
        # rigid_contact_point0/1: a sits centered on the base top (z=0.25), b on a's top (z=0.75).
        shape_base, shape_a, shape_b = 0, 1, 2
        row_a = sensor.sensing_indices.index(body_a)
        row_b = sensor.sensing_indices.index(body_b)
        positions = sensor.position_matrix.numpy()
        np.testing.assert_allclose(
            positions[row_a, sensor.counterpart_indices[row_a].index(shape_base)], [0.0, 0.0, 0.25], atol=0.05
        )
        np.testing.assert_allclose(
            positions[row_a, sensor.counterpart_indices[row_a].index(shape_b)], [0.0, 0.0, 0.75], atol=0.05
        )
        np.testing.assert_allclose(
            positions[row_b, sensor.counterpart_indices[row_b].index(shape_a)], [0.0, 0.0, 0.75], atol=0.05
        )
        # b never touches the base
        np.testing.assert_array_equal(positions[row_b, sensor.counterpart_indices[row_b].index(shape_base)], 0.0)

    def test_stacking_friction(self):
        """Friction forces are near zero for boxes at rest on a flat surface."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.default_shape_cfg.density = 100.0

        builder.add_shape_box(body=-1, hx=1.0, hy=1.0, hz=0.25, label="base")
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.8), wp.quat_identity()), label="a")
        builder.add_shape_box(body_a, hx=0.15, hy=0.15, hz=0.25)

        model = builder.finalize()
        mass_a = 4.5

        try:
            solver = SolverMuJoCo(model, njmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo not available: {e}")

        sensor = SensorContact(model, sensing_bodies=["a"])
        contacts = newton.Contacts(
            solver.get_max_contact_count(),
            0,
            device=model.device,
            requested_attributes=model.get_requested_contact_attributes(),
        )

        state_in, state_out, control = model.state(), model.state(), model.control()
        sim_dt = 1.0 / 240.0
        num_steps = 240 * 2
        avg_steps = 10  # average forces over last few steps for stability
        for _ in range(num_steps - avg_steps):
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in

        total_acc = np.zeros((1, 3))
        friction_acc = np.zeros((1, 3))
        for _ in range(avg_steps):
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in
            solver.update_contacts(contacts, state_in)
            sensor.update(state_in, contacts)
            total_acc += sensor.total_force.numpy()
            friction_acc += sensor.total_force_friction.numpy()
        total = total_acc / avg_steps
        friction = friction_acc / avg_steps

        g = 9.81
        # Normal force should match weight
        self.assertAlmostEqual(total[0, 2], mass_a * g, delta=mass_a * g * 0.02)
        # Friction should be near zero (box at rest on flat ground, no lateral forces)
        np.testing.assert_allclose(friction[0], [0.0, 0.0, 0.0], atol=mass_a * g * 0.02)

    def test_parallel_scenario(self):
        """Test contact forces with a, b, c side-by-side on base."""
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 1000.0
        builder.default_shape_cfg.density = 100.0

        builder.add_shape_box(body=-1, hx=2.0, hy=2.0, hz=0.25, label="base")
        body_a = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0, 0.8), wp.quat_identity()), label="a")
        builder.add_shape_box(body_a, hx=0.15, hy=0.15, hz=0.25)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.6), wp.quat_identity()), label="b")
        builder.add_shape_box(body_b, hx=0.1, hy=0.1, hz=0.05)
        body_c = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0, 0.8), wp.quat_identity()), label="c")
        builder.add_shape_box(body_c, hx=0.1, hy=0.1, hz=0.25)

        model = builder.finalize()
        mass_a, mass_b, mass_c = 4.5, 0.4, 2.0  # kg

        try:
            solver = SolverMuJoCo(model, njmax=200)
        except ImportError as e:
            self.skipTest(f"MuJoCo not available: {e}")

        sensor_abc = SensorContact(model, sensing_bodies=["a", "b", "c"])
        sensor_base = SensorContact(model, sensing_shapes=["base"])
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
        use_graph = is_graph_capture_allocation_enabled(device)
        if use_graph:
            # warmup (2 steps to allocate both buffers)
            solver.step(state_in, state_out, control, None, sim_dt)
            solver.step(state_out, state_in, control, None, sim_dt)
            with wp.ScopedCapture(device) as capture:
                solver.step(state_in, state_out, control, None, sim_dt)
                solver.step(state_out, state_in, control, None, sim_dt)
            graph = capture.graph

        avg_steps = 10  # average forces over last few steps for stability
        remaining = num_steps - avg_steps - (4 if use_graph else 0)
        for _ in range(remaining // 2 if use_graph else remaining):
            if use_graph:
                wp.capture_launch(graph)
            else:
                solver.step(state_in, state_out, control, None, sim_dt)
                state_in, state_out = state_out, state_in
        if use_graph and remaining % 2 == 1:
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in

        forces_acc = np.zeros((3, 3))
        base_acc = np.zeros((1, 3))
        for _ in range(avg_steps):
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in
            solver.update_contacts(contacts, state_in)
            sensor_abc.update(state_in, contacts)
            sensor_base.update(state_in, contacts)
            forces_acc += sensor_abc.total_force.numpy()
            base_acc += sensor_base.total_force.numpy()
        forces = forces_acc / avg_steps
        base_force = base_acc / avg_steps

        g = 9.81
        self.assertAlmostEqual(forces[0, 2], mass_a * g, delta=mass_a * g * 0.01)
        self.assertAlmostEqual(forces[1, 2], mass_b * g, delta=mass_b * g * 0.01)
        self.assertAlmostEqual(forces[2, 2], mass_c * g, delta=mass_c * g * 0.01)

        total_weight = (mass_a + mass_b + mass_c) * g
        self.assertAlmostEqual(base_force[0, 2], -total_weight, delta=total_weight * 0.01)


class TestSensorContactKamino(unittest.TestCase):
    """End-to-end contact-sensor tests using the Kamino solver.

    Regression coverage that the Kamino->Newton conversion populates ``Contacts.force`` so
    ``SensorContact.total_force`` matches ``ContactAggregation``.
    """

    def test_box_on_plane_total_force_matches_aggregation(self):
        # Build a single box resting on a static ground plane.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        basics.build_box_on_plane(builder=builder)
        model = builder.finalize(skip_validation_joints=True)

        # Use Kamino's internal collision detector (steps run with ``contacts=None``).
        config = SolverKamino.Config.from_model(model)
        config.use_collision_detector = True
        config.collision_detector.max_contacts = 200
        solver = SolverKamino(model=model, config=config)

        # SensorContact requests the ``force`` contact attribute, so allocate the
        # output ``Contacts`` buffer afterwards to pick it up.
        sensor = SensorContact(model, sensing_bodies=["box"])
        contacts = newton.Contacts(
            rigid_contact_max=200,
            soft_contact_max=0,
            device=model.device,
            requested_attributes=model.get_requested_contact_attributes(),
        )
        self.assertIsNotNone(contacts.force, "force attribute must be allocated for the sensor path")

        # Kamino's per-body aggregation reads the same internal contacts the conversion does.
        aggregation = ContactAggregation(solver._model_kamino, solver._contacts_kamino)

        state_in, state_out, control = model.state(), model.state(), model.control()
        sim_dt = 1.0 / 240.0

        # Settle the box on the plane.
        for _ in range(120):
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in

        # Average over a few steps for stability.
        avg_steps = 10
        sensor_acc = np.zeros((1, 3))
        agg_acc = np.zeros((1, 3))
        for _ in range(avg_steps):
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in
            solver.update_contacts(contacts, state_in)
            sensor.update(state_in, contacts)
            aggregation.compute(skip_if_no_contacts=False)
            sensor_acc += sensor.total_force.numpy()
            # The box is body 0 of world 0.
            agg_acc += aggregation.body_net_force.numpy()[0, 0]
        sensor_force = sensor_acc[0] / avg_steps
        agg_force = agg_acc[0] / avg_steps

        # Regression: the stock sensor must now report a non-zero, upward contact force.
        self.assertGreater(np.linalg.norm(sensor_force), 1e-6, "SensorContact reported zero force under Kamino")
        self.assertGreater(sensor_force[2], 0.0, "Normal contact force on the box should be upward")
        # Equivalence: it must match Kamino's per-body aggregation.
        np.testing.assert_allclose(sensor_force, agg_force, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
