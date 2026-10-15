# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Focused geometry, POD and integration tests for the single-cap example."""

import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.basic._reduced_elastic_cap import build_cap_rom
from newton.examples.basic.example_basic_reduced_elastic_cup_straw import (
    Example,
    cylinder_sdf,
    drive_straw,
    straw_pose,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestStrawMotion(unittest.TestCase):
    def test_cylinder_distance_is_rotation_invariant(self):
        a = np.pi / 4
        rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        translation = np.array([0.02, -0.04, 0.15])
        points = np.array([[0, 0, 0], [0.003, 0, 0], [0.005, 0, 0], [0, 0, 0.082]])
        pose = np.concatenate([translation, [0, np.sin(a / 2), 0, np.cos(a / 2)]])
        np.testing.assert_allclose(
            cylinder_sdf(points @ rotation.T + translation, pose, 0.004, 0.08),
            [-0.004, -0.001, 0.001, 0.002],
            atol=1e-8,
        )

    def test_stir_has_constant_tilt_and_returns_to_start(self):
        cfg = Example.Config(stir=True)

        def pose(t):
            return np.array(
                straw_pose(
                    float(t),
                    cfg.lid_z,
                    cfg.straw_half_height,
                    0.0,
                    True,
                    float(np.deg2rad(cfg.tilt_degrees)),
                    cfg.stir_radius,
                )
            )

        for t in np.linspace(0, 9.5, 40):
            value = pose(float(t))
            axis = np.array(wp.quat_rotate(wp.quat(*value[3:]), wp.vec3(0, 0, 1)))
            self.assertAlmostEqual(float(axis[2]), np.cos(np.deg2rad(cfg.tilt_degrees)), places=6)
        np.testing.assert_allclose(pose(0), pose(9.5), atol=1e-7)
        np.testing.assert_allclose(pose(1.8), pose(7.0), atol=1e-7)

    def test_angular_velocity_matches_prescribed_rotation(self):
        cfg = Example.Config(stir=True)
        dt, t = 1 / 240, 4.1
        pose0, pose1 = wp.zeros(1, dtype=wp.transform), wp.zeros(1, dtype=wp.transform)
        vel0, vel1 = wp.zeros(1, dtype=wp.spatial_vector), wp.zeros(1, dtype=wp.spatial_vector)
        wp.launch(
            drive_straw,
            dim=1,
            inputs=[
                wp.array([t], dtype=float),
                dt,
                0,
                cfg.lid_z,
                cfg.straw_half_height,
                0.0,
                True,
                float(np.deg2rad(cfg.tilt_degrees)),
                cfg.stir_radius,
                pose0,
                vel0,
                pose1,
                vel1,
            ],
        )
        now = pose0.numpy()[0]
        nxt = np.array(
            straw_pose(
                t + dt,
                cfg.lid_z,
                cfg.straw_half_height,
                0.0,
                True,
                float(np.deg2rad(cfg.tilt_degrees)),
                cfg.stir_radius,
            )
        )
        velocity = vel0.numpy()[0]
        np.testing.assert_allclose(now[:3] + dt * velocity[:3], nxt[:3], atol=1e-7)
        angular = velocity[3:]
        speed = float(np.linalg.norm(angular))
        self.assertGreater(speed, 0.1)
        predicted = wp.quat_from_axis_angle(wp.vec3(*(angular / speed)), speed * dt) * wp.quat(*now[3:])
        np.testing.assert_allclose(np.array(predicted), nxt[3:], atol=1e-6)


class TestCapBasis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_cap_rom(Example.Config())

    def test_connected_closed_skin(self):
        vertices, faces, *_ = self.data
        neighbours = [set() for _ in vertices]
        edges = Counter()
        for a, b, c in faces:
            for u, v in ((a, b), (b, c), (c, a)):
                neighbours[u].add(v)
                neighbours[v].add(u)
                edges[tuple(sorted((u, v)))] += 1
        self.assertTrue(all(count == 2 for count in edges.values()))
        seen = set()
        todo = [0]
        while todo:
            vertex = todo.pop()
            if vertex not in seen:
                seen.add(vertex)
                todo.extend(neighbours[vertex] - seen)
        self.assertEqual(len(seen), len(vertices))

    def test_projected_mass_and_positive_stiffness(self):
        _, _, basis, _, _, _, _, _, _ = self.data
        phi = np.asarray(basis.sample_phi).transpose(0, 2, 1).reshape(-1, 4)
        mass = np.repeat(basis.sample_mass, 3)
        np.testing.assert_allclose(phi.T @ (mass[:, None] * phi), np.diag(basis.mode_mass), atol=1e-12, rtol=2e-6)
        self.assertTrue(np.all(np.asarray(basis.mode_stiffness) > 0))
        self.assertTrue(np.all(np.asarray(basis.mode_damping) > 0))

    def test_fixed_annulus_and_movable_four_tongues(self):
        _, _, basis, _, mid, _, _, sector, depth = self.data
        phi = np.asarray(basis.sample_phi)[: len(mid)]
        np.testing.assert_allclose(phi[(sector < 0) | (depth == 0)], 0, atol=1e-12)
        for flap in range(4):
            self.assertGreater(np.linalg.norm(phi[(sector == flap) & (depth > 0.98)]), 0.9)

    def test_snapshot_fit_is_bounded(self):
        audit = self.data[3]
        self.assertLess(audit["snapshot_max_vertex_error_m"], 0.0015)
        self.assertLess(audit["bending_energy_relative_rms_fit_error"], 0.04)


def test_stirring_requires_contact(test, device):
    with wp.ScopedDevice(device):
        example = Example(None, SimpleNamespace(test=True), config=Example.Config(contact_enabled=False))
        for _ in range(570):
            example.step()
        example.test_final()
        test.assertGreater(example.summary()["max_penetration_m"], 0.002)
        test.assertLess(example.summary()["max_deformation_m"], 1e-5)
        # The same trajectory must fail the contact-driven opening checks.
        example.cfg.contact_enabled = True
        with test.assertRaises(AssertionError):
            example.test_final()


def test_metrics_are_opt_in(test, device):
    with wp.ScopedDevice(device):
        example = Example(None)

        def unexpected_metrics():
            raise AssertionError("Normal stepping must not collect CPU diagnostics")

        example.update_metrics = unexpected_metrics
        example.step()
        test.assertEqual(example.records, [])
        test.assertEqual(example.model.elastic_body_count, 1)
        test.assertEqual(example.basis.mode_count, 4)
        test.assertEqual(example.graph is not None, wp.get_device(device).is_cuda)


def test_short_validation_has_clear_error(test, device):
    with wp.ScopedDevice(device):
        example = Example(None, SimpleNamespace(test=True))
        example.step()
        with test.assertRaisesRegex(AssertionError, "570 frames"):
            example.test_final()


class TestCapDynamics(unittest.TestCase):
    pass


for device in get_test_devices(mode="basic"):
    add_function_test(
        TestCapDynamics, "test_stirring_requires_contact", test_stirring_requires_contact, devices=[device]
    )
    add_function_test(TestCapDynamics, "test_metrics_are_opt_in", test_metrics_are_opt_in, devices=[device])
    add_function_test(
        TestCapDynamics,
        "test_short_validation_has_clear_error",
        test_short_validation_has_clear_error,
        devices=[device],
    )


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
