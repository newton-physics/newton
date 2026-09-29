# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the contact bridge without motion files or a calibrated shoe download."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import newton
from newton.tests.test_digital_shoe import _tiny_artifact
from projects.digital_shoe.runtime import FoundationConfig
from projects.impedance_instron.cartesian.shoe import Shoe


class TestCartesianShoe(unittest.TestCase):
    """Keep contact a force response rather than prescribed foot motion."""

    def setUp(self):
        """Write a small standalone two-column shoe and fixture."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        raw = _tiny_artifact()
        raw["visual_meshes"] = {
            "fullfoot_last": {
                "vertices_m": [[-0.02, -0.01, 0.02], [0.02, -0.01, 0.02], [0.02, 0.01, 0.02], [-0.02, 0.01, 0.02]],
                "triangles": [[0, 1, 2], [0, 2, 3]],
            }
        }
        raw["instron_fixtures"] = {
            "fullfoot_last": {
                "carrier_anchor_m": [[-0.01, 0.0, 0.02], [0.01, 0.0, 0.02]],
                "foam_free_top_m": [0.02, 0.02],
                "foam_bottom_m": [0.0, 0.0],
                "rest_length_m": [0.02, 0.02],
                "area_m2": [0.0001, 0.0001],
                "neighbors": [[1, -1, -1, -1], [0, -1, -1, -1]],
                "spacing_m": 0.01,
            }
        }
        path = Path(self.directory.name) / "shoe.json"
        path.write_text(json.dumps(raw))
        self.shoe = Shoe(path, [0, 0, 0.1], 0.0)

    def test_flight_and_compression(self):
        """Produce zero flight force and an upward response to compression."""
        force, compression = self.shoe.apply([0, 0.2], [0, 0], 0, 0, 0.0001)
        np.testing.assert_array_equal(force, 0)
        self.assertEqual(compression, 0)
        force, compression = self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)
        self.assertGreater(force[1], 0)
        self.assertAlmostEqual(force[2], 0, places=6)
        self.assertAlmostEqual(compression, 0.001, places=6)

    def test_elastic_coulomb_is_the_leg_default_and_scales_with_column_geometry(self):
        """Use area-scaled equilibrium shear stiffness without a Maxwell branch."""
        self.assertEqual(self.shoe.foundation.config.friction_model, "elastic_coulomb")
        self.assertEqual(int(self.shoe.foundation.friction_solver.settings.numpy()[0, 0]), 9)
        self.assertEqual(FoundationConfig().friction_model, "elastic_coulomb")
        cfg = FoundationConfig(friction_model="column_maxwell")
        self.assertEqual(cfg.friction_model, "column_maxwell")
        material = self.shoe.shoe.material
        g_eq = material.equilibrium_shear_modulus_pa
        area = np.asarray(self.shoe.shoe.column_bed.area_m2)
        rest = np.asarray(self.shoe.shoe.column_bed.rest_length_m)
        kt = g_eq * area / rest
        dt, speed = 0.001, 0.01
        expected = -np.sum(kt * dt * speed)
        actual, _compression = self.shoe.apply([0, 0.099], [speed, 0], 0, 0, dt)
        np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=1e-7)

        explicit_maxwell = Shoe(self.shoe.artifact_path, [0, 0, 0.1], 0.0, friction_model="maxwell")
        prior_force, _compression = explicit_maxwell.apply([0, 0.099], [speed, 0], 0, 0, dt)
        self.assertGreater(abs(prior_force[0]), 10.0 * abs(actual[0]))

        explicit_column_maxwell = Shoe(self.shoe.artifact_path, [0, 0, 0.1], 0.0, friction_model="column_maxwell")
        self.assertEqual(explicit_column_maxwell.foundation.config.friction_model, "column_maxwell")

    def test_pitch_sign_and_virtual_power(self):
        """Map Newton wrench signs into mathematical planar angular power."""
        force, _ = self.shoe.apply([0, 0.1], [0.2, -0.1], 0.1, 0.3, 0.0001)
        self.assertAlmostEqual(force[2], -float(self.shoe.state.body_f.numpy()[0, 4]))
        expected = np.dot(force, [0.2, -0.1, 0.3])
        reported = float(self.shoe.foundation.contact_power.numpy()[0])
        self.assertAlmostEqual(expected, reported, places=5)

    def test_translation_invariance(self):
        """Preserve normal loading under a stationary-ground horizontal translation."""
        a, _ = self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)
        self.shoe.foundation.reset()
        b, _ = self.shoe.apply([3, 0.099], [0, 0], 0, 0, 0.0001)
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_reject_ambiguous_footprint(self):
        """Reject duplicate fixture coordinates instead of silently dropping support."""
        path = self.shoe.artifact_path
        raw = json.loads(path.read_text())
        fixture = raw["instron_fixtures"]["fullfoot_last"]
        fixture["carrier_anchor_m"][1] = fixture["carrier_anchor_m"][0]
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "unique planar"):
            Shoe(path, [0, 0, 0.1], 0)

    def test_last_mesh_shares_carrier_without_second_contact(self):
        """Attach the actual last mesh without changing mass or adding a second contact path."""
        model = self.shoe.model
        self.assertEqual(model.shape_count, 1)
        self.assertEqual(int(model.shape_body.numpy()[0]), 0)
        flags = model.shape_flags.numpy()[0]
        self.assertFalse(flags & newton.ShapeFlags.COLLIDE_SHAPES)
        self.assertFalse(flags & newton.ShapeFlags.COLLIDE_PARTICLES)
        self.assertAlmostEqual(float(model.body_mass.numpy()[0]), 1.0)
        geometry = self.shoe.geometry()
        raw = json.loads(self.shoe.artifact_path.read_text())
        vertices = np.asarray(raw["visual_meshes"]["fullfoot_last"]["vertices_m"])
        np.testing.assert_allclose(geometry["last_vertices_local_m"] + self.shoe.mount_m, vertices)

    def test_rendering_preserves_previous_foundation_wrench(self):
        """Preserve the old mesh-free carrier's forces while adding mesh and spring replay."""
        with patch.object(newton.ModelBuilder, "add_shape_mesh", return_value=-1):
            previous = Shoe(self.shoe.artifact_path, [0, 0, 0.1], 0)
        self.assertEqual(previous.model.shape_count, 0)
        for height, pitch in ((0.2, 0.0), (0.099, 0.0), (0.099, 0.05), (0.098, -0.03), (0.2, 0.0)):
            expected, _ = previous.apply([0.01, height], [0.02, -0.1], pitch, 0.2, 0.0001)
            actual, _ = self.shoe.apply([0.01, height], [0.02, -0.1], pitch, 0.2, 0.0001)
            self.shoe.snapshot()
            np.testing.assert_array_equal(actual, expected)

    def test_distributed_pressure_and_moment(self):
        """Recover the ankle wrench from distributed ground forces rather than a point load."""
        wrench, _ = self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)
        snapshot = self.shoe.snapshot()
        self.assertTrue(np.all(snapshot["pressure_pa"] > 0))
        wrench, _ = self.shoe.apply([0, 0.099], [0.1, 0], 0.05, 0.1, 0.0001)
        snapshot = self.shoe.snapshot()
        self.assertNotAlmostEqual(float(snapshot["pressure_pa"][0]), float(snapshot["pressure_pa"][1]))
        force = self.shoe.foundation.ground_force.numpy().astype(float)
        point = self.shoe.foundation.contact_point.numpy().astype(float)
        total = force.sum(axis=0)
        moment = np.cross(point - [0, 0, 0.099], force).sum(axis=0)
        np.testing.assert_allclose(wrench, [total[0], total[2], -moment[1]], atol=1e-6)
        np.testing.assert_allclose(snapshot["pressure_pa"] * self.shoe.shoe.column_bed.area_m2, force[:, 2], rtol=1e-6)

    def test_rigid_sites_and_passive_visualization_do_not_advance_state(self):
        """Rotate spring sites with the last while leaving material and friction history untouched."""
        self.shoe.apply([0.4, 0.3], [0, 0], 0.4, 0, 0.0001)
        fields = ("q_state", "peq_prev", "tangent_anchor", "compression", "z_free")
        before = {name: getattr(self.shoe.foundation, name).numpy().copy() for name in fields}
        snapshot = self.shoe.snapshot()
        geometry = self.shoe.geometry()
        local_top = geometry["anchor_local_m"].copy()
        local_top[:, 2] += geometry["rest_length_m"]
        c, s = np.cos(0.4), np.sin(0.4)
        rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        expected_top = local_top @ rotation.T + np.array([0.4, 0, 0.3])
        np.testing.assert_allclose(snapshot["top_m"], expected_top, atol=1e-7)
        self.shoe.snapshot()
        for name in fields:
            np.testing.assert_array_equal(before[name], getattr(self.shoe.foundation, name).numpy())
        self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)
        np.testing.assert_array_equal(snapshot["compression_m"], 0)

    def test_keep_coupled_passive_margin(self):
        """Retain outer foam columns without rigidly attaching their tops to the last."""
        path = self.shoe.artifact_path
        raw = json.loads(path.read_text())
        bed = raw["column_bed"]
        bed["anchor_bottom_m"].append([0.03, 0, 0])
        bed["rest_length_m"].append(0.02)
        bed["area_m2"].append(0.0001)
        bed["neighbors"] = [[1, -1, -1, -1], [0, 2, -1, -1], [1, -1, -1, -1]]
        path.write_text(json.dumps(raw))
        shoe = Shoe(path, [0, 0, 0.1], 0)
        self.assertEqual(shoe.foundation.free_column_count, 1)
        np.testing.assert_array_equal(shoe.geometry()["driven"], [1, 1, 0])
        shoe.apply([0, 0.098], [0, 0], 0, 0, 0.0001)
        snapshot = shoe.snapshot()
        self.assertEqual(snapshot["pressure_pa"].shape, (3,))
        self.assertLessEqual(snapshot["compression_m"][2], 0.002 + 1e-7)
        np.testing.assert_allclose(snapshot["top_m"][:2, 2], 0.018, atol=1e-7)
        self.assertGreaterEqual(snapshot["top_m"][2, 2], 0.018 - 1e-7)

    def test_fixed_attachment_offsets_preserve_rest_geometry(self):
        """Keep fixture-to-foam offsets rigid without changing calibrated spring rest lengths."""
        path = self.shoe.artifact_path
        raw = json.loads(path.read_text())
        for point in raw["instron_fixtures"]["fullfoot_last"]["carrier_anchor_m"]:
            point[2] += 0.005
        for vertex in raw["visual_meshes"]["fullfoot_last"]["vertices_m"]:
            vertex[2] += 0.005
        path.write_text(json.dumps(raw))
        shoe = Shoe(path, [0, 0, 0.1], 0)
        geometry = shoe.geometry()
        np.testing.assert_array_equal(geometry["rest_length_m"], self.shoe.geometry()["rest_length_m"])
        np.testing.assert_array_equal(geometry["anchor_local_m"], self.shoe.geometry()["anchor_local_m"])
        nominal_top = geometry["anchor_local_m"].copy()
        nominal_top[:, 2] += geometry["rest_length_m"]
        np.testing.assert_allclose(geometry["attachment_local_m"] - nominal_top, [[0, 0, 0.005]] * 2, atol=1e-12)
        expected = self.shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)[0]
        actual = shoe.apply([0, 0.099], [0, 0], 0, 0, 0.0001)[0]
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
