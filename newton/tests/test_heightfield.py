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

import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton import Heightfield
from newton.tests.unittest_utils import assert_np_equal


class TestHeightfield(unittest.TestCase):
    """Test suite for heightfield support."""

    def test_heightfield_creation(self):
        """Test creating a Heightfield with auto-normalization."""
        nrow, ncol = 10, 10
        raw_data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32) * 5.0  # 0-5 meters

        hfield = Heightfield(data=raw_data, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0)

        self.assertEqual(hfield.nrow, nrow)
        self.assertEqual(hfield.ncol, ncol)
        self.assertEqual(hfield.hx, 5.0)
        self.assertEqual(hfield.hy, 5.0)
        self.assertEqual(hfield.data.dtype, np.float32)
        self.assertEqual(hfield.data.shape, (nrow, ncol))

        # Data should be normalized to [0, 1]
        self.assertAlmostEqual(float(hfield.data.min()), 0.0, places=5)
        self.assertAlmostEqual(float(hfield.data.max()), 1.0, places=5)

        # min_z/max_z should be auto-derived from raw data
        self.assertAlmostEqual(hfield.min_z, float(raw_data.min()), places=5)
        self.assertAlmostEqual(hfield.max_z, float(raw_data.max()), places=5)

    def test_heightfield_explicit_z_range(self):
        """Test creating a Heightfield with explicit min_z/max_z."""
        nrow, ncol = 5, 5
        data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32)

        hfield = Heightfield(data=data, nrow=nrow, ncol=ncol, hx=3.0, hy=3.0, min_z=-1.0, max_z=4.0)

        self.assertEqual(hfield.min_z, -1.0)
        self.assertEqual(hfield.max_z, 4.0)
        # Data still normalized
        self.assertAlmostEqual(float(hfield.data.min()), 0.0, places=5)
        self.assertAlmostEqual(float(hfield.data.max()), 1.0, places=5)

    def test_heightfield_flat(self):
        """Test that flat (constant) data produces zeros."""
        nrow, ncol = 5, 5
        flat_data = np.full((nrow, ncol), 3.0, dtype=np.float32)

        hfield = Heightfield(data=flat_data, nrow=nrow, ncol=ncol, hx=1.0, hy=1.0)

        assert_np_equal(hfield.data, np.zeros((nrow, ncol)), tol=1e-6)
        self.assertAlmostEqual(hfield.min_z, 3.0, places=5)
        self.assertAlmostEqual(hfield.max_z, 3.0, places=5)

    def test_heightfield_hash(self):
        """Test that heightfield hashing works for deduplication."""
        nrow, ncol = 5, 5
        data_a = np.array([[i + j for j in range(ncol)] for i in range(nrow)], dtype=np.float32)
        data_b = np.array([[i + j for j in range(ncol)] for i in range(nrow)], dtype=np.float32)
        data_c = np.array([[i * j for j in range(ncol)] for i in range(nrow)], dtype=np.float32)

        hfield1 = Heightfield(data=data_a, nrow=nrow, ncol=ncol, hx=1.0, hy=1.0)
        hfield2 = Heightfield(data=data_b, nrow=nrow, ncol=ncol, hx=1.0, hy=1.0)
        hfield3 = Heightfield(data=data_c, nrow=nrow, ncol=ncol, hx=1.0, hy=1.0)

        # Same data should produce same hash
        self.assertEqual(hash(hfield1), hash(hfield2))

        # Different data should produce different hash
        self.assertNotEqual(hash(hfield1), hash(hfield3))

    def test_add_shape_heightfield(self):
        """Test adding a heightfield shape via ModelBuilder."""
        builder = newton.ModelBuilder()

        nrow, ncol = 8, 8
        elevation_data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32)
        hfield = Heightfield(data=elevation_data, nrow=nrow, ncol=ncol, hx=4.0, hy=4.0)

        shape_id = builder.add_shape_heightfield(heightfield=hfield)

        self.assertGreaterEqual(shape_id, 0)
        self.assertEqual(builder.shape_count, 1)
        self.assertEqual(builder.shape_type[shape_id], newton.GeoType.HFIELD)
        self.assertIs(builder.shape_source[shape_id], hfield)

    def test_mjcf_hfield_parsing(self):
        """Test parsing MJCF file with hfield asset."""
        mjcf = """
        <mujoco model="test_heightfield">
          <compiler autolimits="true"/>
          <asset>
            <hfield name="terrain" nrow="10" ncol="10"
                    size="5 5 1 0"/>
          </asset>
          <worldbody>
            <geom type="hfield" hfield="terrain"/>
            <body pos="0 0 2">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, parse_meshes=True)

        hfield_shapes = [i for i in range(builder.shape_count) if builder.shape_type[i] == newton.GeoType.HFIELD]
        self.assertEqual(len(hfield_shapes), 1)

        hfield = builder.shape_source[hfield_shapes[0]]
        self.assertIsInstance(hfield, Heightfield)
        self.assertEqual(hfield.nrow, 10)
        self.assertEqual(hfield.ncol, 10)
        # MuJoCo size (5, 5, 1, 0) → hx=5, hy=5, min_z=0, max_z=1
        self.assertAlmostEqual(hfield.hx, 5.0)
        self.assertAlmostEqual(hfield.hy, 5.0)
        self.assertAlmostEqual(hfield.min_z, 0.0)
        self.assertAlmostEqual(hfield.max_z, 1.0)

        # Data should be all zeros (no file, no elevation → flat)
        assert_np_equal(hfield.data, np.zeros((10, 10)), tol=1e-6)

    def test_mjcf_hfield_binary_file(self):
        """Test parsing MJCF with binary heightfield file."""
        nrow, ncol = 4, 6
        rng = np.random.default_rng(42)
        elevation = rng.random((nrow, ncol)).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            tmp_path = f.name
            np.array([nrow, ncol], dtype=np.int32).tofile(f)
            elevation.tofile(f)

        def resolver(_base_dir, _file_path):
            return tmp_path

        mjcf = """
        <mujoco>
          <asset>
            <hfield name="terrain" nrow="4" ncol="6"
                    size="3 2 1 0" file="terrain.bin"/>
          </asset>
          <worldbody>
            <geom type="hfield" hfield="terrain"/>
          </worldbody>
        </mujoco>
        """

        try:
            builder = newton.ModelBuilder()
            builder.add_mjcf(mjcf, parse_meshes=True, path_resolver=resolver)

            hfield_shapes = [i for i in range(builder.shape_count) if builder.shape_type[i] == newton.GeoType.HFIELD]
            self.assertEqual(len(hfield_shapes), 1)

            hfield = builder.shape_source[hfield_shapes[0]]
            self.assertEqual(hfield.nrow, nrow)
            self.assertEqual(hfield.ncol, ncol)
            self.assertAlmostEqual(hfield.hx, 3.0)
            self.assertAlmostEqual(hfield.hy, 2.0)
            # Data is normalized — check shape and range
            self.assertAlmostEqual(float(hfield.data.min()), 0.0, places=4)
            self.assertAlmostEqual(float(hfield.data.max()), 1.0, places=4)
        finally:
            os.unlink(tmp_path)

    def test_mjcf_hfield_inline_elevation(self):
        """Test parsing MJCF with inline elevation attribute."""
        mjcf = """
        <mujoco>
          <asset>
            <hfield name="terrain" nrow="3" ncol="3"
                    size="2 2 1 0"
                    elevation="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"/>
          </asset>
          <worldbody>
            <geom type="hfield" hfield="terrain"/>
          </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, parse_meshes=True)

        hfield_shapes = [i for i in range(builder.shape_count) if builder.shape_type[i] == newton.GeoType.HFIELD]
        self.assertEqual(len(hfield_shapes), 1)

        hfield = builder.shape_source[hfield_shapes[0]]
        self.assertEqual(hfield.nrow, 3)
        self.assertEqual(hfield.ncol, 3)
        # Data is normalized from [0.1, 0.9] to [0, 1]
        self.assertAlmostEqual(float(hfield.data.min()), 0.0, places=5)
        self.assertAlmostEqual(float(hfield.data.max()), 1.0, places=5)
        self.assertAlmostEqual(hfield.min_z, -0.0)  # size_base=0 → min_z=0
        self.assertAlmostEqual(hfield.max_z, 1.0)  # size_z=1 → max_z=1

    def test_solver_mujoco_hfield(self):
        """Test converting Newton model with heightfield to MuJoCo."""
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("MuJoCo not installed")

        builder = newton.ModelBuilder()

        nrow, ncol = 5, 5
        elevation_data = np.zeros((nrow, ncol), dtype=np.float32)
        hfield = Heightfield(data=elevation_data, nrow=nrow, ncol=ncol, hx=2.0, hy=2.0, min_z=0.0, max_z=0.5)

        builder.add_shape_heightfield(heightfield=hfield)

        sphere_body = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_sphere(body=sphere_body, radius=0.1)

        model = builder.finalize()

        try:
            newton.solvers.SolverMuJoCo(model)
        except Exception as e:
            self.fail(f"Failed to create MuJoCo solver with heightfield: {e}")

    def test_heightfield_collision(self):
        """Test that a sphere doesn't fall through a heightfield."""
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("MuJoCo not installed")

        builder = newton.ModelBuilder()

        nrow, ncol = 10, 10
        elevation = np.zeros((nrow, ncol), dtype=np.float32)
        hfield = Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0, min_z=0.0, max_z=1.0)
        builder.add_shape_heightfield(heightfield=hfield)

        sphere_radius = 0.1
        start_z = 0.5
        sphere_body = builder.add_body(xform=wp.transform((0.0, 0.0, start_z), wp.quat_identity()))
        builder.add_shape_sphere(body=sphere_body, radius=sphere_radius)

        model = builder.finalize()
        solver = newton.solvers.SolverMuJoCo(model)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        sim_dt = 1.0 / 240.0

        for _ in range(500):
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in

        final_z = float(state_in.body_q.numpy()[sphere_body, 2])

        self.assertGreater(
            final_z,
            -sphere_radius,
            f"Sphere fell through heightfield: z={final_z:.4f}",
        )

    def test_heightfield_always_static(self):
        """Test that heightfields are always static (zero mass, zero inertia)."""
        nrow, ncol = 10, 10
        elevation_data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32)

        hfield = Heightfield(data=elevation_data, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0)

        self.assertEqual(hfield.mass, 0.0)
        self.assertFalse(hfield.has_inertia)

    def test_heightfield_radius_computation(self):
        """Test bounding sphere radius computation for heightfield."""
        from newton._src.geometry.utils import compute_shape_radius  # noqa: PLC0415

        nrow, ncol = 10, 10
        elevation_data = np.zeros((nrow, ncol), dtype=np.float32)

        hfield = Heightfield(data=elevation_data, nrow=nrow, ncol=ncol, hx=4.0, hy=3.0, min_z=0.0, max_z=2.0)

        scale = (1.0, 1.0, 1.0)
        radius = compute_shape_radius(newton.GeoType.HFIELD, scale, hfield)

        # Expected: sqrt(hx^2 + hy^2 + ((max_z - min_z)/2)^2)
        expected_radius = np.sqrt(4.0**2 + 3.0**2 + ((2.0 - 0.0) / 2) ** 2)
        self.assertAlmostEqual(radius, expected_radius, places=5)

    def test_heightfield_finalize(self):
        """Test heightfield finalization to Warp array."""
        nrow, ncol = 5, 5
        elevation_data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32)

        hfield = Heightfield(data=elevation_data, nrow=nrow, ncol=ncol, hx=2.0, hy=2.0)

        ptr = hfield.finalize()
        self.assertIsInstance(ptr, int)
        self.assertGreater(ptr, 0)
        self.assertIsNotNone(hfield.warp_array)

        # Finalized array should be 1D (flattened)
        self.assertEqual(len(hfield.warp_array.shape), 1)
        self.assertEqual(hfield.warp_array.shape[0], nrow * ncol)

    def test_heightfield_native_collision_flat(self):
        """Test native CollisionPipeline detects contact between sphere and flat heightfield."""
        builder = newton.ModelBuilder()

        # Flat heightfield at z=0
        nrow, ncol = 10, 10
        elevation = np.zeros((nrow, ncol), dtype=np.float32)
        hfield = Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0, min_z=0.0, max_z=1.0)
        builder.add_shape_heightfield(heightfield=hfield)

        # Sphere slightly above the heightfield surface
        sphere_body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()))
        builder.add_shape_sphere(body=sphere_body, radius=0.1)

        model = builder.finalize()
        state = model.state()

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        # Should detect at least one contact (sphere is within contact margin of heightfield)
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(contact_count, 0, "No contacts detected between sphere and heightfield")

    def test_heightfield_native_collision_no_contact(self):
        """Test that no contacts are generated when sphere is far above heightfield."""
        builder = newton.ModelBuilder()

        nrow, ncol = 10, 10
        elevation = np.zeros((nrow, ncol), dtype=np.float32)
        hfield = Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=5.0, hy=5.0, min_z=0.0, max_z=1.0)
        builder.add_shape_heightfield(heightfield=hfield)

        # Sphere far above the heightfield
        sphere_body = builder.add_body(xform=wp.transform((0.0, 0.0, 5.0), wp.quat_identity()))
        builder.add_shape_sphere(body=sphere_body, radius=0.1)

        model = builder.finalize()
        state = model.state()

        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(contact_count, 0, f"Unexpected contacts detected: {contact_count}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
