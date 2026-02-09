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
        """Test creating a Heightfield object and verifying its properties."""
        nrow, ncol = 10, 10
        elevation_data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32)
        size = (5.0, 5.0, 1.0, 0.0)

        hfield = Heightfield(
            data=elevation_data,
            nrow=nrow,
            ncol=ncol,
            size=size,
        )

        # Check properties
        self.assertEqual(hfield.nrow, nrow)
        self.assertEqual(hfield.ncol, ncol)
        self.assertEqual(hfield.size, size)
        self.assertEqual(hfield.data.dtype, np.float32)
        self.assertEqual(hfield.data.shape, (nrow, ncol))
        assert_np_equal(hfield.data, elevation_data, tol=1e-6)

    def test_heightfield_hash(self):
        """Test that heightfield hashing works for deduplication."""
        nrow, ncol = 5, 5
        data1 = np.zeros((nrow, ncol), dtype=np.float32)
        data2 = np.zeros((nrow, ncol), dtype=np.float32)
        data3 = np.ones((nrow, ncol), dtype=np.float32)

        hfield1 = Heightfield(data=data1, nrow=nrow, ncol=ncol)
        hfield2 = Heightfield(data=data2, nrow=nrow, ncol=ncol)
        hfield3 = Heightfield(data=data3, nrow=nrow, ncol=ncol)

        # Same data should produce same hash
        self.assertEqual(hash(hfield1), hash(hfield2))

        # Different data should produce different hash
        self.assertNotEqual(hash(hfield1), hash(hfield3))

    def test_add_shape_heightfield(self):
        """Test adding a heightfield shape via ModelBuilder."""
        builder = newton.ModelBuilder()

        nrow, ncol = 8, 8
        elevation_data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32)
        hfield = Heightfield(
            data=elevation_data,
            nrow=nrow,
            ncol=ncol,
            size=(4.0, 4.0, 0.5, 0.0),
        )

        shape_id = builder.add_shape_heightfield(
            heightfield=hfield,
        )

        # Verify shape was added
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

        # Find the heightfield shape
        hfield_shapes = [i for i in range(builder.shape_count) if builder.shape_type[i] == newton.GeoType.HFIELD]

        # Should have exactly one heightfield
        self.assertEqual(len(hfield_shapes), 1)

        shape_id = hfield_shapes[0]
        hfield = builder.shape_source[shape_id]

        # Verify heightfield properties from MJCF
        self.assertIsInstance(hfield, Heightfield)
        self.assertEqual(hfield.nrow, 10)
        self.assertEqual(hfield.ncol, 10)
        self.assertEqual(hfield.size, (5.0, 5.0, 1.0, 0.0))

        # Data should be all zeros (no file specified)
        assert_np_equal(
            hfield.data,
            np.zeros((10, 10)),
            tol=1e-6,
        )

    def test_mjcf_hfield_binary_file(self):
        """Test parsing MJCF with binary heightfield file."""
        nrow, ncol = 4, 6
        rng = np.random.default_rng(42)
        elevation = rng.random((nrow, ncol)).astype(np.float32)

        # Write MuJoCo binary format: int32 header + float32 data
        with tempfile.NamedTemporaryFile(
            suffix=".bin",
            delete=False,
        ) as f:
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
            builder.add_mjcf(
                mjcf,
                parse_meshes=True,
                path_resolver=resolver,
            )

            hfield_shapes = [i for i in range(builder.shape_count) if builder.shape_type[i] == newton.GeoType.HFIELD]
            self.assertEqual(len(hfield_shapes), 1)

            hfield = builder.shape_source[hfield_shapes[0]]
            self.assertEqual(hfield.nrow, nrow)
            self.assertEqual(hfield.ncol, ncol)
            self.assertEqual(hfield.size, (3.0, 2.0, 1.0, 0.0))
            assert_np_equal(hfield.data, elevation, tol=1e-6)
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
        expected = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
        assert_np_equal(hfield.data, expected, tol=1e-6)

    def test_solver_mujoco_hfield(self):
        """Test converting Newton model with heightfield to MuJoCo."""
        try:
            import mujoco  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("MuJoCo not installed")

        builder = newton.ModelBuilder()

        # Create a simple heightfield
        nrow, ncol = 5, 5
        elevation_data = np.zeros((nrow, ncol), dtype=np.float32)
        hfield = Heightfield(
            data=elevation_data,
            nrow=nrow,
            ncol=ncol,
            size=(2.0, 2.0, 0.5, 0.01),
        )

        # Add heightfield to world
        builder.add_shape_heightfield(
            heightfield=hfield,
        )

        # Add a sphere that will interact with the heightfield
        sphere_body = builder.add_body(
            xform=wp.transform(
                (0.0, 0.0, 1.0),
                wp.quat_identity(),
            ),
        )
        builder.add_shape_sphere(
            body=sphere_body,
            radius=0.1,
        )

        # Finalize model
        model = builder.finalize()

        # Create MuJoCo solver (this will convert heightfield to MuJoCo format)
        try:
            newton.solvers.SolverMuJoCo(model)
        except Exception as e:
            self.fail(f"Failed to create MuJoCo solver with heightfield: {e}")

    def test_heightfield_collision(self):
        """Test that a sphere doesn't fall through a heightfield."""
        try:
            import mujoco  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("MuJoCo not installed")

        builder = newton.ModelBuilder()

        # Flat heightfield at z=0
        nrow, ncol = 10, 10
        elevation = np.zeros((nrow, ncol), dtype=np.float32)
        hfield = Heightfield(
            data=elevation,
            nrow=nrow,
            ncol=ncol,
            size=(5.0, 5.0, 1.0, 0.01),
        )
        builder.add_shape_heightfield(heightfield=hfield)

        # Sphere starting above the heightfield
        sphere_radius = 0.1
        start_z = 0.5
        sphere_body = builder.add_body(
            xform=wp.transform(
                (0.0, 0.0, start_z),
                wp.quat_identity(),
            ),
        )
        builder.add_shape_sphere(body=sphere_body, radius=sphere_radius)

        model = builder.finalize()
        solver = newton.solvers.SolverMuJoCo(model)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        sim_dt = 1.0 / 240.0

        # Let sphere settle on heightfield
        for _ in range(500):
            solver.step(state_in, state_out, control, None, sim_dt)
            state_in, state_out = state_out, state_in

        final_z = float(state_in.body_q.numpy()[sphere_body, 2])

        # Sphere should rest on the surface, not fall through
        self.assertGreater(
            final_z,
            -sphere_radius,
            f"Sphere fell through heightfield: z={final_z:.4f}",
        )

    def test_heightfield_always_static(self):
        """Test that heightfields are always static (zero mass, zero inertia)."""
        nrow, ncol = 10, 10
        elevation_data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32)

        hfield = Heightfield(
            data=elevation_data,
            nrow=nrow,
            ncol=ncol,
            size=(5.0, 5.0, 1.0, 0.0),
        )

        self.assertEqual(hfield.mass, 0.0)
        self.assertFalse(hfield.has_inertia)

    def test_heightfield_radius_computation(self):
        """Test bounding sphere radius computation for heightfield."""
        from newton._src.geometry.utils import compute_shape_radius  # noqa: PLC0415

        nrow, ncol = 10, 10
        size = (4.0, 3.0, 2.0, 0.0)  # size_x, size_y, size_z, size_base
        elevation_data = np.zeros((nrow, ncol), dtype=np.float32)

        hfield = Heightfield(
            data=elevation_data,
            nrow=nrow,
            ncol=ncol,
            size=size,
        )

        scale = (1.0, 1.0, 1.0)
        radius = compute_shape_radius(newton.GeoType.HFIELD, scale, hfield)

        # Expected: sqrt((size_x/2)^2 + (size_y/2)^2 + ((size_z+size_base)/2)^2)
        expected_radius = np.sqrt((4.0 / 2) ** 2 + (3.0 / 2) ** 2 + ((2.0 + 0.0) / 2) ** 2)
        self.assertAlmostEqual(radius, expected_radius, places=5)

    def test_heightfield_finalize(self):
        """Test heightfield finalization to Warp array."""
        nrow, ncol = 5, 5
        elevation_data = np.random.default_rng(42).random((nrow, ncol)).astype(np.float32)

        hfield = Heightfield(
            data=elevation_data,
            nrow=nrow,
            ncol=ncol,
            size=(2.0, 2.0, 1.0, 0.0),
        )

        # Finalize should return a pointer
        ptr = hfield.finalize()
        self.assertIsInstance(ptr, int)
        self.assertGreater(ptr, 0)

        # Warp array should be created
        self.assertIsNotNone(hfield.warp_array)


if __name__ == "__main__":
    unittest.main(verbosity=2)
