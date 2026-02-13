# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Tests for the SDF API: enable/disable, USD parsing with fallback, and per-shape overrides."""

import tempfile
import unittest
from pathlib import Path

import warp as wp

import newton
from newton._src.utils.import_usd import parse_usd
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices

CUBE_POINTS = [
    (-0.5, -0.5, -0.5),
    (0.5, -0.5, -0.5),
    (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (-0.5, -0.5, 0.5),
    (0.5, -0.5, 0.5),
    (0.5, 0.5, 0.5),
    (-0.5, 0.5, 0.5),
]

CUBE_FACE_VERTEX_COUNTS = [4, 4, 4, 4, 4, 4]

CUBE_FACE_VERTEX_INDICES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    0,
    1,
    5,
    4,
    2,
    3,
    7,
    6,
    0,
    3,
    7,
    4,
    1,
    2,
    6,
    5,
]


def _add_rigid_body(stage, path: str):
    """Define an Xform prim with RigidBodyAPI at *path*."""
    from pxr import UsdPhysics

    prim = stage.DefinePrim(path, "Xform")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    return prim


def _add_collision_mesh(stage, path: str):
    """Create a cube collision-mesh prim and return it."""
    from pxr import UsdGeom, UsdPhysics

    mesh = UsdGeom.Mesh.Define(stage, path)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh.CreatePointsAttr(CUBE_POINTS)
    mesh.CreateFaceVertexCountsAttr(CUBE_FACE_VERTEX_COUNTS)
    mesh.CreateFaceVertexIndicesAttr(CUBE_FACE_VERTEX_INDICES)
    return mesh


def _set_sdf_attrs(prim, *, resolution=None, inner=None, outer=None):
    """Write optional newton:sdf* custom attributes on *prim*."""
    from pxr import Sdf

    if resolution is not None:
        prim.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(resolution)
    if inner is not None:
        prim.CreateAttribute("newton:sdfNarrowBandInner", Sdf.ValueTypeNames.Float, custom=True).Set(inner)
    if outer is not None:
        prim.CreateAttribute("newton:sdfNarrowBandOuter", Sdf.ValueTypeNames.Float, custom=True).Set(outer)


def _count_sdf_shapes(model):
    """Return the number of mesh shapes that have a valid SDF pointer."""
    if not hasattr(model, "shape_sdf_data") or model.shape_sdf_data is None:
        return 0
    types = model.shape_type.numpy()
    sdf = model.shape_sdf_data.numpy()
    return sum(1 for i in range(model.shape_count) if types[i] == newton.GeoType.MESH and sdf[i]["sparse_sdf_ptr"] > 0)


class TestSDFAPI(unittest.TestCase):
    """Tests for builder.enable_sdf / disable_sdf, USD attribute parsing, and override_shape_sdf."""

    def test_defaults_enable_disable(self, device=None):
        """enable_sdf sets defaults; disable_sdf clears them; invalid args raise ValueError."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        builder = newton.ModelBuilder()
        self.assertIsNone(builder.default_shape_cfg.sdf_max_resolution)

        builder.enable_sdf()
        self.assertEqual(builder.default_shape_cfg.sdf_max_resolution, 64)
        self.assertEqual(builder.default_shape_cfg.sdf_narrow_band_range, (-0.01, 0.01))
        self.assertEqual(builder.default_shape_cfg.contact_margin, 0.01)

        builder.enable_sdf(max_resolution=128, narrow_band_range=(-0.02, 0.02), contact_margin=0.02)
        self.assertEqual(builder.default_shape_cfg.sdf_max_resolution, 128)
        self.assertEqual(builder.default_shape_cfg.sdf_narrow_band_range, (-0.02, 0.02))
        self.assertEqual(builder.default_shape_cfg.contact_margin, 0.02)

        builder.disable_sdf()
        self.assertIsNone(builder.default_shape_cfg.sdf_max_resolution)
        self.assertIsNone(builder.default_shape_cfg.sdf_target_voxel_size)

        with self.assertRaises(ValueError):
            newton.ModelBuilder().enable_sdf(max_resolution=65)

        with self.assertRaises(ValueError):
            newton.ModelBuilder().enable_sdf(narrow_band_range=(0.01, -0.01))

    def test_usd_with_defaults(self, device=None):
        """USD attributes override builder defaults; shapes without attrs fall back to defaults."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Usd

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))

            # Body1: USD-defined resolution=128 and narrow band (-0.02, 0.02)
            _add_rigid_body(stage, "/World/Body1")
            m1 = _add_collision_mesh(stage, "/World/Body1/CollisionMesh")
            _set_sdf_attrs(m1.GetPrim(), resolution=128, inner=-0.02, outer=0.02)

            # Body2: no USD SDF attributes → should use builder defaults
            _add_rigid_body(stage, "/World/Body2")
            _add_collision_mesh(stage, "/World/Body2/CollisionMesh")

            # Body3: USD-defined resolution=256, no narrow band → defaults for narrow band
            _add_rigid_body(stage, "/World/Body3")
            m3 = _add_collision_mesh(stage, "/World/Body3/CollisionMesh")
            _set_sdf_attrs(m3.GetPrim(), resolution=256)

            stage.Save()

            builder = newton.ModelBuilder()
            builder.enable_sdf(max_resolution=64, narrow_band_range=(-0.01, 0.01))

            result = parse_usd(builder, str(usd_path))
            psm = result["path_shape_map"]
            s1 = psm["/World/Body1/CollisionMesh"]
            s2 = psm["/World/Body2/CollisionMesh"]
            s3 = psm["/World/Body3/CollisionMesh"]

            # Body1: USD values
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 128)
            self.assertAlmostEqual(builder.shape_sdf_narrow_band_range[s1][0], -0.02, places=5)
            self.assertAlmostEqual(builder.shape_sdf_narrow_band_range[s1][1], 0.02, places=5)

            # Body2: builder defaults
            self.assertEqual(builder.shape_sdf_max_resolution[s2], 64)
            self.assertAlmostEqual(builder.shape_sdf_narrow_band_range[s2][0], -0.01, places=5)
            self.assertAlmostEqual(builder.shape_sdf_narrow_band_range[s2][1], 0.01, places=5)

            # Body3: USD resolution, builder default narrow band
            self.assertEqual(builder.shape_sdf_max_resolution[s3], 256)
            self.assertAlmostEqual(builder.shape_sdf_narrow_band_range[s3][0], -0.01, places=5)
            self.assertAlmostEqual(builder.shape_sdf_narrow_band_range[s3][1], 0.01, places=5)

            model = builder.finalize(device=device)
            self.assertEqual(_count_sdf_shapes(model), 3)

    def test_override_shape_sdf(self, device=None):
        """override_shape_sdf modifies individual shapes after USD import."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")

        from pxr import Usd

        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))

            _add_rigid_body(stage, "/World/Body1")
            _add_collision_mesh(stage, "/World/Body1/CollisionMesh")

            _add_rigid_body(stage, "/World/Body2")
            _add_collision_mesh(stage, "/World/Body2/CollisionMesh")

            stage.Save()

            builder = newton.ModelBuilder()
            builder.enable_sdf(max_resolution=64)
            result = parse_usd(builder, str(usd_path))
            psm = result["path_shape_map"]
            s1 = psm["/World/Body1/CollisionMesh"]
            s2 = psm["/World/Body2/CollisionMesh"]

            # Both shapes start with the builder default
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 64)
            self.assertEqual(builder.shape_sdf_max_resolution[s2], 64)

            # Override shape1
            builder.override_shape_sdf(s1, sdf_max_resolution=128, sdf_narrow_band_range=(-0.02, 0.02))
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 128)
            self.assertEqual(builder.shape_sdf_narrow_band_range[s1], (-0.02, 0.02))
            self.assertEqual(builder.shape_sdf_max_resolution[s2], 64)  # unchanged

            # Override shape2
            builder.override_shape_sdf(s2, sdf_max_resolution=256, sdf_narrow_band_range=(-0.01, 0.01))
            self.assertEqual(builder.shape_sdf_max_resolution[s2], 256)
            self.assertEqual(builder.shape_sdf_narrow_band_range[s2], (-0.01, 0.01))

            # Partial override keeps other fields
            builder.override_shape_sdf(s1, sdf_max_resolution=96)
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 96)
            self.assertEqual(builder.shape_sdf_narrow_band_range[s1], (-0.02, 0.02))

            # Hydroelastic flag
            builder.override_shape_sdf(s1, is_hydroelastic=True, k_hydro=1.0e11)
            self.assertTrue(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)
            self.assertEqual(builder.shape_material_k_hydro[s1], 1.0e11)

            builder.override_shape_sdf(s1, is_hydroelastic=False)
            self.assertFalse(builder.shape_flags[s1] & newton.ShapeFlags.HYDROELASTIC)

            # Finalize and check SDF pointers
            model = builder.finalize(device=device)
            self.assertEqual(_count_sdf_shapes(model), 2)

            # Error cases
            with self.assertRaises(IndexError):
                builder.override_shape_sdf(9999, sdf_max_resolution=64)

            with self.assertRaises(ValueError):
                builder.override_shape_sdf(s1, sdf_max_resolution=65)

            with self.assertRaises(ValueError):
                builder.override_shape_sdf(s1, sdf_narrow_band_range=(0.01, -0.01))


devices = get_selected_cuda_test_devices()
add_function_test(TestSDFAPI, "test_defaults_enable_disable", TestSDFAPI.test_defaults_enable_disable, devices=devices)
add_function_test(TestSDFAPI, "test_usd_with_defaults", TestSDFAPI.test_usd_with_defaults, devices=devices)
add_function_test(TestSDFAPI, "test_override_shape_sdf", TestSDFAPI.test_override_shape_sdf, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
