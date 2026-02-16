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

"""Tests for the SDF API: USD attribute parsing and per-shape overrides via override_shape_sdf."""

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
    """Tests for USD SDF attribute parsing and override_shape_sdf."""

    def test_usd_sdf_attributes(self, device=None):
        """USD newton:sdf* attributes are parsed into per-shape SDF lists on the builder."""
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

            # Body2: USD-defined resolution=256, no narrow band
            _add_rigid_body(stage, "/World/Body2")
            m2 = _add_collision_mesh(stage, "/World/Body2/CollisionMesh")
            _set_sdf_attrs(m2.GetPrim(), resolution=256)

            stage.Save()

            builder = newton.ModelBuilder()
            result = parse_usd(builder, str(usd_path))
            psm = result["path_shape_map"]
            s1 = psm["/World/Body1/CollisionMesh"]
            s2 = psm["/World/Body2/CollisionMesh"]

            # Body1: USD values for both resolution and narrow band
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 128)
            self.assertAlmostEqual(builder.shape_sdf_narrow_band_range[s1][0], -0.02, places=5)
            self.assertAlmostEqual(builder.shape_sdf_narrow_band_range[s1][1], 0.02, places=5)

            # Body2: USD resolution only, narrow band from ShapeConfig default
            self.assertEqual(builder.shape_sdf_max_resolution[s2], 256)

            model = builder.finalize(device=device)
            self.assertEqual(_count_sdf_shapes(model), 2)

    def test_override_shape_sdf(self, device=None):
        """override_shape_sdf sets and modifies SDF properties on individual shapes."""
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
            result = parse_usd(builder, str(usd_path))
            psm = result["path_shape_map"]
            s1 = psm["/World/Body1/CollisionMesh"]
            s2 = psm["/World/Body2/CollisionMesh"]

            # Set SDF on shape1 via override
            builder.override_shape_sdf(s1, sdf_max_resolution=128, sdf_narrow_band_range=(-0.02, 0.02))
            self.assertEqual(builder.shape_sdf_max_resolution[s1], 128)
            self.assertEqual(builder.shape_sdf_narrow_band_range[s1], (-0.02, 0.02))

            # Set SDF on shape2 via override
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
add_function_test(TestSDFAPI, "test_usd_sdf_attributes", TestSDFAPI.test_usd_sdf_attributes, devices=devices)
add_function_test(TestSDFAPI, "test_override_shape_sdf", TestSDFAPI.test_override_shape_sdf, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
