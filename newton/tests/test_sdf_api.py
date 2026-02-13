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

"""
Tests for SDF API.

Verifies SDF configuration via:
- Builder defaults (enable_sdf/disable_sdf)
- USD attribute parsing with defaults
- Per-shape overrides (override_shape_sdf)
"""

import tempfile
import unittest
from pathlib import Path

import warp as wp
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

import newton
from newton._src.utils.import_usd import parse_usd
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


def create_cube_mesh_prim(stage, prim_path: str):
    """Create a cube mesh prim at the given path."""
    mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)
    UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
    
    mesh_prim.CreatePointsAttr([
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
    ])
    mesh_prim.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh_prim.CreateFaceVertexIndicesAttr([
        0, 1, 2, 3,  # bottom
        4, 5, 6, 7,  # top
        0, 1, 5, 4,  # front
        2, 3, 7, 6,  # back
        0, 3, 7, 4,  # left
        1, 2, 6, 5,  # right
    ])
    return mesh_prim


class TestSDFAPI(unittest.TestCase):
    """Test SDF API: builder defaults, USD attributes, and per-shape overrides."""

    def test_defaults_enable_disable(self, device=None):
        """Test that enable_sdf() and disable_sdf() work correctly."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")
        
        builder = newton.ModelBuilder()
        
        # Initially SDF should be disabled
        self.assertIsNone(builder.default_shape_cfg.sdf_max_resolution)
        
        # Enable SDF with default parameters
        builder.enable_sdf()
        self.assertEqual(builder.default_shape_cfg.sdf_max_resolution, 64)
        self.assertEqual(builder.default_shape_cfg.sdf_narrow_band_range, (-0.01, 0.01))
        self.assertEqual(builder.default_shape_cfg.contact_margin, 0.01)
        
        # Enable SDF with custom parameters
        builder2 = newton.ModelBuilder()
        builder2.enable_sdf(
            max_resolution=128,
            narrow_band_range=(-0.02, 0.02),
            contact_margin=0.02,
        )
        self.assertEqual(builder2.default_shape_cfg.sdf_max_resolution, 128)
        self.assertEqual(builder2.default_shape_cfg.sdf_narrow_band_range, (-0.02, 0.02))
        self.assertEqual(builder2.default_shape_cfg.contact_margin, 0.02)
        
        # Test disable_sdf()
        builder2.disable_sdf()
        self.assertIsNone(builder2.default_shape_cfg.sdf_max_resolution)
        self.assertIsNone(builder2.default_shape_cfg.sdf_target_voxel_size)
        
        # Test validation errors
        builder3 = newton.ModelBuilder()
        with self.assertRaises(ValueError):
            builder3.enable_sdf(max_resolution=65)  # Not divisible by 8
        
        with self.assertRaises(ValueError):
            builder3.enable_sdf(narrow_band_range=(0.01, -0.01))  # inner >= outer

    def test_usd_with_defaults(self, device=None):
        """Test USD attributes with some shapes having USD-defined SDF and others using defaults."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            
            # Create body1 with USD-defined SDF attributes
            body1_prim = stage.DefinePrim("/World/Body1", "Xform")
            UsdPhysics.RigidBodyAPI.Apply(body1_prim)
            mesh1_prim = create_cube_mesh_prim(stage, "/World/Body1/CollisionMesh")
            prim1 = mesh1_prim.GetPrim()
            prim1.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(128)
            prim1.CreateAttribute("newton:sdfNarrowBandInner", Sdf.ValueTypeNames.Float, custom=True).Set(-0.02)
            prim1.CreateAttribute("newton:sdfNarrowBandOuter", Sdf.ValueTypeNames.Float, custom=True).Set(0.02)
            
            # Create body2 WITHOUT USD SDF attributes (will use builder default)
            body2_prim = stage.DefinePrim("/World/Body2", "Xform")
            UsdPhysics.RigidBodyAPI.Apply(body2_prim)
            create_cube_mesh_prim(stage, "/World/Body2/CollisionMesh")
            
            # Create body3 with different USD SDF resolution
            body3_prim = stage.DefinePrim("/World/Body3", "Xform")
            UsdPhysics.RigidBodyAPI.Apply(body3_prim)
            mesh3_prim = create_cube_mesh_prim(stage, "/World/Body3/CollisionMesh")
            prim3 = mesh3_prim.GetPrim()
            prim3.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int, custom=True).Set(256)
            
            stage.Save()
            
            # Import with builder default (different from USD values)
            builder = newton.ModelBuilder()
            builder.enable_sdf(max_resolution=64, narrow_band_range=(-0.01, 0.01))
            
            parse_usd(builder, str(usd_path))
            model = builder.finalize(device=device)
            
            if not hasattr(model, 'shape_sdf_data') or model.shape_sdf_data is None:
                self.skipTest("Model does not have SDF data")
            
            shape_types = model.shape_type.numpy()
            shape_sdf_data = model.shape_sdf_data.numpy()
            
            # Verify all mesh shapes have SDF enabled
            sdf_enabled_count = 0
            for i in range(model.shape_count):
                if shape_types[i] == newton.GeoType.MESH:
                    sdf_ptr = shape_sdf_data[i]["sparse_sdf_ptr"]
                    if sdf_ptr > 0:
                        sdf_enabled_count += 1
            
            self.assertGreater(sdf_enabled_count, 0, "SDF should be enabled for mesh shapes")
            self.assertEqual(sdf_enabled_count, 3, "All three mesh shapes should have SDF enabled")

    def test_override_shape_sdf(self, device=None):
        """Test that override_shape_sdf() can modify SDF parameters after USD import."""
        if device is None or not wp.get_device(device).is_cuda:
            self.skipTest("SDF tests require CUDA device")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            usd_path = Path(tmpdir) / "test_sdf.usda"
            stage = Usd.Stage.CreateNew(str(usd_path))
            
            # Create two bodies with mesh collision
            body1_prim = stage.DefinePrim("/World/Body1", "Xform")
            UsdPhysics.RigidBodyAPI.Apply(body1_prim)
            mesh1_prim = create_cube_mesh_prim(stage, "/World/Body1/CollisionMesh")
            
            body2_prim = stage.DefinePrim("/World/Body2", "Xform")
            UsdPhysics.RigidBodyAPI.Apply(body2_prim)
            mesh2_prim = create_cube_mesh_prim(stage, "/World/Body2/CollisionMesh")
            
            stage.Save()
            
            # Import USD
            builder = newton.ModelBuilder()
            builder.enable_sdf(max_resolution=64)
            parse_usd(builder, str(usd_path))
            
            # Find shape indices (assuming first two mesh shapes)
            shape_indices = []
            for i in range(builder.shape_count):
                if builder.shape_type[i] == newton.GeoType.MESH:
                    shape_indices.append(i)
            
            self.assertGreaterEqual(len(shape_indices), 2, "Should have at least 2 mesh shapes")
            shape1_idx = shape_indices[0]
            shape2_idx = shape_indices[1]
            
            # Verify initial state
            self.assertEqual(builder.shape_sdf_max_resolution[shape1_idx], 64)
            self.assertEqual(builder.shape_sdf_max_resolution[shape2_idx], 64)
            
            # Override shape1 with specific SDF parameters
            builder.override_shape_sdf(shape1_idx, sdf_max_resolution=128, sdf_narrow_band_range=(-0.02, 0.02))
            self.assertEqual(builder.shape_sdf_max_resolution[shape1_idx], 128)
            self.assertEqual(builder.shape_sdf_narrow_band_range[shape1_idx], (-0.02, 0.02))
            
            # Verify shape2 is unchanged
            self.assertEqual(builder.shape_sdf_max_resolution[shape2_idx], 64)
            
            # Override shape2 using different values
            builder.override_shape_sdf(shape2_idx, sdf_max_resolution=256, sdf_narrow_band_range=(-0.01, 0.01))
            self.assertEqual(builder.shape_sdf_max_resolution[shape2_idx], 256)
            self.assertEqual(builder.shape_sdf_narrow_band_range[shape2_idx], (-0.01, 0.01))
            
            # Partial override - only change resolution
            builder.override_shape_sdf(shape1_idx, sdf_max_resolution=96)
            self.assertEqual(builder.shape_sdf_max_resolution[shape1_idx], 96)
            self.assertEqual(builder.shape_sdf_narrow_band_range[shape1_idx], (-0.02, 0.02))  # Unchanged
            
            # Test hydroelastic override
            builder.override_shape_sdf(shape1_idx, is_hydroelastic=True, k_hydro=1.0e11)
            self.assertTrue(builder.shape_flags[shape1_idx] & newton.ShapeFlags.HYDROELASTIC)
            self.assertEqual(builder.shape_material_k_hydro[shape1_idx], 1.0e11)
            
            # Test disabling hydroelastic
            builder.override_shape_sdf(shape1_idx, is_hydroelastic=False)
            self.assertFalse(builder.shape_flags[shape1_idx] & newton.ShapeFlags.HYDROELASTIC)
            
            # Finalize and verify SDF is actually enabled
            model = builder.finalize(device=device)
            
            if not hasattr(model, 'shape_sdf_data') or model.shape_sdf_data is None:
                self.skipTest("Model does not have SDF data")
            
            shape_sdf_data = model.shape_sdf_data.numpy()
            sdf_ptr1 = shape_sdf_data[shape1_idx]["sparse_sdf_ptr"]
            sdf_ptr2 = shape_sdf_data[shape2_idx]["sparse_sdf_ptr"]
            
            self.assertGreater(sdf_ptr1, 0, "Shape1 should have SDF enabled")
            self.assertGreater(sdf_ptr2, 0, "Shape2 should have SDF enabled")
            
            # Test error cases
            with self.assertRaises(IndexError):
                builder.override_shape_sdf(9999, sdf_max_resolution=64)
            
            with self.assertRaises(ValueError):
                builder.override_shape_sdf(shape1_idx, sdf_max_resolution=65)  # Not divisible by 8
            
            with self.assertRaises(ValueError):
                builder.override_shape_sdf(shape1_idx, sdf_narrow_band_range=(0.01, -0.01))  # inner >= outer


# Add CUDA tests
devices = get_selected_cuda_test_devices()
add_function_test(TestSDFAPI, "test_defaults_enable_disable", TestSDFAPI.test_defaults_enable_disable, devices=devices)
add_function_test(TestSDFAPI, "test_usd_with_defaults", TestSDFAPI.test_usd_with_defaults, devices=devices)
add_function_test(TestSDFAPI, "test_override_shape_sdf", TestSDFAPI.test_override_shape_sdf, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)

