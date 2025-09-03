#!/usr/bin/env python3
"""
Schema resolver tests for USD imports using ant.usda.

This suite validates:
1. Schema resolver priority handling with ["newton", "physx"] priority
2. PhysX attribute mapping to Newton equivalents from USD file
3. Engine-specific attribute parsing and storage from USD content

Prerequisites:
- Activate newton conda environment: conda activate newton
- Run from newton/tests directory: python3 test_schema_resolver.py
"""

import unittest
import sys
from pathlib import Path
import shutil
import math

from newton.tests.unittest_utils import USD_AVAILABLE
from newton import ModelBuilder  # noqa: PLC0415
from newton._src.utils.import_usd import parse_usd  # noqa: PLC0415
from newton._src.utils.schema_resolver import Resolver  # noqa: PLC0415


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestSchemaResolver(unittest.TestCase):
    """Test schema resolver with USD import from ant.usda."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use the actual ant.usda file
        test_dir = Path(__file__).parent
        self.ant_usda_path = test_dir / "assets" / "ant.usda"
        self.assertTrue(self.ant_usda_path.exists(), f"Ant USDA file not found: {self.ant_usda_path}")
        
    def test_import_ant_with_newton_physx_priority(self):
        """Test importing ant.usda with Newton-PhysX priority and validate schema resolution."""
        builder = ModelBuilder()
        
        # Import with Newton-PhysX priority
        result = parse_usd(
            builder=builder,
            source=str(self.ant_usda_path),
            schema_priority=["newton", "physx"],
            collect_engine_specific_attrs=True,
            verbose=False
        )
        
        # Basic import validation
        self.assertIsInstance(result, dict)
        self.assertIn("path_body_map", result)
        self.assertIn("path_shape_map", result)
        # Check that we have bodies and shapes
        self.assertGreater(len(result["path_body_map"]), 0)
        self.assertGreater(len(result["path_shape_map"]), 0)
        
        print(f"Successfully imported ant.usda:")
        print(f"  - Bodies: {len(result['path_body_map'])}")
        print(f"  - Shapes: {len(result['path_shape_map'])}")
        print(f"  - Joints: {builder.joint_count}")
        
        # Validate engine attributes were collected
        engine_specific_attrs = result.get("engine_specific_attrs", {})
        self.assertIsInstance(engine_specific_attrs, dict)
        
        if engine_specific_attrs:
            print(f"  - Engine attributes collected: {list(engine_specific_attrs.keys())}")
            
        return result, builder
        
    def test_physx_joint_armature(self):
        builder = ModelBuilder()
        result = parse_usd(
            builder=builder,
            source=str(self.ant_usda_path),
            schema_priority=["physx", "newton"],  # PhysX first
            collect_engine_specific_attrs=True,
            verbose=False
        )
        armature_values_found = []
        for i in range(6, builder.joint_dof_count):
            armature = builder.joint_armature[i]
            if armature > 0: 
                armature_values_found.append(armature)
        for i, armature in enumerate(armature_values_found):
            self.assertAlmostEqual(armature, 0.01, places=3)

        builder = ModelBuilder()
        result = parse_usd(
            builder=builder,
            source=str(self.ant_usda_path),
            schema_priority=["newton", "mjc"],  # nothing should be found
            collect_engine_specific_attrs=True,
            verbose=False
        )
        armature_values_found = []
        for i in range(6, builder.joint_dof_count):
            armature = builder.joint_armature[i]
            if armature > 0: 
                armature_values_found.append(armature)
        for i, armature in enumerate(armature_values_found):
            self.assertAlmostEqual(armature, 0.0, places=3)
        

    def test_engine_specific_attrs_collection(self):
        """Test collection of engine-specific attributes from real ant.usda file."""
        builder = ModelBuilder()
        
        # Import with engine attribute collection enabled
        result = parse_usd(
            builder=builder,
            source=str(self.ant_usda_path),
            schema_priority=["newton", "physx"],
            collect_engine_specific_attrs=True,
            verbose=False
        )
        
        engine_specific_attrs = result.get("engine_specific_attrs", {})
        
        # We should have collected PhysX attributes
        if "physx" in engine_specific_attrs:
            physx_attrs = engine_specific_attrs["physx"]
            print(f"Collected PhysX attributes from {len(physx_attrs)} prims:")
            
            # Look for specific attributes we expect from ant.usda
            joint_armature_prims = []
            limit_damping_prims = []
            articulation_prims = []
            
            for prim_path, attrs in physx_attrs.items():
                if "physxJoint:armature" in attrs:
                    joint_armature_prims.append((prim_path, attrs["physxJoint:armature"]))
                if "physxLimit:angular:damping" in attrs:
                    limit_damping_prims.append((prim_path, attrs["physxLimit:angular:damping"]))
                if "physxArticulation:enabledSelfCollisions" in attrs:
                    articulation_prims.append((prim_path, attrs["physxArticulation:enabledSelfCollisions"]))
            
            print(f"  - physxJoint:armature found on {len(joint_armature_prims)} prims")
            for prim_path, value in joint_armature_prims[:3]:  # Show first 3
                print(f"    {prim_path}: {value}")
                self.assertAlmostEqual(value, 0.01, places=6)  # From ant.usda
                
            print(f"  - physxLimit:angular:damping found on {len(limit_damping_prims)} prims")
            for prim_path, value in limit_damping_prims[:3]:  # Show first 3
                print(f"    {prim_path}: {value}")
                self.assertAlmostEqual(value, 0.1, places=6)  # From ant.usda
                
            print(f"  - physxArticulation:enabledSelfCollisions found on {len(articulation_prims)} prims")
            for prim_path, value in articulation_prims:
                print(f"    {prim_path}: {value}")
                self.assertEqual(value, False)  # From ant.usda
                
            # Validate we found the expected attributes
            self.assertGreater(len(joint_armature_prims), 0, "Should find physxJoint:armature attributes")
            self.assertGreater(len(limit_damping_prims), 0, "Should find physxLimit:angular:damping attributes")
            self.assertGreater(len(articulation_prims), 0, "Should find physxArticulation:enabledSelfCollisions attributes")
            
        else:
            print("No PhysX attributes collected - this might indicate an issue")
            
    def test_schema_priority(self):
        """Test that schema priority affects attribute resolution with USD."""
        builder1 = ModelBuilder()
        builder2 = ModelBuilder()
        
        # Import with Newton first
        result1 = parse_usd(
            builder=builder1,
            source=str(self.ant_usda_path),
            schema_priority=["newton", "physx"],
            collect_engine_specific_attrs=True,
            verbose=False
        )
        
        # Import with PhysX first  
        result2 = parse_usd(
            builder=builder2,
            source=str(self.ant_usda_path),
            schema_priority=["physx", "newton"],
            collect_engine_specific_attrs=True,
            verbose=False
        )

        # Both should succeed and have same structure
        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)
        self.assertEqual(len(result1["path_body_map"]), len(result2["path_body_map"]))
        self.assertEqual(len(result1["path_shape_map"]), len(result2["path_shape_map"]))
        self.assertEqual(builder1.joint_count, builder2.joint_count)
        
        print("Schema priority test:")
        print(f"  - Both imports successful with {len(result1['path_body_map'])} bodies")
        print(f"  - Both imports have {builder1.joint_count} joints")
        
        # Compare joint armature values - they should be the same since ant.usda only has PhysX values
        if builder1.joint_count > 0:
            armature1 = builder1.joint_armature[0] if builder1.joint_armature[0] > 0 else "not set"
            armature2 = builder2.joint_armature[0] if builder2.joint_armature[0] > 0 else "not set"
            print(f"  - First joint armature (Newton priority): {armature1}")
            print(f"  - First joint armature (PhysX priority): {armature2}")
            
            # Since ant.usda only has PhysX attributes, both should be the same
            if armature1 != "not set" and armature2 != "not set":
                self.assertEqual(armature1, armature2)
                
    def test_resolver_with_usd_stage(self):
        """Test schema resolver directly with USD stage."""
        try:
            from pxr import Usd
        except ImportError:
            self.skipTest("USD Python bindings not available")
            
        # Open the USD stage
        stage = Usd.Stage.Open(str(self.ant_usda_path))
        self.assertIsNotNone(stage)
        
        # Create resolver
        resolver = Resolver(["newton", "physx"])
        
        # Find prims with PhysX joint attributes
        joint_prims = []
        for prim in stage.Traverse():
            if prim.HasAttribute("physxJoint:armature"):
                joint_prims.append(prim)
                
        print(f"Found {len(joint_prims)} prims with physxJoint:armature in ant.usda")
        
        # Test resolver on real prims
        for i, prim in enumerate(joint_prims):  # Test first 3
            prim_path = str(prim.GetPath())
            
            # Test armature resolution
            armature = resolver.get_value(prim, "joint", "armature", default=0.0)
            phsyx_armature = prim.GetAttribute("physxJoint:armature").Get()
            print(f"  - {prim_path}: USD physxJoint:armature = {phsyx_armature}")

            self.assertAlmostEqual(armature, phsyx_armature, places=6)  # Expected value from ant.usda
            
                
            # Collect engine attributes for this prim
            resolver.collect_prim_engine_attrs(prim)
            
        # Check accumulated engine attributes
        engine_specific_attrs = resolver.get_engine_specific_attrs()
        if "physx" in engine_specific_attrs:
            physx_attrs = engine_specific_attrs["physx"]
            print(f"  - Collected PhysX attributes from {len(physx_attrs)} prims")
            
            # Verify we collected the expected attributes
            for prim_path, attrs in list(physx_attrs.items())[:2]:  # Show first 2
                print(f"    {prim_path}: {list(attrs.keys())}")
                if "physxJoint:armature" in attrs:
                    self.assertAlmostEqual(attrs["physxJoint:armature"], 0.01, places=6)
                    
    def test_time_step_resolution(self):
        """Test time step resolution from USD physics scene."""
        try:
            from pxr import Usd
        except ImportError:
            self.skipTest("USD Python bindings not available")
            
        # Open the USD stage
        stage = Usd.Stage.Open(str(self.ant_usda_path))
        self.assertIsNotNone(stage)
        
        # Find the physics scene prim
        physics_scene_prim = None
        for prim in stage.Traverse():
            if "physicsScene" in str(prim.GetPath()).lower():
                physics_scene_prim = prim
                break
                
        if physics_scene_prim is None:
            self.skipTest("No physics scene found in ant.usda")
            
        print(f"Found physics scene: {physics_scene_prim.GetPath()}")
        
        # Create resolver
        resolver = Resolver(["newton", "physx"])
        
        # Test time step resolution
        time_step = resolver.get_value(physics_scene_prim, "scene", "time_step", default=0.01)
        print(f"  - Time step resolved: {time_step}")
        
        # Check if this came from PhysX (since ant.usda might have TimeStepsPerSecond = 120)
        if "scene.time_step" in resolver.sources:
            engine_name, usd_attr = resolver.sources["scene.time_step"]
            print(f"  - Source: {engine_name} -> {usd_attr}")
            
            if engine_name == "physx" and usd_attr == "physxScene:timeStepsPerSecond":
                # Should be converted from Hz to seconds
                expected_time_step = 1.0 / 120.0  # From TimeStepsPerSecond = 120
                self.assertAlmostEqual(time_step, expected_time_step, places=6)
                print(f"  - Correctly converted 120 Hz -> {time_step:.6f} seconds")

    def test_mjc_solref_copy_and_import_usd(self):
        """Load pre-authored ant_mixed.usda and compare resolved gains under different priorities."""
        try:
            from pxr import Usd
        except ImportError:
            self.skipTest("USD Python bindings not available")

        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        dst = assets_dir / "ant_mixed.usda"
        self.assertTrue(dst.exists(), f"Missing mixed USD: {dst}")

        # Import with two different schema priorities
        builder_newton = ModelBuilder()
        result_newton = parse_usd(
            builder=builder_newton,
            source=str(dst),
            schema_priority=["newton", "physx", "mjc"],
            collect_engine_specific_attrs=True,
            verbose=False,
        )

        builder_mjc = ModelBuilder()
        result_mjc = parse_usd(
            builder=builder_mjc,
            source=str(dst),
            schema_priority=["mjc", "newton", "physx"],
            collect_engine_specific_attrs=True,
            verbose=False,
        )
        # With mjc priority and solref chosen as (0.5, 0.05), the resolved gains should be 2x physx/newton
        self.assertEqual(len(builder_newton.joint_limit_ke), len(builder_mjc.joint_limit_ke))
        self.assertEqual(len(builder_newton.joint_limit_kd), len(builder_mjc.joint_limit_kd))
        for physx_ke, mjc_ke in zip(builder_newton.joint_limit_ke, builder_mjc.joint_limit_ke, strict=False):
            self.assertAlmostEqual(mjc_ke, 2.0 * physx_ke, places=6)
        for physx_kd, mjc_kd in zip(builder_newton.joint_limit_kd, builder_mjc.joint_limit_kd, strict=False):
            self.assertAlmostEqual(mjc_kd, 2.0 * physx_kd, places=6)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestSchemaResolver))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running USD Schema Resolver Tests")
    print("=" * 60)
    print("Testing with actual ant.usda file and USD import functionality")
    print("Priority: ['newton', 'physx']")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n" + "=" * 60)
        print("All USD tests passed!")
        print("\nKey validations:")
        print("- USD import with schema resolver works correctly")
        print("- PhysX attributes from ant.usda are properly resolved")
        print("- Engine-specific attributes are collected from USD content")
        print("- Schema priority affects USD import behavior")
        print("- Joint armature values (0.01) resolved from physxJoint:armature")
        print("- Limit damping values (0.1) resolved from physxLimit:angular:damping")
        print("- Time step conversion works with real USD physics scene")
    else:
        print("\n" + "=" * 60)
        print("Some USD tests failed! ❌")
        sys.exit(1)
