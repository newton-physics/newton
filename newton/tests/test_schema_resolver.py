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
from pathlib import Path

from newton import ModelBuilder
from newton._src.utils.import_usd import parse_usd
from newton._src.utils.schema_resolver import Resolver
from newton.tests.unittest_utils import USD_AVAILABLE

try:
    from pxr import Usd
except ImportError:
    self.skipTest("USD Python bindings not available")


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
            verbose=False,
        )

        # Basic import validation
        self.assertIsInstance(result, dict)
        self.assertIn("path_body_map", result)
        self.assertIn("path_shape_map", result)
        # Check that we have bodies and shapes
        self.assertGreater(len(result["path_body_map"]), 0)
        self.assertGreater(len(result["path_shape_map"]), 0)

        print("Successfully imported ant.usda:")
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
        parse_usd(
            builder=builder,
            source=str(self.ant_usda_path),
            schema_priority=["physx", "newton"],  # PhysX first
            collect_engine_specific_attrs=True,
            verbose=False,
        )
        armature_values_found = []
        for i in range(6, builder.joint_dof_count):
            armature = builder.joint_armature[i]
            if armature > 0:
                armature_values_found.append(armature)
        for _i, armature in enumerate(armature_values_found):
            self.assertAlmostEqual(armature, 0.01, places=3)

        builder = ModelBuilder()
        parse_usd(
            builder=builder,
            source=str(self.ant_usda_path),
            schema_priority=["newton", "mjc"],  # nothing should be found
            collect_engine_specific_attrs=True,
            verbose=False,
        )
        armature_values_found = []
        for i in range(6, builder.joint_dof_count):
            armature = builder.joint_armature[i]
            if armature > 0:
                armature_values_found.append(armature)
        for _i, armature in enumerate(armature_values_found):
            self.assertAlmostEqual(armature, 0.01, places=3)

    def test_engine_specific_attrs_collection(self):
        """Test collection of engine-specific attributes from real ant.usda file."""
        builder = ModelBuilder()

        # Import with engine attribute collection enabled
        result = parse_usd(
            builder=builder,
            source=str(self.ant_usda_path),
            schema_priority=["newton", "physx"],
            collect_engine_specific_attrs=True,
            verbose=False,
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
            self.assertGreater(
                len(articulation_prims), 0, "Should find physxArticulation:enabledSelfCollisions attributes"
            )

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
            verbose=False,
        )

        # Import with PhysX first
        result2 = parse_usd(
            builder=builder2,
            source=str(self.ant_usda_path),
            schema_priority=["physx", "newton"],
            collect_engine_specific_attrs=True,
            verbose=False,
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
        self.assertEqual(builder1.joint_armature[6], builder2.joint_armature[6])

    def test_resolver(self):
        """Test schema resolver directly with USD stage."""

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
        for _i, prim in enumerate(joint_prims):
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

    def test_mjc_solref(self):
        """Load pre-authored ant_mixed.usda and compare resolved gains under different priorities."""

        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        dst = assets_dir / "ant_mixed.usda"
        self.assertTrue(dst.exists(), f"Missing mixed USD: {dst}")

        # Import with two different schema priorities
        builder_newton = ModelBuilder()
        parse_usd(
            builder=builder_newton,
            source=str(dst),
            schema_priority=["newton", "physx", "mjc"],
            collect_engine_specific_attrs=True,
            verbose=False,
        )

        builder_mjc = ModelBuilder()
        parse_usd(
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

    def test_newton_custom_properties(self):
        """Read pre-authored newton custom properties in ant_mixed.usda and validate import/modelBuilder."""
        # Use ant_mixed.usda which contains authored custom attributes
        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        dst = assets_dir / "ant_mixed.usda"
        self.assertTrue(dst.exists(), f"Missing mixed USD: {dst}")

        builder = ModelBuilder()
        result = parse_usd(
            builder=builder,
            source=str(dst),
            schema_priority=["newton", "physx", "mjc"],
            collect_engine_specific_attrs=True,
            verbose=False,
        )

        engine_attrs = result.get("engine_specific_attrs", {})
        self.assertIn("newton", engine_attrs)

        # Body property checks
        body_path = "/ant/front_left_leg"
        self.assertIn(body_path, engine_attrs["newton"])
        self.assertIn("newton:model:body:testBodyScalar", engine_attrs["newton"][body_path])
        self.assertIn("newton:model:body:testBodyVec", engine_attrs["newton"][body_path])
        self.assertIn("newton:model:body:testBodyBool", engine_attrs["newton"][body_path])
        self.assertIn("newton:model:body:testBodyInt", engine_attrs["newton"][body_path])
        self.assertIn("newton:state:body:testBodyVec3B", engine_attrs["newton"][body_path])
        self.assertAlmostEqual(engine_attrs["newton"][body_path]["newton:model:body:testBodyScalar"], 1.5, places=6)
        # also validate vector value in engine attrs
        vec_val = engine_attrs["newton"][body_path]["newton:model:body:testBodyVec"]
        self.assertAlmostEqual(float(vec_val[0]), 0.1, places=6)
        self.assertAlmostEqual(float(vec_val[1]), 0.2, places=6)
        self.assertAlmostEqual(float(vec_val[2]), 0.3, places=6)
        # Joint property checks (authored on front_left_leg joint)
        joint_name = "/ant/joints/front_left_leg"
        self.assertIn(joint_name, engine_attrs["newton"])  # engine attrs recorded
        self.assertIn("newton:state:joint:testJointScalar", engine_attrs["newton"][joint_name])
        # also validate state/control joint custom attrs in engine attrs
        self.assertIn("newton:state:joint:testStateJointScalar", engine_attrs["newton"][joint_name])
        self.assertIn("newton:control:joint:testControlJointScalar", engine_attrs["newton"][joint_name])
        self.assertIn("newton:state:joint:testStateJointBool", engine_attrs["newton"][joint_name])
        self.assertIn("newton:control:joint:testControlJointInt", engine_attrs["newton"][joint_name])
        self.assertIn("newton:model:joint:testJointVec", engine_attrs["newton"][joint_name])

        model = builder.finalize()
        state = model.state()
        self.assertEqual(model.get_attribute_frequency("testBodyVec"), "body")

        body_map = result["path_body_map"]
        idx = body_map[body_path]
        # Custom properties are currently materialized on Model (not State)
        body_scalar = model.testBodyScalar.numpy()
        self.assertAlmostEqual(float(body_scalar[idx]), 1.5, places=6)

        body_vec = model.testBodyVec.numpy()
        self.assertAlmostEqual(float(body_vec[idx, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(body_vec[idx, 1]), 0.2, places=6)
        self.assertAlmostEqual(float(body_vec[idx, 2]), 0.3, places=6)
        self.assertTrue(hasattr(model, "testBodyBool"))
        self.assertTrue(hasattr(model, "testBodyInt"))
        self.assertTrue(hasattr(state, "testBodyVec3B"))
        body_bool = model.testBodyBool.numpy()
        body_int = model.testBodyInt.numpy()
        body_vec_b = state.testBodyVec3B.numpy()
        self.assertEqual(int(body_bool[idx]), 1)
        self.assertEqual(int(body_int[idx]), 7)
        self.assertAlmostEqual(float(body_vec_b[idx, 0]), 1.1, places=6)
        self.assertAlmostEqual(float(body_vec_b[idx, 1]), 2.2, places=6)
        self.assertAlmostEqual(float(body_vec_b[idx, 2]), 3.3, places=6)

        # For prims without authored values, ensure defaults are present:
        # Pick a different body (e.g., front_right_leg) that didn't author testBodyScalar
        other_body = "/ant/front_right_leg"
        self.assertIn(other_body, body_map)
        other_idx = body_map[other_body]
        # The default for float is 0.0
        self.assertAlmostEqual(float(body_scalar[other_idx]), 0.0, places=6)
        # The default for vector3f is (0,0,0)
        self.assertAlmostEqual(float(body_vec[other_idx, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(body_vec[other_idx, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(body_vec[other_idx, 2]), 0.0, places=6)

        # Joint custom property materialization and defaults
        self.assertEqual(model.get_attribute_frequency("testJointScalar"), "joint")
        # Authored joint value
        self.assertIn(joint_name, builder.joint_key)
        joint_idx = builder.joint_key.index(joint_name)
        joint_arr = model.testJointScalar.numpy()
        self.assertAlmostEqual(float(joint_arr[joint_idx]), 2.25, places=6)
        # Non-authored joint should be default 0.0
        other_joint = "/ant/joints/front_right_leg"
        self.assertIn(other_joint, builder.joint_key)
        other_joint_idx = builder.joint_key.index(other_joint)
        self.assertAlmostEqual(float(joint_arr[other_joint_idx]), 0.0, places=6)

        # Validate state-assigned custom property mirrors initial values
        # testStateJointScalar is authored on a joint with assignment="state"
        self.assertTrue(hasattr(state, "testStateJointScalar"))
        state_joint = state.testStateJointScalar.numpy()
        self.assertAlmostEqual(float(state_joint[joint_idx]), 4.0, places=6)
        self.assertAlmostEqual(float(state_joint[other_joint_idx]), 0.0, places=6)
        # bool state property
        self.assertTrue(hasattr(state, "testStateJointBool"))
        state_joint_bool = state.testStateJointBool.numpy()
        self.assertEqual(int(state_joint_bool[joint_idx]), 1)
        self.assertEqual(int(state_joint_bool[other_joint_idx]), 0)

        # Validate control-assigned custom property mirrors initial values
        control = model.control()
        self.assertTrue(hasattr(control, "testControlJointScalar"))
        control_joint = control.testControlJointScalar.numpy()
        self.assertAlmostEqual(float(control_joint[joint_idx]), 5.5, places=6)
        self.assertAlmostEqual(float(control_joint[other_joint_idx]), 0.0, places=6)
        # int control property
        self.assertTrue(hasattr(control, "testControlJointInt"))
        control_joint_int = control.testControlJointInt.numpy()
        self.assertEqual(int(control_joint_int[joint_idx]), 3)
        self.assertEqual(int(control_joint_int[other_joint_idx]), 0)

    def test_physx_engine_specific_attrs_in_ant_mixed(self):
        """Validate PhysX engine-specific attributes are collected from ant_mixed.usda."""
        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        usd_path = assets_dir / "ant_mixed.usda"
        self.assertTrue(usd_path.exists(), f"Missing mixed USD: {usd_path}")

        builder = ModelBuilder()
        result = parse_usd(
            builder=builder,
            source=str(usd_path),
            schema_priority=["newton", "physx", "mjc"],
            collect_engine_specific_attrs=True,
            verbose=False,
        )

        engine_attrs = result.get("engine_specific_attrs", {})
        self.assertIn("physx", engine_attrs, "PhysX engine attributes should be collected")
        physx_attrs = engine_attrs["physx"]
        self.assertIsInstance(physx_attrs, dict)

        # Accumulate authored PhysX properties of interest
        articulation_found = []
        joint_armature_found = []
        limit_damping_found = []

        for prim_path, attrs in physx_attrs.items():
            if "physxArticulation:enabledSelfCollisions" in attrs:
                articulation_found.append((prim_path, attrs["physxArticulation:enabledSelfCollisions"]))
            if "physxJoint:armature" in attrs:
                joint_armature_found.append((prim_path, attrs["physxJoint:armature"]))
            if "physxLimit:angular:damping" in attrs:
                limit_damping_found.append((prim_path, attrs["physxLimit:angular:damping"]))

        # We expect at least one instance of each from ant_mixed.usda
        self.assertGreater(
            len(articulation_found), 0, "Should find physxArticulation:enabledSelfCollisions on articulation root"
        )
        self.assertGreater(len(joint_armature_found), 0, "Should find physxJoint:armature on joints")
        self.assertGreater(len(limit_damping_found), 0, "Should find physxLimit:angular:damping on joints")

        # Validate values against authored USD
        # Articulation self-collisions should be false/0 on /ant
        for prim_path, val in articulation_found:
            if str(prim_path) == "/ant" or "/ant" in str(prim_path):
                self.assertEqual(bool(val), False)
                break

        # Joint armature and limit damping should match authored values
        for _prim_path, val in joint_armature_found[:3]:
            self.assertAlmostEqual(float(val), 0.01, places=6)
        for _prim_path, val in limit_damping_found[:3]:
            self.assertAlmostEqual(float(val), 0.1, places=6)


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
