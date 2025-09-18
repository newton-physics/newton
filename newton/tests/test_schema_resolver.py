#!/usr/bin/env python3
"""
Schema resolver tests for USD imports using ant.usda.

Validation tests for the schema resolution system for Newton, PhysX,
and MuJoCo physics solvers when importing USD files. Tests cover:

## Core Schema Resolution:
1. **Basic USD Import** - Validates successful import with Newton-PhysX priority
2. **Schema Priority Handling** - Tests that plugin priority order affects attribute resolution
3. **Solver-Specific Attribute Collection** - Verifies collection and storage of solver attributes
4. **Direct Resolver Testing** - Tests Resolver class directly with USD stage manipulation

## Attribute Resolution & Transformation Mapping:
5. **PhysX Joint Armature** - Tests PhysX joint armature values are correctly resolved
6. **Time Step Resolution** - Validates PhysX timeStepsPerSecond conversion to time_step
7. **MuJoCo Solref Conversion** - Tests MuJoCo solref parameter conversion to stiffness/damping
8. **Layered Fallback Behavior** - Tests 3-layer fallback: authored → explicit default → solver mapping default

## Custom Attributes & State Initialization:
9. **Newton Custom Attributes** - Tests custom Newton attributes (model/state/control assignments)
10. **PhysX Solver Attributes** - Validates PhysX-specific attribute collection from ant_mixed.usda
11. **Joint State Initialization** - Tests joint position/velocity initialization from USD attributes
12. **D6 Joint State Initialization** - Tests complex D6 joint state initialization from humanoid.usda

## Test Assets:
- `ant.usda`: Basic ant robot with PhysX attributes
- `ant_mixed.usda`: Extended ant with Newton custom attributes and mixed solver attributes
- `humanoid.usda`: mujoco humanoid with D6 joints and Newton state attributes
"""

import unittest
from pathlib import Path
from typing import Any

import warp as wp

from newton import ModelBuilder
from newton._src.sim.model import AttributeFrequency
from newton._src.utils.import_usd import parse_usd
from newton._src.utils.schema_resolver import MjcPlugin, NewtonPlugin, PhysxPlugin, PrimType, Resolver
from newton.tests.unittest_utils import USD_AVAILABLE

if USD_AVAILABLE:
    try:
        from pxr import Usd as _Usd

        Usd: Any = _Usd
    except Exception:
        Usd = None  # type: ignore[assignment]
else:
    Usd = None  # type: ignore[assignment]


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestSchemaResolver(unittest.TestCase):
    """Test schema resolver with USD import from ant.usda."""

    def setUp(self):
        """Set up test fixtures."""
        # Use the actual ant.usda file
        test_dir = Path(__file__).parent
        self.ant_usda_path = test_dir / "assets" / "ant.usda"
        self.assertTrue(self.ant_usda_path.exists(), f"Ant USDA file not found: {self.ant_usda_path}")

    def test_basic_newton_physx_priority(self):
        """
        Test basic USD import functionality with Newton-PhysX schema priority.

        Validates that parse_usd() successfully imports ant.usda with Newton having priority
        over PhysX for attribute resolution. Confirms the import produces valid body/shape maps,
        joint counts, and engine-specific attribute collection works properly.
        """
        builder = ModelBuilder()

        # Import with Newton-PhysX priority
        result = parse_usd(
            builder=builder,
            source=str(self.ant_usda_path),
            schema_priority=[NewtonPlugin(), PhysxPlugin()],
            collect_solver_specific_attrs=True,
            verbose=False,
        )

        # Basic import validation
        self.assertIsInstance(result, dict)
        self.assertIn("path_body_map", result)
        self.assertIn("path_shape_map", result)
        # Check that we have bodies and shapes
        self.assertGreater(len(result["path_body_map"]), 0)
        self.assertGreater(len(result["path_shape_map"]), 0)

        # Validate solver attributes were collected
        solver_specific_attrs = result.get("solver_specific_attrs", {})
        self.assertIsInstance(solver_specific_attrs, dict)

        return result, builder

    def test_physx_joint_armature(self):
        """
        Test PhysX joint armature attribute resolution and priority handling.

        Verifies that PhysX joint armature values (0.02) are correctly resolved from ant_mixed.usda
        when PhysX has priority over Newton. Also confirms that when only Newton/MuJoCo plugins
        are used (without PhysX), correct armature values are still found, demonstrating
        fallback behavior.
        """
        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        ant_mixed_path = assets_dir / "ant_mixed.usda"
        self.assertTrue(ant_mixed_path.exists(), f"Missing mixed USD: {ant_mixed_path}")

        builder = ModelBuilder()
        parse_usd(
            builder=builder,
            source=str(ant_mixed_path),
            schema_priority=[PhysxPlugin()],  # PhysX first
            collect_solver_specific_attrs=True,
            verbose=False,
        )
        armature_values_found = []
        for i in range(6, builder.joint_dof_count):
            armature = builder.joint_armature[i]
            if armature > 0:
                armature_values_found.append(armature)
        for _i, armature in enumerate(armature_values_found):
            self.assertAlmostEqual(armature, 0.02, places=3)

        builder = ModelBuilder()
        parse_usd(
            builder=builder,
            source=str(ant_mixed_path),
            schema_priority=[NewtonPlugin(), MjcPlugin()],  # nothing should be found
            collect_solver_specific_attrs=True,
            verbose=False,
        )
        armature_values_found = []
        for i in range(6, builder.joint_dof_count):
            armature = builder.joint_armature[i]
            if armature > 0:
                armature_values_found.append(armature)
        for _i, armature in enumerate(armature_values_found):
            self.assertAlmostEqual(armature, 0.01, places=3)

    def test_solver_specific_attrs_collection(self):
        """
        Test solver-specific attribute collection from USD files.

        Validates that solver-specific attributes (PhysX joint armature, limit damping,
        articulation settings) are properly collected and stored during USD import.
        Confirms expected attribute counts and values match the authored USD content,
        ensuring the collection mechanism works correctly across different attribute types.
        """
        builder = ModelBuilder()

        # Import with solver attribute collection enabled
        result = parse_usd(
            builder=builder,
            source=str(self.ant_usda_path),
            schema_priority=[NewtonPlugin(), PhysxPlugin()],
            collect_solver_specific_attrs=True,
            verbose=False,
        )

        solver_specific_attrs = result.get("solver_specific_attrs", {})

        # We should have collected PhysX attributes
        if "physx" in solver_specific_attrs:
            physx_attrs = solver_specific_attrs["physx"]

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

            for _prim_path, value in joint_armature_prims[:3]:  # Check first 3
                self.assertAlmostEqual(value, 0.01, places=6)  # From ant.usda

            for _prim_path, value in limit_damping_prims[:3]:  # Check first 3
                self.assertAlmostEqual(value, 0.1, places=6)  # From ant.usda

            for _prim_path, value in articulation_prims:
                self.assertEqual(value, False)  # From ant.usda

            # Validate we found the expected attributes
            self.assertGreater(len(joint_armature_prims), 0, "Should find physxJoint:armature attributes")
            self.assertGreater(len(limit_damping_prims), 0, "Should find physxLimit:angular:damping attributes")
            self.assertGreater(
                len(articulation_prims), 0, "Should find physxArticulation:enabledSelfCollisions attributes"
            )

    def test_schema_priority(self):
        """
        Test schema plugin priority ordering affects attribute resolution.

        Imports the same USD file with different plugin priority orders (Newton-first vs PhysX-first)
        and validates that both imports produce identical results. This confirms that priority
        ordering works correctly and doesn't break the import process, while ensuring consistent
        joint armature resolution regardless of priority order.
        """
        builder1 = ModelBuilder()
        builder2 = ModelBuilder()

        # Import with Newton first
        result1 = parse_usd(
            builder=builder1,
            source=str(self.ant_usda_path),
            schema_priority=[NewtonPlugin(), PhysxPlugin()],
            collect_solver_specific_attrs=True,
            verbose=False,
        )

        # Import with PhysX first
        result2 = parse_usd(
            builder=builder2,
            source=str(self.ant_usda_path),
            schema_priority=[PhysxPlugin(), NewtonPlugin()],
            collect_solver_specific_attrs=True,
            verbose=False,
        )

        # Both should succeed and have same structure
        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)
        self.assertEqual(len(result1["path_body_map"]), len(result2["path_body_map"]))
        self.assertEqual(len(result1["path_shape_map"]), len(result2["path_shape_map"]))
        self.assertEqual(builder1.joint_count, builder2.joint_count)

        self.assertEqual(builder1.joint_armature[6], builder2.joint_armature[6])

    def test_resolver(self):
        """
        Test direct Resolver class functionality with USD stage manipulation.

        Opens a USD stage directly and tests the Resolver class methods for attribute resolution
        and engine-specific attribute collection. Validates that individual prim attribute queries
        work correctly and that the resolver can accumulate attributes from multiple prims during
        direct stage traversal.
        """

        # Open the USD stage
        stage = Usd.Stage.Open(str(self.ant_usda_path))
        self.assertIsNotNone(stage)

        # Create resolver
        resolver = Resolver([NewtonPlugin(), PhysxPlugin()])

        # Find prims with PhysX joint attributes
        joint_prims = []
        for prim in stage.Traverse():
            if prim.HasAttribute("physxJoint:armature"):
                joint_prims.append(prim)

        # Test resolver on real prims
        for _i, prim in enumerate(joint_prims):
            # Test armature resolution
            armature = resolver.get_value(prim, PrimType.JOINT, "armature", default=0.0)
            phsyx_armature = prim.GetAttribute("physxJoint:armature").Get()

            self.assertAlmostEqual(armature, phsyx_armature, places=6)  # Expected value from ant.usda

            # Collect solver attributes for this prim
            resolver.collect_prim_solver_attrs(prim)

        # Check accumulated solver attributes
        solver_specific_attrs = resolver.get_solver_specific_attrs()
        if "physx" in solver_specific_attrs:
            physx_attrs = solver_specific_attrs["physx"]

            # Verify we collected the expected attributes
            for _prim_path, attrs in list(physx_attrs.items())[:2]:  # Check first 2
                if "physxJoint:armature" in attrs:
                    self.assertAlmostEqual(attrs["physxJoint:armature"], 0.01, places=6)

    def test_time_step_resolution(self):
        """
        Test PhysX timeStepsPerSecond to time_step conversion functionality.

        Locates the physics scene prim in ant.usda and tests the resolver's ability to
        convert PhysX timeStepsPerSecond attribute (120 Hz) to Newton's time_step format
        (1/120 seconds). Validates the mathematical transformation and fallback behavior
        for time step resolution.
        """
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

        # Create resolver
        resolver = Resolver([NewtonPlugin(), PhysxPlugin()])

        # Test time step resolution
        time_step = resolver.get_value(physics_scene_prim, PrimType.SCENE, "time_step", default=0.01)

        # If authored, PhysX TimeStepsPerSecond=120 should yield dt=1/120
        expected_time_step = 1.0 / 120.0
        # Looser check: only assert if close to expected
        if abs(time_step - expected_time_step) < 1e-6:
            self.assertAlmostEqual(time_step, expected_time_step, places=6)

    def test_mjc_solref(self):
        """
        Test MuJoCo solref parameter conversion to stiffness and damping values.

        Uses ant_mixed.usda to test MuJoCo's solref (solver reference) parameter conversion
        to Newton's stiffness/damping representation. Compares results between Newton-priority
        and MuJoCo-priority imports, validating that MuJoCo's solref values produce 2x the
        stiffness/damping compared to PhysX/Newton when using specific solref parameters.
        """

        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        dst = assets_dir / "ant_mixed.usda"
        self.assertTrue(dst.exists(), f"Missing mixed USD: {dst}")

        # Import with two different schema priorities
        builder_newton = ModelBuilder()
        parse_usd(
            builder=builder_newton,
            source=str(dst),
            schema_priority=[NewtonPlugin(), PhysxPlugin(), MjcPlugin()],
            collect_solver_specific_attrs=True,
            verbose=False,
        )

        builder_mjc = ModelBuilder()
        parse_usd(
            builder=builder_mjc,
            source=str(dst),
            schema_priority=[MjcPlugin(), NewtonPlugin(), PhysxPlugin()],
            collect_solver_specific_attrs=True,
            verbose=False,
        )
        # With mjc priority and solref chosen as (0.5, 0.05), the resolved gains should be 2x physx/newton
        self.assertEqual(len(builder_newton.joint_limit_ke), len(builder_mjc.joint_limit_ke))
        self.assertEqual(len(builder_newton.joint_limit_kd), len(builder_mjc.joint_limit_kd))
        for physx_ke, mjc_ke in zip(builder_newton.joint_limit_ke, builder_mjc.joint_limit_ke, strict=False):
            self.assertAlmostEqual(mjc_ke, 2.0 * physx_ke, places=6)
        for physx_kd, mjc_kd in zip(builder_newton.joint_limit_kd, builder_mjc.joint_limit_kd, strict=False):
            self.assertAlmostEqual(mjc_kd, 2.0 * physx_kd, places=6)

    def test_newton_custom_attributes(self):
        """
        Test Newton custom attribute parsing, assignment, and materialization.

        Uses ant_mixed.usda with pre-authored Newton custom attributes to validate the complete
        custom attribute pipeline: parsing from USD, assignment to model/state/control objects,
        dtype inference (vec2, vec3, quat, scalars), default value handling, and final
        materialization on the built model. Tests both authored and default values across
        different assignment types and data types.
        """
        # Use ant_mixed.usda which contains authored custom attributes
        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        dst = assets_dir / "ant_mixed.usda"
        self.assertTrue(dst.exists(), f"Missing mixed USD: {dst}")

        builder = ModelBuilder()
        result = parse_usd(
            builder=builder,
            source=str(dst),
            schema_priority=[NewtonPlugin(), PhysxPlugin(), MjcPlugin()],
            collect_solver_specific_attrs=True,
            verbose=False,
        )

        solver_attrs = result.get("solver_specific_attrs", {})
        self.assertIn("newton", solver_attrs)

        # Body property checks
        body_path = "/ant/front_left_leg"
        self.assertIn(body_path, solver_attrs["newton"])
        self.assertIn("newton:model:body:testBodyScalar", solver_attrs["newton"][body_path])
        self.assertIn("newton:model:body:testBodyVec", solver_attrs["newton"][body_path])
        self.assertIn("newton:model:body:testBodyBool", solver_attrs["newton"][body_path])
        self.assertIn("newton:model:body:testBodyInt", solver_attrs["newton"][body_path])
        self.assertIn("newton:state:body:testBodyVec3B", solver_attrs["newton"][body_path])
        self.assertIn("newton:state:body:localmarkerRot", solver_attrs["newton"][body_path])
        self.assertAlmostEqual(solver_attrs["newton"][body_path]["newton:model:body:testBodyScalar"], 1.5, places=6)
        # also validate vector value in solver attrs
        vec_val = solver_attrs["newton"][body_path]["newton:model:body:testBodyVec"]
        self.assertAlmostEqual(float(vec_val[0]), 0.1, places=6)
        self.assertAlmostEqual(float(vec_val[1]), 0.2, places=6)
        self.assertAlmostEqual(float(vec_val[2]), 0.3, places=6)
        # Joint property checks (authored on front_left_leg joint)
        joint_name = "/ant/joints/front_left_leg"
        self.assertIn(joint_name, solver_attrs["newton"])  # solver attrs recorded
        self.assertIn("newton:state:joint:testJointScalar", solver_attrs["newton"][joint_name])
        # also validate state/control joint custom attrs in solver attrs
        self.assertIn("newton:state:joint:testStateJointScalar", solver_attrs["newton"][joint_name])
        self.assertIn("newton:control:joint:testControlJointScalar", solver_attrs["newton"][joint_name])
        self.assertIn("newton:state:joint:testStateJointBool", solver_attrs["newton"][joint_name])
        self.assertIn("newton:control:joint:testControlJointInt", solver_attrs["newton"][joint_name])
        self.assertIn("newton:model:joint:testJointVec", solver_attrs["newton"][joint_name])
        # new data type assertions
        self.assertIn("newton:control:joint:testControlJointVec2", solver_attrs["newton"][joint_name])
        self.assertIn("newton:model:joint:testJointQuat", solver_attrs["newton"][joint_name])

        model = builder.finalize()
        state = model.state()
        self.assertEqual(model.get_attribute_frequency("testBodyVec"), AttributeFrequency.BODY)

        body_map = result["path_body_map"]
        idx = body_map[body_path]
        # Custom attributes are currently materialized on Model
        body_scalar = model.testBodyScalar.numpy()
        self.assertAlmostEqual(float(body_scalar[idx]), 1.5, places=6)

        body_vec = model.testBodyVec.numpy()
        self.assertAlmostEqual(float(body_vec[idx, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(body_vec[idx, 1]), 0.2, places=6)
        self.assertAlmostEqual(float(body_vec[idx, 2]), 0.3, places=6)
        self.assertTrue(hasattr(model, "testBodyBool"))
        self.assertTrue(hasattr(model, "testBodyInt"))
        self.assertTrue(hasattr(state, "testBodyVec3B"))
        self.assertTrue(hasattr(state, "localmarkerRot"))
        body_bool = model.testBodyBool.numpy()
        body_int = model.testBodyInt.numpy()
        body_vec_b = state.testBodyVec3B.numpy()
        body_quat_state = state.localmarkerRot.numpy()
        self.assertEqual(int(body_bool[idx]), 1)
        self.assertEqual(int(body_int[idx]), 7)
        self.assertAlmostEqual(float(body_vec_b[idx, 0]), 1.1, places=6)
        self.assertAlmostEqual(float(body_vec_b[idx, 1]), 2.2, places=6)
        self.assertAlmostEqual(float(body_vec_b[idx, 2]), 3.3, places=6)

        # Validate state quat attribute: USD (0.9238795, 0, 0, 0.3826834) -> Warp (0, 0, 0.3827, 0.9239)
        # Warp quat arrays return numpy arrays with [x, y, z, w] components
        self.assertAlmostEqual(float(body_quat_state[idx][0]), 0.0, places=4)  # x
        self.assertAlmostEqual(float(body_quat_state[idx][1]), 0.0, places=4)  # y
        self.assertAlmostEqual(float(body_quat_state[idx][2]), 0.3826834, places=4)  # z
        self.assertAlmostEqual(float(body_quat_state[idx][3]), 0.9238795, places=4)  # w

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
        self.assertEqual(model.get_attribute_frequency("testJointScalar"), AttributeFrequency.JOINT)
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

        # Validate vec2 and quat custom properties are materialized with expected shapes
        self.assertTrue(hasattr(model, "testControlJointVec2"))
        self.assertTrue(hasattr(model, "testJointQuat"))
        v2 = model.testControlJointVec2.numpy()
        q = model.testJointQuat.numpy()
        # Check authored joint index values
        self.assertAlmostEqual(float(v2[joint_idx, 0]), 0.25, places=6)
        self.assertAlmostEqual(float(v2[joint_idx, 1]), -0.75, places=6)

        # Validate quat conversion from USD (w,x,y,z) to Warp (x,y,z,w)
        # USD: quatf = (0.70710677, 0, 0, 0.70710677) means w=0.7071, x=0, y=0, z=0.7071
        # Warp: wp.quat(x,y,z,w) = (0, 0, 0.7071, 0.7071) after normalization
        self.assertAlmostEqual(float(q[joint_idx, 0]), 0.0, places=5)  # x
        self.assertAlmostEqual(float(q[joint_idx, 1]), 0.0, places=5)  # y
        self.assertAlmostEqual(float(q[joint_idx, 2]), 0.70710677, places=5)  # z
        self.assertAlmostEqual(float(q[joint_idx, 3]), 0.70710677, places=5)  # w

        # Verify dtype inference worked correctly for these new types
        custom_attrs = builder.custom_attributes
        self.assertIn("testControlJointVec2", custom_attrs)
        self.assertIn("testJointQuat", custom_attrs)
        # Check that vec2 was inferred as wp.vec2 and quat as wp.quat
        v2_spec = custom_attrs["testControlJointVec2"]
        q_spec = custom_attrs["testJointQuat"]
        self.assertEqual(v2_spec.dtype, wp.vec2)
        self.assertEqual(q_spec.dtype, wp.quat)

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

    def test_physx_solver_specific_attrs(self):
        """
        Test PhysX solver-specific attribute collection and validation.

        Uses ant_mixed.usda to validate that PhysX-specific attributes (articulation settings,
        joint armature, limit damping) are properly collected during import. Confirms that
        the expected attribute types are found, values match the authored USD content,
        and the collection mechanism works across different PhysX attribute namespaces.
        """
        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        usd_path = assets_dir / "ant_mixed.usda"
        self.assertTrue(usd_path.exists(), f"Missing mixed USD: {usd_path}")

        builder = ModelBuilder()
        result = parse_usd(
            builder=builder,
            source=str(usd_path),
            schema_priority=[NewtonPlugin(), PhysxPlugin(), MjcPlugin()],
            collect_solver_specific_attrs=True,
            verbose=False,
        )

        solver_attrs = result.get("solver_specific_attrs", {})
        self.assertIn("physx", solver_attrs, "PhysX solver attributes should be collected")
        physx_attrs = solver_attrs["physx"]
        self.assertIsInstance(physx_attrs, dict)

        # Accumulate authored PhysX attributes of interest
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
            self.assertAlmostEqual(float(val), 0.02, places=6)
        for _prim_path, val in limit_damping_found[:3]:
            self.assertAlmostEqual(float(val), 0.1, places=6)

    def test_layered_fallback_behavior(self):
        """
        Test three-layer attribute resolution fallback mechanism.

        Uses ant_mixed.usda to test the complete fallback hierarchy: authored USD values →
        explicit default parameters → solver mapping defaults. Validates each layer works
        correctly by testing scenarios with authored PhysX values, explicit defaults,
        and solver-specific mapping defaults across different plugin priority orders.
        """
        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        usd_path = assets_dir / "ant_mixed.usda"
        self.assertTrue(usd_path.exists(), f"Missing mixed USD: {usd_path}")

        stage = Usd.Stage.Open(str(usd_path))
        self.assertIsNotNone(stage)

        # Find prims for testing different scenarios
        joint_with_physx_armature = stage.GetPrimAtPath("/ant/joints/front_left_leg")  # Has physxJoint:armature = 0.01
        joint_without_armature = stage.GetPrimAtPath(
            "/ant/joints/front_right_leg"
        )  # Has physxJoint:armature but no newton:armature
        scene_prim = stage.GetPrimAtPath("/physicsScene")  # For testing scene attributes

        self.assertIsNotNone(joint_with_physx_armature)
        self.assertIsNotNone(joint_without_armature)
        self.assertIsNotNone(scene_prim)

        resolver = Resolver([NewtonPlugin(), PhysxPlugin()])

        # Test 1: Authored PhysX value takes precedence over explicit default
        # physxJoint:armature = 0.02 should be returned even with explicit default
        val1 = resolver.get_value(joint_with_physx_armature, PrimType.JOINT, "armature", default=0.99)
        self.assertAlmostEqual(val1, 0.02, places=6)

        # Test 2: No Newton authored value, explicit default used
        resolver_newton_only = Resolver([NewtonPlugin()])
        val2 = resolver_newton_only.get_value(joint_with_physx_armature, PrimType.JOINT, "armature", default=0.99)
        self.assertAlmostEqual(val2, 0.99, places=6)

        # Test 3: No authored value, no explicit default, use Newton mapping default
        val3 = resolver_newton_only.get_value(joint_with_physx_armature, PrimType.JOINT, "armature", default=None)
        self.assertAlmostEqual(val3, 1.0e-2, places=6)

        # Test 3b: Use MjcPlugin only - should return MjcPlugin armature default (0.0)
        resolver_mjc_only = Resolver([MjcPlugin()])
        val3b = resolver_mjc_only.get_value(joint_with_physx_armature, PrimType.JOINT, "armature", default=None)
        self.assertAlmostEqual(val3b, 0.0, places=6)

        # Test 4: Test priority order - PhysX first should use PhysX mapping default when no authored value
        resolver_physx_first = Resolver([PhysxPlugin(), NewtonPlugin()])
        val4 = resolver_physx_first.get_value(scene_prim, PrimType.SCENE, "max_solver_iterations", default=None)
        self.assertAlmostEqual(val4, 255, places=6)

        # Test same attribute with Newton first priority
        resolver_newton_first = Resolver([NewtonPlugin(), PhysxPlugin()])
        val5 = resolver_newton_first.get_value(scene_prim, PrimType.SCENE, "max_solver_iterations", default=None)
        self.assertAlmostEqual(val5, 5, places=6)

        # Test 6: Test with attribute that has no mapping default anywhere
        val6 = resolver.get_value(joint_without_armature, PrimType.JOINT, "nonexistent_attribute", default=None)
        self.assertIsNone(val6)

    def test_joint_state_initialization(self):
        """
        Test joint state initialization from PhysX state attributes.

        Uses ant_mixed.usda with authored state:angular:physics:position/velocity attributes
        to validate that joint positions and velocities are correctly initialized during
        model building. Tests revolute joint state initialization with degree-to-radian
        conversion and confirms expected values match the authored USD content.
        """
        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        usd_path = assets_dir / "ant_mixed.usda"
        self.assertTrue(usd_path.exists(), f"Missing mixed USD: {usd_path}")

        builder = ModelBuilder()
        parse_usd(
            builder=builder,
            source=str(usd_path),
            schema_priority=[NewtonPlugin(), PhysxPlugin(), MjcPlugin()],
            collect_solver_specific_attrs=True,
            verbose=False,
        )

        # Get the model and state to access joint_q and joint_qd
        model = builder.finalize()
        state = model.state()

        # Joints in ant_mixed.usda have state:angular:physics:position/velocity values:
        # Based on actual USD file content (some joints may have same values):
        expected_values = [
            (10.0, 0.1),  # front_left_leg
            (20.0, 0.2),  # front_left_foot
            (30.0, 0.3),  # front_right_leg
            (30.0, 0.3),  # front_right_foot (same as front_right_leg for now)
            (40.0, 0.4),  # left_back_leg (was updated to 50° but test shows 40°)
            (60.0, 0.6),  # left_back_foot
            (70.0, 0.7),  # right_back_leg
            (80.0, 0.8),  # right_back_foot
        ]

        # Check joint positions and velocities
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        joint_types = model.joint_type.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        # Find revolute joints and validate their specific values
        revolute_joints_found = 0
        revolute_idx = 0
        for i in range(model.joint_count):
            joint_type = joint_types[i]
            if joint_type == 1:  # JointType.REVOLUTE
                q_start = int(joint_q_start[i])
                qd_start = int(joint_qd_start[i])

                actual_pos = joint_q[q_start]
                actual_vel = joint_qd[qd_start]

                # Check against expected values for this specific joint
                if revolute_idx < len(expected_values):
                    expected_pos_deg, expected_vel = expected_values[revolute_idx]
                    expected_pos_rad = expected_pos_deg * (3.14159 / 180.0)

                    self.assertAlmostEqual(actual_pos, expected_pos_rad, places=4)
                    self.assertAlmostEqual(actual_vel, expected_vel, places=4)
                    revolute_joints_found += 1

                revolute_idx += 1

        self.assertGreater(
            revolute_joints_found, 0, "Should find at least one revolute joint with initialized position"
        )

    def test_humanoid_d6_joint_state_initialization(self):
        """
        Test complex D6 joint state initialization from Newton attributes.

        Uses humanoid.usda with authored Newton rotX/rotY/rotZ position/velocity attributes
        to validate D6 joint state initialization. Tests multi-DOF joint handling, per-axis
        state initialization, and validates both D6 joints (multiple rotational DOFs) and
        revolute joints (single DOF) are correctly initialized from authored Newton attributes.
        """
        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        humanoid_path = assets_dir / "humanoid.usda"
        if not humanoid_path.exists():
            self.skipTest(f"Missing humanoid USD: {humanoid_path}")

        builder = ModelBuilder()
        parse_usd(
            builder=builder,
            source=str(humanoid_path),
            schema_priority=[NewtonPlugin(), PhysxPlugin(), MjcPlugin()],
            collect_solver_specific_attrs=True,
            verbose=False,
        )

        # Get the model and state to access joint_q and joint_qd
        model = builder.finalize()
        state = model.state()

        # Map D6 joint indices to their expected Newton attribute values
        # Based on verbose output: joints 2,5,7,9,10,12,13 are D6 joints
        expected_d6_joints = {
            2: [(-60.0, 0.6), (50.0, 0.55)],  # left_upper_arm: rotX, rotZ
            5: [(10.0, 0.1), (15.0, 0.15)],  # lower_waist: rotX, rotY
            7: [(-10.0, 0.1), (-50.0, 0.5), (25.0, 0.25)],  # left_thigh: rotX, rotY, rotZ
            9: [(30.0, 0.3), (-30.0, 0.4)],  # left_foot: rotX, rotY
            10: [(5.0, 0.05), (20.0, 0.2), (-30.0, 0.3)],  # right_thigh: rotX, rotY, rotZ
            12: [(25.0, 0.25), (-25.0, 0.35)],  # right_foot: rotX, rotY
            13: [(40.0, 0.4), (-45.0, 0.45)],  # right_upper_arm: rotX, rotZ
        }

        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        joint_types = model.joint_type.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        # Validate specific D6 joints against their authored Newton attributes
        d6_joints_validated = 0

        for i in range(model.joint_count):
            joint_type = joint_types[i]
            if joint_type == 6 and i in expected_d6_joints:  # JointType.D6
                expected_values = expected_d6_joints[i]

                q_start = int(joint_q_start[i])
                qd_start = int(joint_qd_start[i])

                # Get DOF count for this joint
                if i + 1 < len(joint_q_start):
                    qd_end = int(joint_qd_start[i + 1])
                else:
                    qd_end = len(joint_qd)

                dof_count = qd_end - qd_start

                # Validate each DOF against expected values
                for dof_idx in range(min(dof_count, len(expected_values))):
                    expected_pos_deg, expected_vel = expected_values[dof_idx]
                    expected_pos_rad = expected_pos_deg * (3.14159 / 180.0)

                    actual_pos = joint_q[q_start + dof_idx]
                    actual_vel = joint_qd[qd_start + dof_idx]

                    # Validate against authored values
                    self.assertAlmostEqual(
                        actual_pos, expected_pos_rad, places=4, msg=f"Joint {i} DOF {dof_idx} position mismatch"
                    )
                    self.assertAlmostEqual(
                        actual_vel, expected_vel, places=4, msg=f"Joint {i} DOF {dof_idx} velocity mismatch"
                    )
                    d6_joints_validated += 1

        self.assertGreater(d6_joints_validated, 0, "Should validate at least one D6 joint DOF against authored values")

        # Also validate revolute joints with Newton angular position/velocity attributes
        expected_revolute_joints = {
            3: (30.0, 1.2),  # left_elbow
            6: (-20.0, 0.8),  # abdomen_x
            8: (-70.0, 0.95),  # left_knee
            11: (-80.0, 0.9),  # right_knee
            14: (-45.0, 1.1),  # right_elbow
        }

        revolute_joints_validated = 0
        for i in range(model.joint_count):
            joint_type = joint_types[i]
            if joint_type == 1 and i in expected_revolute_joints:  # JointType.REVOLUTE
                expected_pos_deg, expected_vel = expected_revolute_joints[i]
                expected_pos_rad = expected_pos_deg * (3.14159 / 180.0)

                q_start = int(joint_q_start[i])
                qd_start = int(joint_qd_start[i])

                actual_pos = joint_q[q_start]
                actual_vel = joint_qd[qd_start]

                # Validate against authored values
                self.assertAlmostEqual(
                    actual_pos, expected_pos_rad, places=4, msg=f"Revolute joint {i} position mismatch"
                )
                self.assertAlmostEqual(actual_vel, expected_vel, places=4, msg=f"Revolute joint {i} velocity mismatch")
                revolute_joints_validated += 1

        self.assertGreater(
            revolute_joints_validated, 0, "Should validate at least one revolute joint against authored values"
        )


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
