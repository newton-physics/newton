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

"""
Schema resolver tests for USD imports using ant.usda.

Validation tests for the schema resolution system for Newton, PhysX,
and MuJoCo physics solvers when importing USD files. Tests cover:

## Core Schema Resolution:
1. **Basic USD Import** - Validates successful import with Newton-PhysX priority
2. **Schema Priority Handling** - Tests that plugin priority order affects attribute resolution
3. **Solver-Specific Attribute Collection** - Verifies collection and storage of solver attributes
4. **Direct SchemaResolverManager Testing** - Tests SchemaResolverManager class directly with USD stage manipulation

## Attribute Resolution & Transformation Mapping:
5. **PhysX Joint Armature** - Tests PhysX joint armature values are correctly resolved
6. **Time Step Resolution** - Validates PhysX timeStepsPerSecond conversion to time_step
7. **MuJoCo Solref Conversion** - Tests MuJoCo solref parameter conversion to stiffness/damping
8. **Layered Fallback Behavior** - Tests 3-layer fallback: authored → explicit default → solver mapping default

## Custom Attributes & State Initialization:
9. **Newton Custom Attributes** - Tests custom Newton attributes (model/state/control assignments)
10. **Namespaced Custom Attributes** - Tests namespace isolation and independent attributes with same name
11. **PhysX Solver Attributes** - Validates PhysX-specific attribute collection from ant_mixed.usda
12. **Joint State Initialization** - Tests joint position/velocity initialization from USD attributes
13. **D6 Joint State Initialization** - Tests complex D6 joint state initialization from humanoid.usda

## Test Assets:
- `ant.usda`: Basic ant robot with PhysX attributes
- `ant_mixed.usda`: Extended ant with Newton custom attributes, namespaced attributes, and mixed solver attributes
- `humanoid.usda`: mujoco humanoid with D6 joints and Newton state attributes
"""

import unittest
from pathlib import Path
from typing import Any

import warp as wp

from newton import Model, ModelBuilder
from newton._src.usd.schema_resolver import (
    PrimType,
    SchemaResolverManager,
)
from newton._src.usd.schemas import (
    SchemaResolverMjc,
    SchemaResolverNewton,
    SchemaResolverPhysx,
)
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE

AttributeFrequency = Model.AttributeFrequency

if USD_AVAILABLE:
    try:
        from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade
        from pxr import Usd as _Usd

        Usd: Any = _Usd
    except (ImportError, ModuleNotFoundError):
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
        result = builder.add_usd(
            source=str(self.ant_usda_path),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
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
        schema_attrs = result.get("schema_attrs", {})
        self.assertIsInstance(schema_attrs, dict)

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
        builder.add_usd(
            source=str(ant_mixed_path),
            schema_resolvers=[SchemaResolverPhysx()],  # PhysX first
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
        builder.add_usd(
            source=str(ant_mixed_path),
            schema_resolvers=[SchemaResolverNewton()],  # nothing should be found
            verbose=False,
        )
        armature_values_found = []
        for i in range(6, builder.joint_dof_count):
            armature = builder.joint_armature[i]
            if armature > 0:
                armature_values_found.append(armature)
        for _i, armature in enumerate(armature_values_found):
            self.assertAlmostEqual(armature, 0.01, places=3)

    def test_schema_attrs_collection(self):
        """
        Test solver-specific attribute collection from USD files.

        Validates that solver-specific attributes (PhysX joint armature, limit damping,
        articulation settings) are properly collected and stored during USD import.
        Confirms expected attribute counts and values match the authored USD content,
        ensuring the collection mechanism works correctly across different attribute types.
        """
        builder = ModelBuilder()

        # Import with solver attribute collection enabled
        result = builder.add_usd(
            source=str(self.ant_usda_path),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
            verbose=False,
        )

        schema_attrs = result.get("schema_attrs", {})

        # We should have collected PhysX attributes
        if "physx" in schema_attrs:
            physx_attrs = schema_attrs["physx"]

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

    def test_schema_resolvers(self):
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
        result1 = builder1.add_usd(
            source=str(self.ant_usda_path),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
            verbose=False,
        )

        # Import with PhysX first
        result2 = builder2.add_usd(
            source=str(self.ant_usda_path),
            schema_resolvers=[SchemaResolverPhysx(), SchemaResolverNewton()],
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
        Test direct SchemaResolverManager class functionality with USD stage manipulation.

        Opens a USD stage directly and tests the SchemaResolverManager class methods for attribute resolution
        and engine-specific attribute collection. Validates that individual prim attribute queries
        work correctly and that the resolver can accumulate attributes from multiple prims during
        direct stage traversal.
        """

        # Open the USD stage
        stage = Usd.Stage.Open(str(self.ant_usda_path))
        self.assertIsNotNone(stage)

        # Create resolver
        resolver = SchemaResolverManager([SchemaResolverNewton(), SchemaResolverPhysx()])

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
            resolver.collect_prim_attrs(prim)

        # Check accumulated solver attributes
        schema_attrs = resolver.schema_attrs
        if "physx" in schema_attrs:
            physx_attrs = schema_attrs["physx"]

            # Verify we collected the expected attributes
            for _prim_path, attrs in list(physx_attrs.items())[:2]:  # Check first 2
                if "physxJoint:armature" in attrs:
                    self.assertAlmostEqual(attrs["physxJoint:armature"], 0.01, places=6)

    def test_max_solver_iterations(self):
        """
        Test maxSolverIterations priority.
        """
        # Open the USD stage
        stage = Usd.Stage.Open(str(self.ant_usda_path))
        self.assertIsNotNone(stage)

        # Find the physics scene prim
        physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
        self.assertTrue(physics_scene_prim.IsValid())
        self.assertTrue(physics_scene_prim.IsA(UsdPhysics.Scene))
        self.assertTrue(physics_scene_prim.HasAPI("NewtonSceneAPI"))

        # Create resolver
        resolver = SchemaResolverManager([SchemaResolverPhysx(), SchemaResolverNewton()])

        # newton is the only authored schema in the asset
        max_solver_iterations = resolver.get_value(physics_scene_prim, PrimType.SCENE, "max_solver_iterations")
        self.assertEqual(max_solver_iterations, 100)

        # physx can be used to override the newton value
        physics_scene_prim.CreateAttribute("physxScene:maxVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(200)
        max_solver_iterations = resolver.get_value(physics_scene_prim, PrimType.SCENE, "max_solver_iterations")
        self.assertEqual(max_solver_iterations, 200)

        # resolver priority can be reversed, so newton overrides physx
        resolver = SchemaResolverManager([SchemaResolverNewton(), SchemaResolverPhysx()])
        max_solver_iterations = resolver.get_value(physics_scene_prim, PrimType.SCENE, "max_solver_iterations")
        self.assertEqual(max_solver_iterations, 100)

        # mujoco will be converted from iterations to max_solver_iterations
        physics_scene_prim.CreateAttribute("mjc:option:iterations", Sdf.ValueTypeNames.Int).Set(300)
        resolver = SchemaResolverManager([SchemaResolverMjc(), SchemaResolverNewton()])
        max_solver_iterations = resolver.get_value(physics_scene_prim, PrimType.SCENE, "max_solver_iterations")
        self.assertEqual(max_solver_iterations, 300)

    def test_time_steps_per_second(self):
        """
        Test time_steps_per_second priority.
        """
        # Open the USD stage
        stage = Usd.Stage.Open(str(self.ant_usda_path))
        self.assertIsNotNone(stage)

        # Find the physics scene prim
        physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
        self.assertTrue(physics_scene_prim.IsValid())
        self.assertTrue(physics_scene_prim.IsA(UsdPhysics.Scene))
        self.assertTrue(physics_scene_prim.HasAPI("NewtonSceneAPI"))

        # Create resolver
        resolver = SchemaResolverManager([SchemaResolverPhysx(), SchemaResolverNewton()])

        # newton is the only authored schema in the asset
        time_steps_per_second = resolver.get_value(physics_scene_prim, PrimType.SCENE, "time_steps_per_second")
        self.assertEqual(time_steps_per_second, 120)

        # physx can be used to override the newton value
        physics_scene_prim.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.Int).Set(60)
        time_steps_per_second = resolver.get_value(physics_scene_prim, PrimType.SCENE, "time_steps_per_second")
        self.assertEqual(time_steps_per_second, 60)

        # resolver priority can be reversed, so newton overrides physx
        resolver = SchemaResolverManager([SchemaResolverNewton(), SchemaResolverPhysx()])
        time_steps_per_second = resolver.get_value(physics_scene_prim, PrimType.SCENE, "time_steps_per_second")
        self.assertEqual(time_steps_per_second, 120)

        # mujoco will be converted from time_step to time_steps_per_second
        physics_scene_prim.CreateAttribute("mjc:option:timestep", Sdf.ValueTypeNames.Float).Set(0.01)
        resolver = SchemaResolverManager([SchemaResolverMjc(), SchemaResolverNewton()])
        time_steps_per_second = resolver.get_value(physics_scene_prim, PrimType.SCENE, "time_steps_per_second")
        self.assertEqual(time_steps_per_second, 100)

    def test_gravity_enabled(self):
        """
        Test gravity_enabled priority.
        """
        # Open the USD stage
        stage = Usd.Stage.Open(str(self.ant_usda_path))
        self.assertIsNotNone(stage)

        # Find the physics scene prim
        physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
        self.assertTrue(physics_scene_prim.IsValid())
        self.assertTrue(physics_scene_prim.IsA(UsdPhysics.Scene))
        self.assertTrue(physics_scene_prim.HasAPI("NewtonSceneAPI"))

        # Create resolver
        resolver = SchemaResolverManager([SchemaResolverPhysx(), SchemaResolverNewton()])

        # there is no authored value in the asset, but the global default is True
        gravity_enabled = resolver.get_value(physics_scene_prim, PrimType.SCENE, "gravity_enabled")
        self.assertEqual(gravity_enabled, True)

        # newton can be used to override the default value
        physics_scene_prim.GetAttribute("newton:gravityEnabled").Set(False)
        gravity_enabled = resolver.get_value(physics_scene_prim, PrimType.SCENE, "gravity_enabled")
        self.assertEqual(gravity_enabled, False)

        # physx can be used to override the newton value
        physics_scene_prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(False)
        gravity_enabled = resolver.get_value(physics_scene_prim, PrimType.SCENE, "gravity_enabled")
        self.assertEqual(gravity_enabled, True)

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
        SolverMuJoCo.register_custom_attributes(builder_newton)
        builder_newton.add_usd(
            source=str(dst),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
            verbose=False,
        )

        builder_mjc = ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder_mjc)
        builder_mjc.add_usd(
            source=str(dst),
            schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton(), SchemaResolverPhysx()],
            verbose=False,
        )
        # With mjc priority and solref chosen as (0.5, 0.05):
        # Stiffness: k = 1/(timeconst^2 * dampratio^2) = 1/0.000625 = 1600.0 (800x physx/newton value of 2.0)
        # Damping: b = 2/timeconst = 2/0.5 = 4.0
        self.assertEqual(len(builder_newton.joint_limit_ke), len(builder_mjc.joint_limit_ke))
        self.assertEqual(len(builder_newton.joint_limit_kd), len(builder_mjc.joint_limit_kd))
        for physx_ke, mjc_ke in zip(builder_newton.joint_limit_ke, builder_mjc.joint_limit_ke, strict=False):
            self.assertAlmostEqual(mjc_ke / 800.0, physx_ke, places=5)

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
        result = builder.add_usd(
            source=str(dst),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
            verbose=False,
        )

        solver_attrs = result.get("schema_attrs", {})
        self.assertIn("newton", solver_attrs)

        # Body property checks
        body_path = "/ant/front_left_leg"
        self.assertIn(body_path, solver_attrs["newton"])
        self.assertIn("newton:testBodyScalar", solver_attrs["newton"][body_path])
        self.assertIn("newton:testBodyVec", solver_attrs["newton"][body_path])
        self.assertIn("newton:testBodyBool", solver_attrs["newton"][body_path])
        self.assertIn("newton:testBodyInt", solver_attrs["newton"][body_path])
        self.assertIn("newton:testBodyVec3B", solver_attrs["newton"][body_path])
        self.assertIn("newton:localmarkerRot", solver_attrs["newton"][body_path])
        self.assertAlmostEqual(solver_attrs["newton"][body_path]["newton:testBodyScalar"], 1.5, places=6)
        # also validate vector value in solver attrs
        vec_val = solver_attrs["newton"][body_path]["newton:testBodyVec"]
        self.assertAlmostEqual(float(vec_val[0]), 0.1, places=6)
        self.assertAlmostEqual(float(vec_val[1]), 0.2, places=6)
        self.assertAlmostEqual(float(vec_val[2]), 0.3, places=6)
        # Joint property checks (authored on front_left_leg joint)
        joint_name = "/ant/joints/front_left_leg"
        self.assertIn(joint_name, solver_attrs["newton"])  # solver attrs recorded
        self.assertIn("newton:testJointScalar", solver_attrs["newton"][joint_name])
        # also validate state/control joint custom attrs in solver attrs
        self.assertIn("newton:testStateJointScalar", solver_attrs["newton"][joint_name])
        self.assertIn("newton:testControlJointScalar", solver_attrs["newton"][joint_name])
        self.assertIn("newton:testStateJointBool", solver_attrs["newton"][joint_name])
        self.assertIn("newton:testControlJointInt", solver_attrs["newton"][joint_name])
        self.assertIn("newton:testJointVec", solver_attrs["newton"][joint_name])
        # new data type assertions
        self.assertIn("newton:testControlJointVec2", solver_attrs["newton"][joint_name])
        self.assertIn("newton:testJointQuat", solver_attrs["newton"][joint_name])

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

    def test_physx_schema_attrs(self):
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
        result = builder.add_usd(
            source=str(usd_path),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
            verbose=False,
        )

        solver_attrs = result.get("schema_attrs", {})
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

        resolver = SchemaResolverManager([SchemaResolverNewton(), SchemaResolverPhysx()])

        # Test 1: Authored PhysX value takes precedence over explicit default
        # physxJoint:armature = 0.02 should be returned even with explicit default
        val1 = resolver.get_value(joint_with_physx_armature, PrimType.JOINT, "armature", default=0.99)
        self.assertAlmostEqual(val1, 0.02, places=6)

        # Test 2: No Newton authored value, explicit default used
        resolver_newton_only = SchemaResolverManager([SchemaResolverNewton()])
        val2 = resolver_newton_only.get_value(joint_with_physx_armature, PrimType.JOINT, "armature", default=0.99)
        self.assertAlmostEqual(val2, 0.99, places=6)

        # Test 3: No authored value, no explicit default, use Newton mapping default
        val3 = resolver_newton_only.get_value(joint_with_physx_armature, PrimType.JOINT, "armature", default=None)
        self.assertAlmostEqual(val3, 1.0e-2, places=6)

        # Test 3b: Use SchemaResolverMjc only - should return SchemaResolverMjc armature default (0.0)
        resolver_mjc_only = SchemaResolverManager([SchemaResolverMjc()])
        val3b = resolver_mjc_only.get_value(joint_with_physx_armature, PrimType.JOINT, "armature", default=None)
        self.assertAlmostEqual(val3b, 0.0, places=6)

        # Test 4: Test priority order - PhysX first should use PhysX mapping default when no authored value
        resolver_physx_first = SchemaResolverManager([SchemaResolverPhysx(), SchemaResolverNewton()])
        val4 = resolver_physx_first.get_value(scene_prim, PrimType.SCENE, "max_solver_iterations", default=None)
        self.assertAlmostEqual(val4, 255, places=6)

        # Test same attribute with Newton first priority
        resolver_newton_first = SchemaResolverManager([SchemaResolverNewton(), SchemaResolverPhysx()])
        val5 = resolver_newton_first.get_value(scene_prim, PrimType.SCENE, "max_solver_iterations", default=None)
        self.assertEqual(val5, -1)  # there is no authored value & the schema default is -1

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
        builder.add_usd(
            source=str(usd_path),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
            verbose=False,
        )

        # Get the model and state to access joint_q and joint_qd
        model = builder.finalize()
        state = model.state()

        # Joints in ant_mixed.usda have state:angular:physics:position/velocity values

        # Check joint positions and velocities
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        joint_types = model.joint_type.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        # Map joint keys to expected values for more robust testing
        expected_joint_values = {
            "/ant/joints/front_left_leg": (10.0, 0.1),
            "/ant/joints/front_left_foot": (20.0, 0.2),
            "/ant/joints/front_right_leg": (30.0, 0.3),
            "/ant/joints/front_right_foot": (30.0, 0.3),
            "/ant/joints/left_back_leg": (40.0, 0.4),
            "/ant/joints/left_back_foot": (60.0, 0.6),
            "/ant/joints/right_back_leg": (70.0, 0.7),
            "/ant/joints/right_back_foot": (80.0, 0.8),
        }

        # Find revolute joints and validate their specific values
        revolute_joints_found = 0
        for i in range(model.joint_count):
            joint_type = joint_types[i]
            if joint_type == 1:  # JointType.REVOLUTE
                joint_key = builder.joint_key[i] if i < len(builder.joint_key) else None
                if joint_key not in expected_joint_values:
                    continue

                q_start = int(joint_q_start[i])
                qd_start = int(joint_qd_start[i])

                actual_pos = joint_q[q_start]
                actual_vel = joint_qd[qd_start]

                expected_pos_deg, expected_vel = expected_joint_values[joint_key]
                expected_pos_rad = expected_pos_deg * (3.14159 / 180.0)

                self.assertAlmostEqual(
                    actual_pos,
                    expected_pos_rad,
                    places=4,
                    msg=f"Joint {joint_key} position mismatch: expected {expected_pos_deg}°, got {actual_pos * 180 / 3.14159:.1f}°",
                )
                self.assertAlmostEqual(
                    actual_vel,
                    expected_vel,
                    places=4,
                    msg=f"Joint {joint_key} velocity mismatch: expected {expected_vel}, got {actual_vel}",
                )
                revolute_joints_found += 1

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
        builder.add_usd(
            source=str(humanoid_path),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
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

    def test_d6_dof_index_mapping_correctness(self):
        """
        Test D6 joint DOF index mapping correctness when some axes have no authored values.

        This test validates D6 DOF index mapping to ensure that dof_idx would not
        desync when some DOFs existed but had no authored initial position/velocity values.
        Uses humanoid.usda to test scenarios where D6 joints have selective axis values.

        The test ensures that:
        1. DOF indices correctly map to the actual DOF axes that were added
        2. Missing initial values don't cause index shifts for subsequent axes
        3. Only axes that were actually added as DOFs are processed
        """
        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        humanoid_path = assets_dir / "humanoid.usda"
        if not humanoid_path.exists():
            self.skipTest(f"Missing humanoid USD: {humanoid_path}")

        # Create a custom USD stage to test specific D6 DOF mapping scenarios
        if Usd is None:
            self.skipTest("USD not available")

        stage = Usd.Stage.Open(str(humanoid_path))
        self.assertIsNotNone(stage)

        # Test the specific case that would trigger the bug:
        # Find a D6 joint and verify its DOF mapping behavior
        builder = ModelBuilder()
        builder.add_usd(
            source=str(humanoid_path),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
            verbose=False,
        )

        model = builder.finalize()
        state = model.state()

        # Get joint data
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        joint_types = model.joint_type.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        # Test specific D6 joints that have selective axis values
        # Joint 7 (left_thigh) has rotX, rotY, rotZ values: (-10°, 0.1), (-50°, 0.5), (25°, 0.25)
        # Joint 9 (left_foot) has only rotX, rotY values: (30°, 0.3), (-30°, 0.4) - missing rotZ
        # Joint 10 (right_thigh) has rotX, rotY, rotZ values: (5°, 0.05), (20°, 0.2), (-30°, 0.3)

        test_cases = [
            {
                "joint_idx": 7,  # left_thigh - has all 3 rotational DOFs
                "expected_values": [(-10.0, 0.1), (-50.0, 0.5), (25.0, 0.25)],
                "description": "D6 joint with all rotational DOFs authored",
            },
            {
                "joint_idx": 9,  # left_foot - has only 2 rotational DOFs
                "expected_values": [(30.0, 0.3), (-30.0, 0.4)],
                "description": "D6 joint with partial rotational DOFs authored",
            },
            {
                "joint_idx": 10,  # right_thigh - has all 3 rotational DOFs
                "expected_values": [(5.0, 0.05), (20.0, 0.2), (-30.0, 0.3)],
                "description": "D6 joint with all rotational DOFs authored (different values)",
            },
        ]

        validated_joints = 0

        for test_case in test_cases:
            joint_idx = test_case["joint_idx"]
            expected_values = test_case["expected_values"]
            description = test_case["description"]

            if joint_idx >= len(joint_types):
                continue

            joint_type = joint_types[joint_idx]
            if joint_type != 6:  # Not a D6 joint
                continue

            q_start = int(joint_q_start[joint_idx])
            qd_start = int(joint_qd_start[joint_idx])

            # Get DOF count for this joint
            if joint_idx + 1 < len(joint_q_start):
                qd_end = int(joint_qd_start[joint_idx + 1])
            else:
                qd_end = len(joint_qd)

            dof_count = qd_end - qd_start

            # Validate that we have the expected number of DOFs
            self.assertEqual(
                dof_count, len(expected_values), f"{description}: Expected {len(expected_values)} DOFs, got {dof_count}"
            )

            # Validate each DOF maps to the correct expected value
            for dof_idx in range(dof_count):
                expected_pos_deg, expected_vel = expected_values[dof_idx]
                expected_pos_rad = expected_pos_deg * (3.14159 / 180.0)

                actual_pos = joint_q[q_start + dof_idx]
                actual_vel = joint_qd[qd_start + dof_idx]

                # This is the critical test: if DOF indices were incorrectly mapped,
                # these values would be wrong or zero
                self.assertAlmostEqual(
                    actual_pos,
                    expected_pos_rad,
                    places=4,
                    msg=f"{description}: Joint {joint_idx} DOF {dof_idx} position mapping incorrect. "
                    f"Expected {expected_pos_deg}° ({expected_pos_rad:.4f} rad), got {actual_pos:.4f} rad",
                )
                self.assertAlmostEqual(
                    actual_vel,
                    expected_vel,
                    places=4,
                    msg=f"{description}: Joint {joint_idx} DOF {dof_idx} velocity mapping incorrect. "
                    f"Expected {expected_vel}, got {actual_vel}",
                )

            validated_joints += 1

        # Ensure we actually tested some joints
        self.assertGreater(
            validated_joints, 0, "Should have validated at least one D6 joint for DOF index mapping correctness"
        )

        # Additional verification: check that joints with missing axes don't have incorrect values
        # Joint 9 (left_foot) should only have 2 DOFs, not 3, so accessing a 3rd DOF should be invalid
        joint_9_qd_start = int(joint_qd_start[9])
        joint_9_qd_end = int(joint_qd_start[10]) if 10 < len(joint_qd_start) else len(joint_qd)
        joint_9_dof_count = joint_9_qd_end - joint_9_qd_start

        # This joint should have exactly 2 DOFs (rotX, rotY), not 3
        self.assertEqual(
            joint_9_dof_count,
            2,
            f"Joint 9 (left_foot) should have 2 DOFs, got {joint_9_dof_count}. "
            "This indicates the DOF mapping fix is working correctly.",
        )

    def test_attribute_parsing(self):
        """
        Test that both Newton and MuJoCo custom attributes are correctly parsed and collected.
        """
        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        ant_mixed_path = assets_dir / "ant_mixed.usda"
        self.assertTrue(ant_mixed_path.exists(), f"Missing mixed USD: {ant_mixed_path}")

        # Test with all three plugins to ensure attribute collection works
        builder = ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        result = builder.add_usd(
            source=str(ant_mixed_path),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
            verbose=False,
        )

        solver_attrs = result.get("schema_attrs", {})

        # Verify Newton attributes are collected
        self.assertIn("newton", solver_attrs, "Newton solver attributes should be collected")
        newton_attrs = solver_attrs["newton"]
        joint_path = "/ant/joints/front_left_leg"
        self.assertIn(joint_path, newton_attrs, f"Newton attributes should be found on {joint_path}")

        # Check specific Newton custom attributes
        newton_joint_attrs = newton_attrs[joint_path]
        self.assertIn("newton:testJointScalar", newton_joint_attrs)
        self.assertAlmostEqual(newton_joint_attrs["newton:testJointScalar"], 2.25, places=2)
        self.assertIn("newton:testJointVec", newton_joint_attrs)

        # Verify MuJoCo attributes are collected
        self.assertIn("mjc", solver_attrs, "MuJoCo solver attributes should be collected")
        mjc_attrs = solver_attrs["mjc"]
        self.assertIn(joint_path, mjc_attrs, f"MuJoCo attributes should be found on {joint_path}")

        # Check specific MuJoCo custom attributes
        mjc_joint_attrs = mjc_attrs[joint_path]
        self.assertIn("mjc:model:joint:testMjcJointScalar", mjc_joint_attrs)
        self.assertAlmostEqual(mjc_joint_attrs["mjc:model:joint:testMjcJointScalar"], 3.14, places=2)
        self.assertIn("mjc:state:joint:testMjcJointVec3", mjc_joint_attrs)
        mjc_vec = mjc_joint_attrs["mjc:state:joint:testMjcJointVec3"]
        self.assertAlmostEqual(float(mjc_vec[0]), 1.0, places=1)
        self.assertAlmostEqual(float(mjc_vec[1]), 2.0, places=1)
        self.assertAlmostEqual(float(mjc_vec[2]), 3.0, places=1)

    def test_namespaced_custom_attributes(self):
        """
        Test that custom attributes with namespaces are isolated from default namespace attributes.

        This test verifies:
        1. Attributes with the same name in different namespaces are treated as separate attributes
        2. Each namespace maintains its own values independent of other namespaces
        3. After finalization, separate attribute objects are created for each namespace
        4. Namespace attributes are accessible via namespace prefix on model/state/control objects
        """
        test_dir = Path(__file__).parent
        assets_dir = test_dir / "assets"
        ant_mixed_path = assets_dir / "ant_mixed.usda"
        self.assertTrue(ant_mixed_path.exists(), f"Missing mixed USD: {ant_mixed_path}")

        builder = ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        result = builder.add_usd(
            source=str(ant_mixed_path),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
            verbose=False,
        )

        model = builder.finalize()
        state = model.state()
        control = model.control()

        body_map = result["path_body_map"]
        body_path = "/ant/front_left_leg"
        self.assertIn(body_path, body_map)
        body_idx = body_map[body_path]

        joint_name = "/ant/joints/front_left_leg"
        self.assertIn(joint_name, builder.joint_key)
        joint_idx = builder.joint_key.index(joint_name)

        # Test 1: Verify that testBodyScalar exists in both default and namespace_a
        # Default namespace: newton:testBodyScalar = 1.5 (model assignment)
        self.assertTrue(hasattr(model, "testBodyScalar"), "Default namespace testBodyScalar should exist on model")
        default_body_scalar = model.testBodyScalar.numpy()
        self.assertAlmostEqual(
            float(default_body_scalar[body_idx]), 1.5, places=6, msg="Default namespace testBodyScalar should be 1.5"
        )

        # Namespace_a: newton:namespace_a:testBodyScalar = 2.5 (model assignment)
        self.assertTrue(hasattr(model, "namespace_a"), "namespace_a should exist on model")
        self.assertTrue(hasattr(model.namespace_a, "testBodyScalar"), "testBodyScalar should exist in namespace_a")
        namespaced_body_scalar = model.namespace_a.testBodyScalar.numpy()
        self.assertAlmostEqual(
            float(namespaced_body_scalar[body_idx]),
            2.5,
            places=6,
            msg="namespace_a testBodyScalar should be 2.5 (different from default)",
        )

        # Test 2: Verify that testBodyInt exists in both default and namespace_b with different assignments
        # Default namespace: newton:testBodyInt = 7 (model assignment)
        self.assertTrue(hasattr(model, "testBodyInt"), "Default namespace testBodyInt should exist on model")
        default_body_int = model.testBodyInt.numpy()
        self.assertEqual(int(default_body_int[body_idx]), 7, msg="Default namespace testBodyInt should be 7")

        # Namespace_b: newton:namespace_b:testBodyInt = 42 (state assignment)
        self.assertTrue(hasattr(state, "namespace_b"), "namespace_b should exist on state")
        self.assertTrue(hasattr(state.namespace_b, "testBodyInt"), "testBodyInt should exist in namespace_b on state")
        namespaced_body_int = state.namespace_b.testBodyInt.numpy()
        self.assertEqual(
            int(namespaced_body_int[body_idx]), 42, msg="namespace_b testBodyInt should be 42 (different from default)"
        )

        # Test 3: Verify that testJointVec exists in both default and namespace_a with different assignments
        # Default namespace: newton:testJointVec = (0.5, 0.6, 0.7) (model assignment)
        self.assertTrue(hasattr(model, "testJointVec"), "Default namespace testJointVec should exist on model")
        default_joint_vec = model.testJointVec.numpy()
        self.assertAlmostEqual(float(default_joint_vec[joint_idx, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(default_joint_vec[joint_idx, 1]), 0.6, places=6)
        self.assertAlmostEqual(float(default_joint_vec[joint_idx, 2]), 0.7, places=6)

        # Namespace_a: newton:namespace_a:testJointVec = (1.5, 2.5, 3.5) (control assignment)
        self.assertTrue(hasattr(control, "namespace_a"), "namespace_a should exist on control")
        self.assertTrue(
            hasattr(control.namespace_a, "testJointVec"), "testJointVec should exist in namespace_a on control"
        )
        namespaced_joint_vec = control.namespace_a.testJointVec.numpy()
        self.assertAlmostEqual(
            float(namespaced_joint_vec[joint_idx, 0]), 1.5, places=6, msg="namespace_a testJointVec[0] should be 1.5"
        )
        self.assertAlmostEqual(
            float(namespaced_joint_vec[joint_idx, 1]), 2.5, places=6, msg="namespace_a testJointVec[1] should be 2.5"
        )
        self.assertAlmostEqual(
            float(namespaced_joint_vec[joint_idx, 2]), 3.5, places=6, msg="namespace_a testJointVec[2] should be 3.5"
        )

        # Test 4: Verify unique namespace attributes that don't exist in default namespace
        # namespace_a:uniqueBodyAttr = 100.0 (state assignment)
        self.assertTrue(hasattr(state, "namespace_a"), "namespace_a should exist on state")
        self.assertTrue(
            hasattr(state.namespace_a, "uniqueBodyAttr"), "uniqueBodyAttr should exist in namespace_a on state"
        )
        unique_body_attr = state.namespace_a.uniqueBodyAttr.numpy()
        self.assertAlmostEqual(float(unique_body_attr[body_idx]), 100.0, places=6)

        # namespace_b:uniqueJointAttr = 999.0 (model assignment)
        self.assertTrue(hasattr(model, "namespace_b"), "namespace_b should exist on model")
        self.assertTrue(
            hasattr(model.namespace_b, "uniqueJointAttr"), "uniqueJointAttr should exist in namespace_b on model"
        )
        unique_joint_attr = model.namespace_b.uniqueJointAttr.numpy()
        self.assertAlmostEqual(float(unique_joint_attr[joint_idx]), 999.0, places=6)

        # Test 5: Verify that default namespace attributes don't have the unique namespace attributes
        self.assertFalse(
            hasattr(model, "uniqueBodyAttr"), "uniqueBodyAttr should NOT exist in default namespace on model"
        )
        self.assertFalse(
            hasattr(state, "uniqueBodyAttr"), "uniqueBodyAttr should NOT exist in default namespace on state"
        )
        self.assertFalse(
            hasattr(model, "uniqueJointAttr"), "uniqueJointAttr should NOT exist in default namespace on model"
        )
        self.assertFalse(
            hasattr(control, "uniqueJointAttr"), "uniqueJointAttr should NOT exist in default namespace on control"
        )

    def test_articulation_frequency_attributes(self):
        """
        Test ARTICULATION frequency attributes from USD import.

        Uses ant_mixed.usda which has an articulation with PhysicsArticulationRootAPI
        and tests that custom articulation attributes are correctly parsed and materialized.
        """
        test_dir = Path(__file__).parent
        ant_usd_path = test_dir / "assets" / "ant_mixed.usda"

        # Import the ant USD file
        builder = ModelBuilder()
        builder.add_usd(
            source=str(ant_usd_path),
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
            verbose=False,
        )

        # Finalize the model
        model = builder.finalize()
        state = model.state()
        control = model.control()

        # Validate ARTICULATION frequency attributes exist
        self.assertTrue(hasattr(model, "articulation_default_stiffness"))
        self.assertTrue(hasattr(state, "articulation_default_damping"))

        # Check attribute frequencies
        self.assertEqual(
            model.get_attribute_frequency("articulation_default_stiffness"), AttributeFrequency.ARTICULATION
        )
        self.assertEqual(model.get_attribute_frequency("articulation_default_damping"), AttributeFrequency.ARTICULATION)

        # Validate namespaced attributes
        self.assertTrue(hasattr(control, "pd_control"))
        self.assertTrue(hasattr(control.pd_control, "articulation_default_pd_gains"))

        # Check that the ant articulation has the custom attribute values we set
        # The ant USD file defines:
        #   - articulation_stiffness = 150.0 (on ant Xform prim)
        #   - articulation_damping = 15.0 (on ant Xform prim)
        #   - pd_control:pd_gains = (2.0, 0.2) (on ant Xform prim)
        arctic_stiff = model.articulation_default_stiffness.numpy()
        arctic_damp = state.articulation_default_damping.numpy()
        pd_gains = control.pd_control.articulation_default_pd_gains.numpy()

        # The ant is the first (and likely only) articulation
        self.assertGreater(len(arctic_stiff), 0)
        self.assertAlmostEqual(arctic_stiff[0], 150.0, places=5)
        self.assertAlmostEqual(arctic_damp[0], 15.0, places=5)
        self.assertAlmostEqual(pd_gains[0][0], 2.0, places=5)
        self.assertAlmostEqual(pd_gains[0][1], 0.2, places=5)

    def test_contact_margin(self):
        """
        Test contact_margin priority.
        """
        stage = Usd.Stage.CreateInMemory()
        xform = UsdGeom.Xform.Define(stage, "/xform").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(xform)
        collider = UsdGeom.Cube.Define(stage, "/xform/collider").GetPrim()
        collider.ApplyAPI("NewtonCollisionAPI")
        self.assertTrue(collider.HasAPI("NewtonCollisionAPI"))
        self.assertTrue(collider.HasAPI("PhysicsCollisionAPI"))
        self.assertTrue(UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get())
        UsdPhysics.CollisionAPI.Apply(collider)

        resolver = SchemaResolverManager([SchemaResolverPhysx(), SchemaResolverNewton()])

        # Explicit newton:contactMargin -> "margin"
        collider.GetAttribute("newton:contactMargin").Set(0.2)
        margin = resolver.get_value(collider, PrimType.SHAPE, "margin")
        self.assertAlmostEqual(margin, 0.2)

        # PhysX restOffset overrides margin when PhysX is first
        collider.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.15)
        margin = resolver.get_value(collider, PrimType.SHAPE, "margin")
        self.assertAlmostEqual(margin, 0.15)

        # Newton first: margin from Newton again
        resolver = SchemaResolverManager([SchemaResolverNewton(), SchemaResolverPhysx()])
        margin = resolver.get_value(collider, PrimType.SHAPE, "margin")
        self.assertAlmostEqual(margin, 0.2)

        # Mjc first: mjc:margin (geom margin) used for "margin"
        resolver = SchemaResolverManager([SchemaResolverMjc(), SchemaResolverNewton()])
        collider.CreateAttribute("mjc:margin", Sdf.ValueTypeNames.Float).Set(0.4)
        margin = resolver.get_value(collider, PrimType.SHAPE, "margin")
        self.assertAlmostEqual(margin, 0.4)

        builder = ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        result = builder.add_usd(
            source=stage,
            schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()],
            verbose=False,
        )
        schema_attrs = result.get("schema_attrs", {})
        self.assertAlmostEqual(schema_attrs["mjc"]["/xform/collider"]["mjc:margin"], 0.4)

    def test_contact_gap(self):
        """
        Test gap (contact processing distance) priority: Newton, PhysX contactOffset, Mjc.
        """
        stage = Usd.Stage.CreateInMemory()
        xform = UsdGeom.Xform.Define(stage, "/xform").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(xform)
        collider = UsdGeom.Cube.Define(stage, "/xform/collider").GetPrim()
        collider.ApplyAPI("NewtonCollisionAPI")
        UsdPhysics.CollisionAPI.Apply(collider)

        resolver = SchemaResolverManager([SchemaResolverPhysx(), SchemaResolverNewton()])

        collider.CreateAttribute("newton:contactGap", Sdf.ValueTypeNames.Float).Set(0.02)
        gap = resolver.get_value(collider, PrimType.SHAPE, "gap")
        self.assertAlmostEqual(gap, 0.02)

        collider.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.03)
        gap = resolver.get_value(collider, PrimType.SHAPE, "gap")
        self.assertAlmostEqual(gap, 0.03)

        resolver = SchemaResolverManager([SchemaResolverNewton(), SchemaResolverPhysx()])
        gap = resolver.get_value(collider, PrimType.SHAPE, "gap")
        self.assertAlmostEqual(gap, 0.02)

        resolver = SchemaResolverManager([SchemaResolverMjc(), SchemaResolverNewton()])
        collider.CreateAttribute("mjc:gap", Sdf.ValueTypeNames.Float).Set(0.01)
        gap = resolver.get_value(collider, PrimType.SHAPE, "gap")
        self.assertAlmostEqual(gap, 0.01)

    def test_max_hull_vertices(self):
        """
        Test max_hull_vertices priority.
        """
        stage = Usd.Stage.CreateInMemory()
        xform = UsdGeom.Xform.Define(stage, "/xform").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(xform)
        collider = UsdGeom.Mesh.Define(stage, "/xform/collider").GetPrim()
        collider.ApplyAPI("NewtonMeshCollisionAPI")
        self.assertTrue(collider.HasAPI("NewtonMeshCollisionAPI"))
        self.assertTrue(collider.HasAPI("NewtonCollisionAPI"))
        self.assertTrue(collider.HasAPI("PhysicsCollisionAPI"))

        # Create resolver
        resolver = SchemaResolverManager([SchemaResolverPhysx(), SchemaResolverNewton()])

        # there is no authored max_hull_vertices in the asset, so it should be the physx default (64)
        max_hull_vertices = resolver.get_value(collider, PrimType.SHAPE, "max_hull_vertices")
        self.assertEqual(max_hull_vertices, 64)

        # an explicit newton value should be used
        collider.GetAttribute("newton:maxHullVertices").Set(32)
        max_hull_vertices = resolver.get_value(collider, PrimType.SHAPE, "max_hull_vertices")
        self.assertEqual(max_hull_vertices, 32)

        # an explicit physx value should override the newton value
        collider.CreateAttribute("physxConvexHullCollision:hullVertexLimit", Sdf.ValueTypeNames.Int).Set(64)
        max_hull_vertices = resolver.get_value(collider, PrimType.SHAPE, "max_hull_vertices")
        self.assertEqual(max_hull_vertices, 64)

        # reversed resolver priority should use the newton value
        resolver = SchemaResolverManager([SchemaResolverNewton(), SchemaResolverPhysx()])
        max_hull_vertices = resolver.get_value(collider, PrimType.SHAPE, "max_hull_vertices")
        self.assertEqual(max_hull_vertices, 32)

        # mujoco mjc:maxhullvert is equivalent to max_hull_vertices
        resolver = SchemaResolverManager([SchemaResolverMjc(), SchemaResolverNewton()])
        collider.CreateAttribute("mjc:maxhullvert", Sdf.ValueTypeNames.Int).Set(128)
        max_hull_vertices = resolver.get_value(collider, PrimType.SHAPE, "max_hull_vertices")
        self.assertEqual(max_hull_vertices, 128)

        # with mujoco lower priority, newton value should be used
        resolver = SchemaResolverManager([SchemaResolverNewton(), SchemaResolverMjc()])
        max_hull_vertices = resolver.get_value(collider, PrimType.SHAPE, "max_hull_vertices")
        self.assertEqual(max_hull_vertices, 32)

    def test_material_friction_attributes(self):
        """
        Test mu_rolling and mu_torsional priority on materials.
        """

        stage = Usd.Stage.CreateInMemory()
        material = UsdShade.Material.Define(stage, "/material").GetPrim()
        material.ApplyAPI("NewtonMaterialAPI")
        self.assertTrue(material.HasAPI("NewtonMaterialAPI"))
        self.assertTrue(material.HasAPI("PhysicsMaterialAPI"))

        # Create resolver with Newton priority
        resolver = SchemaResolverManager([SchemaResolverNewton()])

        # there is no authored value, so it should return the default (0)
        rolling = resolver.get_value(material, PrimType.MATERIAL, "mu_rolling")
        torsional = resolver.get_value(material, PrimType.MATERIAL, "mu_torsional")
        self.assertEqual(rolling, 0.0005)
        self.assertEqual(torsional, 0.25)

        # an explicit newton value should be used
        material.GetAttribute("newton:rollingFriction").Set(0.1)
        material.GetAttribute("newton:torsionalFriction").Set(0.2)
        rolling = resolver.get_value(material, PrimType.MATERIAL, "mu_rolling")
        torsional = resolver.get_value(material, PrimType.MATERIAL, "mu_torsional")
        self.assertAlmostEqual(rolling, 0.1)
        self.assertAlmostEqual(torsional, 0.2)

        # mujoco mjc:rollingfriction and mjc:torsionalfriction are equivalent
        resolver = SchemaResolverManager([SchemaResolverMjc(), SchemaResolverNewton()])
        material.CreateAttribute("mjc:rollingfriction", Sdf.ValueTypeNames.Float).Set(0.3)
        material.CreateAttribute("mjc:torsionalfriction", Sdf.ValueTypeNames.Float).Set(0.4)
        rolling = resolver.get_value(material, PrimType.MATERIAL, "mu_rolling")
        torsional = resolver.get_value(material, PrimType.MATERIAL, "mu_torsional")
        self.assertAlmostEqual(rolling, 0.3)
        self.assertAlmostEqual(torsional, 0.4)

        # with mujoco lower priority, newton values should be used
        resolver = SchemaResolverManager([SchemaResolverNewton(), SchemaResolverMjc()])
        rolling = resolver.get_value(material, PrimType.MATERIAL, "mu_rolling")
        torsional = resolver.get_value(material, PrimType.MATERIAL, "mu_torsional")
        self.assertAlmostEqual(rolling, 0.1)
        self.assertAlmostEqual(torsional, 0.2)

        # physx does not have these attributes, so newton values should still be used
        resolver = SchemaResolverManager([SchemaResolverPhysx(), SchemaResolverNewton()])
        rolling = resolver.get_value(material, PrimType.MATERIAL, "mu_rolling")
        torsional = resolver.get_value(material, PrimType.MATERIAL, "mu_torsional")
        self.assertAlmostEqual(rolling, 0.1)
        self.assertAlmostEqual(torsional, 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
