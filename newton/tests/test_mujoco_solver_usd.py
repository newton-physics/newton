# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE

try:
    import newton_usd_schemas  # noqa: F401
except ImportError:
    pass


def _has_mujoco_scene_api() -> bool:
    """Return whether the installed schema package contains ``NewtonMuJoCoSceneAPI``."""
    if not USD_AVAILABLE:
        return False
    from pxr import Usd

    return bool(Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("NewtonMuJoCoSceneAPI"))


def _stage_source(scene_attrs: str = "", scene_apis: str = '"NewtonMuJoCoSceneAPI"') -> str:
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World"
{{
    def PhysicsScene "PhysicsScene" (
        prepend apiSchemas = [{scene_apis}]
    )
    {{
{scene_attrs}
    }}

    def Xform "Articulation" (
        prepend apiSchemas = ["PhysicsArticulationRootAPI"]
    )
    {{
        def Xform "Base" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        )
        {{
            float physics:mass = 1.0
            float3 physics:diagonalInertia = (0.1, 0.1, 0.1)
        }}

        def PhysicsFixedJoint "Anchor"
        {{
            rel physics:body0 = </World/Articulation/Base>
        }}

        def Xform "Body" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        )
        {{
            double3 xformOp:translate = (0, 0, 1)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            float physics:mass = 1.0

            def Sphere "Geom" (
                prepend apiSchemas = ["PhysicsCollisionAPI"]
            )
            {{
                double radius = 0.1
            }}
        }}

        def PhysicsRevoluteJoint "Joint"
        {{
            rel physics:body0 = </World/Articulation/Base>
            rel physics:body1 = </World/Articulation/Body>
            uniform token physics:axis = "Y"
            point3f physics:localPos0 = (0, 0, 2)
            point3f physics:localPos1 = (0, 0, 1)
        }}
    }}
}}
"""


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
@unittest.skipUnless(_has_mujoco_scene_api(), "Requires newton-usd-schemas with NewtonMuJoCoSceneAPI")
class TestSolverMuJoCoCreateFromUsd(unittest.TestCase):
    def _write(self, source: str) -> str:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, "scene.usda")
        with open(path, "w", encoding="utf-8") as file:
            file.write(source)
        return path

    def _build_model(self, path: str) -> newton.Model:
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(path)
        return builder.finalize()

    def setUp(self):
        self._stages = []

    def _stage(self, path: str):
        from pxr import Usd

        stage = Usd.Stage.Open(path)
        self._stages.append(stage)  # prims are only valid while their stage is alive
        return stage

    def _scene(self, path: str):
        from pxr import UsdPhysics

        return next(prim for prim in self._stage(path).Traverse() if prim.IsA(UsdPhysics.Scene))

    def test_missing_api_raises(self):
        """Reject a stage whose scene does not apply NewtonMuJoCoSceneAPI."""
        path = self._write(_stage_source(scene_apis='"NewtonSceneAPI"'))
        with self.assertRaisesRegex(ValueError, "NewtonMuJoCoSceneAPI is not applied"):
            SolverMuJoCo.create_from_usd(self._build_model(path), self._scene(path))

    def test_scene_prim_selection(self):
        """Configure the solver from the scene prim that is passed in."""
        from pxr import UsdPhysics

        path = self._write(
            """#usda 1.0
(
    defaultPrim = "World"
)

class Xform "Proto"
{
    def PhysicsScene "Scene" (
        prepend apiSchemas = ["NewtonMuJoCoSceneAPI"]
    )
    {
        uniform bool newton:mujoco:useMujocoCpu = true
        uniform int newton:mujoco:updateDataInterval = 4
    }
}

def Xform "World"
{
    def PhysicsScene "A_Mujoco" (
        prepend apiSchemas = ["NewtonMuJoCoSceneAPI"]
    )
    {
        uniform bool newton:mujoco:useMujocoCpu = true
        uniform int newton:mujoco:updateDataInterval = 2
    }

    def PhysicsScene "B_Mujoco" (
        prepend apiSchemas = ["NewtonMuJoCoSceneAPI"]
    )
    {
        uniform bool newton:mujoco:useMujocoCpu = true
        uniform int newton:mujoco:updateDataInterval = 5
    }

    def Xform "Instance" (
        instanceable = true
        references = </Proto>
    )
    {
    }
}
"""
        )
        stage = self._stage(path)
        instance_scene = stage.GetPrimAtPath("/World/Instance/Scene")
        self.assertTrue(instance_scene.IsInstanceProxy())
        # The scenes carry no bodies, so simulate the articulation from a separate stage.
        model = self._build_model(self._write(_stage_source()))
        cases = (
            ("first scene", stage.GetPrimAtPath("/World/A_Mujoco"), 2),
            ("second scene", stage.GetPrimAtPath("/World/B_Mujoco"), 5),
            ("typed scene", UsdPhysics.Scene(stage.GetPrimAtPath("/World/B_Mujoco")), 5),
            ("instance proxy", instance_scene, 4),
        )
        for name, scene, expected in cases:
            with self.subTest(scene=name):
                solver = SolverMuJoCo.create_from_usd(model, scene)
                self.assertEqual(solver.update_data_interval, expected)

    def test_non_scene_prim_raises(self):
        """Reject a prim that is not a ``UsdPhysics.Scene``, and a missing prim."""
        path = self._write(_stage_source())
        model = newton.ModelBuilder().finalize()
        articulation = self._stage(path).GetPrimAtPath("/World/Articulation")
        for prim in (articulation, None):
            with self.subTest(prim=prim), self.assertRaisesRegex(TypeError, "UsdPhysics.Scene"):
                SolverMuJoCo.create_from_usd(model, prim)

    def test_overrides_take_precedence(self):
        """Pass keyword overrides to the constructor ahead of authored values."""
        path = self._write(
            _stage_source(
                """
        uniform bool newton:mujoco:useMujocoCpu = true
        uniform int newton:mujoco:updateDataInterval = 3
        uniform int newton:mjwarp:njmax = 123
        """
            )
        )
        model = self._build_model(path)
        scene = self._scene(path)

        solver = SolverMuJoCo.create_from_usd(model, scene, update_data_interval=5, iterations=7)
        self.assertEqual(solver.update_data_interval, 5)  # override replaces the authored value
        self.assertEqual(solver.mj_model.opt.iterations, 7)  # override of an option that is not authored
        self.assertTrue(solver.use_mujoco_cpu)  # authored values without an override are kept
        self.assertEqual(solver.mjw_data.njmax, 123)

        # An explicit None replaces the authored value with the constructor default.
        solver = SolverMuJoCo.create_from_usd(model, scene, njmax=None)
        reference = SolverMuJoCo(model, use_mujoco_cpu=True)
        self.assertEqual(solver.mjw_data.njmax, reference.mjw_data.njmax)

        # Unknown keywords are rejected by the constructor.
        with self.assertRaises(TypeError):
            SolverMuJoCo.create_from_usd(model, scene, not_an_option=1)

    def test_invalid_value_raises(self):
        """Reject out-of-range authored integers."""
        path = self._write(_stage_source("uniform int newton:mjwarp:nconmax = -5"))
        with self.assertRaisesRegex(ValueError, "nconmax"):
            SolverMuJoCo.create_from_usd(self._build_model(path), self._scene(path))

    def test_invalid_deterministic_raises(self):
        """Reject an unknown determinism token."""
        path = self._write(_stage_source('uniform token newton:deterministic = "bogus"'))
        with self.assertRaisesRegex(ValueError, "deterministic"):
            SolverMuJoCo.create_from_usd(self._build_model(path), self._scene(path))

    def test_cpu_solver_values(self):
        """Check authored values on a real MuJoCo CPU solver."""
        try:
            mujoco, _ = SolverMuJoCo.import_mujoco()
        except ImportError:
            self.skipTest("MuJoCo is not installed")
        path = self._write(
            _stage_source(
                """
        uniform bool newton:mujoco:useMujocoCpu = true
        uniform bool newton:mujoco:useMujocoContacts = true
        uniform int newton:mujoco:updateDataInterval = 2
        uniform int newton:mjwarp:njmax = -1
        uniform token newton:deterministic = "runToRun"
        uniform int mjc:option:iterations = 7
        uniform int mjc:option:ls_iterations = 9
        uniform double mjc:option:tolerance = 1e-6
        uniform double mjc:option:impratio = 2.0
        uniform token mjc:option:cone = "elliptic"
        uniform token mjc:option:solver = "cg"
        uniform token mjc:option:integrator = "euler"
        uniform bool mjc:flag:multiccd = true
        uniform bool mjc:flag:contact = false
        uniform bool mjc:flag:sensor = false
        """,
                scene_apis='"NewtonMuJoCoSceneAPI", "MjcSceneAPI"',
            )
        )
        model = self._build_model(path)
        solver = SolverMuJoCo.create_from_usd(model, self._scene(path))
        opt = solver.mj_model.opt
        self.assertIs(solver.model, model)
        self.assertTrue(solver.use_mujoco_cpu)
        self.assertTrue(solver._use_mujoco_contacts)
        self.assertEqual(solver.update_data_interval, 2)
        self.assertEqual(solver._deterministic, wp.DeterministicMode.RUN_TO_RUN)
        # -1 requests the solver's automatic capacity, as if njmax were unauthored
        reference = SolverMuJoCo(model, use_mujoco_cpu=True)
        self.assertEqual(solver.mjw_data.njmax, reference.mjw_data.njmax)
        self.assertEqual(opt.iterations, 7)
        self.assertEqual(opt.ls_iterations, 9)
        self.assertAlmostEqual(opt.tolerance, 1e-6, places=9)
        self.assertAlmostEqual(opt.impratio, 2.0)
        self.assertEqual(opt.cone, mujoco.mjtCone.mjCONE_ELLIPTIC)
        self.assertEqual(opt.solver, mujoco.mjtSolver.mjSOL_CG)
        self.assertEqual(opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)
        # multiccd is enabled by clearing its disable bit; contacts and sensors are disabled
        self.assertFalse(opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        self.assertTrue(opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.assertTrue(opt.disableflags & mujoco.mjtDisableBit.mjDSBL_SENSOR)

    def test_cpu_solver_defaults(self):
        """Leave the constructor defaults in place when nothing is authored."""
        try:
            mujoco, _ = SolverMuJoCo.import_mujoco()
        except ImportError:
            self.skipTest("MuJoCo is not installed")
        path = self._write(_stage_source("uniform bool newton:mujoco:useMujocoCpu = true"))
        solver = SolverMuJoCo.create_from_usd(self._build_model(path), self._scene(path))
        opt = solver.mj_model.opt
        self.assertEqual(solver.update_data_interval, 1)
        self.assertTrue(solver._use_mujoco_contacts)
        self.assertTrue(opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        self.assertFalse(opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.assertFalse(opt.disableflags & mujoco.mjtDisableBit.mjDSBL_SENSOR)
        self.assertFalse(solver.enable_sleeping)

    def test_gpu_solver_values(self):
        """Check capacity and sleeping values on a real MuJoCo Warp solver."""
        if not wp.get_cuda_device_count():
            self.skipTest("Requires a CUDA device")
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError:
            self.skipTest("MuJoCo is not installed")
        path = self._write(
            _stage_source(
                """
        uniform int newton:mjwarp:nconmax = 64
        uniform int newton:mjwarp:njmax = 128
        uniform int newton:mjwarp:nvmax = 1
        uniform double mjc:option:sleep_tolerance = 0.02
        """,
                scene_apis='"NewtonMuJoCoSceneAPI", "MjcSceneAPI"',
            )
        )
        model = self._build_model(path)
        # NewtonMuJoCoSceneAPI has no sleeping flag, so enable sleeping through the model attribute.
        model.mujoco.enable_sleeping.assign(np.array([True]))
        solver = SolverMuJoCo.create_from_usd(model, self._scene(path))
        self.assertFalse(solver.use_mujoco_cpu)
        self.assertEqual(solver.mjw_data.naconmax, 64)
        self.assertEqual(solver.mjw_data.njmax, 128)
        self.assertTrue(solver.enable_sleeping)
        self.assertEqual(solver.nvmax, 1)
        self.assertAlmostEqual(float(solver.mjw_model.opt.sleep_tolerance.numpy()[0]), 0.02, places=6)

    def test_schema_defaults_match_constructor_defaults(self):
        """Each schema fallback value corresponds to the constructor's default argument."""
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        scene = UsdPhysics.Scene.Define(stage, "/Scene").GetPrim()
        scene.ApplyAPI("NewtonMuJoCoSceneAPI")
        parameters = inspect.signature(SolverMuJoCo.__init__).parameters

        # (attribute, constructor argument, schema value that means "use the constructor default")
        automatic = {
            "njmax": "njmax",
            "njmax_nnz": "njmax_nnz",
            "nconmax": "nconmax",
            "nvmax": "nvmax",
        }
        plain = {
            "useMujocoCpu": "use_mujoco_cpu",
            "useMujocoContacts": "use_mujoco_contacts",
            "updateDataInterval": "update_data_interval",
            "includeSites": "include_sites",
            "skipVisualOnlyGeoms": "skip_visual_only_geoms",
        }
        for name, argument in automatic.items():
            with self.subTest(attribute=name):
                self.assertEqual(scene.GetAttribute(f"newton:mjwarp:{name}").Get(), -1)
                self.assertIsNone(parameters[argument].default)
        for name, argument in plain.items():
            with self.subTest(attribute=name):
                self.assertEqual(scene.GetAttribute(f"newton:mujoco:{name}").Get(), parameters[argument].default)
        # "inherit" is the schema's spelling of deterministic=None
        self.assertEqual(scene.GetAttribute("newton:deterministic").Get(), "inherit")
        self.assertIsNone(parameters["deterministic"].default)

    def test_unauthored_attributes_match_default_solver(self):
        """A scene that authors nothing yields the same solver as ``SolverMuJoCo(model)``."""
        if not wp.get_cuda_device_count():
            self.skipTest("Requires a CUDA device")
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError:
            self.skipTest("MuJoCo is not installed")
        path = self._write(_stage_source())
        model = self._build_model(path)
        solver = SolverMuJoCo.create_from_usd(model, self._scene(path))
        reference = SolverMuJoCo(model)

        for attribute in (
            "use_mujoco_cpu",
            "update_data_interval",
            "enable_sleeping",
            "nvmax",
            "_use_mujoco_contacts",
            "_deterministic",
        ):
            with self.subTest(attribute=attribute):
                self.assertEqual(getattr(solver, attribute), getattr(reference, attribute))
        for attribute in ("naconmax", "njmax", "nvmax"):
            with self.subTest(attribute=f"mjw_data.{attribute}"):
                self.assertEqual(getattr(solver.mjw_data, attribute), getattr(reference.mjw_data, attribute))
        for attribute in ("ngeom", "nsite", "nbody", "njnt"):
            with self.subTest(attribute=f"mj_model.{attribute}"):
                self.assertEqual(getattr(solver.mj_model, attribute), getattr(reference.mj_model, attribute))
        opt, reference_opt = solver.mj_model.opt, reference.mj_model.opt
        for attribute in (
            "iterations",
            "ls_iterations",
            "ccd_iterations",
            "sdf_iterations",
            "sdf_initpoints",
            "solver",
            "integrator",
            "cone",
            "jacobian",
            "impratio",
            "tolerance",
            "ls_tolerance",
            "ccd_tolerance",
            "density",
            "viscosity",
            "disableflags",
            "enableflags",
        ):
            with self.subTest(attribute=f"mj_model.opt.{attribute}"):
                self.assertEqual(getattr(opt, attribute), getattr(reference_opt, attribute))
        for attribute in ("wind", "magnetic"):
            with self.subTest(attribute=f"mj_model.opt.{attribute}"):
                self.assertEqual(list(getattr(opt, attribute)), list(getattr(reference_opt, attribute)))
        self.assertEqual(
            float(solver.mjw_model.opt.sleep_tolerance.numpy()[0]),
            float(reference.mjw_model.opt.sleep_tolerance.numpy()[0]),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
