# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import tempfile
import unittest
from unittest import mock

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

    def test_missing_api_raises(self):
        """Reject a stage whose scene does not apply NewtonMuJoCoSceneAPI."""
        path = self._write(_stage_source(scene_apis='"NewtonSceneAPI"'))
        with self.assertRaisesRegex(ValueError, "NewtonMuJoCoSceneAPI is not applied"):
            SolverMuJoCo.create_from_usd(path, self._build_model(path))

    def test_scene_inside_instance(self):
        """Find a scene that is only reachable through an instance proxy."""
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
        uniform int newton:mujoco:nconmax = 11
    }
}

def Xform "World"
{
    def Xform "Instance" (
        instanceable = true
        references = </Proto>
    )
    {
    }
}
"""
        )
        model = newton.ModelBuilder().finalize()
        with mock.patch.object(SolverMuJoCo, "__init__", return_value=None) as init:
            SolverMuJoCo.create_from_usd(path, model)
        self.assertEqual(init.call_args.kwargs, {"nconmax": 11})

    def test_first_scene_must_apply_api(self):
        """Reject a stage whose first scene differs from the one that applies the API."""
        path = self._write(
            """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def PhysicsScene "A_Other"
    {
    }

    def PhysicsScene "B_Mujoco" (
        prepend apiSchemas = ["NewtonMuJoCoSceneAPI"]
    )
    {
        uniform int newton:mujoco:nconmax = 5
    }
}
"""
        )
        model = newton.ModelBuilder().finalize()
        with self.assertRaisesRegex(ValueError, "A_Other"):
            SolverMuJoCo.create_from_usd(path, model)

    def test_options_come_from_importer_scene(self):
        """Read the arguments from the same scene that supplies the model's options."""
        path = self._write(
            """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def PhysicsScene "A_Mujoco" (
        prepend apiSchemas = ["NewtonMuJoCoSceneAPI"]
    )
    {
        uniform int newton:mujoco:nconmax = 5
    }

    def PhysicsScene "B_Mujoco" (
        prepend apiSchemas = ["NewtonMuJoCoSceneAPI"]
    )
    {
        uniform int newton:mujoco:nconmax = 99
    }
}
"""
        )
        model = newton.ModelBuilder().finalize()
        with mock.patch.object(SolverMuJoCo, "__init__", return_value=None) as init:
            SolverMuJoCo.create_from_usd(path, model)
        self.assertEqual(init.call_args.kwargs, {"nconmax": 5})

    def test_invalid_value_raises(self):
        """Reject out-of-range authored integers."""
        path = self._write(_stage_source("uniform int newton:mujoco:nconmax = -5"))
        with self.assertRaisesRegex(ValueError, "nconmax"):
            SolverMuJoCo.create_from_usd(path, self._build_model(path))

    def test_invalid_deterministic_raises(self):
        """Reject an unknown determinism token."""
        path = self._write(_stage_source('uniform token newton:mujoco:deterministic = "bogus"'))
        with self.assertRaisesRegex(ValueError, "deterministic"):
            SolverMuJoCo.create_from_usd(path, self._build_model(path))

    def test_authored_values_reach_constructor(self):
        """Pass authored attributes to the constructor and omit unauthored or -1 values."""
        path = self._write(
            _stage_source(
                """
        uniform int newton:mujoco:nconmax = 123
        uniform int newton:mujoco:njmax = -1
        uniform bool newton:mujoco:useMujocoCpu = true
        uniform bool mjc:flag:multiccd = true
        uniform bool mjc:flag:contact = false
        uniform int newton:mujoco:updateDataInterval = 3
        uniform token newton:mujoco:deterministic = "runToRun"
        """
            )
        )
        with mock.patch.object(SolverMuJoCo, "__init__", return_value=None) as init:
            solver = SolverMuJoCo.create_from_usd(path, self._build_model(path))
        self.assertIsInstance(solver, SolverMuJoCo)
        args, kwargs = init.call_args
        self.assertIsInstance(args[0], newton.Model)
        self.assertEqual(
            kwargs,
            {
                "nconmax": 123,
                "use_mujoco_cpu": True,
                "enable_multiccd": True,
                "disable_contacts": True,
                "update_data_interval": 3,
                "deterministic": wp.DeterministicMode.RUN_TO_RUN,
            },
        )

    def test_uses_provided_model(self):
        """Pass the caller's model to the constructor unchanged."""
        path = self._write(_stage_source("uniform int newton:mujoco:nconmax = 7"))
        model = newton.ModelBuilder().finalize()
        with mock.patch.object(SolverMuJoCo, "__init__", return_value=None) as init:
            SolverMuJoCo.create_from_usd(path, model)
        args, kwargs = init.call_args
        self.assertIs(args[0], model)
        self.assertEqual(kwargs, {"nconmax": 7})

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
        uniform token newton:mujoco:deterministic = "runToRun"
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
        solver = SolverMuJoCo.create_from_usd(path, self._build_model(path))
        opt = solver.mj_model.opt
        self.assertTrue(solver.use_mujoco_cpu)
        self.assertTrue(solver._use_mujoco_contacts)
        self.assertEqual(solver.update_data_interval, 2)
        self.assertEqual(solver._deterministic, wp.DeterministicMode.RUN_TO_RUN)
        self.assertEqual(solver.model.body_count, 1)
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
        solver = SolverMuJoCo.create_from_usd(path, self._build_model(path))
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
        uniform int newton:mujoco:nconmax = 64
        uniform int newton:mujoco:njmax = 128
        uniform bool newton:mujoco:enableSleeping = true
        uniform int newton:mujoco:nvmax = 1
        uniform double mjc:option:sleep_tolerance = 0.02
        """,
                scene_apis='"NewtonMuJoCoSceneAPI", "MjcSceneAPI"',
            )
        )
        solver = SolverMuJoCo.create_from_usd(path, self._build_model(path))
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
                self.assertEqual(scene.GetAttribute(f"newton:mujoco:{name}").Get(), -1)
                self.assertIsNone(parameters[argument].default)
        for name, argument in plain.items():
            with self.subTest(attribute=name):
                self.assertEqual(scene.GetAttribute(f"newton:mujoco:{name}").Get(), parameters[argument].default)
        # enable_sleeping defaults to None, which resolves to the model attribute (default False)
        self.assertFalse(scene.GetAttribute("newton:mujoco:enableSleeping").Get())
        self.assertIsNone(parameters["enable_sleeping"].default)
        # "inherit" is the schema's spelling of deterministic=None
        self.assertEqual(scene.GetAttribute("newton:mujoco:deterministic").Get(), "inherit")
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
        solver = SolverMuJoCo.create_from_usd(path, model)
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
