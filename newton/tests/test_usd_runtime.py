# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import newton
from newton.tests.unittest_utils import USD_AVAILABLE


def _make_stage(solver_api="NewtonSolverXpbdAPI", scene_attrs=None, num_bodies=1, num_scenes=1):
    """Author a minimal in-memory stage: N falling unit cubes above a static ground box."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for i in range(num_scenes):
        scene = UsdPhysics.Scene.Define(stage, f"/physicsScene{i}" if i else "/physicsScene")
        prim = scene.GetPrim()
        if i == 0:
            if solver_api:
                for token in [solver_api] if isinstance(solver_api, str) else solver_api:
                    prim.AddAppliedSchema(token)
            for name, (type_name, value) in (scene_attrs or {}).items():
                prim.CreateAttribute(name, type_name).Set(value)

    ground = UsdGeom.Cube.Define(stage, "/ground")
    ground.CreateSizeAttr().Set(1.0)
    ground.AddScaleOp().Set(Gf.Vec3f(20.0, 20.0, 1.0))
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.5))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    for i in range(num_bodies):
        cube = UsdGeom.Cube.Define(stage, f"/box{i}")
        cube.CreateSizeAttr().Set(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0 + 1.5 * i))
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return stage


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdRuntime(unittest.TestCase):
    def test_no_physics_scene_errors(self):
        from pxr import Usd

        import newton.usd.runtime as runtime  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        with self.assertRaisesRegex(ValueError, "PhysicsScene"):
            runtime.load_usd(stage)

    def test_multiple_physics_scenes_error(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        stage = _make_stage(num_scenes=2)
        with self.assertRaisesRegex(ValueError, "exactly one PhysicsScene"):
            runtime.load_usd(stage)

    def test_missing_solver_api_errors(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        stage = _make_stage(solver_api=None)
        with self.assertRaisesRegex(ValueError, "NewtonSolverXpbdAPI"):
            # error message must list the registered schema names
            runtime.load_usd(stage)

    def test_unknown_solver_api_errors(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        stage = _make_stage(solver_api="NewtonSolverWarpDriveAPI")
        with self.assertRaisesRegex(ValueError, "NewtonSolverWarpDriveAPI"):
            runtime.load_usd(stage)

    def test_multiple_solver_apis_error(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        stage = _make_stage(solver_api=["NewtonSolverXpbdAPI", "NewtonSolverVbdAPI"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runtime.load_usd(stage)

    def test_load_populates_simulation(self):
        from pxr import Sdf

        import newton.usd.runtime as runtime  # noqa: PLC0415

        stage = _make_stage(
            scene_attrs={
                "newton:timeStepsPerSecond": (Sdf.ValueTypeNames.Float, 240.0),
                "newton:xpbd:iterations": (Sdf.ValueTypeNames.Int, 7),
                "newton:collisionInterval": (Sdf.ValueTypeNames.Int, 2),
            }
        )
        sim = runtime.load_usd(stage, use_graph=False)
        self.assertIsInstance(sim.model, newton.Model)
        self.assertEqual(len(sim.solvers), 1)
        self.assertIs(sim.solver, sim.solvers[0])
        self.assertIsInstance(sim.solver, newton.solvers.SolverXPBD)
        self.assertEqual(sim.solver.iterations, 7)  # authored param
        self.assertAlmostEqual(sim.dt, 1.0 / 240.0)  # authored timestep
        self.assertEqual(sim.collision_interval, 2)
        self.assertEqual(sim.time, 0.0)
        self.assertEqual(sim.step_count, 0)
        self.assertIsNotNone(sim.contacts)
        self.assertEqual(sim.model.body_count, 1)

    def test_unauthored_params_use_constructor_defaults(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        sim = runtime.load_usd(_make_stage(), use_graph=False)
        self.assertEqual(sim.solver.iterations, 2)  # SolverXPBD default
        self.assertAlmostEqual(sim.dt, 1.0 / 1000.0)  # resolver default 1000 Hz
        self.assertEqual(sim.collision_interval, 1)

    def test_unknown_solver_namespace_attrs_are_inert(self):
        from pxr import Sdf

        import newton.usd.runtime as runtime  # noqa: PLC0415

        stage = _make_stage(scene_attrs={"newton:xpbd:iterattions": (Sdf.ValueTypeNames.Int, 99)})
        sim = runtime.load_usd(stage, use_graph=False)  # no warning, no error
        self.assertEqual(sim.solver.iterations, 2)

    def test_source_stage_not_mutated(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        stage = _make_stage()
        before = stage.GetRootLayer().ExportToString()
        runtime.load_usd(stage, use_graph=False)
        self.assertEqual(stage.GetRootLayer().ExportToString(), before)

    def test_requires_grad_passthrough(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        sim = runtime.load_usd(_make_stage(), requires_grad=True, use_graph=False)
        self.assertTrue(sim.state.body_q.requires_grad)

    def test_vbd_stage_loads_with_coloring(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        sim = runtime.load_usd(_make_stage(solver_api="NewtonSolverVbdAPI"), use_graph=False)
        self.assertIsInstance(sim.solver, newton.solvers.SolverVBD)

    def test_load_emits_no_custom_attr_warnings(self):
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415

        from pxr import Sdf

        import newton.usd.runtime as runtime  # noqa: PLC0415

        stage = _make_stage(
            scene_attrs={
                "newton:timeStepsPerSecond": (Sdf.ValueTypeNames.Float, 240.0),
                "newton:xpbd:iterations": (Sdf.ValueTypeNames.Int, 7),
                "newton:collisionInterval": (Sdf.ValueTypeNames.Int, 2),
            }
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runtime.load_usd(stage, use_graph=False)
        self.assertNotIn("Warning: Custom attribute", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
