# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

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


if __name__ == "__main__":
    unittest.main()
