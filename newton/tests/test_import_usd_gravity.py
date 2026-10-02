# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test PhysX gravity exclusions imported into MuJoCo's gravity compensation."""

import unittest

import numpy as np

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE, get_test_devices
from newton.usd import SchemaResolverPhysx


def _gravity_stage():
    """Create independent bodies with authored, absent, and native gravity settings."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    scene = UsdPhysics.Scene.Define(stage, "/scene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(10.0)
    for index, (disabled, native) in enumerate(((True, None), (False, None), (None, None), (True, 0.5), (True, 0.0))):
        cube = UsdGeom.Cube.Define(stage, f"/body_{index}")
        cube.CreateSizeAttr().Set(0.1)
        cube.AddTranslateOp().Set(Gf.Vec3d(index * 2.0, 0.0, 2.0))
        prim = cube.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(1.0)
        if disabled is not None:
            prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(disabled)
        if native is not None:
            prim.CreateAttribute("mjc:gravcomp", Sdf.ValueTypeNames.Float).Set(native)
    return stage


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestImportUsdGravity(unittest.TestCase):
    def test_physx_gravity_compensation_import(self):
        """Import gravity exclusions only with the resolver and preserve native values."""
        for enabled in (False, True):
            with self.subTest(physx_resolver=enabled):
                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                resolvers = [SchemaResolverPhysx()] if enabled else None
                result = builder.add_usd(_gravity_stage(), schema_resolvers=resolvers)
                model = builder.finalize(device="cpu")
                values = model.mujoco.gravcomp.numpy()
                for index, expected in enumerate((float(enabled), 0.0, 0.0, 0.5, 0.0)):
                    body = result["path_body_map"][f"/body_{index}"]
                    self.assertEqual(float(values[body]), expected)

    def test_physx_gravity_compensation_dynamics(self):
        """Keep an excluded body floating beside falling and partially compensated bodies."""
        for device in get_test_devices():
            with self.subTest(device=device):
                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                result = builder.add_usd(_gravity_stage(), schema_resolvers=[SchemaResolverPhysx()])
                model = builder.finalize(device=device)
                solver = SolverMuJoCo(model, use_mujoco_cpu=device.is_cpu)
                state, output = model.state(), model.state()
                control = model.control()
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                initial = state.body_q.numpy().copy()
                for _ in range(8):
                    state.clear_forces()
                    solver.step(state, output, control, None, 0.01)
                    state, output = output, state
                velocities = state.body_qd.numpy()
                poses = state.body_q.numpy()
                for index, expected in enumerate((0.0, -0.8, -0.8, -0.4, -0.8)):
                    body = result["path_body_map"][f"/body_{index}"]
                    np.testing.assert_allclose(velocities[body, :3], (0.0, 0.0, expected), atol=2.0e-6)
                    np.testing.assert_allclose(velocities[body, 3:], 0.0, atol=1.0e-7)
                    if index == 0:
                        np.testing.assert_allclose(poses[body], initial[body], atol=1.0e-7)
                    else:
                        self.assertLess(poses[body, 2], initial[body, 2])

    def test_physx_gravity_compensation_replication(self):
        """Preserve imported compensation across copied and replicated worlds."""
        source = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(source)
        source.add_usd(_gravity_stage(), schema_resolvers=[SchemaResolverPhysx()])
        builder = newton.ModelBuilder()
        builder.replicate(source, 2)
        copied = newton.ModelBuilder()
        copied.add_builder(builder)
        for device in get_test_devices():
            with self.subTest(device=device):
                model = copied.finalize(device=device)
                np.testing.assert_array_equal(model.mujoco.gravcomp.numpy(), [1.0, 0.0, 0.0, 0.5, 0.0] * 2)

    def test_physx_gravity_without_mujoco_attributes(self):
        """Allow other solver imports without creating MuJoCo custom attributes."""
        builder = newton.ModelBuilder()
        builder.add_usd(_gravity_stage(), schema_resolvers=[SchemaResolverPhysx()])
        self.assertNotIn("mujoco:gravcomp", builder.custom_attributes)
        self.assertEqual(builder.body_count, 5)


if __name__ == "__main__":
    unittest.main()
