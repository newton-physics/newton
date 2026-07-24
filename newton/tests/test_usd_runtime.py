# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal

# Check for mujoco availability via SolverMuJoCo's lazy import mechanism
try:
    SolverMuJoCo.import_mujoco()
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


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

    def test_step_advances_falling_body(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        sim = runtime.load_usd(_make_stage(), use_graph=False)
        z0 = sim.state.body_q.numpy()[0, 2]
        for _ in range(50):
            runtime.step(sim)
        z1 = sim.state.body_q.numpy()[0, 2]
        self.assertLess(z1, z0)  # gravity acted
        self.assertTrue(np.isfinite(sim.state.body_q.numpy()).all())
        self.assertAlmostEqual(sim.time, 50 * sim.dt, places=6)
        self.assertEqual(sim.step_count, 50)

    def test_state_identity_is_stable(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        sim = runtime.load_usd(_make_stage(), use_graph=False)
        state_ref = sim.state
        body_q_ref = sim.state.body_q
        for _ in range(3):
            runtime.step(sim)
        self.assertIs(sim.state, state_ref)
        self.assertIs(sim.state.body_q, body_q_ref)

    def test_applied_force_acts_for_one_step(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        sim = runtime.load_usd(_make_stage(), use_graph=False)
        force = np.zeros((1, 6), dtype=np.float32)
        force[0, 3] = 1.0e4  # linear force +x; spatial vector layout [ang, lin]
        sim.state.body_f.assign(force)
        runtime.step(sim)
        vx_after_one = sim.state.body_qd.numpy()[0, 3]
        self.assertGreater(vx_after_one, 0.0)  # force was consumed
        assert_np_equal(sim.state.body_f.numpy(), np.zeros((1, 6), dtype=np.float32))  # and cleared

    def test_parity_with_manual_wiring(self):
        from pxr import Sdf

        import newton.usd.runtime as runtime  # noqa: PLC0415

        scene_attrs = {"newton:xpbd:iterations": (Sdf.ValueTypeNames.Int, 4)}
        sim = runtime.load_usd(_make_stage(scene_attrs=scene_attrs), use_graph=False)

        # Manual canonical wiring of the identical stage.
        builder = newton.ModelBuilder()
        newton.solvers.SolverXPBD.register_custom_attributes(builder)
        builder.add_usd(
            _make_stage(scene_attrs=scene_attrs),
            schema_resolvers=[
                newton.usd.SchemaResolverNewton(),
                newton.usd.SchemaResolverPhysx(),
                newton.usd.SchemaResolverMjc(),
            ],
            apply_up_axis_from_stage=True,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverXPBD(model, iterations=4)
        state = model.state()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        dt = sim.dt
        for _ in range(20):
            runtime.step(sim)
            collision_pipeline.collide(state, contacts)
            solver.step(state, state, None, contacts, dt)
            state.clear_forces()
        assert_np_equal(sim.state.body_q.numpy(), state.body_q.numpy(), tol=1e-6)

    def test_empty_stage_steps_as_noop(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        sim = runtime.load_usd(_make_stage(num_bodies=0), use_graph=False)
        for _ in range(5):
            runtime.step(sim)
        self.assertEqual(sim.step_count, 5)

    def test_collision_interval_skips_collide(self):
        from unittest import mock  # noqa: PLC0415

        from pxr import Sdf

        import newton.usd.runtime as runtime  # noqa: PLC0415

        stage = _make_stage(scene_attrs={"newton:collisionInterval": (Sdf.ValueTypeNames.Int, 4)})
        sim = runtime.load_usd(stage, use_graph=False)
        with mock.patch.object(sim.collision_pipeline, "collide", wraps=sim.collision_pipeline.collide) as spy:
            for _ in range(8):
                runtime.step(sim)
        self.assertEqual(spy.call_count, 2)  # steps 0 and 4

    def test_invalid_collision_interval_errors_at_load(self):
        from pxr import Sdf

        import newton.usd.runtime as runtime  # noqa: PLC0415

        stage = _make_stage(scene_attrs={"newton:collisionInterval": (Sdf.ValueTypeNames.Int, 0)})
        with self.assertRaisesRegex(ValueError, "newton:collisionInterval"):
            runtime.load_usd(stage, use_graph=False)

    def test_solver_selection_is_pure_data(self):
        from newton._src.usd.runtime import _SOLVER_REGISTRY, _resolve_params, _select_solver  # noqa: PLC0415

        entry = _select_solver(["PhysicsSceneAPI", "NewtonSolverXpbdAPI"])
        self.assertIs(entry, _SOLVER_REGISTRY["NewtonSolverXpbdAPI"])
        params = _resolve_params(entry, {"newton:xpbd:iterations": 7, "unrelated:attr": 1})
        self.assertEqual(params, {"iterations": 7})
        with self.assertRaisesRegex(ValueError, "NewtonSolverXpbdAPI"):
            _select_solver([])

    def test_all_registered_solvers_load_and_step(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415
        from newton._src.usd.runtime import _SOLVER_REGISTRY  # noqa: PLC0415

        for schema, entry in _SOLVER_REGISTRY.items():
            with self.subTest(solver=schema):
                if entry.cls is newton.solvers.SolverMuJoCo and not MUJOCO_AVAILABLE:
                    self.skipTest("Requires mujoco-warp")
                sim = runtime.load_usd(_make_stage(solver_api=schema), use_graph=False)
                self.assertIsInstance(sim.solver, entry.cls)
                for _ in range(10):
                    runtime.step(sim)
                self.assertTrue(np.isfinite(sim.state.body_q.numpy()).all())
                self.assertEqual(sim.step_count, 10)

    def _load_with_capture(self, runtime, stage):
        try:
            return runtime.load_usd(stage, use_graph=True)
        except RuntimeError:
            self.skipTest("graph capture unsupported on this device")

    def test_capture_does_not_advance_sim(self):
        import warp as wp  # noqa: PLC0415

        import newton.usd.runtime as runtime  # noqa: PLC0415

        sim_graph = self._load_with_capture(runtime, _make_stage())
        sim_plain = runtime.load_usd(_make_stage(), use_graph=False)
        self.assertEqual(sim_graph.time, 0.0)
        self.assertEqual(sim_graph.step_count, 0)
        assert_np_equal(sim_graph.state.body_q.numpy(), sim_plain.state.body_q.numpy())
        assert_np_equal(sim_graph.state.body_qd.numpy(), sim_plain.state.body_qd.numpy())
        if wp.get_device().is_cuda:
            self.assertIsNotNone(sim_graph._graphs)

    def test_graph_and_direct_trajectories_match(self):
        import newton.usd.runtime as runtime  # noqa: PLC0415

        sim_graph = self._load_with_capture(runtime, _make_stage())
        sim_plain = runtime.load_usd(_make_stage(), use_graph=False)
        for _ in range(50):
            runtime.step(sim_graph)
            runtime.step(sim_plain)
        assert_np_equal(sim_graph.state.body_q.numpy(), sim_plain.state.body_q.numpy(), tol=1e-6)

    def test_collision_interval_graph_pair(self):
        from pxr import Sdf

        import newton.usd.runtime as runtime  # noqa: PLC0415

        attrs = {"newton:collisionInterval": (Sdf.ValueTypeNames.Int, 4)}
        sim_graph = self._load_with_capture(runtime, _make_stage(scene_attrs=attrs))
        sim_plain = runtime.load_usd(_make_stage(scene_attrs=attrs), use_graph=False)
        self.assertIsNotNone(sim_graph._graphs)
        self.assertIsNotNone(sim_graph._graphs[0])
        self.assertIsNotNone(sim_graph._graphs[1])
        self.assertIsNot(sim_graph._graphs[0], sim_graph._graphs[1])
        for _ in range(10):
            runtime.step(sim_graph)
            runtime.step(sim_plain)
        assert_np_equal(sim_graph.state.body_q.numpy(), sim_plain.state.body_q.numpy(), tol=1e-6)

    def test_cli_runs_headless(self):
        import os  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        from newton._src.usd.runtime import _main  # noqa: PLC0415

        stage = _make_stage()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.usda")
            stage.GetRootLayer().Export(path)
            _main([path, "--viewer", "null", "--num-steps", "20"])


if __name__ == "__main__":
    unittest.main()
