# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify selective gravity on imported and replicated rigid bodies."""

import unittest
from itertools import pairwise

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS, SolverFeatherstone
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx


class TestFeatherPGSBodyGravity(unittest.TestCase):
    def test_parallel_tree_selective_world_gravity(self):
        """Match analytic axis accelerations with enabled and disabled tree leaves in distinct worlds."""
        if not wp.get_device().is_cuda:
            self.skipTest("Parallel tree execution requires CUDA")
        builder = newton.ModelBuilder(gravity=(2.0, -3.0, 1.0))
        for gravity in ((0.0, 0.0, -10.0), (4.0, -5.0, 2.0)):
            builder.add_world(_gravity_tree(gravity))
        _add_free_body(builder, "global", False)
        model = builder.finalize(device=wp.get_device())
        expected = np.zeros((model.body_count, 3), dtype=np.float32)
        gravity = model.gravity.numpy()[model.body_world.numpy()]
        for body in range(model.body_count - 1):
            leaf = body % 4 - 1
            if leaf >= 0:
                expected[body, leaf] = gravity[body, leaf] * 0.08
        expected[-1] = gravity[-1] * 0.08
        expected[model.body_disable_gravity.numpy()] = 0.0
        for mode in ("split", "matrix_free"):
            for parallel in (False, True):
                with self.subTest(mode=mode, parallel_tree=parallel):
                    state = _advance(model, SolverFeatherPGS, {"pgs_mode": mode, "parallel_tree": parallel}, steps=8)
                    np.testing.assert_allclose(state.body_qd.numpy()[:, :3], expected, atol=2.0e-6)
                    np.testing.assert_allclose(state.body_qd.numpy()[:, 3:], 0.0, atol=1.0e-7)

    def test_parallel_tree_mixed_com_wrench(self):
        """Match branched selective gravity to external forces at enabled centers of mass."""
        if not wp.get_device().is_cuda:
            self.skipTest("Parallel tree execution requires CUDA")
        gravity = np.array((2.0, -3.0, -10.0), dtype=np.float32)
        model = _gravity_tree(gravity, floating=True).finalize(device=wp.get_device())
        reference = _gravity_tree((0.0, 0.0, 0.0), floating=True).finalize(device=wp.get_device())
        forces = np.zeros((model.body_count, 6), dtype=np.float32)
        forces[:, :3] = model.body_mass.numpy()[:, None] * gravity
        forces[model.body_disable_gravity.numpy()] = 0.0
        for mode in ("split", "matrix_free"):
            serial = None
            for parallel in (False, True):
                with self.subTest(mode=mode, parallel_tree=parallel):
                    options = {"pgs_mode": mode, "parallel_tree": parallel}
                    state = _advance(model, SolverFeatherPGS, options, steps=8)
                    control = _advance(reference, SolverFeatherPGS, options, steps=8, forces=forces)
                    for name in ("body_q", "body_qd"):
                        np.testing.assert_allclose(
                            getattr(state, name).numpy(), getattr(control, name).numpy(), atol=3.0e-6
                        )
                        if serial is not None:
                            np.testing.assert_allclose(
                                getattr(state, name).numpy(), getattr(serial, name).numpy(), atol=3.0e-6
                            )
                    self.assertGreater(float(np.linalg.norm(state.body_qd.numpy()[:, 3:])), 0.01)
                    serial = state

    def test_parallel_tree_runtime_flags_and_graph(self):
        """Refresh selective tree gravity and world vectors during notified graph replay."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Parallel tree graph capture requires CUDA")
        for mode in ("split", "matrix_free"):
            with self.subTest(mode=mode):
                builder = newton.ModelBuilder()
                for gravity in ((1.0, -2.0, -10.0), (4.0, -5.0, 2.0)):
                    builder.add_world(_gravity_tree(gravity))
                model = builder.finalize(device=device)
                solvers, states, graphs, outputs, controls = [], [], [], [], []
                for parallel in (False, True):
                    options = {"pgs_mode": mode, "parallel_tree": parallel}
                    _advance(model, SolverFeatherPGS, options, steps=2)
                    solver = SolverFeatherPGS(model, angular_damping=0.0, **options)
                    self.assertEqual(solver._tree_plan is not None, parallel)
                    self.assertTrue(solver._fk_id_cache_enabled)
                    state, output = model.state(), model.state()
                    control = model.control()
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    with wp.ScopedCapture(device) as capture:
                        state.clear_forces()
                        solver.step(state, output, control, None, 0.01)
                        output.clear_forces()
                        solver.step(output, state, control, None, 0.01)
                    solvers.append(solver)
                    states.append(state)
                    graphs.append(capture.graph)
                    # Keep captured state and control allocations alive for replay.
                    outputs.append(output)
                    controls.append(control)
                for phase in range(3):
                    if phase == 1:
                        flags = model.body_disable_gravity.numpy()
                        flags[5:7] = ~flags[5:7]
                        model.body_disable_gravity.assign(flags)
                        for solver in solvers:
                            solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
                    elif phase == 2:
                        gravity = model.gravity.numpy()
                        gravity[1] = (3.0, 7.0, -5.0)
                        model.gravity.assign(gravity)
                        for solver in solvers:
                            solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
                    before = states[0].body_qd.numpy().copy()
                    for graph in graphs:
                        wp.capture_launch(graph)
                    delta = np.zeros((model.body_count, 3), dtype=np.float32)
                    gravity = model.gravity.numpy()[model.body_world.numpy()]
                    for body in range(model.body_count):
                        leaf = body % 4 - 1
                        if leaf >= 0:
                            delta[body, leaf] = gravity[body, leaf] * 0.02
                    delta[model.body_disable_gravity.numpy()] = 0.0
                    for state in states:
                        np.testing.assert_allclose(state.body_qd.numpy()[:, :3] - before[:, :3], delta, atol=2.0e-6)
                    for name in ("body_q", "body_qd"):
                        np.testing.assert_allclose(
                            getattr(states[1], name).numpy(), getattr(states[0], name).numpy(), atol=3.0e-6
                        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_selective_gravity(self):
        """Keep a gravity-disabled body stationary while its neighbor falls."""
        # USD is an optional importer dependency.
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(10.0)
        for index, disabled in enumerate((True, False, None)):
            cube = UsdGeom.Cube.Define(stage, f"/body_{index}")
            cube.CreateSizeAttr().Set(0.1)
            cube.AddTranslateOp().Set(Gf.Vec3d(index, 0.0, 2.0))
            prim = cube.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(1.0)
            if disabled is not None:
                prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(disabled)
        builder = newton.ModelBuilder()
        SolverFeatherPGS.register_custom_attributes(builder)
        builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
        self.assertEqual(builder.body_disable_gravity, [True, False, False])
        model = builder.finalize(device=wp.get_device())
        solver = SolverFeatherPGS(model, angular_damping=0.0)
        state, output = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        initial = state.body_q.numpy().copy()
        for _ in range(8):
            state.clear_forces()
            solver.step(state, output, control, None, 0.01)
            state, output = output, state
        frozen = model.body_label.index("/body_0")
        falling = model.body_label.index("/body_1")
        np.testing.assert_allclose(state.body_q.numpy()[frozen], initial[frozen], atol=1.0e-7)
        self.assertAlmostEqual(float(state.body_qd.numpy()[falling, 2]), -0.8, places=5)
        self.assertLess(float(state.body_q.numpy()[falling, 2]), float(initial[falling, 2]) - 0.02)
        absent = model.body_label.index("/body_2")
        np.testing.assert_allclose(state.body_qd.numpy()[absent], state.body_qd.numpy()[falling], atol=1.0e-7)
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/body_0"))
        for parent, child in ((0, 1), (1, 2)):
            joint = UsdPhysics.FixedJoint.Define(stage, f"/joint_{child}")
            joint.CreateBody0Rel().SetTargets([f"/body_{parent}"])
            joint.CreateBody1Rel().SetTargets([f"/body_{child}"])
        imported = newton.ModelBuilder()
        imported.add_usd(
            stage, collapse_fixed_joints=True, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()]
        )
        self.assertEqual(imported.body_disable_gravity, [True, False])
        clones = newton.ModelBuilder()
        clones.replicate(imported, 4)
        np.testing.assert_array_equal(clones.finalize(device="cpu").body_disable_gravity.numpy(), [True, False] * 4)

    def test_world_gravity_and_replication(self):
        """Preserve selective gravity across distinct local worlds and global bodies."""
        builder = newton.ModelBuilder(gravity=(2.0, -3.0, 1.0))
        for gravity in ((0.0, 0.0, -10.0), (4.0, -5.0, 2.0)):
            source = _free_pair(gravity)
            builder.add_world(source)
        _add_free_body(builder, "global", False)
        self.assertEqual(builder.body_disable_gravity, [True, False, True, False, False])
        model = builder.finalize(device=wp.get_device())
        expected = model.gravity.numpy()[model.body_world.numpy()] * 0.08
        expected[model.body_disable_gravity.numpy()] = 0.0
        initial = model.state()
        newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
        gravity_force = wp.zeros(model.joint_dof_count, dtype=float, device=model.device)
        newton.eval_inverse_dynamics_passive(model, initial, gravity_force=gravity_force)
        np.testing.assert_allclose(gravity_force.numpy().reshape(-1, 6)[:, :3], -expected / 0.08, atol=1.0e-6)
        for solver_type, options in _solver_options():
            with self.subTest(solver=solver_type.__name__, options=options):
                state = _advance(model, solver_type, options, steps=8)
                np.testing.assert_allclose(state.body_qd.numpy()[:, :3], expected, atol=2.0e-6)
                np.testing.assert_allclose(state.body_qd.numpy()[:, 3:], 0.0, atol=1.0e-7)
        replicas = newton.ModelBuilder()
        replicas.replicate(_free_pair(), 4)
        self.assertEqual(replicas.body_disable_gravity, [True, False] * 4)
        copied = newton.ModelBuilder()
        copied.add_builder(replicas)
        np.testing.assert_array_equal(copied.finalize(device="cpu").body_disable_gravity.numpy(), [True, False] * 4)

    def test_mixed_articulation_gravity_wrench(self):
        """Match selective gravity to an equivalent external force at the enabled COM."""
        gravity = np.array((2.0, -3.0, -10.0), dtype=np.float32)
        model = _mixed_articulation(gravity).finalize(device=wp.get_device())
        reference = _mixed_articulation((0.0, 0.0, 0.0)).finalize(device=wp.get_device())
        forces = np.zeros((2, 6), dtype=np.float32)
        forces[1, :3] = reference.body_mass.numpy()[1] * gravity
        for solver_type, options in _solver_options():
            with self.subTest(solver=solver_type.__name__, options=options):
                state = _advance(model, solver_type, options, steps=8)
                control = _advance(reference, solver_type, options, steps=8, forces=forces)
                np.testing.assert_allclose(state.body_q.numpy(), control.body_q.numpy(), atol=3.0e-6)
                np.testing.assert_allclose(state.body_qd.numpy(), control.body_qd.numpy(), atol=3.0e-6)
                self.assertGreater(float(np.linalg.norm(state.body_qd.numpy()[:, 3:])), 0.01)

    def test_collapse_preserves_mixed_gravity(self):
        """Retain mixed-gravity fixed joints and remap flags after equal-gravity merges."""
        for flags in ((False, False, True, True, False), (True, True, False, True, True)):
            builder = newton.ModelBuilder()
            world_fixed = builder.add_link(mass=1.0, label="world_fixed", disable_gravity=True)
            builder.add_articulation([builder.add_joint_fixed(-1, world_fixed)])
            joints = []
            parent = -1
            for index, disabled in enumerate(flags):
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), disable_gravity=disabled)
                joints.append(builder.add_joint_free(body) if index == 0 else builder.add_joint_fixed(parent, body))
                parent = body
            builder.add_articulation(joints)
            builder.collapse_fixed_joints()
            expected = [flags[0], *[right for left, right in pairwise(flags) if left != right]]
            self.assertEqual(builder.body_disable_gravity, expected)
            self.assertEqual(builder.body_count, len(expected))
            np.testing.assert_array_equal(builder.finalize(device="cpu").body_disable_gravity.numpy(), expected)

    def test_runtime_flags_and_graph(self):
        """Refresh selective gravity after notification during cached graph replay."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires CUDA")
        for mode in ("split", "matrix_free"):
            for notify in (False, True):
                with self.subTest(mode=mode, notify=notify):
                    builder = newton.ModelBuilder()
                    builder.replicate(_free_pair(), 4)
                    model = builder.finalize(device=device)
                    _advance(model, SolverFeatherPGS, {"pgs_mode": mode}, steps=2)
                    solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_mode=mode)
                    state, output = model.state(), model.state()
                    control = model.control()
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                    self.assertTrue(solver._fk_id_cache_enabled)
                    with wp.ScopedCapture(device) as capture:
                        state.clear_forces()
                        solver.step(state, output, control, None, 0.01)
                        output.clear_forces()
                        solver.step(output, state, control, None, 0.01)
                    self.assertIs(solver._fk_id_cache_source_state, state)
                    for _ in range(4):
                        wp.capture_launch(capture.graph)
                    expected = np.array([0.0, -0.8] * 4, dtype=np.float32)
                    np.testing.assert_allclose(state.body_qd.numpy()[:, 2], expected, atol=2.0e-6)
                    old_flags = model.body_disable_gravity.numpy()
                    flags = old_flags.copy()
                    flags[2:4] = [False, True]
                    model.body_disable_gravity.assign(flags)
                    np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 1)
                    if notify:
                        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
                        np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 0)
                    wp.capture_launch(capture.graph)
                    expected += np.where(flags, 0.0, -0.1)
                    expected += np.where(flags if notify else old_flags, 0.0, -0.1)
                    np.testing.assert_allclose(state.body_qd.numpy()[:, 2], expected, atol=2.0e-6)
                    old_gravity = model.gravity.numpy()
                    gravity = old_gravity.copy()
                    gravity[2] = [3.0, 0.0, -5.0]
                    model.gravity.assign(gravity)
                    if notify:
                        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
                    before = state.body_qd.numpy().copy()
                    wp.capture_launch(capture.graph)
                    world = model.body_world.numpy()
                    delta = (gravity[world] + (gravity if notify else old_gravity)[world]) * 0.01
                    delta[flags] = 0.0
                    np.testing.assert_allclose(state.body_qd.numpy()[:, :3] - before[:, :3], delta, atol=2.0e-6)


def _add_free_body(builder, label, disabled):
    """Add an isolated body with explicit inertial properties."""
    return builder.add_body(
        label=label,
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
        disable_gravity=disabled,
        xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()),
    )


def _free_pair(gravity=(0.0, 0.0, -10.0)):
    """Build one disabled and one enabled free body."""
    builder = newton.ModelBuilder(gravity=gravity)
    _add_free_body(builder, "disabled", True)
    _add_free_body(builder, "enabled", False)
    return builder


def _mixed_articulation(gravity):
    """Build a floating fixed pair with separated centers of mass."""
    builder = newton.ModelBuilder(gravity=gravity)
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), disable_gravity=True)
    child = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)), com=(0.3, 0.2, -0.1))
    joints = [builder.add_joint_free(root)]
    joints.append(builder.add_joint_fixed(root, child, parent_xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity())))
    builder.add_articulation(joints)
    return builder


def _solver_options():
    """Exercise serial and cached dynamics in both Feather solvers."""
    choices = [(SolverFeatherstone, {}), (SolverFeatherPGS, {"pgs_mode": "split"})]
    if wp.get_device().is_cuda:
        choices.append((SolverFeatherPGS, {"pgs_mode": "matrix_free"}))
    return choices


def _advance(model, solver_type, options, *, steps, forces=None):
    """Step an isolated model with an optional fixed COM wrench."""
    solver = solver_type(model, angular_damping=0.0, **options)
    if options.get("parallel_tree"):
        assert solver._tree_plan is not None
        assert any(group.lanes > 1 for group in solver._tree_plan.groups)
    state, output = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    for _ in range(steps):
        state.clear_forces()
        if forces is not None:
            state.body_f.assign(forces)
        solver.step(state, output, control, None, 0.01)
        state, output = output, state
    return state


def _gravity_tree(gravity, *, floating=False):
    """Build three independently scheduled leaves with alternating gravity flags."""
    builder = newton.ModelBuilder(gravity=gravity)
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), disable_gravity=True)
    joints = [builder.add_joint_free(root) if floating else builder.add_joint_fixed(-1, root)]
    for index, axis in enumerate((newton.Axis.X, newton.Axis.Y, newton.Axis.Z)):
        child = builder.add_link(
            mass=2.0 + index,
            inertia=wp.mat33(np.eye(3)),
            com=(0.3, 0.2, -0.1) if floating else (0.0, 0.0, 0.0),
            disable_gravity=index == 1,
        )
        anchor = wp.transform((1.0, 0.4 * index, 0.1 * index), wp.quat_identity())
        if floating:
            joint = builder.add_joint_fixed(root, child, parent_xform=anchor)
        else:
            joint = builder.add_joint_prismatic(root, child, axis=axis, parent_xform=anchor)
        joints.append(joint)
    builder.add_articulation(joints)
    return builder


if __name__ == "__main__":
    unittest.main()
