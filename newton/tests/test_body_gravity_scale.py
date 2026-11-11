# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-body gravity scales (:attr:`newton.Model.body_gravity_scale`)."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import (
    SolverFeatherstone,
    SolverKamino,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverSteel,
    SolverVBD,
    SolverXPBD,
)
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM, SolverCoupledProxy
from newton.tests.unittest_utils import add_function_test, get_test_devices

SCALES = (1.0, 0.5, 0.0, -1.0)
INERTIA = wp.mat33(np.eye(3) * 0.01)
DT = 0.01


class TestBodyGravityScale(unittest.TestCase):
    def test_builder_rejects_nonfinite_scale(self):
        """Reject non-finite gravity scales in the builder and at finalize."""
        builder = newton.ModelBuilder()
        for value in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "gravity_scale must be finite"):
                builder.add_body(mass=1.0, gravity_scale=value)
        builder.add_body(mass=1.0, gravity_scale=0.5)
        self.assertEqual(builder.body_gravity_scale, [0.5])
        builder.body_gravity_scale[0] = float("-inf")
        with self.assertRaisesRegex(ValueError, "body_gravity_scale of body 0 must be finite"):
            builder.finalize(device="cpu")

    def test_replicate_and_selection_keep_scale(self):
        """Carry scales through replicate and add_builder, and read and write them through a view."""
        template = newton.ModelBuilder()
        root = template.add_link(mass=1.0, inertia=INERTIA, label="root", gravity_scale=0.0)
        tip = template.add_link(mass=1.0, inertia=INERTIA, label="tip", gravity_scale=0.5)
        joints = [
            template.add_joint_revolute(-1, root, axis=(0.0, 1.0, 0.0)),
            template.add_joint_revolute(root, tip, axis=(0.0, 1.0, 0.0)),
        ]
        template.add_articulation(joints, label="arm")

        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=INERTIA, gravity_scale=-1.0)
        builder.replicate(template, 3)
        builder.add_world(template)
        model = builder.finalize(device="cpu")
        np.testing.assert_array_equal(model.body_gravity_scale.numpy(), [-1.0, *([0.0, 0.5] * 4)])

        view = newton.selection.ArticulationView(model, "arm")
        scales = view.get_attribute("body_gravity_scale", model).numpy()
        np.testing.assert_array_equal(scales.reshape(-1, 2), [[0.0, 0.5]] * 4)
        view.set_attribute("body_gravity_scale", model, np.full(scales.shape, 2.0, dtype=np.float32))
        np.testing.assert_array_equal(model.body_gravity_scale.numpy(), [-1.0, *([2.0] * 8)])

    def test_collapse_keeps_joints_between_different_scales(self):
        """Merge fixed joints between equal scales and keep them between different ones."""
        builder = newton.ModelBuilder()
        base = builder.add_link(mass=1.0, inertia=INERTIA)
        same = builder.add_link(mass=1.0, inertia=INERTIA)
        off = builder.add_link(mass=1.0, inertia=INERTIA, gravity_scale=0.0)
        welded = builder.add_link(mass=1.0, inertia=INERTIA, gravity_scale=0.0)
        joints = [
            builder.add_joint_revolute(-1, base, axis=(0.0, 1.0, 0.0)),
            builder.add_joint_fixed(base, same),
            builder.add_joint_fixed(same, off),
            builder.add_joint_fixed(off, welded),
        ]
        builder.add_articulation(joints)
        builder.collapse_fixed_joints()
        self.assertEqual(builder.body_count, 2)
        self.assertEqual(builder.body_gravity_scale, [1.0, 0.0])
        np.testing.assert_allclose(builder.body_mass, [2.0, 2.0])
        self.assertEqual(builder.joint_type, [newton.JointType.REVOLUTE, newton.JointType.FIXED])


def _free_body_model(device, *, worlds: tuple, global_scales: tuple, mujoco: bool):
    """Free bodies with every scale in :data:`SCALES` in each world of gravity ``worlds`` (or in the implicit
    world if ``worlds`` is empty), plus global bodies with ``global_scales`` under the global gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -2.0))
    if mujoco:
        SolverMuJoCo.register_custom_attributes(builder)

    def add_bodies(scales):
        for i, scale in enumerate(scales):
            builder.add_body(
                xform=wp.transform((2.0 * i, 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=INERTIA,
                gravity_scale=scale,
            )

    add_bodies(global_scales)
    if not worlds:
        add_bodies(SCALES)
    for gravity in worlds:
        builder.begin_world(gravity=gravity)
        add_bodies(SCALES)
        builder.end_world()
    return builder.finalize(device=device)


def _expected_gravity(model):
    gravity = model.gravity.numpy()
    return model.body_gravity_scale.numpy()[:, None] * gravity[model.body_world.numpy()]


def _advance(solver, model, states, control, steps):
    for _ in range(steps):
        states[0].clear_forces()
        solver.step(states[0], states[1], control, None, DT)
        states.reverse()


def test_free_fall(test, device, solver_fn, scene, mujoco):
    """Accelerate each free body from rest at its scaled world gravity, and report it through the coupling hook."""
    model = _free_body_model(device, **scene, mujoco=mujoco)
    solver = solver_fn(model)
    states = [model.state(), model.state()]
    _advance(solver, model, states, model.control(), 10)
    np.testing.assert_allclose(states[0].body_qd.numpy()[:, :3], _expected_gravity(model) * 10 * DT, atol=1.0e-4)

    acceleration = wp.empty(model.body_count, dtype=wp.vec3, device=device)
    solver.coupling_eval_gravity_acceleration(acceleration, None)
    np.testing.assert_allclose(acceleration.numpy(), _expected_gravity(model), atol=1.0e-6)


def _chain_model(device, *, mujoco: bool):
    """Two horizontal links on Y-axis revolute joints; the outer link has no gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    if mujoco:
        SolverMuJoCo.register_custom_attributes(builder)
    inner = builder.add_link(mass=2.0, com=(0.5, 0.0, 0.0), inertia=INERTIA)
    outer = builder.add_link(
        xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()),
        mass=3.0,
        com=(0.5, 0.0, 0.0),
        inertia=INERTIA,
        gravity_scale=0.0,
    )
    joints = [
        builder.add_joint_revolute(-1, inner, axis=(0.0, 1.0, 0.0)),
        builder.add_joint_revolute(
            inner, outer, parent_xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()), axis=(0.0, 1.0, 0.0)
        ),
    ]
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def test_chain_hold(test, device, solver_fn, mujoco):
    """Hold a chain with inverse-dynamics gravity torques and report reactions without the removed weight."""
    model = _chain_model(device, mujoco=mujoco)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # U = sum(s m |g| z) and z = -x sin(q) about +Y; only the inner link (m = 2, x = 0.5) has gravity.
    gravity_force = wp.zeros(model.joint_dof_count, dtype=float, device=device)
    newton.eval_inverse_dynamics_passive(model, state_0, gravity_force=gravity_force)
    np.testing.assert_allclose(gravity_force.numpy(), [-10.0, 0.0], atol=1.0e-4)

    solver = solver_fn(model)
    observables = solver.observables(kinds={newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F})
    control = model.control()
    wp.copy(control.joint_f, gravity_force)
    solver.step(state_0, state_1, control, None, 1.0e-3, observables=observables)

    np.testing.assert_allclose(state_1.joint_qd.numpy(), [0.0, 0.0], atol=1.0e-4)
    parent_f = observables.body_parent_f.numpy()
    np.testing.assert_allclose(parent_f[0], [0.0, 0.0, 20.0, 0.0, 0.0, 0.0], atol=1.0e-3)
    np.testing.assert_allclose(parent_f[1], np.zeros(6), atol=1.0e-3)


def test_graph_replay_after_notify(test, device, solver_fn, mujoco):
    """Apply an in-place scale change to a graph captured with every scale at 1."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    if mujoco:
        SolverMuJoCo.register_custom_attributes(builder)
    builder.add_body(mass=1.0, inertia=INERTIA)
    model = builder.finalize(device=device)
    solver = solver_fn(model)
    states = [model.state(), model.state()]
    control = model.control()
    _advance(solver, model, states, control, 2)
    with wp.ScopedCapture(device=device) as capture:
        _advance(solver, model, states, control, 2)

    def replay(scale):
        model.body_gravity_scale.fill_(scale)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        before = states[0].body_qd.numpy()[0, 2]
        wp.capture_launch(capture.graph)
        return states[0].body_qd.numpy()[0, 2] - before

    np.testing.assert_allclose(replay(1.0), -10.0 * 2 * DT, atol=1.0e-4)
    np.testing.assert_allclose(replay(0.0), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(replay(-0.5), 5.0 * 2 * DT, atol=1.0e-4)


def test_notify_rejects_nonfinite_scale(test, device, solver_fn, mujoco):
    """Reject a non-finite scale written into the model on the next eager notification."""
    builder = newton.ModelBuilder()
    if mujoco:
        SolverMuJoCo.register_custom_attributes(builder)
    builder.add_body(mass=1.0, inertia=INERTIA)
    model = builder.finalize(device=device)
    solver = solver_fn(model)
    model.body_gravity_scale.fill_(float("nan"))
    with test.assertRaisesRegex(ValueError, "body_gravity_scale of body 0 must be finite"):
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)


def test_unsupported_solver_raises(test, device, solver_fn):
    """Raise for scaled dynamic bodies at construction and on notification, including a kinematic body made
    dynamic, and accept massless or kinematic ones."""
    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0, inertia=INERTIA)
    builder.add_shape_sphere(body, radius=0.1)
    builder.add_body(mass=1.0, inertia=INERTIA, is_kinematic=True, gravity_scale=0.0)
    builder.add_body(mass=0.0, gravity_scale=0.0)
    builder.color()
    model = builder.finalize(device=device)
    solver = solver_fn(model)

    flags = model.body_flags.numpy()
    model.body_flags.fill_(int(newton.BodyFlags.DYNAMIC))
    with test.assertRaisesRegex(NotImplementedError, "body_gravity_scale"):
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
    model.body_flags.assign(flags)

    model.body_gravity_scale.assign([0.5, 0.0, 0.0])
    with test.assertRaisesRegex(NotImplementedError, "body_gravity_scale"):
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    with test.assertRaisesRegex(NotImplementedError, "body_gravity_scale"):
        solver_fn(model)


def test_unsupported_solver_notify_after_unit_scales(test, device, solver_fn):
    """Skip the host read on body notifications while every scale is 1, and check again after a scale change."""
    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0, inertia=INERTIA)
    builder.add_shape_sphere(body, radius=0.1)
    builder.add_body(mass=1.0, inertia=INERTIA, is_kinematic=True)
    builder.color()
    model = builder.finalize(device=device)
    solver = solver_fn(model)
    with mock.patch.object(wp.array, "numpy", side_effect=AssertionError("host read")):
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)

    model.body_gravity_scale.assign([1.0, 0.0])
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    model.body_flags.fill_(int(newton.BodyFlags.DYNAMIC))
    with test.assertRaisesRegex(NotImplementedError, "body_gravity_scale"):
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)


def test_coupled_proxy(test, device, destination_fn):
    """Proxy scaled bodies owned by Featherstone into a solver that rejects scales, and fall at the owner's gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    bodies = [
        builder.add_body(
            xform=wp.transform((2.0 * i, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=INERTIA,
            gravity_scale=scale,
        )
        for i, scale in enumerate(SCALES)
    ]
    builder.color()
    model = builder.finalize(device=device)
    coupled = SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupled.Entry(name="owner", solver=SolverFeatherstone, bodies=bodies),
            SolverCoupled.Entry(name="proxy", solver=destination_fn),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(source="owner", destination="proxy", bodies=bodies)]
        ),
    )
    states = [model.state(), model.state()]
    _advance(coupled, model, states, model.control(), 10)
    np.testing.assert_allclose(states[0].body_qd.numpy()[:, :3], _expected_gravity(model) * 10 * DT, atol=1.0e-4)


def test_coupled_admm_joint_proxy_raises(test, device, destination_fn):
    """Raise when a solver that rejects scales would apply its own gravity to a scaled ADMM joint proxy."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    parent = builder.add_link(mass=1.0, inertia=INERTIA, gravity_scale=0.0)
    child = builder.add_link(xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=INERTIA)
    joints = [
        builder.add_joint_revolute(-1, parent, axis=(0.0, 1.0, 0.0)),
        builder.add_joint_ball(parent, child, parent_xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity())),
    ]
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    with test.assertRaisesRegex(NotImplementedError, "body_gravity_scale"):
        SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(name="parent", solver=SolverFeatherstone, bodies=[parent], joints=joints[:1]),
                SolverCoupled.Entry(name="child", solver=destination_fn, bodies=[child]),
            ],
            coupling=SolverCoupledADMM.Config(),
        )


class TestBodyGravityScaleSolvers(unittest.TestCase):
    pass


class TestBodyGravityScaleUnsupported(unittest.TestCase):
    pass


class TestBodyGravityScaleCoupled(unittest.TestCase):
    pass


supported_solvers = {
    "featherstone": (SolverFeatherstone, False),
    "steel": (SolverSteel, False),
    "mujoco_warp": (lambda model: SolverMuJoCo(model, use_mujoco_cpu=False), True),
    # One MuJoCo world holding every body, so bodies of the second and global worlds see another world's gravity.
    "mujoco_warp_shared": (lambda model: SolverMuJoCo(model, use_mujoco_cpu=False, separate_worlds=False), True),
    "mujoco_cpu": (lambda model: SolverMuJoCo(model, use_mujoco_cpu=True), True),
}
# Separate MuJoCo worlds cannot hold global bodies, a shared MuJoCo world holds one Newton world and native
# MuJoCo holds one world. In the shared world, the global bodies see the local world's non-collinear gravity.
two_worlds = ((0.0, 0.0, -5.0), (1.0, 0.0, -3.0))
free_body_scenes = {
    "featherstone": {"worlds": two_worlds, "global_scales": SCALES},
    "steel": {"worlds": two_worlds, "global_scales": SCALES},
    "mujoco_warp": {"worlds": two_worlds, "global_scales": ()},
    "mujoco_warp_shared": {"worlds": two_worlds[1:], "global_scales": SCALES},
    "mujoco_cpu": {"worlds": (), "global_scales": ()},
}

unsupported_solvers = {
    "semi_implicit": SolverSemiImplicit,
    "xpbd": SolverXPBD,
    "vbd": lambda model: SolverVBD(model, rigid_compliant_alm=True),
    "kamino": SolverKamino,
}

for device in get_test_devices():
    for name, (solver_fn, mujoco) in supported_solvers.items():
        if name == "mujoco_cpu" and device.is_cuda:
            continue
        add_function_test(
            TestBodyGravityScaleSolvers,
            f"test_free_fall_{name}",
            test_free_fall,
            devices=[device],
            solver_fn=solver_fn,
            scene=free_body_scenes[name],
            mujoco=mujoco,
        )
        if name == "mujoco_warp_shared":
            continue
        add_function_test(
            TestBodyGravityScaleSolvers,
            f"test_chain_hold_{name}",
            test_chain_hold,
            devices=[device],
            solver_fn=solver_fn,
            mujoco=mujoco,
        )
        add_function_test(
            TestBodyGravityScaleSolvers,
            f"test_notify_rejects_nonfinite_scale_{name}",
            test_notify_rejects_nonfinite_scale,
            devices=[device],
            solver_fn=solver_fn,
            mujoco=mujoco,
        )
        if device.is_cuda and name in ("featherstone", "steel", "mujoco_warp"):
            add_function_test(
                TestBodyGravityScaleSolvers,
                f"test_graph_replay_after_notify_{name}",
                test_graph_replay_after_notify,
                devices=[device],
                solver_fn=solver_fn,
                mujoco=mujoco,
            )
    for name, solver_fn in unsupported_solvers.items():
        add_function_test(
            TestBodyGravityScaleUnsupported,
            f"test_unsupported_solver_raises_{name}",
            test_unsupported_solver_raises,
            devices=[device],
            solver_fn=solver_fn,
        )
        # Kamino's own body checks read the model on the host.
        if name != "kamino":
            add_function_test(
                TestBodyGravityScaleUnsupported,
                f"test_unsupported_solver_notify_after_unit_scales_{name}",
                test_unsupported_solver_notify_after_unit_scales,
                devices=[device],
                solver_fn=solver_fn,
            )
    for name in ("semi_implicit", "xpbd", "vbd"):
        add_function_test(
            TestBodyGravityScaleCoupled,
            f"test_coupled_proxy_{name}",
            test_coupled_proxy,
            devices=[device],
            destination_fn=unsupported_solvers[name],
        )
    add_function_test(
        TestBodyGravityScaleCoupled,
        "test_coupled_admm_joint_proxy_raises",
        test_coupled_admm_joint_proxy_raises,
        devices=[device],
        destination_fn=SolverXPBD,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
