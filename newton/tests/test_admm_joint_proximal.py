# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that ADMM joint proxies and gamma preserve the coupled fixed point."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherstone, SolverMuJoCo, SolverVBD, SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM
from newton.tests.unittest_utils import get_cuda_test_devices


def _cross(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


class TestAdmmJointProximal(unittest.TestCase):
    def _cuda_device(self):
        devices = get_cuda_test_devices()
        if not devices:
            self.skipTest("CUDA device required")
        return devices[0]

    def _make_solver(
        self,
        *,
        kind="ball",
        anchor=(0, 0, 0),
        gravity=(0, 0, 0),
        device="cpu",
        factory=None,
        free_bodies=True,
        **config,
    ):
        builder = newton.ModelBuilder(gravity=gravity)
        for mass in (2.0, 5.0):
            builder.add_body(mass=mass, inertia=wp.mat33(np.diag([1.0, 2.0, 3.0])))
        kwargs = {"parent": 0, "child": 1, "parent_xform": wp.transform(anchor, wp.quat_identity())}
        if kind == "ball":
            builder.add_joint_ball(**kwargs)
        elif kind == "fixed":
            builder.add_joint_fixed(**kwargs)
        else:
            builder.add_joint_revolute(**kwargs, axis=wp.vec3(1, 0, 0))
        builder.color()
        model = builder.finalize(device=device)
        solver = SolverCoupledADMM(
            model,
            [
                SolverCoupled.Entry(
                    name=str(body),
                    bodies=[body],
                    joints=[body] if free_bodies else [],
                    solver=factory or (lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="euler")),
                )
                for body in range(2)
            ],
            SolverCoupledADMM.Config(**config),
        )
        return model, solver

    def test_linearized_iterations_match_scalar_recurrence(self):
        """Match the linearized local update with gamma to a scalar recurrence."""
        cases = ((False, None, set()), (True, (), set()))
        dt, rho, stiffness = 0.02, 0.1, 5000.0
        for enabled, destinations, selected in cases:
            for gamma in (0.0, 10.0):
                with self.subTest(enabled=enabled, destinations=destinations, gamma=gamma):
                    model, solver = self._make_solver(
                        factory=SolverFeatherstone,
                        joint_proximal_bodies=enabled,
                        joint_proximal_destination_entries=destinations,
                        iterations=5,
                        rho=rho,
                        gamma=gamma,
                        joint_stiffness=stiffness,
                        gravity=(3, 0, 0),
                    )
                    self.assertFalse(solver._admm_joint_proxies)
                    state, out = model.state(), model.state()
                    vn = np.array([1.3, -0.7])
                    velocity = np.zeros((2, 6), dtype=np.float32)
                    velocity[:, 0] = vn
                    state.body_qd.assign(velocity)
                    qd = state.joint_qd.numpy()
                    qd[:12] = velocity.ravel()
                    state.joint_qd.assign(qd)
                    force = np.array([2.0, -3.0])
                    body_f = np.zeros((2, 6), dtype=np.float32)
                    body_f[:, 0] = force
                    state.body_f.assign(body_f)
                    mass = model.body_mass.numpy()
                    w2 = 1.0 / np.sum(1.0 / mass)
                    w = np.sqrt(w2)
                    P = rho * w2 * (gamma + np.array([str(i) in selected for i in range(2)]))
                    v = vn.copy()
                    u = dual = 0.0
                    for _ in range(5):
                        impulse = w * (dual + rho * w * (u - (v[1] - v[0])))
                        v = (mass * vn + P * v + dt * (force + 3 * mass) + np.array([-impulse, impulse])) / (mass + P)
                        z = v[1] - v[0]
                        u = (rho * w2 * z - w * dual) / (dt * dt * stiffness + rho * w2)
                        dual += rho * w * (u - z)
                    with ExitStack() as stack:
                        steps = []
                        for entry in solver._entries.values():
                            steps.append(
                                stack.enter_context(patch.object(entry.solver, "step", wraps=entry.solver.step))
                            )
                        solver.step(state, out, None, None, dt)
                        for step in steps:
                            self.assertEqual(step.call_count, 5)
                    np.testing.assert_allclose(out.body_qd.numpy()[:, 0], v, rtol=3e-6, atol=3e-6)
                    group = solver._admm_rr_groups[0]
                    self.assertAlmostEqual(float(group.u.numpy()[0, 0]), u, places=5)
                    self.assertAlmostEqual(float(group.lambda_.numpy()[0, 0]), dual, places=4)

    def test_proxy_selection_validates_names_and_skips_unused_construction(self):
        """Reject unknown destinations and keep no proxies when none are selected."""
        with self.assertRaisesRegex(ValueError, "Unknown ADMM joint proximal destination"):
            self._make_solver(joint_proximal_destination_entries=("missing",))
        for config in ({"joint_proximal_bodies": False}, {"joint_proximal_destination_entries": ()}):
            with self.subTest(config=config):
                _, solver = self._make_solver(
                    factory=lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="implicitfast"),
                    **config,
                )
                self.assertFalse(solver._admm_joint_proxies)

    def test_proxy_selection_preserves_scalar_fixed_point(self):
        """Converge selected and disabled proxies to the same physical equations."""
        dt, rho, stiffness = 0.02, 0.1, 5000.0
        force = np.array([2.0, -3.0])
        for destinations in ((), ("0",), ("1",)):
            for gamma in (0.0, 10.0):
                with self.subTest(destinations=destinations, gamma=gamma):
                    model, solver = self._make_solver(
                        factory=SolverFeatherstone,
                        joint_proximal_destination_entries=destinations,
                        iterations=64,
                        rho=rho,
                        gamma=gamma,
                        joint_stiffness=stiffness,
                    )
                    state, out = model.state(), model.state()
                    f = np.zeros((2, 6), dtype=np.float32)
                    f[:, 0] = force
                    state.body_f.assign(f)
                    mass = model.body_mass.numpy()
                    expected = np.linalg.solve(
                        np.diag(mass) + dt * dt * stiffness * np.array([[1, -1], [-1, 1]]), dt * force
                    )
                    solver.step(state, out, None, None, dt)
                    np.testing.assert_allclose(out.body_qd.numpy()[:, 0], expected, atol=2e-6)
                    group = solver._admm_rr_groups[0]
                    np.testing.assert_allclose(group.u.numpy(), group.Jv.numpy(), atol=2e-6)

    def test_proxies_preserve_native_fixed_point(self):
        """Replay the converged coupling forces on each entry without proxies."""
        factories = {
            "featherstone": SolverFeatherstone,
            "xpbd": lambda view: SolverXPBD(view, iterations=10, angular_damping=0.0),
            "vbd": lambda view: SolverVBD(view, iterations=10, rigid_compliant_alm=True, rigid_joint_linear_ke=1e5),
            "mujoco": lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="euler"),
        }
        self._check_proxy_fixed_point(factories)

    def _check_proxy_fixed_point(self, factories):
        """Replay converged coupling forces using only the physical model."""
        anchor = np.array([0.3, -0.4, 0.2])
        initial = np.array([[1.3, 0.2, -0.3, 0.7, -0.4, 0.3], [-0.7, -0.1, 0.2, -0.2, 0.1, 0.3]], dtype=np.float32)
        forces = np.array([[1, 3, 2, 0.5, -0.2, 0.4], [-2, 1, 4, -0.4, 0.3, 0.2]], dtype=np.float32)
        jacobians = [np.hstack((np.eye(3), -_cross(anchor))), np.hstack((np.eye(3), np.zeros((3, 3))))]
        dt, stiffness = 0.02, 5000.0
        # The replay checks the answer itself, so one high-rho case with gamma covers each backend.
        rho, gamma = 2.0, 1.0
        for backend, factory in factories.items():
            with self.subTest(backend=backend):
                model, solver = self._make_solver(
                    factory=factory,
                    free_bodies=True,
                    anchor=anchor,
                    gravity=(0, -9.81, 0),
                    iterations=48,
                    rho=rho,
                    gamma=gamma,
                    joint_stiffness=stiffness,
                )
                self.assertEqual(len(solver._admm_joint_proxies), 2)
                state, out = model.state(), model.state()
                state.body_qd.assign(initial)
                joint_velocity = state.joint_qd.numpy()
                joint_velocity[:12] = initial.ravel()
                state.joint_qd.assign(joint_velocity)
                state.body_f.assign(forces)
                solver.step(state, out, None, None, dt)
                velocity = out.body_qd.numpy()
                self.assertTrue(np.all(np.isfinite(velocity)))
                group = solver._admm_rr_groups[0]
                np.testing.assert_allclose(group.u.numpy(), group.Jv.numpy(), atol=2e-5)
                jv = jacobians[1] @ velocity[1] - jacobians[0] @ velocity[0]
                _, replay_solver = self._make_solver(
                    factory=factory,
                    anchor=anchor,
                    gravity=(0, -9.81, 0),
                    gamma=0.0,
                    joint_proximal_bodies=False,
                )
                for body in range(2):
                    entry = replay_solver._entries[str(body)]
                    physical, replay = entry.view.state(), entry.view.state()
                    physical.body_qd.assign(initial[body : body + 1])
                    physical.joint_qd.assign(initial[body])
                    physical.body_f.assign(
                        [forces[body] + (1 if body == 0 else -1) * dt * stiffness * jacobians[body].T @ jv]
                    )
                    if backend == "vbd":
                        entry.solver.coupling_notify_input_state_update(physical, newton.StateFlags.BODY, dt=0.0)
                    entry.solver.step(physical, replay, None, None, dt)
                    np.testing.assert_allclose(velocity[body], replay.body_qd.numpy()[0], atol=4e-5, rtol=3e-5)
                    np.testing.assert_allclose(out.body_q.numpy()[body], replay.body_q.numpy()[0], atol=2e-6)

    def test_gamma_preserves_native_fixed_point_without_all_proxies(self):
        """Cancel gamma for spinning anisotropic bodies with optional proxy destinations."""
        anchor = np.array([0.3, -0.4, 0.2])
        initial = np.array([[1.3, 0.2, -0.3, 0.7, -0.4, 0.3], [-0.7, -0.1, 0.2, -0.2, 0.1, 0.3]], dtype=np.float32)
        forces = np.array([[1, 3, 2, 0.5, -0.2, 0.4], [-2, 1, 4, -0.4, 0.3, 0.2]], dtype=np.float32)
        jacobians = [np.hstack((np.eye(3), -_cross(anchor))), np.hstack((np.eye(3), np.zeros((3, 3))))]
        dt, stiffness = 0.02, 5000.0
        # The backend-specific fixed point is covered by test_proxies_preserve_native_fixed_point.
        factories = {"featherstone": SolverFeatherstone}
        for backend, factory in factories.items():
            for enabled, destinations in ((False, None), (True, ("0",))):
                with self.subTest(backend=backend, enabled=enabled, destinations=destinations):
                    model, solver = self._make_solver(
                        factory=factory,
                        anchor=anchor,
                        gravity=(0, -9.81, 0),
                        rho=0.4,
                        gamma=1.0,
                        iterations=32,
                        joint_stiffness=stiffness,
                        joint_proximal_bodies=enabled,
                        joint_proximal_destination_entries=destinations,
                    )
                    _, reference = self._make_solver(
                        factory=factory,
                        anchor=anchor,
                        gravity=(0, -9.81, 0),
                        gamma=0.0,
                        joint_proximal_bodies=False,
                    )
                    state, out = model.state(), model.state()
                    state.body_qd.assign(initial)
                    qd = state.joint_qd.numpy()
                    qd[:12] = initial.ravel()
                    state.joint_qd.assign(qd)
                    state.body_f.assign(forces)
                    solver.step(state, out, None, None, dt)
                    velocity = out.body_qd.numpy()
                    group = solver._admm_rr_groups[0]
                    np.testing.assert_allclose(group.u.numpy(), group.Jv.numpy(), atol=2e-5)
                    jv = jacobians[1] @ velocity[1] - jacobians[0] @ velocity[0]
                    for body in range(2):
                        entry = reference._entries[str(body)]
                        physical, replay = entry.view.state(), entry.view.state()
                        physical.body_qd.assign(initial[body : body + 1])
                        physical.joint_qd.assign(initial[body])
                        physical.body_f.assign(
                            [forces[body] + (1 if body == 0 else -1) * dt * stiffness * jacobians[body].T @ jv]
                        )
                        entry.solver.coupling_notify_input_state_update(
                            physical, newton.StateFlags.BODY | newton.StateFlags.JOINT, dt=0.0
                        )
                        entry.solver.step(physical, replay, None, None, dt)
                        np.testing.assert_allclose(velocity[body], replay.body_qd.numpy()[0], atol=4e-5, rtol=3e-5)
                        np.testing.assert_allclose(out.body_q.numpy()[body], replay.body_q.numpy()[0], atol=2e-6)

    def test_gamma_preserves_articulated_fixed_point(self):
        """Match gamma and proxies to the plain answer on moving fixed and floating chains."""
        # The backend-specific fixed point is covered by test_proxies_preserve_native_fixed_point.
        factories = {"featherstone": (SolverFeatherstone, 1.0)}
        for backend, (factory, rho) in factories.items():
            for floating in (False, True):
                reference = None
                for gamma, proxies in ((0.0, False), (1.0, False), (1.0, True)):
                    with self.subTest(backend=backend, floating=floating, gamma=gamma, proxies=proxies):
                        builder = newton.ModelBuilder(gravity=(0, -9.81, 0))
                        for i in range(2):
                            base = builder.add_link(
                                mass=2.0 + 3.0 * i,
                                com=wp.vec3(0.3, 0.4, 0.2),
                                inertia=wp.mat33(np.diag([1.0, 2.0, 3.0])),
                            )
                            root = (
                                builder.add_joint_free(base)
                                if floating
                                else builder.add_joint_revolute(-1, base, axis=wp.vec3(0, 0, 1))
                            )
                            tip = builder.add_link(
                                mass=1.0, com=wp.vec3(0.2, 0.3, 0.1), inertia=wp.mat33(np.diag([0.2, 0.3, 0.4]))
                            )
                            hinge = builder.add_joint_revolute(
                                base,
                                tip,
                                axis=wp.vec3(0, 1, 0),
                                parent_xform=wp.transform((0.6, 0.2, 0), wp.quat_identity()),
                            )
                            builder.add_articulation([root, hinge])
                        builder.add_joint_ball(1, 3, parent_xform=wp.transform((0.2, -0.1, 0.3), wp.quat_identity()))
                        model = builder.finalize(device="cpu")
                        solver = SolverCoupledADMM(
                            model,
                            [
                                SolverCoupled.Entry(
                                    name=str(i), bodies=[2 * i, 2 * i + 1], joints=[2 * i, 2 * i + 1], solver=factory
                                )
                                for i in range(2)
                            ],
                            SolverCoupledADMM.Config(
                                rho=rho,
                                gamma=gamma,
                                iterations=64,
                                joint_stiffness=5000.0,
                                joint_proximal_bodies=proxies,
                            ),
                        )
                        state, out = model.state(), model.state()
                        q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
                        qs, ds = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
                        for i in range(2):
                            if floating:
                                qd[ds[2 * i] : ds[2 * i] + 6] = [1.3, 0.2, -0.3, 0.7, -0.4, 0.3]
                            else:
                                q[qs[2 * i]] = (0.3, -0.4)[i]
                                qd[ds[2 * i]] = (1.3, -0.9)[i]
                            q[qs[2 * i + 1]] = (-0.2, 0.6)[i]
                            qd[ds[2 * i + 1]] = (-0.7, 0.8)[i]
                        state.joint_q.assign(q)
                        state.joint_qd.assign(qd)
                        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                        solver.step(state, out, None, None, 0.02)
                        group = solver._admm_rr_groups[0]
                        np.testing.assert_allclose(group.u.numpy(), group.Jv.numpy(), atol=2e-5)
                        actual = (out.joint_qd.numpy(), out.body_q.numpy(), out.body_qd.numpy())
                        if reference is None:
                            reference = actual
                        else:
                            for value, expected in zip(actual, reference, strict=True):
                                np.testing.assert_allclose(value, expected, atol=3e-5, rtol=3e-5)

    def test_off_center_convergence_is_independent_of_rho(self):
        """Check stationarity as well as primal feasibility at the converged solution."""
        dt, stiffness = 0.02, 5000.0
        anchor = np.array([0.3, -0.4, 0.2])
        M = np.diag([2.0] * 3 + [1.0, 2.0, 3.0] + [5.0] * 3 + [1.0, 2.0, 3.0])
        J = np.hstack([-np.eye(3), _cross(anchor), np.eye(3), np.zeros((3, 3))])
        initial = np.array([[1.3, 0.2, -0.3, 0, 0, 0], [-0.7, -0.1, 0.2, 0, 0, 0]])
        forces = np.array([[1, 3, 2, 0.5, -0.2, 0.4], [-2, 1, 4, -0.4, 0.3, 0.2]])
        system = M + dt * dt * stiffness * J.T @ J
        rhs = M @ initial.ravel() + dt * forces.ravel()
        expected = np.linalg.solve(system, rhs)
        for rho in (0.4, 2.0):
            for gamma in (0.0, 1.0):
                with self.subTest(rho=rho, gamma=gamma):
                    model, solver = self._make_solver(
                        anchor=anchor, rho=rho, gamma=gamma, iterations=64, joint_stiffness=stiffness
                    )
                    state, out = model.state(), model.state()
                    state.body_qd.assign(initial.astype(np.float32))
                    joint_qd = state.joint_qd.numpy()
                    joint_qd[:12] = initial.ravel()
                    state.joint_qd.assign(joint_qd)
                    state.body_f.assign(forces.astype(np.float32))
                    solver.step(state, out, None, None, dt)
                    velocity = out.body_qd.numpy().ravel()
                    np.testing.assert_allclose(velocity, expected, rtol=3e-5, atol=3e-6)
                    np.testing.assert_allclose(system @ velocity, rhs, rtol=3e-5, atol=1e-5)
                    group = solver._admm_rr_groups[0]
                    np.testing.assert_allclose(group.Jv.numpy(), group.u.numpy(), atol=2e-6)
                    # The native integrator must advance the physical COM with
                    # the returned velocity, without a post-step pose correction.
                    np.testing.assert_allclose(out.body_q.numpy()[:, :3], dt * velocity.reshape(2, 6)[:, :3], atol=2e-7)

    def test_vbd_uses_saved_pose_for_joint_jacobian(self):
        """VBD can modify its input pose; the dual update must keep the original J."""
        anchor = np.array([0.3, -0.4, 0.2])
        model, solver = self._make_solver(
            anchor=anchor,
            factory=lambda view: SolverVBD(view, iterations=1, rigid_compliant_alm=True),
            free_bodies=True,
            iterations=4,
            rho=0.4,
            gamma=1.0,
        )
        state, out = model.state(), model.state()
        forces = np.zeros((2, 6), dtype=np.float32)
        forces[0, 3:] = [20, -40, 70]
        state.body_f.assign(forces)
        solver.step(state, out, None, None, 0.02)
        velocity = out.body_qd.numpy()
        expected = velocity[1, :3] - velocity[0, :3] - np.cross(velocity[0, 3:], anchor)
        np.testing.assert_allclose(solver._admm_rr_groups[0].Jv.numpy()[0], expected, atol=1e-7)

    def test_local_solve_preserves_native_hinge(self):
        """The final physical pose must still satisfy the subsolver's own joint."""
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        parent = builder.add_link(
            xform=wp.transform((0.5, 0, 0), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.eye(3)),
        )
        hinge = builder.add_joint_revolute(
            -1,
            parent,
            child_xform=wp.transform((-0.5, 0, 0), wp.quat_identity()),
            axis=wp.vec3(0, 0, 1),
            armature=0.0,
            damping=0.0,
        )
        builder.add_articulation([hinge])
        child = builder.add_body(
            xform=wp.transform((0.8, 0.2, 0), wp.quat_identity()),
            mass=5.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_joint_ball(parent, child, parent_xform=wp.transform((0.3, 0.2, 0), wp.quat_identity()))
        builder.color()
        model = builder.finalize(device="cpu")
        solver = SolverCoupledADMM(
            model,
            [
                SolverCoupled.Entry(
                    name=str(body),
                    bodies=[body],
                    joints=[body],
                    solver=lambda view: SolverMuJoCo(view, use_mujoco_cpu=True, integrator="euler"),
                )
                for body in (parent, child)
            ],
            SolverCoupledADMM.Config(iterations=16, rho=0.4, gamma=1.0, joint_stiffness=5000.0),
        )
        self.assertEqual(len(solver._admm_joint_proxies), 2)
        state, out = model.state(), model.state()
        forces = np.zeros((2, 6), dtype=np.float32)
        forces[child, 1] = 100.0
        for _ in range(3):
            state.body_f.assign(forces)
            solver.step(state, out, None, None, 0.02)
            q = out.body_q.numpy()[parent]
            transform = wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:]))
            anchor = wp.transform_point(transform, wp.vec3(-0.5, 0, 0))
            np.testing.assert_allclose(np.array(anchor), 0.0, atol=1e-7)
            np.testing.assert_array_equal(solver.view(str(parent)).body_com.numpy()[0], [0, 0, 0])
            state, out = out, state
        self.assertGreater(abs(float(state.body_qd.numpy()[parent, 5])), 0.01)

    def test_cuda_graph(self):
        """Replay proxy targets, gamma compensation, and native solves inside a CUDA graph."""
        device = self._cuda_device()
        featherstone = SolverFeatherstone
        xpbd = lambda view: SolverXPBD(view, iterations=10, angular_damping=0.0)  # noqa: E731
        vbd = lambda view: SolverVBD(view, iterations=10, rigid_compliant_alm=True)  # noqa: E731
        cases = (
            ("featherstone", featherstone, "ball", True),
            ("featherstone", featherstone, "revolute", True),
            ("featherstone", featherstone, "ball", False),
            ("xpbd", xpbd, "ball", True),
            ("vbd", vbd, "ball", True),
        )
        for backend, factory, kind, proxies in cases:
            with self.subTest(backend=backend, kind=kind, proxies=proxies):
                # VBD carries history across steps, so its replayed third step
                # is compared with a twin solver's eager third step.
                runs = []
                for _ in range(2 if backend == "vbd" else 1):
                    model, solver = self._make_solver(
                        device=device,
                        factory=factory,
                        kind=kind,
                        anchor=(0.3, -0.2, 0.1),
                        rho=0.2,
                        gamma=2.0,
                        iterations=8,
                        joint_proximal_bodies=proxies,
                    )
                    self.assertEqual(len(solver._admm_joint_proxies), 2 if proxies else 0)
                    state, out = model.state(), model.state()
                    state.body_qd.assign([[1, 0.2, -0.3, 0.7, -0.4, 0.3], [-0.7, 0, 0.2, -0.2, 0.1, 0.3]])
                    joint_qd = state.joint_qd.numpy()
                    joint_qd[:12] = state.body_qd.numpy().ravel()
                    state.joint_qd.assign(joint_qd)
                    state.body_f.assign([[0, 5, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]])
                    groups = [*solver._admm_rr_groups, *solver._admm_rr_revolute_angular_groups]

                    def step(solver=solver, state=state, out=out, groups=groups):
                        for group in groups:
                            group.u.zero_()
                            group.lambda_.zero_()
                        solver.step(state, out, None, None, 0.01)

                    step()
                    step()
                    runs.append((step, out))
                eager_step, eager_out = runs[0]
                if len(runs) > 1:
                    eager_step()
                expected = eager_out.body_qd.numpy()
                step, out = runs[-1]
                # A captured transfer into a temporary CPU array can corrupt
                # the host heap when the graph is replayed after it is freed.
                with patch.object(SolverVBD, "_to_numpy", side_effect=AssertionError("host read during capture")):
                    with wp.ScopedCapture(device=model.device) as capture:
                        step()
                wp.capture_launch(capture.graph)
                self.assertTrue(np.all(np.isfinite(out.body_qd.numpy())))
                np.testing.assert_allclose(out.body_qd.numpy(), expected, rtol=3e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
