# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regress physical compliance beyond legacy contact-impedance bounds.

Expected velocities follow independent force-space backward Euler. The frozen
800/1600-contact cases exercise the actual coupled MuJoCo Warp solver, including
its native pyramidal cone; neither mass nor inverse reference weight is edited.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo


def _fixture(capacity, *, device="cpu", mass=1.0, friction=False, hydro=True, enabled=True, cone=None):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMuJoCo.register_custom_attributes(builder)
    config = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.4 if friction else 0.0, margin=0.0, gap=0.0)
    attrs = {"mujoco:condim": 3 if friction else 1}
    ground = builder.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1, cfg=config, custom_attributes=attrs)
    body = builder.add_body(mass=mass, inertia=wp.mat33(np.eye(3) * 0.1), lock_inertia=True)
    shape = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=config, custom_attributes=attrs)
    model = builder.finalize(device=device)
    if hydro:
        flags = model.shape_flags.numpy()
        flags |= int(newton.ShapeFlags.HYDROELASTIC)
        model.shape_flags.assign(flags)
    # The backend preserves its FP32 numerical safeguard: the requested 1e-10
    # is floored to an effective 1e-6. These tests never override that guard.
    solver = SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        use_hydroelastic_force_response=enabled,
        nconmax=capacity,
        njmax=capacity * (4 if friction else 1) + 16,
        iterations=200,
        tolerance=1.0e-10,
        cone=cone or ("pyramidal" if friction else "elliptic"),
        integrator="euler",
        impratio=1.0,
    )
    np.testing.assert_allclose(solver.mjw_model.opt.tolerance.numpy(), 1.0e-6, rtol=1.0e-6, atol=0.0)
    contacts = newton.Contacts(capacity, 0, device=device, per_contact_shape_properties=True)
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    return model, solver, contacts, state_in, state_out, (ground, shape)


def _publish(contacts, shapes, weights, *, stiffness, damping=0.0, gap=-1.0e-5):
    weights = np.asarray(weights, dtype=np.float32)
    count, capacity = len(weights), contacts.rigid_contact_max
    contacts.clear()
    contacts.rigid_contact_count.assign(np.array([count], dtype=np.int32))
    contacts.rigid_contact_shape0.fill_(shapes[0])
    contacts.rigid_contact_shape1.fill_(shapes[1])
    points0 = np.zeros((capacity, 3), dtype=np.float32)
    points1 = points0.copy()
    points0[:, 2] = -0.5 * gap
    points1[:, 2] = 0.5 * gap
    contacts.rigid_contact_point0.assign(points0)
    contacts.rigid_contact_point1.assign(points1)
    contacts.rigid_contact_normal.fill_(wp.vec3(0.0, 0.0, 1.0))
    contacts.rigid_contact_offset0.zero_()
    contacts.rigid_contact_offset1.zero_()
    contacts.rigid_contact_margin0.zero_()
    contacts.rigid_contact_margin1.zero_()
    ke, kd = np.zeros(capacity, dtype=np.float32), np.zeros(capacity, dtype=np.float32)
    ke[:count], kd[:count] = stiffness * weights, damping * weights
    contacts.rigid_contact_stiffness.assign(ke)
    contacts.rigid_contact_damping.assign(kd)
    contacts.rigid_contact_friction.fill_(1.0)


def _initial_velocity(model, state, velocity):
    state.joint_qd.assign(np.array([0.0, 0.0, velocity, 0.0, 0.0, 0.0]))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _normal_reference(mass, stiffness, damping, step, gap, velocity, external):
    # Independently eliminate x_next=x+h*v_next from material force and momentum.
    velocity_next = (mass * velocity + step * external - step * stiffness * gap) / (
        mass + step * damping + step**2 * stiffness
    )
    force = -stiffness * (gap + step * velocity_next) - damping * velocity_next
    if force < 0.0:
        return velocity + step * external / mass, 0.0
    return velocity_next, force


def _check_backend(solver):
    # Preserve the numerical negative control on the old backend: the scalar
    # regression below fails on dynamics, not on an absent experimental API.
    import mujoco_warp

    checker = getattr(mujoco_warp, "check_contact_force_params", None)
    if checker is not None:
        checker(solver.mjw_data)


class TestMuJoCoPhysicalContact(unittest.TestCase):
    def test_frozen_pyramidal_800_1600_rows(self):
        """Recover fixed-J acceleration outside the historical impedance floor."""
        mass, stiffness, step, gap, velocity, external = 1.0, 1.28e6, 1.0 / 1920.0, -1.0e-5, -0.1, -9.81
        expected, force = _normal_reference(mass, stiffness, 0.0, step, gap, velocity, external)
        for count in (1, 800, 1600):
            with self.subTest(count=count):
                model, solver, contacts, si, so, shapes = _fixture(count, friction=True)
                _initial_velocity(model, si, velocity)
                si.body_f.assign(np.array([[0.0, 0.0, external, 0.0, 0.0, 0.0]]))
                _publish(contacts, shapes, np.full(count, 1.0 / count), stiffness=stiffness, gap=gap)
                solver.step(si, so, model.control(), contacts, step)
                _check_backend(solver)
                actual = so.body_qd.numpy()[0]
                np.testing.assert_allclose(actual, [0.0, 0.0, expected, 0.0, 0.0, 0.0], rtol=2.0e-5, atol=2.0e-7)
                # Use acceleration as a separate check, so initial velocity cannot
                # make a large impulse discrepancy look like small state error.
                np.testing.assert_allclose((actual[2] - velocity) / step, (expected - velocity) / step, rtol=2.0e-5)
                np.testing.assert_allclose(solver.mjw_data.qfrc_constraint.numpy()[0, 2], force, rtol=2.0e-5)
                # Row force and the solver's reported generalized force can
                # differ after a stable-active-set shortcut. Independently
                # verify global momentum using the actual rows in FP64.
                data = solver.mjw_data
                row_count = int(data.nefc.numpy()[0])
                jacobian = data.efc.J.numpy()[0, :row_count, :6].astype(np.float64)
                row_force = data.efc.force.numpy()[0, :row_count].astype(np.float64)
                wrench = jacobian.T @ row_force
                acceleration = data.qacc.numpy()[0, :6].astype(np.float64)
                mass_diagonal = np.array([mass, mass, mass, 0.1, 0.1, 0.1])
                external_wrench = np.array([0.0, 0.0, external, 0.0, 0.0, 0.0])
                residual = mass_diagonal * acceleration - external_wrench - wrench
                scale = max(1.0, np.linalg.norm(external_wrench, np.inf), np.linalg.norm(wrench, np.inf))
                self.assertLessEqual(np.linalg.norm(residual, np.inf) / scale, 2.0e-5)

    def test_unequal_soft_and_damped_rows(self):
        """Retain physical weights with small compliance, varying mass and damping."""
        weights = np.array([0.03, 0.07, 0.11, 0.29, 0.5])
        for mass, stiffness, damping, step in ((1.0, 731.0, 0.0, 1.0 / 7680.0), (0.07, 270000.0, 1.3, 1.0 / 1920.0)):
            with self.subTest(mass=mass, damping=damping):
                model, solver, contacts, si, so, shapes = _fixture(len(weights), mass=mass)
                velocity, gap, external = -0.1, -0.00013, -mass * 9.81
                _initial_velocity(model, si, velocity)
                si.body_f.assign(np.array([[0.0, 0.0, external, 0.0, 0.0, 0.0]]))
                _publish(contacts, shapes, weights, stiffness=stiffness, damping=damping, gap=gap)
                solver.step(si, so, model.control(), contacts, step)
                _check_backend(solver)
                expected, force = _normal_reference(mass, stiffness, damping, step, gap, velocity, external)
                np.testing.assert_allclose(so.body_qd.numpy()[0, 2], expected, rtol=2.0e-5, atol=2.0e-7)
                np.testing.assert_allclose(
                    solver.mjw_data.qfrc_constraint.numpy()[0, 2], force, rtol=2.0e-5, atol=1.0e-7
                )

    def test_floor_static_equilibrium(self):
        """Support the exact spring load where the old dynamic mapping was bounded."""
        stiffness, step = 731.0, 1.0 / 7680.0
        model, solver, contacts, si, so, shapes = _fixture(1600)
        si.body_f.assign(np.array([[0.0, 0.0, -9.81, 0.0, 0.0, 0.0]]))
        _publish(contacts, shapes, np.full(1600, 1.0 / 1600), stiffness=stiffness, gap=-9.81 / stiffness)
        solver.step(si, so, model.control(), contacts, step)
        _check_backend(solver)
        np.testing.assert_allclose(so.body_qd.numpy(), 0.0, atol=2.0e-7)

    def test_count_activation_and_timestep_refresh(self):
        """Do not reuse physical coefficients after removal or a parameter update."""
        model, solver, contacts, si, so, shapes = _fixture(64)
        for count, stiffness, damping, gap, step in (
            (0, 731.0, 0.0, -0.001, 1.0 / 7680.0),
            (64, 731.0, 0.0, -0.001, 1.0 / 7680.0),
            (4, 731.0, 3.0, -0.001, 1.0 / 3840.0),
            (0, 731.0, 0.0, -0.001, 1.0 / 7680.0),
            (1, 10000.0, 0.0, 0.001, 1.0 / 1920.0),
            (17, 3711.0, 0.03, -0.001, 1.0 / 7680.0),
        ):
            with self.subTest(count=count, step=step, gap=gap):
                weights = np.full(count, 1.0 / count) if count else []
                _publish(contacts, shapes, weights, stiffness=stiffness, damping=damping, gap=gap)
                solver.step(si, so, model.control(), contacts, step)
                _check_backend(solver)
                expected = (
                    _normal_reference(1.0, stiffness, damping, step, gap, 0.0, 0.0)[0] if count and gap < 0 else 0.0
                )
                np.testing.assert_allclose(so.body_qd.numpy()[0, 2], expected, rtol=2.0e-5, atol=2.0e-7)

    def test_passive_fixed_active_energy(self):
        """No energy is added by the declared closed-system implicit contact step."""
        mass, stiffness, velocity, gap, step = 1.0, 1000.0, -0.05, -0.001, 1.0 / 7680.0
        for damping in (0.0, 2.0):
            model, solver, contacts, si, so, shapes = _fixture(64)
            _initial_velocity(model, si, velocity)
            _publish(contacts, shapes, np.full(64, 1.0 / 64), stiffness=stiffness, damping=damping, gap=gap)
            solver.step(si, so, model.control(), contacts, step)
            _check_backend(solver)
            actual = float(so.body_qd.numpy()[0, 2])
            next_gap = gap + step * actual
            before = 0.5 * mass * velocity**2 + 0.5 * stiffness * gap**2
            after = 0.5 * mass * actual**2 + 0.5 * stiffness * next_gap**2
            self.assertLessEqual(after, before + 1.0e-9)
            expected, _ = _normal_reference(mass, stiffness, damping, step, gap, velocity, 0.0)
            self.assertAlmostEqual(actual, expected, delta=2.0e-7)

    def test_nonhydro_default_unchanged(self):
        """Preserve the ordinary response when physical mode has no hydro contacts."""
        outputs = []
        for enabled in (False, True):
            model, solver, contacts, si, so, shapes = _fixture(16, hydro=False, enabled=enabled)
            _publish(contacts, shapes, np.full(16, 1.0 / 16), stiffness=731.0, damping=2.0, gap=-0.001)
            solver.step(si, so, model.control(), contacts, 1.0 / 7680.0)
            _check_backend(solver)
            outputs.append(so.body_qd.numpy())
        np.testing.assert_array_equal(*outputs)

    def test_zero_tuple_retains_elliptic_friction_policy(self):
        """The legacy (0, 0) sentinel must not suppress positive shape kf."""
        outputs = []
        for enabled in (False, True):
            model, solver, contacts, si, so, shapes = _fixture(4, friction=True, enabled=enabled, cone="elliptic")
            model.shape_material_kf.fill_(123.0)
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
            si.joint_qd.assign(np.array([0.2, -0.1, -0.05, 0.0, 0.0, 0.0]))
            newton.eval_fk(model, si.joint_q, si.joint_qd, si)
            _publish(contacts, shapes, np.full(4, 0.25), stiffness=0.0, damping=0.0, gap=-0.001)
            solver.step(si, so, model.control(), contacts, 1.0 / 1920.0)
            _check_backend(solver)
            outputs.append((so.body_qd.numpy(), solver.mjw_data.contact.solreffriction.numpy()))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])

    def test_missing_backend_extension_is_rejected(self):
        """An explicit physical policy must never silently select the old backend."""
        import mujoco_warp

        required = (
            "enable_contact_force_params",
            "check_contact_force_params",
            "launch_contact_force_graph",
            "reset_contact_force_params",
        )
        if not all(hasattr(mujoco_warp, name) for name in required):
            self.skipTest("This check requires a backend containing the extension")
        for name in required:
            with self.subTest(missing=name):
                original = getattr(mujoco_warp, name)
                delattr(mujoco_warp, name)
                try:
                    with self.assertRaisesRegex(ImportError, "physical-contact support"):
                        _fixture(1)
                finally:
                    setattr(mujoco_warp, name, original)

    def test_cuda_floor_eager_and_graph(self):
        """Match eager and explicitly checked graph responses below the impedance floor."""
        if not wp.get_cuda_device_count():
            self.skipTest("CUDA unavailable")
        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires mempool")
        count, stiffness, step, gap = 1600, 1.28e6, 1.0 / 1920.0, -1.0e-5
        model, solver, contacts, si, so, shapes = _fixture(count, device=device, friction=True)
        _publish(contacts, shapes, np.full(count, 1.0 / count), stiffness=stiffness, gap=gap)
        control = model.control()
        solver.step(si, so, control, contacts, step)
        _check_backend(solver)
        expected, _ = _normal_reference(1.0, stiffness, 0.0, step, gap, 0.0, 0.0)
        np.testing.assert_allclose(so.body_qd.numpy()[0, 2], expected, rtol=2.0e-5, atol=2.0e-7)
        with wp.ScopedCapture(device=device) as capture:
            solver.step(si, so, control, contacts, step)
        for _ in range(3):
            wp.capture_launch(capture.graph)
            _check_backend(solver)
            np.testing.assert_allclose(so.body_qd.numpy()[0, 2], expected, rtol=2.0e-5, atol=2.0e-7)


if __name__ == "__main__":
    unittest.main()
