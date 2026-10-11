# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the opt-in hydroelastic normal response against force-space mechanics.

These synthetic contacts hold the geometry fixed while partitioning an additive
spring/damper law. They intentionally bypass SDF extraction and its quadrature
error, but exercise Newton's contact conversion and the actual MuJoCo Warp solve.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo


def _make_fixture(
    *,
    device="cpu",
    mass=1.0,
    dynamic_pair=False,
    hydroelastic=True,
    force_response=True,
    worlds=1,
    friction=False,
    cone=None,
):
    """Build free rigid bodies with exactly prescribed mass and inertia."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMuJoCo.register_custom_attributes(builder)
    config = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5 if friction else 0.0, margin=0.0, gap=0.0)
    body0 = -1
    if dynamic_pair:
        body0 = builder.add_body(mass=2.0 * mass, inertia=wp.mat33(np.eye(3) * 0.2), lock_inertia=True)
    shape0 = builder.add_shape_box(
        body0, hx=0.1, hy=0.1, hz=0.1, cfg=config, custom_attributes={"mujoco:condim": 3 if friction else 1}
    )
    body1 = builder.add_body(mass=mass, inertia=wp.mat33(np.eye(3) * 0.1), lock_inertia=True)
    shape1 = builder.add_shape_box(
        body1, hx=0.1, hy=0.1, hz=0.1, cfg=config, custom_attributes={"mujoco:condim": 3 if friction else 1}
    )
    if worlds > 1:
        template = builder
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template, worlds)
    model = builder.finalize(device=device)
    if hydroelastic:
        # No collision pipeline is used: set the classification without building SDFs.
        flags = model.shape_flags.numpy()
        flags |= int(newton.ShapeFlags.HYDROELASTIC)
        model.shape_flags.assign(flags)
    options = {"use_hydroelastic_force_response": True} if force_response else {}
    solver = SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        nconmax=32,
        njmax=128,
        iterations=100,
        tolerance=1.0e-9,
        integrator="euler",
        cone=cone or ("pyramidal" if friction else "elliptic"),
        **options,
    )
    contacts = newton.Contacts(32, 0, device=device, per_contact_shape_properties=True)
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    return model, solver, contacts, state_in, state_out, (shape0, shape1)


def _fill_contacts(contacts, shapes, weights, *, stiffness=1.0e4, damping=0.0, gap=-0.001, lever=(0.0, 0.0, 0.0)):
    """Publish collinear contact rows whose individual force laws sum unchanged."""
    weights = np.asarray(weights, dtype=np.float32)
    count = len(weights)
    capacity = contacts.rigid_contact_max
    contacts.clear()
    contacts.rigid_contact_count.assign(np.array([count], dtype=np.int32))
    contacts.rigid_contact_shape0.fill_(shapes[0])
    contacts.rigid_contact_shape1.fill_(shapes[1])
    points0 = np.tile(np.asarray(lever, dtype=np.float32), (capacity, 1))
    points1 = points0.copy()
    points0[:, 2] -= 0.5 * gap
    points1[:, 2] += 0.5 * gap
    contacts.rigid_contact_point0.assign(points0)
    contacts.rigid_contact_point1.assign(points1)
    contacts.rigid_contact_normal.fill_(wp.vec3(0.0, 0.0, 1.0))
    contacts.rigid_contact_offset0.zero_()
    contacts.rigid_contact_offset1.zero_()
    contacts.rigid_contact_margin0.zero_()
    contacts.rigid_contact_margin1.zero_()
    ke = np.zeros(capacity, dtype=np.float32)
    kd = np.zeros(capacity, dtype=np.float32)
    ke[:count] = stiffness * weights
    kd[:count] = damping * weights
    contacts.rigid_contact_stiffness.assign(ke)
    contacts.rigid_contact_damping.assign(kd)
    contacts.rigid_contact_friction.fill_(1.0)


def _dense_reference(masses, inertias, lever, *, gap, stiffness, damping, dt, velocity=None):
    """Solve the independent backward-Euler force balance in generalized coordinates.

    ``J`` maps world body velocities (linear then angular) to separation speed.
    For the fixed normal row, f = -K*(gap+h*J*u_next)-C*J*u_next.
    Combining it with M*(u_next-u)=h*J.T*f gives the dense linear system below.
    """
    bodies = len(masses)
    j = np.zeros(6 * bodies)
    mass_matrix = np.zeros((6 * bodies, 6 * bodies))
    for body, (mass, inertia) in enumerate(zip(masses, inertias, strict=True)):
        sign = 1.0 if body == bodies - 1 else -1.0
        j[6 * body : 6 * body + 3] = sign * np.array([0.0, 0.0, 1.0])
        j[6 * body + 3 : 6 * body + 6] = sign * np.cross(lever, [0.0, 0.0, 1.0])
        mass_matrix[6 * body : 6 * body + 3, 6 * body : 6 * body + 3] = mass * np.eye(3)
        mass_matrix[6 * body + 3 : 6 * body + 6, 6 * body + 3 : 6 * body + 6] = inertia * np.eye(3)
    velocity = np.zeros(6 * bodies) if velocity is None else np.asarray(velocity).reshape(-1)
    matrix = mass_matrix + dt * (damping + dt * stiffness) * np.outer(j, j)
    rhs = mass_matrix @ velocity - dt * stiffness * gap * j
    next_velocity = np.linalg.solve(matrix, rhs)
    force = -stiffness * (gap + dt * j @ next_velocity) - damping * (j @ next_velocity)
    return next_velocity.reshape(bodies, 6), force


class TestMuJoCoContactPartition(unittest.TestCase):
    def test_weighted_normal_partition(self):
        """Preserve motion and summed impulse when one force law is partitioned."""
        partitions = [np.full(n, 1.0 / n) for n in (1, 2, 4, 8, 16)]
        partitions.append(np.array([0.1, 0.2, 0.3, 0.4]))
        for mass in (0.1, 1.0):
            for damping in (0.0, 20.0):
                for weights in partitions:
                    with self.subTest(mass=mass, damping=damping, weights=weights):
                        model, solver, contacts, state_in, state_out, shapes = _make_fixture(mass=mass)
                        _fill_contacts(contacts, shapes, weights, damping=damping)
                        solver.step(state_in, state_out, model.control(), contacts, 0.001)
                        expected, force = _dense_reference(
                            [mass], [0.1], np.zeros(3), gap=-0.001, stiffness=1.0e4, damping=damping, dt=0.001
                        )
                        actual = state_out.body_qd.numpy()
                        np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=2.0e-7)
                        self.assertAlmostEqual(float(mass * actual[-1, 2] / 0.001), force, delta=force * 2.0e-4)

    def test_dynamic_pair_and_contact_torque(self):
        """Retain coupled body inertia and the actual rotational contact lever arm."""
        for dynamic_pair in (False, True):
            for count in (1, 16):
                with self.subTest(dynamic_pair=dynamic_pair, count=count):
                    model, solver, contacts, state_in, state_out, shapes = _make_fixture(dynamic_pair=dynamic_pair)
                    lever = np.array([0.2, -0.1, 0.0])
                    _fill_contacts(contacts, shapes, np.full(count, 1.0 / count), lever=lever, damping=20.0)
                    solver.step(state_in, state_out, model.control(), contacts, 0.001)
                    masses = [2.0, 1.0] if dynamic_pair else [1.0]
                    inertias = [0.2, 0.1] if dynamic_pair else [0.1]
                    expected, force = _dense_reference(
                        masses, inertias, lever, gap=-0.001, stiffness=1.0e4, damping=20.0, dt=0.001
                    )
                    actual = state_out.body_qd.numpy()
                    np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=2.0e-7)
                    # Momentum/torque checks use output velocities, independently of efc_force.
                    expected_wrench = force * np.r_[0.0, 0.0, 1.0, np.cross(lever, [0.0, 0.0, 1.0])]
                    actual_wrench = actual[-1] * np.r_[np.full(3, masses[-1]), np.full(3, inertias[-1])] / 0.001
                    np.testing.assert_allclose(actual_wrench, expected_wrench, rtol=2.0e-4, atol=2.0e-5)
                    if dynamic_pair:
                        np.testing.assert_allclose(2.0 * actual[0, :3] + actual[1, :3], 0.0, atol=2.0e-7)

    def test_prescribed_damping_and_time_resolution(self):
        """Match approach damping and closed-system energy loss at three timesteps."""
        for damping in (0.0, 20.0):
            for dt in (0.001, 0.0005, 0.00025):
                with self.subTest(damping=damping, dt=dt):
                    model, solver, contacts, state_in, state_out, shapes = _make_fixture()
                    velocity = np.array([[0.0, 0.0, -0.05, 0.0, 0.0, 0.0]])
                    state_in.joint_qd.assign(velocity.reshape(-1))
                    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                    # Keep every row inside the documented impedance range at h/4.
                    _fill_contacts(contacts, shapes, [0.25] * 4, damping=damping)
                    solver.step(state_in, state_out, model.control(), contacts, dt)
                    expected, _ = _dense_reference(
                        [1.0],
                        [0.1],
                        np.zeros(3),
                        gap=-0.001,
                        stiffness=1.0e4,
                        damping=damping,
                        dt=dt,
                        velocity=velocity,
                    )
                    actual = state_out.body_qd.numpy()
                    np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=2.0e-7)
                    actual_translation = state_out.body_q.numpy()[0, :3]
                    np.testing.assert_allclose(actual_translation, dt * expected[0, :3], rtol=2.0e-4, atol=2.0e-9)
                    energy_before = 0.5 * 0.05**2 + 0.5 * 1.0e4 * 0.001**2
                    next_gap = -0.001 + actual_translation[2]
                    energy_after = 0.5 * actual[0, 2] ** 2 + 0.5 * 1.0e4 * next_gap**2
                    self.assertLessEqual(energy_after, energy_before)

    def test_static_force_balance(self):
        """Support different gravitational loads with the same spring coefficient."""
        for mass in (0.1, 1.0):
            with self.subTest(mass=mass):
                model, solver, contacts, state_in, state_out, shapes = _make_fixture(mass=mass)
                load = mass * 9.81
                state_in.body_f.assign(np.array([[0.0, 0.0, -load, 0.0, 0.0, 0.0]]))
                _fill_contacts(contacts, shapes, np.full(16, 1.0 / 16), gap=-load / 1.0e4)
                solver.step(state_in, state_out, model.control(), contacts, 0.001)
                np.testing.assert_allclose(state_out.body_qd.numpy(), 0.0, atol=2.0e-7)

    def test_lower_impedance_bound_preserves_static_force(self):
        """Preserve static spring support outside the legacy impedance interval."""
        for count in (1, 16):
            with self.subTest(count=count):
                model, solver, contacts, state_in, state_out, shapes = _make_fixture()
                state_in.body_f.assign(np.array([[0.0, 0.0, -9.81, 0.0, 0.0, 0.0]]))
                _fill_contacts(contacts, shapes, np.full(count, 1.0 / count), gap=-9.81 / 1.0e4)
                solver.step(state_in, state_out, model.control(), contacts, 0.00005)
                solver.check_hydroelastic_force_response()
                np.testing.assert_allclose(state_out.body_qd.numpy(), 0.0, atol=2.0e-7)

    def test_lower_impedance_bound_dynamic_response(self):
        """Match physical dynamics below the former impedance floor."""
        mass, stiffness, gap, dt, velocity = 1.0, 1.0e4, -0.001, 0.00025, -0.05
        for weights in (np.array([0.1, 0.2, 0.3, 0.4]), np.full(16, 1.0 / 16)):
            with self.subTest(weights=weights):
                model, solver, contacts, state_in, state_out, shapes = _make_fixture(mass=mass)
                state_in.joint_qd.assign(np.array([0.0, 0.0, velocity, 0.0, 0.0, 0.0]))
                newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                _fill_contacts(contacts, shapes, weights, gap=gap, stiffness=stiffness)
                solver.step(state_in, state_out, model.control(), contacts, dt)
                solver.check_hydroelastic_force_response()
                expected = (mass * velocity - dt * stiffness * gap) / (mass + dt**2 * stiffness)
                actual = float(state_out.body_qd.numpy()[0, 2])
                self.assertAlmostEqual(actual, expected, delta=2.0e-8)

    def test_contact_buffer_and_timestep_reuse(self):
        """Refresh parameters after contact removal, reordering, recontact, and timestep changes."""
        model, solver, contacts, state_in, state_out, shapes = _make_fixture()
        control = model.control()
        # Keep initial state fixed to isolate contact-buffer lifetime from changing geometry.
        cases = (
            ([], -0.001, 0.001),
            ([0.1, 0.2, 0.3, 0.4], -0.001, 0.001),
            ([0.4, 0.3, 0.2, 0.1], -0.001, 0.001),
            ([], -0.001, 0.001),
            ([1.0], 0.001, 0.001),
            ([1.0], -0.001, 0.0005),
        )
        for weights, gap, dt in cases:
            with self.subTest(weights=weights, gap=gap, dt=dt):
                _fill_contacts(contacts, shapes, weights, gap=gap, damping=20.0)
                solver.step(state_in, state_out, control, contacts, dt)
                if weights and gap < 0:
                    expected, _ = _dense_reference(
                        [1.0], [0.1], np.zeros(3), gap=gap, stiffness=1.0e4, damping=20.0, dt=dt
                    )
                else:
                    expected = np.zeros((1, 6))
                np.testing.assert_allclose(state_out.body_qd.numpy(), expected, rtol=2.0e-4, atol=2.0e-7)
        # The unchanged generation exercises any fast path while h changes again.
        solver.step(state_in, state_out, control, contacts, 0.001)
        expected, _ = _dense_reference([1.0], [0.1], np.zeros(3), gap=-0.001, stiffness=1.0e4, damping=20.0, dt=0.001)
        np.testing.assert_allclose(state_out.body_qd.numpy(), expected, rtol=2.0e-4, atol=2.0e-7)

    def test_nonhydro_contact_override_is_unchanged(self):
        """Leave existing custom-contact semantics unchanged for non-hydro shapes."""
        outputs = []
        for force_response in (False, True):
            model, solver, contacts, state_in, state_out, shapes = _make_fixture(
                hydroelastic=False, force_response=force_response
            )
            _fill_contacts(contacts, shapes, [0.1, 0.2, 0.3, 0.4], damping=20.0)
            solver.step(state_in, state_out, model.control(), contacts, 0.001)
            outputs.append((state_out.body_qd.numpy(), solver.mjw_data.contact.solref.numpy()[:4]))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])

    def test_multiple_normal_directions(self):
        """Preserve coupled torque when a body contacts two different surfaces."""
        model, solver, contacts, state_in, state_out, shapes = _make_fixture()
        normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        levers = np.array([[0.2, 0.0, 0.0], [0.0, 0.0, 0.1]])
        stiffness = np.array([1.0e4, 2.0e4])
        damping = np.array([20.0, 10.0])
        gap = np.array([-0.001, -0.0005])
        dt = 0.001
        _fill_contacts(contacts, shapes, [1.0, 2.0])
        points0 = np.zeros((contacts.rigid_contact_max, 3), dtype=np.float32)
        points1 = points0.copy()
        normal_buffer = points0.copy()
        points0[:2] = levers - 0.5 * gap[:, None] * normals
        points1[:2] = levers + 0.5 * gap[:, None] * normals
        normal_buffer[:2] = normals
        contacts.rigid_contact_point0.assign(points0)
        contacts.rigid_contact_point1.assign(points1)
        contacts.rigid_contact_normal.assign(normal_buffer)
        damping_buffer = np.zeros(contacts.rigid_contact_max, dtype=np.float32)
        damping_buffer[:2] = damping
        contacts.rigid_contact_damping.assign(damping_buffer)
        j = np.hstack((normals, np.cross(levers, normals)))
        mass = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
        matrix = mass + dt * j.T @ np.diag(damping + dt * stiffness) @ j
        expected = np.linalg.solve(matrix, -dt * j.T @ (stiffness * gap))
        solver.step(state_in, state_out, model.control(), contacts, dt)
        np.testing.assert_allclose(state_out.body_qd.numpy()[0], expected, rtol=2.0e-4, atol=2.0e-7)

    def test_world_isolation(self):
        """Keep one world's response unchanged as another world's contact count changes."""
        model, solver, contacts, state_in, state_out, shapes = _make_fixture(worlds=2)
        expected, _ = _dense_reference([1.0], [0.1], np.zeros(3), gap=-0.001, stiffness=1.0e4, damping=0.0, dt=0.001)
        for other_count, other_scale in ((1, 1.0), (16, 4.0), (0, 0.0)):
            with self.subTest(other_count=other_count, other_scale=other_scale):
                weights = [1.0] + ([other_scale / other_count] * other_count if other_count else [])
                _fill_contacts(contacts, shapes, weights)
                shape0 = np.full(contacts.rigid_contact_max, shapes[0] + 2, dtype=np.int32)
                shape1 = np.full(contacts.rigid_contact_max, shapes[1] + 2, dtype=np.int32)
                shape0[0], shape1[0] = shapes
                contacts.rigid_contact_shape0.assign(shape0)
                contacts.rigid_contact_shape1.assign(shape1)
                solver.step(state_in, state_out, model.control(), contacts, 0.001)
                np.testing.assert_allclose(state_out.body_qd.numpy()[:1], expected, rtol=2.0e-4, atol=2.0e-7)

    def test_pyramidal_friction_at_normal_incidence(self):
        """Preserve the declared normal force at symmetric zero tangential velocity."""
        for count in (1, 16):
            with self.subTest(count=count):
                model, solver, contacts, state_in, state_out, shapes = _make_fixture(friction=True)
                _fill_contacts(contacts, shapes, np.full(count, 1.0 / count), damping=20.0)
                solver.step(state_in, state_out, model.control(), contacts, 0.001)
                expected, _ = _dense_reference(
                    [1.0], [0.1], np.zeros(3), gap=-0.001, stiffness=1.0e4, damping=20.0, dt=0.001
                )
                np.testing.assert_allclose(state_out.body_qd.numpy(), expected, rtol=2.0e-4, atol=2.0e-7)

    def test_frictional_partition_preserves_wrench(self):
        """Preserve the full contact impulse under splitting with tangential slip."""
        for cone in ("pyramidal", "elliptic"):
            reference = None
            for weights in ([1.0], [1.0 / 16] * 16, [0.1, 0.2, 0.3, 0.4]):
                with self.subTest(cone=cone, weights=weights):
                    model, solver, contacts, state_in, state_out, shapes = _make_fixture(friction=True, cone=cone)
                    velocity = np.array([[0.2, -0.1, -0.05, 0.0, 0.0, 0.0]])
                    state_in.joint_qd.assign(velocity.reshape(-1))
                    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                    _fill_contacts(contacts, shapes, weights, lever=(0.2, -0.1, 0.0), damping=20.0)
                    solver.step(state_in, state_out, model.control(), contacts, 0.001)
                    actual = state_out.body_qd.numpy()
                    # Cone-active normal and tangential forces are coupled. The oracle
                    # here is unchanged representation, not an uncoupled scalar spring.
                    impulse = (actual - velocity) * np.array([[1.0, 1.0, 1.0, 0.1, 0.1, 0.1]])
                    self.assertGreater(float(impulse[0, 2]), 0.0)
                    self.assertLess(float(impulse[0, 0]), 0.0)
                    if reference is None:
                        reference = (actual, impulse)
                    else:
                        np.testing.assert_allclose(actual, reference[0], rtol=2.0e-4, atol=2.0e-7)
                        np.testing.assert_allclose(impulse, reference[1], rtol=2.0e-4, atol=2.0e-7)

    def test_cuda_eager_and_graph_replay(self):
        """Match CUDA eager and captured solves to the independent dense reference."""
        if not wp.get_cuda_device_count():
            self.skipTest("CUDA is unavailable")
        device = wp.get_cuda_device(0)
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the mempool allocator")
        model, solver, contacts, state_in, state_out, shapes = _make_fixture(device=device, dynamic_pair=True)
        control = model.control()
        lever = np.array([0.2, -0.1, 0.0])
        _fill_contacts(contacts, shapes, [0.1, 0.2, 0.3, 0.4], lever=lever, damping=20.0)
        solver.step(state_in, state_out, control, contacts, 0.001)
        expected, _ = _dense_reference(
            [2.0, 1.0], [0.2, 0.1], lever, gap=-0.001, stiffness=1.0e4, damping=20.0, dt=0.001
        )
        eager = state_out.body_qd.numpy()
        np.testing.assert_allclose(eager, expected, rtol=2.0e-4, atol=2.0e-7)
        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_in, state_out, control, contacts, 0.001)
        for _ in range(3):
            wp.capture_launch(capture.graph)
            solver.check_hydroelastic_force_response()
            np.testing.assert_allclose(state_out.body_qd.numpy(), expected, rtol=2.0e-4, atol=2.0e-7)
            np.testing.assert_allclose(state_out.body_qd.numpy(), eager, rtol=2.0e-4, atol=2.0e-7)


if __name__ == "__main__":
    unittest.main()
