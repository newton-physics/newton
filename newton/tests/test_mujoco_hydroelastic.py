# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the force-space response of hydroelastic MuJoCo contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.geometry import HydroelasticSDF
from newton.tests.unittest_utils import MUJOCO_MODULES, get_selected_cuda_test_devices, requires

_cuda_devices = get_selected_cuda_test_devices()


def _make_contacts(
    count,
    *,
    mass=1.0,
    stiffness=1.0e4,
    damping=0.0,
    velocity=-0.1,
    cone="elliptic",
    mu=0.4,
    condim=3,
    gravity=0.0,
    hydroelastic_force_space=True,
    use_speculative_contacts=False,
    integrator=None,
):
    """Build a planar hydroelastic patch with prescribed quadrature weights."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu, margin=0.0, gap=0.0, is_hydroelastic=True, sdf_max_resolution=8)
    builder.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.1), wp.quat_identity()), hx=0.2, hy=0.2, hz=0.1, cfg=cfg)
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.095), wp.quat_identity()))
    cfg.density = mass / 0.008
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    model = builder.finalize()
    model.mujoco.condim.fill_(condim)
    qd = model.joint_qd.numpy()
    qd[2] = velocity
    model.joint_qd.assign(qd)
    contacts = newton.Contacts(
        rigid_contact_max=count, soft_contact_max=0, device=model.device, per_contact_shape_properties=True
    )
    contacts.rigid_contact_count.fill_(count)
    contacts.rigid_contact_shape0.fill_(0)
    contacts.rigid_contact_shape1.fill_(1)
    contacts.rigid_contact_point0.fill_(wp.vec3(0.0, 0.0, 0.0))
    contacts.rigid_contact_point1.fill_(wp.vec3(0.0, 0.0, -0.1))
    contacts.rigid_contact_normal.fill_(wp.vec3(0.0, 0.0, 1.0))
    weights = np.arange(1, count + 1, dtype=np.float32)
    weights /= weights.sum()
    contacts.rigid_contact_stiffness.assign(stiffness * weights)
    contacts.rigid_contact_damping.assign(damping * weights)
    contacts.contact_generation.fill_(1)
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        use_speculative_contacts=use_speculative_contacts,
        **({} if hydroelastic_force_space is None else {"hydroelastic_force_space": hydroelastic_force_space}),
        nconmax=max(count, 16),
        njmax=max(count * 4, 64),
        iterations=50,
        tolerance=1.0e-10,
        cone=cone,
        integrator=integrator,
    )
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    return model, solver, contacts, state_in, state_out


def _make_box_on_slab(*, slab_stiffness, mass, clearance, gap):
    """Build the unreduced 4 mm box and 1 mm slab from issue #4561."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.rigid_gap = gap
    common = {
        "mu": 0.4,
        "gap": gap,
        "margin": 0.0,
        "is_hydroelastic": True,
        "sdf_narrow_band_range": (-0.001, 0.001),
        "sdf_padding": 0.001,
    }
    builder.add_shape_box(
        -1,
        xform=wp.transform((0.0, 0.0, -0.0005), wp.quat_identity()),
        hx=0.0155,
        hy=0.008,
        hz=0.0005,
        cfg=newton.ModelBuilder.ShapeConfig(kh=slab_stiffness, sdf_target_voxel_size=0.0002, **common),
    )
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.002 + clearance), wp.quat_identity()))
    builder.add_shape_box(
        body,
        hx=0.002,
        hy=0.002,
        hz=0.002,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=mass / 0.004**3,
            kh=1.0e12,
            sdf_target_voxel_size=0.00025,
            sdf_texture_format="float32",
            **common,
        ),
    )
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=8192,
        sdf_hydroelastic_config=HydroelasticSDF.Config(
            reduce_contacts=False,
            buffer_fraction=1.0,
        ),
    )
    solver = newton.solvers.SolverMuJoCo(
        model,
        use_mujoco_contacts=False,
        hydroelastic_force_space=True,
        cone="elliptic",
        nconmax=2048,
        njmax=8192,
    )
    return model, body, pipeline, pipeline.contacts(), solver


@requires(*MUJOCO_MODULES)
@unittest.skipUnless(_cuda_devices, "Requires a CUDA device")
class TestMuJoCoHydroelastic(unittest.TestCase):
    """Exercise hydroelastic material conversion through the actual solver."""

    def setUp(self):
        """Select a CUDA device for the solver and contact buffers."""
        scope = wp.ScopedDevice(_cuda_devices[0])
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def test_default_preserves_legacy_contact_response(self):
        """Keep default and explicit-False motion equal to the legacy mapping."""
        for cone in ("elliptic", "pyramidal"):
            for mass in (0.2, 2.0):
                with self.subTest(cone=cone, mass=mass):
                    outputs = []
                    for hydroelastic, mode in ((False, None), (True, None), (True, False)):
                        model, solver, contacts, state_in, state_out = _make_contacts(
                            4, mass=mass, cone=cone, hydroelastic_force_space=mode
                        )
                        if not hydroelastic:
                            flags = model.shape_flags.numpy()
                            flags &= ~int(newton.ShapeFlags.HYDROELASTIC)
                            model.shape_flags.assign(flags)
                        self.assertIsNone(solver._pressure_contact)
                        self.assertIsNone(solver._pressure_contact_key)
                        self.assertIsNone(solver._last_contact_timestep)
                        samples = []
                        for dt in (0.002, 0.006):
                            solver.step(state_in, state_out, None, contacts, dt)
                            with wp.ScopedCapture() as capture:
                                solver.step(state_in, state_out, None, contacts, dt)
                            wp.capture_launch(capture.graph)
                            samples.append(state_out.body_qd.numpy().copy())
                        outputs.append(samples)
                    for actual in outputs[1:]:
                        np.testing.assert_allclose(actual, outputs[0], atol=1.0e-6, rtol=1.0e-5)

    def test_force_space_preserves_control_callback(self):
        """Apply control forces and physical contacts without replacing callbacks."""
        _, solver, contacts, state_in, state_out = _make_contacts(4)
        calls = []
        force = wp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=float, device=solver.device)

        def control(model, data):
            calls.append(model)
            wp.copy(data.qfrc_applied, force)

        solver.mjw_model.callback.control = control
        dt = 0.002
        solver.step(state_in, state_out, None, contacts, dt)
        expected = (-0.1 + dt * (50.0 + 1.0)) / (1.0 + dt * dt * 1.0e4)
        self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)
        self.assertEqual(len(calls), 1)
        self.assertIs(solver.mjw_model.callback.control, control)

    def test_force_space_integrators(self):
        """Preserve the spring response through each supported split-step integrator."""
        for integrator in ("euler", "implicit", "implicitfast"):
            with self.subTest(integrator=integrator):
                _, solver, contacts, state_in, state_out = _make_contacts(4, integrator=integrator)
                dt = 0.006
                solver.step(state_in, state_out, None, contacts, dt)
                expected = (-0.1 + dt * 50.0) / (1.0 + dt * dt * 1.0e4)
                self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)

    def test_force_space_rejects_unsupported_options(self):
        """Reject cones and integrators that cannot give the promised response."""
        for options in ({"integrator": "rk4"}, {"integrator": "discrete"}, {"cone": "pyramidal"}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "hydroelastic_force_space=True"):
                _make_contacts(1, **options)
        with self.assertWarnsRegex(newton.NewtonWarning, "RK4"):
            _, solver, contacts, state_in, state_out = _make_contacts(
                1, integrator="rk4", hydroelastic_force_space=False
            )
        solver.step(state_in, state_out, None, contacts, 0.002)
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())
        _, solver, contacts, state_in, state_out = _make_contacts(1)
        solver.mjw_model.opt.integrator = solver._mujoco_warp.IntegratorType.RK4
        with self.assertRaisesRegex(ValueError, "hydroelastic_force_space=True.*RK4"):
            solver.step(state_in, state_out, None, contacts, 0.002)

    def test_force_space_rejects_unsupported_backends(self):
        """Reject an opt-in that the selected contact backend cannot apply."""
        model = newton.ModelBuilder().finalize()
        for options in ({"use_mujoco_contacts": True}, {"use_mujoco_cpu": True, "use_mujoco_contacts": False}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "hydroelastic_force_space=True"):
                newton.solvers.SolverMuJoCo(model, hydroelastic_force_space=True, **options)

    def test_contact_subdivision_preserves_spring_response(self):
        """Preserve the implicit spring response when subdividing a contact."""
        stiffness, dt, velocity, penetration = 1.0e4, 0.006, -0.1, 0.005
        for mass, damping in ((0.2, 0.0), (2.0, 20.0)):
            expected = (mass * velocity + dt * stiffness * penetration) / (mass + dt * damping + dt * dt * stiffness)
            for count in (1, 16):
                with self.subTest(mass=mass, damping=damping, count=count):
                    _model, solver, contacts, state_in, state_out = _make_contacts(
                        count, mass=mass, stiffness=stiffness, damping=damping, velocity=velocity
                    )
                    solver.step(state_in, state_out, None, contacts, dt)
                    actual = float(state_out.body_qd.numpy()[0, 2])
                    self.assertAlmostEqual(actual, expected, delta=2.0e-5)
                    # Verify the solve's reaction, not a reconstruction of ke * depth.
                    force = float(solver.mjw_data.qfrc_constraint.numpy()[0, 2])
                    self.assertAlmostEqual(force, mass * (expected - velocity) / dt, delta=0.02)
                    self.assertAlmostEqual(
                        float(contacts.rigid_contact_stiffness.numpy().sum()), stiffness, delta=0.002
                    )

    def test_small_weights_and_low_friction_preserve_response(self):
        """Preserve spring response beyond MuJoCo's impedance parameter bounds."""
        stiffness, dt, velocity = 1.0e4, 0.0002, -0.1
        expected = (velocity + dt * stiffness * 0.005) / (1.0 + dt * dt * stiffness)
        for mu in (0.0, 1.0e-4, 0.4):
            for count in (1, 16):
                with self.subTest(mu=mu, count=count):
                    _, solver, contacts, state_in, state_out = _make_contacts(count, mu=mu)
                    solver.step(state_in, state_out, None, contacts, dt)
                    self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)
                    if mu < 1.0e-3:
                        cids = solver._contact_tid_to_cid.numpy()[:count]
                        scale = np.linalg.norm(solver.mjw_data.contact.frame.numpy()[cids, 1], axis=1)
                        np.testing.assert_allclose(scale, max(mu / 1.0e-5, 1.0), rtol=1.0e-4)
                        np.testing.assert_allclose(solver.mjw_data.contact.friction.numpy()[cids, 0], 1.0e-5)

    def test_soft_contacts_preserve_elliptic_friction_metric(self):
        """Retain the cone metric for small sliding, torsional and rolling weights."""
        dt = 1.0 / 7680.0
        for condim in (3, 4, 6):
            with self.subTest(condim=condim):
                _, solver, contacts, state_in, state_out = _make_contacts(4, stiffness=1.0, velocity=0.0, condim=condim)
                solver.step(state_in, state_out, None, contacts, dt)
                data = solver.mjw_data
                mapping = solver._contact_tid_to_cid.numpy()[:4]
                scale = np.linalg.norm(data.contact.frame.numpy()[mapping, 1], axis=1)
                sliding, torsional, rolling = solver.mjw_model.geom_friction.numpy()[0, 0]
                np.testing.assert_allclose(
                    data.contact.friction.numpy()[mapping]
                    * np.stack([scale, scale, np.ones_like(scale), scale, scale], axis=1),
                    np.tile([sliding, sliding, torsional, rolling, rolling], (4, 1)),
                    rtol=1.0e-5,
                )
                addresses = data.contact.efc_address.numpy()
                friction = data.contact.friction.numpy()
                diagonal = data.efc.D.numpy()[0]
                weights = contacts.rigid_contact_stiffness.numpy()
                ir = float(solver.mjw_model.opt.impratio_invsqrt.numpy()[0])
                for tid, cid in enumerate(mapping):
                    rows = addresses[cid, :condim]
                    self.assertTrue(np.all(rows >= 0))
                    expected = np.full(condim, dt * dt * float(weights[tid]))
                    # MuJoCo's elliptic projection requires one common metric:
                    # D_normal = D_tangent * (ir * mu_sliding / mu_axis)^2.
                    mu = friction[cid, : condim - 1].astype(np.float64)
                    expected[1:] *= (mu / (ir * mu[0])) ** 2
                    np.testing.assert_allclose(diagonal[rows], expected, rtol=2.0e-6, atol=0.0)

    def test_static_pressure_force_matches_load(self):
        """Balance a known load at the material's physical equilibrium depth."""
        stiffness, penetration = 1.0e4, 0.005
        for mass in (0.5, 2.0):
            with self.subTest(mass=mass):
                _, solver, contacts, state_in, state_out = _make_contacts(
                    16,
                    mass=mass,
                    velocity=0.0,
                    mu=0.01,
                    gravity=-stiffness * penetration / mass,
                )
                solver.step(state_in, state_out, None, contacts, 0.002)
                self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 2]), 0.0, delta=2.0e-5)
                self.assertAlmostEqual(
                    float(solver.mjw_data.qfrc_constraint.numpy()[0, 2]), stiffness * penetration, delta=0.02
                )

    def test_nonuniform_patch_preserves_force_and_torque(self):
        """Match an implicit spring patch's force and torque after subdivision."""
        dt, stiffness, damping, mass = 0.002, 1.0e4, 20.0, 1.0
        points = np.array([[-0.01, -0.01], [0.01, -0.01], [-0.01, 0.01], [0.01, 0.01]])
        for subdivisions in (1, 4):
            with self.subTest(subdivisions=subdivisions):
                count = 4 * subdivisions
                model, solver, contacts, state_in, state_out = _make_contacts(count, condim=1, damping=damping)
                xy = np.repeat(points, subdivisions, axis=0)
                weights = np.repeat(np.array([0.1, 0.2, 0.3, 0.4]) / subdivisions, subdivisions)
                contacts.rigid_contact_point0.assign(np.column_stack((xy, np.zeros(count))))
                contacts.rigid_contact_point1.assign(np.column_stack((xy, np.full(count, -0.1))))
                contacts.rigid_contact_stiffness.assign(stiffness * weights)
                contacts.rigid_contact_damping.assign(damping * weights)
                jacobian = np.zeros((count, 6))
                jacobian[:, 2] = 1.0
                jacobian[:, 3] = xy[:, 1]
                jacobian[:, 4] = -xy[:, 0]
                mass_inverse = np.zeros((6, 6))
                mass_inverse[:3, :3] = np.eye(3) / mass
                mass_inverse[3:, 3:] = np.linalg.inv(model.body_inertia.numpy()[0])
                velocity = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0])
                k = stiffness * weights
                c = damping * weights
                # Solve the coupled backward-Euler spring equations independently.
                compliance = 1.0 / (dt * (c + dt * k))
                rhs = compliance * (k * 0.005 - (c + dt * k) * (jacobian @ velocity))
                force = np.linalg.solve(jacobian @ mass_inverse @ jacobian.T + np.diag(compliance), rhs)
                self.assertTrue(np.all(force > 0.0))
                expected = velocity + dt * mass_inverse @ jacobian.T @ force
                solver.step(state_in, state_out, None, contacts, dt)
                np.testing.assert_allclose(state_out.body_qd.numpy()[0], expected, atol=2.0e-5, rtol=2.0e-4)
                np.testing.assert_allclose(
                    solver.mjw_data.qfrc_constraint.numpy()[0], jacobian.T @ force, atol=0.02, rtol=2.0e-4
                )

    def test_dynamic_pairs_are_independent_across_worlds(self):
        """Preserve two-body mass response independently in batched worlds."""
        dt, stiffness = 0.002, 1.0e4
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        masses = (0.2, 2.0)
        for mass in masses:
            world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            newton.solvers.SolverMuJoCo.register_custom_attributes(world)
            for z, body_mass in ((0.0, 2.0 * mass), (0.195, mass)):
                body = world.add_body(xform=wp.transform((0.0, 0.0, z), wp.quat_identity()))
                cfg = newton.ModelBuilder.ShapeConfig(
                    density=body_mass / 0.008,
                    mu=0.01,
                    gap=0.0,
                    margin=0.0,
                    is_hydroelastic=True,
                    sdf_max_resolution=8,
                )
                world.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
            builder.add_world(world)
        model = builder.finalize()
        qd = model.joint_qd.numpy().reshape(4, 6)
        qd[1::2, 2] = -0.1
        model.joint_qd.assign(qd.reshape(-1))
        contacts = newton.Contacts(
            rigid_contact_max=8, soft_contact_max=0, device=model.device, per_contact_shape_properties=True
        )
        contacts.rigid_contact_count.fill_(8)
        contacts.rigid_contact_shape0.assign(np.repeat([0, 2], 4))
        contacts.rigid_contact_shape1.assign(np.repeat([1, 3], 4))
        contacts.rigid_contact_point0.fill_(wp.vec3(0.0, 0.0, 0.1))
        contacts.rigid_contact_point1.fill_(wp.vec3(0.0, 0.0, -0.1))
        contacts.rigid_contact_normal.fill_(wp.vec3(0.0, 0.0, 1.0))
        contacts.rigid_contact_stiffness.assign(stiffness * np.tile([0.1, 0.2, 0.3, 0.4], 2))
        contacts.contact_generation.fill_(1)
        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_contacts=False,
            hydroelastic_force_space=True,
            separate_worlds=True,
            cone="elliptic",
            nconmax=16,
            njmax=64,
            iterations=50,
            tolerance=1.0e-10,
        )
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver.step(state_in, state_out, None, contacts, dt)
        velocity = state_out.body_qd.numpy()
        for world, mass in enumerate(masses):
            inv_mass = 1.5 / mass
            relative = (-0.1 + dt * inv_mass * stiffness * 0.005) / (1.0 + dt * dt * inv_mass * stiffness)
            force = (relative + 0.1) / (dt * inv_mass)
            self.assertAlmostEqual(float(velocity[2 * world, 2]), -dt * force / (2.0 * mass), delta=2.0e-5)
            self.assertAlmostEqual(float(velocity[2 * world + 1, 2]), -0.1 + dt * force / mass, delta=2.0e-5)

    def test_high_impratio_friction_does_not_reverse_sliding(self):
        """Bound tangential relaxation for soft hydroelastic rows at high impratio."""
        dt, stiffness, mass, velocity = 1.0 / 600.0, 1000.0, 0.01, 0.001
        for count in (1, 16):
            with self.subTest(count=count):
                model, solver, contacts, state_in, state_out = _make_contacts(
                    count, mass=mass, stiffness=stiffness, velocity=0.0
                )
                model.shape_material_kf.fill_(1000.0)
                solver.mjw_model.opt.impratio_invsqrt.fill_(1.0 / np.sqrt(1000.0))
                contacts.rigid_contact_point0.fill_(wp.vec3(0.0, 0.0, 0.0975))
                contacts.rigid_contact_point1.fill_(wp.vec3(0.0, 0.0, -0.0025))
                qd = state_in.joint_qd.numpy()
                qd[0] = velocity
                state_in.joint_qd.assign(qd)
                newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                solver.step(state_in, state_out, None, contacts, dt)
                actual = float(state_out.body_qd.numpy()[0, 0])
                force = solver.mjw_data.qfrc_constraint.numpy()[0]
                self.assertLess(abs(float(force[0])), 0.4 * float(force[2]))
                self.assertGreaterEqual(actual, -1.0e-7)
                self.assertLessEqual(abs(actual), velocity)
                data = solver.mjw_data
                cids = solver._contact_tid_to_cid.numpy()[:count]
                scale2 = np.sum(data.contact.frame.numpy()[cids, 1] ** 2, axis=1)
                rows = data.contact.efc_address.numpy()[cids, 1]
                weight = np.sum(scale2 * data.efc.D.numpy()[0, rows])
                expected = velocity / (1.0 + weight / mass)
                self.assertAlmostEqual(actual, expected, delta=1.0e-7)

        # A deliberately slow authored friction slope must also survive scaling.
        model, solver, contacts, state_in, state_out = _make_contacts(1, mass=mass, stiffness=stiffness, velocity=0.0)
        model.shape_material_kf.fill_(0.1)
        contacts.rigid_contact_point0.fill_(wp.vec3(0.0, 0.0, 0.0975))
        contacts.rigid_contact_point1.fill_(wp.vec3(0.0, 0.0, -0.0025))
        qd = state_in.joint_qd.numpy()
        qd[0] = velocity
        state_in.joint_qd.assign(qd)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        solver.step(state_in, state_out, None, contacts, dt)
        self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 0]), velocity * (1.0 - 0.1 * dt / mass), delta=2.0e-8)

    def test_stiff_contact_preserves_elliptic_friction(self):
        """Stop sliding inside the cone without overweighting stiff friction rows."""
        model, solver, contacts, state_in, state_out = _make_contacts(1, stiffness=1.0e10, velocity=0.0)
        model.shape_material_kf.fill_(1000.0)
        solver.mjw_model.opt.impratio_invsqrt.fill_(0.1)
        # Keep the 5 mm separation while placing the midpoint at the center of mass.
        contacts.rigid_contact_point0.fill_(wp.vec3(0.0, 0.0, 0.0975))
        contacts.rigid_contact_point1.fill_(wp.vec3(0.0, 0.0, -0.0025))
        qd = model.joint_qd.numpy()
        qd[0] = 0.1
        state_in.joint_qd.assign(qd)
        model.joint_qd.assign(qd)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver.step(state_in, state_out, None, contacts, 0.006)
        force = solver.mjw_data.qfrc_constraint.numpy()[0]
        self.assertLess(abs(float(force[0])), 0.4 * float(force[2]))
        self.assertAlmostEqual(float(force[0]), -0.1 / 0.006, delta=0.01)
        self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 0]), 0.0, delta=2.0e-5)
        cid = int(solver._contact_tid_to_cid.numpy()[0])
        scale = float(np.linalg.norm(solver.mjw_data.contact.frame.numpy()[cid, 1]))
        self.assertAlmostEqual(scale, 0.1, delta=1.0e-4)

    def test_cached_speculative_contact_keeps_activation_response(self):
        """Do not reinterpret a speculative placeholder as a pressure weight."""
        velocities = []
        for hydroelastic in (False, True):
            model, solver, contacts, state_in, state_out = _make_contacts(
                1, mass=0.01, stiffness=5.0e8, velocity=0.0, mu=0.0
            )
            if not hydroelastic:
                flags = model.shape_flags.numpy()
                flags &= ~int(newton.ShapeFlags.HYDROELASTIC)
                model.shape_flags.assign(flags)
            q = state_in.joint_q.numpy()
            q[2] = 0.105
            state_in.joint_q.assign(q)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            with wp.ScopedCapture() as capture:
                solver.step(state_in, state_out, None, contacts, 1.0 / 600.0)
            wp.capture_launch(capture.graph)
            q[2] = 0.0999
            state_in.joint_q.assign(q)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            wp.capture_launch(capture.graph)
            samples = [float(state_out.body_qd.numpy()[0, 2])]
            for dt in (1.0 / 600.0, 1.0 / 1200.0):
                solver._invalidate_contact_fast_path()
                solver.step(state_in, state_out, None, contacts, dt)
                samples.append(float(state_out.body_qd.numpy()[0, 2]))
            velocities.append(samples)
            if hydroelastic:
                # A newly generated penetrating patch carries physical weights.
                contacts.contact_generation.fill_(2)
                contacts.rigid_contact_stiffness.fill_(1.0e4)
                solver.step(state_in, state_out, None, contacts, dt)
                expected = dt * 1.0e4 * 0.0001 / (0.01 + dt * dt * 1.0e4)
                self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)
        np.testing.assert_allclose(velocities[1], velocities[0], atol=1.0e-6, rtol=1.0e-5)

    def test_force_space_with_speculative_contact_braking(self):
        """Preserve speculative braking when a cached pressure contact separates."""
        velocities = []
        for enabled in (False, True):
            model, solver, contacts, state_in, state_out = _make_contacts(
                1, hydroelastic_force_space=enabled, use_speculative_contacts=True
            )
            dt = 0.006
            solver.step(state_in, state_out, None, contacts, dt)
            if enabled:
                expected = (-0.1 + dt * 50.0) / (1.0 + dt * dt * 1.0e4)
                self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)

            # Reuse the pressure contact after moving apart, with enough closing
            # velocity that the speculative normal constraint must brake it.
            q = state_in.joint_q.numpy()
            q[2] = 0.105
            state_in.joint_q.assign(q)
            qd = state_in.joint_qd.numpy()
            qd[2] = -2.0
            state_in.joint_qd.assign(qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            samples = []
            for dt in (0.006, 0.003):
                solver.step(state_in, state_out, None, contacts, dt)
                with wp.ScopedCapture() as capture:
                    solver.step(state_in, state_out, None, contacts, dt)
                wp.capture_launch(capture.graph)
                samples.append(float(state_out.body_qd.numpy()[0, 2]))
                self.assertEqual(int(solver.mjw_data.nacon.numpy()[0]), 1)
                self.assertEqual(int(solver.mjw_data.contact.dim.numpy()[0]), 1)
                self.assertEqual(int(solver._contact_count.numpy()[0]), 0)
            velocities.append(samples)
        np.testing.assert_allclose(velocities[1], velocities[0], atol=1.0e-6, rtol=1.0e-5)

    def test_cached_timestep_changes_and_graph_replay(self):
        """Refresh material parameters when a graph changes timestep or contact buffer."""
        _, solver, contacts, state_in, state_out = _make_contacts(16, mu=0.01)
        steps = (0.002, 0.006, 0.0002)
        for dt in steps:
            solver.step(state_in, state_out, None, contacts, dt)
        outputs = [solver.model.state() for _ in steps]
        with wp.ScopedCapture() as capture:
            for dt, output in zip(steps, outputs, strict=True):
                solver.step(state_in, output, None, contacts, dt)
        for _ in range(2):
            wp.capture_launch(capture.graph)
            for dt, output in zip(steps, outputs, strict=True):
                expected = (-0.1 + dt * 50.0) / (1.0 + dt * dt * 1.0e4)
                self.assertAlmostEqual(float(output.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)
        self.assertEqual(int(contacts.contact_generation.numpy()[0]), 1)

        # Warm both buffers before capture so replay cannot rely on Python's reset.
        _, solver, contacts, state_in, state_out = _make_contacts(1)
        other = newton.Contacts(
            rigid_contact_max=1, soft_contact_max=0, device=solver.model.device, per_contact_shape_properties=True
        )
        for name, value in vars(contacts).items():
            target = getattr(other, name, None)
            if isinstance(value, wp.array) and isinstance(target, wp.array):
                wp.copy(target, value)
        other.rigid_contact_stiffness.assign(4.0 * contacts.rigid_contact_stiffness.numpy())
        for buffer in (contacts, other):
            buffer.rigid_contact_point0.fill_(wp.vec3(0.0, 0.0, 0.0975))
            buffer.rigid_contact_point1.fill_(wp.vec3(0.0, 0.0, -0.0025))
        qd = state_in.joint_qd.numpy()
        qd[0] = 0.001
        state_in.joint_qd.assign(qd)
        newton.eval_fk(solver.model, state_in.joint_q, state_in.joint_qd, state_in)
        graphs, expected = [], []
        for buffer in (contacts, other):
            solver.step(state_in, state_out, None, buffer, 0.002)
            expected.append(state_out.body_qd.numpy().copy())
            with wp.ScopedCapture() as capture:
                solver.step(state_in, state_out, None, buffer, 0.002)
            graphs.append(capture.graph)
        for index in (0, 1, 0):
            wp.capture_launch(graphs[index])
            np.testing.assert_allclose(state_out.body_qd.numpy(), expected[index], atol=1.0e-8, rtol=1.0e-5)

    def test_disabled_passive_forces_keep_hydroelastic_response(self):
        """Retain hydroelastic constraints with passive forces, actuation and sensors disabled."""
        import mujoco_warp

        _, solver, contacts, state_in, state_out = _make_contacts(16, mu=0.01)
        solver.mjw_model.opt.disableflags |= (
            mujoco_warp.DisableBit.SPRING
            | mujoco_warp.DisableBit.DAMPER
            | mujoco_warp.DisableBit.ACTUATION
            | mujoco_warp.DisableBit.SENSOR
        )
        dt = 0.002
        solver.step(state_in, state_out, None, contacts, dt)
        expected = (-0.1 + dt * 50.0) / (1.0 + dt * dt * 1.0e4)
        self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)

    def test_disabled_contacts_ignore_stale_constraint_addresses(self):
        """Leave free motion intact when contact constraints are disabled after a step."""
        import mujoco_warp

        _, solver, contacts, state_in, state_out = _make_contacts(16)
        solver.step(state_in, state_out, None, contacts, 0.002)
        solver.mjw_model.opt.disableflags |= mujoco_warp.DisableBit.CONTACT
        solver.step(state_in, state_out, None, contacts, 0.002)
        self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 2]), -0.1, delta=2.0e-6)

    def test_thin_slab_impact(self):
        """Stop the reported falling box before it crosses the slab mid-plane."""
        for mass in (0.01, 1.0):
            with self.subTest(mass=mass):
                model, body, pipeline, contacts, solver = _make_box_on_slab(
                    slab_stiffness=1.0e11, mass=mass, clearance=0.0005, gap=0.0005
                )
                state_in, state_out = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
                peak = 0.0
                for _ in range(120):
                    state_in.clear_forces()
                    pipeline.collide(state_in, contacts)
                    solver.step(state_in, state_out, None, contacts, 1.0 / 1920.0)
                    state_in, state_out = state_out, state_in
                    peak = max(peak, 0.002 - float(state_in.body_q.numpy()[body, 2]))
                self.assertLess(peak, 0.0002)

    def test_soft_contact_friction_holds_to_coulomb_limit(self):
        """Hold a soft patch below the Coulomb limit and release it at that limit."""
        stiffness, penetration, mass, mu, dt = 1.0e4, 0.005, 1.0, 0.4, 0.002
        load = stiffness * penetration
        for count in (1, 16):
            with self.subTest(count=count):
                model, solver, contacts, state_in, state_out = _make_contacts(
                    count, mass=mass, stiffness=stiffness, velocity=0.0, mu=mu, gravity=-load / mass
                )
                # Keep the 5 mm separation while placing the midpoint at the center of mass.
                contacts.rigid_contact_point0.fill_(wp.vec3(0.0, 0.0, 0.0975))
                contacts.rigid_contact_point1.fill_(wp.vec3(0.0, 0.0, -0.0025))
                observables = solver.observables(contacts=contacts, kinds={solver.ObservableKind.CONTACT_F})

                def push(force, steps, solver=solver, contacts=contacts, observables=observables):
                    nonlocal state_in, state_out
                    for _ in range(steps):
                        state_in.clear_forces()
                        state_in.body_f.assign(np.array([[force, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
                        solver.step(state_in, state_out, None, contacts, dt, observables=observables)
                        state_in, state_out = state_out, state_in

                # Below the limit, friction balances the push and the patch only
                # creeps slowly. The original mapping reaches about 0.27 m/s.
                push(0.5 * mu * load, 20)
                self.assertLess(abs(float(state_in.body_qd.numpy()[0, 0])), 0.002)
                reaction = solver.mjw_data.qfrc_constraint.numpy()[0]
                self.assertAlmostEqual(float(reaction[0]), -0.5 * mu * load, delta=0.02)
                self.assertAlmostEqual(float(reaction[2]), load, delta=0.02)
                # The scaled tangent rows must not leak into world-frame forces.
                force = observables.contact_f.numpy()[:count, :3].sum(axis=0)
                np.testing.assert_allclose(force, [0.5 * mu * load, 0.0, -load], atol=0.02)

                # Above the limit, friction saturates at mu times the normal force.
                speed = float(state_in.body_qd.numpy()[0, 0])
                push(1.5 * mu * load, 5)
                reaction = solver.mjw_data.qfrc_constraint.numpy()[0]
                self.assertAlmostEqual(float(reaction[0]), -mu * float(reaction[2]), delta=0.02 * mu * load)
                gained = float(state_in.body_qd.numpy()[0, 0]) - speed
                self.assertAlmostEqual(
                    gained, 5 * dt * 0.5 * mu * load / mass, delta=0.1 * 5 * dt * 0.5 * mu * load / mass
                )

        # Rolling rows put the scaled tangents on the angular Jacobian. In the world frame
        # their torque must still saturate at the rolling coefficient times the normal force.
        model, solver, contacts, state_in, state_out = _make_contacts(
            4, mass=mass, stiffness=stiffness, velocity=0.0, mu=mu, gravity=-load / mass, condim=6
        )
        contacts.rigid_contact_point0.fill_(wp.vec3(0.0, 0.0, 0.0975))
        contacts.rigid_contact_point1.fill_(wp.vec3(0.0, 0.0, -0.0025))
        qd = model.joint_qd.numpy()
        qd[4] = 20.0
        state_in.joint_qd.assign(qd)
        model.joint_qd.assign(qd)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver.step(state_in, state_out, None, contacts, dt)
        mapping = solver._contact_tid_to_cid.numpy()[:4]
        self.assertTrue(np.all(np.linalg.norm(solver.mjw_data.contact.frame.numpy()[mapping, 1], axis=1) > 1.0))
        rolling = float(solver.mjw_model.geom_friction.numpy()[0, 0, 2])
        reaction = solver.mjw_data.qfrc_constraint.numpy()[0]
        self.assertAlmostEqual(float(reaction[4]), -rolling * float(reaction[2]), delta=0.02 * rolling * load)

    def test_soft_slab_friction_hold(self):
        """Press the reported box onto the soft slab and push it sideways past the friction limit."""
        mass, mu, load, dt = 0.02, 0.4, 1.0, 1.0 / 1920.0
        model, body, pipeline, contacts, solver = _make_box_on_slab(
            slab_stiffness=1.5e9, mass=mass, clearance=0.0, gap=0.0
        )
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        # Press the box onto the slab with a 1 N total normal load.
        wrench = np.zeros((model.body_count, 6), dtype=np.float32)
        wrench[body, 2] = -(load + mass * model.gravity.numpy()[0, 2])

        def push(force, steps):
            nonlocal state_in, state_out
            wrench[body, 0] = force
            state_in.body_f.assign(wrench)
            for _ in range(steps):
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, None, contacts, dt)
                state_in, state_out = state_out, state_in
                # The constant wrench is the only applied force.
                wp.copy(state_in.body_f, state_out.body_f)

        push(0.0, 150)
        spin = 0.0
        for _ in range(50):
            push(0.0, 1)
            spin = max(spin, float(np.abs(state_in.body_qd.numpy()[body, 3:]).max()))
        self.assertLess(spin, 0.02)
        start = float(state_in.body_q.numpy()[body, 0])
        # Half the friction limit only tilts the box on the compliant slab. With
        # friction as soft as the normal row, the box slides at 3 % of the limit.
        for step in range(100):
            push(0.5 * mu * load * (step + 1) / 100, 1)
        push(0.5 * mu * load, 100)
        self.assertLess(abs(float(state_in.body_q.numpy()[body, 0]) - start), 6.0e-5)
        self.assertLess(abs(float(state_in.body_qd.numpy()[body, 0])), 1.0e-3)
        # Beyond the limit the box accelerates at the excess force over its mass.
        speed = float(state_in.body_qd.numpy()[body, 0])
        push(1.5 * mu * load, 20)
        gained = float(state_in.body_qd.numpy()[body, 0]) - speed
        expected = 20 * dt * 0.5 * mu * load / mass
        self.assertAlmostEqual(gained, expected, delta=0.2 * expected)


if __name__ == "__main__":
    unittest.main()
