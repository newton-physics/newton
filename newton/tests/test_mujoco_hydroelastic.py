# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test opt-in hydroelastic compliance on MuJoCo Warp."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.geometry import HydroelasticSDF
from newton.tests.unittest_utils import MUJOCO_MODULES, get_selected_cuda_test_devices, requires

_cuda_devices = get_selected_cuda_test_devices()


def _contacts(count=1, stiffness=1.0e4, damping=0.0):
    """Split a 5 mm penetrating patch into unequal pressure weights."""
    contacts = newton.Contacts(rigid_contact_max=count, soft_contact_max=0, per_contact_shape_properties=True)
    contacts.rigid_contact_count.fill_(count)
    contacts.rigid_contact_shape0.fill_(0)
    contacts.rigid_contact_shape1.fill_(1)
    # Place the midpoint at the body's center to isolate translation.
    contacts.rigid_contact_point0.fill_(wp.vec3(0.0, 0.0, 0.0975))
    contacts.rigid_contact_point1.fill_(wp.vec3(0.0, 0.0, -0.0025))
    contacts.rigid_contact_normal.fill_(wp.vec3(0.0, 0.0, 1.0))
    weights = np.arange(1, count + 1, dtype=np.float32)
    weights /= weights.sum()
    contacts.rigid_contact_stiffness.assign(stiffness * weights)
    contacts.rigid_contact_damping.assign(damping * weights)
    contacts.contact_generation.fill_(1)
    return contacts


def _solver(*, mass=1.0, velocity=-0.1, condim=3, hydroelastic_force_space=True, **options):
    """Build a box above a fixed hydroelastic surface."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.4, margin=0.0, gap=0.0, is_hydroelastic=True, sdf_max_resolution=8)
    builder.add_shape_box(-1, hx=0.2, hy=0.2, hz=0.1, cfg=cfg)
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.095), wp.quat_identity()))
    cfg.density = mass / 0.008
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    model = builder.finalize()
    model.mujoco.condim.fill_(condim)
    options.setdefault("cone", "elliptic")
    if hydroelastic_force_space is not None:
        options["hydroelastic_force_space"] = hydroelastic_force_space
    solver = newton.solvers.SolverMuJoCo(
        model, use_mujoco_contacts=False, nconmax=16, njmax=128, iterations=50, tolerance=1.0e-10, **options
    )
    state_in, state_out = model.state(), model.state()
    state_in.joint_qd.assign(np.array([0.0, 0.0, velocity, 0.0, 0.0, 0.0]))
    newton.eval_fk(model, model.joint_q, state_in.joint_qd, state_in)
    return model, solver, state_in, state_out


@requires(*MUJOCO_MODULES)
@unittest.skipUnless(_cuda_devices, "Requires a CUDA device")
class TestMuJoCoHydroelastic(unittest.TestCase):
    """Check normal compliance, friction, and cached contact conversion."""

    def setUp(self):
        """Select the CUDA device for all test buffers."""
        self.enterContext(wp.ScopedDevice(_cuda_devices[0]))

    def test_default_preserves_legacy_response(self):
        """Keep default and explicit-False behavior equal to ordinary contacts."""
        for cone in ("elliptic", "pyramidal"):
            with self.subTest(cone=cone):
                velocities = []
                for hydroelastic, mode in ((False, None), (True, None), (True, False)):
                    model, solver, state_in, state_out = _solver(cone=cone, hydroelastic_force_space=mode)
                    if not hydroelastic:
                        flags = model.shape_flags.numpy()
                        flags &= ~int(newton.ShapeFlags.HYDROELASTIC)
                        model.shape_flags.assign(flags)
                    solver.step(state_in, state_out, None, _contacts(4), 0.002)
                    velocities.append(state_out.body_qd.numpy())
                for actual in velocities[1:]:
                    np.testing.assert_allclose(actual, velocities[0], atol=1.0e-6, rtol=1.0e-5)

    def test_normal_response(self):
        """Preserve spring response across masses, subdivision, and timestep."""
        for mass, damping, integrator in ((0.2, 0.0, "euler"), (2.0, 20.0, "implicit"), (1.0, 0.0, "implicitfast")):
            _, solver, state_in, state_out = _solver(mass=mass, integrator=integrator)
            for count in (1, 16):
                contacts = _contacts(count, damping=damping)
                # The shorter step puts small weights below MuJoCo's impedance floor.
                for dt in (0.006, 0.0002):
                    with self.subTest(mass=mass, count=count, dt=dt):
                        solver.step(state_in, state_out, None, contacts, dt)
                        expected = (-0.1 * mass + dt * 50.0) / (mass + dt * damping + dt * dt * 1.0e4)
                        self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)
                        force = float(solver.mjw_data.qfrc_constraint.numpy()[0, 2])
                        self.assertAlmostEqual(force, mass * (expected + 0.1) / dt, delta=0.02)

    def test_unsupported_options(self):
        """Reject backends, cones, and integrators that cannot apply the fix."""
        model, _, _, _ = _solver()
        for options in (
            {"use_mujoco_cpu": True},
            {"use_mujoco_contacts": True},
            {"integrator": "rk4"},
            {"integrator": "discrete"},
            {"cone": "pyramidal"},
        ):
            settings = {"use_mujoco_contacts": False, "cone": "elliptic", **options}
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "hydroelastic_force_space=True"):
                newton.solvers.SolverMuJoCo(model, hydroelastic_force_space=True, **settings)

    def test_friction_metric(self):
        """Preserve sliding, torsional, and rolling limits for tiny weights."""
        _, solver, state_in, state_out = _solver(condim=6, velocity=0.0)
        contacts, dt = _contacts(4, stiffness=1.0), 1.0 / 7680.0
        solver.step(state_in, state_out, None, contacts, dt)
        data = solver.mjw_data
        cids = solver._contact_tid_to_cid.numpy()[:4]
        scale = np.linalg.norm(data.contact.frame.numpy()[cids, 1], axis=1)
        friction = data.contact.friction.numpy()[cids]
        sliding, torsional, rolling = solver.mjw_model.geom_friction.numpy()[0, 0]
        np.testing.assert_allclose(
            friction * np.column_stack((scale, scale, np.ones(4), scale, scale)),
            np.tile([sliding, sliding, torsional, rolling, rolling], (4, 1)),
            rtol=1.0e-5,
        )
        rows = data.contact.efc_address.numpy()[cids, :6]
        self.assertTrue(np.all(rows >= 0))
        expected = np.repeat((dt * dt * contacts.rigid_contact_stiffness.numpy())[:, None], 6, axis=1)
        ir = float(solver.mjw_model.opt.impratio_invsqrt.numpy()[0])
        # The cone projection requires the same metric across all six rows.
        expected[:, 1:] *= (friction / (ir * friction[:, :1])) ** 2
        np.testing.assert_allclose(data.efc.D.numpy()[0, rows], expected, rtol=2.0e-6, atol=0.0)

    def test_captured_contact_changes(self):
        """Refresh cached material data when a graph changes its timestep or buffer."""
        model, solver, state_in, state_out = _solver()
        state_in.joint_qd.assign(np.array([0.001, 0.0, -0.1, 0.0, 0.0, 0.0]))
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        a, b = _contacts(), _contacts(stiffness=4.0e4)
        graphs, expected = [], []
        # Build fresh references outside capture. A's graphs share a buffer identity.
        for contacts, dt in ((a, 0.002), (a, 0.006), (b, 0.002)):
            solver._invalidate_contact_fast_path()
            solver.step(state_in, state_out, None, contacts, dt)
            expected.append(state_out.body_qd.numpy().copy())
            with wp.ScopedCapture() as capture:
                solver.step(state_in, state_out, None, contacts, dt)
            graphs.append(capture.graph)
        # Change timestep and buffer separately, so neither check masks the other.
        for index in (0, 1, 0, 2, 0):
            wp.capture_launch(graphs[index])
            np.testing.assert_allclose(state_out.body_qd.numpy(), expected[index], atol=1.0e-8, rtol=1.0e-5)

    def test_cached_speculative_contact(self):
        """Keep activation stiffness until collision generates a pressure contact."""
        velocities = []
        for enabled in (False, True):
            model, solver, state_in, state_out = _solver(
                mass=0.01, velocity=0.0, condim=1, hydroelastic_force_space=enabled
            )
            contacts, dt = _contacts(stiffness=5.0e8), 1.0 / 600.0
            q = state_in.joint_q.numpy()
            q[2] = 0.105
            state_in.joint_q.assign(q)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            with wp.ScopedCapture() as capture:
                solver.step(state_in, state_out, None, contacts, dt)
            wp.capture_launch(capture.graph)
            q[2] = 0.0999
            state_in.joint_q.assign(q)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            wp.capture_launch(capture.graph)
            velocities.append(state_out.body_qd.numpy())
            if enabled:
                contacts.contact_generation.fill_(2)
                contacts.rigid_contact_stiffness.fill_(1.0e4)
                solver.step(state_in, state_out, None, contacts, dt)
                expected = dt * 1.0e4 * 0.0001 / (0.01 + dt * dt * 1.0e4)
                self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)
        np.testing.assert_allclose(velocities[1], velocities[0], atol=1.0e-6, rtol=1.0e-5)

    def test_soft_slab_friction(self):
        """Hold the reported box below the friction limit and release it above."""
        mass, mu, load, dt = 0.02, 0.4, 1.0, 1.0 / 1920.0
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        common = {
            "mu": mu,
            "gap": 0.0,
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
            cfg=newton.ModelBuilder.ShapeConfig(kh=1.5e9, sdf_target_voxel_size=0.0002, **common),
        )
        body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.002), wp.quat_identity()))
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
            sdf_hydroelastic_config=HydroelasticSDF.Config(reduce_contacts=False, buffer_fraction=1.0),
        )
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverMuJoCo(
            model, use_mujoco_contacts=False, hydroelastic_force_space=True, cone="elliptic", nconmax=2048, njmax=8192
        )
        observables = solver.observables(contacts=contacts, kinds={solver.ObservableKind.CONTACT_F})
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        wrench = np.zeros((model.body_count, 6), dtype=np.float32)
        wrench[body, 2] = -(load + mass * model.gravity.numpy()[0, 2])

        def push(force, steps):
            nonlocal state_in, state_out
            wrench[body, 0] = force
            state_in.body_f.assign(wrench)
            for _ in range(steps):
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, None, contacts, dt, observables=observables)
                state_in, state_out = state_out, state_in
                wp.copy(state_in.body_f, state_out.body_f)

        push(0.0, 200)
        self.assertLess(float(np.abs(state_in.body_qd.numpy()[body, 3:]).max()), 0.02)
        start = float(state_in.body_q.numpy()[body, 0])
        for step in range(100):
            push(0.5 * mu * load * (step + 1) / 100, 1)
        push(0.5 * mu * load, 100)
        self.assertLess(abs(float(state_in.body_q.numpy()[body, 0]) - start), 6.0e-5)
        self.assertLess(abs(float(state_in.body_qd.numpy()[body, 0])), 1.0e-3)
        # Public forces must remain in world coordinates despite scaled tangent rows.
        count = int(contacts.rigid_contact_count.numpy()[0])
        np.testing.assert_allclose(
            observables.contact_f.numpy()[:count, :3].sum(axis=0), [0.5 * mu * load, 0.0, -load], atol=0.02
        )
        speed = float(state_in.body_qd.numpy()[body, 0])
        push(1.5 * mu * load, 20)
        gained = float(state_in.body_qd.numpy()[body, 0]) - speed
        expected = 20 * dt * 0.5 * mu * load / mass
        self.assertAlmostEqual(gained, expected, delta=0.2 * expected)


if __name__ == "__main__":
    unittest.main()
