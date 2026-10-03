# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test opt-in contact compliance on native dense and matrix-free routes."""

import os
import unittest

import numpy as np
import warp as wp

import newton
from newton.geometry import HydroelasticSDF
from newton.solvers import SolverFeatherPGS


def run_fixture(
    *,
    articulated,
    enabled,
    steps=200,
    dt=0.005,
    iterations=8,
    stock=False,
    lateral_force=0.0,
    friction_scale=1.0,
    stiffness=3000.0,
    height=0.05,
    kinematic=False,
    solver_options=None,
):
    """Run actual Newton sphere/plane contacts through one physical step per tick."""
    device = os.environ.get("HYDRO_TEST_DEVICE", "cuda:0")
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.rigid_gap = 0.005
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        xform = wp.transform(wp.vec3(0, 0, height), wp.quat_identity())
        if articulated:
            body = builder.add_link(xform=xform)
            if lateral_force:
                joint = builder.add_joint_d6(
                    parent=-1,
                    child=body,
                    parent_xform=xform,
                    linear_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
                    ],
                )
            else:
                joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.Z, parent_xform=xform)
            builder.add_articulation([joint])
        else:
            body = builder.add_body(xform=xform)
        builder.add_shape_sphere(
            body, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.3 / (4 / 3 * np.pi * 0.05**3), mu=0.5)
        )
        model = builder.finalize()
        if kinematic:
            flags = model.body_flags.numpy()
            flags[body] |= int(newton.BodyFlags.KINEMATIC)
            model.body_flags.assign(flags)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
        model.rigid_contact_max = 32
        contacts = pipeline.contacts()
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(32, dtype=float))
        extra = {} if stock else {"contact_compliance": enabled}
        options = dict(
            # Isolate the normal material law from positional patch friction.
            friction_anchor_beta=0.0,
            pgs_mode="matrix_free",
            pgs_schedule="interleaved",
            articulated_contact_response="immediate",
            enable_restitution=False,
            pgs_iterations=iterations,
            pgs_velocity_iterations=0,
            dense_max_constraints=32,
            mf_max_constraints=32,
            pgs_beta=0.05,
            **extra,
        )
        options.update(solver_options or {})
        solver = SolverFeatherPGS(model, **options)
        state, next_state = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        trace = []
        seen_paths = set()
        control = model.control()
        for step in range(steps):
            state.clear_forces()
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            contacts.rigid_contact_stiffness.fill_(stiffness)
            contacts.rigid_contact_damping.fill_(20.0)
            contacts.rigid_contact_friction.fill_(friction_scale)
            if lateral_force and step >= steps // 2:
                control.joint_f.assign(np.array([lateral_force, 0.0], dtype=np.float32))
            solver.step(state, next_state, control, contacts, dt)
            state, next_state = next_state, state
            seen_paths.update(solver.contact_path.numpy()[:count].tolist())
            trace.append(state.body_q.numpy()[body].copy())
        return np.asarray(trace), seen_paths, solver, float(model.body_mass.numpy()[body])


def run_native_hydro_fixture(*, articulated=True):
    """Consume emitted SDF hydro coefficients unchanged on dense or MF sphere rows."""
    with wp.ScopedDevice(os.environ.get("HYDRO_TEST_DEVICE", "cuda:0")):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.5,
            is_hydroelastic=True,
            sdf_max_resolution=32,
            sdf_narrow_band_range=(-0.01, 0.01),
            sdf_padding=0.006,
            gap=0.005,
            kh=1e7,
        )
        builder.add_shape_box(
            body=-1, hx=0.1, hy=0.1, hz=0.025, xform=wp.transform(wp.vec3(0, 0, -0.025), wp.quat_identity())
        )
        transform = wp.transform(wp.vec3(0, 0, 0.048), wp.quat_identity())
        if articulated:
            body = builder.add_link(xform=transform)
            joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.Z, parent_xform=transform)
            builder.add_articulation([joint])
        else:
            body = builder.add_body(xform=transform)
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 0.3 / (4 / 3 * np.pi * 0.05**3)
        builder.add_shape_sphere(body, radius=0.05, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            rigid_contact_max=512,
            deterministic=True,
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                reduce_contacts=True, moment_matching=True, anchor_contact=True, buffer_fraction=1.0
            ),
        )
        model.rigid_contact_max = 512
        inertia = tuple(x.numpy().copy() for x in (model.body_mass, model.body_com, model.body_inertia))
        initial = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, initial)
        contacts = pipeline.contacts()
        pipeline.collide(initial, contacts)
        input_contract = {
            name: getattr(initial, name).numpy().copy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")
        }
        count = int(contacts.rigid_contact_count.numpy()[0])
        stiffness = contacts.rigid_contact_stiffness.numpy()[:count].copy()
        damping = contacts.rigid_contact_damping.numpy()[:count].copy()
        friction = contacts.rigid_contact_friction.numpy()[:count].copy()
        if count <= 0 or count > 512 or not (stiffness > 0).any():
            raise AssertionError("Expected valid native hydro contacts with emitted positive stiffness")
        results = {}
        for enabled in (False, True):
            solver = SolverFeatherPGS(
                model,
                contact_compliance=enabled,
                friction_anchor_beta=0.0,
                pgs_mode="matrix_free",
                pgs_schedule="interleaved",
                articulated_contact_response="immediate",
                enable_restitution=False,
                pgs_iterations=128,
                pgs_contact_regularization=0.0,
                dense_max_constraints=2048,
                mf_max_constraints=32 if articulated else 2048,
                pgs_beta=0.05,
            )
            output = model.state()
            step_input = model.state()
            for name, value in input_contract.items():
                getattr(step_input, name).assign(value)
                np.testing.assert_array_equal(getattr(step_input, name).numpy(), value)
            step_input.clear_forces()
            solver.step(step_input, output, model.control(), contacts, 0.0025)
            paths = solver.contact_path.numpy()[:count]
            if not np.all(paths == (0 if articulated else 1)):
                raise AssertionError("Native hydro fixture did not use the requested contact route")
            impulses = solver.impulses if articulated else solver.mf_impulses
            row_types = solver.row_type if articulated else solver.mf_row_type
            results["compliant" if enabled else "stock_law"] = {
                "body_z_m": float(output.body_q.numpy()[body, 2]),
                "joint_velocity_m_s": float(output.joint_qd.numpy()[0]),
                "normal_impulse_N_s": float(impulses.numpy()[0, row_types.numpy()[0] == 0].sum()),
                "consumed_compliant_contacts": solver.compliance_contact_count,
            }
        for before, after in zip(inertia, (model.body_mass, model.body_com, model.body_inertia), strict=True):
            np.testing.assert_array_equal(before, after.numpy())
        np.testing.assert_array_equal(stiffness, contacts.rigid_contact_stiffness.numpy()[:count])
        np.testing.assert_array_equal(damping, contacts.rigid_contact_damping.numpy()[:count])
        np.testing.assert_array_equal(friction, contacts.rigid_contact_friction.numpy()[:count])
        results["native_material"] = {
            "contacts": count,
            "positive_stiffness_contacts": int((stiffness > 0).sum()),
            "stiffness_min_N_per_m": float(stiffness.min()),
            "stiffness_max_N_per_m": float(stiffness.max()),
            "damping_max_N_s_per_m": float(damping.max()),
            "friction_weight_min": float(friction.min()),
            "friction_weight_max": float(friction.max()),
            "mass_kg": float(inertia[0][body]),
        }
        return results


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestContactComplianceIntegration(unittest.TestCase):
    """Gate the experimental adapter against actual dense and MF contact paths."""

    def test_dense_articulated_support(self):
        """Recover finite-stiffness indentation on real dense articulated rows."""
        trace, paths, solver, mass = run_fixture(articulated=True, enabled=True)
        self.assertEqual(paths, {0})
        self.assertGreater(solver.compliance_contact_count, 0)
        self.assertLess(abs(trace[-1, 2] - (0.05 - mass * 9.81 / 3000)), 2e-6)

    def test_mf_free_body_support(self):
        """Recover the same indentation on real matrix-free free-body rows."""
        trace, paths, solver, mass = run_fixture(articulated=False, enabled=True)
        self.assertEqual(paths, {1})
        self.assertGreater(solver.compliance_contact_count, 0)
        self.assertLess(abs(trace[-1, 2] - (0.05 - mass * 9.81 / 3000)), 2e-6)

    def test_default_off_exact_state(self):
        """Match an omitted option to explicit OFF, not to a separate pristine solver."""
        for articulated in (False, True):
            original, _, _, _ = run_fixture(articulated=articulated, enabled=False, stock=True, steps=30)
            disabled, _, _, _ = run_fixture(articulated=articulated, enabled=False, steps=30)
            np.testing.assert_array_equal(original, disabled)

    def test_enabled_zero_stiffness_preserves_hard_contacts(self):
        """Preserve hard-contact motion with zero stiffness even when compliance is ON."""
        for articulated in (False, True):
            hard, paths, _, _ = run_fixture(articulated=articulated, enabled=False, stiffness=0, steps=60)
            noop, actual_paths, solver, _ = run_fixture(articulated=articulated, enabled=True, stiffness=0, steps=60)
            self.assertEqual(actual_paths, paths)
            self.assertEqual(solver.compliance_contact_count, 0)
            np.testing.assert_array_equal(hard, noop)

    def test_friction_uses_compliant_load_and_reducer_weight(self):
        """Hold below mu*mg and slip after reducing the exported friction weight."""
        held, _, solver, mass = run_fixture(articulated=True, enabled=True, lateral_force=0.5)
        slipped, _, _, _ = run_fixture(articulated=True, enabled=True, lateral_force=0.5, friction_scale=0.25)
        self.assertLess(abs(held[-1, 0]), 1e-5)
        self.assertGreater(slipped[-1, 0], 0.005)
        types = solver.row_type.numpy()[0]
        normal = solver.impulses.numpy()[0, types == 0].sum() / 0.005
        tangent = np.linalg.norm(solver.impulses.numpy()[0, types == 2]) / 0.005
        self.assertAlmostEqual(float(normal), mass * 9.81, delta=1e-4)
        self.assertAlmostEqual(float(tangent), 0.5, delta=1e-4)

    def test_native_hydro_material_pipeline(self):
        """Apply native pressure-weighted hydro coefficients without overwriting fields."""
        result = run_native_hydro_fixture()
        self.assertEqual(
            result["compliant"]["consumed_compliant_contacts"], result["native_material"]["positive_stiffness_contacts"]
        )
        self.assertGreater(
            abs(result["compliant"]["normal_impulse_N_s"] - result["stock_law"]["normal_impulse_N_s"]), 1e-5
        )
        self.assertTrue(
            all(np.isfinite(list(result[key].values())).all() for key in ("compliant", "stock_law", "native_material"))
        )

    def test_native_hydro_free_body_manifold(self):
        """Handle multiple MF hydro normals without indexing a dummy prescribed-target buffer."""
        result = run_native_hydro_fixture(articulated=False)
        self.assertGreater(result["native_material"]["positive_stiffness_contacts"], 1)
        self.assertEqual(
            result["compliant"]["consumed_compliant_contacts"], result["native_material"]["positive_stiffness_contacts"]
        )
        self.assertTrue(np.isfinite(list(result["compliant"].values())).all())


if __name__ == "__main__":
    unittest.main()
