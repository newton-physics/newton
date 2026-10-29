# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact compliance and contact regularization of SolverSteel."""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.contact_compliance import (
    material_coefficients,
    normal_coefficients,
    prepare_rows,
    start_step,
)
from newton._src.solvers.steel.kernels import PGS_CONSTRAINT_TYPE_CONTACT, compute_world_contact_bias
from newton.geometry import HydroelasticSDF
from newton.solvers import SolverSteel
from newton.tests.steel._compliance_reference import solve_reference, support_trajectory
from newton.tests.steel._fixtures import _build_mixed_response_model, _build_scissor_model, run_fixture
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def run_native_hydro_fixture(*, device, articulated=True):
    """Consume the emitted SDF hydroelastic coefficients unchanged on dense or free-body sphere rows."""
    with wp.ScopedDevice(device):
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
            raise AssertionError("Expected valid hydroelastic contacts with positive stiffness")
        results = {}
        for enabled in (False, True):
            solver = SolverSteel(
                model,
                solve_mode="matrix_free",
                contact_compliance=enabled,
                friction_anchor_beta=0.0,
                iterations=128,
                contact_regularization=0.0,
                articulation_constraint_max=2048,
                free_body_constraint_max=32 if articulated else 2048,
                baumgarte=0.05,
            )
            output = model.state()
            step_input = model.state()
            for name, value in input_contract.items():
                getattr(step_input, name).assign(value)
                np.testing.assert_array_equal(getattr(step_input, name).numpy(), value)
            step_input.clear_forces()
            solver.step(step_input, output, model.control(), contacts, 0.0025)
            paths = solver._contact_path.numpy()[:count]
            if not np.all(paths == (0 if articulated else 1)):
                raise AssertionError("Hydroelastic fixture did not use the requested contact route")
            impulses = solver._impulses if articulated else solver._mf_impulses
            row_types = solver._row_type if articulated else solver._mf_row_type
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


def test_dense_articulated_support(test, device):
    """Recover the finite-stiffness indentation on dense articulated rows."""
    trace, paths, solver, mass = run_fixture(device=device, articulated=True, enabled=True)
    test.assertEqual(paths, {0})
    test.assertGreater(solver.compliance_contact_count, 0)
    test.assertLess(abs(trace[-1, 2] - (0.05 - mass * 9.81 / 3000)), 2e-6)


def test_mf_free_body_support(test, device):
    """Recover the same indentation on free-body rows."""
    trace, paths, solver, mass = run_fixture(device=device, articulated=False, enabled=True)
    test.assertEqual(paths, {1})
    test.assertGreater(solver.compliance_contact_count, 0)
    test.assertLess(abs(trace[-1, 2] - (0.05 - mass * 9.81 / 3000)), 2e-6)


def test_default_off_exact_state(test, device):
    """Match an omitted option to explicit OFF, bitwise."""
    for articulated in (False, True):
        original, _, _, _ = run_fixture(device=device, articulated=articulated, enabled=False, stock=True, steps=30)
        disabled, _, _, _ = run_fixture(device=device, articulated=articulated, enabled=False, steps=30)
        np.testing.assert_array_equal(original, disabled)


def test_enabled_zero_stiffness_preserves_hard_contacts(test, device):
    """Preserve the rigid contact motion with zero stiffness even when compliance is on."""
    for articulated in (False, True):
        hard, paths, _, _ = run_fixture(device=device, articulated=articulated, enabled=False, stiffness=0, steps=60)
        noop, actual_paths, solver, _ = run_fixture(
            device=device, articulated=articulated, enabled=True, stiffness=0, steps=60
        )
        test.assertEqual(actual_paths, paths)
        test.assertEqual(solver.compliance_contact_count, 0)
        np.testing.assert_array_equal(hard, noop)


def test_friction_uses_compliant_load_and_contact_weight(test, device):
    """Hold below mu*m*g and slip after reducing the contact friction weight."""
    held, _, solver, mass = run_fixture(device=device, articulated=True, enabled=True, lateral_force=0.5)
    slipped, _, _, _ = run_fixture(
        device=device, articulated=True, enabled=True, lateral_force=0.5, friction_scale=0.25
    )
    test.assertLess(abs(held[-1, 0]), 1e-5)
    test.assertGreater(slipped[-1, 0], 0.005)
    types = solver._row_type.numpy()[0]
    normal = solver._impulses.numpy()[0, types == 0].sum() / 0.005
    tangent = np.linalg.norm(solver._impulses.numpy()[0, types == 2]) / 0.005
    test.assertAlmostEqual(float(normal), mass * 9.81, delta=1e-4)
    test.assertAlmostEqual(float(tangent), 0.5, delta=1e-4)


def test_native_hydro_material_pipeline(test, device):
    """Apply the pressure-weighted hydroelastic coefficients without overwriting the contact fields."""
    result = run_native_hydro_fixture(device=device)
    test.assertEqual(
        result["compliant"]["consumed_compliant_contacts"], result["native_material"]["positive_stiffness_contacts"]
    )
    test.assertGreater(abs(result["compliant"]["normal_impulse_N_s"] - result["stock_law"]["normal_impulse_N_s"]), 1e-5)
    test.assertTrue(
        all(np.isfinite(list(result[key].values())).all() for key in ("compliant", "stock_law", "native_material"))
    )


def test_native_hydro_free_body_manifold(test, device):
    """Handle several free-body hydroelastic normals without indexing the placeholder target buffer."""
    result = run_native_hydro_fixture(device=device, articulated=False)
    test.assertGreater(result["native_material"]["positive_stiffness_contacts"], 1)
    test.assertEqual(
        result["compliant"]["consumed_compliant_contacts"], result["native_material"]["positive_stiffness_contacts"]
    )
    test.assertTrue(np.isfinite(list(result["compliant"].values())).all())


def test_compliance_opt_in_selects_point_friction(test, device):
    """Select point friction, with a warning, when friction_anchor_beta is omitted under compliance."""
    for articulated in (False, True):
        with test.subTest(articulated=articulated):
            options = {"articulated": articulated, "enabled": True, "steps": 20, "friction_scale": 0.4}
            with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "point friction"):
                actual, paths, solver, _ = run_fixture(
                    device=device, **options, solver_options={"friction_anchor_beta": None}
                )
            reference, reference_paths, _, _ = run_fixture(
                device=device, **options, solver_options={"friction_anchor_beta": 0.0}
            )
            np.testing.assert_array_equal(actual, reference)
            test.assertEqual(paths, reference_paths)
            test.assertGreater(solver.compliance_contact_count, 0)
            test.assertFalse(solver._friction_anchors_enabled)
            test.assertEqual(solver.friction_anchor_beta, 0.0)


def test_omitted_friction_anchor_beta_keeps_patches_without_compliance(test, device):
    """Keep the default patch friction when friction_anchor_beta is omitted and compliance is off."""
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.05)
    model = builder.finalize(device=device)
    for kwargs in ({}, {"friction_anchor_beta": None}, {"contact_compliance": False}):
        with test.subTest(**kwargs):
            solver = SolverSteel(model, solve_mode="matrix_free", **kwargs)
            test.assertEqual(solver.friction_anchor_beta, 0.2)
            test.assertTrue(solver._friction_anchors_enabled)


def test_explicit_patch_compliance_conflict_fails_before_step(test, device):
    """Reject an explicit positive patch gain with compliance before any state can advance."""
    with patch.object(SolverSteel, "step", side_effect=AssertionError("step must not run")):
        with test.assertRaisesRegex(ValueError, "contact_compliance.*friction_anchor_beta"):
            run_fixture(
                device=device, articulated=False, enabled=True, steps=1, solver_options={"friction_anchor_beta": 0.2}
            )


def test_shape_restitution_is_rejected(test, device):
    """Reject a positive shape restitution, which has no defined composition with the compliant law."""
    with test.assertRaisesRegex(ValueError, "restitution"):
        run_fixture(device=device, articulated=True, enabled=True, steps=1, restitution=0.5)
    # Restitution keeps working without compliance.
    trace, _, _, _ = run_fixture(device=device, articulated=True, enabled=False, steps=1, restitution=0.5)
    test.assertTrue(np.isfinite(trace).all())


def test_runtime_shape_restitution_is_rejected(test, device):
    """Reject a positive restitution assigned after construction once a shape notification publishes it."""
    for replace in (False, True):
        with test.subTest(replace=replace), wp.ScopedDevice(device):
            _, _, solver, _ = run_fixture(device=device, articulated=True, enabled=True, steps=1)
            model = solver.model
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
            contacts = pipeline.contacts()
            for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
                setattr(contacts, name, wp.full(32, 3000.0, dtype=float))
            states = [model.state(), model.state()]
            newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])

            def step(states=states, pipeline=pipeline, contacts=contacts, solver=solver):
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], None, contacts, 0.005)
                states.reverse()

            restitution = np.full(model.shape_count, 0.5, dtype=np.float32)
            if replace:
                model.shape_material_restitution = wp.array(restitution, dtype=float)
            else:
                model.shape_material_restitution.assign(restitution)
            # The solver reads its own restitution buffer, which only a notification refreshes.
            step()
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
            with test.assertRaisesRegex(ValueError, "contact_compliance requires zero shape restitution"):
                step()
            model.shape_material_restitution.zero_()
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
            step()
            test.assertTrue(np.isfinite(states[0].body_q.numpy()).all())


class TestContactComplianceIntegration(unittest.TestCase):
    """Gate the experimental compliance against the dense and free-body contact rows."""


devices = get_cuda_test_devices()


for _fn in (
    test_dense_articulated_support,
    test_mf_free_body_support,
    test_default_off_exact_state,
    test_enabled_zero_stiffness_preserves_hard_contacts,
    test_friction_uses_compliant_load_and_contact_weight,
    test_native_hydro_material_pipeline,
    test_native_hydro_free_body_manifold,
    test_compliance_opt_in_selects_point_friction,
    test_omitted_friction_anchor_beta_keeps_patches_without_compliance,
    test_explicit_patch_compliance_conflict_fails_before_step,
    test_shape_restitution_is_rejected,
    test_runtime_shape_restitution_is_rejected,
):
    add_function_test(TestContactComplianceIntegration, _fn.__name__, _fn, devices=devices)


def _trajectory(device, *, dt=0.0025, iterations=16, damping=20.0, x=0.0, v=0.0, compliant=True):
    """Return an eight-second support trace for a 0.3 kg, 3000 N/m pad."""
    with wp.ScopedDevice(device):
        result = wp.zeros((round(8.0 / dt), 3), dtype=float)
        wp.launch(
            support_trajectory, 1, inputs=[0.3, 9.81, 3000.0, damping, dt, x, v, iterations, int(compliant), result]
        )
        return result.numpy()


def test_material_fallbacks_and_units(test, device):
    """Preserve weighted stiffness and apply friction weighting only once."""
    test.assertEqual(material_coefficients(100, 0, 0, shape_friction=2), (100, 0, 2))
    test.assertEqual(material_coefficients(100, 3, 0.25, shape_friction=2), (100, 3, 0.5))
    with test.assertRaises(ValueError):
        material_coefficients(0, 0, 0)


def test_unsupported_solver_modes_rejected(test, device):
    """Reject settings whose compliant update has not been wired or validated, before any allocation."""
    model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
    for key, value in (
        ("velocity_iterations", 1),
        ("contact_regularization", 0.01),
        ("contact_warmstart", True),
        ("iterations", 0),
        ("contact_shared_anchor", True),
        ("contact_friction_shared_anchor", True),
    ):
        with test.assertRaisesRegex(ValueError, key):
            SolverSteel(model, solve_mode="matrix_free", contact_compliance=True, **{key: value})


def test_static_indentation_and_baseline_failure(test, device):
    """Recover mg/k indentation that the hard-normal baseline cannot supply."""
    target = -0.3 * 9.81 / 3000.0
    compliant = _trajectory(
        device,
    )
    hard = _trajectory(device, compliant=False)
    test.assertLess(abs(compliant[-1, 0] - target), 2e-7)
    test.assertGreater(abs(hard[-1, 0] - target), 0.0009)
    test.assertLess(abs(compliant[-1, 2] - 0.3 * 9.81), 2e-4)


def test_timestep_and_iteration_convergence(test, device):
    """Keep static stiffness invariant across substeps and iteration counts."""
    target = -0.3 * 9.81 / 3000.0
    for dt in (0.005, 0.0025, 0.00125):
        for iterations in (1, 8, 16, 128):
            trace = _trajectory(device, dt=dt, iterations=iterations)
            test.assertLess(abs(trace[-1, 0] - target), 2e-7)


def test_no_attraction(test, device):
    """Return zero tensile impulse even for fast separation while indented."""
    for phi, velocity in ((0.01, 0.0), (0.0, 1.0), (-0.001, 10.0)):
        impulse, residual = solve_reference([[1 / 0.3]], [velocity], [phi], [3000.0], [20.0], dt=0.0025, device=device)
        test.assertEqual(float(impulse[0]), 0.0)
        test.assertGreaterEqual(float(residual[0]), 0.0)


def test_damping_reduces_oscillation(test, device):
    """Reduce early normal oscillation without changing static indentation."""
    undamped = _trajectory(device, damping=0, x=-0.002)
    damped = _trajectory(device, damping=20, x=-0.002)
    test.assertLess(np.sqrt(np.mean(damped[:80, 1] ** 2)), np.sqrt(np.mean(undamped[:80, 1] ** 2)))
    test.assertLess(abs(damped[-1, 0] - undamped[-1, 0]), 2e-7)


def test_no_dashpot_force_across_open_gap(test, device):
    """Avoid a damping force before the predicted trajectory closes the gap."""
    result, _ = solve_reference([[1.0]], [-0.1], [0.01], [3000.0], [10000.0], dt=0.0025, device=device)
    test.assertEqual(float(result[0]), 0.0)


def test_coupled_rows_match_independent_linear_solve(test, device):
    """Match an independent exact solve on an all-active coupled fixture."""
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    phi = [-0.002, -0.001]
    stiffness, damping = [3000.0, 5000.0], [20.0, 10.0]
    gamma, bias = np.array(
        [normal_coefficients(k, c, p, dt=0.0025) for k, c, p in zip(stiffness, damping, phi, strict=True)]
    ).T
    velocity = np.array([-0.1, -0.2])
    expected = np.linalg.solve(matrix + np.diag(gamma), -(velocity + bias))
    test.assertTrue((expected > 0).all())
    result, residual = solve_reference(matrix, velocity, phi, stiffness, damping, dt=0.0025, device=device)
    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(residual, 0, atol=1e-6)


def test_area_partition_preserves_normal_load(test, device):
    """Preserve compliance when one patch is split into equal quadrature rows."""
    for count in (1, 4, 16):
        result, residual = solve_reference(
            np.full((count, count), 1 / 0.3),
            np.full(count, -0.1),
            np.full(count, -0.001),
            np.full(count, 3000 / count),
            np.full(count, 20 / count),
            dt=0.0025,
            iterations=128,
            device=device,
        )
        gamma, bias = normal_coefficients(3000, 20, -0.001, dt=0.0025)
        expected = (0.1 - bias) / (1 / 0.3 + gamma)
        test.assertAlmostEqual(float(result.sum()), expected, places=6)
        np.testing.assert_allclose(residual, 0, atol=1e-6)


class TestContactComplianceLaw(unittest.TestCase):
    """Test the isolated compliant contact equations independently of the integration."""


law_devices = get_test_devices()


for _fn in (
    test_material_fallbacks_and_units,
    test_unsupported_solver_modes_rejected,
    test_static_indentation_and_baseline_failure,
    test_timestep_and_iteration_convergence,
    test_no_attraction,
    test_damping_reduces_oscillation,
    test_no_dashpot_force_across_open_gap,
    test_coupled_rows_match_independent_linear_solve,
    test_area_partition_preserves_normal_load,
):
    add_function_test(TestContactComplianceLaw, _fn.__name__, _fn, devices=law_devices)


def _contacts(device, *, count=0):
    """Create bounded material buffers for guard-only tests."""
    return SimpleNamespace(
        rigid_contact_count=wp.array([count], dtype=int, device=device),
        rigid_contact_max=4,
        rigid_contact_stiffness=wp.zeros(4, device=device),
        rigid_contact_damping=wp.zeros(4, device=device),
        rigid_contact_friction=wp.zeros(4, device=device),
    )


def test_input_overflow_rejected(test, device):
    """Fail explicitly for lost contact candidates."""
    scene = _CompliantScene(device, articulated=True)
    scene.collide()
    scene.contacts.rigid_contact_count.fill_(scene.contacts.rigid_contact_max + 1)
    with test.assertRaisesRegex(RuntimeError, "overflowing contact"):
        scene.solve()


def test_missing_material_arrays_rejected(test, device):
    """Require the per-contact material arrays that the compliant law reads."""
    solver = SimpleNamespace(model=SimpleNamespace(device=device))
    for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
        contacts = _contacts(device)
        setattr(contacts, name, None)
        with test.subTest(missing=name), test.assertRaisesRegex(ValueError, "hydroelastic material arrays"):
            start_step(solver, contacts, 0.005)
    with test.assertRaisesRegex(ValueError, "hydroelastic material arrays"):
        start_step(solver, None, 0.005)


def test_invalid_contact_stiffness_rejected(test, device):
    """Reject a negative or non-finite contact stiffness before writing compliant rows."""
    for stiffness in (-1.0, float("nan")):
        with test.subTest(stiffness=stiffness), test.assertRaisesRegex(ValueError, "Invalid contact stiffness"):
            run_fixture(device=device, articulated=True, enabled=True, steps=1, stiffness=stiffness)


def test_nonfinite_inputs_rejected(test, device):
    """Reject invalid time steps instead of emitting infinite row coefficients."""
    solver = SimpleNamespace(model=SimpleNamespace(device=device))
    contacts = _contacts(device)
    for dt in (0, -1, float("nan"), float("inf")):
        with test.assertRaisesRegex(ValueError, "dt"):
            start_step(solver, contacts, dt)


def test_persistent_patch_guard(test, device):
    """Reject positive persistent-anchor bias but permit a disabled dummy patch buffer."""
    solver = SimpleNamespace(
        model=SimpleNamespace(device=device),
        _friction_anchor_beta=0.0,
        _friction_patches=object(),
    )
    start_step(solver, _contacts(device), 0.005)
    solver._friction_anchor_beta = 0.2
    with test.assertRaisesRegex(ValueError, "friction_anchor_beta"):
        start_step(solver, _contacts(device), 0.005)
    solver._contact_compliance = True
    with test.assertRaisesRegex(ValueError, "friction_anchor_beta"):
        SolverSteel.step(solver, None, None, None, None, 0.005)


def test_filtered_speculative_contacts_are_skipped(test, device):
    """Honor the normal allocator's positive-gap exclusion for compliant contacts."""
    for articulated in (False, True):
        _, paths, solver, _ = run_fixture(
            device=device,
            articulated=articulated,
            enabled=True,
            steps=1,
            height=0.052,
            solver_options={"contact_gap_gate": 0.001},
        )
        test.assertEqual(paths, {-1})
        test.assertEqual(solver.compliance_contact_count, 0)
        test.assertEqual(solver.compliance_skipped_contact_count, 1)


def test_nonresponding_contact_is_skipped(test, device):
    """Ignore a kinematic-ground pair without treating it as lost capacity."""
    _, paths, solver, _ = run_fixture(device=device, articulated=True, enabled=True, steps=1, kinematic=True)
    test.assertEqual(paths, {-1})
    test.assertEqual(solver.compliance_contact_count, 0)
    test.assertEqual(solver.compliance_skipped_contact_count, 1)


def test_rolled_back_capacity_fails_without_warnings(test, device):
    """Reject an actual failed reservation even when its counter was rolled back."""
    for articulated in (False, True):
        with test.assertRaisesRegex(RuntimeError, "overflowing solver rows"):
            run_fixture(
                device=device,
                articulated=articulated,
                enabled=True,
                steps=1,
                solver_options={
                    "articulation_constraint_max": 1,
                    "free_body_constraint_max": 1,
                    "warn_constraint_overflow": False,
                },
            )


def test_solver_overflow_and_dropped_contact_rejected(test, device):
    """Reject overflowing rows and missing mappings for positive stiffness."""
    scene = _CompliantScene(device, articulated=True)
    scene.step()
    solver = scene.solver
    solver._constraint_count.fill_(solver.articulation_constraint_max + 1)
    with test.assertRaisesRegex(RuntimeError, "overflowing solver rows"):
        prepare_rows(solver, scene.contacts, 0.005)
    solver._constraint_count.fill_(1)
    solver._contact_slot.fill_(-1)
    with test.assertRaisesRegex(RuntimeError, "dropped"):
        prepare_rows(solver, scene.contacts, 0.005)


def test_reset_and_open_gap_release(test, device):
    """Clear only step-local compliance data and exert no force across an open gap."""
    _, _, solver, _ = run_fixture(device=device, articulated=True, enabled=True, steps=5)
    model = solver.model
    state = model.state()
    state.joint_q.assign(np.array([0.01], dtype=np.float32))
    state.joint_qd.assign(np.array([0.2], dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    before = {name: getattr(state, name).numpy().copy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
    solver.reset(state)
    test.assertEqual(solver.compliance_contact_count, 0)
    test.assertEqual(solver.compliance_skipped_contact_count, 0)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(state, name).numpy(), value)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
    contacts = pipeline.contacts()
    for field in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
        setattr(contacts, field, wp.zeros(32, device=device))
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
    output = model.state()
    solver.step(state, output, model.control(), contacts, 0.005)
    test.assertEqual(solver.compliance_contact_count, 0)
    test.assertAlmostEqual(float(output.joint_qd.numpy()[0]), 0.2 - 9.81 * 0.005, delta=1e-6)
    test.assertTrue(np.isfinite(output.body_q.numpy()).all())


def test_actual_timestep_and_iterations(test, device):
    """Keep native row stiffness invariant under smaller timesteps and more iterations."""
    for dt, iterations in ((0.005, 1), (0.0025, 16)):
        trace, _, _, mass = run_fixture(
            device=device, articulated=True, enabled=True, dt=dt, iterations=iterations, steps=round(1 / dt)
        )
        test.assertAlmostEqual(float(trace[-1, 2]), 0.05 - mass * 9.81 / 3000, delta=2e-6)


def test_public_defaults_remain_hard(test, device):
    """Expose an explicit default-off public constructor option."""
    parameter = inspect.signature(SolverSteel.__init__).parameters["contact_compliance"]
    test.assertIs(parameter.default, False)
    test.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)


class TestContactComplianceSafety(unittest.TestCase):
    """Reject invalid routing and preserve the physical reset and release behavior."""


for _fn in (
    test_input_overflow_rejected,
    test_nonfinite_inputs_rejected,
    test_missing_material_arrays_rejected,
    test_invalid_contact_stiffness_rejected,
    test_persistent_patch_guard,
    test_filtered_speculative_contacts_are_skipped,
    test_nonresponding_contact_is_skipped,
    test_rolled_back_capacity_fails_without_warnings,
    test_solver_overflow_and_dropped_contact_rejected,
    test_reset_and_open_gap_release,
    test_actual_timestep_and_iterations,
    test_public_defaults_remain_hard,
):
    add_function_test(TestContactComplianceSafety, _fn.__name__, _fn, devices=devices)


class _CompliantScene:
    """A compliant sphere-on-plane solver stepped from fixed buffers, so its steps can be captured."""

    def __init__(self, device, *, articulated, stiffness=3000.0, damping=5.0):
        # Zero steps: the solver is constructed but never stepped, so the first capture is cold.
        _, _, self.solver, _ = run_fixture(device=device, articulated=articulated, enabled=True, steps=0)
        self.model = model = self.solver.model
        with wp.ScopedDevice(device):
            self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
            self.contacts = self.pipeline.contacts()
            # Collision rewrites the contact material arrays, so each step copies them from these sources.
            self.stiffness = wp.full(32, stiffness, dtype=float)
            self.damping = wp.full(32, damping, dtype=float)
            self.friction = wp.full(32, 1.0, dtype=float)
            for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
                setattr(self.contacts, name, wp.zeros(32, dtype=float))
            self.states = [model.state(), model.state()]
            self.control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.states[0])
        self.solver.reset(self.states[0])

    def collide(self):
        self.pipeline.collide(self.states[0], self.contacts)
        wp.copy(self.contacts.rigid_contact_stiffness, self.stiffness)
        wp.copy(self.contacts.rigid_contact_damping, self.damping)
        wp.copy(self.contacts.rigid_contact_friction, self.friction)

    def solve(self):
        self.solver.step(self.states[0], self.states[1], self.control, self.contacts, 0.005)
        wp.copy(self.states[0].body_q, self.states[1].body_q)
        wp.copy(self.states[0].body_qd, self.states[1].body_qd)
        wp.copy(self.states[0].joint_q, self.states[1].joint_q)
        wp.copy(self.states[0].joint_qd, self.states[1].joint_qd)

    def step(self):
        self.collide()
        self.solve()

    def capture(self):
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.step()
        return capture.graph

    def pose(self):
        return np.concatenate([self.states[0].body_q.numpy().ravel(), self.states[0].body_qd.numpy().ravel()])


def test_captured_steps_match_eager_steps(test, device):
    """Replay a compliant step captured cold, before any eager step, bitwise like eager stepping."""
    for articulated in (True, False):
        eager, captured = (_CompliantScene(device, articulated=articulated) for _ in range(2))
        graph = captured.capture()
        # The capture records the step without running it.
        np.testing.assert_array_equal(captured.pose(), eager.pose())
        for step in range(40):
            eager.step()
            wp.capture_launch(graph)
            with test.subTest(articulated=articulated, step=step):
                np.testing.assert_array_equal(captured.pose(), eager.pose())
        test.assertEqual(captured.solver.compliance_contact_count, 1)
        test.assertEqual(captured.solver.compliance_contact_count, eager.solver.compliance_contact_count)
        captured.solver.check_constraint_capacity()


def test_capture_interleaves_with_eager_steps_and_reset(test, device):
    """Keep eager steps, replays and resets on one trajectory."""
    reference, mixed = (_CompliantScene(device, articulated=True) for _ in range(2))
    initial = [wp.clone(a) for a in (mixed.states[0].body_q, mixed.states[0].body_qd)]
    initial_joint = [wp.clone(a) for a in (mixed.states[0].joint_q, mixed.states[0].joint_qd)]
    trajectory = []
    for _ in range(12):
        reference.step()
        trajectory.append(reference.pose())
    for _ in range(3):
        mixed.step()
    graph = mixed.capture()
    for _ in range(6):
        wp.capture_launch(graph)
    for _ in range(3):
        mixed.step()
    np.testing.assert_array_equal(mixed.pose(), trajectory[11])
    for target, source in zip(
        (mixed.states[0].body_q, mixed.states[0].body_qd, mixed.states[0].joint_q, mixed.states[0].joint_qd),
        (*initial, *initial_joint),
        strict=True,
    ):
        wp.copy(target, source)
    mixed.solver.reset(mixed.states[0])
    for step in range(4):
        wp.capture_launch(graph)
        np.testing.assert_array_equal(mixed.pose(), trajectory[step])


def test_replay_reads_edited_contact_material(test, device):
    """Read per-contact material edits made in place between replays."""
    captured = _CompliantScene(device, articulated=True)
    eager = _CompliantScene(device, articulated=True, stiffness=1500.0)
    graph = captured.capture()
    captured.stiffness.fill_(1500.0)
    for _ in range(30):
        wp.capture_launch(graph)
        eager.step()
    np.testing.assert_array_equal(captured.pose(), eager.pose())


def test_captured_errors_are_deferred_until_reset(test, device):
    """Report invalid input from replay through check_constraint_capacity until reset, then recover."""
    scene = _CompliantScene(device, articulated=True)
    graph = scene.capture()
    wp.capture_launch(graph)
    scene.solver.check_constraint_capacity()
    scene.stiffness.fill_(float("nan"))
    wp.capture_launch(graph)
    for _ in range(2):
        with test.assertRaisesRegex(RuntimeError, "contact compliance.*Invalid contact stiffness"):
            scene.solver.check_constraint_capacity()
    scene.solver.reset(scene.states[0])
    scene.solver.check_constraint_capacity()
    scene.stiffness.fill_(3000.0)
    wp.capture_launch(graph)
    scene.solver.check_constraint_capacity()
    # A shape notification publishes the restitution that the captured step reads.
    scene.model.shape_material_restitution.fill_(0.5)
    scene.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    wp.capture_launch(graph)
    with test.assertRaisesRegex(RuntimeError, "contact compliance.*zero shape restitution"):
        scene.solver.check_constraint_capacity()
    scene.model.shape_material_restitution.zero_()
    scene.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    scene.solver.reset(scene.states[0])
    wp.capture_launch(graph)
    scene.solver.check_constraint_capacity()
    test.assertTrue(np.isfinite(scene.pose()).all())


def _run_paired_fixture(device, *, hinv_jt_kernel, friction, steps=40):
    """Drop a free box onto a 23-DOF arm (one articulation/free-body pair per world) with compliant contacts."""
    with wp.ScopedDevice(device):
        model = _build_mixed_response_model(device, dof_count=23, friction=friction)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, broad_phase="nxn", reduce_contacts=False)
        model.rigid_contact_max = 64
        contacts = pipeline.contacts()
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(64, dtype=float))
        with kernel_overrides({"hinv_jt_kernel": hinv_jt_kernel}):
            solver = SolverSteel(
                model,
                contact_compliance=True,
                friction_anchor_beta=0.0,
                solve_mode="matrix_free",
                enable_restitution=False,
                contact_friction_scale=1.0 if friction > 0.0 else 0.0,
                joint_limit_activation_gap=0.0,
                iterations=8,
                articulation_constraint_max=96,
                free_body_constraint_max=32,
            )
        state, output = model.state(), model.state()
        free = int(np.flatnonzero(solver._model_plan.is_free_rigid)[0])
        start = int(solver._model_plan.articulation_dof_start[free])
        qd = state.joint_qd.numpy()
        qd[start] = 0.5
        qd[start + 2] = -3.0
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        control = model.control()
        trace = []
        compliant = 0
        for _ in range(steps):
            state.clear_forces()
            pipeline.collide(state, contacts)
            contacts.rigid_contact_stiffness.fill_(2.0e4)
            contacts.rigid_contact_damping.fill_(10.0)
            contacts.rigid_contact_friction.fill_(1.0)
            solver.step(state, output, control, contacts, 1.0 / 240.0)
            state, output = output, state
            trace.append(state.joint_q.numpy().copy())
            compliant = max(compliant, solver.compliance_contact_count)
        return solver, np.asarray(trace), compliant


def test_paired_factor_owner_matches_general_owner(test, device):
    """Apply the compliant residual in the paired factor-coordinate owner as in the general owner."""
    # Frictional contacts take the contact-triple branch; frictionless normal-only rows take the general branch.
    for friction in (0.7, 0.0):
        with test.subTest(friction=friction):
            general_solver, general, _ = _run_paired_fixture(device, hinv_jt_kernel="par_row", friction=friction)
            paired_solver, paired, compliant = _run_paired_fixture(device, hinv_jt_kernel="auto", friction=friction)
            test.assertFalse(general_solver._paired_factor_coordinates)
            test.assertTrue(paired_solver._paired_factor_coordinates)
            test.assertGreater(compliant, 0)
            np.testing.assert_allclose(paired, general, rtol=5.0e-4, atol=1.0e-5)


def _run_local_owner_scene(device, *, compliance, steps=40):
    """Drop a free box onto a 13-DOF arm with joint limits, the scene that selects the articulation-local owners."""
    with wp.ScopedDevice(device):
        model = _build_mixed_response_model(device, dof_count=13, friction=0.0)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, broad_phase="nxn", reduce_contacts=False)
        contacts = pipeline.contacts()
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(64, dtype=float))
        with kernel_overrides({"hinv_jt_kernel": "par_row"}):
            solver = SolverSteel(
                model,
                solve_mode="matrix_free",
                contact_compliance=compliance,
                friction_anchor_beta=0.0,
                enable_joint_limits=True,
                joint_limit_activation_gap=0.0,
                iterations=8,
                articulation_constraint_max=32,
                free_body_constraint_max=32,
            )
        state, output = model.state(), model.state()
        qd = state.joint_qd.numpy()
        free = int(np.flatnonzero(solver._model_plan.is_free_rigid)[0])
        qd[int(solver._model_plan.articulation_dof_start[free]) + 2] = -3.0
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        control = model.control()
        trace = []
        for _ in range(steps):
            state.clear_forces()
            pipeline.collide(state, contacts)
            solver.step(state, output, control, contacts, 1.0 / 240.0)
            state, output = output, state
            trace.append(np.concatenate([state.joint_q.numpy(), state.joint_qd.numpy()]))
        return solver, np.asarray(trace)


def test_enabled_option_keeps_rigid_contacts_on_local_owner_scene(test, device):
    """Leave rigid contacts unchanged by the option on a scene whose rigid solve takes the local owners."""
    rigid_solver, rigid = _run_local_owner_scene(device, compliance=False)
    compliant_solver, enabled = _run_local_owner_scene(device, compliance=True)
    np.testing.assert_allclose(enabled, rigid, rtol=5.0e-4, atol=2.0e-5)
    test.assertEqual(compliant_solver.compliance_contact_count, 0)
    test.assertTrue(rigid_solver._local_internal_fast_path)
    # The local owners do not carry the compliant residual, so the option keeps the general owner.
    test.assertFalse(compliant_solver._local_internal_fast_path)


add_function_test(
    TestContactComplianceIntegration,
    test_enabled_option_keeps_rigid_contacts_on_local_owner_scene.__name__,
    test_enabled_option_keeps_rigid_contacts_on_local_owner_scene,
    devices=devices,
)


add_function_test(
    TestContactComplianceIntegration,
    test_paired_factor_owner_matches_general_owner.__name__,
    test_paired_factor_owner_matches_general_owner,
    devices=devices,
)


class TestContactComplianceCapture(unittest.TestCase):
    pass


for _fn in (
    test_captured_steps_match_eager_steps,
    test_capture_interleaves_with_eager_steps_and_reset,
    test_replay_reads_edited_contact_material,
    test_captured_errors_are_deferred_until_reset,
):
    add_function_test(TestContactComplianceCapture, _fn.__name__, _fn, devices=devices)


DT = 1.0 / 60.0


PATH_DENSE = 0  # contact routing id of the dense rows


def _contact_rows(solver, path_id: int, contact_count: int) -> dict[int, int]:
    """Map each contact of world 0 routed to ``path_id`` to its first row."""
    contact_path = solver._contact_path.numpy()
    contact_slot = solver._contact_slot.numpy()
    contact_world = solver._contact_world.numpy()
    return {
        c: int(contact_slot[c])
        for c in range(contact_count)
        if int(contact_world[c]) == 0 and int(contact_path[c]) == path_id and int(contact_slot[c]) >= 0
    }


def _stack_scene(device, g, velocity_iterations=2):
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.003
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    bodies = []
    for k in range(3):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.002 * k, 0.0, 0.0505 + 0.101 * k), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        bodies.append(b)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=128,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=8,
        velocity_iterations=velocity_iterations,
        contact_regularization=g,
        contact_warmstart=True,
    )
    return model, pipeline, solver, bodies


def _run(model, pipeline, solver, frames, dt=DT):
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    zs = []
    for _k in range(frames):
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
        zs.append(float(s0.body_q.numpy()[-1][2]))
    return s0.body_q.numpy(), zs


def _resting_box(device, g, rate, frames, articulated=False, **solver_kwargs):
    """Rest a box on the ground; ``articulated`` mounts it on a vertical prismatic joint (dense rows)."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.01
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    xform = wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity())
    if articulated:
        b = builder.add_link(xform=xform)
        joint = builder.add_joint_prismatic(-1, b, axis=wp.vec3(0.0, 0.0, 1.0), parent_xform=xform)
        builder.add_articulation([joint])
    else:
        b = builder.add_body(xform=xform)
    builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=32,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverSteel(
        model, solve_mode="matrix_free", iterations=12, contact_regularization=g, **solver_kwargs
    )
    _, zs = _run(model, pipeline, solver, frames, dt=1.0 / rate)
    return 0.05 - zs[-1]


def _sag_formula(g, rate):
    """Resting sag of a body under gravity: ``g * a * dt^2 / baumgarte`` with the default beta."""
    return g * 9.81 / (rate * rate) / 0.2


def _scissor_step(device, g, velocity_iterations):
    """Step the same-articulation scissor scene once; return the solver, contact count and joint velocity."""
    model = _build_scissor_model(device)
    solver = newton.solvers.SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=8,
        velocity_iterations=velocity_iterations,
        contact_regularization=g,
        articulation_constraint_max=64,
        free_body_constraint_max=16,
    )
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in.clear_forces()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 200.0)
    return solver, int(contacts.rigid_contact_count.numpy()[0]), state_out.joint_qd.numpy().copy()


def test_regularization_documented_sag(test: unittest.TestCase, device):
    """Sag a resting box by the regularizer's damped-compliance law.

    The sag is ``g * a * dt^2 / beta``: 6.8 mm at g = 0.5 and 60 Hz, 0.43 mm at 240 Hz.
    """
    for rate in (60, 240):
        sag = _resting_box(device, 0.5, rate, 3 * rate, contact_warmstart=True)
        expected = _sag_formula(0.5, rate)
        test.assertAlmostEqual(sag, expected, delta=0.15 * expected, msg=f"{rate} Hz: sag {sag * 1000:.2f} mm")


def test_regularization_documented_sag_on_dense_rows(test: unittest.TestCase, device):
    """Apply the same sag law to dense (articulated) contact rows.

    At rest each row satisfies ``beta * phi / dt = -g * d * lambda``. A box on a vertical
    prismatic joint has ``d = 1 / m`` per contact and its four contacts share the weight,
    ``lambda = m * a * dt / 4``, so it sags by ``g * a * dt^2 / (4 * beta)``.
    """
    for rate in (60, 240):
        sag = _resting_box(device, 0.5, rate, 3 * rate, articulated=True, contact_warmstart=True)
        expected = 0.25 * _sag_formula(0.5, rate)
        test.assertAlmostEqual(sag, expected, delta=0.15 * expected, msg=f"{rate} Hz: sag {sag * 1000:.2f} mm")
        rigid = _resting_box(device, 0.0, rate, 3 * rate, articulated=True, contact_warmstart=True)
        test.assertLess(abs(rigid), 0.1 * expected, msg=f"{rate} Hz: rigid sag {rigid * 1000:.3f} mm")


def test_regularization_velocity_pass_exempt(test: unittest.TestCase, device):
    """Solve the exact rigid law in the velocity-only pass.

    A settled stack holds its height to well under a millimetre over the last second.
    """
    model, pipeline, solver, _bodies = _stack_scene(device, g=0.05, velocity_iterations=4)
    _, zs = _run(model, pipeline, solver, 240)
    drift = abs(zs[-1] - zs[179])
    test.assertLess(drift, 5.0e-4, f"top box kept sinking ({drift * 1000:.2f} mm over the last second)")


def test_dense_contact_rows_carry_the_regularization_weight(test: unittest.TestCase, device):
    """Give penetrating articulated (dense) contact rows the weight ``1 / (1 + g)``."""
    solver, count, _ = _scissor_step(device, 0.5, 0)
    rows = _contact_rows(solver, PATH_DENSE, count)
    test.assertGreater(len(rows), 0, "scene produced no dense self-contact row")
    phi = solver._phi.numpy()[0]
    weights = solver._row_w.numpy()[0]
    for slot in rows.values():
        test.assertLess(float(phi[slot]), 0.0)
        test.assertAlmostEqual(float(weights[slot]), 2.0 / 3.0, places=6)


def test_velocity_pass_is_rigid_on_dense_rows(test: unittest.TestCase, device):
    """Ignore the regularizer on dense rows in the velocity-only pass.

    After the pass the joint velocity is the same as with ``g = 0``, since the rigid law does not depend on ``g``.
    """
    solver, count, qd_soft = _scissor_step(device, 0.5, 8)
    test.assertGreater(len(_contact_rows(solver, PATH_DENSE, count)), 0)
    _, _, qd_rigid = _scissor_step(device, 0.0, 8)
    _, _, qd_position_soft = _scissor_step(device, 0.5, 0)
    np.testing.assert_allclose(qd_soft, qd_rigid, rtol=0.0, atol=1.0e-4, err_msg="velocity pass depends on g")
    # Positive control: the regularized position solve alone does depend on g.
    test.assertGreater(float(np.max(np.abs(qd_position_soft - qd_rigid))), 1.0e-3)


def test_regularization_indeterminate_split(test: unittest.TestCase, device):
    """Select the symmetric force split for a plank on three identical supports.

    The rigid split is not unique; the regularizer must make the outer supports equal.
    """
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.003
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    supports = []
    for k in range(3):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.15 * (k - 1), 0.0, 0.0255), wp.quat_identity()))
        s = builder.add_shape_box(b, hx=0.025, hy=0.025, hz=0.025, cfg=cfg)
        supports.append(s)
    plank = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0605), wp.quat_identity()))
    builder.add_shape_box(plank, hx=0.2, hy=0.03, hz=0.01, cfg=cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=256,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=8,
        velocity_iterations=2,
        contact_regularization=0.02,
        contact_warmstart=True,
    )
    contacts = pipeline.contacts()
    observables = solver.observables({newton.solvers.SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _k in range(120):
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, DT, observables=observables)
        s0, s1 = s1, s0

    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    force = observables.contact_f.numpy()[:count, :3]
    loads = []
    for s in supports:
        fz = 0.0
        for i in range(count):
            # plank-support pairs only (exclude support-ground)
            if s in (shape0[i], shape1[i]) and 0 not in (shape0[i], shape1[i]):
                fz += abs(float(force[i][2]))
        loads.append(fz)
    test.assertGreater(min(loads), 0.0, f"a support carries no load: {loads}")
    test.assertAlmostEqual(loads[0], loads[2], delta=0.1 * max(loads), msg=f"outer supports asymmetric: {loads}")


def test_regularization_validation(test: unittest.TestCase, device):
    """Accept only finite, non-negative regularization values up to 1e6."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    for bad in (-0.1, float("nan"), float("inf"), 1.0e7):
        with test.subTest(value=bad), test.assertRaises(ValueError):
            newton.solvers.SolverSteel(model, solve_mode="matrix_free", contact_regularization=bad)
    newton.solvers.SolverSteel(model, solve_mode="matrix_free", contact_regularization=0.0)
    newton.solvers.SolverSteel(model, solve_mode="matrix_free", contact_regularization=1.0e6)


def test_zero_regularization_shares_one_weight_slot(test: unittest.TestCase, device):
    """Allocate no capacity-sized per-row weight buffers on the exact-rigid path."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free")
    test.assertFalse(solver._regularization_enabled)
    test.assertEqual(solver._row_w.shape, (1, 1))
    test.assertEqual(solver._mf_row_w.shape, (1, 1))
    regularized = newton.solvers.SolverSteel(model, solve_mode="matrix_free", contact_regularization=0.1)
    test.assertTrue(regularized._regularization_enabled)
    test.assertEqual(regularized._row_w.shape, regularized._impulses.shape)
    test.assertEqual(regularized._mf_row_w.shape, regularized._mf_impulses.shape)


def test_exact_surface_contact_is_regularized(test: unittest.TestCase, device):
    """Treat a zero-gap contact as active and only a strictly positive gap as speculative."""
    row_w = wp.zeros((1, 1), dtype=wp.float32, device=device)
    wp.launch(
        compute_world_contact_bias,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device=device),
            wp.zeros((1, 1), dtype=wp.float32, device=device),
            wp.full((1, 1), PGS_CONSTRAINT_TYPE_CONTACT, dtype=wp.int32, device=device),
            wp.zeros((1, 1), dtype=wp.float32, device=device),
            0.2,
            0.2,
            1.0,
            0.5,
            DT,
        ],
        outputs=[wp.zeros((1, 1), dtype=wp.float32, device=device), row_w],
        device=device,
    )
    test.assertEqual(float(row_w.numpy()[0, 0]), 0.5)


def test_compliance_parameters_removed(test: unittest.TestCase, device):
    """Reject the dense compliance knobs as unknown keyword arguments."""
    names = tuple(inspect.signature(newton.solvers.SolverSteel).parameters)
    test.assertNotIn("dense_contact_compliance", names)
    test.assertNotIn("speculative_dense_contact_compliance", names)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    with test.assertRaises(TypeError):
        newton.solvers.SolverSteel(model, solve_mode="matrix_free", dense_contact_compliance=1.0e-4)


def test_restitution_rows_stay_rigid(test: unittest.TestCase, device):
    """Solve a row whose rebound target fires rigidly, so the rebound is e * v_in for any regularizer.

    Without the exemption the regularized fixed point would be (e - g)/(1 + g) * v_in and vanish at g = e.
    """

    def rebound(g):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.003
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7, restitution=0.8)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, restitution=0.8))
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
        builder.add_shape_sphere(b, radius=0.05, cfg=cfg)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=32,
            broad_phase="nxn",
            deterministic=True,
            contact_matching="latest",
        )
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverSteel(
            model,
            solve_mode="matrix_free",
            iterations=16,
            velocity_iterations=0,
            contact_regularization=g,
            contact_warmstart=True,
        )
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        max_up = 0.0
        for _k in range(90):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, DT)
            s0, s1 = s1, s0
            max_up = max(max_up, float(s0.body_qd.numpy()[b][2]))
        return max_up

    v_ref = rebound(0.0)
    test.assertGreater(v_ref, 1.0, "no bounce measured")
    for g in (0.02, 0.5):
        ratio = rebound(g) / v_ref
        test.assertAlmostEqual(ratio, 1.0, delta=0.05, msg=f"g={g}: rebound ratio {ratio:.3f}")


class TestSteelRegularization(unittest.TestCase):
    pass


for _fn in (
    test_regularization_documented_sag,
    test_regularization_documented_sag_on_dense_rows,
    test_regularization_velocity_pass_exempt,
    test_dense_contact_rows_carry_the_regularization_weight,
    test_velocity_pass_is_rigid_on_dense_rows,
    test_regularization_indeterminate_split,
    test_regularization_validation,
    test_zero_regularization_shares_one_weight_slot,
    test_exact_surface_contact_is_regularized,
    test_compliance_parameters_removed,
    test_restitution_rows_stay_rigid,
):
    add_function_test(TestSteelRegularization, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main()
