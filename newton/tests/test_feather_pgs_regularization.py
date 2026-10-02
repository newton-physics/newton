# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FeatherPGS contact regularizer (``pgs_contact_regularization``)."""

import inspect
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT, compute_world_contact_bias
from newton.tests.test_feather_pgs_propagation_same_articulation import _build_scissor_model
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DT = 1.0 / 60.0
PATH_DENSE = 0  # contact routing id of the dense rows


def _contact_rows(solver, path_id: int, contact_count: int) -> dict[int, int]:
    """Map each contact of world 0 routed to ``path_id`` to its first row."""
    contact_path = solver.contact_path.numpy()
    contact_slot = solver.contact_slot.numpy()
    contact_world = solver.contact_world.numpy()
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
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_iterations=8,
        pgs_velocity_iterations=velocity_iterations,
        pgs_contact_regularization=g,
        pgs_warmstart=True,
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
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=12, pgs_contact_regularization=g, **solver_kwargs)
    _, zs = _run(model, pipeline, solver, frames, dt=1.0 / rate)
    return 0.05 - zs[-1]


def _sag_formula(g, rate):
    """Resting sag of a body under gravity: ``g * a * dt^2 / pgs_beta`` with the default beta."""
    return g * 9.81 / (rate * rate) / 0.2


def _scissor_step(device, g, velocity_iterations):
    """Step the same-articulation scissor scene once; return the solver, contact count and joint velocity."""
    model = _build_scissor_model(device)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_iterations=8,
        pgs_velocity_iterations=velocity_iterations,
        pgs_contact_regularization=g,
        dense_max_constraints=64,
        mf_max_constraints=16,
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
    """The regularizer is a numerical damped compliance: a resting box sags by
    ``g * a * dt^2 / beta`` (6.8 mm at g = 0.5 and 60 Hz, 0.43 mm at 240 Hz)."""
    for rate in (60, 240):
        sag = _resting_box(device, 0.5, rate, 3 * rate, pgs_warmstart=True)
        expected = _sag_formula(0.5, rate)
        test.assertAlmostEqual(sag, expected, delta=0.15 * expected, msg=f"{rate} Hz: sag {sag * 1000:.2f} mm")


def test_regularization_documented_sag_on_dense_rows(test: unittest.TestCase, device):
    """Dense (articulated) contact rows follow the same law.

    At rest each row satisfies ``beta * phi / dt = -g * d * lambda``. A box on a vertical
    prismatic joint has ``d = 1 / m`` per contact and its four contacts share the weight,
    ``lambda = m * a * dt / 4``, so it sags by ``g * a * dt^2 / (4 * beta)``.
    """
    for rate in (60, 240):
        sag = _resting_box(device, 0.5, rate, 3 * rate, articulated=True, pgs_warmstart=True)
        expected = 0.25 * _sag_formula(0.5, rate)
        test.assertAlmostEqual(sag, expected, delta=0.15 * expected, msg=f"{rate} Hz: sag {sag * 1000:.2f} mm")
        rigid = _resting_box(device, 0.0, rate, 3 * rate, articulated=True, pgs_warmstart=True)
        test.assertLess(abs(rigid), 0.1 * expected, msg=f"{rate} Hz: rigid sag {rigid * 1000:.3f} mm")


def test_regularization_velocity_pass_exempt(test: unittest.TestCase, device):
    """The velocity-only pass solves the exact rigid law: a settled stack holds
    its height to well under a millimetre over the last second."""
    model, pipeline, solver, _bodies = _stack_scene(device, g=0.05, velocity_iterations=4)
    _, zs = _run(model, pipeline, solver, 240)
    drift = abs(zs[-1] - zs[179])
    test.assertLess(drift, 5.0e-4, f"top box kept sinking ({drift * 1000:.2f} mm over the last second)")


def test_dense_contact_rows_carry_the_regularization_weight(test: unittest.TestCase, device):
    """Penetrating articulated (dense) contact rows get the weight ``1 / (1 + g)``."""
    solver, count, _ = _scissor_step(device, 0.5, 0)
    rows = _contact_rows(solver, PATH_DENSE, count)
    test.assertGreater(len(rows), 0, "scene produced no dense self-contact row")
    phi = solver.phi.numpy()[0]
    weights = solver.row_w.numpy()[0]
    for slot in rows.values():
        test.assertLess(float(phi[slot]), 0.0)
        test.assertAlmostEqual(float(weights[slot]), 2.0 / 3.0, places=6)


def test_velocity_pass_is_rigid_on_dense_rows(test: unittest.TestCase, device):
    """The velocity-only pass ignores the regularizer on dense rows: after it, the joint
    velocity is the same as with ``g = 0``, since the rigid law does not depend on ``g``."""
    solver, count, qd_soft = _scissor_step(device, 0.5, 8)
    test.assertGreater(len(_contact_rows(solver, PATH_DENSE, count)), 0)
    _, _, qd_rigid = _scissor_step(device, 0.0, 8)
    _, _, qd_position_soft = _scissor_step(device, 0.5, 0)
    np.testing.assert_allclose(qd_soft, qd_rigid, rtol=0.0, atol=1.0e-4, err_msg="velocity pass depends on g")
    # Positive control: the regularized position solve alone does depend on g.
    test.assertGreater(float(np.max(np.abs(qd_position_soft - qd_rigid))), 1.0e-3)


def test_regularization_indeterminate_split(test: unittest.TestCase, device):
    """A plank on three identical supports has no unique rigid force split;
    the regularizer must select the symmetric one (outer supports equal)."""
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
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_iterations=8,
        pgs_velocity_iterations=2,
        pgs_contact_regularization=0.02,
        pgs_warmstart=True,
    )
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _k in range(120):
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0
    solver.update_contacts(contacts)

    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    force = contacts.rigid_contact_force.numpy()[:count]
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
    """Parameter contract: finite non-negative values up to 1e6 only."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    for bad in (-0.1, float("nan"), float("inf"), 1.0e7):
        with test.subTest(value=bad), test.assertRaises(ValueError):
            newton.solvers.SolverFeatherPGS(model, pgs_contact_regularization=bad)
    newton.solvers.SolverFeatherPGS(model, pgs_contact_regularization=0.0)
    newton.solvers.SolverFeatherPGS(model, pgs_contact_regularization=1.0e6)


def test_zero_regularization_shares_one_weight_slot(test: unittest.TestCase, device):
    """The exact-rigid path carries no capacity-sized per-row weight buffers."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(model)
    test.assertFalse(solver._regularization_enabled)
    test.assertEqual(solver.row_w.shape, (1, 1))
    test.assertEqual(solver.mf_row_w.shape, (1, 1))
    regularized = newton.solvers.SolverFeatherPGS(model, pgs_contact_regularization=0.1)
    test.assertTrue(regularized._regularization_enabled)
    test.assertEqual(regularized.row_w.shape, regularized.impulses.shape)
    test.assertEqual(regularized.mf_row_w.shape, regularized.mf_impulses.shape)


def test_exact_surface_contact_is_regularized(test: unittest.TestCase, device):
    """A zero-gap contact is active; only a strictly positive gap is speculative."""
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
    """The no-op dense compliance knobs are gone; passing them fails loudly."""
    names = tuple(inspect.signature(newton.solvers.SolverFeatherPGS).parameters)
    test.assertNotIn("dense_contact_compliance", names)
    test.assertNotIn("speculative_dense_contact_compliance", names)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    with test.assertRaises(TypeError):
        newton.solvers.SolverFeatherPGS(model, dense_contact_compliance=1.0e-4)


def test_restitution_rows_stay_rigid(test: unittest.TestCase, device):
    """A row whose rebound target fires is solved rigid, so the rebound is
    e * v_in whatever the regularizer. Without the exemption the regularized
    fixed point would be (e - g)/(1 + g) * v_in and vanish at g = e."""

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
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_iterations=16,
            pgs_velocity_iterations=0,
            pgs_contact_regularization=g,
            pgs_warmstart=True,
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


class TestFeatherPGSRegularization(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
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
    add_function_test(TestFeatherPGSRegularization, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
