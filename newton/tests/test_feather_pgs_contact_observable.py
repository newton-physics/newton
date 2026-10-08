# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The CONTACT_F solver observable of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
import newton._src.solvers.feather_pgs.kernels
from newton._src.geometry.sdf_hydroelastic import HydroelasticSDF
from newton.solvers import SolverFeatherPGS, SolverObservableFlags
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 120.0
LOWER_MASS = 2.0
UPPER_MASS = 0.5
FILL = -7.0


def _stacked_boxes(device):
    """Two free boxes stacked on the ground: ground-box and box-box contacts."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for z, half, mass in ((0.1, 0.1, LOWER_MASS), (0.25, 0.05, UPPER_MASS)):
        inertia = wp.mat33(np.eye(3) * mass * (2.0 * half) ** 2 / 6.0)
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), mass=mass, inertia=inertia
        )
        builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    return model, pipeline


def _net_body_forces(model, contacts, contact_f):
    """Sum each body's contact forces from the per-contact rows (force on shape 0's body)."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
    body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
    linear = contact_f[:count, :3]
    net = np.zeros((model.body_count, 3))
    for body in range(model.body_count):
        net[body] = linear[body0 == body].sum(axis=0) - linear[body1 == body].sum(axis=0)
    return net


def _check_rows(test, contacts, contact_f, legacy):
    """Live rows hold the legacy linear force and no torque; every other row is zero."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(contact_f[:count, :3], legacy[:count])
    np.testing.assert_array_equal(contact_f[:count, 3:], 0.0)
    np.testing.assert_array_equal(contact_f[count:], 0.0)


def test_contact_f_matches_legacy_force_and_weight(test, device, steps=150, **options):
    """Report the legacy update_contacts force per contact, and each box's weight at rest."""
    model, pipeline = _stacked_boxes(device)
    solver = SolverFeatherPGS(model, pgs_iterations=32, **options)
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    test.assertEqual(observables.contact_f.shape[0], model.rigid_contact_max + model.soft_contact_max)
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    contacts = pipeline.contacts()
    states = [model.state(), model.state()]
    control = model.control()
    for _ in range(steps):
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        states.reverse()
    contact_f = observables.contact_f.numpy()
    solver.update_contacts(contacts)
    _check_rows(test, contacts, contact_f, contacts.rigid_contact_force.numpy())
    if options.get("mf_max_constraints") is not None:
        # Rows beyond the capacity were dropped; those contacts report zero.
        count = int(contacts.rigid_contact_count.numpy()[0])
        dropped = solver.contact_slot.numpy()[:count] < 0
        test.assertTrue(dropped.any())
        np.testing.assert_array_equal(contact_f[:count][dropped], 0.0)
    else:
        weight = np.linalg.norm(model.gravity.numpy()[0])
        net = _net_body_forces(model, contacts, contact_f)
        np.testing.assert_allclose(net[0], [0.0, 0.0, LOWER_MASS * weight], atol=0.03 * LOWER_MASS * weight)
        np.testing.assert_allclose(net[1], [0.0, 0.0, UPPER_MASS * weight], atol=0.03 * UPPER_MASS * weight)


def test_unrequested_contact_f_is_left_untouched(test, device):
    """Write nothing when the observable is allocated but not selected for the step."""
    model, pipeline = _stacked_boxes(device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free" if wp.get_device(device).is_cuda else "split")
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DT, observables=observables.select(set()))
    np.testing.assert_array_equal(observables.contact_f.numpy(), FILL)
    solver.step(state_in, state_out, model.control(), contacts, DT, observables=observables)
    test.assertFalse(np.any(observables.contact_f.numpy() == FILL))


def test_contact_f_in_a_captured_graph(test, device):
    """Write the observable from the replayed step's impulses in a captured graph."""
    model, pipeline = _stacked_boxes(device)
    solver = SolverFeatherPGS(model, pgs_iterations=32)
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    contacts = pipeline.contacts()
    states = [model.state(), model.state()]
    control = model.control()

    def step():
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        pipeline.collide(states[1], contacts)
        solver.step(states[1], states[0], control, contacts, DT, observables=observables)

    step()
    with wp.ScopedCapture(device) as recorded:
        step()
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    for _ in range(60):
        wp.capture_launch(recorded.graph)
    solver.update_contacts(contacts)
    _check_rows(test, contacts, observables.contact_f.numpy(), contacts.rigid_contact_force.numpy())
    weight = np.linalg.norm(model.gravity.numpy()[0])
    net = _net_body_forces(model, contacts, observables.contact_f.numpy())
    np.testing.assert_allclose(net[1, 2], UPPER_MASS * weight, rtol=0.03)


def test_contact_f_of_sleeping_islands(test, device, skip_constraints=True):
    """Report the legacy force of a sleeping island's contacts: zero with skipped rows, else the resting force."""
    model, pipeline = _stacked_boxes(device)
    solver = SolverFeatherPGS(
        model,
        pgs_iterations=32,
        enable_sleeping=True,
        sleep_quiet_time=0.05,
        sleep_skip_constraints=skip_constraints,
        friction_anchor_beta=0.0,
    )
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    contacts = pipeline.contacts()
    states = [model.state(), model.state()]
    control = model.control()
    for step in range(150):
        if step == 149:
            observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        states.reverse()
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    contact_f = observables.contact_f.numpy()
    solver.update_contacts(contacts)
    _check_rows(test, contacts, contact_f, contacts.rigid_contact_force.numpy())
    net = _net_body_forces(model, contacts, contact_f)
    if skip_constraints:
        np.testing.assert_array_equal(net, 0.0)
    else:
        weight = np.linalg.norm(model.gravity.numpy()[0])
        np.testing.assert_allclose(net[0, 2], LOWER_MASS * weight, rtol=0.03)
        np.testing.assert_allclose(net[1, 2], UPPER_MASS * weight, rtol=0.03)


def test_contact_f_of_compliant_hydroelastic_rows(test, device):
    """Report the legacy force of hydroelastic contacts solved with contact compliance."""
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
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.048), wp.quat_identity()))
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
        solver = SolverFeatherPGS(
            model, contact_compliance=True, friction_anchor_beta=0.0, pgs_iterations=64, mf_max_constraints=2048
        )
        observables = solver.observables({SolverObservableFlags.CONTACT_F})
        contacts = pipeline.contacts()
        states = [model.state(), model.state()]
        newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
        for _ in range(5):
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], model.control(), contacts, 0.0025, observables=observables)
            states.reverse()
        test.assertGreater(solver.compliance_contact_count, 0)
        contact_f = observables.contact_f.numpy()
        solver.update_contacts(contacts)
        _check_rows(test, contacts, contact_f, contacts.rigid_contact_force.numpy())
        test.assertGreater(np.abs(contact_f[:, 2]).sum(), 0.0)


def test_overflowing_contact_count_stays_in_the_rigid_rows(test, device, capture=False):
    """Bound rigid rows by the capacity when the contact count overflows it, leaving soft rows zero."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    # Built before the pipeline, the solver's contact scratch is larger than the contact buffer.
    pgs_mode = "matrix_free" if wp.get_device(device).is_cuda else "split"
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, warn_constraint_overflow=False)
    rigid_capacity, soft_capacity = 4, 2
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=rigid_capacity, soft_contact_max=soft_capacity)
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    test.assertEqual(observables.contact_f.shape[0], rigid_capacity + soft_capacity)
    contacts = pipeline.contacts()
    state_in, state_out, control = model.state(), model.state(), model.control()
    overflow_count = wp.array([rigid_capacity + soft_capacity + 3], dtype=wp.int32, device=device)

    def step():
        pipeline.collide(state_in, contacts)
        # The collision counter keeps counting past the capacity when the buffer overflows.
        wp.copy(contacts.rigid_contact_count, overflow_count)
        solver.step(state_in, state_out, control, contacts, DT, observables=observables)

    pipeline.collide(state_in, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), rigid_capacity)
    step()
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    if capture:
        with wp.ScopedCapture(device) as recorded:
            step()
        observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
        wp.capture_launch(recorded.graph)
    else:
        step()
    contact_f = observables.contact_f.numpy()
    solver.update_contacts(contacts)
    np.testing.assert_array_equal(contact_f[:rigid_capacity, :3], contacts.rigid_contact_force.numpy())
    np.testing.assert_array_equal(contact_f[:rigid_capacity, 3:], 0.0)
    np.testing.assert_array_equal(contact_f[rigid_capacity:], 0.0)


def test_contact_force_kernel_reads_only_rigid_rows(test, device):
    """Read no rigid contact data past the capacity when the count overflows into the soft rows."""
    rigid_capacity, soft_capacity = 1, 2
    rows = rigid_capacity + soft_capacity
    backing = []

    def rigid_view(values, dtype):
        # Valid storage past the view makes a read beyond the rigid capacity deterministic.
        full = wp.array(values, dtype=dtype, device=device)
        backing.append(full)
        return wp.array(ptr=full.ptr, shape=(rigid_capacity,), dtype=dtype, device=device)

    impulses = wp.array([[3.0]], dtype=wp.float32, device=device)
    row_count = wp.array([1], dtype=wp.int32, device=device)
    contact_f = wp.full(rows, wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL), device=device)
    wp.launch(
        newton._src.solvers.feather_pgs.kernels.compute_contact_spatial_force_from_impulses,
        dim=rows,
        inputs=[
            wp.array([rows + 4], dtype=wp.int32, device=device),
            rigid_view([[0.0, 0.0, -1.0]] * rows, wp.vec3),
            rigid_view([0] * rows, wp.int32),
            rigid_view([0] * rows, wp.int32),
            rigid_view([0] * rows, wp.int32),
            rigid_view([1] * rows, wp.int32),
            impulses,
            impulses,
            impulses,
            row_count,
            row_count,
            row_count,
            10.0,
            rigid_capacity,
        ],
        outputs=[contact_f],
        device=device,
    )
    np.testing.assert_allclose(contact_f.numpy()[0], [0.0, 0.0, 30.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(contact_f.numpy()[rigid_capacity:], 0.0)


class TestFeatherPGSContactObservable(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
all_devices = get_test_devices()

add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_matches_legacy_force_and_weight",
    test_contact_f_matches_legacy_force_and_weight,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_matches_legacy_force_and_weight_split",
    test_contact_f_matches_legacy_force_and_weight,
    devices=all_devices,
    pgs_mode="split",
)
for _suffix, _options in (
    ("propagation", {"articulated_contact_response": "propagation"}),
    ("point_friction", {"friction_anchor_beta": 0.0}),
    ("torsion", {"contact_torsion_radius": 0.01}),
    ("device_torsion", {"contact_torsion_radius": 0.01, "contact_torsion_device": True}),
    ("double_buffer", {"double_buffer": True}),
    ("row_overflow", {"mf_max_constraints": 6, "warn_constraint_overflow": False}),
):
    add_function_test(
        TestFeatherPGSContactObservable,
        f"test_contact_f_matches_legacy_force_and_weight_{_suffix}",
        test_contact_f_matches_legacy_force_and_weight,
        devices=cuda_devices,
        **_options,
    )
add_function_test(
    TestFeatherPGSContactObservable,
    "test_unrequested_contact_f_is_left_untouched",
    test_unrequested_contact_f_is_left_untouched,
    devices=all_devices,
)
add_function_test(
    TestFeatherPGSContactObservable,
    "test_overflowing_contact_count_stays_in_the_rigid_rows",
    test_overflowing_contact_count_stays_in_the_rigid_rows,
    devices=all_devices,
)
add_function_test(
    TestFeatherPGSContactObservable,
    "test_overflowing_contact_count_stays_in_the_rigid_rows_captured",
    test_overflowing_contact_count_stays_in_the_rigid_rows,
    devices=cuda_devices,
    capture=True,
)
add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_force_kernel_reads_only_rigid_rows",
    test_contact_force_kernel_reads_only_rigid_rows,
    devices=all_devices,
)
add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_in_a_captured_graph",
    test_contact_f_in_a_captured_graph,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_of_sleeping_islands",
    test_contact_f_of_sleeping_islands,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_of_sleeping_islands_without_skipped_rows",
    test_contact_f_of_sleeping_islands,
    devices=cuda_devices,
    skip_constraints=False,
)
add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_of_compliant_hydroelastic_rows",
    test_contact_f_of_compliant_hydroelastic_rows,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main()
