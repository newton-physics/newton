# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for routed tendons in the VBD rigid solver."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.tests.test_tendon_capstan import (
    _moving_rolling_route_material_error,
    build_dynamic_pulley_atwood,
    build_force_driven_dynamic_route,
    build_kinematic_pulley_atwood,
    build_kinematic_rolling_transport,
    build_motorized_pulley_drive,
    build_pinhole_atwood,
    build_simple_cable_gravity,
    build_slack_pinhole_route,
)
from newton.tests.test_tendon_equilibrium import build_atwood_equal_weights
from newton.tests.unittest_utils import sanitize_identifier

TENDON_VBD_SOLVER_KWARGS = {
    "iterations": 20,
    "tendon_max_sweeps": 32,
    "tendon_settle_tol": 1.0e-3,
    "rigid_compliant_alm": False,
    "rigid_avbd_beta": 1.0e6,
    "rigid_joint_linear_k_start": 1.0e7,
    "rigid_joint_angular_k_start": 1.0e5,
    "rigid_joint_linear_kd": 5.0e-2,
    "rigid_joint_angular_kd": 2.0e-2,
}


def add_test(cls, name, devices, test_fn):
    for device in devices:
        test_name = f"test_{sanitize_identifier(name)}_{sanitize_identifier(device)}"

        def test_method(self, d=device, fn=test_fn):
            return fn(self, d)

        expected_failure_devices = getattr(test_fn, "__unittest_expected_failure_devices__", set())
        if getattr(test_fn, "__unittest_expecting_failure__", False) or device in expected_failure_devices:
            test_method = unittest.expectedFailure(test_method)
        setattr(cls, test_name, test_method)


def _set_serial_body_coloring(model):
    model.body_color_groups = [
        wp.array([body], dtype=wp.int32, device=model.device) for body in range(model.body_count)
    ]


def _hinge_y_angle(body_q, body_idx):
    q = body_q[body_idx]
    return float(2.0 * np.arctan2(float(q[4]), float(q[6])))


def _hinge_z_angle(body_q, body_idx):
    q = body_q[body_idx]
    return float(2.0 * np.arctan2(float(q[5]), float(q[6])))


def _make_tendon_vbd_solver(model):
    return newton.solvers.SolverVBD(model, **TENDON_VBD_SOLVER_KWARGS)


def build_single_span_tendon():
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))

    anchor = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)),
        mass=0.0,
        is_kinematic=True,
    )
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
        mass=1.0,
    )
    builder.add_shape_sphere(anchor, radius=0.01)
    builder.add_shape_sphere(body, radius=0.01)

    builder.add_tendon()
    builder.add_tendon_link(
        body=anchor,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
    )
    builder.add_tendon_link(
        body=body,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        compliance=1.0e-5,
        damping=0.0,
        rest_length=0.5,
    )

    builder.color()
    model = builder.finalize()
    return model, body


def build_fixed_rolling_chain(compliance=1.0e-5):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    positions = [(-1.0, 0.0, 0.0), (0.0, 0.25, 0.0), (1.0, -0.25, 0.0), (2.0, 0.25, 0.0)]
    bodies = [
        builder.add_body(mass=0.0, is_kinematic=True),
        builder.add_body(mass=0.0, is_kinematic=True),
        builder.add_body(mass=0.0, is_kinematic=True),
        builder.add_body(
            xform=wp.transform(p=wp.vec3(*positions[-1])),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        ),
    ]

    builder.add_tendon()
    builder.add_tendon_link(
        body=bodies[0],
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=positions[0],
    )
    builder.add_tendon_link(
        body=bodies[1],
        link_type=int(TendonLinkType.ROLLING),
        radius=0.1,
        orientation=-1,
        offset=positions[1],
        compliance=compliance,
    )
    builder.add_tendon_link(
        body=bodies[2],
        link_type=int(TendonLinkType.ROLLING),
        radius=0.1,
        orientation=1,
        offset=positions[2],
        compliance=compliance,
    )
    builder.add_tendon_link(
        body=bodies[3],
        link_type=int(TendonLinkType.ATTACHMENT),
        compliance=compliance,
    )
    builder.color()
    return builder.finalize(), bodies[-1]


def build_dynamic_rolling_route():
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
    lower = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5)),
        mass=0.0,
        is_kinematic=True,
    )
    upper = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5)),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    candidate = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.25, 0.0, 0.0)),
        mass=0.0,
        is_kinematic=True,
    )

    builder.add_tendon()
    builder.add_tendon_link(body=lower, link_type=int(TendonLinkType.ATTACHMENT), axis=(0.0, 1.0, 0.0))
    candidate_link = builder.add_tendon_link(
        body=candidate,
        link_type=int(TendonLinkType.ROLLING),
        radius=0.1,
        orientation=1,
        dynamic=True,
        axis=(0.0, 1.0, 0.0),
        compliance=1.0e-4,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=upper,
        link_type=int(TendonLinkType.ATTACHMENT),
        axis=(0.0, 1.0, 0.0),
        compliance=1.0e-4,
        rest_length=-1.0,
    )
    builder.color()
    return builder.finalize(), candidate, candidate_link, lower, upper


def run_vbd_model(model, num_frames=80, substeps=12, fps=60):
    _set_serial_body_coloring(model)
    dt = 1.0 / fps / substeps
    solver = _make_tendon_vbd_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    return state_0, solver


def run_vbd_motorized_model(model, drive_joint, target=2.0, num_frames=70, substeps=10, fps=60):
    _set_serial_body_coloring(model)
    dt = 1.0 / fps / substeps
    solver = _make_tendon_vbd_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    q_start = int(model.joint_q_start.numpy()[drive_joint])
    for _ in range(num_frames):
        control.joint_target_q[q_start : q_start + 1].fill_(target)
        for _ in range(substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    return state_0, solver


def _run_dynamic_pulley(mu, device, num_frames=40):
    model, left_idx, right_idx, pulley_idx = build_dynamic_pulley_atwood(mu=mu)
    _set_serial_body_coloring(model)

    solver = _make_tendon_vbd_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    substeps = 12
    dt = 1.0 / 60.0 / substeps
    last_angle = _hinge_y_angle(state_0.body_q.numpy(), pulley_idx)
    theta = 0.0

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

        angle = _hinge_y_angle(state_0.body_q.numpy(), pulley_idx)
        theta += float((angle - last_angle + np.pi) % (2.0 * np.pi) - np.pi)
        last_angle = angle

    body_q = state_0.body_q.numpy()
    return body_q, left_idx, right_idx, theta


def _dynamic_capstan_metrics(device, mu, num_frames=40):
    model, left_idx, right_idx, pulley_idx = build_dynamic_pulley_atwood(mu=mu)
    _set_serial_body_coloring(model)
    substeps = 12
    dt = 1.0 / 60.0 / substeps
    solver = _make_tendon_vbd_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    body_q = state_0.body_q.numpy()
    last_angle = _hinge_y_angle(body_q, pulley_idx)
    theta = 0.0

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        angle = _hinge_y_angle(body_q, pulley_idx)
        theta += float((angle - last_angle + np.pi) % (2.0 * np.pi) - np.pi)
        last_angle = angle

    left_travel = float(body_q[left_idx][2]) - 2.0
    right_travel = 2.0 - float(body_q[right_idx][2])
    cable_travel = 0.5 * (left_travel + right_travel)
    radius = float(model.tendon_link_radius.numpy()[1])
    rim_travel = theta * radius
    slip = abs(cable_travel - rim_travel)
    return body_q, left_travel, right_travel, cable_travel, theta, rim_travel, slip


def _kinematic_capstan_metrics(device, mu, num_frames=40):
    model, left_idx, right_idx, pulley_idx = build_kinematic_pulley_atwood(mu=mu)
    state, _ = run_vbd_model(model, num_frames=num_frames)
    body_q = state.body_q.numpy()
    left_travel = float(body_q[left_idx][2]) - 2.0
    right_travel = 2.0 - float(body_q[right_idx][2])
    theta = _hinge_y_angle(body_q, pulley_idx)
    return body_q, left_travel, right_travel, theta


class TestTendonVBD(unittest.TestCase):
    pass


def test_vbd_tendon_stretch_pulls_toward_anchor(test, device):
    """A taut VBD tendon segment should pull the dynamic endpoint toward rest length."""
    with wp.ScopedDevice(device):
        model, body_idx = build_single_span_tendon()
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        initial_x = float(state_0.body_q.numpy()[body_idx][0])
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        final_x = float(state_1.body_q.numpy()[body_idx][0])

        test.assertLess(final_x, initial_x - 0.1, f"Tendon should pull endpoint inward: {initial_x} -> {final_x}")
        test.assertTrue(np.isfinite(state_1.body_q.numpy()).all(), "Non-finite VBD single-span state")


def test_vbd_tendon_zero_compliance_uses_floor(test, device):
    """VBD should approximate zero tendon compliance with its stiffness floor."""
    with wp.ScopedDevice(device):
        results = []
        for compliance in (0.0, 1.0e-8):
            model, body_idx = build_single_span_tendon()
            model.tendon_seg_compliance.fill_(compliance)
            solver = _make_tendon_vbd_solver(model)
            state_0 = model.state()
            state_1 = model.state()
            solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
            results.append(
                (
                    state_1.body_q.numpy()[body_idx].copy(),
                    float(solver.tendon_seg_active_compliance.numpy()[0]),
                )
            )

        test.assertTrue(np.isfinite(results[0][0]).all())
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=0.0, atol=1.0e-6)
        test.assertEqual(results[0][1], results[1][1])


def test_vbd_rolling_chain_tension_matches_accepted_pose(test, device):
    """VBD material relaxation should converge at the final accepted rigid pose."""
    with wp.ScopedDevice(device):
        model, slider = build_fixed_rolling_chain()
        solver_kwargs = dict(TENDON_VBD_SOLVER_KWARGS)
        solver_kwargs.update(tendon_max_sweeps=256, tendon_settle_tol=1.0e-5)
        solver = newton.solvers.SolverVBD(model, **solver_kwargs)
        state_0 = model.state()
        state_1 = model.state()

        direction = solver.tendon_seg_attachment_r.numpy()[-1] - solver.tendon_seg_attachment_l.numpy()[-1]
        direction /= np.linalg.norm(direction)
        body_f = np.zeros((model.body_count, 6), dtype=np.float32)
        body_f[slider, :3] = 10.0 * direction
        state_0.body_f.assign(wp.array(body_f, dtype=wp.spatial_vector))
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)

        lengths = np.linalg.norm(
            solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1
        )
        tension = np.maximum(lengths - solver.tendon_seg_rest_length.numpy(), 0.0) / np.maximum(
            solver.tendon_seg_active_compliance.numpy(), 1.0e-30
        )
        test.assertGreater(np.min(tension), 1.0, f"Every rolling-chain span should carry load: {tension}")
        test.assertLess(np.ptp(tension), 5.0e-3, f"Frictionless rolling-chain tensions did not converge: {tension}")
        test.assertGreater(
            int(solver.tendon_cone_sweep_count.numpy()[0]),
            4,
            "The regression must exercise adaptive material relaxation beyond the legacy VBD sweep count",
        )


def test_vbd_dynamic_tendon_route_switches_inside_solver(test, device):
    """VBD should update dynamic routing and its active segment endpoints inside ``step``."""
    with wp.ScopedDevice(device):
        model, candidate, candidate_link, lower, upper = build_dynamic_rolling_route()
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        rest_length = solver.tendon_seg_rest_length.numpy()
        rest_length[0] -= 0.02
        solver.tendon_seg_rest_length.assign(rest_length)

        body_colors = np.full(model.body_count, -1, dtype=np.int32)
        for color, group in enumerate(model.body_color_groups):
            body_colors[group.numpy()] = color
        test.assertNotEqual(body_colors[lower], body_colors[upper], "The possible bypass span must be colored")

        body_q = state_0.body_q.numpy()
        active_history = []
        segment_history = []
        material_tension_history = []
        first_upper_z = None
        for x in (0.25, 0.05, -0.25, 0.25):
            body_q[candidate, :3] = (x, 0.0, 0.0)
            state_0.body_q.assign(body_q)
            solver.step(state_0, state_1, control, None, 1.0 / 120.0)
            active_history.append(bool(solver.tendon_link_active.numpy()[candidate_link]))
            segment_history.append(solver.tendon_seg_active.numpy().tolist())
            material_tension_history.append(solver.tendon_seg_material_tension.numpy().tolist())
            if first_upper_z is None:
                first_upper_z = float(state_1.body_q.numpy()[upper][2])
            state_0, state_1 = state_1, state_0
            body_q = state_0.body_q.numpy()

        test.assertEqual(active_history, [False, True, True, False])
        test.assertEqual(segment_history, [[1, 0], [1, 1], [1, 1], [1, 0]])
        test.assertEqual(material_tension_history[0][1], 0.0)
        test.assertEqual(material_tension_history[-1][1], 0.0)
        test.assertLess(first_upper_z, 0.499, "The inactive bypass span must load its new right endpoint")
        test.assertTrue(np.isfinite(state_0.body_q.numpy()).all())


def test_vbd_dynamic_route_activation_uses_accepted_pose(test, device):
    """Route activation should not use the VBD inertial predictor pose."""
    with wp.ScopedDevice(device):
        model, candidate, candidate_link = build_force_driven_dynamic_route(device)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        test.assertFalse(solver.tendon_link_active.numpy()[candidate_link])
        body_qd = state_0.body_qd.numpy()
        body_qd[candidate, 0] = -24.0
        state_0.body_qd.assign(body_qd)

        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)

        test.assertLess(abs(float(state_1.body_q.numpy()[candidate, 0])), 0.1)
        test.assertFalse(
            solver.tendon_link_active.numpy()[candidate_link],
            "The predictor crossed the cable, but activation belongs to the accepted step-start pose",
        )


def test_vbd_tendon_coloring_separates_segment_endpoints(test, device):
    """Rigid-body coloring should include tendon force-element connectivity."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body_0 = builder.add_body(mass=1.0)
        body_1 = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body_0, radius=0.01)
        builder.add_shape_sphere(body_1, radius=0.01)
        builder.add_tendon()
        builder.add_tendon_link(body=body_0, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body_1,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-3,
        )
        builder.color()
        model = builder.finalize()

        body_colors = np.full(model.body_count, -1, dtype=int)
        for color, group in enumerate(model.body_color_groups):
            body_colors[group.numpy()] = color
        test.assertNotEqual(body_colors[body_0], body_colors[body_1])
        _make_tendon_vbd_solver(model)

        model.body_color_groups = [wp.array([body_0, body_1], dtype=wp.int32, device=model.device)]
        with test.assertRaisesRegex(ValueError, "does not separate tendon segment endpoints"):
            _make_tendon_vbd_solver(model)


def test_vbd_tendon_compliance_uses_physical_tension(test, device):
    """A compliant VBD tendon should settle to the same physical equilibrium at different time steps."""
    with wp.ScopedDevice(device):
        results = []
        for dt in (1.0 / 60.0, 1.0 / 120.0):
            model, body_idx, mass, compliance, initial_z = build_simple_cable_gravity()
            model.tendon_seg_damping.fill_(100.0)
            _set_serial_body_coloring(model)
            solver = _make_tendon_vbd_solver(model)
            state_0 = model.state()
            state_1 = model.state()
            control = model.control()
            collision_pipeline = newton.CollisionPipeline(model)
            contacts = collision_pipeline.contacts()

            load = mass * 9.81
            expected_z = initial_z - load * compliance
            for _ in range(round(1.5 / dt)):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            attachment_l = solver.tendon_seg_attachment_l.numpy()[0]
            attachment_r = solver.tendon_seg_attachment_r.numpy()[0]
            rest_length = float(solver.tendon_seg_rest_length.numpy()[0])
            stretch = float(np.linalg.norm(attachment_r - attachment_l)) - rest_length
            material_tension = stretch / compliance
            final_z = float(state_0.body_q.numpy()[body_idx][2])
            final_vz = float(state_0.body_qd.numpy()[body_idx][2])
            results.append((material_tension, final_z, final_vz))

            test.assertAlmostEqual(
                material_tension,
                load,
                delta=0.05 * load,
                msg=f"VBD material tension should balance the load: {material_tension:.3f} vs {load:.3f}",
            )
            test.assertAlmostEqual(
                final_z,
                expected_z,
                delta=5.0e-3,
                msg=f"VBD equilibrium position changed: {final_z:.6f} vs {expected_z:.6f}",
            )
            test.assertAlmostEqual(final_vz, 0.0, delta=2.0e-2, msg=f"VBD equilibrium velocity changed: {final_vz:.6f}")

        test.assertAlmostEqual(
            results[0][0],
            results[1][0],
            delta=0.05 * mass * 9.81,
            msg=f"VBD material tension should be time-step independent: {results}",
        )


def test_vbd_slack_damped_tendon_opposes_extension(test, device):
    """Positive length rate may activate unilateral damping before positive stretch."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        anchor = builder.add_body(mass=0.0, is_kinematic=True)
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.25, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_tendon()
        builder.add_tendon_link(body=anchor, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-3,
            damping=100.0,
            rest_length=1.0,
        )
        builder.color()
        model = builder.finalize()
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        state_0.body_qd.assign(
            [
                wp.spatial_vector(0.0),
                wp.spatial_vector(30.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ]
        )

        dt = 1.0 / 120.0
        solver.step(state_0, state_1, model.control(), None, dt)

        final_vx = float(state_1.body_qd.numpy()[body][0])
        material_tension = float(solver.tendon_seg_material_tension.numpy()[0])
        reported_lambda = float(solver.tendon_seg_lambda.numpy()[0])
        test.assertLess(final_vx, 25.0)
        test.assertEqual(material_tension, 0.0)
        test.assertEqual(reported_lambda, 0.0, "VBD must not report a synthetic XPBD multiplier")


def test_vbd_reset_restores_tendon_state(test, device):
    """VBD reset should restore persistent tendon material and routing state."""
    with wp.ScopedDevice(device):
        model, _ = build_kinematic_rolling_transport(mu=10.0)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()
        initial_local_l = solver.tendon_seg_attachment_l_local.numpy().copy()
        initial_active = solver.tendon_link_active.numpy().copy()
        initial_total = solver.tendon_total_cable.numpy().copy()

        solver.tendon_seg_rest_length.fill_(0.123)
        solver.tendon_seg_attachment_l_local.zero_()
        solver.tendon_link_active.zero_()
        solver.tendon_total_cable.zero_()

        world_mask = wp.array([False] * model.world_count + [True], dtype=wp.bool, device=device)
        solver.reset(model.state(), world_mask=world_mask, flags=0)

        np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy(), initial_rest, atol=1.0e-7)
        np.testing.assert_allclose(solver.tendon_seg_attachment_l_local.numpy(), initial_local_l, atol=1.0e-7)
        np.testing.assert_array_equal(solver.tendon_link_active.numpy(), initial_active)
        np.testing.assert_allclose(solver.tendon_total_cable.numpy(), initial_total, atol=1.0e-7)


def test_vbd_reset_rebaselines_tendon_at_custom_pose(test, device):
    """A custom reset pose should not be interpreted as new roller travel."""
    with wp.ScopedDevice(device):
        model, pulley = build_kinematic_rolling_transport(mu=10.0)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()

        body_q = state_0.body_q.numpy()
        angle = 0.4
        body_q[pulley, 3:] = np.array([0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle)], dtype=np.float32)
        state_0.body_q.assign(body_q)
        solver.reset(state_0, flags=0)
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)

        np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy(), initial_rest, atol=1.0e-7)


def test_vbd_tendon_diagnostics_match_final_pose(test, device):
    """Reported tendon geometry and damping should use the completed VBD pose."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        body_l = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.5, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        body_r = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.5, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        compliance = 1.0e-5
        damping = 1.0
        initial_length = 1.0
        rest_length = 0.5
        builder.add_tendon()
        builder.add_tendon_link(body=body_l, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body_r,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=compliance,
            damping=damping,
            rest_length=rest_length,
        )
        builder.color()
        model = builder.finalize()
        solver = newton.solvers.SolverVBD(
            model,
            iterations=1,
            rigid_compliant_alm=False,
            rigid_avbd_beta=1.0e6,
        )
        state_0 = model.state()
        state_1 = model.state()

        dt = 1.0 / 60.0
        solver.step(state_0, state_1, model.control(), None, dt)

        body_q = state_1.body_q.numpy()
        final_length = float(np.linalg.norm(body_q[body_r][:3] - body_q[body_l][:3]))
        attachment_l = solver.tendon_seg_attachment_l.numpy()[0]
        attachment_r = solver.tendon_seg_attachment_r.numpy()[0]
        attachment_length = float(np.linalg.norm(attachment_r - attachment_l))
        length_rate = (final_length - initial_length) / dt
        material_tension = max((final_length - rest_length) / compliance, 0.0)
        total_tension = max(material_tension + damping * length_rate, 0.0)
        reported_material_tension = float(solver.tendon_seg_material_tension.numpy()[0])
        reported_lambda = float(solver.tendon_seg_lambda.numpy()[0])
        routing_damping_tension = float(solver.tendon_seg_damping_tension.numpy()[0])
        test.assertLess(final_length, 0.75, "The solve must move the endpoints enough to expose stale diagnostics")
        test.assertGreater(total_tension, 0.0, "The regression must exercise a loaded tendon")
        test.assertAlmostEqual(attachment_length, final_length, delta=1.0e-6)
        test.assertAlmostEqual(reported_material_tension, material_tension, delta=1.0e-3)
        test.assertAlmostEqual(
            routing_damping_tension,
            damping * length_rate,
            delta=max(1.0e-3, 1.0e-4 * abs(damping * length_rate)),
            msg="VBD material routing must use the same pose-difference damping rate as its force evaluation",
        )
        test.assertEqual(reported_lambda, 0.0, "VBD must not report a synthetic XPBD multiplier")


def test_vbd_nonrolling_rotation_uses_discrete_length_rate(test, device):
    """Non-rolling damping should use the discrete free-span length change."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        rotating = builder.add_body(mass=0.0, is_kinematic=True)
        fixed = builder.add_body(
            xform=wp.transform(p=wp.vec3(2.0, 1.0, 0.0)),
            mass=0.0,
            is_kinematic=True,
        )
        damping = 10.0
        builder.add_tendon()
        builder.add_tendon_link(
            body=rotating,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(1.0, 0.0, 0.0),
        )
        builder.add_tendon_link(
            body=fixed,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-3,
            damping=damping,
            rest_length=-1.0,
        )
        builder.color()
        model = builder.finalize()
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()

        dt = 1.0 / 60.0
        # The first VBD step establishes pose history from the supplied state.
        solver.step(state_0, state_1, model.control(), None, dt)

        body_q = state_1.body_q.numpy()
        angle = 0.5 * np.pi
        body_q[rotating, 3:] = np.array([0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle)])
        state_1.body_q.assign(body_q)

        solver.step(state_1, state_0, model.control(), None, dt)

        initial_length = np.sqrt(2.0)
        final_length = 2.0
        expected_damping_tension = damping * (final_length - initial_length) / dt
        test.assertAlmostEqual(
            float(solver.tendon_seg_damping_tension.numpy()[0]),
            expected_damping_tension,
            delta=1.0e-3,
        )


def test_vbd_rolling_spin_does_not_add_damping_tension(test, device):
    """Pure roller-axis spin should transport material without stretching the free spans."""
    with wp.ScopedDevice(device):
        model, pulley = build_kinematic_rolling_transport(mu=10.0)
        model.tendon_seg_damping.fill_(100.0)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()

        body_q = state_0.body_q.numpy()
        angle = 0.4
        body_q[pulley, 3:] = np.array([0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle)], dtype=np.float32)
        state_0.body_q.assign(body_q)

        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)

        rest_delta = solver.tendon_seg_rest_length.numpy() - initial_rest
        damping_tension = solver.tendon_seg_damping_tension.numpy()
        test.assertGreater(
            float(np.max(np.abs(rest_delta))),
            1.0e-4,
            f"Prescribed rolling spin should still transport material: delta={rest_delta}",
        )
        np.testing.assert_allclose(
            damping_tension,
            0.0,
            atol=1.0e-4,
            err_msg="Roller-axis spin must not change free-span length or create damping tension",
        )


def test_vbd_frictionless_off_center_roller_retains_body_torque(test, device):
    """A frictionless roller still transmits force through its off-center axis."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        anchor_l = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.5, 1.0, 0.0)),
            mass=0.0,
            is_kinematic=True,
        )
        roller = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            lock_inertia=True,
        )
        anchor_r = builder.add_body(
            xform=wp.transform(p=wp.vec3(1.5, 1.0, 0.0)),
            mass=0.0,
            is_kinematic=True,
        )

        builder.add_tendon()
        builder.add_tendon_link(body=anchor_l, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=roller,
            link_type=int(TendonLinkType.ROLLING),
            radius=0.2,
            orientation=1,
            axis=(0.0, 0.0, 1.0),
            offset=(0.5, 0.0, 0.0),
            mu=0.0,
            compliance=1.0e-2,
            rest_length=1.3,
        )
        builder.add_tendon_link(
            body=anchor_r,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=1.0e-2,
            rest_length=1.3,
        )
        builder.color()
        model = builder.finalize()
        _set_serial_body_coloring(model)

        solver_kwargs = dict(TENDON_VBD_SOLVER_KWARGS)
        solver_kwargs["iterations"] = 1
        solver = newton.solvers.SolverVBD(model, **solver_kwargs)
        state_0 = model.state()
        state_1 = model.state()
        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)

        tension = solver.tendon_seg_material_tension.numpy()
        angle = _hinge_z_angle(state_1.body_q.numpy(), roller)
        np.testing.assert_allclose(tension[0], tension[1], rtol=1.0e-4, atol=1.0e-4)
        test.assertGreater(
            angle,
            4.0e-4,
            "The resultant tendon force at an off-center roller must retain its moment about the body COM",
        )


def test_vbd_same_body_rolling_span_applies_net_torque(test, device):
    """A rolling span with both endpoints on one body should retain its net torque."""
    with wp.ScopedDevice(device):
        initial_angle = 0.23
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        anchor = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.8, -0.5, 0.0)),
            mass=0.0,
            is_kinematic=True,
        )
        body = builder.add_body(
            xform=wp.transform(q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), initial_angle)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            lock_inertia=True,
        )

        builder.add_tendon()
        builder.add_tendon_link(body=anchor, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ROLLING),
            radius=0.12,
            orientation=-1,
            mu=0.0,
            offset=(0.1, 0.15, 0.0),
            compliance=1.0e-3,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.5, -0.3, 0.0),
            compliance=1.0e-3,
            rest_length=-1.0,
        )
        builder.color()
        model = builder.finalize()
        _set_serial_body_coloring(model)

        solver_kwargs = dict(TENDON_VBD_SOLVER_KWARGS)
        solver_kwargs.update(iterations=1, tendon_max_sweeps=1)
        solver = newton.solvers.SolverVBD(model, **solver_kwargs)
        span_lengths = np.linalg.norm(
            solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1
        )
        rest_lengths = span_lengths.copy()
        rest_lengths[0] += 1.0
        rest_lengths[1] -= 0.01
        solver.tendon_seg_rest_length.assign(rest_lengths)

        state_0, state_1 = model.state(), model.state()
        initial_pose = state_0.body_q.numpy()[body].copy()
        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)

        final_pose = state_1.body_q.numpy()[body]
        np.testing.assert_allclose(final_pose[:3], initial_pose[:3], rtol=0.0, atol=1.0e-7)
        test.assertGreater(_hinge_z_angle(state_1.body_q.numpy(), body), initial_angle + 1.0e-5)
        test.assertGreater(float(solver.tendon_seg_material_tension.numpy()[1]), 1.0)


def test_vbd_tendon_cuda_graph_capture(test, device):
    """Native VBD tendon force accumulation should remain CUDA graph-capturable."""
    device = wp.get_device(device)
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    with wp.ScopedDevice(device):
        model, body_idx, _, _, initial_z = build_simple_cable_gravity()
        model.tendon_seg_damping.fill_(100.0)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        dt = 1.0 / 120.0

        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_0, state_1, control, None, dt)
            solver.step(state_1, state_0, control, None, dt)

        for _ in range(10):
            wp.capture_launch(capture.graph)

        body_q = state_0.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Captured VBD tendon steps produced non-finite poses")
        test.assertLess(float(body_q[body_idx][2]), initial_z, "Captured VBD tendon should respond to gravity")


def test_vbd_dynamic_tendon_cuda_graph_capture(test, device):
    """VBD dynamic active-set updates should remain CUDA graph-capturable."""
    device = wp.get_device(device)
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires a CUDA device")
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    with wp.ScopedDevice(device):
        warm_model, _, _, _, _ = build_dynamic_rolling_route()
        warm_solver = _make_tendon_vbd_solver(warm_model)
        warm_0 = warm_model.state()
        warm_1 = warm_model.state()
        warm_solver.step(warm_0, warm_1, warm_model.control(), None, 1.0 / 120.0)

        model, candidate, candidate_link, _, _ = build_dynamic_rolling_route()
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        for state in (state_0, state_1):
            body_q = state.body_q.numpy()
            body_q[candidate, :3] = (0.05, 0.0, 0.0)
            state.body_q.assign(body_q)

        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_0, state_1, control, None, 1.0 / 120.0)
            solver.step(state_1, state_0, control, None, 1.0 / 120.0)

        wp.capture_launch(capture.graph)
        test.assertTrue(solver.tendon_link_active.numpy()[candidate_link])


def test_vbd_pinhole_slip_atwood(test, device):
    """A pinhole is a frictionless slip waypoint: heavy descends, light rises."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx = build_pinhole_atwood()
        state, _ = run_vbd_model(model, num_frames=120)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD pinhole Atwood state")

        left_z = float(body_q[left_idx][2])
        right_z = float(body_q[right_idx][2])
        test.assertGreater(left_z, 2.05, f"Light side should rise through VBD pinhole slip: z={left_z:.4f}")
        test.assertLess(right_z, 1.95, f"Heavy side should descend through VBD pinhole slip: z={right_z:.4f}")


def test_vbd_slack_pinhole_does_not_redistribute(test, device):
    """Pinholes should transfer only taut excess, not proportionally repartition slack."""
    with wp.ScopedDevice(device):
        model = build_slack_pinhole_route()
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        initial = solver.tendon_seg_rest_length.numpy().copy()
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        final = solver.tendon_seg_rest_length.numpy()

        np.testing.assert_allclose(
            final,
            initial,
            rtol=0.0,
            atol=1.0e-6,
            err_msg=f"VBD slack pinhole rest lengths should not be repartitioned: {initial} -> {final}",
        )


def test_vbd_dynamic_pulley_uses_angular_jacobian(test, device):
    """High friction recovers the no-slip angular Jacobian baseline."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx, pulley_idx = build_dynamic_pulley_atwood(mu=10.0)
        state, _ = run_vbd_model(model, num_frames=60)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD dynamic pulley Atwood state")

        left_z = float(body_q[left_idx][2])
        right_z = float(body_q[right_idx][2])
        theta = abs(_hinge_y_angle(body_q, pulley_idx))

        test.assertGreater(left_z, 2.05, f"VBD light side should rise: z={left_z:.4f}")
        test.assertLess(right_z, 1.95, f"VBD heavy side should descend: z={right_z:.4f}")
        test.assertGreater(theta, 0.05, f"VBD pulley should rotate from no-slip coupling: theta={theta:.4f}")


def test_vbd_pulley_inertia_limit_locks_cable_travel(test, device):
    """As pulley inertia tends to infinity, no-slip cable travel tends to zero."""
    with wp.ScopedDevice(device):
        radius = 0.15
        model, _, _, pulley_idx = build_dynamic_pulley_atwood(
            mu=10.0,
            mass_left=1.0,
            mass_right=3.0,
            pulley_mass=20000.0,
            pulley_radius=radius,
        )
        state, _ = run_vbd_model(model, num_frames=60)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD high-inertia pulley state")

        theta = abs(_hinge_y_angle(body_q, pulley_idx))
        rim_travel = theta * radius

        test.assertLess(
            rim_travel,
            5.0e-3,
            f"VBD high-inertia no-slip pulley should lock cable travel: R*theta={rim_travel:.6f}, theta={theta:.6f}",
        )


def test_vbd_dynamic_capstan_mu_controls_pulley_rotation(test, device):
    """Dynamic capstan: zero mu slips, mid mu grips partially, high mu approaches no-slip."""
    with wp.ScopedDevice(device):
        low = _dynamic_capstan_metrics(device, mu=0.0)
        mid = _dynamic_capstan_metrics(device, mu=0.04)
        high = _dynamic_capstan_metrics(device, mu=10.0)

        for label, metrics in [("low", low), ("mid", mid), ("high", high)]:
            body_q, left_travel, right_travel, *_ = metrics
            test.assertTrue(np.isfinite(body_q).all(), f"Non-finite VBD dynamic capstan state for {label} mu")
            test.assertGreater(right_travel, 0.03, f"VBD heavy side should descend for {label} mu: {right_travel:.5f}")
            test.assertGreater(left_travel, -0.03, f"VBD light side should not sink for {label} mu: {left_travel:.5f}")

        _, _, _, _, theta_low, _, slip_low = low
        _, _, _, _, theta_mid, rim_mid, slip_mid = mid
        _, _, _, cable_high, theta_high, rim_high, slip_high = high

        test.assertLess(abs(theta_low), 0.035, f"Zero-mu VBD dynamic pulley should not rotate: theta={theta_low:.5f}")
        test.assertGreater(
            theta_mid, theta_low + 0.02, f"Mid-mu VBD pulley should rotate in cable direction: {theta_mid:.5f}"
        )
        test.assertGreater(
            theta_high, theta_mid + 0.03, f"High-mu VBD pulley should rotate more than mid mu: {theta_high:.5f}"
        )
        test.assertLess(
            abs(cable_high - rim_high),
            0.06,
            f"High-mu VBD dynamic capstan should approach no-slip: cable={cable_high:.5f}, rim={rim_high:.5f}",
        )
        test.assertGreater(
            slip_low, slip_mid, f"VBD dynamic slip should decrease from low to mid mu: {slip_low:.5f} <= {slip_mid:.5f}"
        )
        test.assertGreater(
            slip_mid,
            slip_high,
            f"VBD dynamic slip should decrease from mid to high mu: {slip_mid:.5f} <= {slip_high:.5f}",
        )
        test.assertGreater(rim_mid, 0.0, f"VBD mid-mu rim travel should be positive: {rim_mid:.5f}")


def test_vbd_kinematic_capstan_mu_controls_slip_and_locking(test, device):
    """Kinematic capstan: zero mu slips, mid mu slips less, high mu locks."""
    with wp.ScopedDevice(device):
        low = _kinematic_capstan_metrics(device, mu=0.0)
        mid = _kinematic_capstan_metrics(device, mu=0.08)
        high = _kinematic_capstan_metrics(device, mu=10.0)

        for label, metrics in [("low", low), ("mid", mid), ("high", high)]:
            body_q, _, _, theta = metrics
            test.assertTrue(np.isfinite(body_q).all(), f"Non-finite VBD kinematic capstan state for {label} mu")
            test.assertLess(abs(theta), 1.0e-5, f"VBD kinematic pulley should not rotate for {label} mu: {theta:.6f}")

        _, left_low, right_low, _ = low
        _, _, right_mid, _ = mid
        _, _, right_high, _ = high

        test.assertGreater(right_low, 0.18, f"Zero-mu VBD kinematic capstan should freely slip: dz={right_low:.5f}")
        test.assertGreater(left_low, 0.08, f"Zero-mu VBD light side should rise through slip: dz={left_low:.5f}")
        test.assertGreater(
            right_low, right_mid + 0.03, f"VBD mid mu should slip less than zero mu: {right_low:.5f} vs {right_mid:.5f}"
        )
        test.assertGreater(
            right_mid,
            right_high + 0.03,
            f"VBD high mu should lock more than mid mu: {right_mid:.5f} vs {right_high:.5f}",
        )
        test.assertLess(
            right_high, 0.06, f"High-mu VBD kinematic capstan should lock cable motion: dz={right_high:.5f}"
        )


def test_vbd_motorized_pulley_drives_slider(test, device):
    """A rolling drive pulley must convert rotation into cable sliding."""
    with wp.ScopedDevice(device):
        model, slider_idx, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=10.0)
        state, _ = run_vbd_motorized_model(model, drive_joint)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD motorized pulley state")

        slider_x = float(body_q[slider_idx][0])
        theta = abs(_hinge_z_angle(body_q, pulley_idx))

        test.assertGreater(theta, 0.5, f"VBD drive pulley should rotate under its target: theta={theta:.4f}")
        test.assertGreater(
            slider_x, -0.2, f"VBD no-slip drive should pull the slider through the cable: x={slider_x:.4f}"
        )


def test_vbd_frictionless_motorized_pulley_does_not_drive_slider(test, device):
    """With mu=0, pulley spin should not inject cable sliding through rolling transfer."""
    with wp.ScopedDevice(device):
        model, slider_idx, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=0.0)
        state, _ = run_vbd_motorized_model(model, drive_joint)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD frictionless motorized pulley state")

        slider_x = float(body_q[slider_idx][0])
        theta = abs(_hinge_z_angle(body_q, pulley_idx))

        test.assertGreater(theta, 0.5, f"VBD frictionless drive pulley should still rotate: theta={theta:.4f}")
        test.assertLess(
            abs(slider_x + 0.4),
            0.02,
            f"VBD frictionless pulley spin should not pull cable/slider: x={slider_x:.4f}",
        )


def test_vbd_motorized_pulley_couples_without_delay(test, device):
    """A driven pulley should move the cable during the initial rotation, not later."""
    with wp.ScopedDevice(device):
        model, slider_idx, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=10.0)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        q_start = int(model.joint_q_start.numpy()[drive_joint])
        initial_x = float(state_0.body_q.numpy()[slider_idx][0])
        dt = 1.0 / 60.0 / 10.0
        for _ in range(30):
            control.joint_target_q[q_start : q_start + 1].fill_(1.0)
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()
        slider_dx = float(body_q[slider_idx][0]) - initial_x
        theta = abs(_hinge_z_angle(body_q, pulley_idx))

        test.assertGreater(theta, 0.1, f"VBD pulley should have started rotating: theta={theta:.4f}")
        test.assertGreater(slider_dx, 0.02, f"VBD pulley rotation should immediately pull cable: dx={slider_dx:.4f}")


def test_vbd_motorized_pulley_updates_rest_in_first_step(test, device):
    """Rolling surface transfer should happen in the same VBD step as pulley rotation."""
    with wp.ScopedDevice(device):
        model, _, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=10.0)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        q_start = int(model.joint_q_start.numpy()[drive_joint])
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()
        control.joint_target_q[q_start : q_start + 1].fill_(1.0)

        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0 / 10.0)

        body_q = state_1.body_q.numpy()
        theta = abs(_hinge_z_angle(body_q, pulley_idx))
        rest_delta = solver.tendon_seg_rest_length.numpy() - initial_rest

        test.assertGreater(theta, 1.0e-3, f"VBD drive pulley should rotate in the first step: theta={theta:.6f}")
        test.assertGreater(
            float(np.max(np.abs(rest_delta))),
            1.0e-4,
            f"VBD pulley rotation should transfer rolling rest length in the first step: delta={rest_delta}",
        )


def test_vbd_moving_rolling_route_conserves_material(test, device):
    """VBD should conserve material while a rolling link's wrap angle changes."""
    with wp.ScopedDevice(device):

        def solver_factory(model):
            _set_serial_body_coloring(model)
            return _make_tendon_vbd_solver(model)

        cases = ((0.0, None), (0.2, None), (100.0, None), (100.0, 1))
        for mu, short_segment in cases:
            with test.subTest(mu=mu, short_segment=short_segment):
                error, min_rest = _moving_rolling_route_material_error(solver_factory, mu, short_segment)
                test.assertLess(
                    error,
                    1.0e-4,
                    f"VBD cable material drifted by {error} m for mu={mu}, short_segment={short_segment}",
                )
                if short_segment is not None:
                    test.assertLess(min_rest, 1.1e-6, "The VBD material-donor span should exercise the clamp")


def test_vbd_equal_weight_atwood(test, device):
    """Equal masses over a high-friction pulley should remain close to equilibrium."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx, pulley_idx = build_atwood_equal_weights()
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        dt = 1.0 / 60.0 / 16

        bq0 = state_0.body_q.numpy()
        y_left_0 = float(bq0[left_idx][2])
        y_right_0 = float(bq0[right_idx][2])

        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

        att_r = solver.tendon_seg_attachment_r.numpy()
        att_l = solver.tendon_seg_attachment_l.numpy()
        pulley_z = float(state_0.body_q.numpy()[pulley_idx][2])
        test.assertGreater(
            att_r[0][2],
            pulley_z,
            f"VBD cable should wrap over pulley: left tangent z={att_r[0][2]:.3f} <= center z={pulley_z:.3f}",
        )
        test.assertGreater(
            att_l[1][2],
            pulley_z,
            f"VBD cable should wrap over pulley: right tangent z={att_l[1][2]:.3f} <= center z={pulley_z:.3f}",
        )

        for _ in range(40):
            for _ in range(16):
                state_0.clear_forces()
                collision_pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

        bq = state_0.body_q.numpy()
        y_left = float(bq[left_idx][2])
        y_right = float(bq[right_idx][2])

        drift_left = abs(y_left - y_left_0)
        drift_right = abs(y_right - y_right_0)
        drift_diff = abs((y_left - y_left_0) - (y_right - y_right_0))

        test.assertLess(drift_left, 0.05, f"VBD left weight drifted {drift_left:.4f} m")
        test.assertLess(drift_right, 0.05, f"VBD right weight drifted {drift_right:.4f} m")
        test.assertLess(drift_diff, 0.02, f"VBD asymmetric drift: {drift_diff:.4f} m")
        test.assertTrue(np.isfinite(bq).all(), "Non-finite VBD equal-weight body positions")


devices = ["cpu"]
if wp.is_cuda_available():
    devices.append("cuda:0")

add_test(TestTendonVBD, "vbd_tendon_stretch_pulls_toward_anchor", devices, test_vbd_tendon_stretch_pulls_toward_anchor)
add_test(
    TestTendonVBD,
    "vbd_tendon_zero_compliance_uses_floor",
    devices,
    test_vbd_tendon_zero_compliance_uses_floor,
)
add_test(
    TestTendonVBD,
    "vbd_rolling_chain_tension_matches_accepted_pose",
    devices,
    test_vbd_rolling_chain_tension_matches_accepted_pose,
)
add_test(
    TestTendonVBD,
    "vbd_dynamic_tendon_route_switches_inside_solver",
    devices,
    test_vbd_dynamic_tendon_route_switches_inside_solver,
)
add_test(
    TestTendonVBD,
    "vbd_dynamic_route_activation_uses_accepted_pose",
    devices,
    test_vbd_dynamic_route_activation_uses_accepted_pose,
)
add_test(
    TestTendonVBD,
    "vbd_tendon_coloring_separates_segment_endpoints",
    devices,
    test_vbd_tendon_coloring_separates_segment_endpoints,
)
add_test(
    TestTendonVBD,
    "vbd_tendon_compliance_uses_physical_tension",
    devices,
    test_vbd_tendon_compliance_uses_physical_tension,
)
add_test(
    TestTendonVBD,
    "vbd_slack_damped_tendon_opposes_extension",
    devices,
    test_vbd_slack_damped_tendon_opposes_extension,
)
add_test(TestTendonVBD, "vbd_reset_restores_tendon_state", devices, test_vbd_reset_restores_tendon_state)
add_test(
    TestTendonVBD,
    "vbd_reset_rebaselines_tendon_at_custom_pose",
    devices,
    test_vbd_reset_rebaselines_tendon_at_custom_pose,
)
add_test(
    TestTendonVBD,
    "vbd_tendon_diagnostics_match_final_pose",
    devices,
    test_vbd_tendon_diagnostics_match_final_pose,
)
add_test(
    TestTendonVBD,
    "vbd_nonrolling_rotation_uses_discrete_length_rate",
    devices,
    test_vbd_nonrolling_rotation_uses_discrete_length_rate,
)
add_test(
    TestTendonVBD,
    "vbd_rolling_spin_does_not_add_damping_tension",
    devices,
    test_vbd_rolling_spin_does_not_add_damping_tension,
)
add_test(
    TestTendonVBD,
    "vbd_frictionless_off_center_roller_retains_body_torque",
    devices,
    test_vbd_frictionless_off_center_roller_retains_body_torque,
)
add_test(
    TestTendonVBD,
    "vbd_same_body_rolling_span_applies_net_torque",
    devices,
    test_vbd_same_body_rolling_span_applies_net_torque,
)
add_test(TestTendonVBD, "vbd_tendon_cuda_graph_capture", devices, test_vbd_tendon_cuda_graph_capture)
add_test(TestTendonVBD, "vbd_dynamic_tendon_cuda_graph_capture", devices, test_vbd_dynamic_tendon_cuda_graph_capture)
add_test(TestTendonVBD, "vbd_pinhole_slip_atwood", devices, test_vbd_pinhole_slip_atwood)
add_test(
    TestTendonVBD, "vbd_slack_pinhole_does_not_redistribute", devices, test_vbd_slack_pinhole_does_not_redistribute
)
add_test(
    TestTendonVBD, "vbd_dynamic_pulley_uses_angular_jacobian", devices, test_vbd_dynamic_pulley_uses_angular_jacobian
)
add_test(
    TestTendonVBD,
    "vbd_pulley_inertia_limit_locks_cable_travel",
    devices,
    test_vbd_pulley_inertia_limit_locks_cable_travel,
)
add_test(
    TestTendonVBD,
    "vbd_dynamic_capstan_mu_controls_pulley_rotation",
    devices,
    test_vbd_dynamic_capstan_mu_controls_pulley_rotation,
)
add_test(
    TestTendonVBD,
    "vbd_kinematic_capstan_mu_controls_slip_and_locking",
    devices,
    test_vbd_kinematic_capstan_mu_controls_slip_and_locking,
)
add_test(TestTendonVBD, "vbd_motorized_pulley_drives_slider", devices, test_vbd_motorized_pulley_drives_slider)
add_test(
    TestTendonVBD,
    "vbd_frictionless_motorized_pulley_does_not_drive_slider",
    devices,
    test_vbd_frictionless_motorized_pulley_does_not_drive_slider,
)
add_test(
    TestTendonVBD,
    "vbd_motorized_pulley_couples_without_delay",
    devices,
    test_vbd_motorized_pulley_couples_without_delay,
)
add_test(
    TestTendonVBD,
    "vbd_motorized_pulley_updates_rest_in_first_step",
    devices,
    test_vbd_motorized_pulley_updates_rest_in_first_step,
)
add_test(
    TestTendonVBD,
    "vbd_moving_rolling_route_conserves_material",
    devices,
    test_vbd_moving_rolling_route_conserves_material,
)
add_test(TestTendonVBD, "vbd_equal_weight_atwood", devices, test_vbd_equal_weight_atwood)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
