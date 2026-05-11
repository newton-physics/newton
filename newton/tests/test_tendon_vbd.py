# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for routed tendons in the VBD rigid solver."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.example_tendon_3d_routing import Example as Routing3DExample
from newton.examples.cable.example_tendon_cable_machine import Example as CableMachineExample
from newton.examples.cable.example_tendon_capstan_friction import Example as DynamicCapstanExample
from newton.examples.cable.example_tendon_capstan_kinematic import Example as KinematicCapstanExample
from newton.examples.cable.example_tendon_compound_pulley import Example as CompoundPulleyExample
from newton.examples.cable.example_tendon_equilibrium import Example as EquilibriumExample
from newton.examples.cable.example_tendon_gear_pulley import Example as GearPulleyExample
from newton.examples.cable.example_tendon_pinhole import Example as PinholeExample
from newton.examples.cable.example_tendon_pulley import Example as PulleyExample
from newton.examples.cable.example_tendon_rolling_pulley import Example as RollingPulleyExample
from newton.examples.cable.example_tendon_xy_table import Example as XYTableExample
from newton.examples.cable.example_tendon_xy_table import _desired_table_xy
from newton.tests.test_tendon_capstan import (
    build_dynamic_pulley_atwood,
    build_kinematic_pulley_atwood,
    build_motorized_pulley_drive,
    build_pinhole_atwood,
    build_slack_pinhole_route,
)
from newton.tests.test_tendon_equilibrium import build_atwood_equal_weights
from newton.tests.unittest_utils import find_nan_members, sanitize_identifier

TENDON_VBD_SOLVER_KWARGS = {
    "iterations": 20,
    "rigid_avbd_beta": 1.0e6,
    "rigid_joint_linear_k_start": 1.0e7,
    "rigid_joint_angular_k_start": 1.0e5,
    "rigid_joint_linear_kd": 5.0e-2,
    "rigid_joint_angular_kd": 2.0e-2,
    "rigid_tendon_relaxation": 0.7,
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


def _make_example_args(num_frames):
    return type(
        "Args",
        (),
        {"headless": True, "record": False, "num_frames": num_frames, "episode_frames": None, "test": True},
    )()


def _install_vbd_example_solver(example):
    _set_serial_body_coloring(example.model)
    solver_kwargs = dict(TENDON_VBD_SOLVER_KWARGS)

    is_xy_table = all(hasattr(example, attr) for attr in ("p2_dof_start", "p6_dof_start", "_apply_cable_pretension"))
    if is_xy_table:
        example.model.joint_target_kd[example.p2_dof_start : example.p2_dof_start + 1].fill_(200.0)
        example.model.joint_target_kd[example.p6_dof_start : example.p6_dof_start + 1].fill_(200.0)
        solver_kwargs["iterations"] = 30
        solver_kwargs["rigid_tendon_relaxation"] = 0.6

    example.solver = newton.solvers.SolverVBD(example.model, **solver_kwargs)
    if is_xy_table:
        example._apply_cable_pretension(0.99995)


def _assert_example_members_have_no_nans(test, example):
    for member_name in ("state_0", "state_1", "model", "control", "contacts"):
        if not hasattr(example, member_name):
            continue
        nan_members = find_nan_members(getattr(example, member_name))
        test.assertFalse(nan_members, f"NaN members found in {member_name}: {nan_members}")


def _run_vbd_example_assertions(test, device, example_cls, num_frames):
    with wp.ScopedDevice(device):
        example = example_cls(None, _make_example_args(num_frames))
        _install_vbd_example_solver(example)

        has_post_step = hasattr(example, "test_post_step")
        has_final = hasattr(example, "test_final")
        if not (has_post_step or has_final):
            test.fail(f"{example_cls.__module__}.Example does not define example assertions")

        for _ in range(num_frames):
            example.step()
            if has_post_step:
                example.test_post_step()

        if has_final:
            example.test_final()
        _assert_example_members_have_no_nans(test, example)


def _make_vbd_example_assertion_test(example_cls, num_frames, *, skip_cpu=False):
    def test_fn(test, device, cls=example_cls, frames=num_frames, cpu_skip=skip_cpu):
        if cpu_skip and not wp.get_device(device).is_cuda:
            test.skipTest("VBD example-assertion regression is CUDA-only because the scene is expensive")
        _run_vbd_example_assertions(test, device, cls, frames)

    return test_fn


def build_single_span_tendon():
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=0.0)

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

    model = builder.finalize()
    _set_serial_body_coloring(model)
    return model, body


def run_vbd_model(model, num_frames=80, substeps=12, fps=60):
    _set_serial_body_coloring(model)
    dt = 1.0 / fps / substeps
    solver = _make_tendon_vbd_solver(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
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
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dof_start = int(model.joint_qd_start.numpy()[drive_joint])
    for _ in range(num_frames):
        control.joint_target_pos[dof_start : dof_start + 1].fill_(target)
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
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
    contacts = model.contacts()

    substeps = 12
    dt = 1.0 / 60.0 / substeps
    last_angle = _hinge_y_angle(state_0.body_q.numpy(), pulley_idx)
    theta = 0.0

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
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
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    body_q = state_0.body_q.numpy()
    last_angle = _hinge_y_angle(body_q, pulley_idx)
    theta = 0.0

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
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


def _dynamic_capstan_example_metrics(device, num_frames=40):
    example = DynamicCapstanExample(None, None)
    _set_serial_body_coloring(example.model)
    example.solver = _make_tendon_vbd_solver(example.model)
    for _ in range(num_frames):
        example.step()
    theta = np.array(example._pulley_rotation_history[-1], dtype=np.float64)
    left_z = np.array(example._left_z_history[-1], dtype=np.float64)
    right_z = np.array(example._right_z_history[-1], dtype=np.float64)
    cable_travel = 0.5 * ((left_z - 2.0) + (2.0 - right_z))
    rim_travel = theta * example.pulley_radius
    slip = np.abs(cable_travel - rim_travel)
    return tuple(example.mus), theta, cable_travel, rim_travel, slip


def _dynamic_capstan_example_history(device, num_frames=100):
    example = DynamicCapstanExample(None, None)
    _set_serial_body_coloring(example.model)
    example.solver = _make_tendon_vbd_solver(example.model)
    for _ in range(num_frames):
        example.step()

    theta = np.asarray(example._pulley_rotation_history, dtype=np.float64)
    left_z = np.asarray(example._left_z_history, dtype=np.float64)
    right_z = np.asarray(example._right_z_history, dtype=np.float64)
    return example, theta, left_z, right_z


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
        contacts = model.contacts()

        initial_x = float(state_0.body_q.numpy()[body_idx][0])
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        final_x = float(state_1.body_q.numpy()[body_idx][0])

        test.assertLess(final_x, initial_x - 0.1, f"Tendon should pull endpoint inward: {initial_x} -> {final_x}")
        test.assertTrue(np.isfinite(state_1.body_q.numpy()).all(), "Non-finite VBD single-span state")


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
        contacts = model.contacts()

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


def test_vbd_dynamic_capstan_example_friction_modes(test, device):
    """The rendered VBD dynamic capstan cases should show zero, partial, and no-slip limits."""
    with wp.ScopedDevice(device):
        mus, theta, _cable_travel, rim_travel, slip = _dynamic_capstan_example_metrics(device)
        test.assertEqual(mus[0], 0.0, f"VBD dynamic capstan example should keep zero friction first: {mus}")
        test.assertGreater(mus[1], 0.0, f"VBD dynamic capstan example mid friction should be finite: {mus}")
        test.assertGreaterEqual(
            mus[2], 10.0, f"VBD dynamic capstan example high friction should be no-slip-like: {mus}"
        )

        theta_low, theta_mid, theta_high = theta
        rim_mid = rim_travel[1]
        slip_mid = slip[1]
        slip_high = slip[2]
        test.assertLess(abs(theta_low), 0.08, f"VBD example zero-mu pulley should not rotate: theta={theta_low:.5f}")
        test.assertGreater(
            theta_mid, 0.25, f"VBD example mid-mu pulley should rotate in cable direction: {theta_mid:.5f}"
        )
        test.assertGreater(
            theta_high,
            theta_mid + 0.10,
            f"VBD example pulley rotation should increase with friction: "
            f"low={theta_low:.5f}, mid={theta_mid:.5f}, high={theta_high:.5f}, mus={mus}",
        )
        test.assertLess(
            rim_mid, rim_travel[2], f"VBD example mid-mu rim travel should stay below high mu: {rim_travel}"
        )
        test.assertLess(
            slip_high,
            0.03,
            f"VBD example high-mu pulley should be no-slip/locked: "
            f"theta={theta_high:.5f}, slip={slip_high:.5f}, all_slip={slip}",
        )
        test.assertLess(
            slip_mid,
            slip[0],
            f"VBD example mid-mu slip should be below zero-mu slip: mid={slip_mid:.5f}, low={slip[0]:.5f}",
        )


def test_vbd_dynamic_capstan_example_rotation_transients(test, device):
    """Rendered VBD dynamic capstan trajectories should stay ordered and non-reversing."""
    with wp.ScopedDevice(device):
        example, theta, left_z, right_z = _dynamic_capstan_example_history(device, num_frames=60)
        dtheta = np.diff(theta, axis=0)

        test.assertTrue(np.isfinite(theta).all(), "Non-finite VBD dynamic capstan rotation history")
        test.assertTrue(np.isfinite(left_z).all(), "Non-finite VBD dynamic capstan payload history")
        test.assertLess(
            np.max(np.abs(theta[:, 0])),
            0.08,
            f"Zero-mu VBD dynamic pulley should remain nearly stationary over the whole render: {theta[:, 0]}",
        )
        test.assertGreaterEqual(
            np.min(dtheta[:, 2]),
            -5.0e-3,
            f"High-mu VBD dynamic pulley should not reverse direction: min_dtheta={np.min(dtheta[:, 2]):.5f}",
        )
        test.assertLessEqual(
            np.max(theta[:, 1] - theta[:, 2]),
            0.10,
            f"Mid-mu VBD dynamic pulley should not overtake the high-mu pulley: "
            f"max_mid_minus_high={np.max(theta[:, 1] - theta[:, 2]):.5f}, final={theta[-1]}",
        )
        test.assertLessEqual(
            np.max(dtheta[:, 1]),
            0.16,
            f"Mid-mu VBD dynamic pulley should not develop a delayed angular-velocity spike: "
            f"max_dtheta={np.max(dtheta[:, 1]):.5f}",
        )

        light_limit = example.pulley_z + example.pulley_radius - 0.02
        test.assertLess(
            np.max(left_z),
            light_limit,
            f"VBD dynamic capstan light payloads should not crest over the pulleys: "
            f"max_z={np.max(left_z, axis=0)}, limit={light_limit:.5f}",
        )

        cable_travel = 0.5 * ((left_z[-1] - 2.0) + (2.0 - right_z[-1]))
        rim_travel = theta[-1] * example.pulley_radius
        slip = np.abs(cable_travel - rim_travel)
        test.assertGreater(
            slip[1],
            slip[2],
            f"Mid-mu VBD dynamic capstan should retain more slip than high mu over the pre-contact prefix: {slip}",
        )


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
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        dof_start = int(model.joint_qd_start.numpy()[drive_joint])
        initial_x = float(state_0.body_q.numpy()[slider_idx][0])
        dt = 1.0 / 60.0 / 10.0
        for _ in range(30):
            control.joint_target_pos[dof_start : dof_start + 1].fill_(1.0)
            state_0.clear_forces()
            model.collide(state_0, contacts)
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
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        dof_start = int(model.joint_qd_start.numpy()[drive_joint])
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()
        control.joint_target_pos[dof_start : dof_start + 1].fill_(1.0)

        state_0.clear_forces()
        model.collide(state_0, contacts)
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


def test_vbd_rolling_transfer_saturates_at_zero_span(test, device):
    """Rolling transfer should clamp before a free span goes negative."""
    with wp.ScopedDevice(device):
        model, slider_idx, _, drive_joint = build_motorized_pulley_drive(mu=10.0)
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        dof_start = int(model.joint_qd_start.numpy()[drive_joint])
        dt = 1.0 / 60.0 / 10.0
        saturated_x = None

        for _frame in range(60):
            control.joint_target_pos[dof_start : dof_start + 1].fill_(8.0)
            for _ in range(10):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            rest = solver.tendon_seg_rest_length.numpy()
            if saturated_x is None and np.min(rest) <= 1.1e-6:
                body_q = state_0.body_q.numpy()
                saturated_x = float(body_q[slider_idx][0])

        body_q = state_0.body_q.numpy()
        final_x = float(body_q[slider_idx][0])
        rest = solver.tendon_seg_rest_length.numpy()

        test.assertIsNotNone(saturated_x, "VBD driven pulley should exhaust one adjacent free span")
        test.assertGreaterEqual(float(np.min(rest)), 0.99e-6, f"VBD rest lengths must stay non-negative: {rest}")
        test.assertLess(
            abs(final_x - saturated_x),
            2.0e-2,
            f"VBD slider should lock once a free span is exhausted: {final_x:.6f} vs {saturated_x:.6f}",
        )


def test_vbd_equal_weight_atwood(test, device):
    """Equal masses over a high-friction pulley should remain close to equilibrium."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx, pulley_idx = build_atwood_equal_weights()
        _set_serial_body_coloring(model)
        solver = _make_tendon_vbd_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        dt = 1.0 / 60.0 / 16

        bq0 = state_0.body_q.numpy()
        y_left_0 = float(bq0[left_idx][2])
        y_right_0 = float(bq0[right_idx][2])

        state_0.clear_forces()
        model.collide(state_0, contacts)
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
                model.collide(state_0, contacts)
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


def test_vbd_compound_pulley_stays_balanced(test, device):
    """The VBD compound-pulley render path must not pass while visibly unstable."""
    _run_vbd_example_assertions(test, device, CompoundPulleyExample, 220)


def test_vbd_rolling_pulley_revolute_axis_stays_stiff(test, device):
    """VBD tendon scenes need stiff revolute anchors so routing geometry stays coherent."""
    with wp.ScopedDevice(device):
        args = type("Args", (), {"headless": True, "record": False, "num_frames": 24, "episode_frames": None})()
        example = RollingPulleyExample(None, args)
        _set_serial_body_coloring(example.model)
        example.solver = _make_tendon_vbd_solver(example.model)

        body_q0 = example.state_0.body_q.numpy().copy()
        pulley_idx = example.pulley_idx
        max_pulley_drift = 0.0

        for _ in range(24):
            example.step()
            body_q = example.state_0.body_q.numpy()
            test.assertTrue(np.isfinite(body_q).all(), "Non-finite VBD rolling-pulley state")
            drift = float(np.linalg.norm(body_q[pulley_idx][:3] - body_q0[pulley_idx][:3]))
            max_pulley_drift = max(max_pulley_drift, drift)

        test.assertLess(
            max_pulley_drift,
            5.0e-3,
            f"VBD rolling-pulley revolute anchor drift is too soft: {max_pulley_drift:.6f} m",
        )


def test_vbd_gear_pulley_direction_and_mechanical_advantage(test, device):
    """The VBD gear example must match the XPBD direction: free light weight falls."""
    with wp.ScopedDevice(device):
        example = GearPulleyExample(None, None)
        _set_serial_body_coloring(example.model)
        example.solver = _make_tendon_vbd_solver(example.model)

        for _ in range(example._direction_validation_frames):
            example.step()

        lower_positions = np.asarray(example._lower_history, dtype=np.float64)
        right_positions = np.asarray(example._right_history, dtype=np.float64)
        rotations = np.asarray(example._pulley_rotation_history, dtype=np.float64)

        sample = len(lower_positions) - 1
        lower_rise = float(lower_positions[sample, 2]) - example._initial_lower_z
        right_drop = example._initial_right_z - float(right_positions[sample, 2])
        right_travel = float(np.linalg.norm(right_positions[sample] - right_positions[0]))
        travel_ratio = right_travel / max(lower_rise, 1.0e-6)
        final_rim_travel = np.abs(rotations[sample] * example._pulley_radii)

        test.assertGreater(lower_rise, 0.05, f"VBD gear moving block should rise: dz={lower_rise:.4f}")
        test.assertGreater(
            right_drop,
            0.20,
            f"VBD gear free counterweight should fall, not rise: dz={right_drop:.4f}",
        )
        test.assertGreater(travel_ratio, 2.5, f"VBD gear should amplify free-weight travel: ratio={travel_ratio:.3f}")
        test.assertLess(
            travel_ratio, 5.5, f"VBD gear travel ratio is not the expected block-and-tackle range: {travel_ratio:.3f}"
        )
        inactive = [example._pulley_names[i] for i, travel in enumerate(final_rim_travel) if float(travel) <= 0.025]
        test.assertFalse(inactive, f"All VBD gear pulleys should rotate: inactive={inactive}, rim={final_rim_travel}")


def test_vbd_xy_table_tracks_reference_prefix(test, device):
    """VBD XY table should follow the same reference direction as the XPBD example."""
    if not wp.get_device(device).is_cuda:
        test.skipTest("XY table VBD reference-prefix regression is CUDA-only because the scene is expensive")

    with wp.ScopedDevice(device):
        example = XYTableExample(None, None)
        _set_serial_body_coloring(example.model)

        example.model.joint_target_kd[example.p2_dof_start : example.p2_dof_start + 1].fill_(200.0)
        example.model.joint_target_kd[example.p6_dof_start : example.p6_dof_start + 1].fill_(200.0)

        solver_kwargs = dict(TENDON_VBD_SOLVER_KWARGS)
        solver_kwargs["iterations"] = 30
        solver_kwargs["rigid_tendon_relaxation"] = 0.6
        example.solver = newton.solvers.SolverVBD(example.model, **solver_kwargs)
        example._apply_cable_pretension(0.99995)

        for _ in range(30):
            example.step()
            example.test_post_step()

        example.test_final()

        table_xy = np.asarray(example._table_xy_history, dtype=np.float64)
        drive_rot = np.asarray(example._drive_rotation_history, dtype=np.float64)
        sample_times = (np.arange(len(table_xy), dtype=np.float64) + 1.0) * example.frame_dt
        reference_xy = np.asarray([_desired_table_xy(float(t)) for t in sample_times], dtype=np.float64)
        reference_error = np.linalg.norm(table_xy - reference_xy, axis=1)
        reference_rms = float(np.sqrt(np.mean(reference_error * reference_error)))
        reference_max = float(np.max(reference_error))
        y_drift = float(np.max(np.abs(table_xy[:, 1])))
        drive_pair_error = float(np.max(np.abs(drive_rot[:, 0] - drive_rot[:, 1])))

        test.assertLess(reference_rms, 0.008, f"VBD XY table prefix RMS drifted from reference: {reference_rms:.5f}")
        test.assertLess(reference_max, 0.012, f"VBD XY table prefix max error too high: {reference_max:.5f}")
        test.assertLess(y_drift, 0.004, f"VBD XY table should stay on the X reference prefix: y={y_drift:.5f}")
        test.assertLess(
            drive_pair_error,
            0.02,
            f"VBD XY table drive pulleys should stay coupled in the symmetric prefix: error={drive_pair_error:.5f}",
        )


test_vbd_gear_pulley_direction_and_mechanical_advantage.__unittest_expected_failure_devices__ = {"cuda:0"}


VBD_EXAMPLE_ASSERTION_CASES = (
    ("vbd_tendon_pulley_example_assertions", PulleyExample, 100, True, set()),
    ("vbd_tendon_pinhole_example_assertions", PinholeExample, 100, True, set()),
    ("vbd_tendon_rolling_pulley_example_assertions", RollingPulleyExample, 180, True, set()),
    ("vbd_tendon_equilibrium_example_assertions", EquilibriumExample, 100, True, set()),
    ("vbd_tendon_gear_pulley_example_assertions", GearPulleyExample, 180, True, {"cuda:0"}),
    ("vbd_tendon_cable_machine_example_assertions", CableMachineExample, 100, True, set()),
    ("vbd_tendon_3d_routing_example_assertions", Routing3DExample, 70, True, set()),
    ("vbd_tendon_capstan_friction_example_assertions", DynamicCapstanExample, 100, True, set()),
    ("vbd_tendon_capstan_kinematic_example_assertions", KinematicCapstanExample, 100, True, set()),
)


devices = ["cpu"]
if wp.is_cuda_available():
    devices.append("cuda:0")

add_test(TestTendonVBD, "vbd_tendon_stretch_pulls_toward_anchor", devices, test_vbd_tendon_stretch_pulls_toward_anchor)
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
    "vbd_dynamic_capstan_example_friction_modes",
    devices,
    test_vbd_dynamic_capstan_example_friction_modes,
)
add_test(
    TestTendonVBD,
    "vbd_dynamic_capstan_example_rotation_transients",
    devices,
    test_vbd_dynamic_capstan_example_rotation_transients,
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
    "vbd_rolling_transfer_saturates_at_zero_span",
    devices,
    test_vbd_rolling_transfer_saturates_at_zero_span,
)
add_test(TestTendonVBD, "vbd_equal_weight_atwood", devices, test_vbd_equal_weight_atwood)
add_test(TestTendonVBD, "vbd_compound_pulley_stays_balanced", devices, test_vbd_compound_pulley_stays_balanced)
add_test(
    TestTendonVBD,
    "vbd_rolling_pulley_revolute_axis_stays_stiff",
    devices,
    test_vbd_rolling_pulley_revolute_axis_stays_stiff,
)
add_test(
    TestTendonVBD,
    "vbd_gear_pulley_direction_and_mechanical_advantage",
    devices,
    test_vbd_gear_pulley_direction_and_mechanical_advantage,
)
add_test(TestTendonVBD, "vbd_xy_table_tracks_reference_prefix", devices, test_vbd_xy_table_tracks_reference_prefix)
for name, example_cls, num_frames, skip_cpu, expected_failure_devices in VBD_EXAMPLE_ASSERTION_CASES:
    test_fn = _make_vbd_example_assertion_test(example_cls, num_frames, skip_cpu=skip_cpu)
    test_fn.__unittest_expected_failure_devices__ = expected_failure_devices
    add_test(TestTendonVBD, name, devices, test_fn)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
